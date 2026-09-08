# RotaryPosEmb 项目交接文档（供后续 Agent）

> 最后更新：2026-09-08  
> 目标读者：后续继续本项目的 Agent / 开发者  
> 原则：先读本文 + `docs/problem.md`，再动代码或提交

---

## 0. 一句话状态

- **工程**：Ascend C Kernel 直调实现 RoPE（`kernel.asc`）
- **本地 PV**：20 组题面对齐用例，**20/20 精度 Pass**（远端 NPU）
- **探测提交**（Awaycsdn）：**12/15 Pass**，case12 **Runtime Error**，**不计分**
- **队长榜**（Poetratume）：**#25 / 43.41 分**，15/15 Pass（性能落后）
- **下一优先**：(1) 修 case12 RE (2) 压官方 2.5–3.3× case (3) 再碰 case9 秒级毒瘤

---

## 1. 仓库与路径

| 角色 | 路径 |
|------|------|
| 本地工程 | `E:\Learn\CANN\rotaryposemb\` |
| GitHub | https://github.com/AwayGithub/rotaryposemb |
| 远端 DevEnv 工程 | `/mnt/workspace/rotaryposemb/` |
| 官方题面（完整） | `docs/problem.md` |
| 题面原始 JSON | `docs/problem.raw.json` |
| 上板数据 | `perf_onboard/`（含 `ANALYSIS.md`、`PROBE_SUBMIT.md`、tgz） |
| cann-learning-hub | `E:\Learn\CANN\cann-learning-hub\`（含 `skills/cannjudge-submit`） |

### 1.1 远端连接（AtomGit / CANNlab）

- SSH Host 示例：`devenvc_unmn2.<id>.atomgit.0` → `127.0.0.1:20022`
- 用户：`developer`
- CANN：`source /home/developer/Ascend/cann-9.0.0/set_env.sh`
- NPU：`/dev/davinci2`、`davinci3`（Ascend910，dav-2201）
- 转发依赖本机 VS Code 扩展 **CANNlab Dev Space** 的 hub（端口 20022/10875）

### 1.2 关键文件

```text
rotaryposemb/
├── kernel.asc          # 唯一核心：vec kernel + reorder kernel + run_kernel
├── main.asc            # 本地 CLI 测试入口
├── CMakeLists.txt
├── build.sh / run.py / run.sh
├── scripts/
│   ├── cases_spec.py   # 20 组 PV 定义（题面对齐）
│   ├── gen_data.py     # 造数 + golden
│   ├── verify_result.py
│   └── RotaryPosEmb.py # golden（FP32 inv_freq/cos/sin）
├── docs/
│   ├── problem.md      # 官方完整题面
│   ├── problem.raw.json
│   └── AGENT_HANDOFF.md  # 本文件
└── perf_onboard/       # 上板与探测提交产物
```

---

## 2. 赛题要点（官方）

- **页面**：https://cannjudge.cn/public/op_challenge_jingdong_prelim/rotaryposemb  
- **problem_id**：`6a9a9befbf41025d60143218`  
- **模板**：`npu_kernel_dev`（提交 **`kernel.asc`**，不是 op_host/op_kernel 四件套）  
- **CANN**：9.0.0，vector  

### 2.1 IO

| 名 | shape | dtype |
|----|-------|-------|
| x | (B,S,H,rope_dim) | fp16/fp32 |
| positions | (B,S) | int32，[0,131072] |
| theta | attr | float，默认 10000 |
| y | 同 x | 同 x |

范围：B∈[1,128]，S∈[1,8192]，H∈[1,256]，rope_dim∈[16,1024] 且**偶数**。

### 2.2 数学（rotate-half）

```text
inv_freq[i] = theta^(-2i/dim), i=0..dim/2-1
θ = p * inv_freq
y[..., i]           = x[..., i]*cosθ - x[..., i+dim/2]*sinθ
y[..., i+dim/2]     = x[..., i+dim/2]*cosθ + x[..., i]*sinθ
```

- cos/sin、inv_freq **必须 FP32 中间精度**  
- 精度：fp32 rtol/atol **1e-4**；fp16 **1e-3**  

### 2.3 计分

```text
score_i = 100 / (1 + log_1.5(t_i / T_i))
final   = mean(score_i)   # 仅当 15/15 精度 Pass
```

- `T_i` = 该 case **全场最优**（不是榜一每一列）  
- **15 个官方 case 全过精度才计分**  
- 慢 1.5×≈50 分；慢 3×≈27 分；**多个 2.5×+ 会钉死在 40 分段**

### 2.4 官方示例 shape

1. `(1,8,1,16)` fp32  
2. `(2,512,128,64)` fp16（DeepSeek 风格）  
3. `(2,16,3,24)` fp16 非对齐 dim  

---

## 3. 实现架构（kernel.asc）

### 3.1 两条路径

| 条件 | 路径 | 说明 |
|------|------|------|
| `dim % 16 == 0` | `rotary_pos_emb_vec_kernel` | 纯 device，TPipe/TQue |
| 否则 | host gather → `rotary_pos_emb_reorder_kernel` → host scatter | half-slot 32B 对齐布局 |

### 3.2 平台硬限制（踩坑结论，勿回退）

1. 无可靠设备端 Sin/Cos/libm → 多项式 + 分段 2π 归约  
2. DMA **32B 对齐**；DataCopyPad 在 2201 上不可用  
3. 手动 SetFlag/WaitFlag 不可靠 → 用 **TQue**  
4. TBuf 与 TQue 混用易踩内存 → trig scratch 走 TQue  
5. `iBuf_` 必须初始化（`trig_[8*MAX_HALF].ReinterpretCast<int32_t>()`），否则 Cast 写地址 0  
6. invf 的 DataCopy count 必须 32B 对齐长度（用 `alignHalf`）  

### 3.3 已做优化（2026-09-08）

- invf 按 `(theta,dim)` 缓存，避免重复 H2D  
- TILE 放大；大 H 多 tile + vec 路径 TQue depth-2  
- reorder 首 tile 预取；host pack 用独立 unpack buffer  
- **本地 msprof 相对旧 baseline 几乎无加速**（见 `perf_onboard`）  
- 提交版勿含：`ROPE_TIMING` / `chrono` / `thread` / `fprintf` / `getenv`（会触发「不合规内容」）

### 3.4 编译运行（远端）

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
cd /mnt/workspace/rotaryposemb
bash build.sh
python3 run.py all          # 20 组
python3 run.py 17           # 单组
```

产物：`build/rotary_pos_emb_custom`  
CLI：`./rotary_pos_emb_custom dtype B S H D theta x.bin p.bin y.out [coreNum]`  
dtype：0=fp32，1=fp16  

### 3.5 Ascend C Toolkit

- settings 已指 `build.sh` + `python ../run.py`  
- 「运行」= 20 组 PV；「上板分析」= `msprof op`（命令行已跑通）  
- 输出目录权限需 `chmod 700`，否则 msprof 拒绝写入  

---

## 4. 本地 20 组 PV（cases_spec）

定义：`scripts/cases_spec.py`（id 0–19）

| 覆盖 | case 例 |
|------|---------|
| 官方 ex1 | 0: (1,8,1,16) fp32 |
| 官方 ex3 非对齐 | 3: (2,16,3,24) fp16 |
| 官方 ex2 DeepSeek | 17: (2,512,128,64) fp16 |
| pos=0 | 6/7 |
| theta 变化 | 11–14 |
| 大 dim | 15/16 |
| 长序列 | 18 |
| 大非对齐 host | 19: (2,256,32,24) fp16 |

Golden：`RotaryPosEmb.impl`（与题面 PyTorch 语义一致）。  
校验阈值：与官方相同（1e-4 / 1e-3）。

**注意**：本地 20 组 **复现不出官方 case9 的秒级**；kernel 侧最大约数百 μs。秒级更像 host/路径/ harness。

---

## 5. 性能与榜单分析摘要

详细报告：`perf_onboard/ANALYSIS.md`  
上板原始：`perf_onboard/perf_onboard_complete.tgz`、`data/perf_onboard/`

### 5.1 队长（Poetratume）vs 全场最优 T

- 榜：**#25 / 43.41**，队「张雪峰老师我们还记得你队」  
- **最亏**：case5/6/7/8/10/11 约 **2.1–3.3×T**（单 case 估分 ~25–34）  
- case9：队长 ~11s，最优仍 ~3.9s（全场毒瘤）  
- case12/13 已接近最优  

### 5.2 本地上板模式

- 小 shape：**pipeline bound**（MTE2/3 利用率低）  
- 大 shape（17/18/10）：**compute bound**  
- case17 task_us ≈ **325 μs**（监控锚点）  

### 5.3 进步空间优先级

| 优先级 | 内容 |
|--------|------|
| **P0** | 把官方 2.5–3.3× case 压到 1.2–1.5×（名次跃迁）；host/device 拆分 |
| **P0b** | **修 case12 Runtime Error**（否则永远不计分） |
| **P1** | case9 秒级毒瘤：反推 shape + 路径 |
| **P2** | 大 shape 双缓冲/Trig（谨慎，历史踩坑） |
| **P3** | 小 shape pipeline（收益低） |

---

## 6. CANNJudge 提交

### 6.1 账号

| 账号 | 说明 |
|------|------|
| **Awaycsdn** | 用户本人（GitCode OAuth），uid `6aa01359c76b321ca60bc0af` |
| **Poetratume** | **组长**，非本人；榜上 43.41 分是组长号 |

登录方式：**GitCode OAuth**，不是邮箱密码。  
会话 cookie 名：`cannjudge_auth`（JWT：`{uid, exp}.sig`）。  
Cookie 易过期；勿提交到 git。临时可放 `%TEMP%\opencode\cj_session.json`。

### 6.2 提交格式（npu_kernel_dev）

```http
POST /api/submissions/submit
Cookie: cannjudge_auth=...
{
  "problemId": "6a9a9befbf41025d60143218",
  "userId": "<uid>",
  "files": [ { "path": "kernel.asc", "content": "..." } ],
  "tiling_h": "", "tiling_key_h": "", "host_cpp": "", "kernel_cpp": ""
}
```

- **不要**交含 Debug/计时/thread/getenv/fprintf 的代码 → `提交代码中存在不合规内容`  
- 查询：`GET /api/submissions/{id}?userId=`  
- 排行榜公开：`GET /api/problems/{pid}/ranking`  

### 6.3 探测提交结果（2026-09-08）

- ID：`6aa01f67c76b321ca61031fb`（244171）  
- **12/15 Pass**，case12 **RE**，13–14 Skipped  
- 时延与队长几乎同量级 → 同一实现路径  
- 明细：`perf_onboard/PROBE_SUBMIT.md`  
- 原始 JSON：`%TEMP%\opencode\rope_result.json`  

| case | Awaycsdn ms | 状态 |
|------|-------------|------|
| 0–11 | 与队长相近 | Pass |
| 9 | ~11179 | Pass（秒级） |
| 12 | 0 | **Runtime Error** |
| 13–14 | 0 | Skipped |

---

## 7. 相关工具链

| 工具 | 用途 |
|------|------|
| `cannjudge-submit` skill | 登录/下载/提交/查分（learning-hub） |
| 本机 skill 链接 | `.opencode/skills/cannjudge-submit` → learning-hub |
| Ascend C Toolkit | 编译/运行/上板（需适配 build.sh/run.py） |
| `msprof op` | 上板分析；output 目录权限 700 |

### 7.1 上板命令模板

```bash
source ~/Ascend/cann-9.0.0/set_env.sh
cd /mnt/workspace/rotaryposemb/build
mkdir -m 700 -p /tmp/opprof_out
msprof op --output=/tmp/opprof_out \
  --aic-metrics=Default,Roofline,PipeUtilization,Memory,BasicInfo \
  --warm-up=2 --launch-count=3 \
  ./rotary_pos_emb_custom 0 1 8 1 16 10000.0 \
  input/case0/x.bin input/case0/positions.bin output/y.bin 0
```

---

## 8. 安全与红线

1. **禁止**把密码、完整 cookie、private.pem 写入仓库  
2. 用户曾在对话中提供过密码 → 建议用户自行修改；Agent 勿回显  
3. 提交前扫描：`Debug/debug/fprintf/getenv/chrono/thread/ROPE`  
4. 本地 HANDOFF 旧文件可能含敏感信息 → **不要 git add**  
5. Poetratume ≠ 当前用户；勿把队长成绩说成用户本人  

---

## 9. 建议工作流（后续 Agent）

```text
1. 读 docs/problem.md + 本文件
2. 远端 source CANN；git pull；bash build.sh；python3 run.py all
3. 优先：定位/修复官方 case12 RE
   - 在 cases_spec 增加疑似大 dim/大 S 回归
   - 本地复现 RE 再改 kernel
4. 其次：host vs device 计时（大 unaligned）
5. 干净 kernel.asc 探测提交 → 更新 15×ms 表
6. 针对 2.5×+ case 做单点优化，再提交验收 t/T
```

### 9.1 成功标准

| 阶段 | 标准 |
|------|------|
| 本地 | `run.py all` → 20/20 Pass |
| 提交 | 15/15 Pass 且出现 score |
| 性能 | 最差若干 case 的 t/T 从 ~3× 降到 ~1.2–1.5×；榜分上 55–60+ |

---

## 10. 探测提交时延快照（Awaycsdn，供对比）

```text
#   ms
0    3.52
1    3.90
2   12.52
3   16.05
4   27.08
5   39.45
6   44.83
7   30.23
8   79.79
9 11178.81
10 101.58
11 193.58
12 RE
13 skipped
14 skipped
```

全场最优 T 快照（公开榜 min，会随时间变）：

```text
T = [2.27, 3.77, 5.02, 11.04, 13.52, 13.17, 19.67, 10.50, 36.51, 3875.52, 40.07, 57.90, 330.76, 853.76, 291.97]
```

---

## 11. 变更与提交记录（Git）

| commit | 说明 |
|--------|------|
| 初始 | functional baseline |
| Toolkit adapters | build.sh / run.py |
| docs problem | 官方题面入库 |
| 20 cases | cases_spec + gen/verify/run |
| eed6371 等 | P0 工程优化（双缓冲/invf 缓存等） |

远端与 GitHub 以 `main` 为准；大文件 `perf_onboard/` 可不入库（体积大），但 `ANALYSIS.md` / `PROBE_SUBMIT.md` 建议保留或精简入库。

---

## 12. 给 Agent 的最短指令

```text
你在做 CANNJudge RotaryPosEmb。
代码在 E:\Learn\CANN\rotaryposemb 与远端 /mnt/workspace/rotaryposemb。
先读 docs/problem.md 与 docs/AGENT_HANDOFF.md。
本地 20/20 已 Pass；官方探测 12/15，case12 RE 必须先修。
提交只交干净 kernel.asc；GitCode cookie 登录；禁 debug 符号。
性能以官方 15 case 的 t/T 为准，参考 perf_onboard/ANALYSIS.md。
```

---

*本文档由项目会话整理，用于上下文压缩后的连续开发。*
