# RotaryPosEmb 上板性能 × 赛题得分 分析报告

## 1. 数据来源

- 本地 PV：20 组 `scripts/cases_spec.py`
- 上板：`msprof op`，20/20 成功
- 本机目录：`E:\Learn\CANN\rotaryposemb\perf_onboard\`
  - `perf_onboard_complete.tgz`（整包 ~1.5MB）
  - `data/perf_onboard/`（已解压 CSV）
- 计分：15 官方 case 精度全过才计分；`100/(1+log_1.5(t/T))` 取均值
- 榜单：队长 Poetratume **#25 / 43.41**；榜一 **#1 / 67.18**

## 2. 计分机制

```text
score_i = 100 / (1 + log_1.5(t_i / T_i))
final   = mean(score_i)   # only if 15/15 Pass
```

- `T_i` = 该 case **全场最优**（不一定是榜一）
- 慢 1.5× ≈ 50 分；慢 3× ≈ 27 分
- 多个 2.5×+ 的 case 会把总分钉在 40 分段

### 2.1 全场最优 T vs 队长 / 榜一（ms）

| case | T(ms) | 最优者 | 队长 t | t/T | 估分 | 榜一 t |
|------|-------|--------|--------|-----|------|--------|
| 0 | **2.27** | a384649 | 3.63 | **1.60×** | 46.3 | 3.96 |
| 1 | **3.77** | Poetratume | 3.77 | **1.00×** | 100.0 | 5.11 |
| 2 | **5.02** | Piggy | 12.74 | **2.54×** | 30.3 | 6.42 |
| 3 | **11.04** | 王帅 | 15.98 | **1.45×** | 52.3 | 11.15 |
| 4 | **13.52** | 华安9527 | 26.62 | **1.97×** | 37.4 | 13.52 |
| 5 | **13.17** | espressolee | 39.38 | **2.99×** | 27.0 | 14.05 |
| 6 | **19.67** | 华安9527 | 44.76 | **2.28×** | 33.0 | 19.67 |
| 7 | **10.50** | fangzhixiang | 30.55 | **2.91×** | 27.5 | 10.52 |
| 8 | **36.51** | 阿伟sl | 79.37 | **2.17×** | 34.3 | 36.66 |
| 9 | **3875.52** | Piggy | 11090.92 | **2.86×** | 27.8 | 6624.85 |
| 10 | **40.07** | fangzhixiang | 102.37 | **2.55×** | 30.2 | 79.52 |
| 11 | **57.90** | 怪盗基德 | 190.89 | **3.30×** | 25.4 | 89.61 |
| 12 | **330.76** | XingKong_YAMINO | 339.58 | **1.03×** | 93.9 | 348.68 |
| 13 | **853.76** | fangzhixiang | 896.76 | **1.05×** | 89.2 | 1060.23 |
| 14 | **291.97** | swlaird | 407.00 | **1.39×** | 55.0 | 427.43 |
| 均值 |  |  |  |  | **47.31** | 榜一估分 **71.83** |

公式均值队长 **47.31**（榜面 43.41，约差 4 分，可能取整/T 滑动）；榜一 **71.83**（榜面 67.18）。**相对关系可信，可指导优化。**

**队长最亏的 case**

- **case11**: 3.30×T（190.9/57.9 ms）→ **25.4 分**
- **case5**: 2.99×T（39.4/13.2 ms）→ **27.0 分**
- **case7**: 2.91×T（30.6/10.5 ms）→ **27.5 分**
- **case9**: 2.86×T（11090.9/3875.5 ms）→ **27.8 分**
- **case10**: 2.55×T（102.4/40.1 ms）→ **30.2 分**
- **case2**: 2.54×T（12.7/5.0 ms）→ **30.3 分**

### 2.2 优化情景估分

| 情景 | 估分 | Δ |
|------|------|---|
| 现状 | **47.31** | +0.00 |
| 只修 case11 → 1.2×T | **50.22** | +2.91 |
| 修最差 3 个 → 1.2×T | **55.78** | +8.47 |
| 全部 1.5×T | **50.00** | +2.69 |
| 全部 1.2×T | **68.98** | +21.67 |
| 全部 1.05×T | **89.26** | +41.94 |
| 全场最优 | **100.00** | +52.69 |

### 2.3 解读

1. **case9 全场都慢**（最优 3.9s，队长 11.1s，2.86×）——毒瘤，但不是唯一矛盾。
2. **名次真正被 case5/6/7/8/10/11 拖死**（普遍 2.1–3.3×，单 case 25–34 分）。
3. case12/13 已接近最优（~1.03–1.05×）——部分大 case 并不差。
4. 43→60+ 需要把 **一串 2.5×+ case 压到 1.2–1.5×**，不是只优化小 shape。

## 3. 本地 20 组上板（kernel task_us）

> 设备侧 μs；判题 ms 含 host。本地 **没有秒级 kernel** → 榜上秒级更像 host/路径问题。

| case | tag | shape | dtype | task_us | blk | bound |
|------|-----|-------|-------|---------|-----|-------|
| 00 | official-ex1-fp32 | `(1,8,1,16)` | float32 | 3.10 | 8 | pipeline caused |
| 01 | small-fp32 | `(1,7,1,16)` | float32 | 2.92 | 7 | pipeline caused |
| 02 | small-fp16 | `(1,8,1,16)` | float16 | 3.00 | 8 | pipeline caused |
| 03 | official-ex3-dim24 | `(2,16,3,24)` | float16 | 4.14 | 32 | pipeline caused |
| 04 | unaligned-dim18 | `(2,7,5,18)` | float16 | 3.66 | 14 | pipeline caused |
| 05 | unaligned-dim20-fp32 | `(3,11,2,20)` | float32 | 3.96 | 33 | pipeline caused |
| 06 | pos0-fp16 | `(1,9,2,16)` | float16 | 3.20 | 9 | pipeline caused |
| 07 | pos0-fp32 | `(1,9,2,16)` | float32 | 3.08 | 9 | pipeline caused |
| 08 | mid-fp16-wide-pos | `(2,64,8,64)` | float16 | 8.84 | 40 | compute caused |
| 09 | mid-fp32-wide-pos | `(2,64,8,64)` | float32 | 8.78 | 40 | compute caused |
| 10 | mid-fp16-h32 | `(2,128,32,64)` | float16 | 28.46 | 40 | compute caused |
| 11 | theta-1 | `(1,64,4,32)` | float32 | 5.22 | 40 | compute caused |
| 12 | theta-1e3 | `(2,32,8,64)` | float16 | 5.94 | 40 | compute caused |
| 13 | theta-5e5 | `(2,32,8,64)` | float16 | 6.08 | 40 | compute caused |
| 14 | theta-3.5e4 | `(2,32,8,64)` | float16 | 6.14 | 40 | compute caused |
| 15 | dim256-fp32 | `(1,128,4,256)` | float32 | 7.58 | 40 | compute caused |
| 16 | dim1024-edge | `(1,16,1,1024)` | float32 | 3.56 | 16 | pipeline caused |
| 17 | official-ex2-deepseek | `(2,512,128,64)` | float16 | 325.02 | 40 | compute caused |
| 18 | long-seq | `(1,4096,1,64)` | float16 | 80.44 | 40 | compute caused |
| 19 | single-token | `(1,1,1,16)` | float32 | 5.88 | 1 | pipeline caused |

**本地最重**

- case17 `official-ex2-deepseek` `(2,512,128,64)` → **325.0 μs**
- case18 `long-seq` `(1,4096,1,64)` → **80.4 μs**
- case10 `mid-fp16-h32` `(2,128,32,64)` → **28.5 μs**
- case08 `mid-fp16-wide-pos` `(2,64,8,64)` → **8.8 μs**
- case09 `mid-fp32-wide-pos` `(2,64,8,64)` → **8.8 μs**

- 小 shape：**pipeline bound**（MTE 利用率低、启动开销）
- DeepSeek/长序列：**compute bound**（case17 ≈ 325 μs）
- 与官方「秒级」对比：优先查 **host 重排 / 多次 launch**，不要只盯着 vector 峰值

## 4. 样例 vs 官方

| | 官方 15 | 本地 20 |
|--|--------|--------|
| 精度 | fp32 1e-4 / fp16 1e-3 | 相同且 20/20 Pass |
| 题面示例 | (1,8,1,16)/(2,512,128,64)/(2,16,3,24) | case0/17/3 |
| 非对齐 | 要 | case3/4/5 |
| 秒级毒 case | 有（case9 类） | **未复现** |

## 5. 进步空间（优先级）

### P0 — 2.5×–3.3× 官方 case → 1.2×–1.5×（名次跃迁）

对象：**case5/6/7/8/10/11**

- 是否走 host reorder？拆 host vs device 计时
- 减 per-row 同步、增大 tile、减 launch
- 验收：这些 case 估分 28→50+，总分有望 **43→55–60**

### P1 — case9 全场毒瘤（3.9s–11s）

- 先反推 shape / 代码路径
- 收到接近 3.9s 可再涨一截；全场都慢说明业务极重或实现共性瓶颈

### P2 — 大 shape compute（本地 case17/18/10）

- 双缓冲（小步+精度门禁）
- Trig 合并、减 cast、核间负载

### P3 — 小 shape pipeline

- 收益有限，保持正确即可

## 6. 下一步

1. Awaycsdn 提交基线 → 自己的 15×ms
2. 最差 3 case：host/device 拆分
3. 单点优化再提交，用 t/T 表验收
4. 把官方毒 case 镜像进 `cases_spec`

## 7. 本机文件

```text
E:\Learn\CANN\rotaryposemb\perf_onboard\
  perf_onboard_complete.tgz
  data\perf_onboard\case*\OPPROF_*\*.csv
  SUMMARY.md / summary.csv / ANALYSIS.md
```
