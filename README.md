# RotaryPosEmb (Ascend C)

Ascend C Kernel 直调实现的 RoPE（Rotary Position Embedding）。

当前版本功能正确、可本地编译运行；性能仍有优化空间（流水 / 双缓冲等）。

## 功能

- 输入：`x [B, S, H, D]`（fp16 / fp32）、`positions [B, S]`（int32）、`theta`
- 输出：`y [B, S, H, D]`（与输入同 dtype）
- 两条路径：
  - **向量路径**：`dim % 16 == 0`，TPipe + TQue
  - **Host 重排 + 向量路径**：非 16 对齐 dim，half-slot 对齐后全向量处理

## 目录

```
rotaryposemb/
├── kernel.asc          # Kernel + host run_kernel（核心提交文件）
├── main.asc            # 本地通用测试主程序
├── data_utils.h        # 读写 bin 工具
├── CMakeLists.txt
├── run.sh              # 一键编译 + 小例验证
├── test_matrix.py      # 多 shape / dtype 精度矩阵
└── scripts/
    ├── gen_data.py
    ├── RotaryPosEmb.py # golden 参考
    └── verify_result.py
```

## 环境

- Ascend NPU（默认 `dav-2201` / Ascend 910）
- CANN Toolkit（需设置 `ASCEND_HOME_PATH`）
- CMake ≥ 3.16、Python 3 + numpy

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
```

## 快速运行

```bash
bash run.sh
```

自定义单用例：

```bash
./build/rotary_pos_emb_custom <dtype 0/1> <B> <S> <H> <D> <theta> <x.bin> <p.bin> <y.out> [coreNum]
```

全矩阵精度测试：

```bash
python3 test_matrix.py
```

## 实现要点

1. 无设备端向量 Sin/Cos，用多项式 + 分段 2π 归约在 UB 上计算
2. invf 表在 host 用 `powf` 生成，与 numpy golden 对齐
3. DMA 严格 32B 对齐；非对齐 dim 走 host 重排
4. 同步优先依赖 TQue，避免不可靠的手动 SetFlag/WaitFlag

## 后续优化方向

- 全局 tile 双缓冲 / 流水（此前尝试破坏正确性，需结合 TQue depth 语义重做）
- 减少 host 重排开销
- 多核调度与 UB tile 参数调优
- 性能 profiling（msprof）与 bound 分析

## License

仅供学习与竞赛参考使用。
