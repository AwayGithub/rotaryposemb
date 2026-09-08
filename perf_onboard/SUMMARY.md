# RotaryPosEmb 上板分析汇总

- 工具: `msprof op`
- metrics: `Default,Roofline,PipeUtilization,Memory,MemoryL0,MemoryUB,L2Cache,ResourceConflictRatio,ArithmeticUtilization,BasicInfo`
- warm-up=2, launch-count=3
- 输出目录: `/mnt/workspace/rotaryposemb/perf_onboard`

| case | tag | dtype | shape | task_us | block | rc | wall_s | roofline |
|------|-----|-------|-------|---------|-------|----|--------|----------|
| 00 | official-ex1-fp32 | float32 | `(1,8,1,16)` | 3.100000 | 8 | 0 | 6.008 | latency bound:pipeline caused | 2026-09- |
| 01 | small-fp32 | float32 | `(1,7,1,16)` | 2.920000 | 7 | 0 | 6.007 | latency bound:pipeline caused | 2026-09- |
| 02 | small-fp16 | float16 | `(1,8,1,16)` | 3.000000 | 8 | 0 | 5.788 | latency bound:pipeline caused | 2026-09- |
| 03 | official-ex3-dim24 | float16 | `(2,16,3,24)` | 4.140000 | 32 | 0 | 4.965 | latency bound:pipeline caused | 2026-09- |
| 04 | unaligned-dim18 | float16 | `(2,7,5,18)` | 3.660000 | 14 | 0 | 4.965 | latency bound:pipeline caused | 2026-09- |
| 05 | unaligned-dim20-fp32 | float32 | `(3,11,2,20)` | 3.960000 | 33 | 0 | 4.985 | latency bound:pipeline caused | 2026-09- |
| 06 | pos0-fp16 | float16 | `(1,9,2,16)` | 3.200000 | 9 | 0 | 3.922 | latency bound:pipeline caused | 2026-09- |
| 07 | pos0-fp32 | float32 | `(1,9,2,16)` | 3.080000 | 9 | 0 | 5.987 | latency bound:pipeline caused | 2026-09- |
| 08 | mid-fp16-wide-pos | float16 | `(2,64,8,64)` | 8.840000 | 40 | 0 | 5.987 | latency bound:compute caused | 2026-09-0 |
| 09 | mid-fp32-wide-pos | float32 | `(2,64,8,64)` | 8.780000 | 40 | 0 | 5.989 | latency bound:compute caused | 2026-09-0 |
| 10 | mid-fp16-h32 | float16 | `(2,128,32,64)` | 28.459999 | 40 | 0 | 5.987 | latency bound:compute caused | 2026-09-0 |
| 11 | theta-1 | float32 | `(1,64,4,32)` | 5.220000 | 40 | 0 | 5.987 | latency bound:compute caused | 2026-09-0 |
| 12 | theta-1e3 | float16 | `(2,32,8,64)` | 5.940000 | 40 | 0 | 6.007 | latency bound:compute caused | 2026-09-0 |
| 13 | theta-5e5 | float16 | `(2,32,8,64)` | 6.080000 | 40 | 0 | 5.987 | latency bound:compute caused | 2026-09-0 |
| 14 | theta-3.5e4 | float16 | `(2,32,8,64)` | 6.140000 | 40 | 0 | 5.987 | latency bound:compute caused | 2026-09-0 |
| 15 | dim256-fp32 | float32 | `(1,128,4,256)` | 7.580000 | 40 | 0 | 6.007 | latency bound:compute caused | 2026-09-0 |
| 16 | dim1024-edge | float32 | `(1,16,1,1024)` | 3.560000 | 16 | 0 | 5.987 | latency bound:pipeline caused | 2026-09- |
| 17 | official-ex2-deepseek | float16 | `(2,512,128,64)` | 325.019989 | 40 | 0 | 5.987 | latency bound:compute caused | 2026-09-0 |
| 18 | long-seq | float16 | `(1,4096,1,64)` | 80.440002 | 40 | 0 | 6.007 | latency bound:compute caused | 2026-09-0 |
| 19 | single-token | float32 | `(1,1,1,16)` | 5.880000 | 1 | 0 | 5.787 | latency bound:pipeline caused | 2026-09- |

## 性能提示（摘自 log）

### case00 official-ex1-fp32
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:10:54 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 3.100000

### case01 small-fp32
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:00 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 2.920000

### case02 small-fp16
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:06 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 3.000000

### case03 official-ex3-dim24
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:11 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_reorder_kernel | Op Type: vector | Task Duration(us): 4.140000

### case04 unaligned-dim18
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:16 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_reorder_kernel | Op Type: vector | Task Duration(us): 3.660000

### case05 unaligned-dim20-fp32
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:21 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_reorder_kernel | Op Type: vector | Task Duration(us): 3.960000

### case06 pos0-fp16
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:25 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 3.200000

### case07 pos0-fp32
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:11:31 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 3.080000

### case08 mid-fp16-wide-pos
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:11:37 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case09 mid-fp32-wide-pos
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:11:43 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case10 mid-fp16-h32
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:11:49 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case11 theta-1
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:11:55 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case12 theta-1e3
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:12:01 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case13 theta-5e5
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:12:07 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case14 theta-3.5e4
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:12:13 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case15 dim256-fp32
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:12:19 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

### case16 dim1024-edge
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:12:25 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 3.560000

### case17 official-ex2-deepseek
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:12:31 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 325.019989

### case18 long-seq
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 2026-09-08 22:12:37 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector | Task Duration(us): 80.440002

### case19 single-token
- 1) MTE2 bandwidth utilization lower than 80% when active. | 2) MTE3 bandwidth utilization lower than 80% when active. | 3) aivector compute usage lower than 20%. | 2026-09-08 22:12:43 [INFO]  Operator Basic Information: | Op Name: rotary_pos_emb_vec_kernel | Op Type: vector

## 原始产物
每个 case 目录下有 `OPPROF_*/`：
- OpBasicInfo.csv / PipeUtilization.csv / Memory*.csv / L2Cache.csv / ...
- dump/ 原始采集
