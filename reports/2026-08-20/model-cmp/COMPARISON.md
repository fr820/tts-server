# qwen3 1.7B vs 0.6B × 并发 — A10 实测对比（2026-08-20）

同一协议：fast_subtalker 开启（仓库默认路径）、bf16、texts_main.jsonl（35–60 词，
音频 ~15–18s/条）、每格 24 请求（0.6b c1 为上午 30 请求批次）、每格独立服务端
（/metrics 干净）、3 次 warmup 不计入、排除模型加载。client 口径 = bench_http
逐请求（含 HTTP，保守）；srvRTFmean = /metrics `tts_rtf` 直方图均值（含排队）。

## 总表

| 模型 | c | 失败 | rps | RTF p50 | **RTF p90** | TTFA p50 | TTFA p90 | srvRTFmean | 设备显存峰值* |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B | 1 | 0 | 0.11 | 0.57 | **0.58** | 8.4s | 9.8s | 0.57 | 4.2GB |
| 0.6B | 2 | 0 | 0.12 | 1.14 | **1.24** | 16.4s | 17.6s | 1.06 | 5.8GB |
| 0.6B | 4 | 0 | 0.12 | 2.27 | **2.58** | 33.7s | 35.1s | 2.00 | 5.8GB |
| 1.7B | 1 | 0 | 0.11 | 0.57 | **0.58** | 9.2s | 11.3s | 0.58 | 5.8GB |
| 1.7B | 2 | 0 | 0.11 | 1.18 | **1.34** | 18.1s | 20.2s | 1.09 | 5.8GB |
| 1.7B | 4 | 0 | 0.11 | 2.36 | **2.58** | 33.8s | 36.7s | 2.04 | 5.3GB |

*设备显存峰值 = nvidia-smi 全设备口径（含 CUDA context、CUDA Graph 私有池、
caching allocator 高水位），非 torch 活跃分配；torch 口径：0.6B ~2.2GB /
1.7B ~4.2GB。24GB A10 上均余量充足。

## 结论

1. **c=1（单流）两模型几乎同速**：RTF p90 均 0.58（p50 均 0.57）。fast 路径把
   sub-talker 图化后，剩余成本是外层 talker 的每步宿主开销（~32ms/帧）——外层
   单 token 前向同样是 host-bound，A10 显存带宽不是瓶颈，1.7B 只贵在 TTFA p90
   +15%（11.3s vs 9.8s）。与上午的发现一致：本栈瓶颈是宿主调度，不是算力/模型大小。
2. **并发被 `_infer_lock` 完全串行化**：吞吐恒定 ~0.11–0.12 req/s（与模型、c
   无关）；单请求观察到的 RTF ≈ c × 单流 RTF（排队时间计入请求耗时），TTFA
   精确 ×c。**c ≥ 2 即无实时性可言**（RTF p90 1.24 起）。
3. **失败率全部为 0**（含 c=4；对比 2026-08-01 stock 路径 c=4 报 26% 空体失败
   ——串行锁 + 更短的单请求耗时让队列远低于超时线）。
4. **默认维持 0.6B**：速度几乎免费拿到，torch 显存减半（并发/批处理留余量）；
   1.7B 的独有价值是 `instruct` 风格控制（能力位按模型动态置位）。

## 并发实时性的出路（未实现，记录方向）

- **真批处理**：`generate_custom_voice` 接受文本列表，talker 内部已做 left-pad
  批量（qwen-tts 代码有完整 batch 路径）。把锁改为"攒批 + 一次 batch 合成"
  可让 c>1 时吞吐随 batch 增长，而不是排队。
- 多 worker 进程 / 多卡：简单但显存 ×N。
- 现状语义：c>1 时按到达顺序排队，每请求仍 ~8.4s（0.6B）合成，适合"少量长请求"
  而非"多路实时流"。

## 产物

- 每格：`{model}/c{c}/http-qwen3-c{c}.json|md`、`bench.stdout`、`metrics.txt`
  （服务端直方图）、`healthz.json`、`gpu.csv`（5s 采样）
- 0.6B c1 复用上午批次：`../http-qwen3-c1.json`（30 请求）
- 环境：`../env.txt`（同日同机）
- 复现：`/tmp/run_matrix.sh` 思路 = 每格 起新服务端 → 3 warmup →
  `bench_http --concurrency c --requests 24 --texts-file benchmarks/data/texts_main.jsonl`
