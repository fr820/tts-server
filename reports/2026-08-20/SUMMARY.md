# 2026-08-20 qwen3 A10 实测 — 0.6B 默认 + fast sub-talker（CUDA Graph）

全部数字为本机（NVIDIA A10 24GB, driver 580.126.09, CUDA 13.0, torch 2.12.1+cu130,
transformers 4.57.3, qwen-tts 0.1.1, bf16）今日实测，产物在同目录。
模型：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`（本次起为仓库默认；1.7B 对照见文末）。

## RTF 口径（与 goal 一致，全文统一）

RTF = 单请求合成耗时 / 生成音频时长。≥3 次 warmup（服务端启动 warmup + 3 次显式
warmup POST，均不计入统计），排除模型加载。两个口径都记录：

- **client**（bench_http 逐请求，含 HTTP 开销，偏保守）：表中主数字。
- **server**（/metrics 的 `tts_rtf` 直方图，pipeline.py 服务端计时）：交叉验证。

## A-B：fast_subtalker off vs on（同一文本序列，c=1，30 请求，0 失败）

| 档 | RTF p50 | RTF p90 | RTF p95 | latency p90 | TTFA p90 | GPU util p50 | VRAM peak |
|---|---|---|---|---|---|---|---|
| stock（qwen-tts 原生嵌套 generate） | 1.46 | 1.47 | 1.48 | 26.3s | 26.3s | 24% | 3921MB |
| **fast（默认，CUDA Graph 化 sub-talker）** | **0.57** | **0.58** | **0.58** | **9.84s** | **9.83s** | 46% | 4151MB |

**门槛：单流 RTF p90 = 0.58 < 1.0 达成**（client 口径，保守）。
服务端口径：33/33 请求（30 基准 + 3 warmup）`tts_rtf` 全部 ≤ 0.75，均值 0.573
——服务端 p90 = max < 0.75，同样达成。TTFA p90 9.83s ≈ 0.58 × 音频时长
（emulated streaming，TTFA ≈ 总时长，见下）。

数据集 `bench-text-v1-en`（texts_main.jsonl，60 条 35–60 词轮换，音频 ~15–18s/条）。
GPU util：nvidia-smi 2s 采样（gpu_monitor/*.csv）。

## 为什么快了 2.5×

qwen-tts 的 talker 每个音频帧（12Hz）在 `forward` 里对 5 层 code predictor 跑一次
完整 HF `generate()`（15 code-group token），每次请求 ~1000 次 `_sample` 迭代。
单步 GPU 计算 <1ms，但宿主端 Python 调度 ~5ms/步 → GPU 饥饿（stock util 仅 24%），
RTF ~1.46 与 talker 大小无关（0.6B 与 1.7B 实测相同）。

`tts_server/backends/qwen3_fast.py` 把嵌套 generate 换成手写 KV-cache 循环，并把
整个 15 步循环（形状完全静态、token 选择为 GPU→GPU 依赖）捕获为单张 CUDA Graph：
嵌套部分 76ms/帧 → ~5ms/帧。采样语义保持（temperature 0.9 / top_k 50 / top_p 1.0，
或 `subtalker_greedy: true` 切贪心）。`backend.options.fast_subtalker: false` 可回退
原路径（即本 A-B 的 stock 档）。

## gpu_smoke（真实音频校验，6/6 PASS）

- speech: 230444B sr=24000 dur=4.8s peak=26751 rms=3687.3（非静音、时长合理）
- latency: warm p50=2946ms p95=3332ms RTF p50=0.580
- gpu_util: util p50=46% peak=54%（stock 档同机 ~29%）| dev mem p50=4151MB

## 1.7B 对照（同机当日，直连 backend，gpu_validate.py，非服务端路径）

- 1.7B stock: RTF p50 1.461, VRAM alloc 4197MB / peak 4373MB（1.7B 不受 fast 补丁影响的部分未单独测）
- 0.6B stock: RTF p50 1.458, VRAM alloc 2175MB / peak 2343MB
- 0.6B fast:  RTF p50 0.591（直连口径；服务端口径见上表）
- 结论：瓶颈是嵌套生成机器开销而非模型算力；0.6B 显存减半、速度相同 → 默认 0.6B。

## 复现

```bash
uv sync --extra qwen3        # torch 2.12.1+cu130, qwen-tts 0.1.1
# 权重走镜像：export HF_ENDPOINT=https://hf-mirror.com
TTS_CONFIG=<fast 或 stock 的 yaml> uv run uvicorn tts_server.main:app --port 8000
# 3 次 warmup POST /v1/audio/speech 后：
uv run python benchmarks/bench_http.py --backend qwen3 --concurrency 1 \
  --requests 30 --texts-file benchmarks/data/texts_main.jsonl \
  --dataset bench-text-v1-en --out-dir <dir>
curl -s localhost:8000/metrics | grep tts_rtf   # 服务端直方图
uv run python scripts/gpu_smoke.py --iters 8    # 真音频 + GPU 采样
```

两个 yaml 见 `/tmp/tts-stock.yaml`、`/tmp/tts-fast.yaml`（区别仅
`backend.options.fast_subtalker: false/true`）。

## 备注

- emulated streaming：TTFA ≈ 总合成时长（全片段成后切片），能力声明保持
  `streaming_mode="emulated"`，未变。
- 0.6B 不支持 instruct → `supports_emotion_or_style_control` 按加载模型动态置
  False（1.7B 路径仍为 True）。
- CUDA Graph 捕获失败时自动回退 eager fast 循环（日志一条 warning），语义不变。

## 模型 × 并发对比（同日补充）

见 `model-cmp/COMPARISON.md`：1.7B 与 0.6B 在 fast 路径下 c=1 同速（RTF p90 均 0.58），并发被 _infer_lock 串行化（吞吐恒定 ~0.11 req/s，c≥2 单请求 RTF≈c×单流）。
