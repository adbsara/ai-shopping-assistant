# Benchmarks

## Query embedding latency (bge-m3)

| Environment | Hardware | Run 1 | Run 2 | Run 3 | Average |
|---|---|---|---|---|---|
| Local | CPU |  |  |  |  |
| Hugging Face | ZeroGPU |  |  |  |  |

## Vector search latency (Qdrant Cloud)

| Environment | Run 1 | Run 2 | Run 3 | Average |
|---|---|---|---|---|
| Local |  |  |  |  |
| Hugging Face |  |  |  |  |

## Indexing
- 1000 products on local CPU: ~15 minutes (~0.9 s per product)

### Observations
- ZeroGPU was not faster than local CPU for single-query embedding
  (190 ms vs 151 ms). For one short sentence, GPU allocation and data
  transfer overhead dominate the actual compute time.
- GPU is expected to win for batch workloads (e.g., indexing 1000 products).
- Vector search was faster from Hugging Face (160 ms vs 311 ms), because
  most of the latency is network round-trip to Qdrant Cloud.