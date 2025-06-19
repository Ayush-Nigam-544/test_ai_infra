# LLM Load Testing Report: Qwen2.5-3B on BentoML

## 1. Objective
Evaluate the throughput, latency, and scaling behavior of the deployed Qwen2.5-3B LLM service on BentoML by simulating concurrent client requests and measuring:
- First token latency
- Full response time
- Success/failure rates
- System behavior under and over max concurrency

---

## 2. Test Script Overview

We used an async Python script with `httpx` to send concurrent POST requests to the `/generate` endpoint. The script measures:
- Time to first token (when streaming starts)
- Time to full response (when all tokens are received)
- HTTP status and errors

**Note:** All tests were performed with a single replica (`max_replicas = 1`).

**Key code excerpt:**
```python
import asyncio, httpx, time

ENDPOINT = "https://qwen253b-ebc82948.mt-guc1.bentoml.ai/generate"

async def send_request(i, results):
    async with httpx.AsyncClient(timeout=60) as client:
        start = time.perf_counter()
        first_token_time = None
        try:
            async with client.stream(
                "POST", ENDPOINT,
                json={
                    "max_tokens": 128,
                    "prompt": f"Hello from client {i}",
                    "show_reasoning": True
                }
            ) as response:
                async for chunk in response.aiter_bytes():
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start
            elapsed = time.perf_counter() - start
            results.append((i, response.status_code, first_token_time, elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((i, "error", first_token_time, elapsed, str(e)))
```

---

## 3. Parameters Affecting Throughput & Latency

| Parameter                | Description                                                                 | Typical Value / Setting      |
|--------------------------|-----------------------------------------------------------------------------|-----------------------------|
| **max_num_seqs**         | Max concurrent sequences vLLM will process in parallel                      | 256                         |
| **tensor_parallel_size** | Number of GPUs used for tensor parallelism                                  | 1                           |
| **max_tokens**           | Max tokens generated per request                                            | 128 (test), 1024 (prod)     |
| **Prompt length**        | Longer prompts increase compute/memory needs                                | Short (test)                |
| **Batch size**           | Number of concurrent requests sent in a test run                            | 64, 128, 256, 300, 400      |
| **Hardware**             | GPU type and count                                                          | 1x NVIDIA A100              |
| **API endpoint**         | Path and protocol used                                                      | `/generate` (POST, JSON)    |

---

## 4. Results & Observations

### Latest Results
```
=== Testing with 64 concurrent requests ===
Total: 64, Success: 64, Fail: 0
First token - Min: 1.21s, Max: 19.28s, Avg: 10.95s
Full response - Min: 4.65s, Max: 25.53s, Avg: 15.95s

=== Testing with 128 concurrent requests ===
Total: 128, Success: 128, Fail: 0
First token - Min: 2.88s, Max: 37.49s, Avg: 20.37s
Full response - Min: 4.20s, Max: 46.26s, Avg: 27.25s

=== Testing with 256 concurrent requests ===
Total: 256, Success: 256, Fail: 0
First token - Min: 8.99s, Max: 75.64s, Avg: 39.61s
Full response - Min: 12.45s, Max: 93.93s, Avg: 51.77s

=== Testing with 300 concurrent requests ===
Total: 300, Success: 300, Fail: 0
First token - Min: 9.93s, Max: 85.31s, Avg: 44.44s
Full response - Min: 15.08s, Max: 100.75s, Avg: 58.00s

=== Testing with 400 concurrent requests ===
Total: 400, Success: 400, Fail: 0
First token - Min: 7.56s, Max: 101.12s, Avg: 61.22s
Full response - Min: 26.08s, Max: 126.03s, Avg: 76.54s
```

### Additional Results: Low Concurrency Batches (single replica)

| Concurrency | Success | Fail | First Token Min (s) | First Token Max (s) | First Token Avg (s) | Full Response Min (s) | Full Response Max (s) | Full Response Avg (s) |
|:-----------:|:-------:|:----:|:------------------:|:------------------:|:------------------:|:---------------------:|:---------------------:|:---------------------:|
|      1      |    1    |  0   |        0.92        |        0.92        |        0.92        |         1.06          |         1.06          |         1.06          |
|      3      |    3    |  0   |        0.93        |        1.32        |        1.13        |         2.96          |         5.08          |         4.31          |
|      5      |    5    |  0   |        0.97        |        1.73        |        1.34        |         2.32          |         5.55          |         3.83          |
|      9      |    9    |  0   |        1.04        |        2.53        |        1.76        |         1.84          |         6.26          |         4.17          |
|     12      |   12    |  0   |        1.09        |        3.12        |        2.30        |         2.19          |         6.89          |         4.90          |
|     15      |   15    |  0   |        1.11        |        4.38        |        2.58        |         4.25          |         8.37          |         6.13          |
|     20      |   20    |  0   |        1.09        |        4.82        |        2.94        |         2.31          |         8.66          |         6.08          |

### Additional Results: Low Concurrency Batches (latest, 2 replicas)

| Concurrency | Success | Fail | First Token Min (s) | First Token Max (s) | First Token Avg (s) | Full Response Min (s) | Full Response Max (s) | Full Response Avg (s) |
|:-----------:|:-------:|:----:|:------------------:|:------------------:|:------------------:|:---------------------:|:---------------------:|:---------------------:|
|      1      |    1    |  0   |        0.96        |        0.96        |        0.96        |         4.92          |         4.92          |         4.92          |
|      3      |    3    |  0   |        2.19        |        2.57        |        2.38        |         2.88          |         6.01          |         4.64          |
|      5      |    5    |  0   |        0.93        |        1.70        |        1.39        |         1.71          |         5.55          |         3.88          |
|      9      |    9    |  0   |        1.07        |        2.52        |        1.81        |         1.96          |         6.41          |         4.69          |
|     12      |   12    |  0   |        0.93        |        3.10        |        2.03        |         3.21          |         7.18          |         5.02          |
|     15      |   15    |  0   |        0.93        |        3.74        |        2.36        |         2.40          |         7.59          |         5.07          |
|     20      |   20    |  0   |        1.04        |        4.70        |        2.91        |         1.28          |         9.43          |         5.72          |

---


### Interpretation
- **No errors up to 400 requests:** The service queues requests above `max_num_seqs` (256), but processes all eventually.
- **Latency increases with concurrency:** As requests exceed 256, they wait for earlier requests to finish, causing higher response times.
- **First token and full response times diverge at high concurrency:** Indicates queueing and batch processing effects.

---

## 5. Practical Guidance

### Throughput & Latency Control
- **Keep concurrency ≤ max_num_seqs (256)** for best latency.
- **Above 256:** Requests are queued, not processed in parallel—expect linear latency growth.
- **Monitor GPU utilization:** If latency is high even below 256, check for GPU saturation or memory limits.
- **Scale horizontally:** Add more replicas to handle higher concurrency with lower latency.

### Why does first token latency increase with concurrency?
- When you send up to `max_num_seqs` concurrent requests, all are processed in a single batch by the model, so first token latency is determined by the batch processing time.
- When you send more than `max_num_seqs` requests, the extra requests are queued and must wait for the first batch to finish before they are processed in the next batch.
- **This means:**
  - The first `max_num_seqs` requests experience the normal first token latency.
  - Requests above this limit wait for the previous batch to finish, so their first token latency includes both the "wait time and the batch processing time."
  - As concurrency increases, more requests are queued, and the average first token latency rises because more requests are waiting for earlier batches to complete.
- **In summary:**
  - The first batch's first token latency stays about the same, but the average and max first token latency increase as more requests are queued and must wait for their turn to be processed.

### How to Tune
- **Increase `max_num_seqs`** if you have more GPU memory and want higher parallelism.
- **Decrease `max_tokens`** or prompt length for faster responses.
- **Use batching** if your API and model support it, to maximize GPU efficiency.

---

## 6. Example: How to Run the Test

```powershell
python load_test.py
```
- The script will run tests at 64, 128, 256, 300, and 400 concurrent requests.
- Review the printed summary for each batch size.

---

## 7. Key Takeaways
- **max_num_seqs** is your main concurrency control lever.
- **Latency grows rapidly** when you exceed this limit.
- **No errors ≠ good UX:** High latency can be as bad as errors for users.
- **Scale out** for production workloads needing high concurrency and low latency.

---

*Prepared for: Qwen2.5-3B LLM on BentoML Cloud*
*Date: 2025-06-18*
