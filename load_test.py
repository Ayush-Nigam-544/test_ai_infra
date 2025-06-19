import asyncio
import httpx
import time

ENDPOINT = "https://qwen253bn-ebc82948.mt-guc1.bentoml.ai/generate "

async def send_request(i, results):
    async with httpx.AsyncClient(timeout=60) as client:
        start = time.perf_counter()
        first_token_time = None
        try:
            async with client.stream(
                "POST",
                ENDPOINT,
                json={
                    "max_tokens": 128,
                    "prompt": f"Hello from client {i}",
                    "show_reasoning": True
                }
            ) as response:
                async for chunk in response.aiter_bytes():
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start
                    # Continue reading to consume the full response
            elapsed = time.perf_counter() - start
            results.append((i, response.status_code, first_token_time, elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((i, "error", first_token_time, elapsed, str(e)))

async def run_load_test(concurrency):
    results = []
    tasks = [send_request(i, results) for i in range(1, concurrency + 1)]
    await asyncio.gather(*tasks)
    return results

def analyze_results(results):
    successes = [r for r in results if r[1] == 200]
    errors = [r for r in results if r[1] != 200]
    print(f"Total: {len(results)}, Success: {len(successes)}, Fail: {len(errors)}")
    if successes:
        first_tokens = [r[2] for r in successes if r[2] is not None]
        times = [r[3] for r in successes]
        print(f"First token - Min: {min(first_tokens):.2f}s, Max: {max(first_tokens):.2f}s, Avg: {sum(first_tokens)/len(first_tokens):.2f}s")
        print(f"Full response - Min: {min(times):.2f}s, Max: {max(times):.2f}s, Avg: {sum(times)/len(times):.2f}s")
    if errors:
        print("Some errors:", errors[:5])  # Show up to 5 errors

if __name__ == "__main__":
    for concurrency in [1, 3, 5, 9, 12, 15, 20]:
        print(f"\n=== Testing with {concurrency} concurrent requests ===")
        results = asyncio.run(run_load_test(concurrency))
        analyze_results(results)