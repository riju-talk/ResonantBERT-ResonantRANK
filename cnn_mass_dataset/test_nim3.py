import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        for model in ["nv-mistralai/mistral-nemo-12b-instruct", "mistralai/mistral-large"]:
            try:
                resp = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10
                    },
                    headers={
                        "Authorization": "Bearer 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",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0
                )
                print(f"Status {model}:", resp.status_code)
            except Exception as e:
                print(f"Exception {model}:", type(e).__name__, str(e))

asyncio.run(main())
