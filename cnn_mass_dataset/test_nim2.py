import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                json={
                    "model": "mistralai/mixtral-8x7b-instruct-v0.1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10
                },
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json",
                },
                timeout=60.0
            )
            print("Status Mixtral 8x7b:", resp.status_code)
            
            resp2 = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                json={
                    "model": "mistralai/mistral-large-2-instruct",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10
                },
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json",
                },
                timeout=60.0
            )
            print("Status Mistral Large 2:", resp2.status_code)
        except Exception as e:
            print("Exception:", type(e).__name__, str(e))

asyncio.run(main())
