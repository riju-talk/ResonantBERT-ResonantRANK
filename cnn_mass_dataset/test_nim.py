import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                json={
                    "model": "mistralai/mistral-medium-3.5-128b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10
                },
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json",
                },
                timeout=60.0
            )
            print("Status:", resp.status_code)
            print("Body:", resp.text)
        except Exception as e:
            print("Exception:", type(e).__name__, str(e))

asyncio.run(main())
