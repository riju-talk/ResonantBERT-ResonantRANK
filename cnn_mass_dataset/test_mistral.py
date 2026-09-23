from mistralai import Mistral
import os

client = Mistral(api_key=os.environ["Bearer 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"])

response = client.chat.complete(
    model="mistral-medium-3.5",
    messages=[
        {
            "role": "user",
            "content": "Reply with only the word OK."
        }
    ],
)

print(response.choices[0].message.content)