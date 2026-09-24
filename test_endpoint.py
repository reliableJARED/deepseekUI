"""Quick smoke test for the private DeepSeek endpoint (OpenAI chat completions).

Run:  python test_endpoint.py
"""

import os

from openai import OpenAI

BASE_URL = "http://205.196.144.74:11036/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "not-needed")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)

# 1. Which models does it advertise?
models = [m.id for m in client.models.list().data]
print(f"models: {models}")

model = os.environ.get("MODEL", models[0] if models else "DeepSeek-V4-Flash-DSpark")

# 2. One chat completion.
reply = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are terse  content generator."},
        {"role": "user", "content": "how much could a woodchuck chuck if a woodchuck could chuck wood."},
    ],
    max_tokens=100,
    temperature=0.2,
)

print(f"model used: {reply.model}")
print(f"finish_reason: {reply.choices[0].finish_reason}")
print(f"usage: {reply.usage}")
print("---")
print(reply.choices[0].message.content)
