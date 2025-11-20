import os, time, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

"""
FastAPI proxy to xAI Grok-2.
Env:
  XAI_API_KEY   = your xAI API key
  XAI_MODEL_ID  = grok-2 (default) or grok-2-mini, etc.
  PORT          = 5006 (default)
"""

XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_MODEL_ID = os.getenv("XAI_MODEL_ID", "grok-2").strip()
PORT = int(os.getenv("PORT", "5006"))

if not XAI_API_KEY:
    raise RuntimeError("Set XAI_API_KEY in the environment.")

# xAI chat/completions-style endpoint (adjust if your account uses a different base URL)
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai")
COMPLETIONS_URL = f"{XAI_BASE_URL}/v1/chat/completions"

app = FastAPI(title="Grok-2 Proxy", version="0.1.0")

class GenIn(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

@app.get("/health")
def health():
    return {"ok": True, "model": XAI_MODEL_ID}

@app.post("/generate")
def generate(i: GenIn):
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": XAI_MODEL_ID,
        "messages": [{"role": "user", "content": i.prompt}],
        "temperature": i.temperature,
        "top_p": i.top_p,
        # xAI usually uses "max_output_tokens" (alias to OpenAI-style "max_tokens" in some SDKs).
        # Using a safe key name here:
        "max_output_tokens": i.max_new_tokens,
        # You can add system prompts, tools, etc. here if needed.
    }

    t0 = time.time()
    try:
        resp = requests.post(COMPLETIONS_URL, headers=headers, json=payload, timeout=120)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"xAI request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    # Extract text in OpenAI-like shape: choices[0].message.content
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected xAI response: {data}")

    return {
        "text": text,
        "latency_s": round(time.time() - t0, 3),
        "provider": "xai",
        "model": XAI_MODEL_ID,
    }

if __name__ == "__main__":
    uvicorn.run("serve_grok2:app", host="0.0.0.0", port=PORT, log_level="info")
