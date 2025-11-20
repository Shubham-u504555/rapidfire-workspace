import os, torch
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

MODEL_ID = os.getenv("MODEL_ID", "runs/sft_ckpt")
PORT = int(os.getenv("PORT", "5005"))
DTYPE = torch.float16 if os.getenv("DTYPE","fp16").lower() in {"fp16","float16"} else torch.bfloat16

# Optional quant at serve time: SERVE_BITS=4|8|none
SERVE_BITS = os.getenv("SERVE_BITS", "none").lower()
quant_cfg = None
if SERVE_BITS == "4":
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )
elif SERVE_BITS == "8":
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

app = FastAPI(title="RF Transformers Server", version="0.3.0")

def load_base(base_id: str):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=DTYPE,
        quantization_config=quant_cfg,  # <- removes deprecated flags
    )
    return tok, model

adapter_cfg = Path(MODEL_ID) / "adapter_config.json"
if adapter_cfg.exists():
    peft_cfg = PeftConfig.from_pretrained(MODEL_ID)
    base_id = peft_cfg.base_model_name_or_path
    tok, base = load_base(base_id)
    model = PeftModel.from_pretrained(base, MODEL_ID)
else:
    tok, model = load_base(MODEL_ID)

class GenIn(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "serve_bits": SERVE_BITS}

@app.post("/generate")
def generate(i: GenIn):
    inputs = tok(i.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=i.max_new_tokens,
            do_sample=True,
            temperature=i.temperature,
            top_p=i.top_p,
            pad_token_id=tok.eos_token_id,
        )
    return {"text": tok.decode(out[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_transformers:app", host="0.0.0.0", port=PORT, log_level="info")

