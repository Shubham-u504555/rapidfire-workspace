from pathlib import Path
from rapidfire.ft import run_simple_sft, SFTArgs
BASE = Path(__file__).resolve().parent.parent
TRAIN = BASE / "data" / "raw" / "sft_train.jsonl"
EVAL  = BASE / "data" / "raw" / "sft_eval.jsonl"
if __name__ == "__main__":
    if not TRAIN.exists() or not EVAL.exists():
        raise SystemExit("Run: python scripts/01_prepare_data.py")
    run_simple_sft(SFTArgs(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        train_file=str(TRAIN),
        eval_file=str(EVAL),
        output_dir="runs/sft_ckpt",
        max_steps=60,
        lr=2e-4,
        per_device_train_batch_size=4,
        grad_accum=4,
        fp16=True,
        use_qlora=True,   # <- critical
    ))
