import pathlib
from rapidfire.utils import read_text_files, save_jsonl, ensure_dir

BASE = pathlib.Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "corpora"
TRAIN  = BASE / "data" / "raw" / "sft_train.jsonl"
EVAL   = BASE / "data" / "raw" / "sft_eval.jsonl"

def to_examples():
    items = list(read_text_files(CORPUS, ("*.txt","*.md")))
    if not items:
        items = [("hello.txt","Hello there!"), ("bye.txt","Goodbye!")]
    for name, text in items:
        yield {"instruction": f"Summarize {name}", "input": text[:6000], "output": "Summary: (model will learn)"}

def main():
    ex = list(to_examples()); k = max(1, int(0.8 * len(ex)))
    ensure_dir(TRAIN.parent); save_jsonl(TRAIN, ex[:k]); save_jsonl(EVAL, ex[k:])
    print(f"Wrote {TRAIN} and {EVAL} ({len(ex)} rows).")

if __name__ == "__main__": main()
