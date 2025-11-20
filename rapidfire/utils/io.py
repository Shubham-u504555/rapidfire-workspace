from __future__ import annotations
import os, json, glob, pathlib
from typing import Iterable, Dict, Any, List

def ensure_dir(p: os.PathLike | str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def save_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]):
    ensure_dir(pathlib.Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def read_text_files(folder: os.PathLike | str, patterns=("*.txt","*.md")):
    folder = pathlib.Path(folder)
    files = []
    for pat in patterns: files.extend(glob.glob(str(folder / pat)))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                if text: yield pathlib.Path(fp).name, text
        except Exception:
            continue
