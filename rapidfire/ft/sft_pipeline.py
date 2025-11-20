from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from ..utils import get_logger, load_jsonl, ensure_dir

logger = get_logger("rapidfire.ft")

@dataclass
class SFTArgs:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    train_file: str = "data/raw/sft_train.jsonl"
    eval_file:  str = "data/raw/sft_eval.jsonl"
    output_dir: str = "runs/sft_ckpt"
    max_steps: int = 200
    lr: float = 2e-4
    per_device_train_batch_size: int = 4
    grad_accum: int = 4
    fp16: bool = True
    use_qlora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None  # defaults if None

def _build_datasets(train_path: str, eval_path: str):
    from datasets import Dataset
    def to_text(row):
        instr, inp, out = row.get("instruction",""), row.get("input",""), row.get("output","")
        return {"text": f"Instruction:\n{instr}\n\nInput:\n{inp}\n\nResponse:\n{out}"}
    train = Dataset.from_list([to_text(r) for r in load_jsonl(train_path)])
    evald = Dataset.from_list([to_text(r) for r in load_jsonl(eval_path)]) if Path(eval_path).exists() else None
    return train, evald

def _tokenize(train, evald, tok):
    def tok_fn(ex): return tok(ex["text"], truncation=True, max_length=1024)
    train = train.map(tok_fn, batched=True, remove_columns=["text"])
    if evald is not None:
        evald = evald.map(tok_fn, batched=True, remove_columns=["text"])
    return train, evald

def _default_lora_targets() -> list[str]:
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def _trainer_with_qlora(args: SFTArgs):
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train, evald = _build_datasets(args.train_file, args.eval_file)
    train, evald = _tokenize(train, evald, tok)

    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    logger.info("Loading base model in 4-bit (QLoRA): %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_cfg,   # <- NOT load_in_8bit/4bit flags
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules or _default_lora_targets(),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    ensure_dir(args.output_dir)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_steps=0,
        fp16=args.fp16,
        bf16=not args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tok,   # deprecation warning is OK in 4.56.x
        args=targs,
        train_dataset=train,
        eval_dataset=evald,
        data_collator=collator,
    )
    logger.info("Starting QLoRA SFT training …")
    trainer.train()
    logger.info("Saving adapter to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

def _trainer_full_finetune(args: SFTArgs):
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train, evald = _build_datasets(args.train_file, args.eval_file)
    train, evald = _tokenize(train, evald, tok)
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype=dtype)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    ensure_dir(args.output_dir)
    targs = TrainingArguments(
        output_dir=args.output_dir, learning_rate=args.lr, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum, logging_steps=10, save_steps=0,
        fp16=args.fp16, bf16=not args.fp16, report_to="none",
    )
    trainer = Trainer(model=model, tokenizer=tok, args=targs, train_dataset=train, eval_dataset=evald, data_collator=collator)
    logger.info("Starting FULL fine-tune …")
    trainer.train()
    logger.info("Saving full model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

def run_simple_sft(args: Optional[SFTArgs] = None):
    args = args or SFTArgs()
    if args.use_qlora:
        _trainer_with_qlora(args)
    else:
        _trainer_full_finetune(args)
