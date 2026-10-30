"""LoRA fine-tuning on an NVIDIA GPU (Google Colab, RunPod, ...) with PyTorch + PEFT.

    python -m svgen.train_cuda --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out runs/svgen-1.5b

Same data and chat format as the MLX recipe (data/processed/*.jsonl). Built for
flaky machines: it saves a checkpoint every --save-every steps and resumes from
the latest one when you run the same command again. When training finishes it
writes a merged full model to <out>/merged, which MLX on a Mac loads directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device() -> tuple[str, torch.dtype, bool]:
    """Return (device, compute dtype, use grad scaler)."""
    if torch.cuda.is_available():
        # Native bf16 needs compute capability 8.0+ (A100, L4, H100). Recent PyTorch reports
        # bf16 as "supported" on a T4 too, but only through slow emulation.
        if torch.cuda.get_device_capability()[0] >= 8:
            return "cuda", torch.bfloat16, False
        return "cuda", torch.float16, True  # T4, V100
    if torch.backends.mps.is_available():
        return "mps", torch.float32, False  # local smoke tests only
    return "cpu", torch.float32, False


def encode_rows(path: Path, tokenizer, max_len: int) -> list[tuple[list[int], int]]:
    """Tokenize chat rows. Returns (token ids, prompt length) so the loss skips the prompt."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        messages = json.loads(line)["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template prompt is not a prefix of the full conversation")
        if len(full_ids) <= max_len:
            rows.append((full_ids, len(prompt_ids)))
    return rows


def length_batches(rows, batch_size: int, seed: int) -> list[list[int]]:
    """Group similar lengths together (less padding), then shuffle the batches."""
    rng = random.Random(seed)
    order = sorted(range(len(rows)), key=lambda k: (len(rows[k][0]), rng.random()))
    batches = [order[k : k + batch_size] for k in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(rows, indices, pad_id: int, device: str):
    width = max(len(rows[k][0]) for k in indices)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    labels = torch.full((len(indices), width), -100, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, k in enumerate(indices):
        tokens, prompt_len = rows[k]
        ids[row, : len(tokens)] = torch.tensor(tokens)
        labels[row, prompt_len : len(tokens)] = torch.tensor(tokens[prompt_len:])
        mask[row, : len(tokens)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, rows, batch_size, pad_id, device, dtype) -> float:
    model.eval()
    total_loss = total_tokens = 0.0
    for indices in length_batches(rows, batch_size, seed=0):
        ids, labels, mask = collate(rows, indices, pad_id, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = (labels[:, 1:] != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def save_checkpoint(model, optimizer, scaler, step: int, out: Path) -> None:
    ckpt = out / "checkpoint"
    tmp = out / "checkpoint.tmp"
    model.save_pretrained(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None, "step": step}, tmp / "state.pt")
    if ckpt.exists():
        import shutil

        shutil.rmtree(ckpt)
    tmp.rename(ckpt)  # atomic: a disconnect mid-save never leaves a broken checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("runs/svgen-1.5b"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-floor", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--save-every", type=int, default=200, help="optimizer steps between checkpoints")
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--max-steps", type=int, help="stop early (for smoke tests)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype, use_scaler = pick_device()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} dtype={dtype} model={args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_rows = encode_rows(args.data / "train.jsonl", tokenizer, args.max_len)
    valid_rows = encode_rows(args.data / "valid.jsonl", tokenizer, args.max_len)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size / args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"train rows={len(train_rows)} valid rows={len(valid_rows)} optimizer steps={total_steps}", flush=True)

    # On fp16-only GPUs (T4) keep weights in fp32 and let autocast run the matmuls in fp16:
    # Qwen activations can overflow when the whole model is stored in fp16.
    load_dtype = torch.float32 if dtype == torch.float16 else dtype
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    base.config.use_cache = False

    ckpt = args.out / "checkpoint"
    if ckpt.exists():
        model = PeftModel.from_pretrained(base, ckpt, is_trainable=True)
        state = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
        start_step = state["step"]
        print(f"resuming from step {start_step}", flush=True)
    else:
        config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(base, config)
        state, start_step = None, 0
    # Keep trainable LoRA weights in fp32 for stable fp16/bf16 mixed precision.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    if state:
        optimizer.load_state_dict(state["optimizer"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])

    log = (args.out / "train_log.jsonl").open("a", encoding="utf-8")
    model.train()
    step, micro = start_step, 0
    started, tokens_seen, running = time.time(), 0, []
    epoch = start_step * args.grad_accum // max(1, math.ceil(len(train_rows) / args.batch_size))
    while step < total_steps:
        batches = length_batches(train_rows, args.batch_size, seed=args.seed + epoch)
        # On resume, skip the micro-batches this epoch already used.
        skip = start_step * args.grad_accum - epoch * len(batches) if step == start_step else 0
        for indices in batches[max(0, skip):]:
            ids, labels, mask = collate(train_rows, indices, pad_id, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / args.grad_accum
            if not torch.isfinite(loss):
                print(f"warning: non-finite loss at step {step}, batch skipped", flush=True)
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue
            (scaler.scale(loss) if scaler else loss).backward()
            running.append(loss.item() * args.grad_accum)
            tokens_seen += int(mask.sum())
            micro += 1
            if micro % args.grad_accum:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, args.warmup, args.lr_floor)
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 25 == 0 or step == total_steps:
                elapsed = time.time() - started
                entry = {"step": step, "loss": round(sum(running) / len(running), 4), "lr": optimizer.param_groups[0]["lr"],
                         "tok_per_s": round(tokens_seen / elapsed), "eta_min": round((total_steps - step) * elapsed / max(1, step - start_step) / 60, 1)}
                running = []
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(model, valid_rows, args.batch_size, pad_id, device, dtype)
                print(json.dumps({"step": step, "val_loss": round(val, 4)}), flush=True)
                log.write(json.dumps({"step": step, "val_loss": round(val, 4)}) + "\n")
                log.flush()
            if step % args.save_every == 0 or step == total_steps:
                save_checkpoint(model, optimizer, scaler, step, args.out)
            if step >= total_steps:
                break
        epoch += 1

    model.save_pretrained(args.out / "adapter")
    tokenizer.save_pretrained(args.out / "adapter")
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged = merged.to(torch.bfloat16)  # bf16 never overflows; MLX loads it natively
        merged.config.use_cache = True
        merged.save_pretrained(args.out / "merged", safe_serialization=True)
        tokenizer.save_pretrained(args.out / "merged")
        print(f"merged model written to {args.out / 'merged'}", flush=True)


if __name__ == "__main__":
    main()
