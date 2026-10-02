"""Adapt the upstream RLCD loop to Korean choice data and CUDA."""
import argparse
import json
import random
import os
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("USE_TF", "0")

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ({hours * 3600 + minutes * 60 + seconds:.2f}초)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "korean-support",
                        help="Directory for the trained model, checkpoints and preprocessed items")
    args = parser.parse_args()

    if min(args.epochs, args.micro_batch, args.grad_accum) < 1:
        parser.error("epochs, micro-batch and grad-accum must be positive")
        
    total_started = time.perf_counter()

    import torch
    from transformers import AutoTokenizer
    from reference import official_finetune as upstream

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install a CUDA PyTorch build or pass --device cpu.")
    
    torch.manual_seed(42)
    random.seed(42)
    upstream.MODEL_ID = "convaiinnovations/laya-multilingual"

    model_dir = upstream.prepare_model(str(ROOT / "models" / "base"))
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg.update(max_len=1024, head_max_len=256)
    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))["category"]
    labels = list(schema["criteria"])
    items = []
    texts = set()

    for line in args.data.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)

        if row["label"] not in labels or not row["text"].strip():
            raise ValueError(f"Invalid training row: {row['id']}")
        if row["text"] in texts:
            raise ValueError("Duplicate training text")
        
        texts.add(row["text"])
        gold = {"probabilities": {label: float(label == row["label"]) for label in labels}}
        item = upstream.build_training_item(tokenizer, cfg, row["text"], schema, gold)

        if item is None:
            raise ValueError(f"Option markers lost for {row['id']}")
        print(item)
        items.append(item)

    if len(items) < 10:
        raise ValueError("Need at least 10 training examples")
    
    for line in (ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines():
        if json.loads(line)["text"] in texts:
            raise ValueError("Training and test text overlap")
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "train_items.pt"

    torch.save(items, cache)

    print("Synthetic training data: probabilities require validation on real held-out inquiries.", flush=True)

    if len(items) // 10 < 10:
        print("Tiny demo: calibration set is too small to fit a temperature.", flush=True)

    options = SimpleNamespace(epochs=args.epochs, micro_batch=args.micro_batch,
                              grad_accum=args.grad_accum, calib_max=400,
                              no_checkpointing=False, output_dir=str(args.output_dir))
    if args.device == "cuda":
        torch.cuda.synchronize()

    training_started = time.perf_counter()
    upstream.train(options, model_dir, str(cache), torch.device(args.device))

    if args.device == "cuda":
        torch.cuda.synchronize()

    training_seconds = time.perf_counter() - training_started

    for folder in (Path(options.output_dir), Path(options.output_dir) / "checkpoint_latest"):
        config_path = folder / "rl_agent_config.json"
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        saved["model_name"] = "laya-korean-support-demo"
        config_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")

    total_seconds = time.perf_counter() - total_started
    
    print(f"학습 단계 소요 시간 (모델 구성·학습·확률 보정·저장 포함): {format_duration(training_seconds)}", flush=True)
    print(f"총 실행 시간 (패키지 로딩·모델 준비·전처리 포함): {format_duration(total_seconds)}", flush=True)

if __name__ == "__main__":
    main()
