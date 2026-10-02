import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("USE_TF", "0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import laya
    agent = laya.load(args.model, device=args.device)
    questions = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    result = agent.predict(args.text, questions)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
