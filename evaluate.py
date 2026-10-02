"""Compare saved checkpoints on the same held-out toy examples."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("USE_TF", "0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="baseline.json")
    args = parser.parse_args()
    import laya
    agent = laya.load(args.model, device=args.device)
    questions = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    labels = list(questions["category"]["criteria"])
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    predictions = []
    for line in (ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        answer = agent.predict(row["text"], questions)["answers"]["category"]
        pred = answer["choice"]
        matrix[row["label"]][pred] += 1
        predictions.append({**row, "prediction": pred})
    if not predictions:
        raise ValueError("Test data is empty")
    accuracy = sum(r["label"] == r["prediction"] for r in predictions) / len(predictions)
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in labels if g != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        f1s.append(2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    report = {"model": args.model, "accuracy": accuracy, "macro_f1": sum(f1s)/len(f1s),
              "warning": "10 synthetic examples: workflow check only, not production evidence",
              "confusion_matrix_gold_rows": matrix, "predictions": predictions}
    output = ROOT / "reports" / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accuracy={accuracy:.3f}, macro_f1={report['macro_f1']:.3f}; {output}")

if __name__ == "__main__":
    main()
