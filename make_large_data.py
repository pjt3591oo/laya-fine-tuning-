"""Synthetic scale experiments: extra wording does not add independent intents."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from make_scaled_data import ROOT, SEED, REQUESTS, CONTEXTS

PREFIXES = [
    "안녕하세요. ", "안녕하십니까. ", "고객센터 담당자님께 문의드립니다. ",
    "문의 내용을 남깁니다. ", "아래 내용으로 문의를 접수합니다. ",
    "이메일로 내용을 전달드립니다. ", "채팅으로 문의를 남깁니다. ",
    "온라인 문의 게시판에 내용을 적습니다. ", "전화 대신 글로 내용을 남깁니다. ",
    "문의 내용을 간단하게 정리했습니다. ", "연락드릴 내용이 있어 글을 남겨요. ",
    "이번 문의의 내용은 다음과 같습니다. ", "어디로 연락해야 할지 몰라 이곳에 남깁니다. ",
    "고객지원 창구를 통해 연락드립니다. ", "여기에 내용을 남겨도 되는지 모르겠네요. ",
    "관련 부서에 내용을 전달하고 싶습니다. ", "안녕하세요, 고객 문의를 접수하려 합니다. ",
    "안녕하세요. 글로 설명드리겠습니다. ", "문의 양식에 내용을 작성합니다. ",
]

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def main():
    parser = argparse.ArgumentParser(description="Generate 10000/20000 synthetic training sets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "data"
    sizes = (10000, 20000)
    paths = [directory / f"train_{size}.jsonl" for size in sizes]
    manifest_path = directory / "large_data_manifest.json"
    for path in paths + [manifest_path]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {path}; use --overwrite to regenerate")
    base = read_rows(directory / "train_1000.jsonl")
    if len(base) != 1000:
        raise ValueError("Run make_scaled_data.py first to generate the 1000-row dataset")
    groups = defaultdict(list)
    for row in base:
        groups[row["template_group"]].append(row)
    if len(groups) != 100 or any(len(rows) != 10 for rows in groups.values()):
        raise ValueError("Expected 100 template groups with 10 variants each")
    for label, requests in REQUESTS.items():
        for request_id, request in enumerate(requests, 1):
            group = f"{label}-{request_id:02d}"
            variants = []
            for prefix_id, prefix in enumerate(PREFIXES, 1):
                for context_id, context in enumerate(CONTEXTS, 1):
                    variants.append({"id": f"large-{label}-{request_id:02d}-{prefix_id:02d}-{context_id:02d}",
                                     "text": prefix + context.format(request=request),
                                     "label": label, "synthetic": True, "template_group": group})
            random.Random(f"{SEED}:{group}").shuffle(variants)
            groups[group].extend(variants)
    excluded = {row["text"] for name in ("train.jsonl", "test.jsonl")
                for row in read_rows(directory / name)}
    manifest = {"synthetic": True, "seed": SEED,
                "generation": "100 request templates, 10 contexts, 19 shared greeting/channel prefixes",
                "warning": "Only 100 underlying request templates. Extra rows add wording combinations, not new intents. Keep template_group together when splitting data.",
                "nested": "train_1000 subset of train_10000 subset of train_20000", "datasets": {}}
    previous = {row["id"] for row in base}
    for size, path in zip(sizes, paths):
        rows = [row for group in sorted(groups) for row in groups[group][:size//len(groups)]]
        random.Random(SEED + size).shuffle(rows)
        ids = {row["id"] for row in rows}
        texts = {row["text"] for row in rows}
        counts = Counter(row["label"] for row in rows)
        if not (len(rows) == len(ids) == len(texts) == size and previous <= ids
                and not excluded & texts and len(counts) == 5
                and all(count == size//5 for count in counts.values())):
            raise ValueError("Dataset validation failed")
        content = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows)
        path.write_text(content, encoding="utf-8")
        manifest["datasets"][path.name] = {"rows": size, "labels": dict(counts),
                                           "unique_template_groups": len(groups),
                                           "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        previous = ids
        print(f"{path.name}: {size} rows, {dict(counts)}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
