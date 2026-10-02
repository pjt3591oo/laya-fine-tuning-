"""Create small, synthetic learning examples, never a production benchmark."""
import json
import argparse
from pathlib import Path

EXAMPLES = {
    "billing": [
        "같은 주문이 두 번 결제됐어요. 한 건 취소해주세요.",
        "구독을 취소했는데 이번 달 요금이 청구됐습니다. 환불 부탁드려요.",
        "카드로 결제하면 승인 실패라고 나와요.",
        "지난달 결제 영수증을 받고 싶습니다.",
        "결제한 금액과 안내된 가격이 다릅니다. 확인해주세요.",
        "환불 신청한 금액이 아직 입금되지 않았어요.",
        "카드 한도가 충분한데 결제가 거절됩니다.",
        "서비스를 사용하지 않았는데 자동 결제됐어요. 취소해주세요.",
        "계좌에서는 돈이 빠졌는데 주문은 미결제로 나옵니다.",
        "연간 이용권을 구매했는데 환불 가능한가요?",
    ],
    "account": [
        "비밀번호를 잊어버렸어요. 재설정하고 싶습니다.",
        "인증 메일이 오지 않아 가입을 완료할 수 없어요.",
        "휴대폰을 바꿨는데 이중 인증 때문에 로그인할 수 없습니다.",
        "회원 탈퇴를 신청하려고 합니다.",
        "비밀번호가 맞는데 로그인이 안 돼요.",
        "제 계정이 잠겼습니다. 잠금 해제 부탁드립니다.",
        "로그인에 사용하는 이메일 주소를 변경하고 싶어요.",
        "소셜 로그인 계정을 기존 계정과 연결하고 싶습니다.",
        "인증번호가 계속 만료됐다고 나와서 계정에 못 들어갑니다.",
        "탈퇴했던 계정을 복구할 수 있나요?",
    ],
    "technical": [
        "앱을 실행하자마자 종료됩니다.",
        "파일 업로드가 50퍼센트에서 멈춥니다.",
        "검색 버튼을 눌러도 아무 반응이 없어요.",
        "화면에 500 오류가 표시됩니다.",
        "업데이트 이후 알림 기능이 작동하지 않습니다.",
        "저장 버튼을 눌렀는데 작성한 내용이 사라졌어요.",
        "동영상 재생 중 소리는 나오는데 화면이 검게 나옵니다.",
        "다운로드한 파일이 깨져서 열리지 않아요.",
        "웹사이트가 계속 로딩 중이라 이용할 수 없어요.",
        "설정 화면에 들어가면 앱이 멈추고 강제로 종료해야 합니다.",
    ],
    "product": [
        "무료 요금제에서도 파일 내보내기가 가능한가요?",
        "팀 요금제와 개인 요금제의 차이가 궁금합니다.",
        "서비스에서 지원하는 파일 형식을 알려주세요.",
        "팀원을 초대하는 방법을 알려주세요.",
        "모바일에서도 같은 기능을 이용할 수 있나요?",
        "한 번에 올릴 수 있는 파일 크기 제한이 있나요?",
        "기업용 요금제의 기능을 알고 싶습니다.",
        "문서를 다른 사람과 공유하는 방법이 궁금해요.",
        "해외에서도 서비스를 이용할 수 있는지 궁금합니다.",
        "유료 플랜에는 저장 공간이 얼마나 제공되나요?",
    ],
    "other": [
        "귀사와 제휴를 논의하고 싶습니다.",
        "현재 채용 중인 직무가 있나요?",
        "담당자에게 인터뷰를 요청하고 싶습니다.",
        "서비스 로고 색상이 마음에 들어요.",
        "문의드립니다.",
        "행사 후원 제안을 전달하고 싶어요.",
        "오늘도 좋은 하루 보내세요.",
        "마케팅 협업 담당자 연락처를 알고 싶습니다.",
        "어제 이야기했던 건 확인 부탁드려요.",
        "보도자료를 받을 수 있을까요?",
    ],
}

def main():
    parser = argparse.ArgumentParser(description="Generate 40 training and 10 test examples")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing demo data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "data"
    root.mkdir(exist_ok=True)
    for split in ("train", "test"):
        path = root / f"{split}.jsonl"
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {path}; use --overwrite to regenerate")
    rows = {"train": [], "test": []}
    for label, texts in EXAMPLES.items():
        for i, text in enumerate(texts):
            split = "train" if i < 8 else "test"
            rows[split].append({"id": f"demo-{label}-{i+1:02d}", "text": text,
                                "label": label, "synthetic": True})
    for split, items in rows.items():
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} synthetic examples")

if __name__ == "__main__":
    main()
