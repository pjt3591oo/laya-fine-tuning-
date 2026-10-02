"""Build nested, balanced synthetic training sets; no pretrained model required."""
import json
import argparse
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 20261002

# Shared contexts deliberately avoid class-specific cues.
CONTEXTS = [
    "처음 이용하는 고객입니다. {request}",
    "개인 계정으로 서비스를 이용 중이에요. {request}",
    "회사에서 사용하는 서비스와 관련해 문의합니다. {request}",
    "모바일에서 이용하고 있습니다. {request}",
    "PC로 접속해서 사용 중인데요. {request}",
    "오늘 오전에 확인한 내용입니다. {request}",
    "어제부터 알아보던 건인데요. {request}",
    "기존 고객인데 확인할 사항이 있어요. {request}",
    "고객센터로 문의하라는 안내를 받았습니다. {request}",
    "급한 건은 아니지만 확인 부탁드립니다. {request}",
]

REQUESTS = {
    "billing": [
        "결제 버튼을 한 번 눌렀는데 카드 승인 문자가 두 번 왔어요. 중복 승인 취소해주세요.",
        "취소한 주문의 카드 대금이 그대로 청구됐습니다. 환불 처리 여부를 확인해주세요.",
        "무료 체험이 끝나면서 결제됐는데 연장을 원하지 않습니다. 결제 취소를 요청합니다.",
        "환불 완료 안내를 받았지만 통장에 돈이 들어오지 않았어요. 입금 일정을 알려주세요.",
        "체크카드로 결제할 때 잔액이 충분한데 승인 거절 메시지가 나옵니다.",
        "지난번 이용료에 대한 현금영수증을 발급받고 싶습니다.",
        "세금계산서의 사업자 번호를 잘못 입력했어요. 수정 발급을 부탁드립니다.",
        "할인 쿠폰을 적용했는데 정가로 결제됐어요. 차액을 돌려받고 싶습니다.",
        "월 구독료가 안내된 금액보다 많이 청구됐습니다. 추가 청구 내역을 확인해주세요.",
        "결제 수단을 변경했는데 이전 카드와 새 카드에서 모두 돈이 빠졌습니다.",
        "계좌 이체를 마쳤는데 결제 확인이 안 됩니다. 입금 내역을 확인해주세요.",
        "구독을 해지했는데 다음 달 이용료까지 미리 결제됐어요. 취소 가능한가요?",
        "구매 직후 이용권을 잘못 선택한 걸 알았어요. 사용 전이라 환불하고 싶습니다.",
        "해외 카드로 결제하면 인증 단계에서 거절됩니다. 결제 방법을 확인해주세요.",
        "카드 승인에는 성공했는데 서비스 이용권이 지급되지 않았어요. 결제 내역 확인 바랍니다.",
        "사용하지 않은 기간의 구독료를 환불받을 수 있는지 문의합니다.",
        "영수증에 표시된 결제자 이름을 변경해서 다시 발급받고 싶어요.",
        "결제 취소 문자를 받았는데 카드 앱에는 아직 승인 상태입니다. 처리 중인가요?",
        "자동 결제를 껐는데 돈이 빠져나갔어요. 이번 결제 건을 환불해주세요.",
        "주문 취소 후 일부 금액만 반환됐습니다. 나머지 환불액이 어떻게 계산됐나요?",
    ],
    "account": [
        "비밀번호 재설정 링크를 눌러도 만료된 링크라고 합니다. 새 링크를 받고 싶어요.",
        "가입할 때 등록한 이메일에 접근할 수 없어서 로그인 인증을 못 하고 있습니다.",
        "기기를 바꾼 뒤 인증 앱의 코드를 사용할 수 없어요. 계정 접근을 복구해주세요.",
        "로그인 시도가 많다는 이유로 계정이 잠겼어요. 잠금 해제를 요청합니다.",
        "휴대폰 인증번호가 도착하지 않아 회원가입을 진행할 수 없습니다.",
        "회사 이메일 대신 개인 이메일로 로그인 주소를 변경하고 싶어요.",
        "구글로 가입한 계정에 비밀번호 로그인도 연결할 수 있나요?",
        "탈퇴 처리를 요청합니다. 제 계정을 삭제해주세요.",
        "휴면 계정을 다시 사용하려는데 본인 확인을 완료할 수 없어요.",
        "등록한 전화번호를 해지해서 로그인 코드를 받을 수 없습니다. 번호를 바꾸고 싶어요.",
        "맞는 비밀번호를 입력했는데 자꾸 자격 증명이 틀렸다고 나옵니다.",
        "이메일 인증을 완료했는데 계속 미인증 계정이라고 해서 로그인이 막힙니다.",
        "다른 사람이 제 계정에 접속한 기록이 있어요. 모든 기기에서 로그아웃하고 싶습니다.",
        "회원가입을 하려는데 이미 등록된 이메일이라고 합니다. 기존 계정을 찾고 싶어요.",
        "소셜 로그인 제공자를 바꿨더니 새 계정이 생겼어요. 기존 계정과 연결 부탁드립니다.",
        "계정을 탈퇴한 뒤 다시 가입하려는데 이메일을 사용할 수 없다고 나옵니다.",
        "로그인용 아이디를 잊었습니다. 계정 찾기를 도와주세요.",
        "인증 코드가 도착할 때마다 틀린 코드라고 합니다. 로그인 인증을 확인해주세요.",
        "개명해서 계정의 본인 인증 이름과 현재 이름이 다릅니다. 변경을 요청합니다.",
        "계정 접근이 차단됐다는 안내를 받았습니다. 해제 절차를 알려주세요.",
    ],
    "technical": [
        "첨부 파일을 업로드하면 진행 표시가 멈추고 완료되지 않습니다.",
        "문서를 저장한 뒤 다시 열면 마지막 수정 내용이 사라집니다.",
        "앱 실행 후 첫 화면에서 반복적으로 강제 종료됩니다.",
        "검색어를 입력하고 검색을 눌렀는데 버튼이 작동하지 않아요.",
        "동영상 재생 시 소리만 들리고 영상은 검은 화면으로 표시됩니다.",
        "내보낸 PDF를 열면 파일이 손상됐다는 오류가 뜹니다.",
        "새 알림이 있다고 표시되지만 알림 목록을 열면 빈 화면입니다.",
        "캘린더에서 날짜를 선택해도 이전 날짜의 내용만 표시됩니다.",
        "페이지를 이동할 때마다 서버 오류 502가 발생합니다.",
        "설정값을 바꿔도 새로고침하면 변경 전으로 돌아갑니다.",
        "공유 링크를 열면 문서가 있는데도 찾을 수 없다는 오류가 나옵니다.",
        "다운로드가 완료됐다고 나오지만 파일 크기가 0바이트입니다.",
        "작성 중인 화면에서 한글 글자가 겹쳐 보여 내용을 읽기 어렵습니다.",
        "업데이트 이후 푸시 알림이 전혀 전달되지 않습니다.",
        "파일 이름을 변경하면 목록에서 항목이 사라져요. 재접속해도 같습니다.",
        "데이터 가져오기를 실행하면 매번 같은 단계에서 프로그램이 멈춥니다.",
        "댓글을 등록했는데 같은 댓글이 여러 번 생성됩니다.",
        "프로필 사진을 올렸는데 깨진 이미지 아이콘으로 표시됩니다.",
        "목록의 다음 페이지 버튼을 누르면 이전 페이지가 다시 나타납니다.",
        "사이트 접속 시 빈 화면만 표시되고 새로고침해도 달라지지 않아요.",
    ],
    "product": [
        "무료 플랜과 유료 플랜에서 제공하는 기능 차이를 알고 싶습니다.",
        "팀 단위로 이용할 때 초대할 수 있는 인원 제한이 있나요?",
        "업로드할 수 있는 파일 확장자와 최대 용량을 알려주세요.",
        "문서의 읽기 권한과 편집 권한을 따로 설정하는 방법이 궁금합니다.",
        "외부 협업자에게 문서를 공유하는 절차를 알려주세요.",
        "기본 요금제에서도 데이터 내보내기 기능을 사용할 수 있나요?",
        "서비스를 오프라인 상태에서도 이용할 수 있는지 궁금합니다.",
        "기업용 플랜에 포함되는 관리 기능을 설명해주세요.",
        "팀원 초대를 보내는 메뉴가 어디에 있는지 알려주세요.",
        "저장 공간이 부족하면 용량을 추가하는 상품이 있나요?",
        "여러 문서를 폴더별로 정리하는 사용 방법을 알고 싶습니다.",
        "자동 백업 기능을 제공하는지, 제공한다면 주기가 어떻게 되는지 궁금해요.",
        "모바일 앱에서 이용할 수 있는 기능 목록을 알려주세요.",
        "다른 서비스의 자료를 가져오는 기능이 지원되나요?",
        "서비스가 지원하는 운영체제와 브라우저를 알려주세요.",
        "예약 발송 기능은 어느 요금제부터 이용할 수 있나요?",
        "삭제한 문서를 복원하는 메뉴와 보관 기간을 알고 싶어요.",
        "대시보드에 팀별 통계를 표시하는 방법을 안내해주세요.",
        "개인용 계정을 팀 플랜으로 전환하는 절차가 궁금합니다.",
        "API 연동 기능이 제공되는 상품인지 확인하고 싶습니다.",
    ],
    "other": [
        "브랜드 협업 제안을 보내고 싶은데 담당 부서를 알려주세요.",
        "귀사의 개발자 채용 공고를 어디서 확인할 수 있나요?",
        "언론 인터뷰 일정을 협의하고 싶습니다. 홍보 담당자에게 전달해주세요.",
        "행사 후원을 제안하려고 합니다. 제안서 접수 창구가 궁금합니다.",
        "사무실 방문을 위해 회사 주소와 방문 가능 시간을 알고 싶습니다.",
        "지난번 상담 직원이 친절해서 감사 인사를 전하고 싶어요.",
        "대학 연구 프로젝트 협력을 제안하고 싶습니다.",
        "회사 소개 자료와 보도자료를 받아보고 싶습니다.",
        "인턴 채용 지원에 필요한 서류를 문의합니다.",
        "서비스의 브랜드 로고 디자인이 마음에 듭니다. 의견 전달 부탁드려요.",
        "문의할 사항이 있는데 어떤 내용부터 말씀드리면 될까요?",
        "전에 말씀드린 일을 다시 확인해주실 수 있나요? 자세한 내용은 아직 정리 중입니다.",
        "괜찮아졌습니다. 앞서 남긴 문의는 더 이상 답변하지 않으셔도 됩니다.",
        "귀사 행사에 연사로 참여할 수 있을지 제안드립니다.",
        "광고 집행 관련 협의를 하고 싶습니다. 담당자 연결을 부탁드려요.",
        "번역 업무 협업을 제안하려고 합니다. 연락 가능한 부서를 알려주세요.",
        "대표님께 사업 제안서를 전달할 수 있는 방법을 문의합니다.",
        "사회공헌 프로그램에 공동 참여를 제안하고 싶습니다.",
        "답변을 기다리는 건 아니고 응원의 말을 남기고 싶었습니다.",
        "잘못된 곳에 문의를 보냈습니다. 이 메시지는 무시해주세요.",
    ],
}

def main():
    parser = argparse.ArgumentParser(description="Generate balanced 100/400/1000 training sets")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing scaled data")
    args = parser.parse_args()
    directory = ROOT / "data"
    labels = list(json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))["category"]["criteria"])
    excluded = set()
    for name in ("train.jsonl", "test.jsonl"):
        for line in (directory / name).read_text(encoding="utf-8").splitlines():
            excluded.add(json.loads(line)["text"])
    sizes = (100, 400, 1000)
    outputs = [directory / f"train_{size}.jsonl" for size in sizes]
    outputs.append(directory / "scaled_data_manifest.json")
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {path}; use --overwrite to regenerate")
    pool = {}
    for label in labels:
        rows = []
        for request_id, request in enumerate(REQUESTS[label]):
            for context_id, context in enumerate(CONTEXTS):
                text = context.format(request=request)
                if text in excluded:
                    raise ValueError("Generated text overlaps the original data")
                rows.append({"id": f"scaled-{label}-{request_id+1:02d}-{context_id+1:02d}",
                             "text": text, "label": label, "synthetic": True,
                             "template_group": f"{label}-{request_id+1:02d}"})
        # Round-robin request groups before adding more context variants.
        rng = random.Random(SEED)
        request_order = list(range(len(REQUESTS[label])))
        rng.shuffle(request_order)
        context_orders = {}
        for request_id in request_order:
            context_orders[request_id] = rng.sample(range(len(CONTEXTS)), len(CONTEXTS))
        pool[label] = [rows[r*len(CONTEXTS)+context_orders[r][c]]
                       for c in range(len(CONTEXTS)) for r in request_order]
    manifest = {"seed": SEED, "synthetic": True,
                "generation": "100 handwritten request templates x 10 shared contexts",
                "warning": "Variants share templates; size is not independent semantic diversity. Do not randomly split variants for evaluation; group by template_group.",
                "nested": "train_100 is a subset of train_400, which is a subset of train_1000",
                "datasets": {}}
    previous = set()
    for size in sizes:
        rows = [row for label in labels for row in pool[label][:size//len(labels)]]
        random.Random(SEED + size).shuffle(rows)
        texts = {row["text"] for row in rows}
        ids = {row["id"] for row in rows}
        assert len(rows) == len(texts) == len(ids) == size
        assert previous <= ids
        assert not texts & excluded
        counts = dict(Counter(row["label"] for row in rows))
        assert all(count == size//len(labels) for count in counts.values())
        path = directory / f"train_{size}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
        manifest["datasets"][path.name] = {"rows": size, "labels": counts,
                                           "unique_template_groups": len({row["template_group"] for row in rows})}
        previous = ids
        print(f"{path.name}: {size} rows, {counts}")
    outputs[-1].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
