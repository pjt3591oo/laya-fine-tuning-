# 한국어 고객 문의 분류 실습

다국어 Laya로 결제·환불, 계정·로그인, 기술 문제, 상품·서비스, 기타를 분류합니다.
`schema.json`에 분류 기준이 있습니다. 문의의 주된 요청 하나를 선택합니다.
예: 환불 가능 여부는 결제·환불, 요금제 기능 질문은 상품·서비스입니다.
여러 요청이 동등하게 섞인 실무 문의는 별도 라벨 정책이 필요합니다.

## 데이터

`data/train.jsonl` 40개, `data/test.jsonl` 10개는 직접 작성한 **합성 실습 데이터**입니다.
실제 고객 데이터나 공개 벤치마크가 아닙니다. 학습 흐름 확인용이며 일반화 성능을 입증하지 않습니다.
각 행은 `id`, `text`, `label`, `synthetic` 필드를 갖습니다.
학습 내부에서 4개를 확률 보정용으로 분리하므로, 실제 학습에는 36개가 쓰입니다.
보정 데이터가 너무 적어 공식 함수는 온도 1.0을 반환합니다. 확률을 자동 처리 기준으로 쓰지 마세요.
실제 데이터를 사용할 때는 고객/대화 단위로 학습·보정·테스트를 분리하고
각 분류의 사례, 오탈자, 짧은 문의, 여러 의도가 섞인 문의를 확보하세요.

## 실행

### 데이터 생성 순서

현재는 생성된 데이터 파일이 있으므로 바로 학습할 수 있습니다.
데이터 파일이 없는 상태에서는 환경 설치 후 아래 세 스크립트를 순서대로 실행합니다.
`make_scaled_data.py`는 기본 학습·테스트 파일을 읽으므로 `make_demo_data.py`가 먼저입니다.

```bash
uv run make_demo_data.py
uv run make_scaled_data.py
uv run make_large_data.py
```

- `make_demo_data.py`: `data/train.jsonl` 40개, `data/test.jsonl` 10개 생성.
- `make_scaled_data.py`: `data/train_100.jsonl`, `train_400.jsonl`, `train_1000.jsonl` 생성.
- `make_large_data.py`: `data/train_10000.jsonl`, `train_20000.jsonl` 생성. 기존 1,000개 파일을 읽으므로 마지막에 실행합니다.

이미 있는 데이터를 처음부터 재생성하려면 아래 명령을 사용합니다.
`--overwrite`는 해당 스크립트가 만드는 데이터 파일을 덮어씁니다.
같은 코드와 시드로 실행하면 같은 데이터가 만들어집니다.

```bash
uv run make_demo_data.py --overwrite
uv run make_scaled_data.py --overwrite
uv run make_large_data.py --overwrite
```

### 확장 합성 학습 데이터

| 파일 | 전체 문의 수 | 분류별 문의 수 |
|---|---:|---:|
| `data/train_100.jsonl` | 100 | 20 |
| `data/train_400.jsonl` | 400 | 80 |
| `data/train_1000.jsonl` | 1,000 | 200 |
| `data/train_10000.jsonl` | 10,000 | 2,000 |
| `data/train_20000.jsonl` | 20,000 | 4,000 |

100개는 400개에, 400개는 1,000개에 포함됩니다. 기존 40개와 테스트 10개는 유지합니다.
새 파일은 기존 학습·테스트 문의와 텍스트가 중복되지 않습니다.
직접 작성한 요청 문장 100개에 공통 상황 표현 10종을 조합한 합성 데이터입니다.
따라서 1,000개가 모두 서로 독립적인 의미의 사례인 것은 아닙니다.
`template_group`이 같은 행은 같은 요청의 변형이며 평가 분리 시 같은 그룹으로 묶어야 합니다.
기존 테스트 10개는 작은 실습용 테스트로 계속 사용합니다.
기본 실행은 기존 파일을 덮어쓰지 않으며, 재생성 시 `--overwrite`를 지정합니다.
파일 개수와 분류별 구성은 `data/scaled_data_manifest.json`에 기록했습니다.

10,000개는 기존 1,000개를, 20,000개는 10,000개를 포함합니다.
큰 데이터도 동일한 요청 문장 100개를 바탕으로 상황·인사·문의 채널 표현을 조합합니다.
문의 유형의 다양성이 10배·20배 늘어난 데이터가 아니며 학습 규모와 시간 비교용입니다.
큰 데이터의 구성과 SHA-256은 `data/large_data_manifest.json`에 기록했습니다.

`--data`로 학습 데이터를, `--output-dir`로 결과 저장 폴더를 선택합니다.
아래처럼 데이터 크기별로 폴더를 지정하면 각각의 결과를 따로 보관할 수 있습니다.

```bash
uv run train.py --data data/train.jsonl --output-dir models/korean-support-40 --epochs 1
uv run train.py --data data/train_100.jsonl --output-dir models/korean-support-100 --epochs 1
uv run train.py --data data/train_400.jsonl --output-dir models/korean-support-400 --epochs 1
uv run train.py --data data/train_1000.jsonl --output-dir models/korean-support-1000 --epochs 1
uv run train.py --data data/train_10000.jsonl --output-dir models/korean-support-10000 --epochs 1
uv run train.py --data data/train_20000.jsonl --output-dir models/korean-support-20000 --epochs 1
```

지정한 폴더에는 최종 모델과 설정·토크나이저, `checkpoint_latest/`,
전처리한 학습 데이터 `train_items.pt`가 저장됩니다.
`--output-dir`를 생략하면 기존과 같이 `models/korean-support`에 저장합니다.
같은 저장 폴더로 다시 실행하면 이전 결과를 덮어씁니다.
상대 경로는 명령을 실행하는 현재 폴더 기준이며 절대 경로도 사용할 수 있습니다.

학습이 정상 완료되면 소요 시간을 `시:분:초`와 총 초 단위로 출력합니다.
학습 단계 시간은 모델 구성·학습·확률 보정·모델 저장을 포함하고,
총 실행 시간은 패키지 로딩·원본 모델 준비(필요 시 다운로드)·데이터 전처리도 포함합니다.
CUDA 실행은 측정 경계에서 GPU 작업 완료를 기다려 시간을 측정합니다.

평가 결과도 `--output`으로 파일 이름을 달리 지정합니다 (`reports/`에 저장).

```bash
uv run evaluate.py --model ./models/base --output finetuned.json

uv run evaluate.py --model ./models/korean-support-100 --output finetuned-100.json
uv run evaluate.py --model ./models/korean-support-400 --output finetuned-400.json
uv run evaluate.py --model ./models/korean-support-1000 --output finetuned-1000.json
uv run predict.py "두 번 결제됐어요. 한 건 취소해주세요." --model ./models/korean-support-1000
```

확장 데이터의 학습은 아직 실행하지 않았습니다.

Python과 uv가 설치되어 있는 환경에서 프로젝트 폴더에서 실행합니다.
첫 모델 실행은 Hugging Face 모델 다운로드가 필요합니다.

```bash

uv run -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 데이터가 없을 때 생성 (이미 있으면 이 세 줄은 생략)
uv run make_demo_data.py
uv run make_scaled_data.py
uv run make_large_data.py

uv run evaluate.py --output baseline.json
uv run train.py --epochs 1
uv run evaluate.py --model ./models/korean-support --output finetuned.json
uv run predict.py "두 번 결제됐어요. 한 건 취소해주세요." --model ./models/korean-support
```

학습은 공식 RLCD + 교차 엔트로피 루프를 호출해 전체 파라미터를 업데이트합니다.
micro-batch 1, gradient accumulation 8, encoder/head gradient checkpointing을 사용합니다.
16GB GPU에서 시작하기 위한 구성으로 실제 메모리와 CUDA 호환성은 실행 검증이 필요합니다.
메모리 부족 시 다른 GPU 프로그램을 닫고 입력 길이와 학습 구성을 조정하세요.
`reports`의 정확도·macro F1·오분류를 전후 비교합니다. 작은 합성 데이터에서 성능 개선을 보장하지 않습니다.

## System One 호환 API 서버

FastAPI 서버는 TypeSafe의 [공식 HTTP 스펙](https://docs.typesafe.ai/api)에 맞춰
`POST /v1/systemone`의 `state`, `model`, `questions`를 로컬 Laya 모델로 처리합니다.
`choice`, `score`, `noul`을 지원하며 응답은 `model`, `answers`, `usage`입니다.
Jev 서버를 호출하는 것이 아니라 학습된 Laya를 실행합니다.

```bash
uv sync --cache-dir .uv-cache
export LAYA_MODEL="models/korean-support-20000"
export LAYA_DEVICE="cuda" # GPU가 없으면 cpu
export SYSTEM_ONE_API_KEY="your-local-key"
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

기본 모델 경로는 `models/korean-support-20000`, 기본 기기는 `cuda`입니다.
`LAYA_MODEL`은 다른 학습 폴더나 Hugging Face 모델 ID로 지정할 수 있습니다.
`LAYA_MODEL_NAME`으로 공개 모델 이름을 지정합니다 (기본 `laya-korean-support`).
요청의 `jev-latest`, `jev-preview`는 이 로컬 모델의 호환 별칭입니다.
응답에는 실제 로컬 모델 이름을 반환합니다. 요청마다 모델을 바꾸지는 않습니다.
`SYSTEM_ONE_API_KEY`를 설정하면 Bearer 인증을 검사하고, 생략하면 인증 없이 실행합니다.

다른 Bash 터미널에서 고객 문의 분류를 요청합니다.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone  \
 -H 'Authorization: Bearer your-local-key'  \
 -H 'Content-Type: application/json; charset=utf-8'  \
 --data-raw '{
    "model": "jev-latest",
    "state": "두 번 결제됐어요. 한 건 취소해주세요.",
    "questions": {
      "category": {
        "type": "choice",
        "instructions": "고객 문의의 주된 요청을 가장 적합한 유형 하나로 분류하세요.",
        "criteria": {
          "billing": "결제·환불",
          "account": "계정·로그인",
          "technical": "기술 문제",
          "product": "상품·서비스",
          "other": "기타"
        }
      }
    }
  }'
```

`GET /v1/models`에서 공개 모델 이름을 조회하고, `/health`에서 준비 상태를 확인합니다.
Swagger 문서는 `http://127.0.0.1:8000/docs`에 있습니다.
잘못된 요청은 422, 잘못된 인증은 401을 반환합니다.
모델은 서버 시작 시 한 번 로드하며 추론은 하나씩 처리합니다. 워커마다 모델 메모리가 필요합니다.
`usage`의 Laya 입력 잘림 진단도 반환하므로 긴 입력은 `truncated`를 확인하세요.
TypeSafe의 호스팅 서비스와 같은 추론 결과·컨텍스트 한도·과금·요청 제한을 제공하는 것은 아닙니다.

## Todo

https://github.com/pjt3591oo/customer-service-center-chatbot 개선하기

## 출처

`reference/official_finetune.py`는 아래 공식 코드를 내려받은 원본입니다 (Apache-2.0).
Mac용 CLI 대신 `train.py`가 다국어 모델과 로컬 데이터를 지정하고 CUDA 기기를 전달합니다.
내보낸 모델 이름 메타데이터는 `laya-korean-support-demo`이며 로드에는 모델 폴더 경로를 사용합니다.

- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py
- https://huggingface.co/convaiinnovations/laya-multilingual

이전 실습의 학습 모델·평가 결과·학습 중간 파일·실행 결과 기록은 정리했습니다.
가상환경과 원본 모델은 준비되어 있어 위 명령으로 직접 학습·평가할 수 있습니다.
