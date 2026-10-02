# Laya로 한국어 고객 문의 분류 모델 파인튜닝하고 FastAPI 서버 만들기

고객 문의를 읽고 담당 부서를 결정하는 기능을 만들어 보자. 입력은 “두 번 결제됐어요. 한 건 취소해주세요.” 같은 문장이고, 출력은 `billing` 같은 정해진 분류다. 여기에 선택지별 확률도 함께 반환하면 애플리케이션에서 결과를 확인하고 후속 처리를 연결하기 편하다.

이번 글에서는 다국어 Laya 모델을 한국어 고객 문의 데이터로 파인튜닝하고, 저장한 모델을 FastAPI에 연결한다. 마지막에는 `curl`로 요청을 보내 분류 결과를 확인한다. 명령어는 Bash 기준이며, 프로젝트 코드는 [예제 저장소](https://github.com/pjt3591oo/laya-fine-tuning-)를 사용한다.

## 1. 만들려는 서비스

문의의 주된 요청을 다음 다섯 유형 중 하나로 분류한다.

| 라벨 | 분류 | 예시 |
|---|---|---|
| `billing` | 결제·환불 | 중복 결제, 환불, 영수증 |
| `account` | 계정·로그인 | 비밀번호, 인증, 계정 접근 |
| `technical` | 기술 문제 | 앱 종료, 화면 오류, 기능 고장 |
| `product` | 상품·서비스 | 요금제, 기능 안내, 이용 방법 |
| `other` | 기타 | 제휴, 채용, 판단 정보 부족 |

전체 흐름은 다음과 같다.

```text
문의 데이터 + 분류 기준
    → 학습 데이터 전처리
    → 다국어 Laya 파인튜닝
    → 모델 저장 및 평가
    → FastAPI에서 모델 로드
    → HTTP 요청에 분류 결과 반환
```

[Laya 공식 저장소](https://github.com/NandhaKishorM/laya)는 Laya를 텍스트에 대해 정해진 선택, 점수, 예/아니오 확률을 반환하는 System 1 의사결정 엔진으로 소개한다. 이번 프로젝트에서는 `convaiinnovations/laya-multilingual`을 기반으로 `choice` 유형의 고객 문의 분류를 학습한다.

HTTP 인터페이스는 [TypeSafe의 System One API](https://docs.typesafe.ai/api)를 참고한다. `state`, `model`, `questions`를 입력으로 받고, `model`, `answers`, `usage`를 출력하는 구조다. 서버 내부에서는 저장한 Laya 모델을 실행하며, `jev-latest`는 이 서버에서 로컬 모델로 연결되는 호환 별칭이다.

## 2. 프로젝트와 실행 환경 준비

Python 3.13 이상과 `uv`가 설치된 환경에서 시작한다. 프로젝트의 정확한 패키지 버전은 `uv.lock`에 기록되어 있다.

```bash
git clone https://github.com/pjt3591oo/laya-fine-tuning-.git
cd laya-fine-tuning-
uv sync --locked
```

프로젝트의 주요 파일은 다음과 같다.

```text
schema.json                     # 분류 기준
make_demo_data.py               # 기본 합성 데이터 생성
make_scaled_data.py             # 100 / 400 / 1,000건 데이터 생성
make_large_data.py              # 10,000 / 20,000건 데이터 생성
train.py                        # 학습 실행
reference/official_finetune.py  # 기반이 되는 공식 학습 코드
evaluate.py                     # 정확도·macro F1 평가
predict.py                      # 단일 문의 추론
server.py                       # FastAPI 서버
```

### Windows와 WSL을 함께 쓸 때

Windows에서 만든 `.venv`를 WSL에서도 그대로 사용하면 플랫폼이 맞지 않는다. 실제로 WSL에서 `uv run`을 실행하는 과정에서 다음 오류가 발생했다.

```text
failed to remove directory .../.venv/Scripts: Input/output error (os error 5)
```

이 상황에서는 WSL용 가상환경을 별도 경로에 만들면 기존 Windows 가상환경과 충돌을 피할 수 있다. WSL 터미널에서 `uv sync`를 실행하기 전에 설정한다.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/laya-finetuning-wsl"
uv sync --locked
```

`UV_PROJECT_ENVIRONMENT`는 uv의 프로젝트 가상환경 경로를 지정하는 설정이다. 지정한 경로에 환경이 없으면 새로 생성한다. 자세한 동작은 [uv 공식 문서](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)에서 확인할 수 있다.

새 WSL 터미널에서도 위 `export` 설정을 적용한 뒤 프로젝트의 `uv` 명령을 실행한다. Python 패키지는 WSL 환경에 별도로 설치되므로 첫 설치에는 시간이 걸릴 수 있다.

### CUDA 사용 가능 여부 확인

```bash
uv run python -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

`CUDA: True`이면 아래 GPU 학습 명령을 사용할 수 있다. `False`이면 드라이버와 현재 환경의 PyTorch 구성을 확인하거나, 학습·평가·추론 명령에 `--device cpu`를 지정한다. CUDA용 PyTorch 설치 조합은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 환경에 맞게 선택한다.

서버에서는 같은 설정을 `LAYA_DEVICE` 환경변수로 지정한다. 이 프로젝트의 기본 기기는 `cuda`이므로 CPU 실행에서는 기기 설정을 명시해야 한다.

## 3. 분류 기준 정의

학습과 추론에서 같은 분류 기준을 사용하도록 `schema.json`에 질문을 정의한다. 아래는 프로젝트 스키마를 축약한 예다.

```json
{
  "category": {
    "type": "choice",
    "instructions": "고객 문의의 주된 요청을 가장 적합한 유형 하나로 분류하세요.",
    "criteria": {
      "billing": "결제·환불: 중복 청구, 환불, 결제 취소",
      "account": "계정·로그인: 비밀번호, 인증, 계정 접근",
      "technical": "기술 문제: 앱 종료, 화면 오류, 기능 고장",
      "product": "상품·서비스: 기능 안내, 요금제, 이용 방법",
      "other": "기타: 제휴, 채용, 분류 정보 부족"
    }
  }
}
```

`category`는 질문 이름이고, `criteria`의 키가 모델이 선택할 라벨이다. 예를 들어 환불 가능 여부는 `billing`, 요금제의 기능을 묻는 문의는 `product`로 분류한다. 여러 요청이 한 문장에 섞여 있으면 어느 요청을 우선할지 데이터 작성 단계에서 정해야 한다.

## 4. 학습 데이터 생성

기본 데이터는 JSONL 형식이다. 한 줄에 문의 하나와 정답 라벨을 저장한다.

```json
{"id":"demo-billing-01","text":"같은 주문이 두 번 결제됐어요. 한 건 취소해주세요.","label":"billing","synthetic":true}
```

생성된 데이터가 없는 경우 다음 순서로 실행한다.

```bash
uv run make_demo_data.py
uv run make_scaled_data.py
uv run make_large_data.py
```

이미 파일이 있으면 기본 실행은 덮어쓰지 않는다. 의도적으로 다시 생성할 때만 각 명령에 `--overwrite`를 붙인다.

| 파일 | 전체 문의 수 | 분류별 문의 수 |
|---|---:|---:|
| `data/train.jsonl` | 40 | 8 |
| `data/train_100.jsonl` | 100 | 20 |
| `data/train_400.jsonl` | 400 | 80 |
| `data/train_1000.jsonl` | 1,000 | 200 |
| `data/train_10000.jsonl` | 10,000 | 2,000 |
| `data/train_20000.jsonl` | 20,000 | 4,000 |

이번 실습의 데이터는 직접 작성한 합성 데이터다. 확장 데이터는 요청 문장 100개에 상황·인사·문의 채널 표현 등을 조합해서 만든다. 따라서 2만 건이라는 파일 크기가 2만 가지 독립적인 문의 의도를 의미하지는 않는다.

`data/test.jsonl`에는 별도의 합성 문의 10개가 있다. 학습 흐름을 확인하는 데 쓰며, 실제 서비스의 성능을 판단하려면 더 다양한 실제 문의와 별도의 평가 데이터가 필요하다. 같은 요청의 변형은 `template_group`으로 묶어서 학습과 평가에 나누어 섞이지 않도록 분리하는 것이 좋다.

## 5. 학습 전 성능 기록

파인튜닝 효과를 비교하려면 같은 테스트 데이터로 원본 모델부터 평가한다.

```bash
uv run evaluate.py \
  --model convaiinnovations/laya-multilingual \
  --device cuda \
  --output baseline.json
```

결과는 `reports/baseline.json`에 저장한다. 여기에는 정확도, macro F1, 혼동 행렬과 문의별 예측 결과가 들어 있다. macro F1은 각 분류의 F1을 계산한 뒤 평균한 값이다.

이 실습에서는 테스트가 10건이어서 문의 하나의 정답 여부만 바뀌어도 정확도가 10%p 변한다. 숫자와 함께 어떤 문의에서 어떤 라벨을 틀렸는지 확인한다.

## 6. 한국어 문의 데이터로 파인튜닝

먼저 1,000건으로 전체 과정을 확인하려면 다음 명령을 실행한다.

```bash
uv run train.py \
  --data data/train_1000.jsonl \
  --output-dir models/korean-support-1000 \
  --epochs 1 \
  --device cuda
```

2만 건 실험은 저장 폴더를 분리해 실행한다.

```bash
uv run train.py \
  --data data/train_20000.jsonl \
  --output-dir models/korean-support-20000 \
  --epochs 1 \
  --micro-batch 1 \
  --grad-accum 8 \
  --device cuda
```

`train.py`는 [Laya의 공식 파인튜닝 예제](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py)를 프로젝트 데이터에 연결하는 역할을 한다. 원본 모델을 `models/base`에 준비하고, 문의와 분류 기준을 학습용 입력으로 변환한 뒤 공식 학습 루프를 호출한다.

정답은 해당 라벨의 확률이 1이고 나머지는 0인 분포로 만든다. 예를 들어 정답이 `billing`이면 다음과 같다.

```python
gold = {
    "probabilities": {
        "billing": 1.0,
        "account": 0.0,
        "technical": 0.0,
        "product": 0.0,
        "other": 0.0,
    }
}
```

현재 코드의 학습 설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| 기반 모델 | `convaiinnovations/laya-multilingual` |
| 업데이트 범위 | 전체 파라미터 |
| 학습 루프 | 공식 코드의 RLCD와 교차 엔트로피 |
| 입력 길이 설정 | `max_len=1024`, `head_max_len=256` |
| 기본 micro-batch | 1 |
| 기본 gradient accumulation | 8 |
| Gradient checkpointing | encoder와 head에 적용 |
| 확률 보정용 데이터 | 전체의 10%, 최대 400건 |

2만 건을 입력하면 400건을 확률 보정에 쓰고 19,600건을 모델 학습에 사용한다. 기본 40건에서는 학습 36건, 보정 4건으로 나뉘며, 보정 데이터가 너무 적어 공식 함수는 온도 1.0을 반환한다.

학습이 끝나면 지정한 출력 폴더에 다음 파일들이 저장된다.

```text
models/korean-support-20000/
├── model.safetensors
├── rl_agent_config.json
├── checkpoint_meta.json
├── encoder/
├── tokenizer/
├── checkpoint_latest/
└── train_items.pt
```

서버에는 최종 모델 폴더 경로를 전달한다. 같은 출력 폴더로 재학습하면 결과를 덮어쓰므로 실험별로 경로를 구분한다.

완료 로그에는 학습 단계 소요 시간과 총 실행 시간이 출력된다. 학습 단계에는 모델 구성·학습·확률 보정·저장이 포함되고, 총 실행 시간에는 패키지 로딩과 원본 모델 준비, 전처리도 포함된다. 하드웨어와 데이터 크기를 함께 기록하면 다음 실험과 비교하기 쉽다.

## 7. 학습 후 평가와 단일 추론

앞서 사용한 테스트 데이터로 학습 모델을 평가한다.

```bash
uv run evaluate.py \
  --model ./models/korean-support-20000 \
  --device cuda \
  --output finetuned-20000.json

uv run predict.py \
  '두 번 결제됐어요. 한 건 취소해주세요.' \
  --model ./models/korean-support-20000 \
  --device cuda
```

`reports/baseline.json`과 `reports/finetuned-20000.json`의 정확도, macro F1, 오분류를 비교한다. 데이터 수를 늘렸다는 이유만으로 성능이 개선됐다고 결론 내리지는 않는다. 실제 결과를 보고 판단한다.

이번 작업 중 기존 `korean-support-20000` 모델을 CPU로 로드한 API 검증에서는 중복 결제 문의가 `billing`으로 분류되었다. 이 단일 사례는 모델 로딩과 요청 처리의 동작을 확인한 결과이며, 전체 분류 성능을 나타내는 지표는 아니다.

## 8. FastAPI에 모델 연결

`server.py`에서는 FastAPI의 lifespan에서 모델을 한 번 로드한다. 프로젝트 코드의 핵심은 다음과 같다.

```python
configured_model = os.getenv("LAYA_MODEL", "models/korean-support-20000")
local_path = ROOT / configured_model
model_path = str(local_path) if local_path.exists() else configured_model

app.state.agent = laya.load(
    model_path,
    device=os.getenv("LAYA_DEVICE", "cuda"),
)
```

요청이 들어오면 검증된 질문을 Laya에 전달한다.

```python
with request.app.state.inference_lock:
    result = request.app.state.agent.predict(body.state, questions)

return SystemOneResponse(
    model=model_name,
    answers=result["answers"],
    usage=result["usage"],
)
```

모델을 요청마다 다시 로드하지 않도록 앱 상태에 저장하고, 잠금으로 추론을 하나씩 실행한다. 엔드포인트는 일반 `def` 함수이므로 FastAPI의 스레드 풀에서 실행된다. 워커 수를 늘리면 각 프로세스가 모델을 따로 로드하므로 메모리 사용량도 고려해야 한다.

Pydantic으로 요청의 세 질문 유형을 구분하고, 잘못된 입력에는 422를 반환한다. 선택지는 `choice`에서 최대 255개, 점수 기준은 `score`에서 2~10개로 제한한다. 서버가 지원하는 API는 다음과 같다.

| 경로 | 용도 |
|---|---|
| `POST /v1/systemone` | 선택·점수·예/아니오 확률 추론 |
| `GET /v1/models` | 설정된 로컬 모델 이름 조회 |
| `GET /health` | 서버 준비 상태 확인 |
| `/docs` | Swagger API 문서 |

학습한 작업은 고객 문의의 `choice` 분류다. 서버는 Laya의 `score`, `noul`도 처리하지만, 이 두 유형의 도메인 성능까지 검증한 것은 아니다.

## 9. 서버 실행

학습 모델 경로와 기기, 로컬 API 키를 지정한다.

```bash
export LAYA_MODEL="models/korean-support-20000"
export LAYA_MODEL_NAME="laya-korean-support"
export LAYA_DEVICE="cuda"
export SYSTEM_ONE_API_KEY="your-local-key"

uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

CPU로 실행하려면 서버 시작 전에 `export LAYA_DEVICE="cpu"`를 적용한다. 1,000건으로 학습했다면 `LAYA_MODEL`도 `models/korean-support-1000`으로 바꾼다.

`SYSTEM_ONE_API_KEY`는 이 로컬 서버가 검사할 인증 문자열이다. 설정하면 요청의 Bearer 키와 비교하고, 설정하지 않으면 인증 없이 요청을 받는다. 예제 문자열은 실제 운영용 키로 바꾼다.

모델 로딩을 마친 뒤 다른 터미널에서 확인한다.

```bash
curl --fail-with-body http://127.0.0.1:8000/health

curl --fail-with-body http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer your-local-key'
```

브라우저에서 `http://127.0.0.1:8000/docs`를 열면 요청·응답 스키마를 확인할 수 있다.

## 10. curl로 고객 문의 분류 요청

Python이나 별도 SDK 없이 JSON을 직접 보낼 수 있다.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Authorization: Bearer your-local-key' \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data-raw '{
    "model": "laya-korean-support",
    "state": "두 번 결제됐어요. 한 건 취소해주세요.",
    "questions": {
      "category": {
        "type": "choice",
        "instructions": "고객 문의의 주된 요청을 가장 적합한 유형 하나로 분류하세요. 결제나 환불 요청은 결제·환불, 인증이나 계정 접근은 계정·로그인, 프로그램의 고장이나 오류는 기술 문제, 기능·요금제·이용 방법 질문은 상품·서비스입니다. 어느 유형에도 해당하지 않거나 판단할 정보가 부족하면 기타입니다.",
        "criteria": {
          "billing": "결제·환불: 중복 청구, 결제 실패, 환불, 결제 취소, 영수증",
          "account": "계정·로그인: 로그인, 비밀번호, 인증, 계정 접근, 회원 탈퇴",
          "technical": "기술 문제: 앱 종료, 화면 오류, 기능 고장, 서비스 접속 오류",
          "product": "상품·서비스: 기능 안내, 요금제 비교, 이용 방법, 지원 범위",
          "other": "기타: 제휴, 채용, 일반 의견, 분류 정보 부족"
        }
      }
    }
  }'
```

학습에서 사용한 `schema.json`과 동일한 질문을 전송하는 예다. `model`은 `jev-latest`로 보내도 같은 로컬 모델을 사용한다.

다음은 앞서 CPU로 실행한 검증에서 얻은 분류 결과의 발췌다. 당시에는 `state`를 `{"message":"두 번 결제됐어요. 한 건 취소해주세요."}` 형태로 전달하고 세 질문을 함께 요청했다. 위 단일 질문 요청의 확률은 실행 조건에 따라 다를 수 있다.

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 1.0,
    "account": 0.0,
    "technical": 0.0,
    "product": 0.0,
    "other": 0.0
  },
  "confidence": 1.0
}
```

실제 HTTP 응답은 이 객체를 `answers.category`에 담고, 최상위에 `model`과 `usage`도 반환한다. 애플리케이션에서는 `answers.category.choice`를 읽어 담당 부서를 연결할 수 있다.

`confidence`와 선택지 확률은 결과를 검토할 때 함께 읽되, 출력값 1.0만으로 실제 문의에서 오답이 없다고 해석하지 않는다. 자동 처리 기준은 별도의 실제 데이터로 검증해야 한다. 긴 입력에서는 Laya가 반환하는 `usage.truncated`도 확인한다.

## 11. 실행 중 확인할 문제

### CUDA unavailable

학습 스크립트는 CUDA를 사용할 수 없으면 시작 전에 종료한다. 현재 가상환경에서 `torch.cuda.is_available()`을 확인하고, CPU로 확인할 때는 명령에 `--device cpu`를 지정한다. Windows와 WSL은 각각 설치된 PyTorch와 GPU 접근 상태를 확인해야 한다.

### 서버 시작 시 모델 로딩 실패

`LAYA_MODEL`이 학습이 완료된 폴더를 가리키는지 확인한다. 최종 모델에는 가중치, 설정, encoder와 tokenizer 파일이 함께 필요하다. 기본 학습 명령의 출력 경로는 `models/korean-support`이고 서버 기본 경로는 `models/korean-support-20000`이므로, 실제 출력 경로를 환경변수로 맞춘다.

### HTTP 401 또는 422

401이면 서버의 `SYSTEM_ONE_API_KEY`와 요청의 Bearer 키가 일치하는지 확인한다. 422이면 `state`, `model`, `questions`가 있는지와 질문별 `type`, `instructions`, `criteria`를 확인한다. 이 서버는 공개 모델 이름과 `jev-latest`, `jev-preview` 별칭을 허용한다.

### 실험 결과를 기록할 때

게시할 성능 수치에는 데이터 수, 실제 학습·보정 데이터 수, epochs, 하드웨어, 정확도와 macro F1을 함께 적는다. 학습 시간은 완료 로그를 기준으로 기록한다. 이 글의 실행 예시에는 별도로 측정하지 않은 전체 성능이나 학습 시간 수치를 넣지 않았다.

이후 실제 고객 문의를 확보하면 같은 인터페이스를 유지하면서 모델과 평가 데이터를 교체할 수 있다. 다음 실험에서는 합성 문장의 수보다 문의 유형의 다양성, 분류 기준의 일관성, 실제 문의에서의 오분류를 중심으로 개선해 볼 계획이다.
