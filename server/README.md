# OnKiosk Action API (파트 2: FastAPI 코어 서버)

발화 텍스트 + 현재 화면 Context → Action JSON. 지금은 **Mock 백엔드**로 동작하며, 모델(HF/RKLLM)은 같은 자리에 끼워 넣는다.

## 실행 (Git Bash)
```bash
cd server
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-dev.txt
python -m pytest -q                       # 테스트
uvicorn main:app --reload --port 8000     # 서버 (http://localhost:8000/docs)
```

## API
`POST /api/action`
```json
{ "user_text": "따뜻한 아메리카노 라지 두 잔",
  "context": { "screen": "menu", "cart": [], "sold_out": ["콜드브루","마카롱"], "recognition_failure_count": 0 },
  "stt_latency_ms": 420 }
```
응답 — 성공: `{ "ok": true, "fallback": false, "action": {...}, "repairs": [], "source": "model|rule", "screen": "menu" }`
응답 — 실패: `{ "ok": false, "fallback": true, "action": null, "error_code": "TIMEOUT|BACKEND_ERROR|INVALID_JSON|INVALID_SCHEMA|POLICY_VIOLATION" }`

- `fallback: true`일 때만 프론트가 기존 `resolveIntent`(정규식)로 처리한다. `no_action`은 정상 응답이다.
- `recognition_failure_count >= 2`이면 모델을 거치지 않고 `call_staff(VOICE_RECOGNITION_FAILED_TWICE)`를 돌려준다.
- screen은 프론트 9종을 그대로 보내도 서버가 6종으로 바꾼다. `sold_out`을 생략하면 DB(`database.py`) 값을 쓴다.
- 숫자열(전화번호 등)은 모델 입력·로그로 가기 전에 `[숫자]`로 가린다.
- `GET /api/answer-texts`: `answer_question`의 15개 `answer_code` → 안내 문구(초안).
- `GET /health`

`POST /api/orders` (결제 확정 시) — `{ "items":[CartItem...], "payment_method":"KAKAO_PAY", "pickup_type":"VIBRATION_BELL", "print_receipt":true }`
→ `{ "ok":true, "order_number":"A001", "total_price":4500, "order":{...} }` / 실패 `{ "ok":false, "error_code":"SOLD_OUT(409)|INVALID_ORDER(422)|DB_UNAVAILABLE(503)|ORDER_FAILED(500)" }`. 총액은 DB 가격표로 서버가 계산한다.

`POST /api/stamps` — `{ "phone_number":"1234" | "01012345678" | "010-1234-5678", "count":1 }` → `{ "ok":true, "stamp":{...} }` / `INVALID_PHONE(422)` 등. 번호는 로그·응답에 남기지 않는다.

`GET /api/aliases`(별칭 사전), `GET /api/sold-out`(품절 단일 출처)

## 환경변수
| 이름 | 기본값 | 설명 |
|---|---|---|
| `ONKIOSK_BACKEND` | `mock` | `mock` / `hf` / `rkllm` |
| `ONKIOSK_TIMEOUT_S` | `3.0` | 추론 제한 시간(보드 실측 후 조정) |
| `ONKIOSK_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | 쉼표 구분 |
| `ONKIOSK_DB` | `auto` | `off`면 `database.py`를 쓰지 않음 |
| `ONKIOSK_MOCK_REPLAY` | `data/dataset_cases.json` | Mock이 발화를 그대로 재현할 데이터셋 |

## 구조
```
main.py                 uvicorn 진입점
app/schemas.py          Action 19종(Pydantic) + 요청/응답 모델
app/policy.py           메뉴 25종 정책 검증, 안전 보정(품절→안내, 옵션 누락→되묻기)
app/service.py          처리 파이프라인(타임아웃·파싱·검증·폴백·로그)
app/fallback.py         폴백 오류 코드 표
app/prompt.py           SP01 + 모델 입력 조립 (학습 형식과 일치해야 함)
app/db_hooks.py         database.py 연동 지점 (없어도 동작)
app/backends/           base / mock / hf / rkllm
data/dataset_cases.json 데이터셋 240건 (테스트·Mock replay용)
tests/                  정답 240건 검증 + API/정책/폴백 테스트
```

## 확인 필요 (TODO)
1. (완료) `app/prompt.py`·`prompts/sp01.txt`를 v7 학습 형식·SP01(19종)로 교체함. 형식은 `tests/test_prompt.py`가 학습 JSONL 샘플과 비교해 지킨다.
2. HF / RKLLM 백엔드에서 chat template(system/user 래핑)이 모델 변환 설정과 겹치지 않는지 확인
3. `database.py`를 저장소 루트(`server/` 바로 밖) 또는 `server/`에 두면 화면 정규화·품절 조회·추론 로그·주문·적립이 자동 연결됨. 서버 시작 시 `init_db()`를 자동 호출한다.
4. HF / RKLLM 백엔드 구현 (보드 수령 후 타임아웃 값 결정)
