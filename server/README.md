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
1. `app/prompt.py`의 user 메시지 형식을 학습 JSONL과 동일하게 맞출 것 (현재는 임시 형식)
2. `prompts/sp01.txt`를 v5.5 학습에 쓴 SP01로 교체 (현재는 v5 README의 SP01)
3. `answer_question` 필드명을 v5.5 엑셀로 확인
4. `database.py`를 `server/`에 두면 화면 정규화·품절 조회·추론 로그가 자동 연결됨 (함수 시그니처 확인)
5. HF / RKLLM 백엔드 구현 (보드 수령 후 타임아웃 값 결정)
