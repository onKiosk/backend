"""데이터셋 정답(train 192 + eval 48)이 스키마·메뉴 정책 검증을 보정 없이 그대로 통과하는지 확인."""
import json
from pathlib import Path

import pytest

from app.policy import apply_policy
from app.schemas import ACTION_ADAPTER

CASES = json.loads((Path(__file__).resolve().parents[1] / "data" / "dataset_cases.json").read_text("utf-8"))


def test_case_count():
    assert len(CASES) == 240


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_gold_target_passes(case):
    action = ACTION_ADAPTER.validate_python(case["target"])
    ctx = case["context"]
    fixed, repairs = apply_policy(action, ctx, ctx.get("sold_out") or [])
    assert repairs == []
    assert fixed.model_dump(exclude_none=True) == case["target"]
