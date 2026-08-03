"""verify-then-answer — на fake-LLM, без сети."""

from __future__ import annotations

import json

from graphrag.generate.answer import ContextItem
from graphrag.generate.verify import (
    parse_claims,
    supported_items,
    verify_then_answer,
)
from graphrag.llm.base import LLMClient

_CTX = [
    ContextItem(text="NetworkClient переподключается циклически", uri="u1"),
    ContextItem(text="Тикет KAFKA-1 про таймаут", uri="u2"),
]


class FakeLLM(LLMClient):
    """Скриптует два вызова декомпозера (первичный + пере-суд) и одну регенерацию."""

    def __init__(self, initial_json: str, regen_text: str = "", recheck_json: str = '{"claims":[]}'):
        super().__init__("fake")
        self.initial_json = initial_json
        self.regen_text = regen_text
        self.recheck_json = recheck_json
        self.decompose_calls = 0
        self.gen_prompt: str | None = None

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        if "Разложи ОТВЕТ" in prompt:
            self.decompose_calls += 1
            return self.initial_json if self.decompose_calls == 1 else self.recheck_json
        self.gen_prompt = prompt  # генерация (регенерация из подтверждённых)
        return self.regen_text


def _claims(*items) -> str:
    return json.dumps({"claims": list(items)})


# --- чистые ---

def test_parse_claims_with_source():
    p = parse_claims('{"claims":[{"claim":"a","supported":true,"source":"u1"},{"claim":"b","supported":false}]}')
    assert len(p) == 2
    assert p[0] == {"claim": "a", "supported": True, "source": "u1"}
    assert p[1]["supported"] is False and p[1]["source"] is None


def test_parse_claims_code_fence_and_garbage():
    assert parse_claims('```json\n{"claims":[{"claim":"x","supported":true,"source":"u1"}]}\n```')[0]["claim"] == "x"
    assert parse_claims("совсем не json") is None
    assert parse_claims('{"claims":[]}') == []


def test_supported_items_filters():
    claims = [{"claim": "a", "supported": True, "source": "u1"}, {"claim": "b", "supported": False, "source": None}]
    s = supported_items(claims)
    assert len(s) == 1 and s[0]["claim"] == "a"


# --- ветки воздержания ---

def test_parse_fail_abstains():
    llm = FakeLLM("мусор-не-json")
    res = verify_then_answer(llm, "q?", "ответ", _CTX)
    assert res.abstained and res.abstain_reason == "parse_fail"


def test_too_few_supported_abstains():
    # только 1 подтверждённое при min=2
    llm = FakeLLM(_claims(
        {"claim": "a", "supported": True, "source": "u1"},
        {"claim": "b", "supported": False},
    ))
    res = verify_then_answer(llm, "q?", "ответ", _CTX, min_supported=2)
    assert res.abstained and res.abstain_reason == "too_few"


def test_empty_regeneration_abstains():
    # 2 подтверждённых, но регенерация вернула не-grounded (без валидной цитаты)
    llm = FakeLLM(
        _claims(
            {"claim": "a", "supported": True, "source": "u1"},
            {"claim": "c", "supported": True, "source": "u2"},
        ),
        regen_text="ответ без цитат",
    )
    res = verify_then_answer(llm, "q?", "ответ", _CTX)
    assert res.abstained and res.abstain_reason == "empty_regen"


def test_residual_overreach_abstains():
    # регенерация grounded, но пере-суд нашёл домысел выше порога
    llm = FakeLLM(
        _claims(
            {"claim": "a", "supported": True, "source": "u1"},
            {"claim": "c", "supported": True, "source": "u2"},
        ),
        regen_text="Обоснованно [источник: u1] и домысел.",
        recheck_json=_claims(
            {"claim": "обоснованно", "supported": True, "source": "u1"},
            {"claim": "домысел", "supported": False},
        ),  # 1 из 2 не подтверждено = 0.5 > порога 0.2
    )
    res = verify_then_answer(llm, "q?", "ответ", _CTX, residual_threshold=0.2)
    assert res.abstained and res.abstain_reason == "residual"


# --- happy + инвариант «только подтверждённые» ---

def test_happy_path_keeps_citations_and_uses_only_supported():
    llm = FakeLLM(
        _claims(
            {"claim": "NetworkClient переподключается", "supported": True, "source": "u1"},
            {"claim": "таймаут в KAFKA-1", "supported": True, "source": "u2"},
            {"claim": "это полностью решит всё", "supported": False},  # домысел — не должен попасть
        ),
        regen_text="NetworkClient переподключается [источник: u1]; таймаут [источник: u2].",
        recheck_json='{"claims":[{"claim":"x","supported":true,"source":"u1"}]}',  # 0 домысла
    )
    res = verify_then_answer(llm, "q?", "исходный переобобщённый ответ", _CTX)
    assert not res.abstained
    assert res.grounded and set(res.citations) == {"u1", "u2"}
    assert res.n_supported == 2 and res.residual_frac == 0.0
    # регенерация видела только подтверждённые тексты, не домысел
    assert "NetworkClient переподключается" in llm.gen_prompt
    assert "это полностью решит всё" not in llm.gen_prompt
