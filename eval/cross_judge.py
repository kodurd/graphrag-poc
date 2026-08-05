"""Кросс-модельный re-judge: независимый судья (Qwen) оценивает СОХРАНЁННЫЕ ответы.

Разрыв само-оценки: DeepSeek и генерил, и судил. Здесь второй судья (judge-роль,
напр. Qwen через OpenAI-совместимый эндпоинт) оценивает те же (вопрос, ответ,
`context_texts`) — БЕЗ перегенерации и БЕЗ рефетча по id (иначе контекст отличался бы
от того, что видел генератор). DeepSeek-оценки берутся из снимка как есть.

Гейт: пустой `JUDGE_API_KEY` => судья откатился бы на DeepSeek (не независим) — падаем.

Запуск (нужен заполненный JUDGE_* в .env):
    uv run --extra ml python -m eval.cross_judge
"""

from __future__ import annotations

import json
from pathlib import Path

from graphrag.llm.base import LLMClient

from eval.metrics import (
    judge_answer_relevance,
    judge_context_precision,
    judge_faithfulness,
)

_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_RESULTS = "eval/trial/cross_judge_results.json"


def cross_judge_records(records: list[dict], judge_llm: LLMClient) -> list[dict]:
    """Оценивает сохранённые ответы независимым судьёй. Чистая (llm — зависимость).

    Для каждой записи берёт СОХРАНЁННЫЕ answer + context_texts (не рефетчит) и
    прогоняет три judge_*; кладёт Qwen-метрики рядом с DeepSeek из снимка.
    """
    out: list[dict] = []
    for r in records:
        answer = r.get("answer") or ""
        ctx = r.get("context_texts") or []
        faith, faith_abst = judge_faithfulness(judge_llm, answer, ctx)
        out.append({
            "source_id": r.get("source_id"),
            "question": r.get("question"),
            "route": r.get("route"),
            "grounded": r.get("grounded"),
            "abstained": r.get("abstained"),
            "deepseek": r.get("metrics", {}),
            "qwen": {
                "faithfulness": faith,
                "answer_relevance": judge_answer_relevance(judge_llm, r.get("question") or "", answer),
                "context_precision": judge_context_precision(judge_llm, r.get("question") or "", ctx),
            },
            "qwen_faith_abstained": faith_abst,
        })
    return out


def main() -> int:
    from graphrag.config import load_settings
    from graphrag.llm import build_llm

    s = load_settings()
    # Гейт: без независимого judge-ключа судья = DeepSeek (не разрывает круг).
    if not (s.llm.judge_api_key and s.llm.judge_model):
        print("cross-judge: не задан независимый судья — впиши JUDGE_API_KEY / "
              "JUDGE_BASE_URL / JUDGE_MODEL (Qwen) в .env. Тихого фолбэка на DeepSeek нет.")
        return 1

    snap = Path(_SNAPSHOT)
    if not snap.exists():
        print(f"cross-judge: нет снимка {snap} — сначала eval.quality_snapshot")
        return 1
    records = json.loads(snap.read_text(encoding="utf-8-sig"))["records"]
    if records and "context_texts" not in records[0]:
        print("cross-judge: в снимке нет context_texts — пересними quality_snapshot "
              "(нужен патч U2), иначе судья оценит другой контекст.")
        return 1
    print(f"cross-judge: записей {len(records)} · судья {s.llm.judge_model}", flush=True)

    judge = build_llm(s.llm, role="judge")
    results = []
    for i, r in enumerate(records, 1):
        results.extend(cross_judge_records([r], judge))
        if i % 20 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}]", flush=True)

    Path(_RESULTS).write_text(
        json.dumps({"records": results, "judge_model": s.llm.judge_model},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DONE cross-judge: {len(results)} -> {_RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
