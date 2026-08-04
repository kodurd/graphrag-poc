"""Диагностика 8 faith-сбоев Qwen: захватить СЫРОЙ ответ судьи и понять причину.

cross_judge хранит только faith=None, теряя сырой текст. Здесь для сбойных записей
(faith==None & abstained==False) джойним answer/context из снимка по (source_id,
question), зовём judge-LLM НА СЫРОМ (до extract_json) и печатаем ответ + что из него
извлекается. Так видно: невалидный JSON (нужен фикс парсинга) vs валидный JSON без
ключа `faithfulness` (нужен фикс промпта/обработки).

Запуск (нужен JUDGE_API_KEY): uv run --extra ml python -m eval.diagnose_judge
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import _FAITH_PROMPT

_CROSS = "eval/trial/cross_judge_results.json"
_SNAPSHOT = "eval/trial/quality_snapshot_results.json"


def failing_keys(cross: list[dict]) -> list[tuple]:
    """(source_id, question) записей faith==None & abstained==False (сбой, не воздержание)."""
    return [(r.get("source_id"), r.get("question")) for r in cross
            if r["qwen"].get("faithfulness") is None and not r.get("qwen_faith_abstained")]


def main() -> int:
    from graphrag.config import load_settings
    from graphrag.llm import build_llm

    s = load_settings()
    if not (s.llm.judge_api_key and s.llm.judge_model):
        print("diagnose: нет JUDGE_* в .env")
        return 1

    cross = json.loads(Path(_CROSS).read_text(encoding="utf-8-sig"))["records"]
    snap = {(r.get("source_id"), r.get("question")): r
            for r in json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]}
    keys = failing_keys(cross)
    print(f"diagnose: сбойных записей {len(keys)} · судья {s.llm.judge_model}\n", flush=True)

    judge = build_llm(s.llm, role="judge")
    for i, k in enumerate(keys, 1):
        rec = snap.get(k)
        if not rec:
            print(f"[{i}] {k[0]}: НЕТ в снимке (join не сошёлся)"); continue
        ctx = "\n".join(f"- {t}" for t in (rec.get("context_texts") or []))
        prompt = f"{_FAITH_PROMPT}КОНТЕКСТ:\n{ctx}\n\nОТВЕТ:\n{rec.get('answer', '')}"
        raw = judge.complete(prompt)  # СЫРОЙ текст до extract_json
        parsed = judge.extract_json(prompt)
        print(f"[{i}] {k[0]}")
        print(f"    RAW (первые 220): {raw[:220]!r}")
        print(f"    extract_json -> {parsed!r} (dict={isinstance(parsed, dict)}, "
              f"есть 'faithfulness'={isinstance(parsed, dict) and 'faithfulness' in parsed})\n",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
