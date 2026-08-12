"""Диагностик отказов «достали-но-отказ»: содержит ли извлечённый контекст ответ?

Берёт из AFTER-снимка отказы (is_refusal), у которых в top-k был page/KIP-чанк, и
спрашивает судью: достаточно ли контекста, чтобы ответить. Разводит два рычага:
  answerable=yes  -> генерация СЛИШКОМ осторожна (ответ есть, а система отказалась)
  answerable=no   -> чанкинг/жанр не донёс ответ (контент по теме, но без процедуры)

Запуск (нужен JUDGE_API_KEY): uv run python -m eval.diagnose_refusals
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.human_gold import is_refusal

_AFTER = "eval/trial/quality_snapshot_results_after.json"
_OUT = "eval/trial/refusal_diag_report.md"
_PROMPT = (
    "Ты оцениваешь ДОСТАТОЧНОСТЬ контекста. Дан вопрос и извлечённый контекст.\n"
    "Ответь СТРОГО JSON: {\"answerable\": \"yes\"|\"partial\"|\"no\"}.\n"
    "yes — в контексте есть прямой ответ на вопрос;\n"
    "partial — есть частичная/косвенная информация, но не полный ответ;\n"
    "no — контекст по теме, но ответа на заданный вопрос в нём нет.\n\n"
)


def judge_answerable(llm, question: str, context_texts: list[str]) -> str:
    ctx = "\n".join(f"- {t}" for t in context_texts)
    prompt = f"{_PROMPT}ВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{ctx}"
    try:
        data = llm.extract_json(prompt)
        v = (data or {}).get("answerable")
        return v if v in ("yes", "partial", "no") else "err"
    except Exception:
        return "err"


def main() -> int:
    from graphrag.config import load_settings
    from graphrag.llm import build_llm

    s = load_settings()
    if not (s.llm.judge_api_key and s.llm.judge_model):
        print("diag: нет JUDGE_* в .env")
        return 1

    recs = json.loads(Path(_AFTER).read_text(encoding="utf-8-sig"))["records"]
    def has_page(r):
        return any(str(c).startswith("chunk:page:") for c in (r.get("context_ids") or []))
    target = [r for r in recs if is_refusal(r) and has_page(r)]
    print(f"diag: 'достали-но-отказ' {len(target)} · судья {s.llm.judge_model}", flush=True)

    judge = build_llm(s.llm, role="judge")
    rows, verdicts = [], Counter()
    for i, r in enumerate(target, 1):
        v = judge_answerable(judge, r.get("question") or "", r.get("context_texts") or [])
        verdicts[v] += 1
        rows.append((v, r.get("question") or ""))
        if i % 10 == 0 or i == len(target):
            print(f"  [{i}/{len(target)}] {dict(verdicts)}", flush=True)

    n = len(target)
    lines = [
        "# Диагностик отказов «достали-но-отказ»", "",
        f"Отказов с page/KIP-чанком в top-k: **{n}**. Судья: {s.llm.judge_model}.", "",
        "| вердикт контекста | n | доля | рычаг |", "|---|---|---|---|",
        f"| **yes** (ответ ЕСТЬ в контексте) | {verdicts['yes']} | {verdicts['yes']/n:.0%} | генерация слишком осторожна -> ослабить порог/промпт |",
        f"| partial | {verdicts['partial']} | {verdicts['partial']/n:.0%} | пограничные — часть выиграется тем же рычагом |",
        f"| **no** (ответа в контексте нет) | {verdicts['no']} | {verdicts['no']/n:.0%} | чанкинг/жанр не донёс -> нарезка KIP / новые источники |",
        f"| err | {verdicts['err']} | {verdicts['err']/n:.0%} | сбой судьи |",
        "", "## Примеры yes (ответ был, а система отказалась)", "",
    ]
    for v, q in rows:
        if v == "yes":
            lines.append(f"- {q[:110]}")
    Path(_OUT).write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE diag -> {_OUT}: yes={verdicts['yes']} partial={verdicts['partial']} "
          f"no={verdicts['no']} err={verdicts['err']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
