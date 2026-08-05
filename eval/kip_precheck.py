"""Pre-check покрытия KIP — дешёвый гейт ДО любого ingest (U1 плана).

Прежде чем тащить тела KIP в граф и мерить дельту, надо оценить: есть ли вообще
что ловить? Берём ранее-безответные how-to вопросы из снимка (фильтр `is_refusal`
из `eval/human_gold.py`) и характеризуем их по трём осям:

  1. Причина отсутствия ответа — `classify_absence`: «нет нужного жанра в корпусе»
     (missing_genre) vs «не задокументировано нигде» (undocumented).
  2. Жанр вопроса — `classify_genre`: «устроить фичу» (build_feature) vs
     «починить в проде» (fix_prod).
  3. Топикальный overlap вопросов с заголовками KIP — `estimate_overlap`:
     дешёвая лексическая близость в [0,1] (без эмбеддингов).

Все три — ЧИСТЫЕ детерминированные функции над записями (и опц. списком заголовков
KIP), чтобы тест не требовал LLM/эмбеддингов. Классификаторы — заведомо грубые
эвристики-заглушки; позже их можно заменить LLM-разметкой / эмбеддинг-overlap, не
трогая интерфейс.

`precheck` сводит это в вердикт GO/NO-GO: GO, только если средний overlap не ниже
порога И доля build_feature нетривиальна. Иначе — валидный отрицательный результат
(гейт к dev@), без траты на ingest.

Запуск: uv run python -m eval.kip_precheck
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from eval.human_gold import is_refusal

_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_TITLES = "eval/trial/kip_titles.json"
_REPORT = "eval/trial/kip_precheck_report.md"

# Порог по умолчанию для среднего overlap (эвристика, подбирается при прогоне).
_THRESHOLD = 0.15
# Минимальная доля build_feature, чтобы считать её нетривиальной для GO.
_BUILD_SHARE_MIN = 0.2

# --- Ось 2: жанр вопроса (эвристика на ключевых основах слов) -----------------
# fix_prod проверяем ПЕРВЫМ: «обойти exception в проде» важнее, чем совпавшее «как».
_FIX_RE = re.compile(
    r"ошибк|exception|обойти|упал|падает|fix|почин|сбой|падени|исключени|краш|crash", re.I
)
_BUILD_RE = re.compile(
    r"добав|устро|реализ|как сделать|создать|настро|внедри|поддержк", re.I
)

# --- Ось 1: причина отсутствия (эвристика на тексте ответа) --------------------
# Маркеры «нигде не задокументировано» → undocumented; иначе считаем, что нужного
# ЖАНРА (how-to) нет в корпусе тикетов/коммитов → missing_genre.
_UNDOC_RE = re.compile(
    r"не задокументир|нигде не|не описан|не существу|нет такого|не найден", re.I
)

# --- Ось 3: токенизация для лексического overlap -------------------------------
_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁ]+")
_STOPWORDS = {
    "как", "что", "для", "при", "или", "это", "под", "над", "без", "изи",
    "если", "когда", "чтобы", "the", "and", "for", "how", "why", "can",
}


def classify_absence(record: dict) -> str:
    """Причина отсутствия ответа — грубая эвристика на тексте ответа.

    Возвращает "undocumented" при маркерах «нигде не задокументировано», иначе
    "missing_genre" (корпус = тикеты/коммиты, нет how-to-жанра). Это заглушка —
    позже её место может занять LLM-разметка; интерфейс (record -> str) тот же.
    """
    text = (record.get("answer") or "")[:600]
    return "undocumented" if _UNDOC_RE.search(text) else "missing_genre"


def classify_genre(question: str) -> str:
    """Жанр вопроса — детерминированная эвристика на ключевых основах слов.

    "fix_prod" при маркерах починки/сбоя (проверяются первыми), иначе "build_feature".
    Заглушка под возможную замену LLM-классификатором.
    """
    q = question or ""
    if _FIX_RE.search(q):
        return "fix_prod"
    if _BUILD_RE.search(q):
        return "build_feature"
    return "build_feature"


def _tokens(text: str) -> set[str]:
    return {
        t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if len(t) >= 3 and t not in _STOPWORDS
    }


def estimate_overlap(questions: list[str], kip_titles: list[str]) -> float:
    """Дешёвый лексический overlap вопросов с заголовками KIP, в [0,1].

    Для каждого вопроса — максимальная по заголовкам доля его токенов, покрытых
    токенами заголовка; результат — среднее по вопросам. Без эмбеддингов: это
    сознательно дешёвая версия, эмбеддинг-близость — возможное улучшение с тем же
    интерфейсом (questions, titles) -> float. Пустой вход → 0.0.
    """
    if not questions or not kip_titles:
        return 0.0
    title_tok = [_tokens(t) for t in kip_titles]
    title_tok = [t for t in title_tok if t]
    if not title_tok:
        return 0.0
    scores: list[float] = []
    for q in questions:
        qt = _tokens(q)
        if not qt:
            scores.append(0.0)
            continue
        best = max(len(qt & tt) / len(qt) for tt in title_tok)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def precheck(records: list[dict], kip_titles: list[str] | None, *,
             threshold: float = _THRESHOLD) -> dict:
    """Сводит характеристику безответного набора в вердикт GO/NO-GO.

    В pre-check попадают только `is_refusal` записи. Считаем split по причине и по
    жанру, средний overlap с заголовками KIP и вердикт: "GO" только если
    overlap >= threshold И доля build_feature нетривиальна; иначе "NO-GO ...".
    """
    unanswered = [r for r in records if is_refusal(r)]
    n = len(unanswered)

    cause = {"missing_genre": 0, "undocumented": 0}
    genre = {"build_feature": 0, "fix_prod": 0}
    for r in unanswered:
        cause[classify_absence(r)] += 1
        genre[classify_genre(r.get("question") or "")] += 1

    titles = list(kip_titles or [])
    questions = [r.get("question") or "" for r in unanswered]
    overlap = estimate_overlap(questions, titles) if titles else None
    build_share = genre["build_feature"] / n if n else 0.0

    if n == 0:
        verdict = "NO-GO (нет данных: пустой безответный набор)"
    elif not titles or overlap is None:
        verdict = "NO-GO (нет данных KIP)"
    elif overlap >= threshold and build_share >= _BUILD_SHARE_MIN:
        verdict = "GO"
    else:
        verdict = "NO-GO (гейт к dev@)"

    return {
        "n_unanswered": n,
        "cause": cause,
        "genre": genre,
        "build_share": build_share,
        "mean_overlap": overlap,
        "threshold": threshold,
        "n_kip_titles": len(titles),
        "verdict": verdict,
    }


def _load_titles(path: str = _TITLES) -> list[str] | None:
    """Читает опциональный список заголовков KIP. Формат: JSON-список строк или
    объект {"titles": [...]}. Отсутствие файла — не ошибка (возвращает None)."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("titles") or []
    return [str(t) for t in data]


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_report(res: dict) -> str:
    """Отчёт: split по причине, split по жанру, оценка overlap и явный GO/NO-GO."""
    n = res["n_unanswered"]
    cause, genre = res["cause"], res["genre"]

    def share(v: int) -> str:
        return f"{v} ({(v / n if n else 0.0):.0%})"

    lines = [
        "# Pre-check покрытия KIP (гейт до ingest)",
        "",
        "> Дешёвая оценка ДО трат: можно ли KIP-жанром закрыть ранее-безответные "
        "how-to вопросы. Классификаторы — грубые эвристики (заменяемы LLM/эмбеддингами).",
        "",
        f"Безответных (is_refusal) записей: **{n}** · заголовков KIP: "
        f"{res['n_kip_titles']}",
        "",
        "## Причина отсутствия ответа (эвристика по тексту ответа)",
        "",
        "| причина | доля |",
        "|---|---|",
        f"| нет жанра в корпусе (missing_genre) | {share(cause['missing_genre'])} |",
        f"| не задокументировано (undocumented) | {share(cause['undocumented'])} |",
        "",
        "## Жанр вопроса (эвристика по ключевым словам)",
        "",
        "| жанр | доля |",
        "|---|---|",
        f"| устроить фичу (build_feature) | {share(genre['build_feature'])} |",
        f"| починить в проде (fix_prod) | {share(genre['fix_prod'])} |",
        "",
        "## Топикальный overlap с заголовками KIP",
        "",
        f"- средний лексический overlap: **{_fmt(res['mean_overlap'])}** "
        f"(порог {res['threshold']:.3f})",
        f"- доля build_feature: {res['build_share']:.0%} "
        f"(минимум для GO: {_BUILD_SHARE_MIN:.0%})",
        "",
        "## Вердикт",
        "",
        f"**{res['verdict']}**",
        "",
        "> GO — если overlap не ниже порога И доля build_feature нетривиальна. "
        "Иначе валидный отрицательный результат: гейт к dev@ без траты на ingest.",
    ]
    return "\n".join(lines)


def main() -> int:
    snap = Path(_SNAPSHOT)
    if not snap.exists():
        print(f"kip-precheck: NO snapshot {_SNAPSHOT} -- run eval.quality_snapshot first")
        return 1

    records = json.loads(snap.read_text(encoding="utf-8-sig"))["records"]
    titles = _load_titles()

    res = precheck(records, titles)
    Path(_REPORT).write_text(render_report(res), encoding="utf-8")

    tag = res["verdict"].split()[0]  # "GO" | "NO-GO" -- ASCII-only for cp1251 stdout
    titles_note = "no-kip-titles" if titles is None else f"kip-titles={len(titles)}"
    print(f"DONE kip-precheck -> {_REPORT} "
          f"(verdict={tag}, unanswered={res['n_unanswered']}, {titles_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
