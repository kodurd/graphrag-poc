"""kip_precheck: отбор безответных, эвристики жанра/причины, overlap и GO/NO-GO гейт."""

from __future__ import annotations

from eval.kip_precheck import (
    classify_absence,
    classify_genre,
    estimate_overlap,
    precheck,
)


def _r(sid, question, answer, abstained=False):
    return {"source_id": sid, "question": question, "answer": answer,
            "abstained": {"faithfulness": abstained}}


# --- Happy: только refusal-записи входят в pre-check --------------------------

def test_precheck_selects_only_refusals():
    recs = [
        _r("t1", "как добавить поле", "На основе контекста невозможно ответить."),  # refusal (text)
        _r("t2", "как настроить X", "Чтобы настроить, сделай Y [источник: u]."),     # answered
        _r("t3", "как реализовать Z", "любой", abstained=True),                      # refusal (flag)
    ]
    res = precheck(recs, kip_titles=["добавить поле в RPC"], threshold=0.0)
    assert res["n_unanswered"] == 2  # t2 (ответ) не входит
    assert res["genre"]["build_feature"] == 2


# --- Жанр вопроса ------------------------------------------------------------

def test_classify_genre_build_vs_fix():
    assert classify_genre("как добавить поле в RPC") == "build_feature"
    assert classify_genre("как обойти IllegalArgumentException в проде") == "fix_prod"
    # fix-маркер имеет приоритет над build-маркером
    assert classify_genre("как починить и добавить обработку ошибки") == "fix_prod"
    # ни build, ни fix -> "other" (не молча build_feature) — чтобы build_share был честным
    assert classify_genre("что означает поле deprecatedVersions") == "other"


# --- Причина отсутствия ------------------------------------------------------

def test_classify_absence_heuristic():
    assert classify_absence(_r("t1", "q", "Это нигде не задокументировано.")) == "undocumented"
    assert classify_absence(_r("t2", "q", "В контексте нет данных.")) == "missing_genre"


# --- Overlap -----------------------------------------------------------------

def test_estimate_overlap_bounds_and_signal():
    # Полное совпадение токенов вопроса с заголовком -> высокий overlap.
    assert estimate_overlap(["добавить поле RPC"], ["добавить поле RPC"]) == 1.0
    # Нет общих токенов -> 0.0.
    assert estimate_overlap(["совершенно другое"], ["добавить поле RPC"]) == 0.0
    # Пустой вход -> 0.0, без падения.
    assert estimate_overlap([], ["заголовок"]) == 0.0
    assert estimate_overlap(["вопрос"], []) == 0.0


# --- Порог: NO-GO ниже / GO выше --------------------------------------------

def test_threshold_gate_no_go_below():
    # Вопросы почти не пересекаются с заголовками -> overlap низкий -> NO-GO.
    recs = [_r(f"t{i}", "как добавить совершенно уникальную возможность", "невозможно")
            for i in range(3)]
    res = precheck(recs, kip_titles=["метрики коннектора kafka"], threshold=0.5)
    assert "NO-GO" in res["verdict"]


def test_threshold_gate_go_above():
    # Высокий overlap + доминирует build_feature -> GO.
    recs = [_r(f"t{i}", "как добавить метрики коннектора", "невозможно") for i in range(3)]
    res = precheck(recs, kip_titles=["добавить метрики коннектора"], threshold=0.3)
    assert res["mean_overlap"] >= 0.3
    assert res["verdict"] == "GO"


# --- Edge: пустой безответный набор -> честный результат, не падение ----------

def test_empty_unanswered_set_is_honest():
    recs = [_r("t1", "как настроить", "Готово: сделай X [источник: u].")]  # только ответы
    res = precheck(recs, kip_titles=["заголовок"], threshold=0.1)
    assert res["n_unanswered"] == 0
    assert "нет данных" in res["verdict"]


# --- Edge: нет заголовков KIP -> честный NO-GO про отсутствие данных KIP ------

def test_no_kip_titles_reports_no_data():
    recs = [_r("t1", "как добавить поле", "невозможно")]
    res = precheck(recs, kip_titles=None, threshold=0.1)
    assert res["mean_overlap"] is None
    assert "нет данных KIP" in res["verdict"]
