# Замер связи тикет→фикс-коммит

**R0 (предпосылка):** коммитов-чанков в индексе 0 — **НЕТ: замер невалиден** — коммиты не индексированы, already_retrieved структурно 0.

Проваленных how-to: 48. Классы: {'no_fix': 37, 'surfaceable': 11}.
- доли: surfaceable 23% · already_retrieved 0% · no_fix 77%

**Вердикт: INGEST** — no_fix 77% ≥ 50% — связанного коммита нет; нужен ingest ⚠️ НЕ доверенный: R0 не прошёл.

## Спот-чек текстов коммитов (already_retrieved)
  (нет)

## Спот-чек (surfaceable)
  - [task:KAFKA-20791] KAFKA-20791: Bump log4j to 2.25.5 (#22794)
  - [task:KAFKA-20791] KAFKA-20791: Bump log4j to 2.25.5 (#22794)
  - [task:KAFKA-20786] KAFKA-20786: Bound IntegrationTestUtils.readRecords on real elapsed time (#22790)
  - [task:KAFKA-20783] KAFKA-20783: Integrate async log reader fetch in DelayedShareFetch (1/N) (#22780)

⚠️ Достижимость — по пре-реранк пулу (`_candidate_pool`); прод-top_k отмечен справочно (`in_prod_topk`). MENTIONS-полнота зависит от ссылки KAFKA-NNNN в subject коммита.