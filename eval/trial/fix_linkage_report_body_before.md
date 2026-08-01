# Замер связи тикет→фикс-коммит

**R0 (предпосылка):** коммитов-чанков в индексе 3788 — OK.

Проваленных how-to: 48. Классы: {'no_fix': 37, 'already_retrieved': 8, 'surfaceable': 3}.
- доли: surfaceable 6% · already_retrieved 17% · no_fix 77%

**Вердикт: INGEST** — no_fix 77% ≥ 50% — связанного коммита нет; нужен ingest

## Спот-чек текстов коммитов (already_retrieved)
  - [task:KAFKA-20791] KAFKA-20791: Bump log4j to 2.25.5 (#22794)
  - [task:KAFKA-20791] KAFKA-20791: Bump log4j to 2.25.5 (#22794)
  - [task:KAFKA-20783] KAFKA-20783: Integrate async log reader fetch in DelayedShareFetch (1/N) (#22780)
  - [task:KAFKA-20783] KAFKA-20783: Integrate async log reader fetch in DelayedShareFetch (1/N) (#22780)

## Спот-чек (surfaceable)
  - [task:KAFKA-20786] KAFKA-20786: Bound IntegrationTestUtils.readRecords on real elapsed time (#22790)
  - [task:KAFKA-20736] KAFKA-20736: Improve removal of unassigned partitions from share sessions (#22720)
  - [task:KAFKA-20723] KAFKA-20723: Add fail fast check for share group dlq(...). [1/N] (#22635)

⚠️ Достижимость — по пре-реранк пулу (`_candidate_pool`); прод-top_k отмечен справочно (`in_prod_topk`). MENTIONS-полнота зависит от ссылки KAFKA-NNNN в subject коммита.