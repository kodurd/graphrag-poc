# Замер связи тикет→фикс-коммит

**R0 (предпосылка):** коммитов-чанков в индексе 4793 — OK.

Проваленных how-to: 48. Классы: {'no_fix': 36, 'already_retrieved': 10, 'surfaceable': 2}.
- доли: surfaceable 4% · already_retrieved 21% · no_fix 75%

**Вердикт: INGEST** — no_fix 75% ≥ 50% — связанного коммита нет; нужен ingest

## Спот-чек текстов коммитов (already_retrieved)
  - [task:KAFKA-20791] KAFKA-20791: Bump log4j to 2.25.5 (#22794)

Upgrade log4j2 from 2.25.4 to 2.25.5.

Reviewers: Mickael Maison <mickael.maison@gmail.com>

Signed-off-by: Federico Valeri <fedevaleri@gmail.com>
  - [task:KAFKA-20791] KAFKA-20791: Bump log4j to 2.25.5 (#22794)

Upgrade log4j2 from 2.25.4 to 2.25.5.

Reviewers: Mickael Maison <mickael.maison@gmail.com>

Signed-off-by: Federico Valeri <fedevaleri@gmail.com>
  - [task:KAFKA-20783] KAFKA-20783: Integrate async log reader fetch in DelayedShareFetch (1/N) (#22780)

The PR integrates async processing in DelayedShareFetch which ll help
move all the remote fetches out of DelayedShare
  - [task:KAFKA-20776] KAFKA-20703: IQv2 TimestampedKeyWithHeadersQuery for headers-aware key-value stores (KIP-1356) (#22666)

This PR is the **first increment of KIP-1356** — the key/value point
query — for  `TimestampedK

## Спот-чек (surfaceable)
  - [task:KAFKA-20786] KAFKA-20786: Bound IntegrationTestUtils.readRecords on real elapsed time (#22790)

readRecords() polls up to `waitTime` by counting a fixed 100ms per
iteration rather than measuring the wall clock. A 
  - [task:KAFKA-20736] KAFKA-20736: Improve removal of unassigned partitions from share sessions (#22720)

When all partitions in a share session are unassigned with no pending
acknowledgements to be sent, no ShareFetch req

⚠️ Достижимость — по пре-реранк пулу (`_candidate_pool`); прод-top_k отмечен справочно (`in_prod_topk`). MENTIONS-полнота зависит от ссылки KAFKA-NNNN в subject коммита.