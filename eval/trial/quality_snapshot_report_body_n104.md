# Отчёт: качество ответов RAG

⚠️ Само-оценка: одна и та же модель генерирует ответы, судит их и пишет эталоны для размеченного среза. Оценки оптимистичны и ограничены надёжностью самой модели — это потолок метода, а не абсолютная истина.

Вопросов: 102 · размеченный срез: 0 · всего записей: 102

## Метрики

### Faithfulness (не выдумывает)

- среднее **0.776** · p10 0.00 · p50 1.00 · p90 1.00
- оценено 84 из 102 записей
- воздержаний 18/102 (17.6%) · сбоев судьи 0 (воздержание ≠ сбой)
- распределение: {'0.0-0.2': 16, '0.2-0.4': 0, '0.4-0.6': 4, '0.6-0.8': 0, '0.8-1.0': 64}
- по маршруту: factual 0.889, mixed 0.756, multihop 0.796
- по типу источника: Task 0.776

### Answer relevance (отвечает по делу)

- среднее **0.593** · p10 0.00 · p50 0.70 · p90 1.00
- оценено 102 из 102 записей
- распределение: {'0.0-0.2': 26, '0.2-0.4': 6, '0.4-0.6': 9, '0.6-0.8': 11, '0.8-1.0': 50}
- по маршруту: factual 0.956, mixed 0.539, multihop 0.677
- по типу источника: Task 0.593

### Context precision (retrieval релевантен)

- среднее **0.720** · p10 0.40 · p50 0.80 · p90 1.00
- оценено 102 из 102 записей
- распределение: {'0.0-0.2': 1, '0.2-0.4': 6, '0.4-0.6': 11, '0.6-0.8': 28, '0.8-1.0': 56}
- по маршруту: factual 0.900, mixed 0.694, multihop 0.754
- по типу источника: Task 0.720

### Answer correctness (совпадает с эталоном)

_не оценено_ (нет успешных оценок судьи)

### Context recall (эталон покрыт контекстом)

_не оценено_ (нет успешных оценок судьи)

## Примеры провалов

### Faithfulness (не выдумывает) < 0.5

- **Как корректно протестировать и задокументировать поле deprecatedVersions в RPC, чтобы избежать конфликтов с существующими модулями версионирования?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20794', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61320744/KIP-35+-+Retrieving+protocol+version']
- **Как обойти таймаут в TimeWindowedKStreamIntegrationTest.shouldRestoreAfterJoinRestart, если он вызван не изменениями в самом тесте, а накопившимися изменениями в Kafka Streams, и какие модули KS могут быть затронуты этим сбоем?** — faithfulness 0.00, маршрут multihop
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20786', 'graph://module:connect', 'graph://module:clients', 'https://issues.apache.org/jira/browse/KAFKA-20438', 'https://issues.apache.org/jira/browse/KAFKA-20765']
- **Как обойти IllegalStateException в StagedMergeIterator при использовании после закрытия store, чтобы получить InvalidStateStoreException, и какие модули будут затронуты этим изменением?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20777', 'https://issues.apache.org/jira/browse/KAFKA-20760']
- **Как обойти несовместимость Node#idString при обновлении модуля, чтобы не сломать существующие интеграции, и какие другие тикеты или модули это затронет?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20775', 'https://issues.apache.org/jira/browse/KAFKA-20393', 'https://issues.apache.org/jira/browse/KAFKA-20532']
- **Как реализовать readOnly-режим в RocksDB SegmentedStore, чтобы избежать блокировок при параллельном чтении, и какие компромиссы по производительности это вызовет?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20498']

### Answer relevance (отвечает по делу) < 0.5

- **Как корректно обновить share.version с 1 на 2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не поддерживают новую версию?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61318265/KIP-32+-+Add+timestamps+to+Kafka+message']
- **Как перестроить публичный API стримингового ассайнора в Kafka, чтобы он стал общедоступным модулем, и какие существующие тикеты или модули будут затронуты этим изменением?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://cwiki.apache.org/confluence/spaces/KAFKA/pages/51807580/KIP-11+-+Authorization+Interface', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62685862/KIP-50+-+Move+Authorizer+to+o.a.k.common+package', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62693834/Kafka+Streams+Discussions']
- **Как корректно обновить конфигурацию с share.version=1 на share.version=2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не обновились?** — answer_relevance 0.00, маршрут mixed
  источники: ['chunk:commit:a6b1847720bcb5c16487d2a7865fd0e82c826a51#1', 'chunk:task:KAFKA-20793#0', 'chunk:commit:21a080f08ca8087794da0b56ed596c79e17a5eb3#0', 'chunk:commit:06f699664bd016b928bf8064710f68eb7dfb136a#0', 'chunk:page:61318265#16']
- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с другими зависимостями проекта, и какие модули могут сломаться из-за изменений в API или конфигурации?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20510', 'https://issues.apache.org/jira/browse/KAFKA-20509']
- **Как перестроить публичный API стрим-ассайнора в соответствии с KIP-1357, чтобы не сломать обратную совместимость с существующими модулями и тикетами?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20789', 'https://issues.apache.org/jira/browse/KAFKA-20790', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/50859233/Kafka+Improvement+Proposals']
