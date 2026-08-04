# Отчёт: качество ответов RAG

⚠️ Само-оценка: одна и та же модель генерирует ответы, судит их и пишет эталоны для размеченного среза. Оценки оптимистичны и ограничены надёжностью самой модели — это потолок метода, а не абсолютная истина.

Вопросов: 293 · размеченный срез: 0 · всего записей: 293

## Метрики

### Faithfulness (не выдумывает)

- среднее **0.747** · p10 0.00 · p50 1.00 · p90 1.00
- оценено 240 из 293 записей
- воздержаний 50/293 (17.1%) · сбоев судьи 3 (воздержание ≠ сбой)
- распределение: {'0.0-0.2': 51, '0.2-0.4': 0, '0.4-0.6': 11, '0.6-0.8': 11, '0.8-1.0': 167}
- по маршруту: factual 0.798, mixed 0.760, multihop 0.638
- по типу источника: Task 0.747

### Answer relevance (отвечает по делу)

- среднее **0.581** · p10 0.00 · p50 0.70 · p90 1.00
- оценено 290 из 293 записей
- распределение: {'0.0-0.2': 66, '0.2-0.4': 23, '0.4-0.6': 24, '0.6-0.8': 55, '0.8-1.0': 122}
- по маршруту: factual 0.867, mixed 0.540, multihop 0.625
- по типу источника: Task 0.581

### Context precision (retrieval релевантен)

- среднее **0.695** · p10 0.20 · p50 0.80 · p90 1.00
- оценено 292 из 293 записей
- распределение: {'0.0-0.2': 8, '0.2-0.4': 22, '0.4-0.6': 20, '0.6-0.8': 93, '0.8-1.0': 149}
- по маршруту: factual 0.900, mixed 0.662, multihop 0.744
- по типу источника: Task 0.695

### Answer correctness (совпадает с эталоном)

_не оценено_ (нет успешных оценок судьи)

### Context recall (эталон покрыт контекстом)

_не оценено_ (нет успешных оценок судьи)

## Примеры провалов

### Faithfulness (не выдумывает) < 0.5

- **Как корректно протестировать и задокументировать поле deprecatedVersions в RPC, чтобы избежать конфликтов с существующими модулями версионирования?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20794', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61320744/KIP-35+-+Retrieving+protocol+version']
- **Как корректно обновить share.version с 1 на 2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не поддерживают новую версию?** — faithfulness 0.00, маршрут mixed
  источники: ['https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61318265/KIP-32+-+Add+timestamps+to+Kafka+message']
- **Как обойти таймаут в TimeWindowedKStreamIntegrationTest.shouldRestoreAfterJoinRestart, если он вызван не изменениями в самом тесте, а накопившимися изменениями в Kafka Streams, и какие модули KS могут быть затронуты этим сбоем?** — faithfulness 0.00, маршрут multihop
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20786', 'graph://module:connect', 'graph://module:clients', 'https://issues.apache.org/jira/browse/KAFKA-20438', 'https://issues.apache.org/jira/browse/KAFKA-20765']
- **Как обойти IllegalStateException в StagedMergeIterator при использовании после закрытия store, чтобы получить InvalidStateStoreException, и какие модули будут затронуты этим изменением?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20777', 'https://issues.apache.org/jira/browse/KAFKA-20760']
- **Как реализовать readOnly-режим в RocksDB SegmentedStore, чтобы избежать блокировок при параллельном чтении, и какие компромиссы по производительности это вызовет?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20498']

### Answer relevance (отвечает по делу) < 0.5

- **Как корректно обновить share.version с 1 на 2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не поддерживают новую версию?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61318265/KIP-32+-+Add+timestamps+to+Kafka+message']
- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с существующими модулями, использующими старые версии, и какие шаги нужно предпринять для предотвращения конфликтов зависимостей?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20509']
- **Как перейти на публичный API стримового assignor, не сломав обратную совместимость с существующими приватными интеграциями, и какие модули будут затронуты при изменении сигнатуры методов?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20790', 'https://issues.apache.org/jira/browse/KAFKA-20789', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/65144300/KIP-67+Queryable+state+for+Kafka+Streams']
- **Как перестроить публичный API стримингового ассайнора в Kafka, чтобы он стал общедоступным модулем, и какие существующие тикеты или модули будут затронуты этим изменением?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://cwiki.apache.org/confluence/spaces/KAFKA/pages/51807580/KIP-11+-+Authorization+Interface', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62685862/KIP-50+-+Move+Authorizer+to+o.a.k.common+package', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62693834/Kafka+Streams+Discussions']
- **Как корректно обновить конфигурацию с share.version=1 на share.version=2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не обновились?** — answer_relevance 0.00, маршрут mixed
  источники: ['chunk:commit:a6b1847720bcb5c16487d2a7865fd0e82c826a51#1', 'chunk:task:KAFKA-20793#0', 'chunk:commit:21a080f08ca8087794da0b56ed596c79e17a5eb3#0', 'chunk:commit:06f699664bd016b928bf8064710f68eb7dfb136a#0', 'chunk:page:61318265#16']
