# Отчёт: качество ответов RAG

⚠️ Само-оценка: одна и та же модель генерирует ответы, судит их и пишет эталоны для размеченного среза. Оценки оптимистичны и ограничены надёжностью самой модели — это потолок метода, а не абсолютная истина.

Вопросов: 96 · размеченный срез: 0 · всего записей: 96

## Метрики

### Faithfulness (не выдумывает)

- среднее **0.588** · p10 0.00 · p50 1.00 · p90 1.00
- оценено 68 из 96 записей
- воздержаний 21/96 (21.9%) · сбоев судьи 7 (воздержание ≠ сбой)
- распределение: {'0.0-0.2': 28, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 40}
- по маршруту: factual 0.875, mixed 0.588, multihop 0.333
- по типу источника: Task 0.588

### Answer relevance (отвечает по делу)

- среднее **0.715** · p10 0.20 · p50 0.90 · p90 1.00
- оценено 91 из 96 записей
- распределение: {'0.0-0.2': 8, '0.2-0.4': 15, '0.4-0.6': 1, '0.6-0.8': 6, '0.8-1.0': 61}
- по маршруту: factual 0.925, mixed 0.688, multihop 0.736
- по типу источника: Task 0.715

### Context precision (retrieval релевантен)

- среднее **0.607** · p10 0.20 · p50 0.60 · p90 0.95
- оценено 92 из 96 записей
- распределение: {'0.0-0.2': 5, '0.2-0.4': 22, '0.4-0.6': 0, '0.6-0.8': 30, '0.8-1.0': 35}
- по маршруту: factual 0.817, mixed 0.565, multihop 0.720
- по типу источника: Task 0.607

### Answer correctness (совпадает с эталоном)

_не оценено_ (нет успешных оценок судьи)

### Context recall (эталон покрыт контекстом)

_не оценено_ (нет успешных оценок судьи)

## Примеры провалов

### Faithfulness (не выдумывает) < 0.5

- **Как обойти IllegalArgumentException с 'Invalid partition: -1' в Kafka Streams FK LeftJoin, когда Punctuator удаляет записи из state store, а CACHE_MAX_BYTES_BUFFERING_CONFIG не равен 0, и какие модули или настройки топологии будут затронуты этим багом?** — faithfulness 0.00, маршрут multihop
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20792', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/64553012/KIP-63+Unify+store+and+downstream+caching+in+streams', 'graph://module:clients', 'graph://module:connect']
- **Как перестроить публичный API стримингового ассайнора в Kafka, чтобы он стал общедоступным модулем, и какие существующие тикеты или модули будут затронуты этим изменением?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20790', 'https://issues.apache.org/jira/browse/KAFKA-20789', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62693834/Kafka+Streams+Discussions']
- **Как обойти таймаут в TimeWindowedKStreamIntegrationTest.shouldRestoreAfterJoinRestart, если он вызван не изменениями в самом тесте, а накопившимися изменениями в Kafka Streams, и какие модули KS могут быть затронуты этим сбоем?** — faithfulness 0.00, маршрут multihop
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20786', 'graph://module:connect', 'graph://module:clients', 'https://issues.apache.org/jira/browse/KAFKA-20438', 'https://issues.apache.org/jira/browse/KAFKA-20765']
- **Как интеграция api-checker в корневую задачу check через composite-include повлияет на время CI-сборки и какие модули могут сломаться, если в api-checker упадут checkstyle или unit-тесты?** — faithfulness 0.00, маршрут mixed
  источники: ['chunk:task:KAFKA-20779#2', 'chunk:task:KAFKA-20779#1', 'chunk:task:KAFKA-20328#0', 'chunk:task:KAFKA-20779#3', 'chunk:page:38571263#2']
- **Как избежать двойного ребаланса в классическом потребителе при временном отключении реплики, когда идентификатор стойки меняется на null и обратно, и можно ли использовать информацию о потере хоста репликой, чтобы не запускать лишнюю перебалансировку?** — faithfulness 0.00, маршрут factual
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20778']

### Answer relevance (отвечает по делу) < 0.5

- **Как корректно обновить share.version с 1 на 2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не поддерживают новую версию?** — answer_relevance 0.00, маршрут mixed
  источники: ['chunk:task:KAFKA-20793#0', 'chunk:page:61318265#16', 'chunk:page:61320744#15', 'chunk:page:61318265#21', 'chunk:task:KAFKA-20427#7']
- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с существующими модулями, использующими старые версии, и какие шаги нужно предпринять для предотвращения конфликтов зависимостей?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20791']
- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с другими зависимостями проекта, и какие модули могут сломаться из-за изменений в API или конфигурации?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20791']
- **Как интеграция асинхронного чтения логов в DelayedShareFetch повлияет на производительность других модулей, которые зависят от синхронного чтения, и какие шаги нужно предпринять для обратной совместимости?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20505']
- **Как обойти флакинг теста ConsumerGroupHeartbeatWithRegexWithDifferentMemberAcls, который стал нестабильным на trunk после 23 июня, и какие модули или ACL-настройки могут быть затронуты при изменении логики heartbeat для consumer group с regex-подпиской?** — answer_relevance 0.00, маршрут mixed
  источники: ['chunk:task:KAFKA-20781#0', 'chunk:task:KAFKA-20333#0', 'chunk:task:KAFKA-20563#0', 'chunk:task:KAFKA-20565#0', 'chunk:task:KAFKA-20423#0']
