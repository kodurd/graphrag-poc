# Отчёт: качество ответов RAG

⚠️ Само-оценка: одна и та же модель генерирует ответы, судит их и пишет эталоны для размеченного среза. Оценки оптимистичны и ограничены надёжностью самой модели — это потолок метода, а не абсолютная истина.

Вопросов: 300 · размеченный срез: 0 · всего записей: 300

## Метрики

### Faithfulness (не выдумывает)

- среднее **0.711** · p10 0.00 · p50 1.00 · p90 1.00
- оценено 257 из 300 записей
- воздержаний 43/300 (14.3%) · сбоев судьи 0 (воздержание ≠ сбой)
- распределение: {'0.0-0.2': 64, '0.2-0.4': 0, '0.4-0.6': 12, '0.6-0.8': 9, '0.8-1.0': 172}
- по маршруту: factual 0.873, mixed 0.711, multihop 0.571
- по типу источника: Task 0.711

### Answer relevance (отвечает по делу)

- среднее **0.565** · p10 0.00 · p50 0.70 · p90 1.00
- оценено 300 из 300 записей
- распределение: {'0.0-0.2': 77, '0.2-0.4': 22, '0.4-0.6': 23, '0.6-0.8': 55, '0.8-1.0': 123}
- по маршруту: factual 0.900, mixed 0.519, multihop 0.592
- по типу источника: Task 0.565

### Context precision (retrieval релевантен)

- среднее **0.694** · p10 0.20 · p50 0.80 · p90 1.00
- оценено 300 из 300 записей
- распределение: {'0.0-0.2': 8, '0.2-0.4': 25, '0.4-0.6': 19, '0.6-0.8': 96, '0.8-1.0': 152}
- по маршруту: factual 0.897, mixed 0.661, multihop 0.744
- по типу источника: Task 0.694

### Answer correctness (совпадает с эталоном)

_не оценено_ (нет успешных оценок судьи)

### Context recall (эталон покрыт контекстом)

_не оценено_ (нет успешных оценок судьи)

## Примеры провалов

### Faithfulness (не выдумывает) < 0.5

- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с существующими модулями, использующими старые версии, и какие шаги нужно предпринять для предотвращения конфликтов зависимостей?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20509']
- **Как перестроить публичный API стримингового ассайнора в Kafka, чтобы он стал общедоступным модулем, и какие существующие тикеты или модули будут затронуты этим изменением?** — faithfulness 0.00, маршрут mixed
  источники: ['https://cwiki.apache.org/confluence/spaces/KAFKA/pages/51807580/KIP-11+-+Authorization+Interface', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62685862/KIP-50+-+Move+Authorizer+to+o.a.k.common+package', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/62693834/Kafka+Streams+Discussions']
- **Как определить причину внезапного роста rebalance-rate-per-hour в Kafka consumer без просмотра логов, и какие метрики или API нужно добавить, чтобы различать rebalance из-за превышения max.poll.interval.ms, session timeout и вызовов enforceRebalance?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20788', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/63406974/KIP-62+Allow+consumer+to+send+heartbeats+from+a+background+thread']
- **Как корректно обновить конфигурацию с share.version=1 на share.version=2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не обновились?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20793', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61318265/KIP-32+-+Add+timestamps+to+Kafka+message']
- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с другими зависимостями проекта, и какие модули могут сломаться из-за изменений в API или конфигурации?** — faithfulness 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20510', 'https://issues.apache.org/jira/browse/KAFKA-20509']

### Answer relevance (отвечает по делу) < 0.5

- **Как корректно обновить share.version с 1 на 2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не поддерживают новую версию?** — answer_relevance 0.00, маршрут mixed
  источники: ['chunk:task:KAFKA-20793#0', 'chunk:commit:a6b1847720bcb5c16487d2a7865fd0e82c826a51#1', 'chunk:page:61318265#16', 'chunk:commit:06f699664bd016b928bf8064710f68eb7dfb136a#0', 'chunk:commit:21a080f08ca8087794da0b56ed596c79e17a5eb3#0']
- **Как обновление log4j до версии 2.25.5 повлияет на совместимость с существующими модулями, использующими старые версии, и какие шаги нужно предпринять для предотвращения конфликтов зависимостей?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20509']
- **Как перейти на публичный API стримового assignor, не сломав обратную совместимость с существующими приватными интеграциями, и какие модули будут затронуты при изменении сигнатуры методов?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20790', 'https://issues.apache.org/jira/browse/KAFKA-20789', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/65144300/KIP-67+Queryable+state+for+Kafka+Streams']
- **Как корректно обновить конфигурацию с share.version=1 на share.version=2 в стабильной сборке, чтобы не сломать обратную совместимость с клиентами, которые ещё не обновились?** — answer_relevance 0.00, маршрут mixed
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20793', 'https://cwiki.apache.org/confluence/spaces/KAFKA/pages/61318265/KIP-32+-+Add+timestamps+to+Kafka+message']
- **Как обойти IllegalArgumentException с 'Invalid partition: -1' в Kafka Streams FK LeftJoin, когда Punctuator удаляет записи из state store, а CACHE_MAX_BYTES_BUFFERING_CONFIG не равен 0, и какие модули или настройки топологии могут быть затронуты этим багом?** — answer_relevance 0.00, маршрут multihop
  источники: ['https://issues.apache.org/jira/browse/KAFKA-20792']
