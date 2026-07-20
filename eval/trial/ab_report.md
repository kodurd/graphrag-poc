# A/B: lexical vs cross-encoder

⚠️ Само-оценка: одна модель и отвечает, и судит. Дельты индикативны.

Воздержания: lexical 8/13 (62%) · cross-encoder 5/13 (38%)

## answer_relevance

- lexical: n=5 mean=0.92 · cross-encoder: n=8 mean=0.88
- совместно-отвечённых: 5 (K=5) · вердикт: **not_supported**

| вопрос | lexical | cross-encoder | дельта |
|---|---|---|---|
| Как исправить расхождение между метриками num-open-iterators | 0.90 | 0.70 | -0.20 |
| Как переложить реализацию методов RequestConvertToJson#reque | 0.70 | 0.90 | +0.20 |
| Как определить причину внезапного роста rebalance-rate-per-h | 1.00 | 0.90 | -0.10 |
| How can sink connectors track offsets for records dropped by | 1.00 | 1.00 | +0.00 |
| Why does the num-open-iterators metric under-count iterators | 1.00 | 1.00 | +0.00 |

## context_precision

- lexical: n=5 mean=0.61 · cross-encoder: n=8 mean=0.58
- совместно-отвечённых: 5 (K=5) · вердикт: **not_supported**

| вопрос | lexical | cross-encoder | дельта |
|---|---|---|---|
| Как исправить расхождение между метриками num-open-iterators | 0.60 | 0.60 | +0.00 |
| Как переложить реализацию методов RequestConvertToJson#reque | 0.60 | 0.30 | -0.30 |
| Как определить причину внезапного роста rebalance-rate-per-h | 0.75 | 0.90 | +0.15 |
| How can sink connectors track offsets for records dropped by | 0.90 | 0.90 | +0.00 |
| Why does the num-open-iterators metric under-count iterators | 0.20 | 0.20 | +0.00 |

## faithfulness

- lexical: n=5 mean=0.70 · cross-encoder: n=8 mean=0.69
- совместно-отвечённых: 5 (K=5) · вердикт: **not_supported**

| вопрос | lexical | cross-encoder | дельта |
|---|---|---|---|
| Как исправить расхождение между метриками num-open-iterators | 0.50 | 0.50 | +0.00 |
| Как переложить реализацию методов RequestConvertToJson#reque | 0.00 | 0.00 | +0.00 |
| Как определить причину внезапного роста rebalance-rate-per-h | 1.00 | 0.00 | -1.00 |
| How can sink connectors track offsets for records dropped by | 1.00 | 1.00 | +0.00 |
| Why does the num-open-iterators metric under-count iterators | 1.00 | 1.00 | +0.00 |
