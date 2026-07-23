"""CLI-каркас."""

from __future__ import annotations

import typer

from graphrag.config import load_settings

app = typer.Typer(help="GraphRAG PoC — код+тикеты+вики над Apache Kafka.")


@app.command()
def info() -> None:
    """Показать активную конфигурацию (без секретов)."""
    from graphrag.service import info_lines

    for line in info_lines():
        typer.echo(line)


@app.command()
def health() -> None:
    """Проверить связь с Neo4j."""
    from graphrag.service import check_health

    ok = check_health()
    typer.echo("Neo4j: OK" if ok else "Neo4j: НЕДОСТУПЕН")
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def ingest(
    git: bool = typer.Option(True, help="Выгрузить git-историю + импорты"),
    jira: bool = typer.Option(False, help="Выгрузить тикеты JIRA (сеть)"),
    confluence: bool = typer.Option(False, help="Выгрузить страницы Confluence (сеть)"),
    out_dir: str = typer.Option("data/intermediate", help="Каталог JSONL"),
) -> None:
    """Выгрузка источников в промежуточный JSONL."""
    from pathlib import Path

    from graphrag.connectors.confluence import ConfluenceConnector
    from graphrag.connectors.git import GitConnector
    from graphrag.connectors.jira import JiraConnector

    s = load_settings()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if git:
        if not Path(s.corpus.repo_path).exists():
            typer.echo(f"git: пропуск — репозиторий не найден: {s.corpus.repo_path}")
        else:
            conn = GitConnector(s.corpus.repo_path, s.corpus.components, s.corpus.since)
            stats = conn.extract(f"{out_dir}/git.jsonl")
            typer.echo(f"git: {stats}")

    if jira:
        conn = JiraConnector(s.sources.jira_base, s.sources.jira_project, max_issues=s.sources.max_issues)
        typer.echo(f"jira: {conn.extract(f'{out_dir}/jira.jsonl')}")

    if confluence:
        conn = ConfluenceConnector(
            s.sources.confluence_base, s.sources.confluence_space, max_pages=s.sources.max_pages
        )
        typer.echo(f"confluence: {conn.extract(f'{out_dir}/confluence.jsonl')}")


@app.command()
def build(
    in_dir: str = typer.Option("data/intermediate", help="Каталог JSONL от коннекторов"),
    index: bool = typer.Option(True, help="Строить векторный индекс чанков"),
    labels: str = typer.Option(
        "", help="Ограничить индексируемые типы, напр. 'Task,Page' (пусто = все)"
    ),
    batch_size: int = typer.Option(64, help="Размер батча эмбеддинга (память/скорость)"),
) -> None:
    """Построение скелета графа и векторного индекса в Neo4j."""
    import glob

    from graphrag.embeddings import build_embedder
    from graphrag.graph import Neo4jConnection
    from graphrag.graph.schema import apply_schema
    from graphrag.graph.skeleton import load_jsonl
    from graphrag.index.vector import VectorIndexer, collect_text_nodes

    s = load_settings()
    paths = sorted(glob.glob(f"{in_dir}/*.jsonl"))
    if not paths:
        typer.echo(f"build: нет JSONL в {in_dir} — сначала `graphrag ingest`")
        raise typer.Exit(code=1)

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            typer.echo("build: Neo4j недоступен — `docker compose up -d`")
            raise typer.Exit(code=1)
        apply_schema(conn)
        stats = load_jsonl(conn, *paths)
        typer.echo(f"build: скелет загружен {stats} из {len(paths)} файлов")

        if index:
            embedder = build_embedder(s.embeddings)
            indexer = VectorIndexer(
                conn, embedder, size=s.chunk.size, overlap=s.chunk.overlap
            )
            indexer.ensure_index()
            label_list = [x.strip() for x in labels.split(",") if x.strip()] or None
            nodes = collect_text_nodes(conn, label_list)
            typer.echo(f"build: текстовых узлов {len(nodes)} (labels={label_list or 'все'})")
            istats = indexer.index_nodes(nodes, batch_size=batch_size, progress=True)
            conn.run("CALL db.awaitIndexes(300)")
            typer.echo(f"build: индекс {istats} (embedder={s.embeddings.provider})")


@app.command("log-impact")
def log_impact(
    log_file: str = typer.Argument(..., help="Путь к файлу лога с ошибкой"),
    use_llm: bool = typer.Option(False, help="Дополнять извлечение сущностей через LLM"),
) -> None:
    """Анализ 'лог с ошибкой -> что затронуто'."""
    from pathlib import Path

    from graphrag.service import ServiceError, analyze_log

    text = Path(log_file).read_text(encoding="utf-8", errors="ignore")
    try:
        res = analyze_log(text, use_llm=use_llm)
    except ServiceError as e:
        typer.echo(f"log-impact: {e}")
        raise typer.Exit(code=1)

    ent = res["entities"]
    typer.echo(f"Сущности: modules={ent['modules']} exceptions={ent['exceptions']}")
    typer.echo(f"Упавшие модули: {res['failing']}")
    typer.echo("Затронуто:")
    for m in res["affected_modules"]:
        typer.echo(f"  - {m['name']}")
    typer.echo("Владельцы:")
    for o in res["owners"]:
        typer.echo(f"  - {o['name']} ({o['module']})")
    typer.echo("Связанные тикеты (уже чинили):")
    for t in res["related_tasks"]:
        typer.echo(f"  - {t['key']} [{t['status']}] {t['summary']} -> {t['uri']}")
    typer.echo("Страницы вики:")
    for p in res["related_pages"]:
        typer.echo(f"  - {p['title']} -> {p['uri']}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Вопрос к базе"),
) -> None:
    """Ответ на вопрос с цитированием (вектор-поиск -> контекст -> LLM)."""
    from graphrag.service import ServiceError, ask_question

    try:
        res = ask_question(question)
    except ServiceError as e:
        typer.echo(f"ask: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"[маршрут: {res.route}]")
    typer.echo(res.text)
    typer.echo("")
    typer.echo(f"Цитаты: {res.citations}")
    if res.hallucinated_citations:
        typer.echo(f"⚠ отброшены недостоверные ссылки: {res.hallucinated_citations}")
    if not res.grounded:
        typer.echo("⚠ ответ без валидных источников — доверять с осторожностью")


@app.command()
def enrich(
    resolve: bool = typer.Option(True, help="После обогащения склеить дубли (entity resolution)"),
    limit: int = typer.Option(0, help="Ограничить число текстов (0 = все)"),
) -> None:
    """LLM-обогащение графа + entity resolution."""
    from graphrag.graph import Neo4jConnection
    from graphrag.graph.enrich import Enricher
    from graphrag.graph.resolve import EntityResolver
    from graphrag.llm import build_llm

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        typer.echo("enrich: не задан LLM_API_KEY (.env) для извлечения. Ключ или provider: ollama.")
        raise typer.Exit(code=1)

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            typer.echo("enrich: Neo4j недоступен — `docker compose up -d`")
            raise typer.Exit(code=1)
        llm = build_llm(s.llm, role="extraction")
        stats = Enricher(conn, llm).enrich_graph(limit=limit or None)
        typer.echo(f"enrich: {stats}")
        if resolve:
            rstats = EntityResolver(conn).resolve()
            typer.echo(f"resolve: {rstats}")


@app.command()
def sync(
    in_dir: str = typer.Option("data/intermediate", help="Каталог JSONL от коннекторов"),
    manifest_path: str = typer.Option("data/intermediate/manifest.json", help="Файл манифеста"),
) -> None:
    """Инкрементальный ре-sync: пересчёт только затронутого."""
    import glob

    from graphrag.graph import Neo4jConnection
    from graphrag.graph.schema import apply_schema
    from graphrag.incremental.sync import IncrementalSync, load_manifest, save_manifest
    from graphrag.intermediate import read_jsonl

    s = load_settings()
    paths = sorted(p for p in glob.glob(f"{in_dir}/*.jsonl"))
    new_records = [r for p in paths for r in read_jsonl(p)]
    if not new_records:
        typer.echo(f"sync: нет JSONL в {in_dir}")
        raise typer.Exit(code=1)

    prev = load_manifest(manifest_path)
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            typer.echo("sync: Neo4j недоступен — `docker compose up -d`")
            raise typer.Exit(code=1)
        apply_schema(conn)
        stats, new_manifest = IncrementalSync(conn).apply(new_records, prev)
    save_manifest(new_manifest, manifest_path)
    typer.echo(f"sync: {stats}")


@app.command("ru-validate")
def ru_validate(
    k: int = typer.Option(3, help="recall@k"),
    dataset: str = typer.Option("examples/ru_synthetic.json", help="Путь к RU-датасету"),
) -> None:
    """Сравнить эмбеддеры на русском срезе (bge-m3 vs multilingual-e5)."""
    from graphrag.embeddings.embedder import SentenceTransformerEmbedder

    from eval.ru_validation import compare_embedders, load_dataset

    ds = load_dataset(dataset)
    embedders = {
        "bge-m3": SentenceTransformerEmbedder("BAAI/bge-m3"),
        "multilingual-e5": SentenceTransformerEmbedder("intfloat/multilingual-e5-base"),
    }
    try:
        scores = compare_embedders(ds, embedders, k=k)
    except ImportError:
        typer.echo("ru-validate: нужен extra ml — `uv sync --extra ml`")
        raise typer.Exit(code=1)
    for name, score in scores.items():
        typer.echo(f"{name}: recall@{k}={score:.3f}")


@app.command("eval")
def eval_cmd(
    limit: int = typer.Option(100, help="Максимум вопросов в golden set"),
) -> None:
    """Golden set из связей JIRA + retrieval-метрики."""
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.graph import Neo4jConnection
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.golden_set import build_from_graph, evaluate_retrieval

    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            typer.echo("eval: Neo4j недоступен — `docker compose up -d`")
            raise typer.Exit(code=1)
        golden = build_from_graph(conn, limit=limit)
        if not golden:
            typer.echo("eval: golden set пуст — сначала наполните граф (ingest+build)")
            raise typer.Exit(code=1)
        retr = HybridRetriever(
            conn, build_embedder(s.embeddings), build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            min_rerank_score=s.retrieval.min_rerank_score,
        )
        report = evaluate_retrieval(retr, golden)
    typer.echo(
        f"eval: n={report['n']} precision={report['precision']:.3f} "
        f"recall={report['recall']:.3f} f1={report['f1']:.3f}"
    )


@app.command("eval-quality")
def eval_quality(
    n: int = typer.Option(200, help="Сколько вопросов сгенерировать (reference-free)"),
    slice_size: int = typer.Option(40, help="Размер размеченного среза (эталоны)"),
    out_dir: str = typer.Option("eval", help="Куда класть артефакты прогона"),
    threshold: float = typer.Option(0.5, help="Порог «провала» для примеров в отчёте"),
) -> None:
    """Массовая оценка качества ответов: генерация -> прогон -> отчёт."""
    import json
    from pathlib import Path

    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval import labeled_gen, question_gen
    from eval.golden_set import build_from_graph
    from eval.quality_eval import run_quality_eval, run_retrieval_eval
    from eval.quality_report import render_report

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        typer.echo("eval-quality: не задан LLM_API_KEY (.env).")
        raise typer.Exit(code=1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            typer.echo("eval-quality: Neo4j недоступен — `docker compose up -d`")
            raise typer.Exit(code=1)

        llm = build_llm(s.llm, role="generation")
        questions = question_gen.generate_from_graph(conn, llm, limit=n)
        labeled = labeled_gen.generate_from_graph(conn, llm, limit=slice_size)
        typer.echo(f"eval-quality: вопросов {len(questions)}, размеченный срез {len(labeled)}")
        (out / "questions_real.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "labeled_real.json").write_text(
            json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        retr = HybridRetriever(
            conn,
            build_embedder(s.embeddings),
            build_reranker(s.reranker),
            top_k=s.retrieval.top_k,
            rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops,
            min_rerank_score=s.retrieval.min_rerank_score,
        )
        results = run_quality_eval(
            retr, llm, questions, labeled,
            faithfulness_samples=s.eval.faithfulness_judge_samples,
            faithfulness_temperature=s.eval.faithfulness_judge_temperature,
        )
        retrieval = run_retrieval_eval(retr, build_from_graph(conn, limit=200))

    (out / "quality_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(results, retrieval, threshold=threshold)
    (out / "quality_report.md").write_text(report, encoding="utf-8")
    typer.echo(f"eval-quality: отчёт -> {out / 'quality_report.md'}")


@app.command("serve-ui")
def serve_ui(
    host: str = typer.Option("127.0.0.1", help="Адрес прослушивания"),
    port: int = typer.Option(7860, help="Порт"),
    share: bool = typer.Option(False, help="Публичная ссылка Gradio"),
) -> None:
    """Запустить Gradio-интерфейс для ручной проверки (extra ui)."""
    try:
        from graphrag.ui import launch
    except ModuleNotFoundError:
        typer.echo("serve-ui: нужен extra ui — `uv sync --extra ui`")
        raise typer.Exit(code=1)
    typer.echo(f"UI: http://{host}:{port}")
    launch(host=host, port=port, share=share)


if __name__ == "__main__":
    app()
