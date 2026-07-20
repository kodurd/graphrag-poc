"""Gradio-интерфейс для ручной проверки GraphRAG.

Тонкая обёртка над `graphrag.service`. Запуск: `graphrag serve-ui`
(нужен extra ui: `uv sync --extra ui`). Требует поднятый Neo4j
(`docker compose up -d`) и, для вкладки Ask, LLM_API_KEY в .env либо
provider: ollama.
"""

from __future__ import annotations

from graphrag import service


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _ask(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "_Введите вопрос._"
    try:
        res = service.ask_question(question)
    except service.ServiceError as e:
        return f"⚠ **Ошибка:** {e}"
    except Exception as e:  # noqa: BLE001 — показать пользователю, не ронять UI
        return f"⚠ **Непредвиденная ошибка:** {e}"

    lines = [f"**Маршрут:** `{res.route}`", "", res.text, ""]
    if res.citations:
        lines.append("**Цитаты:**")
        lines += [f"- {c}" for c in res.citations]
    else:
        lines.append("_Цитат нет._")
    if res.hallucinated_citations:
        lines.append("")
        lines.append(f"⚠ Отброшены недостоверные ссылки: {res.hallucinated_citations}")
    if not res.grounded:
        lines.append("")
        lines.append("⚠ Ответ без валидных источников — доверять с осторожностью.")
    return "\n".join(lines)


def _impact(log_text: str, use_llm: bool) -> str:
    log_text = log_text or ""
    if not log_text.strip():
        return "_Вставьте текст лога._"
    try:
        res = service.analyze_log(log_text, use_llm=use_llm)
    except service.ServiceError as e:
        return f"⚠ **Ошибка:** {e}"
    except Exception as e:  # noqa: BLE001
        return f"⚠ **Непредвиденная ошибка:** {e}"

    ent = res["entities"]
    out = [
        "### Сущности (кандидаты, до сверки с графом)",
        f"- Модули: `{ent['modules']}`",
        f"- Исключения: `{ent['exceptions']}`",
        "",
        "### Упавшие модули",
        (", ".join(f"`{m}`" for m in res["failing"]) or "_нет_"),
        "",
        "### Затронуто (обход DEPENDS_ON)",
    ]
    out += [f"- {_md_escape(m['name'])}" for m in res["affected_modules"]] or ["_нет_"]
    out += ["", "### Владельцы"]
    out += [
        f"- {_md_escape(o['name'])} ({_md_escape(o['module'])})" for o in res["owners"]
    ] or ["_нет_"]
    out += ["", "### Тикеты «уже чинили»"]
    out += [
        f"- **{t['key']}** [{t['status']}] {_md_escape(t['summary'])} — [ссылка]({t['uri']})"
        for t in res["related_tasks"]
    ] or ["_нет_"]
    out += ["", "### Страницы вики"]
    out += [
        f"- {_md_escape(p['title'])} — [ссылка]({p['uri']})" for p in res["related_pages"]
    ] or ["_нет_"]
    return "\n".join(out)


def _status() -> str:
    try:
        ok = service.check_health()
    except Exception as e:  # noqa: BLE001
        return f"⚠ **Neo4j:** ошибка — {e}"
    badge = "🟢 OK" if ok else "🔴 НЕДОСТУПЕН (`docker compose up -d`)"
    lines = [f"**Neo4j:** {badge}", "", "**Конфигурация:**", "```"]
    lines += service.info_lines()
    lines.append("```")
    return "\n".join(lines)


def build_app():
    """Собрать Gradio Blocks (импорт gradio ленивый — extra ui)."""
    import gradio as gr

    with gr.Blocks(title="GraphRAG PoC") as demo:
        gr.Markdown("# GraphRAG PoC — ручная проверка")

        with gr.Tab("Ask"):
            q = gr.Textbox(label="Вопрос", placeholder="Например: почему брокер недоступен?", lines=2)
            ask_btn = gr.Button("Спросить", variant="primary")
            ask_out = gr.Markdown()
            ask_btn.click(_ask, inputs=q, outputs=ask_out)
            q.submit(_ask, inputs=q, outputs=ask_out)

        with gr.Tab("Log → Impact"):
            log = gr.Textbox(
                label="Лог с ошибкой",
                placeholder="Вставьте stack trace / строки лога…",
                lines=12,
            )
            use_llm = gr.Checkbox(label="Дополнять извлечение сущностей через LLM", value=False)
            imp_btn = gr.Button("Анализировать", variant="primary")
            imp_out = gr.Markdown()
            imp_btn.click(_impact, inputs=[log, use_llm], outputs=imp_out)

        with gr.Tab("Health & Info"):
            st_out = gr.Markdown()
            st_btn = gr.Button("Проверить", variant="primary")
            st_btn.click(_status, outputs=st_out)
            demo.load(_status, outputs=st_out)

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    build_app().launch(server_name=host, server_port=port, share=share)
