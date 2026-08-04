"""Локальный веб-UI для быстрой человеческой разметки gold (без зависимостей).

По одному айтему на экран: вопрос + ответ + контекст, три метрики быстрыми кнопками
(0 / .25 / .5 / .75 / 1) и горячими клавишами. Пишет eval/trial/gold_labels.json
инкрементально (только полностью размеченные тройки → файл всегда валиден для load_gold).

Горячие клавиши: faithfulness = 1..5 · relevance = q w e r t · precision = a s d f g ·
стрелки ←/→ — предыдущий/следующий.

Запуск:  uv run python -m eval.gold_ui    (откроет http://localhost:8765)
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_TEMPLATE = "eval/trial/gold_labels.json.template"
_LABELS = "eval/trial/gold_labels.json"
_PORT = 8765
_METRICS = ("faithfulness", "answer_relevance", "context_precision")

REFUSE_HINT = "невозможно/недостаточно/нет данных"


def _load_items() -> list[dict]:
    """Айтемы для разметки: записи снимка, отобранные в gold-template."""
    from eval.human_gold import is_refusal

    sids = list(json.loads(Path(_TEMPLATE).read_text(encoding="utf-8-sig")).keys())
    by_sid = {r.get("source_id"): r for r in
              json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]}
    items = []
    for sid in sids:
        r = by_sid.get(sid)
        if not r:
            continue
        items.append({
            "sid": sid,
            "question": r.get("question") or "",
            "answer": r.get("answer") or "",
            "context": r.get("context_texts") or [],
            "abstained": is_refusal(r),  # флаг воздержания ИЛИ отказ-фраза в тексте
        })
    return items


def _read_labels() -> dict:
    p = Path(_LABELS)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
    return {}


_PAGE = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<title>Gold разметка</title><style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:0 auto;padding:16px;background:#0f1117;color:#e6e6e6}
 .bar{height:8px;background:#222;border-radius:4px;overflow:hidden;margin:8px 0}
 .bar>i{display:block;height:100%;background:#4caf50;width:0}
 .meta{color:#8a94a6;font-size:13px}
 .q{font-weight:600;font-size:17px;margin:10px 0}
 .a{background:#161a24;border:1px solid #2a3040;border-radius:8px;padding:10px;max-height:260px;overflow:auto;white-space:pre-wrap}
 .ctx{background:#12151d;border:1px solid #232838;border-radius:8px;padding:8px;margin-top:8px;max-height:150px;overflow:auto;font-size:13px;color:#aab}
 .m{margin:14px 0}
 .m b{display:inline-block;width:220px}
 .btns button{width:52px;height:34px;margin:2px;border:1px solid #384;background:#1b1f2a;color:#ddd;border-radius:6px;cursor:pointer}
 .btns button.sel{background:#4caf50;color:#06210a;border-color:#4caf50;font-weight:700}
 .nav{margin-top:16px;display:flex;gap:8px;align-items:center}
 .nav button{padding:8px 16px;border-radius:6px;border:1px solid #345;background:#1b1f2a;color:#ddd;cursor:pointer}
 .tag{background:#5a2;color:#0a0;font-size:11px;padding:1px 6px;border-radius:4px;margin-left:6px;color:#eaffea;background:#274}
 kbd{background:#222;border:1px solid #444;border-radius:3px;padding:0 4px;font-size:11px;color:#9ab}
 .done{color:#4caf50}
</style></head><body>
<div class=meta><span id=pos></span> · размечено <span id=cnt>0</span>/<span id=tot>0</span></div>
<div class=bar><i id=prog></i></div>
<div class=meta id=sid></div>
<div class=q id=q></div>
<div class=a id=a></div>
<details><summary class=meta>контекст (фрагменты)</summary><div class=ctx id=ctx></div></details>
<div id=metrics></div>
<div class=nav>
 <button onclick=go(-1)>← Назад</button>
 <button onclick=go(1)>Дальше →</button>
 <span class=meta>клавиши: faithfulness <kbd>1-5</kbd> · relevance <kbd>q w e r t</kbd> · precision <kbd>a s d f g</kbd> · <kbd>←</kbd><kbd>→</kbd></span>
</div>
<script>
const V=[0,.25,.5,.75,1];
const KEYS={faithfulness:['1','2','3','4','5'],answer_relevance:['q','w','e','r','t'],context_precision:['a','s','d','f','g']};
const NAMES={faithfulness:'faithfulness (не выдумал)',answer_relevance:'answer_relevance (в тему)',context_precision:'context_precision (контекст релевантен)'};
let items=[],labels={},idx=0;
async function boot(){const d=await (await fetch('/data')).json();items=d.items;labels=d.labels;document.getElementById('tot').textContent=items.length;render();}
function cur(){return items[idx];}
function render(){const it=cur();
 document.getElementById('pos').textContent=(idx+1)+' / '+items.length;
 document.getElementById('sid').innerHTML=it.sid+(it.abstained?'<span class=tag>REFUSAL</span>':'');
 document.getElementById('q').textContent=it.question;
 document.getElementById('a').textContent=it.answer;
 document.getElementById('ctx').textContent=it.context.join('\\n— ');
 const lab=labels[it.sid]||{};let h='';
 for(const m of Object.keys(NAMES)){h+='<div class=m><b>'+NAMES[m]+'</b><span class=btns>';
  V.forEach((v,i)=>{const s=(lab[m]===v)?'sel':'';h+='<button class="'+s+'" onclick="setv(\\''+m+'\\','+v+')">'+v+'</button>';});
  h+='</span></div>';}
 document.getElementById('metrics').innerHTML=h;
 let done=Object.values(labels).filter(l=>Object.keys(NAMES).every(m=>typeof l[m]==='number')).length;
 document.getElementById('cnt').textContent=done;
 document.getElementById('cnt').className=done===items.length?'done':'';
 document.getElementById('prog').style.width=(100*done/items.length)+'%';
}
async function setv(m,v){const it=cur();(labels[it.sid]=labels[it.sid]||{})[m]=v;render();
 await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(labels)});}
function go(d){idx=Math.max(0,Math.min(items.length-1,idx+d));render();window.scrollTo(0,0);}
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowRight'){go(1);return;} if(e.key==='ArrowLeft'){go(-1);return;}
 for(const m in KEYS){const i=KEYS[m].indexOf(e.key.toLowerCase());if(i>=0){setv(m,V[i]);return;}}});
boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    items: list[dict] = []

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self._send(200, _PAGE, "text/html")
        elif self.path == "/data":
            self._send(200, json.dumps({"items": self.items, "labels": _read_labels()}))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path != "/save":
            self._send(404, "{}")
            return
        n = int(self.headers.get("Content-Length", 0))
        labels = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        # В файл — только ПОЛНОСТЬЮ размеченные тройки (валидно для load_gold).
        complete = {sid: sc for sid, sc in labels.items()
                    if all(isinstance(sc.get(m), (int, float)) for m in _METRICS)}
        Path(_LABELS).write_text(
            json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
        self._send(200, json.dumps({"saved": len(complete)}))

    def log_message(self, *a):  # тихо
        pass


def main() -> int:
    Handler.items = _load_items()
    if not Handler.items:
        print("gold-ui: нет айтемов — сначала `python -m eval.human_gold build`")
        return 1
    url = f"http://localhost:{_PORT}"
    print(f"gold-ui: {len(Handler.items)} айтемов · {url} · Ctrl-C чтобы остановить\n"
          f"  пишет {_LABELS} инкрементально (полные тройки)", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    HTTPServer(("localhost", _PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
