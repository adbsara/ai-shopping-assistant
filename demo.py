"""
Gradio demo for Hugging Face Spaces (ZeroGPU).

Run locally:
    python demo.py
"""
import spaces  # must be imported before torch on ZeroGPU

import html
import os
import time

import gradio as gr
import torch

from app.embeddings import Embedder
from app.vector_store import ProductStore

# ---------- Backend ----------

ON_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
DEVICE = "cuda" if (ON_ZERO_GPU or torch.cuda.is_available()) else "cpu"

embedder = Embedder(device=DEVICE)
store = ProductStore()
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))


@spaces.GPU(duration=15)
def embed_query(text: str) -> list[float]:
    """Runs on a GPU on Hugging Face; a normal function on your computer."""
    return embedder.embed_one(text)


# ---------- HTML rendering ----------

def render_cards(hits) -> str:
    if not hits:
               return '<div class="empty">😕 محصول مرتبطی در فروشگاه پیدا نشد.<br>فعلاً فروشگاه ما فقط محصولات ورزشی و طبیعت‌گردی دارد.</div>'
    cards = []
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        image = (
            f'<img src="{html.escape(p["image"])}" alt="" loading="lazy">'
            if p.get("image") else '<div class="no-img">🛍️</div>'
        )
        match = max(0, min(100, round(hit.score * 100)))
        cards.append(f"""
        <div class="card">
          <div class="img">
            <span class="rank">#{rank}</span>
            {image}
          </div>
          <div class="body">
            <span class="tag">{html.escape(p.get("category") or "")}</span>
            <div class="title" title="{html.escape(p["title"])}">{html.escape(p["title"])}</div>
            <div class="meta">
              <span class="price">${p["price"]:.2f}</span>
              <span class="rating">★ {p["rating"]} <small>({p["rating_count"]:,})</small></span>
            </div>
            <div class="match">
              <div class="match-bar"><div class="match-fill" style="width:{match}%"></div></div>
              <div class="match-label">Match {match}%</div>
            </div>
          </div>
        </div>""")
    return '<div class="cards">' + "".join(cards) + "</div>"


def render_stats(embed_ms: float, search_ms: float, count: int) -> str:
    return f"""
    <div class="stats">
      <span class="chip">📦 {count} نتیجه</span>
      <span class="chip">🧠 Embedding: {embed_ms:.0f} ms</span>
      <span class="chip">🔎 Vector search: {search_ms:.0f} ms</span>
    </div>"""


def search(query: str, max_price: float, top_k: int):
    query = (query or "").strip()[:500]
    if len(query) < 2:
        return '<div class="empty">✍️ لطفاً چیزی که دنبالش هستید را بنویسید.</div>', ""

    t0 = time.perf_counter()
    vector = embed_query(query)
    t1 = time.perf_counter()
    hits = store.search(vector, limit=int(top_k),
                    max_price=max_price if max_price > 0 else None,
                    score_threshold=MIN_SCORE)

    t2 = time.perf_counter()

    return render_cards(hits), render_stats((t1 - t0) * 1000, (t2 - t1) * 1000, len(hits))


# ---------- Look and feel ----------

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="pink",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Vazirmatn"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1120px !important; margin: auto !important; direction: rtl; }
footer { display: none !important; }

#hero {
  text-align: center; padding: 44px 24px 36px; border-radius: 28px; margin-bottom: 8px;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  box-shadow: 0 20px 50px -20px rgba(79, 70, 229, .6);
}
#hero h1 { color: #fff !important; font-size: 2.3rem; font-weight: 800; margin: 0 0 8px; }
#hero p  { color: rgba(255,255,255,.88) !important; font-size: 1.05rem; margin: 0; }
#hero .badges { margin-top: 18px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
#hero .badges span {
  color: #fff; font-size: .78rem; padding: 5px 12px; border-radius: 99px;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25); direction: ltr;
}

#search-box { align-items: stretch !important; }
#query textarea { font-size: 1.1rem !important; padding: 14px 16px !important; }
#search-btn { min-height: 54px; font-size: 1.05rem; font-weight: 700; }
#suggest button { font-size: .85rem !important; }
#filters input[type=range] { direction: ltr; }

.stats { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 4px; }
.chip {
  font-size: .82rem; padding: 6px 12px; border-radius: 99px;
  background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
}

.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 18px; }
.card {
  display: flex; flex-direction: column; overflow: hidden; border-radius: 18px;
  background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  transition: transform .2s ease, box-shadow .2s ease; animation: rise .35s ease both;
}
.card:hover { transform: translateY(-5px); box-shadow: 0 18px 40px -18px rgba(0,0,0,.35); }
.card .img {
  position: relative; height: 190px; background: #fff;
  display: flex; align-items: center; justify-content: center; padding: 14px;
}
.card .img img { max-width: 100%; max-height: 100%; object-fit: contain; }
.card .no-img { font-size: 3rem; }
.card .rank {
  position: absolute; top: 10px; left: 10px; font-size: .75rem; font-weight: 700; color: #fff;
  padding: 3px 9px; border-radius: 99px; background: linear-gradient(135deg, #4f46e5, #db2777);
}
.card .body {
  direction: ltr; text-align: left; padding: 14px 16px 16px;
  display: flex; flex-direction: column; gap: 10px; flex: 1;
}
.card .tag {
  align-self: flex-start; font-size: .7rem; padding: 3px 9px; border-radius: 99px;
  background: var(--background-fill-secondary); opacity: .85;
}
.card .title {
  font-weight: 600; font-size: .92rem; line-height: 1.45;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.card .meta { display: flex; justify-content: space-between; align-items: center; margin-top: auto; }
.card .price { font-size: 1.25rem; font-weight: 800; color: #16a34a; }
.card .rating { color: #f59e0b; font-weight: 600; font-size: .9rem; }
.card .rating small { color: var(--body-text-color-subdued); font-weight: 400; }
.match-bar { height: 6px; border-radius: 99px; background: var(--border-color-primary); overflow: hidden; }
.match-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #4f46e5, #db2777); }
.match-label { font-size: .72rem; margin-top: 5px; color: var(--body-text-color-subdued); }

.empty { text-align: center; padding: 48px 16px; font-size: 1.05rem; opacity: .75; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
"""

HERO = """
<div id="hero">
  <h1>🛍️ دستیار خرید هوشمند</h1>
  <p>هر چیزی که لازم دارید را به زبان خودتان بنویسید؛ فارسی یا انگلیسی</p>
  <div class="badges">
    <span>bge-m3 multilingual embeddings</span>
    <span>Qdrant vector search</span>
    <span>ZeroGPU</span>
  </div>
</div>
"""

EXAMPLES = [
    ("🎒 کوله سبک کوهنوردی", "کوله پشتی سبک برای کوهنوردی یک روزه", 80),
    ("🎣 هدیه برای ماهیگیر", "هدیه برای کسی که ماهیگیری دوست دارد", 0),
    ("⛺ کمپینگ در هوای سرد", "وسایل کمپینگ برای هوای خیلی سرد", 0),
    ("🧘 تمرین یوگا در خانه", "تجهیزات یوگا برای تمرین در خانه", 50),
]

# Gradio 6 moved theme/css from Blocks() to launch(); support both versions
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
STYLE = {"theme": THEME, "css": CSS}

with gr.Blocks(title="AI Shopping Assistant", **(STYLE if GRADIO_MAJOR < 6 else {})) as demo:
    gr.HTML(HERO)

    with gr.Row(elem_id="search-box"):
        query = gr.Textbox(
            elem_id="query", show_label=False, rtl=True, scale=5, lines=1, max_lines=1,
            placeholder="مثلاً: یه هدیه برای پدرم که عاشق کوهنورده…",
        )
        button = gr.Button("جستجو ✨", variant="primary", elem_id="search-btn", scale=1)

    with gr.Row(elem_id="suggest"):
        example_buttons = [gr.Button(label, size="sm", variant="secondary") for label, _, _ in EXAMPLES]

    with gr.Accordion("⚙️ فیلترها", open=False, elem_id="filters"):
        with gr.Row():
            max_price = gr.Slider(0, 300, value=0, step=5, label="حداکثر قیمت (دلار) — صفر یعنی بدون محدودیت")
            top_k = gr.Slider(1, 20, value=8, step=1, label="تعداد نتایج")

    stats = gr.HTML()
    results = gr.HTML()

    inputs, outputs = [query, max_price, top_k], [results, stats]
    button.click(search, inputs, outputs)
    query.submit(search, inputs, outputs)

    for btn, (_, text, price) in zip(example_buttons, EXAMPLES):
        btn.click(lambda t=text, p=price: (t, p), outputs=[query, max_price]).then(search, inputs, outputs)


if __name__ == "__main__":
    demo.launch(**(STYLE if GRADIO_MAJOR >= 6 else {}))
