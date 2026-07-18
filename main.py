#!/usr/bin/env python3
"""
Claude News Digest
-----------------
Google ニュース RSS(日英)から Anthropic / Claude 関連記事を直近48時間分収集し、
LLM で「無関係除外・重複統合・翻訳・全体要約・トピック分類・重要トピック抽出」を
一括処理して、GitHub Pages 向けのダーク HTML を docs/index.html に書き出す。

設計方針:
  - HTML / SVG 図解はすべてコード側で生成する(LLM には JSON だけ返させる)。
  - LLM プロバイダは summarize() で抽象化。gemini / groq / mistral に対応。
    どれを使っているかは Secrets(環境変数)でのみ決まり、コード・出力・ログには出さない。
  - LLM が失敗した日も、記事リストのみでページを成立させる(壊さない)。

環境変数(すべて GitHub Actions の Repository secrets から注入):
  LLM_PROVIDER  必須  gemini | groq | mistral
  LLM_MODEL     必須  各プロバイダのモデル名
  LLM_API_KEY   必須  API キー
  NEWS_QUERY    任意  既定 "Anthropic Claude"

依存: feedparser, requests
"""

import os
import re
import sys
import json
import html
import datetime as dt
import urllib.parse

import requests
import feedparser

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:
    JST = dt.timezone(dt.timedelta(hours=9))

# ---------------------------------------------------------------- 設定
ROOT    = os.path.dirname(os.path.abspath(__file__))
DOCS    = os.path.join(ROOT, "docs")
ARCHIVE = os.path.join(DOCS, "archive")

QUERY          = os.environ.get("NEWS_QUERY", "Anthropic Claude")
LOOKBACK_HOURS = 48    # 直近この時間内の記事のみ対象
MAX_ITEMS      = 20    # 掲載する最大記事数

# トピック分類(固定5カテゴリ)。増減はこのリストを編集する。
CATEGORIES = [
    "新モデル・アップデート",
    "Claude Code・開発ツール",
    "規制・ポリシー・安全性",
    "研究・技術",
    "その他",
]

FEEDS = [
    ("ja", "JP", "JP:ja"),      # 日本語
    ("en-US", "US", "US:en"),   # 英語
]


# ---------------------------------------------------------------- RSS 取得
def _feed_url(hl, gl, ceid):
    q = urllib.parse.quote(QUERY)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def _norm_title(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


def fetch_articles():
    """直近 LOOKBACK_HOURS の記事を日英フィードから取得(単純なタイトル重複は除去)。"""
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=LOOKBACK_HOURS)
    items, seen_titles = [], set()

    for hl, gl, ceid in FEEDS:
        feed = feedparser.parse(_feed_url(hl, gl, ceid))
        for e in feed.entries:
            pub = None
            if getattr(e, "published_parsed", None):
                pub = dt.datetime(*e.published_parsed[:6], tzinfo=dt.timezone.utc)
            if pub and pub < cutoff:
                continue
            title = getattr(e, "title", "").strip()
            nt = _norm_title(title)
            if not title or nt in seen_titles:
                continue
            seen_titles.add(nt)
            snippet = re.sub(r"<[^>]+>", " ", getattr(e, "summary", "") or "")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            source = ""
            if getattr(e, "source", None) and getattr(e.source, "title", None):
                source = e.source.title
            items.append({
                "title": title,
                "link": getattr(e, "link", ""),
                "published": pub.isoformat() if pub else "",
                "source": source,
                "snippet": snippet[:400],
            })
    return items


# ---------------------------------------------------------------- LLM(プロバイダ抽象化)
# 対応プロバイダ: gemini / groq / mistral
# どれを使うかは環境変数 LLM_PROVIDER の値でのみ決まる(コードに既定値は書かない)。

def _build_prompt(items):
    listing = "\n".join(
        f'[{i}] 見出し: {it["title"]}\n    媒体: {it["source"] or "不明"}\n    抜粋: {it["snippet"] or "(なし)"}'
        for i, it in enumerate(items)
    )
    cats = "、".join(f'"{c}"' for c in CATEGORIES)
    return (
        "あなたはAIニュースのキュレーターです。以下は Google ニュース検索で集めた、"
        "Anthropic 社の AI「Claude」に関係しうる記事リストです(英語または日本語)。\n\n"
        "次の手順で処理してください。\n"
        "1. Claude(Anthropic のAI)に実質的に関係しない記事(同名の人物・画家・無関係な話題など)を除外する。\n"
        "2. 実質的に同じニュースを報じる重複記事を1件に統合する(最も情報量の多いものを代表にする)。\n"
        f"3. 残った記事は重要度順に最大 {MAX_ITEMS} 件まで。各記事に日本語見出し(title_ja)と、"
        f"次の固定カテゴリのうち最も近い1つ(category)を割り当てる: {cats}。"
        "どれにも当たらなければ「その他」。\n"
        "4. 残った全記事を横断して、その日の動向をまとめる:\n"
        "   - headline: 全体を一言で表す日本語見出し(30字以内)\n"
        "   - summary: 3〜4文の日本語要約。抜粋に書かれた範囲でまとめ、推測を混ぜない。\n"
        "5. 特に重要なトピックを2〜4個抽出する(key_topics)。各要素は\n"
        "   title(15字以内の短い題)と note(1文の説明)を持つ。重要度の高い順に並べる。\n\n"
        "出力は次のスキーマの JSON オブジェクトのみ。前置き・コードフェンス・説明を一切付けない。\n"
        "{\n"
        '  "headline": "",\n'
        '  "summary": "",\n'
        '  "key_topics": [{"title": "", "note": ""}],\n'
        '  "articles": [{"index": <元の番号(整数)>, "title_ja": "", "category": ""}]\n'
        "}\n\n"
        "=== 記事リスト ===\n" + listing
    )


def _extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def _call_gemini(prompt, model, key):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
    }
    r = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=body,
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai_compat(base_url, prompt, model, key):
    """Groq / Mistral は OpenAI 互換の chat/completions 形式。"""
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_groq(prompt, model, key):
    return _call_openai_compat("https://api.groq.com/openai/v1", prompt, model, key)


def _call_mistral(prompt, model, key):
    return _call_openai_compat("https://api.mistral.ai/v1", prompt, model, key)


_PROVIDERS = {
    "gemini": _call_gemini,
    "groq": _call_groq,
    "mistral": _call_mistral,
}


def summarize(items):
    """記事リスト -> ダイジェスト dict。失敗時は None(呼び出し側でフォールバック)。

    戻り値: {headline, summary, key_topics:[{title,note}],
             articles:[{title_ja, category, source, url}]}
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model    = os.environ.get("LLM_MODEL", "").strip()
    key      = os.environ.get("LLM_API_KEY", "").strip()

    if not provider or not model or not key:
        raise SystemExit(
            "設定エラー: LLM_PROVIDER / LLM_MODEL / LLM_API_KEY を "
            "Repository secrets に設定してください。"
        )
    if provider not in _PROVIDERS:
        raise SystemExit(
            f"設定エラー: LLM_PROVIDER は {', '.join(sorted(_PROVIDERS))} のいずれかにしてください。"
        )

    try:
        raw = _PROVIDERS[provider](_build_prompt(items), model, key)
        data = _extract_json(raw)
    except Exception:
        # プロバイダ名・応答内容はログに出さない(秘匿要件)
        print("WARN: LLM 呼び出しまたは応答の解析に失敗。記事リストのみで出力します。",
              file=sys.stderr)
        return None

    articles = []
    for row in data.get("articles", []):
        try:
            src = items[int(row["index"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        cat = row.get("category", "")
        if cat not in CATEGORIES:
            cat = "その他"
        articles.append({
            "title_ja": (row.get("title_ja") or src["title"]).strip(),
            "category": cat,
            "source": src["source"],
            "url": src["link"],
        })

    if not articles:
        return None

    return {
        "headline": (data.get("headline") or "").strip(),
        "summary": (data.get("summary") or "").strip(),
        "key_topics": [
            {"title": (t.get("title") or "").strip(),
             "note": (t.get("note") or "").strip()}
            for t in data.get("key_topics", [])
            if isinstance(t, dict) and (t.get("title") or "").strip()
        ][:4],
        "articles": articles[:MAX_ITEMS],
    }


# ---------------------------------------------------------------- SVG 図解(コード生成)
def svg_category_bars(counts):
    """図解①: トピック分類の横棒グラフ。counts: {category: n}"""
    rows = [(c, counts.get(c, 0)) for c in CATEGORIES]
    max_n = max((n for _, n in rows), default=0) or 1
    bar_h, gap, label_w, top = 22, 14, 176, 8
    chart_w = 620
    bar_area = chart_w - label_w - 44
    height = top + len(rows) * (bar_h + gap)

    parts = [
        f'<svg viewBox="0 0 {chart_w} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="トピック分類別の記事件数" '
        f'style="width:100%;height:auto;display:block;">'
    ]
    y = top
    for cat, n in rows:
        w = round(bar_area * n / max_n) if n else 0
        active = n > 0
        bar_fill = "var(--amber)" if active else "var(--line2)"
        txt_fill = "var(--text)" if active else "var(--dim)"
        parts.append(
            f'<text x="{label_w - 10}" y="{y + bar_h - 7}" text-anchor="end" '
            f'fill="{txt_fill}" font-size="12.5" font-family="var(--sans)">{html.escape(cat)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{max(w, 2)}" height="{bar_h}" rx="4" '
            f'fill="{bar_fill}" opacity="{"0.9" if active else "0.5"}"/>'
        )
        parts.append(
            f'<text x="{label_w + max(w, 2) + 8}" y="{y + bar_h - 6}" '
            f'fill="var(--muted)" font-size="12" font-family="var(--mono)">{n}</text>'
        )
        y += bar_h + gap
    parts.append("</svg>")
    return "".join(parts)


def svg_topic_flow(topics):
    """図解②: 重要トピックの流れ。縦のワイヤーにノードを並べる。"""
    if not topics:
        return ""
    node_gap, top_pad = 78, 12
    chart_w = 620
    height = top_pad + len(topics) * node_gap
    line_x = 22

    parts = [
        f'<svg viewBox="0 0 {chart_w} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="本日の重要トピック" '
        f'style="width:100%;height:auto;display:block;">'
    ]
    first_y = top_pad + 12
    last_y = top_pad + (len(topics) - 1) * node_gap + 12
    if len(topics) > 1:
        parts.append(
            f'<line x1="{line_x}" y1="{first_y}" x2="{line_x}" y2="{last_y}" '
            f'stroke="var(--line2)" stroke-width="2"/>'
        )
    for i, t in enumerate(topics):
        cy = top_pad + i * node_gap + 12
        parts.append(
            f'<circle cx="{line_x}" cy="{cy}" r="7" fill="var(--bg)" '
            f'stroke="var(--amber)" stroke-width="2.5"/>'
        )
        parts.append(
            f'<circle cx="{line_x}" cy="{cy}" r="2.6" fill="var(--amber)"/>'
        )
        title = html.escape(t["title"])
        note = html.escape(t["note"])
        parts.append(
            f'<text x="{line_x + 22}" y="{cy + 1}" fill="var(--text)" '
            f'font-size="14.5" font-weight="600" font-family="var(--sans)">{title}</text>'
        )
        # note は foreignObject で折り返し表示
        parts.append(
            f'<foreignObject x="{line_x + 20}" y="{cy + 10}" width="{chart_w - line_x - 32}" height="{node_gap - 20}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" '
            f'style="font-size:12.5px;line-height:1.55;color:var(--muted);'
            f'font-family:var(--sans);">{note}</div></foreignObject>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- HTML 出力
def render_html(digest, articles_fallback, fetched_count, generated_at):
    """digest が None の場合は articles_fallback(原題リスト)のみで描画。"""
    date_label = generated_at.strftime("%Y.%m.%d")
    weekday = "月火水木金土日"[generated_at.weekday()]
    time_label = generated_at.strftime("%H:%M")

    if digest:
        published = len(digest["articles"])
        headline = html.escape(digest["headline"] or "本日のダイジェスト")
        summary = html.escape(digest["summary"])
        counts = {}
        for a in digest["articles"]:
            counts[a["category"]] = counts.get(a["category"], 0) + 1

        summary_html = f"""      <section class="summary">
        <h1 class="headline">{headline}</h1>
        <p class="lede">{summary}</p>
      </section>"""

        charts_html = f"""      <section class="panel">
        <h2 class="panel-title">トピック分類</h2>
        {svg_category_bars(counts)}
      </section>"""
        flow = svg_topic_flow(digest["key_topics"])
        if flow:
            charts_html += f"""
      <section class="panel">
        <h2 class="panel-title">重要トピック</h2>
        {flow}
      </section>"""

        rows = []
        for a in digest["articles"]:
            rows.append(
                f"""        <li class="art">
          <a href="{html.escape(a["url"], quote=True)}" target="_blank" rel="noopener">
            <span class="cat">{html.escape(a["category"])}</span>
            <span class="ttl">{html.escape(a["title_ja"])}</span>
            <span class="src">{html.escape(a["source"] or "")}</span>
          </a>
        </li>"""
            )
        list_html = f"""      <section class="panel">
        <h2 class="panel-title">参照記事(掲載 {published} 件)</h2>
        <ul class="arts">
{chr(10).join(rows)}
        </ul>
      </section>"""
    else:
        published = len(articles_fallback)
        summary_html = """      <section class="summary">
        <h1 class="headline">本日のダイジェスト</h1>
        <p class="lede muted">本日は要約を生成できませんでした。記事一覧のみ掲載しています。</p>
      </section>""" if published else """      <section class="summary">
        <h1 class="headline">本日は新着なし</h1>
        <p class="lede muted">直近48時間に対象記事が見つかりませんでした。</p>
      </section>"""
        charts_html = ""
        if published:
            rows = []
            for a in articles_fallback[:MAX_ITEMS]:
                rows.append(
                    f"""        <li class="art">
          <a href="{html.escape(a["link"], quote=True)}" target="_blank" rel="noopener">
            <span class="ttl">{html.escape(a["title"])}</span>
            <span class="src">{html.escape(a["source"] or "")}</span>
          </a>
        </li>"""
                )
            list_html = f"""      <section class="panel">
        <h2 class="panel-title">記事一覧(掲載 {min(published, MAX_ITEMS)} 件)</h2>
        <ul class="arts">
{chr(10).join(rows)}
        </ul>
      </section>"""
            published = min(published, MAX_ITEMS)
        else:
            list_html = ""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e1116">
<title>Claude News Digest — {date_label}</title>
<style>
  :root {{
    --bg:#0e1116; --surface:#151a21; --line:#232a33; --line2:#2c343e;
    --text:#e6edf3; --muted:#8b98a5; --dim:#5b6672;
    --amber:#f2b45c;
    --mono: ui-monospace,"SF Mono","JetBrains Mono","Roboto Mono",Menlo,monospace;
    --sans: -apple-system,BlinkMacSystemFont,"Hiragino Kaku Gothic ProN",
            "Noto Sans JP","BIZ UDPGothic","Segoe UI",sans-serif;
  }}
  * {{ box-sizing:border-box; }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{
    margin:0; background:var(--bg); color:var(--text);
    font-family:var(--sans); line-height:1.65;
    padding:env(safe-area-inset-top) env(safe-area-inset-right) 3rem env(safe-area-inset-left);
  }}
  .wrap {{ max-width:680px; margin:0 auto; padding:0 18px; }}

  header {{ padding:26px 0 8px; }}
  .eyebrow {{
    font-family:var(--mono); font-size:11px; letter-spacing:.28em;
    text-transform:uppercase; color:var(--amber);
  }}
  .date {{
    font-family:var(--mono); font-weight:600; letter-spacing:.02em;
    font-size:34px; margin:6px 0 2px; font-variant-numeric:tabular-nums;
  }}
  .date .wd {{ color:var(--dim); font-size:20px; margin-left:.4em; }}
  .status {{
    font-family:var(--mono); font-size:11.5px; color:var(--muted);
    border:1px solid var(--line); border-radius:8px;
    padding:9px 12px; margin-top:14px;
    display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center;
  }}
  .status b {{ color:var(--text); font-weight:600; }}
  .status .amber {{ color:var(--amber); }}
  .status .sep {{ color:var(--line2); }}
  .pulse {{
    width:7px; height:7px; border-radius:50%; background:var(--amber);
    box-shadow:0 0 0 0 rgba(242,180,92,.55); animation:pulse 2.6s ease-out infinite;
  }}
  @keyframes pulse {{
    0%{{box-shadow:0 0 0 0 rgba(242,180,92,.5);}}
    70%{{box-shadow:0 0 0 7px rgba(242,180,92,0);}}
    100%{{box-shadow:0 0 0 0 rgba(242,180,92,0);}}
  }}

  .summary {{ padding:22px 0 4px; }}
  .headline {{ font-size:22px; line-height:1.5; margin:0 0 10px; }}
  .lede {{ font-size:15px; color:var(--text); margin:0; }}
  .lede.muted {{ color:var(--muted); }}

  .panel {{
    border:1px solid var(--line); border-radius:12px;
    background:var(--surface); padding:16px 16px 12px; margin-top:18px;
  }}
  .panel-title {{
    font-family:var(--mono); font-size:11px; letter-spacing:.18em;
    text-transform:uppercase; color:var(--amber); margin:0 0 12px;
  }}

  .arts {{ list-style:none; margin:0; padding:0; }}
  .art {{ border-top:1px solid var(--line); }}
  .art:first-child {{ border-top:0; }}
  .art a {{
    display:block; padding:11px 2px; text-decoration:none; color:inherit;
  }}
  .art a:hover .ttl {{ color:var(--amber); }}
  .art a:focus-visible {{ outline:2px solid var(--amber); outline-offset:2px; border-radius:6px; }}
  .cat {{
    display:inline-block; font-family:var(--mono); font-size:10.5px;
    color:var(--amber); border:1px solid var(--line2); border-radius:4px;
    padding:1px 6px; margin-bottom:4px; letter-spacing:.03em;
  }}
  .ttl {{ display:block; font-size:14.5px; font-weight:500; line-height:1.5; }}
  .src {{
    display:block; font-family:var(--mono); font-size:11px;
    color:var(--dim); margin-top:3px;
  }}

  footer {{
    margin-top:26px; padding-top:16px; border-top:1px solid var(--line);
    font-family:var(--mono); font-size:11px; color:var(--dim);
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;
  }}
  footer a {{ color:var(--muted); text-decoration:none; }}

  @media (prefers-reduced-motion: reduce) {{ .pulse {{ animation:none; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Claude News Digest</div>
    <div class="date">{date_label}<span class="wd"> {weekday}</span></div>
    <div class="status">
      <span class="pulse" aria-hidden="true"></span>
      <span>UPDATED <b>{time_label}</b> JST</span>
      <span class="sep">/</span>
      <span>取得 <b>{fetched_count}</b> 件 &rarr; 掲載 <b class="amber">{published}</b> 件</span>
      <span class="sep">/</span>
      <span>直近48h</span>
    </div>
  </header>

  <main>
{summary_html}
{charts_html}
{list_html}
  </main>

  <footer>
    <span>自動生成 · GitHub Actions</span>
    <a href="archive/">過去ログ</a>
  </footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------- main
def main():
    os.makedirs(ARCHIVE, exist_ok=True)
    now = dt.datetime.now(JST)

    items = fetch_articles()
    fetched = len(items)
    print(f"取得: {fetched} 件")

    digest = summarize(items) if items else None
    page = render_html(digest, items, fetched, now)

    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(ARCHIVE, now.strftime("%Y-%m-%d") + ".html"),
              "w", encoding="utf-8") as f:
        f.write(page)

    published = len(digest["articles"]) if digest else min(fetched, MAX_ITEMS)
    print(f"完了: 取得 {fetched} 件 → 掲載 {published} 件({now:%Y-%m-%d %H:%M} JST)")


if __name__ == "__main__":
    main()
