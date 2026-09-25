"""CU Boulder look for the whole app: gold/black theme, header banner, shared CSS.

app.py passes THEME and CSS to launch() and puts HEADER_HTML at the top.
"""

import gradio as gr

CU_GOLD = "#CFB87C"
CU_BLACK = "#000000"
CU_DARK_GRAY = "#565A5C"
CU_LIGHT_GRAY = "#A2A4A3"

_gold = gr.themes.Color(
    name="cu_gold",
    c50="#FBF8EF", c100="#F5EEDA", c200="#EBDDB5", c300="#DFCB95", c400="#CFB87C",
    c500="#B89F5D", c600="#9A8248", c700="#7A6638", c800="#5C4C2A", c900="#3F341D",
    c950="#241E11",
)
_gray = gr.themes.Color(
    name="cu_gray",
    c50="#F7F7F6", c100="#EDEDEC", c200="#D9DADA", c300="#BFC0C0", c400="#A2A4A3",
    c500="#7D8081", c600="#565A5C", c700="#43474A", c800="#2D3033", c900="#1C1E20",
    c950="#111213",
)

THEME = gr.themes.Soft(
    primary_hue=_gold,
    secondary_hue=_gold,
    neutral_hue=_gray,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#FAF8F2",
    body_background_fill_dark="#111213",
    button_primary_background_fill=CU_GOLD,
    button_primary_background_fill_hover="#DFCB95",
    button_primary_background_fill_dark=CU_GOLD,
    button_primary_background_fill_hover_dark="#DFCB95",
    button_primary_text_color=CU_BLACK,
    button_primary_text_color_dark=CU_BLACK,
    block_border_width="1px",
    block_shadow="0 2px 10px rgba(0,0,0,0.06)",
    block_title_text_weight="700",
    # Darker gold for text in light mode so labels stay readable on white.
    color_accent_soft="*primary_100",
    color_accent_soft_dark="*primary_800",
    block_label_text_color="*primary_800",
    block_label_text_color_dark="*primary_200",
    block_title_text_color="*primary_800",
    block_title_text_color_dark="*primary_200",
    link_text_color="*primary_700",
    link_text_color_dark="*primary_300",
)

HEADER_HTML = f"""
<div class="buffs-header">
  <svg class="flatirons" viewBox="0 0 400 80" preserveAspectRatio="none" aria-hidden="true">
    <polygon points="0,80 70,18 120,80" />
    <polygon points="80,80 160,6 225,80" />
    <polygon points="190,80 265,22 320,80" />
    <polygon points="290,80 350,34 400,80" />
  </svg>
  <div class="buffs-title">
    <span class="ralphie" aria-hidden="true">🦬</span>
    <div>
      <h1>Buffs Course Assistant</h1>
      <p>MBAX 6418 · Business Solutions with AI · Ask your course materials anything</p>
    </div>
  </div>
</div>
"""

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@600&display=swap');

/* ---- accent color: darker gold on light backgrounds, lighter gold on dark ---- */
:root, .gradio-container {{ --color-accent: #7A6638; }}
.dark, .dark .gradio-container {{ --color-accent: #DFCB95; }}

/* ---- header ---- */
.buffs-header {{
  position: relative; overflow: hidden;
  background: linear-gradient(135deg, {CU_BLACK} 0%, #1f1f1f 100%);
  border-bottom: 4px solid {CU_GOLD};
  border-radius: 16px; padding: 22px 26px 30px;
}}
.buffs-header .flatirons {{
  position: absolute; right: 0; bottom: 0; width: 55%; height: 70%;
  fill: {CU_GOLD}; opacity: 0.16;
}}
.buffs-title {{ position: relative; display: flex; align-items: center; gap: 16px; }}
.buffs-title h1 {{
  font-family: 'Oswald', sans-serif; letter-spacing: 0.04em; text-transform: uppercase;
  color: {CU_GOLD} !important; margin: 0; font-size: 2rem;
}}
.buffs-title p {{ color: #E6E6E6 !important; margin: 4px 0 0; }}
.ralphie {{ font-size: 3rem; display: inline-block; animation: ralphie-bob 3s ease-in-out infinite; }}
@keyframes ralphie-bob {{
  0%, 100% {{ transform: translateY(0) rotate(0); }}
  50% {{ transform: translateY(-5px) rotate(-4deg); }}
}}

/* ---- tabs ---- */
button[role="tab"][aria-selected="true"] {{ border-bottom-color: {CU_GOLD} !important; font-weight: 700; }}

/* ---- stat tiles ---- */
.buffs-stats {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.buffs-stat {{
  flex: 1 1 140px; padding: 14px 16px; border-radius: 14px;
  background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-top: 4px solid {CU_GOLD};
}}
.buffs-stat .num {{ font-family: 'Oswald', sans-serif; font-size: 1.9rem; line-height: 1; }}
.buffs-stat .label {{ color: var(--body-text-color-subdued); font-size: 0.85rem; margin-top: 4px; }}
.buffs-empty {{
  padding: 18px; border-radius: 14px; text-align: center;
  border: 2px dashed {CU_LIGHT_GRAY}; color: var(--body-text-color-subdued);
}}

/* ---- status messages ---- */
.buffs-msg {{
  padding: 10px 14px; margin: 6px 0; border-radius: 10px;
  background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-left: 5px solid {CU_LIGHT_GRAY};
  animation: msg-in 0.35s ease-out both;
}}
.buffs-msg.added {{ border-left-color: {CU_GOLD}; }}
.buffs-msg.error {{ border-left-color: #C8102E; }}
.buffs-msg.warning {{ border-left-color: #E08A00; }}
@keyframes msg-in {{ from {{ opacity: 0; transform: translateX(-8px); }} to {{ opacity: 1; transform: none; }} }}

/* ---- upload celebration ---- */
.buffs-celebrate {{
  position: relative; overflow: hidden; height: 120px; border-radius: 14px;
  background: linear-gradient(135deg, {CU_BLACK}, #2a2a2a);
  border: 2px solid {CU_GOLD};
  animation: celebrate-fade 4.5s ease-in forwards;
}}
.buffs-celebrate .stampede {{
  position: absolute; bottom: 12px; left: -70px; font-size: 3rem;
  transform: scaleX(-1);
  animation: stampede 2.6s cubic-bezier(.3,.1,.3,1) forwards;
}}
.buffs-celebrate .dust {{
  position: absolute; bottom: 14px; left: -40px; font-size: 1.4rem; opacity: 0.7;
  animation: stampede 2.6s cubic-bezier(.3,.1,.3,1) 0.08s forwards;
}}
.buffs-celebrate .cheer {{
  position: absolute; top: 18px; width: 100%; text-align: center;
  font-family: 'Oswald', sans-serif; font-size: 1.6rem; letter-spacing: 0.06em;
  color: {CU_GOLD}; text-transform: uppercase;
  animation: cheer-pop 0.6s cubic-bezier(.2,1.6,.4,1) 0.2s both;
}}
.buffs-celebrate .sub {{
  position: absolute; top: 58px; width: 100%; text-align: center; color: #E6E6E6;
  animation: msg-in 0.4s ease-out 0.5s both;
}}
.buffs-celebrate .confetti {{
  position: absolute; top: -12px; width: 8px; height: 14px; border-radius: 2px;
  animation: confetti-fall 1.8s ease-in forwards;
}}
@keyframes stampede {{ to {{ left: calc(100% + 70px); }} }}
@keyframes cheer-pop {{ from {{ opacity: 0; transform: scale(0.4); }} to {{ opacity: 1; transform: scale(1); }} }}
@keyframes confetti-fall {{
  to {{ transform: translateY(150px) rotate(540deg); opacity: 0; }}
}}
@keyframes celebrate-fade {{ 0%, 80% {{ opacity: 1; }} 100% {{ opacity: 0.85; }} }}

@media (prefers-reduced-motion: reduce) {{
  .ralphie, .buffs-msg, .buffs-celebrate, .buffs-celebrate * {{ animation: none !important; }}
  .buffs-celebrate .stampede {{ left: auto; right: 24px; }}
  .buffs-celebrate .dust, .buffs-celebrate .confetti {{ display: none; }}
}}
"""
