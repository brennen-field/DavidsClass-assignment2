# Handoffs

Where one person's code depends on another's. When a format is agreed, write it
down here (or in a linked file in `docs/`) so everyone's agent can read it.

| From | To | What gets passed | Spec |
|---|---|---|---|
| Chase | Nick, Shrihari | Processed pages: text, page/slide image, document name, page/slide number | TBD |
| Nick, Shrihari | Chase | A function to delete one document's entries from the text and visual indexes | TBD |
| Nick | Shrihari | Keyword and text-embedding search results | [handoff-text-search.md](handoff-text-search.md) |
| Shrihari | Preston, Brennen | Final ranked evidence (text + images, with source details) | [handoff-visual-search.md](handoff-visual-search.md) |
| Chase, Preston | Brennen | Panels: each exposes `build_panel()` in its package's `panel.py` | `app.py` |
| Brennen | Everyone | Repo structure, `.env.example`, app shell | this repo |
