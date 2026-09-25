# Handoffs

Where one person's code depends on another's. When a format is agreed, write it
down here (or in a linked file in `docs/`) so everyone's agent can read it.

| From | To | What gets passed | Spec |
|---|---|---|---|
| Chase | Nick, Shrihari | Processed pages: text, page/slide image, document name, page/slide number | [handoff-documents.md](handoff-documents.md) |
| Nick, Shrihari | Chase | A function to delete one document's entries from the text and visual indexes | [handoff-documents.md](handoff-documents.md#what-you-give-back-a-delete-function) |
| Nick | Shrihari | Keyword and text-embedding search results | TBD |
| Shrihari | Preston, Brennen | Final ranked evidence (text + images, with source details) | TBD |
| Chase, Preston | Brennen | Panels: each exposes `build_panel()` in its package's `panel.py` | `app.py` |
| Brennen | Everyone | Repo structure, `.env.example`, app shell | this repo |
