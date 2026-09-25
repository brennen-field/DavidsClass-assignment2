"""Launch the course assistant: python app.py"""

import gradio as gr

from course_assistant.answers.panel import build_panel as build_answers_panel
from course_assistant.documents.panel import build_panel as build_documents_panel
from course_assistant.quiz.panel import build_panel as build_quiz_panel
from course_assistant.theme import CSS, HEADER_HTML, THEME


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Buffs Course Assistant") as app:
        gr.HTML(HEADER_HTML)
        with gr.Tab("📚 Documents"):
            build_documents_panel()
        with gr.Tab("💬 Ask"):
            build_answers_panel()
        with gr.Tab("🎯 Quiz"):
            build_quiz_panel()
    return app


if __name__ == "__main__":
    build_app().launch(theme=THEME, css=CSS)
