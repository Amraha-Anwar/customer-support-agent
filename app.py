import threading
import uvicorn
import gradio as gr
from main import app as fastapi_app

with gr.Blocks() as demo:
    gr.Markdown("## WhatsApp AI Ordering Backend — Running ✅")

fastapi_app.mount("/gradio", gr.routes.App.create_app(demo))

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)