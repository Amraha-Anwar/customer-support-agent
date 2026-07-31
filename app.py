import gradio as gr
from main import app as fastapi_app
from fastapi import FastAPI
import uvicorn

# Mount FastAPI inside Gradio
with gr.Blocks() as demo:
    gr.Markdown("# WhatsApp AI Ordering Backend")
    gr.Markdown("API is running.")

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)