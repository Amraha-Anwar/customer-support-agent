import gradio as gr
from main import app as fastapi_app

with gr.Blocks() as demo:
    gr.Markdown("## WhatsApp AI Ordering Backend — Running ✅")

app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)