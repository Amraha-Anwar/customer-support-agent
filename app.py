import threading
import uvicorn
import gradio as gr
from main import app as fastapi_app

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()

with gr.Blocks() as demo:
    gr.Markdown("## Backend Running ✅")

demo.launch(server_name="0.0.0.0", server_port=7860)