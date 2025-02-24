import gradio as gr
import os
from utils.db import clear_folder, clear_sqlite
from chat import predict
from ingest import ingest_file
from models import Models

models = Models()
available_models = ["deepseek-r1:1.5b", "cyberuser42/DeepSeek-R1-Distill-Llama-8B:latest", "llama3.2:latest"]
clear_folder()

def process_files(files):
    for file in files:
        file_path = file.name
        ingest_file(file_path)
    return f"{len(files)} file(s) processed successfully!"

def chat(message, history, model_name):
    # Update model if different from current
    if model_name != models.model_ollama.model:
        models.model_ollama.model = model_name
    
    # Get response from the chain
    return predict(message, history)

def create_interface():
    with gr.Blocks(title="Document Q&A System", theme='Nymbo/Nymbo_Theme') as interface:
        gr.Markdown("# Document Question & Answer System")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    file_count="multiple",
                    label="Upload PDF Document",
                    file_types=[".pdf"],
                    type="filepath"
                )
                status_box = gr.Textbox(value="Ready...", label="Status")
                file_upload.upload(fn=process_files, inputs=[file_upload], outputs=[status_box])
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="Select Model"
                )
            
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat,
                    type="messages",
                    additional_inputs=[model_selector]
                )
                chatbot.textbox.placeholder = "Ask a question about your PDF..."
                clear = gr.Button("Reset All")
        
        clear.click(
            fn=lambda: (clear_sqlite(), []),
            inputs=[],
            outputs=[status_box, file_upload],
            queue=False
        )
        clear.click(
            fn=lambda: [],
            inputs=[],
            outputs=[chatbot.chatbot],
            queue=False
        )

    return interface

if __name__ == "__main__":
    clear_sqlite()
    clear_folder()
    interface = create_interface()
    interface.launch(share=False)