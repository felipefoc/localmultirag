from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self):
        # ollama pull mxbai-embed-large
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )
        
        #ollama pull llama3.2
        self.model_ollama = ChatOllama(
            model="cyberuser42/DeepSeek-R1-Distill-Llama-8B:latest",
            temperature=0
        )