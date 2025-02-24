from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models

models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=".db/chroma_langchain_db"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente muito util! Responda apenas com os dados fornecidos e na linguagem fornecido no input."),
        ("human", "{input} \nPara responder, use apenas o {context}")
    ]
)

retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    ai_response = retrieval_chain.invoke({"input": message})
    return ai_response['answer'].split("\n</think>")[1]

def cli_chat():
    print("Welcome to the CLI chat! Type 'q', 'quit', or 'exit' to end.")
    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break

if __name__ == "__main__":
    cli_chat()

