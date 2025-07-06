# index.py
import sys
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def carregar_texto(caminho_arquivo: str) -> str:
    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Erro de encoding ao ler {caminho_arquivo}. Tente salvar como UTF-8.")
        sys.exit(1)

def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Uso: python index.py faq.txt")
        sys.exit(1)

    caminho = sys.argv[1]
    texto = carregar_texto(caminho)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_text(texto)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(docs, embeddings)

    vector_store.save_local("faqs_index")
    print(f"Índice salvo em 'faqs_index/'. Vetores gerados: {len(docs)}")

if __name__ == "__main__":
    main()
