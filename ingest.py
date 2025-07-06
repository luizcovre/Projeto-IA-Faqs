# ingest.py
import os
import sys
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

def carregar_texto(caminho_arquivo: str) -> str:
    """Lê e retorna todo o conteúdo de um arquivo de texto."""
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        return f.read()

def dividir_em_chunks(texto: str):
    """Divide o texto em chunks de até 1.000 caracteres com 200 de sobreposição."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(texto)

def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print("Uso: python ingest.py arquivo.txt")
        sys.exit(1)

    arquivo_exemplo = sys.argv[1]
    texto = carregar_texto(arquivo_exemplo)
    chunks = dividir_em_chunks(texto)

    print(f"Total de chunks: {len(chunks)}")
    for i, c in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---\n{c}\n")

if __name__ == "__main__":
    main()
