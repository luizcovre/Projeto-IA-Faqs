# qa.py
import sys
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Uso: python qa.py \"sua pergunta aqui\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "faqs_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Substituído run() por invoke()
    resultado = qa.invoke({"query": query})
    resposta = resultado["result"]

    print("\nResposta:\n")
    print(resposta)

if __name__ == "__main__":
    main()
