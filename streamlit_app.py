import os
from dotenv import load_dotenv
import streamlit as st

# Carrega .env localmente
load_dotenv()

# Força uso da key do st.secrets quando em deploy na nuvem
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Teste de Embeddings LangChain + OpenAI")

from langchain_openai import OpenAIEmbeddings

# Inicializa embeddings
embeddings = OpenAIEmbeddings()
st.write("✅ Embeddings inicializado com sucesso!")

# Faz uma consulta de exemplo
vetor = embeddings.embed_query("Olá mundo")
st.write("Exemplo de vetor (primeiros valores):", vetor[:5])
