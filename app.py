# app.py
import io
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Carrega variáveis de ambiente
load_dotenv()

# Configura a página (primeiro comando Streamlit)
st.set_page_config(page_title="Chat FAQ IA", page_icon="🤖")

# Título da aplicação
st.title("Chat FAQ IA")

# Função para processar o arquivo FAQ e reindexar
@st.cache_data(show_spinner=False)
def process_faq(file_bytes: bytes, filename: str) -> int:
    """Extrai e indexa o FAQ retornando apenas o número de chunks."""
    # Extrai texto conforme extensão
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        text = file_bytes.decode("utf-8")

    # Divide em chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(text)

    # Gera embeddings e indexa no FAISS
    embeddings = OpenAIEmbeddings()
    # Usa embed_documents para docs
    vector_store = FAISS.from_texts(docs, embeddings)
    # Ajusta embed_query para as consultas
    vector_store.embedding_function = embeddings.embed_query
    vector_store.save_local("faqs_index")

    return len(docs)

# Sidebar: upload de FAQ e reindexação
st.sidebar.header("🔄 Upload dinâmico de FAQs")
uploaded = st.sidebar.file_uploader(
    "Envie um .txt, .md ou .pdf",
    type=["txt", "md", "pdf"]
)
if uploaded:
    with st.spinner("Reindexando FAQ... ⏳"):
        data = uploaded.read()
        count_chunks = process_faq(data, uploaded.name)
        st.sidebar.success(f"Indexação concluída: {count_chunks} chunks")

# Carrega (ou recarrega) índice existente
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(
    "faqs_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# Ajusta embed_query para consultas
vector_store.embedding_function = embeddings.embed_query

# Histórico de mensagens
if "history" not in st.session_state:
    st.session_state.history = []

# Exibe todo o histórico
for entry in st.session_state.history:
    st.chat_message(entry["role"]).write(entry["content"])

# Campo de entrada do usuário
user_input = st.chat_input("Digite sua pergunta...")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Cria o chain de QA com o vector_store atual
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Executa a consulta e imprime a resposta
    result = qa_chain.invoke({"query": user_input})
    answer = result.get("result", "Desculpe, não consegui gerar uma resposta.")

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
