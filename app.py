# app.py
import io
import os
import shutil
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import stat

# Carrega variáveis de ambiente
load_dotenv()

# Garante que a chave da OpenAI esteja disponível no ambiente antes de qualquer uso
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Chave da OpenAI não encontrada. Verifique o .env local ou os secrets na nuvem.")
    st.stop()


# Configura a página (primeiro comando Streamlit)
st.set_page_config(page_title="Chat FAQ IA", page_icon="🤖")

# Título da aplicação
st.title("Chat FAQ IA")

# Função para processar o arquivo FAQ e reindexar
@st.cache_data(show_spinner=False)
def process_faq(file_bytes: bytes, filename: str) -> int:
    """Extrai e indexa o FAQ retornando apenas o número de chunks."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(docs, embeddings)
    vector_store.embedding_function = embeddings.embed_query
    vector_store.save_local("faqs_index")

    return len(docs)

def on_rm_error(func, path, exc_info):
    # Tenta alterar permissão e remover novamente
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Função para resetar completamente a IA
def resetar_ia():
    pastas_para_apagar = ["faqs_index", "uploads", "__pycache__"]
    extensoes_para_apagar = {".pkl", ".db", ".json", ".txt", ".md", ".pdf"}

    for pasta in pastas_para_apagar:
        if os.path.exists(pasta):
            try:
                shutil.rmtree(pasta, onerror=on_rm_error)
            except Exception as e:
                st.warning(f"Não foi possível remover {pasta}: {e}")

    for arquivo in Path(".").glob("*"):
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes_para_apagar:
            try:
                arquivo.unlink()
            except Exception as e:
                st.warning(f"Não foi possível remover {arquivo}: {e}")

    st.cache_data.clear()
    st.cache_resource.clear()

# Sidebar: upload + reset
st.sidebar.header("🔄 Upload dinâmico de FAQs")
uploaded = st.sidebar.file_uploader("Envie um .txt, .md ou .pdf", type=["txt", "md", "pdf"])

with st.sidebar:
    st.subheader("⚙️ Administração")
    if "confirmar_reset" not in st.session_state:
        st.session_state.confirmar_reset = False

    if st.button("🧹 Resetar IA (limpar tudo)"):
        st.session_state.confirmar_reset = True

    if st.session_state.confirmar_reset:
        st.warning("Tem certeza que deseja resetar a IA? Isso apagará todos os dados e começará do zero.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirmar"):
                resetar_ia()
                st.session_state.confirmar_reset = False  # Limpa o estado antes do sucesso
                st.success("IA resetada com sucesso! Reinicie o app para começar do zero.")
                st.rerun()  # Força atualização da interface
        with col2:
            if st.button("❌ Cancelar"):
                st.session_state.confirmar_reset = False
                st.info("Reset cancelado.")
                st.rerun()

if uploaded:
    with st.spinner("Reindexando FAQ... ⏳"):
        data = uploaded.read()
        count_chunks = process_faq(data, uploaded.name)
        st.sidebar.success(f"Indexação concluída: {count_chunks} chunks")

# Carrega índice existente
index_path = Path("faqs_index/index.faiss")
if index_path.exists():
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "faqs_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    vector_store.embedding_function = embeddings.embed_query
else:
    st.warning("Nenhum índice encontrado. Faça upload de um FAQ para começar.")
    vector_store = None

# Só permita perguntas se o índice existir
if vector_store:
    # Histórico
    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        st.chat_message(entry["role"]).write(entry["content"])

    user_input = st.chat_input("Digite sua pergunta...")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        result = qa_chain.invoke({"query": user_input})
        answer = result.get("result", "Desculpe, não consegui gerar uma resposta.")

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("Envie um FAQ para ativar o chat.")
