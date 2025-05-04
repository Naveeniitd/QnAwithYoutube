import streamlit as st
from dotenv import load_dotenv
from pytube import YouTube
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import requests
from bs4 import BeautifulSoup
from utils import load_or_create_index, extract_video_id, get_transcript
 
#–– Load your .env for OPENAI_API_KEY ––
load_dotenv()

from langsmith import traceable

@traceable(name="YouTube QA Run")
def run_chain(chain, question):
    return chain.invoke(question)

#–– Cache transcript fetching to avoid repeated network calls ––
@st.cache_data
def fetch_transcript(video_url_or_id: str) -> Optional[str]:
    vid = extract_video_id(video_url_or_id)
    return get_transcript(vid)

#–– Session‐state for persistent answer, feedback, and docs ––
if 'answer' not in st.session_state:
    st.session_state['answer'] = None
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = None
if 'retrieved_docs' not in st.session_state:
    st.session_state['retrieved_docs'] = []
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "transcript"

#–– Page setup ––
st.set_page_config(page_title="Q&A with youtube", layout="centered")
st.title("🎥 Q&A with YOUTUBE")

with st.sidebar:
    st.header("🔧 Settings")
    video_url = st.text_input(
        "Enter YouTube Video URL or ID",
        value="https://www.youtube.com/watch?v=H2wDeD0Y1qA"
    )
    model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
    top_k = st.slider("Top-K Chunks", 1, 10, 4)

if video_url.strip():
    vid = extract_video_id(video_url)
    url = f"https://www.youtube.com/watch?v={vid}"
    try:
        yt = YouTube(url)
        title = yt.title
        thumb = yt.thumbnail_url
    except Exception as e:
        # Fallback to HTML scraping if PyTube fails
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            title_tag = soup.find("meta", property="og:title")
            title = title_tag["content"] if title_tag else None
            thumb_tag = soup.find("meta", property="og:image")
            thumb = thumb_tag["content"] if thumb_tag else None
        except Exception as inner_e:
            st.warning(f"⚠️ Could not fetch video metadata. Errors: {e}, {inner_e}")
            title = None
            thumb = None
    col1, col2 = st.columns([1, 2])
    with col1:
        if thumb:
            st.image(thumb, use_container_width=True)
    with col2:
        if title:
            st.subheader(title)
        with st.form("question_form", clear_on_submit=False):
            question = st.text_input("Ask a question about the video")
            submitted = st.form_submit_button("Let's go!")

# Question input moved to column layout

#–– 3) “Let’s go!” button triggers RAG pipeline ––
if 'submitted' in locals() and submitted:
    with st.spinner("Thinking…"):
        # 3a) Transcript preview
        transcript = fetch_transcript(video_url)
        if transcript:
            with st.expander("📜 Transcript preview"):
                snippet = transcript[:1000] + ("…" if len(transcript) > 1000 else "")
                st.write(snippet)
        else:
            st.error("Transcript not available.")

        # 3b) Build/load FAISS index
        vectorstore = load_or_create_index(video_url)
        if not vectorstore:
            st.error("Failed to build index. Check transcript availability.")
        else:
            # Base retriever with MMR (Maximal Marginal Relevance)
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": top_k, "lambda_mult": 0.5}
            )

            # LLM for generating multiple queries
            llm_for_queries = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

            # Multi-query retriever with contextual compression
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm_for_queries
            )
            compressor = LLMChainExtractor.from_llm(llm_for_queries)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=multi_query_retriever
            )
            llm = ChatOpenAI(model=model_choice, temperature=0.2)

            def format_and_store_docs(docs):
                st.session_state['retrieved_docs'] = docs
                max_chars = 15000 if model_choice == "gpt-3.5-turbo" else 30000
                contents = []
                current_length = 0
                for doc in docs:
                    if current_length + len(doc.page_content) > max_chars:
                        break
                    contents.append(doc.page_content)
                    current_length += len(doc.page_content)
                return "\n\n".join(contents)

            prompt = PromptTemplate(
                template="""
You are a helpful assistant. Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
                """,
                input_variables=["context", "question"],
            )

            st.session_state['retrieved_docs'] = []

            chain = (
                RunnableParallel({
                    "context": retriever | RunnableLambda(format_and_store_docs),
                    "question": RunnablePassthrough(),
                })
                | prompt
                | llm
                | StrOutputParser()
            )

            try:
                answer = run_chain(chain, question)
                st.session_state['answer'] = answer
                st.session_state['feedback'] = None
                st.session_state['active_tab'] = "answer"
            except Exception as e:
                st.error(f"Error during answer generation: {e}")

if st.session_state['answer'] is not None:
    tabs = st.tabs(["📜 Transcript", "🔍 Retrieved Chunks", "💡 Answer"])
    active_index = {"transcript": 0, "retrieved": 1, "answer": 2}
    tab = tabs[active_index.get(st.session_state['active_tab'], 2)]
    with tabs[0]:
        snippet = fetch_transcript(video_url)
        if snippet:
            st.write(snippet[:1000] + ("…" if len(snippet) > 1000 else ""))
    with tabs[1]:
        st.info("Retrieved chunks will be shown here (coming soon).")
    with tab:
        st.markdown("✅ **Answer:**", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#d4edda;padding:10px;border-radius:10px;max-height:200px;overflow:auto;color:#000000'><strong>{st.session_state['answer']}</strong></div>",
            unsafe_allow_html=True
        )

        # Show citations
        if st.session_state['retrieved_docs']:
            st.markdown("##### 📚 Sources:")
            for i, doc in enumerate(st.session_state['retrieved_docs'], 1):
                st.markdown(f"**[{i}]** {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")

        col1, col2 = st.columns(2)
        if col1.button("👍", key="like"):
            st.session_state['feedback'] = "Thanks for your feedback!"
        if col2.button("👎", key="dislike"):
            st.session_state['feedback'] = "Sorry to hear that. We'll improve!"

        if st.session_state['feedback']:
            st.info(st.session_state['feedback'])