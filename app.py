import streamlit as st
from dotenv import load_dotenv
from pytube import YouTube
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from utils import load_or_create_index, extract_video_id, get_transcript

#–– Load your .env for OPENAI_API_KEY ––
load_dotenv()

#–– Cache transcript fetching to avoid repeated network calls ––
@st.cache_data
def fetch_transcript(video_url_or_id: str) -> Optional[str]:
    vid = extract_video_id(video_url_or_id)
    return get_transcript(vid)

#–– Session‐state for persistent answer & feedback ––
if 'answer' not in st.session_state:
    st.session_state['answer'] = None
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = None

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
        question = st.text_input("Ask a question about the video")

# Question input moved to column layout

#–– 3) “Let’s go!” button triggers RAG pipeline ––
if st.button("Let's go!"):
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
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )
            llm = ChatOpenAI(model=model_choice, temperature=0.2)

            prompt = PromptTemplate(
                template="""
You are a helpful assistant. Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
                """,
                input_variables=["context", "question"],
            )

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            chain = (
                RunnableParallel({
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                })
                | prompt
                | llm
                | StrOutputParser()
            )

            try:
                answer = chain.invoke(question)
                st.session_state['answer'] = answer
                st.session_state['feedback'] = None
            except Exception as e:
                st.error(f"Error during answer generation: {e}")

if st.session_state['answer'] is not None:
    tabs = st.tabs(["📜 Transcript", "🔍 Retrieved Chunks", "💡 Answer"])
    with tabs[0]:
        snippet = fetch_transcript(video_url)
        if snippet:
            st.write(snippet[:1000] + ("…" if len(snippet) > 1000 else ""))
    with tabs[1]:
        st.info("Retrieved chunks will be shown here (coming soon).")
    with tabs[2]:
        st.success("✅ Answer:")
        st.markdown(f"**{st.session_state['answer']}**")

        col1, col2 = st.columns(2)
        if col1.button("👍", key="like"):
            st.session_state['feedback'] = "Thanks for your feedback!"
        if col2.button("👎", key="dislike"):
            st.session_state['feedback'] = "Sorry to hear that. We'll improve!"

        if st.session_state['feedback']:
            st.info(st.session_state['feedback'])