import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pytube import YouTube

from utils import load_or_create_index, extract_video_id, get_transcript

# Cache transcript fetching to avoid repeated network calls
@st.cache_data
def fetch_transcript(video_url_or_id):
    video_id = extract_video_id(video_url_or_id)
    return get_transcript(video_id)

# Load environment variables
load_dotenv()

# Initialize session state for answer and feedback
if 'answer' not in st.session_state:
    st.session_state['answer'] = None
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = None

# Page config and title
st.set_page_config(page_title="Q&A with Youtube", layout="centered")
st.title("🎥 Q&A with Youtube")

# Inputs
video_url = st.text_input(
    "Enter YouTube Video URL or ID",
    value="https://www.youtube.com/watch?v=QY6yHJC2DIE"
)
question = st.text_input("Ask a question about the video")

# Main handler
if st.button("Let's go!"):
    with st.spinner("Processing..."):
        # Preview transcript
        transcript = fetch_transcript(video_url)
        if transcript:
            with st.expander("Transcript preview"):
                preview = transcript[:1000] + ("..." if len(transcript) > 1000 else "")
                st.write(preview)

        # Show video title and thumbnail
        try:
            yt = YouTube(video_url)
            st.subheader(yt.title)
            st.image(yt.thumbnail_url)
        except Exception:
            pass

        # Load or build the FAISS index
        vectorstore = load_or_create_index(video_url)
        if not vectorstore:
            st.error("Transcript not available or video invalid.")
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

            prompt = PromptTemplate(
                template="""
You are a helpful assistant. Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            try:
                answer = main_chain.invoke(question)
                st.session_state['answer'] = answer
                st.session_state['feedback'] = None
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display stored answer and feedback
if st.session_state['answer'] is not None:
    st.success("Answer generated:")
    st.markdown(f"**{st.session_state['answer']}**")

    col1, col2 = st.columns(2)
    if col1.button("👍", key="like"):
        st.session_state['feedback'] = "Thanks for your feedback!"
    if col2.button("👎", key="dislike"):
        st.session_state['feedback'] = "Sorry to hear that. We'll improve!"

    if st.session_state['feedback']:
        st.write(st.session_state['feedback'])