from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def extract_video_id(url_or_id):
    """
    Extracts the video ID from a YouTube URL, or returns the input if it's already an ID.
    """
    if "youtu.be" in url_or_id or "youtube.com" in url_or_id:
        parsed = urlparse(url_or_id)
        # Short link: youtu.be/VIDEO_ID
        if parsed.hostname and "youtu.be" in parsed.hostname:
            return parsed.path.lstrip("/")
        # Standard URL: youtube.com/watch?v=VIDEO_ID
        query = parse_qs(parsed.query)
        return query.get("v", [None])[0]
    return url_or_id

def get_transcript(video_id):
    """
    Fetches the transcript for the given video ID, or returns None if unavailable.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return None

def create_faiss_index(text, index_path):
    """
    Splits text into chunks, embeds them, and builds a FAISS index saved on disk.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    return db

def load_or_create_index(video_url_or_id):
    """
    Loads an existing FAISS index for the video, or creates one if it doesn't exist.
    """
    video_id = extract_video_id(video_url_or_id)
    index_path = f"indexes/{video_id}"
    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    transcript = get_transcript(video_id)
    if not transcript:
        return None
    return create_faiss_index(transcript, index_path)