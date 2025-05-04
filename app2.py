from googletrans import Translator
import streamlit as st
st.set_page_config(page_title="Multimodal AI Chat", layout="wide")
st.title("📚 Chat with YouTube, PDF or Local Video")
import os
import re
import tempfile
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import time
from pathlib import Path
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
from gtts import gTTS
from vapi import Vapi 
import base64
from langdetect import detect
from deep_translator import GoogleTranslator

st.title("Multimodal Voice Assistant")

if st.button("Start Voice Assistant"):
    vapi = Vapi(api_key='c4e52f7a-4569-4946-8303-83309a75b7d0')
    assistant_config = {
        "firstMessage": "Hello! How can I assist you today?",
        "context": "You are a helpful AI assistant.",
        "model": "vapi",
        "voice": "Neha",
        "recordingEnabled": True,
        "interruptionsEnabled": False
    }
    vapi.start(assistant=assistant_config)

language_options = { ... }  # Keep full dictionary as is

user_lang_name = st.selectbox("Select your known language:", list(language_options.keys()))
output_lang_name = st.selectbox("Select the language you want the answer in:", list(language_options.keys()))
user_lang = language_options[user_lang_name]
output_lang = language_options[output_lang_name]

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("⚠ Google API Key is missing! Please set it in your environment variables.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-1.5-pro-latest"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]+)",
        r"youtube\.com/embed/([a-zA-Z0-9_-]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            return "Invalid YouTube URL."
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_data])
        return transcript_text
    except TranscriptsDisabled:
        return "Transcript is not available for this video."
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"

def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        text = ""
        with open(temp_file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        os.remove(temp_file_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def process_text_for_chat(text, storage_name):
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_text(text)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(storage_name)
        return "✅ Data processed! Now ask questions."
    except Exception as e:
        return f"❌ Error processing text: {e}"

def get_answer(query, storage_name):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(storage_name, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(query)
        if not docs:
            return "🤔 I couldn't find an answer. Try rephrasing your question."
        prompt_template = """
        Answer the question in detail based on the provided document content. If the answer is not found, say:
        \"I couldn't find the answer in the document.\" 

        Document Context:\n {context}\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"⚠ Error processing query: {e}"

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            tts.save(temp_audio.name)
            audio_file = temp_audio.name
        return audio_file
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

def translate_text(text, target_lang):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return text

mode = st.radio("Choose an option:", ["📄 Chat with PDF", "🎥 Chat with YouTube Video", "🎬 Chat with Local Video"])

if mode == "📄 Chat with PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        with st.spinner("⏳ Extracting text..."):
            pdf_text = extract_text_from_pdf(pdf_file)
        if "Error" not in pdf_text:
            with st.spinner("⏳ Processing text for Q&A..."):
                process_status = process_text_for_chat(pdf_text, "faiss_pdf")
                st.success(process_status)
    user_question = st.text_input("Ask a question about the PDF:")
    if user_question:
        translated_question = translate_text(user_question, 'en')
        response = get_answer(translated_question, "faiss_pdf")
        translated_response = translate_text(response, output_lang)
        st.markdown(f"🧑‍💻 You:** {user_question}")
        st.markdown(f"🤖 AI:** {translated_response}")
        audio_file = text_to_speech(translated_response, lang_code=output_lang)
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
            os.remove(audio_file)

elif mode == "🎥 Chat with YouTube Video":
    youtube_link = st.text_input("Enter YouTube Video Link:")
    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
    if st.button("Process Video Transcript"):
        if not youtube_link:
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("⏳ Fetching transcript..."):
                transcript_text = extract_transcript_details(youtube_link)
                if "Error" not in transcript_text and "Transcript is not available" not in transcript_text:
                    with st.spinner("⏳ Processing transcript for Q&A..."):
                        process_status = process_text_for_chat(transcript_text, "faiss_youtube")
                        st.success(process_status)
                else:
                    st.error(transcript_text)
    user_question = st.text_input("Ask a question about the video:")
    if user_question:
        translated_question = translate_text(user_question, 'en')
        response = get_answer(translated_question, "faiss_youtube")
        translated_response = translate_text(response, output_lang)
        st.markdown(f"🧑‍💻 You:** {user_question}")
        st.markdown(f"🤖 AI:** {translated_response}")
        audio_file = text_to_speech(translated_response, lang_code=output_lang)
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
            os.remove(audio_file)

elif mode == "🎬 Chat with Local Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], help="Upload a video for AI analysis")
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        st.video(video_path, format="video/mp4", start_time=0)
        user_query = st.text_area("What insights are you seeking from the video?", placeholder="Ask anything about the video content.")
        if st.button("🔍 Analyze Video"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    translated_query = translate_text(user_query, 'en')
                    with st.spinner("Processing video and gathering insights..."):
                        processed_video = upload_file(video_path)
                        time.sleep(2)
                    response = multimodal_Agent.answer(query=translated_query)
                    translated_response = translate_text(response, output_lang)
                    st.markdown(f"🧑‍💻 You:** {user_query}")
                    st.markdown(f"🤖 AI:** {translated_response}")
                    audio_file = text_to_speech(translated_response, lang_code=output_lang)
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                        os.remove(audio_file)
                except Exception as e:
                    st.error(f"Error analyzing video: {str(e)}")
