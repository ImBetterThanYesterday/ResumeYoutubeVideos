import openai
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

# Función para dividir la transcripción en fragmentos más pequeños
def split_transcript(transcript, max_tokens=2000):
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # Agregar la longitud de la palabra más el espacio
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
    
    # Agregar el último fragmento
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Función para obtener la transcripción del video de YouTube
def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([item['text'] for item in transcript])
    return text

# Función para generar resumen usando LangChain
def summarize_transcript(transcript):
    # Crear un prompt con LangChain
    prompt_template = "Summarize the following video transcript in an engaging and concise way:\n\n{transcript}"
    prompt = PromptTemplate(input_variables=["transcript"], template=prompt_template)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Obtener la clave desde el archivo .env
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    # Usar OpenAI con LangChain para generar el resumen
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Dividir la transcripción en fragmentos más pequeños
    chunks = split_transcript(transcript)
    
    # Generar resumen para cada fragmento y concatenar los resúmenes
    summaries = [chain.run(transcript=chunk) for chunk in chunks]
    full_summary = " ".join(summaries)
    
    return full_summary

# Streamlit: Crear la interfaz de usuario
st.title("YouTube Video Summarizer")

# Input para la URL del video de YouTube
video_url = st.text_input("Enter the YouTube video URL:")

if st.button("Generate Summary"):
    try:
        # Extraer ID del video de YouTube
        video_id = video_url.split('v=')[1]
        
        # Obtener transcripción
        transcript = get_youtube_transcript(video_id)
        
        # Mostrar la transcripción (opcional)
        st.subheader("Transcript:")
        st.write(transcript)
        
        # Generar resumen
        summary = summarize_transcript(transcript)
        
        # Mostrar el resumen
        st.subheader("Summary:")
        st.write(summary)
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
