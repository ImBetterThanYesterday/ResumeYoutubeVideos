from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
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

# Función para generar resumen usando LangChain con Ollama
def summarize_transcript(transcript):
    # Crear un prompt con LangChain
    prompt_template = "Summarize the following video transcript in an engaging and concise way:\n\n{transcript}"
    prompt = PromptTemplate(input_variables=["transcript"], template=prompt_template)
    
    # Usar Ollama con LangChain para generar el resumen
    llm = Ollama(model="granite3-moe:1b")
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Dividir la transcripción en fragmentos más pequeños
    chunks = split_transcript(transcript)
    
    # Generar resumen para cada fragmento y concatenar los resúmenes
    summaries = [chain.run(transcript=chunk) for chunk in chunks]
    full_summary = " ".join(summaries)
    
    return full_summary

# Función para traducir el resumen usando LangChain con Ollama
def translate_summary(summary):
    # Crear un prompt con LangChain
    prompt_template = "Translate the text to Spanish:\n\n{summary}"
    prompt = PromptTemplate(input_variables=["summary"], template=prompt_template)
    
    # Usar Ollama con LangChain para traducir el resumen
    llm = Ollama(model="granite3-moe:1b")
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generar la traducción
    translation = chain.run(summary=summary)
    
    return translation

# Streamlit: Crear la interfaz de usuario
st.title("YouTube Video Summarizer and Translator")

# Input para la URL del video de YouTube
video_url = st.text_input("Enter the YouTube video URL:")

# Inicializar session state para almacenar transcripción, resumen y traducción
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'translation' not in st.session_state:
    st.session_state['translation'] = None

# Botón para generar resumen
if st.button("Generate Summary"):
    try:
        # Extraer ID del video de YouTube
        video_id = video_url.split('v=')[1]
        
        # Obtener transcripción
        st.session_state['transcript'] = get_youtube_transcript(video_id)
        
        # Generar resumen
        st.session_state['summary'] = summarize_transcript(st.session_state['transcript'])
        
        # Restablecer la traducción
        st.session_state['translation'] = None
        
    except Exception as e:
        st.error(f"Error processing video: {e}")

# Mostrar la transcripción y resumen solo si existen
if st.session_state['transcript']:
    st.subheader("Transcript:")
    st.write(st.session_state['transcript'])

if st.session_state['summary']:
    st.subheader("Summary (English):")
    st.write(st.session_state['summary'])

# Botón para traducir resumen
if st.session_state['summary'] and st.button("Translate Summary"):
    try:
        # Traducir resumen
        st.session_state['translation'] = translate_summary(st.session_state['summary'])
        
    except Exception as e:
        st.error(f"Error processing translation: {e}")

# Mostrar la traducción solo si existe
if st.session_state['translation']:
    st.subheader("Translation (Spanish):")
    st.write(st.session_state['translation'])