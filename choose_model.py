from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

class ModelManager:
    """Gestiona diferentes modelos de lenguaje"""
    
    @staticmethod
    def get_model(model_type: str, model_name: Optional[str] = None):
        """
        Retorna una instancia del modelo seleccionado
        
        Args:
            model_type: 'ollama' o 'openai'
            model_name: nombre específico del modelo
        """
        if model_type == "ollama":
            return Ollama(model=model_name or "granite3-moe:1b")
        elif model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            return ChatOpenAI(model=model_name or "gpt-3.5-turbo", api_key=api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

def split_transcript(transcript: str, max_tokens: int = 2000) -> list:
    """Divide la transcripción en fragmentos más pequeños"""
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_youtube_transcript(video_id: str) -> str:
    """Obtiene la transcripción del video de YouTube"""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([item['text'] for item in transcript])

def process_text(text: str, prompt_template: str, model_type: str, model_name: str) -> str:
    """Procesa texto usando el modelo seleccionado"""
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    llm = ModelManager.get_model(model_type, model_name)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    chunks = split_transcript(text)
    results = [chain.run(text=chunk) for chunk in chunks]
    return " ".join(results)

def summarize_transcript(transcript: str, model_type: str, model_name: str) -> str:
    """Genera resumen del texto"""
    prompt_template = """
    Summarize the following video transcript in an engaging and concise way. 
    Include the main points and key takeaways:

    {text}
    """
    return process_text(transcript, prompt_template, model_type, model_name)

def translate_summary(summary: str, language: str, model_type: str, model_name: str) -> str:
    """Traduce el resumen al idioma seleccionado"""
    prompt_template = f"""
    Translate the following text to {language}, maintaining its meaning and style:

    {{text}}
    """
    return process_text(summary, prompt_template, model_type, model_name)

# Configuración de Streamlit
st.title("YouTube Video Summarizer and Translator")

# Selección de modelo
model_type = st.selectbox(
    "Select Model Type",
    ["ollama", "openai"],
    help="Choose the AI model service you want to use"
)

# Opciones específicas de modelo según el tipo seleccionado
if model_type == "ollama":
    model_name = st.selectbox(
        "Select Ollama Model",
        ["granite3-moe:1b", "llama2", "mistral"],
        help="Choose the specific Ollama model"
    )
else:  # openai
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4"],
        help="Choose the specific OpenAI model"
    )

# Input para la URL
video_url = st.text_input("Enter the YouTube video URL:")

# Session state
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'translation' not in st.session_state:
    st.session_state['translation'] = None

# Botón para generar resumen
if st.button("Generate Summary"):
    try:
        with st.spinner("Getting transcript..."):
            video_id = video_url.split('v=')[1]
            st.session_state['transcript'] = get_youtube_transcript(video_id)
        
        with st.spinner("Generating summary..."):
            st.session_state['summary'] = summarize_transcript(
                st.session_state['transcript'],
                model_type,
                model_name
            )
        st.session_state['translation'] = None
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

# Mostrar resultados
if st.session_state['transcript']:
    with st.expander("Show Transcript"):
        st.write(st.session_state['transcript'])

if st.session_state['summary']:
    st.subheader("Summary (English):")
    st.write(st.session_state['summary'])

    # Mostrar el selector de idioma y el botón de traducción después de generar el resumen
    language = st.selectbox(
        "Select Language for Translation",
        ["Spanish", "French", "German", "Chinese", "Japanese"],
        help="Choose the language to translate the summary into"
    )

    # Botón para traducir
    if st.button("Translate Summary"):
        try:
            with st.spinner("Translating..."):
                st.session_state['translation'] = translate_summary(
                    st.session_state['summary'],
                    language,
                    model_type,
                    model_name
                )
        except Exception as e:
            st.error(f"Error translating: {str(e)}")

if st.session_state['translation']:
    st.subheader(f"Translation ({language}):")
    st.write(st.session_state['translation'])
