import streamlit as st
import os
import requests
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import base64
import speech_recognition as sr
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import plotly.express as px

load_dotenv()

# -------------------- API KEYS --------------------
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- PROMPT BUILDER --------------------
def build_prompt(task, product, audience, features):
    if task == "Ad Copy":
        return f"Write a compelling ad copy for {product} targeting {audience}, highlighting features such as {features}."
    elif task == "Slogan":
        return f"Generate 3 catchy slogans for a brand that sells {product} to {audience}."
    elif task == "Campaign Idea":
        return f"Suggest a creative marketing campaign idea for {product}, targeting {audience}. Include the theme, channels to use, and a sample tagline."
    return ""

# -------------------- LLM CALLS --------------------
def together_response(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }
    res = requests.post(url, headers=headers, json=data)
    return res.json()['choices'][0]['message']['content']

def groq_response(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }
    res = requests.post(url, headers=headers, json=data)
    return res.json()['choices'][0]['message']['content']

def ollama_response(prompt):
    try:
        llm = Ollama(model="llama2")
        chain = PromptTemplate.from_template("{question}") | llm | StrOutputParser()
        return chain.invoke({"question": prompt})
    except Exception as e:
        return f"[Ollama not available locally: {str(e)}]"

# -------------------- SPEECH TO TEXT --------------------
def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition."

# -------------------- PDF EXPORT --------------------
def export_as_pdf(text, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split('\n'):
        c.drawString(40, y, line)
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="AI for Marketing", layout="wide")

# -------------------- STYLING --------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');

        body {
            background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
            font-family: 'Quicksand', sans-serif;
        }
        .main > div {
            background-color: #ffffffdd;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h4 {
            color: #2e7d32;
        }
        label, .stTextInput, .stSelectbox, .stTextArea, .stButton button {
            font-family: 'Quicksand', sans-serif;
            color: #3e2723;
        }
        .stButton button {
            background: linear-gradient(to right, #43cea2, #185a9d);
            color: white;
            border-radius: 10px;
            padding: 12px 18px;
            font-weight: bold;
            border: none;
        }
        .stTextInput > div > input, .stTextArea > div > textarea {
            background-color: #e8f5e9;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ App Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"], index=0)
task_type = st.sidebar.selectbox("What do you want to generate?", ["Ad Copy", "Slogan", "Campaign Idea"])
product = st.sidebar.text_input("Enter the Brand/Product", "Red Bull")
audience = st.sidebar.text_input("Describe your target audience", "Gen Z urban youth")
features = st.sidebar.text_input("List key product features", "Energy boost, trendy design, great taste")
model = st.sidebar.selectbox("Choose a model", ["Together.ai (Mistral)", "Ollama (local)", "Groq (LLaMA3)"])

# -------------------- MAIN --------------------
st.title("🎯 AI for Marketing - Multi Model Generator")

st.markdown("### 🎤 Voice Input")
if st.button("🎙️ Start Recording"):
    voice_input = transcribe_audio()
    st.success(f"Recognized: {voice_input}")
    product = voice_input

prompt = build_prompt(task_type, product, audience, features)

st.markdown("#### ✍️ Generated Prompt")
st.text_area("Prompt", value=prompt, height=100, key="main_prompt")

if st.button("🚀 Generate"):
    with st.spinner("Generating..."):
        times = {}
        responses = {}

        start = time.time()
        responses["Together.ai (Mistral)"] = together_response(prompt)
        times["Together.ai (Mistral)"] = time.time() - start

        start = time.time()
        responses["Ollama (local)"] = ollama_response(prompt)
        times["Ollama (local)"] = time.time() - start

        start = time.time()
        responses["Groq (LLaMA3)"] = groq_response(prompt)
        times["Groq (LLaMA3)"] = time.time() - start

    st.markdown("#### 📊 Model Comparison Chart")
    df = pd.DataFrame({"Model": list(times.keys()), "Time Taken (s)": list(times.values())})
    fig = px.bar(df, x="Model", y="Time Taken (s)", color="Model", title="Response Time per Model")
    st.plotly_chart(fig)

    st.markdown("#### 📢 Selected Output")
    st.write(responses[model])

    st.download_button("📥 Download TXT", responses[model], file_name="marketing_output.txt")

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    export_as_pdf(responses[model], pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Export as PDF", f, file_name="marketing_output.pdf")

# -------------------- FOOTER --------------------
st.markdown("""
---
**🔐 Note:** Add your keys in `.streamlit/secrets.toml` or `.env` file:
```
TOGETHER_API_KEY="your-together-key"
GROQ_API_KEY="your-groq-key"
```
Install Ollama and download model with:
```
ollama run llama2
```
""")