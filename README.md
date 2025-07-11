# 🎯 AI for Marketing - Multi-Model Generator

An AI-powered Streamlit app to generate *ad copy, **slogans, and **marketing campaign ideas* using multiple LLMs like *Together.ai (Mistral), **Ollama (local), and **Groq (LLaMA3)*.

---

## 🚀 Features

- ✨ *Ad Content Generation*
  - Generate ad copy, slogans, and creative campaign ideas based on product, audience, and features.

- 🧠 *Multi-Model Support*
  - Choose between:
    - Together.ai (Mistral 7B)
    - Ollama (local LLaMA2)
    - Groq (LLaMA3 70B)

- 🎤 *Voice Input*
  - Speak your product idea directly (uses microphone and Google Speech Recognition).

- 📝 *Prompt Visualization*
  - Shows the final prompt constructed from your inputs.

- 📊 *Model Performance Comparison*
  - Compares the response time of each model visually using a Plotly bar chart.

- 📥 *Export Options*
  - Download the generated content as .txt or .pdf.

---

## 🛠 Tech Stack

- *Frontend*: Streamlit
- *Backend*: Python
- *LLMs*: Together.ai, Groq, Ollama (local)
- *APIs & Tools*:
  - requests, speechrecognition, reportlab, pandas, plotly
  - langchain, langchain_community

---

## ⚙ Installation

1. *Clone the repository*
   ```bash
   git clone https://github.com/yourusername/ai-marketing-generator.git
   cd ai-marketing-generator

   step2:  Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
step3: Install dependencies
pip install -r requirements.txt
step4: Set API keys
Create a .env file in the root folder:
TOGETHER_API_KEY="your_together_api_key"
GROQ_API_KEY="your_groq_api_key"

▶ Running the App
streamlit run main.py
🎤 For voice input, make sure your microphone is enabled.