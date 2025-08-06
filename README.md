# 🧐 NGHA | TAIF AI Assistant

A smart, voice-enabled chatbot that answers user questions based on uploaded **PDF** and **Word (DOCX)** files using **GPT-3.5-Turbo** via OpenAI API. Includes cost tracking, speaker playback, and streamlined UI built with **Streamlit** + **LangChain**.

---

## 🔹 Features

| Feature                       | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| ✅ OpenAI GPT-3.5 (via API)    | Fast and accurate answers using `gpt-3.5-turbo`  |
| 📁 PDF & DOCX support         | Upload documents to query from                   |
| 🔍 RAG (LangChain + ChromaDB) | Retrieves relevant text before answering         |
| 🔊 Voice auto-read (TTS)      | Web Speech API auto-detects Arabic or English    |
| 💵 Token usage & cost tracker | Shows per-response usage based on OpenAI pricing |
| 💡 Suggested questions        | Starts conversation with clickable prompts       |
| 🩱 “New Chat” button          | Clears chat and resets state                     |
| 🔄 “Sync Now” button          | Refreshes vector DB from uploaded files          |
| 📜 Typewriter animation       | Displays text word-by-word for engagement        |

---

## 🤩 Project Structure

```
📆 your-project/
👉 app.py                    ← Main Streamlit chatbot
👉 requirements.txt
👉 .env                      ← Stores your OpenAI API key
👉 logo_image_en.png         ← Logo/avatar for AI
👉 data/
📌 uploads/              ← Your uploaded PDF/DOCX files
📌 chroma/               ← ChromaDB folder for embeddings
👉 chains.py                 ← LangChain retriever logic
👉 tts.py                    ← Text-to-speech (speaker) integration
👉 vectorstore.py            ← PDF/DOCX ingestion & indexing
👉 config.py                 ← Model, settings, costs, constants
👉 speech_utils.py           ← Speaker button HTML logic
```

---

## ⚙️ Installation

### 1. 🐍 Python & VS Code

* Install [Python 3.10 – 3.13](https://www.python.org/downloads/) (tested on 3.13 ✅)
* Install [VS Code](https://code.visualstudio.com/)
* Ensure you have C++ runtime (x64/x86) if using Windows

---

### 2. 🔧 Setup Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux
```

---

### 3. 📆 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. 🔑 Add OpenAI Key

Create a `.env` file with:

```
OPENAI_API_KEY=sk-xxxxxxx
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

> GPT-3.5 is fast and cost-efficient. You can also use `gpt-4` later by changing the model name.

---

### 5. 🚀 Run the App

```bash
streamlit run app.py
```

Then open in your browser:
🔗 [http://localhost:8501/](http://localhost:8501/)

---

## 📟 Token Pricing & Calculation

| Model         | Price per 1K input tokens | Price per 1K output tokens |
| ------------- | ------------------------- | -------------------------- |
| gpt-3.5-turbo | \$0.0005                  | \$0.0015                   |

The app displays:

* Total tokens per message
* Estimated cost
* Running total in the sidebar

---

## 🚩 Voice Output

* Automatically detects Arabic or English
* Uses browser’s built-in voice
* Appears as 🔉 / 🔇 next to AI responses
* No server-side audio processing needed

---

## 🦼️ Reset & Sync Functions

* **🩱 New Chat**: Resets history, token counters, and suggestion state
* **🔄 Sync Now**: Re-indexes documents in `/data/uploads/` (PDF/DOCX) and updates ChromaDB

---

## ✅ Tested On

| Platform        | Status |
| --------------- | ------ |
| Windows 11      | ✅      |
| Chrome/Edge     | ✅      |
| Python 3.13     | ✅      |
| Streamlit 1.35+ | ✅      |

---

## 📌 Notes

* `sqlite` used via Chroma for vector persistence (auto-managed)
* Uses `RecursiveCharacterTextSplitter` and `OllamaEmbeddings` via LangChain
* If TTS doesn't work, check browser support for Web Speech API
* No Ollama needed — this project runs **fully with OpenAI APIs**

---

## 🧠 Credits

Developed by Yousef Banawi
Using: Streamlit • OpenAI • LangChain • ChromaDB ❤️
