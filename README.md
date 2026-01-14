<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Whisper-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="Whisper">
  <img src="https://img.shields.io/badge/LLaMA-3.3-orange?style=for-the-badge&logo=meta&logoColor=white" alt="LLaMA">
  <img src="https://img.shields.io/badge/Groq-API-purple?style=for-the-badge" alt="Groq">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎙️ Voice-Powered Q&A System</h1>

<p align="center">
  <strong>An Intelligent Voice-Based Question Answering System using Speech Recognition, Large Language Models, and Text-to-Speech Synthesis</strong>
</p>

<p align="center">
  <em>Transform spoken questions into comprehensive audio responses powered by state-of-the-art AI</em>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Configuration](#-api-configuration)
- [Examples](#-examples)
- [Performance Metrics](#-performance-metrics)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

The **Voice-Powered Q&A System** is an end-to-end conversational AI pipeline that enables users to ask questions using natural speech and receive intelligent, contextually relevant audio responses. This project demonstrates the seamless integration of three core AI technologies:

1. **Automatic Speech Recognition (ASR)** - Converting voice to text
2. **Large Language Model (LLM)** - Generating intelligent responses
3. **Text-to-Speech (TTS)** - Converting responses back to audio

This system serves as a practical implementation of modern AI capabilities, suitable for educational purposes, accessibility applications, and voice-first interfaces.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Voice Input** | Accept natural speech input in multiple audio formats (WAV, MP3, etc.) |
| 🧠 **Intelligent Responses** | Leverage LLaMA 3.3 70B for comprehensive, accurate answers |
| 🔊 **Audio Output** | Generate natural-sounding speech responses |
| ⚡ **Fast Processing** | Optimized pipeline for minimal latency |
| 🌐 **Multi-Language Support** | Whisper's multilingual capabilities for speech recognition |
| 🔒 **Secure API Integration** | Safe handling of API credentials |
| 📊 **Modular Design** | Clean, extensible architecture for easy customization |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VOICE Q&A SYSTEM PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │              │     │              │     │              │               │
│   │  🎤 AUDIO    │────▶│  📝 TEXT     │────▶│  🔊 AUDIO    │               │
│   │   INPUT      │     │   RESPONSE   │     │   OUTPUT     │               │
│   │              │     │              │     │              │               │
│   └──────┬───────┘     └──────▲───────┘     └──────▲───────┘               │
│          │                    │                    │                        │
│          ▼                    │                    │                        │
│   ┌──────────────┐     ┌──────┴───────┐     ┌──────┴───────┐               │
│   │              │     │              │     │              │               │
│   │   WHISPER    │────▶│   GROQ API   │────▶│    gTTS      │               │
│   │    (ASR)     │     │   (LLaMA)    │     │    (TTS)     │               │
│   │              │     │              │     │              │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│                                                                             │
│   Speech-to-Text      Answer Generation     Text-to-Speech                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Stage**: User provides an audio file containing a spoken question
2. **ASR Stage**: OpenAI Whisper transcribes the audio to text
3. **LLM Stage**: Groq API processes the question using LLaMA 3.3 70B model
4. **TTS Stage**: Google Text-to-Speech converts the response to audio
5. **Output Stage**: Audio response is saved and/or played back

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Speech Recognition** | OpenAI Whisper | Base Model | Transcribe audio to text |
| **Language Model** | LLaMA 3.3 | 70B Versatile | Generate intelligent responses |
| **LLM API** | Groq | Latest | Fast LLM inference |
| **Text-to-Speech** | gTTS | Latest | Convert text to speech |
| **Runtime** | Python | 3.8+ | Primary programming language |

### Dependencies

```
openai-whisper    # Speech recognition
groq              # LLM API client
gtts              # Text-to-speech synthesis
gradio            # Web interface (optional)
ipython           # Interactive computing
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)
- Groq API key ([Get one here](https://console.groq.com/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/voice-qa-system.git
cd voice-qa-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install openai-whisper groq gtts gradio ipython
```

### Step 4: Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

---

## 🚀 Usage

### Basic Usage

```python
import whisper
from groq import Groq
from gtts import gTTS

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize Groq client
client = Groq(api_key="your-api-key")

# Process a voice question
def process_voice_question(audio_path):
    # Step 1: Speech to Text
    result = whisper_model.transcribe(audio_path)
    question = result["text"]
    
    # Step 2: Generate Answer
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=500
    )
    answer = response.choices[0].message.content
    
    # Step 3: Text to Speech
    tts = gTTS(text=answer, lang='en')
    tts.save("answer.mp3")
    
    return question, answer, "answer.mp3"

# Example usage
question, answer, audio_file = process_voice_question("sample_question.wav")
print(f"Question: {question}")
print(f"Answer: {answer}")
```

### Running the Notebook

```bash
jupyter notebook Voice_QA_System.ipynb
```

---

## 📁 Project Structure

```
voice-qa-system/
│
├── 📓 Voice_QA_System.ipynb      # Main Jupyter notebook implementation
├── 📄 README.md                   # Project documentation
├── 📄 Voice_QA_System_Report.pdf  # Academic project report
├── 📄 Voice_QA_System_Report.docx # Editable report document
│
├── 🎵 sample_question.wav         # Sample input audio file
├── 🎵 answer_of_question.mp3      # Generated audio response
│
├── 📁 .venv/                      # Python virtual environment
│
└── 📄 requirements.txt            # Python dependencies (optional)
```

---

## 🔑 API Configuration

### Groq API Setup

1. Visit [Groq Console](https://console.groq.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Store securely and use in your code

```python
# Recommended: Use environment variables
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Or configure directly (not recommended for production)
GROQ_API_KEY = "your-api-key-here"
```

### Security Best Practices

> ⚠️ **Warning**: Never commit API keys to version control!

```bash
# Add to .gitignore
echo "*.env" >> .gitignore
echo ".env" >> .gitignore
```

---

## 📝 Examples

### Example 1: General Knowledge Question

**Input Audio**: "What is artificial intelligence and how does it work?"

**Generated Response**:
> Artificial Intelligence refers to the development of computer systems that can perform tasks typically requiring human intelligence. It works through data collection, processing, learning patterns, decision-making, and taking actions. Types include Narrow AI, General AI, and Superintelligence.

### Example 2: Technical Question

**Input Audio**: "Explain machine learning in simple terms."

**Generated Response**:
> Machine learning is a subset of AI where computers learn from data without explicit programming. It identifies patterns and makes predictions, improving over time with more data exposure.

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **ASR Latency** | ~2-5s | Depends on audio length |
| **LLM Response Time** | ~1-3s | Groq's optimized inference |
| **TTS Generation** | ~1-2s | Varies with response length |
| **Total Pipeline** | ~5-10s | End-to-end processing |
| **Accuracy** | High | Whisper base model performance |

---

## 🔮 Future Enhancements

- [ ] **Real-time Streaming**: Implement live audio streaming for instant responses
- [ ] **Multi-language TTS**: Add support for multiple output languages
- [ ] **Voice Cloning**: Integrate voice cloning for personalized responses
- [ ] **Context Memory**: Add conversation history for multi-turn dialogues
- [ ] **Web Interface**: Deploy with Gradio or Streamlit for web access
- [ ] **Mobile App**: Create mobile applications for iOS/Android
- [ ] **Custom Wake Word**: Add "Hey Assistant" style activation
- [ ] **Emotion Detection**: Analyze sentiment in questions for better responses

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where appropriate
- Write unit tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **[OpenAI Whisper](https://github.com/openai/whisper)** - State-of-the-art speech recognition
- **[Groq](https://groq.com/)** - Ultra-fast LLM inference platform
- **[Meta LLaMA](https://llama.meta.com/)** - Powerful open-source language model
- **[Google TTS](https://cloud.google.com/text-to-speech)** - Natural text-to-speech synthesis
- **Bahçeşehir University** - Academic support and resources

---

<p align="center">
  <strong>Built with ❤️ for the future of voice AI</strong>
</p>

<p align="center">
  <a href="#-voice-powered-qa-system">⬆️ Back to Top</a>
</p>
