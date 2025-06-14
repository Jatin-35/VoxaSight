# VoxaSight – A Voice and Vision AI Agent

**VoxaSight** is a multimodal AI assistant that combines speech and vision to enable interactive learning, especially designed to support inclusive learning. It leverages voice activity detection, real-time speech understanding, contextual image input, and conversational intelligence using state-of-the-art AI models.

---

## 🚀 Features

- 🎤 **Speech Recognition** using Deepgram (STT)
- 🧠 **Conversational Intelligence** powered by Google Gemini LLM
- 🗣️ **Text-to-Speech Synthesis** using Cartesia / Deepgram TTS
- 🧏‍♂️ **Voice Activity Detection** with Silero VAD
- 🌐 **LiveKit Cloud Integration** for real-time video/audio handling
- 👁️ **Image-Based Contextual Understanding** using `ImageContent`
- 🌍 **Multilingual Turn Detection** for smooth interaction

---

## 📁 File Structure

```
VOXASIGHT/
│
├── src/
│ ├── agent.py # Main application entrypoint and session setup (New  Version)
│ └── assistant.py # Code with old version logic v0.x
│
├── .env # API keys and credentials (not committed)
├── .gitignore # Ignore env, pycache, etc.
├── requirements.txt # Python dependencies
└── venv/ # Python virtual environment (ignored)
---

## 🛠️ Tech Stack

- **Language:** Python
- **Frameworks & APIs:**
  - LiveKit Cloud (WebRTC audio/video)
  - Deepgram (STT, TTS)
  - Google Gemini 2.0 (LLM)
  - Silero VAD
  - Cartesia (TTS)
  - WebRTC, asyncio
- **VoicePipelineAgent Architecture:**
  - `STT ➝ LLM ➝ TTS`

---

## 📦 Installation

### 1. Clone the repository
```
git clone https://github.com/Jatin-35/VoxaSight.git
cd VoxaSight
```

### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a .env file in the root directory:
```
LIVEKIT_URL=wss://your-livekit-instance.livekit.cloud
LIVEKIT_API_KEY=ADD_YOUR_LIVEKIT_API_KEY
LIVEKIT_API_SECRET=ADD_YOUR_LIVEKIT_API_SECRET

DEEPGRAM_API_KEY=ADD_YOUR_DEEPGRAM_API_KEY
GOOGLE_API_KEY=ADD_YOUR_GOOGLE_API_KEY
OPENAI_API_KEY=ADD_IF_REQUIRED
OPENAI_API_BASE=https://openrouter.ai/api/v1
```
### ▶️ Run the Project
Make sure you have LiveKit project credentials.

Then run:
```
python src\agent.py dev 
```
---

##  🧠 How It Works

  - Connects to a LiveKit Room with audio and video tracks.
  - Captures user speech using Deepgram STT and video frames from webcam.
  - Converts voice → text using Deepgram STT.
  - Appends webcam frame as ImageContent.
  - Enhances input with Silero VAD and Turn Detection.
  - Passes input (text + image) to Google Gemini LLM.
  - Responds back with natural voice using Cartesia TTS or Deepgram TTS.
    

---

## 🧩 Use Cases

  - Assistive land Inclusive Learning
  - Voice and vision-based interactive tutoring
  - Real-time feedback and support using multimodal input

--- 

🙌 Acknowledgements

  - LiveKit
  - Deepgram
  - Google Gemini
  - Cartesia
  - Silero

---

## 💡 Future Work

  -  Add memory and personalization per child
  -  Integrate real-time feedback loops
  -  Expand support for more languages and disabilities





