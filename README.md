# Historical Professor Agent

## Overview
This project implements an AI-powered historical professor agent named **"Professor Li Bowen"** using OpenAI's GPT-3.5 model. The agent features:

- Expert knowledge in Chinese and world history  
- Emotion-aware responses (5 mood modes)  
- Voice synthesis via Microsoft TTS  
- Web content (urls) learning capability  
- Telegram bot integration  
- Docker container deployment

## Technical Components
- **OpenAI GPT-3.5** for conversation  
- **Microsoft Azure Cognitive Services** for text-to-speech  
- **Qdrant** for vector storage and retrieval  
- **Redis** for conversation memory  
- **FastAPI** backend  
- **Telegram bot** interface  
- **Docker** containerization

## Setup
### Clone the repository:
```bash
git clone https://github.com/DREWdrew53/First-Agent.git
```

### Install dependencies:
```bash
pip install -r requirements.txt
```
### API Configuration
You need to set up the APIs fisrt:
- OpenAi 
- Serpapi 
- LangSmith 
- Telegram bot

## Usage
### Running the Agent
```bash
docker run -p 6379:6379 redis  # start redis
python server.py  # launch the FastAPI server
python tele.py  # run the Telegram bot
```

### Docker Deployment
```bash
docker-compose up -d
```
## API Endpoints

| Method    | Endpoint    | Description    |
|-----------|-------------|----------------|
| POST      | /chat       | Main conversation endpoint |
| POST      | /add_urls   | Add web content to knowledge base |
| WS        | /ws         | WebSocket endpoint for real-time communication |

## Troubleshooting

### Voice synthesis fails:
✅ Verify Microsoft TTS API key  
🌐 Check internet connection  
📌 Ensure correct Azure region endpoint  

#### **Knowledge retrieval problems:**
🔍 Confirm Qdrant is running  
🗂 Check vector storage path permissions  

#### **Memory issues:**
🔄 Verify Redis server is accessible  
🔑 Check session ID consistency  

