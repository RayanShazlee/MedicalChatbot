# Medical Chatbot

An advanced medical chatbot powered by Llama 3.3 and LangChain, featuring multilingual support, real-time translation, and an intuitive user interface.

## Features

- 🌐 **Multilingual Support**: Supports multiple languages including English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Japanese, Korean, and Chinese
- 🤖 **Advanced AI**: Powered by Llama 3.3 70B model for accurate medical information
- 🔄 **Real-time Translation**: Automatic language detection and translation
- 💬 **Chat History**: Persistent chat history with organized message display
- 🛡️ **Security Features**: Input sanitization, rate limiting, and error handling
- 📱 **Responsive Design**: Modern, mobile-friendly interface with smooth animations
- ⚡ **Performance Optimized**: Caching and efficient data handling

## Installation Steps

### Step 1. Clone the repository

```bash
git clone https://github.com/RayanShazlee/MedicalChatbot.git
cd MedicalChatbot/
```

### Step 2. Set Up Python Environment

Using conda:
```bash
conda create -n chatbot python=3.12 -y
conda activate chatbot
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Step 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4. Configure Environment Variables

Create a `.env` file in the root directory with the following:

```ini
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### Step 5. Initialize Vector Store
```bash
python store_index.py
```

### Step 6. Run the Application
```bash
python app.py
```

Access the application at `http://localhost:8080`

## Tech Stack

### Backend
- **Python 3.12**: Core programming language
- **Flask**: Web framework with advanced features:
  - Rate limiting
  - Error handling
  - Session management
- **LangChain**: AI/ML pipeline management
- **Groq**: High-performance LLM hosting
- **Pinecone**: Vector database for semantic search
- **Deep Translator**: Multilingual support

### Frontend
- **Bootstrap 5**: Responsive design framework
- **jQuery**: AJAX and DOM manipulation
- **GSAP**: Smooth animations
- **Font Awesome**: Modern icons

### Features in Detail

#### AI Capabilities
- Medical knowledge base integration
- Context-aware responses
- Multi-turn conversations
- Source citation

#### Security
- Input validation and sanitization
- Rate limiting protection
- Error handling and logging
- Secure API key management

#### User Interface
- Responsive chat interface
- Real-time message updates
- Typing animations
- Message alignment and formatting
- Persistent chat history
- Mobile-friendly design

## Project Structure

```
MedicalChatbot/
├── app.py              # Main application file
├── src/               # Core functionality
│   ├── logger.py     # Logging configuration
│   ├── prompt.py     # LLM prompt templates
│   └── utils.py      # Utility functions
├── static/           # Frontend assets
│   └── stylee.css   # Custom styling
├── templates/        # HTML templates
│   └── chatttt.html # Main chat interface
├── data/            # Data storage
└── logs/            # Application logs
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Llama 3.3 by Meta AI
- LangChain community
- Groq for LLM hosting
- Pinecone for vector search capabilities