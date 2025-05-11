import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, render_template, jsonify, request
import getpass
from deep_translator import GoogleTranslator  # ✅ Replaced googletrans (which is broken)
from src.logger import logging as log
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from src.prompt import prompt_template
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

app = Flask(__name__)

# Configuration
class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass.getpass("Pinecone API Key (recommend to set in .env): ")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or getpass.getpass("Groq API Key (recommend to set in .env): ")
    INDEX_NAME = "medical-chatbot"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.8
    SEARCH_CONFIG = {"k": 3, "score_threshold": 0.2}  # ✅ Improved search threshold

app.config.from_object(Config)

if not all([app.config["PINECONE_API_KEY"], app.config["GROQ_API_KEY"]]):
    raise ValueError("Missing required API keys in environment variables")

# ✅ Initialize Components
@lru_cache(maxsize=None)
def initialize_components():
    log.info("Initializing Pinecone connection")
    pc = pinecone.Pinecone(api_key=app.config["PINECONE_API_KEY"])

    log.info("Downloading embeddings model")
    embeddings = HuggingFaceEmbeddings(model_name=app.config["EMBEDDING_MODEL"])

    log.info("Initializing vector store")
    vector_store = PineconeVectorStore(
        index=pc.Index(app.config["INDEX_NAME"]),
        embedding=embeddings
    )

    log.info("Creating prompt template")
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    log.info("Initializing LLM")
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

    llm = ChatGroq(
        model=app.config["LLM_MODEL"],
        temperature=app.config["LLM_TEMPERATURE"],
        max_tokens=512,
        timeout=None,
        max_retries=2,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=app.config["SEARCH_CONFIG"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,  # ✅ Prevents unnecessary sources
        chain_type_kwargs={"prompt": prompt}
    )

qa_chain = initialize_components()

def detect_language(text):
    """Detect language of the given text using GoogleTranslator."""
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        log.error(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails

def translate_text(text, src_lang, dest_lang):
    """Translate text using GoogleTranslator."""
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception as e:
        log.error(f"Translation failed: {e}")
        return text  # Return original text if translation fails

@app.route("/")
def index():
    return render_template('chatttt.html')

@app.route("/get", methods=["POST"])
def handle_query():
    try:
        msg = request.form["msg"].strip()
        if not msg:
            return jsonify({"response": "Please enter a valid query."})

        print(f"User Query: {msg}")

        # Detect Language
        detected_lang = detect_language(msg)
        print(f"Detected Language: {detected_lang}")

        # Translate to English for Processing
        translated_text = translate_text(msg, detected_lang, "en")
        print(f"Translated Input: {translated_text}")

        # Process Query
        response = qa_chain.invoke({"query": translated_text})
        response_text = response.get("result", "Sorry, I couldn't process that.")

        print("Response: ", response_text)

        # Translate response back to the original language
        # final_response = translate_text(response_text, "en", detected_lang)
        # print(f"Final Translated Response: {final_response}")

        return str(response_text)

    except Exception as e:
        log.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")
