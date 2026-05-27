# 🗾 Japan Travel RAG Chatbot

A RAG-powered (Retrieval-Augmented Generation) chatbot that answers questions about traveling to Japan, built on top of the official JNTO (Japan National Tourism Organization) website.

## 🧠 Architecture

```
japan.travel (JNTO) → Scraper → Raw Data → Q/A Generator → qa_dataset.csv
                                          ↓
                                   ChromaDB (Vector Store)
                                          ↓
                             Hybrid Retriever (Q/A + Vector Search)
                                          ↓
                              Claude LLM → Streamlit Chat UI
```

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/josejulian1604/agai03-rag-japantravel.git
cd agai03-rag-japantravel
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

4. Run the data pipeline (first time only):
```bash
python src/scraper.py
python src/qa_generator.py
python src/vector_store.py
```

5. Launch the chatbot:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
rag-chatbot/
├── data/
│   ├── raw/              # Scraped pages (JSON)
│   ├── processed/        # Cleaned text
│   └── qa_dataset.csv    # Generated Q/A pairs
├── src/
│   ├── scraper.py        # Web scraper (JNTO)
│   ├── qa_generator.py   # Synthetic Q/A generation
│   ├── vector_store.py   # ChromaDB setup & indexing
│   ├── retriever.py      # Hybrid retrieval logic
│   └── chatbot.py        # Chatbot orchestration
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠️ Tech Stack

- **Scraping:** requests, BeautifulSoup4, playwright
- **LLM:** Claude (Anthropic) via LangChain
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **UI:** Streamlit

## 👤 Author

José Julián Gutiérrez Badilla  
Software Engineering — ITCR  
[LinkedIn](https://www.linkedin.com/in/jose-julian-gutierrez-badilla-8095a528a) | [GitHub](https://github.com/josejulian1604)

## 📚 Course

AGAI-03 — Agentic AI Development Bootcamp
