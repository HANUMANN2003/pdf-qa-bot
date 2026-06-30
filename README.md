# PDF Question Answering System

Upload a PDF, ask questions about it, get answers with page-level source references.

## How it works

1. PDF is loaded and split into ~1000-character chunks
2. Each chunk is embedded (OpenAI embeddings) and stored in ChromaDB (in-memory, per session)
3. Your question is embedded too, and the 3 most similar chunks are retrieved
4. Those chunks + your question go to `gpt-4o-mini`, which writes the answer
5. The source chunks are shown below the answer so you can verify it

---

## 1. Run it locally

```bash
# 1. Clone/unzip this folder, then cd into it
cd pdf-qa-bot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux
# then open .env and paste your real key in place of the placeholder

# 5. Run the app
streamlit run app.py
```

Your browser will open at `http://localhost:8501`. If `.env` has a valid key, the sidebar shows "API key loaded ✅" automatically — otherwise you can paste a key directly into the sidebar at runtime (handy for testing without saving it to disk).

Get an API key at: https://platform.openai.com/api-keys

---

## 2. Deploy to Streamlit Community Cloud (free)

1. Push this folder to a new GitHub repo (e.g. `pdf-qa-bot`):
   ```bash
   git init
   git add .
   git commit -m "PDF Q&A system"
   git branch -M main
   git remote add origin https://github.com/<your-username>/pdf-qa-bot.git
   git push -u origin main
   ```
   (`.gitignore` already excludes `.env`, so your key won't be pushed.)

2. Go to https://share.streamlit.io and sign in with GitHub.

3. Click **New app**, select your `pdf-qa-bot` repo, branch `main`, file `app.py`.

4. Before deploying, click **Advanced settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```
   This is the cloud equivalent of your local `.env` file — the app reads it automatically via `st.secrets`.

5. Click **Deploy**. After a minute or two, you'll get a live URL like:
   `https://pdf-qa-bot-<random>.streamlit.app`

---

## Notes for beginners

- **Cost**: each question costs a small fraction of a cent (embeddings) plus a `gpt-4o-mini` call (also very cheap). Fine for testing/portfolio use.
- **No OpenAI budget?** Swap `OpenAIEmbeddings`/`ChatOpenAI` for free alternatives — ask and I'll give you a Groq + HuggingFace embeddings version (same architecture, different providers).
- **Chroma storage**: this version uses an in-memory Chroma collection per uploaded file (simplest for deployment — no disk persistence issues on Streamlit Cloud's ephemeral filesystem). If you want it to remember PDFs across restarts, that needs a persistent vector DB (e.g. Chroma Cloud, Pinecone) — let me know if you want that upgrade.
