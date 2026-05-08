# BSA AI Chatbot

AI-powered customer support chatbot for BSA motorcycle dealers.

## Stack

| Layer | Technology |
|-------|-----------|
| RAG / LLM | Python · FastAPI · OpenAI · Pinecone |
| Front layer | Python · FastAPI · asyncpg |
| Database | PostgreSQL |
| Admin UI | React 18 · Vite · React Router |
| PDF pipeline | Python · FastAPI · PyMuPDF · AWS S3 |

## Architecture

```
Browser / Dealer site
        │
        ▼
Port 3000  →  Front layer  (server/main.py)   — sessions, auth, SSE relay
        │                  └── PostgreSQL      — sessions, messages, handoffs
        ├──▶  Port 8000  →  RAG API  (api.py)           — chat answers
        └──▶  Port 8001  →  Pipeline (production/pipeline_api.py) — PDF ingest
```

## Setup

### 1. Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install -r server/requirements.txt
```

### 2. Environment variables

```bash
# Root .env — RAG core (api.py) + pipeline
copy .env.example .env
# Fill in: OPENAI_API_KEY, PINECONE_API_KEY, AWS_* etc.

# Server .env — front layer (server/main.py)
copy server\.env.example server\.env
# Fill in: DATABASE_URL, ADMIN_PASSWORD etc.
```

### 3. PostgreSQL

Create the database — tables are created automatically on first startup:

```sql
CREATE DATABASE bsa_chat;
```

### 4. React admin build

```bash
cd server\admin-react
npm install
npm run build
```

The build outputs to `server/public/app/` which is served by the front layer.

## Running

Open 3 terminals from the `deploy/` folder:

**Terminal 1 — RAG API (port 8000)**
```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Pipeline API (port 8001)**
```bash
cd production
uvicorn pipeline_api:app --host 127.0.0.1 --port 8001
```

**Terminal 3 — Front layer (port 3000)**
```bash
cd server
uvicorn main:app --host 127.0.0.1 --port 3000
```

## URLs

| Page | URL |
|------|-----|
| Chat Widget demo | http://localhost:3000/test.html |
| Knowledge Base | http://localhost:3000/app/kb |
| Admin — Handoffs | http://localhost:3000/app/admin |
| Admin — Analytics | http://localhost:3000/app/analytics |
| Admin — Content | http://localhost:3000/app/content |
| Admin — Pipeline | http://localhost:3000/app/pipeline |

## Dealer Embed

Add this script tag to any dealer website:

```html
<script src="https://YOUR-DOMAIN/widget.js"
        data-dealer="dealer-id">
</script>
```

## Environment variables reference

### `deploy/.env`

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (embeddings + LLM) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX` | Pinecone index name |
| `AWS_ACCESS_KEY_ID` | AWS key (S3 image storage) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret |
| `S3_BUCKET` | S3 bucket name |
| `AWS_REGION` | AWS region (default: ap-southeast-2) |
| `CONFIDENCE_THRESHOLD` | Below this score → suggest handoff (default: 0.45) |

### `deploy/server/.env`

| Variable | Description |
|----------|-------------|
| `PORT` | Front layer port (default: 3000) |
| `DATABASE_URL` | PostgreSQL connection string |
| `DATABASE_SSL` | Set `true` for cloud providers |
| `ADMIN_PASSWORD` | Password for the admin panel |
| `PYTHON_API_URL` | RAG API URL (default: http://127.0.0.1:8000) |
| `PIPELINE_API_URL` | Pipeline API URL (default: http://127.0.0.1:8001) |
| `HISTORY_TURNS` | Chat context window in Q/A pairs (default: 5) |
| `CORS_ORIGINS` | Allowed widget origins, comma-separated (default: `*`) |
