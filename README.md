# BSA AI Chatbot

AI-powered customer support chatbot for BSA motorcycle dealers.

## Architecture

```
3 servers running simultaneously:

Port 8000  →  RAG API (Python/FastAPI)     — api.py
Port 8001  →  Pipeline API (Python/FastAPI) — production/pipeline_api.py
Port 3000  →  Node.js Server (Express)      — server/server.js
```

## Setup

### 1. Python environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Environment variables
```bash
# Root .env (for Python)
cp .env.example .env
# Fill in: OPENAI_API_KEY, PINECONE_API_KEY, S3_BUCKET etc.

# Server .env (for Node.js)
cp server/.env.example server/.env
# Fill in: DATABASE_URL, ADMIN_PASSWORD etc.
```

### 3. Node.js setup
```bash
cd server
npm install
```

### 4. React app build
```bash
cd server/admin-react
npm install
npm run build
```

### 5. PostgreSQL
Create database `bsa_chat` and run the server — tables auto-create on startup.

## Running

Open 3 terminals:

**Terminal 1 — RAG API**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Pipeline API**
```bash
cd production
uvicorn pipeline_api:app --host 0.0.0.0 --port 8001
```

**Terminal 3 — Node.js**
```bash
cd server
node server.js
```

## URLs

| Page | URL |
|------|-----|
| Chat Widget | http://localhost:3000/test.html |
| Knowledge Base | http://localhost:3000/app/kb |
| Admin — Handoffs | http://localhost:3000/app/admin |
| Admin — Analytics | http://localhost:3000/app/analytics |
| Admin — Content | http://localhost:3000/app/content |
| Admin — Pipeline | http://localhost:3000/app/pipeline |

## Dealer Embed

```html
<script src="https://YOUR-DOMAIN/widget.js"
        data-dealer="dealer-id">
</script>
```
