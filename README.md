# Frank Chipana Portfolio (React + RAG Chatbot)

This repo contains:

- A React + TypeScript portfolio website (deployed on GitHub Pages).
- A Python RAG backend (deployed on Google Cloud Run) that powers the floating chatbot.

## Tech Stack

- Frontend: Vite, React, TypeScript, Tailwind CSS, TanStack Router
- Backend: Python, LangChain, HuggingFace Embeddings, Azure Cosmos DB (vector store), Gemini API
- Deploy:
  - Frontend: GitHub Pages (`gh-pages`)
  - Backend: Google Cloud Run

## Project Structure

- `src/`: React app + Python backend (`src/main.py`)
- `public/`: static files (includes `cv-frank-chipana.pdf` and `404.html` for GitHub Pages SPA fallback)
- `render.yaml`: legacy Render config (not used if you deploy on GCP)

## Frontend (Local Dev)

Install and run:

```bash
npm install
npm run dev
```

Local dev API URL is configured in `/.env.local`:

```env
VITE_CHAT_API_URL=http://127.0.0.1:8080
```

## Backend (Local)

### Option A: Run with venv (CLI or API)

API mode:

```bash
venv/bin/python src/main.py --mode api --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Option B (Recommended): Run Backend with Docker

Build:

```bash
docker build -t portfolio-rag .
```

Create `./.env.docker` (do not commit secrets):

```env
AZURE_COSMOS_DB_ENDPOINT=https://...
AZURE_COSMOS_DB_KEY=...
COSMOS_DB_NAME=...
COSMOS_CONTAINER_NAME=...
GEMINI_API_KEY=...
GEMINI_MODEL=...
CORS_ORIGINS=http://localhost:5173
```

Run:

```bash
docker run --rm -p 8080:8080 --env-file ./.env.docker portfolio-rag
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

## Backend API Endpoints

- `GET /health` (also works on `/`)
- `POST /connect` (use existing Cosmos index)
- `POST /reindex` (rebuild index from PDFs in `DOCS_DIR`)
- `POST /chat` body: `{ "message": "..." }`

## Environment Variables

### Frontend

- `VITE_CHAT_API_URL` (backend base URL)

Production example (GitHub Pages):
- `/.env.production` should point to your Cloud Run URL.

### Backend

- `AZURE_COSMOS_DB_ENDPOINT`
- `AZURE_COSMOS_DB_KEY`
- `COSMOS_DB_NAME`
- `COSMOS_CONTAINER_NAME`
- `EMBEDDING_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_DIM` (default: `384`)
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `DOCS_DIR` (default: `./src/documentos_subidos`)
- `CORS_ORIGINS` (example: `https://frankchip2023.github.io`)

## Deploy Frontend: GitHub Pages

This repo is configured for GitHub Pages with Vite basepath `/portfolio/`.

Deploy:

```bash
npm run build
npm run deploy
```

GitHub → Settings → Pages:
- Branch: `gh-pages`
- Folder: `/ (root)`

Notes:
- After deploy, GitHub Pages may cache old assets. Use hard refresh: `Cmd + Shift + R`.
- `public/404.html` is included to support SPA routing on GitHub Pages.

## Deploy Backend: Google Cloud Run (Step-by-Step)

Assumptions used below:

- `PROJECT_ID`: `portfolio-486616`
- Region: `europe-west4`
- Artifact Registry repo name: `portfolio` (Docker)
- Image name: `porfolio`

### 1) Enable APIs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 2) Configure Docker auth

```bash
gcloud auth configure-docker europe-west4-docker.pkg.dev
```

### 3) Build + push image (Cloud Build)

```bash
gcloud builds submit --tag europe-west4-docker.pkg.dev/portfolio-486616/portfolio/porfolio:latest .
```

### 4) Deploy to Cloud Run

```bash
gcloud run deploy porfolio \
  --image europe-west4-docker.pkg.dev/portfolio-486616/portfolio/porfolio:latest \
  --platform managed \
  --region europe-west4 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 3
```

### 5) Set backend env vars in Cloud Run

Cloud Run → Service → Edit & Deploy new revision → Environment variables:

- Set all Azure + Gemini variables listed above.
- Set `CORS_ORIGINS=https://frankchip2023.github.io`

### 6) Connect frontend to Cloud Run

Update `/.env.production`:

```env
VITE_CHAT_API_URL=https://YOUR_CLOUD_RUN_URL
```

Redeploy frontend:

```bash
npm run build
npm run deploy
```

## Security Notes

- Never commit API keys. Use `/.env.local`, `/.env.docker`, and Cloud Run environment variables.
- If a key was ever committed or shared, rotate it immediately.

## Troubleshooting

- GitHub Pages shows a blank page:
  - Hard refresh: `Cmd + Shift + R`
  - Confirm Pages points to `gh-pages`
- Chat shows `Failed to fetch`:
  - Check `VITE_CHAT_API_URL` is correct
  - Check `CORS_ORIGINS` includes `https://frankchip2023.github.io`
- Gemini model error `NOT_FOUND`:
  - Verify `GEMINI_MODEL` value (typos cause 404)

