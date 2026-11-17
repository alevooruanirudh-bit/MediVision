# Free deployment

Everything below uses free tiers only — no credit card.

## Local, one command

```bash
docker compose up --build         # backend :8000 (weights baked in) + frontend :3000
```
Or without Docker: `make setup && make train-starter && make seed && make dev`.

## Backend → Hugging Face Spaces (Docker SDK, free CPU)

1. Train the starter models locally (`make train-starter`) so `artifacts/` contains the weights.
2. Create a Space at <https://huggingface.co/new-space>: SDK **Docker**, hardware **CPU basic (free)**.
3. Add this header to the Space's `README.md` (the Space repo, not this one):

   ```yaml
   ---
   title: MediVision AI
   sdk: docker
   app_port: 7860
   ---
   ```
4. Push the repo contents (Dockerfile at the root, `ml/`, `backend/`, `artifacts/`, `sample_data/`,
   `requirements.txt`). Model files are larger than 10 MB, so enable Git LFS on the Space for
   `*.onnx`/`*.pt`: `git lfs track "*.onnx" "*.pt"`.
5. Set Space variables: `CORS_ORIGINS=https://<your-app>.vercel.app`. The container listens on 7860,
   seeds the demo studies on start, and stores data in `/tmp` (ephemeral on the free tier).

The image installs CPU-only PyTorch; inference runs on ONNX Runtime.

## Frontend → Vercel (Hobby, free)

1. Import the repository on <https://vercel.com/new>, set **Root Directory** to `frontend`.
2. Environment variable: `NEXT_PUBLIC_API_URL=https://<user>-<space>.hf.space`.
3. Deploy. The backend's CORS already allows `*.vercel.app` and `*.hf.space` origins.

## Environment variables

See `.env.example`; none of them are API keys for AI services. The optional local LLM plug-in
(`COPILOT_LOCAL_LLM`) only accepts a localhost/compose-internal Ollama URL and is off by default.

## Notes

- Free CPU Spaces sleep when idle; the first request after waking reloads models (~10 s).
- Full-profile weights (from Kaggle) can be shipped the same way; the registry prefers them.
- `make docker` builds both images locally to verify before pushing.
