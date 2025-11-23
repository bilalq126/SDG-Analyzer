# SDG Analyzer

A Streamlit app that analyzes project descriptions for alignment with the UN Sustainable Development Goals (SDGs) and generates targeted project ideas using Google Generative AI (Gemini).

This repository contains:

- `main.py` — Streamlit app entrypoint (Analyzer + Reverse Idea Generator).
- `gemini.py` — LLM wrapper and helpers (calls Google Generative API with SDK and REST fallbacks).
- `utils.py` — helper utilities (charts, lookup helpers).

## Quick local run

1. Create a virtual environment (recommended) and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Provide your Google API key locally (one of these):
- Set environment variable in PowerShell:

```powershell
$env:GOOGLE_API_KEY = "YOUR_API_KEY"
```

- Or add the key to Streamlit secrets when deploying (recommended for cloud).

3. Run the Streamlit app:

```powershell
streamlit run "D:\Python\SDG Analyzer\main.py"
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (create a new repo if needed). Example commands:

```powershell
git init
git add .
git commit -m "Initial commit: SDG Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

2. Go to https://share.streamlit.io and click **New app** → **Deploy from GitHub**.
3. Select your repository and branch (`main`) and set the main file path to `main.py`.

4. Add your Google API key as a secret: In the app's settings on Streamlit Cloud, go to **Secrets** and add:

```
GOOGLE_API_KEY = "your_api_key_here"
```

The app reads `GOOGLE_API_KEY` from environment variables or `st.secrets`; using Streamlit secrets is recommended for production deployment.

## Notes & troubleshooting

- If you see a 404 or permission error when calling Gemini/Generative API, check:
  - That `GOOGLE_API_KEY` is valid, has access to the Generative Models API, and is not restricted in a way that blocks the endpoint.
  - If using service account credentials or OAuth in restricted environments, adjust permissions accordingly.

- Local import warnings about `ScriptRunContext` are normal when importing Streamlit modules outside `streamlit run`.

- If you want to reduce API usage while testing, stub `gemini.call_gemini()` to return canned JSON output.

## Files to edit for customization

- `gemini.py`: prompts, model name, and fallback behaviors.
- `main.py`: UI layouts, keys, and widget behavior.

## Next steps I can help with

- Push the repo to GitHub and connect Streamlit Cloud (I can generate the git commands or push for you if you provide access).
- Add a one-click deploy badge or GitHub Action.
- Add an on-demand "Suggest improvements" action per idea.

---

If you want, I can now push these files to a GitHub repo (I'll show the commands first), or add a `Procfile` or GitHub Actions workflow for automatic deploys. Which would you like?