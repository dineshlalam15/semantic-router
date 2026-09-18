# Semantic LLM Router

A decoupled, configuration-driven **Semantic LLM Router** in Python that classifies user queries into semantic domains using local HuggingFace embeddings and selects the optimal LLM model across configured providers (**OpenAI**, **Gemini**, **Anthropic**, and **LiteLLM**).

> [!NOTE]
> **Zero External LLM / Provider Calls**: The router acts strictly as a decision-making layer. Provider and model names are metadata only; no external API calls or API keys are required. The router runs 100% locally and offline after downloading the embedding model.

---

## How It Works

```
User Query
    ↓
HuggingFace Encoder (sentence-transformers/all-MiniLM-L6-v2)
    ↓
Cosine Similarity against Pre-computed Route Utterances
    ↓
Domain Match (Highest similarity)
    ↓
Lookup Recommended Model for Domain (from config/models.yaml)
    ↓
Routing Decision: { query, domain, recommended_model, llm_provider }
```

1. **Decoupled Architecture**: All domains, utterances, models, providers, and capabilities live in `config/routes.yaml` and `config/models.yaml`. No routing logic is hardcoded in Python.
2. **Deterministic Matching**: Evaluates queries against route utterances using HuggingFace embeddings and returns the best matching domain.
3. **Sub-5ms Latency**: Route utterances are pre-computed at startup; query matching is a fast vectorized dot product.

---

## How to Run

### On macOS / Linux

```bash
# 1. Clone or navigate to the project directory
cd semantic-router

# 2. Create and activate a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests (optional)
pytest -v tests/

# 5. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### On Windows

#### Using PowerShell:
```powershell
# 1. Navigate to the project directory
cd semantic-router

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests (optional)
pytest -v tests/

# 5. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Using Command Prompt (`cmd.exe`):
```cmd
cd semantic-router
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
pytest -v tests/
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Testing the API

Once the server is running at `http://localhost:8000`:

### macOS / Linux (cURL)
```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I implement an LRU cache in Python?"}'
```

### Windows (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/route" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query": "How do I implement an LRU cache in Python?"}' | ConvertTo-Json
```

### Example Response
```json
{
  "query": "How do I implement an LRU cache in Python?",
  "domain": "software_engineering",
  "recommended_model": "claude-3-5-sonnet-20241022",
  "llm_provider": "Anthropic"
}
```

---

## Interactive Documentation

Open in your browser:
* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
