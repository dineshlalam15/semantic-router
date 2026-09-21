# Semantic LLM Router for Marketing

A decoupled, configuration-driven **Semantic LLM Router** in Python that classifies user queries into marketing-specific domains using local HuggingFace embeddings and selects the optimal LLM model across configured providers (**Anthropic**, **Gemini**, **OpenAI**, and **LiteLLM / Open-Weights**).

> [!NOTE]
> **Zero External LLM / Provider Calls**: The router acts strictly as a low-latency decision-making layer. Provider and model names are metadata; no external API calls or API keys are required. The router runs 100% locally and offline after downloading the embedding model.

---

## Architecture Diagram

```mermaid
flowchart TD
    %% Startup Phase
    subgraph Startup["Application Startup (Initialization Phase)"]
        direction TB
        CFG_R["Load Routes & Utterances Config"]
        CFG_M["Load Models & Providers Config"]
        ENC["Initialize Local Embedding Model"]
        EMB_INIT["Pre-compute Normalized Utterance Embeddings"]
        MATRIX[("In-Memory Utterance\nVector Matrix")]
        REGISTRY[("In-Memory Model Registry\n(Domain Mappings)")]

        CFG_R --> ENC
        ENC --> EMB_INIT
        EMB_INIT --> MATRIX
        CFG_M --> REGISTRY
    end

    %% Runtime Phase
    subgraph Runtime["Request Lifecycle (Runtime Flow)"]
        direction TB
        REQ["Incoming Client Request\nPOST /route"]
        VAL["Request Validation\n(Pydantic Schema)"]
        ENCODE_REQ["Encode Query to\nVector Embedding"]
        NORM["L2-Normalize Query Vector"]
        SIM["Vectorized Dot Product\n(Cosine Similarity Computation)"]
        BEST["Identify Best Matching Domain\n(Argmax Similarity)"]
        SELECT["Select Recommended Model & Provider\n(Domain Lookup)"]
        RESP["Return JSON Response\n(Domain, Model, Provider)"]

        REQ --> VAL
        VAL --> ENCODE_REQ
        ENCODE_REQ --> NORM
        NORM --> SIM
        SIM --> BEST
        BEST --> SELECT
        SELECT --> RESP
    end

    %% Interactions
    MATRIX -.->|Pre-computed Vectors| SIM
    REGISTRY -.->|Domain Configuration| SELECT
```

---

## Core Marketing Domains

All routing logic is 100% dedicated to marketing workloads defined in [`config/routes.yaml`](file:///Users/dineshlalam15/Desktop/semantic-router/config/routes.yaml):

| Domain | Focus & Capabilities | Primary Model Recommendation |
|---|---|---|
| **`commercial_visual_production`** | Cinematic campaign hero imagery, 3D product renders, turntable sequences, digital billboards, and showroom loops. | `dall-e-3` (OpenAI) / `imagen-3` (Gemini) / `flux-1-dev` (LiteLLM) |
| **`marketing_collateral_and_layout`** | Studio pack shots, packaging design, candid customer lifestyle imagery, technical infographics, trade show banners, and brochure layouts. | `dall-e-3` (OpenAI) / `firefly-vector` (Firefly) |
| **`campaign_and_advertising_copy`** | Launch press releases, configurator UI copy, TV/video commercial scripts, PPC search ad headlines, and promotional messaging. | `claude-3-5-haiku-20241022` (Anthropic) / `gpt-4o` (OpenAI) |
| **`email_marketing_and_retention`** | Customer onboarding drip sequences, win-back campaigns, churn prevention messaging, and dynamic personalization. | `claude-3-5-haiku-20241022` (Anthropic) / `gemini-2.0-flash` (Gemini) |
| **`social_media_and_brand_storytelling`** | Omnichannel social launch kits (LinkedIn, Instagram, X, TikTok), viral video hooks, organic social calendars, and brand narratives. | `claude-3-5-haiku-20241022` (Anthropic) / `gpt-4o` (OpenAI) |
| **`seo_and_content_strategy`** | In-depth SEO pillar articles, keyword intent clustering, Generative Engine Optimization (GEO), and topical authority planning. | `claude-3-5-sonnet-20241022` (Anthropic) / `gemini-1.5-pro` (Gemini) |
| **`technical_and_evidence_content`** | High-stakes B2B evidence whitepapers, conference research abstracts, enterprise sales presentation decks, and ROI business cases. | `gemini-1.5-pro` (Gemini) / `claude-opus-4-5` (Anthropic) |
| **`market_intelligence_and_research`** | Competitor teardowns, Ideal Customer Profile (ICP) buyer personas, Voice of Customer (VoC) sentiment mining, and positioning matrices. | `o1` (OpenAI) / `gemini-1.5-pro` (Gemini) / `deepseek-r1` (LiteLLM) |

---

## Supported Providers & Models

Model configurations and metadata live in [`config/text_models.yaml`](file:///Users/dineshlalam15/Desktop/semantic-router/config/text_models.yaml) and [`config/image_models.yaml`](file:///Users/dineshlalam15/Desktop/semantic-router/config/image_models.yaml):

* **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-opus-4-5`
* **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, `dall-e-3`
* **Gemini**: `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`, `imagen-3`
* **LiteLLM**: `deepseek-r1`, `llama-3.3-70b-instruct`, `qwen-2.5-coder-32b`, `flux-1-dev`, `stable-diffusion-3-5-large`
* **Adobe Firefly**: `firefly-image-3`, `firefly-vector`

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

# 4. Start the API server
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

# 4. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Using Command Prompt (`cmd.exe`):
```cmd
cd semantic-router
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Testing the API

Once the server is running at `http://localhost:8000`:

### 1. Copywriting Query (cURL)
```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a 600-word press release for the global commercial launch of an innovative product line."}'
```

**Response:**
```json
{
  "query": "Write a 600-word press release for the global commercial launch of an innovative product line.",
  "domain": "campaign_and_advertising_copy",
  "recommended_model": "claude-3-5-haiku-20241022",
  "llm_provider": "Anthropic"
}
```

### 2. Commercial Visual Production Query (cURL)
```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{"query": "Render a cinematic exterior hero shot of a flagship smart hardware device on a sleek minimalist desk."}'
```

**Response:**
```json
{
  "query": "Render a cinematic exterior hero shot of a flagship smart hardware device on a sleek minimalist desk.",
  "domain": "commercial_visual_production",
  "recommended_model": "dall-e-3",
  "llm_provider": "OpenAI"
}
```

### 3. Market Research Query (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/route" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query": "Analyze competitor pricing tiers and messaging claims across the top 4 players in project management software."}' | ConvertTo-Json
```

**Response:**
```json
{
  "query": "Analyze competitor pricing tiers and messaging claims across the top 4 players in project management software.",
  "domain": "market_intelligence_and_research",
  "recommended_model": "claude-3-5-sonnet-20241022",
  "llm_provider": "Anthropic"
}
```

---

## Interactive Documentation

Open in your browser:
* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)