# Door & Window AI Assistant

A multi-role AI assistant for a windows/doors/accessories business, built with Streamlit, LangChain/LangGraph, OpenRouter, Chroma, and a small FastAPI inventory service. The app itself is company-agnostic (branding is entirely env-driven — see `backend/company_config.py`); this particular deployment's RAG knowledge base is built from **Marinig Serramenti**'s real public catalogue and manuals (`data/website.txt`, `data/installation manuals/`, `data/maintainance manuals/`), so the assistant answers with real product data even though the surrounding app, database, and calendar are demo infrastructure (see "Mock Backends vs. Real Integration" below).

Three portals share one assistant core, each with its own tools, system prompt, and permissions — enforced architecturally (a role's LLM simply cannot call a tool that isn't in its list), not just by prompt instruction:

- **Client** — ask about products/specifications/maintenance, request a site survey, respond to a priced estimate, look up their own tickets.
- **Installer** — installation-manual access (separate knowledge base from the client-facing one), view/claim/release assigned jobs, report site issues, submit survey measurements and job completions, request inventory reservations, check travel distance to a job.
- **Manager** — everything above plus full inventory control, the ticket approval/pricing workflow, cost configuration, model selection, ticket editing, reservation approval, and a semantic Q&A cache review queue.

## Mock Backends vs. Real Integration

This project's persistence and scheduling are **mocks built for the demo**, not a claim about how a real deployment would be wired:

- **Database**: a local SQLite file (`db/company.db`) holds inventory, tickets, clients, users, and cost config. A real deployment would point this at whatever the client already runs (an ERP, a CRM, a proper RDBMS) instead.
- **Calendar**: Google Calendar via OAuth2 is used for availability checks and bookings, purely because it's a realistic, easy-to-demo stand-in for *some* scheduling system. A real deployment would integrate whatever scheduling/dispatch system the client already uses (which might not be Google Calendar at all).

What's meant to carry over to a real deployment is the **tool layer's shape**, not its current backend: every mutating action the agent can take — checking availability, booking an appointment, moving inventory, pricing an estimate — is already isolated behind a single function in `tools.py` (e.g. `book_installation_tool`, `inventory_api_tool`, `check_calendar_availability_tool`). Swapping a mock backend for a client's real system means changing the *inside* of that one function, not the agent's prompts, the LangGraph structure, or the HITL confirmation flow around it. The database schema (`backend/database.py`) and calendar calls (`tools.py`'s Google Calendar helpers) are the two places to replace first; everything upstream of them (routing, confirmation, RAG, UI) is designed to keep working unmodified against a different backend as long as the tool functions' return shape doesn't change.

## Example Usage

**Client** — asking about a product, then booking a site survey:
> "Tell me about the model IRIS" → the assistant answers from the catalogue, with sources.
> "I want to book a survey" → a compact booking form opens inline in the chat, pre-filled from anything already mentioned in the conversation; submitting it creates a ticket for manager approval.

**Installer** — closing out a survey visit:
> "I measured TKT-00007: 150x100x8cm, standard glass" → measurements are saved and the ticket moves to the manager's pricing queue. The installer's dashboard also shows a job-briefing panel for their active ticket: an estimated time budget, whatever the client said in the original request, and installation guidance for that product pulled from the installer-only manuals.

**Manager** — pricing a survey and handling a correction:
> "Approve and calculate the pricing for TKT-00007" → `approve_survey_estimate_tool` computes the price from the measurements, glass type, and travel distance automatically, then emails the client.

## Features

- **RAG knowledge base**, split into role-scoped Chroma collections rather than one shared store:
  - Product catalogue, scraped from the company website (`data/website.txt`) — available to everyone.
  - Maintenance/care manuals, parsed from PDFs — available to everyone.
  - Installation manuals, parsed from PDFs — **installer/manager only**, a hard tool-level restriction (`search_installation_manuals_tool` isn't even in the client's tool list).
  - Glossary (50 terms auto-extracted from the ingested content, each with an LLM-generated definition — see `ingestion/build_glossary_content.py`) and FAQ entries (JSONL).
- **Hybrid retrieval**: vector similarity + BM25 keyword search + multi-query expansion + FlashRank reranking, with product-name detection so a question about one model doesn't get diluted by chunks from other models. Two extra retrieval candidates are always added alongside the user's own phrasing: a bare-keyword variant (stripping conversational lead-ins like "tell me about") and a translation into the catalogue's actual content language — cross-lingual embedding/lexical matching is measurably weaker than same-language matching, so a question asked in a UI language other than the content's own language still retrieves reliably. Answers cite the source webpage's full URL inline (never a generic "click here"-style link) instead of showing raw retrieved chunks.
- **Semantic Q&A cache** — near-identical questions are answered from a reviewed cache instead of re-running the full RAG + LLM pipeline; new Q&A pairs go through manager approval before being cached.
- **Human-in-the-loop confirmation** — an explicit LangGraph `interrupt()` pauses the graph for any mutating action (bookings, inventory changes, cost updates, ticket edits) until the user confirms or cancels; read-only actions (search, availability checks, price estimates, ticket lookups) execute immediately. See "Visualizing the Graph" below to inspect the actual node graph.
- **Ticket lifecycle**: client requests a site survey → manager approves → installer measures on site → manager prices the estimate → client accepts/rejects → installer installs → manager approves the completion report → automatic client emails at every stage. A client can accept/reject a priced estimate either conversationally in chat, or from a small in-portal widget (email lookup, then a couple of clicks) that works even from a brand-new session — no need to remember the original conversation.
- **Ticket audit log** — every status change on every ticket (created, approved, claimed, measured, priced, accepted, blocked, closed, edited...) is recorded with who and when, visible in a "History" panel on the ticket card.
- **Manager ticket editing** — contact info, schedule, quantity, or notes can be corrected after the fact (e.g. after a phone call with the client) via chat or a dashboard form; ticket status/model/price still only change through their own dedicated approval tools.
- **Standardized client addresses** — a client-typed address is normalized on save (via Nominatim's structured breakdown) into "City, Street Number, Postcode (Province)", so the client directory doesn't accumulate as many spellings of one address as there are ways to type it.
- **Ticket map** — open/in-progress tickets plotted on a map (pydeck), colored by status, with a hover tooltip; coordinates are geocoded once at request time (or lazily backfilled for older tickets).
- **Ticket status summary + installer job briefing** — stat-tile ticket counts on both dashboards; each of the installer's active-job ticket cards additionally shows an estimated time budget (on-site labor + travel), the client's original notes, the surveyor's own site note (LLM-rephrased for clarity, content unchanged), and installation guidance for that product — falling back to general documented procedures, clearly labeled as such, when no model-specific guide exists.
- **Installer alerts** — structured, AI-formatted problem reports from the field to the manager, optionally linked to a ticket (which then shows as BLOCKED).
- **Inventory reservation requests** — installers request stock reservations; the manager reviews and approves them (which deducts stock via the inventory API) or rejects them.
- **Price estimates** — combines a driving-time/distance calculation (OpenStreetMap Nominatim + OSRM) with configurable labour/travel/overhead/VAT rates; frame cost scales with perimeter, glass cost with area.
- **Google Calendar integration** — OAuth2-based availability check and installation/survey booking; past/invalid dates and times are rejected with a clear warning instead of silently failing.
- **Voice mode** — speech-to-text (OpenAI Whisper) and text-to-speech (OpenAI TTS) for chat input/output.
- **Multi-language UI** — English, Italiano, Deutsch, Español, Slovenščina (chat replies always match the user's own message language, independent of the UI language setting).
- **Two analytics views for the manager**: **LLM Usage** (per-role token/cost tracking) and **Analytics** (business metrics — closed tickets, total revenue, hours worked; the chart is always a full, zero-filled 7-day or 30-day rolling window plotting daily revenue, never a sparse list of only the days that happened to close a ticket).
- **In-app PDF catalogue viewer**.
- **Layout**: a fixed header and chat input, two independently scrollable panes (chat vs. dashboard), and a draggable handle between them to resize the split by hand.

## Project Structure

```text
app/
  app.py                  Streamlit entry point / page router
  api_server.py           FastAPI inventory API
  ui_shared.py            Shared UI helpers: sidebar, chat rendering, translations, CSS, ticket widgets
  pages/
    login.py               Role selection / installer & manager login
    client.py               Client chat portal
    installer.py            Installer portal (jobs, alerts, completions, job briefing)
    manager.py               Manager dashboard (tickets, map, clients, alerts, cache, analytics)
backend/
  assistant.py            Agent orchestration, per-role tool sets (system prompts loaded from prompts/)
  prompts/                 Per-role system prompt templates (shared.txt, client.txt, installer.txt, manager.txt)
  graph.py                Explicit LangGraph StateGraph with HITL interrupts
  studio_graphs.py        Entry points for LangGraph Studio (see langgraph.json)
  auth.py                 bcrypt-based installer/manager authentication
  database.py             SQLite schema, inventory seeding, tickets/bookings/alerts/audit log/analytics
  company_config.py       Company-specific naming/branding, env-overridable
  llm_provider.py         OpenRouter model registry + cost calculation
  qa_cache.py             Semantic Q&A cache (Chroma-backed)
  email_utils.py          SMTP notifications
  voice.py                Whisper STT / OpenAI TTS
  secrets_provider.py     Centralized environment/secret loading
  log_context.py          Request-scoped logging context
rag/
  rag_engine.py           Hybrid retrieval, reranking, answer generation, role-scoped collection routing
ingestion/
  build_catalogue_rag.py       Scrapes data/website.txt into the catalogue_specs collection
  build_catalogue_sections.py  Derives sections/catalogue/catalogue_sections.json (used for inventory seeding + product detection)
  build_installation_rag.py    Parses PDFs (LlamaParse) into installation_guides + maintenance_guides collections
  build_glossary_content.py    Generates data/glossary/glossary.jsonl (50 terms + LLM definitions) from already-ingested content
  build_faq_and_glossary_rag.py  Loads FAQ/glossary JSONL into their collections
  webscraping_utils.py     Shared scraping/chunking helpers
evaluation/
  evaluate_rag.py          Standalone RAGAS evaluation script (manual, real API cost — see below)
data/                     Local source data (website.txt, PDFs, glossary — PDFs are gitignored)
sections/catalogue/       Generated catalogue section map (product name ↔ URL)
secrets/                  Google OAuth credentials.json / token.json (gitignored)
db/                       SQLite (company.db) + Chroma vector store (gitignored)
tests/                    Pytest test suite (conftest.py fixtures + unit/integration tests)
tools.py                  LangChain tools exposed to the assistant, per role
run.py                    Starts the inventory API and Streamlit UI together
langgraph.json            LangGraph Studio graph registry (dev-only, see below)
```

## Requirements

Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root with:

```env
# Required
OPENAI_API_KEY=...           # embeddings (text-embedding-3-small) + Whisper/TTS
OPENROUTER_API_KEY=...       # chat models
LLAMA_CLOUD_API_KEY=...      # PDF parsing (LlamaParse) for installation manuals

# Optional — email notifications
SMTP_HOST=...
SMTP_PORT=...
SMTP_USERNAME=...
SMTP_PASSWORD=...
MANAGER_EMAIL=...

# Optional — client email verification. Clients verify their email with a
# one-time code before seeing or answering their existing tickets (codes are
# sent via the SMTP settings above). With SMTP unset, DEMO_MODE=true shows the
# code on screen instead — for demos only, never in production.
DEMO_MODE=false

# Optional — inventory API / installer routing
API_URL=http://127.0.0.1:8000
# Shared secret between the assistant and the inventory API. The API refuses
# every request without it. run.py generates one per session when unset; set
# it explicitly when you start the two services separately.
INVENTORY_API_KEY=...
# Interface the inventory API binds to (loopback by default). Use 0.0.0.0
# only when the API runs in its own container.
API_HOST=127.0.0.1
INSTALLER_BASE_LABEL=...
INSTALLER_BASE_LAT=...
INSTALLER_BASE_LON=...
GEOCODING_COUNTRY_CODES=it

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_ENDPOINT=...
LANGCHAIN_PROJECT=...

# Optional — company branding overrides (see backend/company_config.py for defaults)
COMPANY_NAME=Doors & Windows Co.
CONTENT_LANGUAGE=Italian          # language the ingested catalogue/manuals are actually written in
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### Google Calendar (booking/availability)

1. In [Google Cloud Console](https://console.cloud.google.com): create/select a project, enable the **Google Calendar API**, configure the OAuth consent screen (add your account under Test users if it's in Testing status), then create an **OAuth client ID** (Desktop app type) and download the JSON.
2. Save it as `secrets/credentials.json`.
3. Trigger the one-time interactive login once:
   ```powershell
   .venv\Scripts\python.exe -c "from tools import get_calendar_service; get_calendar_service()"
   ```
   This opens a browser for consent and writes `secrets/token.json`. After that, calendar availability checks and bookings work without further prompts.

## Build Local Data

Run once (and again whenever source content changes):

```powershell
# 1. Derive the product/category section map from data/website.txt
.venv\Scripts\python.exe ingestion\build_catalogue_sections.py

# 2. Seed/update the SQLite database (users, inventory, tickets schema, costs)
.venv\Scripts\python.exe backend\database.py

# 3. Build the RAG collections (each clears and rebuilds its own Chroma collection)
.venv\Scripts\python.exe ingestion\build_catalogue_rag.py
.venv\Scripts\python.exe ingestion\build_installation_rag.py    # builds BOTH installation_guides and maintenance_guides

# 4. Optional — (re)generate the glossary from the content ingested in step 3, then load it
.venv\Scripts\python.exe ingestion\build_glossary_content.py
.venv\Scripts\python.exe ingestion\build_faq_and_glossary_rag.py
```

Notes:
- The catalogue ingestion scrapes the URLs listed in `data/website.txt` and chunks each page (~1600 characters/chunk); product pages get a `product_name` metadata tag used later for retrieval filtering.
- The installation ingestion parses whatever PDFs are in `data/installation manuals/` and `data/maintainance manuals/` and groups consecutive pages into chunks (~1600 characters/chunk), preserving page ranges — not one giant chunk per PDF. The two folders become two separate Chroma collections (`installation_guides`, `maintenance_guides`), which is what makes the installer-only access restriction possible at the tool level.
- `build_glossary_content.py` reads the chunks already sitting in the collections built in step 3 (no re-scraping) and asks the LLM to propose/define the 50 most relevant technical terms — real API cost, run it deliberately, not on every rebuild.
- Mock login accounts are created via `backend/auth.py`'s `register_user()` (clients need no account — they pick the Client role directly from the login page).

## Run The App

```powershell
.venv\Scripts\python.exe run.py
```

Or run the two services separately (requires `INVENTORY_API_KEY` in `.env`):

```powershell
.venv\Scripts\python.exe app\api_server.py
.venv\Scripts\python.exe -m streamlit run app\app.py
```

- Streamlit UI: `http://localhost:8501`
- Inventory API: `http://127.0.0.1:8000`

## Model Configuration

Models are registered in `backend/llm_provider.py` (OpenRouter):

- `openai/gpt-4o-mini` (default)
- `google/gemini-2.5-flash`
- `meta-llama/llama-3.3-70b-instruct`
- `openai/gpt-4o`

The manager can switch the active model from the dashboard sidebar; the choice is persisted in SQLite (`app_config` table) and shared across all sessions. Embeddings use OpenAI `text-embedding-3-small`.

The main conversational LLM runs at `temperature=0.2` + `top_p=0.9` (factual grounding comes from RAG/tool-calling, not sampling temperature, so a little variation keeps replies from sounding robotic across many sessions); RAG answer generation, the input-safety guard, and structured-extraction calls stay at `temperature=0` where precision/determinism matters; multi-query expansion uses its own `temperature=0.5` LLM instance, since diversity of phrasing is the entire point there.

## Docker Deployment

The app runs as a single container: `run.py` starts both the FastAPI inventory
API and the Streamlit UI in one process, exactly as it does locally. Stateful
data (SQLite, Chroma, logs, Google OAuth secrets, source data, the generated
catalogue section map) is bind-mounted from the host so it survives image
rebuilds and stays out of the built image.

**Prerequisites (do these on the host first, not in the container):**

1. `.env` file present in the project root (see Environment Setup above).
2. Google Calendar OAuth already completed locally — `secrets/credentials.json`
   and `secrets/token.json` must already exist. The interactive browser login
   can't run inside a container, so this step has to happen on the host first
   (see the Google Calendar section above); the container only ever reuses
   the resulting `token.json`, refreshing it in place when it expires.
3. `data/website.txt` and any source PDFs in `data/` present.

**Build and start:**

```powershell
docker compose build
docker compose up -d
```

**First run only — build the local data inside the container's mounted volumes:**

```powershell
docker compose run --rm app python ingestion/build_catalogue_sections.py
docker compose run --rm app python backend/database.py
docker compose run --rm app python ingestion/build_catalogue_rag.py
docker compose run --rm app python ingestion/build_installation_rag.py
docker compose run --rm app python ingestion/build_glossary_content.py
docker compose run --rm app python ingestion/build_faq_and_glossary_rag.py
```

These write into `./db` and `./sections` on the host (via the bind mounts), so
they only need to be re-run when source content changes — not on every
container restart.

- Streamlit UI: `http://localhost:8501`
- Inventory API: not published to the host — only the assistant inside the
  container calls it.

**Notes:**
- `API_URL` (used by `tools.py` to reach the inventory API) defaults to
  `http://127.0.0.1:8000`, which is correct here since both services share
  one container's network namespace — no override needed for this
  single-container setup.
- Logs land in `./logs` on the host; check `docker compose logs -f app` for
  live container output.
- To run the two services as independent containers instead (e.g. to scale
  or restart them separately), split `docker-compose.yml` into two services
  from the same image, overriding `command:` to run
  `uvicorn app.api_server:app --host 0.0.0.0 --port 8000` and
  `streamlit run app/app.py` respectively, point `API_URL` at the other
  service's container name (e.g. `http://api:8000`), and give both the same
  `INVENTORY_API_KEY`. Keep port 8000 on the internal network only.

## Testing

```powershell
pip install pytest pytest-mock
pytest tests/
```

The suite (`tests/conftest.py` + `tests/test_*.py`) covers `backend/graph.py`
(routing helpers plus full integration tests of all four HITL paths, using a
scripted fake chat model — no real LLM calls), `backend/database.py` (the
entire ticket lifecycle and its audit log, business analytics, reservations),
`tools.py` (pure helpers plus DB-backed tools with external services
mocked — geocoding, calendar, email, LLM, inventory API), `backend/qa_cache.py`
(with a deterministic fake embeddings model), and `rag/rag_engine.py`'s pure
functions. Every test runs against an isolated throwaway SQLite database
(`tests/conftest.py`'s `test_db` fixture) — the real `db/company.db` is never
opened by the test suite.

## RAG Quality Evaluation (RAGAS)

```powershell
python evaluation/evaluate_rag.py
```

Runs a hand-written golden question set (grounded in the real ingested
catalogue/installation/maintenance content) through the actual production RAG
pipeline, then scores it with RAGAS' faithfulness, answer relevancy, context
precision, and context recall metrics (LLM-as-judge). Prints a summary and
saves per-question scores to `evaluation/last_run.json` for manual review.
This is a manual, deliberate command — it makes real LLM calls (both to
answer the questions and to judge them), so it's never run from tests or CI.

## Visualizing the Graph (LangGraph Studio)

LangSmith's default trace view (enabled via the `LANGCHAIN_TRACING_V2` env
vars above) shows the *executed* path of a single conversation turn as a
run tree — useful, but not the same as seeing the graph's actual shape. For
an interactive diagram of the compiled StateGraph (agent/tools/confirm/
execute_confirmed nodes, with the live path highlighted per thread):

```powershell
pip install "langgraph-cli[inmem]"
langgraph dev
```

This starts a local LangGraph API server using the `langgraph.json` graph
registry (`backend/studio_graphs.py` — one entry per role, each just building
a real `DoorAssistantAgent` and returning its compiled graph, no duplicated
logic) and prints a Studio URL pointing at `smith.langchain.com/studio` with
`baseUrl` set to the local server. `langgraph-cli` is a dev-only tool and is
deliberately not in `requirements.txt`, same as `pytest`.

## Notes

- This is a local/demo application (see "Mock Backends vs. Real Integration" above — SQLite + Google Calendar are demo stand-ins, not a hardened production deployment; no authN beyond bcrypt + Streamlit session state).
- Travel-time estimates use OpenStreetMap Nominatim (geocoding) and OSRM (routing); traffic conditions are not factored in.
- `secrets/`, `.env`, `db/`, and `logs/` are gitignored — see `.gitignore` for the full list.
