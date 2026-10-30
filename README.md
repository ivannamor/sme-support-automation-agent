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

## Personal Data & Retention

> **Read this before deploying with real users.** The assistant stores
> conversations, and clients routinely type personal data into them (name,
> email, phone, address). Under the GDPR, personal data may be kept only as
> long as it is needed for its purpose (*storage limitation*, Art. 5(1)(e)).
> The retention periods below are **defaults for the demo, not legal
> advice** — agree the real values with whoever is responsible for data
> protection in your organisation before going live.

What the app stores about conversations, and for how long:

| Store | What it holds | Who reads it | Kept for | Setting |
|---|---|---|---|---|
| `db/checkpoints.db` | The agent's working memory: the full message history of each chat session, incl. tool calls and results | Only the LLM, during that conversation | **7 days** since the session was last used | `CHECKPOINT_RETENTION_DAYS` |
| `conversation_log` table in `db/company.db` | One row per turn: time, role, user (installer/manager username, the client's *verified* email, or `anonymous`), user message, assistant reply, tools used, model, confirm/cancel decision | People (an archive — never fed back to the LLM) | **365 days** | `CONVERSATION_RETENTION_DAYS` |

Expired data is deleted automatically when the app starts and then once a
day (on the first page load after 24 hours), see `app/app.py`.

Other places personal data ends up, **not** covered by the automatic deletion:

- **Tickets and clients** (`pending_bookings`, `clients` and related tables) —
  the business records themselves, kept until deleted by hand.
- **Log files** (`logs/`) — operational logs; rotated by size (10 MB × 5
  files), not by age. Error messages can quote user input.
- **Third-party services**: every message is sent to the LLM provider
  (OpenRouter, and OpenAI for embeddings/voice). With LangSmith tracing
  enabled (`LANGCHAIN_TRACING_V2=true`) full conversations are uploaded to
  LangSmith too — their retention is configured on their side.

**Erasure requests** (Art. 17, right to be forgotten) are not automated yet.
An archived client's turns can be found by their verified email:

```sql
DELETE FROM conversation_log WHERE user_identity = 'client@example.com';
```

Conversations by `anonymous` clients can't be matched to a person this way,
even though their text may still contain personal data — that is one more
reason to keep `CONVERSATION_RETENTION_DAYS` no longer than necessary.

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
  - Glossary (50 terms auto-extracted from the ingested content, each with an LLM-generated definition — see `ingestion/build_glossary_content.py`), plus optional static FAQs (see "Optional: static FAQs" below — none are shipped; frequent questions are covered by the Q&A cache instead).
- **Hybrid retrieval**: vector similarity + BM25 keyword search + multi-query expansion + FlashRank reranking, with product-name detection so a question about one model doesn't get diluted by chunks from other models. Two extra retrieval candidates are always added alongside the user's own phrasing: a bare-keyword variant (stripping conversational lead-ins like "tell me about") and a translation into the catalogue's actual content language — cross-lingual embedding/lexical matching is measurably weaker than same-language matching, so a question asked in a UI language other than the content's own language still retrieves reliably. Answers cite the source webpage's full URL inline (never a generic "click here"-style link) instead of showing raw retrieved chunks. All query variants are embedded in one request and searched by vector in every collection; the multi-query expansion and the translation run in parallel; the Chroma collections, the reranker model and the embeddings client are loaded once per process.
- **Semantic Q&A cache** — near-identical questions are answered from a reviewed cache instead of re-running the full RAG + LLM pipeline; new Q&A pairs go through manager approval before being cached.
- **Conversation memory** — chat state is checkpointed to SQLite (`backend/checkpointing.py`), shared by every agent, so changing the active model doesn't wipe ongoing conversations or pending confirmations. Only the most recent ~4000 tokens of history (whole turns, never splitting a tool call from its result) are sent to the LLM on each turn, so long chats don't get ever more expensive. Every turn is also archived in a human-readable `conversation_log` table — see "Personal Data & Retention" above.
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
- **Two analytics views for the manager**: **LLM Usage** (per-role token/cost tracking, see the next point) and **Analytics** (business metrics — closed tickets, total revenue, hours worked; the chart is always a full, zero-filled 7-day or 30-day rolling window plotting daily revenue, never a sparse list of only the days that happened to close a ticket).
- **Resilient LLM calls** — every LLM and embeddings request has a 30 s timeout and 2 retries (`backend/llm_provider.py`); while streaming, the timeout is the longest allowed silence between two chunks, not the length of the reply. If the input safety guard can't run (e.g. the provider is down), the message is still blocked, but the user is told the assistant is temporarily unavailable instead of seeing a security alert.
- **Token & cost accounting** — token counts are the ones the provider reports for each call (`usage_metadata`, requested in streamed responses too), not estimates. Every LLM call is counted: the agent's own, the input safety guard, calls made inside tools (query expansion, RAG, form extraction, emails) and calls made directly by dashboard buttons. Each call is priced at the model that actually served it (tools use the default model, whatever model the chat uses), at the exact cost OpenRouter reports for it when there is one — on non-streamed calls (guard, tools, RAG); langchain-openai drops it from streamed ones — otherwise from the price table in `backend/llm_provider.py`. When both exist and differ by more than 5%, a `PRICE_TABLE_STALE` warning says the table needs updating (the guard checks the default model's price on every message). A chat turn is logged to `usage_log` as one row per model used, a dashboard-button call as one row. Offline scripts (ingestion, evaluation) are not logged. If a provider ever reports no usage, an approximate count is logged instead, with a `USAGE_ESTIMATED` warning.
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
  graph.py                Explicit LangGraph StateGraph with HITL interrupts + history trimming
  checkpointing.py        Shared SQLite checkpointer (conversation state) + pruning of idle threads
  studio_graphs.py        Entry points for LangGraph Studio (see langgraph.json)
  auth.py                 bcrypt-based installer/manager authentication
  database.py             SQLite schema, inventory seeding, tickets/bookings/alerts/audit log/analytics
  company_config.py       Company-specific naming/branding, env-overridable
  llm_provider.py         OpenRouter model registry + cost calculation
  usage_tracking.py       Per-call token/cost accounting from provider-reported usage
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
  datasets/golden.yaml     Golden set: 42 hand-checked cases (tool routing + answers)
  run_evals.py / judge.py  Agent evals: run the cases, then score them (manual, real API cost — see below)
  compare.py               Side-by-side comparison of two judged runs
  calibrate.py             Picks the judge model by agreement with human labels
  results/                 Judged runs, committed so the numbers have a history
data/                     Local source data (website.txt, PDFs, glossary — PDFs are gitignored)
sections/catalogue/       Generated catalogue section map (product name ↔ URL)
secrets/                  Google OAuth credentials.json / token.json (gitignored)
db/                       SQLite (company.db, checkpoints.db) + Chroma vector store (gitignored)
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

# Optional — data retention, in days (see "Personal Data & Retention" above)
CHECKPOINT_RETENTION_DAYS=7       # agent working memory of idle chat sessions
CONVERSATION_RETENTION_DAYS=365   # conversation archive (conversation_log)

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

Run once (and again whenever source content changes), from the project root.
Scripts are run as modules (`python -m package.module`), which puts the
project root on the import path:

```powershell
# 1. Derive the product/category section map from data/website.txt
.venv\Scripts\python.exe -m ingestion.build_catalogue_sections

# 2. Seed/update the SQLite database (users, inventory, tickets schema, costs)
.venv\Scripts\python.exe -m backend.database

# 3. Build the RAG collections (each clears and rebuilds its own Chroma collection)
.venv\Scripts\python.exe -m ingestion.build_catalogue_rag
.venv\Scripts\python.exe -m ingestion.build_installation_rag    # builds BOTH installation_guides and maintenance_guides

# 4. Optional — (re)generate the glossary from the content ingested in step 3, then load it
.venv\Scripts\python.exe -m ingestion.build_glossary_content
.venv\Scripts\python.exe -m ingestion.build_faq_and_glossary_rag
```

Notes:
- The catalogue ingestion scrapes the URLs listed in `data/website.txt` and chunks each page (~1600 characters/chunk); product pages get a `product_name` metadata tag used later for retrieval filtering.
- The installation ingestion parses whatever PDFs are in `data/installation manuals/` and `data/maintainance manuals/` and groups consecutive pages into chunks (~1600 characters/chunk), preserving page ranges — not one giant chunk per PDF. The two folders become two separate Chroma collections (`installation_guides`, `maintenance_guides`), which is what makes the installer-only access restriction possible at the tool level.
- `build_glossary_content.py` reads the chunks already sitting in the collections built in step 3 (no re-scraping) and asks the LLM to propose/define the 50 most relevant technical terms — real API cost, run it deliberately, not on every rebuild.
- Every build script replaces its collection, so any of them can be re-run safely.

### Optional: static FAQs

None are shipped: frequent questions are meant to come from the semantic Q&A cache, which grows as the manager approves real answers. A company that already has a FAQ list can add it as its own retrieval collection:

1. Copy `data/faqs/faq.example.jsonl` to `data/faqs/faq.jsonl` and replace the examples — one JSON object per line, `{"question": "...", "answer": "..."}` (extra keys are kept too; write them in the content language).
2. Run `.venv\Scripts\python.exe -m ingestion.build_faq_and_glossary_rag`.

Edit the file and re-run to update them. Delete it and re-run to switch them off: the collection is emptied.
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
docker compose run --rm app python -m ingestion.build_catalogue_sections
docker compose run --rm app python -m backend.database
docker compose run --rm app python -m ingestion.build_catalogue_rag
docker compose run --rm app python -m ingestion.build_installation_rag
docker compose run --rm app python -m ingestion.build_glossary_content
docker compose run --rm app python -m ingestion.build_faq_and_glossary_rag
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

## Development

Dev tools (pytest, ruff, mypy) are pinned in `requirements-dev.txt` and
configured in `pyproject.toml`:

```powershell
pip install -r requirements-dev.txt

ruff check .          # lint
ruff format .         # format
mypy                  # type-check backend/
pytest                # tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs all four on every push to
`main` and every pull request. The mypy check on `backend/` is blocking; the
rest of the codebase is type-checked too, but only reported, not enforced.

## Testing

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

## Agent Evals

```powershell
python -m evaluation.run_evals --label baseline --repeats 3   # 1. run the agent, save what it did
python -m evaluation.judge evaluation/results/<date>_baseline.json   # 2. score it
python -m evaluation.compare <A>.json <B>.json               # baseline vs experiment
```

The golden set ([evaluation/datasets/golden.yaml](evaluation/datasets/golden.yaml))
has two kinds of case:

- **Routing** (one per role and main tool): only the agent's first decision is
  run, with the real system prompt and tools. Checked: the tool chosen (or a
  plain reply), its key arguments, and *forbidden* tools, e.g. booking a
  survey with contact details the user never gave.
- **Knowledge** (products, maintenance, installation, edge cases, in Italian,
  English and German): a whole chat turn. Checked: every expected fact (an
  LLM judge, one yes/no call per fact), that the reply is in the user's
  language, and that retrieval found an expected source document. RAGAS
  (faithfulness, answer relevancy, context precision/recall) scores the RAG
  layer against hand-written reference answers, only on runs that retrieved
  documents.

Facts are checked rather than whole answers compared, because a correct
answer can be phrased in endless ways: "MAGIS 40 has a Uw of 0.64" either is
in the reply or isn't. Each fact quotes its source document in the case's
`evidence` field.

**Every bug found in the app becomes a new golden case**, so it can't come
back unnoticed.

The agent isn't deterministic, so a comparison runs every case 3 times
(`--repeats 3`) and reports pass *rates*. Running the same configuration
twice (an A/A test) gives the noise floor: a difference smaller than that
isn't evidence of anything. Repeats of one run tend to agree with each other
even when two runs disagree, so 3 repeats are not 3 independent measurements.

The judge model is fixed (`JUDGE_MODEL` in `evaluation/judge.py`, temperature
0) and was picked by `evaluation/calibrate.py` against human labels on tricky
(answer, fact) pairs. On 24 pairs, `claude-sonnet-5.5` agreed on all of them;
`claude-haiku-5.5` and `gemini-3.8-flash` agreed on 23, each with one false
YES.

### Baseline

`gpt-4o-mini`, 42 cases, 3 repeats, the same configuration run twice:

| Metric | Run A | Run B |
|---|---|---|
| Overall pass rate | 0.762 | 0.714 |
| Routing pass rate | 0.978 | 0.956 |
| Knowledge pass rate | 0.642 | 0.580 |
| Fact recall | 0.674 | 0.600 |
| Retrieval hit rate | 0.476 | 0.460 |
| Faithfulness | 0.934 | 0.910 |
| Answer relevancy | 0.742 | 0.700 |
| Context precision | 0.264 | 0.226 |
| Context recall | 0.843 | 0.839 |

The noise is large: about 5 points overall and 6–7 on knowledge, from a single
A/A pair. The main weakness is routing to the documents: most maintenance and
installation questions go to `general_door_knowledge_tool`, which answers
without reading them, and those runs are absent from the RAGAS averages.

Runs happen in a sandbox (a fresh temporary database): the real
`db/company.db` is never touched, and tools that change data stop at the HITL
confirmation. Judged results are committed to `evaluation/results/`. These
are manual commands with a real API cost; CI only validates the golden set. A
3-repeat run costs about $0.15 for the agent and $3 for the judge, nearly all
of it RAGAS: pass `--no-ragas` to the judge unless a change touches retrieval.

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
