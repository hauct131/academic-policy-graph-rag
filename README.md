# Academic Policy Graph RAG

An intelligent, Graph RAG-based Academic Policy Assistant designed to provide student guidance and policy interpretation.

## Goal
Build a Graph RAG system that acts as an assistant to guide students through various university rules and procedures. Unlike standard vector-based RAG, a Graph RAG approach models the complex, interconnected relationships between policies, prerequisites, semesters, department structures, and credit requirements.

## Supported Academic Domains
The assistant supports guidance across the following topics:
- **Course Registration**: Prerequisites, corequisites, load limits, and registration workflows.
- **Study-Ahead / Over-Credit Registration**: Requirements and processes for registering extra credits or higher-level courses.
- **Retake & Grade Improvement**: Policies on repeating courses, grade replacement, and GPA impact.
- **Academic Warning**: Criteria, repercussions, and probation recovery paths.
- **Graduation Requirements**: Credit allocation, major-specific requirements, and GPA thresholds.
- **Academic Calendar & Deadlines**: Add/drop periods, withdrawal dates, and registration windows.

## Project Structure
```text
academic-policy-graph-rag/
├── app/                      # FastAPI application
│   └── main.py               # API entrypoint and health routes
├── core/                     # Domain-independent Graph RAG pipeline logic
│   └── __init__.py
├── domains/                  # Academic policy domain-specific configurations
│   └── ou_academic_policy_v1/   # V1 policy definitions, schemas, and prompts
│       └── __init__.py
├── scripts/                  # Data ingestion and pipeline utilities
│   └── __init__.py
├── data/                     # Raw, processed, and generated data files
│   └── .gitkeep
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_health.py        # Integration and unit tests
├── requirements-api.txt      # API & Development dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusion patterns
└── AGENTS.md                 # Project-specific guidelines for AI coding assistants
```

## Quickstart & Setup

1. **Set up virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-api.txt
   ```

3. **Build generated data (Ingestion Pipeline):**
   ```bash
   python scripts/01_build_policy_chunks.py
   python scripts/02_annotate_policy_chunks.py
   python scripts/03_build_policy_graph.py
   ```

## Running the Assistant

### 1. Run CLI QA Query
Query the pipeline directly via terminal command:
```bash
python scripts/06_answer_policy_question.py --question "Điều kiện xét tốt nghiệp là gì?"
```

### 2. Run API Server
Start the FastAPI server locally:
```bash
uvicorn app.main:app --reload
```
API endpoints:
- Health Check: `GET http://localhost:8000/api/v1/graph-rag/health`
- Ask Question: `POST http://localhost:8000/policy/ask`

Example `curl` query:
```bash
curl -X POST http://localhost:8000/policy/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Điều kiện xét tốt nghiệp là gì?","top_k":5}'
```

### 3. Run Smoke Demo
Run a quick deterministic query validation script (without starting the API server):
```bash
python scripts/09_smoke_policy_qa.py
```

### 4. Run Evaluations
Run the evaluation test suite against the target QA scenarios:
```bash
python scripts/08_eval_policy_cases.py --verbose
```

The eval dataset contains at least 30 cases covering: graduation (điều kiện xét tốt nghiệp, cấp bằng, hạng tốt nghiệp, GPA, tín chỉ), course exemption (điều kiện miễn môn, hồ sơ, 50% giới hạn, đề cương, thủ tục, thời hạn), foreign language (IELTS, TOEIC, Cambridge, thời hạn chứng chỉ, kỳ thi đầu ra, ngoại ngữ hai), course registration (đăng ký, điều chỉnh), retake and grade improvement, academic standing (cảnh báo, buộc thôi học), and temporal/deadline notice warnings.

### 5. Run Unit & Integration Tests
Run pytest to execute the full validation suite:
```bash
pytest
```

### Retrieval Service Layer

- Retrieval is now wrapped by `PolicyRetrievalService`.
- Current retrieval backend: `lexical_v0`.
- Current backend is deterministic lexical/metadata/graph retrieval.
- Future BM25/vector/hybrid retrieval should be implemented as new retrieval backends behind `PolicyRetrievalService`, not by bypassing source selection or strict evidence pruning.
- Future retrieval changes must not replace citation guardrails, source selector, strict evidence pruning, and temporal notice checks.

**Available retrieval backends:**

| Backend | Name | Status | Description |
|---|---|---|---|
| `LexicalPolicyRetrievalBackend` | `lexical_v0` | **Default / Production** | Deterministic lexical token overlap + metadata tag scoring + graph bonus |
| `BM25LikePolicyRetrievalBackend` | `bm25_like_v0` | Experimental | Standard-library BM25-like scoring with Vietnamese accent-insensitive tokenization |

**Example backend comparison (smoke script):**
```bash
python scripts/09_smoke_policy_qa.py --retrieval-backend lexical_v0
python scripts/09_smoke_policy_qa.py --retrieval-backend bm25_like_v0
```

> **Note:** Production default remains `lexical_v0` until evaluation proves another backend is better across all 9 eval cases.

### Compare Retrieval Backends

Run the deterministic retrieval comparison script to measure `first_chunk`, `chunk_any`, and `negative_chunk` retrieval quality across backends on the existing evaluation cases:

```bash
python scripts/10_compare_retrieval_backends.py
python scripts/10_compare_retrieval_backends.py --backends lexical_v0,bm25_like_v0 --verbose
```

- `bm25_like_v0` is experimental. Do not change the production default until expanded eval cases show consistent improvement over `lexical_v0`.
- This script evaluates **retrieval only** (which chunks were selected). For full end-to-end evaluation including answer text, use `python scripts/08_eval_policy_cases.py --verbose`.


## Documentation Links
For detailed guides on deployment, updates, and architecture:
- [Production Readiness Checklist](docs/PRODUCTION_READINESS_CHECKLIST.md)
- [Data Update Playbook](docs/DATA_UPDATE_PLAYBOOK.md)
- [AI Agent Guidelines](AGENTS.md)
