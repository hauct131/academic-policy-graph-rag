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

### 5. Run Unit & Integration Tests
Run pytest to execute the full validation suite:
```bash
pytest
```

### Retrieval Service Layer

- Retrieval is now wrapped by `PolicyRetrievalService`.
- Current backend is deterministic lexical/metadata/graph retrieval.
- Future vector/BM25/hybrid retrieval should be added behind this service layer.
- Future retrieval changes must not replace citation guardrails, source selector, strict evidence pruning, and temporal notice checks.

## Documentation Links
For detailed guides on deployment, updates, and architecture:
- [Production Readiness Checklist](docs/PRODUCTION_READINESS_CHECKLIST.md)
- [Data Update Playbook](docs/DATA_UPDATE_PLAYBOOK.md)
- [AI Agent Guidelines](AGENTS.md)
