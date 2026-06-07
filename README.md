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
│   └── academic_policy_v1/   # V1 policy definitions, schemas, and prompts
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

## Setup & Installation

1. **Set up virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-api.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

4. **Run the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://localhost:8000`. You can check the health endpoint at:
   `http://localhost:8000/api/v1/graph-rag/health`

5. **Run tests:**
   ```bash
   pytest
   ```
