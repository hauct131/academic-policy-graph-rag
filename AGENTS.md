# AI Agent Guidelines

Welcome! You are assisting in building the **Academic Policy Graph RAG** system.
Follow these guidelines strictly to keep the project clean, safe, and aligned with the target domain.

## 1. Project Goal

This project builds a **Graph RAG-based Academic Policy Assistant** for student academic guidance.

The system helps students retrieve and understand academic regulations related to:

* Course registration
* Study-ahead / over-credit registration
* Retake and grade improvement
* Academic warning
* Graduation requirements
* Academic calendar and deadlines

The system should not behave like a generic chatbot. It must ground its answers in retrieved academic policy contexts and cite the retrieved sources.

## 2. Domain Scope & Naming Conventions

### Domain

The target domain is:

* Higher education academic policy
* Student academic guidance
* Course progression
* Registration rules
* Graduation audit
* Academic calendar and deadline guidance

### Naming Constraints

Do **not** copy or reuse names from legal/labor RAG systems, including:

* `LegalPack`
* `legal-rag`
* `labor law`
* `article`
* `decree`
* `legal citation`
* `legal_reference_extractor`
* `case_rag`

Use academic-domain terms instead, such as:

* `policy`
* `regulation`
* `section`
* `clause`
* `requirement`
* `prerequisite`
* `academic_policy_v1`
* `degree_requirement`
* `academic_calendar`
* `student_status`
* `course_registration`
* `graduation_check`

## 3. Architecture & Directory Boundaries

Maintain clean separation of concerns.

### `core/`

Contains domain-independent Graph RAG pipeline logic, such as:

* Document parsing interfaces
* Chunking utilities
* Embedding interfaces
* Vector store interfaces
* Graph construction logic
* Graph traversal logic
* Retrieval and reranking logic
* Citation guardrail logic

`core/` should not hardcode academic policy rules, university-specific rules, or document-specific assumptions.

### `domains/academic_policy_v1/`

Contains academic domain-specific configuration, such as:

* Taxonomy
* Extraction config
* Prompt templates
* Entity definitions
* Metadata schema definitions
* Domain-specific retrieval hints

This folder should contain configuration and domain definitions, not generated data.

### `data/raw/`

Contains raw academic policy documents used for ingestion.

These may include:

* Simulated academic policy documents
* Public academic regulations
* Academic calendar documents
* Graduation requirement documents

Raw demo documents must clearly state if they are simulated and not official university policy.

### `data/`

Stores raw files and generated artifacts, such as:

* Raw documents
* Parsed chunks
* Annotated chunks
* Vector database files
* Graph JSONL files
* Evaluation outputs

`data/` must not contain application code.

### `app/`

Contains FastAPI code:

* Routers
* Request and response schemas
* API services
* Runtime application setup

### `scripts/`

Contains standalone scripts for:

* Data preparation
* Chunk generation
* Metadata annotation
* Vector DB building
* Graph building
* Evaluation
* Demo queries

Scripts should be self-contained and should call reusable logic from `core/` when appropriate.

### `tests/`

Contains unit tests and API tests.

When adding important modules or endpoints, add or update tests.

## 4. Technology Stack & Principles

Keep dependencies minimal.

Prefer:

* FastAPI for API
* Pydantic for validation
* Standard Python logging
* Qdrant for vector search
* JSONL for intermediate generated artifacts
* Simple, readable Python modules

Avoid:

* Heavy dependencies without clear need
* Complex frameworks too early
* Hardcoded academic rules inside retrieval code
* Domain-specific logic leaking into `core/`
* Large generated files committed to Git

## 5. API & Runtime Safety

Use safe FastAPI patterns.

Rules:

* Use FastAPI lifespan for heavy service initialization.
* Do not create lazy global singleton services inside route files.
* Do not expose internal exception details to API clients.
* Do not return `str(exc)` directly in HTTP responses.
* Log internal errors server-side.
* Return safe generic error messages to clients.
* Keep endpoint naming domain-neutral when possible.

Preferred endpoint style:

```text
GET  /api/v1/graph-rag/health
POST /api/v1/graph-rag/answer
```

Avoid endpoint names such as:

```text
/api/v1/legal-rag/...
/api/v1/labor-rag/...
```

## 6. Data & Git Policy

Raw demo documents may be stored under:

```text
data/raw/
```

Generated artifacts should not be committed unless explicitly requested.

Generated artifacts include:

* `data/chunks/*.jsonl`
* `data/vector_db/`
* `data/graph/*.jsonl`
* `data/eval/results/`
* Logs
* Temporary outputs

Keep `.env` out of Git.

Only commit:

```text
.env.example
```

Before committing, always check:

```bash
git status
git diff --stat
git diff
```

## 7. Retrieval Rules

The retrieval pipeline should follow this general design:

```text
User question
-> baseline vector search
-> metadata or intent detection
-> graph expansion
-> rerank and deduplicate
-> final contexts
-> answer generation
-> citation guardrail
```

Rules:

* Vector search retrieves semantically similar chunks.
* Metadata graph expands related conditions, procedures, deadlines, and risks.
* Graph expansion must not blindly add noisy contexts.
* Final scores should be normalized and must not exceed `1.0`.
* Deduplicate by stable source identity, preferably `chunk_id`.
* Retrieval debug information should be available for development and evaluation.

Useful debug fields:

```text
base_contexts
graph_contexts
final_contexts
retrieval_ms
total_ms
```

## 8. Citation Guardrail Rules

Final answers must be grounded in retrieved contexts.

Rules:

* Answers should cite retrieved context markers.
* Do not invent source names.
* Do not cite documents that were not retrieved.
* If citation validation fails, return a safe fallback answer.
* Fallback answers should still use the retrieved contexts as much as possible.
* If information is missing, clearly state what information is missing.

## 9. Answer Style

Default answer language: Vietnamese.

Academic guidance answers should be structured as:

1. Nhận định sơ bộ
2. Điều kiện cần đối chiếu
3. Thông tin còn thiếu
4. Rủi ro cần lưu ý
5. Hướng dẫn bước tiếp theo
6. Nguồn quy định

The assistant should not claim certainty when the retrieved policy context is incomplete.

Use wording such as:

* "Theo các nguồn được truy xuất..."
* "Cần đối chiếu thêm..."
* "Thông tin hiện chưa đủ để kết luận chắc chắn..."
* "Sinh viên nên kiểm tra thêm với phòng đào tạo/cố vấn học tập..."

## 10. Development Workflow

When implementing a task:

1. Inspect the current project first.
2. Propose a minimal plan.
3. Modify only necessary files.
4. Run relevant checks.
5. Show changed files.
6. Explain how to test.
7. Avoid implementing future phases unless explicitly asked.

For every meaningful change, run at least:

```bash
python -m compileall app core scripts
```

For API changes, also test:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/api/v1/graph-rag/health | python -m json.tool
```

## 11. Safety Rules for AI Agents

Do not:

* Delete files without asking.
* Rename public folders without asking.
* Change project scope without asking.
* Add legal/labor-specific naming.
* Add heavy dependencies without justification.
* Commit generated data unless explicitly requested.
* Rewrite the whole project when a small change is enough.
* Modify `.env`.
* Run destructive Git commands without asking.

Ask before running:

```bash
rm
mv
git reset
git clean
docker-compose down
docker-compose up
pip install
```

Allowed safe inspection commands include:

```bash
ls
pwd
cat
head
tail
grep
find
git status
git diff
git diff --stat
python -m compileall
```

## 12. Current MVP Scope

The MVP should cover only these five use cases:

1. Study-ahead / over-credit registration
2. Retake and grade improvement
3. Academic warning
4. Graduation check
5. Academic calendar and deadlines

Do not expand to these areas yet:

* Scholarship
* Dormitory
* Tuition exemption
* Student discipline
* Major transfer
* Temporary leave
* Internship management
* Thesis management

These can be future extensions after the MVP is stable.

## 13. Definition of Done

A task is considered done only when:

* The implementation matches the requested scope.
* The code compiles.
* Relevant endpoint or script can run.
* README or command notes are updated if needed.
* Generated data is not accidentally committed.
* The changed files are clearly explained.
