# AI Agent Guidelines (AGENTS.md)

Welcome! You are assisting in building the **Academic Policy Graph RAG** system. To maintain high code quality, consistency, and alignment with project goals, please follow these guidelines strictly.

## 1. Domain Scope & Naming Conventions
- **Domain**: Higher education academic policy, student academic guidance, course progression, registration guidelines, and graduation audits.
- **Naming Constraints**: 
  - **Do NOT** copy or reuse names from legal/labor RAG systems (e.g., *LegalPack*, *legal-rag*, *labor law*, *article*, *decree*, *legal citation*, *legal_reference_extractor*, or *case_rag*).
  - **DO** use terminology native to the academic domain (e.g., *policy*, *regulation*, *prerequisite*, *requirement*, *section*, *clause*, *academic_policy_v1*, *degree_requirement*, *academic_calendar*).

## 2. Architecture & Directory Boundaries
Maintain clean separation of concerns:
- **`core/`**: Contains domain-independent Graph RAG pipeline logic (e.g., graph construction, entity extraction, vector storage interfaces, traversal/retrieval patterns). This folder must know nothing about "academic policies" or "courses".
- **`domains/academic_policy_v1/`**: Academic domain-specific schemas, extraction configurations, prompt templates, entity definitions (e.g., Course, Prerequisite, Department, Term), and policy raw texts.
- **`app/`**: FastAPI routers, request/response models, and API endpoints connecting the core pipeline with domain configs.
- **`scripts/`**: Ad-hoc ingestion scripts, loaders, or visualization tools. Keep these self-contained.
- **`data/`**: Storage for raw files, intermediate JSON representation, and DB dumps. Must not contain code.

## 3. Technology Stack & Principles
- Keep dependencies minimal. Avoid bloat.
- Utilize standard tools (e.g., Pydantic for validation, standard logging, standard FastAPI patterns).
- Test endpoints and schemas in `tests/`.

## 4. Development Workflow
- When adding new modules, write appropriate tests in `tests/`.
- Ensure all FastAPI endpoints include comprehensive validation and error handling.
- Verify status endpoints before finalizing tasks.
