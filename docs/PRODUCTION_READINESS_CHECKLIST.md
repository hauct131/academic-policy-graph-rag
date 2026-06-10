# Production Readiness Checklist

This document details the checklist and evaluation required before deploying the **Academic Policy Graph RAG** system to a production environment serving students.

## Current Status
- **Phase**: Deterministic MVP / V0 QA pipeline.
- **Approach**: Purely deterministic (no LLM, no external APIs, no vector/semantic embedding models yet).
- **Correctness**: High grounding because it only utilizes deterministic templates, explicit keyword mappings, and strict source selection.

---

## 1. What is Safe in the Current MVP
- **Evidence-Grounded Answers**: The QA system strictly utilizes retrieved chunks and does not invent facts or hallucinate.
- **Safety Disclaimers**: Fallback language is automatically appended to prompt students to verify details with Phòng QLĐT.
- **No LLM / External Calls**: No risk of API cost spikes, token limit exhaustion, latency issues, or prompt injections.
- **Temporal Warnings**: When queries ask for semester schedules/deadlines (e.g., *"Học kỳ này..."*), the system detects the lack of active notices and prints a clear temporal warning instead of inventing or assuming deadlines.

---

## 2. What is NOT Ready for Student-Wide Production
- **Limited Knowledge Base**: Currently only contains 3 ingested core documents.
- **No Semantic/Vector Search**: Standard keyword retrieval cannot handle complex synonyms, typing errors, or deep semantic query intents.
- **No Monitoring Dashboard**: Lacks analytics for tracking queries, failure rates, latency, or student feedback.
- **Manual Notice Updates**: Active semester notices are not automatically fetched; updating active notices requires manual ingestion.
- **Lack of Official Sign-Off**: The current templates and outputs must be reviewed and approved by academic advisors or administrators.

---

## 3. Must-Have Before Student-Wide Production
1. **Ingest More Official Regulations**: Import all student handbooks, credit system regulations, and major-specific guidelines.
2. **Setup Real Active Notices**: Populate actual semester notices in `document_registry.jsonl` rather than only relying on fallback warnings.
3. **Expand Evaluation Dataset**: Scale evaluation cases (`data/eval/ou_policy_cases.jsonl`) to at least 50-100 scenarios.
4. **Human-in-the-Loop Review**: Run validation workshops with Phòng Đào tạo (QLĐT) to review answer templates.
5. **Secure Logging & Privacy**: Log metrics (timestamp, length, status, top_k) but keep full question logging opt-in and redacted.
6. **Deployment Configurations**: Containerize with Docker, configure persistent volumes, and setup HTTPS.
7. **Backup & Versioning**: Version control `document_registry.jsonl` and auto-archive old semester notices to keep the registry clean.
8. **Contact & Route Info**: Provide clear contact information for Phòng QLĐT, student advisors, and support desks.

---

## 4. Privacy & Safety Guidelines
### Request Logging Policy
- **Default**: Question text logging is disabled.
- **Opt-in Only**: Enabled exclusively via `POLICY_QA_ENABLE_REQUEST_LOGGING=1`.
- **Logs Redaction**: Minimize and redact sensitive personal identifiers (student IDs, names, contact numbers) from logs if question logging is ever enabled.

### Deadline & Schedule Safety
- **Rule**: Never invent deadlines, exams, or submission dates.
- **Fallback**: If the query asks for temporal info and there is no active semester notice matching the policy area, return:
  > *"Thông tin hiện chưa đủ để kết luận chắc chắn. Sinh viên cần đối chiếu trực tiếp với thông báo chính thức của học kỳ hiện tại hoặc liên hệ Phòng Quản lý Đào tạo."*
