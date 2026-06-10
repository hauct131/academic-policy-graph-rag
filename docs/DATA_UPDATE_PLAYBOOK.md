# Data Update Playbook

This playbook provides step-by-step instructions for adding or updating academic policy documents, regulations, and semester notices in the Academic Policy Graph RAG system.

---

## 1. Adding a New Official Regulation

Follow these steps to ingest a new stable regulation document (e.g., training regulation, foreign language requirements):

### Step 1: Save the Source File
Save the official source PDF to the raw documents directory:
`data/raw/documents/<filename>.pdf`

### Step 2: Extract & Clean Content
1. If the PDF needs OCR, run the OCR extraction and save any intermediate outputs under:
   `data/raw/ocr/` (which is excluded from Git).
2. Save raw extracted text under:
   `data/raw/parsed/<filename>.txt`
3. Refine and format the text into clean Markdown under:
   `data/raw/cleaned/<filename>.md`
   * Ensure standard markdown headers (`#`, `##`, `###`) are used.
   * Verify that frontmatter containing metadata (Decision No, Issued Date, etc.) is preserved.

### Step 3: Register the Document
Open `domains/ou_academic_policy_v1/document_registry.jsonl` and append a new JSON record representing the document:
```json
{
  "doc_id": "ou_document_name_year",
  "title": "Quyết định ban hành Quy định...",
  "document_type": "regulation",
  "policy_area": ["course_exemption"],
  "decision_no": "XXXX/QĐ-ĐHM",
  "issued_date": "YYYY-MM-DD",
  "effective_from": "YYYY-MM-DD",
  "effective_to": null,
  "status": "active",
  "temporal_scope": "general",
"update_cadence": "stable",
  "source_pdf": "data/raw/documents/<filename>.pdf",
  "source_path": "data/raw/cleaned/<filename>.md"
}
```

### Step 4: Re-build & Verify Pipeline
Run the data building pipeline to chunk, annotate, and graph-expand the new content:
```bash
# 1. Build parsed chunks
python scripts/01_build_policy_chunks.py

# 2. Add metadata annotations
python scripts/02_annotate_policy_chunks.py

# 3. Re-build policy graph nodes & edges
python scripts/03_build_policy_graph.py
```

### Step 5: Test & Validate
1. Add new verification test cases into `data/eval/ou_policy_cases.jsonl` targeting the new regulation.
2. Run evaluation cases to verify everything works:
   ```bash
   python scripts/08_eval_policy_cases.py --verbose
   ```
3. Run the unit and integration test suite:
   ```bash
   python -m pytest
   ```

### Step 6: Commit Source & Config
Commit only the source markdown, registry config, and tests. Do NOT commit generated outputs:
```bash
git add data/raw/cleaned/<filename>.md domains/ou_academic_policy_v1/document_registry.jsonl data/eval/ou_policy_cases.jsonl tests/
git commit -m "Ingest new regulation: <document title>"
```

---

## 2. Adding Semester Notices

Semester notices (e.g., registration deadlines, exit exam notices) are temporary operational schedules.

### Rules for Notices
- **Real Evidence Only**: Never invent dates, deadlines, or schedules.
- **Accurate Scope**: Set `document_type` to `semester_notice` or `annual_notice` as appropriate.
- **Explicit Expiration**: Always supply `effective_from` and `effective_to` to ensure the notice expires automatically when the semester ends.

### Step-by-Step Notice Registration
1. Save the notice PDF and clean markdown under `data/raw/documents/` and `data/raw/cleaned/`.
2. Append the notice metadata to `domains/ou_academic_policy_v1/document_registry.jsonl`:
   ```json
   {
     "doc_id": "ou_exemption_notice_hk1_2025",
     "title": "Thông báo nộp hồ sơ miễn môn học kỳ 1 năm học 2025-2026",
     "document_type": "semester_notice",
     "policy_area": ["course_exemption"],
     "decision_no": "YYYY/TB-ĐHM",
     "issued_date": "2025-08-01",
     "effective_from": "2025-08-01",
     "effective_to": "2026-01-31",
     "status": "active",
     "temporal_scope": "semester",
     "update_cadence": "semester",
     "source_pdf": "data/raw/documents/ou_exemption_notice_hk1_2025.pdf",
     "source_path": "data/raw/cleaned/ou_exemption_notice_hk1_2025.md",
     "semester": 1,
     "academic_year": "2025-2026"
   }
   ```
3. Re-build the chunks and graph outputs by running:
   ```bash
   python scripts/01_build_policy_chunks.py
   python scripts/02_annotate_policy_chunks.py
   python scripts/03_build_policy_graph.py
   ```
4. Run evaluations and pytest as described above to ensure no regressions.
