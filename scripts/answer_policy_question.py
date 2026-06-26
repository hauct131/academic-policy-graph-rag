#!/usr/bin/env python3
"""
scripts/06_answer_policy_question.py

Template/rule-based QA answer utility for OU Academic Policy Graph RAG.
Infers policy issues from a user question, retrieves evidence, and outputs
a grounded, cited response in Vietnamese.

Usage:
    python scripts/06_answer_policy_question.py --question "Điều kiện xét tốt nghiệp là gì?"
"""

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.absolute()))

import retrieve_policy_chunks as _retriever
import select_policy_sources as _selector
import policy_retrieval_service as _service
import policy_domain_config as _domain
import policy_document_registry as _reg

normalize_text = _retriever.normalize_text
read_jsonl = _retriever.read_jsonl
retrieve_chunks = _retriever.retrieve_chunks
load_graph_expansion = _retriever.load_graph_expansion
select_sources_for_issue = _selector.select_sources_for_issue
prune_selected_sources_for_issue = _selector.prune_selected_sources_for_issue
PolicyRetrievalService = _service.PolicyRetrievalService
load_domain_config = _domain.load_domain_config
infer_issues_from_domain = _domain.infer_issues_from_domain
load_document_registry = _reg.load_document_registry
should_warn_missing_current_notice = _reg.should_warn_missing_current_notice


# ---------------------------------------------------------------------------
# Issue inference
# ---------------------------------------------------------------------------

def infer_case_issues(question: str, domain_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Infer academic policy issues from the user question.
    Returns a list of dicts representing issues with:
      - issue_type
      - policy_area
      - query (search query to run)
      - label (Vietnamese title)
    """
    if domain_config is not None:
        return infer_issues_from_domain(question, domain_config)

    norm_q = normalize_text(question)
    issues = []

    # 1. Graduation
    if any(kw in norm_q for kw in ["tot nghiep", "xet tot nghiep", "bang tot nghiep"]):
        issues.append({
            "issue_type": "graduation",
            "policy_area": "graduation",
            "query": "dieu kien xet tot nghiep",
            "label": "Quy định về tốt nghiệp"
        })

    # 2. Course exemption
    if any(kw in norm_q for kw in ["mien mon", "giam mon", "bang diem", "da hoc truong khac", "chuyen diem", "ho so mien"]):
        query = "ho so xin mien giam mon hoc" if "ho so" in norm_q else "dieu kien xet mien giam mon hoc"
        issues.append({
            "issue_type": "course_exemption",
            "policy_area": "course_exemption",
            "query": query,
            "label": "Quy định về miễn giảm môn học và chuyển điểm"
        })

    # 3. Foreign language requirement
    if any(kw in norm_q for kw in ["tieng anh", "ielts", "toeic", "toefl", "aptis", "cambridge"]):
        cert_kws = ["ielts", "toeic", "toefl", "aptis", "cambridge", "chung chi"]
        if any(ckw in norm_q for ckw in cert_kws):
            query = "xet mien tieng anh chung chi ielts toeic toefl aptis cambridge"
        else:
            query = "chuan dau ra tieng anh ngoai ngu khong chuyen"
        issues.append({
            "issue_type": "foreign_language_requirement",
            "policy_area": "foreign_language_requirement",
            "query": query,
            "label": "Quy định về chuẩn ngoại ngữ và miễn tiếng Anh"
        })

    # 4. Course registration
    if any(kw in norm_q for kw in ["dang ky mon", "hoc vuot", "khoi luong hoc tap"]):
        issues.append({
            "issue_type": "course_registration",
            "policy_area": "course_registration",
            "query": "dang ky mon hoc hoc vuot khoi luong hoc tap",
            "label": "Quy định về đăng ký môn học và khối lượng học tập"
        })

    # 5. Retake & grade improvement
    if any(kw in norm_q for kw in ["hoc lai", "cai thien diem"]):
        issues.append({
            "issue_type": "retake_and_grade_improvement",
            "policy_area": "retake_and_grade_improvement",
            "query": "dang ky hoc lai cai thien diem mon hoc",
            "label": "Quy định về học lại và cải thiện điểm"
        })

    # 6. Academic warning
    if any(kw in norm_q for kw in ["canh bao", "hoc luc kem", "buoc thoi hoc"]):
        issues.append({
            "issue_type": "academic_standing",
            "policy_area": "academic_standing",
            "query": "canh bao ket qua hoc tap buoc thoi hoc",
            "label": "Quy định về cảnh báo học tập và buộc thôi học"
        })

    # Fallback
    if not issues:
        issues.append({
            "issue_type": "generic",
            "policy_area": None,
            "query": question,
            "label": "Quy định liên quan"
        })

    return issues


# ---------------------------------------------------------------------------
# Answer templates
# ---------------------------------------------------------------------------

def get_vietnamese_answer(issue_type: str, query: str, domain_config: dict[str, Any] | None = None) -> str:
    """Generate static cautious response template based on issue type."""
    if domain_config is not None:
        defs = domain_config.get("issue_definitions", [])
        target_item = None
        for item in defs:
            if item.get("issue_type") == issue_type:
                target_item = item
                break
        if target_item:
            norm_q = normalize_text(query)
            overridden_template = None
            for o_cond in target_item.get("answer_template_override", []):
                cond_kws = [normalize_text(ckw) for ckw in o_cond.get("condition_keywords", [])]
                if any(ckw in norm_q for ckw in cond_kws):
                    overridden_template = o_cond.get("answer_template")
                    break
            if overridden_template:
                return overridden_template
            return target_item.get("answer_template", "")

    if issue_type == "graduation":
        return (
            "Dựa trên các đoạn tìm được, căn cứ chính là Điều 27 về điều kiện xét tốt nghiệp và công nhận tốt nghiệp. "
            "Sinh viên cần đối chiếu các điều kiện được nêu trong điều này, như việc hoàn thành chương trình/tín chỉ, "
            "điểm trung bình chung tích lũy và các yêu cầu liên quan khác nếu có trong chương trình. "
            "Vui lòng đối chiếu thêm chi tiết tại các tài liệu/căn cứ được liệt kê dưới đây."
        )
    elif issue_type == "course_exemption":
        if "ho so" in query.lower():
            return (
                "Dựa trên các đoạn tìm được, hồ sơ liên quan đến đơn xin miễn/giảm, bảng điểm hoặc đề cương môn học/chứng chỉ "
                "nếu được yêu cầu. Cần đối chiếu Điều 5 và thông báo học kỳ hiện hành nếu câu hỏi hỏi thời hạn."
            )
        return (
            "Dựa trên các đoạn tìm được, việc xét miễn môn học/học phần cần đối chiếu Điều 4 về điều kiện xét miễn, giảm môn học. "
            "Nếu câu hỏi liên quan đến hồ sơ, cần đối chiếu thêm Điều 5 về hồ sơ xin miễn, giảm môn học. "
            "Chưa nên kết luận chắc chắn nếu chưa đối chiếu đủ điểm, số tín chỉ, đề cương môn học và thời hạn áp dụng."
        )
    elif issue_type == "foreign_language_requirement":
        return (
            "Dựa trên các đoạn tìm được, câu hỏi về IELTS/TOEIC/TOEFL/Aptis/Cambridge cần đối chiếu Điều 9 và Phụ lục I. "
            "Chưa nên kết luận chắc chắn được miễn nếu chưa kiểm tra điểm, kỹ năng, thời hạn chứng chỉ và chương trình áp dụng."
        )
    elif issue_type == "course_registration":
        return (
            "Dựa trên các đoạn tìm được, việc đăng ký môn học và khối lượng học tập cần đối chiếu với các quy định chung của nhà trường. "
            "Sinh viên cần kiểm tra kỹ kế hoạch học tập cá nhân và các điều kiện ràng buộc đi kèm trước khi thực hiện đăng ký."
        )
    elif issue_type == "retake_and_grade_improvement":
        return (
            "Dựa trên các đoạn tìm được, việc học lại hoặc đăng ký cải thiện điểm số cần đối chiếu với các quy tắc tính điểm hiện hành. "
            "Chưa nên khẳng định kết quả cải thiện nếu chưa hoàn thành khóa học lại và có điểm số chính thức được cập nhật."
        )
    elif issue_type == "academic_standing":
        return (
            "Dựa trên các đoạn tìm được, việc xếp loại học lực kém, cảnh báo học tập hoặc buộc thôi học cần đối chiếu với các điều kiện "
            "và số lần vi phạm liên tiếp. Sinh viên cần chủ động theo dõi kết quả rèn luyện và học tập để tránh các rủi ro đáng tiếc."
        )
    else:
        return (
            "Dựa trên các đoạn tìm được, vui lòng đối chiếu kỹ điều kiện thực tế của bạn với các văn bản quy định chi tiết bên dưới. "
            "Chưa thể đưa ra kết luận chắc chắn cho trường hợp này."
        )


# ---------------------------------------------------------------------------
# Answer formulation
# ---------------------------------------------------------------------------

def asks_about_current_semester(question: str, domain_config: dict[str, Any] | None = None) -> bool:
    norm = normalize_text(question)
    if domain_config is not None and "current_semester_keywords" in domain_config:
        kws = [normalize_text(kw) for kw in domain_config["current_semester_keywords"]]
    else:
        kws = ["hoc ky nay", "hoc ky hien tai", "deadline", "thoi han", "han chot", "khi nao", "bao gio", "lich thi", "lich hoc"]
    return any(kw in norm for kw in kws)


def answer_question(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
    policy_area_filter: str | None = None,
    action_tag_filter: str | None = None,
    requirement_tag_filter: str | None = None,
    risk_tag_filter: str | None = None,
    graph_bonus_map: dict[str, float] | None = None,
    show_evidence_text: bool = False,
    domain_config: dict[str, Any] | None = None,
    document_registry: list[dict[str, Any]] | None = None,
    retrieval_service: PolicyRetrievalService | None = None,
) -> str:
    """
    Given a question and the chunks list, infer issues, retrieve evidence,
    and format the structured Vietnamese QA answer.
    """
    issues = infer_case_issues(question, domain_config=domain_config)

    if retrieval_service is None:
        retrieval_service = PolicyRetrievalService(chunks=chunks)

    # Gather evidence for each issue
    evidence_by_issue = {}
    total_matches = 0

    for issue in issues:
        selected = retrieval_service.retrieve_for_issue(
            issue=issue,
            question=question,
            top_k=top_k,
            max_sources=min(3, top_k),
            use_graph=True,
            strict_pruning=True,
            policy_area_filter=policy_area_filter,
            action_tag_filter=action_tag_filter,
            requirement_tag_filter=requirement_tag_filter,
            risk_tag_filter=risk_tag_filter,
            graph_bonus_map=graph_bonus_map,
        )
        evidence_by_issue[issue["issue_type"]] = selected
        total_matches += len(selected)

    # 1. No-answer path
    if total_matches == 0:
        return "Chưa tìm thấy quy định phù hợp trong dữ liệu hiện có."

    # 2. Case-level intro
    if len(issues) > 1:
        labels = [issue["label"] for issue in issues]
        intro = f"Mình tách trường hợp này thành {len(issues)} vấn đề: {', '.join(labels)}."
    else:
        intro = "Sau đây là nhận định sơ bộ dựa trên các quy định hiện hành:"

    # 3. Formulate markdown sections
    lines = [
        "# Câu trả lời",
        "",
        intro,
        ""
    ]

    # Assign global citation index starting from 1
    global_citations = []
    citation_index = 1

    for idx, issue in enumerate(issues, 1):
        results = evidence_by_issue[issue["issue_type"]]
        if not results:
            continue

        lines.append(f"## {idx}. {issue['label']}")
        lines.append("")
        
        answer_text = get_vietnamese_answer(issue["issue_type"], issue["query"], domain_config=domain_config)
        lines.append(answer_text)
        lines.append("")

        lines.append("Căn cứ chính:")
        for chunk, score in results:
            title = chunk.get("section_title") or chunk.get("chunk_id")
            doc = chunk.get("doc_id")
            
            # Map this chunk to a global citation number
            citation_num = None
            for g_num, g_chunk, g_score in global_citations:
                if g_chunk["chunk_id"] == chunk["chunk_id"]:
                    citation_num = g_num
                    break
            
            if citation_num is None:
                citation_num = citation_index
                global_citations.append((citation_num, chunk, score))
                citation_index += 1

            lines.append(f"* [{citation_num}] {title} — {doc} — score {score:.2f}")
        lines.append("")

    # 4. Formulate details
    lines.append("# Căn cứ chi tiết")
    lines.append("")
    for c_num, chunk, score in global_citations:
        title = chunk.get("section_title") or chunk.get("chunk_id")
        if show_evidence_text:
            preview = chunk.get("text", "")
        else:
            preview = chunk.get("text", "")[:300].replace("\n", " ").strip()
            if len(chunk.get("text", "")) > 300:
                preview += "..."

        lines.append(f"[{c_num}] {title}")
        lines.append(f"* chunk_id: {chunk.get('chunk_id')}")
        lines.append(f"* doc_id: {chunk.get('doc_id')}")
        lines.append(f"* source_pdf: {chunk.get('source_pdf') or 'N/A'}")
        lines.append(f"* score: {score:.2f}")
        lines.append(f"* preview: {preview}")
        lines.append("")

    # 5. Formulate suggestions
    lines.append("# Gợi ý kiểm tra thêm")
    lines.append("")
    warn_missing = False
    if document_registry is not None:
        p_area = None
        if issues:
            p_area = issues[0].get("policy_area")
        warn_missing = should_warn_missing_current_notice(
            question=question,
            records=document_registry,
            policy_area=p_area,
            chunks=chunks
        )
    else:
        warn_missing = asks_about_current_semester(question, domain_config=domain_config)

    if warn_missing:
        notice = "Dữ liệu hiện có chủ yếu là quy định chung, chưa có thông báo học kỳ hiện tại nên chưa thể kết luận thời hạn cụ thể."
        if domain_config and "current_semester_missing_notice_message" in domain_config:
            notice = domain_config["current_semester_missing_notice_message"]
        lines.append(notice)
    else:
        disclaimer = "Sinh viên nên đối chiếu kỹ điều kiện của mình với các căn cứ nêu trên."
        if domain_config and "fallback_scope_disclaimer" in domain_config:
            disclaimer = domain_config["fallback_scope_disclaimer"]
        lines.append(disclaimer)
    
    lines.append("Khuyến nghị liên hệ Phòng Quản lý đào tạo (Phòng QLĐT) hoặc Cố vấn học tập để nhận thông tin cập nhật mới nhất cho học kỳ hiện tại.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formulate grounded QA answers for Academic Policy queries."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question or student case text",
    )
    parser.add_argument(
        "--chunks-file",
        default="data/chunks/policy_chunks.annotated.jsonl",
        help="Path to annotated chunks JSONL",
    )
    parser.add_argument(
        "--nodes-file",
        default="data/graph/policy_graph_nodes.jsonl",
        help="Path to graph nodes JSONL",
    )
    parser.add_argument(
        "--edges-file",
        default="data/graph/policy_graph_edges.jsonl",
        help="Path to graph edges JSONL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve per issue (default: 5)",
    )
    parser.add_argument(
        "--policy-area",
        help="Override/filter by specific policy area",
    )
    parser.add_argument(
        "--action-tag",
        help="Override/filter by specific action tag",
    )
    parser.add_argument(
        "--requirement-tag",
        help="Override/filter by specific requirement tag",
    )
    parser.add_argument(
        "--risk-tag",
        help="Override/filter by specific risk tag",
    )
    parser.add_argument(
        "--show-evidence-text",
        action="store_true",
        help="Show full evidence text in details instead of preview",
    )
    parser.add_argument(
        "--domain-config",
        default="domains/ou_academic_policy_v1/domain.json",
        help="Path to domain configuration JSON",
    )
    parser.add_argument(
        "--document-registry",
        default="domains/ou_academic_policy_v1/document_registry.jsonl",
        help="Path to document registry JSONL",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    chunks_path = Path(args.chunks_file)
    nodes_path = Path(args.nodes_file)
    edges_path = Path(args.edges_file)

    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    # Load chunks
    chunks = read_jsonl(chunks_path)

    # Graph bonus map
    graph_bonus_map = load_graph_expansion(args.question, nodes_path, edges_path)

    # Load domain config if path exists
    domain_config = None
    config_path = Path(args.domain_config)
    if config_path.exists():
        domain_config = load_domain_config(config_path)

    # Load document registry if path exists
    document_registry = None
    registry_path = Path(args.document_registry)
    if registry_path.exists():
        document_registry = load_document_registry(registry_path)

    # Generate answer
    ans = answer_question(
        question=args.question,
        chunks=chunks,
        top_k=args.top_k,
        policy_area_filter=args.policy_area,
        action_tag_filter=args.action_tag,
        requirement_tag_filter=args.requirement_tag,
        risk_tag_filter=args.risk_tag,
        graph_bonus_map=graph_bonus_map,
        show_evidence_text=args.show_evidence_text,
        domain_config=domain_config,
        document_registry=document_registry,
    )

    print(ans)


if __name__ == "__main__":
    main()
