import json
import time
from collections import deque
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
import re
from debug_logger import log, reset_logs, get_logs

from llm_utils import (
    verification_query_35,
    format_output_4,
    enrich_query_35,
    search_queries_35,
    web_output_35,
    refine_search_queries_35,
    refine_format_output_4,
    GOLD_FIELDS,
    RAW_FIELDS,
)
from selection_engine import run_selection_engine
from web_search import search_web, get_web_snippets
load_dotenv(dotenv_path="QDRANT.env")

# --- Qdrant setup ---
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ias_officers"

client = QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY
)

# --- Constants ---
THRESHOLD_SELECTION = 0.55
THRESHOLD_RAW = 0.60
THRESHOLD_REFINE = 0.5

# --- Rate limiting setup ---
LLM_RATE_LIMIT = 10
MIN_INTERVAL = 60.0 / LLM_RATE_LIMIT
llm_timestamps = deque()

def rate_limited_call(fn, *args, **kwargs):
    now = time.time()
    while llm_timestamps and now - llm_timestamps[0] > 60:
        llm_timestamps.popleft()
    if len(llm_timestamps) >= LLM_RATE_LIMIT:
        return None
    result = fn(*args, **kwargs)
    llm_timestamps.append(time.time())
    return result

# --- Load vectorstore only ---
@st.cache_resource
def load_vectorstore():
    from embedding import get_embedding_model
    return get_embedding_model()

EMBEDDING_FUNC = load_vectorstore()

# --- Markdown formatter ---
def format_markdown(data):
    if isinstance(data, dict):
        data = [data]

    def render_value(val):
        if not val or val in ['N/A', [], ['N/A'], None]:
            return "None"
        if isinstance(val, list):
            return "\n- " + "\n- ".join(v.strip() for v in val if v and v != 'N/A')
        return str(val).strip()

    def clean_officer_name(name):
        if not name:
            return "Unknown Officer"
        return re.sub(r"\bIdentity\s*No\.?\\?.*", "", name).strip()

    output = []
    for idx, officer in enumerate(data, 1):
        if not isinstance(officer, dict):
            continue

        source = officer.get("source", "gold")

        if source == "raw":
            name = clean_officer_name(officer.get("officer_name") or officer.get("name"))
            source_file = officer.get("source_file", "")
            block = [f"**{idx}) {name}**"]

            if source_file.startswith("https://"):
                block.append(f"**Reference:** [📄 View Resume]({source_file})")
            else:
                block.append(f"**Source File**: {render_value(source_file)}")

            output.append("\n".join(block))
            continue

        name = clean_officer_name(officer.get("officer_name"))
        block = [f"**{idx}) {name}**"]

        fields = [
            ("Allotment Year", officer.get("allotment_year")),
            ("Cadre", officer.get("cadre")),
            ("Recruitment Mode", officer.get("recruitment_mode")),
            ("Education", officer.get("education")),
            ("Postings", officer.get("postings")),
            ("Training Details", officer.get("training_details")),
        ]

        for label, raw_val in fields:
            val = render_value(raw_val)
            if "\n- " in val:
                block.append(f"**{label}**:{val}")
            else:
                block.append(f"**{label}**: {val}")

        source_file = officer.get("source_file", "")
        if source_file.startswith("https://"):
            block.append(f"**Reference:** [📄 View Resume]({source_file})")
        else:
            block.append(f"**Source File**: {render_value(source_file)}")

        reasoning = officer.get("selection_reasoning")
        if reasoning:
            block.append(reasoning.strip())

        output.append("\n".join(block))

    return "\n\n".join(output).replace("\n", "\n\n")

def safe_format_markdown(data):
    import json
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return format_markdown(parsed)
        except json.JSONDecodeError:
            return data.strip()
    return format_markdown(data)

def format_web_results(data):
    if isinstance(data, dict):
        data = [data]

    if all(isinstance(item, str) for item in data):
        return "\n\n".join(data)

    output = []
    for idx, item in enumerate(data, 1):
        officer = item.get("officer", "Unknown Officer")
        block = [f"**{idx}) {officer}**"]

        for key, val in item.items():
            if key == "officer":
                continue
            if not val or val in ['N/A', [], None]:
                val = "None"
            elif isinstance(val, list):
                val = "\n- " + "\n- ".join(str(v).strip() for v in val if v)
            else:
                val = str(val).strip()
            block.append(f"**{key.replace('_', ' ').title()}**: {val}")

        output.append("\n".join(block))

    return "\n\n".join(output).replace("\n", "\n\n")

def strip_heavy_fields(officer: dict) -> dict:
    """Remove large or unnecessary fields before sending to LLM."""
    keys_to_remove = {
        "vector", "embedding", "document", "content", "raw_text", "text_chunks",
        "full_doc", "original_pdf", "text", "metadata"  # adjust as per your real structure
    }
    return {k: v for k, v in officer.items() if k not in keys_to_remove}


# --- Main Query Handler ---
def handle_user_query(query, session, user_feedback=None):
    reset_logs()
    log(f"Incoming Query: {query}")
    log(f"User Feedback: {user_feedback}")

    enriched_query = rate_limited_call(enrich_query_35, query) or query
    log(f"Enriched Query: {enriched_query}")

    query_vector = EMBEDDING_FUNC.embed_query(enriched_query)
    log(f"Query Vector: {query_vector[:5]}...")

    if user_feedback == "force_raw":
        resp, res, src = fallback_to_raw_or_web(query, query_vector, session, user_feedback)
        return resp, res, src, get_logs()

    if user_feedback == "force_web":
        session.log_score(query, score=0.0, source="web", was_fallback=True, force_override=True)
        resp = web_fallback(query, session, origin="manual_force_web")
        return resp, [], "web", get_logs()

    gold_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=10,
        with_payload=True,
        with_vectors=True,
        query_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="gold"))])
    )

    log(f"GOLD Hits: {len(gold_hits)}")
    if gold_hits:
        log(f"Top GOLD Score: {gold_hits[0].score:.4f}")

    if not gold_hits:
        log("No GOLD hits found. Falling back to RAW.")
        session.log_score(query, score=0.0, source="gold", was_fallback=True, force_override=False)
        resp, res, src = fallback_to_raw_or_web(query, query_vector, session, user_feedback)
        return resp, res, src, get_logs()

    result = [
        {
            **hit.payload,
            "embedding": hit.vector,
            "_vector_score": hit.score
        } for hit in gold_hits if hit.vector is not None
    ]

    summary_md, top_3 = run_selection_engine(query, result)
    valid_top = [r for r in top_3 if r.get("total_score", 0) >= THRESHOLD_SELECTION]

    if not valid_top:
        log("No valid GOLD results passed selection threshold. Falling back to RAW.")
        resp, res, src = fallback_to_raw_or_web(query, query_vector, session, user_feedback)
        return resp, res, src, get_logs()

    light_officers = [strip_heavy_fields(o) for o in valid_top]
    summary = rate_limited_call(format_output_4, query, json.dumps(light_officers))
    log(f"LLM Summary: {summary}")

    parsed_summary = []
    try:
        parsed_summary = json.loads(summary)
    except Exception as e:
        log(f"⚠️ JSON parse error in LLM gold output: {e}")

    # Check if parsed summary is empty
    if not parsed_summary:
        log("LLM returned no officers. Falling back to RAW.")
        resp, res, src = fallback_to_raw_or_web(query, query_vector, session, user_feedback)
        return resp, res, src, get_logs()

    session.update(query, valid_top, is_refinement=False)

    return (
        safe_format_markdown(summary) + "\n\n🟢 **Source: GOLD**",
        valid_top,
        "gold",
        get_logs()
    )

# --- RAW Fallback ---
def fallback_to_raw_or_web(query, query_vector, session, user_feedback):
    raw_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
        with_vectors=True,
        query_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="raw"))])
    )

    if not raw_hits:
        session.log_score(query, score=0.0, source="raw", was_fallback=True, force_override=False)
        resp = web_fallback(query, session, origin="no_raw")
        return resp, [], "web"

    top_score = raw_hits[0].score
    force_web = user_feedback == "force_web"
    session.log_score(query, score=top_score, source="raw", was_fallback=top_score < THRESHOLD_RAW, force_override=force_web)

    if top_score >= THRESHOLD_RAW and not force_web:
        result = [
            {
                "officer_name": r.payload.get("officer_name"),
                "source_file": r.payload.get("source_file"),
                "source": r.payload.get("source"),
                "embedding": r.vector
            } for r in raw_hits if r.vector is not None
        ]
        session.update(query, result, is_refinement=False)
        return format_markdown(result) + "\n\n🟠 **Source: RAW**", result, "raw"
    else:
        resp = web_fallback(query, session, origin="no_threshold")
        return resp, [], "web"
    
# --- Web Fallback ---
def web_fallback(query, session, origin="unknown"):
    web_queries = rate_limited_call(search_queries_35, query)
    results, urls = [], []

    for wq in web_queries.get("web_queries", [query]):
        snippets = search_web(wq)
        urls.extend(snippets.get("urls", []))
        results.extend(snippets.get("snippets", []))
        time.sleep(1)

    if not results:
        return "❌ No results found. Please try another query."

    summary = rate_limited_call(web_output_35, query, snippets=results)
    session.update(query, summary, is_refinement=False)
    session.log_score(query, score=0.0, source="web", was_fallback=True, force_override=False)

    if isinstance(summary, str):
        markdown = summary
    else:
        markdown = format_web_results(summary)

    url_block = "\n\n" + "\n\n".join(f"🔗 {url}" for url in urls)
    return markdown + url_block + "\n\n🌐 **Source: WEB**"

# --- Refine Handler ---
def refine_query(refine_prompt, previous_result, session):
    reset_logs()
    first_result = previous_result[0]
    source_type = first_result.get("source", "gold")
    FIELD_SET = list(GOLD_FIELDS if source_type == "gold" else RAW_FIELDS)

    check = rate_limited_call(verification_query_35, refine_prompt, FIELD_SET)
    log(f"Verification result: {check}")
    if check.get("irrelevant", False):
        log("Query marked as irrelevant. Falling back to web.")
        return run_web_fallback(refine_prompt, session)

    embedded_query = EMBEDDING_FUNC.embed_query(refine_prompt)
    filtered_hits = []
    log(f"Embedded refine prompt vector: {embedded_query[:5]}...")

    for officer in previous_result:
        name = officer.get("officer_name") or officer.get("name", "Unknown")
        if "embedding" not in officer:
            log(f"Officer '{name}' has no embedding.")
            continue

        score = cosine_similarity(officer["embedding"], embedded_query)
        log(f"Cosine score for '{name}': {score:.4f}")
        if score >= THRESHOLD_REFINE:
            officer["_score"] = score
            filtered_hits.append(officer)

    log(f"Total officers above threshold ({THRESHOLD_REFINE}): {len(filtered_hits)}")

    if not filtered_hits:
        log("No officers passed refinement threshold. Falling back to web.")
        return run_web_fallback(refine_prompt, session)

    filtered_hits_for_llm = [{k: v for k, v in officer.items() if k != "embedding"} for officer in filtered_hits]
    summary = rate_limited_call(refine_format_output_4, refine_prompt, filtered_hits_for_llm)
    session.update(refine_prompt, filtered_hits, is_refinement=True)

    return (
        safe_format_markdown(summary) + "\n\n🟢 **Source: REFINED**",
        filtered_hits,
        "refined",
        get_logs()
    )

def run_web_fallback(refine_prompt, session):
    web_queries = rate_limited_call(search_queries_35, refine_prompt)
    results, urls = [], []
    for wq in web_queries.get("web_queries", [refine_prompt]):
        snippets = search_web(wq)
        urls.extend(snippets.get("urls", []))
        results.extend(snippets.get("snippets", []))
        time.sleep(1)

    if not results:
        return "❌ No results found. Please try another query.", [], "web"

    summary = rate_limited_call(web_output_35, refine_prompt, results)
    session.update(refine_prompt, summary, is_refinement=True)

    if isinstance(summary, str):
        markdown = summary
    else:
        markdown = format_web_results(summary)

    url_block = "\n\n" + "\n\n".join(f"🔗 {url}" for url in urls)
    return markdown + url_block + "\n\n🌐 **Source: WEB**", summary, "web"

# --- Rollback ---
def rollback_last_step(session):
    reset_logs()
    restored = session.undo_last()
    if not restored:
        log("Nothing to undo.")
        return "⚠️ Nothing to undo.", None, get_logs()

    try:
        restored_data = json.loads(restored)
    except:
        log("Undo failed: Invalid JSON format.")
        return "⚠️ Undo failed: Invalid format.", None, get_logs()

    results = restored_data.get("results", [])
    markdown = safe_format_markdown(results)
    return markdown, results, get_logs()
# --- Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))
