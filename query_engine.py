import json
import time
from collections import deque
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv

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
THRESHOLD_GOLD = 0.65
THRESHOLD_RAW = 0.60
THRESHOLD_REFINE = 0.60

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

# --- Helper: format doc ---
def format_doc(doc):
    m = getattr(doc, "metadata", None) or getattr(doc, "payload", {})
    return {
        "name": m.get("name"),
        "present_post": m.get("present_post"),
        "qualifications": m.get("qualifications"),
        "domicile": m.get("domicile"),
        "allotment_year": m.get("allotment_year"),
        "remarks": m.get("remarks"),
    }

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

    output = []
    for idx, officer in enumerate(data, 1):
        if not isinstance(officer, dict):
            continue

        block = [f"**{idx}) {officer.get('officer_name', 'Unknown Officer')}**"]

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

        # Add clickable resume link if it's a SAS URL
        source_file = officer.get("source_file", "")
        if source_file.startswith("https://"):
            block.append(f"[📄 View Resume]({source_file})")
        else:
            block.append(f"**Source File**: {render_value(source_file)}")

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

# --- Main Query Handler ---
def handle_user_query(query, session, user_feedback=None):
    GOLD_FIELDS = {
        "officer_name", "cadre", "allotment_year", "recruitment_mode",
        "education", "postings", "training_details", "awards_publications",
        "source_file", "source"
    }

    print(f"\n---\n[DEBUG] Incoming Query: {query}")
    print(f"[DEBUG] User Feedback: {user_feedback}")

    if user_feedback == "force_raw":
        query_vector = EMBEDDING_FUNC.embed_query(query)
        return fallback_to_raw_or_web(query, query_vector, session, user_feedback)

    if user_feedback == "force_web":
        session.log_score(query, score=0.0, source="web", was_fallback=True, force_override=True)
        return web_fallback(query, session, origin="manual_force_web"), [], "web"

    # Step 1: Structured verification
    check = rate_limited_call(verification_query_35, query, list(GOLD_FIELDS))
    print(f"[DEBUG] Verification result: {check}")
    if check.get("irrelevant", False):
        print("[DEBUG] Query flagged as irrelevant to GOLD. Switching to WEB fallback.")
        session.log_score(query, score=0.0, source="web", was_fallback=True, force_override=False)
        return web_fallback(query, session, origin="direct"), [], "web"

    # Step 2: Embed and search GOLD
    query_vector = EMBEDDING_FUNC.embed_query(query)
    print(f"[DEBUG] Query Vector: {query_vector[:5]}...")

    gold_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
        query_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="gold"))])
    )

    print(f"[DEBUG] GOLD Hits: {len(gold_hits)}")
    if gold_hits:
        print(f"[DEBUG] Top GOLD Score: {gold_hits[0].score:.4f}")
        print(f"[DEBUG] Top GOLD Payload: {json.dumps(gold_hits[0].payload, indent=2)}")

    if not gold_hits:
        print("[DEBUG] No GOLD hits found. Falling back to RAW.")
        session.log_score(query, score=0.0, source="gold", was_fallback=True, force_override=False)
        return fallback_to_raw_or_web(query, query_vector, session, user_feedback)

    top_score = gold_hits[0].score
    session.log_score(query, score=top_score, source="gold", was_fallback=top_score < THRESHOLD_GOLD, force_override=False)

    # Step 3: Format with LLM
    if top_score >= THRESHOLD_GOLD:
        print("[DEBUG] GOLD score is sufficient. Formatting result with LLM...")
        summary = rate_limited_call(format_output_4, query, [hit.payload for hit in gold_hits])
        print(f"[DEBUG] LLM Summary: {summary}")
        session.update(query, summary, is_refinement=False)
        return safe_format_markdown(summary) + "\n\n🟢 **Source: GOLD**", summary, "gold"
    else:
        print(f"[DEBUG] GOLD score below threshold ({top_score:.4f} < {THRESHOLD_GOLD}). Falling back to RAW.")
        return fallback_to_raw_or_web(query, query_vector, session, user_feedback)


# --- RAW Fallback ---
def fallback_to_raw_or_web(query, old_vector, session, user_feedback):
    enriched_query = rate_limited_call(enrich_query_35, query)
    enriched_vector = EMBEDDING_FUNC.embed_query(enriched_query)

    raw_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=enriched_vector,
        limit=5,
        with_payload=True,
        with_vectors=True,
        query_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="raw"))])
    )

    if not raw_hits:
        session.log_score(query, score=0.0, source="raw", was_fallback=True, force_override=False)
        return web_fallback(query, session, origin="no_raw"), [], "web"

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
        return web_fallback(query, session, origin="no_threshold"), [], "web"


# --- Web Fallback ---
def web_fallback(query, session, origin="unknown"):
    web_queries = rate_limited_call(search_queries_35, query)
    results, urls = [], []

    for wq in web_queries.get("web_queries", [query]):
        snippets = search_web(wq)
        urls.extend(snippets.get("urls", []))
        results.extend(snippets.get("snippets", []))
        time.sleep(1)

    summary = rate_limited_call(web_output_35, query, snippets=results)
    session.update(query, summary, is_refinement=False)
    session.log_score(query, score=0.0, source="web", was_fallback=True, force_override=False)

    if isinstance(summary, str):
        markdown = summary
    else:
        markdown = format_web_results(summary)

    url_block = "\n\n" + "\n".join(f"🔗 {url}" for url in urls)
    return markdown + url_block + "\n\n🌐 **Source: WEB**"


# --- Refine Handler ---
def refine_query(refine_prompt, previous_result, session):
    # Fallback to web if no previous results available
    if not previous_result:
        web_queries = rate_limited_call(search_queries_35, refine_prompt)
        results, urls = [], []

        for wq in web_queries.get("web_queries", [refine_prompt]):
            snippets = search_web(wq)
            urls.extend(snippets.get("urls", []))
            results.extend(snippets.get("snippets", []))
            time.sleep(1)

        summary = rate_limited_call(web_output_35, refine_prompt, results)
        session.update(refine_prompt, summary, is_refinement=True)

        if isinstance(summary, str):
            markdown = summary
        else:
            markdown = format_web_results(summary)

        url_block = "\n\n" + "\n".join(f"🔗 {url}" for url in urls)
        return markdown + url_block + "\n\n🌐 **Source: WEB**", summary, "web"

    # Determine which field set to use based on previous results
    first_result = previous_result[0]
    source_type = first_result.get("source", "gold")

    FIELD_SET = GOLD_FIELDS if source_type == "gold" else RAW_FIELDS

    # Step 1: Structured Verification
    check = rate_limited_call(verification_query_35, refine_prompt, FIELD_SET)
    if check.get("irrelevant", False):
        officer_names = [officer.get("officer_name") or officer.get("name") for officer in previous_result]
        query_with_names = f"{refine_prompt}\n\nOfficer names:\n" + ", ".join(officer_names)

        web_queries = rate_limited_call(refine_search_queries_35, query_with_names)
        results, urls = [], []

        for wq in web_queries.get("web_queries", [refine_prompt]):
            snippets = search_web(wq)
            urls.extend(snippets.get("urls", []))
            results.extend(snippets.get("snippets", []))
            time.sleep(1)

        summary = rate_limited_call(web_output_35, refine_prompt, results)
        session.update(refine_prompt, summary, is_refinement=True)

        if isinstance(summary, str):
            markdown = summary
        else:
            markdown = format_web_results(summary)

        url_block = "\n\n" + "\n".join(f"🔗 {url}" for url in urls)
        return markdown + url_block + "\n\n🌐 **Source: WEB**", summary, "web"

    # Step 2: Filter previous officers based on semantic match
    embedded_query = EMBEDDING_FUNC.embed_query(refine_prompt)
    filtered_hits = []

    for officer in previous_result:
        if "embedding" not in officer:
            continue
        score = cosine_similarity(officer["embedding"], embedded_query)
        if score >= THRESHOLD_REFINE:
            officer["_score"] = score
            filtered_hits.append(officer)

    # Step 3: Fallback to web if no relevant filtered officers
    if not filtered_hits:
        web_queries = rate_limited_call(search_queries_35, refine_prompt)
        results, urls = [], []

        for wq in web_queries.get("web_queries", [refine_prompt]):
            snippets = search_web(wq)
            urls.extend(snippets.get("urls", []))
            results.extend(snippets.get("snippets", []))
            time.sleep(1)

        summary = rate_limited_call(web_output_35, refine_prompt, results)
        session.update(refine_prompt, summary, is_refinement=True)

        if isinstance(summary, str):
            markdown = summary
        else:
            markdown = format_web_results(summary)

        url_block = "\n\n" + "\n".join(f"🔗 {url}" for url in urls)
        return markdown + url_block + "\n\n🌐 **Source: WEB**", summary, "web"

    # Step 4: Local refinement on filtered officers
    summary = rate_limited_call(refine_format_output_4, refine_prompt, filtered_hits)
    session.update(refine_prompt, summary, is_refinement=True)
    return safe_format_markdown(summary) + "\n\n🟢 **Source: REFINED**", summary, "refined"


# --- Rollback ---
def rollback_last_step(session):
    restored = session.undo_last()
    if not restored:
        return "⚠️ Nothing to undo.", None

    try:
        restored_data = json.loads(restored)
    except:
        return "⚠️ Undo failed: Invalid format.", None

    results = restored_data.get("results", [])
    markdown = safe_format_markdown(results)
    return markdown, results

# --- Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))
