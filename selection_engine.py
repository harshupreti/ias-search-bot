import re
from rapidfuzz import fuzz
import numpy as np
import json
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv(dotenv_path="GITHUB_TOKEN.env")
token = os.environ["GITHUB_TOKEN"]
from debug_logger import log, reset_logs, get_logs

model = "gpt-4o-mini"
client = OpenAI(api_key=token)
# # Setup OpenAI client
# endpoint = "https://models.github.ai/inference"
# model = "openai/gpt-4.1"
# client = OpenAI(
#     base_url=endpoint,
#     api_key=token,
# )

# Configuration: weights for combining match scores
WEIGHTS = {
    'exact_match': 0.7,
    'partial_match': 0.5,
    'fuzzy': 0.3,
    'token_overlap': 0.1,
    'vector': 0.2,             # weight for vector similarity
    'year_boost': 0.15,        # boost for matching year conditions
    'year_exact_boost': 0.2,   # boost for exact year match
    'penalty': 0.35            # penalty weight for NOT conditions
}

# Metadata fields for trait extraction
METADATA_FIELDS = [
    'education', 'postings', 'training_details',
    'awards_publications', 'cadre', 'recruitment_mode'
]


def extract_traits(query: str) -> dict:
    """
    Use an LLM to extract relevant traits from the query.
    Returns a dict mapping each metadata field to list of terms or null, plus 'year_filter'.
    """
    system_prompt = (
        "You are an assistant that extracts structured query traits for matching IAS officer data. "
        "Given the user query, identify which of the following metadata fields are referenced: "
        f"{', '.join(METADATA_FIELDS)}. "
        "Also detect any year constraint (e.g., after 2005, before 2010) with operator and year. "
        "Strictly return a JSON object with each field name mapping to a list of terms or null, "
        "plus 'year_filter' as [operator, year] or null."
    )
    user_prompt = f"User Query: {query}\n\nRespond only with JSON."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```json"):
        raw = raw.lstrip("```json").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        traits = json.loads(raw)
    except Exception as e:
        log(f"❌ JSON parsing failed in extract_traits: {e}")
        log("Fallback triggered. Raw content was:\n" + raw)
        traits = {field: None for field in METADATA_FIELDS}
        traits['year_filter'] = None
    return traits


def parse_year_constraint(traits: dict) -> tuple | None:
    """
    Normalize year_filter from traits dict to (op, year) tuple or None.
    """
    yf = traits.get('year_filter')
    if isinstance(yf, list) and len(yf) == 2:
        op, year = yf
        if isinstance(year, int) and op in ('>', '<', '==', '!='):
            log(f" parse_year_constraint -> {(op, year)}")
            return op, year
    return None


def exact_match_score(terms: list, text: str) -> float:
    if not text or not terms:
        return 0.0
    words = set(re.findall(r"\b\w+\b", text.lower()))
    matches = sum(1 for t in terms if t.lower() in words)
    score = min(matches / len(terms), 1.0)
    log(f" exact_match_score terms={terms} score={score}")
    return score


def partial_match_score(terms: list, text: str) -> float:
    if not text or not terms:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for t in terms if t.lower() in text_lower)
    score = min(matches / len(terms), 1.0)
    log(f" partial_match_score terms={terms} score={score}")
    return score


def fuzzy_score(terms: list, text: str) -> float:
    if not text or not terms:
        return 0.0
    scores = [fuzz.partial_ratio(t, text) / 100.0 for t in terms]
    max_score = max(scores)
    log(f" fuzzy_score terms={terms} max_score={max_score}")
    return max_score


def token_overlap_score(terms: list, text: str) -> float:
    if not text or not terms:
        return 0.0
    text_tokens = set(re.findall(r"\w+", text.lower()))
    term_tokens = set(t.lower() for t in terms)
    overlap = len(text_tokens & term_tokens) / max(len(term_tokens), 1)
    log(f" token_overlap_score terms={terms} overlap={overlap}")
    return overlap


def score_officer(officer: dict, traits: dict) -> dict:
    """
    Compute scores across metadata fields using match functions, vector, and year logic.
    """
    combined = []
    for field in METADATA_FIELDS:
        val = officer.get(field)
        if isinstance(val, list): combined.append(' '.join(val))
        elif isinstance(val, str): combined.append(val)
    combined_text = ' '.join(combined)

    # Gather trait terms
    all_terms = []
    for field in METADATA_FIELDS:
        terms = traits.get(field)
        if terms:
            all_terms.extend(terms if isinstance(terms, list) else [terms])

    # Tiered matching
    s_ex = exact_match_score(all_terms, combined_text)
    s_pa = partial_match_score(all_terms, combined_text)
    s_fz = fuzzy_score(all_terms, combined_text)
    s_to = token_overlap_score(all_terms, combined_text)

    trait_score = (
        WEIGHTS['exact_match'] * s_ex +
        WEIGHTS['partial_match'] * s_pa +
        WEIGHTS['fuzzy'] * s_fz +
        WEIGHTS['token_overlap'] * s_to
    )
    log(f" trait_score={trait_score}")

    vec = officer.get('vector_score', 0.0)
    total = WEIGHTS['vector'] * vec + (1 - WEIGHTS['vector']) * trait_score
    log(f" base total (with vector)={total}")

    # Year boost/penalty
    yf = parse_year_constraint(traits)
    if yf:
        op, y = yf
        try:
            oy = int(officer.get('allotment_year', 0))
            if op == '>' and oy > y:
                total += WEIGHTS['year_boost']
            elif op == '<' and oy < y:
                total += WEIGHTS['year_boost']
            elif op == '==' and oy == y:
                total += WEIGHTS['year_exact_boost']
            elif op == '!=' and oy == y:
                total -= WEIGHTS['penalty']
            log(f" total after year logic={total}")
        except:
            pass

    # Filter out weird or zero
    if total <= 0:
        log(f" Removing officer {officer.get('officer_name')} due to non-positive total_score")
        return None

    total = max(0.0, min(1.0, total))
    officer_out = officer.copy()
    officer_out.update({
        'exact_match_score': s_ex,
        'partial_match_score': s_pa,
        'fuzzy_score': s_fz,
        'token_overlap_score': s_to,
        'trait_score': trait_score,
        'vector_score': vec,
        'total_score': total
    })
    log(f" Scored officer {officer_out.get('officer_name')} total_score={total}")
    return officer_out


def select_best_officers(scored: list, top_n: int = 3) -> list:
    valid = [o for o in scored if o]
    sorted_officers = sorted(valid, key=lambda o: o['total_score'], reverse=True)
    log(f" Top {top_n} after sorting: {[o.get('officer_name') for o in sorted_officers[:top_n]]}")
    return sorted_officers[:top_n]


def generate_reasoning(officer: dict) -> str:
    reasons = []

    # Semantic-only fallback marker
    if officer.get("semantic_only", False):
        return "Selected due to high semantic similarity with the query when explicit metadata traits were insufficient."

    if officer['exact_match_score'] > 0.5:
        reasons.append('strong exact matches')
    if officer['partial_match_score'] > 0.5:
        reasons.append('partial keyword matches')
    if officer['fuzzy_score'] > 0.5:
        reasons.append('close term matches')
    if officer['token_overlap_score'] > 0.3:
        reasons.append('shared terms')
    if officer['vector_score'] > 0.6:
        reasons.append('high semantic similarity')

    if not reasons:
        reasons.append('overall relevance')
    return 'Selected due to ' + ', '.join(reasons) + '.'


def run_selection_engine(query: str, officers: list):
    log(f" Starting selection pipeline for query: {query}")
    traits = extract_traits(query)

    scored = []
    trait_scores = []

    for o in officers:
        scored_officer = score_officer(o, traits)
        if scored_officer:
            scored.append(scored_officer)
            trait_scores.append(scored_officer["trait_score"])

    # ✅ Semantic-only fallback condition
    if trait_scores and all(score < 0.3 for score in trait_scores):
        log("All trait scores below 0.3 — using semantic-only fallback (total_score = vector_score)")
        for officer in scored:
            officer['total_score'] = officer.get('_vector_score', officer.get('vector_score', 0.0))
            officer['semantic_only'] = True
            log(f"Total score: {officer['total_score']}")

    selected = select_best_officers(scored)

    for o in selected:
        o['selection_reasoning'] = generate_reasoning(o)

    summary = []
    for i, o in enumerate(selected, 1):
        summary.append(f"**{i}) {o.get('officer_name','Unknown Officer')}**")
        summary.append(f"- **Total Score**: {o['total_score']:.2f}")
        summary.append(f"- **Why Selected**: {o['selection_reasoning']}")
        summary.append('')
    summary_md = '\n'.join(summary)

    log(f"Pipeline complete.")
    return summary_md, selected