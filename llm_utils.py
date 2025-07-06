import os
import json
import re
import ast
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="GITHUB_TOKEN.env")
token = os.environ["GITHUB_TOKEN"]

# Setup OpenAI client
# endpoint = "https://models.github.ai/inference"
model = "gpt-4o-mini"
client = OpenAI(api_key=token)

# endpoint = "https://models.github.ai/inference"
# model = "openai/gpt-4.1"
# client = OpenAI(
#     base_url=endpoint,
#     api_key=token,
# )

# Available metadata fields for structured verification
GOLD_FIELDS = {
    "officer_name", "cadre", "allotment_year", "recruitment_mode",
    "education", "postings", "training_details", "awards_publications",
    "source_file", "source"
}

RAW_FIELDS = {
    "officer_name", "source_file", "source"
}

# --- Utility: Safe JSON parser ---
def safe_json_parse(text, fallback="[]"):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return json.loads(fallback)

# --- 1. Check if query is valid for metadata search ---
def verification_query_35(query, fields):
    metadata_fields = sorted(list(fields))
    metadata_str = ", ".join(metadata_fields)

    system_prompt = f"""
You are a strict classifier that checks if a user query can be answered using structured IAS officer metadata.

Here are the **available metadata fields**:
{metadata_str}

✅ You SHOULD mark the query as `"irrelevant": false` if it:
- Uses filters on any of these fields (e.g. year > 2000, cadre = Bihar)
- Asks about education, training, recruitment, postings, or awards
- Requests grouping, sorting, or selection based on structured fields

❌ Mark as `"irrelevant": true` only if:
- It asks for personal views, opinions, ethics, media news, behavior, controversies, or recent public reports
- It requires real-time info or family background
- It has nothing to do with officer data

Return ONLY a JSON object like: {{ "irrelevant": true/false }}

DO NOT include any extra text or explanation.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": query}
        ],
        temperature=0.0  # deterministic
    )
    raw = response.choices[0].message.content
    print("[DEBUG] Raw verification LLM response:", raw)
    return safe_json_parse(raw)


# --- 2. Generate final gold response ---
def generate_final_response(query: str, officer_data_json: str) -> str:
    system_prompt = """
You are an AI assistant that formats structured IAS officer metadata based on a user query.

Your task is to:
1. Return a VALID JSON list of officers
2. For each officer:
   - Only include the latest and most relevant postings based on the user query
   - Omit any fields that are None or contain 'N/A'
   - Add a "selection_reasoning" field that briefly explains (in 3–4 lines) why this officer is a good match for the query
3. Remove any officer that is not even remotely relevant to the query or is a bad match.
Be precise and only use details from the metadata. Do NOT invent.

Format:
[
  {
    "officer_name": "...",
    "cadre": "...",
    "allotment_year": "...",
    "recruitment_mode": "...",
    "education": "...",
    "postings": ["..."],  # Only latest and relevant to query
    "training_details": ["..."],
    "awards_publications": "...",
    "source_file": "...",
    "source": "gold",
    "selection_reasoning": "This officer was selected because ..."
  },
  ...
]

Keep officer_name clean (remove any 'Identity No.' etc.)
Always include ALL officers given in the metadata.
"""


    user_content = f"User Query:\n{query}\n\nOfficer Metadata:\n{officer_data_json}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content}
            ],
            temperature=0.25,
            top_p=1.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Failed in generate_final_response:", str(e))
        return "[]"

def format_output_4(query, officer_data):
    response = generate_final_response(query, json.dumps(officer_data))
    return safe_json_parse(response)

# --- 3. Enrich query for raw search ---
def enrich_query_35(query):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Improve the query to search noisy raw resume data. Return only the enriched query string."},
            {"role": "user", "content": query}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# --- 4. Web search queries from original query ---
def search_queries_35(query):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query generator for a system that finds information about IAS officers in India only.\n"
                    "Generate a list of 3–5 precise web search queries that explicitly mention 'IAS officers' or similar phrases like 'Indian Administrative Service'.\n"
                    "Avoid queries that might match police officers, foreign officers, or unrelated roles.\n"
                    "Return ONLY a valid JSON list of short search queries. No explanations, no bullet points, no preambles."
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.4
    )
    parsed = safe_json_parse(response.choices[0].message.content)
    return {"web_queries": parsed if isinstance(parsed, list) else [query]}

# --- 5. Summarize web search results ---
def web_output_35(query, snippets):
    joined = "\n".join(map(str, snippets))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a summarizer bot helping a user find IAS officers based on web snippets.\n"
                    "Group the findings by officer if names are available, otherwise use 'Unknown Officer'.\n"
                    "Format your response in clean markdown. Each section should begin with the officer name like\n"
                    "`**1) Officer Name**`. If the officer is unknown, say `**1) Unknown Officer**`.\n\n"
                    "Use bullet points where appropriate. Avoid returning structured data like JSON or dictionaries.\n"
                    "Conclude with '🌐 Source: WEB'."
                )
            },
            {"role": "user", "content": f"Query: {query}\n\nSnippets:\n{joined}"}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# --- 6. Refine web search queries ---
def refine_search_queries_35(refine_query_with_names):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You're an assistant helping refine IAS officer search queries for web lookup.\n"
                    "Generate 3–5 highly specific and useful **web search queries** based on:\n"
                    "1. The user's refined query (which indicates what additional info they seek).\n"
                    "2. The **list of officer names** provided (these should be incorporated into search queries).\n\n"
                    "⚠️ Return ONLY a JSON array of plain strings like:\n"
                    "[\"Query 1\", \"Query 2\", ...]\n\n"
                    "Do NOT include comments, explanations, or any text outside the JSON list."
                )
            },
            {"role": "user", "content": refine_query_with_names}
        ],
        temperature=0.4
    )

    content = response.choices[0].message.content.strip()
    parsed = safe_json_parse(content)
    return {"web_queries": parsed} if isinstance(parsed, list) else {"web_queries": []}

# --- 7. Refine final output for follow-up ---
def refine_results(refine_prompt: str, previous_officers: list) -> str:
    system_prompt = """
You are an AI assistant refining IAS officer search results based on a user's follow-up query.

You MUST return only a valid JSON list, like this:

[
  {
    "officer_name": "...",
    "cadre": "...",
    "allotment_year": "...",
    "education": "...",
    "postings": ["..."],
    "training_details": ["..."],
    "awards_publications": "...",
    "source_file": "...",
    "source": "gold" or "raw"
  },
  ...
]
"""

    user_prompt = f"Refinement Query: {refine_prompt}\n\nPrevious Officer Results:\n{json.dumps(previous_officers, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            top_p=1.0
        )
        return json.dumps(safe_json_parse(response.choices[0].message.content.strip()))
    except Exception as e:
        print("❌ LLM call failed during refinement:", str(e))
        return "[]"

def refine_format_output_4(refine_prompt, filtered_officers):
    from llm_utils import refine_results
    return safe_json_parse(refine_results(refine_prompt, filtered_officers))
