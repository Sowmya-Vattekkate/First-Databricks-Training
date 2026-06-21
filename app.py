import streamlit as st

from databricks.ai_search.client import AISearchClient

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole
)

# ----------------------------
# Configuration
# ----------------------------

ENDPOINT_NAME = "hr_policy_endpoint"

INDEX_NAME = "databricks_poc_ws.hr_chatbot.hr_policy_index"

MODEL_NAME = "databricks-meta-llama-3-1-8b-instruct"

# ----------------------------
# databricks Clients
# ----------------------------

search_client = AISearchClient()

index = search_client.get_index(
    endpoint_name=ENDPOINT_NAME,
    index_name=INDEX_NAME
)

w = WorkspaceClient()

# ----------------------------
# Retreival 
# ----------------------------

def retrieve_chunks(question):

    results = index.similarity_search(
        query_text=question,
        columns=[
            "content",
            "page_number",
            "source"
        ],
        num_results=8
    )

    return results

# ----------------------------
# Context Builder
# ----------------------------

def build_context(results):

    rows = results["result"]["data_array"]

    context = []

    for row in rows:

        context.append(row[0])

    return "\n\n".join(context)

# ------------------------------------
# Source Extraction
# ------------------------------------

import re

def extract_sources(results):

    rows = results["result"]["data_array"]

    pages = set()

    for row in rows:

        source = row[2]

        match = re.search(r'Page\s+(\d+)', source)

        if match:
            pages.add(int(match.group(1)))

    return sorted(pages)

# ------------------------------------
# Define Prompt
# ------------------------------------

def create_prompt(question, context):

    return f"""
You are an HR Policy Assistant.

Rules:
1. Answer ONLY using the provided HR policy context.
2. Do NOT use external knowledge.
3. Provide a concise and business-friendly answer.
4. Summarize information instead of copying large sections verbatim.
5. Use bullet points when multiple policy rules apply.
6. If information is partially available, provide only what is available.
7. If information is not available, respond exactly with:
   Information not found in HR policy.

Context:
{context}

Question:
{question}

Answer:
"""

# ------------------------------------
# Main Chat function
# ------------------------------------

def ask_hr_bot(question):

    results = retrieve_chunks(question)

    context = build_context(results)

    prompt = create_prompt(
        question,
        context
    )

    response = w.serving_endpoints.query(
        name=MODEL_NAME,
        temperature=0,
        max_tokens=500,
        messages=[
            ChatMessage(
                role=ChatMessageRole.USER,
                content=prompt
            )
        ]
    )

    answer = response.choices[0].message.content

    sources = extract_sources(results)

    return answer, sources


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    layout="wide"
)

st.title("📘 HR Policy Assistant")

question = st.text_input(
    "Ask a question from the HR Policy"
)

if st.button("Submit"):

    if question:

        with st.spinner("Searching policy..."):

            answer, sources = ask_hr_bot(question)

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for s in sources:

            st.write(f"• {s}")