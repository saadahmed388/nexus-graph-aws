import json
from typing import Literal
from pydantic import BaseModel, Field

from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAIResponses

from schema_and_prompts.system_prompts import intent_resolution_prompt
from schema_and_prompts.db_schema import GRAPH_DB_SCHEMA_JSON
from retrieval_agent import function_tools

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = OpenAIResponses(
    model="gpt-5.6-terra", 
    temperature=0.0, 
    strict=True, 
    reasoning_options={
        "effort": "xhigh",
        "summary": "auto",
    },
    api_key=OPENAI_API_KEY
)

class QueryIntentResult(BaseModel):
    intent: Literal["METADATA", "ANALYSIS", "ROOT_CAUSE", "RESOLUTION", "COMMENTS", "CODE_CHANGES", \
                    "RELATED_TICKETS", "SEMANTIC_SEARCH", "EXACT_SEARCH", "AGGREGATION", "GENERAL"]
    sub_intent: str
    required_evidence: list[str]
    retrieval_strategy: list[str]


def analyze_query_intent_tool(query: str) -> str:

    prompt = PromptTemplate("""
    User Query : {query}
    {intent_resolution_prompt}
    """)

    result = llm.structured_predict(
        QueryIntentResult, 
        prompt=prompt, 
        query=query, 
        intent_resolution_prompt = intent_resolution_prompt
    )
    
    return result.model_dump_json()
