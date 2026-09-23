"""
Pattern extraction and detection module for query operational constraints.

Contains functions that analyze user queries using regular expressions and keyword
heuristics to extract latency SLAs, context window requirements, data privacy/on-premise
needs, budget caps, and capability signals.
"""

import re
from typing import List, Optional


def _extract_max_latency_ms(query: str) -> Optional[int]:
    """
    Extracts latency threshold constraint if mentioned in the user query.
    params: input query.
        1. Checks for millisecond pattern if present in the query ("under 500ms", "latency under 200 ms", "sub-500ms", "< 500ms").
        2. Checks for seconds pattern if present in the query ("under 2.5s", "under 2 seconds", "within 1 sec").
        3. Converts seconds to milliseconds if required.
    returns: latency threshold in milliseconds.
    """
    ms_match = re.search(r"(?:under|sub-|below|<|within)\s*(\d+)\s*(?:ms|milliseconds?)", query, re.IGNORECASE)
    if ms_match:
        return int(ms_match.group(1))

    sec_match = re.search(r"(?:under|sub-|below|<|within)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)", query, re.IGNORECASE)
    if sec_match:
        return int(float(sec_match.group(1)) * 1000)

    return None


def _extract_min_context_k(query: str) -> Optional[int]:
    """
    Extracts minimum context window constraint if mentioned in the user query.
    params: input query.
        1. Extracts context window information based on number of pages or documents mentioned in the query.
            e.g. 400 pages ~ 260K Tokens, 40 pages ~ 26K Tokens. 
        2. Estimates context window based on document volume in query.
            e.g. 600 articles, 500 documents.
    returns: minimum token capacity required in thousands of tokens (k).
    """

    page_match = re.search(r"(\d+)\s*-?\s*pages?", query, re.IGNORECASE)
    if page_match:
        pages = int(page_match.group(1))
        estimated_tokens_k = max(1, int(pages * 0.65))
        return estimated_tokens_k

    doc_match = re.search(r"(\d+)\s+(?:articles?|documents?|knowledge base articles?)", query, re.IGNORECASE)
    if doc_match:
        docs = int(doc_match.group(1))
        return max(1, int(docs * 0.5))  # roughly 500 words/tokens per article

    return None


def _requires_on_premise(query: str) -> bool:
    """
    Checks for privacy and data governance requirement.
    returns: True/False
    """
    patterns = [
        r"on-premise",
        r"on-prem",
        r"no data (?:may )?leave",
        r"internal (?:infrastructure|servers)",
        r"air-gapped",
        r"open-weights",
        r"data governance",
        r"private infrastructure",
    ]
    return any(re.search(pat, query, re.IGNORECASE) for pat in patterns)


def _requires_low_cost(query: str) -> bool:
    """
    Checks for explicit budget phrases(e.g. "budget cap 400 USD", "cost under 15 USD", "500,000 SKUs"). 
    returns: True/False 
    """
    patterns = [
        r"budget cap",
        r"cost under",
        r"target total cost",
        r"budget.*?(?:under|\$)",
        r"low-cost",
        r"cost-effective",
        r"\d{3,},\d{3}\s+(?:skus?|items?|records?)",  # high volume e.g. 500,000 SKUs
    ]
    return any(re.search(pat, query, re.IGNORECASE) for pat in patterns)


def _extract_required_capabilities(query: str) -> List[str]:
    """
    Extracts specific capability signals mentioned in the query.
    params: input query.
    returns: List of capability tags required by the query.
    """
    required = []
    q_lower = query.lower()

    if "json" in q_lower or "structured output" in q_lower or "structured data" in q_lower:
        required.append("structured")
    if "typography" in q_lower or "legibl" in q_lower or "banner" in q_lower or "poster" in q_lower:
        required.append("typography")
    if "cinematic" in q_lower or "camera" in q_lower or "motion" in q_lower or "video" in q_lower:
        required.append("motion")
    if "photoreal" in q_lower or "pack shot" in q_lower or "hero image" in q_lower:
        required.append("photoreal")

    return required
