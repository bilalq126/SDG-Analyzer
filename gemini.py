# gemini.py
"""
Gemini API wrapper and domain-specific helper functions for EcoMind AI.

This module:
- Reads API key from environment (GOOGLE_API_KEY) or streamlit secrets (optional).
- Contains call_gemini() to send prompts to Google Generative API (Gemini).
- Contains analyze_sdg(), generate_pitch(), reverse_ideas() helpers that:
  - construct structured prompts
  - ask Gemini to return JSON ONLY
  - parse that JSON and return Python dicts

NOTE: The exact endpoint payload/response shape can vary depending on Google API versions.
This wrapper attempts to be resilient by detecting common response shapes.
"""

import os
import json
import time
import requests
from typing import Dict, Any, List, Optional

# Default model - change if you have a different stable name available
DEFAULT_MODEL = "gemini-1.5-flash"

# Endpoint base for Google Generative Language API (v1). Adjust if your account uses a different path.
BASE_URL = "https://generativelanguage.googleapis.com/v1/models"

def _get_api_key() -> Optional[str]:
    # prefer environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    # Allow using GOOGLE_API_KEY_STAGING or other names
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY_STAGING")

    # If running under Streamlit, users may have set the key in `st.secrets`.
    # Try to read that as a fallback without making Streamlit a hard dependency.
    if not api_key:
        try:
            import streamlit as _st
            api_key = _st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            # ignore if streamlit is not installed or secrets not available
            pass

    return api_key

def call_gemini(prompt: str,
                model: str = DEFAULT_MODEL,
                temperature: float = 0.2,
                max_output_tokens: int = 700,
                retry: int = 2) -> str:
    """
    Call the Google Generative Language API (Gemini) via REST and return the text output.
    Expects the model to produce textual output (we request JSON in the prompt).
    Set GOOGLE_API_KEY in env vars before calling.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Google API key not set. Set environment variable GOOGLE_API_KEY.")

    headers = {"Content-Type": "application/json; charset=utf-8"}

    payload = {
        "prompt": {
            "text": prompt
        },
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }

    # If the official google.generativeai SDK is installed and an API key is available,
    # prefer using it (simpler auth and model handling). Fall back to REST if the SDK
    # is not available or the call fails.
    try:
        import google.generativeai as genai
        api_key = _get_api_key()
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # The SDK uses a model object; try a simple generate call.
                gm = genai.GenerativeModel(model)
                # Try a couple of common call signatures depending on SDK version.
                try:
                    resp = gm.generate_content(prompt)
                except TypeError:
                    # some SDK versions expect keyword args
                    resp = gm.generate_content(prompt=prompt, temperature=temperature)

                # Extract text from response if possible
                if hasattr(resp, 'text'):
                    return resp.text
                # some SDKs return a dict-like object
                try:
                    rj = resp if isinstance(resp, dict) else resp.to_dict()
                    # look for common keys
                    if isinstance(rj, dict):
                        # try to find text in nested keys
                        def _find_text(obj):
                            if isinstance(obj, str):
                                return obj
                            if isinstance(obj, dict):
                                for v in obj.values():
                                    t = _find_text(v)
                                    if t:
                                        return t
                            if isinstance(obj, list):
                                for item in obj:
                                    t = _find_text(item)
                                    if t:
                                        return t
                            return None
                        t = _find_text(rj)
                        if t:
                            return t
                except Exception:
                    pass
            except Exception:
                # If SDK call fails, continue to REST fallback
                pass
    except Exception:
        # SDK not installed — ignore and use REST logic below
        pass

    # Try multiple common endpoint suffixes to handle API surface differences.
    suffix_candidates = [":generateText", ":generate"]
    attempted_urls = []

    for suffix in suffix_candidates:
        url = f"{BASE_URL}/{model}{suffix}"
        attempted_urls.append(url)

        for attempt in range(retry + 1):
            try:
                resp = requests.post(url, headers=headers, params={"key": api_key}, json=payload, timeout=60)
            except requests.RequestException as re:
                # network-level error — retry if possible
                if attempt < retry:
                    time.sleep(1 + attempt * 2)
                    continue
                raise RuntimeError(f"Network error calling Gemini endpoint {url}: {re}")

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    return resp.text

                # Typical response shapes — extract textual content if present
                text = None
                if isinstance(data, dict):
                    if "candidates" in data:
                        c0 = data["candidates"][0]
                        text = c0.get("output") or (c0.get("content") and "".join([seg.get("text", "") for seg in c0.get("content", [])]))
                    elif "result" in data and "candidates" in data["result"]:
                        c0 = data["result"]["candidates"][0]
                        text = c0.get("output") or (c0.get("content") and "".join([seg.get("text", "") for seg in c0.get("content", [])]))
                    elif "output" in data:
                        text = data.get("output")

                if not text:
                    def extract_text_recursive(obj):
                        if isinstance(obj, str):
                            return obj
                        if isinstance(obj, dict):
                            for v in obj.values():
                                t = extract_text_recursive(v)
                                if t:
                                    return t
                        if isinstance(obj, list):
                            for item in obj:
                                t = extract_text_recursive(item)
                                if t:
                                    return t
                        return None
                    text = extract_text_recursive(data)

                if not text:
                    return json.dumps(data, indent=2)
                return text.strip()

            else:
                # 404 often means incorrect model name or endpoint — try next suffix
                if resp.status_code == 404:
                    # break to try next suffix
                    break
                # retry on 5xx
                if resp.status_code >= 500 and attempt < retry:
                    time.sleep(1 + attempt * 2)
                    continue
                # other error — raise with context
                raise RuntimeError(f"Gemini API error {resp.status_code} from {url}: {resp.text}")

    # If we get here, none of the suffixes returned success
    raise RuntimeError(
        "Gemini API returned 404/Not found. Tried the following endpoints: " + ", ".join(attempted_urls) +
        ".\nCheck that your GOOGLE_API_KEY is correct and that the model name (e.g., 'gemini-2.5-flash') is available for your account."
    )


def _parse_json_from_text(text: str) -> Any:
    """
    Attempts to extract a JSON object from the model output text.
    The model is instructed to return JSON only, but sometimes there are leading/trailing words.
    This function finds the first '{' and last '}' and tries to json.loads that slice.
    """
    text = text.strip()
    # if it already seems to be JSON:
    try:
        return json.loads(text)
    except Exception:
        pass

    # find first '{' and last '}'
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # try to find JSON array
    first_a = text.find("[")
    last_a = text.rfind("]")
    if first_a != -1 and last_a != -1 and last_a > first_a:
        candidate = text[first_a:last_a+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # As fallback, try to replace single quotes with double quotes (best-effort)
    alt = text.replace("'", '"')
    try:
        return json.loads(alt)
    except Exception:
        # give up and return raw text
        return {"raw_text": text}


def analyze_sdg(project_text: str,
                model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Send project_text to Gemini and ask for a structured SDG analysis.
    Returns a dict with keys:
    - sdgs: list of {id:int, short_name:str, score:int (0-100), explanation:str}
    - sustainability_impact: "Low"/"Medium"/"High"
    - feasibility_score: int (1-10)
    - risks: {environmental:[], social:[], economic:[]}
    - recommendations: [str...]
    - notes: optional additional notes

    The prompt forces Gemini to return JSON ONLY with that structure.
    """
    prompt = f"""
You are EcoMind AI, an expert sustainability analyst. Given a project description, produce a structured JSON analysis strictly following the JSON schema below. Do NOT include any extra commentary, only output valid JSON.

Schema:
{{
  "sdgs": [
    {{
      "id": <int 1-17>,
      "short_name": "<short SDG name>",
      "score": <int 0-100>,            // relevance 0 to 100
      "explanation": "<1-2 sentence explanation of why this SDG matches>"
    }}
  ],
  "sustainability_impact": "<Low|Medium|High>",
  "feasibility_score": <int 1-10>,
  "risks": {{
    "environmental": ["<short risk statements>"],
    "social": ["<short risk statements>"],
    "economic": ["<short risk statements>"]
  }},
  "recommendations": ["<actionable recommendation 1>", "<actionable recommendation 2>"],
  "notes": "<optional short note>"
}}

Instructions:
- Analyze the following project and return between 3 and 5 most relevant SDGs only.
- For each SDG give a numeric score (0-100). Ensure scores reflect the project's likely alignment.
- Keep explanations concise (1-2 sentences).
- Determine sustainability_impact conservatively.
- Feasibility_score should be 1-10 assessing technical feasibility.
- Provide up to 3 short risks per category (environmental/social/economic).
- Provide 3-6 practical, AI-aware improvement recommendations (short bullet sentences).
- Return only the JSON object (no markdown, no text).

Project:
\"\"\"{project_text}\"\"\"
"""

    response_text = call_gemini(prompt, model=model, temperature=0.15, max_output_tokens=700)
    parsed = _parse_json_from_text(response_text)

    # If we successfully parsed and have SDGs, return immediately.
    if isinstance(parsed, dict) and parsed.get("sdgs"):
        return parsed

    # Fallback: sometimes the model returns useful text but misses the JSON schema.
    # Ask a focused SDG classifier prompt to extract 3 relevant SDGs when initial parse failed.
    try:
        fallback_prompt = (
            "You are EcoMind AI, an expert sustainability analyst. The following project description may be short or informal.\n"
            "Identify the 3 most relevant Sustainable Development Goals (SDG numbers 1-17) that the project aligns with.\n"
            "Return ONLY a JSON object with a single key \"sdgs\" which is a list of objects with keys:\n"
            "  - id: integer (1-17)\n"
            "  - short_name: short SDG name\n"
            "  - score: integer 0-100 indicating relevance\n"
            "  - explanation: one brief sentence explaining the match\n\n"
            "Project:\n"
            f"{project_text}\n"
        )

        fb_text = call_gemini(fallback_prompt, model=model, temperature=0.0, max_output_tokens=300)
        fb_parsed = _parse_json_from_text(fb_text)
        if isinstance(fb_parsed, dict) and fb_parsed.get("sdgs"):
            # Merge into a full analysis structure, preserving any other fields from original parse if present
            result = {
                "sdgs": fb_parsed.get("sdgs", []),
                "sustainability_impact": parsed.get("sustainability_impact", "Unknown") if isinstance(parsed, dict) else "Unknown",
                "feasibility_score": parsed.get("feasibility_score", 0) if isinstance(parsed, dict) else 0,
                "risks": parsed.get("risks", {"environmental": [], "social": [], "economic": []}) if isinstance(parsed, dict) else {"environmental": [], "social": [], "economic": []},
                "recommendations": parsed.get("recommendations", []) if isinstance(parsed, dict) else [],
                "notes": parsed.get("notes", "Parser fallback — SDG classifier used.") if isinstance(parsed, dict) else "Parser fallback — SDG classifier used.",
            }
            return result
    except Exception:
        # ignore fallback errors and fall through to final fallback
        pass

    # If parsing failed entirely, return fallback structure with raw text
    return {
        "sdgs": [],
        "sustainability_impact": "Unknown",
        "feasibility_score": 0,
        "risks": {"environmental": [], "social": [], "economic": []},
        "recommendations": [],
        "notes": "Parser fallback — raw model output attached.",
        "raw": parsed
    }


def generate_pitch(project_text: str,
                   model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Request Gemini to generate a concise, copyable pitch summary for the project.
    Returns:
    {
      "pitch": "<one-paragraph pitch, 2-4 sentences>",
      "elevator": "<single-sentence elevator pitch>",
      "bullet_points": ["...","..."]
    }
    Output MUST be JSON only.
    """
    prompt = f"""
You are EcoMind AI, an expert startup / sustainability pitch writer. Given a project description, produce JSON only with these keys:
{{
  "pitch": "<2-4 sentence marketing & technical pitch summary suitable for a slide or email>",
  "elevator": "<one-sentence concise hook>",
  "bullet_points": ["<3 short bullets summarizing benefits/impact/ask>"]
}}

Constraints:
- No extra commentary. Only valid JSON object.
- Keep language clear and suitable for investors / partners; emphasize SDG impact.
Project:
\"\"\"{project_text}\"\"\"
"""
    response_text = call_gemini(prompt, model=model, temperature=0.25, max_output_tokens=360)
    parsed = _parse_json_from_text(response_text)
    if isinstance(parsed, dict) and "pitch" in parsed:
        return parsed
    else:
        return {"pitch": "", "elevator": "", "bullet_points": [], "raw": parsed}


def reverse_ideas(selected_sdg: int,
                                    model: str = DEFAULT_MODEL,
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Given an SDG number (1-17) and optional contextual constraints, ask Gemini to return
        3-5 project ideas that align with the SDG and the provided context.

        `context` may include keys such as:
            - sector (e.g., 'software', 'civil', 'agriculture')
            - region (e.g., 'East Africa', 'India')
            - beneficiaries (e.g., 'smallholder farmers', 'urban youth')
            - budget (e.g., '<$50k', '$50k-$200k', '>$200k')
            - technologies (e.g., 'AI, IoT, solar')
            - constraints (free-text constraints or priorities)

        Returns a dict similar to before but with ideas tailored to the context.
        """
        ctx = context or {}
        # Build a short context description for the prompt
        ctx_lines = []
        if ctx.get("sector"):
                ctx_lines.append(f"Sector: {ctx.get('sector')}")
        if ctx.get("region"):
                ctx_lines.append(f"Region: {ctx.get('region')}")
        if ctx.get("beneficiaries"):
                ctx_lines.append(f"Beneficiaries: {ctx.get('beneficiaries')}")
        if ctx.get("budget"):
                ctx_lines.append(f"Budget: {ctx.get('budget')}")
        if ctx.get("technologies"):
                ctx_lines.append(f"Technologies: {ctx.get('technologies')}")
        if ctx.get("constraints"):
                ctx_lines.append(f"Constraints: {ctx.get('constraints')}")

        context_text = "\n".join(ctx_lines) if ctx_lines else "None"

        prompt = f"""
You are EcoMind AI, an ideation engine for Sustainable Development Goals (SDGs).
Given the SDG number and the project context, return JSON only with the structure:
{{
    "sdg": {selected_sdg},
    "sdg_name": "<short sdg name>",
    "ideas": [
        {{"title": "<short idea title>", "description": "<1-2 sentence description>", "why_it_fits": "<short reasoning>", "key_steps": ["..."], "estimated_budget": "<budget range>", "improvement_suggestions": ["<short suggestion 1>"]}}
    ]
}}

Context (use these to tailor ideas):
{context_text}

Requirements:
- Return 3 to 5 distinct, realistic, and actionable project ideas aligned to SDG #{selected_sdg} and the context above.
- Each description must be 1-2 sentences describing scope, impact, and one practical implementation note.
- Include a short `why_it_fits` and 2-3 `key_steps` for implementation, plus an `estimated_budget` suggestion.
- Output ONLY valid JSON (no markdown, no commentary).
"""

        response_text = call_gemini(prompt, model=model, temperature=0.6, max_output_tokens=700)
        parsed = _parse_json_from_text(response_text)
        if isinstance(parsed, dict) and "ideas" in parsed:
            return parsed
        else:
            return {"sdg": selected_sdg, "sdg_name": "", "ideas": [], "raw": parsed}


def reverse_ideas_multi(selected_sdgs: List[int],
                        model: str = DEFAULT_MODEL,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Given a list of SDG numbers, ask Gemini to propose 3-5 project ideas that intentionally
    address all of the provided SDGs (i.e., cross-cutting projects). Returns a dict:

    {
        "covered_sdgs": [1,5,13],
        "ideas": [ {title, description, why_it_fits, key_steps, estimated_budget, improvement_suggestions, covered_sdgs}, ... ]
    }

    Each idea MUST explicitly state which of the selected SDGs it covers (use the key
    `covered_sdgs`) and the prompt enforces that every returned idea addresses ALL
    selected SDGs so judges can evaluate multi-goal solutions.
    """
    ctx = context or {}
    ctx_lines = []
    if ctx.get("sector"):
        ctx_lines.append(f"Sector: {ctx.get('sector')}")
    if ctx.get("region"):
        ctx_lines.append(f"Region: {ctx.get('region')}")
    if ctx.get("beneficiaries"):
        ctx_lines.append(f"Beneficiaries: {ctx.get('beneficiaries')}")
    if ctx.get("budget"):
        ctx_lines.append(f"Budget: {ctx.get('budget')}")
    if ctx.get("technologies"):
        ctx_lines.append(f"Technologies: {ctx.get('technologies')}")
    if ctx.get("constraints"):
        ctx_lines.append(f"Constraints: {ctx.get('constraints')}")

    context_text = "\n".join(ctx_lines) if ctx_lines else "None"
    sdg_list_str = ", ".join(str(s) for s in selected_sdgs)

    prompt = f"""
You are EcoMind AI, an ideation engine for Sustainable Development Goals (SDGs).
Given the list of SDG numbers below and an optional project context, propose 3 to 5
distinct, realistic, and actionable project ideas that ADDRESS ALL of the listed SDGs.

Return ONLY valid JSON with this structure:
{{
  "covered_sdgs": [{sdg_list_str}],
  "ideas": [
    {{
      "title": "<short idea title>",
      "description": "<1-2 sentence description>",
      "why_it_fits": "<short explanation why this idea advances each listed SDG>",
      "key_steps": ["step 1", "step 2"],
      "estimated_budget": "<budget range>",
      "improvement_suggestions": ["..."],
      "covered_sdgs": [{sdg_list_str}]
    }}
  ]
}}

Context (use to tailor ideas):
{context_text}

Requirements:
- Each returned idea MUST address ALL of the SDG numbers: {sdg_list_str}.
- For `why_it_fits` explicitly mention how the idea advances each SDG by number/name.
- Provide 2-4 `key_steps` and an `estimated_budget` for implementation.
- Output only the JSON object (no markdown or extra text).
"""

    response_text = call_gemini(prompt, model=model, temperature=0.65, max_output_tokens=900)
    parsed = _parse_json_from_text(response_text)
    if isinstance(parsed, dict) and parsed.get("ideas"):
        # Defensive: ensure top-level covered_sdgs is present
        parsed.setdefault("covered_sdgs", selected_sdgs)
        # Ensure each idea has covered_sdgs
        for idea in parsed.get("ideas", []):
            idea.setdefault("covered_sdgs", selected_sdgs)
        return parsed
    else:
        return {"covered_sdgs": selected_sdgs, "ideas": [], "raw": parsed}


def suggest_improvements(project_text: str, sdg_id: int, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Given a project description and an SDG id, return focused, prioritized suggestions
    to improve alignment with that SDG, reduce risks, and increase feasibility.

    Returns a dict: {"sdg": sdg_id, "suggestions": ["..."], "notes": "..."}
    """
    prompt = f"""
You are EcoMind AI, an expert sustainability advisor. Given a short project description and a target SDG number, provide a concise, prioritized list of up to 6 actionable recommendations that would increase the project's alignment to SDG #{sdg_id}, reduce relevant risks, and improve feasibility in practical terms.

Return ONLY valid JSON with keys:
{{
  "sdg": {sdg_id},
  "suggestions": ["<actionable suggestion 1>", "<actionable suggestion 2>"],
  "notes": "<short caveats or context>"
}}

Project:
{project_text}
"""

    response_text = call_gemini(prompt, model=model, temperature=0.2, max_output_tokens=400)
    parsed = _parse_json_from_text(response_text)
    if isinstance(parsed, dict) and "suggestions" in parsed:
        return parsed
    else:
        return {"sdg": sdg_id, "suggestions": [], "notes": "Parser fallback", "raw": parsed}
