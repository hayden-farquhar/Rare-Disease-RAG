"""LLM diagnostic generation with retrieved context."""

import json
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class DiagnosisCandidate(BaseModel):
    rank: int
    disease_name: str
    orpha_code: Optional[str] = None
    omim_id: Optional[str] = None
    confidence: str  # high/medium/low
    supporting_features: list[str] = []
    against_features: list[str] = []
    evidence_sources: list[str] = []
    reasoning: str = ""


class DiagnosticOutput(BaseModel):
    clinical_summary: str = ""
    phenotype_analysis: dict = {}
    differential_diagnosis: list[DiagnosisCandidate] = []
    recommended_investigations: list[dict] = []
    diagnostic_uncertainty: str = ""
    additional_history_needed: list[str] = []
    raw_response: str = ""


def generate_diagnosis(
    vignette: str,
    context: str,
    client=None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> DiagnosticOutput:
    """Generate a diagnostic assessment using retrieved context.

    Args:
        vignette: Clinical vignette text.
        context: Assembled retrieved evidence.
        client: Anthropic client (created if None).
        model: Model to use for generation.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    prompt_template = (PROMPT_DIR / "diagnostic_reasoning.md").read_text()
    prompt = prompt_template.replace("{vignette}", vignette).replace("{context}", context)

    response = _call_with_retry(client, model, max_tokens, temperature, prompt)

    raw_text = response.content[0].text.strip()
    return _parse_diagnostic_output(raw_text)


def _call_with_retry(client, model, max_tokens, temperature, prompt, max_retries=5):
    """Call API with retry on overloaded errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            err_str = str(e).lower()
            if "overloaded" in err_str or "529" in str(e) or "500" in str(e) or "internal server" in err_str or "server error" in err_str:
                wait = 10 * (attempt + 1)
                print(f"    API overloaded, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    # Final attempt without catch
    return client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )


def generate_diagnosis_no_rag(
    vignette: str,
    client=None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> DiagnosticOutput:
    """Generate diagnosis WITHOUT retrieved context (baseline comparison)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    prompt = (
        "You are an expert clinical geneticist and diagnostician specialising in rare diseases.\n\n"
        "Analyse the following clinical vignette and provide a ranked differential diagnosis.\n"
        "For each candidate, explain supporting and opposing features, your confidence level, "
        "and suggested next investigations.\n\n"
        f"## Clinical Vignette\n{vignette}\n\n"
        "Respond with a JSON object containing:\n"
        '{\n'
        '  "clinical_summary": "...",\n'
        '  "differential_diagnosis": [\n'
        '    {"rank": 1, "disease_name": "...", "confidence": "high/medium/low", '
        '"supporting_features": [...], "against_features": [...], "reasoning": "..."}\n'
        '  ],\n'
        '  "recommended_investigations": [{"test": "...", "rationale": "..."}],\n'
        '  "diagnostic_uncertainty": "..."\n'
        '}'
    )

    response = _call_with_retry(client, model, max_tokens, temperature, prompt)

    raw_text = response.content[0].text.strip()
    return _parse_diagnostic_output(raw_text)


def _parse_diagnostic_output(raw_text: str) -> DiagnosticOutput:
    """Parse LLM JSON response into structured output."""
    output = DiagnosticOutput(raw_response=raw_text)

    try:
        # Find JSON in response
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start < 0 or end <= start:
            return output

        json_str = raw_text[start:end]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common issues: trailing commas, unescaped newlines
            import re
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Last resort: extract just the differential_diagnosis array
                dd_match = re.search(r'"differential_diagnosis"\s*:\s*\[(.+?)\]', json_str, re.DOTALL)
                if dd_match:
                    # Try to parse individual disease entries
                    data = {"differential_diagnosis": []}
                    for m in re.finditer(r'\{[^{}]+\}', dd_match.group(1)):
                        try:
                            entry = json.loads(m.group())
                            data["differential_diagnosis"].append(entry)
                        except json.JSONDecodeError:
                            continue
                    if not data["differential_diagnosis"]:
                        return output
                else:
                    return output

        output.clinical_summary = data.get("clinical_summary", "")
        output.phenotype_analysis = data.get("phenotype_analysis", {})
        output.diagnostic_uncertainty = data.get("diagnostic_uncertainty", "")
        output.additional_history_needed = data.get("additional_history_needed", [])
        output.recommended_investigations = data.get("recommended_investigations", [])

        for i, dx in enumerate(data.get("differential_diagnosis", [])):
            try:
                rank_val = dx.get("rank", i + 1)
                if not isinstance(rank_val, int):
                    rank_val = i + 1
                output.differential_diagnosis.append(DiagnosisCandidate(
                    rank=rank_val,
                    disease_name=dx.get("disease_name", ""),
                    orpha_code=dx.get("orpha_code"),
                    omim_id=dx.get("omim_id"),
                    confidence=dx.get("confidence", "low"),
                    supporting_features=dx.get("supporting_features", []),
                    against_features=dx.get("against_features", []),
                    evidence_sources=dx.get("evidence_sources", []),
                    reasoning=dx.get("reasoning", ""),
                ))
            except Exception:
                continue

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Could not parse diagnostic output: {e}")

    return output
