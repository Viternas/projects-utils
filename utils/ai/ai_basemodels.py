from pydantic import BaseModel
from typing import Optional, Dict, Any


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int = 0
    audio_tokens: int = 0
    reasoning_tokens: int = 0
    rejected_prediction_tokens: int = 0

class PromptTokensDetails(BaseModel):
    audio_tokens: int = 0
    cached_tokens: int = 0

class CompletionUsage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Optional[CompletionTokensDetails]
    prompt_tokens_details: Optional[PromptTokensDetails]


def convert_ollama_to_openai_usage(ollama_output: Dict[str, Any]) -> CompletionUsage:
    """
    Convert ollama output format to OpenAI's CompletionUsage format.

    Args:
        ollama_output: Dictionary containing the ollama output

    Returns:
        CompletionUsage object following the OpenAI standard
    """
    prompt_tokens = ollama_output.get("prompt_eval_count", 0)
    completion_tokens = ollama_output.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    completion_tokens_details = CompletionTokensDetails(
        accepted_prediction_tokens=completion_tokens,
        audio_tokens=0,
        reasoning_tokens=0,
        rejected_prediction_tokens=0
    )

    prompt_tokens_details = PromptTokensDetails(
        audio_tokens=0,
        cached_tokens=0
    )

    return CompletionUsage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        completion_tokens_details=completion_tokens_details,
        prompt_tokens_details=prompt_tokens_details
    )
