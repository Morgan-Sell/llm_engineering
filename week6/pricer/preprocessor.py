import os
from functools import lru_cache

from anthropic import Anthropic, RateLimitError
from dotenv import load_dotenv

DEFAULT_MODEL_NAME = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 200
MODEL_PRICES_PER_MILLION = {
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


@lru_cache(maxsize=1)
def client() -> Anthropic:
    load_dotenv(override=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY before running preprocessing.")
    return Anthropic(api_key=api_key)


class Preprocessor:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, max_tokens=DEFAULT_MAX_TOKENS):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.max_tokens = max_tokens

    def messages_for(self, text: str) -> list[dict]:
        return [{"role": "user", "content": text}]

    @staticmethod
    def text_from_blocks(blocks) -> str:
        texts = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)
        return "\n".join(texts).strip()

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICES_PER_MILLION.get(self.model_name)
        if not pricing:
            return 0
        input_cost = input_tokens * pricing["input"] / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000
        return input_cost + output_cost

    def preprocess(self, text: str) -> str:
        messages = self.messages_for(text)
        try:
            response = client().messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "Claude rate limit exceeded. Retry later or use a smaller model/batch."
            ) from exc

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_cost += self.estimate_cost(
            response.usage.input_tokens, response.usage.output_tokens
        )
        return self.text_from_blocks(response.content)
