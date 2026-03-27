"""Thin wrapper around the Anthropic Messages API for non-streaming completions."""

from anthropic import Anthropic

# Large enough for JSON schema / batched rows; API will cap per model.
_MAX_OUTPUT_TOKENS = 16_384


class AnthropicCompletor:
    """
    Calls Claude with a system prompt and a user prompt, returns text only.

    The model id is passed per instance so it always matches the user's Gradio input.
    """

    def __init__(self, model: str) -> None:
        """
        Create a client for the given model.

        Args:
            model: Anthropic model ID selected in the UI.
        """
        m = (model or "").strip()
        if not m:
            raise ValueError("model must be a non-empty Anthropic model id.")
        self._model = m
        self._client = Anthropic()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Run a single non-streaming request and return assistant text.

        Args:
            system_prompt: Rules (e.g. JSON-only output, schema rules).
            user_prompt: Task content (dataset description, schema, batch instructions).

        Returns:
            Raw assistant string to parse as JSON (possibly inside markdown fences).

        Raises:
            API errors from the ``anthropic`` SDK on failure or rate limits.
        """
        message = self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Anthropic API returns a list of blocks, we only want the text blocks.
        chunks: list[str] = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks)