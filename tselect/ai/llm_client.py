"""
tselect/ai/llm_client.py
------------------------
Groq API client using the official groq package.
Reads config from tselect.yaml:
    ai:
      groq_api_key: gsk_xxxx
      model: llama-3.3-70b-versatile   # optional
      timeout: 15                      # optional
"""
from groq import Groq
from tselect.utils.logger import setup_logger
logger = setup_logger()
DEFAULT_MODEL   = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT = 15
class LLMClientError(Exception):
    pass
class LLMClient:
    def __init__(self, config: dict):
        ai_cfg       = config.get("ai", {})
        # support both new (groq_api_key) and old (api_key) field names
        self.api_key = ai_cfg.get("groq_api_key") or ai_cfg.get("api_key", "")
        self.model   = ai_cfg.get("model", DEFAULT_MODEL)
        self.timeout = ai_cfg.get("timeout", DEFAULT_TIMEOUT)
        if not self.api_key:
            raise LLMClientError(
                "\n  ❌ Groq API key not found.\n"
                "  Add to tselect.yaml:\n"
                "      ai:\n"
                "        groq_api_key: gsk_xxxx\n"
                "  Get a free key at: https://console.groq.com\n"
            )
        self.client = Groq(api_key=self.api_key)
    def complete(self, prompt: str) -> str:
        """
        Send prompt to Groq, return response string.
        Raises LLMClientError on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.1,
                max_tokens  = 512,
                timeout     = self.timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMClientError(f"Groq error: {e}") from e
    def safe_complete(self, prompt: str) -> str | None:
        """
        Like complete() but returns None instead of raising.
        Use when you want graceful fallback.
        """
        try:
            return self.complete(prompt)
        except LLMClientError as e:
            logger.warning(f"LLM call failed (skipping AI filter): {e}")
            return None