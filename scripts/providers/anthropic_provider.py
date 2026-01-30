"""Anthropic Claude VLM Provider."""

import json
import os
import re
import time
import base64
from pathlib import Path

from .base import VLMProvider, VLMResponse


class AnthropicProvider(VLMProvider):
    """Anthropic Claude Vision provider."""
    
    name = "anthropic"
    
    def __init__(self, model: str = None, **kwargs):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _encode_image_base64(self, image_path: Path) -> tuple[str, str]:
        """Encode image as base64 and return with media type."""
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        
        suffix = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif"
        }
        media_type = media_types.get(suffix, "image/png")
        
        return base64_data, media_type
    
    def call(self, image_path: Path, system_prompt: str, assertion: str) -> VLMResponse:
        """Call Anthropic Claude Vision API."""
        base64_data, media_type = self._encode_image_base64(image_path)
        
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Assertion: {assertion}"
                            }
                        ]
                    }
                ]
            )
            latency_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
        
        raw_text = response.content[0].text
        
        # Get usage
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        
        # Parse response
        result, confidence, evidence, reasoning = self._parse_response(raw_text)
        
        return VLMResponse(
            result=result,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            raw=raw_text,
            cost={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "api_calls": 1
            }
        )
    
    def _parse_response(self, raw_text: str) -> tuple:
        """Parse VLM response to extract structured fields."""
        result = None
        confidence = None
        evidence = None
        reasoning = None
        
        # Try to parse JSON
        json_match = re.search(r'\{[^}]+\}', raw_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result = parsed.get("result")
                confidence = parsed.get("confidence")
                evidence = parsed.get("evidence")
                reasoning = parsed.get("reasoning")
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract PASS/FAIL from text
        if result is None:
            text_upper = raw_text.upper()
            if "PASS" in text_upper:
                result = "PASS"
            elif "FAIL" in text_upper:
                result = "FAIL"
            elif "TRUE" in text_upper:
                result = "PASS"
            elif "FALSE" in text_upper:
                result = "FAIL"
        
        return result, confidence, evidence, reasoning
