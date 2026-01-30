"""Google Gemini VLM Provider."""

import json
import os
import re
import time
import base64
from pathlib import Path

from .base import VLMProvider, VLMResponse


class GeminiProvider(VLMProvider):
    """Google Gemini Vision provider."""
    
    name = "gemini"
    
    def __init__(self, model: str = None, **kwargs):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        # Import and configure
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        genai.configure(api_key=self.api_key)
        self.genai = genai
        self.client = genai.GenerativeModel(self.model)
    
    def _load_image(self, image_path: Path) -> dict:
        """Load image for Gemini API."""
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        return {"mime_type": mime_type, "data": image_data}
    
    def call(self, image_path: Path, system_prompt: str, assertion: str) -> VLMResponse:
        """Call Gemini Vision API."""
        image_part = self._load_image(image_path)
        
        # Combine system prompt with user message (Gemini doesn't have separate system role in older API)
        full_prompt = f"""{system_prompt}

Assertion: {assertion}"""
        
        try:
            start_time = time.time()
            response = self.client.generate_content(
                [
                    {"inline_data": image_part},
                    full_prompt
                ],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 500,
                }
            )
            latency_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
        
        raw_text = response.text
        
        # Get usage metadata if available
        usage_metadata = getattr(response, 'usage_metadata', None)
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
        
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
