"""OpenAI VLM Provider (GPT-4 Vision)."""

import json
import os
import re
import time
import base64
from pathlib import Path

import openai

from .base import VLMProvider, VLMResponse


class OpenAIProvider(VLMProvider):
    """OpenAI GPT-4 Vision provider."""
    
    name = "openai"
    
    def __init__(self, model: str = None, image_cache: dict = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.image_cache = image_cache if image_cache is not None else {}
        
        # Try to import imgbb_uploader for image hosting
        try:
            from imgbb_uploader import get_image_for_api
            self.get_image_for_api = get_image_for_api
        except ImportError:
            self.get_image_for_api = None
    
    def _encode_image_base64(self, image_path: Path) -> dict:
        """Encode image as base64 data URL."""
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}
        }
    
    def _get_image_content(self, image_path: Path) -> tuple[dict, str]:
        """Get image content for API, using cache and ImgBB if available."""
        cache_key = str(image_path)
        
        if cache_key in self.image_cache:
            content = self.image_cache[cache_key].copy()
            method = content.pop("_method", "cached")
            return content, method
        
        # Try ImgBB first if available
        if self.get_image_for_api:
            content = self.get_image_for_api(image_path)
            method = content.pop("_method", "imgbb")
            self.image_cache[cache_key] = {**content, "_method": method}
            return content, method
        
        # Fallback to base64
        content = self._encode_image_base64(image_path)
        self.image_cache[cache_key] = {**content, "_method": "base64"}
        return content, "base64"
    
    def call(self, image_path: Path, system_prompt: str, assertion: str) -> VLMResponse:
        """Call OpenAI Vision API."""
        image_content, image_method = self._get_image_content(image_path)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": f"Assertion: {assertion}"}
                ]
            }
        ]
        
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.0
                )
                latency_ms = int((time.time() - start_time) * 1000)
                break
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # If ImgBB URL failed, retry with base64
                if attempt == 0 and image_method == "imgbb" and (
                    "timeout" in error_msg or "invalid_image_url" in error_msg
                ):
                    print(f"\n⚠️  ImgBB URL failed, retrying with base64...", end=" ")
                    image_content = self._encode_image_base64(image_path)
                    messages[1]["content"][0] = image_content
                    image_method = "base64_fallback"
                    continue
                else:
                    raise
        else:
            raise last_error
        
        raw_text = response.choices[0].message.content
        usage = response.usage
        
        # Parse response
        result, confidence, evidence, reasoning = self._parse_response(raw_text)
        
        return VLMResponse(
            result=result,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            raw=raw_text,
            cost={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
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
