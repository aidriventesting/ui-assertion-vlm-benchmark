"""OpenAI VLM Provider (GPT-4 Vision)."""

import json
import os
import re
import time
import base64
from pathlib import Path

import openai
from typing import Optional, Any

from .base import VLMProvider, VLMResponse


class OpenAIProvider(VLMProvider):
    """OpenAI GPT-4 Vision provider."""
    
    name = "openai"
    
    def __init__(self, model: str = None, image_cache: dict = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
    
    def call(self, image_path: Path, system_prompt: str, assertion: str,
             params: Optional[dict] = None) -> VLMResponse:
        """Call OpenAI Vision API.
        
        Args:
            params: Dictionary of parameters like:
                - temperature (float)
                - max_tokens (int)
                - logprobs (bool)
                - top_logprobs (int)
                - output_format (str): 'abc' or 'json'
        """
        params = params or {}
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
        
        # Extract parameters with defaults
        temperature = params.get("temperature", 0.0)
        max_tokens = params.get("max_tokens", 500)
        use_logprobs = params.get("logprobs", False)
        top_logprobs = params.get("top_logprobs", 5 if use_logprobs else None)
        output_format = params.get("output_format", "json")
        
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if use_logprobs:
                    api_params["logprobs"] = True
                    api_params["top_logprobs"] = top_logprobs
                
                response = self.client.chat.completions.create(**api_params)
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
        
        # Parse result based on output_format
        if output_format == "abc":
            # Handle A/B/C scoring format → map to PASS/FAIL/UNCLEAR
            raw_upper = raw_text.strip().upper() if raw_text else ""
            if raw_upper == "A":
                result = "PASS"
            elif raw_upper == "B":
                result = "FAIL"
            elif raw_upper == "C":
                result = "UNCLEAR"
            else:
                result = None
            # These are typically not in ABC mode
            confidence, evidence, reasoning = None, None, None
        else:
            # Parse response (handles JSON and plain text)
            result, confidence, evidence, reasoning = self._parse_response(raw_text)
        
        # Extract logprobs-based confidence (more reliable than verbalized)
        logprob_confidence, logprobs_dist = self._extract_logprob_confidence(response)
        if logprob_confidence is not None:
            confidence = logprob_confidence
        
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
            },
            logprobs=logprobs_dist
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
            if "UNCLEAR" in text_upper:
                result = "UNCLEAR"
            elif "PASS" in text_upper:
                result = "PASS"
            elif "FAIL" in text_upper:
                result = "FAIL"
            elif "TRUE" in text_upper:
                result = "PASS"
            elif "FALSE" in text_upper:
                result = "FAIL"
        
        return result, confidence, evidence, reasoning
    
    def _extract_logprob_confidence(self, response) -> tuple[int | None, dict | None]:
        """Extract probability-based confidence from OpenAI logprobs.
        
        Handles both formats:
        - A/B/C scoring format (A=PASS, B=FAIL, C=UNCLEAR)
        - Direct PASS/FAIL format
        
        Returns:
            (confidence, logprobs_dict): confidence as 0-100, and full probability dist
        """
        import math
        
        try:
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or choice.logprobs is None:
                return None, None
            
            content = choice.logprobs.content
            if not content:
                return None, None
            
            # Look for A/B/C or PASS/FAIL in first few tokens
            for token_info in content[:5]:
                token_upper = token_info.token.upper().strip()
                
                # Build probability distribution from top_logprobs
                prob_dist = {"p_pass": 0.0, "p_fail": 0.0, "p_unclear": 0.0}
                
                # A/B/C format (for scoring track)
                if token_upper in ("A", "B", "C"):
                    # Collect all alternatives
                    all_tokens = [(token_info.token.upper().strip(), token_info.logprob)]
                    if token_info.top_logprobs:
                        for alt in token_info.top_logprobs:
                            all_tokens.append((alt.token.upper().strip(), alt.logprob))
                    
                    for tok, logprob in all_tokens:
                        prob = math.exp(logprob)
                        if tok == "A":
                            prob_dist["p_pass"] = prob
                        elif tok == "B":
                            prob_dist["p_fail"] = prob
                        elif tok == "C":
                            prob_dist["p_unclear"] = prob
                    
                    # Confidence = probability of chosen token
                    main_prob = math.exp(token_info.logprob)
                    confidence = int(main_prob * 100)
                    return min(confidence, 100), prob_dist
                
                # PASS/FAIL format (for JSON track)
                if token_upper in ("PASS", "FAIL", "UNCLEAR"):
                    all_tokens = [(token_info.token.upper().strip(), token_info.logprob)]
                    if token_info.top_logprobs:
                        for alt in token_info.top_logprobs:
                            all_tokens.append((alt.token.upper().strip(), alt.logprob))
                    
                    for tok, logprob in all_tokens:
                        prob = math.exp(logprob)
                        if tok == "PASS":
                            prob_dist["p_pass"] = prob
                        elif tok == "FAIL":
                            prob_dist["p_fail"] = prob
                        elif tok == "UNCLEAR":
                            prob_dist["p_unclear"] = prob
                    
                    main_prob = math.exp(token_info.logprob)
                    confidence = int(main_prob * 100)
                    return min(confidence, 100), prob_dist
            
            return None, None
            
        except Exception:
            return None, None
