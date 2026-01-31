"""Google Gemini VLM Provider."""

import json
import os
import re
import time
import base64
from pathlib import Path

from typing import Optional, Any
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
    
    def call(self, image_path: Path, system_prompt: str, assertion: str,
             params: Optional[dict] = None) -> VLMResponse:
        """Call Gemini Vision API.
        
        Args:
            params: Dictionary of parameters like:
                - temperature (float)
                - max_tokens (int)
                - logprobs (bool)
                - top_logprobs (int)
                - output_format (str): 'abc' or 'json'
        """
        params = params or {}
        image_part = self._load_image(image_path)
        
        # Extract parameters with defaults
        temperature = params.get("temperature", 0.0)
        max_tokens = params.get("max_tokens", 500)
        use_logprobs = params.get("logprobs", False)
        top_logprobs = params.get("top_logprobs", 5 if use_logprobs else None)
        output_format = params.get("output_format", "json")
        
        # Combine system prompt with user message
        full_prompt = f"""{system_prompt}

Assertion: {assertion}"""
        
        try:
            start_time = time.time()
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if use_logprobs:
                generation_config["response_logprobs"] = True
                generation_config["logprobs"] = top_logprobs

            response = self.client.generate_content(
                [
                    {"inline_data": image_part},
                    full_prompt
                ],
                generation_config=generation_config
            )
            latency_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
        
        raw_text = response.text
        
        # Get usage metadata if available
        usage_metadata = getattr(response, 'usage_metadata', None)
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
        
        # Parse result based on output_format
        if output_format == "abc":
            # Handle A/B/C scoring format
            raw_upper = raw_text.strip().upper() if raw_text else ""
            if raw_upper == "A":
                result = "PASS"
            elif raw_upper == "B":
                result = "FAIL"
            elif raw_upper == "C":
                result = "UNCLEAR"
            else:
                result = None
            confidence, evidence, reasoning = None, None, None
        else:
            # Parse response (JSON track)
            result, confidence, evidence, reasoning = self._parse_response(raw_text)
        
        # Extract logprobs-based confidence
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
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
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
        """Extract probability-based confidence from Gemini logprobs."""
        import math
        try:
            # Gemini SDK response structure for logprobs
            candidate = response.candidates[0]
            if not hasattr(candidate, 'logprobs_result') or candidate.logprobs_result is None:
                return None, None
            
            top_candidates = candidate.logprobs_result.top_candidates
            if not top_candidates:
                return None, None
            
            # Look at the first token's logprobs
            token_info = top_candidates[0]
            
            prob_dist = {"p_pass": 0.0, "p_fail": 0.0, "p_unclear": 0.0}
            
            # Map A/B/C or PASS/FAIL
            for cand in token_info.candidates:
                tok = cand.token.upper().strip()
                prob = math.exp(cand.logprob)
                
                # Scoring track (A/B/C)
                if tok == "A": prob_dist["p_pass"] = prob
                elif tok == "B": prob_dist["p_fail"] = prob
                elif tok == "C": prob_dist["p_unclear"] = prob
                
                # Direct labels
                elif tok == "PASS": prob_dist["p_pass"] = prob
                elif tok == "FAIL": prob_dist["p_fail"] = prob
                elif tok == "UNCLEAR": prob_dist["p_unclear"] = prob

            # Confidence is the probability of the most likely token
            main_cand = token_info.candidates[0]
            confidence = int(math.exp(main_cand.logprob) * 100)
            
            return min(confidence, 100), prob_dist
        except Exception:
            return None, None
