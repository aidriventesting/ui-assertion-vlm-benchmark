"""Abstract base class for VLM providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMResponse:
    """Standardized response from VLM providers."""
    result: Optional[str]  # PASS, FAIL, or None (abstain)
    confidence: Optional[int]  # 0-100
    evidence: Optional[str]
    reasoning: Optional[str]
    raw: str  # Raw response text
    cost: dict  # {input_tokens, output_tokens, latency_ms, api_calls}
    
    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "raw": self.raw,
            "cost": self.cost,
        }


class VLMProvider(ABC):
    """Abstract base class for VLM providers."""
    
    name: str = "base"
    model: str = ""
    
    @abstractmethod
    def call(self, image_path: Path, system_prompt: str, assertion: str) -> VLMResponse:
        """Call the VLM with an image and assertion.
        
        Args:
            image_path: Path to the screenshot image
            system_prompt: System prompt template
            assertion: The assertion to verify
            
        Returns:
            VLMResponse with result, confidence, evidence, etc.
        """
        pass
    
    def get_model_name(self) -> str:
        """Return the full model identifier for results tracking."""
        return f"{self.name}/{self.model}"
