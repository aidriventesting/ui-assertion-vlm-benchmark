"""
Image uploader utility for ImgBB.
Uploads images and returns URL for cheaper VLM API calls.
Falls back to base64 if upload fails.
"""

import os
import base64
import requests
from pathlib import Path
from typing import Optional


class ImgBBUploader:
    """Upload images to ImgBB and get public URLs."""
    
    BASE_URL = "https://api.imgbb.com/1/upload"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("IMGBB_API_KEY")
    
    @property
    def is_available(self) -> bool:
        """Check if uploader is configured."""
        return bool(self.api_key)
    
    def upload_from_file(self, file_path: str, expiration: Optional[int] = 600) -> Optional[str]:
        """
        Upload image file to ImgBB.
        
        Args:
            file_path: Path to image file
            expiration: Seconds until image expires (default 10 min, None = never)
        
        Returns:
            Image URL or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            with open(file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                'key': self.api_key,
                'image': image_data,
                'name': Path(file_path).stem
            }
            if expiration is not None:
                payload['expiration'] = str(expiration)
            
            response = requests.post(self.BASE_URL, data=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', {}).get('display_url')
            
        except Exception as e:
            print(f"⚠️  ImgBB upload failed: {e}")
            return None
    
    def upload_from_base64(self, base64_data: str, filename: str = "screenshot", 
                           expiration: Optional[int] = 600) -> Optional[str]:
        """
        Upload base64 image to ImgBB.
        
        Args:
            base64_data: Base64 encoded image
            filename: Name for the image
            expiration: Seconds until image expires
        
        Returns:
            Image URL or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            payload = {
                'key': self.api_key,
                'image': base64_data,
                'name': filename
            }
            if expiration is not None:
                payload['expiration'] = str(expiration)
            
            response = requests.post(self.BASE_URL, data=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', {}).get('display_url')
            
        except Exception as e:
            print(f"⚠️  ImgBB upload failed: {e}")
            return None


# Singleton instance
_uploader: Optional[ImgBBUploader] = None


def get_uploader() -> ImgBBUploader:
    """Get or create the uploader instance."""
    global _uploader
    if _uploader is None:
        _uploader = ImgBBUploader()
    return _uploader


def get_image_for_api(image_path: Path) -> dict:
    """
    Get image ready for VLM API call.
    Tries ImgBB upload first, falls back to base64.
    
    Returns:
        dict with 'type' and 'image_url' for API consumption
    """
    uploader = get_uploader()
    
    # Try ImgBB upload first (cheaper)
    if uploader.is_available:
        url = uploader.upload_from_file(str(image_path))
        if url:
            return {
                "type": "image_url",
                "image_url": {"url": url},
                "_method": "imgbb"
            }
        print(f"⚠️  Falling back to base64 for {image_path.name}")
    
    # Fallback to base64 (expensive but reliable)
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Detect mime type
    suffix = image_path.suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime_type = mime_types.get(suffix, "image/png")
    
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
        "_method": "base64"
    }
