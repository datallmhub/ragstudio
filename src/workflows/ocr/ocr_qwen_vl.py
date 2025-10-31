"""
Qwen-VL OCR Integration for ChunkForge Pipeline

This module provides integration with Qwen-VL models via Ollama API
for advanced OCR on complex documents with rich layouts, tables, and multilingual content.
"""

import json
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import base64
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing will be limited.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. PDF processing will be limited.")

from src.core.config.ocr_settings import get_ocr_settings

LOGGER = logging.getLogger(__name__)


class QwenVLConfig:
    """Configuration for Qwen-VL OCR processing via Ollama API."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        max_length: int = 16384,  # Increased from 4096 to allow full document extraction
        temperature: Optional[float] = None,
        batch_size: int = 1,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        # Load from global settings if not provided
        settings = get_ocr_settings()

        self.model_name = model_name or settings.qwen_vl.model_name
        self.api_base_url = (api_base_url or settings.qwen_vl.api_base_url).rstrip('/')
        self.max_length = max_length
        self.temperature = temperature if temperature is not None else settings.qwen_vl.temperature
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds or settings.qwen_vl.timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.qwen_vl.max_retries
        self.retry_delay = retry_delay if retry_delay is not None else settings.qwen_vl.retry_delay


class QwenVLProcessor:
    """
    Qwen-VL processor for advanced OCR on complex documents via Ollama API.
    
    Supports:
    - Multi-page PDF processing
    - Rich layout understanding
    - Table extraction
    - Multilingual text recognition
    - Structured output in JSON format
    - API-based processing (no local model loading)
    """
    
    def __init__(self, config: Optional[QwenVLConfig] = None):
        self.config = config or QwenVLConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL library is required for image processing")
        
        self._test_api_connection()
    
    def _test_api_connection(self):
        """Test connection to Ollama API and verify model availability."""
        try:
            self.logger.info(f"Testing connection to Ollama API: {self.config.api_base_url}")
            
            # Test API connection
            response = requests.get(
                f"{self.config.api_base_url}/v1/models",
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json()
            available_models = [model.get('id', '') for model in models.get('data', [])]
            
            self.logger.info(f"Available models: {available_models}")
            
            # Check if our model is available
            if self.config.model_name not in available_models:
                self.logger.warning(f"Model {self.config.model_name} not found in available models")
                self.logger.info("Available models: " + ", ".join(available_models))
                # Try to find a similar model
                for model in available_models:
                    if 'qwen' in model.lower() and 'vl' in model.lower():
                        self.logger.info(f"Found similar model: {model}")
                        self.config.model_name = model
                        break
            
            self.logger.info(f"Using model: {self.config.model_name}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama API: {e}")
            raise ConnectionError(f"Cannot connect to Ollama API at {self.config.api_base_url}")
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            raise
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process PDF document with Qwen-VL OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Processing PDF with Qwen-VL: {pdf_path}")
        
        start_time = time.time()
        results = {
            "source": str(pdf_path),
            "pages": [],
            "total_text": "",
            "processing_time": 0.0,
            "model_used": self.config.model_name,
            "api_base_url": self.config.api_base_url,
            "processing_method": "ollama_api",
            "errors": []
        }
        
        try:
            # Convert PDF to images
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError("pdf2image is required for PDF processing")
            
            images = convert_from_path(str(pdf_path))
            self.logger.info(f"Converted PDF to {len(images)} images")
            
            # Process each page
            page_texts = []
            for i, image in enumerate(images):
                try:
                    page_result = self._process_image(image, page_num=i+1)
                    results["pages"].append(page_result)
                    page_texts.append(page_result["text"])
                    
                except Exception as e:
                    error_msg = f"Failed to process page {i+1}: {e}"
                    self.logger.warning(error_msg)
                    results["errors"].append(error_msg)
                    
                    # Add empty page result
                    results["pages"].append({
                        "page": i+1,
                        "text": "",
                        "layout": "unknown",
                        "confidence": 0.0,
                        "error": str(e)
                    })
            
            # Combine all page texts
            results["total_text"] = "\n\n".join(page_texts)
            results["processing_time"] = time.time() - start_time
            
            self.logger.info(f"Qwen-VL processing completed: {len(results['total_text'])} chars in {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Qwen-VL PDF processing failed: {e}")
            results["errors"].append(f"Processing failed: {str(e)}")
            results["processing_time"] = time.time() - start_time
            raise
    
    def _process_image(self, image: Image.Image, page_num: int) -> Dict[str, any]:
        """
        Process a single image with Qwen-VL via Ollama API.
        
        Args:
            image: PIL Image object
            page_num: Page number for logging
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Prepare the prompt for OCR
            prompt = self._create_ocr_prompt()
            
            # Convert image to base64 for the API
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Prepare API request payload
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "max_tokens": self.config.max_length,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            # Make API request with retries
            response_text = self._make_api_request(payload, page_num)
            
            # Parse the response
            parsed_result = self._parse_response(response_text, page_num)
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Failed to process image for page {page_num}: {e}")
            return {
                "page": page_num,
                "text": "",
                "layout": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _make_api_request(self, payload: Dict[str, any], page_num: int) -> str:
        """
        Make API request to Ollama with retry logic.
        
        Args:
            payload: Request payload
            page_num: Page number for logging
            
        Returns:
            Response text from the API
        """
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Making API request for page {page_num} (attempt {attempt + 1})")
                
                response = requests.post(
                    f"{self.config.api_base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                    headers={"Content-Type": "application/json"}
                )

                # CRITICAL: Check status code explicitly
                if response.status_code != 200:
                    error_details = ""
                    try:
                        error_json = response.json()
                        error_details = f" - {error_json.get('error', {}).get('message', 'Unknown error')}"
                    except:
                        error_details = f" - {response.text[:200]}"

                    raise RuntimeError(
                        f"Qwen-VL API call failed with status {response.status_code}{error_details}"
                    )

                # Verify response is valid JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response from API: {e}")

                # Extract text from response
                if "choices" in result and len(result["choices"]) > 0:
                    response_text = result["choices"][0]["message"]["content"]

                    # Verify we got actual content
                    if not response_text or not response_text.strip():
                        raise ValueError("Empty response from Qwen-VL API")

                    self.logger.debug(f"API request successful for page {page_num}")
                    return response_text
                else:
                    raise ValueError(f"No choices in API response. Response keys: {list(result.keys())}")

            except requests.exceptions.Timeout as e:
                self.logger.warning(f"API request timeout for page {page_num} (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise RuntimeError(f"Qwen-VL API timeout after {self.config.max_retries} attempts") from e

            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"API connection error for page {page_num} (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Cannot connect to Qwen-VL API at {self.config.api_base_url}. Is Ollama running?") from e

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request failed for page {page_num} (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in API request for page {page_num}: {e}")
                raise
        
        raise RuntimeError(f"All {self.config.max_retries} API request attempts failed for page {page_num}")
    
    def _create_ocr_prompt(self) -> str:
        """Create the OCR prompt for Qwen-VL."""
        return """Extract ALL text from this document image. Read EVERYTHING visible on the page.

CRITICAL REQUIREMENTS:
1. Extract EVERY SINGLE word, number, and symbol - DO NOT skip anything
2. Preserve the original layout and structure exactly as shown
3. Handle French and English text with proper accents (é, è, à, ç, etc.)
4. Include ALL sections: headers, body text, tables, footers, signatures, fine print
5. Maintain table structure if present
6. DO NOT summarize, DO NOT skip sections, DO NOT truncate

Output format: Plain text only (no markdown, no code blocks, no explanations)

Start extracting now - read EVERYTHING:
"""
    
    def _parse_response(self, response: str, page_num: int) -> Dict[str, any]:
        """
        Parse Qwen-VL response and extract structured information.
        
        Args:
            response: Raw response from Qwen-VL
            page_num: Page number
            
        Returns:
            Structured result dictionary
        """
        try:
            # Clean the response
            cleaned_text = response.strip()
            
            # Determine layout type based on content analysis
            layout_type = self._detect_layout_type(cleaned_text)
            
            # Calculate confidence based on text quality indicators
            confidence = self._calculate_confidence(cleaned_text)
            
            result = {
                "page": page_num,
                "text": cleaned_text,
                "layout": layout_type,
                "confidence": confidence,
                "text_length": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
            }
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to parse response for page {page_num}: {e}")
            return {
                "page": page_num,
                "text": response,
                "layout": "unknown",
                "confidence": 0.5,
                "error": f"Parse error: {str(e)}"
            }
    
    def _detect_layout_type(self, text: str) -> str:
        """Detect the layout type based on text content."""
        if not text:
            return "empty"
        
        # Simple heuristics for layout detection
        lines = text.split('\n')
        
        # Check for table-like content
        if any('|' in line or '\t' in line for line in lines):
            return "table"
        
        # Check for list-like content
        if any(line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')) for line in lines):
            return "list"
        
        # Check for heading patterns
        if any(len(line.strip()) < 50 and line.strip().isupper() for line in lines):
            return "mixed"
        
        # Default to paragraph layout
        return "paragraph"
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality."""
        if not text:
            return 0.0
        
        # Simple confidence calculation based on text characteristics
        confidence_factors = []
        
        # Length factor (longer text = higher confidence)
        length_factor = min(1.0, len(text) / 1000)
        confidence_factors.append(length_factor * 0.3)
        
        # Character diversity factor
        unique_chars = len(set(text))
        char_diversity = unique_chars / max(1, len(text))
        confidence_factors.append(char_diversity * 0.2)
        
        # Word count factor
        word_count = len(text.split())
        word_factor = min(1.0, word_count / 100)
        confidence_factors.append(word_factor * 0.2)
        
        # Punctuation factor (proper punctuation = higher confidence)
        punct_count = sum(1 for c in text if c in '.,;:!?')
        punct_factor = min(1.0, punct_count / 50)
        confidence_factors.append(punct_factor * 0.3)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors)
        
        return min(1.0, overall_confidence)
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process a single image file with Qwen-VL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Process the image
            result = self._process_image(image, page_num=1)
            result["source"] = str(image_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process image {image_path}: {e}")
            raise


def create_qwen_vl_processor(
    config: Optional[QwenVLConfig] = None,
    api_base_url: str = "http://192.168.8.133:11434",
    model_name: str = "qwen2-vl:2b"
) -> QwenVLProcessor:
    """
    Create a Qwen-VL processor instance with Ollama API.
    
    Args:
        config: Optional configuration
        api_base_url: Ollama API base URL
        model_name: Model name to use
        
    Returns:
        QwenVLProcessor instance
    """
    if config is None:
        config = QwenVLConfig(
            model_name=model_name,
            api_base_url=api_base_url
        )
    return QwenVLProcessor(config)


def process_pdf_with_qwen_vl(
    pdf_path: Union[str, Path], 
    config: Optional[QwenVLConfig] = None,
    api_base_url: str = "http://192.168.8.133:11434",
    model_name: str = "qwen2-vl:2b"
) -> Dict[str, any]:
    """
    Convenience function to process PDF with Qwen-VL via Ollama API.
    
    Args:
        pdf_path: Path to the PDF file
        config: Optional configuration
        api_base_url: Ollama API base URL
        model_name: Model name to use
        
    Returns:
        Processing results
    """
    processor = create_qwen_vl_processor(config, api_base_url, model_name)
    return processor.process_pdf(pdf_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ocr_qwen_vl.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Create processor with Ollama API config
        config = QwenVLConfig(
            model_name="qwen2-vl:2b",
            api_base_url="http://192.168.8.133:11434"
        )
        processor = create_qwen_vl_processor(config)
        
        # Process PDF
        results = processor.process_pdf(pdf_path)
        
        print(f"PDF processed: {results['source']}")
        print(f"Pages: {len(results['pages'])}")
        print(f"Total text length: {len(results['total_text'])} chars")
        print(f"Processing time: {results['processing_time']:.2f}s")
        print(f"Model: {results['model_used']}")
        print(f"API URL: {results['api_base_url']}")
        print(f"Processing method: {results['processing_method']}")
        
        # Show sample text
        if results['total_text']:
            print(f"\nSample text (first 500 chars):")
            print(results['total_text'][:500])
        
        # Save results
        output_path = Path(pdf_path).with_suffix('.qwen_vl.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)
