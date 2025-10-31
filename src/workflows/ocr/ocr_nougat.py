"""
Nougat OCR Processor for Scientific Documents.

This module integrates Meta AI's Nougat model for OCR processing
of scientific and mathematical documents.
"""

import logging
import time
import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import requests

try:
    import torch
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    import PIL.Image
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
    logging.warning("Nougat dependencies not available. Install with: pip install transformers torch pillow")

from src.core.config.ocr_settings import get_ocr_settings

LOGGER = logging.getLogger(__name__)


class NougatConfig:
    """Configuration for Nougat OCR processing."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 4096,
        min_length: int = 0,
        num_beams: Optional[int] = None,
        early_stopping: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        batch_size: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        # Load from global settings if not provided
        settings = get_ocr_settings()

        self.model_name = model_name or settings.nougat.model_name
        self.device = device or settings.nougat.device
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams if num_beams is not None else settings.nougat.beam_size
        self.early_stopping = early_stopping
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size if batch_size is not None else settings.nougat.batch_size
        self.timeout_seconds = timeout_seconds or settings.nougat.timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay


class NougatOCRProcessor:
    """
    Nougat OCR processor for scientific documents.
    
    This class handles the integration with Meta AI's Nougat model
    for processing scientific and mathematical documents.
    """
    
    def __init__(self, config: Optional[NougatConfig] = None):
        """
        Initialize the Nougat processor.
        
        Args:
            config: Configuration for Nougat processing
        """
        self.config = config or NougatConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not NOUGAT_AVAILABLE:
            raise ImportError(
                "Nougat dependencies not available. "
                "Install with: pip install transformers torch pillow"
            )
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Nougat model and processor."""
        try:
            self.logger.info(f"Loading Nougat model: {self.config.model_name}")
            
            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.logger.info("Using CUDA for Nougat processing")
                else:
                    self.device = "cpu"
                    self.logger.info("Using CPU for Nougat processing")
            else:
                self.device = self.config.device
            
            # Load processor and model (model name from config, user-controlled)
            self.processor = NougatProcessor.from_pretrained(self.config.model_name)  # nosec B615
            self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)  # nosec B615
            self.model.to(self.device)
            
            self.logger.info("Nougat model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Nougat model: {e}")
            raise
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF document using Nougat OCR.
        
        Args:
            pdf_path: Path to the PDF document
            
        Returns:
            Dictionary with processing results:
            {
                "total_text": str,
                "pages": List[Dict],
                "processing_time": float,
                "model_used": str,
                "processing_method": str,
                "success": bool,
                "metadata": Dict
            }
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Processing PDF with Nougat: {pdf_path}")
        
        start_time = time.time()
        
        try:
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path)
            
            if not images:
                raise ValueError("No images extracted from PDF")
            
            # Process each page
            pages = []
            total_text = ""
            
            for i, image in enumerate(images):
                self.logger.info(f"Processing page {i+1}/{len(images)}")
                
                page_result = self._process_image(image, page_num=i+1)
                pages.append(page_result)
                total_text += page_result["text"] + "\n\n"
            
            processing_time = time.time() - start_time
            
            result = {
                "total_text": total_text.strip(),
                "pages": pages,
                "processing_time": processing_time,
                "model_used": self.config.model_name,
                "processing_method": "nougat_ocr",
                "success": True,
                "metadata": {
                    "source": str(pdf_path),
                    "pages_processed": len(pages),
                    "total_text_length": len(total_text),
                    "avg_confidence": sum(p.get("confidence", 0) for p in pages) / max(1, len(pages)),
                    "device_used": self.device
                }
            }
            
            self.logger.info(
                f"Nougat processing completed: {len(pages)} pages, "
                f"{len(total_text)} chars, {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Nougat processing failed: {e}")
            
            return {
                "total_text": "",
                "pages": [],
                "processing_time": processing_time,
                "model_used": self.config.model_name,
                "processing_method": "nougat_ocr",
                "success": False,
                "metadata": {
                    "source": str(pdf_path),
                    "error": str(e),
                    "device_used": self.device
                }
            }
    
    def _pdf_to_images(self, pdf_path: Path) -> List[PIL.Image.Image]:
        """Convert PDF pages to PIL images."""
        try:
            from pdf2image import convert_from_path

            # Get DPI from settings
            settings = get_ocr_settings()
            dpi = settings.nougat.dpi

            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt='RGB'
            )

            self.logger.info(f"Converted PDF to {len(images)} images (DPI: {dpi})")
            return images

        except Exception as e:
            self.logger.error(f"Failed to convert PDF to images: {e}")
            raise
    
    def _process_image(self, image: PIL.Image.Image, page_num: int) -> Dict[str, Any]:
        """
        Process a single image using Nougat.
        
        Args:
            image: PIL Image to process
            page_num: Page number for logging
            
        Returns:
            Dictionary with page processing results
        """
        try:
            # Prepare image for Nougat
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text using Nougat
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.config.max_length,
                    min_length=self.config.min_length,
                    num_beams=self.config.num_beams,
                    early_stopping=self.config.early_stopping,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Clean and format the text
            cleaned_text = self._clean_nougat_output(generated_text)
            
            return {
                "page": page_num,
                "text": cleaned_text,
                "confidence": 0.95,  # Nougat doesn't provide confidence scores
                "layout": "scientific",
                "processing_time": 0.0  # Individual page time not tracked
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process page {page_num}: {e}")
            return {
                "page": page_num,
                "text": "",
                "confidence": 0.0,
                "layout": "error",
                "error": str(e)
            }
    
    def _clean_nougat_output(self, text: str) -> str:
        """
        Clean and format Nougat output text.
        
        Args:
            text: Raw text from Nougat
            
        Returns:
            Cleaned and formatted text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Fix common Nougat formatting issues
        cleaned = cleaned.replace("\\n", "\n")
        cleaned = cleaned.replace("\\t", "\t")
        
        # Ensure proper line breaks for equations
        cleaned = re.sub(r'(\$[^$]+\$)', r'\n\1\n', cleaned)
        
        # Clean up multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()


def create_nougat_processor(config: Optional[NougatConfig] = None) -> NougatOCRProcessor:
    """
    Create a Nougat processor instance.
    
    Args:
        config: Configuration for Nougat processing
        
    Returns:
        NougatProcessor instance
    """
    return NougatOCRProcessor(config)


def process_pdf_with_nougat(pdf_path: Union[str, Path], config: Optional[NougatConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to process a PDF with Nougat.
    
    Args:
        pdf_path: Path to the PDF document
        config: Configuration for Nougat processing
        
    Returns:
        Dictionary with processing results
    """
    processor = create_nougat_processor(config)
    return processor.process_pdf(pdf_path)


# Example usage
if __name__ == "__main__":
    # Test on math.pdf
    pdf_path = Path("tests/data/math.pdf")
    
    if pdf_path.exists():
        print("Testing Nougat OCR on math.pdf...")
        
        config = NougatConfig(
            model_name="facebook/nougat-base",
            device="auto"
        )
        
        result = process_pdf_with_nougat(pdf_path, config)
        
        print(f"Processing completed: {result['success']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Pages processed: {len(result['pages'])}")
        print(f"Text length: {len(result['total_text'])} chars")
        
        if result['success']:
            print("\nSample text:")
            print(result['total_text'][:500])
    else:
        print(f"Test file {pdf_path} not found")
