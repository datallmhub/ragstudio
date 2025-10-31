"""
Document Complexity Analyzer for ChunkForge OCR Pipeline

This module analyzes document complexity to determine the optimal OCR strategy:
- Simple documents → Classic OCR (Tesseract/docTR)
- Medium complexity → MiniCPM-V 2.7B
- High complexity → Qwen-VL-2B/7B
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Image analysis will be limited.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. PDF analysis will be limited.")

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("unstructured not available. Structure analysis will be limited.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. OCR confidence analysis will be limited.")

LOGGER = logging.getLogger(__name__)


class DocumentComplexityAnalyzer:
    """
    Analyzes document complexity to determine optimal OCR strategy.
    
    Complexity factors:
    - Text density and quality
    - Presence of images, tables, diagrams
    - Layout complexity
    - OCR confidence scores
    - Multilingual content
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_document(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Analyze document complexity and return comprehensive metrics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing complexity analysis results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Analyzing document complexity: {pdf_path}")
        
        analysis = {
            "source": str(pdf_path),
            "complexity_score": 0.0,
            "recommended_strategy": "classic_ocr",
            "metrics": {},
            "features": {},
            "warnings": []
        }
        
        try:
            # Step 1: Basic PDF analysis
            basic_metrics = self._analyze_basic_structure(pdf_path)
            analysis["metrics"].update(basic_metrics)
            
            # Step 2: Image analysis (if available)
            if PDF2IMAGE_AVAILABLE and OPENCV_AVAILABLE:
                image_metrics = self._analyze_images(pdf_path)
                analysis["metrics"].update(image_metrics)
            else:
                analysis["warnings"].append("Image analysis unavailable (missing dependencies)")
            
            # Step 3: Text quality analysis
            text_metrics = self._analyze_text_quality(pdf_path)
            analysis["metrics"].update(text_metrics)
            
            # Step 4: Layout complexity analysis
            layout_metrics = self._analyze_layout_complexity(pdf_path)
            analysis["metrics"].update(layout_metrics)
            
            # Step 5: Calculate overall complexity score
            complexity_score = self._calculate_complexity_score(analysis["metrics"])
            analysis["complexity_score"] = complexity_score
            
            # Step 6: Determine recommended strategy
            analysis["recommended_strategy"] = self._recommend_strategy(complexity_score)
            
            # Step 7: Extract features for ML routing
            analysis["features"] = self._extract_features(analysis["metrics"])
            
            self.logger.info(f"Complexity analysis completed: score={complexity_score:.3f}, strategy={analysis['recommended_strategy']}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            analysis["warnings"].append(f"Analysis error: {str(e)}")
            analysis["complexity_score"] = 0.5  # Default to medium complexity
            analysis["recommended_strategy"] = "minicpm_v"
            return analysis
    
    def _analyze_basic_structure(self, pdf_path: Path) -> Dict[str, any]:
        """Analyze basic PDF structure and metadata."""
        metrics = {}
        
        try:
            # Get PDF page count
            if PDF2IMAGE_AVAILABLE:
                images = convert_from_path(str(pdf_path), first_page=1, last_page=1)
                # Estimate page count by checking file size vs first page
                file_size = pdf_path.stat().st_size
                first_page_size = len(images[0].tobytes()) if images else 0
                estimated_pages = max(1, file_size // (first_page_size * 2)) if first_page_size > 0 else 1
                metrics["page_count"] = min(estimated_pages, 50)  # Cap at 50 for analysis
            else:
                metrics["page_count"] = 1
            
            # Analyze file size
            metrics["file_size_mb"] = pdf_path.stat().st_size / (1024 * 1024)
            
            # Check if PDF has text layer
            if UNSTRUCTURED_AVAILABLE:
                try:
                    elements = partition_pdf(str(pdf_path), strategy="fast")
                    text_elements = [e for e in elements if hasattr(e, 'text') and e.text.strip()]
                    metrics["has_text_layer"] = len(text_elements) > 0
                    metrics["text_elements_count"] = len(text_elements)
                except Exception:
                    metrics["has_text_layer"] = False
                    metrics["text_elements_count"] = 0
            else:
                metrics["has_text_layer"] = False
                metrics["text_elements_count"] = 0
            
        except Exception as e:
            self.logger.warning(f"Basic structure analysis failed: {e}")
            metrics["page_count"] = 1
            metrics["file_size_mb"] = 1.0
            metrics["has_text_layer"] = False
            metrics["text_elements_count"] = 0
        
        return metrics
    
    def _analyze_images(self, pdf_path: Path) -> Dict[str, any]:
        """Analyze images within the PDF."""
        metrics = {}
        
        try:
            # Convert first few pages to images for analysis
            max_pages = min(3, metrics.get("page_count", 3))
            images = convert_from_path(str(pdf_path), first_page=1, last_page=max_pages)
            
            total_pixels = 0
            image_regions = 0
            text_regions = 0
            table_regions = 0
            
            for img in images:
                # Convert PIL to OpenCV format
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                height, width = img_cv.shape[:2]
                total_pixels += height * width
                
                # Analyze image content
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Detect text regions using edge detection
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum area threshold
                        # Analyze contour shape to determine region type
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        if aspect_ratio > 2.0:  # Likely text line
                            text_regions += 1
                        elif 0.8 < aspect_ratio < 1.2:  # Likely table cell
                            table_regions += 1
                        else:
                            image_regions += 1
            
            metrics["image_regions"] = image_regions
            metrics["text_regions"] = text_regions
            metrics["table_regions"] = table_regions
            metrics["total_pixels"] = total_pixels
            metrics["image_density"] = image_regions / max(1, len(images))
            metrics["text_density"] = text_regions / max(1, len(images))
            metrics["table_density"] = table_regions / max(1, len(images))
            
        except Exception as e:
            self.logger.warning(f"Image analysis failed: {e}")
            metrics.update({
                "image_regions": 0,
                "text_regions": 0,
                "table_regions": 0,
                "total_pixels": 0,
                "image_density": 0.0,
                "text_density": 0.0,
                "table_density": 0.0
            })
        
        return metrics
    
    def _analyze_text_quality(self, pdf_path: Path) -> Dict[str, any]:
        """Analyze text quality and OCR confidence."""
        metrics = {}
        
        try:
            if TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE:
                # Convert first page to image for OCR analysis
                images = convert_from_path(str(pdf_path), first_page=1, last_page=1)
                if images:
                    # Get OCR data with confidence scores
                    ocr_data = pytesseract.image_to_data(images[0], output_type=pytesseract.Output.DICT)
                    
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    
                    if confidences:
                        metrics["avg_ocr_confidence"] = sum(confidences) / len(confidences)
                        metrics["min_ocr_confidence"] = min(confidences)
                        metrics["max_ocr_confidence"] = max(confidences)
                        metrics["low_confidence_ratio"] = len([c for c in confidences if c < 60]) / len(confidences)
                    else:
                        metrics["avg_ocr_confidence"] = 0.0
                        metrics["min_ocr_confidence"] = 0.0
                        metrics["max_ocr_confidence"] = 0.0
                        metrics["low_confidence_ratio"] = 1.0
                    
                    # Analyze text patterns
                    text = pytesseract.image_to_string(images[0])
                    metrics["text_length"] = len(text)
                    metrics["word_count"] = len(text.split())
                    metrics["char_density"] = len(text) / max(1, len(text.split()))
                    
                    # Detect multilingual content
                    metrics["multilingual_score"] = self._detect_multilingual(text)
                    
                else:
                    metrics.update({
                        "avg_ocr_confidence": 0.0,
                        "min_ocr_confidence": 0.0,
                        "max_ocr_confidence": 0.0,
                        "low_confidence_ratio": 1.0,
                        "text_length": 0,
                        "word_count": 0,
                        "char_density": 0.0,
                        "multilingual_score": 0.0
                    })
            else:
                metrics.update({
                    "avg_ocr_confidence": 0.0,
                    "min_ocr_confidence": 0.0,
                    "max_ocr_confidence": 0.0,
                    "low_confidence_ratio": 1.0,
                    "text_length": 0,
                    "word_count": 0,
                    "char_density": 0.0,
                    "multilingual_score": 0.0
                })
                
        except Exception as e:
            self.logger.warning(f"Text quality analysis failed: {e}")
            metrics.update({
                "avg_ocr_confidence": 0.0,
                "min_ocr_confidence": 0.0,
                "max_ocr_confidence": 0.0,
                "low_confidence_ratio": 1.0,
                "text_length": 0,
                "word_count": 0,
                "char_density": 0.0,
                "multilingual_score": 0.0
            })
        
        return metrics
    
    def _analyze_layout_complexity(self, pdf_path: Path) -> Dict[str, any]:
        """Analyze layout complexity using unstructured."""
        metrics = {}
        
        try:
            if UNSTRUCTURED_AVAILABLE:
                elements = partition_pdf(str(pdf_path), strategy="fast")
                
                # Count different element types
                element_types = {}
                for element in elements:
                    element_type = type(element).__name__
                    element_types[element_type] = element_types.get(element_type, 0) + 1
                
                metrics["total_elements"] = len(elements)
                metrics["element_diversity"] = len(element_types)
                metrics["has_tables"] = element_types.get("Table", 0) > 0
                metrics["has_images"] = element_types.get("Image", 0) > 0
                metrics["has_lists"] = element_types.get("ListItem", 0) > 0
                metrics["has_titles"] = element_types.get("Title", 0) > 0
                
                # Calculate layout complexity score
                complexity_factors = [
                    element_types.get("Table", 0) * 0.3,
                    element_types.get("Image", 0) * 0.2,
                    element_types.get("ListItem", 0) * 0.1,
                    len(elements) * 0.01,  # More elements = more complex
                ]
                metrics["layout_complexity"] = min(1.0, sum(complexity_factors))
                
            else:
                metrics.update({
                    "total_elements": 0,
                    "element_diversity": 0,
                    "has_tables": False,
                    "has_images": False,
                    "has_lists": False,
                    "has_titles": False,
                    "layout_complexity": 0.0
                })
                
        except Exception as e:
            self.logger.warning(f"Layout complexity analysis failed: {e}")
            metrics.update({
                "total_elements": 0,
                "element_diversity": 0,
                "has_tables": False,
                "has_images": False,
                "has_lists": False,
                "has_titles": False,
                "layout_complexity": 0.0
            })
        
        return metrics
    
    def _detect_multilingual(self, text: str) -> float:
        """Detect multilingual content in text."""
        if not text:
            return 0.0
        
        # Simple heuristic: count non-ASCII characters
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
        
        non_ascii_ratio = (total_chars - ascii_chars) / total_chars
        
        # Detect French-specific characters
        french_chars = sum(1 for c in text if c in 'àâäéèêëïîôöùûüÿç')
        french_ratio = french_chars / total_chars
        
        # Combine metrics
        multilingual_score = min(1.0, non_ascii_ratio * 2 + french_ratio * 3)
        
        return multilingual_score
    
    def _calculate_complexity_score(self, metrics: Dict[str, any]) -> float:
        """Calculate overall complexity score from metrics."""
        
        # Weighted factors for complexity calculation
        factors = []
        
        # OCR confidence factor (lower confidence = higher complexity)
        avg_confidence = metrics.get("avg_ocr_confidence", 0)
        confidence_factor = max(0, (100 - avg_confidence) / 100)
        factors.append(("ocr_confidence", confidence_factor, 0.25))
        
        # Layout complexity factor
        layout_complexity = metrics.get("layout_complexity", 0)
        factors.append(("layout_complexity", layout_complexity, 0.20))
        
        # Image density factor
        image_density = metrics.get("image_density", 0)
        factors.append(("image_density", image_density, 0.15))
        
        # Table density factor
        table_density = metrics.get("table_density", 0)
        factors.append(("table_density", table_density, 0.15))
        
        # Multilingual factor
        multilingual_score = metrics.get("multilingual_score", 0)
        factors.append(("multilingual", multilingual_score, 0.10))
        
        # Low confidence ratio factor
        low_conf_ratio = metrics.get("low_confidence_ratio", 0)
        factors.append(("low_confidence_ratio", low_conf_ratio, 0.10))
        
        # File size factor (larger files = more complex)
        file_size_mb = metrics.get("file_size_mb", 0)
        size_factor = min(1.0, file_size_mb / 10)  # Normalize to 10MB
        factors.append(("file_size", size_factor, 0.05))
        
        # Calculate weighted complexity score
        weighted_score = sum(factor_value * weight for _, factor_value, weight in factors)
        
        return min(1.0, weighted_score)
    
    def _recommend_strategy(self, complexity_score: float) -> str:
        """Recommend OCR strategy based on complexity score."""
        
        if complexity_score < 0.4:
            return "classic_ocr"
        elif complexity_score < 0.7:
            return "minicpm_v"
        else:
            return "qwen_vl"
    
    def _extract_features(self, metrics: Dict[str, any]) -> Dict[str, any]:
        """Extract features for ML routing decisions."""
        
        return {
            "ocr_confidence": metrics.get("avg_ocr_confidence", 0),
            "layout_complexity": metrics.get("layout_complexity", 0),
            "image_density": metrics.get("image_density", 0),
            "table_density": metrics.get("table_density", 0),
            "multilingual_score": metrics.get("multilingual_score", 0),
            "low_confidence_ratio": metrics.get("low_confidence_ratio", 0),
            "file_size_mb": metrics.get("file_size_mb", 0),
            "page_count": metrics.get("page_count", 1),
            "has_tables": metrics.get("has_tables", False),
            "has_images": metrics.get("has_images", False),
            "element_diversity": metrics.get("element_diversity", 0),
        }


def detect_complexity(pdf_path: Union[str, Path]) -> Dict[str, any]:
    """
    Convenience function to detect document complexity.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Complexity analysis results
    """
    analyzer = DocumentComplexityAnalyzer()
    return analyzer.analyze_document(pdf_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        analysis = detect_complexity(pdf_path)
        
        print(f"Document: {analysis['source']}")
        print(f"Complexity Score: {analysis['complexity_score']:.3f}")
        print(f"Recommended Strategy: {analysis['recommended_strategy']}")
        print(f"\nMetrics:")
        for key, value in analysis['metrics'].items():
            print(f"  {key}: {value}")
        
        print(f"\nFeatures:")
        for key, value in analysis['features'].items():
            print(f"  {key}: {value}")
        
        if analysis['warnings']:
            print(f"\nWarnings:")
            for warning in analysis['warnings']:
                print(f"  - {warning}")
        
    except Exception as e:
        print(f"Error analyzing document: {e}")
        sys.exit(1)
