"""
Scientific Document Detection Module for ChunkForge OCR Pipeline.

This module detects scientific and mathematical documents that require
specialized OCR processing using Nougat (Meta AI) or other scientific OCR engines.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import json

try:
    from pdfminer.high_level import extract_text, extract_pages
    from pdfminer.layout import LTTextContainer, LTFigure, LTImage
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logging.warning("pdfminer.six not available. Scientific detection will be limited.")

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract/PIL not available. OCR-based detection will be limited.")

LOGGER = logging.getLogger(__name__)


class ScientificDocumentDetector:
    """
    Detects scientific and mathematical documents that require specialized OCR.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the scientific document detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for scientific detection."""
        return {
            "image_text_ratio_threshold": 0.6,
            "ocr_coverage_threshold": 0.8,
            "math_symbol_weight": 0.1,
            "equation_pattern_weight": 0.05,
            "keyword_weight": 0.02,
            "min_math_score": 0.15,
            "scientific_keywords": [
                "équation", "fonction", "dérivée", "intégrale", "limite", "somme",
                "produit", "racine", "logarithme", "exponentielle", "trigonométrie",
                "géométrie", "algèbre", "calcul", "théorème", "démonstration",
                "solution", "résoudre", "calculer", "déterminer", "montrer",
                "vérifier", "exercice", "problème", "formule", "expression",
                "matrice", "vecteur", "espace", "dimension", "transformation",
                "polynôme", "série", "suite", "convergence", "divergence",
                "probabilité", "statistique", "distribution", "variance",
                "physique", "chimie", "biologie", "mécanique", "thermodynamique",
                "électromagnétisme", "optique", "quantique", "relativité"
            ],
            "math_symbols": [
                r'√', r'π', r'θ', r'∫', r'∑', r'∏', r'∞', r'±', r'≤', r'≥', 
                r'≠', r'≈', r'∈', r'∉', r'∪', r'∩', r'⊂', r'⊃', r'→', r'↔',
                r'α', r'β', r'γ', r'δ', r'ε', r'ζ', r'η', r'λ', r'μ', r'ν',
                r'ξ', r'ρ', r'σ', r'τ', r'φ', r'χ', r'ψ', r'ω', r'Δ', r'∇',
                r'∂', r'ℜ', r'ℑ', r'ℵ', r'ℶ', r'ℷ', r'ℸ', r'ℹ', r'℺', r'℻'
            ],
            "equation_patterns": [
                r'[a-zA-Z]²',  # Exposants
                r'[a-zA-Z]³',  # Cubes
                r'[a-zA-Z]ⁿ',  # Puissances n
                r'[a-zA-Z]₀',  # Indices
                r'[a-zA-Z]₁',  # Indices
                r'[a-zA-Z]₂',  # Indices
                r'[a-zA-Z]₃',  # Indices
                r'[a-zA-Z]ₙ',  # Indices n
                r'[0-9]+/[0-9]+',  # Fractions
                r'[a-zA-Z]+\([^)]+\)',  # Fonctions
                r'[0-9]+\.[0-9]+',  # Nombres décimaux
                r'[a-zA-Z]+²\s*[+\-]\s*[a-zA-Z]+²',  # Équations quadratiques
                r'[a-zA-Z]+\s*[+\-×÷]\s*[a-zA-Z]+',  # Opérations
                r'[a-zA-Z]+\s*=\s*[a-zA-Z0-9]+',  # Égalités
                r'[a-zA-Z]+\s*[<>≤≥]\s*[a-zA-Z0-9]+',  # Inégalités
            ]
        }
    
    def detect_scientific_document(self, doc_path: Union[str, Path]) -> Dict[str, any]:
        """
        Detect if a document is scientific/mathematical and requires specialized OCR.
        
        Args:
            doc_path: Path to the document to analyze
            
        Returns:
            Dictionary with detection results:
            {
                "is_scientific": bool,
                "math_density": float,
                "confidence": float,
                "indicators": dict,
                "recommended_engine": str,
                "reasoning": str
            }
        """
        doc_path = Path(doc_path)
        self.logger.info(f"Analyzing document for scientific content: {doc_path}")
        
        try:
            # Step 1: Basic document analysis
            doc_info = self._analyze_document_structure(doc_path)
            
            # Step 2: Extract text for analysis
            raw_text = self._extract_raw_text(doc_path)
            
            # Step 3: Analyze mathematical content
            math_analysis = self._analyze_mathematical_content(raw_text)
            
            # Step 4: Calculate scientific score
            scientific_score = self._calculate_scientific_score(doc_info, math_analysis)
            
            # Step 5: Determine if scientific
            is_scientific = scientific_score >= self.config["min_math_score"]
            
            # Step 6: Recommend OCR engine
            recommended_engine = self._recommend_ocr_engine(is_scientific, scientific_score, doc_info)
            
            # Step 7: Generate reasoning
            reasoning = self._generate_reasoning(is_scientific, scientific_score, doc_info, math_analysis)
            
            result = {
                "is_scientific": is_scientific,
                "math_density": scientific_score,
                "confidence": min(scientific_score * 2, 1.0),
                "indicators": {
                    "image_text_ratio": doc_info["image_text_ratio"],
                    "ocr_coverage": doc_info["ocr_coverage"],
                    "math_symbols_count": math_analysis["symbol_count"],
                    "equations_count": math_analysis["equation_count"],
                    "keywords_count": math_analysis["keyword_count"],
                    "file_size_mb": doc_info["file_size_mb"],
                    "page_count": doc_info["page_count"]
                },
                "recommended_engine": recommended_engine,
                "reasoning": reasoning,
                "source": str(doc_path)
            }
            
            self.logger.info(
                f"Scientific detection result: is_scientific={is_scientific}, "
                f"score={scientific_score:.3f}, engine={recommended_engine}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in scientific detection: {e}")
            return {
                "is_scientific": False,
                "math_density": 0.0,
                "confidence": 0.0,
                "indicators": {},
                "recommended_engine": "classic_ocr",
                "reasoning": f"Error during analysis: {str(e)}",
                "source": str(doc_path)
            }
    
    def _analyze_document_structure(self, doc_path: Path) -> Dict[str, any]:
        """Analyze document structure and extract basic metrics."""
        doc_info = {
            "file_size_mb": doc_path.stat().st_size / (1024 * 1024),
            "page_count": 0,
            "image_text_ratio": 0.0,
            "ocr_coverage": 0.0,
            "has_text_layer": False
        }
        
        if not PDFMINER_AVAILABLE:
            return doc_info
        
        try:
            # Extract pages and analyze structure
            pages = list(extract_pages(str(doc_path)))
            doc_info["page_count"] = len(pages)
            
            total_text_length = 0
            total_image_area = 0
            total_page_area = 0
            
            for page in pages:
                page_area = page.width * page.height
                total_page_area += page_area
                
                for element in page:
                    if isinstance(element, LTTextContainer):
                        total_text_length += len(element.get_text())
                        doc_info["has_text_layer"] = True
                    elif isinstance(element, (LTFigure, LTImage)):
                        # Estimate image area
                        img_area = element.width * element.height
                        total_image_area += img_area
            
            # Calculate ratios
            if total_page_area > 0:
                doc_info["image_text_ratio"] = total_image_area / total_page_area
            
            # Estimate OCR coverage based on text length
            if doc_info["page_count"] > 0:
                expected_text_length = doc_info["page_count"] * 2000  # Rough estimate
                doc_info["ocr_coverage"] = min(total_text_length / expected_text_length, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing document structure: {e}")
        
        return doc_info
    
    def _extract_raw_text(self, doc_path: Path) -> str:
        """Extract raw text from document for analysis."""
        if not PDFMINER_AVAILABLE:
            return ""
        
        try:
            return extract_text(str(doc_path))
        except Exception as e:
            self.logger.warning(f"Error extracting text: {e}")
            return ""
    
    def _analyze_mathematical_content(self, text: str) -> Dict[str, any]:
        """Analyze mathematical content in the text."""
        if not text:
            return {
                "symbol_count": 0,
                "equation_count": 0,
                "keyword_count": 0,
                "math_score": 0.0
            }
        
        # Count mathematical symbols
        symbol_count = sum(len(re.findall(pattern, text)) for pattern in self.config["math_symbols"])
        
        # Count equation patterns
        equation_count = sum(len(re.findall(pattern, text)) for pattern in self.config["equation_patterns"])
        
        # Count scientific keywords
        keyword_count = sum(
            1 for keyword in self.config["scientific_keywords"] 
            if keyword.lower() in text.lower()
        )
        
        # Calculate math score
        total_length = len(text)
        math_score = (
            symbol_count * self.config["math_symbol_weight"] +
            equation_count * self.config["equation_pattern_weight"] +
            keyword_count * self.config["keyword_weight"]
        ) / max(total_length / 1000, 1)
        
        return {
            "symbol_count": symbol_count,
            "equation_count": equation_count,
            "keyword_count": keyword_count,
            "math_score": min(math_score, 1.0)
        }
    
    def _calculate_scientific_score(self, doc_info: Dict, math_analysis: Dict) -> float:
        """Calculate overall scientific score."""
        score = 0.0
        
        # Mathematical content score
        score += math_analysis["math_score"] * 0.4
        
        # Image-to-text ratio (high ratio suggests scanned scientific papers)
        if doc_info["image_text_ratio"] > self.config["image_text_ratio_threshold"]:
            score += 0.3
        
        # Low OCR coverage suggests complex content
        if doc_info["ocr_coverage"] < self.config["ocr_coverage_threshold"]:
            score += 0.2
        
        # Large file size might indicate complex scientific content
        if doc_info["file_size_mb"] > 5.0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _recommend_ocr_engine(self, is_scientific: bool, score: float, doc_info: Dict) -> str:
        """Recommend the best OCR engine based on analysis."""
        if is_scientific:
            if score > 0.7:
                return "nougat"  # High confidence scientific document
            elif score > 0.4:
                return "qwen_vl"  # Medium confidence, use Qwen-VL as fallback
            else:
                return "classic_ocr"  # Low confidence, use classic OCR
        else:
            return "classic_ocr"
    
    def _generate_reasoning(self, is_scientific: bool, score: float, doc_info: Dict, math_analysis: Dict) -> str:
        """Generate human-readable reasoning for the decision."""
        reasons = []
        
        if math_analysis["symbol_count"] > 0:
            reasons.append(f"Found {math_analysis['symbol_count']} mathematical symbols")
        
        if math_analysis["equation_count"] > 0:
            reasons.append(f"Found {math_analysis['equation_count']} equation patterns")
        
        if math_analysis["keyword_count"] > 0:
            reasons.append(f"Found {math_analysis['keyword_count']} scientific keywords")
        
        if doc_info["image_text_ratio"] > self.config["image_text_ratio_threshold"]:
            reasons.append(f"High image-to-text ratio ({doc_info['image_text_ratio']:.2f})")
        
        if doc_info["ocr_coverage"] < self.config["ocr_coverage_threshold"]:
            reasons.append(f"Low OCR coverage ({doc_info['ocr_coverage']:.2f})")
        
        if is_scientific:
            return f"Scientific document detected (score: {score:.3f}). Indicators: {', '.join(reasons)}"
        else:
            return f"Not detected as scientific document (score: {score:.3f}). Indicators: {', '.join(reasons) if reasons else 'None'}"


def detect_scientific_document(doc_path: Union[str, Path]) -> Dict[str, any]:
    """
    Convenience function to detect scientific documents.
    
    Args:
        doc_path: Path to the document to analyze
        
    Returns:
        Dictionary with detection results
    """
    detector = ScientificDocumentDetector()
    return detector.detect_scientific_document(doc_path)


# Example usage
if __name__ == "__main__":
    # Test on math.pdf
    pdf_path = Path("tests/data/math.pdf")
    
    if pdf_path.exists():
        result = detect_scientific_document(pdf_path)
        print("Scientific Document Detection Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Test file {pdf_path} not found")


