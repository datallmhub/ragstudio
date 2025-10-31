"""
Intelligent OCR Router for ChunkForge Pipeline

This module provides intelligent routing between different OCR engines
based on document complexity analysis and fallback strategies.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from prefect import flow, task, get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Define dummy decorators
    def flow(func):
        return func
    def task(func):
        return func
    def get_run_logger():
        return logging.getLogger(__name__)

from src.core.config.ocr_settings import get_ocr_settings
from src.workflows.analyzer.complexity import DocumentComplexityAnalyzer, detect_complexity
from src.workflows.analyzer.scientific_detector import ScientificDocumentDetector, detect_scientific_document
from src.workflows.analyzer.image_quality_detector import ImageQualityDetector
from src.workflows.analyzer.ocr_quality_detector import OCRQualityDetector
from src.workflows.ingest.loader import ingest_file
from src.workflows.ocr.ocr_qwen_vl import QwenVLProcessor, QwenVLConfig, process_pdf_with_qwen_vl
from src.workflows.ocr.ocr_nougat import NougatOCRProcessor, NougatConfig, process_pdf_with_nougat

LOGGER = logging.getLogger(__name__)


class OCRRouterConfig:
    """Configuration for the intelligent OCR router."""

    def __init__(
        self,
        enable_qwen_vl: Optional[bool] = None,
        enable_nougat: Optional[bool] = None,
        enable_minicpm_v: bool = False,  # Placeholder for future implementation
        complexity_threshold_low: Optional[float] = None,
        complexity_threshold_high: Optional[float] = None,
        scientific_threshold: Optional[float] = None,
        fallback_enabled: Optional[bool] = None,
        timeout_seconds: int = 300,
        max_retries: Optional[int] = None,
    ):
        # Load from global settings if not provided
        settings = get_ocr_settings()

        self.enable_qwen_vl = enable_qwen_vl if enable_qwen_vl is not None else settings.qwen_vl.enabled
        self.enable_nougat = enable_nougat if enable_nougat is not None else settings.nougat.enabled
        self.enable_minicpm_v = enable_minicpm_v
        self.complexity_threshold_low = complexity_threshold_low if complexity_threshold_low is not None else settings.routing.complexity_threshold_low
        self.complexity_threshold_high = complexity_threshold_high if complexity_threshold_high is not None else settings.routing.complexity_threshold_high
        self.scientific_threshold = scientific_threshold if scientific_threshold is not None else settings.routing.scientific_threshold
        self.fallback_enabled = fallback_enabled if fallback_enabled is not None else settings.routing.fallback_enabled
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.routing.max_retries_per_engine


class OCRRouter:
    """
    Intelligent OCR router that selects the optimal engine based on document complexity and type.
    
    Routing logic:
    1. Scientific documents (math_density ≥ 0.15) → Nougat (Meta AI)
    2. complexity_score < 0.4 → Classic OCR (Tesseract/docTR)
    3. 0.4 ≤ score < 0.7 → MiniCPM-V 2.7B (placeholder)
    4. score ≥ 0.7 → Qwen-VL-2B/7B
    """
    
    def __init__(self, config: Optional[OCRRouterConfig] = None):
        self.config = config or OCRRouterConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.complexity_analyzer = DocumentComplexityAnalyzer()
        self.scientific_detector = ScientificDocumentDetector()
        self.image_quality_detector = ImageQualityDetector()
        self.ocr_quality_detector = OCRQualityDetector()

        # Initialize OCR engine configs (will load from global settings)
        self.qwen_vl_config = QwenVLConfig()
        self.nougat_config = NougatConfig()
        
        # Initialize engines
        self.qwen_vl_processor = None
        self.nougat_processor = None
        
        if self.config.enable_qwen_vl:
            try:
                self.qwen_vl_processor = QwenVLProcessor(self.qwen_vl_config)
                self.logger.info("Qwen-VL processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Qwen-VL: {e}")
                self.config.enable_qwen_vl = False
        
        if self.config.enable_nougat:
            try:
                self.nougat_processor = NougatOCRProcessor(self.nougat_config)
                self.logger.info("Nougat processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Nougat: {e}")
                self.config.enable_nougat = False
    
    def process_document(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process document with intelligent OCR routing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and processing metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting intelligent OCR routing for: {pdf_path}")
        
        start_time = time.time()
        result = {
            "source": str(pdf_path),
            "processing_time": 0.0,
            "routing_decisions": [],
            "extracted_text": "",
            "metadata": {},
            "errors": []
        }
        
        try:
            # Step 0: Detect image/document quality (for images only)
            is_image = pdf_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

            if is_image:
                self.logger.info(f"Detecting quality for image: {pdf_path}")

                # Test OCR quality with quick extraction
                try:
                    ocr_metrics = self.ocr_quality_detector.detect_ocr_quality(pdf_path, languages="eng+fra")

                    result["metadata"]["ocr_quality_test"] = ocr_metrics.to_dict()
                    result["routing_decisions"].append({
                        "step": "ocr_quality_detection",
                        "ocr_quality_score": ocr_metrics.overall_ocr_quality,
                        "ocr_quality_category": ocr_metrics.quality_category,
                        "recommended_engine": self.ocr_quality_detector.get_recommended_ocr_engine(ocr_metrics),
                        "timestamp": time.time()
                    })

                    self.logger.info(
                        f"OCR quality test: {ocr_metrics.quality_category} "
                        f"(score={ocr_metrics.overall_ocr_quality:.3f})"
                    )

                    # If OCR quality is LOW, route directly to Qwen-VL
                    if ocr_metrics.quality_category == "LOW":
                        self.logger.info("Low OCR quality detected, routing to advanced OCR (Qwen-VL)")
                        routing_result = {
                            "engine": "qwen_vl",
                            "reason": f"low_ocr_quality (score={ocr_metrics.overall_ocr_quality:.3f})",
                            "text": "",
                            "metrics": {}
                        }

                        if self.config.enable_qwen_vl and self.qwen_vl_processor:
                            routing_result = self._execute_qwen_vl(pdf_path)
                        else:
                            # Fallback to classic if Qwen-VL not available
                            self.logger.warning("Qwen-VL not available, using classic OCR despite low quality")
                            routing_result = self._execute_classic_ocr(pdf_path)

                        result["extracted_text"] = routing_result["text"]
                        result["metadata"]["ocr_engine"] = routing_result["engine"]
                        result["metadata"]["ocr_metrics"] = routing_result["metrics"]
                        result["routing_decisions"].append({
                            "step": "ocr_routing",
                            "engine_used": routing_result["engine"],
                            "routing_reason": routing_result["reason"],
                            "timestamp": time.time()
                        })

                        # Post-process and return early
                        if result["extracted_text"]:
                            result["metadata"]["text_length"] = len(result["extracted_text"])
                            result["metadata"]["word_count"] = len(result["extracted_text"].split())
                            result["metadata"]["success"] = True
                        else:
                            result["metadata"]["success"] = False
                            result["errors"].append("No text extracted")

                        result["processing_time"] = time.time() - start_time
                        return result

                except Exception as e:
                    self.logger.warning(f"OCR quality detection failed: {e}, continuing with normal routing")

            # Step 1: Detect scientific content (for PDFs only, skip for images)
            if is_image:
                # Skip scientific detection for images
                scientific_analysis = {
                    "is_scientific": False,
                    "math_density": 0.0,
                    "recommended_engine": "classic_ocr"
                }
                is_scientific = False
                math_density = 0.0
            else:
                # Analyze PDF documents for scientific content
                scientific_analysis = self.scientific_detector.detect_scientific_document(pdf_path)
                is_scientific = scientific_analysis["is_scientific"]
                math_density = scientific_analysis["math_density"]
            
            result["metadata"]["scientific_analysis"] = scientific_analysis
            result["routing_decisions"].append({
                "step": "scientific_detection",
                "is_scientific": is_scientific,
                "math_density": math_density,
                "recommended_engine": scientific_analysis["recommended_engine"],
                "timestamp": time.time()
            })
            
            self.logger.info(f"Scientific detection: is_scientific={is_scientific}, math_density={math_density:.3f}")
            
            # Step 2: Analyze document complexity (if not scientific and not an image)
            if not is_scientific and not is_image:
                complexity_analysis = self.complexity_analyzer.analyze_document(pdf_path)
                complexity_score = complexity_analysis["complexity_score"]
                recommended_strategy = complexity_analysis["recommended_strategy"]

                result["metadata"]["complexity_analysis"] = complexity_analysis
                result["routing_decisions"].append({
                    "step": "complexity_analysis",
                    "complexity_score": complexity_score,
                    "recommended_strategy": recommended_strategy,
                    "timestamp": time.time()
                })

                self.logger.info(f"Complexity analysis: score={complexity_score:.3f}, strategy={recommended_strategy}")
            elif is_image:
                # For images, use OCR quality as the primary routing factor
                complexity_score = 0.5  # Default medium complexity for images
                recommended_strategy = "classic_ocr"  # Will be overridden by OCR quality test
            else:
                complexity_score = 0.0
                recommended_strategy = "nougat"
            
            # Step 3: Route to appropriate OCR engine
            routing_result = self._route_to_engine(pdf_path, complexity_score, recommended_strategy, is_scientific, math_density)
            
            result["extracted_text"] = routing_result["text"]
            result["metadata"]["ocr_engine"] = routing_result["engine"]
            result["metadata"]["ocr_metrics"] = routing_result["metrics"]
            result["routing_decisions"].append({
                "step": "ocr_routing",
                "engine_used": routing_result["engine"],
                "routing_reason": routing_result["reason"],
                "timestamp": time.time()
            })
            
            # Step 3: Post-process results
            if result["extracted_text"]:
                result["metadata"]["text_length"] = len(result["extracted_text"])
                result["metadata"]["word_count"] = len(result["extracted_text"].split())
                result["metadata"]["success"] = True
            else:
                result["metadata"]["success"] = False
                result["errors"].append("No text extracted")
            
            result["processing_time"] = time.time() - start_time
            
            self.logger.info(f"OCR routing completed: {len(result['extracted_text'])} chars in {result['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCR routing failed: {e}")
            result["errors"].append(f"Routing failed: {str(e)}")
            result["processing_time"] = time.time() - start_time
            
            # Attempt fallback if enabled
            if self.config.fallback_enabled:
                self.logger.info("Attempting fallback to classic OCR")
                try:
                    fallback_result = self._fallback_to_classic_ocr(pdf_path)
                    result["extracted_text"] = fallback_result["text"]
                    result["metadata"]["ocr_engine"] = "classic_ocr_fallback"
                    result["metadata"]["success"] = True
                    result["routing_decisions"].append({
                        "step": "fallback",
                        "engine_used": "classic_ocr",
                        "reason": "primary_routing_failed",
                        "timestamp": time.time()
                    })
                except Exception as fallback_error:
                    result["errors"].append(f"Fallback failed: {str(fallback_error)}")
            
            return result
    
    def _route_to_engine(self, pdf_path: Path, complexity_score: float, recommended_strategy: str, is_scientific: bool = False, math_density: float = 0.0) -> Dict[str, any]:
        """
        Route document to appropriate OCR engine based on complexity and scientific content.
        
        Args:
            pdf_path: Path to the PDF file
            complexity_score: Calculated complexity score
            recommended_strategy: Recommended strategy from analysis
            is_scientific: Whether document is detected as scientific
            math_density: Mathematical content density score
            
        Returns:
            Dictionary containing OCR results
        """
        
        # Priority 1: Scientific documents → Nougat
        if is_scientific and math_density >= self.config.scientific_threshold:
            if self.config.enable_nougat and self.nougat_processor:
                strategy = "nougat"
                reason = f"scientific_document (math_density={math_density:.3f})"
            else:
                # Fallback to Qwen-VL for scientific documents if Nougat unavailable
                if self.config.enable_qwen_vl and self.qwen_vl_processor:
                    strategy = "qwen_vl"
                    reason = f"scientific_document_fallback (math_density={math_density:.3f}, nougat_unavailable)"
                else:
                    strategy = "classic_ocr"
                    reason = f"scientific_document_fallback (math_density={math_density:.3f}, advanced_engines_unavailable)"
        
        # Priority 2: High complexity → Qwen-VL
        elif complexity_score >= self.config.complexity_threshold_high:
            if self.config.enable_qwen_vl and self.qwen_vl_processor:
                strategy = "qwen_vl"
                reason = f"high_complexity (score={complexity_score:.3f})"
            else:
                strategy = "classic_ocr"
                reason = f"high_complexity_fallback (score={complexity_score:.3f}, qwen_vl_unavailable)"
        
        # Priority 3: Medium complexity → MiniCPM-V (placeholder)
        elif complexity_score >= self.config.complexity_threshold_low:
            if self.config.enable_minicpm_v:
                strategy = "minicpm_v"
                reason = f"medium_complexity (score={complexity_score:.3f})"
            else:
                strategy = "classic_ocr"
                reason = f"medium_complexity_fallback (score={complexity_score:.3f}, minicpm_v_unavailable)"
        
        # Priority 4: Low complexity → Classic OCR
        else:
            strategy = "classic_ocr"
            reason = f"low_complexity (score={complexity_score:.3f})"
        
        self.logger.info(f"Routing to {strategy}: {reason}")
        
        # Execute OCR with selected strategy
        if strategy == "classic_ocr":
            return self._execute_classic_ocr(pdf_path)
        elif strategy == "nougat":
            return self._execute_nougat(pdf_path)
        elif strategy == "minicpm_v":
            return self._execute_minicpm_v(pdf_path)
        elif strategy == "qwen_vl":
            return self._execute_qwen_vl(pdf_path)
        else:
            raise ValueError(f"Unknown OCR strategy: {strategy}")
    
    def _execute_classic_ocr(self, pdf_path: Path) -> Dict[str, any]:
        """Execute classic OCR using unstructured loader."""
        try:
            self.logger.info(f"Executing classic OCR on {pdf_path}")

            # Use unstructured loader for OCR
            document = ingest_file(str(pdf_path))
            text = document.text if document else ""

            metrics = {
                "source": str(pdf_path),
                "text_length": len(text),
                "method": "unstructured",
            }

            return {
                "text": text,
                "engine": "classic_ocr",
                "metrics": metrics,
                "reason": "classic_ocr_execution"
            }

        except Exception as e:
            self.logger.error(f"Classic OCR failed: {e}")
            raise
    
    def _execute_nougat(self, pdf_path: Path) -> Dict[str, any]:
        """Execute Nougat OCR for scientific documents."""
        try:
            self.logger.info(f"Executing Nougat OCR on {pdf_path}")
            
            if not self.nougat_processor:
                raise RuntimeError("Nougat processor not available")
            
            results = self.nougat_processor.process_pdf(pdf_path)
            
            return {
                "text": results["total_text"],
                "engine": "nougat",
                "metrics": {
                    "model_used": results["model_used"],
                    "processing_method": results["processing_method"],
                    "processing_time": results["processing_time"],
                    "pages_processed": len(results["pages"]),
                    "avg_confidence": sum(p.get("confidence", 0) for p in results["pages"]) / max(1, len(results["pages"])),
                    "device_used": results["metadata"].get("device_used", "unknown")
                },
                "reason": "nougat_execution"
            }
            
        except Exception as e:
            self.logger.error(f"Nougat OCR failed: {e}")
            raise
    
    def _execute_minicpm_v(self, pdf_path: Path) -> Dict[str, any]:
        """Execute MiniCPM-V OCR (placeholder implementation)."""
        self.logger.warning("MiniCPM-V not implemented, falling back to classic OCR")
        
        # For now, fall back to classic OCR
        return self._execute_classic_ocr(pdf_path)
    
    def _execute_qwen_vl(self, pdf_path: Path) -> Dict[str, any]:
        """Execute Qwen-VL OCR."""
        try:
            self.logger.info(f"Executing Qwen-VL OCR on {pdf_path}")

            if not self.qwen_vl_processor:
                raise RuntimeError("Qwen-VL processor not available")

            # Check if it's an image or PDF
            is_image = pdf_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

            if is_image:
                # Process image directly
                self.logger.info(f"Processing {pdf_path.name} as image with Qwen-VL")
                from PIL import Image
                img = Image.open(pdf_path)
                result = self.qwen_vl_processor._process_image(img, page_num=1)

                return {
                    "text": result["text"],
                    "engine": "qwen_vl",
                    "metrics": {
                        "model_used": self.qwen_vl_processor.config.model_name,
                        "api_base_url": self.qwen_vl_processor.config.api_base_url,
                        "processing_method": "image_direct",
                        "processing_time": 0.0,
                        "pages_processed": 1,
                        "avg_confidence": result.get("confidence", 0.0)
                    },
                    "reason": "qwen_vl_image_execution"
                }

            else:
                # Process PDF (convert to images first)
                self.logger.info(f"Processing {pdf_path.name} as PDF with Qwen-VL")
                results = self.qwen_vl_processor.process_pdf(pdf_path)

                return {
                    "text": results["total_text"],
                    "engine": "qwen_vl",
                    "metrics": {
                        "model_used": results["model_used"],
                        "api_base_url": results["api_base_url"],
                        "processing_method": results["processing_method"],
                        "processing_time": results["processing_time"],
                        "pages_processed": len(results["pages"]),
                        "avg_confidence": sum(p.get("confidence", 0) for p in results["pages"]) / max(1, len(results["pages"]))
                    },
                    "reason": "qwen_vl_pdf_execution"
                }

        except Exception as e:
            self.logger.error(f"Qwen-VL OCR failed: {e}")

            # Automatic fallback to classic OCR if enabled in routing config
            if hasattr(self.config, 'fallback_enabled') and self.config.fallback_enabled:
                self.logger.warning("Falling back to classic OCR due to Qwen-VL failure")
                try:
                    fallback_result = self._execute_classic_ocr(pdf_path)
                    # Add metadata indicating this was a fallback
                    fallback_result["metadata"] = fallback_result.get("metadata", {})
                    fallback_result["metadata"]["fallback_from"] = "qwen_vl"
                    fallback_result["metadata"]["fallback_reason"] = str(e)
                    fallback_result["reason"] = f"qwen_vl_failed_fallback_to_classic: {str(e)[:100]}"
                    return fallback_result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback to classic OCR also failed: {fallback_error}")
                    raise RuntimeError(
                        f"Both Qwen-VL and classic OCR failed. "
                        f"Qwen-VL error: {e}. Fallback error: {fallback_error}"
                    ) from e
            else:
                # Fallback disabled, raise original error
                raise RuntimeError(f"Qwen-VL OCR failed and fallback is disabled: {e}") from e

    def _fallback_to_classic_ocr(self, pdf_path: Path) -> Dict[str, any]:
        """Fallback to classic OCR when other engines fail."""
        self.logger.info(f"Executing fallback classic OCR on {pdf_path}")
        return self._execute_classic_ocr(pdf_path)


@task(name="Analyze Document Complexity")
def analyze_complexity_task(pdf_path: str) -> Dict[str, any]:
    """
    Prefect task for document complexity analysis.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Complexity analysis results
    """
    logger = get_run_logger()
    logger.info(f"Analyzing complexity for: {pdf_path}")
    
    analyzer = DocumentComplexityAnalyzer()
    return analyzer.analyze_document(pdf_path)


@task(name="Route OCR Engine")
def route_ocr_engine_task(
    pdf_path: str, 
    complexity_score: float, 
    config: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """
    Prefect task for OCR engine routing.
    
    Args:
        pdf_path: Path to the PDF file
        complexity_score: Document complexity score
        config: Optional router configuration
        
    Returns:
        OCR routing results
    """
    logger = get_run_logger()
    logger.info(f"Routing OCR engine for: {pdf_path} (complexity={complexity_score:.3f})")
    
    # Convert config dict to OCRRouterConfig if provided
    router_config = None
    if config:
        router_config = OCRRouterConfig(**config)
    
    router = OCRRouter(router_config)
    
    # Create a temporary result structure for the routing decision
    pdf_path_obj = Path(pdf_path)
    routing_result = router._route_to_engine(pdf_path_obj, complexity_score, "auto")
    
    return {
        "pdf_path": pdf_path,
        "complexity_score": complexity_score,
        "routing_result": routing_result
    }


@flow(name="ChunkForge OCR Router")
def chunkforge_ocr_router_flow(
    pdf_path: str,
    config: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """
    Prefect flow for intelligent OCR routing.
    
    Args:
        pdf_path: Path to the PDF file
        config: Optional router configuration
        
    Returns:
        Complete OCR processing results
    """
    logger = get_run_logger()
    logger.info(f"Starting ChunkForge OCR Router flow for: {pdf_path}")
    
    # Convert config dict to OCRRouterConfig if provided
    router_config = None
    if config:
        router_config = OCRRouterConfig(**config)
    
    # Initialize router
    router = OCRRouter(router_config)
    
    # Process document
    result = router.process_document(pdf_path)
    
    logger.info(f"ChunkForge OCR Router flow completed: {len(result['extracted_text'])} chars")
    
    return result


# Convenience functions
def process_document_intelligent(pdf_path: Union[str, Path], config: Optional[OCRRouterConfig] = None) -> Dict[str, any]:
    """
    Process document with intelligent OCR routing.
    
    Args:
        pdf_path: Path to the PDF file
        config: Optional router configuration
        
    Returns:
        Processing results
    """
    router = OCRRouter(config)
    return router.process_document(pdf_path)


def analyze_and_route(pdf_path: Union[str, Path]) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Analyze document complexity and get routing recommendation.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (complexity_analysis, routing_recommendation)
    """
    analyzer = DocumentComplexityAnalyzer()
    complexity_analysis = analyzer.analyze_document(pdf_path)
    
    router = OCRRouter()
    complexity_score = complexity_analysis["complexity_score"]
    
    # Get routing decision without executing OCR
    routing_result = router._route_to_engine(Path(pdf_path), complexity_score, "auto")
    
    return complexity_analysis, routing_result


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ocr_router.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Process document with intelligent routing
        result = process_document_intelligent(pdf_path)
        
        print(f"Document processed: {result['source']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"OCR engine: {result['metadata']['ocr_engine']}")
        print(f"Text length: {result['metadata']['text_length']} chars")
        print(f"Success: {result['metadata']['success']}")
        
        print(f"\nRouting decisions:")
        for decision in result['routing_decisions']:
            print(f"  {decision['step']}: {decision}")
        
        if result['errors']:
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        # Show sample text
        if result['extracted_text']:
            print(f"\nSample text (first 500 chars):")
            print(result['extracted_text'][:500])
        
    except Exception as e:
        print(f"Error processing document: {e}")
        sys.exit(1)
