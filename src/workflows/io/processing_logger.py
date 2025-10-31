"""
Processing Logger pour ChunkForge

Ce module trace tous les moteurs OCR, modèles LLM et stratégies utilisés
pour chaque document traité.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Étapes du pipeline de traitement."""
    INGESTION = "ingestion"
    OCR_DETECTION = "ocr_detection"
    OCR_PROCESSING = "ocr_processing"
    OCR_REPAIR = "ocr_repair"
    LANGUAGE_DETECTION = "language_detection"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    STRATEGY_SELECTION = "strategy_selection"
    CHUNKING = "chunking"
    EXPORT = "export"


class OCREngine(Enum):
    """Moteurs OCR disponibles."""
    NATIVE_TEXT = "native_text"
    TESSERACT = "tesseract"
    DOCTR = "doctr"
    QWEN_VL_2B = "qwen_vl_2b"
    QWEN_VL_7B = "qwen_vl_7b"
    NOUGAT = "nougat"
    MINICPM_V = "minicpm_v"
    UNSTRUCTURED = "unstructured"


class LLMModel(Enum):
    """Modèles LLM disponibles."""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    OLLAMA_LLAMA = "ollama_llama"
    OLLAMA_MISTRAL = "ollama_mistral"


class ChunkingStrategy(Enum):
    """Stratégies de chunking."""
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"
    FIXED_SIZE = "fixed_size"
    LATE = "late"


@dataclass
class ProcessingStep:
    """Une étape de traitement avec ses métadonnées."""
    stage: ProcessingStage
    engine_used: Optional[str] = None
    model_used: Optional[str] = None
    strategy_used: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.duration_seconds = self.end_time - self.start_time


@dataclass
class ProcessingTrace:
    """Trace complète du traitement d'un document."""
    document_id: str
    source_path: str
    file_name: str
    file_type: str
    file_size: int
    processing_start: float
    processing_end: Optional[float] = None
    total_duration: Optional[float] = None
    steps: List[ProcessingStep] = None
    final_ocr_engine: Optional[str] = None
    final_llm_model: Optional[str] = None
    final_chunking_strategy: Optional[str] = None
    chunks_generated: int = 0
    text_length: int = 0
    language_detected: Optional[str] = None
    complexity_score: Optional[float] = None
    scientific_document: bool = False
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.processing_end:
            self.total_duration = self.processing_end - self.processing_start


class ProcessingLogger:
    """
    Logger principal pour tracer le traitement des documents.
    
    Enregistre chaque étape, moteur utilisé, et métadonnées complètes.
    """
    
    def __init__(self, log_dir: str = "logs/processing"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: Optional[ProcessingTrace] = None
        self.logger = logging.getLogger(f"{__name__}.ProcessingLogger")
    
    def start_document_processing(
        self,
        document_id: str,
        source_path: str,
        file_name: str,
        file_type: str,
        file_size: int
    ) -> ProcessingTrace:
        """Démarre le traçage pour un nouveau document."""
        
        self.current_trace = ProcessingTrace(
            document_id=document_id,
            source_path=source_path,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            processing_start=time.time()
        )
        
        self.logger.info(
            f"🚀 Début du traitement: {file_name} "
            f"(ID: {document_id}, Taille: {file_size:,} bytes)"
        )
        
        return self.current_trace
    
    def log_step(
        self,
        stage: ProcessingStage,
        engine_used: Optional[str] = None,
        model_used: Optional[str] = None,
        strategy_used: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingStep:
        """Enregistre une étape de traitement."""
        
        if not self.current_trace:
            raise ValueError("Aucun document en cours de traitement")
        
        step = ProcessingStep(
            stage=stage,
            engine_used=engine_used,
            model_used=model_used,
            strategy_used=strategy_used,
            start_time=time.time(),
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        self.current_trace.steps.append(step)
        
        # Mettre à jour les informations finales
        if engine_used:
            self.current_trace.final_ocr_engine = engine_used
        if model_used:
            self.current_trace.final_llm_model = model_used
        if strategy_used:
            self.current_trace.final_chunking_strategy = strategy_used
        
        # Logging
        status = "✅" if success else "❌"
        self.logger.info(
            f"{status} {stage.value}: {engine_used or model_used or strategy_used or 'N/A'}"
        )
        
        if error_message:
            self.logger.error(f"Erreur dans {stage.value}: {error_message}")
        
        return step
    
    def log_ocr_decision(
        self,
        pdf_type: str,
        engine_selected: str,
        reason: str,
        complexity_score: Optional[float] = None,
        scientific_document: bool = False
    ):
        """Enregistre la décision de routage OCR."""
        
        metadata = {
            "pdf_type": pdf_type,
            "complexity_score": complexity_score,
            "scientific_document": scientific_document,
            "reason": reason
        }
        
        self.log_step(
            stage=ProcessingStage.OCR_DETECTION,
            engine_used=engine_selected,
            metadata=metadata
        )
        
        # Mettre à jour les métadonnées du document
        if self.current_trace:
            self.current_trace.complexity_score = complexity_score
            self.current_trace.scientific_document = scientific_document
    
    def log_llm_usage(
        self,
        model: str,
        purpose: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None
    ):
        """Enregistre l'utilisation d'un modèle LLM."""
        
        metadata = {
            "purpose": purpose,
            "tokens_used": tokens_used,
            "cost": cost
        }
        
        self.log_step(
            stage=ProcessingStage.OCR_REPAIR,
            model_used=model,
            metadata=metadata
        )
    
    def log_chunking_strategy(
        self,
        strategy: str,
        reason: str,
        ml_confidence: Optional[float] = None,
        chunks_count: int = 0
    ):
        """Enregistre la stratégie de chunking utilisée."""
        
        metadata = {
            "reason": reason,
            "ml_confidence": ml_confidence,
            "chunks_count": chunks_count
        }
        
        self.log_step(
            stage=ProcessingStage.STRATEGY_SELECTION,
            strategy_used=strategy,
            metadata=metadata
        )
        
        if self.current_trace:
            self.current_trace.chunks_generated = chunks_count
    
    def finish_document_processing(
        self,
        success: bool = True,
        error_message: Optional[str] = None,
        text_length: Optional[int] = None,
        language_detected: Optional[str] = None
    ) -> ProcessingTrace:
        """Termine le traçage du document."""
        
        if not self.current_trace:
            raise ValueError("Aucun document en cours de traitement")
        
        self.current_trace.processing_end = time.time()
        self.current_trace.success = success
        self.current_trace.error_message = error_message
        
        if text_length:
            self.current_trace.text_length = text_length
        if language_detected:
            self.current_trace.language_detected = language_detected
        
        # Sauvegarder la trace
        self._save_trace()
        
        # Logging final
        duration = self.current_trace.total_duration or 0
        status = "✅" if success else "❌"
        
        self.logger.info(
            f"{status} Traitement terminé: {self.current_trace.file_name} "
            f"(Durée: {duration:.2f}s, Chunks: {self.current_trace.chunks_generated})"
        )
        
        if self.current_trace.final_ocr_engine:
            self.logger.info(f"🔍 OCR utilisé: {self.current_trace.final_ocr_engine}")
        if self.current_trace.final_llm_model:
            self.logger.info(f"🤖 LLM utilisé: {self.current_trace.final_llm_model}")
        if self.current_trace.final_chunking_strategy:
            self.logger.info(f"📝 Stratégie: {self.current_trace.final_chunking_strategy}")
        
        trace = self.current_trace
        self.current_trace = None
        
        return trace
    
    def _save_trace(self):
        """Sauvegarde la trace dans un fichier JSON."""
        
        if not self.current_trace:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processing_trace_{self.current_trace.document_id}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        try:
            # Convertir les enums en strings pour la sérialisation JSON
            trace_dict = asdict(self.current_trace)
            
            # Convertir les ProcessingStage en strings
            for step in trace_dict.get('steps', []):
                if 'stage' in step and hasattr(step['stage'], 'value'):
                    step['stage'] = step['stage'].value
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trace_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Trace sauvegardée: {filepath}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la trace: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des traitements récents."""
        
        if not self.current_trace:
            return {"error": "Aucun document en cours de traitement"}
        
        return {
            "document_id": self.current_trace.document_id,
            "file_name": self.current_trace.file_name,
            "processing_time": self.current_trace.total_duration,
            "ocr_engine": self.current_trace.final_ocr_engine,
            "llm_model": self.current_trace.final_llm_model,
            "chunking_strategy": self.current_trace.final_chunking_strategy,
            "chunks_generated": self.current_trace.chunks_generated,
            "success": self.current_trace.success,
            "steps_count": len(self.current_trace.steps)
        }


# Instance globale du logger
processing_logger = ProcessingLogger()


def enrich_chunk_metadata(chunk_metadata: Dict[str, Any], processing_trace: ProcessingTrace) -> Dict[str, Any]:
    """
    Enrichit les métadonnées d'un chunk avec les informations de traitement.
    
    Args:
        chunk_metadata: Métadonnées existantes du chunk
        processing_trace: Trace complète du traitement
        
    Returns:
        Métadonnées enrichies
    """
    
    enriched = chunk_metadata.copy()
    
    # Informations de traitement
    enriched["processing"] = {
        "document_id": processing_trace.document_id,
        "ocr_engine": processing_trace.final_ocr_engine,
        "llm_model": processing_trace.final_llm_model,
        "chunking_strategy": processing_trace.final_chunking_strategy,
        "processing_time": processing_trace.total_duration,
        "complexity_score": processing_trace.complexity_score,
        "scientific_document": processing_trace.scientific_document,
        "language_detected": processing_trace.language_detected
    }
    
    # Détails des étapes
    enriched["processing_steps"] = [
        {
            "stage": step.stage.value,
            "engine": step.engine_used,
            "model": step.model_used,
            "strategy": step.strategy_used,
            "duration": step.duration_seconds,
            "success": step.success
        }
        for step in processing_trace.steps
    ]
    
    return enriched


def create_processing_report(traces: List[ProcessingTrace]) -> Dict[str, Any]:
    """
    Crée un rapport de traitement basé sur plusieurs traces.
    
    Args:
        traces: Liste des traces de traitement
        
    Returns:
        Rapport consolidé
    """
    
    if not traces:
        return {"error": "Aucune trace disponible"}
    
    successful_traces = [t for t in traces if t.success]
    failed_traces = [t for t in traces if not t.success]
    
    # Statistiques par moteur OCR
    ocr_engines = {}
    for trace in successful_traces:
        engine = trace.final_ocr_engine or "unknown"
        if engine not in ocr_engines:
            ocr_engines[engine] = {"count": 0, "total_time": 0, "avg_chunks": 0}
        ocr_engines[engine]["count"] += 1
        ocr_engines[engine]["total_time"] += trace.total_duration or 0
        ocr_engines[engine]["avg_chunks"] += trace.chunks_generated
    
    # Calculer les moyennes
    for engine_stats in ocr_engines.values():
        if engine_stats["count"] > 0:
            engine_stats["avg_time"] = engine_stats["total_time"] / engine_stats["count"]
            engine_stats["avg_chunks"] = engine_stats["avg_chunks"] / engine_stats["count"]
    
    # Statistiques par stratégie de chunking
    chunking_strategies = {}
    for trace in successful_traces:
        strategy = trace.final_chunking_strategy or "unknown"
        chunking_strategies[strategy] = chunking_strategies.get(strategy, 0) + 1
    
    return {
        "summary": {
            "total_documents": len(traces),
            "successful": len(successful_traces),
            "failed": len(failed_traces),
            "success_rate": len(successful_traces) / len(traces) if traces else 0,
            "total_processing_time": sum(t.total_duration or 0 for t in traces),
            "avg_processing_time": sum(t.total_duration or 0 for t in traces) / len(traces) if traces else 0
        },
        "ocr_engines": ocr_engines,
        "chunking_strategies": chunking_strategies,
        "llm_usage": {
            "documents_with_llm": len([t for t in successful_traces if t.final_llm_model]),
            "models_used": list(set(t.final_llm_model for t in successful_traces if t.final_llm_model))
        },
        "quality_metrics": {
            "avg_complexity_score": sum(t.complexity_score or 0 for t in successful_traces) / len(successful_traces) if successful_traces else 0,
            "scientific_documents": len([t for t in successful_traces if t.scientific_document]),
            "avg_chunks_per_document": sum(t.chunks_generated for t in successful_traces) / len(successful_traces) if successful_traces else 0
        }
    }
