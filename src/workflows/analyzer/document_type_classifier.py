"""
Document Type Classifier

Classifie les documents selon leur type et leurs caractéristiques pour
adapter le traitement OCR et le routage approprié.

Types de documents :
- IMAGE : JPEG, PNG, TIFF, BMP (nécessitent OCR)
- PDF_NATIVE : PDF avec texte extractible (pas d'OCR nécessaire)
- PDF_SCANNED : PDF scanné/image (nécessite OCR)
- PDF_HYBRID : PDF mixte (texte + images scannées)
- OFFICE : DOCX, XLSX, PPTX (texte structuré)
- TEXT : TXT, MD, HTML (texte brut)
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Union
from enum import Enum

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Types de documents supportés."""
    IMAGE = "image"
    PDF_NATIVE = "pdf_native"
    PDF_SCANNED = "pdf_scanned"
    PDF_HYBRID = "pdf_hybrid"
    OFFICE = "office"
    TEXT = "text"
    UNKNOWN = "unknown"


class DocumentCharacteristics:
    """Caractéristiques d'un document."""

    def __init__(
        self,
        doc_type: DocumentType,
        needs_ocr: bool,
        has_text: bool,
        has_images: bool,
        is_structured: bool,
        page_count: int = 1,
        text_coverage: float = 0.0,
        image_coverage: float = 0.0,
        recommended_processor: str = "classic",
        metadata: Dict = None,
    ):
        self.doc_type = doc_type
        self.needs_ocr = needs_ocr
        self.has_text = has_text
        self.has_images = has_images
        self.is_structured = is_structured
        self.page_count = page_count
        self.text_coverage = text_coverage  # % du document qui est du texte extractible
        self.image_coverage = image_coverage  # % du document qui est des images
        self.recommended_processor = recommended_processor
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "doc_type": self.doc_type.value,
            "needs_ocr": self.needs_ocr,
            "has_text": self.has_text,
            "has_images": self.has_images,
            "is_structured": self.is_structured,
            "page_count": self.page_count,
            "text_coverage": self.text_coverage,
            "image_coverage": self.image_coverage,
            "recommended_processor": self.recommended_processor,
            "metadata": self.metadata,
        }


class DocumentTypeClassifier:
    """
    Classifie les documents selon leur type et caractéristiques.

    Stratégies de traitement :

    1. IMAGE → OCR direct (Qwen-VL ou Classic)
       - Test de qualité OCR
       - Pas d'analyse de structure PDF

    2. PDF_NATIVE → Extraction texte simple
       - PyMuPDF/pdfplumber
       - Pas d'OCR nécessaire

    3. PDF_SCANNED → OCR sur toutes les pages
       - Conversion PDF → images
       - OCR avec routage intelligent

    4. PDF_HYBRID → Extraction texte + OCR sur images
       - Extraire texte des zones texte
       - OCR sur les zones image
       - Combiner les résultats

    5. OFFICE → Extraction texte structuré
       - python-docx, openpyxl, etc.
       - Pas d'OCR

    6. TEXT → Lecture simple
       - Pas de traitement spécial
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def classify(self, file_path: Union[str, Path]) -> DocumentCharacteristics:
        """
        Classifie un document selon son type et ses caractéristiques.

        Args:
            file_path: Path to the document

        Returns:
            DocumentCharacteristics object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Classifying document: {file_path}")

        suffix = file_path.suffix.lower()

        # Classification par extension
        if suffix in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}:
            return self._classify_image(file_path)

        elif suffix == '.pdf':
            return self._classify_pdf(file_path)

        elif suffix in {'.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'}:
            return self._classify_office(file_path)

        elif suffix in {'.txt', '.md', '.rst', '.html', '.htm', '.xml'}:
            return self._classify_text(file_path)

        else:
            self.logger.warning(f"Unknown document type: {suffix}")
            return DocumentCharacteristics(
                doc_type=DocumentType.UNKNOWN,
                needs_ocr=False,
                has_text=False,
                has_images=False,
                is_structured=False,
                recommended_processor="unknown"
            )

    def _classify_image(self, file_path: Path) -> DocumentCharacteristics:
        """Classifie une image."""
        self.logger.info(f"Classifying as IMAGE: {file_path.name}")

        metadata = {}

        if PIL_AVAILABLE:
            try:
                img = Image.open(file_path)
                metadata['size'] = img.size
                metadata['mode'] = img.mode
                metadata['format'] = img.format
            except Exception as e:
                self.logger.warning(f"Could not read image metadata: {e}")

        return DocumentCharacteristics(
            doc_type=DocumentType.IMAGE,
            needs_ocr=True,
            has_text=False,
            has_images=True,
            is_structured=False,
            page_count=1,
            text_coverage=0.0,
            image_coverage=1.0,
            recommended_processor="ocr_with_quality_test",
            metadata=metadata
        )

    def _classify_pdf(self, file_path: Path) -> DocumentCharacteristics:
        """
        Classifie un PDF (native, scanned, or hybrid).

        Stratégie :
        1. Essayer d'extraire du texte
        2. Si texte > 50% du contenu → PDF_NATIVE
        3. Si texte < 10% → PDF_SCANNED
        4. Si 10-50% → PDF_HYBRID
        """
        self.logger.info(f"Classifying PDF: {file_path.name}")

        if not PYMUPDF_AVAILABLE:
            self.logger.warning("PyMuPDF not available, assuming PDF_SCANNED")
            return DocumentCharacteristics(
                doc_type=DocumentType.PDF_SCANNED,
                needs_ocr=True,
                has_text=False,
                has_images=True,
                is_structured=True,
                recommended_processor="pdf_to_images_then_ocr"
            )

        try:
            doc = fitz.open(file_path)
            page_count = len(doc)

            # Analyser les premières pages pour déterminer le type
            sample_size = min(5, page_count)
            total_text_length = 0
            total_images = 0

            for page_num in range(sample_size):
                page = doc[page_num]

                # Extraire texte
                text = page.get_text()
                total_text_length += len(text.strip())

                # Compter images
                images = page.get_images()
                total_images += len(images)

            doc.close()

            # Calculer le ratio texte/image
            avg_text_per_page = total_text_length / sample_size
            avg_images_per_page = total_images / sample_size

            # Déterminer le type
            # Heuristique : plus de 100 caractères par page = texte natif
            text_coverage = min(1.0, avg_text_per_page / 500)  # Normalize to 0-1

            metadata = {
                "page_count": page_count,
                "avg_text_per_page": avg_text_per_page,
                "avg_images_per_page": avg_images_per_page,
                "sample_size": sample_size,
            }

            if avg_text_per_page > 500:
                # PDF avec beaucoup de texte extractible
                if avg_images_per_page > 0.5:
                    # PDF hybride (texte + images)
                    self.logger.info("PDF classified as HYBRID (text + images)")
                    return DocumentCharacteristics(
                        doc_type=DocumentType.PDF_HYBRID,
                        needs_ocr=True,  # Pour les images
                        has_text=True,
                        has_images=True,
                        is_structured=True,
                        page_count=page_count,
                        text_coverage=text_coverage,
                        image_coverage=1.0 - text_coverage,
                        recommended_processor="hybrid_extract_text_and_ocr_images",
                        metadata=metadata
                    )
                else:
                    # PDF natif (principalement du texte)
                    self.logger.info("PDF classified as NATIVE (extractable text)")
                    return DocumentCharacteristics(
                        doc_type=DocumentType.PDF_NATIVE,
                        needs_ocr=False,
                        has_text=True,
                        has_images=False,
                        is_structured=True,
                        page_count=page_count,
                        text_coverage=text_coverage,
                        image_coverage=0.0,
                        recommended_processor="extract_text_only",
                        metadata=metadata
                    )

            elif avg_text_per_page > 100:
                # PDF avec un peu de texte (possiblement OCR déjà appliqué)
                self.logger.info("PDF classified as HYBRID (some text)")
                return DocumentCharacteristics(
                    doc_type=DocumentType.PDF_HYBRID,
                    needs_ocr=True,
                    has_text=True,
                    has_images=True,
                    is_structured=True,
                    page_count=page_count,
                    text_coverage=text_coverage,
                    image_coverage=0.5,
                    recommended_processor="hybrid_extract_text_and_ocr_images",
                    metadata=metadata
                )

            else:
                # PDF scanné (presque pas de texte extractible)
                self.logger.info("PDF classified as SCANNED (no extractable text)")
                return DocumentCharacteristics(
                    doc_type=DocumentType.PDF_SCANNED,
                    needs_ocr=True,
                    has_text=False,
                    has_images=True,
                    is_structured=True,
                    page_count=page_count,
                    text_coverage=0.0,
                    image_coverage=1.0,
                    recommended_processor="pdf_to_images_then_ocr",
                    metadata=metadata
                )

        except Exception as e:
            self.logger.error(f"Failed to classify PDF: {e}")
            # Fallback : assumer PDF scanné
            return DocumentCharacteristics(
                doc_type=DocumentType.PDF_SCANNED,
                needs_ocr=True,
                has_text=False,
                has_images=True,
                is_structured=True,
                recommended_processor="pdf_to_images_then_ocr",
                metadata={"error": str(e)}
            )

    def _classify_office(self, file_path: Path) -> DocumentCharacteristics:
        """Classifie un document Office."""
        self.logger.info(f"Classifying as OFFICE: {file_path.name}")

        return DocumentCharacteristics(
            doc_type=DocumentType.OFFICE,
            needs_ocr=False,
            has_text=True,
            has_images=False,
            is_structured=True,
            recommended_processor="extract_structured_text",
            metadata={"format": file_path.suffix}
        )

    def _classify_text(self, file_path: Path) -> DocumentCharacteristics:
        """Classifie un document texte."""
        self.logger.info(f"Classifying as TEXT: {file_path.name}")

        return DocumentCharacteristics(
            doc_type=DocumentType.TEXT,
            needs_ocr=False,
            has_text=True,
            has_images=False,
            is_structured=False,
            recommended_processor="read_text",
            metadata={"format": file_path.suffix}
        )


def classify_document(file_path: Union[str, Path]) -> DocumentCharacteristics:
    """
    Convenience function to classify a document.

    Args:
        file_path: Path to the document

    Returns:
        DocumentCharacteristics object
    """
    classifier = DocumentTypeClassifier()
    return classifier.classify(file_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python document_type_classifier.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        classifier = DocumentTypeClassifier()
        characteristics = classifier.classify(file_path)

        print(f"Document Classification: {file_path}")
        print(f"=" * 80)
        print(f"Type:                  {characteristics.doc_type.value}")
        print(f"Needs OCR:             {characteristics.needs_ocr}")
        print(f"Has Text:              {characteristics.has_text}")
        print(f"Has Images:            {characteristics.has_images}")
        print(f"Is Structured:         {characteristics.is_structured}")
        print(f"Page Count:            {characteristics.page_count}")
        print(f"Text Coverage:         {characteristics.text_coverage:.1%}")
        print(f"Image Coverage:        {characteristics.image_coverage:.1%}")
        print(f"Recommended Processor: {characteristics.recommended_processor}")

        if characteristics.metadata:
            print(f"\nMetadata:")
            for key, value in characteristics.metadata.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error classifying document: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
