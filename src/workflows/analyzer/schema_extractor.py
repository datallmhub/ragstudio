"""
Mathematical Schema Extractor for ChunkForge OCR Pipeline.

This module extracts and saves mathematical schemas, diagrams, and geometric
elements from OCR-processed scientific documents.
"""

import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class MathematicalSchema:
    """Represents a mathematical schema or diagram."""
    schema_id: str
    schema_type: str  # 'geometric', 'algebraic', 'graph', 'table', 'figure'
    description: str
    elements: List[str]  # Geometric elements found
    equations: List[str]  # Related equations
    references: List[str]  # References to this schema
    page_number: Optional[int] = None
    confidence: float = 0.0


class MathematicalSchemaExtractor:
    """
    Extracts mathematical schemas, diagrams, and geometric elements
    from OCR-processed scientific documents.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the schema extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "geometric_patterns": [
                r"cercle de centre ([A-Z]) et de rayon (\d+)",
                r"triangle ([A-Z]{3})",
                r"droite \(([A-Z]{2})\)",
                r"segment \[([A-Z]{2})\]",
                r"point ([A-Z]) d'affixe",
                r"angle orienté",
                r"repère orthonormé",
                r"courbe représentative",
                r"asymptote",
                r"tangente"
            ],
            "equation_patterns": [
                r"\\\\([^\\\\]+)\\\\",  # LaTeX equations
                r"([a-zA-Z]+\s*[+\-×÷]\s*[a-zA-Z]+)",  # Simple equations
                r"([a-zA-Z]+\s*=\s*[a-zA-Z0-9]+)",  # Equalities
                r"([a-zA-Z]+\s*[<>≤≥]\s*[a-zA-Z0-9]+)"  # Inequalities
            ],
            "figure_patterns": [
                r"Figure (\d+)",
                r"figure (\d+)",
                r"annexe",
                r"schéma",
                r"diagramme"
            ],
            "schema_types": {
                "geometric": ["cercle", "triangle", "droite", "segment", "point", "angle"],
                "algebraic": ["équation", "fonction", "polynôme", "matrice"],
                "graph": ["courbe", "graphe", "asymptote", "tangente"],
                "table": ["tableau", "matrice", "système"],
                "figure": ["figure", "annexe", "schéma", "diagramme"]
            }
        }
    
    def extract_schemas(self, text: str, source_path: Optional[str] = None) -> List[MathematicalSchema]:
        """
        Extract mathematical schemas from OCR text.
        
        Args:
            text: OCR-processed text
            source_path: Path to source document
            
        Returns:
            List of extracted mathematical schemas
        """
        self.logger.info("Extracting mathematical schemas from text")
        
        schemas = []
        
        # Extract geometric schemas
        geometric_schemas = self._extract_geometric_schemas(text)
        schemas.extend(geometric_schemas)
        
        # Extract algebraic schemas
        algebraic_schemas = self._extract_algebraic_schemas(text)
        schemas.extend(algebraic_schemas)
        
        # Extract graph schemas
        graph_schemas = self._extract_graph_schemas(text)
        schemas.extend(graph_schemas)
        
        # Extract table schemas
        table_schemas = self._extract_table_schemas(text)
        schemas.extend(table_schemas)
        
        # Extract figure references
        figure_schemas = self._extract_figure_schemas(text)
        schemas.extend(figure_schemas)
        
        self.logger.info(f"Extracted {len(schemas)} mathematical schemas")
        
        return schemas
    
    def _extract_geometric_schemas(self, text: str) -> List[MathematicalSchema]:
        """Extract geometric schemas (circles, triangles, etc.)."""
        schemas = []
        
        # Extract circles
        circle_matches = re.findall(r"cercle de centre ([A-Z]) et de rayon (\d+)", text)
        for center, radius in circle_matches:
            schema = MathematicalSchema(
                schema_id=f"circle_{center}_{radius}",
                schema_type="geometric",
                description=f"Cercle de centre {center} et de rayon {radius}",
                elements=[f"cercle", f"centre_{center}", f"rayon_{radius}"],
                equations=[],
                references=self._find_references(text, f"cercle de centre {center}"),
                confidence=0.9
            )
            schemas.append(schema)
        
        # Extract triangles
        triangle_matches = re.findall(r"triangle ([A-Z]{3})", text)
        for triangle in triangle_matches:
            schema = MathematicalSchema(
                schema_id=f"triangle_{triangle}",
                schema_type="geometric",
                description=f"Triangle {triangle}",
                elements=[f"triangle", f"vertices_{triangle}"],
                equations=[],
                references=self._find_references(text, f"triangle {triangle}"),
                confidence=0.8
            )
            schemas.append(schema)
        
        # Extract lines and segments
        line_matches = re.findall(r"droite \(([A-Z]{2})\)", text)
        for line in line_matches:
            schema = MathematicalSchema(
                schema_id=f"line_{line}",
                schema_type="geometric",
                description=f"Droite ({line})",
                elements=[f"droite", f"points_{line}"],
                equations=[],
                references=self._find_references(text, f"droite ({line})"),
                confidence=0.7
            )
            schemas.append(schema)
        
        segment_matches = re.findall(r"segment \[([A-Z]{2})\]", text)
        for segment in segment_matches:
            schema = MathematicalSchema(
                schema_id=f"segment_{segment}",
                schema_type="geometric",
                description=f"Segment [{segment}]",
                elements=[f"segment", f"points_{segment}"],
                equations=[],
                references=self._find_references(text, f"segment [{segment}]"),
                confidence=0.7
            )
            schemas.append(schema)
        
        return schemas
    
    def _extract_algebraic_schemas(self, text: str) -> List[MathematicalSchema]:
        """Extract algebraic schemas (equations, functions)."""
        schemas = []
        
        # Extract LaTeX equations
        latex_equations = re.findall(r"\\\\([^\\\\]+)\\\\", text)
        for i, equation in enumerate(latex_equations):
            if len(equation.strip()) > 5:  # Filter out very short equations
                schema = MathematicalSchema(
                    schema_id=f"equation_{i+1}",
                    schema_type="algebraic",
                    description=f"Équation: \\\\{equation}\\\\",
                    elements=["équation", "latex"],
                    equations=[f"\\\\{equation}\\\\"],
                    references=self._find_references(text, f"\\\\{equation}\\\\"),
                    confidence=0.9
                )
                schemas.append(schema)
        
        return schemas
    
    def _extract_graph_schemas(self, text: str) -> List[MathematicalSchema]:
        """Extract graph schemas (curves, asymptotes)."""
        schemas = []
        
        # Extract curves
        curve_matches = re.findall(r"courbe ([A-Z])", text)
        for curve in curve_matches:
            schema = MathematicalSchema(
                schema_id=f"curve_{curve}",
                schema_type="graph",
                description=f"Courbe {curve}",
                elements=[f"courbe", f"function_{curve}"],
                equations=[],
                references=self._find_references(text, f"courbe {curve}"),
                confidence=0.8
            )
            schemas.append(schema)
        
        # Extract asymptotes
        asymptote_matches = re.findall(r"asymptote", text)
        if asymptote_matches:
            schema = MathematicalSchema(
                schema_id="asymptote",
                schema_type="graph",
                description="Asymptote",
                elements=["asymptote", "limite"],
                equations=[],
                references=self._find_references(text, "asymptote"),
                confidence=0.7
            )
            schemas.append(schema)
        
        return schemas
    
    def _extract_table_schemas(self, text: str) -> List[MathematicalSchema]:
        """Extract table schemas."""
        schemas = []
        
        # Extract tables
        table_matches = re.findall(r"tableau", text)
        if table_matches:
            schema = MathematicalSchema(
                schema_id="table",
                schema_type="table",
                description="Tableau de variation",
                elements=["tableau", "variation"],
                equations=[],
                references=self._find_references(text, "tableau"),
                confidence=0.6
            )
            schemas.append(schema)
        
        return schemas
    
    def _extract_figure_schemas(self, text: str) -> List[MathematicalSchema]:
        """Extract figure references."""
        schemas = []
        
        # Extract figure references
        figure_matches = re.findall(r"Figure (\d+)", text)
        for figure_num in figure_matches:
            schema = MathematicalSchema(
                schema_id=f"figure_{figure_num}",
                schema_type="figure",
                description=f"Figure {figure_num}",
                elements=[f"figure", f"annexe"],
                equations=[],
                references=self._find_references(text, f"Figure {figure_num}"),
                confidence=0.8
            )
            schemas.append(schema)
        
        return schemas
    
    def _find_references(self, text: str, pattern: str) -> List[str]:
        """Find references to a pattern in the text."""
        references = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                # Extract context around the reference
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = '\n'.join(lines[start:end])
                references.append(context.strip())
        
        return references
    
    def save_schemas(self, schemas: List[MathematicalSchema], output_path: Path) -> Dict[str, Any]:
        """
        Save extracted schemas to files.
        
        Args:
            schemas: List of extracted schemas
            output_path: Path to save schemas
            
        Returns:
            Dictionary with save results
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save schemas as JSON
        schemas_data = []
        for schema in schemas:
            schemas_data.append({
                "schema_id": schema.schema_id,
                "schema_type": schema.schema_type,
                "description": schema.description,
                "elements": schema.elements,
                "equations": schema.equations,
                "references": schema.references,
                "page_number": schema.page_number,
                "confidence": schema.confidence
            })
        
        schemas_file = output_path / "mathematical_schemas.json"
        with open(schemas_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_data, f, indent=2, ensure_ascii=False)
        
        # Save schemas by type
        schemas_by_type = {}
        for schema in schemas:
            if schema.schema_type not in schemas_by_type:
                schemas_by_type[schema.schema_type] = []
            schemas_by_type[schema.schema_type].append(schema)
        
        for schema_type, type_schemas in schemas_by_type.items():
            type_file = output_path / f"schemas_{schema_type}.json"
            type_data = []
            for schema in type_schemas:
                type_data.append({
                    "schema_id": schema.schema_id,
                    "description": schema.description,
                    "elements": schema.elements,
                    "equations": schema.equations,
                    "confidence": schema.confidence
                })
            
            with open(type_file, 'w', encoding='utf-8') as f:
                json.dump(type_data, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        summary = {
            "total_schemas": len(schemas),
            "schemas_by_type": {k: len(v) for k, v in schemas_by_type.items()},
            "files_generated": [
                str(schemas_file),
                *[str(output_path / f"schemas_{t}.json") for t in schemas_by_type.keys()]
            ]
        }
        
        summary_file = output_path / "schemas_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(schemas)} schemas to {output_path}")
        
        return summary


def extract_and_save_schemas(text: str, output_path: Path, source_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract and save mathematical schemas.
    
    Args:
        text: OCR-processed text
        output_path: Path to save schemas
        source_path: Path to source document
        
    Returns:
        Dictionary with extraction and save results
    """
    extractor = MathematicalSchemaExtractor()
    schemas = extractor.extract_schemas(text, source_path)
    summary = extractor.save_schemas(schemas, output_path)
    
    return {
        "schemas_extracted": len(schemas),
        "save_summary": summary,
        "schemas": schemas
    }


# Example usage
if __name__ == "__main__":
    # Test on Nougat output
    text_file = Path("qwen_vl_demo/nougat_test/math_pdf_nougat_result.txt")
    output_dir = Path("qwen_vl_demo/schemas_extraction")
    
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = extract_and_save_schemas(text, output_dir, str(text_file))
        
        print(f"Extracted {result['schemas_extracted']} mathematical schemas")
        print(f"Files saved to: {output_dir}")
        
        # Print summary
        summary = result['save_summary']
        print(f"Schemas by type: {summary['schemas_by_type']}")
    else:
        print(f"Text file {text_file} not found")
