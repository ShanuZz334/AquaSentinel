"""
AquaSentinel - Integrated Vision-Language Pipeline
Orchestrates Physics-Based Optical Restoration, Visual Grounding with Domain Adapter,
Linguistic Reasoning, Spatial Risk Heatmaps, and Forensic Dossier Export.
"""

import os
from PIL import Image
from typing import Dict, Any, Tuple
from .optical_enhancer import UnderwaterOpticalEnhancer
from .vision_grounding import VisualGroundingEngine
from .mission_analyzer import MissionLanguageAnalyzer
from .report_exporter import ForensicReportExporter


class AquaSentinelPipeline:
    def __init__(self):
        self.enhancer = UnderwaterOpticalEnhancer()
        self.grounder = VisualGroundingEngine()
        self.analyzer = MissionLanguageAnalyzer()
        self.exporter = ForensicReportExporter()

    def process_mission(
        self,
        raw_image: Image.Image,
        mission_prompt: str,
        apply_enhancement: bool = True
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Dict[str, Any]]:
        """
        Executes the institutional multimodal pipeline:
        1. Physics-based underwater optical restoration (UDCP + Beer-Lambert + Color Constancy)
        2. Visual Grounding with Fine-Tuned Marine Domain Adapter
        3. Spatial Risk Density Heatmap generation
        4. Neural Autoregressive linguistic reasoning & severity scoring
        5. Forensic Dossier creation

        Returns:
            - Enhanced Image (PIL)
            - Grounded Annotated Image with HUD overlays (PIL)
            - Spatial Risk Heatmap Image (PIL)
            - Complete Mission Intelligence JSON Report
        """
        # Step 1: Physics Optical Restoration & Spectral Analytics
        if apply_enhancement:
            enhanced_image, optical_metrics = self.enhancer.enhance(raw_image)
        else:
            enhanced_image = raw_image.copy()
            optical_metrics = {
                "enhancement_bypassed": True,
                "optical_cast_reduction_pct": 0.0,
                "raw_histograms": self.enhancer.compute_spectral_histogram(enhanced_image),
                "enhanced_histograms": self.enhancer.compute_spectral_histogram(enhanced_image)
            }

        # Step 2: Visual Grounding Detection with Domain Adapter (executed on natural optical representation)
        detections = self.grounder.detect_and_ground(raw_image, mission_prompt)

        # Step 3: Draw Grounded HUD Boxes on Image (rendered onto enhanced frame)
        display_frame = enhanced_image if apply_enhancement else raw_image
        annotated_image = self.grounder.draw_grounded_boxes(display_frame, detections)

        # Step 4: Generate Spatial Hazard Density Heatmap
        heatmap_image = self.enhancer.generate_density_heatmap(display_frame, detections)

        # Step 5: Linguistic Reasoning & Report Synthesis
        report = self.analyzer.generate_report(mission_prompt, detections, optical_metrics)

        # Step 6: Export Forensic Dossier
        try:
            dossier_html_path = self.exporter.export_html_inspection_dossier(report)
            dossier_filename = os.path.basename(dossier_html_path)
            report["dossier_export_url"] = f"/api/exports/{dossier_filename}"
        except Exception as e:
            report["dossier_export_url"] = None

        return enhanced_image, annotated_image, heatmap_image, report
