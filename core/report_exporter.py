"""
AquaSentinel - Forensic Inspection Reporting & Export Engine
Generates publication-ready inspection dossiers and standardized audit JSONs
with cryptographic SHA-256 integrity verification and simulated geospatial waypoints.
"""

import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple


class ForensicReportExporter:
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def generate_verification_hash(self, report_data: Dict[str, Any]) -> str:
        serialized = json.dumps(report_data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def export_json_dossier(self, report: Dict[str, Any], mission_id: str = None) -> Tuple[str, Dict[str, Any]]:
        if not mission_id:
            mission_id = f"MISSION-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        report_payload = {
            "dossier_id": mission_id,
            "system": "AquaSentinel Underwater Vision-Language System",
            "benchmark_basis": "UWBench-2026",
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
            "simulated_coordinates": {
                "latitude": 18.2871,
                "longitude": -64.7938,
                "depth_meters": 14.5,
                "acoustic_beacon_id": "AUV-TRX-08"
            },
            "inspection_results": report,
        }
        report_payload["integrity_sha256"] = self.generate_verification_hash(report_payload)

        file_path = self.export_dir / f"{mission_id}_dossier.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        return str(file_path), report_payload

    def export_html_inspection_dossier(self, report: Dict[str, Any], mission_id: str = None) -> str:
        if not mission_id:
            mission_id = f"MISSION-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        json_path, payload = self.export_json_dossier(report, mission_id)

        entities_rows = ""
        for e in report.get("grounded_entities", []):
            entities_rows += f"""
            <tr>
                <td style="font-family: monospace; font-size: 11px;">{e['id']}</td>
                <td><strong>{e['name']}</strong></td>
                <td>{e['spatial_region']}</td>
                <td style="font-family: monospace;">{e['confidence']}</td>
                <td><span class="badge {e['hazard_level'].lower()}">{e['hazard_level']}</span></td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AquaSentinel Forensic Inspection Dossier - {mission_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #0F172A; line-height: 1.5; }}
        .header {{ border-bottom: 2px solid #0284C7; padding-bottom: 15px; margin-bottom: 25px; }}
        .title {{ font-size: 22px; font-weight: 700; color: #0F172A; margin: 0; }}
        .subtitle {{ font-size: 13px; color: #64748B; margin-top: 4px; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 15px; margin-bottom: 25px; font-size: 12px; }}
        .meta-label {{ color: #64748B; font-weight: 500; text-transform: uppercase; font-size: 10px; display: block; }}
        .meta-value {{ font-weight: 600; font-family: monospace; margin-top: 2px; }}
        .section-heading {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; margin: 25px 0 10px 0; border-bottom: 1px solid #E2E8F0; padding-bottom: 5px; }}
        .narrative-box {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 15px; font-size: 13px; white-space: pre-line; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }}
        th, td {{ border: 1px solid #E2E8F0; padding: 8px 12px; text-align: left; }}
        th {{ background: #F1F5F9; font-weight: 600; color: #475569; }}
        .badge {{ padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; }}
        .badge.critical {{ background: #FEE2E2; color: #DC2626; }}
        .badge.moderate, .badge.warning {{ background: #FEF3C7; color: #D97706; }}
        .badge.none {{ background: #DCFCE7; color: #16A34A; }}
        .protocol-box {{ background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 3px solid #16A34A; border-radius: 4px; padding: 12px; font-size: 12px; margin-top: 15px; }}
        .footer {{ margin-top: 40px; border-top: 1px solid #E2E8F0; padding-top: 15px; font-size: 10px; color: #94A3B8; display: flex; justify-content: space-between; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">AQUASENTINEL FORENSIC INSPECTION DOSSIER</h1>
        <div class="subtitle">Autonomous Underwater Vision-Language Anomaly & Habitat Diagnostic Analysis</div>
    </div>

    <div class="meta-grid">
        <div>
            <span class="meta-label">Dossier ID</span>
            <div class="meta-value">{mission_id}</div>
        </div>
        <div>
            <span class="meta-label">Timestamp</span>
            <div class="meta-value">{payload['generated_at']}</div>
        </div>
        <div>
            <span class="meta-label">Coordinates / Depth</span>
            <div class="meta-value">18.287°N, 64.793°W (14.5m)</div>
        </div>
        <div>
            <span class="meta-label">Severity Rating</span>
            <div class="meta-value" style="color: #DC2626;">{report.get('hazard_severity_score', 0)} / 10.0</div>
        </div>
    </div>

    <div class="section-heading">Executive Scene Assessment</div>
    <div class="narrative-box">{report.get('natural_language_synthesis', 'No narrative recorded.')}</div>

    <div class="section-heading">Identified Entities & Spatial Localization</div>
    <table>
        <thead>
            <tr>
                <th>Identifier</th>
                <th>Classification</th>
                <th>Spatial Region</th>
                <th>Confidence</th>
                <th>Risk Tier</th>
            </tr>
        </thead>
        <tbody>
            {entities_rows}
        </tbody>
    </table>

    <div class="section-heading">Actionable Operational Protocol</div>
    <div class="protocol-box">
        <strong>Strategic Intervention Directive:</strong><br>
        {report.get('strategic_recommendation', 'Maintain cruising altitude.')}
    </div>

    <div class="footer">
        <div>SHA-256 Verification: {payload['integrity_sha256']}</div>
        <div>AquaSentinel Marine Vision-Language Framework • UWBench-2026</div>
    </div>
</body>
</html>"""

        html_path = self.export_dir / f"{mission_id}_dossier.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(html_path)
