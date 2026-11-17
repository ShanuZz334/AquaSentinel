"""
AquaSentinel - Main Web Application Server
Natural-Language Guided Underwater Anomaly and Scene Analysis System
Institutional Grade • Powered by Real Pre-Trained & Fine-Tuned Models
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core.pipeline import AquaSentinelPipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
EXPORTS_DIR = BASE_DIR / "exports"

app = FastAPI(
    title="AquaSentinel",
    description="Underwater Vision-Language Mission Analysis System",
    version="2.5.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize Pipeline with real models & fine-tuned adapter
pipeline = AquaSentinelPipeline()


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    return FileResponse(html_path)


@app.get("/api/samples/{filename}")
async def get_sample_image(filename: str):
    file_path = SAMPLES_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Sample image not found"})


@app.get("/api/exports/{filename}")
async def get_exported_dossier(filename: str):
    file_path = EXPORTS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Dossier not found"})


@app.get("/api/benchmarks")
async def get_benchmarks():
    benchmark_file = DATA_DIR / "benchmark_data.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            return json.load(f)
    return JSONResponse(status_code=404, content={"error": "Benchmark data not found"})


@app.get("/api/presets")
async def get_presets():
    return {
        "presets": [
            {
                "id": "ghost_net",
                "title": "Ghost Net Hazard",
                "filename": "ghost_net_reef.jpg",
                "environment": "Coral Reef (12m)",
                "recommended_prompt": "Identify objects that could be harmful to marine life and ocean fauna."
            },
            {
                "id": "sunken_drum",
                "title": "Industrial Container",
                "filename": "sunken_drum_debris.jpg",
                "environment": "Benthic Sediment (35m)",
                "recommended_prompt": "Perform an anomaly scan and detect anthropogenic waste or hazard containers."
            },
            {
                "id": "coral_bleaching",
                "title": "Bleached Substrate",
                "filename": "coral_bleaching_reef.jpg",
                "environment": "Shallow Atoll (6m)",
                "recommended_prompt": "Assess coral reef health and identify signs of bleaching or degraded substrate."
            }
        ],
        "adapter_status": {
            "loaded": pipeline.grounder.adapter_active,
            "weights_file": "weights/aquasentinel_marine_adapter.pt",
            "model_architecture": "Vision Transformer + Autoregressive Language Model"
        }
    }


@app.post("/api/analyze")
async def analyze_mission(
    image: Optional[UploadFile] = File(None),
    preset_filename: Optional[str] = Form(None),
    mission_prompt: str = Form("Identify objects that could be harmful to marine life."),
    apply_enhancement: bool = Form(True)
):
    try:
        if image and image.filename:
            contents = await image.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        elif preset_filename:
            file_path = SAMPLES_DIR / preset_filename
            if not file_path.exists():
                return JSONResponse(status_code=400, content={"error": f"Preset {preset_filename} not found"})
            pil_img = Image.open(file_path).convert("RGB")
        else:
            file_path = SAMPLES_DIR / "ghost_net_reef.jpg"
            pil_img = Image.open(file_path).convert("RGB")

        # Execute Full Pipeline
        enhanced_pil, annotated_pil, heatmap_pil, report = pipeline.process_mission(
            raw_image=pil_img,
            mission_prompt=mission_prompt,
            apply_enhancement=apply_enhancement
        )

        return {
            "success": True,
            "raw_image_b64": pil_to_base64(pil_img),
            "enhanced_image_b64": pil_to_base64(enhanced_pil),
            "annotated_image_b64": pil_to_base64(annotated_pil),
            "heatmap_image_b64": pil_to_base64(heatmap_pil),
            "report": report
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/optical-enhance")
async def optical_lab_tune(
    preset_filename: Optional[str] = Form("ghost_net_reef.jpg"),
    red_gain: float = Form(1.35),
    contrast_factor: float = Form(1.25),
    blue_damping: float = Form(0.92)
):
    try:
        file_path = SAMPLES_DIR / preset_filename
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"error": "Sample image not found"})

        pil_img = Image.open(file_path).convert("RGB")
        enhancer = pipeline.enhancer
        enhancer.red_gain = red_gain
        enhancer.contrast_factor = contrast_factor
        enhancer.blue_damping = blue_damping

        enhanced_pil, metrics = enhancer.enhance(pil_img)
        return {
            "success": True,
            "enhanced_b64": pil_to_base64(enhanced_pil),
            "metrics": metrics
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(">> AQUASENTINEL: Underwater Vision-Language Mission Assistant")
    print(">> Dashboard URL: http://localhost:7860")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860)
