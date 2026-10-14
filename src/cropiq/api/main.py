"""FastAPI service.

    CROPIQ_CONFIG=configs/default.yaml uvicorn cropiq.api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health               service + model availability
    POST /predict/image        multipart file -> disease, severity, anomaly score
    POST /predict/irrigation   JSON {"readings": [24+ hourly {soil_moisture, temperature, humidity}]}
    POST /advisory             multipart: file (optional) + readings (JSON string, optional) -> fused advisory
"""
from __future__ import annotations

import io
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from cropiq.config import load_config
from cropiq.pipeline.inference import CropAdvisor, ModelUnavailable

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class Reading(BaseModel):
    soil_moisture: float = Field(..., ge=-5, le=100, description="% volumetric water content")
    temperature: float = Field(..., ge=-20, le=70, description="air temperature, deg C")
    humidity: float = Field(..., ge=0, le=100, description="relative humidity, %")


class IrrigationRequest(BaseModel):
    readings: list[Reading] = Field(..., min_length=1, description="hourly readings, oldest first")


def create_app(config_path: str | None = None, overrides: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="Crop Health & Irrigation Intelligence", version="0.1.0")
    app.state.advisor = None

    def advisor() -> CropAdvisor:
        if app.state.advisor is None:
            path = config_path or os.environ.get("CROPIQ_CONFIG", "configs/default.yaml")
            app.state.advisor = CropAdvisor(load_config(path, overrides))
        return app.state.advisor

    async def read_image(file: UploadFile) -> Image.Image:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image too large (max 8 MB).")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Could not decode the uploaded file as an image.")

    def readings_or_422(raw: str) -> list[dict]:
        try:
            payload = json.loads(raw)
            items = payload["readings"] if isinstance(payload, dict) else payload
            return [Reading(**r).model_dump() for r in items]
        except Exception as exc:
            raise HTTPException(422, f"Invalid readings: {exc}")

    @app.get("/health")
    def health():
        a = advisor()
        return {"status": "ok", "models": a.available()}

    @app.post("/predict/image")
    async def predict_image(file: UploadFile = File(...)):
        img = await read_image(file)
        try:
            out = advisor().predict_image(img)
        except ModelUnavailable as exc:
            raise HTTPException(503, f"Model unavailable: {exc}")
        out.pop("all_probs", None)
        return out

    @app.post("/predict/irrigation")
    def predict_irrigation(req: IrrigationRequest):
        try:
            return advisor().predict_irrigation([r.model_dump() for r in req.readings])
        except ModelUnavailable as exc:
            raise HTTPException(503, f"Model unavailable: {exc}")
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/advisory")
    async def advisory(file: UploadFile | None = File(None), readings: str | None = Form(None)):
        if file is None and readings is None:
            raise HTTPException(422, "Send a leaf image (file), sensor readings, or both.")
        img = await read_image(file) if file is not None else None
        data = readings_or_422(readings) if readings else None
        try:
            return advisor().advise(img, data)
        except ModelUnavailable as exc:
            raise HTTPException(503, f"Model unavailable: {exc}")
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    return app


app = create_app()
