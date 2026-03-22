# ============================================================================== FileResponse
from pydantic import BaseModel
# FILE: main.py
# PURPOSE: FastAPI Backend — รับข้อมูลจาก UI แล้วส่งให้ ML Model ทำนาย
# ==============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from typing import Optional
import uvicorn
import os
import json
from datetime import datetime
from supabase import create_client

from model import ModelManager

app = FastAPI(title="CUI Corrosion Prediction API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

# Serve static files (HTML, images, etc.)
@app.get("/cui_prediction.html", include_in_schema=False)
def serve_main(): return FileResponse(os.path.join(os.getcwd(), "cui_prediction.html"))

@app.get("/upload_history.html", include_in_schema=False)
def serve_upload(): return FileResponse(os.path.join(os.getcwd(), "upload_history.html"))

@app.get("/result.html", include_in_schema=False)
def serve_result(): return FileResponse(os.path.join(os.getcwd(), "result.html"))

@app.get("/gcme_logo.png", include_in_schema=False)
def serve_gcme(): return FileResponse(os.path.join(os.getcwd(), "gcme_logo.png"))

@app.get("/chula_logo.png", include_in_schema=False)
def serve_chula(): return FileResponse(os.path.join(os.getcwd(), "chula_logo.png"))

@app.get("/hero_bg.jpg", include_in_schema=False)
def serve_hero(): return FileResponse(os.path.join(os.getcwd(), "hero_bg.jpg"))


# ==============================================================================
# Schema
# ==============================================================================
class PredictRequest(BaseModel):
    Substrate:               str
    Age_Years:               float
    Operating_Temperature_C: str
    Coating_Prime:           str
    Coating_Second:          str
    Top_Coat:                str
    Insulation_Type:         str
    Vapor_Barrier:           str
    Environment:             str
    Jacket_Damage:           str

class HistoryRecord(BaseModel):
    id:     int
    date:   str
    time:   str
    input:  dict
    result: dict
    note:   Optional[str] = ""

class NoteUpdate(BaseModel):
    note: str


# ==============================================================================
# ENDPOINT 1: Health Check
# ==============================================================================
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_trained": manager.is_trained(),
        "message": "CUI Prediction API is running"
    }


# ==============================================================================
# ENDPOINT 2: Upload Excel → Train Model
# ==============================================================================
@app.post("/api/upload")
async def upload_and_train(file: UploadFile = File(...)):
    filename = file.filename
    if not (filename.endswith(".xlsx") or filename.endswith(".csv")):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ .xlsx หรือ .csv เท่านั้น")

    save_path = os.path.join(os.getcwd(), "final_training_data" + os.path.splitext(filename)[1])
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        result = manager.train(save_path)
        return {
            "status":         "success",
            "message":        f"Train โมเดลสำเร็จจากไฟล์ '{filename}'",
            "accuracy":       result["accuracy"],
            "total_rows":     result["total_rows"],
            "features_count": result["features_count"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการ Train: {str(e)}")


# ==============================================================================
# ENDPOINT 3: Predict
# ==============================================================================
@app.post("/api/predict")
def predict(data: PredictRequest):
    if not manager.is_trained():
        raise HTTPException(
            status_code=400,
            detail="โมเดลยังไม่ได้ถูก Train กรุณา Upload ไฟล์ข้อมูลก่อน"
        )
    try:
        input_dict = data.dict()
        result     = manager.predict(input_dict)
        return {
            "status":               "success",
            "prediction":           result["prediction"],
            "probability":          round(result["probability"], 4),
            "probability_pct":      round(result["probability"] * 100, 2),
            "risk_level":           result["risk_level"],
            "next_inspection_year": result["next_inspection_year"],
            "confidence_pct":       result["confidence_pct"],
            "input_summary":        input_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")


# ==============================================================================
# ENDPOINT 4: Stats
# ==============================================================================
@app.get("/api/stats")
def get_stats():
    return manager.get_stats()


# ==============================================================================
# ENDPOINT 5: History — บันทึกประวัติ
# ==============================================================================
@app.post("/api/history")
def save_history(record: HistoryRecord):
    sb = get_supabase()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase ไม่พร้อมใช้งาน")
    try:
        sb.storage.from_("models").upload(
            f"history/{record.id}.json",
            json.dumps(record.dict()).encode(),
            {"upsert": "true", "content-type": "application/json"}
        )
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINT 6: History — ดึงประวัติทั้งหมด
# ==============================================================================
@app.get("/api/history")
def get_history():
    sb = get_supabase()
    if not sb:
        return []
    try:
        files = sb.storage.from_("models").list("history")
        records = []
        for f in files:
            try:
                data = sb.storage.from_("models").download(f"history/{f['name']}")
                records.append(json.loads(data.decode()))
            except Exception:
                continue
        records.sort(key=lambda x: x.get("id", 0), reverse=True)
        return records
    except Exception:
        return []


# ==============================================================================
# ENDPOINT 7: History — อัพเดต note
# ==============================================================================
@app.patch("/api/history/{record_id}/note")
def update_note(record_id: int, body: NoteUpdate):
    sb = get_supabase()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase ไม่พร้อมใช้งาน")
    try:
        data = sb.storage.from_("models").download(f"history/{record_id}.json")
        record = json.loads(data.decode())
        record["note"] = body.note
        sb.storage.from_("models").upload(
            f"history/{record_id}.json",
            json.dumps(record).encode(),
            {"upsert": "true", "content-type": "application/json"}
        )
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINT 8: History — ลบประวัติ
# ==============================================================================
@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    sb = get_supabase()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase ไม่พร้อมใช้งาน")
    try:
        sb.storage.from_("models").remove([f"history/{record_id}.json"])
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Run Server
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
