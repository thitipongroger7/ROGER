# ==============================================================================
# FILE: main.py
# PURPOSE: FastAPI Backend — CUI Prediction (RF + Bayesian Prior + Conformal)
# ==============================================================================

import os
import json
import threading
import time
import requests as req_lib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from supabase import create_client

from model import ModelManager

app = FastAPI(title="CUI Corrosion Prediction API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager()

SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "admin").strip()
LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "cui2024").strip()


def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[Supabase] connect error: {e}")
        return None


# ==============================================================================
# Static Pages
# ==============================================================================
@app.get("/",                      include_in_schema=False)
def serve_login():       return FileResponse(os.path.join(os.getcwd(), "login.html"))

@app.get("/login.html",            include_in_schema=False)
def serve_login_page():  return FileResponse(os.path.join(os.getcwd(), "login.html"))

@app.get("/cui_prediction.html",   include_in_schema=False)
def serve_main():        return FileResponse(os.path.join(os.getcwd(), "cui_prediction.html"))

@app.get("/upload_history.html",   include_in_schema=False)
def serve_upload():      return FileResponse(os.path.join(os.getcwd(), "upload_history.html"))

@app.get("/result.html",           include_in_schema=False)
def serve_result():      return FileResponse(os.path.join(os.getcwd(), "result.html"))

@app.get("/gcme_logo.png",         include_in_schema=False)
def serve_gcme():        return FileResponse(os.path.join(os.getcwd(), "gcme_logo.png"))

@app.get("/gcme_logo_nobg.png",    include_in_schema=False)
def serve_gcme_nobg():   return FileResponse(os.path.join(os.getcwd(), "gcme_logo_nobg.png"))

@app.get("/chula_logo.png",        include_in_schema=False)
def serve_chula():       return FileResponse(os.path.join(os.getcwd(), "chula_logo.png"))

@app.get("/chula_logo_nobg.png",   include_in_schema=False)
def serve_chula_nobg():  return FileResponse(os.path.join(os.getcwd(), "chula_logo_nobg.png"))

@app.get("/hero_bg.jpg",           include_in_schema=False)
def serve_hero():        return FileResponse(os.path.join(os.getcwd(), "hero_bg.jpg"))

@app.get("/refinery_new.png",      include_in_schema=False)
def serve_refinery():    return FileResponse(os.path.join(os.getcwd(), "refinery_new.png"))


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
    model:                   str = "hybrid"  # "rf" หรือ "hybrid"
    Environment:             str
    Jacket_Damage:           str
    model:                   str = "hybrid"  # "rf" or "hybrid"

class LoginRequest(BaseModel):
    username: str
    password: str

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
# ENDPOINT: Health
# ==============================================================================
@app.head("/")
def head_root():
    return {}

@app.get("/health")
def health_check():
    return {
        "status":        "ok",
        "model_trained": manager.is_trained(),
        "message":       "CUI Prediction API v5.0 is running",
    }


# ==============================================================================
# ENDPOINT: Login
# ==============================================================================
@app.post("/api/login")
def login(data: LoginRequest):
    if data.username.strip() == LOGIN_USERNAME and data.password.strip() == LOGIN_PASSWORD:
        return {"success": True, "message": "Login สำเร็จ"}
    raise HTTPException(status_code=401, detail="Username หรือ Password ไม่ถูกต้อง")


# ==============================================================================
# ENDPOINT: Upload → Train
# ==============================================================================
@app.post("/api/upload")
async def upload_and_train(file: UploadFile = File(...)):
    filename = file.filename
    if not (filename.endswith(".xlsx") or filename.endswith(".csv")):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ .xlsx หรือ .csv")

    ext       = os.path.splitext(filename)[1]
    save_path = os.path.join(os.getcwd(), "final_training_data" + ext)
    content   = await file.read()
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
# ENDPOINT: Predict
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
        result     = manager.predict(input_dict, model=data.model)

        # ── Unknown ──────────────────────────────────────────────────
        if result["prediction"] == "Unknown":
            return {
                "status":      "success",
                "prediction":  "Unknown",
                "unknown_cols": result["unknown_cols"],
                "message":     "ข้อมูลบางรายการไม่เคยอยู่ใน training data — ไม่สามารถทำนายได้",
                "input_summary": input_dict,
            }

        # ── Prediction Set → Final Result ─────────────────────────────
        pred_set   = result["pred_set"] or ("{Yes}" if result["prediction"] == "Yes" else "{No}")
        bayes_pct  = result["bayes_prob"] if result["bayes_prob"] is not None else result["rf_prob"]
        prediction = result["prediction"]

        # Risk level จาก bayes_pct
        if bayes_pct >= 60:   risk_level = "High"
        elif bayes_pct >= 30: risk_level = "Medium"
        else:                 risk_level = "Low"

        # Tier & Action
        tier, action, criteria = _tier_action(pred_set, risk_level, bayes_pct)

        # Next inspection year
        current_year = datetime.now().year
        next_yr = current_year + (1 if tier == 1 else 3 if tier == 2 else 5) if tier else current_year + 5

        return {
            "status":               "success",
            "prediction":           prediction,
            "pred_set":             pred_set,
            "rf_probability":       result["rf_prob"],
            "prior_probability":    result["prior_prob"],
            "bayes_probability":    bayes_pct,
            "risk_level":           risk_level,
            "tier":                 tier,
            "action":               action,
            "tier_criteria":        criteria,
            "q_value":              result["q_value"],
            "next_inspection_year": next_yr,
            "input_summary":        input_dict,
            # Legacy fields (สำหรับ result.html)
            "probability":          result["rf_prob"] / 100,
            "probability_pct":      bayes_pct,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")


def _tier_action(pred_set: str, risk_level: str, bayes_pct: float):
    if pred_set == "{No}":
        return 3, "ตาม Scheduled Inspection (Follow Standard Schedule)", \
               "Prediction Set = {No} — Confident No CUI"
    if pred_set == "{Yes}":
        return 1, "ตรวจสอบทันที (Immediate Inspection)", \
               "Prediction Set = {Yes} — Confident CUI Detected"
    # {Yes,No} หรือ {} → Safety-first → Yes
    if risk_level == "High":
        return 1, "ตรวจสอบทันที (Immediate Inspection)", \
               "Prediction Set = Uncertain · Risk High · Bayes ≥ 60%"
    return 2, "เฝ้าระวัง + ตรวจรอบถัดไป (Monitor & Inspect Next Cycle)", \
           f"Prediction Set = {pred_set} · Safety-first · Bayes = {bayes_pct}%"


# ==============================================================================
# ENDPOINT: Stats
# ==============================================================================
@app.get("/api/stats")
def get_stats():
    return manager.get_stats()


# ==============================================================================
# ENDPOINT: History
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


@app.get("/api/history")
def get_history():
    sb = get_supabase()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not connected")
    try:
        files   = sb.storage.from_("models").list("history")
        records = []
        for f in files:
            try:
                data = sb.storage.from_("models").download(f"history/{f['name']}")
                records.append(json.loads(data.decode()))
            except Exception as e:
                print(f"[history] skip {f['name']}: {e}")
        records.sort(key=lambda x: x.get("id", 0), reverse=True)
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")


@app.patch("/api/history/{record_id}/note")
def update_note(record_id: int, body: NoteUpdate):
    sb = get_supabase()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase ไม่พร้อมใช้งาน")
    try:
        data   = sb.storage.from_("models").download(f"history/{record_id}.json")
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
# Keep Alive
# ==============================================================================
def _self_ping():
    time.sleep(60)
    while True:
        try:
            req_lib.get("https://mtcu-cui.onrender.com/health", timeout=15)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=_self_ping, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
