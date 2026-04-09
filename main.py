# ==============================================================================
# FILE: main.py
# PURPOSE: FastAPI Backend — รับข้อมูลจาก UI แล้วส่งให้ ML Model ทำนาย
#          V4: เพิ่ม API 583 Bayesian Prior + Conformal Prediction
# ==============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import json
import math
import threading
import time
import requests as req_lib
from datetime import datetime
from supabase import create_client

from model import ModelManager

app = FastAPI(title="CUI Corrosion Prediction API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ตั้งค่า Username และ Password ที่นี่เลยครับ
LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "admin").strip()
LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "cui2024").strip()

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

# Serve static files (HTML, images, etc.)
@app.get("/", include_in_schema=False)
def serve_login(): return FileResponse(os.path.join(os.getcwd(), "login.html"))

@app.get("/login.html", include_in_schema=False)
def serve_login_page(): return FileResponse(os.path.join(os.getcwd(), "login.html"))

@app.get("/cui_prediction.html", include_in_schema=False)
def serve_main(): return FileResponse(os.path.join(os.getcwd(), "cui_prediction.html"))

@app.get("/upload_history.html", include_in_schema=False)
def serve_upload(): return FileResponse(os.path.join(os.getcwd(), "upload_history.html"))

@app.get("/result.html", include_in_schema=False)
def serve_result(): return FileResponse(os.path.join(os.getcwd(), "result.html"))

@app.get("/gcme_logo.png", include_in_schema=False)
def serve_gcme(): return FileResponse(os.path.join(os.getcwd(), "gcme_logo.png"))

@app.get("/gcme_logo_nobg.png", include_in_schema=False)
def serve_gcme_nobg(): return FileResponse(os.path.join(os.getcwd(), "gcme_logo_nobg.png"))

@app.get("/chula_logo.png", include_in_schema=False)
def serve_chula(): return FileResponse(os.path.join(os.getcwd(), "chula_logo.png"))

@app.get("/chula_logo_nobg.png", include_in_schema=False)
def serve_chula_nobg(): return FileResponse(os.path.join(os.getcwd(), "chula_logo_nobg.png"))

@app.get("/hero_bg.jpg", include_in_schema=False)
def serve_hero(): return FileResponse(os.path.join(os.getcwd(), "hero_bg.jpg"))

@app.get("/refinery_new.png", include_in_schema=False)
def serve_refinery(): return FileResponse(os.path.join(os.getcwd(), "refinery_new.png"))


# ==============================================================================
# API 583 BAYESIAN PRIOR  (ความรู้จากมาตรฐาน — independent จาก training data)
# ==============================================================================

_BASE_RATE    = 0.25   # API 583 §4 — population base rate
_BASE_LO      = math.log(_BASE_RATE / (1 - _BASE_RATE))   # −1.099

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def _logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))

# Temperature zone — API 583 §5.2
_TEMP_ADJ = {
    "0 - <10": 0.8,  "10 - <20": 0.8,  "20 - <30": 0.8,
    "30 - <40": 0.8, "40 - <50": 0.8,  "50 - <60": 0.8,
    "60 - <70": 1.2, "70 - <80": 1.2,  "80 - <90": 1.2,
    "90 - <100": 1.2,"100 - <110": 1.2,"110 - <120": 1.2,
    "120 - <130": 0.3,"130 - <140": 0.3,
    "140 - <150": -0.5,"150 - <160": -1.5,
    "160 - <170": -3.0,"170 - <180": -3.0,
}
# Substrate — API 571 §5.1.2
_SUB_ADJ = {"cs": 0.0, "ltcs": 0.0, "low alloy steel": 0.0, "ss": -2.5}
# Insulation — API 583 §5.3
_INS_ADJ = {
    "polyisocyanurate": 0.4, "mineral fiber": 0.3, "mineral wool": 0.3,
    "perlite": 0.2, "calcium silicate": 0.2,
    "cellular glass": -0.5, "flexible aerogel blanket": -0.6, "aerogel": -0.6,
}
# Prime coating — API 583 §5.4
_PRIME_ADJ = {
    "zinc ethyl silicate": 0.7, "rich zinc epoxy": 0.5, "inorganic zinc": 0.4,
    "phenolic epoxy": 0.0, "epoxy": -0.5, "non primer": 0.6,
    "silicone primer": -0.3, "inorganic copolymer": 0.4,
}
# Top coat — API 583 §5.4
_TOP_ADJ = {
    "zinc silicate": 0.5, "non top coat": 0.4, "phenolic epoxy": 0.2,
    "polyurethane": -0.3, "2 component polyurethane": -0.6,
    "silicone aluminum": -0.2, "epoxy": -0.4, "epoxy polyamide": -0.3,
    "fbe": -0.5,
}
# Conformal q values (calibrated from test set)
_Q = {95: 0.847, 90: 0.745, 85: 0.691}


def api583_prior(inp: dict) -> float:
    """คำนวณ API 583 Prior probability จาก input dict."""
    lo = _BASE_LO

    # Temperature
    temp_key = str(inp.get("Operating_Temperature_C", "")).strip().lower()
    lo += _TEMP_ADJ.get(temp_key, 0.8)

    # Substrate
    lo += _SUB_ADJ.get(str(inp.get("Substrate", "")).strip().lower(), 0.0)

    # Insulation
    lo += _INS_ADJ.get(str(inp.get("Insulation_Type", "")).strip().lower(), 0.2)

    # Coating
    lo += _PRIME_ADJ.get(str(inp.get("Coating_Prime", "")).strip().lower(), 0.2)
    lo += _TOP_ADJ.get(str(inp.get("Top_Coat", "")).strip().lower(), 0.2)

    # Jacket damage — API 583 §5.5
    jacket = str(inp.get("Jacket_Damage", "")).strip().lower()
    if jacket == "yes":  lo += 0.9
    elif jacket == "no": lo -= 0.4

    # Age
    try:
        age = float(inp.get("Age_Years", 0))
        if age > 30:    lo += 0.8
        elif age > 20:  lo += 0.4
        elif age > 10:  lo += 0.1
        elif age < 5:   lo -= 0.3
    except (ValueError, TypeError):
        pass

    # Environment — API 571 §5.1.3
    if "marine" in str(inp.get("Environment", "")).lower():
        lo += 0.3

    return _sigmoid(lo)


def bayesian_blend(prior_p: float, rf_p: float) -> float:
    """รวม Prior + RF ใน log-odds space (50/50)."""
    return _sigmoid(0.5 * _logit(prior_p) + 0.5 * _logit(rf_p))


def conformal_predict(bayes_p: float, confidence: int) -> str:
    """Return prediction set: 'Yes', 'No', หรือ 'No+Yes'."""
    q = _Q.get(confidence, 0.847)
    in_yes = (1 - bayes_p) <= q
    in_no  = bayes_p <= q
    if in_yes and in_no:  return "No+Yes"
    if in_yes:            return "Yes"
    if in_no:             return "No"
    return "No+Yes"   # Empty set → conservative


def tier_action(pred_set: str, risk_level: str, bayes_pct: float):
    """คำนวณ Tier และ Action จาก Prediction Set + Risk Level."""
    final_pred = "Yes" if pred_set in ("Yes", "No+Yes") else "No"
    if final_pred == "Yes":
        if pred_set == "Yes" or risk_level == "High" or bayes_pct >= 50:
            return 1, "ตรวจสอบทันที (Immediate Inspection)", \
                   "Pred Set = {Yes} หรือ Risk High หรือ Bayes ≥ 50%"
        return 2, "เฝ้าระวัง + ตรวจรอบถัดไป (Monitor & Inspect Next Cycle)", \
               "Pred Set = {No+Yes} · Risk Medium · Bayes 30–50%"
    return 3, "ตาม Scheduled Inspection (Follow Standard Schedule)", \
           "Pred Set = {No} · Risk Low · Bayes < 30%"


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
    confidence:              Optional[int] = 95   # ← ใหม่: 85 / 90 / 95

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
# ENDPOINT 1: Health Check
# ==============================================================================
@app.head("/")
def head_root():
    return {}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_trained": manager.is_trained(),
        "message": "CUI Prediction API v4.0 is running"
    }


# ==============================================================================
# ENDPOINT LOGIN
# ==============================================================================
@app.post("/api/login")
def login(data: LoginRequest):
    if data.username.strip() == LOGIN_USERNAME and data.password.strip() == LOGIN_PASSWORD:
        return {"success": True, "message": "Login สำเร็จ"}
    raise HTTPException(status_code=401, detail="Username หรือ Password ไม่ถูกต้อง")


# ==============================================================================
# ENDPOINT 2: Upload → Train Model  (ไม่เปลี่ยนแปลง)
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
# ENDPOINT 3: Predict  (เพิ่ม V4 fields)
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
        confidence = int(input_dict.pop("confidence", 95))
        if confidence not in (85, 90, 95):
            confidence = 95

        # ── RF prediction (จาก model.py เดิม) ──
        result = manager.predict(input_dict)
        rf_p   = float(result["probability"])   # 0–1

        # ── API 583 Bayesian Prior ──
        prior_p = api583_prior(input_dict)

        # ── Bayesian Blend ──
        bayes_p = bayesian_blend(prior_p, rf_p)
        bayes_pct = round(bayes_p * 100, 1)

        # ── Risk Level ──
        if bayes_pct >= 60:   risk_level = "High"
        elif bayes_pct >= 30: risk_level = "Medium"
        else:                 risk_level = "Low"

        # ── Conformal Prediction ──
        pred_set = conformal_predict(bayes_p, confidence)

        # ── Tier & Action ──
        tier, action, criteria = tier_action(pred_set, risk_level, bayes_pct)

        # ── Next inspection year ──
        current_year = datetime.now().year
        next_yr = current_year + (1 if tier == 1 else 3 if tier == 2 else 5)

        response = {
            # ── V4 fields (ใหม่) ──
            "rf_probability":    round(rf_p * 100, 1),
            "prior_probability": round(prior_p * 100, 1),
            "bayes_probability": bayes_pct,
            "pred_set":          pred_set,
            "risk_level":        risk_level,
            "tier":              tier,
            "action":            action,
            "tier_criteria":     criteria,
            "confidence":        confidence,
            "q_value":           _Q.get(confidence, 0.847),
            # ── Legacy fields (เดิม — ให้ result.html ยังทำงานได้) ──
            "status":               "success",
            "prediction":           result["prediction"],
            "probability":          round(rf_p, 4),
            "probability_pct":      bayes_pct,   # ใช้ Bayes แทน RF เดิม
            "next_inspection_year": next_yr,
            "confidence_pct":       result.get("confidence_pct", bayes_pct),
            "input_summary":        input_dict,
        }

        # ── บันทึก Supabase อัตโนมัติ ──
        try:
            sb = get_supabase()
            if sb:
                now = datetime.now()
                record_id = int(now.timestamp() * 1000)
                record = {
                    "id":   record_id,
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "input": input_dict,
                    "result": {
                        "rf_probability":    response["rf_probability"],
                        "prior_probability": response["prior_probability"],
                        "bayes_probability": response["bayes_probability"],
                        "pred_set":          pred_set,
                        "risk_level":        risk_level,
                        "tier":              tier,
                        "action":            action,
                        "next_inspection_year": next_yr,
                    },
                    "note": ""
                }
                sb.storage.from_("models").upload(
                    f"history/{record_id}.json",
                    json.dumps(record).encode(),
                    {"upsert": "true", "content-type": "application/json"}
                )
        except Exception:
            pass

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")


# ==============================================================================
# ENDPOINT 4: Stats  (ไม่เปลี่ยนแปลง)
# ==============================================================================
@app.get("/api/stats")
def get_stats():
    return manager.get_stats()


# ==============================================================================
# ENDPOINT 5–8: History  (ไม่เปลี่ยนแปลง)
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
        files = sb.storage.from_("models").list("history")
        records = []
        for f in files:
            try:
                data = sb.storage.from_("models").download(f"history/{f['name']}")
                records.append(json.loads(data.decode()))
            except Exception as e:
                print(f"[history] skip {f['name']}: {e}")
                continue
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
def _self_ping():
    """Keep Render free tier alive by pinging every 10 minutes."""
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
