# ==============================================================================
# FILE: model.py
# PURPOSE: ML Logic — อิงจาก ML_model6.py
#          - Random Forest + LabelEncoder (ไม่ใช้ One-Hot)
#          - Oversampling minority class
#          - Threshold 0.50
# ==============================================================================

import re
import os
import pickle
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = os.path.join(os.getcwd(), "cui_model.pkl")
THRESHOLD       = 0.50
N_ESTIMATORS    = 500
RANDOM_STATE    = 42

CAT_COLS = [
    "Substrate", "Coating_Prime", "Coating_Second",
    "Top_Coat", "Insulation_Type", "Vapor_Barrier",
    "Environment", "Jacket_Damage",
]


# ==============================================================================
# Helper Functions
# ==============================================================================
def parse_corrosion_result(val: str) -> float:
    val = str(val).strip()
    if val.lower() == "yes":
        return 1.0
    if val.lower() == "no":
        return 0.0
    m = re.search(r"Yes\(([0-9.]+)%\)", val, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0
    return 0.0


def encode_temperature(t: str) -> float:
    t = str(t).strip()
    if "มากกว่า" in t or ">= 200" in t or "≥ 200" in t:
        return 205.0
    if "Cyclic" in t.lower():
        return 100.0
    m = re.match(r"\(?(-?\d+)\)?\s*-\s*<?(-?\d+)", t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2.0
    return 50.0


def risk_label(prob: float) -> str:
    if prob >= 0.60:
        return "CRITICAL"
    if prob >= 0.30:
        return "WARNING"
    return "SAFE"


def next_inspection_year(prob: float) -> int:
    year = datetime.now().year
    if prob >= 0.60:
        return year + 1
    if prob >= 0.30:
        return year + 3
    return year + 5


# ==============================================================================
# ModelManager
# ==============================================================================
class ModelManager:
    def __init__(self):
        self.model    = None
        self.encoders = {}
        self.stats = {
            "total_datasets": 0,
            "area_counts":    {},
            "last_trained":   None,
            "accuracy":       None,
            "threshold":      THRESHOLD,
        }
        self._load_model()

    def is_trained(self) -> bool:
        return self.model is not None and len(self.encoders) > 0

    # ------------------------------------------------------------------
    def train(self, file_path: str) -> dict:
        print(f"\n[ModelManager] เริ่ม Train จากไฟล์: {file_path}")

        # 1. โหลดข้อมูล
        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            for enc in ['utf-8', 'utf-8-sig', 'cp874', 'tis-620', 'latin-1']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except Exception:
                    continue

        print(f"  โหลดข้อมูลดิบ: {len(df)} แถว")

        # 2. เติมค่าว่าง categorical
        for col in CAT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("Unknow").astype(str).str.strip()
            else:
                df[col] = "Unknow"

        # 3. Parse target
        df["yes_prob"] = df["Corrosion_Result"].apply(parse_corrosion_result)
        df["label"]    = (df["yes_prob"] >= THRESHOLD).astype(int)
        df = df.dropna(subset=["label"])
        print(f"  หลังล้าง Target: {len(df)} แถว")

        # 4. Encode temperature
        df["temp_num"] = df["Operating_Temperature_C"].apply(encode_temperature)

        # 5. Fit LabelEncoders
        self.encoders = {}
        for col in CAT_COLS:
            le = LabelEncoder()
            le.fit(df[col].str.lower())
            self.encoders[col] = le

        # 6. สร้าง Feature Matrix
        X = self._build_features(df)
        y = df["label"]

        label_counts = y.value_counts()
        print(f"  Label distribution → Yes: {label_counts.get(1,0)}, No: {label_counts.get(0,0)}")

        # 7. Oversample minority
        X_min_up, y_min_up = resample(
            X[y == 1], y[y == 1],
            replace=True,
            n_samples=int(label_counts.get(0, len(X))),
            random_state=RANDOM_STATE,
        )
        X_bal = pd.concat([X[y == 0], X_min_up]).reset_index(drop=True)
        y_bal = pd.concat([y[y == 0], y_min_up]).reset_index(drop=True)

        # 8. Cross-validation
        rf = RandomForestClassifier(
            n_estimators     = N_ESTIMATORS,
            max_features     = "sqrt",
            min_samples_leaf = 2,
            min_samples_split= 4,
            class_weight     = "balanced",
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_f1 = cross_val_score(rf, X_bal, y_bal, cv=cv, scoring="f1")
        print(f"  CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

        # 9. Train จริง
        rf.fit(X_bal, y_bal)
        self.model = rf

        # 10. ประเมิน accuracy บน training data
        y_pred   = rf.predict(X)
        accuracy = float((y_pred == y.values).mean()) * 100
        print(f"  Accuracy (train): {accuracy:.2f}%")

        # 11. อัปเดต stats
        area_counts = {}
        if "Area" in df.columns:
            area_counts = df["Area"].value_counts().to_dict()
            area_counts = {str(k): int(v) for k, v in area_counts.items()}

        self.stats.update({
            "total_datasets": len(df),
            "area_counts":    area_counts,
            "last_trained":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "accuracy":       round(accuracy, 2),
            "threshold":      THRESHOLD,
        })

        self._save_model()

        return {
            "accuracy":       round(accuracy, 2),
            "total_rows":     len(df),
            "features_count": X.shape[1],
        }

    # ------------------------------------------------------------------
    def predict(self, input_dict: dict) -> dict:
        if not self.is_trained():
            raise RuntimeError("โมเดลยังไม่ได้ Train")

        # สร้าง DataFrame 1 แถว
        row = {}
        for col in CAT_COLS:
            val = str(input_dict.get(col, "Unknow")).strip()
            row[col] = val
        row["Age_Years"] = float(input_dict.get("Age_Years", 0))
        row["Operating_Temperature_C"] = str(input_dict.get("Operating_Temperature_C", "50"))
        row["temp_num"] = encode_temperature(row["Operating_Temperature_C"])

        df_input = pd.DataFrame([row])
        X_input  = self._build_features(df_input, predict_mode=True)

        prob_yes   = float(self.model.predict_proba(X_input)[0, 1])
        prediction = "Yes" if prob_yes >= THRESHOLD else "No"
        level      = risk_label(prob_yes)
        next_year  = next_inspection_year(prob_yes)
        confidence = round((1 - abs(prob_yes - 0.5) * 2) * 100, 1)

        return {
            "prediction":           prediction,
            "probability":          prob_yes,
            "risk_level":           level,
            "next_inspection_year": next_year,
            "confidence_pct":       confidence,
        }

    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame, predict_mode=False) -> pd.DataFrame:
        feats = {}
        for col in CAT_COLS:
            vals = df[col].astype(str).str.strip().str.lower()
            if predict_mode:
                le    = self.encoders[col]
                known = set(le.classes_)
                vals  = vals.apply(lambda v: v if v in known else le.classes_[0])
            feats[col] = self.encoders[col].transform(vals)
        feats["Age_Years"] = df["Age_Years"].values
        feats["temp_num"]  = df["temp_num"].values
        return pd.DataFrame(feats)

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return self.stats

    # ------------------------------------------------------------------
    def _save_model(self):
        data = {
            "model":    self.model,
            "encoders": self.encoders,
            "stats":    self.stats,
        }
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"  [ModelManager] บันทึก Model → {MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------
    def _load_model(self):
        if not os.path.exists(MODEL_SAVE_PATH):
            print("[ModelManager] ยังไม่มี Model — รอ Upload ไฟล์ Train")
            return
        try:
            with open(MODEL_SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            self.model    = data.get("model")
            self.encoders = data.get("encoders", {})
            self.stats    = data.get("stats", self.stats)

            # ถ้าเป็น model เก่า (One-Hot) ให้ reset
            if not self.encoders:
                print("[ModelManager] พบ Model เก่า — ต้อง Train ใหม่")
                self.model = None
                return

            print(f"[ModelManager] โหลด Model สำเร็จ — Accuracy: {self.stats.get('accuracy')}%")
        except Exception as e:
            print(f"[ModelManager] โหลด Model ไม่สำเร็จ ({e}) — ต้อง Train ใหม่")
