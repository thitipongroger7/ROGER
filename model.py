# ==============================================================================
# FILE: model.py
# PURPOSE: ML Logic — Random Forest + OneHotEncoder + Unknown Detection
#          + Bayesian Prior (API 583 Annex A) + Conformal Prediction
# ==============================================================================

import re
import os
import pickle
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from supabase import create_client

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = os.path.join(os.getcwd(), "cui_model.pkl")
THRESHOLD       = 0.50
N_ESTIMATORS    = 500
RANDOM_STATE    = 42

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
BUCKET_NAME     = "models"
MODEL_FILE_NAME = "cui_model.pkl"

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


def normalize_cat(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        c: df[c].astype(str).str.strip().str.lower()
        for c in CAT_COLS
    })


# ==============================================================================
# Supabase helpers
# ==============================================================================
def _get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[Supabase] เชื่อมต่อไม่ได้: {e}")
        return None


def _upload_model_to_supabase():
    sb = _get_supabase()
    if not sb:
        return
    try:
        with open(MODEL_SAVE_PATH, "rb") as f:
            data = f.read()
        sb.storage.from_(BUCKET_NAME).upload(
            MODEL_FILE_NAME, data,
            {"upsert": "true", "content-type": "application/octet-stream"}
        )
        print("[Supabase] Upload model สำเร็จ ✅")
    except Exception as e:
        print(f"[Supabase] Upload ไม่สำเร็จ: {e}")


def _download_model_from_supabase():
    sb = _get_supabase()
    if not sb:
        return False
    try:
        data = sb.storage.from_(BUCKET_NAME).download(MODEL_FILE_NAME)
        with open(MODEL_SAVE_PATH, "wb") as f:
            f.write(data)
        print("[Supabase] Download model สำเร็จ ✅")
        return True
    except Exception as e:
        print(f"[Supabase] ไม่มี model ใน Supabase หรือ download ไม่สำเร็จ: {e}")
        return False


# ==============================================================================
# ModelManager
# ==============================================================================
class ModelManager:
    def __init__(self):
        self.model      = None
        self.ohe        = None        # OneHotEncoder
        self.seen_vals  = {}          # ค่าที่เคยเห็นใน training (Unknown Detection)
        self.base_lo    = None        # BASE_LO คำนวณจาก training data จริง
        self.q_value    = 0.3352      # Conformal q (conf=90%)
        self.stats = {
            "total_datasets": 0,
            "area_counts":    {},
            "last_trained":   None,
            "accuracy":       None,
            "threshold":      THRESHOLD,
        }
        self._load_model()

    def is_trained(self) -> bool:
        return self.model is not None and self.ohe is not None

    # ------------------------------------------------------------------
    def train(self, file_path: str) -> dict:
        print(f"\n[ModelManager] เริ่ม Train จากไฟล์: {file_path}")

        # โหลดข้อมูล
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

        # เติมค่าว่าง
        for col in CAT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            else:
                df[col] = "Unknown"

        # Label
        df["yes_prob"] = df["Corrosion_Result"].apply(parse_corrosion_result)
        df["label"]    = (df["yes_prob"] >= THRESHOLD).astype(int)
        df = df.dropna(subset=["label"])
        df["temp_num"] = df["Operating_Temperature_C"].apply(encode_temperature)
        print(f"  หลังล้าง: {len(df)} แถว")

        # บันทึกค่าที่เคยเห็น (Unknown Detection)
        self.seen_vals = {
            col: set(df[col].astype(str).str.strip().str.lower().unique())
            for col in CAT_COLS
        }

        # BASE_LO จาก training data จริง
        n_yes = int(df["label"].sum())
        n_total = len(df)
        prev = n_yes / n_total
        self.base_lo = float(np.log(prev / (1 - prev)))
        print(f"  Yes={n_yes}, No={n_total-n_yes}, Prevalence={prev*100:.1f}%")
        print(f"  BASE_LO = {self.base_lo:.4f}")

        # OneHotEncoder
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe.fit(normalize_cat(df))

        # Build features
        X = self._build_features(df)
        y = df["label"].values

        # Oversample minority class
        idx_maj = np.where(y == 0)[0]
        idx_min = np.where(y == 1)[0]
        rng     = np.random.RandomState(RANDOM_STATE)
        idx_up  = rng.choice(idx_min, size=len(idx_maj), replace=True)
        X_bal   = np.vstack([X[idx_maj], X[idx_up]])
        y_bal   = np.hstack([y[idx_maj], y[idx_up]])
        sh      = rng.permutation(len(y_bal))
        X_bal, y_bal = X_bal[sh], y_bal[sh]

        # Train RF
        rf = RandomForestClassifier(
            n_estimators  = N_ESTIMATORS,
            max_features  = "sqrt",
            class_weight  = "balanced",
            random_state  = RANDOM_STATE,
            n_jobs        = -1,
        )
        cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_f1 = cross_val_score(rf, X_bal, y_bal, cv=cv, scoring="f1")
        print(f"  CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

        rf.fit(X_bal, y_bal)
        self.model = rf

        accuracy = float((rf.predict(X) == y).mean()) * 100
        print(f"  Accuracy (train): {accuracy:.2f}%")

        # Calibrate q จาก training data
        self.q_value = self._calibrate_q(df, X, y)
        print(f"  q (conf=90%): {self.q_value:.4f}")

        area_counts = {}
        if "Area" in df.columns:
            area_counts = {str(k): int(v) for k, v in df["Area"].value_counts().items()}

        self.stats.update({
            "total_datasets": len(df),
            "area_counts":    area_counts,
            "last_trained":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "accuracy":       round(accuracy, 2),
            "threshold":      THRESHOLD,
        })

        self._save_model()
        _upload_model_to_supabase()

        return {
            "accuracy":       round(accuracy, 2),
            "total_rows":     len(df),
            "features_count": X.shape[1],
        }

    # ------------------------------------------------------------------
    def _calibrate_q(self, df, X, y, conf=0.90) -> float:
        """Calibrate Conformal q จาก training data"""
        try:
            import prior_v2 as P2
            rf_probs = self.model.predict_proba(X)[:, 1]
            prior_probs = df.apply(
                lambda r: P2.compute_prior(
                    r["Operating_Temperature_C"], r["Jacket_Damage"],
                    r["Environment"], r["Coating_Prime"],
                    r["Age_Years"], r["Insulation_Type"],
                    substrate=r["Substrate"],
                    base_lo=self.base_lo,
                ), axis=1
            ).values
            bayes_probs = np.array([
                P2.bayesian_blend(p, r)
                for p, r in zip(prior_probs, rf_probs)
            ])
            scores = np.where(y == 1, 1 - bayes_probs, bayes_probs)
            return float(np.quantile(scores, conf))
        except Exception as e:
            print(f"  [q calibration] ใช้ค่า default: {e}")
            return 0.3352

    # ------------------------------------------------------------------
    def detect_unknown(self, input_dict: dict) -> list:
        """ตรวจหา categorical value ที่ไม่เคยเห็นใน training"""
        unknown_cols = []
        for col in CAT_COLS:
            val = str(input_dict.get(col, "")).strip().lower()
            if val not in self.seen_vals.get(col, set()):
                unknown_cols.append(col)
        return unknown_cols

    # ------------------------------------------------------------------
    def predict(self, input_dict: dict, model: str = "hybrid") -> dict:
        if not self.is_trained():
            raise RuntimeError("โมเดลยังไม่ได้ Train")

        # Unknown Detection
        unknown_cols = self.detect_unknown(input_dict)
        if unknown_cols:
            return {
                "prediction":    "Unknown",
                "unknown_cols":  unknown_cols,
                "rf_prob":       None,
                "prior_prob":    None,
                "bayes_prob":    None,
                "pred_set":      None,
                "q_value":       self.q_value,
            }

        # Build features
        row = {}
        for col in CAT_COLS:
            row[col] = str(input_dict.get(col, "Unknown")).strip()
        row["Age_Years"] = float(input_dict.get("Age_Years", 0))
        row["Operating_Temperature_C"] = str(input_dict.get("Operating_Temperature_C", "50"))
        row["temp_num"] = encode_temperature(row["Operating_Temperature_C"])

        df_input = pd.DataFrame([row])
        X_input  = self._build_features(df_input)

        # RF Probability
        rf_prob = float(self.model.predict_proba(X_input)[0, 1])

        # ── RF Only ──────────────────────────────────────────────────
        if model == "rf":
            prediction = "Yes" if rf_prob >= THRESHOLD else "No"
            pred_set   = "{Yes}" if rf_prob >= THRESHOLD else "{No}"
            return {
                "prediction":  prediction,
                "pred_set":    pred_set,
                "rf_prob":     round(rf_prob * 100, 1),
                "prior_prob":  None,
                "bayes_prob":  None,
                "q_value":     round(self.q_value, 4),
                "unknown_cols": [],
            }

        # ── Hybrid (RF + Bayesian Prior + Conformal) ─────────────────
        try:
            import prior_v2 as P2
            prior_prob = P2.compute_prior(
                row["Operating_Temperature_C"], row["Jacket_Damage"],
                row["Environment"], row["Coating_Prime"],
                row["Age_Years"], row["Insulation_Type"],
                substrate=row["Substrate"],
                base_lo=self.base_lo,
            )
            bayes_prob = P2.bayesian_blend(prior_prob, rf_prob)
            result_cp  = P2.conformal_predict(bayes_prob, self.q_value)
        except Exception as e:
            print(f"[predict] prior_v2 error: {e}")
            prior_prob = rf_prob
            bayes_prob = rf_prob
            result_cp  = {"pred_set": "{Yes}" if rf_prob >= 0.5 else "{No}",
                          "predict": "Yes" if rf_prob >= 0.5 else "No", "tier": 1}

        return {
            "prediction":  result_cp["predict"],
            "pred_set":    result_cp["pred_set"],
            "rf_prob":     round(rf_prob * 100, 1),
            "prior_prob":  round(prior_prob * 100, 1),
            "bayes_prob":  round(bayes_prob * 100, 1),
            "q_value":     round(self.q_value, 4),
            "unknown_cols": [],
        }

    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        cat_enc = self.ohe.transform(normalize_cat(df))
        num     = np.column_stack([
            df["Age_Years"].values,
            df["temp_num"].values,
        ])
        return np.hstack([cat_enc, num])

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return self.stats

    # ------------------------------------------------------------------
    def _save_model(self):
        data = {
            "model":     self.model,
            "ohe":       self.ohe,
            "seen_vals": self.seen_vals,
            "base_lo":   self.base_lo,
            "q_value":   self.q_value,
            "stats":     self.stats,
        }
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"  [ModelManager] บันทึก Model → {MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------
    def _load_model(self):
        if not os.path.exists(MODEL_SAVE_PATH):
            print("[ModelManager] ไม่มี model local — ลอง download จาก Supabase...")
            _download_model_from_supabase()

        if not os.path.exists(MODEL_SAVE_PATH):
            print("[ModelManager] ยังไม่มี Model — รอ Upload ไฟล์ Train")
            return
        try:
            with open(MODEL_SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            self.model      = data.get("model")
            self.ohe        = data.get("ohe")
            self.seen_vals  = data.get("seen_vals", {})
            self.base_lo    = data.get("base_lo")
            self.q_value    = data.get("q_value", 0.3352)
            self.stats      = data.get("stats", self.stats)

            if self.model is None or self.ohe is None:
                print("[ModelManager] พบ Model เก่า — ต้อง Train ใหม่")
                self.model = None
                self.ohe   = None
                return

            print(f"[ModelManager] โหลด Model สำเร็จ — Accuracy: {self.stats.get('accuracy')}%")
        except Exception as e:
            print(f"[ModelManager] โหลด Model ไม่สำเร็จ ({e}) — ต้อง Train ใหม่")
