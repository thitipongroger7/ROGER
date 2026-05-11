import numpy as np

# ══════════════════════════════════════════════════════════
# CUI Bayesian Prior — API 583 2nd Ed. Annex A
# ══════════════════════════════════════════════════════════
#
# มี 2 ตารางจาก API 583 Annex A:
#   A.2 → CS / Low Alloy Steel   (uniform corrosion)
#   A.3 → Austenitic / Duplex SS (Cl stress corrosion cracking)
#
# โค้ดนี้จะเลือก A.2 หรือ A.3 อัตโนมัติจาก substrate
#
# Formula: Prior% = sigmoid(BASE_LO + (score/MAX_SCORE) × MAX_ADJ)
#
#   BASE_LO   = log(0.166/0.834) = −1.615
#               จาก training prevalence (38/229 = 16.6%)
#   MAX_SCORE = 25  (5 parameters × max rating 5)
#   MAX_ADJ   = 3.0 (engineering assumption)
# ══════════════════════════════════════════════════════════

# BASE_LO คำนวณจาก training data จริง (ไม่ hardcode)
# จะถูก override โดย compute_base_lo() เมื่อ train model
BASE_LO = np.log(0.166 / 0.834)  # ค่า default (จะถูกแทนที่)

def compute_base_lo(y_train: np.ndarray) -> float:
    """
    คำนวณ BASE_LO จาก prevalence ของ training data จริง
    BASE_LO = log(p_yes / p_no)
    """
    n_yes   = int(y_train.sum())
    n_total = len(y_train)
    n_no    = n_total - n_yes
    prev    = n_yes / n_total
    base_lo = float(np.log(prev / (1 - prev)))
    return base_lo
MAX_SCORE = 25.0
MAX_ADJ   = 3.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ══════════════════════════════════════════════════════════
# STEP 0 — จำแนก Material Group จาก Substrate
# ══════════════════════════════════════════════════════════

_CS_KEYWORDS = {
    'cs', 'carbon steel', 'ltcs', 'low temperature carbon steel',
    'low alloy steel', 'low alloy', 'la steel',
    'chrome moly', 'cr-mo', 'cr mo', 'p91', 'p22', 'p11',
    '1.25cr', '2.25cr', '5cr', '9cr',
}

_SS_KEYWORDS = {
    'ss', 'ss304', 'ss316', 'ss321', 'ss347', 'ss310',
    'stainless', 'austenitic',
    '304', '316', '316l', '304l', '321', '347', '310', '317',
    'duplex', 'super duplex', '2205', '2507', 'lean duplex',
    'dss', 'sdss',
    'nickel alloy', 'inconel', 'incoloy', 'hastelloy',
}


def classify_substrate(substrate: str) -> str:
    """
    จำแนก substrate เป็น 3 กลุ่ม:
      'CS'      → Carbon Steel / Low Alloy → ใช้ตาราง A.2
      'SS'      → Austenitic / Duplex SS   → ใช้ตาราง A.3
      'UNKNOWN' → ไม่รู้ → ใช้ A.2 (conservative)

    Returns: 'CS' | 'SS' | 'UNKNOWN'
    """
    s = str(substrate).strip().lower()

    # เช็ค SS ก่อน
    if any(k in s for k in _SS_KEYWORDS):
        return 'SS'

    # เช็ค CS
    if any(k in s for k in _CS_KEYWORDS):
        return 'CS'

    return 'UNKNOWN'


def get_table_name(substrate: str) -> str:
    """คืนชื่อตารางที่จะใช้ เพื่อ logging"""
    if classify_substrate(substrate) == 'SS':
        return 'A.3 (Austenitic/Duplex SS)'
    return 'A.2 (CS/Low Alloy Steel)'


# ══════════════════════════════════════════════════════════
# TABLE A.2 — Rating Functions (CS / Low Alloy Steel)
# Source: API 583 2nd Ed. Annex A Section A.2
# ══════════════════════════════════════════════════════════

def _temp_A2(temp_zone: str) -> int:
    """
    A.2 Operating Temperature (CS)
    Rating 5: 77–110°C or Cyclic
    Rating 3: 38–77°C  or 110–132°C
    Rating 1: −4–38°C  or 132–177°C
    Rating 0: <−4°C    or >177°C
    """
    t = str(temp_zone).strip()
    m = {
        # Rating 0
        'มากกว่าหรือเท่ากับ 200': 0,
        '(30) - <(20)': 0, '(10) - <0': 0,
        '160 - <170': 0, '170 - <180': 0, '180 - <190': 0,
        # Rating 1
        '0 - <10': 1,  '10 - <20': 1, '20 - <30': 1, '30 - <40': 1,
        '140 - <150': 1, '150 - <160': 1,
        # Rating 3
        '40 - <50': 3, '50 - <60': 3,
        '120 - <130': 3, '130 - <140': 3,
        # Rating 5
        '60 - <70': 5,  '70 - <80': 5,  '80 - <90': 5,
        '90 - <100': 5, '100 - <110': 5, '110 - <120': 5,
        'Cyclic temperature': 5, 'Cyclic Temperature': 5,
    }
    return m.get(t, 1)


def _temp_A3(temp_zone: str) -> int:
    """
    A.3 Operating Temperature (SS/Duplex)
    SS เสี่ยง Cl-SCC ที่ช่วงต่างจาก CS:
    Rating 5: 60–121°C  (140–250°F)
    Rating 3: 47–60°C   (120–140°F)  or  121–204°C  (250–400°F)
    Rating 1: อื่นๆ (low or very high temp)
    Note: ไม่มี Rating 0 ใน A.3 temperature
    """
    t = str(temp_zone).strip()
    m = {
        # Rating 1 — outside main Cl-SCC zone
        '(30) - <(20)': 1, '(10) - <0': 1,
        '0 - <10': 1,  '10 - <20': 1, '20 - <30': 1, '30 - <40': 1,
        'มากกว่าหรือเท่ากับ 200': 1,
        '160 - <170': 1, '170 - <180': 1, '180 - <190': 1,
        # Rating 3 — border Cl-SCC zones
        '40 - <50': 3, '50 - <60': 3,                         # 40–60°C
        '120 - <130': 3, '130 - <140': 3, '140 - <150': 3,    # 121–204°C
        '150 - <160': 3,
        # Rating 5 — peak Cl-SCC: 60–121°C
        '60 - <70': 5,  '70 - <80': 5,  '80 - <90': 5,
        '90 - <100': 5, '100 - <110': 5, '110 - <120': 5,
        'Cyclic temperature': 5, 'Cyclic Temperature': 5,
    }
    return m.get(t, 1)


def _jacket_A2(jacket_damage: str) -> int:
    """
    A.2 Jacketing/Insulation Condition
    Rating 5: Damaged condition, several deficiencies
    Rating 3: Average, some deficiencies
    Rating 1: Average, good maintenance
    Rating 0: System age <5yr, no deficiencies
    Dataset มีแค่ Yes/No → 5 or 0
    """
    j = str(jacket_damage).strip().lower()
    if j in ['yes', 'damaged']:   return 5
    if j in ['no', 'intact']:     return 0
    return 1


def _jacket_A3(jacket_damage: str) -> int:
    """
    A.3 Jacketing/Insulation Condition (SS)
    Rating 5: Damaged, several deficiencies
    Rating 3: Average, some deficiencies
    Rating 1: Average, good maintenance
    Rating 0: No deficiencies (note: no system age condition vs A.2)
    """
    j = str(jacket_damage).strip().lower()
    if j in ['yes', 'damaged']:   return 5
    if j in ['no', 'intact']:     return 0
    return 1


def _environment_shared(environment: str) -> int:
    """
    Environment rating เหมือนกันทั้ง A.2 และ A.3
    Rating 5: Coastal/Marine, cooling tower, deluge
    Rating 3: All other locations
    Rating 1: Arid and inland
    Rating 0: No sweating
    """
    e = str(environment).lower()
    if any(k in e for k in ['marine', 'coastal', 'cooling', 'deluge']):
        return 5
    if any(k in e for k in ['arid', 'inland']):
        return 1
    if any(k in e for k in ['no sweating', 'dry', 'indoor']):
        return 0
    return 3


def _coating_A2(coating_prime: str, age_years) -> int:
    """
    A.2 Coating Quality + System Age (CS)
    Rating 5: General coating >15yr OR system age >30yr OR unknown
    Rating 3: General coating 8–15yr OR system age 15–30yr
    Rating 1: Quality coating <15yr OR system age <30yr
    Rating 0: Quality coating <8yr OR system age <15yr
    """
    p = str(coating_prime).lower()
    is_quality = (
        'phenolic epoxy' in p or
        ('epoxy' in p and 'zinc' not in p) or
        'silicone' in p
    )
    try:
        a = float(age_years)
        if is_quality and a < 8:  return 0
        if is_quality and a < 15: return 1
        if a < 15:                return 1
        if a < 30:                return 3
        return 5
    except Exception:
        return 5


def _coating_A3(coating_prime: str, age_years) -> int:
    """
    A.3 Coating Quality + Age (SS)
    ต่างจาก A.2: ไม่ใช้ system age — ดูแค่ coating quality + age เท่านั้น
    Rating 5: General coating >15yr OR unknown
    Rating 3: General coating 8–15yr
    Rating 1: Quality coating <15yr
    Rating 0: Quality coating <8yr
    """
    p = str(coating_prime).lower()
    is_quality = (
        'phenolic epoxy' in p or
        ('epoxy' in p and 'zinc' not in p) or
        'silicone' in p
    )
    try:
        a = float(age_years)
        if is_quality and a < 8:  return 0
        if is_quality and a < 15: return 1
        if a < 15:                return 1
        if a < 30:                return 3
        return 5
    except Exception:
        return 5


def _insulation_shared(insulation_type: str) -> int:
    """
    Insulation Type เหมือนกันทั้ง A.2 และ A.3
    Rating 5: Calcium silicate, Mineral fiber >10ppm Cl, Unknown
    Rating 3: Fiberglass, Mineral fiber, Perlite
    Rating 1: Cellular glass, Foam glass, Aerogel, Closed-cell foam
    Rating 0: Insulating coating, TSA, metallic spray
    """
    ins = str(insulation_type).lower()
    if any(k in ins for k in ['calcium silicate']):
        return 5
    if any(k in ins for k in ['unknown', 'wp']):
        return 5
    if any(k in ins for k in ['mineral fiber', 'mineral wool', 'fiberglass']):
        return 3
    if any(k in ins for k in ['cellular glass', 'foam glass', 'aerogel',
                               'perlite', 'polyisocyanurate', 'polyurethane',
                               'closed']):
        return 1
    if any(k in ins for k in ['coating', 'tsa', 'metallic']):
        return 0
    return 3


# ══════════════════════════════════════════════════════════
# Public Interface — รับ substrate เลือกตารางให้อัตโนมัติ
# ══════════════════════════════════════════════════════════

def rating_temperature(temp_zone: str, substrate: str = 'CS') -> int:
    if classify_substrate(substrate) == 'SS':
        return _temp_A3(temp_zone)
    return _temp_A2(temp_zone)


def rating_jacket(jacket_damage: str, substrate: str = 'CS') -> int:
    if classify_substrate(substrate) == 'SS':
        return _jacket_A3(jacket_damage)
    return _jacket_A2(jacket_damage)


def rating_environment(environment: str, substrate: str = 'CS') -> int:
    return _environment_shared(environment)


def rating_coating_age(coating_prime: str, age_years,
                       substrate: str = 'CS') -> int:
    if classify_substrate(substrate) == 'SS':
        return _coating_A3(coating_prime, age_years)
    return _coating_A2(coating_prime, age_years)


def rating_insulation(insulation_type: str, substrate: str = 'CS') -> int:
    return _insulation_shared(insulation_type)


# ══════════════════════════════════════════════════════════
# STEP 2 — Total API Score
# ══════════════════════════════════════════════════════════

def api_total_score(temp_zone, jacket_damage, environment,
                    coating_prime, age_years, insulation_type,
                    substrate: str = 'CS') -> int:
    """
    รวมคะแนน 5 parameter ตาม API 583 Annex A
    เลือก A.2 (CS) หรือ A.3 (SS) อัตโนมัติจาก substrate
    Returns: score 0–25
    """
    return (
        rating_temperature(temp_zone,     substrate) +
        rating_jacket(jacket_damage,      substrate) +
        rating_environment(environment,   substrate) +
        rating_coating_age(coating_prime,
                           age_years,     substrate) +
        rating_insulation(insulation_type,substrate)
    )


# ══════════════════════════════════════════════════════════
# STEP 3 — Score → Prior%
# ══════════════════════════════════════════════════════════

def compute_prior(temp_zone, jacket_damage, environment,
                  coating_prime, age_years, insulation_type,
                  substrate: str = 'CS',
                  base_lo: float = None) -> float:
    """
    แปลง API score เป็น Prior probability
    เลือกตาราง A.2 หรือ A.3 อัตโนมัติตาม substrate
    base_lo: ถ้าไม่ระบุจะใช้ BASE_LO จาก module (default)
    Returns: float in [0, 1]
    """
    _base = base_lo if base_lo is not None else BASE_LO
    score = api_total_score(
        temp_zone, jacket_damage, environment,
        coating_prime, age_years, insulation_type,
        substrate=substrate,
    )
    lo = _base + (score / MAX_SCORE) * MAX_ADJ
    return float(sigmoid(lo))


# ══════════════════════════════════════════════════════════
# STEP 4 — Bayesian Blend
# ══════════════════════════════════════════════════════════

def bayesian_blend(prior: float, rf_prob: float,
                   w_prior: float = 0.5) -> float:
    """
    รวม Prior% กับ RF% ใน log-odds space
    Bayes% = sigmoid(w_prior × logit(Prior) + w_rf × logit(RF))
    """
    w_rf = 1.0 - w_prior
    lo = (
        w_prior * np.log(prior   / (1 - np.clip(prior,   1e-6, 1-1e-6))) +
        w_rf    * np.log(rf_prob / (1 - np.clip(rf_prob, 1e-6, 1-1e-6)))
    )
    return float(sigmoid(lo))


# ══════════════════════════════════════════════════════════
# STEP 5 — Conformal Prediction
# ══════════════════════════════════════════════════════════

def calibrate_q(bayes_cal: np.ndarray, y_cal: np.ndarray,
                conf: float = 0.90) -> float:
    scores = np.where(y_cal == 1, 1 - bayes_cal, bayes_cal)
    return float(np.quantile(scores, conf))


def conformal_predict(bayes: float, q: float) -> dict:
    in_yes = (1 - bayes) <= q
    in_no  =      bayes  <= q
    if in_yes and not in_no:
        return {'pred_set': '{Yes}',    'predict': 'Yes', 'tier': 1}
    elif not in_yes and in_no:
        return {'pred_set': '{No}',     'predict': 'No',  'tier': 3}
    elif in_yes and in_no:
        return {'pred_set': '{No+Yes}', 'predict': 'Yes', 'tier': 2}
    else:
        return {'pred_set': '{}',       'predict': 'Yes', 'tier': 2}


# ══════════════════════════════════════════════════════════
# ตัวอย่างการใช้งาน
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':

    examples = [
        {
            'name':           'CS example (Table A.2)',
            'substrate':      'CS',
            'temp_zone':      '30 - <40',
            'jacket_damage':  'No',
            'environment':    'Marine',
            'coating_prime':  'phenolic epoxy',
            'age_years':      12,
            'insulation_type':'mineral fiber',
            'rf_prob':        0.2520,
            'q':              0.3199,
        },
        {
            'name':           'SS304 example (Table A.3)',
            'substrate':      'SS304',
            'temp_zone':      '80 - <90',
            'jacket_damage':  'No',
            'environment':    'Marine',
            'coating_prime':  'phenolic epoxy',
            'age_years':      12,
            'insulation_type':'mineral fiber',
            'rf_prob':        0.35,
            'q':              0.3199,
        },
    ]

    for ex in examples:
        sub   = ex['substrate']
        table = get_table_name(sub)
        group = classify_substrate(sub)

        score  = api_total_score(
            ex['temp_zone'], ex['jacket_damage'], ex['environment'],
            ex['coating_prime'], ex['age_years'], ex['insulation_type'],
            substrate=sub,
        )
        prior  = compute_prior(
            ex['temp_zone'], ex['jacket_damage'], ex['environment'],
            ex['coating_prime'], ex['age_years'], ex['insulation_type'],
            substrate=sub,
        )
        bayes  = bayesian_blend(prior, ex['rf_prob'])
        result = conformal_predict(bayes, ex['q'])
        lo_val = BASE_LO + (score / MAX_SCORE) * MAX_ADJ

        print("═" * 62)
        print(f"  {ex['name']}")
        print(f"  Substrate : {sub}  →  Group: {group}")
        print(f"  Table used: API 583 Annex {table}")
        print("═" * 62)
        print(f"\n  Ratings:")
        print(f"    Temperature  : {rating_temperature(ex['temp_zone'], sub)}")
        print(f"    Jacket       : {rating_jacket(ex['jacket_damage'], sub)}")
        print(f"    Environment  : {rating_environment(ex['environment'], sub)}")
        print(f"    Coating+Age  : {rating_coating_age(ex['coating_prime'], ex['age_years'], sub)}")
        print(f"    Insulation   : {rating_insulation(ex['insulation_type'], sub)}")
        print(f"    Total Score  : {score} / 25")
        print(f"\n  Prior%  = sigmoid({BASE_LO:.3f} + ({score}/25)×{MAX_ADJ:.1f})")
        print(f"          = sigmoid({lo_val:.3f}) = {prior*100:.2f}%")
        print(f"  RF%     = {ex['rf_prob']*100:.2f}%")
        print(f"  Bayes%  = {bayes*100:.2f}%")
        print(f"  Result  = {result['pred_set']}  →  {result['predict']}  (Tier {result['tier']})")
        print()
