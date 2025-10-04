# mvp_core.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#감칠맛과 매운맛은 똑같은 양이 늘어나도 맛이 달라지는 수치가 다를것이다. 따라서 맛에 따라 가중치를 다르게 적용하느것이 중요
#맛의 점수에 따라 사람들이 어떻게 느끼는지를 데이터를 계속 쌓아가며 보다 와닿는 맛 표현으로 바꿔야 함. ex) 감칠맛과 짠맛이 일정 수준 이상이고 국물요리라면 깊은 맛이 난다
#식재료와 맛의 증가에 따라 실제로 인간이 느끼는 맛이 어떤식으로 변하는지에 대한 이론도 정확히 알아내어 함수에 적용시키는 편이 좋을 것이다.

TASTE_AXES = ["sweet","salty","sour","bitter","umami","spicy","fatty"]  #맛 벡터
AXIS_WEIGHTS = {ax: 1.0 for ax in TASTE_AXES}  #축별 가중치

def load_data():
    foods = pd.read_csv("foods.csv")
    deltas = pd.read_csv("ingredient_deltas.csv")
    return foods, deltas

UNIT_TO_TSP = {    #단위 환산
    "TBSP" : 3.0, "tbsp" : 3.0, "T" : 3.0,"tbs" : 3.0, "Tbsp" : 3.0,
    "t" : 1.0, "tsp" : 1.0,
    "g" : None, "ml" : None
}

def _to_tsp(amount: float, unit: str): #단위 환산 함수
    if unit in UNIT_TO_TSP and UNIT_TO_TSP[unit] is not None:
        return amount * UNIT_TO_TSP[unit]
    # 모르는 단위는 일단 “1tsp ≈ 1”로 보정해 사용 (경고만 출력)
    print(f"[warn] 미지원 단위: {unit}. 임시로 1tsp 환산 없이 사용합니다.")
    return amount

def compute_final_taste(base_vec: np.ndarray, additions: list, deltas_df: pd.DataFrame, clip_min=0.0, clip_max=10.0):
    final_vec = base_vec.astype(float).copy()
    for add in additions:
        ing = str(add.get("ingredient","")).strip()
        amt = float(add.get("amount", 0))
        unit = add.get("unit", "tsp")

        amt_tbsp = _to_tsp(amt, unit)

        row = deltas_df[(deltas_df["ingredient"]==ing) & (deltas_df["unit"]=="1tsp")]
        if row.empty:
            # 1tsp 기준 데이터가 없으면 단위 그대로 시도
            row = deltas_df[(deltas_df["ingredient"]==ing) & (deltas_df["unit"]==f"1{unit}")]
        if row.empty:
            print(f"[warn] 델타 미존재: {ing} ({unit}) → 무시")
            continue

        delta = row[TASTE_AXES].values[0].astype(float) * amt_tbsp
        # 축별 가중치 적용
        for i, ax in enumerate(TASTE_AXES):
            delta[i] *= AXIS_WEIGHTS[ax]
        final_vec += delta

    return np.clip(final_vec, clip_min, clip_max)

def nearest_foods(target_vec: np.ndarray, foods_df: pd.DataFrame, category=None, topk=3):
    X = foods_df[TASTE_AXES].values
    if category:
        mask = (foods_df["category"]==category)
        foods_sub = foods_df[mask].copy()
        X = foods_sub[TASTE_AXES].values
    else:
        foods_sub = foods_df.copy()

    sims = cosine_similarity([target_vec], X)[0]
    idx = np.argsort(-sims)[:topk]
    return foods_sub.iloc[idx].assign(similarity=sims[idx])

def diff_phrase(delta):
    if abs(delta) < 0.5: return "비슷"
    if abs(delta) < 1.5: return "근소하게 높음" if delta>0 else "근소하게 낮음"
    return "높음" if delta>0 else "낮음"

def compare_sentence(target_vec, ref_vec, ref_name):
    diffs = target_vec - ref_vec
    parts = []
    labels = {"spicy":"매운맛","salty":"짠맛","sour":"신맛","umami":"감칠맛","sweet":"단맛","fatty":"기름짐","bitter":"쓴맛"}
    for i,ax in enumerate(TASTE_AXES):
        phrase = diff_phrase(diffs[i])
        if phrase!="비슷":
            parts.append(f"{labels[ax]}은 {phrase}")
    body = " , ".join(parts[:3]) if parts else "전반적으로 비슷합니다"
    return f"{ref_name}보다 {body}."
