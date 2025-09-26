# api.py
from fastapi import FastAPI, Query, HTTPException
from mvp_core import load_data, compute_final_taste, nearest_foods, compare_sentence, TASTE_AXES
from datetime import datetime
from typing import Optional, List
import numpy as np, pandas as pd
from fastapi.responses import RedirectResponse

app = FastAPI()
foods, deltas = load_data()

def _food_row_or_404(name: str) -> pd.Series:
    name = (name or "").strip()
    row = foods.loc[foods["name"] == name]
    if row.empty:
        examples = foods["name"].head(8).tolist()
        raise HTTPException(status_code=404, detail=f"음식 '{name}'을(를) 찾을 수 없습니다. 예: {examples}")
    return row.iloc[0]

def _validate_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    valid = set(foods["category"].dropna().unique().tolist())
    if cat not in valid:
        raise HTTPException(status_code=400, detail=f"category_filter '{cat}' 미지원. 사용 가능: {sorted(valid)}")
    return cat

def _as_numeric_axes(s: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(s[TASTE_AXES], errors="coerce")
    if vals.isna().any():
        raise HTTPException(status_code=500, detail="맛 축에 NaN/비숫자 값이 있습니다.")
    return vals.astype(float).to_numpy(dtype=float)

def _cosine_neighbors(
    base_food: str,
    category_filter: Optional[str],
    topk: int
) -> pd.DataFrame:
    # 서브셋 선택
    df = foods if not category_filter else foods[foods["category"] == category_filter]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"카테고리 '{category_filter}'에 데이터가 없습니다.")

    # 축 행렬(X)과 기준 벡터(v) 준비 (숫자만!)
    X = df[TASTE_AXES].apply(pd.to_numeric, errors="coerce")
    valid = ~X.isna().any(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].astype(float).to_numpy(dtype=float)

    # 기준 음식 벡터
    base_row = _food_row_or_404(base_food)
    v = _as_numeric_axes(base_row)

    # 코사인 유사도 계산
    denom = (np.linalg.norm(X, axis=1) * (np.linalg.norm(v) + 1e-8)) + 1e-8
    sims = (X @ v) / denom

    out = df[["name", "category"]].copy()
    for i, ax in enumerate(TASTE_AXES):
        out[ax] = X[:, i]
    out["similarity"] = sims

    # 자기 자신 제거 후 상위 k
    out = out[out["name"] != base_food]
    out = out.sort_values("similarity", ascending=False).head(topk).reset_index(drop=True)
    return out

def _summarize(base_food: str, base_vec: np.ndarray, neighbor_row: pd.Series, max_points: int = 3) -> List[str]:
    """
    간단 요약문 생성: 상위 1개 이웃과의 차이를 축별로 정리
    (임계값: 0.75/1.5)
    """
    diffs = base_vec - neighbor_row[TASTE_AXES].astype(float).to_numpy(dtype=float)
    pairs = list(zip(TASTE_AXES, diffs))
    # 절대값 큰 순으로 정렬
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    msgs = []
    count = 0
    for ax, d in pairs:
        if abs(d) < 0.5:  # 너무 미세하면 패스
            continue
        degree = "훨씬 " if abs(d) >= 1.5 else "조금 "
        direction = "높아요" if d > 0 else "낮아요"
        # 한국어 축 라벨(원하면 바꿔도 됩니다)
        ko = {
            "sweet": "단맛", "salty": "짠맛", "sour": "신맛",
            "bitter": "쓴맛", "umami": "감칠맛", "spicy": "매운맛", "fatty": "기름짐"
        }.get(ax, ax)
        msgs.append(f"‘{base_food}’은(는) ‘{neighbor_row['name']}’보다 {ko}이 {degree}{direction}.")
        count += 1
        if count >= max_points:
            break

    if not msgs:
        msgs = [f"‘{base_food}’과(와) ‘{neighbor_row['name']}’의 전반적 맛 프로필은 비슷합니다."]
    return msgs

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)

@app.get("/health")
def health():
    cats = foods["category"].value_counts().to_dict()
    unit_counts = (
    deltas["unit"].astype(str).str.lower().value_counts(dropna=False).to_dict()
    )

    issues = []
    if foods["name"].duplicated().any(): issues.append("duplicate food name")
    if deltas["ingredient"].duplicated().any(): issues.append("duplicate ingredient")
    bad_axes = [ax for ax in TASTE_AXES if ((foods[ax] < 0) | (foods[ax] > 10)).any()]
    if bad_axes: issues.append(f"foods axis out of [0,10]: {bad_axes}")
    unrec = deltas[~deltas["unit"].astype(str).str.lower().str.contains("tsp|tbsp", na=False)]
    if not unrec.empty: issues.append(f"unrecognized units: {sorted(unrec['unit'].astype(str).str.lower().unique())}")
    status = "ok" if not issues else "warn"

    return {
        "status": status,
        "foods_count": int(foods.shape[0]),
        "ingredients_count": int(deltas["ingredient"].nunique()),
        "categories": cats,
        "axes": TASTE_AXES,
        "unit_counts": unit_counts,
        "server_time": datetime.now().isoformat(timespec="seconds")
    }

@app.get("/foods")
def list_foods(
    category: Optional[str] = Query(None, description="카테고리로 필터 (예: soup)"),
    search: Optional[str] = Query(None, description="부분 문자열 매칭(대소문자 무시)"),
    limit: int = Query(50, ge=1, le=200, description="반환 최대 개수"),
    include_scores: bool = Query(False, description="맛 점수(축 값)까지 포함할지 여부")
):
    """
    지원 음식 목록 조회용 엔드포인트.
    - 예) /foods
    - 예) /foods?category=soup
    - 예) /foods?search=곰
    - 예) /foods?limit=20&include_scores=1
    """
    df = foods
    if category:
        # category 열이 없으면 전체 반환(안전)
        if "category" in df.columns:
            df = df[df["category"] == category]
        else:
            df = df.iloc[0:0]  # 카테고리 열이 없다면 빈 목록
    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]
    df = df.head(limit)

    items = []
    for _, row in df.iterrows():
        item = {
            "name": row["name"],
            "category": row.get("category", None)
        }
        if include_scores:
            item["scores"] = {ax: float(row[ax]) for ax in TASTE_AXES}
        items.append(item)

    return {
        "count": len(items),
        "items": items
    }

@app.get("/ingredients")
def list_ingredients(
    search: Optional[str] = Query(None, description="부분 문자열 매칭(대소문자 무시)"),
    limit: int = Query(100, ge=1, le=500, description="반환 최대 개수"),
    sort: str = Query("asc", pattern="^(asc|desc)$", description="정렬 순서: asc | desc")
):
    """
    지원 재료 목록 조회
    - 예) /ingredients
    - 예) /ingredients?search=장
    - 예) /ingredients?limit=50&sort=desc
    """
    if "ingredient" not in deltas.columns:
        return {"count": 0, "items": []}

    names: List[str] = sorted(deltas["ingredient"].dropna().astype(str).unique().tolist(), key=str.lower)
    if sort == "desc":
        names = list(reversed(names))
    if search:
        s = search.lower()
        names = [n for n in names if s in n.lower()]
    names = names[:limit]

    return {
        "count": len(names),
        "items": [{"ingredient": n} for n in names]
    }

@app.get("/axes")
def get_axes():
    """
    맛 축 메타데이터 반환
    예) ["sweet","salty","sour","bitter","umami","spicy","fatty"]
    """
    # 필요시 라벨/설명까지 확장 가능하도록 구조 준비
    axes = [{"key": ax, "label": ax.capitalize(), "range": [0, 10]} for ax in TASTE_AXES]
    return {
        "count": len(axes),
        "axes": axes
    }

@app.post("/predict")
def predict(body: dict):
    """
    body 예:
    {
        "base_food": "곰탕",
        "additions": [{"ingredient":"된장","amount":4,"unit":"Tbsp"}],
        "category_filter": "soup"
    }
    """
    base_vec = foods[foods["name"]==body["base_food"]][TASTE_AXES].values[0]
    final_vec = compute_final_taste(base_vec, body["additions"], deltas)
    neighbors = nearest_foods(final_vec, foods, category=body.get("category_filter"), topk=3)

    comparisons = []
    for _,row in neighbors.iterrows():
        sentence = compare_sentence(final_vec, row[TASTE_AXES].values, row["name"])
        comparisons.append({"ref": row["name"], "similarity": float(row["similarity"]), "text": sentence})

    return {
        "final_scores": {ax: round(float(v),1) for ax,v in zip(TASTE_AXES, final_vec)},
        "comparisons": comparisons
    }

@app.get("/similar")
def similar_foods(
    base_food: str = Query(..., description="기준 음식명 (예: 곰탕)"),
    category_filter: Optional[str] = Query(None, description="필터(예: soup). 미지정시 전체"),
    topk: int = Query(5, ge=1, le=50, description="추천 개수")
):
    try:
        category_filter = _validate_category(category_filter)
        base_row = _food_row_or_404(base_food)
        base_vec = _as_numeric_axes(base_row)  # np.ndarray

        neighbors = _cosine_neighbors(base_food, category_filter, topk)
        sentences = _summarize(base_food, base_vec, neighbors.iloc[0]) if not neighbors.empty else [
            "유사 이웃을 찾지 못했습니다."
        ]

        return {
            "base_food": base_food,
            "category_filter_applied": category_filter,
            "axes": TASTE_AXES,
            "base_vector": {ax: float(base_row[ax]) for ax in TASTE_AXES},
            "neighbors": neighbors.assign(similarity=lambda d: d["similarity"].round(4)).to_dict(orient="records"),
            "sentences": sentences
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar 처리 중 오류: {type(e).__name__}: {e}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)