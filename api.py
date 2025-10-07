# api.py — single-file version (mvp_core merged)
from fastapi import FastAPI, Query, HTTPException, Request, Body
from datetime import datetime
from typing import Optional, List
import numpy as np, pandas as pd
from fastapi.responses import RedirectResponse, JSONResponse
import traceback
from pydantic import BaseModel, Field
import time, threading
import os

#맛 벡터와 가중치
TASTE_AXES = ["sweet","salty","sour","bitter","umami","spicy","fatty"]
AXIS_WEIGHTS = {ax: 1.0 for ax in TASTE_AXES}

UNIT_TO_TSP = {
    "TBSP": 3.0, "tbsp": 3.0, "T": 3.0, "tbs": 3.0, "Tbsp": 3.0,
    "t": 1.0, "tsp": 1.0,
    "g": None, "ml": None
}

# CSV 로딩 (상대경로 또는 /mnt/data 모두 시도)
_DEF_FOODS_CANDIDATES = [
    "foods.csv",
    os.path.join("/mnt/data", "foods.csv")
]
_DEF_DELTAS_CANDIDATES = [
    "ingredient_deltas.csv",
    os.path.join("/mnt/data", "ingredient_deltas.csv")
]

def _first_existing(paths: list[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    # 마지막 경로를 반환해 에러 메시지에 명확히 표기
    return paths[-1]

def load_data():
    foods_path = _first_existing(_DEF_FOODS_CANDIDATES)
    deltas_path = _first_existing(_DEF_DELTAS_CANDIDATES)
    foods = pd.read_csv(foods_path)
    deltas = pd.read_csv(deltas_path)
    return foods, deltas

# -----------------------------
# App setup
# -----------------------------
tags_metadata = [
    {"name": "Health",     "description": "서버 상태와 데이터 개요"},
    {"name": "Predict",    "description": "맛 예측 관련 엔드포인트"},
    {"name": "Similarity", "description": "유사 음식 탐색"},
    {"name": "Data",       "description": "데이터 조회/점검"},
    {"name": "Community",  "description": "초간단 메모 게시판"},
]
app = FastAPI(
    title="맛 예측 API",
    openapi_tags=tags_metadata
)
foods, deltas = load_data()

# 메모리 게시판(데모용)
_POSTS: list[dict] = []
_POST_SEQ = 0
_POST_LOCK = threading.Lock()

# -----------------------------
# Internal helpers (linear taste model + cosine similarity)
# -----------------------------

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


def _unit_factor(u: str) -> float:
    ui = str(u or "").strip().lower()
    if "tbsp" in ui: return 3.0        # 1 Tbsp = 3 tsp
    if "tsp"  in ui: return 1.0
    # 알 수 없는 단위는 거부(이전엔 경고 후 통과였으나 API 일관성을 위해 400)
    raise HTTPException(status_code=400, detail=f"지원 단위가 아닙니다: '{u}' (tsp/Tbsp)")


def _delta_per_tsp(row: pd.Series) -> dict:
    base = str(row.get("unit", "1tsp")).lower()
    base_factor = 3.0 if "tbsp" in base else 1.0   # CSV 단위가 1Tbsp면 3으로 나눠 tsp 기준으로 환산
    return {ax: float(row.get(ax, 0.0)) / base_factor for ax in TASTE_AXES}


def _compute_final_linear(base_food: str, additions: list[dict]) -> pd.Series:
    # 1) 기준 벡터
    r = foods.loc[foods["name"] == (base_food or "").strip()]
    if r.empty:
        examples = foods["name"].head(8).tolist()
        raise HTTPException(status_code=404, detail=f"음식 '{base_food}' 없음. 예: {examples}")
    v = r[TASTE_AXES].iloc[0].apply(pd.to_numeric, errors="coerce").astype(float).to_dict()

    # 2) 재료 적용 (tsp 기준 선형 합산 + 축 가중치)
    for i, a in enumerate(additions or []):
        if not isinstance(a, dict):
            raise HTTPException(status_code=400, detail=f"additions[{i}] 는 객체여야 합니다.")
        ing = a.get("ingredient"); amt = a.get("amount"); unit = a.get("unit","tsp")
        row = deltas.loc[deltas["ingredient"] == ing]
        if row.empty:
            raise HTTPException(status_code=400, detail=f"재료 '{ing}' 없음 (additions[{i}])")
        try:
            amt = float(amt)
        except:
            raise HTTPException(status_code=400, detail=f"amount는 숫자여야 함 (additions[{i}])")
        if amt < 0:
            raise HTTPException(status_code=400, detail=f"amount는 0 이상이어야 함 (additions[{i}])")
        tsp_total = amt * _unit_factor(unit)
        per_tsp = _delta_per_tsp(row.iloc[0])
        for ax in TASTE_AXES:
            v[ax] += per_tsp[ax] * tsp_total * AXIS_WEIGHTS.get(ax, 1.0)

    # 3) 0–10 클리핑 후 Series로 반환
    return pd.Series({ax: float(np.clip(v[ax], 0.0, 10.0)) for ax in TASTE_AXES})


def _cosine_neighbors(final_vec: pd.Series, category_filter: Optional[str], topk: int) -> pd.DataFrame:
    df = foods if not category_filter else foods[foods["category"] == category_filter]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"카테고리 '{category_filter}'에 데이터가 없습니다.")
    X = df[TASTE_AXES].apply(pd.to_numeric, errors="coerce")
    valid = ~X.isna().any(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].astype(float).to_numpy(dtype=float)
    v = final_vec[TASTE_AXES].astype(float).to_numpy(dtype=float) if isinstance(final_vec, pd.Series) else np.asarray([final_vec[a] for a in TASTE_AXES], dtype=float)
    denom = (np.linalg.norm(X, axis=1) * (np.linalg.norm(v) + 1e-8)) + 1e-8
    sims = (X @ v) / denom
    out = df[["name","category"]].copy()
    for i, ax in enumerate(TASTE_AXES):
        out[ax] = X[:, i]
    out["similarity"] = sims
    return out.sort_values("similarity", ascending=False).head(topk).reset_index(drop=True)


def _summarize(base_food: str, base_vec: np.ndarray, neighbor_row: pd.Series, max_points: int = 3) -> List[str]:
    diffs = base_vec - neighbor_row[TASTE_AXES].astype(float).to_numpy(dtype=float)
    pairs = list(zip(TASTE_AXES, diffs))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    msgs = []
    count = 0
    for ax, d in pairs:
        if abs(d) < 0.5:
            continue
        degree = "훨씬 " if abs(d) >= 1.5 else "조금 "
        direction = "높아요" if d > 0 else "낮아요"
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


def _summarize_each(base_vec: pd.Series, neigh_df: pd.DataFrame, max_points: int = 3) -> list[dict]:
    results = []
    b = base_vec[TASTE_AXES].astype(float).to_numpy(dtype=float)
    for _, row in neigh_df.iterrows():
        diffs = b - row[TASTE_AXES].astype(float).to_numpy(dtype=float)
        pairs = list(zip(TASTE_AXES, diffs))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        msgs, cnt = [], 0
        for ax, d in pairs:
            if abs(d) < 0.5:
                continue
            degree = "훨씬 " if abs(d) >= 1.5 else "조금 "
            direction = "높아요" if d > 0 else "낮아요"
            ko = {"sweet":"단맛","salty":"짠맛","sour":"신맛","bitter":"쓴맛","umami":"감칠맛","spicy":"매운맛","fatty":"기름짐"}[ax]
            msgs.append(f"{ko}이 {degree}{direction}.")
            cnt += 1
            if cnt >= max_points:
                break
        if not msgs:
            msgs = ["전반적 맛 프로필이 비슷합니다."]
        results.append({
            "ref": row["name"],
            "similarity": float(row["similarity"]),
            "text": " ".join(msgs)
        })
    return results

#홈페이지로 바로 이동
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)

#예외 처리
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    #오류 위치 정보
    tb = traceback.format_exc().splitlines()[-8:]
    #JSON 형태로 에러 반환
    return JSONResponse(
        status_code=500,
        content={"detail": f"UnhandledError: {type(exc).__name__}: {exc}", "trace": tb},
    )

@app.get("/health", tags=["Health"],
         summary="상태 점검 및 데이터 개요",
         description="서버 상태와 데이터 스냅샷을 반환합니다.\n- 포함: foods/ingredients 개수, 카테고리 분포, 축 목록(axes), 단위 커버리지(unit_counts)\n- 라이트 린트 결과가 있으면 status='warn'으로 표시됩니다.")
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
        "unit_counts": unit_counts,
        "server_time": datetime.now().isoformat(timespec="seconds")
    }

@app.post("/predict", tags=["Predict"], summary="맛 예측",description="기본 음식과 재료 추가로 7가지 맛 축을 예측합니다.\n- 단위: tsp/Tbsp\n- 반환: 최종 맛 벡터, 유사 음식, 설명 문장")
def predict(body: dict):
    try:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON 형식으로 입력해주세요")
        base_food = (body.get("base_food") or "").strip()
        additions = body.get("additions", [])
        if not isinstance(additions, list):
            raise HTTPException(status_code=400, detail="additions는 리스트여야 합니다.")
        category_filter = body.get("category_filter")
        if category_filter is None or (isinstance(category_filter, str) and category_filter.strip() == ""):
            category_filter = None
        elif not isinstance(category_filter, str):
            raise HTTPException(status_code=400, detail="category_filter는 문자열이어야 합니다.")
        topk = int(body.get("topk", 3))
        final_vec = _compute_final_linear(base_food, additions)
        neighbors = _cosine_neighbors(final_vec, category_filter, topk)
        comparisons = _summarize_each(final_vec, neighbors, max_points=3)
        return {
            "input": {"base_food": base_food, "additions": additions, "category_filter": category_filter, "topk": topk},
            "final_scores": {ax: round(float(final_vec[ax]), 1) for ax in TASTE_AXES},
            "neighbors": neighbors.to_dict(orient="records"),
            "comparisons": comparisons
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/predict 처리 중 오류: {type(e).__name__}: {e}")

@app.get("/foods", tags=["Data"],
         summary="음식 리스트",
         description="음식 리스트를 반환합니다.\n- 각 항목: food_id, name, 7개 맛 축, category")
def list_foods():
    items = []
    for _, row in foods.iterrows():
        item = {
            "name": row["name"],
            "category": row.get("category", None)
        }
        items.append(item)
    return {
        "count": len(items),
        "items": items
    }

@app.get("/ingredients", tags=["Data"],
         summary="식재료 리스트",
         description="재료별 맛 변화량(델타)과 기준 단위를 조회합니다.\n- 각 항목: ingredient, unit(예: 1tsp/1Tbsp), 7개 맛 축 델타, category")
def list_ingredients():
    if "ingredient" not in deltas.columns:
        return {"count": 0, "items": []}
    names: List[str] = (
        deltas["ingredient"].dropna().astype(str).unique().tolist()
    )
    return {
        "count": len(names),
        "items": [{"ingredient": n} for n in names]
    }

@app.get("/similar", tags=["Similarity"],
         summary="유사도 추천",
         description="기준 음식과 맛 프로필이 가까운 음식들을 코사인 유사도로 추천합니다.\n- 쿼리: base_food(필수), category_filter(선택), topk(기본 5)\n- 반환: neighbors(이름/카테고리/7축/유사도), sentences(간단 비교 문장)")
def similar_foods(
    base_food: str = Query(..., description="기준 음식명 (예: 곰탕)"),
    category_filter: Optional[str] = Query(None, description="필터(예: soup). 미지정시 전체"),
    topk: int = Query(5, ge=1, le=50, description="추천 개수")
):
    try:
        category_filter = _validate_category(category_filter)
        base_row = _food_row_or_404(base_food)
        base_vec = _as_numeric_axes(base_row)  # np.ndarray
        # 자기 자신 제거를 위해 base_food 이름을 전달하지 않고 벡터 기반 계산만 사용
        neighbors_df = _cosine_neighbors(pd.Series({ax: float(base_row[ax]) for ax in TASTE_AXES}), category_filter, topk+1)
        neighbors_df = neighbors_df[neighbors_df["name"] != base_food].head(topk).reset_index(drop=True)
        sentences = _summarize(base_food, base_vec, neighbors_df.iloc[0]) if not neighbors_df.empty else [
            "유사 이웃을 찾지 못했습니다."
        ]
        return {
            "base_food": base_food,
            "category_filter_applied": category_filter,
            "axes": TASTE_AXES,
            "base_vector": {ax: float(base_row[ax]) for ax in TASTE_AXES},
            "neighbors": neighbors_df.assign(similarity=lambda d: d["similarity"].round(4)).to_dict(orient="records"),
            "sentences": sentences
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar 처리 중 오류: {type(e).__name__}: {e}")

@app.get("/predict1", tags=["Predict"],summary="간단 예측(한 개 재료, 비-JSON)",description=(
             "쿼리스트링 또는 폼으로 입력받아 맛을 예측합니다.\n"
             "- 필수: base_food, ingredient\n"
             "- 옵션: amount(기본 1), unit(tsp|Tbsp, 기본 tsp), category_filter, topk(기본 3)\n"
             "- 반환: 기존 /predict와 유사한 구조"
         ))

@app.post("/predict1", tags=["Predict"], include_in_schema=False)
def predict_one_ingredient(
    # GET 쿼리 또는 POST 폼 양쪽 지원
    base_food: str = Query(..., description="기준 음식명 (예: 곰탕)"),
    ingredient: str = Query(..., description="추가할 재료명 (데이터 기준)"),
    amount: float = Query(1.0, ge=0, description="양 (기본 1)"),
    unit: str = Query("tsp", description="단위: tsp 또는 Tbsp"),
    category_filter: Optional[str] = Query(None, description="추천 필터(예: soup)"),
    topk: int = Query(3, ge=1, le=50, description="추천 개수"),
):
    try:
        # 1) 최종 맛 벡터 계산 (한 개 재료만)
        additions = [{"ingredient": ingredient, "amount": amount, "unit": unit}]
        final_vec = _compute_final_linear(base_food, additions)

        # 2) 유사 음식
        neighbors = _cosine_neighbors(final_vec, category_filter, topk)

        # 3) 간단 비교 문장
        comparisons = _summarize_each(final_vec, neighbors, max_points=3)

        return {
            "input": {
                "base_food": base_food,
                "ingredient": ingredient,
                "amount": amount,
                "unit": unit,
                "category_filter": category_filter,
                "topk": topk,
            },
            "final_scores": {ax: round(float(final_vec[ax]), 1) for ax in TASTE_AXES},
            "neighbors": neighbors.to_dict(orient="records"),
            "comparisons": comparisons,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/predict1 처리 중 오류: {type(e).__name__}: {e}")
    
# 초간단 게시판
class MiniPostIn(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="글 내용")

@app.api_route("/board", methods=["GET", "POST"], tags=["Community"],
              summary="초간단 게시판 (메모리)",
              description="단일 엔드포인트. POST로 글 작성, GET으로 목록 조회. 서버 재시작 시 전체 초기화됩니다.")
def board(request: Request,
          limit: int = 20,
          offset: int = 0,
          payload: Optional[MiniPostIn] = Body(None)):
    """
    - POST /board  { "content": "첫 글!" }  → 글 생성 + 최신 목록 반환
    - GET  /board?limit=20&offset=0       → 최신 목록만 반환
    """
    global _POST_SEQ
    if request.method == "POST":
        if payload is None or not payload.content.strip():
            raise HTTPException(status_code=400, detail="content는 비어있을 수 없습니다.")
        with _POST_LOCK:
            _POST_SEQ += 1
            item = {
                "id": _POST_SEQ,
                "ts": time.time(),
                "content": payload.content.strip()
            }
            _POSTS.append(item)
    items = sorted(_POSTS, key=lambda x: x["ts"], reverse=True)
    sliced = items[offset:offset+limit]
    return {
        "volatile": True,
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "items": sliced
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)