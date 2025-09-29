# api.py
from fastapi import FastAPI, Query, HTTPException, Request, Body
from mvp_core import load_data, compute_final_taste, nearest_foods, compare_sentence, TASTE_AXES
from datetime import datetime
from typing import Optional, List
import numpy as np, pandas as pd
from fastapi.responses import RedirectResponse, JSONResponse
import traceback
from pydantic import BaseModel, Field
import time, threading

tags_metadata = [
    {"name": "Health",     "description": "서버 상태와 데이터 개요"},
    {"name": "Predict",    "description": "맛 예측 관련 엔드포인트"},
    {"name": "Similarity", "description": "유사 음식 탐색"},
    {"name": "Data",       "description": "데이터 조회/점검"},
]
app = FastAPI(
    title="맛 예측 API",
    openapi_tags=tags_metadata
)
foods, deltas = load_data()
_POSTS: list[dict] = []
_POST_SEQ = 0
_POST_LOCK = threading.Lock()
# --- compare_sentence: 버전별 호환 래퍼 ---
def _compare_sentence_safe(base_vec: pd.Series, neigh_df: pd.DataFrame):
    try:
        return compare_sentence(base_vec, neigh_df, max_points=3)
    except TypeError:
        try:
            return compare_sentence(base_vec, neigh_df, 3)
        except TypeError:
            try:
                return compare_sentence(base_vec, neigh_df)
            except Exception:
                return []
            
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

def _summarize_each(base_vec: pd.Series, neigh_df: pd.DataFrame, max_points: int = 3) -> list[dict]:
    results = []
    b = base_vec[TASTE_AXES].astype(float).to_numpy(dtype=float)
    for _, row in neigh_df.iterrows():
        diffs = b - row[TASTE_AXES].astype(float).to_numpy(dtype=float)
        pairs = list(zip(TASTE_AXES, diffs))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        msgs, cnt = [], 0
        for ax, d in pairs:
            if abs(d) < 0.5:  # 너무 미세하면 생략
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

# --- compute_final_taste: 신/구 시그니처 호환 ---
def _compute_final_any_version(base_food: str, additions: list[dict]) -> pd.Series:
    # 1) 신버전: (foods, deltas, base_food, additions)
    try:
        res = compute_final_taste(foods, deltas, base_food, additions)
        final = res["final"] if isinstance(res, dict) and "final" in res else res
        return final if isinstance(final, pd.Series) else pd.Series(final, index=TASTE_AXES)
    except TypeError:
        pass  # 구버전일 가능성

    # 2) 구버전: (base_vec, additions, deltas)
    row = foods.loc[foods["name"] == base_food]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"음식 '{base_food}' 없음.")
    base_vec = row[TASTE_AXES].iloc[0].astype(float).values
    res_old = compute_final_taste(base_vec, additions, deltas)
    if isinstance(res_old, dict) and "final" in res_old:
        fin = res_old["final"]
        return fin if isinstance(fin, pd.Series) else pd.Series(fin, index=TASTE_AXES)
    # numpy 배열 등
    return pd.Series(res_old, index=TASTE_AXES)

# --- nearest_foods: 신/구 시그니처 호환 ---
def _nearest_foods_any_version(final_vec: pd.Series, category_filter: str | None, topk: int) -> pd.DataFrame:
    # 신버전: (foods, final_vec, category_filter=None, topk=3)
    try:
        return nearest_foods(foods, final_vec, category_filter=category_filter, topk=topk)
    except TypeError:
        # 구버전: (final_vec, foods, category=..., topk=...)
        return nearest_foods(final_vec, foods, category=category_filter, topk=topk)

def _unit_factor(u: str) -> float:
    ui = str(u or "").strip().lower()
    if "tbsp" in ui: return 3.0        # 1 Tbsp = 3 tsp
    if "tsp"  in ui: return 1.0
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

    # 2) 재료 적용
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
            v[ax] += per_tsp[ax] * tsp_total

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
    v = final_vec[TASTE_AXES].astype(float).to_numpy(dtype=float)
    denom = (np.linalg.norm(X, axis=1) * (np.linalg.norm(v) + 1e-8)) + 1e-8
    sims = (X @ v) / denom
    out = df[["name","category"]].copy()
    for i, ax in enumerate(TASTE_AXES):
        out[ax] = X[:, i]
    out["similarity"] = sims
    return out.sort_values("similarity", ascending=False).head(topk).reset_index(drop=True)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)

# HTTPException은 기본 처리 유지, 나머지 모든 예외를 JSON으로 노출
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc().splitlines()[-8:]  # 마지막 8줄만
    return JSONResponse(
        status_code=500,
        content={"detail": f"UnhandledError: {type(exc).__name__}: {exc}", "trace": tb},
    )

@app.get("/health", tags=["Health"], 
            summary="상태 점검 및 데이터 개요",
            description="서버 상태와 데이터 스냅샷을 반환합니다.\n- 포함: foods/ingredients 개수, 카테고리 분포, 축 목록(axes), 단위 커버리지(unit_counts)\n- 라이트 린트 결과가 있으면 status='warn'으로 표시됩니다."
            )
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

@app.get("/foods", tags=["Data"], 
            summary="음식 리스트",
            description="음식 리스트를 반환합니다.\n- 각 항목: food_id, name, 7개 맛 축, category")
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

@app.get("/ingredients", tags=["Data"], 
            summary="식재료 리스트",
            description="재료별 맛 변화량(델타)과 기준 단위를 조회합니다.\n- 각 항목: ingredient, unit(예: 1tsp/1Tbsp), 7개 맛 축 델타, category")
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

@app.post("/predict", tags=["Predict"],
            summary="맛 예측",
            description="기본 음식과 재료 추가로 7가지 맛 축을 예측합니다.\n- 단위: tsp/Tbsp\n- 반환: 최종 맛 벡터, 유사 음식, 설명 문장")
def predict(body: dict):
    try:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON 본문을 보내주세요.")
        base_food = (body.get("base_food") or "").strip()
        additions = body.get("additions", [])
        category_filter = body.get("category_filter")
        topk = int(body.get("topk", 3))

        final_vec = _compute_final_linear(base_food, additions)
        neighbors = _cosine_neighbors(final_vec, category_filter, topk)
        comparisons = _summarize_each(final_vec, neighbors, max_points=3)

        return {
            "input": {"base_food": base_food, "additions": additions,
                      "category_filter": category_filter, "topk": topk},
            "final_scores": {ax: round(float(final_vec[ax]), 1) for ax in TASTE_AXES},
            "neighbors": neighbors.to_dict(orient="records"),
            "comparisons": comparisons
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/predict 처리 중 오류: {type(e).__name__}: {e}")
    
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
    
@app.get("/predict/examples", tags=["Predict"], summary="입력 예시 보기")
def predict_examples():
    return {
        "examples": [
            {"id": "minimal", "summary": "추가 재료 없음",
             "payload": {"base_food":"곰탕","additions":[]}},
            {"id": "one_ingredient", "summary": "한 가지 재료",
             "payload": {"base_food":"곰탕","additions":[{"ingredient":"된장","amount":1,"unit":"Tbsp"}],
                         "category_filter":"soup","topk":3}},
            {"id": "multi_ingredients", "summary": "재료 여러 개",
             "payload": {"base_food":"김치찌개",
                         "additions":[
                           {"ingredient":"설탕","amount":1,"unit":"tsp"},
                           {"ingredient":"다시다","amount":0.5,"unit":"tsp"},
                           {"ingredient":"고춧가루","amount":0.5,"unit":"Tbsp"}],
                         "category_filter":"stew","topk":5}}
        ]
    }

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
                "ts": time.time(),   # UNIX epoch
                "content": payload.content.strip()
            }
            _POSTS.append(item)

    # 최신순 정렬 후 페이징
    items = sorted(_POSTS, key=lambda x: x["ts"], reverse=True)
    sliced = items[offset:offset+limit]

    return {
        "volatile": True,            # 재시작 시 사라짐 안내
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "items": sliced
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)