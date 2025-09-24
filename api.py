# api.py
from fastapi import FastAPI, Query
from mvp_core import load_data, compute_final_taste, nearest_foods, compare_sentence, TASTE_AXES
import numpy as np
from datetime import datetime
from typing import Optional

app = FastAPI()
foods, deltas = load_data()

@app.get("/health")
def health():
    """
    서버 헬스 체크용 엔드포인트.
    - 상태: ok
    - 로드된 데이터 개수
    - 맛 축(axes)
    - 서버 시간
    """
    return {
        "status": "ok",
        "foods_count": int(foods.shape[0]),
        "ingredients_count": int(deltas["ingredient"].nunique()),
        "axes": TASTE_AXES,
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

if __name__ == "__main__":
    import uvicorn, webbrowser
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
