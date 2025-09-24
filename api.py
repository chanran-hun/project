# api.py
from fastapi import FastAPI
from mvp_core import load_data, compute_final_taste, nearest_foods, compare_sentence, TASTE_AXES
import numpy as np

app = FastAPI()
foods, deltas = load_data()

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
