# feedback_ramen_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="라면 피드백 API",
    description="사용자가 라면에 대한 불만(예: 싱겁다/짜다/맵다 등)을 보내면 즉시 개선안을 제시합니다.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackIn(BaseModel):
    problem: str = Field(..., min_length=1, description="라면 상태에 대한 자유서술식 피드백 (예: 너무 싱겁다)")

# 키워드 → 이슈 설명 → 권장 재료와 기본 권장량(tsp/기타)
RULES = [
    # 짠맛
    (["싱겁", "간이 약", "심심", "밋밋"],  "짠맛 부족",
     [{"ingredient":"소금", "amount":"0.5 tsp"}, {"ingredient":"간장", "amount":"1 tsp"}]),
    (["짜", "염도 높"],                 "짠맛 과다",
     [{"ingredient":"물", "amount":"100 ml"}, {"ingredient":"면 추가", "amount":"1/2 줌"}]),

    # 감칠맛
    (["밍밍", "감칠맛 없", "맛이 없"],    "감칠맛 부족",
     [{"ingredient":"다시다/조미료", "amount":"0.25 tsp"}, {"ingredient":"멸치액젓", "amount":"0.5 tsp"}]),

    # 매운맛
    (["맵", "너무 얼큰"],               "매운맛 과다",
     [{"ingredient":"물", "amount":"100 ml"}, {"ingredient":"우유", "amount":"50 ml"}]),
    (["안 매", "매운맛 약"],             "매운맛 부족",
     [{"ingredient":"고춧가루", "amount":"0.5 tsp"}, {"ingredient":"청양고추", "amount":"조금"}]),

    # 기름짐
    (["느끼", "기름짐 많"],              "기름짐 과다",
     [{"ingredient":"식초", "amount":"0.5 tsp"}, {"ingredient":"고춧가루", "amount":"0.5 tsp"}]),

    # 단맛
    (["달", "당도 높"],                 "단맛 과다",
     [{"ingredient":"소금", "amount":"0.25 tsp"}, {"ingredient":"식초", "amount":"0.25 tsp"}]),
    (["안 달", "달지 않"],               "단맛 부족",
     [{"ingredient":"설탕", "amount":"0.25 tsp"}]),

    # 신맛
    (["시", "신맛 강"],                  "신맛 과다",
     [{"ingredient":"물", "amount":"100 ml"}, {"ingredient":"설탕", "amount":"0.5 tsp"}]),

    # 쓴맛
    (["써", "탄 맛"],                   "쓴맛 과다",
     [{"ingredient":"설탕", "amount":"0.25 tsp"}, {"ingredient":"물", "amount":"50~100 ml"}]),
]

def detect_issue(text: str):
    t = text.strip()
    for keys, issue, recs in RULES:
        if any(k in t for k in keys):
            return issue, recs
    return None, None

@app.post("/feedback", summary="라면 문제 → 즉시 개선안 제시")
def feedback(data: FeedbackIn):
    issue, recs = detect_issue(data.problem)
    if not issue:
        raise HTTPException(
            status_code=400,
            detail="문제를 인식하지 못했습니다. 예) '너무 싱겁다', '너무 짜다', '너무 맵다', '느끼하다', '밍밍하다' 등"
        )

    # 보수적 1차 권장량 + 재시도 가이드
    guidance = (
        f"라면이 ‘{issue}’로 판단됩니다. 먼저 아래 중 1가지를 추가하고 30초 끓인 뒤 맛을 다시 보세요. "
        f"여전히 부족하면 동일 재료를 동일량만큼 한 번 더 추가하세요."
    )

    return {
        "food": "라면",                         # 음식 입력 없이 고정
        "detected_issue": issue,               # 감지된 문제(짠맛 부족 등)
        "first_actions": recs,                 # 권장 재료와 1차 권장량
        "guidance": guidance,                  # 사용 가이드
        "notes": [
            "간 조정은 항상 ‘조금씩 → 맛보기 → 조금씩’ 순서로 하세요.",
            "물을 추가할 때는 스프 농도도 함께 희석됩니다. 간장/소금으로 재보정하세요.",
            "면/건더기 추가 시는 1~2분 더 끓여 익힘 상태를 맞추세요."
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("feedback_ramen_api:app", host="127.0.0.1", port=8080, reload=True)
