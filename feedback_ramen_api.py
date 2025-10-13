from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI(title="라면 소생자 - 대화형 API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str
    session_id: str | None = None

class ChatOut(BaseModel):
    session_id: str
    reply: str
    state: str

SESSIONS: dict[str, dict] = {}

# 규칙(이슈 → 기본 권장안)
RULES = {
    "salty_high": {
        "detect": ["짜", "염도 높"],
        "ask": "얼마나 짠가요? (조금 / 보통 / 많이)",
        "base": [
            {"ingredient":"물", "amount_ml": 100},
            {"ingredient":"면 추가", "amount_note":"1/2 줌"}
        ]
    },
    "salty_low": {
        "detect": ["싱겁", "간이 약", "심심", "밋밋"],
        "ask": "얼마나 싱거운가요? (조금 / 보통 / 많이)",
        "base": [
            {"ingredient":"소금", "amount_tsp": 0.5},
            {"ingredient":"간장", "amount_tsp": 1.0}
        ]
    },
    "umami_low": {
        "detect": ["밍밍", "감칠맛 없", "맛이 없"],
        "ask": "감칠맛이 얼마나 부족하나요? (조금 / 보통 / 많이)",
        "base": [
            {"ingredient":"다시다/조미료", "amount_tsp": 0.25},
            {"ingredient":"멸치액젓", "amount_tsp": 0.5}
        ]
    },
    "spicy_high": {
        "detect": ["맵", "얼큰"],
        "ask": "매운 정도가 어느 정도인가요? (조금 / 보통 / 많이)",
        "base": [
            {"ingredient":"물", "amount_ml": 100},
            {"ingredient":"우유", "amount_ml": 50}
        ]
    },
    "fatty_high": {
        "detect": ["느끼", "기름짐 많"],
        "ask": "느끼함이 어느 정도인가요? (조금 / 보통 / 많이)",
        "base": [
            {"ingredient":"식초", "amount_tsp": 0.5},
            {"ingredient":"고춧가루", "amount_tsp": 0.5}
        ]
    },
}

INTENSITY = {
    "조금": 0.7, "살짝": 0.7, "살짝만": 0.7,
    "보통": 1.0, "그냥": 1.0,
    "많이": 1.5, "엄청": 1.5, "매우": 1.5
}

def detect_issue(msg: str) -> tuple[str | None, dict | None]:
    m = msg.strip()
    for code, rule in RULES.items():
        if any(k in m for k in rule["detect"]):
            return code, rule
    return None, None


def format_final_suggestion(issue_code: str, scale: float) -> str:
    rule = RULES[issue_code]
    lines = ["알맞은 조치를 안내드릴게요."]
    for item in rule["base"]:
        name = item["ingredient"]
        if "amount_tsp" in item:
            amt = round(item["amount_tsp"] * scale, 2)
            lines.append(f"• {name}: {amt} tsp")
        elif "amount_ml" in item:
            amt = int(item["amount_ml"] * scale)
            lines.append(f"• {name}: {amt} ml")
        else:
            note = item.get("amount_note", "조금")
            # note는 양적 스케일이 애매하므로 텍스트만 표시
            lines.append(f"• {name}: {note}")
    lines.append("\nTip: 추가 → 30초 끓이기 → 맛보기. 부족하면 같은 양 한 번 더!")
    return "\n".join(lines)


@app.post("/chat", response_model=ChatOut)
def chat(input: ChatIn):
    # 세션 준비
    sid = input.session_id or str(uuid.uuid4())
    state = SESSIONS.get(sid, {"state": "await_issue"})

    msg = input.message.strip()

    if state["state"] == "await_issue":
        code, rule = detect_issue(msg)
        if not code:
            return ChatOut(session_id=sid, state="await_issue",
                           reply="말씀을 이해하지 못했어요. 위의 키워드를 이용하여 라면의 상태를 다시 알려주세요")
        # 다음 질문 준비
        SESSIONS[sid] = {"state": "await_intensity", "issue_code": code}
        return ChatOut(session_id=sid, state="await_intensity", reply=rule["ask"])

    elif state["state"] == "await_intensity":
        # 강도 해석
        key = None
        for k in INTENSITY.keys():
            if k in msg:
                key = k; break
        if not key:
            return ChatOut(session_id=sid, state="await_intensity",
                           reply="강도를 알려주세요. (조금 / 보통 / 많이)")
        scale = INTENSITY[key]
        issue_code = state.get("issue_code")
        answer = format_final_suggestion(issue_code, scale)
        # 세션을 다음 대화로 초기화(새 문제 받기)
        SESSIONS[sid] = {"state": "await_issue"}
        return ChatOut(session_id=sid, state="await_issue", reply=answer)

    else:
        # 비정상 상태 복구
        SESSIONS[sid] = {"state": "await_issue"}
        return ChatOut(session_id=sid, state="await_issue",
                       reply="처음부터 다시 도와드릴게요! 라면이 어떤가요?")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("feedback_ramen_chat_api:app", host="127.0.0.1", port=8080, reload=True)