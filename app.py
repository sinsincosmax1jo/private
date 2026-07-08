"""
clozkin - 뷰티 입문 남성을 위한 AI 스킨케어 가이드 MVP (Streamlit 버전)

Streamlit Cloud 배포용. 기존 Flask 단일 파일 버전을 Streamlit으로 포팅했다.
  - 피부 진단: 카메라/사진 업로드 -> Claude Vision 분석
  - 우리 동네 피부랭킹: 진단 점수를 목업 랭킹에 반영
  - D-day 케어 모드: 이벤트 + 목표일 -> Claude가 카운트다운 루틴 생성

실행 방법(로컬):
    pip install -r requirements.txt
    # ANTHROPIC_API_KEY 환경변수 또는 .streamlit/secrets.toml 에 설정
    streamlit run app.py

Streamlit Cloud:
    앱 설정 > Secrets 에 ANTHROPIC_API_KEY = "sk-ant-..." 추가
"""
import os
import re
import html
import json
import base64
from io import BytesIO
from datetime import date

import numpy as np
import streamlit as st
import anthropic
from PIL import Image

MODEL_NAME = "claude-sonnet-5"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clozkin_logo.png")

# ---------------------------------------------------------------------------
# 우리 동네 피부 랭킹 - 목업 데이터 (실제 서비스에서는 DB에서 조회)
# ---------------------------------------------------------------------------
MOCK_RANKING = [
    {"name": "김O우", "score": 91, "product": "라운드랩 자작나무 수분 크림"},
    {"name": "이O훈", "score": 87, "product": "아누아 어성초 77 토너"},
    {"name": "박O진", "score": 83, "product": "달바 백자 크림"},
    {"name": "최O민", "score": 79, "product": "라로슈포제 시카플라스트"},
    {"name": "정O석", "score": 74, "product": "닥터지 블랙스네일 크림"},
    {"name": "강O우", "score": 68, "product": "센카 퍼펙트 워터 클렌징"},
    {"name": "조O현", "score": 63, "product": "이니스프리 그린티 세럼"},
]

EVENT_LABELS = {
    "date": "소개팅",
    "interview": "면접",
    "wedding": "결혼식",
    "meeting": "상견례",
    "dating": "데이트",
}


# ---------------------------------------------------------------------------
# 유틸 & AI 호출
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """환경변수 우선, 없으면 Streamlit secrets 에서 API 키를 읽는다."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:  # noqa: BLE001 - secrets 미설정 시 KeyError/FileNotFoundError 등
        return None


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


@st.cache_data(show_spinner=False)
def logo_data_uri(size: int = 300) -> str | None:
    """브랜드 로고에서 흰 배경을 제거하고 다크 테마용으로 리컬러한 data URI 반환.

    - 흰색 배경 -> 투명
    - 진회색(글자/손 아이콘) -> 밝은 텍스트 색으로 변경 (다크 배경에서 보이도록)
    - 민트 계열 포인트 -> 원색 유지
    """
    try:
        img = Image.open(LOGO_PATH).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)

    arr = np.array(img).astype(np.int16)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    mn = np.minimum(np.minimum(r, g), b)
    mx = np.maximum(np.maximum(r, g), b)
    chroma = mx - mn
    ink = 255 - mn                 # 흰색 배경일수록 0, 잉크(어두움/채색)일수록 큼
    colored = chroma > 45          # 민트 포인트
    gray = (~colored) & (ink > 8)  # 무채색 잉크 = 글자/아이콘

    out = np.zeros_like(arr)
    out[..., 0] = np.where(gray, 238, r)   # 회색 -> 밝은 텍스트색(#eef2f4)
    out[..., 1] = np.where(gray, 242, g)
    out[..., 2] = np.where(gray, 244, b)
    alpha = np.where(colored, np.maximum(ink, 210), ink)
    out[..., 3] = np.clip(np.minimum(alpha, a), 0, 255)

    result = Image.fromarray(out.astype(np.uint8), "RGBA")
    buf = BytesIO()
    result.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text: str) -> dict:
    """모델 응답에서 JSON 블록만 안전하게 추출."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _text_from_response(response) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def diagnose_skin(client: anthropic.Anthropic, image_bytes: bytes, media_type: str) -> dict:
    """얼굴 사진을 Claude Vision으로 분석해 피부 상태 dict 반환."""
    b64_payload = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "너는 남성 뷰티 초보자를 위한 친절한 피부 분석 AI야. "
        "첨부된 얼굴 사진을 보고 피부 상태를 분석해줘. "
        "전문 용어는 최소화하고 초보자도 이해하기 쉽게 설명해. "
        "아래 JSON 형식으로만, 다른 설명 없이 응답해:\n"
        '{"score": 0-100 사이 정수, '
        '"skin_type": "건성/지성/복합성/민감성 중 하나", '
        '"concerns": ["피부 고민 키워드 2~3개, 짧게"], '
        '"summary": "현재 피부 상태에 대한 한 줄 요약 (초보자 친화적 말투)", '
        '"recommended_ingredients": ["추천 성분 2~3개"]}'
    )
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_payload,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return _extract_json(_text_from_response(response))


CHAT_SYSTEM = (
    "너는 'clozkin'의 AI 뷰티 가이드야. 뷰티에 처음 입문하는 남성 고객을 돕는 게 목표야. "
    "스킨케어를 세안처럼 '당연히 하는 행동'으로 느끼게 도와줘. "
    "전문 용어는 최소화하고, 초보자도 부담 없이 따라할 수 있게 아주 친근하고 짧게 답해. "
    "한 번에 너무 많은 걸 시키지 말고 딱 필요한 것만 골라줘. "
    "답변은 2~4문장 이내로 간결하게. 필요하면 이모지를 가볍게 써도 좋아."
)


def chat_reply(client: anthropic.Anthropic, history: list[dict], diagnosis: dict | None) -> str:
    """채팅 히스토리를 기반으로 Claude 답변 생성. history는 {role, content} 리스트."""
    system = CHAT_SYSTEM
    if diagnosis and diagnosis.get("summary"):
        system += (
            f"\n\n[사용자의 최근 피부 진단 결과] "
            f"점수 {diagnosis.get('score', '-')}, "
            f"피부타입 {diagnosis.get('skin_type', '-')}, "
            f"고민 {', '.join(diagnosis.get('concerns', []))}. "
            f"이 정보를 참고해 맞춤형으로 조언해줘."
        )
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        system=system,
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
    )
    return _text_from_response(response).strip()


def generate_routine(client: anthropic.Anthropic, event_label: str, days_left: int,
                     diagnosis: dict) -> dict:
    """이벤트 종류 + D-day를 기반으로 카운트다운 케어 루틴 생성."""
    prompt = (
        f"사용자는 {days_left}일 뒤 '{event_label}'을 앞두고 있어. "
        f"현재 피부 상태: 피부타입 {diagnosis.get('skin_type')}, "
        f"고민 {', '.join(diagnosis.get('concerns', []))}. "
        "뷰티 초보자도 부담 없이 따라할 수 있는 D-day 역산 케어 루틴을 만들어줘. "
        "너무 많은 단계는 부담스러우니 하루에 1~2가지 행동만 제시해. "
        "아래 JSON 형식으로만 응답해:\n"
        '{"routine": [{"day_label": "D-3", "task": "오늘 할 일 한 줄"}, ...], '
        '"today_task": "오늘(가장 가까운 날) 해야 할 일 한 줄"}'
    )
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_json(_text_from_response(response))


# ---------------------------------------------------------------------------
# 스타일 - 미니멀 / 미래지향 다크 (글래스모피즘 + 그라디언트 글로우)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --accent: #43d3b0;
  --accent-2: #5eead4;
  --accent-dim: rgba(67, 211, 176, 0.14);
  --glass: rgba(255, 255, 255, 0.045);
  --glass-brd: rgba(255, 255, 255, 0.09);
  --text: #eef2f4;
  --muted: #8b949e;
  --ink: #06231d;
}

.stApp {
  background:
    radial-gradient(1100px 560px at 50% -12%, rgba(67, 211, 176, 0.13), transparent 60%),
    radial-gradient(900px 500px at 110% 8%, rgba(94, 234, 212, 0.06), transparent 55%),
    linear-gradient(180deg, #0b0e13 0%, #070a0e 100%);
}
.stApp, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp label {
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--text);
}
.block-container { max-width: 560px; padding-top: 2.2rem; padding-bottom: 4rem; }
#MainMenu, header, footer { visibility: hidden; }

/* ---- 브랜드 / 히어로 ---- */
.cl-logo-wrap { display: flex; justify-content: center; margin: 4px 0 0; }
.cl-logo { width: 148px; height: 148px; object-fit: contain;
  filter: drop-shadow(0 10px 30px rgba(67, 211, 176, 0.28)); }
.cl-badge-tag { text-align: center; font-family: 'Space Grotesk', monospace; font-size: 10.5px;
  letter-spacing: 3px; color: var(--muted); font-weight: 600; margin: 14px 0 0; }

/* 로고 로드 실패 시 텍스트 폴백 */
.cl-brand { display: flex; align-items: center; justify-content: center; gap: 9px; margin-bottom: 4px; }
.cl-brand__dot { width: 9px; height: 9px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 14px var(--accent), 0 0 4px var(--accent); }
.cl-brand__name { font-size: 19px; font-weight: 800; letter-spacing: -0.4px; }

.cl-hero__title { text-align: center; font-size: 38px; line-height: 1.16; font-weight: 800;
  letter-spacing: -1.4px; margin: 22px 0 14px; }
.cl-grad { background: linear-gradient(115deg, var(--accent-2), var(--accent));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cl-hero__sub { text-align: center; color: var(--muted); font-size: 15px; line-height: 1.65;
  margin: 0 auto 24px; max-width: 400px; }

.cl-status-wrap { text-align: center; }
.cl-status { display: inline-flex; align-items: center; gap: 8px; margin: 0 0 26px;
  padding: 9px 15px; border-radius: 999px; background: var(--glass);
  border: 1px solid var(--glass-brd); font-size: 13px; color: var(--text); }
.cl-status b { color: var(--accent); font-weight: 700; }

/* ---- 홈 네비게이션 카드 (버튼 자체가 카드) ---- */
[class*="st-key-navbtn_"] { margin-bottom: 14px; }
[class*="st-key-navbtn_"] .stButton > button {
  display: block; text-align: left; width: 100%;
  padding: 22px 22px 40px; border-radius: 22px; position: relative; overflow: hidden;
  background: var(--glass); border: 1px solid var(--glass-brd);
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
[class*="st-key-navbtn_"] .stButton > button::before { content: ""; position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(94, 234, 212, 0.5), transparent); }
[class*="st-key-navbtn_"] .stButton > button::after { content: "→"; position: absolute;
  right: 22px; bottom: 15px; color: var(--accent); font-size: 17px; transition: transform 0.25s ease; }
[class*="st-key-navbtn_"] .stButton > button:hover {
  transform: translateY(-2px); border-color: rgba(67, 211, 176, 0.5); color: var(--text);
  box-shadow: 0 0 0 1px rgba(67, 211, 176, 0.2), 0 16px 44px rgba(67, 211, 176, 0.12);
}
[class*="st-key-navbtn_"] .stButton > button:hover::after { transform: translateX(4px); }
[class*="st-key-navbtn_"] .stButton > button p { margin: 0; }
[class*="st-key-navbtn_"] .stButton > button p:nth-of-type(1) {
  font-family: 'Space Grotesk', monospace; font-size: 11px; letter-spacing: 2px;
  color: var(--accent); text-transform: uppercase; margin-bottom: 12px; }
[class*="st-key-navbtn_"] .stButton > button p:nth-of-type(2) {
  font-size: 21px; font-weight: 800; letter-spacing: -0.6px; color: var(--text); margin-bottom: 7px; }
[class*="st-key-navbtn_"] .stButton > button p:nth-of-type(3) {
  font-size: 13px; font-weight: 500; color: var(--muted); line-height: 1.55; }

/* ---- 일반 버튼 ---- */
.stButton > button {
  border-radius: 14px; font-weight: 700; letter-spacing: -0.2px;
  border: 1px solid var(--glass-brd); background: var(--glass); color: var(--text);
  transition: border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] {
  background: linear-gradient(115deg, var(--accent-2), var(--accent)); color: var(--ink);
  border: 0; box-shadow: 0 10px 34px rgba(67, 211, 176, 0.28);
}
.stButton > button[kind="primary"]:hover { color: var(--ink); filter: brightness(1.05); }

.st-key-back { margin-bottom: 6px; }
.st-key-back .stButton > button { width: auto; background: transparent; border: 0;
  color: var(--muted); padding: 2px 2px; font-weight: 600; }
.st-key-back .stButton > button:hover { color: var(--accent); }

/* ---- 공통 섹션 제목 ---- */
.cl-h { font-size: 24px; font-weight: 800; letter-spacing: -0.7px; margin: 2px 0 4px; }
.cl-sec { font-family: 'Space Grotesk', monospace; font-size: 11px; letter-spacing: 2px;
  color: var(--muted); text-transform: uppercase; margin: 22px 0 12px; }

/* ---- 진단 결과 ---- */
.cl-result { background: var(--glass); border: 1px solid var(--glass-brd); backdrop-filter: blur(16px);
  border-radius: 24px; padding: 28px 24px; text-align: center; margin-top: 8px; }
.cl-result__label { color: var(--muted); font-size: 12px; letter-spacing: 1px; margin: 0; }
.cl-result__score { font-size: 62px; font-weight: 800; letter-spacing: -2px; margin: 2px 0;
  background: linear-gradient(115deg, var(--accent-2), var(--accent));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cl-result__type { color: var(--muted); font-size: 14px; font-weight: 600; margin: 0 0 12px; }
.cl-result__summary { font-size: 15px; line-height: 1.55; margin: 0 0 18px; }
.cl-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 8px; }
.cl-chips span { background: rgba(255,255,255,0.05); border: 1px solid var(--glass-brd);
  font-size: 12px; padding: 6px 13px; border-radius: 999px; color: var(--text); }
.cl-chips--accent span { background: var(--accent-dim); border-color: transparent; color: var(--accent); }

/* ---- 랭킹 ---- */
.cl-rank { display: flex; align-items: center; gap: 14px; background: var(--glass);
  border: 1px solid var(--glass-brd); border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; }
.cl-rank.is-me { background: var(--accent-dim); border-color: rgba(67,211,176,0.5);
  box-shadow: 0 0 0 1px rgba(67,211,176,0.15); }
.cl-rank__num { width: 26px; text-align: center; font-family: 'Space Grotesk', monospace;
  color: var(--accent); font-weight: 700; }
.cl-rank__body { flex: 1; min-width: 0; }
.cl-rank__name { font-size: 14px; font-weight: 700; }
.cl-rank__product { font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.cl-rank__score { font-family: 'Space Grotesk', monospace; font-size: 16px; font-weight: 700; margin-right: 8px; }
.cl-rank__link { font-size: 11px; color: var(--accent); text-decoration: none; white-space: nowrap; }
.cl-rank__link:hover { text-decoration: underline; }

/* ---- D-day ---- */
.cl-countdown { background: var(--glass); border: 1px solid var(--glass-brd); backdrop-filter: blur(16px);
  border-radius: 24px; padding: 26px; text-align: center; margin-bottom: 14px; }
.cl-countdown__dday { font-family: 'Space Grotesk', monospace; font-size: 52px; font-weight: 700;
  letter-spacing: -1px; margin: 0;
  background: linear-gradient(115deg, var(--accent-2), var(--accent));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cl-countdown__label { color: var(--muted); font-size: 13px; margin: 4px 0 0; letter-spacing: 0.5px; }
.cl-today { background: var(--accent-dim); border: 1px solid rgba(67,211,176,0.5); border-radius: 16px;
  padding: 18px; margin-bottom: 14px; }
.cl-today__label { color: var(--accent); font-family: 'Space Grotesk', monospace; font-size: 11px;
  letter-spacing: 2px; text-transform: uppercase; margin: 0 0 5px; }
.cl-today__text { font-size: 15px; font-weight: 600; margin: 0; }
.cl-routine { display: flex; gap: 14px; align-items: baseline; background: var(--glass);
  border: 1px solid var(--glass-brd); border-radius: 12px; padding: 13px 15px; font-size: 13px;
  margin-bottom: 8px; }
.cl-routine__day { font-family: 'Space Grotesk', monospace; color: var(--accent); font-weight: 700;
  flex-shrink: 0; min-width: 44px; }

/* ---- 플로팅 채팅봇 (우측 하단) ---- */
.st-key-chatwidget {
  position: fixed; right: 22px; bottom: 22px; z-index: 1000;
  width: auto; max-width: calc(100vw - 32px);
}
/* 토글 버튼(FAB) - 열림/닫힘 공통 */
.st-key-chat_fab { display: flex; justify-content: flex-end; }
.st-key-chat_fab .stButton > button {
  width: 58px; height: 58px; border-radius: 50%; padding: 0;
  font-size: 24px; line-height: 1; font-weight: 700;
  background: linear-gradient(115deg, var(--accent-2), var(--accent)); color: var(--ink);
  border: 0; box-shadow: 0 12px 34px rgba(67, 211, 176, 0.4);
  transition: transform 0.2s ease, filter 0.2s ease;
}
.st-key-chat_fab .stButton > button:hover {
  transform: translateY(-2px) scale(1.04); color: var(--ink); filter: brightness(1.05);
}
/* 채팅 패널 */
.cl-chat-panel {
  width: min(320px, calc(100vw - 32px)); margin-bottom: 12px;
  background: rgba(13, 18, 24, 0.92); backdrop-filter: blur(20px);
  border: 1px solid var(--glass-brd); border-radius: 22px; overflow: hidden;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.55);
}
.cl-chat-head { padding: 16px 18px; border-bottom: 1px solid var(--glass-brd);
  display: flex; align-items: center; gap: 10px; }
.cl-chat-head__dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 12px var(--accent); flex-shrink: 0; }
.cl-chat-head__name { font-size: 14px; font-weight: 800; letter-spacing: -0.3px; }
.cl-chat-head__sub { font-size: 11px; color: var(--muted); margin-left: auto;
  font-family: 'Space Grotesk', monospace; letter-spacing: 1px; }
.cl-chat-body { max-height: 320px; overflow-y: auto; padding: 16px 16px 4px; }
.cl-chat-body::-webkit-scrollbar { width: 6px; }
.cl-chat-body::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
.cl-msg { font-size: 13px; line-height: 1.5; padding: 9px 13px; border-radius: 14px;
  margin-bottom: 9px; max-width: 85%; word-break: break-word; }
.cl-msg--bot { background: var(--glass); border: 1px solid var(--glass-brd);
  border-bottom-left-radius: 4px; margin-right: auto; }
.cl-msg--user { background: var(--accent-dim); border: 1px solid rgba(67,211,176,0.35);
  color: var(--text); border-bottom-right-radius: 4px; margin-left: auto; text-align: right; }
/* 패널 내부 입력창 */
.st-key-chatwidget .stForm { border: 0; padding: 8px 14px 14px; }
.st-key-chatwidget .stTextInput input {
  background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 12px;
  color: var(--text); font-size: 13px;
}
.st-key-chatwidget .stTextInput input:focus { border-color: var(--accent); box-shadow: none; }
/* 패널·칩·입력 폭을 동일하게 (오른쪽 정렬) */
.st-key-chat_chips, .st-key-chat_form {
  width: min(320px, calc(100vw - 32px)); margin-left: auto;
}
/* 빠른 질문 추천 칩 */
.st-key-chat_chips { padding: 0 14px 2px; }
.st-key-chat_chips [data-testid="column"] { padding: 0 3px; }
.st-key-chat_chips .stButton > button {
  border-radius: 999px; font-size: 11.5px; font-weight: 600; padding: 7px 8px;
  min-height: 0; background: var(--glass); border: 1px solid var(--glass-brd);
  color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.st-key-chat_chips .stButton > button:hover { border-color: var(--accent); color: var(--accent); }

/* ---- 모바일 대응 ---- */
@media (max-width: 480px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem; }
  .cl-logo { width: 120px; height: 120px; }
  .cl-hero__title { font-size: 30px; letter-spacing: -1px; margin: 18px 0 12px; }
  .cl-hero__sub { font-size: 14px; }
  .cl-result__score { font-size: 52px; }
  .cl-countdown__dday { font-size: 42px; }
  [class*="st-key-navbtn_"] .stButton > button p:nth-of-type(2) { font-size: 19px; }
  .st-key-chatwidget { right: 14px; bottom: 14px; }
  .st-key-chat_fab .stButton > button { width: 52px; height: 52px; font-size: 22px; }
  .cl-chat-body { max-height: 42vh; }
}
</style>
"""


# ---------------------------------------------------------------------------
# 화면 렌더링
# ---------------------------------------------------------------------------
def go(screen: str) -> None:
    st.session_state.screen = screen


def back_button() -> None:
    with st.container(key="back"):
        st.button("← 홈으로", key="btn_back", on_click=go, args=("home",))


def section_title(title: str, tag: str) -> None:
    st.markdown(f'<div class="cl-sec">{tag}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cl-h">{title}</div>', unsafe_allow_html=True)


def nav_card(idx: str, tag: str, title: str, desc: str, target: str) -> None:
    # 버튼 라벨을 3개 문단(번호·제목·설명)으로 넘겨 CSS로 카드처럼 스타일링한다.
    label = f"{idx} · {tag}\n\n{title}\n\n{desc}"
    st.button(label, key=f"navbtn_{target}", on_click=go, args=(target,),
              use_container_width=True)


def render_home() -> None:
    uri = logo_data_uri()
    if uri:
        st.markdown(
            f'<div class="cl-logo-wrap"><img class="cl-logo" src="{uri}" alt="clozkin"></div>'
            '<p class="cl-badge-tag">AI BEAUTY GUIDE</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cl-brand"><span class="cl-brand__dot"></span>'
            '<span class="cl-brand__name">clozkin</span></div>'
            '<p class="cl-badge-tag">AI BEAUTY GUIDE</p>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<h1 class="cl-hero__title">세안 다음은,<br>'
        '<span class="cl-grad">당연히 스킨케어.</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="cl-hero__sub">토너·세럼 순서 몰라도 괜찮아요.<br>'
        '사진 한 장으로 지금 내 피부를 읽고, 딱 필요한 것만 알려드릴게요.</p>',
        unsafe_allow_html=True,
    )

    diagnosis = st.session_state.get("last_diagnosis")
    if diagnosis and diagnosis.get("summary"):
        st.markdown(
            f'<div class="cl-status-wrap"><div class="cl-status">'
            f'<b>최근 진단</b> · {diagnosis["summary"]}</div></div>',
            unsafe_allow_html=True,
        )

    nav_card("01", "DIAGNOSIS", "AI 피부 진단",
             "얼굴을 스캔해 유수분·트러블·모공 상태를 30초 만에 분석해요.", "diagnose")
    nav_card("02", "RANKING", "우리 동네 피부 랭킹",
             "같은 동네 남자들의 피부 점수와, 상위권이 실제 쓰는 아이템.", "ranking")
    nav_card("03", "D-DAY", "D-day 케어 모드",
             "소개팅·면접 전, 날짜 역산 집중 관리 루틴을 짜드려요.", "event")


def render_diagnose(client: anthropic.Anthropic | None) -> None:
    back_button()
    section_title("AI 피부 진단", "DIAGNOSIS")

    if client is None:
        st.error("ANTHROPIC_API_KEY가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
        return

    st.caption("얼굴이 잘 보이도록 밝은 곳에서 촬영하거나 사진을 올려주세요.")
    source = st.radio("입력 방식", ["카메라 촬영", "사진 업로드"], horizontal=True,
                      label_visibility="collapsed")

    image_bytes, media_type = None, "image/jpeg"
    if source == "카메라 촬영":
        shot = st.camera_input("사진 촬영", label_visibility="collapsed")
        if shot is not None:
            image_bytes = shot.getvalue()
            media_type = shot.type or "image/jpeg"
    else:
        up = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png", "webp"],
                              label_visibility="collapsed")
        if up is not None:
            image_bytes = up.getvalue()
            media_type = up.type or "image/jpeg"
            st.image(image_bytes, width=240)

    if image_bytes and st.button("이 사진으로 진단하기", type="primary",
                                 use_container_width=True):
        with st.spinner("피부 상태를 분석하는 중..."):
            try:
                result = diagnose_skin(client, image_bytes, media_type)
                st.session_state.last_diagnosis = result
            except json.JSONDecodeError:
                st.error("AI 응답을 해석하지 못했습니다. 다시 시도해주세요.")
            except anthropic.APIError as e:
                st.error(f"AI 호출 중 오류가 발생했습니다: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"알 수 없는 오류: {e}")

    result = st.session_state.get("last_diagnosis")
    if result:
        concerns = "".join(f"<span>{c}</span>" for c in result.get("concerns", []))
        ingredients = "".join(
            f"<span>#{c}</span>" for c in result.get("recommended_ingredients", [])
        )
        st.markdown(
            f'<div class="cl-result">'
            f'<p class="cl-result__label">SKIN SCORE</p>'
            f'<p class="cl-result__score">{result.get("score", "-")}</p>'
            f'<p class="cl-result__type">피부 타입 · {result.get("skin_type", "-")}</p>'
            f'<p class="cl-result__summary">{result.get("summary", "")}</p>'
            f'<div class="cl-chips">{concerns}</div>'
            f'<div class="cl-chips cl-chips--accent">{ingredients}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.button("동네 랭킹 보러가기", use_container_width=True,
                  on_click=go, args=("ranking",))


def render_ranking() -> None:
    back_button()
    section_title("우리 동네 피부 랭킹", "RANKING")

    diagnosis = st.session_state.get("last_diagnosis")
    board = [dict(x) for x in MOCK_RANKING]
    if diagnosis and isinstance(diagnosis.get("score"), (int, float)):
        board.append({
            "name": "나 (진단 결과)",
            "score": diagnosis["score"],
            "product": (diagnosis.get("recommended_ingredients") or ["-"])[0],
            "is_me": True,
        })

    st.caption("피부 진단 결과를 기반으로 순위에 반영했어요" if diagnosis
               else "피부 진단을 하면 내 순위도 함께 볼 수 있어요")

    board.sort(key=lambda x: x["score"], reverse=True)
    for rank, entry in enumerate(board, start=1):
        query = entry["product"].replace(" ", "+")
        link = ("https://www.oliveyoung.co.kr/store/search/getSearchMain.do?query="
                + query)
        st.markdown(
            f'<div class="cl-rank {"is-me" if entry.get("is_me") else ""}">'
            f'<div class="cl-rank__num">{rank}</div>'
            f'<div class="cl-rank__body">'
            f'<div class="cl-rank__name">{entry["name"]}</div>'
            f'<div class="cl-rank__product">{entry["product"]}</div></div>'
            f'<div class="cl-rank__score">{entry["score"]}</div>'
            f'<a class="cl-rank__link" target="_blank" rel="noopener" href="{link}">올리브영 →</a>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_event(client: anthropic.Anthropic | None) -> None:
    back_button()
    section_title("D-day 케어 모드", "D-DAY")

    if client is None:
        st.error("ANTHROPIC_API_KEY가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
        return

    st.markdown('<div class="cl-sec">어떤 이벤트를 준비하시나요?</div>', unsafe_allow_html=True)
    event_key = st.radio(
        "이벤트", list(EVENT_LABELS.keys()),
        format_func=lambda k: EVENT_LABELS[k], horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('<div class="cl-sec">언제인가요?</div>', unsafe_allow_html=True)
    target_date = st.date_input("언제인가요?", min_value=date.today(),
                                label_visibility="collapsed")

    if st.button("케어 루틴 만들기", type="primary", use_container_width=True):
        days_left = (target_date - date.today()).days
        if days_left < 0:
            st.error("목표 날짜는 오늘 이후여야 합니다.")
            return
        diagnosis = st.session_state.get("last_diagnosis") or {
            "skin_type": "정보 없음",
            "concerns": ["일반 컨디션 관리"],
            "summary": "아직 피부 진단을 하지 않았어요.",
        }
        with st.spinner("맞춤 루틴을 짜는 중..."):
            try:
                result = generate_routine(
                    client, EVENT_LABELS.get(event_key, event_key), days_left, diagnosis)
                result["days_left"] = days_left
                result["event_label"] = EVENT_LABELS.get(event_key, event_key)
                st.session_state.last_routine = result
            except json.JSONDecodeError:
                st.error("AI 응답을 해석하지 못했습니다. 다시 시도해주세요.")
            except anthropic.APIError as e:
                st.error(f"AI 호출 중 오류가 발생했습니다: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"알 수 없는 오류: {e}")

    routine = st.session_state.get("last_routine")
    if routine:
        st.markdown(
            f'<div class="cl-countdown">'
            f'<p class="cl-countdown__dday">D-{routine["days_left"]}</p>'
            f'<p class="cl-countdown__label">{routine["event_label"]}까지</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cl-today"><p class="cl-today__label">TODAY</p>'
            f'<p class="cl-today__text">{routine.get("today_task", "")}</p></div>',
            unsafe_allow_html=True,
        )
        for item in routine.get("routine", []):
            st.markdown(
                f'<div class="cl-routine"><span class="cl-routine__day">'
                f'{item.get("day_label", "")}</span><span>{item.get("task", "")}</span></div>',
                unsafe_allow_html=True,
            )


# 빠른 질문 추천 칩 (초보자가 바로 누를 수 있는 예시 질문)
QUICK_QUESTIONS = [
    "토너·세럼 순서 알려줘",
    "지성 피부엔 뭐부터?",
    "여드름 자국 없애는 법",
    "면접 전날 피부 관리",
]


def toggle_chat() -> None:
    st.session_state.chat_open = not st.session_state.get("chat_open", False)


def queue_chat(text: str) -> None:
    """칩/폼에서 보낸 메시지를 다음 렌더에서 처리하도록 예약한다."""
    st.session_state.chat_open = True
    st.session_state.pending_chat = text


def _push_and_reply(client: anthropic.Anthropic | None, text: str) -> None:
    text = text.strip()
    if not text:
        return
    st.session_state.chat_messages.append({"role": "user", "content": text})
    if client is None:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "지금은 AI 연결이 안 돼요. (ANTHROPIC_API_KEY 설정을 확인해주세요)"})
        return
    try:
        reply = chat_reply(client, st.session_state.chat_messages,
                           st.session_state.get("last_diagnosis"))
    except anthropic.APIError as e:
        reply = f"앗, 잠시 문제가 있었어요: {e}"
    except Exception as e:  # noqa: BLE001
        reply = f"앗, 알 수 없는 오류예요: {e}"
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


def render_chat_widget(client: anthropic.Anthropic | None) -> None:
    """모든 화면 우측 하단에 뜨는 플로팅 AI 상담 챗봇."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant",
             "content": "안녕하세요! 뷰티 입문 도와드리는 clozkin 가이드예요 👋 "
                        "스킨케어 뭐든 편하게 물어보세요."}
        ]

    # 칩/폼으로 예약된 메시지를 버블 렌더 전에 처리한다.
    pending = st.session_state.pop("pending_chat", None)
    if pending:
        _push_and_reply(client, pending)

    with st.container(key="chatwidget"):
        chat_open = st.session_state.get("chat_open", False)

        if chat_open:
            # --- 대화 내역 ---
            bubbles = "".join(
                f'<div class="cl-msg cl-msg--{"user" if m["role"] == "user" else "bot"}">'
                f'{html.escape(m["content"]).replace(chr(10), "<br>")}</div>'
                for m in st.session_state.chat_messages
            )
            st.markdown(
                '<div class="cl-chat-panel">'
                '<div class="cl-chat-head"><span class="cl-chat-head__dot"></span>'
                '<span class="cl-chat-head__name">clozkin 가이드</span>'
                '<span class="cl-chat-head__sub">AI</span></div>'
                f'<div class="cl-chat-body">{bubbles}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # --- 빠른 질문 추천 칩 ---
            with st.container(key="chat_chips"):
                cols = st.columns(2, gap="small")
                for i, q in enumerate(QUICK_QUESTIONS):
                    cols[i % 2].button(q, key=f"chip_{i}", on_click=queue_chat,
                                       args=(q,), use_container_width=True)

            # --- 입력창 ---
            with st.form(key="chat_form", clear_on_submit=True):
                user_text = st.text_input(
                    "메시지", placeholder="궁금한 걸 입력해보세요",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("보내기", use_container_width=True)
            if submitted and user_text.strip():
                queue_chat(user_text)
                st.rerun()

        # --- FAB 토글 버튼 ---
        with st.container(key="chat_fab"):
            st.button("✕" if chat_open else "💬", key="btn_chat_fab",
                      on_click=toggle_chat, help="AI 가이드에게 물어보기")


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
def main() -> None:
    page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "◎"
    st.set_page_config(page_title="clozkin", page_icon=page_icon, layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "screen" not in st.session_state:
        st.session_state.screen = "home"

    api_key = get_api_key()
    client = get_client(api_key) if api_key else None

    screen = st.session_state.screen
    if screen == "diagnose":
        render_diagnose(client)
    elif screen == "ranking":
        render_ranking()
    elif screen == "event":
        render_event(client)
    else:
        render_home()

    render_chat_widget(client)


if __name__ == "__main__":
    main()
