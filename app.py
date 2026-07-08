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
import json
import base64
from io import BytesIO
from datetime import date

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
def logo_data_uri(size: int = 320) -> str | None:
    """브랜드 로고를 축소해 data URI 로 반환 (없으면 None)."""
    try:
        img = Image.open(LOGO_PATH).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
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
.cl-logo-wrap { display: flex; justify-content: center; margin: 2px 0 0; }
.cl-logo { width: 120px; height: 120px; object-fit: contain; border-radius: 30px;
  background: #fbfdfd; padding: 15px;
  box-shadow: 0 16px 46px rgba(67, 211, 176, 0.20), 0 0 0 1px rgba(255, 255, 255, 0.08); }
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

/* ---- 홈 네비게이션 카드 (컨테이너 위에 투명 버튼 오버레이) ---- */
[class*="st-key-navcard_"] { position: relative; margin-bottom: 14px; }
.cl-nav {
  position: relative; border-radius: 22px; padding: 22px 22px 46px;
  background: var(--glass); border: 1px solid var(--glass-brd);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  overflow: hidden; transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
.cl-nav::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(94, 234, 212, 0.5), transparent); }
.cl-nav__top { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.cl-nav__idx { font-family: 'Space Grotesk', monospace; font-size: 12px; font-weight: 700;
  color: var(--accent); letter-spacing: 1px; }
.cl-nav__tag { font-family: 'Space Grotesk', monospace; font-size: 10px; letter-spacing: 2.5px;
  color: var(--muted); text-transform: uppercase; }
.cl-nav__title { font-size: 21px; font-weight: 800; letter-spacing: -0.6px; margin-bottom: 7px; }
.cl-nav__desc { font-size: 13px; color: var(--muted); line-height: 1.55; max-width: 84%; }
.cl-nav__arrow { position: absolute; right: 22px; bottom: 18px; color: var(--accent);
  font-size: 17px; transition: transform 0.25s ease; }

[class*="st-key-navcard_"]:hover .cl-nav {
  transform: translateY(-2px); border-color: rgba(67, 211, 176, 0.5);
  box-shadow: 0 0 0 1px rgba(67, 211, 176, 0.2), 0 16px 44px rgba(67, 211, 176, 0.12);
}
[class*="st-key-navcard_"]:hover .cl-nav__arrow { transform: translateX(4px); }

/* 카드 전체를 덮는 투명 클릭 레이어 */
[class*="st-key-navcard_"] .stButton { position: absolute; inset: 0; margin: 0; z-index: 4; }
[class*="st-key-navcard_"] .stButton > button { width: 100%; height: 100%; opacity: 0; border: 0; }

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
    with st.container(key=f"navcard_{target}"):
        st.markdown(
            f'<div class="cl-nav">'
            f'<div class="cl-nav__top"><span class="cl-nav__idx">{idx}</span>'
            f'<span class="cl-nav__tag">{tag}</span></div>'
            f'<div class="cl-nav__title">{title}</div>'
            f'<div class="cl-nav__desc">{desc}</div>'
            f'<span class="cl-nav__arrow">→</span></div>',
            unsafe_allow_html=True,
        )
        st.button(title, key=f"navbtn_{target}", on_click=go, args=(target,),
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


if __name__ == "__main__":
    main()
