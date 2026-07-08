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
from datetime import date

import streamlit as st
import anthropic

MODEL_NAME = "claude-sonnet-5"

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
# 스타일 (모바일 앱 느낌의 다크 테마 유지)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
  --accent: #43d3b0;
  --accent-dim: rgba(67, 211, 176, 0.16);
  --surface: #383d43;
  --base-deep: #24272b;
  --text-muted: #9aa1a8;
}
.block-container { max-width: 520px; padding-top: 1.5rem; }
#MainMenu, header, footer { visibility: hidden; }

.cl-wordmark { font-size: 30px; font-weight: 800; letter-spacing: -0.5px; margin: 0 0 4px; }
.cl-sub { color: var(--text-muted); font-size: 14px; margin: 0 0 20px; }

.cl-banner { background: var(--surface); border-radius: 20px; padding: 20px; margin-bottom: 22px; }
.cl-banner__label { color: var(--accent); font-size: 13px; font-weight: 700; margin: 0 0 6px; }
.cl-banner__text { font-size: 16px; font-weight: 600; margin: 0; }

.cl-result { background: var(--surface); border-radius: 20px; padding: 24px; text-align: center; }
.cl-result__label { color: var(--text-muted); font-size: 13px; margin: 0; }
.cl-result__score { color: var(--accent); font-size: 48px; font-weight: 800; margin: 4px 0 2px; }
.cl-result__type { color: var(--text-muted); font-size: 14px; font-weight: 600; margin: 0 0 10px; }
.cl-result__summary { font-size: 15px; line-height: 1.5; margin: 0 0 16px; }
.cl-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 8px; }
.cl-chips span { background: var(--base-deep); font-size: 12px; padding: 6px 12px; border-radius: 999px; }
.cl-chips--accent span { background: var(--accent-dim); color: var(--accent); }

.cl-rank { display: flex; align-items: center; gap: 14px; background: var(--surface);
  border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }
.cl-rank.is-me { background: var(--accent-dim); border: 1px solid var(--accent); }
.cl-rank__num { width: 26px; text-align: center; color: var(--accent); font-weight: 800; }
.cl-rank__body { flex: 1; }
.cl-rank__name { font-size: 14px; font-weight: 700; }
.cl-rank__product { font-size: 12px; color: var(--text-muted); }
.cl-rank__score { font-size: 15px; font-weight: 800; margin-right: 10px; }
.cl-rank__link { font-size: 11px; color: var(--accent); text-decoration: none; white-space: nowrap; }

.cl-countdown { background: var(--surface); border-radius: 20px; padding: 22px; text-align: center; margin-bottom: 14px; }
.cl-countdown__dday { color: var(--accent); font-size: 38px; font-weight: 800; margin: 0; }
.cl-countdown__label { color: var(--text-muted); font-size: 13px; margin: 4px 0 0; }
.cl-today { background: var(--accent-dim); border: 1px solid var(--accent); border-radius: 14px;
  padding: 16px; margin-bottom: 14px; }
.cl-today__label { color: var(--accent); font-size: 12px; font-weight: 700; margin: 0 0 4px; }
.cl-today__text { font-size: 15px; font-weight: 600; margin: 0; }
.cl-routine { display: flex; gap: 12px; background: var(--surface); border-radius: 10px;
  padding: 12px 14px; font-size: 13px; margin-bottom: 8px; }
.cl-routine__day { color: var(--accent); font-weight: 700; flex-shrink: 0; min-width: 42px; }

div.stButton > button { border-radius: 14px; font-weight: 700; }
</style>
"""


# ---------------------------------------------------------------------------
# 화면 렌더링
# ---------------------------------------------------------------------------
def go(screen: str) -> None:
    st.session_state.screen = screen


def render_home() -> None:
    st.markdown('<p class="cl-wordmark">clozkin</p>', unsafe_allow_html=True)
    st.markdown('<p class="cl-sub">뷰티 입문 남성을 위한 AI 스킨케어 가이드</p>',
                unsafe_allow_html=True)

    diagnosis = st.session_state.get("last_diagnosis")
    banner_text = diagnosis["summary"] if diagnosis and diagnosis.get("summary") \
        else "피부 진단부터 시작해보세요"
    st.markdown(
        f'<div class="cl-banner"><p class="cl-banner__label">오늘의 1분 케어</p>'
        f'<p class="cl-banner__text">{banner_text}</p></div>',
        unsafe_allow_html=True,
    )

    st.button("◎  피부 진단하기\n\n사진 한 장으로 지금 내 피부 상태 확인",
              use_container_width=True, on_click=go, args=("diagnose",))
    st.button("◆  우리 동네 피부랭킹\n\n피부 좋은 남자들은 뭘 쓸까",
              use_container_width=True, on_click=go, args=("ranking",))
    st.button("▲  D-day 케어 모드\n\n소개팅·면접 전 집중 관리",
              use_container_width=True, on_click=go, args=("event",))


def render_diagnose(client: anthropic.Anthropic | None) -> None:
    st.button("‹ 홈으로", on_click=go, args=("home",))
    st.subheader("피부 진단")

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

    if image_bytes and st.button("이 사진으로 진단하기", use_container_width=True):
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
            f'<p class="cl-result__label">피부 스코어</p>'
            f'<p class="cl-result__score">{result.get("score", "-")}</p>'
            f'<p class="cl-result__type">피부 타입: {result.get("skin_type", "-")}</p>'
            f'<p class="cl-result__summary">{result.get("summary", "")}</p>'
            f'<div class="cl-chips">{concerns}</div>'
            f'<div class="cl-chips cl-chips--accent">{ingredients}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.button("동네 랭킹 보러가기", use_container_width=True,
                  on_click=go, args=("ranking",))


def render_ranking() -> None:
    st.button("‹ 홈으로", on_click=go, args=("home",))
    st.subheader("우리 동네 피부랭킹")

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
    st.button("‹ 홈으로", on_click=go, args=("home",))
    st.subheader("D-day 케어 모드")

    if client is None:
        st.error("ANTHROPIC_API_KEY가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
        return

    st.markdown('<p style="color:#9aa1a8;font-size:13px;margin-bottom:6px;">'
                '어떤 이벤트를 준비하시나요?</p>', unsafe_allow_html=True)
    event_key = st.radio(
        "이벤트", list(EVENT_LABELS.keys()),
        format_func=lambda k: EVENT_LABELS[k], horizontal=True,
        label_visibility="collapsed",
    )
    target_date = st.date_input("언제인가요?", min_value=date.today())

    if st.button("케어 루틴 만들기", use_container_width=True):
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
            f'<div class="cl-today"><p class="cl-today__label">오늘 할 일</p>'
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
    st.set_page_config(page_title="clozkin", page_icon="◎", layout="centered")
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
