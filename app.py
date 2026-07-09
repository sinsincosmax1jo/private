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
import time
import base64
import random
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import anthropic
from PIL import Image

try:  # 위치(GPS) 컴포넌트 - 미설치 시에도 앱이 동작하도록 가드
    from streamlit_geolocation import streamlit_geolocation
    _HAS_GEO = True
except Exception:  # noqa: BLE001
    _HAS_GEO = False

MODEL_NAME = "claude-sonnet-5"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clozkin_logo.png")
# 진단 기록을 누적 저장하는 파일 (사이트 전체 랭킹에 반영)
RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin_records.json")

# ---------------------------------------------------------------------------
# 우리 동네 피부 랭킹 - 목업 데이터 (실제 서비스에서는 DB에서 조회)
# ---------------------------------------------------------------------------
MOCK_RANKING = [
    {"name": "김O우", "score": 91, "gain": 7, "product": "라운드랩 자작나무 수분 크림"},
    {"name": "이O훈", "score": 87, "gain": 12, "product": "아누아 어성초 77 토너"},
    {"name": "박O진", "score": 83, "gain": 5, "product": "달바 백자 크림"},
    {"name": "최O민", "score": 79, "gain": 15, "product": "라로슈포제 시카플라스트"},
    {"name": "정O석", "score": 74, "gain": 9, "product": "닥터지 블랙스네일 크림"},
    {"name": "강O우", "score": 68, "gain": 18, "product": "센카 퍼펙트 워터 클렌징"},
    {"name": "조O현", "score": 63, "gain": 3, "product": "이니스프리 그린티 세럼"},
]

# 많이 쓰는 화장품 랭킹 - 목업 데이터 (users = 동네 사용자 수)
MOCK_PRODUCT_RANKING = [
    {"name": "라운드랩 자작나무 수분 크림", "category": "수분크림", "users": 1284},
    {"name": "아누아 어성초 77 토너", "category": "토너", "users": 1102},
    {"name": "닥터지 레드 블레미쉬 진정 크림", "category": "진정크림", "users": 951},
    {"name": "토리든 다이브인 세럼", "category": "세럼", "users": 903},
    {"name": "라로슈포제 시카플라스트 밤", "category": "밤·연고", "users": 812},
    {"name": "센카 퍼펙트 휩 클렌징폼", "category": "클렌저", "users": 774},
    {"name": "이니스프리 노세범 미네랄 파우더", "category": "피지관리", "users": 689},
]

# 구매 내역 - 목업 데이터 (남성용 화장품 / 여러 쇼핑몰)
MOCK_PURCHASES = [
    {"site": "올리브영", "product": "라운드랩 자작나무 수분 크림", "date": "2026-06-28", "price": 19800},
    {"site": "무신사", "product": "그루밍 옴므 올인원 토너", "date": "2026-06-15", "price": 24000},
    {"site": "쿠팡", "product": "우르오스 스킨워시 클렌저", "date": "2026-06-10", "price": 12900},
    {"site": "네이버", "product": "닥터지 레드 블레미쉬 수딩 크림", "date": "2026-05-30", "price": 22500},
    {"site": "올리브영", "product": "토리든 다이브인 저분자 히알루론산 세럼", "date": "2026-05-21", "price": 18700},
    {"site": "쿠팡", "product": "니베아 맨 센서티브 아프터쉐이브 밤", "date": "2026-05-12", "price": 8900},
    {"site": "무신사", "product": "메디힐 남성 진정 마스크팩 10매", "date": "2026-04-30", "price": 13500},
    {"site": "네이버", "product": "라로슈포제 안뗄리오스 선크림", "date": "2026-04-18", "price": 21000},
]
# 쇼핑몰별 대표 색상
SITE_COLORS = {"올리브영": "#a3d977", "네이버": "#03c75a", "쿠팡": "#f7541f", "무신사": "#d9dde1"}

# 피부 우수자(고점수) 지역 분포 - 목업 데이터
MOCK_DISTRICTS = [
    {"area": "강남구", "count": 128},
    {"area": "서초구", "count": 112},
    {"area": "송파구", "count": 97},
    {"area": "마포구", "count": 84},
    {"area": "성동구", "count": 71},
    {"area": "용산구", "count": 63},
]

EVENT_LABELS = {
    "date": "소개팅",
    "interview": "면접",
    "wedding": "결혼식",
    "meeting": "상견례",
    "dating": "데이트",
}

OLIVEYOUNG_SEARCH = "https://www.oliveyoung.co.kr/store/search/getSearchMain.do?query="

# 네이버 쇼핑 최저가 검색 (가격 낮은순 정렬)
NAVER_SHOP_SEARCH = "https://search.shopping.naver.com/search/all?sort=price_asc&query="

# 쇼핑백 아이콘 (feather "shopping-bag") - 올리브영 바로가기 버튼용
SHOP_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/>'
    '<path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>'
)

# 가격표 아이콘 (feather "tag") - 최저가 검색 버튼용
PRICE_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 '
    '.586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/>'
    '<circle cx="7.5" cy="7.5" r="1.2" fill="currentColor" stroke="none"/></svg>'
)

# 피부 타입별 로컬 추천 제품 (API 키 없이 D-day 모드에서 사용)
LOCAL_PRODUCTS = {
    "지성": [("이니스프리 노세범 미네랄 파우더", "번들거림 잡아주는 피지 관리템"),
             ("센카 퍼펙트 휩 클렌징폼", "산뜻하게 유분 정리해주는 클렌저")],
    "건성": [("라운드랩 자작나무 수분 크림", "속건조 잡아주는 고보습 크림"),
             ("토리든 다이브인 세럼", "가볍게 수분 채워주는 히알루론산 세럼")],
    "민감성": [("라로슈포제 시카플라스트 밤 B5", "붉은기·자극 빠르게 진정"),
               ("아누아 어성초 77 토너", "순하게 진정시키는 저자극 토너")],
    "복합성": [("아누아 어성초 77 토너", "T존 유분·볼 건조 밸런스 잡기"),
               ("라운드랩 자작나무 수분 크림", "수분·유분 균형 맞추는 보습")],
}


def shop_button(product_name: str) -> str:
    """제품명으로 올리브영 검색 링크를 여는 아이콘 버튼 HTML을 반환."""
    link = OLIVEYOUNG_SEARCH + product_name.replace(" ", "+")
    return (f'<a class="cl-shop-btn" target="_blank" rel="noopener" href="{link}" '
            f'title="올리브영에서 보기" aria-label="올리브영에서 보기">{SHOP_ICON}</a>')


def price_button(product_name: str) -> str:
    """제품명으로 네이버 최저가(가격 낮은순) 검색을 여는 아이콘 버튼 HTML을 반환."""
    link = NAVER_SHOP_SEARCH + product_name.replace(" ", "+")
    return (f'<a class="cl-price-btn" target="_blank" rel="noopener" href="{link}" '
            f'title="최저가 검색" aria-label="최저가 검색">{PRICE_ICON}</a>')


def buy_buttons(product_name: str) -> str:
    """올리브영 + 최저가 아이콘 버튼을 한 그룹으로 묶어 반환."""
    return (f'<div class="cl-shop-group">{shop_button(product_name)}'
            f'{price_button(product_name)}</div>')


def nearby_points(lat: float, lon: float) -> "pd.DataFrame":
    """내 위치(민트) + 주변 사용자(회색) + 주변 피부과(빨강) 좌표 DataFrame."""
    users = [
        (0.0, 0.0), (0.0042, 0.0031), (-0.0035, 0.0044), (0.0051, -0.0039),
        (-0.0048, -0.0028), (0.0022, 0.0059), (-0.0061, 0.0018),
    ]
    rows = [
        {"lat": lat + dy, "lon": lon + dx,
         "color": "#43d3b0" if i == 0 else "#9aa4ad",
         "size": 130 if i == 0 else 70}
        for i, (dy, dx) in enumerate(users)
    ]
    # 주변 피부과 (빨강)
    clinics = [(0.0028, -0.0018), (-0.0026, 0.0022), (0.0016, 0.0041), (-0.0044, -0.0006)]
    rows += [
        {"lat": lat + dy, "lon": lon + dx, "color": "#ff5a6a", "size": 95}
        for dy, dx in clinics
    ]
    return pd.DataFrame(rows)


def recommend_products(diagnosis: dict) -> list[dict]:
    """진단 결과(피부 타입)에 맞춘 추천 제품 목록. 이벤트 대비 선크림 포함."""
    items = LOCAL_PRODUCTS.get(diagnosis.get("skin_type", ""), LOCAL_PRODUCTS["복합성"])
    picks = [{"name": n, "reason": why} for n, why in items]
    picks.append({"name": "라로슈포제 안뗄리오스 선크림",
                  "reason": "자외선 차단은 피부 관리의 기본이에요"})
    return picks[:3]


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


def mask_name(name: str) -> str:
    """랭킹 노출용 이름 마스킹 (홍길동 -> 홍O동)."""
    name = (name or "").strip()
    if len(name) <= 1:
        return name or "익명"
    if len(name) == 2:
        return name[0] + "O"
    return name[0] + "O" + name[-1]


def load_records() -> list:
    """누적된 진단 기록을 파일에서 읽는다."""
    try:
        with open(RECORDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_record(rec: dict) -> None:
    """진단 기록 1건을 파일에 누적 저장한다."""
    records = load_records()
    records.append(rec)
    try:
        with open(RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 피부 진단 (나이 입력 기반, 결과는 랜덤)
# ---------------------------------------------------------------------------
_SKIN_TYPES = ["건성", "지성", "복합성", "민감성"]
_INGREDIENT_POOL = [
    "히알루론산", "나이아신아마이드", "센텔라", "비타민C",
    "세라마이드", "판테놀", "어성초", "마데카소사이드",
]

# 유수분 선택값 -> 피부 타입 매핑
_MOISTURE_TO_TYPE = {
    "건성": "건성", "약간 건성": "건성", "보통": "복합성",
    "약간 지성": "지성", "지성": "지성",
}
# 고민 -> 추천 성분 매핑
_CONCERN_TO_INGREDIENT = {
    "홍조": "센텔라", "속건조": "세라마이드", "번들거림(유분)": "나이아신아마이드",
    "칙칙함": "비타민C", "트러블": "어성초", "모공": "나이아신아마이드",
    "다크서클": "카페인", "각질": "판테놀",
}


def random_diagnose(age: int, moisture: str = "보통", tone: str = "보통",
                    flush: bool = False, extra: list | None = None) -> dict:
    """나이 + 사용자가 체크한 피부 상태로 진단 결과 생성. 점수는 랜덤 (데모용)."""
    score = random.randint(60, 98)  # 최소 60점 이상 보장
    skin_type = _MOISTURE_TO_TYPE.get(moisture, random.choice(_SKIN_TYPES))

    # 사용자가 체크한 항목을 고민으로 반영
    concerns = list(extra or [])
    if flush:
        concerns.insert(0, "홍조")
    if moisture in ("건성", "약간 건성"):
        concerns.append("속건조")
    if moisture in ("지성", "약간 지성"):
        concerns.append("번들거림(유분)")
    if tone == "어두운 편":
        concerns.append("칙칙함")
    concerns = list(dict.fromkeys(concerns))[:4] or ["전반적으로 안정적"]

    # 고민 기반 추천 성분 (부족하면 랜덤으로 채움)
    ingredients = [_CONCERN_TO_INGREDIENT[c] for c in concerns if c in _CONCERN_TO_INGREDIENT]
    for ing in random.sample(_INGREDIENT_POOL, k=len(_INGREDIENT_POOL)):
        if len(ingredients) >= 3:
            break
        ingredients.append(ing)
    ingredients = list(dict.fromkeys(ingredients))[:3]

    summary = random.choice([
        f"{age}세 평균보다 관리가 잘 되고 있어요! 지금 루틴 유지 추천 👍",
        f"{age}세 기준 딱 평균 정도예요. 보습만 챙겨도 확 좋아질 거예요.",
        f"요즘 컨디션이 살짝 지쳐 보여요. 수분 채우기부터 시작해봐요 💧",
        f"기본기는 탄탄해요. 포인트 케어만 더하면 완벽해요!",
    ])
    return {
        "score": score,
        "gain": random.randint(2, 16),  # 28일 피부 턴오버 동안 상승한 점수
        "skin_type": skin_type,
        "concerns": concerns,
        "summary": summary,
        "recommended_ingredients": ingredients,
    }


def local_routine(event_label: str, days_left: int, diagnosis: dict) -> dict:
    """API 키가 없을 때 규칙 기반으로 D-day 케어 루틴을 생성."""
    steps = [
        "저자극 클렌저로 아침·저녁 세안하기",
        "토너로 수분 채우고 보습 크림 바르기",
        "자기 전 진정 세럼 한 방울 발라주기",
        "외출 시 선크림 꼭 챙겨 바르기",
        "물 자주 마시고 일찍 잠들기",
        "각질 정리 대신 수분팩으로 컨디션 올리기",
    ]
    routine = [
        {"day_label": f"D-{i}", "task": steps[(days_left - i) % len(steps)]}
        for i in range(min(days_left, 6), 0, -1)
    ]
    if not routine:
        routine = [{"day_label": "D-DAY", "task": "가벼운 세안 후 보습 크림으로 마무리하기"}]
    return {
        "routine": routine,
        "today_task": routine[0]["task"],
        "products": recommend_products(diagnosis),
    }


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
        "그리고 이 피부 타입/고민과 이벤트에 어울리는, 한국에서 실제로 쉽게 살 수 있는 "
        "구체적인 시판 화장품 2~3개를 브랜드명 포함 정확한 제품명으로 추천해줘. "
        "아래 JSON 형식으로만 응답해:\n"
        '{"routine": [{"day_label": "D-3", "task": "오늘 할 일 한 줄"}, ...], '
        '"today_task": "오늘(가장 가까운 날) 해야 할 일 한 줄", '
        '"products": [{"name": "브랜드+제품명", "reason": "한 줄 추천 이유"}]}'
    )
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=800,
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
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

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

html, body { background: #05070a; }
.stApp {
  background:
    radial-gradient(1100px 560px at 50% -12%, rgba(67, 211, 176, 0.13), transparent 60%),
    radial-gradient(900px 500px at 110% 8%, rgba(94, 234, 212, 0.06), transparent 55%),
    linear-gradient(180deg, #0b0e13 0%, #070a0e 100%);
  max-width: 430px;
  margin: 0 auto;
  height: 100vh;
  overflow-y: auto;
  overflow-x: hidden;
  border-radius: 44px;
  border: 3px solid var(--accent);
  box-shadow: 0 0 0 1px rgba(67, 211, 176, 0.35), 0 0 30px 6px rgba(67, 211, 176, 0.45);
}
.stApp, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp label {
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--text);
  word-break: keep-all;      /* 한글이 음절 단위로 끊기지 않고 어절 단위로 줄바꿈 */
  overflow-wrap: anywhere;   /* 그래도 넘칠 땐 안전하게 줄바꿈 */
}
/* 머티리얼 아이콘(비밀번호 표시 눈아이콘·확장 화살표 등)이 전역 폰트에 덮여
   'visibility' 같은 글자로 보이지 않도록 아이콘 폰트를 되돌려준다 */
.stApp span[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Rounded' !important;
  word-break: normal; overflow-wrap: normal;
}
.block-container { max-width: 560px; padding-top: 2.2rem; padding-bottom: 7rem; }
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
  letter-spacing: -1.4px; margin: 22px 0 14px; text-wrap: balance; }
.cl-grad { background: linear-gradient(115deg, var(--accent-2), var(--accent));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cl-hero__sub { text-align: center; color: var(--muted); font-size: 15px; line-height: 1.65;
  margin: 0 auto 24px; max-width: 420px; text-wrap: balance; }

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
.cl-rank__gain { color: var(--accent); }
/* 랭킹 탭 */
.stApp [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stApp [data-baseweb="tab"] { font-weight: 700; }
.stApp [data-baseweb="tab-highlight"] { background: var(--accent); }
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

/* ---- 하단 탭 내비게이션 ---- */
.st-key-bottomnav {
  /* 폰 프레임(430px, 가운데 정렬) 바깥 실제 뷰포트 기준 고정이라
     프레임 좌우 가장자리에 맞춰 너비를 맞춰준다 */
  position: fixed; left: max(0px, calc(50% - 215px)); right: max(0px, calc(50% - 215px)); bottom: 0; z-index: 9998;
  background: rgba(10, 14, 19, 0.96); backdrop-filter: blur(16px);
  border-top: 1px solid var(--glass-brd);
  padding: 8px 10px calc(8px + env(safe-area-inset-bottom, 0px));
}
.st-key-bottomnav [data-testid="stHorizontalBlock"] { max-width: 560px; margin: 0 auto; gap: 8px; }
.st-key-bottomnav .stButton > button {
  border: 0; background: transparent; color: var(--muted);
  font-weight: 700; font-size: 13px; border-radius: 12px; box-shadow: none;
}
.st-key-bottomnav .stButton > button:hover { color: var(--text); border: 0; }
.st-key-bottomnav .stButton > button[kind="primary"] {
  background: var(--accent-dim); color: var(--accent); box-shadow: none;
}

/* ---- 플로팅 채팅봇 (우측 하단) ---- */
.st-key-chatwidget {
  /* 폰 프레임(430px, 가운데 정렬) 오른쪽 안쪽에 맞춰 가운데쪽으로,
     좁은 화면에선 화면 끝 16px 로 자동 조정 */
  position: fixed; right: max(16px, calc(50% - 207px)); bottom: 88px; z-index: 9999;
  width: auto; max-width: calc(100vw - 28px);
}
/* 열린 상태의 채팅 카드 - 불투명 배경으로 본문과 겹쳐 글자가 비치는 문제 방지 */
.st-key-chatcard {
  width: min(340px, calc(100vw - 28px)); margin: 0 0 12px auto;
  background: #0e141b; border: 1px solid var(--glass-brd);
  border-radius: 22px; overflow: hidden; box-shadow: 0 24px 70px rgba(0, 0, 0, 0.62);
}
.st-key-chatcard [data-testid="stVerticalBlock"] { gap: 8px; }
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
/* 채팅 패널(헤더+말풍선) - 배경/테두리는 바깥 카드(.st-key-chatcard)가 담당 */
.cl-chat-panel { background: transparent; }
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
/* 빠른 질문 추천 칩 */
.st-key-chat_chips { padding: 2px 14px 0; }
.st-key-chat_chips [data-testid="column"] { padding: 0 3px; }
.st-key-chat_chips .stButton > button {
  border-radius: 999px; font-size: 11.5px; font-weight: 600; padding: 7px 8px;
  min-height: 0; background: var(--glass); border: 1px solid var(--glass-brd);
  color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.st-key-chat_chips .stButton > button:hover { border-color: var(--accent); color: var(--accent); }

/* ---- 안내 배너 (API 키 미설정 등) ---- */
.cl-note { background: var(--accent-dim); border: 1px solid rgba(67,211,176,0.4);
  border-radius: 14px; padding: 13px 15px; font-size: 13px; line-height: 1.55;
  color: var(--text); margin: 6px 0 16px; }
.cl-note b { color: var(--accent); }
.cl-note code { background: rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 6px;
  font-size: 12px; color: var(--accent); }

/* ---- Face ID 인식 상태 ---- */
.cl-faceid { display: flex; align-items: center; gap: 10px; border-radius: 14px;
  padding: 12px 15px; font-size: 13.5px; font-weight: 600; margin: 10px 0; }
.cl-faceid--ok { background: var(--accent-dim); border: 1px solid rgba(67,211,176,0.5); color: var(--accent); }
.cl-faceid--warn { background: rgba(255,180,90,0.12); border: 1px solid rgba(255,180,90,0.45); color: #ffc784; }
.cl-faceid__dot { width: 9px; height: 9px; border-radius: 50%; background: currentColor;
  box-shadow: 0 0 12px currentColor; flex-shrink: 0; animation: cl-pulse 1.4s ease-in-out infinite; }
@keyframes cl-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ---- 많이 쓰는 화장품 랭킹 ---- */
.cl-prank { display: flex; align-items: center; gap: 14px; background: var(--glass);
  border: 1px solid var(--glass-brd); border-radius: 16px; padding: 13px 16px; margin-bottom: 10px; }
.cl-prank__body { flex: 1; min-width: 0; }
.cl-prank__top { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
.cl-prank__name { flex: 1; min-width: 0; font-size: 14px; font-weight: 700;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cl-prank__cat { flex-shrink: 0; font-size: 10.5px; color: var(--accent); background: var(--accent-dim);
  padding: 2px 8px; border-radius: 999px; }
.cl-prank__bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
.cl-prank__bar span { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-2), var(--accent)); }
.cl-prank__meta { font-size: 11.5px; color: var(--muted); margin-top: 6px; }

/* ---- 구매 아이콘 버튼 (올리브영 / 최저가) ---- */
.cl-shop-group { display: inline-flex; align-items: center; gap: 8px; flex-shrink: 0; }
.cl-shop-btn, .cl-price-btn { display: inline-flex; align-items: center; justify-content: center;
  width: 38px; height: 38px; border-radius: 12px; flex-shrink: 0; text-decoration: none;
  transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease, border-color 0.2s ease; }
.cl-shop-btn svg, .cl-price-btn svg { width: 17px; height: 17px; }
/* 올리브영 = 민트 */
.cl-shop-btn { background: var(--accent-dim); border: 1px solid rgba(67,211,176,0.35); color: var(--accent); }
.cl-shop-btn:hover { transform: translateY(-1px); border-color: transparent; color: var(--ink);
  background: linear-gradient(115deg, var(--accent-2), var(--accent)); }
/* 최저가 = 앰버 */
.cl-price-btn { background: rgba(244,193,94,0.14); border: 1px solid rgba(244,193,94,0.4); color: #f4c15e; }
.cl-price-btn:hover { transform: translateY(-1px); border-color: transparent; color: #2a1e05;
  background: linear-gradient(115deg, #ffd98a, #f4c15e); }

/* ---- 상단 브랜드 바 + 로그인 ---- */
.cl-topbrand { font-size: 18px; font-weight: 800; letter-spacing: -0.4px; line-height: 1.1; }
.cl-topbrand span { display: block; font-family: 'Space Grotesk', monospace; font-size: 8.5px;
  letter-spacing: 1.5px; color: var(--muted); font-weight: 600; margin-top: 3px; }
/* 로고 = 홈 이동 (실제 버튼이라 새로고침 없이 이동 -> 로그인 세션이 유지된다)
   버튼(및 조상 wrapper)을 position:absolute로 빼면 streamlit이 마운트 시
   컨테이너의 실제 콘텐츠 높이를 0으로 측정해버려 레이아웃이 깨지므로,
   일반 흐름(normal flow) 그대로 두고 버튼 자체 크기만 !important로 강제한다. */
.st-key-topbar [data-testid="stHorizontalBlock"] { align-items: center; }
.st-key-logohome { line-height: 0; }
.st-key-logohome .stButton button {
  width: 56px !important; height: 56px !important; padding: 0 !important; border: 0 !important; box-shadow: none !important;
  background-color: transparent !important; background-repeat: no-repeat !important; background-position: left center !important;
  background-size: contain !important; color: transparent !important; font-size: 0 !important;
  filter: drop-shadow(0 6px 18px rgba(67, 211, 176, 0.28)); transition: transform 0.2s ease;
}
.st-key-logohome .stButton button:hover {
  background-color: transparent !important; border: 0 !important; box-shadow: none !important; transform: scale(1.05);
}
.st-key-logohome .stButton button * { color: transparent !important; font-size: 0 !important; }

/* 쇼핑몰 배지 (구매내역) */
.cl-site { display: inline-block; font-size: 10.5px; font-weight: 700; padding: 2px 8px;
  border-radius: 999px; border: 1px solid; }
.st-key-logout .stButton > button { font-size: 12px; font-weight: 600; padding: 8px 6px;
  color: var(--muted); }
.st-key-logout .stButton > button:hover { color: var(--accent); border-color: var(--accent); }
.st-key-loginbox { max-width: 360px; margin: 6px auto 0; }

/* 입력창(로그인 / 나이 등) 공통 스타일 */
.stTextInput input, .stNumberInput input {
  background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 12px; color: var(--text);
}
.stTextInput input:focus, .stNumberInput input:focus { border-color: var(--accent); box-shadow: none; }

/* 카메라 프리뷰를 거울(좌우반전) 모드로 - 셀피처럼 자연스럽게 */
.stApp [data-testid="stCameraInput"] video { transform: scaleX(-1); }

/* D-day 케어 모드(챗봇 안 확장 패널) */
.st-key-chat_dday { padding: 0 12px; }
.st-key-chat_dday [data-testid="stExpander"] {
  border: 1px solid var(--glass-brd); border-radius: 12px; background: rgba(255,255,255,0.03); }

/* ---- D-day 추천 제품 ---- */
.cl-rec { display: flex; align-items: center; gap: 12px; background: var(--glass);
  border: 1px solid var(--glass-brd); border-radius: 14px; padding: 12px 14px; margin-bottom: 9px; }
.cl-rec__body { flex: 1; min-width: 0; }
.cl-rec__name { font-size: 13.5px; font-weight: 700; }
.cl-rec__reason { font-size: 11.5px; color: var(--muted); margin-top: 3px; line-height: 1.45; }

/* ---- 모바일 대응 ---- */
@media (max-width: 480px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem; }
  .cl-logo { width: 120px; height: 120px; }
  .cl-hero__title { font-size: 30px; letter-spacing: -1px; margin: 18px 0 12px; }
  .cl-hero__sub { font-size: 14px; }
  .cl-result__score { font-size: 52px; }
  .cl-countdown__dday { font-size: 42px; }
  [class*="st-key-navbtn_"] .stButton > button p:nth-of-type(2) { font-size: 19px; }
  .st-key-chatwidget { right: 14px; bottom: 82px; }
  .st-key-chat_fab .stButton > button { width: 52px; height: 52px; font-size: 22px; }
  .cl-chat-body { max-height: 42vh; }
  .cl-faceid { font-size: 12.5px; padding: 11px 13px; }
  .cl-prank { padding: 12px 13px; gap: 10px; }
  .cl-prank__name { font-size: 13px; }
  .cl-note { font-size: 12.5px; }
  /* 좁은 화면에선 구매 아이콘 버튼을 조금 작게·간격 좁게 해서 이름이 잘리지 않도록 */
  .cl-shop-group { gap: 6px; }
  .cl-shop-btn, .cl-price-btn { width: 34px; height: 34px; border-radius: 10px; }
  .cl-shop-btn svg, .cl-price-btn svg { width: 15px; height: 15px; }
  .cl-rank { gap: 9px; padding: 12px 12px; }
  .st-key-chatcard { width: min(340px, calc(100vw - 24px)); }
}
</style>
"""


# ---------------------------------------------------------------------------
# 화면 렌더링
# ---------------------------------------------------------------------------
def _logout() -> None:
    for k in ("logged_in", "consent", "pending_login", "my_record_id"):
        st.session_state.pop(k, None)


@st.dialog("개인정보 활용 동의")
def consent_dialog() -> None:
    """로그인 시 개인정보 활용 동의 팝업 (동의/비동의)."""
    st.markdown(
        "더 정확한 추천을 위해 아래 개인정보를 활용해요:\n\n"
        "- 이름·나이 등 프로필 정보\n"
        "- 피부 진단 결과 및 피부 정보\n"
        "- 구매 내역"
    )
    st.caption("동의하셔야 서비스를 이용할 수 있어요.")
    c1, c2 = st.columns(2)
    if c1.button("동의", type="primary", use_container_width=True):
        st.session_state.logged_in = True
        st.session_state.consent = True
        st.session_state.pop("pending_login", None)
        st.rerun()
    if c2.button("비동의", use_container_width=True):
        st.session_state.consent = False
        st.session_state.pop("pending_login", None)
        st.rerun()


def render_login() -> None:
    """로그인 화면 - 이름 입력 + 로그인/회원가입/비회원. 진입 시 개인정보 동의 팝업."""
    uri = logo_data_uri()
    if uri:
        st.markdown(
            f'<div class="cl-logo-wrap"><img class="cl-logo" src="{uri}" alt="clozkin"></div>'
            '<p class="cl-badge-tag">SKINCARE, MANDATORY FOR MEN</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cl-brand"><span class="cl-brand__dot"></span>'
            '<span class="cl-brand__name">clozkin</span></div>'
            '<p class="cl-badge-tag">SKINCARE, MANDATORY FOR MEN</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<h1 class="cl-hero__title">남자의 피부,<br>'
        '<span class="cl-grad">이제 관리는 필수.</span></h1>',
        unsafe_allow_html=True,
    )
    with st.container(key="loginbox"):
        with st.form(key="login_form"):
            user_id = st.text_input("아이디", placeholder="아이디", key="login_id")
            st.text_input("비밀번호", type="password",
                          placeholder="비밀번호", key="login_pw")
            c1, c2 = st.columns(2)
            login = c1.form_submit_button("로그인", type="primary",
                                          use_container_width=True)
            signup = c2.form_submit_button("회원가입", use_container_width=True)
        guest = st.button("비회원으로 시작하기", key="guest_btn",
                          use_container_width=True)

        if st.session_state.get("consent") is False:
            st.warning("개인정보 활용에 동의해야 서비스를 이용할 수 있어요.")

    if login:
        # 기존 회원 로그인은 이미 동의한 것으로 간주하고 팝업 없이 바로 입장
        st.session_state.user_name = (user_id or "").strip() or "게스트"
        st.session_state.logged_in = True
        st.session_state.consent = True
        st.session_state.pop("pending_login", None)
        st.rerun()
    elif signup or guest:
        st.session_state.user_name = (user_id or "").strip() or "게스트"
        st.session_state.pending_login = True
        st.session_state.pop("consent", None)

    # 회원가입/비회원 선택 시에만 개인정보 동의 팝업 표시 (로그인은 팝업 없이 바로 입장)
    if st.session_state.get("pending_login"):
        consent_dialog()


def section_title(title: str, tag: str) -> None:
    st.markdown(f'<div class="cl-sec">{tag}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cl-h">{title}</div>', unsafe_allow_html=True)


def render_header() -> None:
    """상단 브랜드 바(로고=홈 이동, 페이지 새로고침 없이 st.button으로 처리) + 로그아웃 (모든 화면 공통)."""
    top_bar = st.container(key="topbar")
    top_l, top_r = top_bar.columns([3, 1])
    with top_l:
        with st.container(key="logohome"):
            uri = logo_data_uri()
            if uri:
                st.markdown(
                    f"<style>.st-key-logohome .stButton button "
                    f"{{background-image: url('{uri}') !important;}}</style>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<style>.st-key-logohome .stButton button "
                    "{color: var(--text) !important; font-size: 19px !important; "
                    "font-weight: 800;}</style>",
                    unsafe_allow_html=True,
                )
            st.button("clozkin 홈", key="btn_logo_home", on_click=set_nav, args=("home",))
    with top_r:
        with st.container(key="logout"):
            st.button("로그아웃", key="btn_logout", on_click=_logout,
                      use_container_width=True)


def render_age_diagnosis() -> None:
    """나이 + 피부 상태 체크 후 촬영하면 (랜덤) 점수를 내고 랭킹에 반영한다."""
    st.markdown('<div class="cl-sec">DIAGNOSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">AI 피부 진단</div>', unsafe_allow_html=True)
    st.caption("나이와 피부 상태를 체크하고 촬영하면 내 피부 점수를 알려드려요.")

    age = st.number_input("나이", min_value=10, max_value=90, value=28, step=1)
    moisture = st.radio("피부 유수분", ["건성", "지성"], horizontal=True)
    tone = st.radio("피부 밝기", ["어두운 편", "밝은 편"], horizontal=True)
    flush = st.radio("홍조(붉은기)", ["없음", "있음"], horizontal=True) == "있음"
    extra = st.multiselect(
        "그 외 신경 쓰이는 부분 (복수 선택)",
        ["트러블", "모공", "칙칙함", "다크서클", "각질", "속건조"])

    run = st.button("피부 진단받기", type="primary", use_container_width=True)
    if run:
        st.session_state.show_camera = True
        st.session_state.pop("last_diagnosis", None)  # 새 진단 시작 - 이전 결과 초기화

    # 진단 버튼을 누르면 카메라가 뜨고, 촬영하면 체크값 + 랜덤 점수로 결과를 낸다.
    if st.session_state.get("show_camera"):
        st.caption("📸 얼굴이 잘 보이도록 정면에서 촬영해주세요.")
        shot = st.camera_input("피부 촬영", label_visibility="collapsed")
        if shot is not None:
            with st.spinner("AI가 피부를 분석하는 중이에요..."):
                time.sleep(3)
            result = random_diagnose(int(age), moisture, tone, flush, extra)
            st.session_state.last_diagnosis = result
            # 진단 기록을 사이트 전체에 누적 저장 (랭킹에 반영)
            rec_id = time.time_ns()
            save_record({
                "id": rec_id,
                "name": mask_name(st.session_state.get("user_name", "")),
                "score": result["score"],
                "gain": result["gain"],
                "product": (result.get("recommended_ingredients") or ["-"])[0],
            })
            st.session_state.my_record_id = rec_id
            st.session_state.show_camera = False
            st.rerun()

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

        # 내 피부에 맞는 추천 제품 + 구매 링크
        st.markdown('<div class="cl-sec">RECOMMENDED</div>', unsafe_allow_html=True)
        st.markdown('<div class="cl-h">내 피부 맞춤 추천 제품</div>', unsafe_allow_html=True)
        for p in recommend_products(result):
            st.markdown(
                f'<div class="cl-rec"><div class="cl-rec__body">'
                f'<div class="cl-rec__name">{p.get("name", "")}</div>'
                f'<div class="cl-rec__reason">{p.get("reason", "")}</div></div>'
                f'{buy_buttons(p.get("name", ""))}</div>',
                unsafe_allow_html=True,
            )


def render_nearby_map() -> None:
    """GPS 위치를 받아 내 주변 사용자 분포를 간단한 지도로 보여준다."""
    st.markdown('<div class="cl-sec">NEARBY</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">내 주변 피부 랭킹 지도</div>', unsafe_allow_html=True)

    lat = lon = None
    if _HAS_GEO:
        st.caption("아래 위치 버튼을 누르면 내 주변 사용자 분포를 지도로 볼 수 있어요.")
        loc = streamlit_geolocation()
        if isinstance(loc, dict):
            lat, lon = loc.get("latitude"), loc.get("longitude")
    else:
        st.caption("위치 컴포넌트를 불러오지 못해 기본 위치(서울 강남)로 표시해요.")

    legend = "🟢 나 · ⚪ 주변 사용자 · 🔴 주변 피부과"
    if lat and lon:
        st.map(nearby_points(lat, lon), color="color", size="size", zoom=13)
        st.caption(f"📍 현재 위치 기준 · {legend} (위도 {lat:.4f}, 경도 {lon:.4f})")
    else:
        # 위치 미허용 시 기본 위치(서울 강남역)로 예시 지도
        st.map(nearby_points(37.4979, 127.0276), color="color", size="size", zoom=13)
        st.caption(f"{legend} · 위치 권한을 허용하면 실제 내 주변으로 바뀌어요. (지금은 예시 위치)")


def _person_row(rank: int, entry: dict, value_html: str) -> None:
    """랭킹 한 줄 렌더. value_html 자리에 점수 또는 상승폭을 넣는다."""
    st.markdown(
        f'<div class="cl-rank {"is-me" if entry.get("is_me") else ""}">'
        f'<div class="cl-rank__num">{rank}</div>'
        f'<div class="cl-rank__body">'
        f'<div class="cl-rank__name">{entry["name"]}</div>'
        f'<div class="cl-rank__product">{entry["product"]}</div></div>'
        f'{value_html}'
        f'{buy_buttons(entry["product"])}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_ranking() -> None:
    render_header()
    section_title("우리 동네 피부 랭킹", "RANKING")
    render_nearby_map()

    # 목업 + 누적된 실제 진단 기록을 합쳐 랭킹 구성
    records = load_records()
    my_id = st.session_state.get("my_record_id")
    board = [dict(x) for x in MOCK_RANKING]
    for r in records:
        board.append({
            "name": r.get("name", "익명"),
            "score": int(r.get("score", 0)),
            "gain": int(r.get("gain", 0)),
            "product": r.get("product", "-"),
            "is_me": r.get("id") == my_id,
        })

    st.caption(f"지금까지 {len(MOCK_RANKING) + len(records):,}명이 진단에 참여했어요. "
               "(진단할수록 기록이 쌓여요)")

    tab_score, tab_gain = st.tabs(["🏆 피부 점수 순위", "📈 28일 상승 순위"])

    with tab_score:
        st.caption("현재 피부 점수가 높은 순이에요.")
        for rank, entry in enumerate(
                sorted(board, key=lambda x: x["score"], reverse=True), start=1):
            _person_row(rank, entry,
                        f'<div class="cl-rank__score">{entry["score"]}</div>')

    with tab_gain:
        st.caption("피부 턴오버 28일 동안 점수가 많이 오른 순이에요. (▲ 상승폭)")
        for rank, entry in enumerate(
                sorted(board, key=lambda x: x.get("gain", 0), reverse=True), start=1):
            _person_row(rank, entry,
                        f'<div class="cl-rank__score cl-rank__gain">▲{entry.get("gain", 0)}</div>')

    # --- 피부 우수자 지역 분포 ---
    st.markdown('<div class="cl-sec">DISTRIBUTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">피부 좋은 남자들, 어디 많을까?</div>', unsafe_allow_html=True)
    st.caption("피부 점수 상위 사용자들의 지역 분포예요.")

    districts = sorted(MOCK_DISTRICTS, key=lambda x: x["count"], reverse=True)
    top_count = districts[0]["count"]
    for rank, d in enumerate(districts, start=1):
        pct = round(d["count"] / top_count * 100)
        st.markdown(
            f'<div class="cl-prank">'
            f'<div class="cl-rank__num">{rank}</div>'
            f'<div class="cl-prank__body">'
            f'<div class="cl-prank__top"><span class="cl-prank__name">{d["area"]}</span></div>'
            f'<div class="cl-prank__bar"><span style="width:{pct}%"></span></div>'
            f'<div class="cl-prank__meta">{d["count"]:,}명</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- 많이 쓰는 화장품 랭킹 ---
    st.markdown('<div class="cl-sec">MOST USED</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">많이 쓰는 화장품 랭킹</div>', unsafe_allow_html=True)
    st.caption("우리 동네 남자들이 지금 가장 많이 쓰는 아이템이에요.")

    products = sorted(MOCK_PRODUCT_RANKING, key=lambda x: x["users"], reverse=True)
    top_users = products[0]["users"]
    for rank, p in enumerate(products, start=1):
        pct = round(p["users"] / top_users * 100)
        st.markdown(
            f'<div class="cl-prank">'
            f'<div class="cl-rank__num">{rank}</div>'
            f'<div class="cl-prank__body">'
            f'<div class="cl-prank__top"><span class="cl-prank__name">{p["name"]}</span>'
            f'<span class="cl-prank__cat">{p["category"]}</span></div>'
            f'<div class="cl-prank__bar"><span style="width:{pct}%"></span></div>'
            f'<div class="cl-prank__meta">{p["users"]:,}명 사용</div></div>'
            f'{buy_buttons(p["name"])}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_diagnosis_screen() -> None:
    render_header()
    render_age_diagnosis()

    st.markdown('<div class="cl-sec">MOVE</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.button("🏠 메인 화면으로", key="diag_home", on_click=set_nav, args=("home",),
              use_container_width=True)
    c2.button("🏆 랭킹 보러가기", key="diag_rank", on_click=set_nav, args=("ranking",),
              use_container_width=True)


def render_purchases_screen() -> None:
    render_header()
    section_title("구매 내역", "PURCHASES")

    if not st.session_state.get("purchases_synced"):
        st.markdown(
            '<div class="cl-note">🔗 올리브영·네이버·쿠팡·무신사에서 구매한 '
            '남성 화장품 내역을 <b>한 번에</b> 불러올 수 있어요.</div>',
            unsafe_allow_html=True,
        )
        if st.button("한번에 연동하기", type="primary", use_container_width=True):
            with st.spinner("여러 쇼핑몰 계정을 연동하는 중..."):
                time.sleep(2)
            st.session_state.purchases_synced = True
            st.rerun()
        st.caption("데모 버전이에요. 실제 계정 연동 없이 예시 내역을 보여줍니다.")
        return

    total = sum(p["price"] for p in MOCK_PURCHASES)
    st.caption(f"최근 {len(MOCK_PURCHASES)}건 · 총 {total:,}원 · 4개 쇼핑몰 연동됨 ✓")
    for p in MOCK_PURCHASES:
        color = SITE_COLORS.get(p["site"], "#43d3b0")
        st.markdown(
            f'<div class="cl-rec"><div class="cl-rec__body">'
            f'<div class="cl-rec__name">{p["product"]}</div>'
            f'<div class="cl-rec__reason">'
            f'<span class="cl-site" style="color:{color};border-color:{color}">{p["site"]}</span>'
            f' · {p["date"]} · {p["price"]:,}원</div></div>'
            f'{buy_buttons(p["product"])}</div>',
            unsafe_allow_html=True,
        )
    st.button("연동 해제", key="unsync", on_click=lambda: st.session_state.update(
        purchases_synced=False), use_container_width=True)


def render_home_screen() -> None:
    render_header()
    uri = logo_data_uri()
    if uri:
        st.markdown(
            f'<div class="cl-logo-wrap"><img class="cl-logo" src="{uri}" alt="clozkin"></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<h1 class="cl-hero__title">세안 다음은,<br>'
        '<span class="cl-grad">당연히 스킨케어.</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="cl-hero__sub">토너·세럼 순서 몰라도 괜찮아요.<br>'
        '내 피부를 진단하고, 우리 동네 랭킹에서 딱 맞는 아이템을 찾아보세요.</p>',
        unsafe_allow_html=True,
    )

    diagnosis = st.session_state.get("last_diagnosis")
    if diagnosis and diagnosis.get("summary"):
        st.markdown(
            f'<div class="cl-status-wrap"><div class="cl-status">'
            f'<b>최근 진단 {diagnosis.get("score", "-")}점</b> · {diagnosis["summary"]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.button("피부 진단 하러가기", type="primary", use_container_width=True,
              on_click=set_nav, args=("diagnose",))
    st.button("우리 동네 랭킹 보기", use_container_width=True,
              on_click=set_nav, args=("ranking",))


def build_dday_message(client: anthropic.Anthropic | None,
                       event_label: str, days_left: int) -> str:
    """D-day 케어 루틴을 만들어 챗봇 말풍선용 텍스트로 반환 (챗봇 기능)."""
    diagnosis = st.session_state.get("last_diagnosis") or {
        "skin_type": "정보 없음",
        "concerns": ["일반 컨디션 관리"],
        "summary": "아직 피부 진단을 하지 않았어요.",
    }
    try:
        if client is not None:
            result = generate_routine(client, event_label, days_left, diagnosis)
        else:
            result = local_routine(event_label, days_left, diagnosis)
    except Exception:  # noqa: BLE001 - 실패 시 규칙 기반 루틴으로 폴백
        result = local_routine(event_label, days_left, diagnosis)
    if not result.get("products"):
        result["products"] = recommend_products(diagnosis)

    lines = [f"📅 {event_label} D-{days_left} 케어 루틴이에요!"]
    if result.get("today_task"):
        lines.append(f"✅ 오늘 할 일: {result['today_task']}")
    for item in result.get("routine", []):
        lines.append(f"• {item.get('day_label', '')} {item.get('task', '')}")
    products = result.get("products", [])
    if products:
        lines.append("\n🛍️ 추천 제품")
        for p in products:
            lines.append(f"• {p.get('name', '')} — {p.get('reason', '')}")
        lines.append("(제품은 랭킹 화면에서 올리브영·최저가로 바로 살 수 있어요)")
    return "\n".join(lines)


def set_nav(screen: str) -> None:
    st.session_state.nav = screen


def render_bottom_nav(active: str) -> None:
    """하단 고정 탭 내비게이션 (홈 / 랭킹 / 진단)."""
    items = [("home", "🏠", "홈"), ("ranking", "🏆", "랭킹"),
             ("diagnose", "✨", "진단"), ("purchases", "🛍", "구매")]
    with st.container(key="bottomnav"):
        cols = st.columns(len(items))
        for col, (key, icon, label) in zip(cols, items):
            col.button(f"{icon} {label}", key=f"nav_{key}", on_click=set_nav,
                       args=(key,), use_container_width=True,
                       type="primary" if active == key else "secondary")


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
            # 패널·칩·입력을 하나의 불투명 카드로 묶어 본문과 겹쳐 보이지 않게 한다.
            with st.container(key="chatcard"):
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

                # --- D-day 케어 모드 (챗봇 기능) ---
                with st.container(key="chat_dday"):
                    with st.expander("📅 D-day 케어 모드"):
                        with st.form(key="dday_form"):
                            ev = st.radio(
                                "이벤트", list(EVENT_LABELS.keys()),
                                format_func=lambda k: EVENT_LABELS[k], horizontal=True,
                                label_visibility="collapsed",
                            )
                            days = st.number_input("며칠 뒤인가요?", min_value=0,
                                                   max_value=60, value=7, step=1)
                            dday_submit = st.form_submit_button(
                                "루틴 만들기", use_container_width=True)
                    if dday_submit:
                        with st.spinner("맞춤 케어 루틴을 짜는 중..."):
                            msg = build_dday_message(
                                client, EVENT_LABELS.get(ev, ev), int(days))
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": msg})
                        st.rerun()

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

    api_key = get_api_key()
    client = get_client(api_key) if api_key else None

    # 가짜 로그인 게이트 - 로그인 전에는 로그인 화면만 보여준다.
    if not st.session_state.get("logged_in"):
        render_login()
        return

    # 로고 이미지 클릭 등 URL(?nav=) 로 들어온 화면 전환 반영
    if "nav" in st.query_params:
        st.session_state.nav = st.query_params["nav"]
        del st.query_params["nav"]

    # 하단 탭으로 화면 전환 (홈 / 랭킹 / 진단 / 구매)
    nav = st.session_state.get("nav", "home")
    if nav == "ranking":
        render_ranking()
    elif nav == "diagnose":
        render_diagnosis_screen()
    elif nav == "purchases":
        render_purchases_screen()
    else:
        render_home_screen()

    render_bottom_nav(nav)
    render_chat_widget(client)


if __name__ == "__main__":
    main()
