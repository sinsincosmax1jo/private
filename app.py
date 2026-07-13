"""
clozkin - 뷰티 입문 남성을 위한 AI 스킨케어 가이드 MVP (Streamlit 버전)

Streamlit Cloud 배포용. 기존 Flask 단일 파일 버전을 Streamlit으로 포팅했다.
발표용 프로토타입이므로 실제 이미지 분석 모델이나 DB 학습은 쓰지 않고, 설문/사진 제출 후
"AI가 분석한 것처럼" 보이는 시뮬레이션 흐름으로 결과를 만든다 (자세한 로직은
simulate_ai_diagnosis 참고).
  - 피부 진단: 사진 촬영/업로드 + 설문 7문항 -> 로딩 화면 -> 설문 70% + 이미지 30%로
    합성한 카테고리별/최종 점수 (전부 랜덤이지만 응답에 따라 그럴듯한 범위 안에서 생성됨)
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
import datetime
from io import BytesIO
from collections import Counter

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import anthropic
from PIL import Image

MODEL_NAME = "claude-sonnet-5"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clozkin_logo.png")
# 진단 기록을 누적 저장하는 파일 (사이트 전체 랭킹에 반영)
RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin_records.json")

# ---------------------------------------------------------------------------
# 우리 동네 피부 랭킹 - 목업 데이터 (실제 서비스에서는 DB에서 조회)
# ---------------------------------------------------------------------------
MOCK_RANKING = [
    {"name": "서초동 피부왕자", "age_group": "20대", "skin_type": "복합성", "score": 91, "gain": 7, "product": "토리든 다이브인 저분자 히알루론산 세럼"},
    {"name": "상도동 도자기피부", "age_group": "30대", "skin_type": "수부지", "score": 87, "gain": 12, "product": "라로슈포제 에빠끌라 토너"},
    {"name": "판교 물광남", "age_group": "20대", "skin_type": "건성", "score": 83, "gain": -4, "product": "라운드랩 자작나무 수분 크림"},
    {"name": "해운대 꿀피부", "age_group": "30대", "skin_type": "지성", "score": 79, "gain": 15, "product": "파티온 노스카나인 트러블 세럼"},
    {"name": "연남동 무결점", "age_group": "40대", "skin_type": "민감성", "score": 74, "gain": -2, "product": "에스트라 에이시카365 수분 진정 크림"},
    {"name": "성수동 광채남", "age_group": "10대", "skin_type": "지성", "score": 68, "gain": 18, "product": "닥터자르트 시카페어 토너"},
    {"name": "잠실 트러블졸업", "age_group": "20대", "skin_type": "복합성", "score": 63, "gain": -7, "product": "라로슈포제 시카플라스트 밤 B5"},
]

# ---------------------------------------------------------------------------
# 리워드 포인트 / 티어 - 목업 데이터 (실제 서비스에서는 DB에서 적립 내역을 조회)
#
# 포인트/티어는 현재 DB 없이 st.session_state["reward_points"]에만 보관된다
# (진단을 완료할 때마다 points_for_diagnosis()만큼 적립). 나중에 실제 계정
# 시스템을 붙일 때는 이 값을 유저 레코드의 필드 하나로 옮기기만 하면 된다.
# 티어 기준(min_points)만 바꾸면 전체 로직이 그대로 따라가도록 상수로 분리했다.
# ---------------------------------------------------------------------------
REWARD_TIERS = [
    {"key": "bronze", "name": "브론즈", "icon": "🥉", "min_points": 0},
    {"key": "silver", "name": "실버", "icon": "🥈", "min_points": 300},
    {"key": "gold", "name": "골드", "icon": "🥇", "min_points": 700},
    {"key": "diamond", "name": "다이아몬드", "icon": "💎", "min_points": 1500},
    {"key": "master", "name": "마스터", "icon": "👑", "min_points": 3000},
]

# 명예의 전당(마스터 티어) 목업 유저 - 랭킹 목업과 같은 동네 캐릭터를 재사용해
# 점수/개선폭이 높은 사람일수록 리워드 포인트도 그럴듯하게 높게 잡았다.
MOCK_REWARD_USERS = [
    {"name": "서초동 피부왕자", "points": 4820},
    {"name": "상도동 도자기피부", "points": 3960},
    {"name": "판교 물광남", "points": 3120},
    {"name": "해운대 꿀피부", "points": 2380},
    {"name": "연남동 무결점", "points": 1050},
    {"name": "성수동 광채남", "points": 520},
    {"name": "잠실 트러블졸업", "points": 150},
]


def points_for_diagnosis(result: dict) -> int:
    """진단 1회 완료 시 적립할 리워드 포인트. 참여 기본점수 + 피부점수 + 개선폭 보너스."""
    base = 80
    score_bonus = int(result.get("score", 0))
    gain_bonus = int(result.get("gain", 0)) * 4
    return base + score_bonus + gain_bonus


def tier_for_points(points: int) -> dict:
    """현재 포인트가 속한 티어 (REWARD_TIERS는 오름차순 정렬돼 있다고 가정)."""
    current = REWARD_TIERS[0]
    for t in REWARD_TIERS:
        if points >= t["min_points"]:
            current = t
        else:
            break
    return current


def tier_progress_pct(points: int, tier: dict, next_tier: dict | None) -> int:
    """티어 한 칸의 진행률(0~100). 아직 도달 못했으면 0, 이미 넘었으면 100."""
    lo = tier["min_points"]
    if points < lo:
        return 0
    if next_tier is None:
        return 100
    hi = next_tier["min_points"]
    return 100 if points >= hi else round((points - lo) / (hi - lo) * 100)

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

# 개선 상승폭이 큰 사람들이 많이 쓰는 제품 랭킹 - 목업 데이터
# (avg_gain = 이 제품을 쓴 급상승 유저들의 평균 점수 상승폭)
MOCK_IMPROVER_PRODUCTS = [
    {"name": "파티온 노스카나인 트러블 세럼", "category": "세럼", "avg_gain": 16},
    {"name": "라로슈포제 시카플라스트 밤 B5", "category": "밤·연고", "avg_gain": 15},
    {"name": "에스트라 에이시카365 수분 진정 크림", "category": "진정크림", "avg_gain": 13},
    {"name": "닥터자르트 시카페어 토너", "category": "토너", "avg_gain": 12},
    {"name": "라로슈포제 에빠끌라 토너", "category": "토너", "avg_gain": 11},
    {"name": "토리든 다이브인 저분자 히알루론산 세럼", "category": "세럼", "avg_gain": 9},
]

# 피부 타입별 인기 제품 랭킹 - 목업 (랭킹 탭 '타입별 인기')
MOCK_TYPE_PRODUCTS = {
    "지성": [
        {"name": "이니스프리 노세범 미네랄 파우더", "category": "피지관리", "users": 842},
        {"name": "라로슈포제 에빠끌라 토너", "category": "토너", "users": 701},
        {"name": "파티온 노스카나인 트러블 세럼", "category": "세럼", "users": 655},
    ],
    "건성": [
        {"name": "라운드랩 자작나무 수분 크림", "category": "수분크림", "users": 913},
        {"name": "토리든 다이브인 세럼", "category": "세럼", "users": 780},
        {"name": "에스트라 에이시카365 수분 진정 크림", "category": "진정크림", "users": 642},
    ],
    "복합성": [
        {"name": "아누아 어성초 77 토너", "category": "토너", "users": 734},
        {"name": "토리든 다이브인 세럼", "category": "세럼", "users": 690},
        {"name": "라운드랩 자작나무 수분 크림", "category": "수분크림", "users": 611},
    ],
    "민감성": [
        {"name": "라로슈포제 시카플라스트 밤 B5", "category": "밤·연고", "users": 688},
        {"name": "닥터자르트 시카페어 토너", "category": "토너", "users": 599},
        {"name": "에스트라 에이시카365 수분 진정 크림", "category": "진정크림", "users": 540},
    ],
    "수부지": [
        {"name": "토리든 다이브인 저분자 히알루론산 세럼", "category": "세럼", "users": 720},
        {"name": "라로슈포제 에빠끌라 토너", "category": "토너", "users": 648},
        {"name": "에스트라 에이시카365 수분 진정 크림", "category": "진정크림", "users": 590},
    ],
}

# 3개월 사용 화장품 내역용 제품 풀 (랭킹 유저별 주문서에 표시)
_HISTORY_POOL = [
    "라운드랩 자작나무 수분 크림", "아누아 어성초 77 토너", "토리든 다이브인 세럼",
    "라로슈포제 에빠끌라 토너", "파티온 노스카나인 트러블 세럼",
    "에스트라 에이시카365 수분 진정 크림", "닥터자르트 시카페어 토너",
    "닥터지 레드 블레미쉬 진정 크림", "센카 퍼펙트 휩 클렌징폼",
    "이니스프리 노세범 미네랄 파우더", "라로슈포제 안뗄리오스 선크림",
    "우르오스 스킨워시 클렌저",
]

# 기존 진단 기록(실명/게스트 등)을 노출할 때 씌우는 닉네임 풀.
# 랭킹에서 실명은 절대 보이지 않게 하고, 서로 '중복되지 않게' 순서대로 배정한다.
# MOCK_RANKING 의 닉네임과도 겹치지 않도록 서로 다른 문자열로 구성한다.
_RECORD_NICKS = [
    "역삼동 뽀샤시", "망원동 촉촉남", "구의동 맑은결", "신촌 유리알", "종로 도자기왕",
    "왕십리 광채남", "목동 꿀광피부", "관악 무결점남", "동탄 물빛남", "송파 반짝이",
    "노원 클린페이스", "은평 뽀송남", "마포 개운한피부", "강서 산뜻남", "성북 정돈남",
    "중랑 안정피부", "금천 생기남", "도봉 투명피부", "용산 밸런스남", "양천 청결남",
    "강동 매끈남", "구로 촉촉핏", "서대문 광채핏", "동대문 클리어", "영등포 뽀얀남",
    "성동 유리막", "광진 물광킹", "노들 도자기핏", "상암 청량남", "가산 반짝핏",
]
# 바(bare) 성분명 - 기록의 product가 이런 성분명이면 실제 제품명으로 치환한다.
_BARE_INGREDIENTS = {
    "히알루론산", "나이아신아마이드", "센텔라", "비타민C", "세라마이드", "판테놀",
    "어성초", "마데카소사이드", "살리실산", "카페인", "중성", "-", "",
}


def unique_nick(idx: int, used: set[str]) -> str:
    """랭킹에서 서로 겹치지 않는 닉네임을 배정한다.
    idx(기록 순번)로 풀에서 하나 고르고, 이미 쓰였으면 번호를 붙여 유일하게 만든다."""
    base = _RECORD_NICKS[idx % len(_RECORD_NICKS)]
    if idx < len(_RECORD_NICKS) and base not in used:
        return base
    # 풀을 다 썼거나 충돌하면 'OOO N호'로 유일하게 보장
    n = idx // len(_RECORD_NICKS) + 2
    cand = f"{base} {n}호"
    while cand in used:
        n += 1
        cand = f"{base} {n}호"
    return cand


def real_product_for(seed_text: str, current: str | None) -> str:
    """기록의 product가 성분명이거나 비어 있으면 실제 제품명으로 바꿔서 반환한다."""
    prod = (current or "").strip()
    if prod and prod not in _BARE_INGREDIENTS:
        return prod
    seed = sum(ord(c) for c in (seed_text or "")) or 1
    return _HISTORY_POOL[seed % len(_HISTORY_POOL)]


# 마이페이지 나이대 드롭다운 / 기록 랜덤 나이대에 쓰는 나이대 목록
AGE_GROUPS = ["10대", "20대", "30대", "40대", "50대", "60대 이상"]


def random_age_group(seed_text: str) -> str:
    """나이대가 없는 기존 기록에 결정적으로 랜덤 나이대를 배정한다 (20~40대 위주)."""
    seed = sum(ord(c) for c in (seed_text or "")) or 1
    weighted = ["10대", "20대", "20대", "30대", "30대", "40대", "40대", "50대"]
    return weighted[seed % len(weighted)]

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

# 피부 좋은 남자들 전국 분포 - 목업 데이터 (x/y = 지도 이미지 내 % 위치)
MOCK_REGIONS = [
    {"name": "서울", "count": 342, "x": 39, "y": 22},
    {"name": "경기", "count": 288, "x": 48, "y": 31},
    {"name": "인천", "count": 121, "x": 29, "y": 27},
    {"name": "강원", "count": 76, "x": 66, "y": 17},
    {"name": "대전", "count": 134, "x": 46, "y": 47},
    {"name": "대구", "count": 118, "x": 61, "y": 54},
    {"name": "부산", "count": 156, "x": 70, "y": 66},
    {"name": "광주", "count": 98, "x": 34, "y": 62},
    {"name": "제주", "count": 41, "x": 31, "y": 90},
]

# 우리 동네(성남시 분당·판교) 동별 피부 우수자 수 - 목업 데이터 (삼평동 1등)
MOCK_DONGS = [
    {"name": "삼평동", "count": 142},
    {"name": "정자동", "count": 128},
    {"name": "판교동", "count": 116},
    {"name": "서현동", "count": 99},
    {"name": "수내동", "count": 81},
    {"name": "이매동", "count": 67},
    {"name": "야탑동", "count": 59},
    {"name": "백현동", "count": 52},
]

EVENT_LABELS = {
    "date": "소개팅",
    "interview": "면접",
    "wedding": "결혼식",
    "meeting": "상견례",
    "dating": "데이트",
}

# 커뮤니티 게시판 - 나이대별 목업 글 (type: concern 고민 / brag 자랑 / poll 투표)
COMMUNITY_GROUPS = ["10대", "20-30대", "40-50대"]
MOCK_COMMUNITY = {
    "10대": [
        {"id": "c10_1", "type": "poll", "author": "성수동 광채남", "tag": "투표",
         "title": "학교 사진 찍는 날, 뭐 바르고 갈까?",
         "option_a": "선크림만 가볍게", "option_b": "쿠션까지 살짝",
         "votes_a": 58, "votes_b": 33, "comments": 6},
        {"id": "c10_2", "type": "brag", "author": "역삼동 뽀샤시", "tag": "자랑",
         "title": "10대 랭킹 3위 찍었습니다 🎉",
         "body": "토너-세럼만 꾸준히 했는데 점수 84점! 다들 화이팅",
         "likes": 41, "comments": 15},
        {"id": "c10_3", "type": "concern", "author": "상도동 뉴비", "tag": "고민",
         "title": "코에 좁쌀 여드름 어떡하죠?",
         "body": "세수만 하는데 코에 좁쌀이 계속 올라와요. 뭐부터 써야 하나요?",
         "likes": 12, "comments": 8},
    ],
    "20-30대": [
        {"id": "c23_1", "type": "poll", "author": "서초동 피부왕자", "tag": "투표",
         "title": "소개팅 전날, 새 팩 써도 될까?",
         "option_a": "그냥 하던 것만", "option_b": "진정팩 하나 추가",
         "votes_a": 121, "votes_b": 87, "comments": 24},
        {"id": "c23_2", "type": "brag", "author": "잠실 반짝이", "tag": "자랑",
         "title": "소개팅 성공했습니다 (피부 칭찬 들음)",
         "body": "3개월 관리했더니 '피부 좋으시네요' 소리 들었어요 ㅎㅎ",
         "likes": 96, "comments": 31},
        {"id": "c23_3", "type": "concern", "author": "판교 개발자피부", "tag": "고민",
         "title": "야근하면 피부가 뒤집혀요…",
         "body": "밤샘 잦은데 트러블+칙칙함 콤보네요. 직장인 루틴 추천 좀요",
         "likes": 33, "comments": 19},
    ],
    "40-50대": [
        {"id": "c45_1", "type": "poll", "author": "목동 꿀광피부", "tag": "투표",
         "title": "자녀 결혼식, 피부 시술 받을까?",
         "option_a": "기본 보습·선크림만", "option_b": "가벼운 시술 받기",
         "votes_a": 74, "votes_b": 68, "comments": 20},
        {"id": "c45_2", "type": "concern", "author": "연남동 무결점", "tag": "고민",
         "title": "눈가 주름·탄력 관리 뭐부터?",
         "body": "이 나이에 시작해도 늦지 않았죠? 기본템부터 알려주세요",
         "likes": 27, "comments": 14},
        {"id": "c45_3", "type": "brag", "author": "구의동 맑은결", "tag": "자랑",
         "title": "40대 턴오버 상승 1위 달성 🚀",
         "body": "동네 랭킹에서 제일 많이 올랐대요. 꾸준함이 답입니다",
         "likes": 52, "comments": 11},
    ],
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
    "중성": [("아누아 어성초 77 토너", "피부 밸런스를 안정적으로 잡아주는 토너"),
             ("라운드랩 자작나무 수분 크림", "무난하게 수분·유분 밸런스를 맞춰주는 크림")],
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


def recommend_products(diagnosis: dict) -> list[dict]:
    """진단 결과(피부 타입)에 맞춘 추천 제품 목록. 민감도 카테고리가 낮으면 진정 제품을 우선하고,
    이벤트 대비 선크림도 포함한다."""
    categories = diagnosis.get("categories") or {}
    if categories.get("sensitivity", 100) < 55:
        skin_type = "민감성"
    else:
        skin_type = diagnosis.get("skin_type", "")
    items = LOCAL_PRODUCTS.get(skin_type, LOCAL_PRODUCTS.get("중성", LOCAL_PRODUCTS["복합성"]))
    picks = [{"name": n, "reason": why} for n, why in items]
    picks.append({"name": "라로슈포제 안뗄리오스 선크림",
                  "reason": "자외선 차단은 피부 관리의 기본이에요"})
    return picks[:3]


def person_history(name: str) -> list[dict]:
    """랭킹 유저의 최근 3개월 사용 화장품 내역(목업)을 이름 기반으로 안정적으로 생성."""
    seed = sum(ord(c) for c in name) or 1
    rnd = random.Random(seed)
    count = rnd.randint(3, 5)
    picks = rnd.sample(_HISTORY_POOL, min(count, len(_HISTORY_POOL)))
    months = ["2026-06", "2026-05", "2026-04"]  # 최근 3개월
    history = [
        {"product": p, "date": f"{rnd.choice(months)}-{rnd.randint(1, 28):02d}"}
        for p in picks
    ]
    history.sort(key=lambda h: h["date"], reverse=True)
    return history


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


SLIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mint_pixel_slime.png")
# 챗봇 진입 버튼용 max 캐릭터(물음표 슬라임) 이미지. 없으면 기본 슬라임으로 대체.
QUESTION_SLIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_slime.png")

# 피부랭킹 지도 배경 이미지. 'map_image' 파일을 우선 찾고, 없으면 기존 한국 지도 파일 사용.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MAP_CANDIDATES = [
    "map_image.png", "map_image.jpg", "map_image.jpeg", "map_image.webp",
    "korea_mint_70_transparent.png",
]
MAP_PATH = next(
    (os.path.join(_BASE_DIR, n) for n in _MAP_CANDIDATES
     if os.path.exists(os.path.join(_BASE_DIR, n))),
    os.path.join(_BASE_DIR, "korea_mint_70_transparent.png"),
)


@st.cache_data(show_spinner=False)
def slime_data_uri(size: int = 96) -> str | None:
    """귀여운 슬라임 마스코트 이미지를 data URI로 반환 (지도 마커용)."""
    try:
        img = Image.open(SLIME_PATH).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False)
def question_slime_data_uri(size: int = 220) -> str | None:
    """챗봇 버튼용 max(물음표 슬라임) 이미지를 data URI로 반환.
    원본이 검은 배경이라 가공 없이 그대로 쓰고, 버튼 배경색도 같은 검정으로 맞춰
    로고 버튼과 동일한 단일 background-image 방식으로 안정적으로 표시한다.
    question_slime.png가 없으면 기본 슬라임 이미지로 폴백한다."""
    path = QUESTION_SLIME_PATH if os.path.exists(QUESTION_SLIME_PATH) else SLIME_PATH
    try:
        img = Image.open(path).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False)
def map_data_uri(size: int = 720) -> str | None:
    """피부랭킹 지도 배경 이미지를 data URI로 반환."""
    try:
        img = Image.open(MAP_PATH).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# 마스터 티어 달성 보상(코스맥스 3WAAU 본품) 이미지. '3WAAU.*' 파일을 우선 찾는다.
# 파일이 없으면 이미지 없이 텍스트 카드로만 표시된다 (다른 이미지들과 동일한 폴백 방식).
_GIFT_CANDIDATES = [
    "3WAAU.png", "3WAAU.jpg", "3WAAU.jpeg", "3WAAU.webp",
    "3waau.png", "3waau.jpg", "3waau.jpeg", "3waau.webp",
]
GIFT_PATH = next(
    (os.path.join(_BASE_DIR, n) for n in _GIFT_CANDIDATES
     if os.path.exists(os.path.join(_BASE_DIR, n))),
    os.path.join(_BASE_DIR, "3WAAU.png"),
)


@st.cache_data(show_spinner=False)
def reward_gift_data_uri(size: int = 640) -> str | None:
    """마스터 보상 제품(3WAAU 본품) 이미지를 data URI로 반환. 없으면 None."""
    try:
        img = Image.open(GIFT_PATH).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 지목 매치 - 닉네임 기반 피부점수 대결
# ---------------------------------------------------------------------------
# 매치 캐릭터 원본 이미지. 사용자가 이 경로에 일러스트를 넣으면 그 이미지를 쓰고,
# 없으면 이미 있는 슬라임 마스코트 이미지로 대체한다 (같은 그림, 색만 다르게 입힌다).
CHARACTER_PATH = os.path.join(_BASE_DIR, "assets", "character.png")

# 민트(#43d3b0) 테마와 어울리는 hue-rotate 각도 팔레트 (8종).
# 닉네임 문자열을 해시해 이 중 하나를 결정적으로 골라 같은 닉네임은 항상 같은 색이 나오게 한다.
_MATCH_HUE_PALETTE = [0, 45, 90, 135, 180, 220, 260, 300]


def match_hue_for_nickname(nickname: str) -> int:
    """닉네임을 해시해 _MATCH_HUE_PALETTE 중 하나를 결정적으로 배정."""
    seed = sum(ord(c) for c in (nickname or "")) or 1
    return _MATCH_HUE_PALETTE[seed % len(_MATCH_HUE_PALETTE)]


# 랜덤 매치용 상대 닉네임 풀 (랜덤 선택 버튼을 누르면 이 중 무작위로 골라 대결)
RANDOM_MATCH_NICKS = [
    "강남 물광남", "홍대 꿀피부", "부산 도자기", "일산 뽀샤시", "수원 광채왕",
    "대구 무결점", "인천 피부요정", "청담 유리알", "노원 촉촉남", "분당 맑은피부",
    "성수 뽀송남", "잠실 반짝이", "판교 개발자피부", "제주 청정남",
]


# 매치 리워드 - 시작하는 순간 사용되는 포인트 / 승리 시 지급되는 포인트
MATCH_ENTRY_COST = 2
MATCH_WIN_REWARD = 20


def random_match_result(my_score: int) -> tuple[int, bool]:
    """상대 점수를 42~100 사이 랜덤으로 뽑고, 내 점수와 비교해 승/패를 정한다.
    (상대방 점수는 항상 42점 이상으로 나온다.)"""
    opp_score = random.randint(42, 100)
    return opp_score, (my_score > opp_score)


@st.cache_data(show_spinner=False)
def character_data_uri(size: int = 220) -> str | None:
    """매치 캐릭터 원본 이미지를 data URI로 반환.
    assets/character.png가 없으면 슬라임 이미지로 대체하고, 그마저 없으면 None."""
    path = CHARACTER_PATH if os.path.exists(CHARACTER_PATH) else SLIME_PATH
    try:
        img = Image.open(path).convert("RGBA")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def match_character_html(css_class: str, hue: int | None, grayscale: bool = False) -> str:
    """매치 화면의 캐릭터 이미지 HTML 한 개를 반환.
    hue가 None이거나 grayscale=True면(상대 미정) 회색조로, 아니면 hue-rotate(deg)로 착색한다."""
    if grayscale or hue is None:
        filt = "grayscale(1) brightness(0.85) contrast(0.9)"
    else:
        filt = f"hue-rotate({hue}deg) saturate(1.4)"
    uri = character_data_uri()
    if uri:
        return f'<img class="{css_class}" src="{uri}" style="filter:{filt}" alt="character">'
    return f'<div class="{css_class} cl-match__emoji" style="filter:{filt}">🥷</div>'


# 픽셀 버블 색상 팔레트 (민트 계열)
_BUBBLE_COLORS = ["rgba(67,211,176,{a})", "rgba(94,234,212,{a})", "rgba(163,217,119,{a})"]


def render_max_loading(message: str) -> None:
    """대표 캐릭터 max가 둥둥 뜨고 뒤로 픽셀 버블이 올라오는 스플래시 오버레이."""
    uri = slime_data_uri(200)
    bubbles = ""
    for _ in range(28):
        left = random.randint(0, 98)
        size = random.randint(6, 20)
        dur = random.uniform(2.4, 5.8)
        delay = random.uniform(0.0, 2.6)
        color = random.choice(_BUBBLE_COLORS).format(a=f"{random.uniform(0.25, 0.75):.2f}")
        bubbles += (
            f'<span class="cl-bubble" style="left:{left}%;width:{size}px;height:{size}px;'
            f'background:{color};animation-duration:{dur:.1f}s;animation-delay:{delay:.1f}s"></span>'
        )
    char = (f'<img class="cl-splash__max" src="{uri}" alt="max">' if uri
            else '<div class="cl-splash__max" style="font-size:80px">🫧</div>')
    st.markdown(
        f'<div class="cl-splash"><div class="cl-splash__bubbles">{bubbles}</div>'
        f'{char}<div class="cl-splash__msg">{message}</div></div>',
        unsafe_allow_html=True,
    )


# 사진+설문 제출 후 보여주는 "AI 분석 중" 문구 (실제 분석은 없고, 문구만 순환한다)
_ANALYSIS_MESSAGES = [
    "피부 데이터를 분석하고 있습니다",
    "유수분 밸런스를 계산하는 중입니다",
    "트러블 및 민감도 지표를 정리하고 있습니다",
    "맞춤 결과를 생성하고 있습니다",
]


def render_analysis_loading() -> None:
    """설문/사진 제출 후 2~4초간 문구가 바뀌는 'AI 분석 중' 스플래시.
    실제 분석은 하지 않고, max 스플래시 위에 문구만 순환시킨다."""
    placeholder = st.empty()
    per_message = random.uniform(0.6, 1.0)
    for msg in _ANALYSIS_MESSAGES:
        with placeholder.container():
            render_max_loading(msg)
        time.sleep(per_message)
    placeholder.empty()


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
# 피부 진단 시뮬레이션
#
# 실제 이미지 분석 모델이나 DB 학습은 전혀 쓰지 않는다. 발표에서 설명할
# "설문 70% + 이미지 30%" 가중치 구조를 코드로도 그대로 흉내 내기 위해:
#   1) 설문 응답 -> 카테고리별 "그럴듯한 범위"를 정하고 그 안에서 랜덤 생성 (설문 점수)
#   2) 카테고리별 랜덤 값을 하나 더 생성 (이미지 점수 - 실제 분석 없이 순수 랜덤)
#   3) 카테고리 최종 점수 = 설문 점수 x 0.7 + 이미지 점수 x 0.3
#   4) 최종 피부 점수는 항상 이 카테고리 점수들의 가중합으로 "역산"해서 계산하므로
#      화면에 보이는 세부 점수와 최종 점수가 항상 정확히 일치한다.
# ---------------------------------------------------------------------------
_INGREDIENT_POOL = [
    "히알루론산", "나이아신아마이드", "센텔라", "비타민C",
    "세라마이드", "판테놀", "어성초", "마데카소사이드",
]
# 고민 -> 추천 성분 매핑
_CONCERN_TO_INGREDIENT = {
    "홍조": "센텔라", "속건조": "세라마이드", "번들거림(유분)": "나이아신아마이드",
    "칙칙함": "비타민C", "트러블": "어성초", "모공": "나이아신아마이드",
    "다크서클": "카페인", "각질": "판테놀",
}

# 결과 화면에 보여줄 5개 세부 카테고리 (0~100점, 가로 바로 표시)
_CATEGORY_KEYS = ["moisture", "trouble", "sensitivity", "tone", "texture"]
_CATEGORY_LABELS = {
    "moisture": "유수분 밸런스", "trouble": "트러블", "sensitivity": "민감도/홍조",
    "tone": "다크서클/톤 균일도", "texture": "피부결/모공",
}
# 최종 피부 점수 = 카테고리별 가중합 (총합 100%)
_CATEGORY_WEIGHTS = {"moisture": 0.25, "trouble": 0.25, "sensitivity": 0.20,
                     "tone": 0.15, "texture": 0.15}
_CATEGORY_CONCERN = {"trouble": "트러블", "sensitivity": "홍조",
                     "tone": "칙칙함", "texture": "모공"}

# 설문 7문항. 각 옵션의 "effects"는 해당 카테고리가 얼마나 안 좋은 쪽(0=좋음~3=나쁨)인지를
# 나타내고, "direction"은 유수분 타입(건성/지성/복합성/중성) 판단에 쓰인다.
SURVEY_QUESTIONS = [
    {"key": "q1", "text": "세안 후 아무것도 바르지 않았을 때 피부 상태는 어떤가요?",
     "options": [
         {"label": "많이 당기고 건조하다", "effects": {"moisture": 3.0}, "direction": "건성"},
         {"label": "약간 당긴다", "effects": {"moisture": 1.5}, "direction": "건성"},
         {"label": "별 느낌 없다", "effects": {"moisture": 0.0}, "direction": "중성"},
         {"label": "금방 번들거린다", "effects": {"moisture": 2.0}, "direction": "지성"},
     ]},
    {"key": "q2", "text": "오후가 되면 얼굴 유분은 어느 정도 올라오나요?",
     "options": [
         {"label": "거의 없다", "effects": {"moisture": 1.2}, "direction": "건성"},
         {"label": "T존만 약간 올라온다", "effects": {"moisture": 0.3, "texture": 0.4}, "direction": "복합성"},
         {"label": "얼굴 전체가 번들거린다", "effects": {"moisture": 2.5, "texture": 1.3}, "direction": "지성"},
         {"label": "날마다 다르다", "effects": {"moisture": 1.5}, "direction": "복합성"},
     ]},
    {"key": "q3", "text": "최근 한 달 동안 트러블은 얼마나 자주 올라왔나요?",
     "options": [
         {"label": "거의 없다", "effects": {"trouble": 0.0}},
         {"label": "가끔 1~2개 올라온다", "effects": {"trouble": 1.0}},
         {"label": "자주 반복된다", "effects": {"trouble": 2.2}},
         {"label": "항상 있는 편이다", "effects": {"trouble": 3.0}},
     ]},
    {"key": "q4", "text": "가장 신경 쓰이는 피부 고민은 무엇인가요?",
     "options": [
         {"label": "트러블/여드름", "effects": {"trouble": 1.6}},
         {"label": "번들거림/피지", "effects": {"moisture": 1.6}, "direction": "지성"},
         {"label": "건조함", "effects": {"moisture": 1.6}, "direction": "건성"},
         {"label": "모공/피부결", "effects": {"texture": 2.0}},
         {"label": "다크서클/칙칙함", "effects": {"tone": 2.0}},
         {"label": "홍조/민감함", "effects": {"sensitivity": 1.6}},
         {"label": "잘 모르겠다", "effects": {}},
     ]},
    {"key": "q5", "text": "피부가 붉어지거나 예민해지는 편인가요?",
     "options": [
         {"label": "거의 없다", "effects": {"sensitivity": 0.0}},
         {"label": "가끔 있다", "effects": {"sensitivity": 1.0}},
         {"label": "자주 있다", "effects": {"sensitivity": 2.2}},
         {"label": "새로운 제품을 쓰면 쉽게 자극이 생긴다", "effects": {"sensitivity": 3.0}},
     ]},
    {"key": "q6", "text": "면도 후 피부 자극이나 트러블이 생기는 편인가요?",
     "options": [
         {"label": "거의 없다", "effects": {"sensitivity": 0.0, "trouble": 0.0}},
         {"label": "가끔 있다", "effects": {"sensitivity": 1.0, "trouble": 0.5}},
         {"label": "자주 있다", "effects": {"sensitivity": 2.0, "trouble": 1.0}},
         {"label": "면도 후 항상 신경 쓰인다", "effects": {"sensitivity": 3.0, "trouble": 1.5}},
     ]},
    {"key": "q7", "text": "지금 내 피부에 가장 가까운 설명은 무엇인가요?",
     "options": [
         {"label": "전체적으로 건조한 편이다", "effects": {"moisture": 2.4}, "direction": "건성"},
         {"label": "전체적으로 번들거리는 편이다", "effects": {"moisture": 2.0}, "direction": "지성"},
         {"label": "부위별로 다르다", "effects": {"moisture": 1.2, "texture": 0.8}, "direction": "복합성"},
         {"label": "트러블이 자주 생긴다", "effects": {"trouble": 2.2}},
         {"label": "예민하고 쉽게 자극받는다", "effects": {"sensitivity": 2.4}},
         {"label": "잘 모르겠다", "effects": {}},
     ]},
]

# 피부 타입 라벨 풀 - 완전 랜덤이 아니라 "가장 낮은 카테고리"를 기준으로 그중 하나를 뽑는다.
_TYPE_LABEL_POOL = {
    "moisture_건성": ["수분부족형 건성", "속건조 건성", "수분부족형 복합성"],
    "moisture_지성": ["과다피지형 지성", "번들거림형 지성", "피지과다 복합성"],
    "moisture_기타": ["밸런스 불균형 복합성", "수분부족형 복합성"],
    "trouble": ["트러블형 지성", "트러블형 복합성", "트러블 케어 필요형"],
    "sensitivity": ["민감성 건성", "홍조 민감형", "민감성 복합성"],
    "tone": ["칙칙톤 관리형", "톤불균일 케어형", "다크서클 케어형"],
    "texture": ["모공관리형 복합성", "피부결 개선형", "모공케어 지성"],
    "balanced": ["균형형 보통 피부", "밸런스 좋은 중성 피부", "안정형 피부"],
}


def _severity_to_range(sev: float) -> tuple[int, int]:
    """설문 응답의 '심각도'(0=좋음~3=나쁨)를 그럴듯한 점수 범위로 변환한다.
    완전 무작위가 아니라 응답에 따라 범위 자체가 달라지게 하는 핵심 로직."""
    sev = max(0.0, min(3.0, sev))
    anchors = [(0.0, (78, 95)), (1.0, (62, 82)), (2.0, (44, 64)), (3.0, (22, 46))]
    for (s0, (lo0, hi0)), (s1, (lo1, hi1)) in zip(anchors, anchors[1:]):
        if s0 <= sev <= s1:
            t = (sev - s0) / (s1 - s0)
            return round(lo0 + (lo1 - lo0) * t), round(hi0 + (hi1 - hi0) * t)
    return anchors[-1][1]


def compute_survey_scores(answers: dict) -> tuple[dict, str]:
    """설문 답변(question key -> 선택한 라벨)으로 카테고리별 '설문 점수'와
    유수분 타입(건성/지성/복합성/중성)을 계산한다."""
    sums = {k: 0.0 for k in _CATEGORY_KEYS}
    counts = {k: 0 for k in _CATEGORY_KEYS}
    direction_votes = []
    for q in SURVEY_QUESTIONS:
        opt = next((o for o in q["options"] if o["label"] == answers.get(q["key"])), None)
        if not opt:
            continue
        for cat, val in opt.get("effects", {}).items():
            sums[cat] += val
            counts[cat] += 1
        if opt.get("direction"):
            direction_votes.append(opt["direction"])

    survey_scores = {}
    for cat in _CATEGORY_KEYS:
        avg_severity = sums[cat] / counts[cat] if counts[cat] else 1.3  # 관련 응답 없으면 보통 수준
        lo, hi = _severity_to_range(avg_severity)
        survey_scores[cat] = random.randint(lo, hi)

    moisture_direction = Counter(direction_votes).most_common(1)[0][0] if direction_votes else "중성"
    return survey_scores, moisture_direction


def build_skin_type_label(categories: dict, moisture_direction: str) -> str:
    """가장 낮은(안 좋은) 카테고리를 기준으로 그럴듯한 피부 타입 라벨을 뽑는다."""
    worst_key = min(categories, key=categories.get)
    if categories[worst_key] >= 72:
        return random.choice(_TYPE_LABEL_POOL["balanced"])
    if worst_key == "moisture":
        pool_key = f"moisture_{moisture_direction}" if f"moisture_{moisture_direction}" in _TYPE_LABEL_POOL \
            else "moisture_기타"
    else:
        pool_key = worst_key
    return random.choice(_TYPE_LABEL_POOL.get(pool_key, _TYPE_LABEL_POOL["balanced"]))


def summary_from_categories(categories: dict) -> str:
    """가장 낮은 1~2개 카테고리를 기준으로 한 줄 요약을 만든다."""
    ranked = sorted(categories.items(), key=lambda x: x[1])
    worst_key, worst_score = ranked[0]
    second_key, second_score = ranked[1]
    best_key, _best_score = ranked[-1]
    if worst_score < 55:
        if second_score < 62 and second_key != worst_key:
            return f"{_CATEGORY_LABELS[worst_key]}과 {_CATEGORY_LABELS[second_key]} 관리가 우선적으로 필요합니다."
        return f"{_CATEGORY_LABELS[worst_key]} 관리가 우선적으로 필요합니다."
    if worst_score < 72:
        return f"{_CATEGORY_LABELS[best_key]}는 양호하지만 {_CATEGORY_LABELS[worst_key]} 개선이 필요합니다."
    if worst_score < 85:
        return f"전반적으로 안정적이지만 {_CATEGORY_LABELS[worst_key]} 관리 여지가 있습니다."
    return "전체적으로 균형 잡힌 우수한 피부 상태예요! 지금 루틴을 그대로 유지해보세요."


def concerns_from_categories(categories: dict, moisture_direction: str) -> list[str]:
    """점수가 낮은 카테고리들을 기존 '고민' 단어로 변환한다 (챗봇 맥락·성분 추천에 재사용)."""
    concerns = []
    for key, score in sorted(categories.items(), key=lambda x: x[1]):
        if score >= 65:
            continue
        if key == "moisture":
            concerns.append("속건조" if moisture_direction == "건성" else "번들거림(유분)")
        else:
            concerns.append(_CATEGORY_CONCERN.get(key, key))
    return concerns[:4] or ["전반적으로 안정적"]


def ingredients_from_concerns(concerns: list[str]) -> list[str]:
    """고민 목록 기반 추천 성분 (부족하면 랜덤으로 채움)."""
    ingredients = [_CONCERN_TO_INGREDIENT[c] for c in concerns if c in _CONCERN_TO_INGREDIENT]
    for ing in random.sample(_INGREDIENT_POOL, k=len(_INGREDIENT_POOL)):
        if len(ingredients) >= 3:
            break
        ingredients.append(ing)
    return list(dict.fromkeys(ingredients))[:3]


def simulate_ai_diagnosis(answers: dict) -> dict:
    """설문 답변으로 'AI가 분석한 것처럼' 보이는 진단 결과를 생성한다.
    실제 이미지 분석/모델 추론은 전혀 하지 않고, 설문 기반 랜덤 점수(70%)와
    순수 랜덤 이미지 점수(30%)를 카테고리별로 합성해 최종 점수를 만든다."""
    survey_scores, moisture_direction = compute_survey_scores(answers)
    # '이미지 분석'을 흉내 내는 값 - 실제 분석 없이 카테고리별로 독립적인 랜덤값만 생성한다.
    image_scores = {cat: random.randint(45, 92) for cat in _CATEGORY_KEYS}

    categories = {}
    for cat in _CATEGORY_KEYS:
        blended = survey_scores[cat] * 0.7 + image_scores[cat] * 0.3
        categories[cat] = max(0, min(100, round(blended)))

    # 최종 피부 점수는 항상 위 카테고리 점수의 가중합으로 계산 -> 화면 표시값과 100% 일치.
    final_score = max(0, min(100, round(
        sum(categories[c] * _CATEGORY_WEIGHTS[c] for c in _CATEGORY_KEYS))))

    concerns = concerns_from_categories(categories, moisture_direction)
    return {
        "score": final_score,
        "categories": categories,
        "survey_categories": survey_scores,
        "image_categories": image_scores,
        "skin_type": moisture_direction,
        "type_label": build_skin_type_label(categories, moisture_direction),
        "summary": summary_from_categories(categories),
        "concerns": concerns,
        "recommended_ingredients": ingredients_from_concerns(concerns),
        "gain": random.randint(3, 18),  # 피부 턴오버(28일) 동안의 예상 개선 점수
        "answers": answers,
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
    "너는 'clozkin'의 대표 캐릭터이자 AI 뷰티 가이드 'max'야. "
    "너는 민트색 픽셀 슬라임 캐릭터 max로서 1인칭으로 직접 대화해. "
    "뷰티에 처음 입문하는 남성 고객을 돕는 게 목표야. "
    "스킨케어를 세안처럼 '당연히 하는 행동'으로 느끼게 도와줘. "
    "전문 용어는 최소화하고, 초보자도 부담 없이 따라할 수 있게 아주 친근하고 짧게 답해. "
    "한 번에 너무 많은 걸 시키지 말고 딱 필요한 것만 골라줘. "
    "가끔 '나 max가~' 처럼 너 자신을 max라고 부르며 캐릭터답게 귀엽고 친근하게 말해도 좋아. "
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

.stApp {
  background:
    radial-gradient(1100px 560px at 50% -12%, rgba(67, 211, 176, 0.13), transparent 60%),
    radial-gradient(900px 500px at 110% 8%, rgba(94, 234, 212, 0.06), transparent 55%),
    linear-gradient(180deg, #0b0e13 0%, #070a0e 100%);
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
/* 본문은 베젤(430px) 안쪽으로 좌우 18px 여백을 둬서 베젤 밖으로 넘치지 않게 한다 */
.block-container { max-width: 430px; margin: 0 auto;
  padding: 2.2rem 18px 7rem; overflow-x: clip; }
#MainMenu, header, footer { visibility: hidden; }
/* 스크롤바가 한쪽에만 생겨 본문이 베젤보다 왼쪽으로 치우치는 현상 방지
   (양쪽에 동일한 스크롤바 여백을 확보해 뷰포트 정중앙 = 베젤 정중앙이 되게 한다) */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  scrollbar-gutter: stable both-edges;
}

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

/* ---- 공통 섹션 제목 (가운데 정렬) ---- */
.cl-h { font-size: 24px; font-weight: 800; letter-spacing: -0.7px; margin: 2px 0 4px;
  text-align: center; }
.cl-sec { font-family: 'Space Grotesk', monospace; font-size: 11px; letter-spacing: 2px;
  color: var(--muted); text-transform: uppercase; margin: 22px 0 12px; text-align: center; }
/* 본문 캡션도 가운데 정렬로 통일 */
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {
  text-align: center; }

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

/* ---- 진단 결과: 카테고리별 점수 바 ---- */
.cl-catbar { margin-bottom: 14px; }
.cl-catbar__top { display: flex; justify-content: space-between; align-items: baseline;
  font-size: 13.5px; font-weight: 700; margin-bottom: 6px; }
.cl-catbar__top span:last-child { font-family: 'Space Grotesk', monospace; color: var(--accent); }
.cl-catbar__track { height: 10px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
.cl-catbar__fill { display: block; height: 100%; border-radius: 999px; transition: width 0.6s ease; }

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
.cl-rank__gain--down { color: #ff7b88; }
/* 랭킹 탭 */
.stApp [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stApp [data-baseweb="tab"] { font-weight: 700; }
.stApp [data-baseweb="tab-highlight"] { background: var(--accent); }
/* 가로 라디오(나이대·피부타입 등)가 좁은 폭에서 베젤 밖으로 넘치지 않고 줄바꿈되게 */
.stApp [data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; row-gap: 4px; }
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
  /* 폰 프레임(430px, 가운데) 폭에 맞추고, 버튼 묶음을 가운데로 정렬.
     width는 반드시 auto!important로 둬야 한다 - streamlit이 자체적으로
     width:100%(= 뷰포트 전체 폭)를 강제해서 left/right로 계산되는 430px
     폭을 무시해버리는 문제가 있었다 (탭 6개부터 화면 밖으로 넘쳐 보였음). */
  position: fixed; left: max(0px, calc(50% - 215px)); right: max(0px, calc(50% - 215px));
  width: auto !important;
  bottom: 0; z-index: 9998;
  background: rgba(10, 14, 19, 0.96); backdrop-filter: blur(16px);
  border-top: 1px solid var(--glass-brd);
  padding: 8px 12px calc(8px + env(safe-area-inset-bottom, 0px));
  display: flex; justify-content: center;
}
.st-key-bottomnav > div { width: 100%; }
.st-key-bottomnav [data-testid="stVerticalBlock"] { width: 100%; }
/* 5칸을 항상 가로 한 줄로, 화면(베젤) 폭 안에 균등 분할.
   Streamlit 1.59는 컬럼에 data-testid="stColumn"을 쓴다(구버전의 "column"이 아님) -
   구 선택자로는 매칭이 안 돼 컬럼이 안 줄어들고 넘쳤던 문제를 같이 고쳤다. */
.st-key-bottomnav [data-testid="stHorizontalBlock"] {
  width: 100%; margin: 0 auto; gap: 2px; flex-direction: row; flex-wrap: nowrap;
}
.st-key-bottomnav [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
  width: auto !important; flex: 1 1 0 !important; min-width: 0 !important;
}
/* 버튼 = 아이콘(위) + 작은 글자(아래) 세로 스택 */
.st-key-bottomnav .stButton > button {
  border: 0; background: transparent; color: var(--muted); box-shadow: none;
  border-radius: 12px; padding: 6px 2px; width: 100%; min-height: 0;
  display: flex; flex-direction: column; align-items: center; gap: 1px;
}
.st-key-bottomnav .stButton > button p { margin: 0; line-height: 1.15; text-align: center; }
.st-key-bottomnav .stButton > button p:first-child { font-size: 20px; }           /* 아이콘 */
.st-key-bottomnav .stButton > button p:last-child {
  font-size: 10.5px; font-weight: 700; letter-spacing: -0.2px; }                   /* 글자 */
.st-key-bottomnav .stButton > button:hover { color: var(--text); border: 0; }
.st-key-bottomnav .stButton > button[kind="primary"] {
  background: var(--accent-dim); color: var(--accent); box-shadow: none;
}

/* ---- 플로팅 채팅봇 (우측 하단) ---- */
.st-key-chatwidget {
  /* 넓은 화면에선 본문(430px) 컬럼 오른쪽 가장자리에 맞춰 가운데쪽으로,
     좁은 화면에선 화면 끝 16px 로 자동 조정 */
  position: fixed; right: max(16px, calc(50% - 199px)); bottom: 88px; z-index: 9999;
  width: auto; max-width: calc(100vw - 28px);
}
/* 열린 상태의 채팅 카드 - 불투명 배경으로 본문과 겹쳐 글자가 비치는 문제 방지 */
.st-key-chatcard {
  width: min(340px, calc(100vw - 28px)); margin: 0 0 12px auto;
  background: #0e141b; border: 1px solid var(--glass-brd);
  border-radius: 22px; overflow: hidden; box-shadow: 0 24px 70px rgba(0, 0, 0, 0.62);
}
.st-key-chatcard [data-testid="stVerticalBlock"] { gap: 8px; }
/* 토글 버튼(FAB) - 열림/닫힘 공통. position:relative로 두어 max 이미지를 버튼 위에 겹친다. */
.st-key-chat_fab { position: relative; width: 58px; margin-left: auto; }
/* 내부 요소 컨테이너는 static으로 두어, 오버레이가 .st-key-chat_fab 기준으로 겹치게 한다 */
.st-key-chat_fab [data-testid="stElementContainer"] { position: static; }
/* 닫힘 상태: 버튼 위에 정확히 겹치는 max 캐릭터 오버레이 (클릭은 아래 버튼으로 통과) */
.cl-fab-over { position: absolute; top: 0; left: 0; width: 58px; height: 58px;
  pointer-events: none; z-index: 5; }
.cl-fab-over img { width: 58px; height: 58px; object-fit: cover; border-radius: 16px;
  border: 1px solid rgba(67,211,176,0.7); background: #0a0d10;
  box-shadow: 0 12px 30px rgba(67,211,176,0.4); }
.st-key-chat_fab .stButton > button {
  width: 58px; height: 58px; border-radius: 16px; padding: 0;
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
.cl-chat-head__ava { width: 30px; height: 30px; object-fit: contain; image-rendering: pixelated;
  flex-shrink: 0; filter: drop-shadow(0 3px 8px rgba(67,211,176,0.4)); }
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
.st-key-chat_chips [data-testid="stColumn"] { padding: 0 3px; }
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

/* ---- 리워드: 티어 시스템 ---- */
.cl-tier-row { background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 16px;
  padding: 14px 16px; margin-bottom: 10px; transition: border-color 0.2s ease, box-shadow 0.2s ease; }
.cl-tier-row.is-current { background: var(--accent-dim); border-color: rgba(67,211,176,0.6);
  box-shadow: 0 0 0 1px rgba(67,211,176,0.25), 0 0 24px rgba(67,211,176,0.25); }
.cl-tier-row__top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.cl-tier-row__name { font-size: 14.5px; font-weight: 800; }
.cl-tier-row__range { font-family: 'Space Grotesk', monospace; font-size: 11.5px; color: var(--muted); }
.cl-tier-row.is-current .cl-tier-row__range { color: var(--accent); }
.cl-tier-row__track { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
.cl-tier-row__fill { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-2), var(--accent)); transition: width 0.6s ease; }

/* ---- 리워드: 명예의 전당 ---- */
.cl-hof-lock { background: var(--glass); border: 1px dashed var(--glass-brd); border-radius: 16px;
  padding: 22px 18px; text-align: center; font-size: 13.5px; line-height: 1.65; color: var(--muted); }
.cl-hof-lock b { color: var(--accent); }
.cl-hof-row { display: flex; align-items: center; gap: 12px; background: var(--glass);
  border: 1px solid var(--glass-brd); border-radius: 14px; padding: 12px 16px; margin-bottom: 8px; }
.cl-hof-row.is-me { background: var(--accent-dim); border-color: rgba(67,211,176,0.5); }
.cl-hof-row__rank { width: 30px; flex-shrink: 0; text-align: center; font-size: 17px; font-weight: 800;
  font-family: 'Space Grotesk', monospace; color: var(--muted); }
.cl-hof-row__name { flex: 1; min-width: 0; font-size: 14px; font-weight: 700;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cl-hof-row__points { font-family: 'Space Grotesk', monospace; font-weight: 700; color: var(--accent);
  flex-shrink: 0; }
.cl-hof-row--top1 { border-color: rgba(244,193,94,0.55); box-shadow: 0 0 22px rgba(244,193,94,0.18); }
.cl-hof-row--top1 .cl-hof-row__rank { font-size: 23px; }
.cl-hof-row--top2 .cl-hof-row__rank, .cl-hof-row--top3 .cl-hof-row__rank { font-size: 20px; }
.cl-hof-row--top2 { border-color: rgba(94,234,212,0.4); }
.cl-hof-row--top3 { border-color: rgba(94,234,212,0.25); }

/* ---- 리워드: 마스터 달성 보상(코스맥스 3WAAU 본품 증정) ---- */
.cl-gift { display: flex; align-items: center; gap: 14px; padding: 14px;
  border-radius: 18px; border: 1px solid var(--glass-brd); background: var(--glass); margin-top: 4px; }
.cl-gift--unlocked { border-color: rgba(67,211,176,0.55);
  background: linear-gradient(135deg, rgba(67,211,176,0.16), rgba(94,234,212,0.05));
  box-shadow: 0 10px 30px rgba(67,211,176,0.18); }
.cl-gift__img { width: 90px; height: 90px; object-fit: contain; border-radius: 14px;
  background: #f6f4ef; padding: 6px; flex-shrink: 0; }
.cl-gift--locked .cl-gift__img { filter: grayscale(0.55) brightness(0.92); opacity: 0.7; }
.cl-gift__img--ph { display: flex; align-items: center; justify-content: center;
  font-size: 42px; background: var(--accent-dim); }
.cl-gift__body { min-width: 0; flex: 1; }
.cl-gift__badge { display: inline-block; font-size: 10.5px; font-weight: 800; letter-spacing: 0.3px;
  color: var(--ink); background: var(--accent); padding: 3px 9px; border-radius: 999px; margin-bottom: 6px; }
.cl-gift--locked .cl-gift__badge { background: rgba(255,255,255,0.1); color: var(--muted); }
.cl-gift__title { font-size: 15.5px; font-weight: 800; letter-spacing: -0.3px; margin-bottom: 4px;
  word-break: keep-all; }
.cl-gift__desc { font-size: 12.5px; color: var(--muted); line-height: 1.55; }
.cl-gift__desc b { color: var(--accent); }

/* ---- 커뮤니티 게시판 ---- */
.cl-post { background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 16px;
  padding: 14px 16px; margin-bottom: 10px; text-align: left; }
.cl-post--brag { border-color: rgba(244,193,94,0.4); background: rgba(244,193,94,0.06); }
.cl-post--poll { border-color: rgba(67,211,176,0.4); background: var(--accent-dim); }
.cl-post__head { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
.cl-post__tag { font-size: 10.5px; font-weight: 800; padding: 2px 9px; border-radius: 999px; }
.cl-post__tag--concern { color: var(--accent); background: var(--accent-dim); }
.cl-post__tag--brag { color: #f4c15e; background: rgba(244,193,94,0.16); }
.cl-post__tag--poll { color: var(--ink); background: linear-gradient(115deg, var(--accent-2), var(--accent)); }
.cl-post__author { font-size: 12px; color: var(--muted); font-weight: 600; }
.cl-post__title { font-size: 15px; font-weight: 800; letter-spacing: -0.3px; margin: 0 0 5px; }
.cl-post__body { font-size: 13px; color: var(--text); line-height: 1.55; margin-bottom: 9px; }
.cl-post__meta { font-size: 11.5px; color: var(--muted); }
/* 투표 막대 */
.cl-poll { display: flex; flex-direction: column; gap: 7px; margin: 8px 0 10px; }
.cl-poll__opt { position: relative; display: flex; align-items: center; padding: 9px 12px;
  border-radius: 10px; background: rgba(255,255,255,0.05); overflow: hidden; }
.cl-poll__opt.is-sel { box-shadow: 0 0 0 1px var(--accent) inset; }
.cl-poll__bar { position: absolute; left: 0; top: 0; bottom: 0; z-index: 0;
  background: linear-gradient(90deg, rgba(67,211,176,0.35), rgba(94,234,212,0.22)); }
.cl-poll__label { position: relative; z-index: 1; flex: 1; font-size: 13px; font-weight: 700; }
.cl-poll__pct { position: relative; z-index: 1; font-family: 'Space Grotesk', monospace;
  font-weight: 700; font-size: 13px; color: var(--accent); }

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
/* 상단 MY 버튼 (로그아웃 옆) */
.st-key-mybtn .stButton > button { font-size: 12px; font-weight: 700; padding: 8px 6px;
  color: var(--accent); border-color: rgba(67,211,176,0.35); background: var(--accent-dim); }
.st-key-mybtn .stButton > button:hover { color: var(--ink);
  background: linear-gradient(115deg, var(--accent-2), var(--accent)); border-color: transparent; }

/* 마이페이지 개인정보 카드 */
.cl-info { background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 16px;
  padding: 6px 16px; margin: 4px 0 8px; }
.cl-info-row { display: flex; align-items: center; justify-content: space-between;
  padding: 11px 0; border-bottom: 1px solid var(--glass-brd); font-size: 14px; }
.cl-info-row:last-child { border-bottom: 0; }
.cl-info-row__k { color: var(--muted); font-weight: 600; }
.cl-info-row__v { color: var(--text); font-weight: 700; text-align: right; }

/* 입력창(로그인 / 나이 등) 공통 스타일 */
.stTextInput input, .stNumberInput input {
  background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 12px; color: var(--text);
}
.stTextInput input:focus, .stNumberInput input:focus { border-color: var(--accent); box-shadow: none; }

/* 카메라 프리뷰 + 촬영된 사진 모두 거울(좌우반전) 모드로 - 셀피처럼 자연스럽게 */
.stApp [data-testid="stCameraInput"] video,
.stApp [data-testid="stCameraInput"] img { transform: scaleX(-1); }
/* 사람 가이드 - 카메라 프리뷰에 '사람(머리+어깨 상반신)' 형태의 점선 아웃라인만
   그린다. 블러/어둡게 처리 없이 깔끔하게 가이드 선만 보이게 한다. (첨부 아이콘 참고) */
.stApp [data-testid="stCameraInput"] div:has(> video) { position: relative; }
.stApp [data-testid="stCameraInput"] div:has(> video)::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 420'>\
<path fill='none' stroke='rgba(94,234,212,0.95)' stroke-width='4' stroke-linejoin='round' stroke-linecap='round' stroke-dasharray='11 9' \
d='M150,44 C198,44 224,86 224,140 C224,196 192,232 150,232 C108,232 76,196 76,140 C76,86 102,44 150,44 Z \
M150,248 C198,248 236,272 258,306 C278,338 288,380 294,432 L6,432 C12,380 22,338 42,306 C64,272 102,248 150,248 Z'/>\
</svg>") no-repeat center / 100% 100%;
}

/* 나이대 배지 (랭킹) */
.cl-agechip { display: inline-block; margin-left: 6px; font-size: 10px; font-weight: 700;
  padding: 1px 7px; border-radius: 999px; color: var(--accent); background: var(--accent-dim);
  vertical-align: middle; }
/* 피부 타입 배지 (랭킹) */
.cl-typechip { display: inline-block; margin-left: 5px; font-size: 10px; font-weight: 700;
  padding: 1px 7px; border-radius: 999px; color: #cfd6dd; background: rgba(255,255,255,0.08);
  border: 1px solid var(--glass-brd); vertical-align: middle; }

/* 랭킹 각 줄 = 하나의 박스(카드 + 주문서 버튼을 함께 감싼다).
   기존엔 주문서(구매내역) 버튼이 카드 박스 '밖'에 나가 있었는데 박스 안으로 넣는다. */
[class*="st-key-rankrow_"] {
  background: var(--glass); border: 1px solid var(--glass-brd); border-radius: 16px;
  margin-bottom: 10px; padding: 4px 8px 4px 4px;
}
[class*="st-key-rankrow_"] .cl-rank {
  background: transparent; border: 0; border-radius: 12px; margin: 0; padding: 8px 6px;
}
[class*="st-key-rankrow_"] .cl-rank.is-me { background: var(--accent-dim); box-shadow: none; }
/* 3개 컬럼(정보 / 쇼핑아이콘 / 주문서)을 세로 가운데로 정렬해 한 줄처럼 보이게 한다.
   모바일(좁은 화면)에서도 세로로 줄바꿈되지 않도록 nowrap + min-width:0 을 강제한다. */
[class*="st-key-rankrow_"] [data-testid="stHorizontalBlock"] {
  gap: 2px; align-items: center; flex-wrap: nowrap; }
[class*="st-key-rankrow_"] [data-testid="stColumn"] { align-self: center; min-width: 0; }
/* 구매(쇼핑백+최저가) 컬럼 - 아이콘 그룹 가운데 정렬 */
[class*="st-key-rankrow_"] [data-testid="stColumn"]:nth-child(2) { display: flex;
  align-items: center; justify-content: center; }
[class*="st-key-rankrow_"] [data-testid="stColumn"]:nth-child(2) .cl-shop-group { gap: 6px; }
/* 주문서 팝오버 열 - 세로 가운데 정렬해서 쇼핑백·최저가 아이콘과 높이를 맞춘다 */
[class*="st-key-rankrow_"] [data-testid="stColumn"]:last-child {
  display: flex; align-items: center; justify-content: center; }
[class*="st-key-rankrow_"] [data-testid="stColumn"]:last-child > div { width: 100%; }
[class*="st-key-rankrow_"] [data-testid="stPopover"] { display: flex; justify-content: center; }
/* 주문서 버튼을 쇼핑백·최저가(38px 정사각) 아이콘과 동일 규격으로 맞춰 3개 정렬 통일 */
[class*="st-key-rankrow_"] [data-testid="stPopover"] button {
  width: 38px; height: 38px; min-height: 38px; padding: 0; margin: 0 auto;
  border: 1px solid var(--glass-brd); background: var(--glass); color: var(--muted);
  border-radius: 12px; font-size: 16px;
  display: flex; align-items: center; justify-content: center; line-height: 1;
}
[class*="st-key-rankrow_"] [data-testid="stPopover"] button:hover {
  color: var(--accent); border-color: var(--accent); background: var(--accent-dim); }
/* 팝오버 버튼의 기본 펼침 화살표(chevron)는 숨겨 아이콘만 깔끔하게 보이게 */
[class*="st-key-rankrow_"] [data-testid="stPopover"] button svg { display: none; }
[class*="st-key-rankrow_"] [data-testid="stPopover"] button p { margin: 0; }
/* 팝오버 안 3개월 사용 내역 한 줄 */
.cl-hist { display: flex; gap: 10px; align-items: baseline; padding: 7px 2px;
  border-bottom: 1px solid var(--glass-brd); font-size: 13px; }
.cl-hist__date { font-family: 'Space Grotesk', monospace; font-size: 11px; color: var(--accent);
  flex-shrink: 0; min-width: 78px; }
.cl-hist__name { color: var(--text); }

/* ---- 전국 피부 지도 (귀여운 슬라임 마커 + 숫자) ---- */
.cl-map { position: relative; width: 100%; aspect-ratio: 1020 / 810; max-height: 460px;
  border-radius: 24px; overflow: hidden; border: 1px solid var(--glass-brd);
  margin: 4px 0 6px;
  background-color: #0c131a;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 100% 100%; }
/* 지도 이미지가 없을 때 쓰는 은은한 그라디언트 폴백 */
.cl-map--nomap {
  background-image:
    radial-gradient(140px 140px at 38% 24%, rgba(67, 211, 176, 0.20), transparent 70%),
    radial-gradient(180px 180px at 66% 66%, rgba(94, 234, 212, 0.14), transparent 70%),
    radial-gradient(120px 120px at 30% 88%, rgba(67, 211, 176, 0.12), transparent 70%),
    linear-gradient(160deg, #0f1822, #0b1016); }
.cl-map__pin { position: absolute; transform: translate(-50%, -50%); text-align: center;
  filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.4)); }
.cl-map__slime { display: block; margin: 0 auto; image-rendering: pixelated; }
.cl-map__dot { border-radius: 50%; margin: 0 auto;
  background: linear-gradient(115deg, var(--accent-2), var(--accent)); }
.cl-map__cnt { font-family: 'Space Grotesk', monospace; font-weight: 700; font-size: 13px;
  color: var(--accent); }
.cl-map__label { font-size: 11px; font-weight: 700; color: var(--text); margin-top: 2px; }
.cl-map__pin--top .cl-map__cnt { color: var(--accent-2); font-size: 15px; }
.cl-map__pin--top .cl-map__label { color: var(--accent-2); }

/* ---- 성남시 순위 달리기 대결 트랙 ---- */
.cl-race { position: relative; border-radius: 20px; padding: 14px 12px 10px; margin: 4px 0 4px;
  overflow: hidden; border: 1px solid var(--glass-brd);
  background: linear-gradient(160deg, #0f1822, #0b1016); }
.cl-race__lane { position: relative; height: 58px; margin-bottom: 6px;
  border-bottom: 2px dashed rgba(255,255,255,0.10); }
.cl-race__lane:last-of-type { border-bottom: 0; }
.cl-race__runner { position: absolute; bottom: 2px; transform: translateX(-50%);
  text-align: center; transition: left 0.6s ease; z-index: 2; }
.cl-race__runner img { width: 42px; height: auto; image-rendering: pixelated; display: block;
  margin: 0 auto; animation: cl-race-run 0.5s ease-in-out infinite; }
.cl-race__runner.is-me img { filter: drop-shadow(0 0 10px var(--accent)) !important; }
.cl-race__name { font-size: 10px; font-weight: 700; color: var(--muted); margin-top: 1px;
  white-space: nowrap; }
.cl-race__runner.is-me .cl-race__name { color: var(--accent); }
/* 결승선 + 깃발 (오른쪽) */
.cl-race__finish { position: absolute; right: 30px; top: 8px; bottom: 8px;
  border-left: 3px dashed var(--accent-2); z-index: 1; }
.cl-race__flag { position: absolute; right: 8px; top: 6px; font-size: 22px; z-index: 3; }
@keyframes cl-race-run { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }

/* AI 피부 분석 중(st.spinner) 로딩 원형 아이콘을 민트색으로 */
.stApp [data-testid="stSpinner"] .ewh6kot0 {
  border-color: rgba(67, 211, 176, 0.22) !important;
  border-top-color: var(--accent) !important;
}

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

/* ---- 스플래시 (로딩/로그인) - max 캐릭터 둥둥 + 픽셀 버블 ----
   전체 화면이 아니라 베젤(430px, 가운데 정렬) 안쪽에만 그려서
   로딩/로그인 중에도 폰 화면처럼 보이는 민트 베젤이 그대로 보이게 한다. */
.cl-splash { position: fixed; top: 0; bottom: 0;
  left: max(0px, calc(50% - 215px)); right: max(0px, calc(50% - 215px));
  z-index: 2147483000; overflow: hidden;
  border: 4px solid var(--accent); border-radius: 40px;
  box-shadow: 0 0 0 1px rgba(67, 211, 176, 0.35) inset, 0 0 26px 4px rgba(67, 211, 176, 0.45);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px;
  background: radial-gradient(700px 500px at 50% 35%, rgba(67,211,176,0.16), transparent 60%),
    linear-gradient(160deg, #0b1016, #0f1822); }
.cl-splash__max { width: 128px; height: auto; image-rendering: pixelated; z-index: 2;
  filter: drop-shadow(0 18px 34px rgba(67,211,176,0.4));
  animation: cl-float 2.4s ease-in-out infinite; }
.cl-splash__msg { z-index: 2; color: var(--text); font-weight: 700; font-size: 15px;
  letter-spacing: -0.2px; text-align: center; max-width: 82vw; padding: 0 16px;
  line-height: 1.45; word-break: keep-all; }
.cl-splash__msg::after { content: ""; }
.cl-splash__bubbles { position: absolute; inset: 0; z-index: 1; pointer-events: none; }
.cl-bubble { position: absolute; bottom: -40px; border-radius: 2px;
  animation-name: cl-bubble-rise; animation-timing-function: linear;
  animation-iteration-count: infinite; }
@keyframes cl-float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-24px); } }
@keyframes cl-bubble-rise {
  0% { transform: translateY(0) scale(0.8); opacity: 0; }
  12% { opacity: 0.9; }
  100% { transform: translateY(-108vh) scale(1.25); opacity: 0; } }

/* ---- 지목 매치 (격투 스타일 대결) ---- */
.cl-match { text-align: center; padding: 8px 0 4px; }
.cl-match__arena { position: relative; display: flex; align-items: center; justify-content: center;
  gap: 10px; min-height: 168px; margin: 10px 0 16px; }
.cl-match__char { width: 108px; height: 108px; object-fit: contain; image-rendering: pixelated; }
.cl-match__emoji { width: 108px; height: 108px; display: flex; align-items: center;
  justify-content: center; font-size: 62px; }
.cl-match__char--left { animation: cl-match-shake-l 0.5s ease-in-out infinite; }
.cl-match__char--right { animation: cl-match-shake-r 0.5s ease-in-out infinite; transform: scaleX(-1); }
.cl-match__char--lose { opacity: 0.35; filter: grayscale(0.4) !important; animation: none !important; }
@keyframes cl-match-shake-l { 0%, 100% { transform: translateX(0); } 50% { transform: translateX(6px); } }
@keyframes cl-match-shake-r {
  0%, 100% { transform: translateX(0) scaleX(-1); } 50% { transform: translateX(-6px) scaleX(-1); } }
.cl-match__vs { font-family: 'Space Grotesk', monospace; font-weight: 800; font-size: 20px;
  color: var(--muted); flex-shrink: 0; }
.cl-match__label { font-size: 12px; color: var(--muted); margin-top: 2px; font-weight: 600; }

/* Ready!! / Fight!! 큐 - 캐릭터와 겹치지 않게 아레나 '위'에 별도 줄로 띄운다.
   (예전엔 아레나 위에 겹쳐 떠서 캐릭터와 글자가 겹쳐 보이는 문제가 있었다.) */
.cl-match__cue { text-align: center; margin: 2px 0 4px; min-height: 46px; }
.cl-match__cue span { display: inline-block; font-family: 'Space Grotesk', monospace;
  font-weight: 800; font-size: 46px; letter-spacing: 2px; opacity: 0; transform: scale(0.4);
  background: linear-gradient(115deg, var(--accent-2), var(--accent));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 22px rgba(67,211,176,0.5));
  animation: cl-match-pop 0.5s cubic-bezier(.2,1.4,.4,1) 1 forwards; }
@keyframes cl-match-pop {
  0% { opacity: 0; transform: scale(0.3); }
  60% { opacity: 1; transform: scale(1.18); }
  100% { opacity: 1; transform: scale(1); } }

/* 닉네임 입력 폼 - Ready!! 배너 뒤에 이어서 페이드인 */
.st-key-match_nickform { opacity: 0; animation: cl-match-fadein 0.5s ease 1.9s 1 forwards; }
@keyframes cl-match-fadein {
  from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

/* 대결 결과 */
.cl-match__score { font-family: 'Space Grotesk', monospace; font-size: 40px; font-weight: 800;
  letter-spacing: -1px; margin: 4px 0 16px; }
.cl-match__score .cl-match__win-num { color: var(--accent); }
.cl-match__score .cl-match__lose-num { color: var(--muted); }
.cl-match__winner-name { font-size: 21px; font-weight: 800; margin: 6px 0 2px; }
.cl-match__badge { display: inline-block; background: linear-gradient(115deg, var(--accent-2), var(--accent));
  color: var(--ink); font-weight: 800; font-size: 12.5px; letter-spacing: 1px; padding: 5px 16px;
  border-radius: 999px; margin-top: 6px; box-shadow: 0 8px 24px rgba(67,211,176,0.35); }
/* 패배 배지 - 내가 졌을 때(상대 승) */
.cl-match__badge--lose { background: rgba(255,90,106,0.16); color: #ff7b88;
  box-shadow: none; border: 1px solid rgba(255,90,106,0.4); }
.cl-match__draw { font-size: 16px; font-weight: 700; margin: 18px 0 6px; color: var(--muted); }

/* 다이얼로그(개인정보 동의 등)가 베젤(430px) 폭을 넘지 않게 살짝 작게 맞춘다.
   role="dialog"가 실제 팝업 패널이라 여기에 폭을 직접 건다. */
div[role="dialog"] {
  width: min(384px, calc(100vw - 52px)) !important;
  max-width: min(384px, calc(100vw - 52px)) !important; }

/* ---- 모바일 대응 ---- */
@media (max-width: 480px) {
  /* 동의 팝업 등 다이얼로그 버튼이 좁은 화면에서 줄바꿈되지 않게 */
  [data-testid="stDialog"] .stButton > button, [role="dialog"] .stButton > button {
    white-space: nowrap; font-size: 13px; padding: 9px 6px; }
  [data-testid="stDialog"] [data-testid="stCheckbox"] label,
  [role="dialog"] [data-testid="stCheckbox"] label { font-size: 14px; font-weight: 700; }
  .cl-match__char, .cl-match__emoji { width: 84px; height: 84px; font-size: 50px; }
  .cl-match__arena { min-height: 132px; gap: 6px; }
  .cl-match__cue span { font-size: 36px; }
  .cl-match__cue { min-height: 40px; }
  .cl-match__score { font-size: 32px; }
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem; }
  .cl-logo { width: 120px; height: 120px; }
  .cl-hero__title { font-size: 30px; letter-spacing: -1px; margin: 18px 0 12px; }
  .cl-hero__sub { font-size: 14px; }
  .cl-result__score { font-size: 52px; }
  .cl-countdown__dday { font-size: 42px; }
  .cl-gift { gap: 11px; padding: 12px; }
  .cl-gift__img { width: 76px; height: 76px; }
  .cl-gift__title { font-size: 14.5px; }
  .cl-gift__desc { font-size: 12px; }
  [class*="st-key-navbtn_"] .stButton > button p:nth-of-type(2) { font-size: 19px; }
  .st-key-chatwidget { right: 14px; bottom: 82px; }
  .st-key-chat_fab { width: 52px; }
  .st-key-chat_fab .stButton > button { width: 52px; height: 52px; font-size: 22px; }
  .cl-fab-over, .cl-fab-over img { width: 52px; height: 52px; }
  .cl-chat-body { max-height: 42vh; }
  .cl-faceid { font-size: 12.5px; padding: 11px 13px; }
  .cl-prank { padding: 12px 13px; gap: 10px; }
  .cl-prank__name { font-size: 13px; }
  .cl-note { font-size: 12.5px; }
  /* 커뮤니티 글 - 모바일 축소 */
  .cl-post { padding: 12px 13px; }
  .cl-post__title { font-size: 14px; }
  .cl-post__body { font-size: 12.5px; }
  .cl-poll__label, .cl-poll__pct { font-size: 12px; }
  /* 좁은 화면에선 구매 아이콘 버튼을 조금 작게·간격 좁게 해서 이름이 잘리지 않도록 */
  .cl-shop-group { gap: 6px; }
  .cl-shop-btn, .cl-price-btn { width: 34px; height: 34px; border-radius: 10px; }
  .cl-shop-btn svg, .cl-price-btn svg { width: 15px; height: 15px; }
  .cl-rank { gap: 9px; padding: 12px 12px; }
  .st-key-chatcard { width: min(340px, calc(100vw - 24px)); }
  .cl-map { max-height: 360px; }
  .cl-map__cnt { font-size: 12px; }
  .cl-map__label { font-size: 10px; }
  /* 달리기 대결 트랙 - 모바일 축소 */
  .cl-race__lane { height: 50px; }
  .cl-race__runner img { width: 36px; }
  .cl-race__name { font-size: 9px; }
  .cl-splash__max { width: 104px; }
  .cl-splash__msg { font-size: 14px; }
  /* 랭킹 줄 + 닉네임/타입 칩 - 좁은 화면에서 잘리지 않게 */
  .cl-rank__name { font-size: 13px; }
  .cl-rank__product { font-size: 11.5px; }
  .cl-rank__num { width: 20px; }
  .cl-agechip, .cl-typechip { font-size: 9px; padding: 1px 6px; margin-left: 4px; }
  /* 주문서 팝오버 열 - 간격/버튼 축소 */
  [class*="st-key-rankrow_"] [data-testid="stHorizontalBlock"] { gap: 3px; }
  [class*="st-key-rankrow_"] [data-testid="stPopover"] button {
    width: 34px; height: 34px; min-height: 34px; border-radius: 10px; font-size: 15px; padding: 0; }
  .cl-hist__date { min-width: 66px; font-size: 10.5px; }
  .cl-hist { font-size: 12px; }
  /* 개선 상승폭 랭킹 / 많이 쓰는 랭킹 바 텍스트 */
  .cl-prank__cat { font-size: 9.5px; }
  .cl-prank__meta { font-size: 11px; }
  /* 챗봇 헤더 아바타 */
  .cl-chat-head__ava { width: 26px; height: 26px; }
  .cl-chat-head { padding: 13px 15px; }
  .cl-msg { font-size: 12.5px; max-width: 88%; }
  /* 하단 내비게이션 - 아이콘/글자 축소로 5칸이 확실히 한 줄에 */
  .st-key-bottomnav { padding: 6px 6px calc(6px + env(safe-area-inset-bottom, 0px)); }
  .st-key-bottomnav [data-testid="stHorizontalBlock"] { gap: 0px; }
  .st-key-bottomnav .stButton > button { padding: 5px 1px; }
  .st-key-bottomnav .stButton > button p:first-child { font-size: 18px; }
  .st-key-bottomnav .stButton > button p:last-child { font-size: 9.5px; }
  /* 상단 MY/로그아웃 버튼 - 좁은 화면에서 잘리지 않게 */
  .st-key-mybtn .stButton > button, .st-key-logout .stButton > button {
    font-size: 11px; padding: 8px 3px; }
  /* 마이페이지 개인정보 카드 */
  .cl-info { padding: 4px 13px; }
  .cl-info-row { font-size: 13px; padding: 10px 0; }
  /* 랭킹 탭 - 여러 탭(최대 5개)이 좁은 폭에도 최대한 들어가게 작게, 넘치면 가로 스크롤 */
  .stApp [data-baseweb="tab-list"] { gap: 2px; overflow-x: auto; }
  .stApp [data-baseweb="tab-list"] button[role="tab"] { padding: 8px 6px; min-width: 0; }
  .stApp [data-baseweb="tab"] { font-size: 11.5px; }
  .stApp [data-baseweb="tab"] p { font-size: 11.5px; margin: 0; white-space: nowrap; }
}

/* 아주 좁은 화면(구형 폰) 대응 - 하단 내비/랭킹 아이콘을 더 축소해 한 줄 유지 */
@media (max-width: 360px) {
  .st-key-bottomnav .stButton > button p:first-child { font-size: 17px; }
  .st-key-bottomnav .stButton > button p:last-child { font-size: 9px; letter-spacing: -0.4px; }
  .cl-agechip, .cl-typechip { font-size: 8.5px; padding: 1px 5px; margin-left: 3px; }
  /* 랭킹 3개 아이콘이 좁은 폭에서도 한 박스 안 한 줄에 들어가게 축소 */
  .cl-rank__name { font-size: 12px; }
  .cl-rank__product { font-size: 10.5px; }
  .cl-rank__num { width: 16px; }
  [class*="st-key-rankrow_"] [data-testid="stHorizontalBlock"] { gap: 1px; }
  [class*="st-key-rankrow_"] { padding: 3px 4px; }
  [class*="st-key-rankrow_"] .cl-shop-btn, [class*="st-key-rankrow_"] .cl-price-btn {
    width: 30px; height: 30px; }
  [class*="st-key-rankrow_"] .cl-shop-btn svg, [class*="st-key-rankrow_"] .cl-price-btn svg {
    width: 14px; height: 14px; }
  [class*="st-key-rankrow_"] [data-testid="stColumn"]:nth-child(2) .cl-shop-group { gap: 3px; }
  [class*="st-key-rankrow_"] [data-testid="stPopover"] button {
    width: 30px; height: 30px; min-height: 30px; font-size: 14px; }
}

/* 폰 화면처럼 보이는 민트색 베젤 - .stApp 자체는 손대지 않고(스크롤/레이아웃 안전),
   본문(430px, 가운데 정렬)과 같은 폭·위치로 계산한 장식용 오버레이만 그린다.
   좁은(모바일) 화면에서는 자동으로 화면 전체 폭 베젤이 된다. */
.stApp::after {
  content: "";
  position: fixed; top: 0; bottom: 0;
  left: max(0px, calc(50% - 215px)); right: max(0px, calc(50% - 215px));
  border: 4px solid var(--accent);
  border-radius: 40px;
  box-shadow: 0 0 0 1px rgba(67, 211, 176, 0.35) inset, 0 0 26px 4px rgba(67, 211, 176, 0.45);
  pointer-events: none;
  z-index: 9999;
}
</style>
"""


# ---------------------------------------------------------------------------
# 화면 렌더링
# ---------------------------------------------------------------------------
def _logout() -> None:
    # 로그인/로그아웃은 페이지 새로고침 없이 처리되므로(세션 유지 목적),
    # 진단 관련 상태도 여기서 같이 지워야 다음 로그인 사용자에게 남지 않는다.
    for k in ("logged_in", "consent", "location_consent", "pending_login",
              "my_record_id", "diag_stage", "diag_answers", "last_diagnosis",
              "reward_points", "last_reward_earned", "login_loading",
              "agree_privacy", "agree_location", "signup_stage",
              "signup_agree_privacy", "signup_agree_location",
              "user_age_group", "user_neighborhood",
              "match_stage", "match_opponent", "match_last_reward", "gift_claimed"):
        st.session_state.pop(k, None)
    _reset_survey_answers()


@st.dialog("서비스 이용 동의")
def consent_dialog() -> None:
    """회원가입/비회원 시작 시 개인정보 + 위치기반 서비스 이용 동의 팝업 (모두 필수)."""
    st.markdown("clozkin 서비스 이용을 위해 아래 <b>필수 항목</b>에 동의해주세요.",
                unsafe_allow_html=True)

    agree_privacy = st.checkbox("(필수) 개인정보 활용 동의", key="agree_privacy")
    st.caption("이름·나이 등 프로필, 피부 진단 결과, 구매 내역을 맞춤 추천에 활용해요.")

    agree_location = st.checkbox("(필수) 위치기반 서비스 이용 동의", key="agree_location")
    st.caption("우리 동네 피부 랭킹·주변 지역 정보 제공을 위해 위치 정보를 활용해요.")

    if st.session_state.get("consent") is False:
        st.warning("필수 항목(개인정보·위치기반)에 모두 동의해야 서비스를 이용할 수 있어요.")

    c1, c2 = st.columns(2)
    if c1.button("동의하고 시작", type="primary", use_container_width=True):
        if agree_privacy and agree_location:
            st.session_state.logged_in = True
            st.session_state.consent = True
            st.session_state.location_consent = True
            st.session_state.login_loading = True  # 2초 max 로딩 (최상위 스플래시가 팝업을 덮음)
            st.session_state.pop("pending_login", None)
            st.rerun()
        else:
            st.warning("개인정보·위치기반 서비스 이용에 모두 동의해주세요.")
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
            '<p class="cl-badge-tag">SKINCARE, (WO)MANDATORY FOR MEN</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cl-brand"><span class="cl-brand__dot"></span>'
            '<span class="cl-brand__name">clozkin</span></div>'
            '<p class="cl-badge-tag">SKINCARE, (WO)MANDATORY FOR MEN</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<h1 class="cl-hero__title">남자의 피부,<br>'
        '<span class="cl-grad">이제 관리는 필수.</span></h1>',
        unsafe_allow_html=True,
    )

    # 회원가입을 누르면 로그인 폼 대신 가입 정보 입력 화면을 보여준다.
    if st.session_state.get("signup_stage"):
        render_signup()
        return

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
            st.warning("개인정보·위치기반 서비스에 동의해야 서비스를 이용할 수 있어요.")

    if login:
        # 기존 회원 로그인은 이미 동의한 것으로 간주하고 팝업 없이 바로 입장
        st.session_state.user_name = (user_id or "").strip() or "게스트"
        st.session_state.logged_in = True
        st.session_state.consent = True
        st.session_state.location_consent = True
        st.session_state.login_loading = True  # 로그인 시 2초 max 캐릭터 로딩
        st.session_state.pop("pending_login", None)
        st.rerun()
    elif signup:
        # 회원가입은 닉네임·비밀번호·나이대·사는동네를 입력받는 화면으로 이동
        st.session_state.signup_stage = True
        st.session_state.pop("consent", None)
        st.rerun()
    elif guest:
        st.session_state.user_name = (user_id or "").strip() or "게스트"
        st.session_state.pending_login = True
        st.session_state.pop("consent", None)

    # 비회원 시작 시에만 개인정보 동의 팝업 표시 (로그인은 팝업 없이 바로 입장)
    if st.session_state.get("pending_login"):
        consent_dialog()


def render_signup() -> None:
    """회원가입 화면 - 닉네임·비밀번호·나이대·사는동네 입력 + 필수 동의 후 바로 로그인."""
    _dong_names = [d["name"] for d in MOCK_DONGS]
    st.markdown('<div class="cl-sec">SIGN UP</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">회원가입</div>', unsafe_allow_html=True)
    st.caption("몇 가지만 입력하면 바로 시작할 수 있어요. 맞춤 추천에 사용돼요.")

    with st.container(key="signupbox"):
        with st.form(key="signup_form"):
            nickname = st.text_input("닉네임", placeholder="닉네임", key="signup_nickname")
            password = st.text_input("비밀번호", type="password",
                                     placeholder="비밀번호", key="signup_pw")
            age_group = st.selectbox("나이대", AGE_GROUPS, index=1, key="signup_age_group")
            neighborhood = st.selectbox("사는 동네", _dong_names, index=0,
                                        key="signup_neighborhood")
            st.markdown("---")
            agree_privacy = st.checkbox("(필수) 개인정보 활용 동의", key="signup_agree_privacy")
            agree_location = st.checkbox("(필수) 위치기반 서비스 이용 동의",
                                         key="signup_agree_location")
            submitted = st.form_submit_button("가입하고 시작하기", type="primary",
                                              use_container_width=True)
        back = st.button("← 뒤로", key="signup_back", use_container_width=True)

    if back:
        st.session_state.pop("signup_stage", None)
        st.rerun()

    if submitted:
        if not (nickname or "").strip():
            st.warning("닉네임을 입력해주세요.")
        elif not (password or "").strip():
            st.warning("비밀번호를 입력해주세요.")
        elif not (agree_privacy and agree_location):
            st.warning("개인정보·위치기반 서비스 이용에 모두 동의해야 가입할 수 있어요.")
        else:
            st.session_state.user_name = nickname.strip()
            st.session_state.user_age_group = age_group
            st.session_state.user_neighborhood = neighborhood
            st.session_state.logged_in = True
            st.session_state.consent = True
            st.session_state.location_consent = True
            st.session_state.login_loading = True  # 가입 직후 2초 max 캐릭터 로딩
            st.session_state.pop("signup_stage", None)
            st.session_state.pop("pending_login", None)
            st.rerun()


def section_title(title: str, tag: str) -> None:
    st.markdown(f'<div class="cl-sec">{tag}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cl-h">{title}</div>', unsafe_allow_html=True)


def render_header() -> None:
    """상단 브랜드 바(로고=홈 이동, 페이지 새로고침 없이 st.button으로 처리) + 로그아웃 (모든 화면 공통)."""
    top_bar = st.container(key="topbar")
    top_l, top_m, top_r = top_bar.columns([3, 1, 1.35])
    with top_l:
        with st.container(key="logohome"):
            uri = logo_data_uri()
            if uri:
                # 버튼을 로고 '이미지'로만 보이게 (글자 숨김) + 클릭 시 홈 이동
                st.markdown(
                    "<style>.st-key-logohome .stButton>button{"
                    f"background:url('{uri}') left center/contain no-repeat !important;"
                    "height:46px;width:100%;border:0!important;box-shadow:none!important;"
                    "background-color:transparent!important;color:transparent!important;"
                    "font-size:0!important;padding:0!important;}"
                    ".st-key-logohome .stButton>button:hover{filter:brightness(1.12);}"
                    "</style>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<style>.st-key-logohome .stButton>button{border:0!important;"
                    "background:transparent!important;color:var(--text)!important;"
                    "font-size:19px!important;font-weight:800;box-shadow:none!important;}</style>",
                    unsafe_allow_html=True,
                )
            st.button("clozkin", key="btn_logo_home", on_click=set_nav, args=("home",))
    with top_m:
        with st.container(key="mybtn"):
            st.button("MY", key="btn_my", on_click=set_nav, args=("my",),
                      use_container_width=True)
    with top_r:
        with st.container(key="logout"):
            st.button("로그아웃", key="btn_logout", on_click=_logout,
                      use_container_width=True)


def _catbar_colors(score: int) -> tuple[str, str]:
    """카테고리 점수대에 맞는 바 그라디언트 색상 (낮을수록 앰버, 높을수록 민트)."""
    if score >= 80:
        return "#5eead4", "#43d3b0"
    if score >= 60:
        return "#a8e8d6", "#5eead4"
    return "#ffd98a", "#f4c15e"


def render_category_bars(categories: dict) -> None:
    """5개 세부 카테고리 점수를 가로 바(progress bar)로 보여준다."""
    for key in _CATEGORY_KEYS:
        score = categories.get(key, 0)
        c1, c2 = _catbar_colors(score)
        st.markdown(
            f'<div class="cl-catbar"><div class="cl-catbar__top">'
            f'<span>{_CATEGORY_LABELS[key]}</span><span>{score}점</span></div>'
            f'<div class="cl-catbar__track"><span class="cl-catbar__fill" '
            f'style="width:{score}%;background:linear-gradient(90deg,{c1},{c2})"></span></div></div>',
            unsafe_allow_html=True,
        )


def _reset_survey_answers() -> None:
    """설문 라디오 위젯의 session_state를 지운다 - 그러지 않으면 새 진단을 시작해도
    이전 회차에 고른 보기가 그대로 선택된 채로 남아있는다."""
    for q in SURVEY_QUESTIONS:
        st.session_state.pop(f"survey_{q['key']}", None)
    st.session_state.pop("diag_answers", None)


def render_photo_stage() -> None:
    """진단 1단계 - 얼굴 사진을 촬영하거나 업로드한다 (실제 이미지 분석은 하지 않음)."""
    st.markdown('<div class="cl-sec">STEP 1 · PHOTO</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">얼굴 사진을 준비해주세요</div>', unsafe_allow_html=True)
    st.caption("촬영하거나 갤러리에서 업로드해주세요. 다음 설문과 함께 분석에 활용돼요.")

    st.number_input("나이", min_value=10, max_value=90, value=25, step=1, key="diag_age")
    mode = st.radio("사진 입력 방식", ["촬영하기", "업로드하기"], horizontal=True,
                    label_visibility="collapsed", key="diag_photo_mode")

    photo = None
    if mode == "촬영하기":
        st.caption("📸 얼굴이 잘 보이도록 **정면**에서 촬영해주세요.")
        st.caption("👤 화면의 사람 모양 점선 가이드에 얼굴과 어깨를 맞춰주세요.")
        photo = st.camera_input("피부 촬영", label_visibility="collapsed", key="diag_camera")
    else:
        photo = st.file_uploader("얼굴 사진 업로드", type=["png", "jpg", "jpeg"],
                                 label_visibility="collapsed", key="diag_upload")
        if photo is not None:
            st.image(photo, caption="업로드한 사진", width=220)

    if photo is not None:
        if st.button("다음: 설문 진행하기 →", type="primary", use_container_width=True,
                     key="diag_photo_next"):
            st.session_state.diag_stage = "survey"
            st.rerun()


def render_survey_stage() -> None:
    """진단 2단계 - 피부 설문 7문항. 각 문항은 시작 시 아무 것도 선택돼 있지 않다
    (index=None) - 사용자가 직접 고르기 전까지는 체크된 보기가 없어야 하기 때문."""
    st.markdown('<div class="cl-sec">STEP 2 · SURVEY</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">피부 설문 7문항</div>', unsafe_allow_html=True)
    st.caption("모든 문항에 답해주세요. 솔직하게 답할수록 더 정확한 맞춤 결과가 나와요.")

    with st.form(key="survey_form"):
        answers = {}
        for i, q in enumerate(SURVEY_QUESTIONS, start=1):
            st.markdown(f"**Q{i}. {q['text']}**")
            answers[q["key"]] = st.radio(
                q["text"], [o["label"] for o in q["options"]],
                index=None, label_visibility="collapsed", key=f"survey_{q['key']}",
            )
        submitted = st.form_submit_button(
            "AI 분석 시작하기", type="primary", use_container_width=True)

    if submitted:
        unanswered = sum(1 for v in answers.values() if v is None)
        if unanswered:
            st.error(f"아직 답하지 않은 문항이 {unanswered}개 있어요. 모든 문항에 답해주세요.")
        else:
            st.session_state.diag_answers = answers
            st.session_state.diag_stage = "loading"
            st.rerun()

    if st.button("← 사진 다시 선택하기", key="diag_back_photo"):
        st.session_state.diag_stage = "photo"
        st.rerun()


def render_loading_stage() -> None:
    """진단 3단계 - 'AI 분석 중' 로딩 후 결과를 생성하고 기록·랭킹에 반영한다."""
    render_analysis_loading()

    answers = st.session_state.get("diag_answers", {})
    result = simulate_ai_diagnosis(answers)
    st.session_state.last_diagnosis = result

    # 진단 기록을 사이트 전체에 누적 저장 (랭킹에 반영)
    rec_id = time.time_ns()
    recs = recommend_products(result)
    nickname = (st.session_state.get("user_name") or "").strip() or "익명"
    age = int(st.session_state.get("diag_age", 25))
    save_record({
        "id": rec_id,
        "name": nickname,
        # 지목 매치에서 마스킹 없이 정확히 검색하기 위한 별도 필드 (name과 같은 값을
        # 저장하지만, 추후 name이 마스킹되더라도 매치 검색은 영향받지 않게 분리해둔다).
        "nickname": nickname,
        "age_group": f"{(age // 10) * 10}대",
        "skin_type": result.get("skin_type", "-"),
        "type_label": result.get("type_label", "-"),
        "score": result["score"],
        "gain": result.get("gain", 8),
        "product": recs[0]["name"] if recs else "-",
    })
    st.session_state.my_record_id = rec_id

    # 리워드 포인트 적립 (진단 완료 1회당 참여점수 + 피부점수 + 개선폭 보너스)
    earned = points_for_diagnosis(result)
    st.session_state.reward_points = st.session_state.get("reward_points", 0) + earned
    st.session_state.last_reward_earned = earned

    st.session_state.diag_stage = "result"
    st.rerun()


def render_result_stage() -> None:
    """진단 4단계 - 결과 화면: 최종 점수 -> 타입 라벨 -> 한줄요약 -> 카테고리 바 ->
    추천 제품 -> 랭킹 바로가기 -> 28일 후 재촬영 유도 CTA."""
    result = st.session_state.get("last_diagnosis")
    if not result:
        st.session_state.diag_stage = "photo"
        st.rerun()
        return

    st.markdown('<div class="cl-sec">RESULT</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">AI 피부 분석 결과</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cl-result">'
        f'<p class="cl-result__label">SKIN SCORE</p>'
        f'<p class="cl-result__score">{result.get("score", "-")}</p>'
        f'<p class="cl-result__type">피부 타입 · {result.get("type_label", result.get("skin_type", "-"))}</p>'
        f'<p class="cl-result__summary">{result.get("summary", "")}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    earned = st.session_state.get("last_reward_earned")
    if earned:
        st.markdown(
            f'<div class="cl-note">🎁 진단 완료로 <b>+{earned}P</b> 적립됐어요! '
            f'현재 <b>{st.session_state.get("reward_points", 0):,}P</b> · '
            f'하단 <b>리워드</b> 탭에서 내 티어를 확인해보세요.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="cl-sec">ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">카테고리별 분석</div>', unsafe_allow_html=True)
    render_category_bars(result.get("categories", {}))

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

    st.markdown('<div class="cl-sec">NEXT</div>', unsafe_allow_html=True)
    st.button("🏆 우리 동네 랭킹에서 내 순위 보기", type="primary", use_container_width=True,
              on_click=set_nav, args=("ranking",), key="result_goto_rank")

    target_date = (datetime.date.today() + datetime.timedelta(days=28)).isoformat()
    st.markdown(
        f'<div class="cl-note">📅 피부 턴오버 주기는 약 28일이에요. '
        f'<b>{target_date}</b> 즈음 다시 촬영하면 얼마나 좋아졌는지 확인할 수 있어요!</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔁 새로운 진단 시작하기", use_container_width=True, key="result_restart"):
        st.session_state.diag_stage = "photo"
        st.session_state.pop("last_diagnosis", None)
        _reset_survey_answers()
        st.rerun()


def render_skin_map() -> None:
    """피부 좋은 남자들의 분포를 2개 탭으로 보여준다 - 전국 지도 / 우리 동네(성남·판교) 동별."""
    st.markdown('<div class="cl-sec">SKIN MAP</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">피부 좋은 남자들, 어디 많을까?</div>', unsafe_allow_html=True)

    tab_nation, tab_dong, tab_race = st.tabs(["🗺️ 전국", "🏘️ 우리 동네", "🏃 내 순위"])

    # --- 탭 1) 전국 지도 (기존) ---
    with tab_nation:
        st.caption("지역별 피부 우수자 수예요. 슬라임이 클수록 피부 좋은 남자가 많아요 🫧")
        slime = slime_data_uri()
        map_uri = map_data_uri()
        mx = max(r["count"] for r in MOCK_REGIONS)
        ranked = sorted(MOCK_REGIONS, key=lambda x: x["count"], reverse=True)
        pins = ""
        for i, r in enumerate(ranked):
            w = 30 + round(r["count"] / mx * 26)  # 30~56px (지도 위라 살짝 작게)
            top = " cl-map__pin--top" if i == 0 else ""
            icon = (f'<img class="cl-map__slime" src="{slime}" style="width:{w}px">'
                    if slime else
                    f'<div class="cl-map__dot" style="width:{w}px;height:{w}px"></div>')
            pins += (
                f'<div class="cl-map__pin{top}" style="left:{r["x"]}%;top:{r["y"]}%">'
                f'<div class="cl-map__cnt">{r["count"]}</div>'
                f'{icon}'
                f'<div class="cl-map__label">{r["name"]}</div></div>'
            )
        # 지도 이미지가 있으면 배경으로 깔고, 없으면 그라디언트 폴백(--nomap)을 쓴다.
        if map_uri:
            style = f' style="background-image:url({map_uri})"'
            cls = "cl-map"
        else:
            style = ""
            cls = "cl-map cl-map--nomap"
        st.markdown(f'<div class="{cls}"{style}>{pins}</div>', unsafe_allow_html=True)

    # --- 탭 2) 우리 동네(성남시 분당·판교) 동별 순위 ---
    with tab_dong:
        st.caption("우리 동네(성남시 분당·판교) 중 어떤 동에 피부 좋은 남자가 많을까요? 🏘️")
        dongs = sorted(MOCK_DONGS, key=lambda x: x["count"], reverse=True)
        top_cnt = dongs[0]["count"]
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for rank, d in enumerate(dongs, start=1):
            pct = round(d["count"] / top_cnt * 100)
            num = medals.get(rank, str(rank))
            st.markdown(
                f'<div class="cl-prank">'
                f'<div class="cl-rank__num">{num}</div>'
                f'<div class="cl-prank__body">'
                f'<div class="cl-prank__top"><span class="cl-prank__name">{d["name"]}</span>'
                f'<span class="cl-prank__cat">성남시</span></div>'
                f'<div class="cl-prank__bar"><span style="width:{pct}%"></span></div>'
                f'<div class="cl-prank__meta">피부 우수자 {d["count"]}명</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- 탭 3) 성남시에서 내 순위 (max 달리기 대결) ---
    with tab_race:
        diag = st.session_state.get("last_diagnosis")
        if not diag:
            st.info("먼저 피부 진단을 받으면 성남시에서 내 순위를 달리기 대결로 보여드려요! 🏃")
            st.button("진단하러 가기", type="primary", use_container_width=True,
                      on_click=set_nav, args=("diagnose",), key="race_goto_diag")
        else:
            my_score = int(diag.get("score", 0))
            seongnam_total = sum(d["count"] for d in MOCK_DONGS)  # 성남시 참여자(목업)
            # 점수가 높을수록 앞 순위. 결정적으로 계산.
            my_rank = min(seongnam_total,
                          max(1, round((100 - my_score) / 100 * seongnam_total) + 1))
            st.caption("성남시 피부왕들과 달리기 대결! 점수가 높을수록 결승선에 가까워요 🏃")
            st.markdown(
                f'<div class="cl-status-wrap"><div class="cl-status">'
                f'🏃 성남시 <b>{my_rank}등</b> / {seongnam_total:,}명</div></div>',
                unsafe_allow_html=True,
            )
            slime = slime_data_uri(120)
            # 라이벌은 내 점수 기준 앞뒤로 배치해 '대결' 느낌을 준다.
            runners = [
                ("🥊 판교 물광남", min(99, my_score + 7), False, 150),
                (f"나 ({my_rank}등)", my_score, True, 0),
                ("정자동 꿀피부", max(38, my_score - 6), False, 60),
            ]
            lanes = ""
            for name, score, me, hue in runners:
                prog = 8 + round(max(0, min(100, score)) / 100 * 72)  # 8~80%
                hue_style = f"filter:hue-rotate({hue}deg);" if hue else ""
                icon = (f'<img src="{slime}" style="{hue_style}">' if slime
                        else '<div style="font-size:30px">🫧</div>')
                lanes += (
                    f'<div class="cl-race__lane">'
                    f'<div class="cl-race__runner{" is-me" if me else ""}" style="left:{prog}%">'
                    f'{icon}<div class="cl-race__name">{name}</div></div></div>'
                )
            st.markdown(
                f'<div class="cl-race">{lanes}'
                f'<div class="cl-race__finish"></div>'
                f'<div class="cl-race__flag">🏁</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"내 점수 {my_score}점 · 28일 뒤 다시 진단하면 순위가 쭉쭉 올라가요!")


def _person_row(rank: int, entry: dict, value_html: str, key_prefix: str = "") -> None:
    """랭킹 한 줄 렌더 - 순위/이름/제품/점수 + 구매·최저가·주문서 아이콘까지
    '하나의 박스' 안에 3개 컬럼(정보 / 쇼핑아이콘 / 주문서)으로 가지런히 정렬한다."""
    stype = entry.get("skin_type")
    type_chip = f'<span class="cl-typechip">{stype}</span>' if stype else ""
    # 카드(정보) - 구매 버튼은 별도 컬럼으로 빼서 아이콘들을 한 줄로 정렬한다.
    card_html = (
        f'<div class="cl-rank {"is-me" if entry.get("is_me") else ""}">'
        f'<div class="cl-rank__num">{rank}</div>'
        f'<div class="cl-rank__body">'
        f'<div class="cl-rank__name">{entry["name"]}'
        f'<span class="cl-agechip">{entry.get("age_group", "-")}</span>'
        f'{type_chip}</div>'
        f'<div class="cl-rank__product">{entry["product"]}</div></div>'
        f'{value_html}'
        f'</div>'
    )
    with st.container(key=f"rankrow_{key_prefix}_{rank}"):
        col_card, col_buy, col_more = st.columns([5, 2, 1])
        col_card.markdown(card_html, unsafe_allow_html=True)
        col_buy.markdown(buy_buttons(entry["product"]), unsafe_allow_html=True)
        with col_more.popover("🧾", help="최근 3개월 사용 화장품"):
            st.markdown(f"**{entry['name']}** 님의 최근 3개월 사용 화장품 🧾")
            st.caption("근 3개월간 실제로 구매·사용한 제품 내역이에요.")
            for h in person_history(entry["name"]):
                st.markdown(
                    f'<div class="cl-hist"><span class="cl-hist__date">{h["date"]}</span>'
                    f'<span class="cl-hist__name">{h["product"]}</span></div>',
                    unsafe_allow_html=True,
                )


def build_ranking_board() -> list[dict]:
    """목업 + 누적된 실제 진단 기록을 합쳐 랭킹 보드를 만든다 (랭킹/리워드 화면 공용).
    기록의 이름은 모두 닉네임으로, product는 실제 제품명으로 정규화해서 노출한다."""
    records = load_records()
    my_id = st.session_state.get("my_record_id")
    board = [dict(x) for x in MOCK_RANKING]
    # 이미 쓰인 닉네임(목업 포함)을 추적해 중복 없이 배정한다.
    used = {x["name"] for x in MOCK_RANKING}
    # id 순(안정적)으로 배정 → 새 기록이 추가돼도 기존 닉네임이 바뀌지 않는다.
    records_sorted = sorted(records, key=lambda r: r.get("id", 0))
    for idx, r in enumerate(records_sorted):
        orig_name = r.get("nickname") or r.get("name") or "익명"
        nick = unique_nick(idx, used)
        used.add(nick)
        is_me = r.get("id") == my_id
        # 나이대: 기록에 없으면(옛 데이터) 결정적 랜덤으로 채우고,
        # 내 기록은 마이페이지에서 고른 나이대(있으면)를 우선 반영한다.
        age_group = r.get("age_group")
        if not age_group or age_group == "-":
            age_group = random_age_group(str(r.get("id", orig_name)))
        if is_me and st.session_state.get("my_age_group"):
            age_group = st.session_state["my_age_group"]
        board.append({
            # 실명 대신 항상 '중복되지 않는' 닉네임으로 노출 (내 기록은 '(나)' 표시로 구분)
            "name": f"{nick} (나)" if is_me else nick,
            "age_group": age_group,
            "skin_type": r.get("skin_type"),
            "score": int(r.get("score", 0)),
            "gain": int(r.get("gain", 0)),
            "product": real_product_for(orig_name, r.get("product")),
            "is_me": is_me,
        })
    return board


def my_score_rank(board: list[dict]) -> tuple[int | None, int]:
    """전체 피부 점수 순위 기준 내 순위(1부터)와 전체 인원수. 내 기록이 없으면 (None, 총원)."""
    ranked = sorted(board, key=lambda x: x["score"], reverse=True)
    rank = next((i for i, e in enumerate(ranked, start=1) if e.get("is_me")), None)
    return rank, len(ranked)


def gain_badge_html(gain: int) -> str:
    """턴오버 상승/하락 배지 HTML. 양수는 민트 ▲, 음수는 붉은 ▼로 표기한다."""
    g = int(gain or 0)
    if g < 0:
        return f'<div class="cl-rank__score cl-rank__gain cl-rank__gain--down">▼{abs(g)}</div>'
    return f'<div class="cl-rank__score cl-rank__gain">▲{g}</div>'


def weekly_gain(entry: dict) -> int:
    """'이번 주 급상승'용 주간 상승폭(결정적). 28일 상승폭(gain)에서 최근 7일 몫만
    이름 해시로 안정적으로 뽑아, 새로고침해도 같은 값이 나오게 한다."""
    g = int(entry.get("gain", 0))
    if g <= 0:
        return 0
    seed = sum(ord(c) for c in entry.get("name", "")) or 1
    return max(1, round(g * 0.5) + (seed % 3))


def render_ranking() -> None:
    render_header()
    section_title("우리 동네 피부 랭킹", "RANKING")
    render_skin_map()

    board = build_ranking_board()

    # 참여자 수 - '오늘' 참여 인원(세션 동안 고정) + 누적 인원(데모용 고정 수치)
    today_count = st.session_state.setdefault("today_count", random.randint(32, 68))
    st.markdown(
        f'<div class="cl-status-wrap"><div class="cl-status">'
        f'🔥 <b>오늘 {today_count}명</b>이 피부 진단에 참여했어요 · '
        f'누적 31,992명</div></div>',
        unsafe_allow_html=True,
    )

    # 내 현재 순위 + 상위 몇 % (전체 기준, 나이대 필터와 무관하게 항상 보여준다)
    my_rank, total = my_score_rank(board)
    if my_rank:
        top_pct = max(1, round(my_rank / total * 100))  # 최소 1%로 표기
        st.markdown(
            f'<div class="cl-status-wrap"><div class="cl-status">'
            f'<b>내 순위 {my_rank}위</b> / {total}명 중 · '
            f'<b>상위 {top_pct}%</b> 🎯</div></div>',
            unsafe_allow_html=True,
        )

    # 여러 랭킹을 스크롤이 아니라 탭으로 바로바로 볼 수 있게 한다.
    tab_people, tab_weekly, tab_used, tab_type, tab_pick = st.tabs(
        ["🏆 피부 점수", "🔥 주간 급상승", "🧴 인기템", "💧 타입별", "🚀 개선 픽"])

    # === 탭 1) 피부 점수 랭킹 (나이대 필터 + 점수순/턴오버순 하위 탭) ===
    with tab_people:
        # 나이 구분: 전체 / 10~20대 / 30~40대 / 50대 이상
        groups = ["전체", "10~20대", "30~40대", "50대 이상"]
        _group_bands = {
            "10~20대": ("10대", "20대"),
            "30~40대": ("30대", "40대"),
            "50대 이상": ("50대", "60대", "70대", "80대", "90대"),
        }
        picked = st.radio("나이대", groups, horizontal=True, label_visibility="collapsed")

        def _in_group(e: dict) -> bool:
            if picked == "전체":
                return True
            return e.get("age_group") in _group_bands.get(picked, ())

        view = [e for e in board if _in_group(e)]
        sub_score, sub_gain = st.tabs(["🏆 점수순", "📈 턴오버순"])

        with sub_score:
            st.caption("현재 피부 점수가 높은 순이에요.")
            if not view:
                st.caption("이 나이대에는 아직 기록이 없어요.")
            score_sorted = sorted(view, key=lambda x: x["score"], reverse=True)
            show_all_score = st.session_state.get("rank_show_all_score", False)
            visible_count = len(score_sorted) if show_all_score else 10
            for rank, entry in enumerate(score_sorted[:visible_count], start=1):
                _person_row(rank, entry,
                            f'<div class="cl-rank__score">{entry["score"]}</div>',
                            key_prefix="score")
            # 10명 초과일 때만 토글 버튼 노출 (더보기 <-> 접기)
            if len(score_sorted) > 10:
                if show_all_score:
                    if st.button("접기 ▲", key="btn_rank_score_less", use_container_width=True):
                        st.session_state.rank_show_all_score = False
                        st.rerun()
                else:
                    if st.button(f"더보기 ▼ (+{len(score_sorted) - 10})",
                                 key="btn_rank_score_more", use_container_width=True):
                        st.session_state.rank_show_all_score = True
                        st.rerun()

        with sub_gain:
            st.caption("피부 턴오버 28일 동안의 점수 변화 순이에요. (▲ 상승 · ▼ 하락)")
            if not view:
                st.caption("이 나이대에는 아직 기록이 없어요.")
            for rank, entry in enumerate(
                    sorted(view, key=lambda x: x.get("gain", 0), reverse=True), start=1):
                _person_row(rank, entry, gain_badge_html(entry.get("gain", 0)),
                            key_prefix="gain")

    # === 탭 2) 주간 급상승 (최근 7일 상승폭) ===
    with tab_weekly:
        st.caption("최근 7일간 피부 점수가 가장 많이 오른 유저예요. 🔥")
        weekly = [e for e in board if weekly_gain(e) > 0]
        weekly.sort(key=weekly_gain, reverse=True)
        weekly = weekly[:10]
        if not weekly:
            st.caption("아직 이번 주 상승 기록이 없어요.")
        for rank, entry in enumerate(weekly, start=1):
            _person_row(rank, entry,
                        f'<div class="cl-rank__score cl-rank__gain">▲{weekly_gain(entry)}</div>',
                        key_prefix="weekly")

    # === 탭 3) 많이 쓰는 화장품 랭킹 ===
    with tab_used:
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

    # === 탭 4) 피부 타입별 인기템 ===
    with tab_type:
        st.caption("같은 피부 타입 남자들이 많이 쓰는 제품이에요. 내 타입을 골라보세요 💧")
        types = list(MOCK_TYPE_PRODUCTS.keys())  # 지성/건성/복합성/민감성/수부지
        my_type = (st.session_state.get("last_diagnosis") or {}).get("skin_type")
        default_idx = types.index(my_type) if my_type in types else 0
        picked_type = st.radio("피부 타입", types, index=default_idx,
                               horizontal=True, label_visibility="collapsed")
        tprods = MOCK_TYPE_PRODUCTS.get(picked_type, [])
        top_u = tprods[0]["users"] if tprods else 1
        for rank, p in enumerate(tprods, start=1):
            pct = round(p["users"] / top_u * 100)
            st.markdown(
                f'<div class="cl-prank">'
                f'<div class="cl-rank__num">{rank}</div>'
                f'<div class="cl-prank__body">'
                f'<div class="cl-prank__top"><span class="cl-prank__name">{p["name"]}</span>'
                f'<span class="cl-prank__cat">{p["category"]}</span></div>'
                f'<div class="cl-prank__bar"><span style="width:{pct}%"></span></div>'
                f'<div class="cl-prank__meta">{picked_type} {p["users"]:,}명 사용</div></div>'
                f'{buy_buttons(p["name"])}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # === 탭 5) 개선 상승폭 큰 사람들의 픽 ===
    with tab_pick:
        st.caption("피부 점수가 가장 많이 오른 사람들이 즐겨 쓰는 아이템이에요. 🚀")
        improvers = sorted(MOCK_IMPROVER_PRODUCTS, key=lambda x: x["avg_gain"], reverse=True)
        top_gain = improvers[0]["avg_gain"]
        for rank, p in enumerate(improvers, start=1):
            pct = round(p["avg_gain"] / top_gain * 100)
            st.markdown(
                f'<div class="cl-prank">'
                f'<div class="cl-rank__num">{rank}</div>'
                f'<div class="cl-prank__body">'
                f'<div class="cl-prank__top"><span class="cl-prank__name">{p["name"]}</span>'
                f'<span class="cl-prank__cat">{p["category"]}</span></div>'
                f'<div class="cl-prank__bar"><span style="width:{pct}%"></span></div>'
                f'<div class="cl-prank__meta">사용자 평균 <b style="color:var(--accent)">▲{p["avg_gain"]}점</b> 상승</div></div>'
                f'{buy_buttons(p["name"])}'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_diagnosis_screen() -> None:
    render_header()
    stage = st.session_state.get("diag_stage", "photo")
    if stage == "survey":
        render_survey_stage()
    elif stage == "loading":
        render_loading_stage()
        return  # 로딩 직후 st.rerun()으로 넘어가므로 아래 이동 버튼은 그릴 필요 없음
    elif stage == "result":
        render_result_stage()
    else:
        render_photo_stage()

    st.markdown('<div class="cl-sec">MOVE</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.button("🏠 메인 화면으로", key="diag_home", on_click=set_nav, args=("home",),
              use_container_width=True)
    c2.button("🏆 랭킹 보러가기", key="diag_rank", on_click=set_nav, args=("ranking",),
              use_container_width=True)


def _match_arena_html(my_hue: int, opp_hue: int | None, opp_grayscale: bool = False,
                       lose_side: str | None = None, banner_text: str | None = None) -> str:
    """매치 화면의 캐릭터 2체(나/상대) + VS 마크업을 반환.
    lose_side가 'left'/'right'면 그쪽 캐릭터를 패배 스타일(흐리게)로 렌더링한다.
    banner_text가 있으면 (Ready!!/Fight!!) 이 캐릭터 영역 안에만 겹쳐서 띄운다 -
    뷰포트 전체 기준으로 띄우면 실제 페이지 높이와 안 맞아 아래 폼과 겹치기 때문에
    .cl-match__arena(position:relative) 내부에 절대위치로 넣는다."""
    left_cls = "cl-match__char cl-match__char--left"
    right_cls = "cl-match__char cl-match__char--right"
    if lose_side == "left":
        left_cls += " cl-match__char--lose"
    elif lose_side == "right":
        right_cls += " cl-match__char--lose"
    banner_html = (
        f'<div class="cl-match__banner"><span>{banner_text}</span></div>' if banner_text else ""
    )
    return (
        '<div class="cl-match__arena">'
        f'{match_character_html(left_cls, my_hue)}'
        '<div class="cl-match__vs">VS</div>'
        f'{match_character_html(right_cls, opp_hue, grayscale=opp_grayscale)}'
        f'{banner_html}'
        '</div>'
    )


def render_match_screen() -> None:
    """지목 매치 - 닉네임으로 상대를 지목해 피부점수로 대결한다.
    3단계: intro(대기+Ready!!) -> nickname(상대 지목) -> fight(Fight!!) -> result(승패)."""
    render_header()
    section_title("지목 매치", "MATCH")

    stage = st.session_state.get("match_stage", "intro")
    my_diag = st.session_state.get("last_diagnosis")
    my_name = (st.session_state.get("user_name") or "").strip() or "익명"
    my_hue = match_hue_for_nickname(my_name)
    opponent = st.session_state.get("match_opponent")

    if stage not in ("intro", "nickname"):
        # st.container(key=...)는 같은 키로 다시 방문하지 않으면 이전 내용을 그대로
        # 남겨두므로, fight/result 단계에서는 닉네임 입력 폼을 명시적으로 비워야 한다.
        st.container(key="match_nickform").empty()

    if stage in ("intro", "nickname"):
        opp_hue = match_hue_for_nickname(opponent["nickname"]) if opponent else None
        # Ready!! 배너는 최초 진입(intro) 시 한 번만. 캐릭터와 겹치지 않도록
        # 아레나 '위'에 별도 줄(cl-match__cue)로 띄운다.
        cue = ('<div class="cl-match__cue"><span>Ready!!</span></div>'
               if stage == "intro" else "")
        st.markdown(
            f'<div class="cl-match">{cue}'
            f'{_match_arena_html(my_hue, opp_hue, opp_grayscale=opponent is None)}'
            f'<div class="cl-match__label">{my_name} VS '
            f'{opponent["nickname"] if opponent else "???"}</div></div>',
            unsafe_allow_html=True,
        )

        if stage == "intro":
            st.session_state.match_stage = "nickname"

        def _start_match(target: str) -> None:
            """닉네임(직접 입력 또는 랜덤)으로 대결을 시작한다."""
            target = (target or "").strip()
            if not target:
                st.error("상대방 닉네임을 입력해주세요.")
                return
            if not my_diag:
                st.warning("먼저 피부 진단을 받아야 매치를 시작할 수 있어요.")
                st.button("진단하러 가기", type="primary", use_container_width=True,
                          on_click=set_nav, args=("diagnose",), key="match_goto_diag")
                return
            # 매치를 시작하는 순간 리워드 포인트를 사용한다 (0 미만으로는 내려가지 않음).
            st.session_state.reward_points = max(
                0, st.session_state.get("reward_points", 0) - MATCH_ENTRY_COST)
            st.session_state.match_last_reward = 0  # 이번 매치 승리 보상(결과 화면에서 채움)
            # 실제 기록을 찾지 않고, 어떤 닉네임이든 무작위로 승/패가 정해지는 상대를 만든다.
            my_score = int(my_diag.get("score", 0))
            opp_score, _ = random_match_result(my_score)
            st.session_state.match_opponent = {"nickname": target, "score": opp_score}
            st.session_state.match_stage = "fight"
            st.rerun()

        cur_points = st.session_state.get("reward_points", 0)
        st.caption(
            f"⚔️ 대결을 시작하면 리워드 {MATCH_ENTRY_COST}P가 사용돼요 · "
            f"이기면 {MATCH_WIN_REWARD}P 적립! (내 포인트 {cur_points:,}P)")

        with st.container(key="match_nickform"):
            with st.form(key="match_nickname_form"):
                nickname_input = st.text_input(
                    "상대방 닉네임을 입력하세요", placeholder="상대방 닉네임을 입력하세요",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "대결 신청", type="primary", use_container_width=True)
            # 랜덤 선택 - 아무 닉네임이나 골라 랜덤한 사람과 대결
            random_clicked = st.button("🎲 랜덤 상대와 대결하기", use_container_width=True,
                                       key="match_random")

            if submitted:
                _start_match(nickname_input)
            elif random_clicked:
                _start_match(random.choice(RANDOM_MATCH_NICKS))
        return

    if stage == "fight":
        opp_hue = match_hue_for_nickname(opponent.get("nickname", "")) if opponent else None
        st.markdown(
            f'<div class="cl-match"><div class="cl-match__cue"><span>Fight!!</span></div>'
            f'{_match_arena_html(my_hue, opp_hue)}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.9)
        # 승리 시에만 리워드 지급. 이 전환(fight→result)은 매치당 한 번만 실행되므로
        # 결과 화면이 여러 번 다시 그려져도 보상이 중복 적립되지 않는다.
        my_s = int((my_diag or {}).get("score", 0))
        opp_s = int((opponent or {}).get("score", 0))
        if my_s > opp_s:
            st.session_state.reward_points = (
                st.session_state.get("reward_points", 0) + MATCH_WIN_REWARD)
            st.session_state.match_last_reward = MATCH_WIN_REWARD
        else:
            st.session_state.match_last_reward = 0
        st.session_state.match_stage = "result"
        st.rerun()
        return

    # stage == "result"
    opponent = opponent or {}
    my_score = int((my_diag or {}).get("score", 0))
    opp_score = int(opponent.get("score", 0))
    opp_name = opponent.get("nickname", "상대")
    opp_hue = match_hue_for_nickname(opp_name)

    if my_score == opp_score:
        st.markdown(
            f'<div class="cl-match">{_match_arena_html(my_hue, opp_hue)}'
            f'<div class="cl-match__score">{my_score} VS {opp_score}</div>'
            '<div class="cl-match__draw">무승부! 다시 대결해보세요</div></div>',
            unsafe_allow_html=True,
        )
    else:
        i_win = my_score > opp_score
        winner_name = my_name if i_win else opp_name
        lose_side = "right" if i_win else "left"
        win_num, lose_num = (my_score, opp_score) if i_win else (opp_score, my_score)
        score_html = (
            f'<span class="cl-match__win-num">{win_num}</span> VS '
            f'<span class="cl-match__lose-num">{lose_num}</span>'
            if i_win else
            f'<span class="cl-match__lose-num">{lose_num}</span> VS '
            f'<span class="cl-match__win-num">{win_num}</span>'
        )
        # 결과 문구는 '나' 기준: 내가 이기면 WIN!, 상대가 이기면 LOSE.
        if i_win:
            name_html = f'<div class="cl-match__winner-name">🏆 {winner_name}</div>'
            badge_html = '<div class="cl-match__badge">WIN!</div>'
        else:
            name_html = f'<div class="cl-match__winner-name">🏆 {winner_name} 승리</div>'
            badge_html = '<div class="cl-match__badge cl-match__badge--lose">LOSE…</div>'
        st.markdown(
            f'<div class="cl-match">{_match_arena_html(my_hue, opp_hue, lose_side=lose_side)}'
            f'<div class="cl-match__score">{score_html}</div>'
            f'{name_html}{badge_html}</div>',
            unsafe_allow_html=True,
        )

    # 리워드 정산 안내 - 승리 보상(+) / 시작 시 사용된 포인트(-)
    won_reward = st.session_state.get("match_last_reward", 0)
    if won_reward:
        st.success(f"🎁 승리 보상으로 리워드 {won_reward}P를 적립했어요! "
                   f"(현재 {st.session_state.get('reward_points', 0):,}P)")
    else:
        st.caption(f"이번 대결에 리워드 {MATCH_ENTRY_COST}P를 사용했어요. "
                   f"(현재 {st.session_state.get('reward_points', 0):,}P) 다음 판을 노려봐요!")

    if st.button("다시 대결하기", type="primary", use_container_width=True, key="match_retry"):
        st.session_state.match_stage = "intro"
        st.session_state.pop("match_opponent", None)
        st.session_state.pop("match_last_reward", None)
        st.rerun()


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
            render_max_loading("여러 쇼핑몰 계정을 연동하는 중...")
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


def render_rewards_screen() -> None:
    """리워드 탭 - 내 포인트/티어, 티어 시스템, 명예의 전당(마스터 전용).
    포인트는 실제 DB 없이 st.session_state["reward_points"]에만 보관되고
    (진단 완료 시 points_for_diagnosis()로 적립), 랭킹은 build_ranking_board()를
    그대로 재사용한다 - 나중에 실제 계정 시스템을 붙일 때 이 두 지점만 바꾸면 된다."""
    render_header()
    section_title("리워드", "REWARDS")

    points = st.session_state.get("reward_points", 0)
    my_tier = tier_for_points(points)
    tier_idx = REWARD_TIERS.index(my_tier)
    next_tier = REWARD_TIERS[tier_idx + 1] if tier_idx + 1 < len(REWARD_TIERS) else None

    board = build_ranking_board()
    my_rank, total = my_score_rank(board)
    rank_text = (f"우리 동네 랭킹 {my_rank}위 · 상위 {max(1, round(my_rank / total * 100))}%"
                 if my_rank else "아직 피부 진단 기록이 없어요")

    st.markdown(
        f'<div class="cl-result">'
        f'<p class="cl-result__label">MY REWARD POINTS</p>'
        f'<p class="cl-result__score">{points:,}P</p>'
        f'<p class="cl-result__type">{my_tier["icon"]} {my_tier["name"]} 티어</p>'
        f'<p class="cl-result__summary">{rank_text}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if next_tier:
        remain = max(0, next_tier["min_points"] - points)
        st.caption(f"다음 티어 {next_tier['icon']} {next_tier['name']}까지 {remain:,}P 남았어요. "
                   "(피부 진단을 완료할 때마다 포인트가 쌓여요)")
    else:
        st.caption("최고 티어 마스터에 도달했어요! 아래 명예의 전당에서 확인해보세요 👑")

    # --- 티어 시스템 5단계 ---
    st.markdown('<div class="cl-sec">TIER</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">티어 시스템</div>', unsafe_allow_html=True)
    for i, tier in enumerate(REWARD_TIERS):
        nxt = REWARD_TIERS[i + 1] if i + 1 < len(REWARD_TIERS) else None
        pct = tier_progress_pct(points, tier, nxt)
        is_current = tier["key"] == my_tier["key"]
        range_text = (f"{tier['min_points']:,}~{nxt['min_points'] - 1:,}P" if nxt
                      else f"{tier['min_points']:,}P~")
        st.markdown(
            f'<div class="cl-tier-row{" is-current" if is_current else ""}">'
            f'<div class="cl-tier-row__top">'
            f'<span class="cl-tier-row__name">{tier["icon"]} {tier["name"]}</span>'
            f'<span class="cl-tier-row__range">{range_text}</span></div>'
            f'<div class="cl-tier-row__track"><span class="cl-tier-row__fill" '
            f'style="width:{pct}%"></span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- 마스터 달성 보상 (코스맥스 3WAAU 본품 증정) ---
    master_min = REWARD_TIERS[-1]["min_points"]
    st.markdown('<div class="cl-sec">MASTER REWARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">마스터 달성 보상</div>', unsafe_allow_html=True)

    gift_uri = reward_gift_data_uri()
    gift_img = (f'<img class="cl-gift__img" src="{gift_uri}" alt="3WAAU 본품">'
                if gift_uri else '<div class="cl-gift__img cl-gift__img--ph">🧴</div>')

    if points >= master_min:
        st.markdown(
            f'<div class="cl-gift cl-gift--unlocked">{gift_img}'
            f'<div class="cl-gift__body">'
            f'<div class="cl-gift__badge">🎉 마스터 달성</div>'
            f'<div class="cl-gift__title">코스맥스 3WAAU 본품 증정</div>'
            f'<div class="cl-gift__desc">마스터 티어 달성을 축하해요! 코스맥스가 만든 '
            f'<b>3WAAU 맞춤 본품</b>을 드려요.</div></div></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("gift_claimed"):
            st.success("본품 증정 신청이 접수됐어요! 등록된 주소로 배송돼요 🚚")
        elif st.button("🎁 본품 증정 신청하기", type="primary",
                       use_container_width=True, key="gift_claim"):
            st.session_state.gift_claimed = True
            st.balloons()
            st.rerun()
    else:
        remain = master_min - points
        st.markdown(
            f'<div class="cl-gift cl-gift--locked">{gift_img}'
            f'<div class="cl-gift__body">'
            f'<div class="cl-gift__badge">🔒 마스터 전용</div>'
            f'<div class="cl-gift__title">코스맥스 3WAAU 본품 증정</div>'
            f'<div class="cl-gift__desc">마스터 티어({master_min:,}P) 달성 시 '
            f'코스맥스 <b>3WAAU 본품</b>을 드려요. 앞으로 <b>{remain:,}P</b> 남았어요!'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

    # --- 명예의 전당 (마스터 티어 전용) ---
    st.markdown('<div class="cl-sec">HALL OF FAME</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">명예의 전당</div>', unsafe_allow_html=True)

    if points < master_min:
        remain = master_min - points
        st.markdown(
            f'<div class="cl-hof-lock">🔒 마스터 티어({master_min:,}P) 달성 시 '
            f'명예의 전당이 열려요.<br>앞으로 <b>{remain:,}P</b> 더 모으면 도전할 수 있어요!</div>',
            unsafe_allow_html=True,
        )
    else:
        my_name = (st.session_state.get("user_name") or "").strip() or "게스트"
        hof = [dict(u) for u in MOCK_REWARD_USERS if u["points"] >= master_min]
        hof.append({"name": my_name, "points": points, "is_me": True})
        hof.sort(key=lambda x: x["points"], reverse=True)
        trophies = {1: "🥇", 2: "🥈", 3: "🥉"}
        for rank, u in enumerate(hof, start=1):
            trophy = trophies.get(rank, str(rank))
            top_cls = f" cl-hof-row--top{rank}" if rank <= 3 else ""
            me_cls = " is-me" if u.get("is_me") else ""
            st.markdown(
                f'<div class="cl-hof-row{top_cls}{me_cls}">'
                f'<div class="cl-hof-row__rank">{trophy}</div>'
                f'<div class="cl-hof-row__name">{u["name"]}</div>'
                f'<div class="cl-hof-row__points">{u["points"]:,}P</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_my_screen() -> None:
    """마이페이지 - 개인정보 + 리워드 점수 요약 + 구매내역 진입.
    구매 내역은 하단바에서 빠지고 이 화면에서만 들어갈 수 있다."""
    render_header()
    section_title("마이페이지", "MY")

    name = (st.session_state.get("user_name") or "").strip() or "게스트"
    diag = st.session_state.get("last_diagnosis") or {}
    points = st.session_state.get("reward_points", 0)
    tier = tier_for_points(points)

    # 나이대 기본값: 이미 고른 값 > 진단 시 입력한 나이 > 20대
    age = st.session_state.get("diag_age")
    diag_band = f"{(int(age) // 10) * 10}대" if age else None
    if diag_band and diag_band not in AGE_GROUPS:
        diag_band = "60대 이상"
    default_band = st.session_state.get("my_age_group") or diag_band or "20대"
    skin_type = diag.get("type_label") or diag.get("skin_type") or "아직 진단 전"
    score = diag.get("score")
    score_text = f"{score}점" if score is not None else "아직 진단 전"

    # --- 개인정보 ---
    st.markdown('<div class="cl-sec">PROFILE</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">내 정보</div>', unsafe_allow_html=True)
    info = [("닉네임", name), ("피부 타입", skin_type),
            ("최근 피부 점수", score_text), ("리워드 포인트", f"{points:,}P")]
    rows = "".join(
        f'<div class="cl-info-row"><span class="cl-info-row__k">{k}</span>'
        f'<span class="cl-info-row__v">{v}</span></div>' for k, v in info)
    st.markdown(f'<div class="cl-info">{rows}</div>', unsafe_allow_html=True)

    # 나이대 - 드롭다운으로 직접 선택/확인 (랭킹의 내 나이대에도 반영된다)
    if "my_age_group" not in st.session_state:
        st.session_state.my_age_group = default_band
    st.selectbox("나이대", AGE_GROUPS, key="my_age_group",
                 help="내 나이대를 선택하면 랭킹에도 반영돼요.")

    # --- 리워드 점수 요약 ---
    st.markdown('<div class="cl-sec">REWARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">내 리워드</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cl-result"><p class="cl-result__label">MY REWARD POINTS</p>'
        f'<p class="cl-result__score">{points:,}P</p>'
        f'<p class="cl-result__type">{tier["icon"]} {tier["name"]} 티어</p></div>',
        unsafe_allow_html=True,
    )
    st.button("🎁 리워드 · 티어 자세히 보기", use_container_width=True,
              on_click=set_nav, args=("rewards",), key="my_go_rewards")

    # --- 구매 내역 진입 (여기서만 접근 가능) ---
    st.markdown('<div class="cl-sec">PURCHASES</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-h">구매 내역</div>', unsafe_allow_html=True)
    st.caption("내가 구매한 화장품 내역을 한 곳에서 확인할 수 있어요.")
    st.button("🛍️ 구매 내역 보러가기", type="primary", use_container_width=True,
              on_click=set_nav, args=("purchases",), key="my_go_purchases")


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


def scroll_to_top() -> None:
    """다음 화면/단계로 넘어갈 때 페이지를 맨 위로 스크롤한다.
    Streamlit 본문 스크롤 컨테이너(버전에 따라 셀렉터가 다름)를 모두 시도한다."""
    components.html(
        """
        <script>
        const w = window.parent, d = w.document;
        const sels = ['section.main', '[data-testid="stMain"]',
                      '[data-testid="stAppViewContainer"]', '.main', '.block-container'];
        for (const s of sels) { const e = d.querySelector(s); if (e) { e.scrollTop = 0; } }
        if (d.scrollingElement) d.scrollingElement.scrollTop = 0;
        w.scrollTo(0, 0);
        </script>
        """,
        height=0,
    )


def _render_poll_post(post: dict) -> None:
    """소개팅 고민 등 투표 글 - 두 선택지 막대 + 투표 버튼(1인 1표, 세션 저장)."""
    pid = post["id"]
    voted = st.session_state.get(f"poll_{pid}")  # None / "a" / "b"
    va = post["votes_a"] + (1 if voted == "a" else 0)
    vb = post["votes_b"] + (1 if voted == "b" else 0)
    total = max(1, va + vb)
    pa = round(va / total * 100)
    pb = 100 - pa
    a_sel = " is-sel" if voted == "a" else ""
    b_sel = " is-sel" if voted == "b" else ""
    st.markdown(
        f'<div class="cl-post cl-post--poll">'
        f'<div class="cl-post__head"><span class="cl-post__tag cl-post__tag--poll">투표</span>'
        f'<span class="cl-post__author">{html.escape(post["author"])}</span></div>'
        f'<div class="cl-post__title">{html.escape(post["title"])}</div>'
        f'<div class="cl-poll">'
        f'<div class="cl-poll__opt{a_sel}"><div class="cl-poll__bar" style="width:{pa}%"></div>'
        f'<span class="cl-poll__label">{html.escape(post["option_a"])}</span>'
        f'<span class="cl-poll__pct">{pa}%</span></div>'
        f'<div class="cl-poll__opt{b_sel}"><div class="cl-poll__bar" style="width:{pb}%"></div>'
        f'<span class="cl-poll__label">{html.escape(post["option_b"])}</span>'
        f'<span class="cl-poll__pct">{pb}%</span></div>'
        f'</div>'
        f'<div class="cl-post__meta">🗳️ {total}표 · 💬 {post.get("comments", 0)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if voted:
        st.caption(f"내 선택: {post['option_a'] if voted == 'a' else post['option_b']} ✓")
    else:
        c1, c2 = st.columns(2)
        if c1.button(post["option_a"], key=f"pa_{pid}", use_container_width=True):
            st.session_state[f"poll_{pid}"] = "a"
            st.rerun()
        if c2.button(post["option_b"], key=f"pb_{pid}", use_container_width=True):
            st.session_state[f"poll_{pid}"] = "b"
            st.rerun()


def _render_community_post(post: dict) -> None:
    """커뮤니티 글 한 개 렌더 (고민/자랑/투표)."""
    if post.get("type") == "poll":
        _render_poll_post(post)
        return
    tag = post.get("tag", "고민")
    tag_cls = {"고민": "concern", "자랑": "brag"}.get(tag, "concern")
    body = html.escape(post.get("body", "")).replace(chr(10), "<br>")
    st.markdown(
        f'<div class="cl-post cl-post--{tag_cls}">'
        f'<div class="cl-post__head">'
        f'<span class="cl-post__tag cl-post__tag--{tag_cls}">{tag}</span>'
        f'<span class="cl-post__author">{html.escape(post.get("author", "익명"))}</span></div>'
        f'<div class="cl-post__title">{html.escape(post.get("title", ""))}</div>'
        f'<div class="cl-post__body">{body}</div>'
        f'<div class="cl-post__meta">❤️ {post.get("likes", 0)} · 💬 {post.get("comments", 0)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_community_screen() -> None:
    """커뮤니티 - 나이대별(10대/20-30대/40-50대) 게시판.
    피부 고민, 랭킹 자랑, 소개팅 투표 등 여러 유형의 글을 볼 수 있고 직접 글도 쓸 수 있다."""
    render_header()
    section_title("커뮤니티", "COMMUNITY")
    st.caption("나이대별 게시판에서 피부 고민을 나누고, 자랑도 하고, 투표도 해보세요 💬")

    my_nick = (st.session_state.get("user_name") or "").strip() or "익명"
    tabs = st.tabs(COMMUNITY_GROUPS)
    for tab, group in zip(tabs, COMMUNITY_GROUPS):
        with tab:
            # 글쓰기 - 팝오버 폼으로 간단히 고민 글 등록 (세션에만 저장)
            with st.popover("✏️ 글쓰기", use_container_width=True):
                with st.form(key=f"cform_{group}", clear_on_submit=True):
                    ctitle = st.text_input("제목", key=f"ctitle_{group}",
                                           placeholder="예) 지성 피부 수분크림 추천요")
                    cbody = st.text_area("내용", key=f"cbody_{group}",
                                         placeholder="고민 내용을 적어주세요", height=90)
                    if st.form_submit_button("등록", type="primary",
                                             use_container_width=True):
                        if ctitle.strip():
                            posts = st.session_state.setdefault(f"cposts_{group}", [])
                            posts.insert(0, {
                                "id": f"user_{group}_{len(posts)}", "type": "concern",
                                "author": my_nick, "tag": "고민",
                                "title": ctitle.strip(), "body": cbody.strip(),
                                "likes": 0, "comments": 0,
                            })
                            st.rerun()

            user_posts = st.session_state.get(f"cposts_{group}", [])
            if not user_posts and not MOCK_COMMUNITY.get(group):
                st.caption("아직 글이 없어요. 첫 글을 남겨보세요!")
            for post in user_posts + MOCK_COMMUNITY.get(group, []):
                _render_community_post(post)


def render_bottom_nav(active: str) -> None:
    """하단 고정 탭 내비게이션. 홈을 가운데에 두고 좌우로 배치한다 (구매내역은 MY에서만).
    아이콘 위 + 작은 글자 아래로 쌓아, 웹·모바일 모두 화면 폭 안에 5칸이 들어가게 한다."""
    items = [("ranking", "🏆", "랭킹"), ("match", "⚔️", "매치"),
             ("home", "🏠", "홈"), ("diagnose", "📷", "진단"),
             ("community", "💬", "커뮤니티")]
    with st.container(key="bottomnav"):
        cols = st.columns(len(items))
        for col, (key, icon, label) in zip(cols, items):
            # 라벨을 "아이콘\n\n글자"로 넘겨 버튼 안에서 세로로 쌓이게 한다.
            col.button(f"{icon}\n\n{label}", key=f"nav_{key}", on_click=set_nav,
                       args=(key,), use_container_width=True,
                       type="primary" if active == key else "secondary")


# 빠른 질문 추천 칩 (버튼은 2개만 노출. 나머지 주제도 직접 입력하면 max가 답한다.)
QUICK_QUESTIONS = [
    "토너·세럼 순서 알려줘",
    "여드름 자국 없애는 법",
]

# max가 특정 질문에 정해진 대로 답하는 스크립트 답변 (키워드가 모두 포함되면 매칭)
SCRIPTED_ANSWERS = [
    {
        "keysets": [["토너", "세럼", "순서"], ["토너", "세럼", "먼저"]],
        "answer": (
            "순서는 세안 → 토너 → 세럼 → 크림이에요.\n"
            "토너는 피부결 정리, 세럼은 피부 고민 집중 케어예요.\n"
            "아무것도 모르겠으면 '토너 다음 세럼'만 기억하면 돼요! 😉\n\n"
            "🛍️ 추천 제품\n"
            "• 라로슈포제 에빠끌라 토너\n"
            "• 파티온 노스카나인 트러블 세럼"
        ),
    },
    {
        "keysets": [["지성", "뭐부터"], ["지성", "먼저"], ["지성", "순서"], ["지성", "무엇부터"]],
        "answer": (
            "지성 피부는 가볍고 산뜻한 제품부터 바르면 돼요.\n"
            "보통 토너 → 세럼 → 가벼운 수분크림 순서가 잘 맞아요.\n"
            "번들거려도 가벼운 보습은 꼭 해주는 게 좋아요! 💧\n\n"
            "🛍️ 추천 제품\n"
            "• 라로슈포제 에빠끌라 토너\n"
            "• 에스트라 에이시카365 수분 진정 크림"
        ),
    },
    {
        "keysets": [["여드름", "자국"], ["여드름자국"], ["여드름", "흉터"]],
        "answer": (
            "여드름 자국은 진정 관리 + 자외선 차단 + 꾸준함이 중요해요.\n"
            "붉은 자국은 진정, 갈색 자국은 미백 기능성 세럼이 도움 돼요.\n"
            "손으로 만지거나 짜면 더 오래가니 최대한 안 건드리는 게 좋아요!\n\n"
            "🛍️ 추천 제품\n"
            "• 파티온 노스카나인 트러블 세럼\n"
            "• 닥터자르트 시카페어 토너"
        ),
    },
    {
        "keysets": [["면접"]],
        "answer": (
            "면접 전날은 새 제품 말고, 순하고 간단하게 관리하는 게 좋아요.\n"
            "각질 제거, 압출, 강한 팩은 피하는 게 안전해요.\n"
            "진정 토너 + 가벼운 보습크림 정도만 해도 깔끔해 보여요! ✨\n\n"
            "🛍️ 추천 제품\n"
            "• 닥터자르트 시카페어 토너\n"
            "• 에스트라 에이시카365 수분 진정 크림"
        ),
    },
    {
        "keysets": [["결혼식"], ["상견례"], ["중요한날"], ["중요한", "날"]],
        "answer": (
            "중요한 날 전에는 '좋아지게' 하기보다 '뒤집히지 않게' 관리하는 게 핵심이에요.\n"
            "1~2주 전부터 세안, 보습, 선크림만 꾸준히 해도 충분히 깔끔해 보여요.\n"
            "당일엔 번들거림 적은 보습제와 선크림만 써도 인상이 정돈돼 보여요!\n\n"
            "🛍️ 추천 제품\n"
            "• 닥터자르트 시카페어 토너\n"
            "• 에스트라 에이시카365 수분 진정 크림"
        ),
    },
    {
        # 스킨·토너·에센스·로션 이름 차이가 헷갈릴 때
        "keysets": [
            ["스킨", "토너"], ["토너", "에센스"], ["에센스", "로션"],
            ["스킨", "에센스"], ["스킨", "로션"], ["토너", "로션"],
            ["스킨", "차이"], ["토너", "차이"], ["에센스", "차이"], ["로션", "차이"],
        ],
        "answer": (
            "스킨, 토너, 에센스, 로션 이름이 헷갈려도 괜찮습니다.\n"
            "초보자는 이렇게만 이해하면 됩니다.\n\n"
            "• 토너/스킨: 세안 후 가볍게 정리\n"
            "• 에센스/세럼: 기능성 집중\n"
            "• 로션/크림: 보습\n"
            "• 선크림: 자외선 차단\n\n"
            "이름이 회사마다 달라서 복잡해 보이지만, 실제로는\n"
            "닦고 → 보충하고 → 막아주는 구조라고 보면 쉽습니다."
        ),
    },
    {
        # 여자(여성) 화장품 써도 되는지
        "keysets": [["여자", "화장품"], ["여성", "화장품"], ["여자화장품"], ["여성화장품"]],
        "answer": "네. 피부에 맞으면 문제 없습니다.",
    },
    {
        # 피부 관리 팁
        "keysets": [
            ["피부", "관리", "팁"], ["피부관리", "팁"], ["피부", "팁"],
            ["관리", "팁"], ["피부관리", "방법"], ["피부", "관리", "방법"],
        ],
        "answer": (
            "좋은 피부관리 = 세안 + 보습 + 선크림\n"
            "• 남성용 여부보다 피부 타입이 중요\n"
            "• 비싼 것보다 안 자극적이고 꾸준히 쓸 수 있는 것이 중요\n"
            "• 처음부터 여러 개 사지 말고 2~3개로 시작"
        ),
    },
    {
        # PDRN 성분 관련
        "keysets": [["pdrn"], ["피디알엔"]],
        "answer": (
            "PDRN은 피부 회복과 진정을 내세우는 성분으로, 쉽게 말해 DNA 조각 계열 원료입니다.\n"
            "주로 연어 유래 성분으로 알려져 있고, 화장품이나 피부 시술 분야에서 자주 언급됩니다.\n"
            "다만 바르는 화장품에서의 효과는 제한적일 수 있어서, 만능 재생 성분처럼 기대하는 건 "
            "과장일 수 있습니다."
        ),
    },
]


def scripted_reply(text: str) -> str | None:
    """사용자 질문이 정해진 스크립트 질문과 맞으면 max의 고정 답변을 반환, 아니면 None."""
    norm = re.sub(r"[\s·,.?!]", "", text).lower()
    for item in SCRIPTED_ANSWERS:
        for keyset in item["keysets"]:
            if all(re.sub(r"[\s·,.?!]", "", k).lower() in norm for k in keyset):
                return item["answer"]
    return None


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

    # 1) 정해진 스크립트 질문이면 max가 고정 답변 (API 없이도 동작)
    scripted = scripted_reply(text)
    if scripted is not None:
        st.session_state.chat_messages.append({"role": "assistant", "content": scripted})
        return

    if client is None:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "앗, 지금은 max가 잠깐 쉬는 중이에요 🫧 "
                       "위의 빠른 질문 버튼을 눌러보면 바로 답해드릴 수 있어요!"})
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
             "content": "안녕하세요! 저는 clozkin 대표 캐릭터 max예요 🫧 "
                        "뷰티 입문, 저 max한테 뭐든 편하게 물어보세요!"}
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
                head_uri = slime_data_uri(72)
                avatar = (f'<img class="cl-chat-head__ava" src="{head_uri}" alt="max">'
                          if head_uri else '<span class="cl-chat-head__dot"></span>')
                st.markdown(
                    '<div class="cl-chat-panel">'
                    f'<div class="cl-chat-head">{avatar}'
                    '<span class="cl-chat-head__name">max</span>'
                    '<span class="cl-chat-head__sub">clozkin</span></div>'
                    f'<div class="cl-chat-body">{bubbles}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                # --- D-day 케어 모드 (챗봇 기능) ---
                with st.container(key="chat_dday"):
                    # 직전에 '루틴 만들기'를 눌렀으면 이번 렌더에서 토글을 접는다(1회성).
                    dday_kwargs = {}
                    if st.session_state.pop("dday_collapse", False):
                        dday_kwargs["expanded"] = False
                    with st.expander("📅 D-day 케어 모드", **dday_kwargs):
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
                        render_max_loading("max가 맞춤 케어 루틴을 짜는 중...")
                        msg = build_dday_message(
                            client, EVENT_LABELS.get(ev, ev), int(days))
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": msg})
                        st.session_state.dday_collapse = True  # 다음 렌더에서 토글 닫기
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

        # --- FAB 토글 버튼 (닫힌 상태에선 대표 캐릭터 max 이미지가 버튼 위에 얹힌다) ---
        with st.container(key="chat_fab"):
            max_uri = None if chat_open else question_slime_data_uri(240)
            # 버튼을 먼저 그려 컨테이너 맨 위(top:0)에 오게 하고, 그 위에 불투명 max 이미지를
            # 정확히 겹친다(이미지가 버튼을 완전히 덮으므로 버튼 배경 스타일은 필요 없다).
            st.button("✕" if chat_open else ("" if max_uri else "💬"), key="btn_chat_fab",
                      on_click=toggle_chat, help="max에게 물어보기 (클릭하면 챗봇이 열려요)")
            if max_uri:
                # div로 감싼 오버레이(div class는 이 앱에서 안정적으로 적용됨)로 버튼 위에 겹친다
                st.markdown(f'<div class="cl-fab-over"><img src="{max_uri}" alt="max"></div>',
                            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
def main() -> None:
    page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "◎"
    st.set_page_config(page_title="clozkin", page_icon=page_icon, layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    api_key = get_api_key()
    client = get_client(api_key) if api_key else None

    # 앱 최초 로딩 시 5초 스플래시 (max 캐릭터 둥둥 + 픽셀 버블)
    if not st.session_state.get("splashed"):
        render_max_loading("clozkin 불러오는 중...")
        time.sleep(5)
        st.session_state.splashed = True
        st.rerun()

    # 가짜 로그인 게이트 - 로그인 전에는 로그인 화면만 보여준다.
    if not st.session_state.get("logged_in"):
        render_login()
        return

    # 로그인 직후 2초 max 캐릭터 로딩 (스플래시가 최상위라 동의 팝업까지 덮으며 닫힌다)
    if st.session_state.get("login_loading"):
        render_max_loading("로그인 중...")
        time.sleep(2)
        st.session_state.login_loading = False
        st.rerun()

    # 로고 이미지 클릭 등 URL(?nav=) 로 들어온 화면 전환 반영
    if "nav" in st.query_params:
        st.session_state.nav = st.query_params["nav"]
        del st.query_params["nav"]

    # 화면/단계가 바뀌면 페이지 최상단으로 스크롤 (다음 페이지로 넘어갈 때 위로).
    # 실제 스크롤 스크립트는 main 끝에서 주입한다 (최상단에 빈 iframe 여백이 생기지 않도록).
    nav = st.session_state.get("nav", "home")
    view_sig = (f"{nav}|{st.session_state.get('diag_stage', '')}"
                f"|{st.session_state.get('match_stage', '')}")
    need_scroll = st.session_state.get("_view_sig") != view_sig
    st.session_state._view_sig = view_sig

    # 하단 탭으로 화면 전환 (홈 / 랭킹 / 진단 / 매치 / 커뮤니티), 구매·리워드는 MY에서만
    if nav == "ranking":
        render_ranking()
    elif nav == "diagnose":
        render_diagnosis_screen()
    elif nav == "match":
        render_match_screen()
    elif nav == "community":
        render_community_screen()
    elif nav == "my":
        render_my_screen()
    elif nav == "purchases":
        render_purchases_screen()
    elif nav == "rewards":
        render_rewards_screen()
    else:
        render_home_screen()

    render_bottom_nav(nav)
    render_chat_widget(client)

    if need_scroll:
        scroll_to_top()


if __name__ == "__main__":
    main()
