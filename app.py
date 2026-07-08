"""
clozkin - 뷰티 입문 남성을 위한 AI 스킨케어 가이드 MVP
단일 파일 버전: app.py 하나에 Flask 서버 + HTML/CSS/JS(프론트엔드)를 모두 포함.

실행 방법:
    pip install -r requirements.txt
    cp .env.example .env   # (없다면 직접 만들어 ANTHROPIC_API_KEY=... 입력)
    python app.py
    -> http://localhost:5000
"""
import os
import re
import json
from datetime import datetime, date

from flask import Flask, request, jsonify, session, Response
from dotenv import load_dotenv
import anthropic

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "clozkin-dev-secret-change-me")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-5"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

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


def _extract_json(text: str) -> dict:
    """모델 응답에서 JSON 블록만 안전하게 추출."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# 프론트엔드 (HTML + CSS + JS 전부 포함)
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>clozkin</title>
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root {
  --base: #2f3338;
  --base-deep: #24272b;
  --surface: #383d43;
  --accent: #43d3b0;
  --accent-dim: rgba(67, 211, 176, 0.16);
  --text: #f5f7f8;
  --text-muted: #9aa1a8;
  --radius-lg: 24px;
  --radius-md: 16px;
  --radius-sm: 10px;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  background: var(--base);
  color: var(--text);
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
  overscroll-behavior: none;
}

button { font-family: inherit; }
.is-hidden { display: none !important; }

.screen {
  display: none;
  flex-direction: column;
  min-height: 100vh;
  max-width: 480px;
  margin: 0 auto;
  padding: 24px 20px 40px;
  position: relative;
}
.screen.is-active { display: flex; }

.screen-splash {
  align-items: center;
  justify-content: center;
  gap: 28px;
  background: var(--base-deep);
}
.wordmark { font-size: 34px; font-weight: 700; letter-spacing: -0.5px; color: var(--text); margin: 0; }
.wordmark-sm { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }

.dot-loader { display: flex; gap: 6px; }
.dot-loader span {
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  opacity: 0.35; animation: dot-pulse 1.1s infinite ease-in-out;
}
.dot-loader span:nth-child(2) { animation-delay: 0.15s; }
.dot-loader span:nth-child(3) { animation-delay: 0.3s; }
.dot-loader--dark span { background: var(--accent); }
@keyframes dot-pulse {
  0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
  40% { opacity: 1; transform: scale(1.15); }
}

.topbar { padding: 6px 0 20px; }
.status-banner { background: var(--surface); border-radius: var(--radius-lg); padding: 22px; margin-bottom: 28px; }
.status-banner__label { margin: 0 0 6px; color: var(--accent); font-size: 13px; font-weight: 600; letter-spacing: 0.2px; }
.status-banner__text { margin: 0; font-size: 17px; font-weight: 600; }

.menu-list { display: flex; flex-direction: column; gap: 14px; }
.menu-card {
  display: flex; align-items: center; gap: 16px; background: var(--surface); border: none;
  border-radius: var(--radius-lg); padding: 22px; color: var(--text); text-align: left; cursor: pointer;
  transition: transform 0.15s ease, background 0.15s ease;
}
.menu-card:active { transform: scale(0.98); background: #40454b; }
.menu-card__icon {
  font-size: 20px; color: var(--accent); width: 36px; height: 36px; display: flex;
  align-items: center; justify-content: center; background: var(--accent-dim); border-radius: 50%; flex-shrink: 0;
}
.menu-card__body { display: flex; flex-direction: column; gap: 4px; }
.menu-card__title { font-size: 16px; font-weight: 700; }
.menu-card__desc { font-size: 13px; color: var(--text-muted); }

.subbar { display: flex; align-items: center; gap: 12px; padding: 4px 0 24px; }
.subbar__title { font-size: 17px; font-weight: 700; }
.btn-back { background: var(--surface); border: none; color: var(--text); width: 34px; height: 34px; border-radius: 50%; font-size: 18px; cursor: pointer; }

.scan-stage { position: relative; width: 260px; height: 260px; margin: 12px auto 20px; border-radius: 50%; overflow: hidden; background: var(--base-deep); }
.scan-stage video, .scan-stage img { width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }
.scan-ring { position: absolute; inset: -2px; border-radius: 50%; border: 2px solid var(--accent); box-shadow: 0 0 0 4px rgba(67, 211, 176, 0.12); pointer-events: none; }
.scan-ring.is-scanning { animation: ring-pulse 1.4s infinite ease-in-out; }
@keyframes ring-pulse {
  0% { box-shadow: 0 0 0 4px rgba(67, 211, 176, 0.12); }
  50% { box-shadow: 0 0 0 12px rgba(67, 211, 176, 0.02); }
  100% { box-shadow: 0 0 0 4px rgba(67, 211, 176, 0.12); }
}
.scan-hint { text-align: center; color: var(--text-muted); font-size: 14px; margin: 0 0 24px; }
.scan-actions { display: flex; flex-direction: column; align-items: center; gap: 12px; }
.analyzing-text { color: var(--text-muted); font-size: 14px; margin: 0; }

.btn-primary { width: 100%; background: var(--accent); color: var(--base-deep); border: none; border-radius: var(--radius-md); padding: 16px; font-size: 15px; font-weight: 700; cursor: pointer; }
.btn-primary:active { opacity: 0.85; }
.btn-secondary { width: 100%; background: transparent; color: var(--text); border: 1px solid var(--surface); border-radius: var(--radius-md); padding: 15px; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 10px; }

.result-card { background: var(--surface); border-radius: var(--radius-lg); padding: 24px; text-align: center; margin-top: 8px; }
.result-score-label { margin: 0; font-size: 13px; color: var(--text-muted); }
.result-score { margin: 4px 0 2px; font-size: 48px; font-weight: 800; color: var(--accent); }
.result-type { margin: 0 0 10px; font-size: 14px; font-weight: 600; color: var(--text-muted); }
.result-summary { margin: 0 0 16px; font-size: 15px; line-height: 1.5; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 10px; }
.chip-row span { background: var(--base-deep); color: var(--text); font-size: 12px; padding: 6px 12px; border-radius: 999px; }
.chip-row--accent span { background: var(--accent-dim); color: var(--accent); }

.error-banner { background: rgba(255, 99, 99, 0.12); color: #ff8a8a; border-radius: var(--radius-md); padding: 14px 16px; font-size: 13px; margin-top: 14px; text-align: center; }

.ranking-note { color: var(--text-muted); font-size: 13px; margin: 0 0 18px; }
.ranking-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.ranking-item { display: flex; align-items: center; gap: 14px; background: var(--surface); border-radius: var(--radius-md); padding: 14px 16px; }
.ranking-item.is-me { background: var(--accent-dim); border: 1px solid var(--accent); }
.ranking-item__rank { width: 28px; font-weight: 800; color: var(--accent); font-size: 14px; text-align: center; }
.ranking-item__body { flex: 1; display: flex; flex-direction: column; gap: 2px; }
.ranking-item__name { font-size: 14px; font-weight: 700; }
.ranking-item__product { font-size: 12px; color: var(--text-muted); }
.ranking-item__score { font-size: 15px; font-weight: 800; }
.ranking-item__link { font-size: 11px; color: var(--accent); text-decoration: none; white-space: nowrap; }

.section-label { font-size: 13px; color: var(--text-muted); margin: 18px 0 10px; }
.event-chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.event-chip { background: var(--surface); border: 1px solid transparent; color: var(--text); border-radius: 999px; padding: 10px 16px; font-size: 13px; cursor: pointer; }
.event-chip.is-selected { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); font-weight: 700; }
.date-input { width: 100%; background: var(--surface); border: none; border-radius: var(--radius-md); padding: 14px; color: var(--text); font-size: 15px; margin-bottom: 20px; color-scheme: dark; }

.countdown-banner { background: var(--surface); border-radius: var(--radius-lg); padding: 24px; text-align: center; margin-bottom: 16px; }
.countdown-banner__dday { margin: 0; font-size: 40px; font-weight: 800; color: var(--accent); }
.countdown-banner__label { margin: 4px 0 0; font-size: 13px; color: var(--text-muted); }

.today-card { background: var(--accent-dim); border: 1px solid var(--accent); border-radius: var(--radius-md); padding: 18px; margin-bottom: 16px; }
.today-card__label { margin: 0 0 4px; font-size: 12px; color: var(--accent); font-weight: 700; }
.today-card__text { margin: 0; font-size: 15px; font-weight: 600; }

.routine-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
.routine-item { display: flex; gap: 12px; background: var(--surface); border-radius: var(--radius-sm); padding: 12px 14px; font-size: 13px; }
.routine-item__day { color: var(--accent); font-weight: 700; flex-shrink: 0; }

@media (min-width: 481px) { .screen { padding-top: 40px; } }
</style>
</head>
<body>

  <section id="screen-splash" class="screen screen-splash is-active">
    <h1 class="wordmark">clozkin</h1>
    <div class="dot-loader"><span></span><span></span><span></span></div>
  </section>

  <section id="screen-home" class="screen">
    <header class="topbar"><span class="wordmark-sm">clozkin</span></header>

    <div class="status-banner" id="status-banner">
      <p class="status-banner__label">오늘의 1분 케어</p>
      <p class="status-banner__text" id="status-banner-text">피부 진단부터 시작해보세요</p>
    </div>

    <nav class="menu-list">
      <button class="menu-card" data-nav="diagnose">
        <span class="menu-card__icon">◎</span>
        <span class="menu-card__body">
          <span class="menu-card__title">피부 진단하기</span>
          <span class="menu-card__desc">사진 한 장으로 지금 내 피부 상태 확인</span>
        </span>
      </button>
      <button class="menu-card" data-nav="ranking">
        <span class="menu-card__icon">◆</span>
        <span class="menu-card__body">
          <span class="menu-card__title">우리 동네 피부랭킹</span>
          <span class="menu-card__desc">피부 좋은 남자들은 뭘 쓸까</span>
        </span>
      </button>
      <button class="menu-card" data-nav="event">
        <span class="menu-card__icon">▲</span>
        <span class="menu-card__body">
          <span class="menu-card__title">D-day 케어 모드</span>
          <span class="menu-card__desc">소개팅·면접 전 집중 관리</span>
        </span>
      </button>
    </nav>
  </section>

  <section id="screen-diagnose" class="screen">
    <header class="subbar">
      <button class="btn-back" data-nav="home">‹</button>
      <span class="subbar__title">피부 진단</span>
    </header>

    <div class="scan-stage">
      <video id="camera-video" autoplay playsinline muted></video>
      <canvas id="camera-canvas" class="is-hidden"></canvas>
      <img id="captured-preview" class="is-hidden" alt="촬영된 사진">
      <div class="scan-ring" id="scan-ring"></div>
    </div>

    <p class="scan-hint" id="scan-hint">얼굴이 원 안에 오도록 맞춰주세요</p>

    <div class="scan-actions" id="scan-actions">
      <button class="btn-primary" id="btn-capture">사진 촬영</button>
    </div>

    <div class="scan-actions is-hidden" id="analyzing-actions">
      <div class="dot-loader dot-loader--dark"><span></span><span></span><span></span></div>
      <p class="analyzing-text">피부 상태를 분석하는 중...</p>
    </div>

    <div class="result-card is-hidden" id="diagnose-result">
      <p class="result-score-label">피부 스코어</p>
      <p class="result-score" id="result-score">-</p>
      <p class="result-type" id="result-type">-</p>
      <p class="result-summary" id="result-summary">-</p>
      <div class="chip-row" id="result-concerns"></div>
      <div class="chip-row chip-row--accent" id="result-ingredients"></div>
      <button class="btn-primary" data-nav="home">홈으로</button>
      <button class="btn-secondary" data-nav="ranking">동네 랭킹 보러가기</button>
    </div>

    <div class="error-banner is-hidden" id="diagnose-error"></div>
  </section>

  <section id="screen-ranking" class="screen">
    <header class="subbar">
      <button class="btn-back" data-nav="home">‹</button>
      <span class="subbar__title">우리 동네 피부랭킹</span>
    </header>
    <p class="ranking-note" id="ranking-note">피부 진단을 하면 내 순위도 함께 볼 수 있어요</p>
    <ul class="ranking-list" id="ranking-list"></ul>
  </section>

  <section id="screen-event" class="screen">
    <header class="subbar">
      <button class="btn-back" data-nav="home">‹</button>
      <span class="subbar__title">D-day 케어 모드</span>
    </header>

    <div id="event-setup">
      <p class="section-label">어떤 이벤트를 준비하시나요?</p>
      <div class="event-chip-row" id="event-chip-row">
        <button class="event-chip" data-event="date">소개팅</button>
        <button class="event-chip" data-event="interview">면접</button>
        <button class="event-chip" data-event="wedding">결혼식</button>
        <button class="event-chip" data-event="meeting">상견례</button>
        <button class="event-chip" data-event="dating">데이트</button>
      </div>

      <p class="section-label">언제인가요?</p>
      <input type="date" id="event-date" class="date-input">

      <button class="btn-primary" id="btn-generate-routine">케어 루틴 만들기</button>
    </div>

    <div class="scan-actions is-hidden" id="routine-loading">
      <div class="dot-loader dot-loader--dark"><span></span><span></span><span></span></div>
      <p class="analyzing-text">맞춤 루틴을 짜는 중...</p>
    </div>

    <div class="countdown-banner is-hidden" id="countdown-banner">
      <p class="countdown-banner__dday" id="countdown-dday">D-0</p>
      <p class="countdown-banner__label" id="countdown-label">이벤트까지</p>
    </div>

    <div class="today-card is-hidden" id="today-card">
      <p class="today-card__label">오늘 할 일</p>
      <p class="today-card__text" id="today-card-text">-</p>
    </div>

    <ul class="routine-list is-hidden" id="routine-list"></ul>

    <div class="error-banner is-hidden" id="routine-error"></div>
  </section>

<script>
const screens = {
  splash: document.getElementById('screen-splash'),
  home: document.getElementById('screen-home'),
  diagnose: document.getElementById('screen-diagnose'),
  ranking: document.getElementById('screen-ranking'),
  event: document.getElementById('screen-event'),
};

let cameraStream = null;

function showScreen(name) {
  Object.values(screens).forEach((el) => el.classList.remove('is-active'));
  screens[name].classList.add('is-active');
  if (name === 'diagnose') { startCamera(); } else { stopCamera(); }
  if (name === 'ranking') { loadRanking(); }
}

document.addEventListener('click', (e) => {
  const target = e.target.closest('[data-nav]');
  if (target) { showScreen(target.dataset.nav); }
});

setTimeout(() => showScreen('home'), 1400);

const video = document.getElementById('camera-video');
const canvas = document.getElementById('camera-canvas');
const preview = document.getElementById('captured-preview');
const scanRing = document.getElementById('scan-ring');
const scanHint = document.getElementById('scan-hint');
const scanActions = document.getElementById('scan-actions');
const analyzingActions = document.getElementById('analyzing-actions');
const resultCard = document.getElementById('diagnose-result');
const diagnoseError = document.getElementById('diagnose-error');
const btnCapture = document.getElementById('btn-capture');

async function startCamera() {
  resetDiagnoseScreen();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = cameraStream;
    video.classList.remove('is-hidden');
    preview.classList.add('is-hidden');
    scanRing.classList.add('is-scanning');
  } catch (err) {
    scanHint.textContent = '카메라를 사용할 수 없어요. 권한을 확인해주세요.';
  }
}

function stopCamera() {
  if (cameraStream) { cameraStream.getTracks().forEach((t) => t.stop()); cameraStream = null; }
}

function resetDiagnoseScreen() {
  scanHint.textContent = '얼굴이 원 안에 오도록 맞춰주세요';
  scanActions.classList.remove('is-hidden');
  analyzingActions.classList.add('is-hidden');
  resultCard.classList.add('is-hidden');
  diagnoseError.classList.add('is-hidden');
}

btnCapture.addEventListener('click', async () => {
  const w = video.videoWidth || 480;
  const h = video.videoHeight || 480;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  preview.src = dataUrl;
  preview.classList.remove('is-hidden');
  video.classList.add('is-hidden');
  scanRing.classList.remove('is-scanning');
  stopCamera();

  scanActions.classList.add('is-hidden');
  analyzingActions.classList.remove('is-hidden');
  scanHint.textContent = '분석 중...';

  try {
    const res = await fetch('/api/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    analyzingActions.classList.add('is-hidden');

    if (!res.ok) {
      diagnoseError.textContent = data.error || '진단에 실패했어요. 다시 시도해주세요.';
      diagnoseError.classList.remove('is-hidden');
      scanActions.classList.remove('is-hidden');
      return;
    }
    renderDiagnosisResult(data);
    updateHomeBanner(data);
  } catch (err) {
    analyzingActions.classList.add('is-hidden');
    diagnoseError.textContent = '네트워크 오류가 발생했어요.';
    diagnoseError.classList.remove('is-hidden');
    scanActions.classList.remove('is-hidden');
  }
});

function renderDiagnosisResult(data) {
  document.getElementById('result-score').textContent = data.score ?? '-';
  document.getElementById('result-type').textContent = data.skin_type ? `피부 타입: ${data.skin_type}` : '';
  document.getElementById('result-summary').textContent = data.summary || '';

  const concernsEl = document.getElementById('result-concerns');
  concernsEl.innerHTML = '';
  (data.concerns || []).forEach((c) => {
    const span = document.createElement('span');
    span.textContent = c;
    concernsEl.appendChild(span);
  });

  const ingredientsEl = document.getElementById('result-ingredients');
  ingredientsEl.innerHTML = '';
  (data.recommended_ingredients || []).forEach((c) => {
    const span = document.createElement('span');
    span.textContent = `#${c}`;
    ingredientsEl.appendChild(span);
  });

  resultCard.classList.remove('is-hidden');
}

function updateHomeBanner(diagnosis) {
  const bannerText = document.getElementById('status-banner-text');
  if (diagnosis && diagnosis.summary) { bannerText.textContent = diagnosis.summary; }
}

async function loadRanking() {
  const list = document.getElementById('ranking-list');
  const note = document.getElementById('ranking-note');
  list.innerHTML = '<li class="ranking-item">불러오는 중...</li>';

  try {
    const res = await fetch('/api/ranking');
    const data = await res.json();

    note.textContent = data.has_diagnosis
      ? '피부 진단 결과를 기반으로 순위에 반영했어요'
      : '피부 진단을 하면 내 순위도 함께 볼 수 있어요';

    list.innerHTML = '';
    data.board.forEach((entry) => {
      const li = document.createElement('li');
      li.className = 'ranking-item' + (entry.is_me ? ' is-me' : '');
      li.innerHTML = `
        <span class="ranking-item__rank">${entry.rank}</span>
        <span class="ranking-item__body">
          <span class="ranking-item__name">${entry.name}</span>
          <span class="ranking-item__product">${entry.product}</span>
        </span>
        <span class="ranking-item__score">${entry.score}</span>
        <a class="ranking-item__link" target="_blank" rel="noopener"
           href="https://www.oliveyoung.co.kr/store/search/getSearchMain.do?query=${encodeURIComponent(entry.product)}">
           올리브영 →
        </a>
      `;
      list.appendChild(li);
    });
  } catch (err) {
    list.innerHTML = '<li class="ranking-item">랭킹을 불러오지 못했어요.</li>';
  }
}

let selectedEvent = null;

document.getElementById('event-chip-row').addEventListener('click', (e) => {
  const chip = e.target.closest('.event-chip');
  if (!chip) return;
  document.querySelectorAll('.event-chip').forEach((c) => c.classList.remove('is-selected'));
  chip.classList.add('is-selected');
  selectedEvent = chip.dataset.event;
});

document.getElementById('btn-generate-routine').addEventListener('click', async () => {
  const routineError = document.getElementById('routine-error');
  routineError.classList.add('is-hidden');

  const targetDate = document.getElementById('event-date').value;
  if (!selectedEvent || !targetDate) {
    routineError.textContent = '이벤트 종류와 날짜를 모두 선택해주세요.';
    routineError.classList.remove('is-hidden');
    return;
  }

  document.getElementById('event-setup').classList.add('is-hidden');
  document.getElementById('routine-loading').classList.remove('is-hidden');

  try {
    const res = await fetch('/api/routine', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event: selectedEvent, target_date: targetDate }),
    });
    const data = await res.json();
    document.getElementById('routine-loading').classList.add('is-hidden');

    if (!res.ok) {
      document.getElementById('event-setup').classList.remove('is-hidden');
      routineError.textContent = data.error || '루틴 생성에 실패했어요.';
      routineError.classList.remove('is-hidden');
      return;
    }
    renderRoutine(data);
  } catch (err) {
    document.getElementById('routine-loading').classList.add('is-hidden');
    document.getElementById('event-setup').classList.remove('is-hidden');
    routineError.textContent = '네트워크 오류가 발생했어요.';
    routineError.classList.remove('is-hidden');
  }
});

function renderRoutine(data) {
  const countdownBanner = document.getElementById('countdown-banner');
  document.getElementById('countdown-dday').textContent = `D-${data.days_left}`;
  document.getElementById('countdown-label').textContent = `${data.event_label}까지`;
  countdownBanner.classList.remove('is-hidden');

  const todayCard = document.getElementById('today-card');
  document.getElementById('today-card-text').textContent = data.today_task || '';
  todayCard.classList.remove('is-hidden');

  const listEl = document.getElementById('routine-list');
  listEl.innerHTML = '';
  (data.routine || []).forEach((item) => {
    const li = document.createElement('li');
    li.className = 'routine-item';
    li.innerHTML = `<span class="routine-item__day">${item.day_label}</span><span>${item.task}</span>`;
    listEl.appendChild(li);
  });
  listEl.classList.remove('is-hidden');
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """업로드된 얼굴 사진을 Claude Vision으로 분석해 피부 상태 JSON 반환."""
    if client is None:
        return jsonify({"error": "ANTHROPIC_API_KEY가 설정되지 않았습니다. .env를 확인하세요."}), 500

    data = request.get_json(silent=True) or {}
    image_data_url = data.get("image")
    if not image_data_url or "," not in image_data_url:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    try:
        header, b64_payload = image_data_url.split(",", 1)
        media_type = "image/jpeg"
        if "image/png" in header:
            media_type = "image/png"
        elif "image/webp" in header:
            media_type = "image/webp"

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

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        result = _extract_json(raw_text)

        session["last_diagnosis"] = result
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "AI 응답을 해석하지 못했습니다. 다시 시도해주세요."}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"AI 호출 중 오류가 발생했습니다: {str(e)}"}), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"알 수 없는 오류: {str(e)}"}), 500


@app.route("/api/ranking", methods=["GET"])
def ranking():
    """진단 점수를 기반으로 우리 동네 랭킹에 사용자 삽입."""
    diagnosis = session.get("last_diagnosis")
    board = list(MOCK_RANKING)

    if diagnosis and isinstance(diagnosis.get("score"), (int, float)):
        board.append({
            "name": "나 (진단 결과)",
            "score": diagnosis["score"],
            "product": (diagnosis.get("recommended_ingredients") or ["-"])[0],
            "is_me": True,
        })

    board.sort(key=lambda x: x["score"], reverse=True)
    for i, entry in enumerate(board, start=1):
        entry["rank"] = i

    return jsonify({"board": board, "has_diagnosis": diagnosis is not None})


@app.route("/api/routine", methods=["POST"])
def routine():
    """이벤트 종류 + D-day를 기반으로 카운트다운 케어 루틴 생성."""
    if client is None:
        return jsonify({"error": "ANTHROPIC_API_KEY가 설정되지 않았습니다. .env를 확인하세요."}), 500

    data = request.get_json(silent=True) or {}
    event_key = data.get("event", "date")
    target_date_str = data.get("target_date")

    if not target_date_str:
        return jsonify({"error": "목표 날짜(target_date)가 필요합니다."}), 400

    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "날짜 형식이 올바르지 않습니다 (YYYY-MM-DD)."}), 400

    days_left = (target_date - date.today()).days
    if days_left < 0:
        return jsonify({"error": "목표 날짜는 오늘 이후여야 합니다."}), 400

    event_label = EVENT_LABELS.get(event_key, event_key)
    diagnosis = session.get("last_diagnosis") or {
        "skin_type": "정보 없음",
        "concerns": ["일반 컨디션 관리"],
        "summary": "아직 피부 진단을 하지 않았어요.",
    }

    try:
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

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        result = _extract_json(raw_text)
        result["days_left"] = days_left
        result["event_label"] = event_label
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "AI 응답을 해석하지 못했습니다. 다시 시도해주세요."}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"AI 호출 중 오류가 발생했습니다: {str(e)}"}), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"알 수 없는 오류: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
