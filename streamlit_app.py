"""
AI Interview Arena — 완성형 Streamlit 앱
- Judge Agent (Claude API) 실제 연결
- Competitor 답변 LLM 동적 생성
- Elo Rating + Pairwise 비교 평가
- 레이더 차트 / 순위 시각화
"""

import os
import json
import asyncio
import random
import pandas as pd
import streamlit as st
from anthropic import Anthropic

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Arena",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 글로벌 스타일 ─────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* 사이드바 */
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

/* 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.5rem; font-weight: 600;
    transition: all 0.2s; width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(79,70,229,0.4); }

/* 카드 */
.arena-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.arena-card-gold  { border-left: 4px solid #f59e0b; }
.arena-card-blue  { border-left: 4px solid #3b82f6; }
.arena-card-green { border-left: 4px solid #10b981; }
.arena-card-red   { border-left: 4px solid #ef4444; }

/* 순위 뱃지 */
.rank-badge {
    display: inline-block; border-radius: 50%;
    width: 32px; height: 32px; line-height: 32px;
    text-align: center; font-weight: 700; font-size: 0.9rem;
}
.rank-1 { background: #f59e0b; color: #000; }
.rank-2 { background: #9ca3af; color: #000; }
.rank-3 { background: #92400e; color: #fff; }
.rank-4 { background: #374151; color: #fff; }

/* 점수 바 */
.score-bar-bg { background:#21262d; border-radius:6px; height:10px; margin:4px 0 12px; }
.score-bar-fill { border-radius:6px; height:10px; transition:width 0.6s ease; }

/* 헤더 타이틀 */
.arena-title { font-size:2.4rem; font-weight:800; letter-spacing:-0.5px; }
.arena-sub   { color:#8b949e; font-size:0.95rem; margin-top:0.2rem; }

/* 구분선 */
.arena-divider { border:none; border-top:1px solid #30363d; margin:1.5rem 0; }

/* metric 숫자 */
.big-score { font-size:2.2rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Anthropic 클라이언트 ──────────────────────────────────
@st.cache_resource
def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit secrets를 확인하세요.")
        st.stop()
    return Anthropic(api_key=api_key)

# ── 데이터셋 로드 ─────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_step2.csv")

# ── Elo Rating ────────────────────────────────────────────
INITIAL_ELO = 1000
K_FACTOR = 32

def elo_expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def elo_update(ratings: dict, winner_key: str, loser_key: str):
    """winner vs loser 1:1 Elo 업데이트"""
    ra, rb = ratings[winner_key], ratings[loser_key]
    ea = elo_expected(ra, rb)
    ratings[winner_key] = round(ra + K_FACTOR * (1.0 - ea), 1)
    ratings[loser_key]  = round(rb + K_FACTOR * (0.0 - (1 - ea)), 1)
    return ratings

# ── LLM 경쟁자 답변 생성 ──────────────────────────────────
COMPETITOR_PERSONAS = {
    "하": {
        "name": "김지훈",
        "emoji": "🥉",
        "label": "초보 준비생",
        "color": "#ef4444",
        "system": (
            "당신은 첫 면접을 보는 취업 준비생 '김지훈'입니다.\n"
            "답변은 2~4문장, 추상적이고 수치 없이, '열심히', '최선' 같은 표현을 씁니다.\n"
            "마크다운 없이 순수 텍스트로만 답변하세요."
        ),
        "score_range": (20, 45),
    },
    "중": {
        "name": "박서연",
        "emoji": "🥈",
        "label": "보통 준비생",
        "color": "#9ca3af",
        "system": (
            "당신은 면접을 어느 정도 준비한 취업 준비생 '박서연'입니다.\n"
            "STAR 기법을 느슨하게 따르되 수치는 없고, 5~8문장으로 답변합니다.\n"
            "마크다운 없이 순수 텍스트로만 답변하세요."
        ),
        "score_range": (45, 70),
    },
    "상": {
        "name": "이준혁",
        "emoji": "🥇",
        "label": "우수 준비생",
        "color": "#f59e0b",
        "system": (
            "당신은 철저히 준비한 우수한 면접 후보자 '이준혁'입니다.\n"
            "완벽한 STAR 구조, 구체적 수치 1개 이상, 전문 용어 2개 이상, 8~12문장으로 답변합니다.\n"
            "마크다운 없이 순수 텍스트로만 답변하세요."
        ),
        "score_range": (75, 95),
    },
}

def generate_competitor_answer(client: Anthropic, question: str, level: str, job: str) -> str:
    """경쟁자 LLM 답변 동적 생성"""
    persona = COMPETITOR_PERSONAS[level]
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=persona["system"] + f"\n[지원 직무]: {job}",
            messages=[{"role": "user", "content": f"면접 질문: {question}"}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"(답변 생성 실패: {e})"

# ── Judge Agent ───────────────────────────────────────────
JUDGE_SYSTEM = """당신은 공정한 AI 면접 평가 전문가입니다.
주어진 면접 질문과 두 답변(A, B)을 비교 평가하고, 반드시 아래 JSON 형식으로만 응답하세요.
다른 텍스트나 마크다운 없이 JSON만 출력하세요.

{
  "answer_a": {
    "논리성": <0-100 정수>,
    "전문성": <0-100 정수>,
    "직무적합성": <0-100 정수>,
    "명확성": <0-100 정수>,
    "구체성": <0-100 정수>,
    "피드백": "<2~3문장 구체적 피드백>"
  },
  "answer_b": {
    "논리성": <0-100 정수>,
    "전문성": <0-100 정수>,
    "직무적합성": <0-100 정수>,
    "명확성": <0-100 정수>,
    "구체성": <0-100 정수>,
    "피드백": "<2~3문장 구체적 피드백>"
  },
  "winner": "A" or "B" or "TIE",
  "reason": "<승자 선택 이유 1~2문장>"
}"""

def judge_pairwise(client: Anthropic, question: str, answer_a: str, label_a: str,
                   answer_b: str, label_b: str, criteria: str = "") -> dict:
    """두 답변 Pairwise 비교 — Claude Judge"""
    criteria_text = f"\n[평가 기준]: {criteria}" if criteria else ""
    user_msg = (
        f"[면접 질문]: {question}{criteria_text}\n\n"
        f"[답변 A ({label_a})]:\n{answer_a}\n\n"
        f"[답변 B ({label_b})]:\n{answer_b}"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        # Fallback: 랜덤 더미 점수
        def rand_scores(lo, hi):
            return {k: random.randint(lo, hi) for k in ["논리성","전문성","직무적합성","명확성","구체성"]}
        return {
            "answer_a": {**rand_scores(40, 80), "피드백": f"(Judge 오류: {e})"},
            "answer_b": {**rand_scores(40, 80), "피드백": f"(Judge 오류: {e})"},
            "winner": "TIE",
            "reason": "평가 중 오류가 발생했습니다.",
        }

def avg_scores(score_dict: dict) -> float:
    keys = ["논리성", "전문성", "직무적합성", "명확성", "구체성"]
    return round(sum(score_dict[k] for k in keys) / len(keys), 1)

# ── Radar Chart (SVG) ────────────────────────────────────
import math

def make_radar_svg(scores: dict, color: str, label: str, size: int = 220) -> str:
    """5각형 레이더 차트 SVG 생성"""
    keys = ["논리성", "전문성", "직무적합성", "명확성", "구체성"]
    n = len(keys)
    cx, cy, r = size / 2, size / 2, size * 0.38
    angles = [math.pi / 2 + 2 * math.pi * i / n for i in range(n)]

    def pt(angle, radius):
        return cx + radius * math.cos(angle), cy - radius * math.sin(angle)

    # 배경 그리드
    grid_lines = ""
    for level in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{pt(a, r*level)[0]:.1f},{pt(a, r*level)[1]:.1f}" for a in angles)
        grid_lines += f'<polygon points="{pts}" fill="none" stroke="#30363d" stroke-width="1"/>\n'

    # 축선
    axis_lines = ""
    for a in angles:
        x2, y2 = pt(a, r)
        axis_lines += f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#30363d" stroke-width="1"/>\n'

    # 데이터 폴리곤
    data_pts = " ".join(
        f"{pt(angles[i], r * scores.get(keys[i], 0) / 100)[0]:.1f},"
        f"{pt(angles[i], r * scores.get(keys[i], 0) / 100)[1]:.1f}"
        for i in range(n)
    )
    fill_color = color + "44"  # 투명도
    polygon = f'<polygon points="{data_pts}" fill="{fill_color}" stroke="{color}" stroke-width="2"/>\n'

    # 점
    dots = ""
    for i, key in enumerate(keys):
        val = scores.get(key, 0) / 100
        x, y = pt(angles[i], r * val)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>\n'

    # 레이블
    labels = ""
    for i, key in enumerate(keys):
        x, y = pt(angles[i], r * 1.22)
        val = scores.get(key, 0)
        labels += (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#8b949e">{key}</text>\n'
            f'<text x="{x:.1f}" y="{y+13:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="700" fill="{color}">{val}</text>\n'
        )

    return f"""
<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{size}" height="{size}" fill="#0d1117" rx="10"/>
  {grid_lines}{axis_lines}{polygon}{dots}{labels}
  <text x="{cx:.0f}" y="{size-8}" text-anchor="middle"
        font-size="11" font-weight="600" fill="{color}">{label}</text>
</svg>"""

# ── 점수 바 HTML ──────────────────────────────────────────
def score_bar(label: str, value: int, color: str) -> str:
    return f"""
<div style="margin-bottom:6px">
  <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8b949e;margin-bottom:3px">
    <span>{label}</span><span style="color:{color};font-weight:700">{value}</span>
  </div>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{value}%;background:{color}"></div>
  </div>
</div>"""

# ── 메인 UI ───────────────────────────────────────────────
def main():
    client = get_client()
    df = load_dataset()

    # ── 헤더 ─────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem">
      <div class="arena-title">🏟️ AI Interview <span style="color:#4f46e5">Arena</span></div>
      <div class="arena-sub">Multi-Agent 기반 경쟁형 AI 면접 시뮬레이션 | 자연어처리 2조</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 사이드바 설정 ────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 면접 설정")

        categories = ["전체"] + sorted(df["카테고리"].dropna().unique().tolist())
        selected_cat = st.selectbox("📂 카테고리", categories)

        difficulties = ["전체", "하", "중", "상"]
        selected_diff = st.selectbox("🎯 난이도", difficulties)

        job = st.text_input("💼 지원 직무", value="AI 개발자",
                            placeholder="예: 백엔드 개발자, 데이터 사이언티스트")

        st.markdown("---")
        st.markdown("### 🤖 경쟁자 설정")
        use_llm = st.toggle("LLM 동적 생성 사용", value=True,
                            help="ON: Claude가 실시간으로 경쟁자 답변 생성 / OFF: 데이터셋 예시 사용")

        levels_to_show = st.multiselect(
            "경쟁자 수준 선택",
            ["하", "중", "상"],
            default=["하", "중", "상"],
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#8b949e;line-height:1.6">
        <b>평가 방식</b><br>
        • LLM-as-a-Judge<br>
        • Pairwise Comparison<br>
        • Elo Rating (K=32, 초기=1000)
        </div>
        """, unsafe_allow_html=True)

    # ── 질문 필터링 ──────────────────────────────────────
    df_view = df.copy()
    if selected_cat != "전체":
        df_view = df_view[df_view["카테고리"] == selected_cat]
    if selected_diff != "전체":
        df_view = df_view[df_view["난이도"] == selected_diff]

    if df_view.empty:
        st.warning("선택한 조건에 해당하는 질문이 없습니다.")
        return

    # ── 질문 선택 ────────────────────────────────────────
    question_options = df_view["질문"].tolist()
    selected_q = st.selectbox("📋 면접 질문 선택", question_options)
    row = df_view[df_view["질문"] == selected_q].iloc[0]

    col_q, col_meta = st.columns([3, 1])
    with col_q:
        st.markdown(f"""
        <div class="arena-card arena-card-blue">
          <div style="font-size:0.75rem;color:#3b82f6;font-weight:700;margin-bottom:6px">💬 면접 질문</div>
          <div style="font-size:1.05rem;font-weight:600;line-height:1.6">{selected_q}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_meta:
        st.markdown(f"""
        <div class="arena-card" style="text-align:center">
          <div style="font-size:0.75rem;color:#8b949e">카테고리</div>
          <div style="font-weight:700;color:#4f46e5">{row['카테고리']}</div>
          <hr class="arena-divider">
          <div style="font-size:0.75rem;color:#8b949e">난이도</div>
          <div style="font-weight:700;color:#f59e0b">{row['난이도']}</div>
        </div>
        """, unsafe_allow_html=True)

    if "좋은 답변 기준" in df.columns and pd.notna(row.get("좋은 답변 기준")):
        with st.expander("📌 좋은 답변 기준 보기"):
            st.info(row["좋은 답변 기준"])

    # ── 사용자 답변 입력 ─────────────────────────────────
    st.markdown("#### ✏️ 내 답변 입력")
    user_answer = st.text_area(
        label="user_answer",
        label_visibility="collapsed",
        placeholder="면접 답변을 자유롭게 작성하세요. STAR 기법(상황-과제-행동-결과)을 활용하면 좋습니다.",
        height=180,
        key="user_answer_input",
    )

    char_count = len(user_answer)
    st.markdown(
        f'<div style="text-align:right;font-size:0.75rem;color:{"#10b981" if char_count > 50 else "#8b949e"}">'
        f'{char_count}자 {"✓ 충분한 분량" if char_count > 50 else "(50자 이상 권장)"}</div>',
        unsafe_allow_html=True,
    )

    # ── 평가 시작 버튼 ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    start_btn = st.button("⚔️ AI 경쟁 면접 평가 시작", type="primary")

    if not start_btn:
        return

    if not user_answer.strip():
        st.warning("답변을 먼저 입력해주세요.")
        return

    if not levels_to_show:
        st.warning("경쟁자 수준을 1개 이상 선택해주세요.")
        return

    # ── Step 1: 경쟁자 답변 생성 ─────────────────────────
    st.markdown("<hr class='arena-divider'>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI 경쟁자 답변")

    competitor_answers = {}
    level_map = {"하": "하 답변", "중": "중 답변", "상": "상 답변"}
    cols = st.columns(len(levels_to_show))

    for idx, level in enumerate(levels_to_show):
        persona = COMPETITOR_PERSONAS[level]
        with cols[idx]:
            with st.spinner(f"{persona['emoji']} {persona['name']} 답변 생성 중..."):
                if use_llm:
                    answer = generate_competitor_answer(client, selected_q, level, job)
                else:
                    col_key = level_map[level]
                    answer = row[col_key] if col_key in df.columns and pd.notna(row[col_key]) else "(데이터 없음)"
                competitor_answers[level] = answer

            st.markdown(f"""
            <div class="arena-card" style="border-left:4px solid {persona['color']}">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <span style="font-size:1.3rem">{persona['emoji']}</span>
                <div>
                  <div style="font-weight:700;font-size:0.9rem">{persona['name']}</div>
                  <div style="font-size:0.75rem;color:{persona['color']}">{persona['label']}</div>
                </div>
              </div>
              <div style="font-size:0.88rem;line-height:1.65;color:#c9d1d9">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Step 2: Judge Pairwise 평가 ──────────────────────
    st.markdown("<hr class='arena-divider'>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Judge AI 비교 평가")

    criteria = str(row.get("좋은 답변 기준", "")) if pd.notna(row.get("좋은 답변 기준")) else ""

    # 사용자 vs 각 경쟁자 평가
    all_scores = {"나": {"논리성":0,"전문성":0,"직무적합성":0,"명확성":0,"구체성":0,"피드백":"","total":0}}
    for level in levels_to_show:
        all_scores[level] = {"논리성":0,"전문성":0,"직무적합성":0,"명확성":0,"구체성":0,"피드백":"","total":0}

    pairwise_results = {}

    progress_bar = st.progress(0, text="Judge AI가 답변을 비교 분석 중...")
    total_pairs = len(levels_to_show)

    for i, level in enumerate(levels_to_show):
        persona = COMPETITOR_PERSONAS[level]
        with st.spinner(f"나 vs {persona['name']} 비교 중..."):
            result = judge_pairwise(
                client, selected_q,
                user_answer, "나 (사용자)",
                competitor_answers[level], f"{persona['name']} ({level}수준)",
                criteria,
            )
        pairwise_results[level] = result

        # 점수 누적 (나중에 평균)
        for k in ["논리성","전문성","직무적합성","명확성","구체성"]:
            all_scores["나"][k] += result["answer_a"].get(k, 0)
            all_scores[level][k] = result["answer_b"].get(k, 0)

        all_scores["나"]["피드백"] = result["answer_a"].get("피드백", "")
        all_scores[level]["피드백"] = result["answer_b"].get("피드백", "")
        all_scores[level]["total"] = avg_scores(result["answer_b"])

        progress_bar.progress((i + 1) / total_pairs, text=f"비교 완료: {i+1}/{total_pairs}")

    progress_bar.empty()

    # 나의 점수 평균 (여러 경쟁자와 비교한 결과의 평균)
    n = len(levels_to_show)
    for k in ["논리성","전문성","직무적합성","명확성","구체성"]:
        all_scores["나"][k] = round(all_scores["나"][k] / n)
    all_scores["나"]["total"] = avg_scores(all_scores["나"])

    # ── Step 3: Elo Rating 계산 ──────────────────────────
    elo_ratings = {"나": INITIAL_ELO}
    for level in levels_to_show:
        elo_ratings[level] = INITIAL_ELO

    for level, result in pairwise_results.items():
        winner = result.get("winner", "TIE")
        if winner == "A":
            elo_ratings = elo_update(elo_ratings, "나", level)
        elif winner == "B":
            elo_ratings = elo_update(elo_ratings, level, "나")
        else:  # TIE
            pass  # Elo 변동 없음

    # ── Step 4: 최종 순위 산정 ───────────────────────────
    ranked = sorted(
        [("나", elo_ratings["나"], all_scores["나"])] +
        [(level, elo_ratings[level], all_scores[level]) for level in levels_to_show],
        key=lambda x: x[1],
        reverse=True,
    )
    user_rank = next(i + 1 for i, (name, _, _) in enumerate(ranked) if name == "나")

    # ── 결과 헤더 ────────────────────────────────────────
    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(user_rank, "4️⃣")
    rank_msg = {
        1: ("상위권", "#10b981", "AI 경쟁자 전원을 능가했습니다!"),
        2: ("중상위권", "#3b82f6", "대부분의 경쟁자보다 우수합니다."),
        3: ("중위권", "#f59e0b", "일부 경쟁자보다 아쉬운 결과입니다."),
        4: ("하위권", "#ef4444", "더 구체적이고 논리적인 답변이 필요합니다."),
    }.get(user_rank, ("—", "#8b949e", ""))

    st.markdown("<hr class='arena-divider'>", unsafe_allow_html=True)
    st.markdown("### 🏆 최종 결과")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    with res_col1:
        st.markdown(f"""
        <div class="arena-card" style="text-align:center">
          <div style="font-size:0.75rem;color:#8b949e">최종 순위</div>
          <div style="font-size:2.5rem">{rank_emoji}</div>
          <div style="font-weight:700;font-size:1.2rem">{user_rank}위 / {len(ranked)}명</div>
        </div>
        """, unsafe_allow_html=True)
    with res_col2:
        st.markdown(f"""
        <div class="arena-card" style="text-align:center">
          <div style="font-size:0.75rem;color:#8b949e">내 총점</div>
          <div class="big-score" style="color:{rank_msg[1]}">{all_scores['나']['total']}</div>
          <div style="font-size:0.75rem;color:#8b949e">/ 100점</div>
        </div>
        """, unsafe_allow_html=True)
    with res_col3:
        st.markdown(f"""
        <div class="arena-card" style="text-align:center">
          <div style="font-size:0.75rem;color:#8b949e">Elo Rating</div>
          <div class="big-score" style="color:#4f46e5">{elo_ratings['나']}</div>
          <div style="font-size:0.75rem;color:#8b949e">초기 1000점 기준</div>
        </div>
        """, unsafe_allow_html=True)
    with res_col4:
        st.markdown(f"""
        <div class="arena-card" style="text-align:center">
          <div style="font-size:0.75rem;color:#8b949e">수준 판정</div>
          <div style="font-size:1.3rem;font-weight:700;color:{rank_msg[1]}">{rank_msg[0]}</div>
          <div style="font-size:0.75rem;color:#8b949e;margin-top:4px">{rank_msg[2]}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 전체 순위표 ──────────────────────────────────────
    st.markdown("#### 📊 전체 순위표")
    for rank_i, (name, elo, scores) in enumerate(ranked):
        is_user = name == "나"
        rank_num = rank_i + 1
        badge_cls = f"rank-{min(rank_num, 4)}"

        if is_user:
            persona_label = "나 (사용자)"
            color = "#4f46e5"
            emoji = "👤"
        else:
            p = COMPETITOR_PERSONAS[name]
            persona_label = f"{p['emoji']} {p['name']} ({name}수준)"
            color = p["color"]
            emoji = p["emoji"]

        total = scores.get("total", 0)
        border = "border:1px solid #4f46e5;" if is_user else ""

        st.markdown(f"""
        <div class="arena-card" style="{border}">
          <div style="display:flex;align-items:center;gap:12px">
            <span class="rank-badge {badge_cls}">{rank_num}</span>
            <div style="flex:1">
              <div style="font-weight:700;{"color:#4f46e5" if is_user else ""}">{persona_label}</div>
              <div style="font-size:0.75rem;color:#8b949e">Elo: {elo} | 총점: {total}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:1.4rem;font-weight:800;color:{color}">{total}</div>
              <div style="font-size:0.7rem;color:#8b949e">점</div>
            </div>
          </div>
          <div style="margin-top:10px">
            {score_bar("논리성",   scores.get("논리성",0),   color)}
            {score_bar("전문성",   scores.get("전문성",0),   color)}
            {score_bar("직무적합성", scores.get("직무적합성",0), color)}
            {score_bar("명확성",   scores.get("명확성",0),   color)}
            {score_bar("구체성",   scores.get("구체성",0),   color)}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 레이더 차트 ──────────────────────────────────────
    st.markdown("#### 🕸️ 역량 레이더 차트")
    radar_names = ["나"] + levels_to_show
    radar_cols = st.columns(len(radar_names))
    for ci, name in enumerate(radar_names):
        with radar_cols[ci]:
            if name == "나":
                color = "#4f46e5"
                label = "나 (사용자)"
            else:
                color = COMPETITOR_PERSONAS[name]["color"]
                label = f"{COMPETITOR_PERSONAS[name]['name']}"
            svg = make_radar_svg(all_scores[name], color, label)
            st.markdown(svg, unsafe_allow_html=True)

    # ── Pairwise 비교 상세 ───────────────────────────────
    st.markdown("#### 🔍 1:1 비교 상세")
    for level, result in pairwise_results.items():
        persona = COMPETITOR_PERSONAS[level]
        winner = result.get("winner", "TIE")
        win_label = "🏆 내가 이겼습니다" if winner == "A" else ("😔 경쟁자가 이겼습니다" if winner == "B" else "🤝 무승부")

        with st.expander(f"나 vs {persona['emoji']} {persona['name']} ({level}수준) — {win_label}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**👤 나의 답변 평가**")
                st.markdown(result["answer_a"].get("피드백", ""), unsafe_allow_html=False)
            with c2:
                st.markdown(f"**{persona['emoji']} {persona['name']} 답변 평가**")
                st.markdown(result["answer_b"].get("피드백", ""), unsafe_allow_html=False)
            st.info(f"**Judge 판단 이유**: {result.get('reason', '')}")

    # ── 개선 피드백 ──────────────────────────────────────
    st.markdown("<hr class='arena-divider'>", unsafe_allow_html=True)
    st.markdown("### 💡 종합 피드백 & 개선 포인트")

    # 가장 낮은 두 항목 찾기
    score_items = {k: all_scores["나"][k] for k in ["논리성","전문성","직무적합성","명확성","구체성"]}
    weakest = sorted(score_items.items(), key=lambda x: x[1])[:2]

    feedback_tips = {
        "논리성":    "주장 → 근거 → 결론의 흐름을 명확히 하고, STAR 기법을 활용해 보세요.",
        "전문성":    "직무 관련 키워드와 기술 용어를 자연스럽게 포함시켜 보세요.",
        "직무적합성": "지원 직무와 본인의 경험을 직접적으로 연결하는 문장을 추가하세요.",
        "명확성":    "두괄식으로 핵심 메시지를 먼저 말하고, 이후에 부연 설명을 더하세요.",
        "구체성":    "수치(%, 기간, 건수 등)나 구체적 프로젝트명을 1개 이상 포함하세요.",
    }

    tip_cols = st.columns(2)
    for ci, (item, score) in enumerate(weakest):
        with tip_cols[ci]:
            st.markdown(f"""
            <div class="arena-card arena-card-red">
              <div style="font-size:0.75rem;color:#ef4444;font-weight:700">개선 필요 · {item} ({score}점)</div>
              <div style="margin-top:6px;font-size:0.9rem;line-height:1.6">{feedback_tips[item]}</div>
            </div>
            """, unsafe_allow_html=True)

    # 내 답변 & 상 수준 모범 답변 비교
    with st.expander("📝 내 답변 vs 모범 답변 비교"):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**👤 내 답변**")
            st.write(user_answer)
        with cc2:
            if "상" in levels_to_show:
                st.markdown("**🥇 상 수준 AI 답변 (참고용)**")
                st.write(competitor_answers.get("상", "—"))
            elif row.get("상 답변") and pd.notna(row.get("상 답변")):
                st.markdown("**🥇 모범 답변 (데이터셋)**")
                st.write(row["상 답변"])

    # ── 다시 하기 ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 다른 질문으로 다시 도전"):
        st.rerun()

# ── 진입점 ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
