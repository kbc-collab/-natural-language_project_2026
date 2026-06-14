"""
AI Interview Arena — 채팅형 완성 버전
- 좌: 채팅 UI (면접관 질문 + 사용자 답변 연속)
- 우: AI 경쟁자 답변 + Judge 순위 패널
- 꼬리질문 자동 생성 (이전 답변 컨텍스트 전달)
- Judge: rubric.py 가중치 + Claude API 실제 연결
- Elo Rating 누적
"""

import os, json, random
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

# ── 스타일 ────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #161b22; border-right:1px solid #30363d; }

/* 채팅 버블 — 면접관 */
.bubble-interviewer {
    background: #1c2333; border:1px solid #30363d;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px; margin: 8px 0 8px 0;
    max-width: 90%;
}
/* 채팅 버블 — 사용자 */
.bubble-user {
    background: #1a1f6e; border:1px solid #3b4dc8;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px; margin: 8px 0 8px 40px;
}
/* 경쟁자 카드 */
.comp-card {
    background: #161b22; border:1px solid #30363d;
    border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;
    font-size: 0.82rem; line-height: 1.55;
}
/* 순위 카드 */
.rank-card {
    background: #0d1117; border:1px solid #30363d;
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
}
.rank-gold   { border-left: 4px solid #f59e0b; }
.rank-silver { border-left: 4px solid #9ca3af; }
.rank-bronze { border-left: 4px solid #92400e; }
.rank-low    { border-left: 4px solid #374151; }

/* 점수바 */
.bar-bg   { background:#21262d; border-radius:4px; height:7px; margin:3px 0 8px; }
.bar-fill { border-radius:4px; height:7px; }

/* 섹션 헤더 */
.section-label {
    font-size:0.7rem; font-weight:700; letter-spacing:0.08em;
    color:#8b949e; text-transform:uppercase; margin: 14px 0 6px;
}
.stButton > button { width:100%; }
</style>
""", unsafe_allow_html=True)

# ── Anthropic 클라이언트 ──────────────────────────────────
@st.cache_resource
def get_client():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
    if not key:
        st.error("⚠️ ANTHROPIC_API_KEY가 없습니다. .env 또는 Streamlit secrets를 확인하세요.")
        st.stop()
    return Anthropic(api_key=key)

# ── 데이터셋 ──────────────────────────────────────────────
@st.cache_data
def load_df():
    return pd.read_csv("dataset_step2.csv")

# ── rubric (C 브랜치 rubric.py 인라인) ───────────────────
RUBRICS = {
    "인성": {
        "desc": "구체적 상황 제시, 본인의 역할 명확화, 갈등/성장 과정 논리적 서술, 결과와 배운 점 포함",
        "weights": {"논리성":2, "구체성":2, "직무적합성":1, "완성도":1},
    },
    "AI/NLP 기초": {
        "desc": "핵심 개념 정확히 설명, 기술 간 차이점 이해, 실제 적용 사례 언급",
        "weights": {"논리성":1, "구체성":1, "직무적합성":3, "완성도":1},
    },
    "프로젝트 경험": {
        "desc": "문제를 구체적으로 정의, 접근 방법 논리적 서술, 결과 수치화, 재현 가능한 해결책 제시",
        "weights": {"논리성":2, "구체성":3, "직무적합성":2, "완성도":1},
    },
    "문제해결/실무": {
        "desc": "요구사항 분석 우선, 복수의 접근법 비교, 선택 이유 명확, 실무 경험 기반",
        "weights": {"논리성":2, "구체성":2, "직무적합성":3, "완성도":1},
    },
}
SCORE_KEYS = ["논리성", "구체성", "직무적합성", "완성도"]

def max_score(cat):
    return sum(RUBRICS[cat]["weights"].values()) * 5

def weighted_score(scores: dict, cat: str) -> float:
    w = RUBRICS[cat]["weights"]
    return sum(scores[k] * w[k] for k in w)

def normalize(raw, cat):
    return round(raw / max_score(cat) * 100, 1)

# ── Elo ───────────────────────────────────────────────────
K = 32
def elo_expected(ra, rb): return 1 / (1 + 10**((rb - ra) / 400))
def elo_update(ratings, winner, loser):
    ra, rb = ratings[winner], ratings[loser]
    ea = elo_expected(ra, rb)
    ratings[winner] = round(ra + K*(1.0 - ea), 1)
    ratings[loser]  = round(rb + K*(0.0 - (1-ea)), 1)

# ── 경쟁자 프롬프트 ───────────────────────────────────────
COMPETITOR_SYSTEM = {
    "상": (
        "당신은 철저히 준비한 우수한 면접 후보자입니다.\n"
        "STAR 구조를 완벽하게 따르고, 수치를 1개 이상 포함하며, "
        "전문 용어를 자연스럽게 사용해 8~12문장으로 답변합니다.\n"
        "마크다운 없이 자연어 텍스트로만 출력하세요."
    ),
    "중": (
        "당신은 어느 정도 준비된 면접 후보자입니다.\n"
        "STAR 구조를 느슨하게 따르되 구체적 수치가 부족하고, "
        "5~8문장으로 결론이 약하게 끝납니다.\n"
        "마크다운 없이 자연어 텍스트로만 출력하세요."
    ),
    "하": (
        "당신은 면접 준비가 부족한 초보 후보자입니다.\n"
        "'열심히', '최선'처럼 추상적인 표현을 쓰며 2~4문장으로 짧게 답합니다.\n"
        "마크다운 없이 자연어 텍스트로만 출력하세요."
    ),
}
COMP_TEMP = {"상": 0.4, "중": 0.7, "하": 0.9}
COMP_COLOR = {"상": "#f59e0b", "중": "#9ca3af", "하": "#6b7280"}
COMP_EMOJI = {"상": "🥇", "중": "🥈", "하": "🥉"}

# ── 면접관 꼬리질문 생성 ──────────────────────────────────
INTERVIEWER_SYSTEM = """\
당신은 AI/NLP 분야 전문 면접관입니다.
면접 흐름을 이어가며 자연스러운 꼬리질문을 합니다.

규칙:
1. 반드시 한국어로 질문 1개만 출력하세요.
2. 질문 본문만 출력하고 부연 설명은 없습니다.
3. 이전에 한 질문은 반복하지 마세요.
4. 이전 답변의 핵심 내용에 이어지는 구체적인 꼬리질문을 합니다.
5. STAR 기법으로 답할 수 있는 질문을 선호합니다.\
"""

def generate_followup(client, question_history, last_answer, job):
    prev = "\n".join(f"- {q}" for q in question_history[:-1]) or "없음"
    user_msg = (
        f"[지원 직무]: {job}\n"
        f"[이전 질문들]: {prev}\n"
        f"[현재 질문]: {question_history[-1]}\n"
        f"[지원자 답변]: {last_answer}\n\n"
        "위 답변에 이어지는 꼬리질문 1개를 생성하세요."
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0.6,
        system=INTERVIEWER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()

def generate_competitor_answer(client, question, level, job):
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        temperature=COMP_TEMP[level],
        system=COMPETITOR_SYSTEM[level] + f"\n[지원 직무]: {job}",
        messages=[{"role": "user", "content": f"면접 질문: {question}"}],
    )
    return resp.content[0].text.strip()

# ── Judge 평가 ────────────────────────────────────────────
JUDGE_SYSTEM = """\
당신은 AI/NLP 분야 전문 면접 평가관입니다.
지원자(A)와 AI 경쟁자(B)의 답변을 비교하고 반드시 JSON만 출력하세요.

{
  "answer_a": {"논리성":점수,"구체성":점수,"직무적합성":점수,"완성도":점수,"피드백":"1~2문장"},
  "answer_b": {"논리성":점수,"구체성":점수,"직무적합성":점수,"완성도":점수,"피드백":"1~2문장"},
  "winner": "A" or "B" or "TIE",
  "reason": "2~3문장"
}

점수는 1~5 정수. 답변 순서(A/B) 편향 없이 내용만 평가하세요.\
"""

def judge(client, question, category, user_ans, comp_ans):
    rubric = RUBRICS.get(category, RUBRICS["인성"])
    weight_txt = ", ".join(f"{k}×{v}" for k,v in rubric["weights"].items())
    msg = (
        f"[질문] {question}\n[카테고리] {category}\n"
        f"[좋은 답변 기준] {rubric['desc']}\n[가중치] {weight_txt}\n\n"
        f"[답변 A] (지원자)\n{user_ans}\n\n[답변 B] (AI 경쟁자)\n{comp_ans}"
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[{"role":"user","content":msg}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())

# ── 세션 초기화 ───────────────────────────────────────────
def init_session():
    defaults = {
        "started": False,
        "chat": [],           # [{"role":"interviewer"|"user", "text":..., "q_idx":int}]
        "q_history": [],      # 질문 텍스트 목록
        "comp_results": [],   # 라운드별 경쟁자+judge 결과
        "elo": {},            # {"상":1000,"중":1000,"하":1000,"나":1000}
        "round": 0,
        "waiting_answer": False,
        "current_question": "",
        "current_category": "",
        "job": "AI/NLP 개발자",
        "level_a": "상",
        "level_b": "중",
        "max_rounds": 3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏟️ AI Interview Arena")
    st.markdown("---")

    if not st.session_state.started:
        st.markdown("**면접 설정**")
        job = st.text_input("지원 직무", value="AI/NLP 개발자")
        cat_opts = ["인성", "AI/NLP 기초", "프로젝트 경험", "문제해결/실무"]
        category = st.selectbox("질문 카테고리", cat_opts)
        diff_opts = ["하", "중", "상"]
        difficulty = st.selectbox("난이도", diff_opts, index=1)

        st.markdown("**AI 경쟁자 설정**")
        level_a = st.selectbox("경쟁자 A 수준", ["상", "중", "하"], index=0)
        level_b = st.selectbox("경쟁자 B 수준", ["상", "중", "하"], index=1)
        max_rounds = st.selectbox("질문 라운드 수", [2, 3, 4, 5], index=1)

        st.markdown("---")
        if st.button("🎯 AI 면접 시작", type="primary"):
            df = load_df()
            pool = df[(df["카테고리"] == category) & (df["난이도"] == difficulty)]
            if pool.empty:
                pool = df[df["카테고리"] == category]
            row = pool.sample(1).iloc[0]

            st.session_state.update({
                "started": True,
                "job": job,
                "level_a": level_a,
                "level_b": level_b,
                "max_rounds": max_rounds,
                "current_question": row["질문"],
                "current_category": category,
                "q_history": [row["질문"]],
                "waiting_answer": True,
                "round": 1,
                "elo": {"나": 1000, level_a: 1000, level_b: 1000},
                "chat": [{"role":"interviewer","text":row["질문"],"q_idx":1}],
                "comp_results": [],
            })
            st.rerun()
    else:
        st.markdown(f"**직무:** {st.session_state.job}")
        st.markdown(f"**카테고리:** {st.session_state.current_category}")
        st.markdown(f"**라운드:** {st.session_state.round} / {st.session_state.max_rounds}")
        st.markdown("---")

        # Elo 현황
        st.markdown("**📊 현재 Elo 순위**")
        elo = st.session_state.elo
        sorted_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_elo):
            medal = ["🥇","🥈","🥉","4️⃣"][min(i,3)]
            color = "#4f46e5" if name == "나" else COMP_COLOR.get(name, "#9ca3af")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:4px 0;font-size:0.85rem">'
                f'<span>{medal} <b style="color:{color}">{name}</b></span>'
                f'<span style="color:{color};font-weight:700">{score}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        if st.button("🔄 면접 초기화"):
            for k in ["started","chat","q_history","comp_results","elo",
                      "round","waiting_answer","current_question","current_category"]:
                del st.session_state[k]
            st.rerun()

# ── 메인 화면 ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("""
    <div style="text-align:center;padding:80px 0 40px">
      <div style="font-size:2.5rem;font-weight:800">🏟️ AI Interview Arena</div>
      <div style="color:#8b949e;margin-top:8px">
        AI 면접관 + AI 경쟁자 + AI Judge 기반 경쟁형 면접 시뮬레이션
      </div>
      <div style="color:#6b7280;font-size:0.85rem;margin-top:16px">
        ← 왼쪽 사이드바에서 설정 후 시작하세요
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

client = get_client()
left, right = st.columns([3, 2], gap="large")

# ─────────────────────────────────────────
# 왼쪽: 채팅 패널
# ─────────────────────────────────────────
with left:
    st.markdown("### 💬 면접 진행")

    # 채팅 히스토리 렌더링
    for msg in st.session_state.chat:
        if msg["role"] == "interviewer":
            st.markdown(
                f'<div class="bubble-interviewer">'
                f'<span style="font-size:0.7rem;color:#8b949e;font-weight:700">🤖 면접관 질문</span><br>'
                f'<span style="font-size:0.95rem">{msg["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bubble-user">'
                f'<span style="font-size:0.7rem;color:#818cf8;font-weight:700">👤 내 답변</span><br>'
                f'<span style="font-size:0.9rem">{msg["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # 면접 완료
    if st.session_state.round > st.session_state.max_rounds:
        st.markdown("---")
        final_elo = sorted(st.session_state.elo.items(), key=lambda x: x[1], reverse=True)
        my_rank = next(i+1 for i,(n,_) in enumerate(final_elo) if n=="나")
        rank_msg = {1:"🥇 1위! 모든 경쟁자를 앞질렀습니다.", 2:"🥈 2위! 우수한 성과입니다.",
                    3:"🥉 3위", 4:"4위 — 더 구체적인 답변 연습이 필요합니다."}.get(my_rank,"")
        st.success(f"**면접 종료** — 최종 {my_rank}위 / {len(final_elo)}명  {rank_msg}")
        st.stop()

    # 답변 입력창 (대기 중일 때만)
    if st.session_state.waiting_answer:
        with st.form("answer_form", clear_on_submit=True):
            user_input = st.text_area(
                "답변을 입력하세요",
                placeholder="면접 답변을 입력하세요...",
                height=120,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("➤ 제출", type="primary")

        if submitted and user_input.strip():
            answer = user_input.strip()
            q = st.session_state.current_question
            cat = st.session_state.current_category
            la = st.session_state.level_a
            lb = st.session_state.level_b

            # 채팅에 사용자 답변 추가
            st.session_state.chat.append({"role":"user","text":answer,"q_idx":st.session_state.round})
            st.session_state.waiting_answer = False

            with st.spinner("🤖 AI 경쟁자 답변 생성 중..."):
                ans_a = generate_competitor_answer(client, q, la, st.session_state.job)
                ans_b = generate_competitor_answer(client, q, lb, st.session_state.job)

            with st.spinner("⚖️ Judge 평가 중..."):
                res_a = judge(client, q, cat, answer, ans_a)
                res_b = judge(client, q, cat, answer, ans_b)

            # 정규화 점수
            def norm_result(res, cat):
                for key in ["answer_a","answer_b"]:
                    scores = {k: res[key][k] for k in SCORE_KEYS}
                    res[key]["normalized"] = normalize(weighted_score(scores, cat), cat)
                return res

            res_a = norm_result(res_a, cat)
            res_b = norm_result(res_b, cat)

            # Elo 업데이트
            elo = st.session_state.elo
            if la not in elo: elo[la] = 1000
            if lb not in elo: elo[lb] = 1000

            for res, level in [(res_a, la), (res_b, lb)]:
                w = res["winner"]
                if w == "A":   elo_update(elo, "나", level)
                elif w == "B": elo_update(elo, level, "나")

            # 결과 저장
            st.session_state.comp_results.append({
                "round": st.session_state.round,
                "question": q,
                "user_answer": answer,
                "comp_a": {"level": la, "answer": ans_a, "result": res_a},
                "comp_b": {"level": lb, "answer": ans_b, "result": res_b},
            })

            # 다음 라운드 처리
            next_round = st.session_state.round + 1
            st.session_state.round = next_round

            if next_round <= st.session_state.max_rounds:
                with st.spinner("🤔 꼬리질문 생성 중..."):
                    followup = generate_followup(
                        client,
                        st.session_state.q_history,
                        answer,
                        st.session_state.job,
                    )
                st.session_state.q_history.append(followup)
                st.session_state.current_question = followup
                st.session_state.chat.append({"role":"interviewer","text":followup,"q_idx":next_round})
                st.session_state.waiting_answer = True

            st.rerun()

# ─────────────────────────────────────────
# 오른쪽: 경쟁자 답변 + 순위 패널
# ─────────────────────────────────────────
with right:
    results = st.session_state.comp_results

    if not results:
        st.markdown("### 🤖 AI 경쟁자 답변")
        st.markdown(
            '<div class="comp-card" style="color:#8b949e;text-align:center;padding:30px">'
            '답변을 제출하면<br>경쟁자 답변이 여기에 표시됩니다.</div>',
            unsafe_allow_html=True
        )
        st.markdown("### 🏆 순위 및 평가")
        st.markdown(
            '<div class="rank-card" style="color:#8b949e;text-align:center;padding:20px">'
            '평가 결과가 여기에 표시됩니다.</div>',
            unsafe_allow_html=True
        )
    else:
        latest = results[-1]
        la = latest["comp_a"]["level"]
        lb = latest["comp_b"]["level"]
        res_a = latest["comp_a"]["result"]
        res_b = latest["comp_b"]["result"]

        # ── 경쟁자 답변 ──────────────────────────────
        st.markdown("### 🤖 AI 경쟁자 답변")

        if len(results) > 1:
            tab_labels = [f"Q{r['round']}" for r in results]
            tabs = st.tabs(tab_labels)
        else:
            tabs = [st.container()]

        for idx, (tab, rnd) in enumerate(zip(tabs, results)):
            with tab:
                for comp_key, comp_label in [("comp_a","경쟁자 A"), ("comp_b","경쟁자 B")]:
                    c = rnd[comp_key]
                    color = COMP_COLOR[c["level"]]
                    emoji = COMP_EMOJI[c["level"]]
                    st.markdown(
                        f'<div class="comp-card">'
                        f'<span style="font-size:0.7rem;font-weight:700;color:{color}">'
                        f'{emoji} {comp_label} ({c["level"]}수준)</span><br><br>'
                        f'{c["answer"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ── 순위 및 평가 ──────────────────────────────
        st.markdown("### 🏆 순위 및 평가")

        # 최신 라운드 점수 계산
        my_score_a = latest["comp_a"]["result"]["answer_a"]["normalized"]
        my_score_b = latest["comp_b"]["result"]["answer_a"]["normalized"]
        my_score = round((my_score_a + my_score_b) / 2, 1)

        ca_score = latest["comp_a"]["result"]["answer_b"]["normalized"]
        cb_score = latest["comp_b"]["result"]["answer_b"]["normalized"]

        elo = st.session_state.elo
        entries = [
            ("나", my_score, elo.get("나", 1000), "#4f46e5"),
            (f"경쟁자A ({la})", ca_score, elo.get(la, 1000), COMP_COLOR[la]),
            (f"경쟁자B ({lb})", cb_score, elo.get(lb, 1000), COMP_COLOR[lb]),
        ]
        entries.sort(key=lambda x: x[2], reverse=True)

        rank_cls = ["rank-gold","rank-silver","rank-bronze","rank-low"]
        medals = ["🥇","🥈","🥉","4️⃣"]

        for i, (name, score, elo_score, color) in enumerate(entries):
            st.markdown(
                f'<div class="rank-card {rank_cls[min(i,3)]}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span>{medals[i]} <b style="color:{color}">{name}</b></span>'
                f'<span style="font-size:1.1rem;font-weight:700;color:{color}">{score}점</span>'
                f'</div>'
                f'<div style="font-size:0.72rem;color:#8b949e;margin-top:2px">Elo: {elo_score}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # 내 피드백
        st.markdown('<div class="section-label">📝 내 답변 피드백</div>', unsafe_allow_html=True)
        fb_a = res_a["answer_a"].get("피드백","")
        fb_b = res_b["answer_a"].get("피드백","")
        combined_fb = fb_a if fb_a else fb_b
        if combined_fb:
            st.info(combined_fb)

        # 항목별 점수바
        st.markdown('<div class="section-label">📊 항목별 점수 (최신 라운드)</div>', unsafe_allow_html=True)
        score_dict_a = {k: res_a["answer_a"][k] for k in SCORE_KEYS}
        score_dict_b = res_b["answer_a"]

        avg_scores = {k: round((score_dict_a[k] + score_dict_b.get(k,0))/2, 1) for k in SCORE_KEYS}

        for k, v in avg_scores.items():
            pct = int(v / 5 * 100)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;'
                f'color:#8b949e;margin-bottom:2px"><span>{k}</span>'
                f'<span style="color:#818cf8;font-weight:700">{v}/5</span></div>'
                f'<div class="bar-bg"><div class="bar-fill" '
                f'style="width:{pct}%;background:#4f46e5"></div></div>',
                unsafe_allow_html=True
            )

        # 경쟁자 판정 이유
        with st.expander("⚖️ Judge 판정 상세"):
            st.markdown(f"**vs 경쟁자A ({la}수준)**")
            st.write(res_a.get("reason",""))
            st.markdown(f"**vs 경쟁자B ({lb}수준)**")
            st.write(res_b.get("reason",""))
