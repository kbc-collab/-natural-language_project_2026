import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="AI Interview Arena",
    page_icon="🏟️",
    layout="wide"
)

st.title("🏟️ AI Interview Arena")
st.caption("Multi-Agent 기반 경쟁형 AI 면접 시뮬레이션 시스템")

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_step2.csv")

df = load_dataset()

st.sidebar.header("면접 설정")

category_col = "카테고리" if "카테고리" in df.columns else None
difficulty_col = "난이도" if "난이도" in df.columns else None
question_col = "질문" if "질문" in df.columns else df.columns[0]

if category_col:
    categories = ["전체"] + sorted(df[category_col].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("카테고리", categories)

    if selected_category != "전체":
        df_view = df[df[category_col] == selected_category]
    else:
        df_view = df.copy()
else:
    df_view = df.copy()

if difficulty_col:
    difficulties = ["전체"] + sorted(df_view[difficulty_col].dropna().unique().tolist())
    selected_difficulty = st.sidebar.selectbox("난이도", difficulties)

    if selected_difficulty != "전체":
        df_view = df_view[df_view[difficulty_col] == selected_difficulty]

question = st.selectbox("면접 질문 선택", df_view[question_col].tolist())

row = df_view[df_view[question_col] == question].iloc[0]

st.subheader("면접 질문")
st.info(question)

user_answer = st.text_area(
    "내 답변 입력",
    placeholder="여기에 면접 답변을 입력하세요.",
    height=180
)

st.divider()

if st.button("AI 경쟁 면접 평가 시작", type="primary"):
    if not user_answer.strip():
        st.warning("먼저 답변을 입력해주세요.")
        st.stop()

    st.subheader("AI 경쟁자 답변")

    col1, col2, col3 = st.columns(3)

    low_col = "하 답변"
    mid_col = "중 답변"
    high_col = "상 답변"

    with col1:
        st.markdown("### 🥉 하 수준 경쟁자")
        st.write(row[low_col] if low_col in df.columns else "하 답변 컬럼 없음")

    with col2:
        st.markdown("### 🥈 중 수준 경쟁자")
        st.write(row[mid_col] if mid_col in df.columns else "중 답변 컬럼 없음")

    with col3:
        st.markdown("### 🥇 상 수준 경쟁자")
        st.write(row[high_col] if high_col in df.columns else "상 답변 컬럼 없음")

    st.divider()
    st.subheader("평가 결과")

    st.info(
        "현재는 Streamlit 통합 1차 버전입니다. "
        "다음 단계에서 judge/evaluator.py 또는 app/agents/judge_agent.py를 연결해 "
        "점수, 순위, 피드백을 자동 출력합니다."
    )

    with st.expander("내 답변 보기"):
        st.write(user_answer)

    if "좋은 답변 기준" in df.columns:
        with st.expander("좋은 답변 기준"):
            st.write(row["좋은 답변 기준"])
