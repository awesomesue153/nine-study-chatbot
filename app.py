import os
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

# ★ 여기서 모델·토큰 모두 명시
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",   # ← 더 작은 버전
    huggingfacehub_api_token=hf_token
)



import streamlit as st
import json
import pandas as pd
import csv
from serpapi import GoogleSearch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# — 0) 설정 파일 로드 —
with open("config.json", "r", encoding="utf-8") as f:
    menu_cfg = json.load(f)
with open("publishers.json", "r", encoding="utf-8") as f:
    pub_cfg = json.load(f)

# — 1) CSV → content 사전 생성 —
concepts_df = pd.read_csv("concepts.csv")
problems_df = pd.read_csv("problems.csv")
self_df     = pd.read_csv("self_check.csv")

# exam_tips.csv 수동 파싱
tips = []
with open("exam_tips.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if not row:
            continue
        unit_id  = row[0]
        tip_text = ",".join(row[1:])
        tips.append({"unit_id": unit_id, "tip": tip_text})
tips_df = pd.DataFrame(tips)

# content dict 구성
content = {}
for uid, grp in concepts_df.groupby("unit_id"):
    content[uid] = {"concept": grp["concept"].iloc[0]}
for uid, grp in problems_df.groupby("unit_id"):
    content.setdefault(uid, {})["problems"] = grp.to_dict(orient="records")
for uid, grp in self_df.groupby("unit_id"):
    content.setdefault(uid, {})["self_check"] = grp.to_dict(orient="records")
for uid, grp in tips_df.groupby("unit_id"):
    content.setdefault(uid, {})["exam_tips"] = grp["tip"].tolist()

# — 2) SerpAPI 설정 —
API_KEY = "YOUR_SERPAPI_API_KEY"
def web_search(query):
    return GoogleSearch({"engine":"google","q":query,"api_key":API_KEY}) \
           .get_dict().get("organic_results", [])

# — 3) RAG 초기화 —
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_store = FAISS.load_local("rag_index", emb, allow_dangerous_deserialization=True)
gen_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.7
)
llm_rag = HuggingFacePipeline(pipeline=gen_pipe)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_rag,
    retriever=rag_store.as_retriever(search_kwargs={"k":3})
)

# — 4) Streamlit 초기 설정 —
st.set_page_config(page_title="나인스터디 챗봇", layout="wide")
if "step" not in st.session_state:
    st.session_state.step = 0
if "path" not in st.session_state:
    st.session_state.path = []

# — 상단 네비게이션 바 —
col1, col2, col3 = st.columns([1,6,1])
with col1:
    if st.button("🏠 홈"):
        st.session_state.step = 0
        st.session_state.path.clear()
        st.rerun()
with col3:
    if st.session_state.step > 0:
        if st.button("🔙 뒤로가기"):
            st.session_state.step -= 1
            if st.session_state.path:
                st.session_state.path.pop()
            st.rerun()

st.title("🧑‍🎓 나인스터디 챗봇")
st.write("나인이에게 ‘스터디위드미? 스윗미!’ 해 보세요 😊")

# — 5) 단계별 화면 분기 —
if st.session_state.step == 0:
    st.header("메인 메뉴")
    st.write("원하시는 서비스를 선택하세요.")
    if st.button("🔍 레벨테스트 받기"):
        st.info("레벨테스트 기능은 준비 중입니다.")
    if st.button("📚 학습 및 질문하기"):
        st.session_state.step = 1
        st.rerun()

elif st.session_state.step == 1:
    opts1 = menu_cfg["학습 및 질문하기"]
    sel = st.selectbox("대상 선택", list(opts1.keys()))
    if st.button("선택"):
        st.session_state.path = ["학습 및 질문하기", sel]
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    opts2 = menu_cfg["학습 및 질문하기"][st.session_state.path[1]]
    sel = st.selectbox("세부 선택", list(opts2.keys()))
    if st.button("선택"):
        st.session_state.path.append(sel)
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    last_opts = menu_cfg["학습 및 질문하기"][st.session_state.path[1]][st.session_state.path[2]]
    if isinstance(last_opts, dict) and "분류" in last_opts:
        sel = st.selectbox("분류 선택", last_opts["분류"])
        if st.button("선택"):
            st.session_state.path.append(sel)
            st.session_state.step = 4
            st.rerun()
    else:
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    grade = st.session_state.path[1]
    cat   = st.session_state.path[3] if len(st.session_state.path) > 3 else None
    pubs  = pub_cfg.get(grade, {}).get(cat, {})
    sel   = st.selectbox("교재 선택", list(pubs.keys()))
    if st.button("선택"):
        st.session_state.path.append(sel)
        st.session_state.step = 5
        st.rerun()

elif st.session_state.step == 5:
    grade, cat, pub = st.session_state.path[1], st.session_state.path[3], st.session_state.path[4]
    units = pub_cfg[grade][cat][pub]
    sel   = st.selectbox("과 선택", units)
    if st.button("선택"):
        st.session_state.path.append(sel)
        st.session_state.step = 6
        st.rerun()

elif st.session_state.step == 6:
    unit = st.session_state.path[-1]
    uid  = unit.replace(" ", "_")
    data = content.get(uid, {})

    st.header(f"🔖 {unit}")
    st.write("---")

    # 1️⃣ 개념 설명 + 하이브리드 QA
    if st.button("1️⃣ 개념 설명"):
        st.markdown(data.get("concept","준비 중입니다."))
        st.write("---")
        mode2 = st.radio("❓ 질문 유형 선택", ["교재 범위 질문","심화 질문(웹 검색)"], horizontal=True)
        q = st.text_input("질문 입력", key="hybrid_q")
        if q:
            if mode2 == "교재 범위 질문":
                res = rag_chain({"question":q,"chat_history":[]})
                st.markdown(res["answer"])
            else:
                for r in web_search(q)[:3]:
                    st.markdown(f"**{r['title']}**\n{r['snippet']}\n[{r['link']}]")
        st.write("---")

    # 2️⃣ 문제 풀기
    if st.button("2️⃣ 문제 풀기"):
        for p in data.get("problems", []):
            ans = st.radio(p["question"], eval(p["choices"]), key=p["q_id"])
            if st.button("제출", key=p["q_id"]):
                st.success("✔ 정답!" if ans == p["answer"] else "❌ 오답!")
        st.write("---")

    # 3️⃣ 실력 체크
    if st.button("3️⃣ 실력 체크"):
        for sc in data.get("self_check", []):
            resp = st.text_input(sc["question"], key=sc["question"])
            if st.button("확인", key=sc["question"] + "_chk"):
                st.write("정답:", sc["answer"])
        st.write("---")

    # 4️⃣ 시험 팁
    if st.button("4️⃣ 시험 팁"):
        for tip in data.get("exam_tips", []):
            st.write("•", tip)
        st.write("---")

    # 🔗 공유하기
    if st.button("🔗 공유하기"):
        st.success("https://n9study.example.com 를 친구에게 공유하세요!")
