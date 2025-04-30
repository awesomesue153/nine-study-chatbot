import os, streamlit as st
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)                                   # 토큰 로그인
# ⬇︎ 키워드 인자 제거
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)


import os, json
import streamlit as st
import csv, pandas as pd
from huggingface_hub import login
from serpapi import GoogleSearch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── 0) Hugging Face 토큰 로그인 ─────────────────────────
hf_token = os.getenv("HF_TOKEN")  # Streamlit Secrets 에 저장한 키
if hf_token:
    login(hf_token)

# ── 1) 설정 파일 로드 ──────────────────────────────────
with open("config.json", encoding="utf-8")     as f:
    menu_cfg = json.load(f)
with open("publishers.json", encoding="utf-8") as f:
    pub_cfg  = json.load(f)

# ── 2) CSV → content 딕셔너리 ─────────────────────────
concepts_df = pd.read_csv("concepts.csv")
problems_df = pd.read_csv("problems.csv")
self_df     = pd.read_csv("self_check.csv")

# 🔸 exam_tips.csv (unit_id, tip) 는 쉼표가 섞여 있어 수동 파싱
tips = []
with open("exam_tips.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)        # 헤더 건너뛰기 (없으면 자동 무시)
    for row in reader:
        if not row:
            continue
        unit_id  = row[0]
        tip_text = ",".join(row[1:])     # 쉼표 갯수 상관없이 뒤를 전부 결합
        tips.append({"unit_id": unit_id, "tip": tip_text})
tips_df = pd.DataFrame(tips)

content = {}
for uid, grp in concepts_df.groupby("unit_id"):
    content[uid] = {"concept": grp["concept"].iloc[0]}
for uid, grp in problems_df.groupby("unit_id"):
    content.setdefault(uid, {})["problems"] = grp.to_dict(orient="records")
for uid, grp in self_df.groupby("unit_id"):
    content.setdefault(uid, {})["self_check"] = grp.to_dict(orient="records")
for uid, grp in tips_df.groupby("unit_id"):
    content.setdefault(uid, {})["exam_tips"] = grp["tip"].tolist()

# ── 3) SerpAPI 검색 함수 ───────────────────────────────
API_KEY = os.getenv("SERPAPI_API_KEY", "demo")  # demo 키는 제한됨

def web_search(query: str):
    params = {"engine": "google", "q": query, "api_key": API_KEY}
    return GoogleSearch(params).get_dict().get("organic_results", [])

# ── 4) RAG 체인 초기화 ─────────────────────────────────
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # 20 MB 모델
    huggingfacehub_api_token=hf_token
)
rag_store = FAISS.load_local("rag_index", emb, allow_dangerous_deserialization=True)

gen_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.7,
)
llm_rag = HuggingFacePipeline(pipeline=gen_pipe)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_rag,
    retriever=rag_store.as_retriever(search_kwargs={"k": 3})
)

# ── 5) Streamlit UI ────────────────────────────────────
st.set_page_config(page_title="나인스터디 챗봇", layout="wide")
st.title("🧑‍🎓 나인스터디 챗봇")
st.write("나인이에게 ‘스터디위드미? 스윗미!’ 해 보세요 😊")

mode = st.radio("원하시는 서비스를 선택하세요", ["레벨테스트 받기", "학습 및 질문하기"])
if mode == "레벨테스트 받기":
    st.info("레벨테스트 기능은 준비 중입니다.")
    st.stop()

# ── 6) 학습 & 질문 경로 탐색 ──────────────────────────
path = ["학습 및 질문하기"]

def step(opts: dict, label: str):
    choice = st.selectbox(label, list(opts.keys()))
    path.append(choice)
    return opts[choice]

opts1 = menu_cfg["학습 및 질문하기"]
opts2 = step(opts1, "대상 선택")
opts3 = step(opts2, "세부 선택")
if isinstance(opts3, dict) and "분류" in opts3:
    cat = st.selectbox("분류 선택", opts3["분류"])
    path.append(cat)
    pubs = pub_cfg[path[-2]][cat]
    pub  = st.selectbox("교재 선택", list(pubs.keys()))
    path.append(pub)
    unit = st.selectbox("과 선택", pubs[pub])
    path.append(unit)
else:
    unit = path[-1]

# ── 7) 콘텐츠 렌더 + 하이브리드 QA ────────────────────
uid = unit.replace(" ", "_")
data = content.get(uid, {})

st.header(f"🔖 {unit}")
st.write("---")

if st.button("1️⃣ 개념 자세히 설명"):
    st.markdown(data.get("concept", "준비 중입니다."))
    st.write("---")
    qtype = st.radio("질문 범위", ["교재 범위", "웹 심화"], horizontal=True)
    q = st.text_input("질문 입력", key="hybrid_q")
    if q:
        if qtype == "교재 범위":
            res = rag_chain({"question": q, "chat_history": []})
            st.markdown(res["answer"])
        else:
            for r in web_search(q)[:3]:
                st.markdown(f"**{r['title']}**\n{r['snippet']}\n[{r['link']}]")
    st.write("---")

if st.button("2️⃣ 단원 문제 풀기"):
    for p in data.get("problems", []):
        ans = st.radio(p["question"], eval(p["choices"]), key=p["q_id"])
        if st.button("제출", key=p["q_id"]):
            st.success("✔ 정답!" if ans == p["answer"] else "❌ 오답!")
    st.write("---")

if st.button("3️⃣ 실력 체크"):
    for sc in data.get("self_check", []):
        resp = st.text_input(sc["question"], key=sc["question"])
        if st.button("확인", key=sc["question"] + "_chk"):
            st.write("정답:", sc["answer"])
    st.write("---")

if st.button("4️⃣ 시험 포인트"):
    for tip in data.get("exam_tips", []):
        st.write("•", tip)
    st.write("---")

if st.button("🔗 추천 공유하기"):
    st.success("https://n9study.example.com  를 친구에게 공유하세요!")