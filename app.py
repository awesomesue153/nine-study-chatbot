import streamlit as st, json, pandas as pd
from serpapi import GoogleSearch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# — 0) 설정 로드 —
with open("config.json")     as f: menu_cfg = json.load(f)
with open("publishers.json") as f: pub_cfg  = json.load(f)

# — 1) CSV → content 사전 —
concepts_df = pd.read_csv("concepts.csv")
problems_df = pd.read_csv("problems.csv")
self_df     = pd.read_csv("self_check.csv")
tips_df     = pd.read_csv("exam_tips.csv")

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
    .get_dict().get("organic_results",[])

# — 3) RAG 체인 초기화 —
#   이미 생성된 rag_index 폴더 필요
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_store = FAISS.load_local("rag_index", emb)
gen_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    max_new_tokens=128, temperature=0.7)
llm_rag = HuggingFacePipeline(pipeline=gen_pipe)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_rag,
    retriever=rag_store.as_retriever(search_kwargs={"k":3})
)

# — 4) UI 시작 —
st.set_page_config(page_title="나인스터디 챗봇", layout="wide")
st.title("🧑‍🎓 나인스터디 챗봇")
st.write("나인이에게 ‘스터디위드미? 스윗미!’ 해 보세요 😊")

mode = st.radio("원하시는 서비스를 선택하세요",
                ["레벨테스트 받기","학습 및 질문하기"])
if mode=="레벨테스트 받기":
    st.info("레벨테스트 기능은 준비 중입니다.")
    st.stop()

# — 5) 학습 및 질문하기 분기 —
path=["학습 및 질문하기"]
def step(opts,label):
    choice = st.selectbox(label, list(opts.keys()))
    path.append(choice)
    return opts[choice]

opts1 = menu_cfg["학습 및 질문하기"]
opts2 = step(opts1, "대상 선택")
opts3 = step(opts2, "세부 선택")
if isinstance(opts3, dict) and "분류" in opts3:
    cat = st.selectbox("분류 선택", opts3["분류"])
    path.append(cat)
    pubs   = pub_cfg[path[-2]][cat]
    pub    = st.selectbox("교재 선택", list(pubs.keys()))
    path.append(pub)
    unit   = st.selectbox("과 선택", pubs[pub])
    path.append(unit)
else:
    unit = path[-1]

# — 6) 콘텐츠 화면 & 하이브리드 QA —
uid  = unit.replace(" ","_")
data = content.get(uid, {})

st.header(f"🔖 {unit}")
st.write("---")

# 6-1) 개념 설명 + 하이브리드 질문
if st.button("1️⃣ 개념을 자세히 설명해줘요"):
    st.markdown(data.get("concept","준비 중입니다."))
    st.write("---")
    st.write("❓ 질문 유형을 선택하세요:")
    mode2 = st.radio("", ["교재 범위 질문","심화 질문(웹 검색)"], horizontal=True)
    q = st.text_input("질문 입력", key="hybrid_q")
    if q:
        if mode2=="교재 범위 질문":
            res = rag_chain({"question":q, "chat_history":[]})
            st.markdown(res["answer"])
        else:
            results = web_search(q)[:3]
            for r in results:
                st.markdown(f"**{r['title']}**\n{r['snippet']}\n[{r['link']}]")
    st.write("---")

# 6-2) 문제 풀기
if st.button("2️⃣ 해당 단원 문제를 풀고 싶어요"):
    for p in data.get("problems", []):
        ans = st.radio(p["question"], eval(p["choices"]), key=p["q_id"])
        if st.button("제출", key=p["q_id"]):
            st.success("✔ 정답!" if ans==p["answer"] else "❌ 오답!")
    st.write("---")

# 6-3) 내 실력 체크하기
if st.button("3️⃣ 내 실력을 체크하고 싶어요"):
    for sc in data.get("self_check", []):
        resp = st.text_input(sc["question"], key=sc["question"])
        if st.button("확인", key=sc["question"]+"_chk"):
            st.write("정답:", sc["answer"])
    st.write("---")

# 6-4) 시험에 나올 포인트
if st.button("4️⃣ 시험에 나올 포인트 알려줘요"):
    for tip in data.get("exam_tips", []):
        st.write("•", tip)
    st.write("---")

# — 7) 추천 공유하기 —
if st.button("🔗 추천 공유하기"):
    st.success("https://n9study.example.com  를 친구에게 공유하세요!")
