###############################################################################
#  나인스터디 챗봇 v0.3.5  –  RAG + Web 하이브리드  (모바일 친화 레이아웃)
#  · 캐싱 : CSV / FAISS / LLM            · DEV↔PROD 스위치  (USE_SMALL_LLM)
#  · UI   : Stage 머신 (0 서비스→1 네비→2 학습) + 상단 고정 홈버튼 + 토스트 안내
###############################################################################
import os, json, csv, logging
from pathlib import Path
import streamlit as st
import pandas as pd

logging.basicConfig(level=logging.WARN)

# ────────────────────────────── utils
def parse_choices(raw: str) -> list[str]:
    raw = raw.strip()

    # ① JSON 표준 (더블쿼트) → 바로 리턴
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return v
    except Exception:
        pass

    # ② 대괄호 감싸기 제거  […],  ['a','b']
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]

    # ③ 쉼표 분리 후, 남은 따옴표 / 공백 제거
    return [s.strip().strip("'\"")          # 바깥 ', " 제거
            for s in raw.split(",")
            if s.strip()]

# ────────────────────────────── 페이지 설정
st.set_page_config(page_title="나인스터디 챗봇", layout="wide")

# ────────────────────────────── 고정 TOP Bar
top   = st.container()
col_btn, _ = top.columns([1, 9], gap="small")   # _ : unused spacer

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button{
        position:sticky;top:6px;z-index:998;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────── 설정 JSON
with open("config.json", encoding="utf-8")     as f:
    menu_cfg = json.load(f)
with open("publishers.json", encoding="utf-8") as f:
    pub_cfg  = json.load(f)

# ────────────────────────────── CSV 로딩
CSV_PATHS = [Path(p) for p in
             ("concepts.csv", "problems.csv", "self_check.csv", "exam_tips.csv")]

@st.cache_data(show_spinner="📂 CSV 로딩 중…",
               hash_funcs={Path: lambda p: p.stat().st_mtime})
def load_csvs() -> dict[str, pd.DataFrame]:
    c, p, s = [pd.read_csv(p) for p in CSV_PATHS[:3]]

    tips = []
    with CSV_PATHS[3].open(encoding="utf-8") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            tips.append({"unit_id": row[0], "tip": ",".join(row[1:])})
    t = pd.DataFrame(tips)
    return {"concepts": c, "problems": p, "selfcheck": s, "tips": t}

dfs = load_csvs()

# ────────────────────────────── content dict
def build_content(d: dict[str, pd.DataFrame]) -> dict:
    c = {}
    for uid, g in d["concepts"].groupby("unit_id"):
        c[uid] = {"concept": g["concept"].iloc[0]}
    for uid, field in (("problems", "problems"),
                       ("selfcheck", "self_check")):
        for u, g in d[uid].groupby("unit_id"):
            c.setdefault(u, {})[field] = g.to_dict("records")
    for uid, g in d["tips"].groupby("unit_id"):
        c.setdefault(uid, {})["exam_tips"] = g["tip"].tolist()
    return c

content = build_content(dfs)

# ────────────────────────────── RAG + LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (pipeline, AutoTokenizer,
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM)
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain

@st.cache_resource(show_spinner="🧠 모델·인덱스 초기화…")
def init_rag_chain():
    emb      = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    idx_path = Path("rag_index")

    if idx_path.exists():
        store = FAISS.load_local(idx_path, emb, allow_dangerous_deserialization=True)
    else:
        if not Path("chunks.csv").exists():
            st.error("❗ chunks.csv 가 없습니다. RAG 인덱스를 만들 수 없습니다.")
            st.stop()
        chunks_df = pd.read_csv("chunks.csv")
        store     = FAISS.from_texts(chunks_df["chunk"].tolist(), emb)
        store.save_local(idx_path)

    MODEL_ID = ("google/flan-t5-small"
                if os.getenv("USE_SMALL_LLM")
                else "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    if "t5" in MODEL_ID:                       # seq‑to‑seq
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl  = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        task = "text2text-generation"
    else:                                      # causal
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl  = AutoModelForCausalLM.from_pretrained(
                  MODEL_ID, device_map="auto", torch_dtype="auto")
        task = "text-generation"

    llm_pipe = pipeline(task, model=mdl, tokenizer=tok,
                        max_new_tokens=96, temperature=0.6)
    llm      = HuggingFacePipeline(pipeline=llm_pipe)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=store.as_retriever(search_kwargs={"k": 3})
    )

rag_chain = init_rag_chain()

# ────────────────────────────── SerpAPI
from dotenv import load_dotenv
load_dotenv()                 # .env → 환경변수 등록

SERP_KEY = os.getenv("SERPAPI_KEY", "")
if SERP_KEY:
    from serpapi import GoogleSearch
    # ① 순수 호출 함수 (캐싱 X)
    def _raw_search(q: str):
        return (
            GoogleSearch(
                {
                    "engine": "google",
                    "q":      q,
                    "api_key": SERP_KEY,
                    "num":    2    # ← 결과 2개만
                }
            )
            .get_dict()
            .get("organic_results", [])
        )
    # ② 1시간 캐싱 래퍼
    @st.cache_data(ttl=3600, show_spinner="🔍 검색 중…")
    def web_search(q: str):
        return _raw_search(q)
else:
    web_search = lambda q: []

# ────────────────────────────── Stage 머신
if "stage" not in st.session_state:
    st.session_state.stage = 0               # 0 서비스 · 1 네비 · 2 학습
if "nav" not in st.session_state:
    st.session_state.nav = {}

# ↩️ Home 버튼 (콜백에서 rerun 제거)
def reset_app():
    st.session_state.stage = 0
    st.session_state.nav.clear()

with col_btn:
    st.button("↩️ 처음", on_click=reset_app, use_container_width=True)

###############################################################################
#  STAGE 0 ─ 서비스 선택
###############################################################################
if st.session_state.stage == 0:
    st.title("🧑‍🎓 나인스터디 챗봇")
    svc = st.radio("서비스 선택",
                   ["레벨테스트 받기", "학습 및 질문하기"],
                   key="svc_choice", horizontal=True)
    if svc == "레벨테스트 받기":
        st.info("🚧 레벨테스트 기능은 준비 중입니다.")
    if st.button("다음 ▶", use_container_width=True):
        if svc == "학습 및 질문하기":
            st.session_state.stage = 1
        st.session_state.pop("svc_choice", None)
        st.rerun()

###############################################################################
#  STAGE 1 ─ 네비게이션
###############################################################################
if st.session_state.stage == 1:
    nav = st.session_state.nav

    # 1) 대상
    if "target" not in nav:
        nav["target"] = st.selectbox(
            "대상 선택", list(menu_cfg["학습 및 질문하기"].keys()))
        st.stop()

    # 2) 세부
    tgt_cfg = menu_cfg["학습 및 질문하기"][nav["target"]]
    if "detail" not in nav:
        nav["detail"] = st.selectbox("세부 선택", list(tgt_cfg.keys()))
        st.stop()

    # 3) (옵션) 분류
    detail_cfg = tgt_cfg[nav["detail"]]
    if "cat" not in nav and "분류" in detail_cfg:
        nav["cat"] = st.selectbox("분류 선택", detail_cfg["분류"])
        st.stop()

    # 4) 교재
    pubs = (pub_cfg.get(nav["detail"], {})
                     .get(nav.get("cat", ""), {}))
    if not pubs:
        st.error("📚 교재 정보가 없습니다."); st.stop()

    if "pub" not in nav:
        nav["pub"] = st.selectbox("교재 선택", list(pubs.keys()))
        st.stop()

    # 5) 단원
    units = pubs.get(nav["pub"], [])
    if not units:
        st.error("📑 단원 목록이 없습니다."); st.stop()

    nav["unit"] = st.selectbox("과 선택", units)

    if st.button("학습 화면으로 ▶", use_container_width=True):
        st.session_state.stage = 2
        st.rerun()

###############################################################################
#  STAGE 2 ─ 단원 학습
###############################################################################
# ───── Stage 2 ─ 단원 학습 화면 ─────
if st.session_state.stage == 2:
    nav = st.session_state.nav
    uid = nav["unit"].replace(" ", "_")
    data = content.get(uid, {})

    st.header(f"📑 {nav['unit']}")
    st.divider()

    # ---------- 1️⃣ 개념 설명 ----------
    # ❶ 클릭 시 세션 플래그 ON
    def open_concept():
        st.session_state.concept_open = True

    st.button("1️⃣ 개념 설명", key="btn_concept",
              on_click=open_concept, use_container_width=True)

    # ❷ 플래그가 켜져 있으면 항상 표시
    if st.session_state.get("concept_open"):
        st.markdown(data.get("concept", "준비 중입니다."))
        q = st.text_input("추가 질문 :", key="concept_q")
        if q:
            if st.toggle("웹 심화 질문으로 전환", False, key="tg_web"):
                for r in web_search(q):
                    st.write("🔗", r["title"], "→", r["link"])
                    st.caption(r["snippet"])
            else:
                try:
                    ans = rag_chain({"question": q, "chat_history": []})["answer"]
                    st.write(ans)
                except Exception as e:
                    st.error("질문 처리 실패"); logging.error(e, exc_info=True)
        st.divider()

    # 2️⃣ 단원 문제
    if st.button("2️⃣ 단원 문제", use_container_width=True):
        for p in data.get("problems", []):
            key = f"{uid}_{p['q_id']}"
            ans = st.radio(p["question"], parse_choices(p["choices"]), key=key)
            if st.button("제출", key=f"chk_{key}"):
                st.success("✅ 정답!" if ans == p["answer"] else "❌ 오답")
        st.divider()

    # 3️⃣ 셀프 체크
    if st.button("3️⃣ 셀프 체크", use_container_width=True):
        for s in data.get("self_check", []):
            r = st.text_input(s["question"], key=s["question"])
            if st.button("정답 확인", key=f"self_{s['question']}"):
                st.write("정답:", s["answer"])
        st.divider()

    # 4️⃣ 시험 포인트
    if st.button("4️⃣ 시험 포인트", use_container_width=True):
        for t in data.get("exam_tips", []):
            st.write("•", t)
        st.divider()

    # 🔗 공유
    if st.button("🔗 추천 링크 복사", use_container_width=True):
        st.success("https://n9study.example.com 를 복사해 친구에게 보내세요!")
