###############################################################################
#  나인스터디 챗봇  v0.6.1  –  RAG + Web 하이브리드 (모바일 친화 레이아웃)
#  • 캐싱 :  CSV ⟶ @cache_data      ·   FAISS Index·LLM ⟶ @cache_resource
#  • 스위치 :  DEV(Flan‑T5‑small) ←→ PROD(TinyLlama‑1.1B)   |  USE_SMALL_LLM=1
#  • UI  :  Stage 머신 (0 서비스 → 1 네비 → 2 학습) + 상단 고정 Home 버튼
#            └─ 단원 문제 👉 « 1 문항 표시 + Submit → Next 루프 »
#  • v0.6.1 :  문제 섹션 진입 시 남은 채점 상태(prob_check) 초기화 버그 Fix
###############################################################################

import os, json, csv, logging
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv; load_dotenv()

logging.basicConfig(level=logging.WARN)

# ───────────────────────── utils
def parse_choices(raw: str) -> list[str]:
    raw = raw.strip()
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [s.strip().strip("'\"") for s in raw.split(",") if s.strip()]

# ───────────────────────── 페이지 & TOP Bar
st.set_page_config(page_title="나인스터디 챗봇", layout="wide")
top = st.container()
col_btn, _ = top.columns([1, 9], gap="small")
st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button{
        position:sticky;top:6px;z-index:998;}
    h2 {font-size:28px !important;}   /* 단원 타이틀 28 px 공통 적용 */
    </style>""", unsafe_allow_html=True)

# ───────────────────────── 설정(JSON) · CSV
with open("config.json", encoding="utf-8") as f: menu_cfg = json.load(f)
with open("publishers.json", encoding="utf-8") as f: pub_cfg = json.load(f)

CSV_PATHS = [Path(p) for p in
             ("concepts.csv", "problems.csv", "self_check.csv", "exam_tips.csv")]
@st.cache_data(show_spinner="📂 CSV 로딩 중…",
               hash_funcs={Path: lambda p: p.stat().st_mtime})
def load_csvs():
    c, p, s = [pd.read_csv(p) for p in CSV_PATHS[:3]]
    tips_rows=[]
    with CSV_PATHS[3].open(encoding="utf-8") as f:
        r=csv.reader(f); next(r, None)
        for row in r: tips_rows.append({"unit_id":row[0],"tip":",".join(row[1:])})
    t=pd.DataFrame(tips_rows)
    return {"concepts":c,"problems":p,"selfcheck":s,"tips":t}
dfs = load_csvs()

def build_content(d):
    c={}
    for uid,g in d["concepts"].groupby("unit_id"):
        c[uid]={"concept":g["concept"].iloc[0]}
    for src,field in (("problems","problems"),):
        for uid,g in d[src].groupby("unit_id"):
            c.setdefault(uid,{})[field]=g.to_dict("records")
    for uid,g in d["tips"].groupby("unit_id"):
        c.setdefault(uid,{})["exam_tips"]=g["tip"].tolist()
    return c
content = build_content(dfs)

# ───────────────────────── RAG + LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# 수정된 프롬프트 (질문-컨텍스트-응답 구조)
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""{context}"""
)

@st.cache_resource(show_spinner="🧠 모델·인덱스 초기화…")
def init_rag_chain():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    store = FAISS.load_local("rag_index", emb, allow_dangerous_deserialization=True)

    MODEL_ID = "google/flan-t5-small" if os.getenv("USE_SMALL_LLM") else \
               "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if "t5" in MODEL_ID:
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl  = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        task = "text2text-generation"
    else:
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl  = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
        task = "text-generation"

    # 토큰 & 생성 매개변수 조정
    gen = pipeline(
        task,
        model=mdl,
        tokenizer=tok,
        max_new_tokens=128,  # 60 → 128로 증가
        temperature=0.7,
        repetition_penalty=1.2,  # 반복 억제 강화
        top_p=0.9
    )

    llm = HuggingFacePipeline(pipeline=gen)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )
    return chain

rag_chain = init_rag_chain()

# ───────────────────────── SerpAPI (옵션)
SERP_KEY=os.getenv("SERPAPI_KEY","")
if SERP_KEY:
    from serpapi import GoogleSearch
    @st.cache_data(ttl=3600,show_spinner="🔍 검색 중…")
    def web_search(q):
        return (GoogleSearch({"engine":"google","q":q,
                              "api_key":SERP_KEY,"num":2})
                .get_dict().get("organic_results",[]))
else:
    web_search=lambda q:[]

# ───────────────────────── Stage & Session State
if "stage"    not in st.session_state: st.session_state.stage = 0
if "nav"      not in st.session_state: st.session_state.nav   = {}
if "open_sec" not in st.session_state: st.session_state.open_sec = None
if "prob_idx" not in st.session_state: st.session_state.prob_idx = 0
if "prob_check" not in st.session_state: st.session_state.prob_check = None

def reset_app():
    st.session_state.stage=0
    st.session_state.nav.clear()
    for k in ("open_sec","prob_idx","prob_check"):
        st.session_state[k]= None if k=="open_sec" else 0
with col_btn: st.button("↩️ Home",on_click=reset_app,use_container_width=True)

###############################################################################
#  STAGE 0 ─ 서비스 선택
###############################################################################
if st.session_state.stage==0:
    st.title("🧑‍🎓 나인스터디 챗봇")
    svc=st.radio("서비스 선택",["레벨테스트 받기","학습 및 질문하기"],
                 horizontal=True,key="svc_choice")
    if svc=="레벨테스트 받기": st.info("🚧 레벨테스트 기능은 준비 중입니다.")
    if st.button("다음 ▶",use_container_width=True):
        if svc=="학습 및 질문하기": st.session_state.stage=1
        st.session_state.pop("svc_choice",None); st.rerun()

###############################################################################
#  STAGE 1 ─ 네비게이션
###############################################################################
if st.session_state.stage==1:
    def reset_lower(level: str):
        """level 변경 시 그 아래 단계 state 초기화"""
        order = ["target", "detail", "cat", "pub", "unit"]
        idx   = order.index(level)
        for k in order[idx+1:]:
            st.session_state.nav.pop(k, None)

    nav = st.session_state.nav

    # 1) 대상 (중학교·고등학교 …)
    tgt = st.selectbox(
            "대상 선택",
            list(menu_cfg["학습 및 질문하기"].keys()),
            key="sb_target",
        )
    if nav.get("target") != tgt:
        nav["target"] = tgt
        reset_lower("target")
        st.rerun()

    # 2) 세부 (중1/고1 …)
    tgt_cfg  = menu_cfg["학습 및 질문하기"][nav["target"]]
    det = st.selectbox("세부 선택", list(tgt_cfg.keys()), key="sb_detail")
    if nav.get("detail") != det:
        nav["detail"] = det
        reset_lower("detail")
        st.rerun()

    # 3) (옵션) 분류
    detail_cfg = tgt_cfg[nav["detail"]]
    if "분류" in detail_cfg:
        cat = st.selectbox("분류 선택", detail_cfg["분류"], key="sb_cat")
        if nav.get("cat") != cat:
            nav["cat"] = cat
            reset_lower("cat")
            st.rerun()
    else:
        nav.pop("cat", None)

    # 4) 교재
    pubs_dict = (
        pub_cfg.get(nav["detail"], {})      # "중1" / "고1" …
            .get(nav.get("cat", ""), {}) # "내신" / "수능" …
    )
    if not pubs_dict:
        st.error("📚 교재 정보가 없습니다."); st.stop()

    pub = st.selectbox("교재 선택", list(pubs_dict.keys()), key="sb_pub")
    if nav.get("pub") != pub:
        nav["pub"] = pub
        reset_lower("pub")
        st.rerun()

    # 5) 단원
    units = pubs_dict[nav["pub"]]
    unit  = st.selectbox("과 선택", units, key="sb_unit")
    nav["unit"] = unit

    if st.button("학습 화면으로 ▶", use_container_width=True):
        st.session_state.stage = 2
        st.rerun()


###############################################################################
#  STAGE 2 ─ 단원 학습
###############################################################################
if st.session_state.stage==2:
    nav=st.session_state.nav
    uid=nav["unit"].replace(" ","_")
    data=content.get(uid,{})
    st.header(f"📑 {nav['unit']}")
    # st.divider()

    # ── 버튼 Row
    bcols=st.columns(3)
    bcols[0].button("1️⃣ 개념 설명",use_container_width=True,
        on_click=lambda: st.session_state.__setitem__("open_sec","concept"))
    
    bcols[1].button(
    "2️⃣ 단원 문제",
    use_container_width=True,
    on_click=lambda: st.session_state.update({
        "open_sec": "problems",         # 문제 섹션 열기
        "prob_check": None              # ← 전에 기억해둔 정/오답 값 지우기
        })
    )
    bcols[2].button("3️⃣ 시험 포인트",use_container_width=True,
        on_click=lambda: st.session_state.__setitem__("open_sec","tips"))

    place=st.container()
    st.button("🔗 추천 링크 복사",use_container_width=True,
        on_click=lambda: st.session_state.__setitem__("open_sec","share"))

    # ── 문제 렌더링 헬퍼
    def render_problem(qdata):
        q_id=qdata["q_id"]; q_key=f"{uid}_{q_id}"; chk_key=f"chk_{q_key}"
        ans=st.radio(qdata["question"],parse_choices(qdata["choices"]),
                     key=q_key,index=None)
        if st.button("제출",key=chk_key):
            st.session_state.prob_check = (ans == qdata["answer"])
        if st.session_state.prob_check is not None:
            if st.session_state.prob_check: st.success("✅ 정답!")
            else: st.error("❌ 오답")
        if st.button("다음"):
            st.session_state.prob_idx = (st.session_state.prob_idx + 1) % len(data["problems"])
            st.session_state.prob_check=None
            # 선택 초기화
            st.session_state.pop(q_key,None); st.session_state.pop(chk_key,None)
            st.rerun()

    # ── 섹션
    with place:
        sec=st.session_state.open_sec
        if sec=="concept":
            st.markdown(data.get("concept","준비 중입니다."))
            q=st.text_input("추가 질문 :",key="concept_q")
            if q:
                if st.toggle("웹 심화 질문으로 전환",key="tg_web"):
                    for r in web_search(q):
                        st.write("🔗",r["title"],"→",r["link"]); st.caption(r["snippet"])
                else:
                    try:
                        ans=rag_chain({"question":q,"chat_history":[]})["answer"]
                        st.write(ans)
                    except Exception as e:
                        st.error("질문 처리 실패"); logging.error(e,exc_info=True)

        elif sec=="problems":
            if not data.get("problems"): st.info("문제가 없습니다.")
            else:
                cur=data["problems"][st.session_state.prob_idx]
                render_problem(cur)

        elif sec=="tips":
            for t in data.get("exam_tips",[]): st.write("•",t)

        elif sec=="share":
            st.success("https://ninestudy.co.kr 를 복사해 친구에게 보내세요!")