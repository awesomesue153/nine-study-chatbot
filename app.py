###############################################################################
#  NineStudy Chatbot  ▪︎  v 0.8.0   (2025‑05‑13)
#
#  ► 변경 핵심
#    • NEW  레벨테스트 분기(Stage 9, 99) 추가
#          – Adaptive‑Lite 엔진(20 문항·난이도 자동 조절)
#          – 결과 PDF 다운로드 / SQLite 저장
#    • NEW  세션 상태: lt, lt_block, lt_idx, user_id
#    • NEW  의존성: fpdf2 · matplotlib  (requirements.txt 반영)
#    • NEW  외부 모듈  leveltest.py  로 엔진 / PDF / DB 분리
#
#  ► 폴더/파일
#      leveltest_questions.csv     (25 문항 샘플)
#      leveltest.py                (레벨테스트 로직 전담)
#
#  ► 사용 방법
#      - 서비스 선택 → “레벨테스트 받기” 클릭
#      - 20 문항 완료 → 레벨·섹션 점수·PDF 리포트 확인
#
#  © 2025  Chapter9 — Creative Flow Labs
###############################################################################

import os, json, csv, logging
from leveltest import LevelEngine, ITEMS as LT_ITEMS, make_pdf, save_result
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
import uuid, io

ROOT_DIR = Path(__file__).resolve().parent   # ★ 절대경로 상수

# Hugging Face 토큰 등록 (Streamlit Secret)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

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

# CSV 파일 경로 리스트
CSV_PATHS = ["concepts.csv", "problems.csv", "self_check.csv", "exam_tips.csv"]

@st.cache_data(show_spinner="📂 CSV 로딩 중…",
               hash_funcs={Path: lambda p: p.stat().st_mtime})
def load_csvs():
    # ── app.py가 위치한 'nine-study-chatbot' 폴더 절대 경로
    base_dir = Path(__file__).parent  # :contentReference[oaicite:0]{index=0}

    # 첫 세 개 CSV 읽기
    df_concepts    = pd.read_csv(base_dir / CSV_PATHS[0])
    df_problems    = pd.read_csv(base_dir / CSV_PATHS[1])
    df_self_check  = pd.read_csv(base_dir / CSV_PATHS[2])

    # exam_tips.csv 읽어서 DataFrame 생성
    tips_rows = []
    with (base_dir / CSV_PATHS[3]).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            tips_rows.append({"unit_id": row[0], "tip": ",".join(row[1:])})
    df_tips = pd.DataFrame(tips_rows)

    return {
        "concepts":   df_concepts,
        "problems":   df_problems,
        "selfcheck":  df_self_check,
        "tips":       df_tips
    }

# 실제 로딩
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
from langchain.chat_models import ChatOpenAI   # ← 추가
from langchain.callbacks.base import BaseCallbackHandler   # ← 새 import
from importlib import util

# ── OpenAI 키 · SDK 유효성 체크 ───────────────────────────────────────────
OPENAI_KEY = (
    st.secrets.get("OPENAI_API_KEY")        # secrets.toml / Cloud Secrets
    or os.getenv("OPENAI_API_KEY")          # 컨테이너 ENV
)
HAS_SDK = bool(util.find_spec("openai"))    # 패키지 설치 여부
USE_OPENAI = bool(OPENAI_KEY and HAS_SDK)   # 두 조건 모두 충족해야 사용


# 수정된 프롬프트 (질문-컨텍스트-응답 구조)
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""{context}"""
)

@st.cache_resource(show_spinner="🧠 모델 & 인덱스 로딩…")
def init_rag_chain():
    # 1) 임베딩 · 벡터스토어
    emb   = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    store = FAISS.load_local(str(ROOT_DIR / "rag_index"), emb,
                             allow_dangerous_deserialization=True)

    llm = None   # ← 먼저 초기화해 두어 안전장치

    # 2) OpenAI 시도
    if USE_OPENAI:
        try:
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=OPENAI_KEY,
                streaming=True,
                temperature=0.7,
                max_tokens=256,
                timeout=15,
                max_retries=2,
            )
        except Exception as e:
            logging.warning("⚠️ ChatOpenAI 로드 실패 → TinyLlama 폴백 (%s)", e)

    # 3) TinyLlama 폴백 (llm 이 아직 None 이면)
    if llm is None:
        MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl  = AutoModelForCausalLM.from_pretrained(
                   MODEL_ID, device_map="auto", torch_dtype="auto")
        pipe = pipeline(
            "text-generation",
            model=mdl, tokenizer=tok,
            max_new_tokens=128, temperature=0.7, top_p=0.9,
            repetition_penalty=1.2)
        llm = HuggingFacePipeline(pipeline=pipe)

    # 4) RAG 체인
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

# 레벨테스트 전용 세션값 --------------------------
if "lt"      not in st.session_state: st.session_state.lt       = None   # 엔진 객체
if "lt_block" not in st.session_state: st.session_state.lt_block = []    # 현재 블록
if "lt_idx"   not in st.session_state: st.session_state.lt_idx   = 0     # 블록 내 인덱스
# -----------------------------------------------

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
    st.title("🧐 나인스터디 챗봇")
    svc = st.radio(
            "서비스 선택",
            ["레벨테스트 받기", "학습 및 질문하기"],
            horizontal=True, key="svc_choice")

    if st.button("다음 ▶", use_container_width=True):
        if   svc == "학습 및 질문하기":
            st.session_state.stage = 1
        elif svc == "레벨테스트 받기":
            st.session_state.stage = 9        #  ← 새 레벨테스트 스테이지
        st.session_state.pop("svc_choice", None)
        st.rerun()


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
            # 추가질문 처리 시 안내문구 
            if q:
                with st.status("현재 AI 튜터링 챗봇이 질문에 대해 생각 중입니다.", expanded=False) as status:
                    try:
                        ans = rag_chain({"question": q, "chat_history": []})["answer"]
                        status.update(label="✅ 답변 완료!", state="complete")
                        st.write(ans)
                    except Exception as e:
                        status.update(label="❌ 처리 실패", state="error")
                        st.error("질문 처리 중 문제가 발생했습니다.")

        elif sec=="problems":
            if not data.get("problems"): st.info("문제가 없습니다.")
            else:
                cur=data["problems"][st.session_state.prob_idx]
                render_problem(cur)

        elif sec=="tips":
            for t in data.get("exam_tips",[]): st.write("•",t)

        elif sec=="share":
            st.success("http://ninestudy.co.kr 를 복사해 친구에게 보내세요!")


###############################################################################
#  STAGE 9 ─ 레벨테스트 진행
###############################################################################
if st.session_state.stage == 9:
    st.title("🎯 레벨테스트")

    # 초기화
    if st.session_state.lt is None:
        st.session_state.user_id  = str(uuid.uuid4())[:8]
        st.session_state.lt       = LevelEngine(LT_ITEMS)
        st.session_state.lt_block = st.session_state.lt.next_block()
        st.session_state.lt_idx   = 0

    eng   = st.session_state.lt
    block = st.session_state.lt_block
    idx   = st.session_state.lt_idx
    q     = block[idx]

    st.progress(len(eng.history) / eng.MAX_Q)
    st.subheader(f"문항 {len(eng.history)+1} / {eng.MAX_Q}")
    st.markdown(q["question"])

    # choices 문자열 → 리스트
    choices = json.loads(q["choices"]) if q["choices"].strip()[0] == "[" else \
              [c.strip() for c in q["choices"].split(",")]

    choice = st.radio("선택하세요:", choices, key=f"lt_{q['q_id']}", index=None)

    if st.button("제출"):
        correct = int(choice == q["answer"])
        eng.record(q["q_id"], correct, q["difficulty"])

        if idx + 1 == len(block):            # 블록 완료
            eng.adjust_diff()
            if eng.done_flag():              # 테스트 종료
                st.session_state.stage = 99
            else:
                st.session_state.lt_block = eng.next_block()
                st.session_state.lt_idx   = 0
        else:                                # 블록 내 다음 문제
            st.session_state.lt_idx += 1
        st.rerun()

###############################################################################
#  STAGE 99 ─ 결과 & PDF
###############################################################################
if st.session_state.stage == 99:
    eng = st.session_state.lt
    # ── 점수 집계
    sec_scores = {"vocab":0,"grammar":0,"writing":0,"reading":0}
    for q_id, correct, _ in eng.history:
        if correct:
            skill = next(i for i in LT_ITEMS if i["q_id"] == q_id)["skill"]
            sec_scores[skill] += 4    # 25점/섹션 = 6문항→4점

    total = sum(sec_scores.values())
    level_map = [(14,"A1"),(34,"A2"),(54,"B1"),(74,"B2"),(89,"C1"),(100,"C2")]
    level = next(l for t,l in level_map if total <= t)

    st.success(f"당신의 레벨은 **{level}**입니다! (총점 {total}/100)")
    st.write("섹션별 점수:", sec_scores)

    # ── PDF 리포트
    result = dict(user_id = st.session_state.user_id,
                  total_score = total,
                  level_code = level,
                  section_scores = sec_scores)
    pdf_bytes = make_pdf(result)
    st.download_button("📄 PDF 리포트 다운로드", pdf_bytes,
                       file_name="NineStudy_LevelReport.pdf")

    # ── DB 저장
    save_result(st.session_state.user_id, result)

