###############################################################################
#  NineStudy Chatbot  ▪︎  v 1.1.0   (2025-06-03)
#
#  ► 변경 핵심
#    • Data : exam_tips.csv 를 표준 CSV 스키마(헤더+콤마)로 통일
#    • Code : load_csvs() 단순화 → 4개 CSV 모두 pd.read_csv() 한 줄 처리
#    • Bug  : 시험 포인트 탭 미노출 오류 수정
#    • Docs : 릴리스 노트 / 헤더 주석 1.1.0 반영
#
#  © 2025 Chapter9 — Creative Flow Labs
###############################################################################

import os, json, csv, logging, uuid
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
from leveltest import LevelEngine, ITEMS as LT_ITEMS, make_pdf, save_result

# ───────────────────────── 상수 · 토큰 주입
ROOT_DIR = Path(__file__).resolve().parent
IMG_DIR  = ROOT_DIR / "images"           # ★ 이미지 전용 경로

HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)

# ───────────────────────── 유틸
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

# ───────────────────────── 페이지 & 상단 Bar
st.set_page_config(page_title="나인스터디 챗봇", layout="wide")
st.markdown("""
<style>
/* 기존 숨김 --------------------------- */
footer, #MainMenu {visibility:hidden;}
div[data-testid="stFullscreenButton"],
.stViewFullscreenButton {display:none;}
/* …(나머지 이전 코드 그대로)…          */

/* NEW — Streamlit 1.34+ selector 보강 -- */

/* 1) 상단 Share 버튼·툴바 전체 숨김  */
header[data-testid="stHeader"]      {visibility:hidden !important;}
div[data-testid="stToolbarActions"] {display:none  !important;}

/* 2) 하단 Manage app / Version 바지 */
div[data-testid="stStatusWidget"]   {display:none  !important;}
a[href*="manage.app"]               {display:none  !important;}
</style>
""", unsafe_allow_html=True)

top = st.container()
col_btn, _ = top.columns([1, 9], gap="small")

# ───────────────────────── 설정(JSON) · CSV
with open(ROOT_DIR / "config.json", encoding="utf-8") as f:
    menu_cfg = json.load(f)
with open(ROOT_DIR / "publishers.json", encoding="utf-8") as f:
    pub_cfg = json.load(f)

CSV_PATHS = ["concepts.csv", "problems.csv", "self_check.csv", "exam_tips.csv"]

@st.cache_data(show_spinner="📂 CSV 로딩 중…",
               hash_funcs={Path: lambda p: p.stat().st_mtime})
def load_csvs():
    base_dir = ROOT_DIR
    df_concepts    = pd.read_csv(base_dir / CSV_PATHS[0])
    df_problems    = pd.read_csv(base_dir / CSV_PATHS[1])
    df_self_check  = pd.read_csv(base_dir / CSV_PATHS[2])
    df_tips        = pd.read_csv(base_dir / CSV_PATHS[3])

    return {
        "concepts":   df_concepts,
        "problems":   df_problems,
        "selfcheck":  df_self_check,
        "tips":       df_tips
    }

dfs = load_csvs()

def build_content(d):
    c={}
    for uid,g in d["concepts"].groupby("unit_id"):
        c[uid]={"concept":g["concept"].iloc[0]}
    for uid,g in d["problems"].groupby("unit_id"):
        c.setdefault(uid,{})["problems"]=g.to_dict("records")
    for uid,g in d["tips"].groupby("unit_id"):
        c.setdefault(uid,{})["exam_tips"]=g["tip"].tolist()
    return c
content = build_content(dfs)

# ───────────────────────── RAG + LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from importlib import util

HAS_SDK = bool(util.find_spec("openai"))
USE_OPENAI = bool(OPENAI_KEY and HAS_SDK)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="{context}"
)

@st.cache_resource(show_spinner="🧠 모델 & 인덱스 로딩…")
def init_rag_chain():
    emb   = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    store = FAISS.load_local(str(ROOT_DIR / "rag_index"), emb,
                             allow_dangerous_deserialization=True)

    llm = None
    if USE_OPENAI:
        try:
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=OPENAI_KEY,
                streaming=True, temperature=0.7, max_tokens=256,
                timeout=15, max_retries=2,
            )
            logging.info("💡 OpenAI LLM 사용(gpt-3.5-turbo)")
        except Exception as e:
            logging.error("OpenAI 초기화 실패→폴백 (%s)", e, exc_info=True)

    if llm is None:
        MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl  = AutoModelForCausalLM.from_pretrained(
                   MODEL_ID, device_map="auto", torch_dtype="auto")
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                        max_new_tokens=128, temperature=0.7, top_p=0.9,
                        repetition_penalty=1.2)
        llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("💡 TinyLlama LLM 사용")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )
    return chain

rag_chain = init_rag_chain()

# ── 스트리밍 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, box): self.box, self.txt = box, ""
    def on_llm_new_token(self, t, **_):
        self.txt += t
        self.box.markdown(self.txt + "▌")

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

    st.progress(min(len(eng.history), eng.MAX_Q) / eng.MAX_Q)
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
#  STAGE 99 ─ 결과 & PDF  (트렌디 UI v2)
###############################################################################
if st.session_state.stage == 99:
    eng = st.session_state.lt

    # 1) 점수 집계 -----------------------------------------------------------
    sec_scores = {"vocab":0,"grammar":0,"writing":0,"reading":0}
    for q_id, correct, _ in eng.history:
        if correct:
            skill = next(i for i in LT_ITEMS if i["q_id"] == q_id)["skill"]
            sec_scores[skill] += 4
    total = sum(sec_scores.values())
    level = next(l for t,l in [(14,"A1"),(34,"A2"),(54,"B1"),
                               (74,"B2"),(89,"C1"),(100,"C2")] if total<=t)
    
    # ── ❶ 4-색 형광 그래프 BytesIO 생성 ─────────────────────────────
    import matplotlib.pyplot as plt
    from io import BytesIO
    plt.clf()
    cols = ["#5ABFA3", "#FF6F6C", "#F9F871", "#6DD9FF"]  # 형광 Green·Cyan·Yellow·Pink
    labels = list(sec_scores.keys())
    values = list(sec_scores.values())
    fig, ax = plt.subplots(figsize=(4, 2.2))             # ← 크기 DOWN
    bars = ax.bar(labels, values, color=cols, width=0.55)
    ax.set_ylim(0, 25) ; ax.spines[['right','top']].set_visible(False)
    ax.set_ylabel("점수")
    # 점수 숫자 표시
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', va='bottom', fontsize=8)
    buf = BytesIO() ; plt.tight_layout(pad=0.4)
    plt.savefig(buf, format="png", dpi=140) ; buf.seek(0)
    chart_img = buf.getvalue()            # ← PDF로 넘길 변수
    plt.close(fig)

    # 2) 헤더 이미지 -----------------------------------------------------------
    HEADER_IMG = IMG_DIR / "header_leveltest(2).png"

    st.image(
        str(HEADER_IMG),                     # Path → str 로 변환
        caption="NineStudy Level Test Result",
        use_container_width=True,
    )


    # 3) 레이아웃 (2:1)  -----------------------------------------------------
    left, right = st.columns([2, 1], gap="large")

    # ── 3-1  Altair 형광 막대 + 점수 레이블 ---------------------------
    import altair as alt, pandas as pd
    df_chart = pd.DataFrame({
        "section": list(sec_scores.keys()),
        "score":   list(sec_scores.values()),
    })

    neon_palette = ["#5ABFA3", "#FF6F6C", "#F9F871", "#6DD9FF"]  # vocab→reading

    base = alt.Chart(df_chart).encode(
        x=alt.X("section:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("score:Q",   title=None, scale=alt.Scale(domain=[0, 25]))
    )

    bars = base.mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6
    ).encode(
        color=alt.Color("section:N",
                        scale=alt.Scale(domain=list(sec_scores.keys()),
                                        range=neon_palette),
                        legend=None),
        tooltip=["section", "score"],
    )

    labels = base.mark_text(
        dy=-8,                         # 막대 위쪽 약간 띄우기
        color="black",                 # 필요하면 'white' 로
        fontSize=13,
        fontWeight="bold"
    ).encode(
        text="score:Q"
    )

    with left:
        st.altair_chart((bars + labels), use_container_width=True)


    # 3-2  Overview & Level 카드 -----------------------------------------------
    with right:
        st.markdown("#### 🏆 Overview")

        neon = "#5ABFA3"          # 형광 그린 (그래프와 통일)

        # ── 총점 커스텀 표시 ─────────────────────────────
        st.markdown(
            f"""
            <div style='line-height:1; margin-bottom:8px'>
                <span style='font-size:50px; font-weight:900; color:{neon};'>
                    {total}
                </span>
                <span style='font-size:24px; font-weight:600;'> / 100</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")                # Streamlit 빈 줄
        st.write("")                # Streamlit 빈 줄
        st.write("")                # Streamlit 빈 줄

        # ── 레벨 커스텀 표시 ─────────────────────────────
        st.markdown("#### 🏆 Level")
        st.markdown(
            f"""
            <div style='line-height:1; margin-bottom:8px'>
                <span style='font-size:50px; font-weight:900; color:{neon};'>
                    {level}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        weakest = min(sec_scores, key=sec_scores.get)
        best    = max(sec_scores, key=sec_scores.get)
        st.markdown(
            f"<div style='margin-top:12px'>"
            f"😍 Best : <b>{best.capitalize()}</b><br>"
            f"😭 Weak : <b>{weakest.capitalize()}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # 4) PDF 리포트 + DB 저장 -------------------------------------------
    result = dict(
        user_id=st.session_state.user_id,
        total_score=total,
        level_code=level,
        section_scores=sec_scores,
    )
    pdf_bytes = make_pdf(result, chart_img, HEADER_IMG)
    st.download_button(
        "📄  PDF 리포트 다운로드",
        pdf_bytes,
        file_name="NineStudy_LevelReport.pdf",
    )
    save_result(st.session_state.user_id, result)

    # 5) 약점 보완 바로 가기 (선택) --------------------------------------
    if st.button(f"💡  {weakest.capitalize()} 보완 학습 시작"):
        st.session_state.stage = 1
        st.rerun()
