###############################################################################
#  leveltest.py  ▪︎  v 0.8.0   (2025‑05‑13)
#
#  ▪︎ 기능
#    - LevelEngine : Adaptive‑Lite 난이도 조절(블록당 3문항, 최대 20문항)
#    - PDF Report  : fpdf2 로 결과 요약서를 메모리 bytes 로 생성
#    - SQLite Log  : test_results 테이블 자동 생성·저장
#
#  ▪︎ 주요 상수
#    CSV_FILE  = leveltest_questions.csv
#    DB_FILE   = leveltest.db
#    BLOCK     = 3     # 문항/블록
#    MAX_Q     = 20    # 총 문항
#
#  ▪︎ 사용 흐름
#      app.py(Stage 9) → LevelEngine.next_block() → record() → adjust_diff()
#      종료 → make_pdf() → save_result()
###############################################################################

import random, uuid, datetime as dt, sqlite3, json, pandas as pd
from pathlib import Path
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent   # 현재 파일 위치 → 절대경로
CSV_FILE = ROOT_DIR / "leveltest_questions.csv"
DB_FILE  = ROOT_DIR / "leveltest.db"

# ───────────────────── 데이터 로드
def load_items():
    df = pd.read_csv(CSV_FILE)        # 절대경로 사용
    return [row._asdict() for row in df.itertuples(index=False)]

ITEMS = load_items()

# ───────────────────── 간단 DB
def save_result(user_id: str, result: dict):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS test_results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, timestamp TEXT, detail_json TEXT)""")
    cur.execute("INSERT INTO test_results(user_id,timestamp,detail_json) VALUES(?,?,?)",
                (user_id, dt.datetime.now().isoformat(), json.dumps(result, ensure_ascii=False)))
    con.commit(); con.close()

# ───────────────────── Adaptive‑Lite 엔진
class LevelEngine:
    BLOCK  = 3          # 블록당 문항
    MAX_Q  = 20         # 최대 문항 수
    HI, LO = 0.8, 0.4   # 난이도 상·하향 임계값

    def __init__(self, items):
        self.items = items.copy()
        self.history = []         # [(q_id, correct, diff), ...]
        self.cur_diff = 3         # 시작 난이도
        self.done = set()

    def next_block(self):
        pool = [q for q in self.items if q["difficulty"] == self.cur_diff and q["q_id"] not in self.done]
        if len(pool) < self.BLOCK:    # 예비 부족 시 랜덤 충원
            pool += random.sample([q for q in self.items if q["q_id"] not in self.done], self.BLOCK-len(pool))
        block = random.sample(pool, self.BLOCK)
        return block

    def record(self, q_id, correct, diff):
        self.history.append((q_id, correct, diff))
        self.done.add(q_id)

    def adjust_diff(self):
        last = self.history[-self.BLOCK:]
        rate = sum(c for _, c, _ in last) / len(last)
        if   rate >= self.HI and self.cur_diff < 5: self.cur_diff += 1
        elif rate <= self.LO and self.cur_diff > 1: self.cur_diff -= 1

    def done_flag(self):
        return len(self.history) >= self.MAX_Q

# ───────────────────── PDF 리포트
from io import BytesIO
from datetime import datetime

# 폰트 파일 (프로젝트에 fonts/NotoSansKR-*.ttf 위치)
FONT_DIR   = ROOT_DIR / "fonts"
TTF_REG    = FONT_DIR / "NotoSansKR-Regular.ttf"
TTF_BOLD   = FONT_DIR / "NotoSansKR-Bold.ttf"

def _register_noto(pdf):
    """
    NotoSansKR Regular / Bold 를 한 번만 등록
    (pdf.fonts 사전 구조 : {"Family": {"": .. , "B": ..}})
    """
    fam = pdf.fonts.get("NotoSansKR", {})
    if "" not in fam:
        pdf.add_font("NotoSansKR", "", str(TTF_REG),  uni=True)
    if "B" not in fam:
        pdf.add_font("NotoSansKR", "B", str(TTF_BOLD), uni=True)

def make_pdf(result: dict, chart_buf: bytes, header_path) -> bytes:
    """
    ▪︎ result    : {"user_id", "total_score", "level_code", "section_scores":{…}}
    ▪︎ chart_buf : Stage 99 에서 Bytes 로 넘겨준 막대그래프 PNG
    ▪︎ header_path : 헤더 이미지(Path 객체)
    """
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    _register_noto(pdf)
    pdf.add_page()

    # ① 헤더 이미지 + 타이틀 ------------------------------------------------
    # pdf.image(str(header_path), x=10, y=8, w=190) #배경 이미지 미노출
    pdf.set_xy(10, 24)
    pdf.set_font("NotoSansKR", "B", 24)
    pdf.cell(0, 10, "NineStudy 레벨테스트 결과", ln=1, align="C")

    # ② Overview & Level 카드 ---------------------------------------------
    neon      = (57, 255, 20)     # 형광 그린
    card_y    = 40                # 공통 Y 좌표
    card_h    = 30                # 높이
    card_w    = 100               # 전체 폭 (A4 좌우 여백 고려)
    x_left    = 55                # 중앙 정렬용 X (≈ (210-card_w)/2 )

    # 2-1) 배경 카드
    pdf.set_fill_color(230, 255, 240)                  # 옅은 민트
    pdf.rect(x_left, card_y, card_w, card_h, style="F")
    
    # 2-2) 텍스트: 점수와 레벨을 ‘ | ’ 로 구분해 같은 줄에 출력
    pdf.set_xy(x_left, card_y + 6)                     # 카드 안쪽으로 살짝 내림
    pdf.set_font("NotoSansKR", "B", 24)                # 굵은 24 pt
    pdf.set_text_color(*neon)
    pdf.cell(card_w, 10, f"{result['total_score']} / 100", align="C")

    pdf.set_xy(x_left, card_y + 18)  
    pdf.set_font("NotoSansKR", "", 14)
    pdf.set_text_color(0)
    pdf.cell(card_w, 8, f"Level : {result['level_code']}", align="C")

    # ③ 섹션 그래프 --------------------------------------------------------
    chart_io = BytesIO(chart_buf)
    chart_top = 80          # ← 그래프 Y
    chart_h   = 65
    pdf.image(chart_io, x=45, y=chart_top, w=120, h=chart_h)  # W:H = 120:65

    # ④ 상세 점수표 --------------------------------------------------------
    scores  = result["section_scores"]
    row_h   = 8
    col_w   = (35, 35)            # (왼쪽, 오른쪽) ― 헤더·데이터 동일 사용
    table_x = 70                  # 왼쪽 여백
    table_y = chart_top + chart_h + 10   # 그래프 아래 10 mm

    pdf.set_xy(table_x, table_y)
    pdf.set_font("NotoSansKR", "B", 12)   # 헤더 굵게
    
    # ── 헤더행 -------------------------------------------------------
    pdf.set_fill_color(230, 255, 240)         # 연한 민트
    pdf.cell(col_w[0], row_h, "Sections", border=1, align="C", fill=True)
    pdf.cell(col_w[1], row_h, "Score", border=1, ln=1, align="C", fill=True)      # ln=1 → 같은 X 로 줄바꿈
    
    # ← 헤더 줄바꿈 후 X 좌표를 다시 table_x 로 맞춘다
    pdf.set_x(table_x)

    # ── 본문행 -------------------------------------------------------
    pdf.set_font("NotoSansKR", "", 12)        # 일반체로 다시 설정
    for sec in ("vocab", "grammar", "writing", "reading"):
        pdf.set_x(table_x)                    # ★ 매 행마다 시작 X 재설정
        pdf.cell(col_w[0], row_h, sec.capitalize(), border=1, align="C")
        pdf.cell(col_w[1], row_h, f"{scores[sec]}/25",
                 border=1, ln=1, align="C")

    # ⑤ 맞춤 피드백 --------------------------------------------------------
    fb_txt = build_feedback(
        result["level_code"],
        min(result["section_scores"], key=result["section_scores"].get),
    )
    FEED_W = 120                     # 폭 고정
    X_FEED = (210 - FEED_W) / 2      # 중앙 정렬
    Y_FEED = pdf.get_y() + 10        # 표와 8 mm 간격
    PAD    = 4                       # 박스 내부 패딩(mm)

    # ⑤-1 피드백 문장 생성
    fb_txt = build_feedback(
        result["level_code"],
        min(result["section_scores"], key=result["section_scores"].get),
    )

    # ⑤-2 라인 수를 먼저 계산해 박스 높이 산출
    line_h   = 6                                     # 줄 간격
    lines    = pdf.multi_cell(                       # split_only=True → 줄바꿈 계산만
                FEED_W - PAD*2, line_h, fb_txt,
                split_only=True)
    box_h    = len(lines) * line_h + PAD*2           # 패딩을 위·아래로 더함

    # ⑤-3 배경(필) + 테두리 사각형
    pdf.set_fill_color(245, 245, 245)                # 옅은 그레이
    pdf.set_draw_color(80, 80, 80)                   # 블랙 컬러
    pdf.rect(X_FEED, Y_FEED, FEED_W, box_h, style="FD")  # F=Fill, D=Draw

    # ⑤-4 텍스트 출력 (패딩만큼 안쪽으로 이동)
    pdf.set_xy(X_FEED + PAD, Y_FEED + PAD)
    pdf.set_font("NotoSansKR", "", 11)
    pdf.set_text_color(0)                            # 검정
    pdf.multi_cell(FEED_W - PAD*2, line_h, fb_txt, align="L")

    # ⑥ 푸터 --------------------------------------------------------------
    pdf.set_y(-15)
    pdf.set_font("NotoSansKR", "", 8)
    pdf.set_text_color(120)
    pdf.cell(
        0,
        5,
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  https://ninestudy.co.kr",
        align="C",
    )

    # ⑦ 바이트스트림 반환 ---------------------------------------------------
    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out.read()


LEVEL_FEEDBACK = {
    "A1": (
        "🌱 막 영어 학습을 시작한 단계예요. 알파벳과 기초 발음, 인사·숫자·날짜처럼 "
        "일상 단어를 익히는 중이라면 아주 정상적인 수준입니다!\n\n"
        "📝학습 팁: 파닉스(발음 규칙) + ‘I am…, You are…’ 같은 현재형 패턴을 반복하세요. "
        "짧은 동요·플래시카드·듣고 따라 말하기 활동이 효과적입니다."
    ),
    "A2": (
        "🍀간단한 생활 영어는 이미 할 수 있지만, 두 개 이상의 절이 결합되는 복합 문장에서 "
        "어순·시제 실수가 자주 보여요.\n\n"
        "📝학습 팁: ‘because/when/if’ 같은 접속사로 문장을 연결해 보세요. "
        "과거형-현재완료 구분, 관사(a/an/the) 사용법을 집중 점검하면 빠르게 안정됩니다."
    ),
    "B1": (
        "🌿핵심 어휘와 표현은 잘 구사하지만, 문법 일관성(시제 일치·주어-동사 수 일치)이 흔들릴 때가 있습니다. "
        "자유 회화에서 ‘중간 난이도’ 의사소통은 가능하나 주제 변화에 따라 망설임이 생길 수 있어요.\n\n"
        "📝학습 팁: 짧은 에세이를 쓰고 Grammarly·Languagetool 등으로 오류를 교정해 보세요. "
        "또한 동사구(look forward to, end up ing)·분사구문 꾸준히 노출하면 표현 폭이 넓어집니다."
    ),
    "B2": (
        "🌳대부분의 상황에서 영어로 명확히 의사를 표현할 수 있으나 어휘 범위가 일부 제한적입니다. "
        "추상적·전문적 주제에서 적절한 단어 선택이 어려워 중복 표현을 쓰곤 합니다.\n\n"
        "📝학습 팁: 시사 기사·TED 영상을 ‘주제별 단어장’으로 정리하고, "
        "동의어·반의어까지 함께 묶어 보세요. 또, 조건·가정법·도치 같은 고급 문법을 회화에 적용해 보며 "
        "문장 다양성을 높이면 좋습니다."
    ),
    "C1": (
        "🌲대부분의 사회·학문적 상황에서 자연스럽고 유창하게 소통할 수 있습니다. 다만 세부 뉘앙스(유머·지역 특유 표현)나 미세한 어조 조절이 아쉬울 때가 있네요.\n\n"
        "📝학습 팁: 원어민 팟캐스트·소설을 ‘쉐도잉+요약’으로 학습해 보세요. 또 발표·디베이트에서 rhetorical devices(비유·반어)를 의도적으로 사용해 보면 표현 깊이가 더해집니다."
    ),
    "C2": (
        "🌳🌳원어민과 거의 대등한 이해·표현 능력을 보유하고 있습니다! "
        "전문 문헌·문학 작품도 큰 무리 없이 읽고, 복잡한 논증을 전개할 수 있는 수준이에요.\n\n"
        "📝유지·발전 팁: 특정 분야(법·의학·공학 등)의 jargon을 체계적으로 익혀 ‘전문 번역’ "
        "또는 ‘학술 발표’에 도전해 보세요. 또한 작문에서 tone·register(격식/비격식) 조절 훈련을 지속하면 "
        "언어 유연성이 더욱 공고해집니다."
    )
}

def build_feedback(level:str, weakest:str)->str:
    base = LEVEL_FEEDBACK.get(level, "")
    tip  = f"\n\n특히 **{weakest.capitalize()}** 파트 보강을 권장합니다."
    return base + tip
