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

import random, uuid, datetime, sqlite3, json, pandas as pd
from pathlib import Path
from fpdf import FPDF

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
                (user_id, datetime.datetime.now().isoformat(), json.dumps(result,ensure_ascii=False)))
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
def make_pdf(result: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "NineStudy Level Test Report", ln=1, align="C")
    pdf.set_font_size(12)
    pdf.cell(0, 8, f"User ID: {result['user_id']}", ln=1)
    pdf.cell(0, 8, f"Level: {result['level_code']}   Score: {result['total_score']}/100", ln=1)
    pdf.ln(4); pdf.cell(0, 8, "Section Scores:", ln=1)
    for sec, sc in result["section_scores"].items():
        pdf.cell(0, 6, f"  {sec.capitalize():<8}: {sc} / 25", ln=1)
    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        return f.read()
