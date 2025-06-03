import csv, pandas as pd, re, sys
from pathlib import Path

csv_path = Path("problems.csv") if len(sys.argv) == 1 else Path(sys.argv[1])

def clean_line(line:str) -> str:
    """
    • 큰따옴표 개수가 홀수 → 마지막 ," 로 끝나면 따옴표 제거
    • 큰따옴표 0개인데 choices 필드가 ['..','..'] 형태면
      choices 전체를 큰따옴표로 감싸 줌
    """
    if line.count('"') % 2 == 1 and line.rstrip().endswith('",'):
        line = line.replace('",', ',', 1)           # 잉여 " 제거
    if "['" in line and "\"['" not in line:
        line = re.sub(r"\['", r"\"['", line, 1)     # 맨 앞에 "
        line = re.sub(r"'\]", r"']\"", line, 1)     # 맨 뒤에 "
    return line

# ── 클린된 임시 버퍼 -----------------------------------------------
buf = []
with csv_path.open(encoding="utf-8-sig") as f:
    for raw in f:
        buf.append(clean_line(raw))

tmp_path = csv_path.with_suffix(".tmp.csv")
tmp_path.write_text("".join(buf), encoding="utf-8-sig")

# ── 이제 정상 파싱 ---------------------------------------------------
df = pd.read_csv(tmp_path, encoding="utf-8-sig")    # sep="," 기본
print("✅ rows:", len(df), "| columns:", df.columns.tolist())

# … 이후 rename_units 로직 이어서 …
