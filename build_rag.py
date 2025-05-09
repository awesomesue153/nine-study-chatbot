import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── 스크립트 파일(build_rag.py)이 있는 폴더 절대 경로
base_dir = Path(__file__).parent  # :contentReference[oaicite:0]{index=0}

# ── CSV 파일을 base_dir 기준으로 읽기
chunks_path = base_dir / "chunks.csv"
chunks = pd.read_csv(chunks_path)  # :contentReference[oaicite:1]{index=1}

# ── 임베딩 & 인덱스 생성
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
FAISS.from_texts(chunks["chunk"].tolist(), emb).save_local(str(base_dir / "rag_index"))
print("✅ rag_index 생성 완료")
