import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

chunks = pd.read_csv("chunks.csv")  # chunk, source
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
FAISS.from_texts(chunks["chunk"].tolist(), emb).save_local("rag_index")
print("✅ rag_index 생성 완료")