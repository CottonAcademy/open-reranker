from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from FlagEmbedding import FlagReranker
import torch
import uvicorn

app = FastAPI()

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FlagReranker('./huggingface.co/BAAI/bge-reranker-v2-m3', device=device)

class RerankRequest(BaseModel):
    query: str
    passages: List[str]

class ScoredPassage(BaseModel):
    passage: str
    score: float

@app.post("/rerank", response_model=List[ScoredPassage])
async def rerank(request: RerankRequest):
    try:
        # 计算得分
        scores = model.compute_score([[request.query, p] for p in request.passages])
        
        # 直接返回 ScoredPassage 对象列表
        results = [
            ScoredPassage(passage=p, score=float(s))
            for p, s in zip(request.passages, scores)
        ]
        
        # 按得分降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results  # 直接返回列表，不要用元组包装
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)