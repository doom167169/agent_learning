import os 
import time
import jieba
import bm25s
import torch
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

# 0. 模型初始化
embedding_model = DashScopeEmbeddings(
    model="text_embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

rewrite_model = init_chat_model("deepseek-v4-flash")   # 查询重写
chat_model = "deepseek-v4-pro"
reranker = CrossEncoder(   # CrossEncoder模型
    "Qwen/Qwen3-Reranker-0.6B",
    evice="cuda" if torch.cuda.is_available() else "cpu"
)



# 1. 文档加载 & 切分
def load_and_split(md_path: str) -> List[Document]:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = MarkdownHeaderTextSplitter([
        ("#", "Header1"), ("##", "Header2"), ("###", "Header3"),
    ])

    # 对文本进行切分
    chunks = splitter.split_text(text)

    for i, doc in enumerate(chunks):
        doc.id = f"doc_{i+1}"
        if "Header3" in doc.metadata:
            doc.page_content = f"### {doc.metadata["Header3"]}\n{doc.page_content}"


    return chunks



# 2. 稠密检索引擎（向量）
def build_vector_store(docs: List[Document]) -> InMemoryVectorStore:
    vs = InMemoryVectorStore(embedding_model)
    vs.add_documents(docs)




# 3. 稀疏检索引擎（BM25）
def build_bm25_index(docs: List[Document], index_path: str = "my_index.bm25"):
    if os.path.exists(index_path):
        return bm25s.BM25.load(index_path, load_corpus=True)

    corpus = [{"id": d.id, "content": d.page_content} for d in docs]
    tokens = [jieba.lcut(d.page_content) for d in docs]
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(tokens)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    retriever.save(index_path)
    return retriever

def bm25_search(retriever, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
    tokens = jieba.lcut(query)
    results, scores = retriever.retrieve(tokens, k=k)
    return [(results[0, i], scores[0, i]) for i in range(results.shape[1])]



# 4. 查询改写
def query_rewrite(query: str) -> str:
    prompt = f"""将以下问题改写为适合检索的关键词形式，提取核心概念，用空格分隔，只输出关键词不要解释
    问题：{query}
    关键词：
    """

    return rewrite_model.invoke(prompt).content.strip()



# 5. RRF 融合
def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """ranked_lists: 每个检索器返回的 [{"id":..., "content":...}, ...]"""
    scores, registry = {}, {}
    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1.0 / (k + rank)
            registry[doc["id"]] = doc

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [registry[did] for did in sorted_ids]




# 6. Cross-Encoder 精排
def cross_encoder_rerank(query: str, docs: List[Dict], top_k: int = 3) -> List[Dict]:
    pairs = [(query, d["content"]) for d in docs]
    scores = reranker.predict(pairs)

    for i, s in enumerate(scores):
        docs[i]["score"] = float(s)

    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
    return [d for d in sorted_docs if d["score"] > 0][:top_k]




# 7. 综合检索流水线
def comprehensive_search(query: str, top_k: int = 3) -> str: 
    start_total = time.perf_counter()

    # 7.1 查询改写
    start = time.perf_counter()
    rewritten = query_rewrite(query)
    print(f"[1.查询改写] '{query}' → '{rewritten}' ({time_ms(start)}ms)")

    # 7.2 稠密检索
    start = time.perf_counter()
    dense_results = vectorstore.similarity_search_with_score(rewritten, k=top_k)
    dense_docs = [{"id": d.id, "content": d.page_content} for d, _ in dense_docs]
    print(f"[2.稠密检索] {len(dense_docs)} docs ({time_ms(start)}ms)")

    # 7.3 稀疏检索
    start = time.perf_counter()
    sparse_results = bm25_search(bm25_retriever, rewritten, k=top_k)
    sparse_docs = [doc for doc, _ in sparse_results]
    print(f"[3.稀疏检索] {len(sparse_docs)} docs ({time_ms(start)}ms)")

    # 7.4 RRF 融合
    start = time.perf_counter()
    fused_docs = reciprocal_rank_fusion([dense_docs, sparse_docs])
    print(f"[4.RRF融合] {len(fused_docs)} docs ({time_ms(start)}ms)")


    # 7.5 Cross-Encoder 精排
    start = time.perf_counter()
    final_docs = cross_encoder_rerank(query, fused_docs, top_k=top_k)
    print(f"[5.CrossEncoder] {len(final_docs)} docs ({time_ms(start)}ms)")
    

    print(f"[总耗时] {time_ms(start_total)}ms\n")

    return "\n\n".join(
        f"[score: {d["score"]:.4f}] {d["content"]}" for d in final_docs
    )

def time_ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.0f}"


# 8. 封装为 Agent Tool
@tool
def search_knowledge_base(query: str) -> str:
    """检索知识库。先改写查询，再走稠密+稀疏双路检索，RRF融合，CrossEncoder精排。"""
    return comprehensive_search(query)



# 9. 启动
if __name__ == "__main__":
    # 加载文档
    docs = load_and_split("中二知识笔记.md")
    print(f"文档切分完成：{len(docs)} 个 chunk")

    # 构建双路检索引擎
    vectorstore = build_vector_store(docs)
    bm25_retriever = build_bm25_index(docs)
    print("检索引擎就绪")

    # 创建 Agent
    agent = create_agent(
        model=chat_model,
        tools=[search_knowledge_base],
        system_prompt=(
            "你是一个知识助手。根据工具检索到的资料回答问题。"
            "如果资料中没有相关信息，直接说不知道，不要编造。"
        )
    )
    print("Agent 启动完成\n")

    # 交互
    query = "因材施教是谁提出的？"
    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage) and chunk.content:
            print(chunk.content, end="", flush=True)