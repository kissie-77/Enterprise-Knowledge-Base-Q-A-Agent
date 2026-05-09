"""
高级检索模块 - 多路召回 + Reranker 重排序
面试高频考点：为什么单一向量检索不够？如何提升召回率和精排质量？

核心设计：
1. 多路召回（Hybrid Retrieval）
   - 语义路：向量相似度检索（捕捉语义相关）
   - 关键词路：BM25 稀疏检索（精确关键词匹配）
   - 融合：RRF（Reciprocal Rank Fusion）倒排融合

2. Reranker 重排序
   - Cross-Encoder 精排：对候选集做精细化打分
   - 比 Bi-Encoder 更准确，但计算成本高，所以只对候选集做

3. 管道流程
   Query → 多路召回(粗排) → 候选集合并去重 → Reranker(精排) → Top-K 结果

面试考点：
- Bi-Encoder vs Cross-Encoder 的区别和使用场景
- 为什么用 RRF 而不是简单分数相加？
- 召回阶段 vs 精排阶段各自的设计考量
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .rag_engine import RAGEngine
from .query_rewriter import QueryRewriter, RewriteResult


@dataclass
class RetrievalResult:
    """单条检索结果"""
    content: str
    score: float                          # 最终得分（越高越好）
    source: str = ""                      # 来源文件
    chunk_index: int = 0                  # 片段索引
    retrieval_method: str = ""            # 检索方式
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalPipelineResult:
    """完整检索管道结果"""
    query: str                                           # 原始查询
    rewrite_result: Optional[RewriteResult] = None       # 查询改写结果
    initial_candidates: int = 0                          # 粗排候选数
    final_results: List[RetrievalResult] = field(default_factory=list)  # 精排后结果
    methods_used: List[str] = field(default_factory=list)  # 使用的检索方法
    reranker_used: bool = False                          # 是否使用了重排序

    def get_context(self, max_length: int = 3000) -> str:
        """构建上下文文本（带来源标注）"""
        if not self.final_results:
            return "知识库中未找到相关内容。"

        parts = []
        current_length = 0
        for i, r in enumerate(self.final_results):
            content = r.content
            if current_length + len(content) > max_length:
                remaining = max_length - current_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            source_info = f"(来源: {r.source})" if r.source else ""
            parts.append(f"[参考{i+1}]{source_info} {content}")
            current_length += len(content)

        return "\n\n".join(parts)


class BM25Retriever:
    """
    BM25 稀疏检索器
    基于关键词的经典信息检索算法，与向量检索互补

    优势：精确匹配关键词、专有名词、数字编号等
    劣势：无法理解语义相似（"汽车" vs "轿车"）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数（越大，高频词权重越高）
            b: 文档长度归一化参数（0=不归一化, 1=完全归一化）
        """
        self.k1 = k1
        self.b = b
        self._documents: List[str] = []
        self._doc_metadata: List[Dict[str, Any]] = []
        self._doc_freqs: List[Dict[str, int]] = []     # 每个文档的词频
        self._idf: Dict[str, float] = {}               # 逆文档频率
        self._avg_doc_len: float = 0
        self._is_built: bool = False

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """添加文档到索引"""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        for doc, meta in zip(documents, metadatas):
            self._documents.append(doc)
            self._doc_metadata.append(meta)
            # 分词并计算词频
            tokens = self._tokenize(doc)
            freq = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self._doc_freqs.append(dict(freq))

        self._build_index()

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """BM25 检索"""
        if not self._is_built or not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for doc_idx, doc_freq in enumerate(self._doc_freqs):
            score = 0.0
            doc_len = sum(doc_freq.values())

            for token in query_tokens:
                if token not in doc_freq:
                    continue
                tf = doc_freq[token]
                idf = self._idf.get(token, 0)

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scores[:top_k]:
            if score <= 0:
                continue
            meta = self._doc_metadata[doc_idx]
            results.append(RetrievalResult(
                content=self._documents[doc_idx],
                score=score,
                source=meta.get("source", ""),
                chunk_index=meta.get("chunk_index", 0),
                retrieval_method="bm25",
                metadata=meta,
            ))

        return results

    def _build_index(self):
        """构建 IDF 索引"""
        n = len(self._documents)
        if n == 0:
            return

        # 计算每个词出现在多少个文档中
        doc_containing: Dict[str, int] = defaultdict(int)
        total_len = 0

        for doc_freq in self._doc_freqs:
            total_len += sum(doc_freq.values())
            for token in doc_freq.keys():
                doc_containing[token] += 1

        self._avg_doc_len = total_len / n

        # 计算 IDF
        for token, df in doc_containing.items():
            # 使用 BM25 的 IDF 变体
            self._idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        self._is_built = True

    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词（中英文混合）
        生产环境可以替换为 jieba 分词
        """
        import re
        # 英文按空格/标点分词，中文按字分词
        # 先提取英文单词
        english_tokens = re.findall(r'[a-zA-Z]+', text.lower())
        # 提取中文字符（按字分）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 提取中文双字组合（模拟二元分词）
        chinese_bigrams = []
        chinese_text = ''.join(chinese_chars)
        for i in range(len(chinese_text) - 1):
            chinese_bigrams.append(chinese_text[i:i+2])

        return english_tokens + chinese_chars + chinese_bigrams

    @property
    def document_count(self) -> int:
        return len(self._documents)


class Reranker:
    """
    重排序器（Cross-Encoder）
    对粗排结果进行精细化重排序

    原理：Cross-Encoder 将 query 和 document 拼接后一起编码，
    比 Bi-Encoder（分别编码再算相似度）更准确，但速度慢，
    所以只在候选集（通常 20-50 条）上使用。

    策略：
    - 有 cross-encoder 模型时：使用模型精排
    - 无模型时：使用 LLM 打分作为降级方案
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_llm_fallback: bool = True,
        llm: Optional[Any] = None,
    ):
        """
        Args:
            model_name: Cross-Encoder 模型名称
            use_llm_fallback: 无模型时是否使用 LLM 降级打分
            llm: LLM 客户端（用于降级方案）
        """
        self.model_name = model_name
        self.use_llm_fallback = use_llm_fallback
        self.llm = llm
        self._model = None
        self._model_available = False

        # 尝试加载 cross-encoder
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            self._model_available = True
            print(f"✅ Reranker 模型加载成功: {model_name}")
        except Exception as e:
            print(f"⚠️ Reranker 模型加载失败 ({model_name}): {e}")
            print("  将使用 LLM 降级打分方案")

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        对候选结果进行重排序

        Args:
            query: 查询文本
            candidates: 候选结果列表
            top_k: 返回 top-k 个结果

        Returns:
            重排序后的结果（按得分降序）
        """
        if not candidates:
            return []

        if self._model_available:
            return self._rerank_with_model(query, candidates, top_k)
        elif self.use_llm_fallback and self.llm:
            return await self._rerank_with_llm(query, candidates, top_k)
        else:
            # 无重排能力，直接返回原结果
            return candidates[:top_k]

    def _rerank_with_model(
        self, query: str, candidates: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        """使用 Cross-Encoder 模型重排"""
        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)

        # 更新分数并排序
        scored_results = []
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            candidate.retrieval_method += "+reranker"
            scored_results.append(candidate)

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    async def _rerank_with_llm(
        self, query: str, candidates: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        """
        LLM 降级打分方案
        让 LLM 对每个候选结果的相关性打分（1-5分）
        """
        prompt = f"""请对以下文档片段与用户问题的相关性进行评分（1-5分）。
5分 = 完全相关且能直接回答问题
3分 = 部分相关
1分 = 完全不相关

用户问题: {query}

请对每个片段打分，每行格式为: 编号:分数

"""
        for i, c in enumerate(candidates[:10]):  # 限制最多10个，避免 token 过多
            content_preview = c.content[:200]
            prompt += f"片段{i+1}: {content_preview}\n\n"

        prompt += "评分结果（每行一个，格式如 1:4）："

        try:
            result = await self.llm.think(prompt, temperature=0)

            # 解析分数
            import re
            score_map = {}
            for match in re.finditer(r'(\d+)\s*[:：]\s*(\d+)', result):
                idx = int(match.group(1)) - 1
                score = int(match.group(2))
                if 0 <= idx < len(candidates) and 1 <= score <= 5:
                    score_map[idx] = score

            # 更新分数
            for idx, score in score_map.items():
                candidates[idx].score = score / 5.0  # 归一化到 0-1
                candidates[idx].retrieval_method += "+llm_rerank"

            # 排序
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]

        except Exception as e:
            print(f"⚠️ LLM 重排序失败: {e}")
            return candidates[:top_k]


def rrf_fusion(
    result_lists: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """
    RRF（Reciprocal Rank Fusion）倒数排名融合

    将多路检索结果融合为统一排名。
    比简单分数相加更鲁棒，因为不同检索方法的分数量纲不同。

    公式: RRF_score(d) = Σ 1/(k + rank_i(d))

    Args:
        result_lists: 多路检索结果列表
        k: 平滑常数（默认60，防止高排名文档权重过大）

    Returns:
        融合后的排序结果
    """
    # 以 content 为 key 进行融合
    content_scores: Dict[str, float] = defaultdict(float)
    content_results: Dict[str, RetrievalResult] = {}
    content_methods: Dict[str, List[str]] = defaultdict(list)

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            content_key = result.content[:100]  # 用前100字符作为去重 key
            rrf_score = 1.0 / (k + rank + 1)
            content_scores[content_key] += rrf_score

            if content_key not in content_results:
                content_results[content_key] = result

            if result.retrieval_method and result.retrieval_method not in content_methods[content_key]:
                content_methods[content_key].append(result.retrieval_method)

    # 按 RRF 分数排序
    sorted_keys = sorted(content_scores.keys(), key=lambda x: content_scores[x], reverse=True)

    fused_results = []
    for key in sorted_keys:
        result = content_results[key]
        result.score = content_scores[key]
        result.retrieval_method = "+".join(content_methods[key])
        fused_results.append(result)

    return fused_results


class HybridRetriever:
    """
    混合检索器 - 完整的检索管道

    流程:
    1. 查询改写（Query Rewriting）
    2. 多路召回（向量 + BM25）
    3. RRF 融合
    4. Reranker 精排
    5. 返回 Top-K

    这是整个 RAG 系统检索侧的核心组件。
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        query_rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[Reranker] = None,
        enable_bm25: bool = True,
        enable_reranker: bool = True,
    ):
        """
        Args:
            rag_engine: 向量检索引擎
            query_rewriter: 查询改写器
            reranker: 重排序器
            enable_bm25: 是否启用 BM25 检索路
            enable_reranker: 是否启用重排序
        """
        self.rag_engine = rag_engine
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.enable_bm25 = enable_bm25
        self.enable_reranker = enable_reranker

        # BM25 索引（延迟构建）
        self._bm25: Optional[BM25Retriever] = None
        self._bm25_synced = False

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rewrite_strategy: str = "auto",
        conversation_history: str = "",
        candidate_multiplier: int = 3,
    ) -> RetrievalPipelineResult:
        """
        执行完整检索管道

        Args:
            query: 用户查询
            top_k: 最终返回结果数
            rewrite_strategy: 查询改写策略
            conversation_history: 对话历史
            candidate_multiplier: 粗排候选数倍数（相对于 top_k）

        Returns:
            完整检索管道结果
        """
        pipeline_result = RetrievalPipelineResult(query=query)
        candidate_count = top_k * candidate_multiplier

        # ===== 第1步：查询改写 =====
        rewrite_result = None
        search_queries = [query]  # 默认只用原始查询

        if self.query_rewriter:
            rewrite_result = await self.query_rewriter.rewrite(
                query=query,
                strategy=rewrite_strategy,
                conversation_history=conversation_history,
            )
            pipeline_result.rewrite_result = rewrite_result
            search_queries = rewrite_result.get_all_queries()

        # ===== 第2步：多路召回 =====
        all_result_lists: List[List[RetrievalResult]] = []

        # 路1: 向量语义检索
        for sq in search_queries[:4]:  # 限制查询数量
            vector_results = await self._vector_search(sq, candidate_count)
            if vector_results:
                all_result_lists.append(vector_results)
                if "vector" not in pipeline_result.methods_used:
                    pipeline_result.methods_used.append("vector")

        # 路2: BM25 关键词检索
        if self.enable_bm25:
            await self._ensure_bm25_index()
            if self._bm25 and self._bm25.document_count > 0:
                for sq in search_queries[:2]:
                    bm25_results = self._bm25.search(sq, top_k=candidate_count)
                    if bm25_results:
                        all_result_lists.append(bm25_results)
                        if "bm25" not in pipeline_result.methods_used:
                            pipeline_result.methods_used.append("bm25")

        if not all_result_lists:
            return pipeline_result

        # ===== 第3步：RRF 融合 =====
        fused_results = rrf_fusion(all_result_lists)
        pipeline_result.initial_candidates = len(fused_results)

        # ===== 第4步：Reranker 精排 =====
        primary_query = rewrite_result.rewritten_query if rewrite_result else query

        if self.enable_reranker and self.reranker and len(fused_results) > top_k:
            final_results = await self.reranker.rerank(
                query=primary_query,
                candidates=fused_results[:candidate_count],
                top_k=top_k,
            )
            pipeline_result.reranker_used = True
        else:
            final_results = fused_results[:top_k]

        pipeline_result.final_results = final_results
        return pipeline_result

    async def _vector_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """向量检索"""
        try:
            results = await self.rag_engine.search(query, top_k=top_k)
            return [
                RetrievalResult(
                    content=r["content"],
                    score=1.0 / (1.0 + r["score"]),  # L2距离转相似度分数
                    source=r.get("metadata", {}).get("source", ""),
                    chunk_index=r.get("metadata", {}).get("chunk_index", 0),
                    retrieval_method="vector",
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ]
        except Exception as e:
            print(f"⚠️ 向量检索失败: {e}")
            return []

    async def _ensure_bm25_index(self):
        """确保 BM25 索引已构建"""
        if self._bm25_synced:
            return

        try:
            # 从 ChromaDB 获取所有文档构建 BM25 索引
            collection = self.rag_engine.collection
            count = collection.count()
            if count == 0:
                return

            # 获取所有文档
            all_docs = collection.get(
                limit=min(count, 10000),  # 限制最大量
                include=["documents", "metadatas"],
            )

            if all_docs["documents"]:
                self._bm25 = BM25Retriever()
                self._bm25.add_documents(
                    documents=all_docs["documents"],
                    metadatas=all_docs["metadatas"],
                )
                self._bm25_synced = True
                print(f"✅ BM25 索引构建完成: {self._bm25.document_count} 条文档")
        except Exception as e:
            print(f"⚠️ BM25 索引构建失败: {e}")

    def invalidate_bm25_cache(self):
        """使 BM25 缓存失效（文档更新时调用）"""
        self._bm25_synced = False
        self._bm25 = None
