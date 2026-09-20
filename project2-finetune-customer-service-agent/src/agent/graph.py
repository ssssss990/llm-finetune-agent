"""
LangGraph 客服 Agent 状态图

流程：理解 -> 检索 -> 澄清 -> 生成 -> 人工审核

- conditional edges 按置信度分流：
  - 高置信度直接生成
  - 低置信度走澄清节点
  - 退款金额超阈值走 human-in-the-loop

- checkpoint 落 Redis，支持断点续跑
"""

from typing import TypedDict, Optional, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger


class AgentState(TypedDict):
    messages: list
    query: str
    intent: str
    confidence: float
    retrieved_docs: list[dict]
    generated_response: str
    needs_clarification: bool
    needs_human_review: bool
    refund_amount: Optional[float]
    history: list


class CustomerServiceGraph:
    """客服 Agent 状态图"""

    def __init__(self, llm=None, retriever=None, checkpointer=None):
        self.llm = llm
        self.retriever = retriever
        self.checkpointer = checkpointer or MemorySaver()
        self.refund_threshold = 500.0
        self.confidence_threshold = 0.7
        self._graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 状态图"""
        graph = StateGraph(AgentState)

        graph.add_node("understand", self._understand_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("clarify", self._clarify_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("human_review", self._human_review_node)

        graph.set_entry_point("understand")

        graph.add_conditional_edges(
            "understand",
            self._route_after_understanding,
            {
                "retrieve": "retrieve",
                "clarify": "clarify",
            },
        )

        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "generate": "generate",
                "clarify": "clarify",
            },
        )

        graph.add_conditional_edges(
            "generate",
            self._route_after_generation,
            {
                "human_review": "human_review",
                "end": END,
            },
        )

        graph.add_edge("clarify", "retrieve")
        graph.add_edge("human_review", END)

        return graph.compile(checkpointer=self.checkpointer)

    async def run(
        self, query: str, thread_id: str = "default"
    ) -> dict:
        """执行客服流程"""
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            intent="",
            confidence=0.0,
            retrieved_docs=[],
            generated_response="",
            needs_clarification=False,
            needs_human_review=False,
            refund_amount=None,
            history=[],
        )

        result = await self._graph.ainvoke(initial_state, config=config)
        return result

    async def _understand_node(self, state: AgentState) -> dict:
        """理解节点：意图识别 + 置信度评估"""
        prompt = (
            f"用户问题：{state['query']}\n"
            "请判断意图类别（退换货/物流/发票/优惠券/账户异常）"
            "并给出置信度（0-1）。返回 JSON："
            '{"intent": "xxx", "confidence": 0.0}'
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        import json, re
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return {
                    "intent": parsed["intent"],
                    "confidence": parsed["confidence"],
                }
            except (json.JSONDecodeError, KeyError):
                pass
        return {"intent": "unknown", "confidence": 0.0}

    async def _retrieve_node(self, state: AgentState) -> dict:
        """检索节点：获取相关文档"""
        if self.retriever:
            docs = await self.retriever.ainvoke(state["query"])
            return {"retrieved_docs": docs}
        return {"retrieved_docs": []}

    async def _clarify_node(self, state: AgentState) -> dict:
        """澄清节点：向用户提问以补充信息"""
        prompt = (
            f"用户问题：{state['query']}\n"
            f"已识别意图：{state['intent']}\n"
            "信息不足，请生成一个简短的追问。"
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "generated_response": response.content,
            "needs_clarification": True,
        }

    async def _generate_node(self, state: AgentState) -> dict:
        """生成节点：生成客服回复"""
        context = "\n".join(
            [d.get("text", "") for d in state["retrieved_docs"][:3]]
        )
        prompt = (
            f"用户问题：{state['query']}\n"
            f"意图：{state['intent']}\n"
            f"参考信息：{context}\n\n"
            "请生成客服回复。要求：简洁、专业、有温度。"
            "不能做越权承诺。"
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        # 检查是否涉及退款金额
        import re
        amount_match = re.search(r'(\d+)\s*元', response.content)
        refund_amount = None
        if amount_match:
            refund_amount = float(amount_match.group(1))

        return {
            "generated_response": response.content,
            "refund_amount": refund_amount,
        }

    async def _human_review_node(self, state: AgentState) -> dict:
        """人工审核节点：等待人工确认"""
        return {
            "needs_human_review": True,
            "generated_response": (
                f"[待审核] 拟回复：{state['generated_response']}"
                f"\n退款金额：{state.get('refund_amount', '未知')}元"
                "\n请人工审核后确认发送。"
            ),
        }

    def _route_after_understanding(self, state: AgentState) -> str:
        """理解后路由：高置信度走检索，低置信度走澄清"""
        if state["confidence"] >= self.confidence_threshold:
            return "retrieve"
        return "clarify"

    def _route_after_retrieval(self, state: AgentState) -> str:
        """检索后路由：有足够文档走生成，否则澄清"""
        if len(state["retrieved_docs"]) >= 2:
            return "generate"
        return "clarify"

    def _route_after_generation(self, state: AgentState) -> str:
        """生成后路由：退款超阈值走人工审核"""
        if (state.get("refund_amount") and
                state["refund_amount"] > self.refund_threshold):
            return "human_review"
        return "end"
