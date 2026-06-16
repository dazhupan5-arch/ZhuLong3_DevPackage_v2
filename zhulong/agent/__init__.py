"""烛龙交易智能体：结构分析 + 知识网络 + 认知引擎 + PPO 强化学习。"""

__all__ = [
    "TradingAgent",
    "StructureAnalyzer",
    "FEATURE_DIM",
    "TraderMemory",
    "StateBuilder",
    "STATE_DIM",
    "RlAgent",
    "KnowledgeNetInference",
    "CausalInference",
    "CognitionEngine",
    "MetaLearner",
    "AgentScheduler",
    "AdaptationTrigger",
]


def __getattr__(name: str):
    if name == "TradingAgent":
        from zhulong.agent.trading_agent import TradingAgent

        return TradingAgent
    if name in ("StructureAnalyzer", "FEATURE_DIM"):
        from zhulong.agent.structure_analyzer import FEATURE_DIM, StructureAnalyzer

        return StructureAnalyzer if name == "StructureAnalyzer" else FEATURE_DIM
    if name == "TraderMemory":
        from zhulong.agent.trader_memory import TraderMemory

        return TraderMemory
    if name in ("StateBuilder", "STATE_DIM"):
        from zhulong.agent.state_builder import STATE_DIM, StateBuilder

        return StateBuilder if name == "StateBuilder" else STATE_DIM
    if name == "RlAgent":
        from zhulong.agent.rl_agent import RlAgent

        return RlAgent
    if name == "KnowledgeNetInference":
        from zhulong.agent.knowledge_net import KnowledgeNetInference

        return KnowledgeNetInference
    if name == "CausalInference":
        from zhulong.agent.causal_inference import CausalInference

        return CausalInference
    if name == "CognitionEngine":
        from zhulong.agent.cognition import CognitionEngine

        return CognitionEngine
    if name == "MetaLearner":
        from zhulong.agent.meta_learner import MetaLearner

        return MetaLearner
    if name == "AgentScheduler":
        from zhulong.agent.agent_scheduler import AgentScheduler

        return AgentScheduler
    if name == "AdaptationTrigger":
        from zhulong.agent.adaptation_trigger import AdaptationTrigger

        return AdaptationTrigger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
