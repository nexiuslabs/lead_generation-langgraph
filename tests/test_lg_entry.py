from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.constants import CONFIG_KEY_CHECKPOINTER
from langchain_core.runnables.config import var_child_runnable_config

from app import lg_entry


def test_normalize_node_merges_existing_state():
    existing_state = {
        "messages": [HumanMessage(content="confirm profile"), AIMessage(content="Prompting...")],
        "profile_workflow_focus": "company",
        "ask_counts": {"seeds": 1},
    }
    payload = {
        "state": existing_state,
        "input": {
            "messages": [
                {
                    "role": "human",
                    "content": "Acme — https://acme.com\nGlobex — https://globex.com",
                }
            ]
        },
    }

    merged = lg_entry.normalize_payload(payload)

    assert merged.get("profile_workflow_focus") == "company"
    assert merged.get("ask_counts") == {"seeds": 1}
    assert len(merged.get("messages") or []) == 1
    last = merged.get("messages")[-1]
    assert isinstance(last, HumanMessage)
    assert "Acme" in (last.content or "")


def test_normalize_node_passthrough_for_graph_state():
    graph_state = {
        "messages": [HumanMessage(content="hello"), AIMessage(content="hi!")],
        "profile_workflow_focus": "icp_profile",
    }

    out = lg_entry.normalize_payload(graph_state)

    assert out is graph_state


def test_load_checkpoint_state_returns_channel_values():
    class DummySaver:
        def __init__(self, state):
            self.state = state

        def get_tuple(self, config):
            return CheckpointTuple(
                config=config,
                checkpoint={"channel_values": self.state},
                metadata={},
                parent_config=None,
                pending_writes=[],
            )

    saver = DummySaver({"company_profile_confirmed": True})
    config = {"configurable": {"thread_id": "thread-123", CONFIG_KEY_CHECKPOINTER: saver}}

    restored = lg_entry._load_checkpoint_state(config)

    assert restored == {"company_profile_confirmed": True}


def test_load_checkpoint_state_missing_thread_returns_none():
    class NoopSaver:
        def get_tuple(self, config):
            raise AssertionError("should not be called")

    config = {"configurable": {CONFIG_KEY_CHECKPOINTER: NoopSaver()}}

    restored = lg_entry._load_checkpoint_state(config)

    assert restored is None


def test_load_checkpoint_state_uses_context_config():
    class DummySaver:
        def __init__(self, state):
            self.state = state

        def get_tuple(self, config):
            return CheckpointTuple(
                config=config,
                checkpoint={"channel_values": self.state},
                metadata={},
                parent_config=None,
                pending_writes=[],
            )

    saver = DummySaver({"company_profile_confirmed": True})
    config = {"configurable": {"thread_id": "ctx-thread", CONFIG_KEY_CHECKPOINTER: saver}}

    token = var_child_runnable_config.set(config)
    try:
        restored = lg_entry._load_checkpoint_state(None)
    finally:
        var_child_runnable_config.reset(token)

    assert restored == {"company_profile_confirmed": True}
