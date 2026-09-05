import asyncio
import atexit
import queue
import threading
import time
import traceback

import streamlit as st

from retrieval_agent import create_agent
from reasoning_agent import analyze_query_intent_tool
from llama_index.core.workflow import Context
from llama_index.core.memory import Memory
from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult

st.set_page_config(
    page_title="JIRA Retrieval Assistant",
    page_icon="🤖",
    layout="wide",
)


IDLE_TIMEOUT_SECONDS = 120


class AgentRuntime:

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._build_agent(), self.loop)
        self.agent, self.driver = future.result()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _build_agent(self):
        return create_agent()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self):
        async def _close():
            await self.driver.close()

        try:
            self.run(_close())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


@st.cache_resource
def get_runtime() -> AgentRuntime:
    runtime = AgentRuntime()
    atexit.register(runtime.close)
    return runtime


runtime = get_runtime()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_logs" not in st.session_state:
    st.session_state.tool_logs = []


async def _build_context():
    return Context(runtime.agent)

retrieval_context = runtime.run(_build_context())
memory = Memory.from_defaults(session_id="jira_chat", token_limit=8000)


def reset_conversation():

    for key in ("messages", "tool_logs", "retrieval_context", "memory"):
        st.session_state.pop(key, None)


async def process_query_stream(query: str, q: queue.Queue):

    assistant_response = ""
    call_counter = 0

    try:
        intent = analyze_query_intent_tool(query)

        agent_input = f"""
        USER QUERY:
        {query}

        ANALYZED INTENT:
        {intent}

        Use the analyzed intent to choose the appropriate retrieval
        strategy and answer the user's question.
        """

        handler = runtime.agent.run(
            user_msg=agent_input,
            ctx=retrieval_context,
            memory=memory,
        )

        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                assistant_response += event.delta
                q.put({"type": "delta", "content": event.delta})

            elif isinstance(event, ToolCall):
                call_counter += 1
                tool_id = event.tool_id or f"{event.tool_name}-{call_counter}"
                q.put({
                    "type": "tool_start",
                    "id": tool_id,
                    "tool_name": event.tool_name,
                    "args": event.tool_kwargs,
                })

            elif isinstance(event, ToolCallResult):
                tool_id = event.tool_id or f"{event.tool_name}-{call_counter}"
                q.put({
                    "type": "tool_end",
                    "id": tool_id,
                    "tool_name": event.tool_name,
                    "output": str(event.tool_output),
                    "is_error": bool(getattr(event.tool_output, "is_error", False)),
                })

        response = await handler

        if not assistant_response:
            assistant_response = str(response)
            q.put({"type": "delta", "content": assistant_response})

        q.put({"type": "done", "full_response": assistant_response})

    except Exception as e:
        q.put({
            "type": "error",
            "message": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "partial_response": assistant_response,
        })


st.title("🤖 JIRA Retrieval Assistant")
col_caption, col_reset = st.columns([5, 1])

with col_caption:
    st.caption("Hybrid retrieval + reasoning agent")
with col_reset:
    if st.button("🔄 Reset", help="Clear the conversation and start a fresh context"):
        reset_conversation()
        st.rerun()

chat_col, tool_col = st.columns(2, gap="large")

with chat_col:
    st.subheader("💬 Conversation")
    chat_container = st.container(height=650)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with tool_col:
    st.subheader("🔧 Agent Activity")
    tool_container = st.container(height=650)
    with tool_container:
        if not st.session_state.tool_logs:
            st.info("Tool calls and retrieval activity will appear here.")
        else:
            for log in st.session_state.tool_logs:
                state = "error" if log.get("is_error") else "complete"
                icon = "⚠️" if log.get("is_error") else "✅"
                with st.status(f"{icon} `{log['tool_name']}`", state=state, expanded=False):
                    st.caption("Arguments")
                    st.json(log["args"])
                    st.caption("Result")
                    st.write(log["output"])

query = st.chat_input("Ask something about your JIRA incidents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(query)

    with chat_container:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("🤔 Thinking...")

    q = queue.Queue()
    runtime.submit(process_query_stream(query, q))

    full_response = ""
    new_logs = []
    active_statuses = {} 
    finished = False
    timed_out = False
    last_activity = time.monotonic()

    while not finished:
        try:
            item = q.get(timeout=0.1)
            last_activity = time.monotonic()
        except queue.Empty:
            if time.monotonic() - last_activity > IDLE_TIMEOUT_SECONDS:
                timed_out = True
                break
            continue

        if item["type"] == "delta":
            full_response += item["content"]
            response_placeholder.markdown(full_response + "▌")

        elif item["type"] == "tool_start":
            with tool_container:
                status = st.status(f"🔧 Calling `{item['tool_name']}`", state="running", expanded=True)
                with status:
                    st.caption("Arguments")
                    st.json(item["args"])
            active_statuses[item["id"]] = (status, item["tool_name"], item["args"])

        elif item["type"] == "tool_end":
            entry = active_statuses.pop(item["id"], None)
            tool_name = entry[1] if entry else item["tool_name"]
            args = entry[2] if entry else {}
            is_error = item.get("is_error", False)
            if entry:
                status = entry[0]
                status.update(
                    label=f"{'⚠️' if is_error else '✅'} `{tool_name}`",
                    state="error" if is_error else "complete",
                )
                with status:
                    st.caption("Result")
                    st.write(item["output"])
            new_logs.append({
                "tool_name": tool_name,
                "args": args,
                "output": item["output"],
                "is_error": is_error,
            })

        elif item["type"] == "done":
            full_response = item["full_response"]
            response_placeholder.markdown(full_response)
            finished = True

        elif item["type"] == "error":
            # Mark any tool calls that never got a matching result as interrupted.
            for status, tool_name, _args in active_statuses.values():
                status.update(label=f"⚠️ `{tool_name}` (interrupted)", state="error")
            active_statuses.clear()

            full_response = item.get("partial_response") or ""
            full_response += f"\n\n⚠️ **I ran into an error and had to stop:** {item['message']}"
            response_placeholder.markdown(full_response)

            with tool_container:
                with st.expander("🐛 Error details", expanded=False):
                    st.code(item["traceback"])
            print(item["traceback"])  # also surface in server logs
            finished = True

    if timed_out:
        for status, tool_name, _args in active_statuses.values():
            status.update(label=f"⚠️ `{tool_name}` (timed out)", state="error")
        full_response = full_response or ""
        full_response += (
            "\n\n⚠️ **No response after "
            f"{IDLE_TIMEOUT_SECONDS}s, so I stopped waiting.** "
            "Try again, or hit Reset if this keeps happening."
        )
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.tool_logs.extend(new_logs)

    st.rerun()