"""Langgraph graphs for Home Generative Agent."""
from __future__ import annotations  # noqa: I001

import copy
import json
import logging
from functools import partial
from typing import Any, Literal

import voluptuous as vol
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.helpers import llm
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

from .const import (
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONTEXT_MANAGE_USE_TOKENS,
    EDGE_CHAT_MODEL_REASONING_DELIMITER,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    SUMMARY_INITIAL_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    CONF_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
)
from langchain_google_genai import ChatGoogleGenerativeAI # For type checking if needed

LOGGER = logging.getLogger(__name__)

class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]

async def _call_model(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[BaseMessage]]:
    """Coroutine to call the model."""
    model = config["configurable"]["chat_model"]
    prompt = config["configurable"]["prompt"]
    user_id = config["configurable"]["user_id"]
    hass = config["configurable"]["hass"]
    #current_model_id = model.model_name

    current_model_id = getattr(model, "model_name", None)
    if current_model_id is None:
        current_model_id = getattr(model, "model", "unknown_model_id") # Fallback for Ollama or if model_name not found

    # Retrieve most recent or search for most relevant memories for context.
    # Use semantic search if the last message was from the user.


    # Retrieve most recent or search for most relevant memories for context.
    # Use semantic search if the last message was from the user.
    last_message = state["messages"][-1]
    last_message_from_user = isinstance(last_message, HumanMessage)
    query_prompt = EMBEDDING_MODEL_PROMPT_TEMPLATE.format(
        query=last_message.content
    ) if last_message_from_user else None
    mems = await store.asearch(
        (user_id, "memories"),
        query=query_prompt,
        limit=10
    )

    # Retrieve most recent camera activity from video analysis by the VLM.
    camera_activity: list[dict[str, str]] = []
    for entity_id in hass.states.async_entity_ids():
        if entity_id.startswith("camera."):
            camera_name = entity_id.split(".")[-1]
            s = await store.asearch(
                ("video_analysis", camera_name),
                limit=1
            )
            if s:
                camera_activity.append(
                    {
                        camera_name: {
                            "last activity": s[0].value.get("content"),
                            "datetime": s[0].updated_at.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                )

    # Form the System Message from the base prompt plus memories, recent camera activity
    # and past conversation summaries, if they exist.
    system_message = prompt

    if mems:
        formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
        system_message += f"\n<memories>\n{formatted_mems}\n</memories>"

    if camera_activity:
        ca = "\n".join(str(a) for a in camera_activity)
        system_message += f"\n<recent_camera_activity>\n{ca}\n</recent_camera_activity>"

    if (summary := state.get("summary", "")):
        system_message += (
            f"\n<past_conversation_summary>\n{summary}\n</past_conversation_summary>"
        )

    # Model input is the System Message plus current messages.
    messages = [SystemMessage(content=system_message)] + state["messages"]

    # Trim messages to manage context window length.
    # TODO - if using the token counter from the chat model API, the method
    # 'get_num_tokens_from_messages()' will be called which currently ignores
    # tool schemas and under counts message tokens for the qwen models.
    # Until this is fixed, 'max_tokens' should be set to a value less than
    # the maximum size of the model's context window. See const.py.

    models_to_skip_token_counting = ["gpt-4.1-mini", "gpt-4.1"]
    skip_token_counting_for_model = current_model_id in models_to_skip_token_counting

    if skip_token_counting_for_model:
        LOGGER.debug(
            "Skipping explicit token counting for model %s as it is in the skip list.",
            current_model_id,
        )
        # num_tokens was only used for logging, can be set to None or 0
        # num_tokens = None
    else:
        num_tokens = await hass.async_add_executor_job(
            model.get_num_tokens_from_messages, messages
        )
        LOGGER.debug("Token count in messages from token counter: %s", num_tokens)

    if CONTEXT_MANAGE_USE_TOKENS:
        if skip_token_counting_for_model:
            LOGGER.debug(
                "Model %s is in skip list for token counting. "
                "Using message count for trimming context.",
                current_model_id,
            )
            # This branch implies CONTEXT_MAX_MESSAGES should be used, which is a message count.
            # However, the variable is named max_tokens_for_trimming. This might need review.
            # For now, assuming this path is for models where token counting is skipped.
            token_counter_for_trimming = len
        else:
            max_tokens_for_trimming = config["configurable"]["max_input_tokens_for_trimming"]
            token_counter_for_trimming = model  # Use the model's token counter
    else:
        max_tokens_for_trimming = config["configurable"].get("max_messages_for_trimming", 80) # Default to CONTEXT_MAX_MESSAGES
        token_counter_for_trimming = len

    trimmed_messages = await hass.async_add_executor_job(
        partial(
            trim_messages,
            messages=messages,
            token_counter=token_counter_for_trimming,
            max_tokens=max_tokens_for_trimming,
            strategy="last",
            start_on="human",
            include_system=True,
        )
    )

    LOGGER.debug("Model call messages: %s", trimmed_messages)
    LOGGER.debug("Model call messages length: %s", len(trimmed_messages))

    raw_response = await model.ainvoke(trimmed_messages)
    LOGGER.debug("Raw chat model response: %s", raw_response)
    # Clean up raw response.
    response: str = raw_response.content
    # If model used reasoning, just use the final result.
    first, sep, last = response.partition(
        EDGE_CHAT_MODEL_REASONING_DELIMITER.get("end", "")
    )
    response = last.strip("\n") if sep else first.strip("\n")
    # Create AI message, no need to include tool call metadata if there's none.
    if hasattr(raw_response, "tool_calls"):
        ai_response = AIMessage(content=response, tool_calls=raw_response.tool_calls)
    else:
        ai_response = AIMessage(content=response)
    LOGGER.debug("AI response: %s", ai_response)

    metadata = raw_response.usage_metadata if hasattr(
        raw_response, "usage_metadata"
    ) else {}
    LOGGER.debug("Token counts from metadata: %s", metadata)

    messages_to_remove = [m for m in state["messages"] if m not in trimmed_messages]
    LOGGER.debug("Messages to remove: %s", messages_to_remove)

    return {
        "messages": ai_response,
        "chat_model_usage_metadata": metadata,
        "messages_to_remove": messages_to_remove,
    }

async def _summarize_and_remove_messages(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, str | list[AnyMessage]]:
    """Coroutine to summarize and remove messages."""
    summary = state.get("summary", "")
    msgs_to_remove = state.get("messages_to_remove", [])

    if not msgs_to_remove:
        return {"summary": summary}

    if summary:
        summary_message = SUMMARY_PROMPT_TEMPLATE.format(summary=summary)
    else:
        summary_message = SUMMARY_INITIAL_PROMPT

    # Form the messages that will be used by the summarization model.
    # The summary will be based on the messages that were trimmed away from the main
    # model call, ignoring those from tools since the AI message encapsulates them.
    messages = (
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)] +
        [m.content for m in msgs_to_remove if isinstance(m, HumanMessage|AIMessage)] +
        [HumanMessage(content=summary_message)]
    )

    model = config["configurable"]["summarization_model"]
    options = config["configurable"]["options"]
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_SUMMARIZATION_MODEL,
                RECOMMENDED_SUMMARIZATION_MODEL,
            ),
            "temperature": options.get(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ),
            "num_predict": SUMMARIZATION_MODEL_PREDICT,
            "num_ctx": SUMMARIZATION_MODEL_CTX,
        }
    )

    LOGGER.debug("Summary messages: %s", messages)
    response = await model_with_config.ainvoke(messages)
    LOGGER.debug("Summary response: %s", response)

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in msgs_to_remove],
    }

async def _call_tools(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[ToolMessage]]:
    """Coroutine to call Home Assistant or langchain LLM tools."""
    # Tool calls will be the last message in state.
    tool_calls = state["messages"][-1].tool_calls

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]

    tool_responses: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        LOGGER.debug(
            "Tool call: %s(%s)", tool_name, tool_args
        )

        def _handle_tool_error(err:str, name:str, tid:str) -> ToolMessage:
            return ToolMessage(
                content=TOOL_CALL_ERROR_TEMPLATE.format(error=err),
                name=name,
                tool_call_id=tid,
                status="error",
            )

        # A langchain tool was called.
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]

            # Provide hidden args to tool at runtime.
            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update(
                {
                    "store": store,
                    "config": config,
                }
            )

            try:
                tool_response = await lc_tool.ainvoke(tool_call_copy)
            except (HomeAssistantError, ValidationError) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])
        # A Home Assistant tool was called.
        else:
            tool_input = llm.ToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            try:
                response = await ha_llm_api.async_call_tool(tool_input)

                tool_response = ToolMessage(
                    content=json.dumps(response),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            except (HomeAssistantError, vol.Invalid) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])

        LOGGER.debug("Tool response: %s", tool_response)
        tool_responses.append(tool_response)
    return {"messages": tool_responses}

def _should_continue(
        state: State
    ) -> Literal["action", "summarize_and_remove_messages"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]

    if messages[-1].tool_calls:
        return "action"

    return "summarize_and_remove_messages"

# Define a new graph
workflow = StateGraph(State)

# Define nodes.
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_remove_messages", _summarize_and_remove_messages)

# Define edges.
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("summarize_and_remove_messages", END)
