"""Config flow for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.components import conversation
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TemplateSelector,
)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .const import (
    CONF_CHAT_MODEL,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_CHAT_MODEL,
    CONF_GEMINI_CHAT_MODEL_TEMPERATURE,
    CONF_GEMINI_CHAT_MODEL_TOP_P,
    CONF_CHAT_MODEL_LOCATION,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL,
    CONF_EDGE_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL_TOP_P,
    CONF_EMBEDDING_MODEL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SUMMARIZATION_MODEL,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_DELEGATE_AGENTS,
    CONF_DELEGATE_AGENT_DESCRIPTIONS,
    CONF_VLM,
    CONF_VLM_TEMPERATURE,
    CONF_VLM_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_LOCATION,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_GEMINI_CHAT_MODEL,
    RECOMMENDED_GEMINI_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_GEMINI_CHAT_MODEL_TOP_P,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VLM,
    RECOMMENDED_VLM_TEMPERATURE,
    RECOMMENDED_VLM_TOP_P,
)

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_GEMINI_API_KEY): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_VIDEO_ANALYZER_MODE: "disable",
    CONF_DELEGATE_AGENTS: [],
    CONF_DELEGATE_AGENT_DESCRIPTIONS: {},
    CONF_GEMINI_CHAT_MODEL: RECOMMENDED_GEMINI_CHAT_MODEL,
    CONF_GEMINI_CHAT_MODEL_TEMPERATURE: RECOMMENDED_GEMINI_CHAT_MODEL_TEMPERATURE,
    CONF_GEMINI_CHAT_MODEL_TOP_P: RECOMMENDED_GEMINI_CHAT_MODEL_TOP_P,
}

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """
    Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    openai_api_key = data.get(CONF_API_KEY)
    if openai_api_key:
        client_openai = ChatOpenAI(
            api_key=openai_api_key, async_client=get_async_client(hass)
        )
        try:
            # A simple way to test the API key and connectivity
            await client_openai.ainvoke("Hi")
        except Exception as e:
            LOGGER.error("OpenAI API validation failed: %s", e)
            raise InvalidAuthError(f"OpenAI API key validation failed. Please check the key and network access. Error: {e}") from e

    gemini_api_key = data.get(CONF_GEMINI_API_KEY)
    if gemini_api_key:
        try:
            client_gemini = ChatGoogleGenerativeAI(
                model="gemini-pro",  # Use a common model for validation
                google_api_key=gemini_api_key,
            )
            await client_gemini.ainvoke("Hi")
        except Exception as e:
            LOGGER.error("Gemini API validation failed: %s", e)
            raise InvalidAuthError(f"Gemini API key validation failed. Please check the key and network access. Error: {e}") from e

class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""
    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except InvalidAuthError as e:
            errors["base"] = str(e)
        except CannotConnectError as e: # Assuming CannotConnectError is defined elsewhere or should be generic
            errors["base"] = str(e) if str(e) else "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="HGA",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return HomeGenerativeAgentOptionsFlow(config_entry)

class HomeGenerativeAgentOptionsFlow(OptionsFlow):
    """Config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input.get(CONF_LLM_HASS_API) == "none":
                    user_input.pop(CONF_LLM_HASS_API, None)

                # Store descriptions as plain text.
                # The input from the TextSelector for CONF_DELEGATE_AGENT_DESCRIPTIONS
                # is taken as a raw string. This changes the data type of this option
                # from a dictionary to a string.
                # Downstream code consuming this option will need to be adapted.
                descriptions_as_plain_text = user_input.get(CONF_DELEGATE_AGENT_DESCRIPTIONS, "")
                user_input[CONF_DELEGATE_AGENT_DESCRIPTIONS] = descriptions_as_plain_text

                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
                CONF_VIDEO_ANALYZER_MODE: user_input[CONF_VIDEO_ANALYZER_MODE],
                CONF_DELEGATE_AGENTS: user_input.get(CONF_DELEGATE_AGENTS, []), # list of IDs
                CONF_DELEGATE_AGENT_DESCRIPTIONS: user_input.get(CONF_DELEGATE_AGENT_DESCRIPTIONS, {}), # dict
            }

        schema = config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )

class CannotConnectError(HomeAssistantError):
    """Error to indicate we cannot connect."""

class InvalidAuthError(HomeAssistantError):
    """Error to indicate there is invalid auth."""

def config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> VolDictType:
    """Return a schema for completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    video_analyzer_mode: list[SelectOptionDict] = [
        SelectOptionDict(
            label="Disable",
            value="disable",
        ),
        SelectOptionDict(
            label="Notify on anomaly",
            value="notify_on_anomaly",
        ),
        SelectOptionDict(
            label="Always notify",
            value="always_notify",
        )
    ]

    # Domain for the "Extended OpenAI Conversation" integration
    # Adjust this if the actual domain is different (e.g., "openai_conversation")
    EXTENDED_OPENAI_DOMAIN = "extended_openai_conversation" # Or "openai_conversation"

    # Start with standard conversation entities
    available_agents: list[SelectOptionDict] = [
        SelectOptionDict(
            label=state.attributes.get("friendly_name", state.entity_id),
            value=state.entity_id,
        )
        for state in hass.states.async_all(conversation.DOMAIN)
        # Exclude HGA itself if it's already a conversation entity
        # This check might be more robust if HGA's entity_id is known here
        if not state.entity_id.startswith(f"conversation.{DOMAIN}") # Basic check
    ]
    added_agent_ids = {agent["value"] for agent in available_agents}

    # Add configured instances of "Extended OpenAI Conversation"
    for entry in hass.config_entries.async_entries(EXTENDED_OPENAI_DOMAIN):
        if entry.entry_id not in added_agent_ids:
            available_agents.append(
                SelectOptionDict(
                    label=f"{entry.title} (Extended OpenAI)", # Use entry title
                    value=entry.entry_id, # Use the config entry ID
                )
            )
            added_agent_ids.add(entry.entry_id)
    available_agents.sort(key=lambda x: x["label"])

    schema : VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_VIDEO_ANALYZER_MODE,
            description={"suggested_value": options.get(CONF_VIDEO_ANALYZER_MODE)},
            default=RECOMMENDED_VIDEO_ANALYZER_MODE
            ): SelectSelector(SelectSelectorConfig(options=video_analyzer_mode)),
        vol.Optional(
            CONF_DELEGATE_AGENTS,
            description={"suggested_value": options.get(CONF_DELEGATE_AGENTS)},
            default=[]
        ): SelectSelector(SelectSelectorConfig(options=available_agents, multiple=True, sort=True, custom_value=False)),
        vol.Optional(
            CONF_DELEGATE_AGENT_DESCRIPTIONS,
            description={"suggested_value": options.get(CONF_DELEGATE_AGENT_DESCRIPTIONS, {})},
            default={}
        ): TextSelector(TextSelectorConfig(multiline=True, type="text")),
        vol.Required(
            CONF_RECOMMENDED,
            description={"suggested_value": options.get(CONF_RECOMMENDED)},
            default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }
    if options.get(CONF_RECOMMENDED):
        return schema

    chat_model_location: list[SelectOptionDict] = [
        SelectOptionDict(
            label="cloud",
            value="cloud",
        ),
        SelectOptionDict(
            label="edge",
            value="edge",
        ),
        SelectOptionDict(
            label="gemini",
            value="gemini",
        )
    ]

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL_LOCATION,
                description={"suggested_value": options.get(CONF_CHAT_MODEL_LOCATION)},
                default=RECOMMENDED_CHAT_MODEL_LOCATION
                ): SelectSelector(SelectSelectorConfig(options=chat_model_location)),
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_CHAT_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_CHAT_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_CHAT_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_EDGE_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_EDGE_CHAT_MODEL)},
                default=RECOMMENDED_EDGE_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_EDGE_CHAT_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_EDGE_CHAT_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_EDGE_CHAT_MODEL_TOP_P,
                description={
                    "suggested_value": options.get(CONF_EDGE_CHAT_MODEL_TOP_P)
                },
                default=RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_VLM,
                description={"suggested_value": options.get(CONF_VLM)},
                default=RECOMMENDED_VLM,
            ): str,
            vol.Optional(
                CONF_VLM_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_VLM_TEMPERATURE)
                },
                default=RECOMMENDED_VLM_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_VLM_TOP_P,
                description={"suggested_value": options.get(CONF_VLM_TOP_P)},
                default=RECOMMENDED_VLM_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_SUMMARIZATION_MODEL,
                description={"suggested_value": options.get(CONF_SUMMARIZATION_MODEL)},
                default=RECOMMENDED_SUMMARIZATION_MODEL,
            ): str,
            vol.Optional(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_SUMMARIZATION_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                description={
                    "suggested_value": options.get(CONF_SUMMARIZATION_MODEL_TOP_P)
                },
                default=RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_EMBEDDING_MODEL,
                description={"suggested_value": options.get(CONF_EMBEDDING_MODEL)},
                default=RECOMMENDED_EMBEDDING_MODEL,
            ): str,
            vol.Optional(
                CONF_GEMINI_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_GEMINI_CHAT_MODEL)},
                default=RECOMMENDED_GEMINI_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_GEMINI_CHAT_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_GEMINI_CHAT_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_GEMINI_CHAT_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_GEMINI_CHAT_MODEL_TOP_P,
                description={"suggested_value": options.get(CONF_GEMINI_CHAT_MODEL_TOP_P)},
                default=RECOMMENDED_GEMINI_CHAT_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.01)),
        }
    )

    return schema
