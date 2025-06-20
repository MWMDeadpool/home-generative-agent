"""Constants for Home Generative Agent."""

# from google.genai.types import HarmCategory, HarmBlockThreshold # Removed
from typing import Literal

DOMAIN = "home_generative_agent"

### Configuration parameters that can be overridden in the integration's config UI. ###
# Name of the set of recommended options.
CONF_RECOMMENDED = "recommended"
# Name of system prompt.
CONF_PROMPT = "prompt"
# Run chat model in cloud or at edge.
CONF_CHAT_MODEL_LOCATION = "chat_model_location"
RECOMMENDED_CHAT_MODEL_LOCATION: Literal["cloud", "edge"] = "edge"
### OpenAI chat model parameters.
# See https://platform.openai.com/docs/api-reference/chat/create.
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "gpt-4.1-mini"
CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 1.0
### Ollama edge chat model parameters. ###
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_EDGE_CHAT_MODEL = "edge_chat_model"
RECOMMENDED_EDGE_CHAT_MODEL = "qwen3:8b"
CONF_EDGE_CHAT_MODEL_TEMPERATURE = "edge_chat_model_temperature"
RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE = 0.6
### Google Gemini chat model parameters. ###
CONF_GEMINI_API_KEY = "gemini_api_key"
CONF_GEMINI_CHAT_MODEL = "gemini_chat_model"
RECOMMENDED_GEMINI_CHAT_MODEL = "gemini-1.5-flash-latest" # Changed to a more stable model for v1beta API
CONF_GEMINI_CHAT_MODEL_TEMPERATURE = "gemini_chat_model_temperature"
RECOMMENDED_GEMINI_CHAT_MODEL_TEMPERATURE = 0.7
CONF_GEMINI_CHAT_MODEL_TOP_P = "gemini_chat_model_top_p"
RECOMMENDED_GEMINI_CHAT_MODEL_TOP_P = 0.95

# # Gemini Safety Settings - Removed
# CONF_GEMINI_SAFETY_HATE_SPEECH = "gemini_safety_hate_speech"
# CONF_GEMINI_SAFETY_HARASSMENT = "gemini_safety_harassment"
# CONF_GEMINI_SAFETY_SEXUALLY_EXPLICIT = "gemini_safety_sexually_explicit"
# CONF_GEMINI_SAFETY_DANGEROUS_CONTENT = "gemini_safety_dangerous_content"

# GEMINI_SAFETY_THRESHOLDS_MAP = {
#     "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
#     "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     "HARM_BLOCK_THRESHOLD_UNSPECIFIED": HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
# }

# RECOMMENDED_GEMINI_SAFETY_SETTINGS = {
#     CONF_GEMINI_SAFETY_HATE_SPEECH: "BLOCK_MEDIUM_AND_ABOVE",
#     CONF_GEMINI_SAFETY_HARASSMENT: "BLOCK_MEDIUM_AND_ABOVE",
#     CONF_GEMINI_SAFETY_SEXUALLY_EXPLICIT: "BLOCK_MEDIUM_AND_ABOVE",
#     CONF_GEMINI_SAFETY_DANGEROUS_CONTENT: "BLOCK_MEDIUM_AND_ABOVE",
# }


CONF_EDGE_CHAT_MODEL_TOP_P = "edge_chat_model_top_p"
RECOMMENDED_EDGE_CHAT_MODEL_TOP_P = 0.95
### Ollama vision language model (VLM) parameters. ###
# The VLM is used for vision and summarization tasks.
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_VLM = "vlm"
RECOMMENDED_VLM = "qwen2.5vl:7b"
CONF_VLM_TEMPERATURE = "vlm_temperature"
RECOMMENDED_VLM_TEMPERATURE = 0.0001
CONF_VLM_TOP_P = "vlm_top_p"
RECOMMENDED_VLM_TOP_P = 0.5
CONF_SUMMARIZATION_MODEL = "summarization_model"
RECOMMENDED_SUMMARIZATION_MODEL = "qwen3:1.7b"
CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0.6
CONF_SUMMARIZATION_MODEL_TOP_P = "summarization_model_top_p"
RECOMMENDED_SUMMARIZATION_MODEL_TOP_P = 0.95
### Ollama embedding model parameters. ###
# The embedding model is used for semantic search in long-term memory.
CONF_EMBEDDING_MODEL = "embedding_model"
RECOMMENDED_EMBEDDING_MODEL = "mxbai-embed-large"
### Camera video analyzer parameters. ###
CONF_VIDEO_ANALYZER_MODE = "video_analyzer_mode"
RECOMMENDED_VIDEO_ANALYZER_MODE: Literal[
    "disable", "notify_on_anomaly", "always_notify"
] = "disable"
# List of conversation agents that HGA can delegate tasks to.
CONF_DELEGATE_AGENTS = "delegate_agents"  # Stores list of agent_ids
CONF_DELEGATE_AGENT_DESCRIPTIONS = "delegate_agent_descriptions" # Stores dict[agent_id, description]

### langchain logging level ###
# See https://python.langchain.com/docs/how_to/debugging/
LANGCHAIN_LOGGING_LEVEL: Literal["disable", "verbose", "debug"] = "disable"

### Chat model context-related parameters. ###
# Sets the size of the context window used to generate the next token.
CHAT_MODEL_NUM_CTX = 12288
# Sets the maximum number of output tokens to generate (generic for most models).
CHAT_MODEL_MAX_TOKENS = 2048

# Context window sizes for different models (total tokens).
# These are illustrative; actual values depend on the specific model version.
OLLAMA_DEFAULT_MODEL_CTX = 12288 # Example for a typical Ollama model like qwen
GEMINI_1_5_FLASH_CTX = 1048576
GEMINI_PRO_CTX = 32768 # (30720 input + 2048 output)
OPENAI_GPT4_O_MINI_CTX = 128000 # gpt-4o-mini
OPENAI_GPT4_TURBO_CTX = 128000
OPENAI_GPT3_5_TURBO_16K_CTX = 16385
OPENAI_DEFAULT_CTX = 4096 # Fallback or for models like gpt-3.5-turbo (4k version)

# Fudge factor for Ollama token counting issues (tool schemas, undercounting)
OLLAMA_TOKEN_COUNT_FUDGE_FACTOR = 2048 + 4096

# Next parameters manage chat model context length.
# CONTEXT_MANAGE_USE_TOKENS = True manages chat model context size via token
# counting, if False management is done via message counting.
CONTEXT_MANAGE_USE_TOKENS = True
# CONTEXT_MAX_MESSAGES is messages to keep in context before deletion.
CONTEXT_MAX_MESSAGES = 80
# CONTEXT_MAX_INPUT_TOKENS (calculated in conversation.py) sets the limit on how
# large the input context can grow for the `trim_messages` function.
# This is derived from the model's total context window minus output tokens and any fudge factors.

# Old constant, will be replaced by model-specific calculations.
# For reference, this was: (OLLAMA_DEFAULT_MODEL_CTX - CHAT_MODEL_MAX_TOKENS - OLLAMA_TOKEN_COUNT_FUDGE_FACTOR)
# which is 12288 - 2048 - (2048 + 4096) = 4096
# We'll use this value as a default for Ollama/OpenAI if specific model context isn't identified.
DEFAULT_MAX_INPUT_TOKENS_FOR_TRIMMING = 4096

### Chat model tool error handling parameters. ###
TOOL_CALL_ERROR_SYSTEM_MESSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
"""

### Ollama edge chat model parameters. ###
# Edge chat model server URL.
EDGE_CHAT_MODEL_URL = "192.168.10.2:11434"
# Reasoning delimiters for models that use them in output.
# These may be model dependent, the defaults work for qwen3.
EDGE_CHAT_MODEL_REASONING_DELIMITER: dict[str, str] = {
    "start": "<think>", "end": "</think>"
}

### Ollama VLM parameters. ###
# Ollama VLM server URL.
VLM_URL = "192.168.10.2:11434"
# Ollama VLM maximum number of output tokens to generate.
VLM_NUM_PREDICT = 4096
# Sets the size of the context window used to generate the next token.
VLM_NUM_CTX = 8192
# Ollama VLM model prompts for vision tasks.
VLM_SYSTEM_PROMPT = """
You are a bot that responses with a description of what is visible in a camera image.

Keep your responses simple and to the point.
"""
VLM_USER_PROMPT = "Task: Describe this image:"
VLM_USER_KW_TEMPLATE = """
Task: Tell me if {key_words} are visible in this image:
"""
VLM_IMAGE_WIDTH = 1920
VLM_IMAGE_HEIGHT = 1080

### Ollama summarization model parameters. ###
# Model server URL.
SUMMARIZATION_MODEL_URL = "192.168.10.2:11434"
# Maximum number of tokens to predict when generating text.
SUMMARIZATION_MODEL_PREDICT = 4096
# Sets the size of the context window used to generate the next token.
SUMMARIZATION_MODEL_CTX = 8192
# Reasoning delimiters for models that use them in output.
# These may be model dependent, the defaults work for qwen3.
SUMMARIZATION_MODEL_REASONING_DELIMITER: dict[str, str] = {
    "start": "<think>", "end": "</think>"
}
# Model prompts for summary tasks.
SUMMARY_SYSTEM_PROMPT = "You are a bot that summarizes messages from a smart home AI."
SUMMARY_INITIAL_PROMPT = "Create a summary of the smart home messages above:"
SUMMARY_PROMPT_TEMPLATE = """
This is the summary of the smart home messages so far: {summary}

Update the summary by taking into account the additional smart home messages above:
"""

### Ollama embedding model parameters. ###
EMBEDDING_MODEL_URL = "192.168.10.2:11434"
EMBEDDING_MODEL_DIMS = 1024
EMBEDDING_MODEL_CTX = 512
EMBEDDING_MODEL_PROMPT_TEMPLATE = """
Represent this sentence for searching relevant passages: {query}
"""

### Tool parameters. ###
HISTORY_TOOL_CONTEXT_LIMIT = 50
HISTORY_TOOL_PURGE_KEEP_DAYS = 10 # TO-DO derive actual recorder setting
AUTOMATION_TOOL_EVENT_REGISTERED = "automation_registered_via_home_generative_agent"
AUTOMATION_TOOL_BLUEPRINT_NAME = "goruck/hga_scene_analysis.yaml"

### Camera video analyzer. ###
# Interval units are seconds.
VIDEO_ANALYZER_SCAN_INTERVAL = 1.5
# Root must be in allowlist_external_dirs.
VIDEO_ANALYZER_SNAPSHOT_ROOT = "/media/snapshots"
VIDEO_ANALYZER_SYSTEM_MESSAGE = """
You are a bot that generates a description of a video given descriptions of its frames.
Keep the description to the point and use no more than 250 characters.
"""
VIDEO_ANALYZER_PROMPT = """
Describe what is happening in this video from these frame descriptions:
"""
VIDEO_ANALYZER_MOBILE_APP = "mobile_app_darian_s_phone"
# Time offset units are minutes.
VIDEO_ANALYZER_TIME_OFFSET = 15
VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.8
VIDEO_ANALYZER_DELETE_SNAPSHOTS = True
VIDEO_ANALYZER_TRIGGER_ON_MOTION = True
VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP = 15
VIDEO_ANALYZER_MOTION_CAMERA_MAP: dict = {}

### postgresql db parameters for checkpointer and memory persistent storage. ###
DB_URI = "postgresql://hga:hga@192.168.10.2:5437/hga?sslmode=disable"
