# home-generative-agent

[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)

![Project Maintenance][maintenance-shield]

This README is the documentation for a [Home Assistant](https://www.home-assistant.io/) (HA) integration called home-generative-agent. This project uses [LangChain](https://www.langchain.com/) and [LangGraph](https://www.langchain.com/langgraph) to create a [generative AI agent](https://arxiv.org/abs/2304.03442#) that interacts with and automates tasks within a HA smart home environment. The agent understands your home's context, learns your preferences, and interacts with you and your home to accomplish activities you find valuable. Key features include creating automations, analyzing images, and managing home states using various LLMs (Large Language Models). The architecture involves both cloud-based and edge-based models for optimal performance and cost-effectiveness. Installation instructions, configuration details, and information on the project's architecture and the different models used are included. The project is open-source and welcomes contributions.

These are some of the features currently supported:

- Create complex Home Assistant automations.
- Image scene analysis and understanding.
- Home state analysis of entities, devices, and areas.
- Full agent control of allowed entities in the home.
- Short- and long-term memory using semantic search.
- Automatic summarization of home state to manage LLM context length.

This integration will set up the `conversation` platform, a convenient HA component allowing users to converse directly with the Home Generative Assistant.

## Example Use Cases
### Create an automation.
![Alt text](./assets/automation1.png)

### Create an automation that runs periodically.
![Alt text](./assets/cat_automation.png)

The snippet below shows that the agent is fluent in yaml based on what it generated and registered as an HA automation.

```yaml
alias: Check Litter Box Waste Drawer
triggers:
  - minutes: /30
    trigger: time_pattern
conditions:
  - condition: numeric_state
    entity_id: sensor.litter_robot_4_waste_drawer
    above: 90
actions:
  - data:
      message: The Litter Box waste drawer is more than 90% full!
    action: notify.notify
```

### Check a single camera.
![Alt text](./assets/one_camera.png)

### Check multiple cameras.
https://github.com/user-attachments/assets/230baae5-8702-4375-a3f0-ffa981ee66a3

### Check the history of a light.
![Alt text](./assets/history1.png)

### Report the energy consumption of an appliance.
![Alt text](./assets/fridge_energy_1.png) ![Alt text](./assets/fridge_energy_2.png)

### Summarize home state.
https://github.com/user-attachments/assets/96f834a8-58cc-4bd9-a899-4604c1103a98

You can create an automation of the home state summary that runs periodically from the HA Blueprint `hga_summary.yaml` located in the `blueprints` folder.

### Long-term memory with semantic search.
![Alt text](./assets/semantic1.png)
![Alt text](./assets/semantic2.png)
![Alt text](./assets/semantic3.png)

You can see that the agent correctly generates the automation below.
```yaml
alias: Prepare Home for Arrival
description: Turn on front porch light and unlock garage door lock at 7:30 PM
mode: single
triggers:
 - at: "19:30:00"
    trigger: time
actions:
  - target:
      entity_id: light.front_porch_light
    action: light.turn_on
    data: {}
  - target:
      entity_id: lock.garage_door_lock
    action: lock.unlock
    data: {}
```

### Check a camera for packages.
![Alt text](./assets/check-for-boxes.png)

Below is the camera image the agent analyzed, you can see that two packages are visible. 

![Alt text](./assets/check-for-boxes-pic.png)

### Proactive notification of package delivery.
![Alt text](./assets/proactive-camera-automation.png)

Below is an example notification from this automation if any boxes or packages are visible.

![Alt text](./assets/proactive-notification.png)

The agent uses a tool that in turn uses the HA Blueprint `hga_scene_analysis.yaml` for these requests and so the Blueprint needs to be installed in your HA installation.

### Proactive Camera Video Analysis.

You can enable proactive video scene analysis from cameras visible to Home Assistant. When enabled, motion detection will trigger the analysis which will be stored in a database for use by the agent, and optionally, notifications of the analysis will be sent to the mobile app. You can also enable anomaly detection which will only send notifications based on semantic search of the current analysis vis-a-vis the database. These options are set in the integration's config UI.

The image below is an example of a notification sent to the mobile app.

![Alt text](./assets/video-analysis-screenshot.jpeg)

## Architecture and Design

Below is a high-level view of the architecture.

![Alt text](./assets/hga_arch.png)

The general integration architecture follows the best practices as described in [Home Assistant Core](https://developers.home-assistant.io/docs/development_index/) and is compliant with [Home Assistant Community Store](https://www.hacs.xyz/) (HACS) publishing requirements.

The agent is built using LangGraph and uses the HA `conversation` component to interact with the user. The agent uses the Home Assistant LLM API to fetch the state of the home and understand the HA native tools it has at its disposal. I implemented all other tools available to the agent using LangChain. The agent employs several LLMs, a large and very accurate primary model for high-level reasoning, smaller specialized helper models for camera image analysis, primary model context summarization, and embedding generation for long-term semantic search. The primary model can be either cloud (best accuracy, highest cost) or edge-based (good accuracy, lowest cost) and the helper models are all edge-based. The edge models run under the [Ollama](https://ollama.com/) framework on a computer located in the home. The models currently being used are summarized below.

Model | Location | Purpose
-- | -- | -- |
[GPT-4o](https://platform.openai.com/docs/models#gpt-4o) | OpenAI Cloud | High-level reasoning and planning
[qwen3:8b](https://ollama.com/library/qwen3) | Ollama Edge | High-level reasoning and planning
[qwen2.5vl:7b](https://ollama.com/library/qwen2.5vl) | Ollama Edge | Image scene analysis
[qwen3:1.7bb](https://ollama.com/library/qwen3) | Ollama Edge | Primary model context summarization
[mxbai-embed-large](https://ollama.com/library/mxbai-embed-large) | Ollama Edge | Embedding generation for sematic search

### LangGraph-based Agent
LangGraph powers the conversation agent, enabling you to create stateful, multi-actor applications utilizing LLMs as quickly as possible. It extends LangChain's capabilities, introducing the ability to create and manage cyclical graphs essential for developing complex agent runtimes. A graph models the agent workflow, as seen in the image below.

![Alt text](./assets/graph.png)

The agent workflow has five nodes, each Python module modifying the agent's state, a shared data structure. The edges between the nodes represent the allowed transitions between them, with solid lines unconditional and dashed lines conditional. Nodes do the work, and edges tell what to do next.

The ```__start__``` and ```__end__``` nodes inform the graph where to start and stop. The ```agent``` node runs the primary LLM, and if it decides to use a tool, the ```action``` node runs the tool and then returns control to the ```agent```. The ```summarize_and_remove_messages``` node processes the LLM's context to manage growth while maintaining accuracy if ```agent``` has no tool to call and message trimming is required to manage the LLM context.

### LLM Context Management
You need to carefully manage the context length of LLMs to balance cost, accuracy, and latency and avoid triggering rate limits such as OpenAI's Tokens per Minute restriction. The system controls the context length of the primary model by trimming the messages in the context if they exceed a max parameter which can be expressed in either tokens or messages, and the trimmed messages are replaced by a shorter summary inserted into the system message. These parameters are configurable in `const.py`; their description is below.

Parameter | Description | Default
-- | -- | -- |
`CONTEXT_MAX_MESSAGES` |  Messages to keep in context before deletion | 80
`CONTEXT_MAX_TOKENS` | Tokens to keep in context before deletion | 25600
`CONTEXT_MANAGE_USE_TOKENS` | If True, use tokens to manage context, else use messages | True

### Latency
The latency between user requests or the agent taking timely action on the user's behalf is critical for you to consider in the design. I used several techniques to reduce latency, including using specialized, smaller helper LLMs running on the edge and facilitating primary model prompt caching by structuring the prompts to put static content, such as instructions and examples, upfront and variable content, such as user-specific information at the end. These techniques also reduce primary model usage costs considerably.

You can see the typical latency performance in the table below.

Action | Latency (s) | Remark
-- | -- | -- |
HA intents | < 1 | e.g., turn on a light
Analyze camera image | < 3 | initial request
Add automation | < 1 |
Memory operations | < 1 |

### Tools
The agent can use HA tools as specified in the [LLM API](https://developers.home-assistant.io/docs/core/llm/) and other tools built in the LangChain framework as defined in `tools.py`. Additionally, you can extend the LLM API with tools of your own as well. The code gives the primary LLM the list of tools it can call, along with instructions on using them in its system message and in the docstring of the tool's Python function definition. If the agent decides to use a tool, the LangGraph node `action` is entered, and the node's code runs the tool. The node uses a simple error recovery mechanism that will ask the agent to try calling the tool again with corrected parameters in the event of making a mistake.

The LLM API instructs the agent always to call tools using HA [built-in intents](https://developers.home-assistant.io/docs/intent_builtin) when controlling Home Assistant and to use the intents `HassTurnOn` to lock and `HassTurnOff` to unlock a lock. An intent describes a user's intention generated by user actions.

You can see the list of LangChain tools that the agent can use in the table below.

Langchain Tool | Purpose
-- | -- |
`get_and_analyze_camera_image` | run scene analysis on the image from a camera
`upsert_memory` | add or update a memory
`add_automation` | create and register a HA automation
`get_entity_history` | query HA database for entity history
<del>`get_current_device_state`</del> | <del>get the current state of one or more Home Assistant devices</del> (deprecated, using native HA GetLiveContext tool instead)

### Hardware
I built the HA installation on a Raspberry Pi 5 with SSD storage, Zigbee, and LAN connectivity. I deployed the edge models under Ollama on an Ubuntu-based server with an AMD 64-bit 3.4 GHz CPU, Nvidia 3090 GPU, and 64 GB system RAM. The server is on the same LAN as the Raspberry Pi.

## Installation

**Important Note**: This project has been tested only under Home Assistant Core in a manual development environment as described [here](https://developers.home-assistant.io/docs/development_environment#manual-environment). I have NOT tested it with docker or HAOS.

1. Install the [PostgreSQL](https://www.postgresql.org/) database. This allows for persistence storage of conversations and memories.
2. Install [pgvector](https://github.com/pgvector/pgvector). This enables vector similarity search for PostgresSQL.
3. Using the tool of choice, open your HA configuration's directory (folder) (where you find `configuration.yaml`).
4. If you do not have a `custom_components` directory (folder), you must create it.
5. In the `custom_components` directory (folder), create a new folder called `home_generative_agent`.
6. Download _all_ the files from the `custom_components/home_generative_agent/` directory (folder) in this repository.
7. Place the files you downloaded in the new directory (folder) you created.
8. Restart Home Assistant
9. In the HA UI, go to "Configuration" -> "Integrations" click "+," and search for "Home Generative Agent"
10. Install all the Blueprints in the `blueprints` directory (folder).
11. Install `ollama` on your edge device by following the instructions [here](https://ollama.com/download).
12. Pull `ollama` models `qwen3:8b`, `qwen3:1.7b`, `qwen2.5vl:7b` and `mxbai-embed-large`.

## Configuration
Configuration is done in the UI and via the parameters in `const.py`.

<!---->

## Contributions are welcome!

If you want to contribute to this, please read the [Contribution guidelines](CONTRIBUTING.md)

***

[home_generative_agent]: https://github.com/goruck/home-generative-agent
[commits-shield]: https://img.shields.io/github/commit-activity/y/goruck/home-generative-agent.svg?style=for-the-badge
[commits]: https://github.com/goruck/home-generative-agent/commits/main
[license-shield]: https://img.shields.io/github/license/goruck/home-generative-agent.svg?style=for-the-badge
[maintenance-shield]: https://img.shields.io/badge/maintainer-Lindo%20St%20Angel%20%40goruck-blue.svg?style=for-the-badge
[releases-shield]: https://img.shields.io/github/v/release/goruck/home-generative-agent.svg?style=for-the-badge
[releases]: https://github.com/goruck/home-generative-agent/releases
