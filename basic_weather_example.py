"""
This script sets up an AssistantAgent to provide weather information for a
specified city using a simulated weather tool. The agent is configured to
interact with a model client and stream responses to the console.

How to Run:
1. Ensure you have the necessary dependencies installed, including `autogen_agentchat`, `autogen_ext`, and `dotenv`.
2. Set up your environment variables using a `.env` file, including your OpenAI API key if needed.
3. Run the script using Python, ensuring you have an event loop running, such as with `asyncio.run(main())`.

Output:
The script will output the current weather for the specified city, in this
case, New York, to the console. The output will be a message indicating the
temperature and weather conditions, e.g., "The weather in New York is 73
degrees and Sunny."


---------- user ----------
What is the weather in New York?
---------- weather_agent ----------
[FunctionCall(id='call_6y5GwjCV1YgE2AUMMzLIRIJM', arguments='{"city":"New York"}', name='get_weather')]
---------- weather_agent ----------
[FunctionExecutionResult(content='The weather in New York is 73 degrees and Sunny.', name='get_weather', call_id='call_6y5GwjCV1YgE2AUMMzLIRIJM', is_error=False)]
---------- weather_agent ----------
The weather in New York is currently 73 degrees and sunny.

"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="YOUR_API_KEY",
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
#await main()
asyncio.run(main())
