"""
Usage example:
    $ python funny_counting_example.py

Output:

---------- CounterAgent1 ----------
1
---------- CounterAgent2 ----------
2
---------- CounterAgent1 ----------
3
---------- CounterAgent2 ----------
4
---------- CounterAgent1 ----------
5
---------- CounterAgent2 ----------
6
---------- CounterAgent1 ----------
7
---------- CounterAgent2 ----------
8
---------- CounterAgent1 ----------
9
---------- CounterAgent2 ----------
10
---------- CounterAgent1 ----------
11
---------- CounterAgent2 ----------
12
---------- CounterAgent1 ----------
13
---------- CounterAgent2 ----------
14
---------- CounterAgent1 ----------
15
---------- CounterAgent2 ----------
16
---------- CounterAgent1 ----------
17
---------- CounterAgent2 ----------
18
---------- CounterAgent1 ----------
19
---------- CounterAgent2 ----------
20

"""
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

# Setup the OpenAI model client
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="sk-...",
)

# First Agent: CounterAgent1
counter_agent_1 = AssistantAgent(
    "CounterAgent1",
    model_client=model_client,
    system_message=(
        "You are CounterAgent1. Start counting with '1' and then wait for the next "
        "number from the other agent. Keep counting in ascending order, back and forth."
    ),
    reflect_on_tool_use=False,
    model_client_stream=False,
)

# Second Agent: CounterAgent2
counter_agent_2 = AssistantAgent(
    "CounterAgent2",
    model_client=model_client,
    system_message=(
        "You are CounterAgent2. Listen to CounterAgent1 start the count, then say the "
        "next number, continuing back and forth until reaching 20."
    ),
    reflect_on_tool_use=False,
    model_client_stream=False,
)

text_termination = TextMentionTermination("20")

# Create a team for counting
counting_team = RoundRobinGroupChat([counter_agent_1, counter_agent_2], termination_condition=text_termination )

async def main():
    """
    Agents take turns counting from 1 to 20.
    """
    await Console(counting_team.run_stream())

if __name__ == "__main__":
    asyncio.run(main())

