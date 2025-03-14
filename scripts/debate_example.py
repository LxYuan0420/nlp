"""
Usage example:
    $ python debate_example.py | tee -a debate.log

Output:

---------- Proponent ----------
Sure, let's begin the debate. Please present your argument.
---------- Opponent ----------
The Proponent suggests that implementing a 4-day workweek will increase productivity due to improved employee well-being and work-life balance. However, this could lead to challenges in meeting customer demands and maintaining consistent service levels. Instead, my idea of flexible work hours tailored to each employee's needs allows businesses to cater to individual productivity patterns without disrupting customer service.
---------- Proponent ----------
The Opponent suggests that flexible work hours tailored to each employee's needs is a better solution than a 4-day workweek, as it may help in maintaining customer service levels. This approach might lead to scheduling complexities and inconsistencies in team collaboration, making it difficult to manage overall productivity effectively. In contrast, a 4-day workweek standardizes time off, which can streamline operations and yields substantial productivity gains from highly motivated employees.
---------- Opponent ----------
The Proponent argues that a 4-day workweek standardizes time off and can lead to substantial productivity gains. However, this rigidity does not accommodate the diverse working habits and personal situations of employees, potentially limiting flexibility. My idea of flexible work hours accounts for individual differences, ensuring both personal satisfaction and operational efficiency are maximized.
---------- Proponent ----------
The Opponent claims that flexible work hours accommodate diverse working habits and maximize both personal satisfaction and operational efficiency. Yet, this high degree of personalization can result in fragmentation and inefficiency, where team continuity and collaborative work might suffer. Conversely, a 4-day workweek provides a balanced and unified approach that enhances employee satisfaction while simplifying operational management and boosting productivity across the board.
---------- Opponent ----------
The Proponent believes that a 4-day workweek offers a unified approach that enhances employee satisfaction and simplifies management. Nevertheless, such blanket solutions may not be suitable for all industries or positions, leading to a possible mismatch between work schedules and business needs. On the other hand, my approach of flexible work hours provides tailored solutions that respect both the unique requirements of businesses and the diverse preferences of employees, ensuring continuity and efficient collaboration.
---------- Proponent ----------
The Opponent asserts that flexible work hours respect unique business requirements and employee preferences, ensuring efficient collaboration. Unfortunately, this method can create coordination challenges and lacks the clarity and predictability a business needs to function effectively, particularly across teams and departments. The 4-day workweek, however, provides clear boundaries and consistency that align with many industries' needs while boosting morale and productivity through a well-established rhythm.
---------- Opponent ----------
The Proponent posits that a 4-day workweek provides clarity and consistency while boosting morale and productivity through clear boundaries. However, this rigidity might hinder the adaptability required in today's fast-paced and diverse business environments, making it harder to accommodate urgent tasks or varying workloads. In contrast, flexible work hours offer the necessary versatility to adjust quickly to real-time demands, enhancing operational agility and maintaining a dynamic work culture.
---------- Proponent ----------
The Opponent argues that flexible work hours enhance operational agility and maintain a dynamic work culture by adjusting to real-time demands. However, such elasticity may lead to burnout and blurred work-life boundaries, undermining employees' overall well-being and long-term productivity. A 4-day workweek, on the other hand, enforces a structured respite that empowers employees to perform at their peak while ensuring a sustainable balance between work demands and personal life, ultimately fostering a healthier and more productive work environment.
---------- Opponent ----------
The Proponent suggests that a 4-day workweek enforces structured respite, fostering a healthier and more productive work environment. Yet, this approach might not address the varied personal and professional obligations employees face, potentially leaving some unable to fully benefit from this structure. My approach of flexible work hours allows each person to design their schedule around their life, creating a personalized balance that supports long-term well-being and sustained performance.
---------- Proponent ----------
The Opponent contends that flexible work hours allow for a personalized schedule that caters to individual lifestyles, promoting long-term well-being. However, this high level of personalization can lead to disparities in work distribution and makes it difficult to coordinate team efforts, impairing collective productivity. A 4-day workweek provides a consistent framework that balances individual well-being with organizational coherence, ultimately driving collective productivity and offering everyone the equal opportunity to recharge.
---------- Opponent ----------
The Proponent maintains that a 4-day workweek balances individual well-being with organizational coherence, enhancing collective productivity. However, this uniform approach may overlook the varying demands and peak productivity periods of different roles, potentially sidelining those who do not fit this schedule. By contrast, flexible work hours cater to specific job needs and personal preferences, facilitating better alignment between individual and organizational goals while maintaining team cohesion through strategic planning.
---------- Proponent ----------
The Opponent argues that flexible work hours cater to job-specific needs and personal preferences, potentially aligning individual and organizational goals. Nevertheless, this approach risks creating uneven availability, leading to scheduling conflicts and reduced synchronous collaboration, which can undermine team performance. A 4-day workweek offers structured, collective downtime that synchronizes team efforts and enhances overall efficiency while giving everyone equal access to regular rejuvenation periods.
---------- Opponent ----------
The Proponent argues that a 4-day workweek synchronizes team efforts and enhances efficiency through structured downtime. However, such uniformity could limit responsiveness and adaptability in industries that require continuous operation or flexible deadlines, creating potential bottlenecks. My approach of flexible work hours supports constant coverage and adaptability, allowing teams to strategically align their schedules to meet both business demands and individual work rhythms effectively.
---------- Proponent ----------
The Opponent suggests that flexible work hours support continuous operation and adaptability by allowing strategic alignment of schedules. However, this approach may lead to fragmented team availability and disrupt team dynamics, potentially lowering overall team synergy and cohesiveness. A 4-day workweek, in contrast, offers a standardized schedule that optimizes team alignment, ensures consistent collaboration, and enhances group efficiency while still providing sufficient time for rest and recovery.


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

# Proponent agent setup
proponent_agent = AssistantAgent(
    "Proponent",
    model_client=model_client,
    system_message=(
        "You are the Proponent in this debate. In each turn, summarize the Opponent's idea in one sentence, "
        "then state why it is bad in another sentence, and finally explain why your idea is better in one sentence."
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# Opponent agent setup
opponent_agent = AssistantAgent(
    "Opponent",
    model_client=model_client,
    system_message=(
        "You are the Opponent in this debate. In each turn, summarize the Proponent's idea in one sentence, "
        "then state why it is bad in another sentence, and finally explain why your idea is better in one sentence."
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# Termination condition for the debate
text_termination = TextMentionTermination("End debate")

# Create a team for the debate
debate_team = RoundRobinGroupChat([proponent_agent, opponent_agent], termination_condition=text_termination)

async def main():
    """
    Conduct a debate between Proponent and Opponent.
    Each agent will summarize the opponent's idea, criticize it, and advocate for their idea using three sentences.
    """
    await Console(debate_team.run_stream())

if __name__ == "__main__":
    asyncio.run(main())

