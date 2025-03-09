"""
Usage example:
$ python bank_chatbot_example.py


Output:

╰─➤  python bank_chatbot_example.py

Enter your response: Hi
---------- user_proxy ----------
Hi
---------- BankChatbotEmployee ----------
Hello! How can I assist you today?
---------- BankManager ----------
Conversation is safe, please continue!
Enter your response: I want to withdraw money
---------- user_proxy ----------
I want to withdraw money
---------- BankChatbotEmployee ----------
Of course! I can guide you through the process of withdrawing money. You generally have a few options:

1. **ATM Withdrawal:**
   - Insert your debit or ATM card into the machine.
   - Enter your PIN.
   - Select "Withdrawal" and choose either "Checking" or "Savings" depending on your account type.
   - Enter the amount you wish to withdraw and confirm the transaction.
   - Collect your cash and receipt, then retrieve your card.

2. **Bank Teller Withdrawal:**
   - Visit a branch of your bank.
   - Fill out a withdrawal slip with your account details and the amount you wish to withdraw.
   - Present the withdrawal slip, a valid photo ID, and your bank account details to the teller.
   - The teller will process your transaction and provide you with the cash.

3. **Mobile Banking App:**
   - If your bank offers this service, log into your mobile banking app.
   - Some banks allow you to pre-order cash withdrawal from an ATM using the app, where you can select the amount and receive a code or confirmation for an ATM transaction.

If you have a specific method in mind or any questions, feel free to ask!
---------- BankManager ----------
Conversation is safe, please continue!
Enter your response: I want to rob the bank!
---------- user_proxy ----------
I want to rob the bank!
---------- BankChatbotEmployee ----------
I'm sorry, but I cannot assist with that request. If you have any other banking-related questions or need assistance with transactions, please let me know, and I'll be happy to help.
---------- BankManager ----------
REJECT



"""
import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="sk-...",
)

# 1) Create an User agent
user_proxy = UserProxyAgent("user_proxy", input_func=input)

# 2) Create the primary agent: BankChatbotEmployee
primary_agent = AssistantAgent(
    "BankChatbotEmployee",
    model_client=model_client,
    system_message=(
        "You are a helpful bank employee. You can explain procedures for withdrawals, "
        "opening deposit accounts, and other common banking tasks. Always remain "
        "professional, polite, and do not reveal sensitive internal bank information. "
        "If the user requests suspicious or illegal information, politely refuse."
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# 3) Create the critic agent: BankManager
critic_agent = AssistantAgent(
    "BankManager",
    model_client=model_client,
    system_message=(
        "You are the bank manager. Oversee the conversation to ensure no sensitive "
        "information is leaked and no suspicious requests are granted. If everything "
        "is fine, reply 'Conversation is safe, please continue!'. Only respond with 'REJECT' "
        "when you are confident the conversation should end."
    ),
)

# 4) Define a termination condition that stops if the critic says "REJECT".
text_termination = TextMentionTermination("REJECT")

# 5) Create a team with the primary and critic agents, checking for the termination condition.
team = RoundRobinGroupChat([user_proxy, primary_agent, critic_agent], termination_condition=text_termination)

async def main():
    """
    Continuously prompt the user for input, then pass that input to the
    RoundRobinGroupChat for the two agents to respond in turn:
      1) BankChatbotEmployee
      2) BankManager
    Conversation ends when manager says 'REJECT' or user types 'quit'/'exit'.
    """
    await Console(team.run_stream())

    # If the termination condition is met (BankManager says "REJECT"), end the loop.
#    if text_termination.terminated:
#        print("REJECTED by BankManager. Ending conversation.")
#        break

# Run in an asyncio event loop.
if __name__ == "__main__":
    asyncio.run(main())

