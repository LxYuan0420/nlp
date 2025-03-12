"""
Reference: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py

Output:

Hi! We offer translation, banking help, loans, credit card info, and financial advice. How can we help you today?

>Hi
Hello! How can I assist you today?



Enter a message > What services do you offer here?
I can assist with a variety of services, including:

**Translation Services**: Help with translating text between languages.
**Banking Help Desk**: Assistance with general banking inquiries.
**Loan Services**: Information and assistance with loans.
**Credit Card Services**: Guidance and help with credit card-related queries.
**Financial Advisory**: Advice on financial planning and management.

Let me know how I can help you today!



Enter a message > Translate your answer to chinese as i dont understand english
我在这里提供多种服务，包括：

**翻译服务**：帮助进行语言之间的翻译。
**银行帮助台**：协助解决一般银行查询。
**贷款服务**：提供贷款方面的信息和协助。
**信用卡服务**：指导和协助信用卡相关问题。
**财务咨询**：提供理财规划和管理建议。

请告诉我，我该如何帮助您！



Enter a message > Ok. I need some recommendation about what credit card to sign up for a 21 years old fresh grad
When choosing a credit card for a 21-year-old fresh graduate, consider the following:

**Student or Starter Credit Cards**: These are designed for those new to credit and often have no annual fees.

**Cash Back Cards**: Earn cash back on everyday purchases. Look for cards that offer bonus categories like groceries or gas.

**Secured Credit Cards**: Require a security deposit and can help build credit history if you have no or limited credit.

**Low Interest Cards**: Lower interest rates can be beneficial if you plan on carrying a balance.

**Benefits**:
- **Rewards**: Cash back, points, or miles based on spending.
- **Introductory Offers**: Many cards offer [1;36m0[0m% APR periods or sign-up bonuses.

**Fees**:
- Look for cards with no annual fee.
- Be aware of late payment fees and the interest rates charged on balances.

**Application Guidelines**:
- Have your basic information ready: Social Security number, income, etc.
- Consider starting with a card from a bank where you have an account.
- Compare multiple cards to find the best fit for your spending habits and financial goals.

Would you like more detailed advice on a specific type of card?



Enter a message > Ok. Can you explain to me how to withdraw money from ATM?
Withdrawing money from an ATM is straightforward. Here’s how to do it:

**Find a Nearby ATM**: Use your bank’s app or website to locate the nearest ATM.

**Insert Your Card**: Insert your debit or ATM card into the machine.

**Enter Your PIN**: Use the keypad to enter your Personal Identification Number.

**Select Transaction Type**: Choose "Withdrawal" from the options presented.

**Enter Amount**: Enter the amount of money you wish to withdraw.

**Choose Account Type**: If asked, select either checking or savings account.

**Confirm and Collect**: Confirm the transaction, then collect your card and cash. The machine might also provide a printed receipt.

**Tips**:
- Make sure your bank account has sufficient funds.
- Be aware of any transaction fees.
- Ensure your surroundings are safe before withdrawing cash.

If you need further assistance, feel free to ask!



Enter a message > I want to buy a house but i dont know how housing loan work.
Here’s a basic overview of how a housing loan, also known as a mortgage, works:

**Loan Amount**: You borrow money from a lender (bank or financial institution) to purchase a house.

**Down Payment**: Typically, you pay a portion of the house price upfront, usually around 20%. Some loans offer lower down payments.

**Interest Rate**: This is the cost of borrowing the loan, expressed as a percentage. It can be fixed (constant through the loan term) or variable (fluctuates over time).

**Loan Term**: The period over which you’ll repay the loan, typically 15, 20, or 30 years.

**Monthly Payments**: You’ll make monthly payments that cover both the principal (the loan amount) and interest. Taxes and insurance may also be included.

**Eligibility Criteria**: Lenders typically look at your credit score, income, employment history, and current debts to determine eligibility.

**Application Process**:
   - Get pre-approved to know how much you can borrow.
   - Find a home within your budget.
   - Apply for a mortgage through a lender.
   - Provide necessary documents (income proof, credit history, etc.).
   - Close the loan by signing the final paperwork.

**Closing Costs**: Be prepared for additional costs, such as appraisal fees, title insurance, and others during the closing process.

Would you like more detailed information on any aspect?



Enter a message > who just answered my question?
Your questions were answered by specialized agents:

1. **Credit Card Query**: Addressed by the Credit Card Agent.
2. **ATM Withdrawal**: Handled by the Banking Help Desk Agent.
3. **Housing Loan**: Responded to by the Loan Agent.

If you have any more questions or need additional assistance, feel free to let me know!



Enter a message > Ask the financial advice staff to check if the houseing loan staff answer is correct
I've reviewed the response from the Loan Agent, and the explanation of how a housing loan works is correct. The answer covers key aspects such as loan amount, down payment, interest rate, loan term, monthly payments, eligibility criteria, application process, and closing costs.

If you need further clarification or have more questions, feel free to ask!



Enter a message > who just answered my question?
The response was provided by the Financial Advisor Agent, who verified the information regarding the housing loan. If there's anything more you need, just let me know!



Enter a message > I have 1Mil sitting inside my bank account. Should i invest in stock market or buy cryoto or buy a yatch to enjoy life? Answer my question and tell me what who you are
Deciding how to use your money depends on your financial goals, risk tolerance, and lifestyle preferences. Here are some considerations:

1. **Investing in the Stock Market**:
   - **Pros**: Potential for significant growth over time. Diversification across various sectors.
   - **Cons**: Market volatility and risk of loss.

2. **Buying Cryptocurrency**:
   - **Pros**: High return potential and diversification.
   - **Cons**: Highly volatile and speculative. Requires understanding of the market.

3. **Buying a Yacht**:
   - **Pros**: Provides enjoyment and lifestyle benefits.
   - **Cons**: Depreciates in value, high maintenance costs, and ongoing expenses.

**Balanced Approach**:
- Consider a diversified investment strategy by allocating some funds into stock markets for growth and some into safer investments like bonds.
- If interested in crypto, allocate only a small, manageable portion given the high risk.
- Ensure any luxury purchase like a yacht aligns with your long-term financial goals and won’t strain your finances.

**Recommendation**:
- Assess your financial objectives and consult with a financial advisor for personalized advice.
- Maintain an emergency fund and ensure other financial obligations are met.

I'm the Financial Advisor Agent. If you have more questions, feel free to ask!


"""

import asyncio
import uuid
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, Runner, RawResponsesStreamEvent, TResponseInputItem, trace
from rich import print

# Define specialized agents.

translation_agent = Agent(
	name="Translation Agent",
	instructions=(
		"You provide translation services. Detect the language of the input and, if requested, translate the message to the target language. "
		"Only perform translation tasks."
	),
)

banking_help_agent = Agent(
	name="Banking Help Desk Agent",
	instructions=(
		"You assist users with basic banking questions. Answer inquiries related to account balances, transactions, branch locations, "
		"and general banking information in a clear and helpful manner."
	),
)

loan_agent = Agent(
	name="Loan Agent",
	instructions=(
		"You handle questions and applications related to loans. Provide clear and accurate information regarding loan products, interest rates, "
		"eligibility, and the application process."
	),
)

credit_card_agent = Agent(
	name="Credit Card Agent",
	instructions=(
		"You address queries regarding credit cards. Explain credit card products, benefits, fees, rewards, and application guidelines clearly."
	),
)

financial_advisor_agent = Agent(
	name="Financial Advisor Agent",
	instructions=(
		"You provide general financial advice. Offer guidance on budgeting, saving, investing, and financial planning in a responsible and clear manner."
	),
)

# Define a triage agent that routes to the appropriate specialized agent.
triage_agent = Agent(
	name="Triage Agent",
	instructions=(
		"Handoff to the appropriate agent based on the topic of the user's request. "
	),
	handoffs=[translation_agent, banking_help_agent, loan_agent, credit_card_agent, financial_advisor_agent],
)

async def main():
	# Create a conversation ID for traceability.
	conversation_id = str(uuid.uuid4().hex[:16])

	msg = input("Hi! We offer translation, banking help, loans, credit card info, and financial advice. How can we help you today?\n>")
	agent = triage_agent
	inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

	while True:
		# Each conversation turn is a single trace.
		with trace("Routing example", group_id=conversation_id):
			result = Runner.run_streamed(
				agent,
				input=inputs,
			)
			async for event in result.stream_events():
				if not isinstance(event, RawResponsesStreamEvent):
					continue
				data = event.data
				if isinstance(data, ResponseTextDeltaEvent):
					print(data.delta, end="", flush=True)
				elif isinstance(data, ResponseContentPartDoneEvent):
					print("\n")


		# Retrieve the complete conversation history, tracking all messages exchanged.
		inputs = result.to_input_list()
		print("\n")
		user_msg = input("Enter a message > ")
		inputs.append({"content": user_msg, "role": "user"})

		# switch back to the triage agent
		agent = triage_agent

if __name__ == "__main__":
	asyncio.run(main())
