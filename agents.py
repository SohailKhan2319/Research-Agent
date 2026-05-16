import os
import sys
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import create_react_agent, AgentExecutor

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from tools import scrape_url, web_search
except ImportError:
    print("❌ tools.py missing!")
    sys.exit(1)

load_dotenv()

# --- OLLAMA CONFIGURATION ---
# Note: Using llama3.2:3b for better reliability with ReAct formatting
llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0,
    base_url="http://localhost:11434" 
)

# --- 1. AGENT SECTION (PromptTemplate) ---
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(template)

def build_search_agent():
    tools = [web_search] 
    agent = create_react_agent(llm, tools, react_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def build_reader_agent():
    tools = [scrape_url]
    agent = create_react_agent(llm, tools, react_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --- 2. CHAIN SECTION (ChatPromptTemplate) ---

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Create clear, structured reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure: Introduction, Key Findings, and Conclusion."""),
])
writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp research critic. Provide a score and constructive feedback."),
    ("human", "Review this report and provide a score (X/10) and improvements:\n\n{report}"),
])
critic_chain = critic_prompt | llm | StrOutputParser()