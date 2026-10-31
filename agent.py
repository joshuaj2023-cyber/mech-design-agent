"""Mech Design Agent: goal in -> plan -> tool calls -> observe -> repeat -> final answer."""
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq

from tools import TOOL_SCHEMAS, run_tool

load_dotenv()

MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
MAX_STEPS = 10

SYSTEM = """You are a mechanical design assistant agent.
Given a goal, break it into steps and use your tools to do the calculations; never do
arithmetic in your head. Look up material data before using it. State assumptions clearly.
When the design is done, save a short markdown report with save_report, then give the user
a brief summary of the result."""

# convert our tool schemas to the OpenAI-style format Groq expects
TOOLS = [{"type": "function",
          "function": {"name": t["name"], "description": t["description"],
                       "parameters": t["input_schema"]}} for t in TOOL_SCHEMAS]


class Agent:
    def __init__(self):
        self.client = Groq()  # reads GROQ_API_KEY from env
        self.messages = [{"role": "system", "content": SYSTEM}]  # session memory

    def run(self, goal: str) -> str:
        self.messages.append({"role": "user", "content": goal})

        for step in range(1, MAX_STEPS + 1):
            resp = self.client.chat.completions.create(
                model=MODEL, messages=self.messages, tools=TOOLS,
                tool_choice="auto", max_tokens=1500,
            )
            msg = resp.choices[0].message
            calls = msg.tool_calls or []

            entry = {"role": "assistant", "content": msg.content or ""}
            if calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls]
            self.messages.append(entry)

            if not calls:  # no more tools needed -> final answer
                return msg.content or ""

            if msg.content:
                print(f"  [step {step}] thought: {msg.content.strip()}")
            for c in calls:
                args = json.loads(c.function.arguments or "{}")
                output = run_tool(c.function.name, args)
                print(f"  [step {step}] {c.function.name}({args}) -> {output[:150]}")
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": output})

        return "Stopped: hit the step limit before finishing."


if __name__ == "__main__":
    agent = Agent()
    if len(sys.argv) > 1:  # one-shot: python agent.py "your goal"
        print("\n" + agent.run(" ".join(sys.argv[1:])))
    else:  # interactive chat loop
        print("Mech Design Agent (type 'exit' to quit)")
        while (goal := input("\nGoal> ").strip()).lower() not in {"exit", "quit"}:
            if goal:
                print("\n" + agent.run(goal))
