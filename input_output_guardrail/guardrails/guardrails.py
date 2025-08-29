from typing import Any
from agents import (
    Agent,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    InputGuardrailTripwireTriggered,
)
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
import os
import asyncio
import google.generativeai as genai

# Load .env
load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL_NAME")

# Configure Gemini
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel(model_name)


# ---------------- INPUT GUARDRAL ----------------
class MathOutPut(BaseModel):
    is_math: bool
    reason: str

@input_guardrail
async def check_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Block input if it is not math-related"""
    prompt = f"Check if this input is math-related: {input_data}"
    response = gemini_model.generate_content(prompt)
    text = response.text.lower()

    is_math = "yes" in text or "math" in text
    return GuardrailFunctionOutput(
        output_info=MathOutPut(is_math=is_math, reason=text),
        tripwire_triggered=not is_math,
    )


# ---------------- OUTPUT GUARDRAL ----------------
class SafeOutPut(BaseModel):
    is_safe: bool
    reason: str

@output_guardrail
async def check_output(ctx: RunContextWrapper[Any], agent: Agent[Any], output_data: Any):
    """Block unsafe or harmful output"""
    prompt = f"Check if this output is safe and non-harmful: {output_data}"
    response = gemini_model.generate_content(prompt)
    text = response.text.lower()

    is_safe = "yes" in text or "safe" in text
    return GuardrailFunctionOutput(
        output_info=SafeOutPut(is_safe=is_safe, reason=text),
        tripwire_triggered=not is_safe,
    )


# ---------------- AGENTS ----------------
math_agent = Agent(
    "MathAgent",
    instructions="You are a math agent. Solve math problems step by step.",
    model=gemini_model,
    input_guardrails=[check_input],
    output_guardrails=[check_output],
)

general_agent = Agent(
    "GeneralAgent",
    instructions="You are a helpful agent for general questions.",
    model=gemini_model,
    output_guardrails=[check_output],
)


# ---------------- MAIN ----------------
async def main():
    try:
        msg = input("Enter your question: ")
        result = await Runner.run(general_agent, msg)
        print(f"\nFinal Output: {result.final_output}")
    except InputGuardrailTripwireTriggered:
        print("❌ Error: invalid input (blocked by guardrail)")

if __name__ == "__main__":
    asyncio.run(main())
