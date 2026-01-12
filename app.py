import streamlit as st
from transformers import pipeline
import datetime
import re

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=128
    )

llm = load_llm()

st.title("🎓 Student Assistant AI Agent")
st.subheader("Aligned with SDG 4: Quality Education")

st.write("This AI agent helps students with study notes, reminders, calculations, and academic support.")

# ---------------- TOOLS ----------------

def calculator_tool(text):
    try:
        expression = re.findall(r"[0-9]+|[\+\-\*/]", text)
        return f"Calculation result: {eval(''.join(expression))}"
    except:
        return "Sorry, I couldn't calculate that."

def time_tool():
    return f"Current time is {datetime.datetime.now().strftime('%H:%M:%S')}"

def notes_tool(topic):
    prompt = f"""
Create short and simple study notes in 5 bullet points.
Topic: {topic}
"""
    return llm(prompt)[0]["generated_text"].strip()

# Memory using Streamlit session state
if "reminders" not in st.session_state:
    st.session_state.reminders = []

def reminder_tool(text):
    st.session_state.reminders.append(text)
    return "✅ Reminder saved successfully."

def show_reminders():
    if not st.session_state.reminders:
        return "No reminders set."
    return "\n".join([f"- {r}" for r in st.session_state.reminders])

# ---------------- AGENT BRAIN ----------------

def decide_and_act(user_input):
    decision_prompt = f"""
You are an AI agent with these tools:
CALCULATOR
TIME
NOTES
REMINDER
SHOW_REMINDERS
CHAT

Rules:
- Math → CALCULATOR
- Ask time → TIME
- Ask notes or explain topic → NOTES
- Ask to remember something → REMINDER
- Ask to show reminders → SHOW_REMINDERS
- Otherwise → CHAT

Respond with ONE tool name only.

User Input: {user_input}
Decision:
"""

    decision = llm(decision_prompt)[0]["generated_text"].strip().upper()

    if "CALCULATOR" in decision:
        return calculator_tool(user_input)
    elif "TIME" in decision:
        return time_tool()
    elif "NOTES" in decision:
        return notes_tool(user_input)
    elif "REMINDER" in decision:
        return reminder_tool(user_input)
    elif "SHOW" in decision:
        return show_reminders()
    else:
        return llm(user_input)[0]["generated_text"].strip()

# ---------------- UI INTERACTION ----------------

user_input = st.text_input("Ask the AI agent (notes, time, reminders, calculations):")

if st.button("Submit"):
    if user_input:
        with st.spinner("Agent is thinking..."):
            response = decide_and_act(user_input)

        st.markdown("### 🤖 Agent Response")
        st.write(response)
    else:
        st.warning("Please enter a question.")

# ---------------- ETHICS ----------------

st.markdown("---")
st.markdown("### ⚖️ Ethical & Responsible AI")
st.write("""
- No personal data is stored  
- Agent responses are advisory  
- Designed for student learning support  
- Promotes inclusive education  
- Aligned with SDG 4: Quality Education  
""")
