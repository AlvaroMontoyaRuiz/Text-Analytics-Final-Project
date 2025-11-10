#!/usr/bin/env python3
"""
agents.py
Defines the multi-agent system (Manager, PatientHistory, Discharge).
All worker agents use a single RAG tool.
"""
import os
import json
import traceback
from typing import List, Dict, Optional, Any
from openai import OpenAI

# Import the *one* tool function from tools.py
import tools

# --- OpenAI Tool Definitions ---

# --- MODIFIED: A single tool declaration for all workers ---
# Now includes date fields
WORKER_TOOL_DECLARATION = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_patient_information",
            "description": "Retrieves all relevant patient note chunks for a specific patient. You can filter by ID, name, and dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific search query (e.g., 'symptoms', 'hospital course', 'medications')."
                    },
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's Subject ID or MRN (e.g., '10001401' or 'MRN 12345')."
                    },
                    "patient_name": {
                        "type": "string",
                        "description": "The patient's full name (e.g., 'Casey Gray')."
                    },
                    "admission_date": {
                        "type": "string",
                        "description": "The admission date to filter by. Must be in YYYY-MM-DD format (e.g., '2025-04-09')."
                    },
                    "discharge_date": {
                        "type": "string",
                        "description": "The discharge date to filter by. Must be in YYYY-MM-DD format (e.g., '2025-04-30')."
                    }
                },
                "required": ["query"]
            },
        }
    }
]

# Tool definition for the Manager Agent (Unchanged)
MANAGER_TOOL_DECLARATION = [
    {
        "type": "function",
        "function": {
            "name": "route_query",
            "description": "Routes the user's query to the correct agent based on its intent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "The classified intent of the user's query.",
                        "enum": ["Patient History", "Discharge", "Ambiguous"]
                    }
                },
                "required": ["intent"]
            },
        }
    }
]

# Map the single tool name to the function
TOOL_REGISTRY = {
    "retrieve_patient_information": tools.retrieve_patient_information,
}


# --- Base Agent Class (OpenAI) ---
class BaseAgent:
    """Base class for OpenAI agents with function calling"""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        name: str,
        tools: Optional[List[Dict]] = None,
    ):
        self.client = client
        self.model = model
        self.name = name
        self.tools = tools
        self.last_context: str = "" 
        print(f"Initialized {self.name} with model {self.model}.")

    def execute(
        self, 
        system_prompt: str, 
        user_message: str, 
        patient_id: Optional[str],
        patient_name: Optional[str],
        chat_history: List[Dict],
        max_iterations: int = 10
    ) -> str:
        """
        Execute the agent with a function calling loop.
        Injects patient_id or patient_name into the tool call.
        """
        print(f"\n--- {self.name} EXECUTING ---")
        print(f"Query: {user_message}")
        print(f"Patient ID: {patient_id}")
        print(f"Patient Name: {patient_name}")
        
        self.last_context = "" 
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        try:
            for iteration in range(max_iterations):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto" if self.tools else None,
                    temperature=0.1
                )
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    messages.append(assistant_message)
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name not in TOOL_REGISTRY:
                            return f"Error: Unknown tool '{function_name}'"

                        # --- MODIFIED: Securely merge identifiers ---
                        # Override/add the verified identifiers from the app.
                        # This prevents the LLM from hallucinating an ID.
                        function_args["patient_id"] = patient_id
                        function_args["patient_name"] = patient_name
                        
                        print(f" 🔧 Calling Tool: {function_name}({function_args})")
                        
                        tool_function = TOOL_REGISTRY[function_name]
                        
                        # --- MODIFIED: Use **kwargs to pass all args ---
                        # This now correctly passes query, patient_id, patient_name,
                        # and any dates the LLM extracted.
                        tool_result = tool_function(**function_args)

                        self.last_context = tool_result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_result
                        })
                else:
                    final_answer = assistant_message.content
                    print(f"✅ {self.name} Complete.")
                    return final_answer
            return f"Agent reached max iterations ({max_iterations})"
        except Exception as e:
            print(f"Error during {self.name} execution: {e}")
            traceback.print_exc()
            return f"An error occurred in the {self.name}: {e}. Please check the logs."


# --- Specialized Agents ---

# ManagerAgent is unchanged
class ManagerAgent:
    """
    Classifies user intent by FORCING a function call.
    """
    def __init__(self, client: OpenAI, model_name: str):
        self.name = "Manager Agent"
        self.client = client
        self.model = model_name
        self.system_prompt = """
You are the central manager for a clinical AI assistant. Your job is to classify the user's intent based on their query.
Analyze the query and determine if it relates to "Patient History" or "Discharge".

- **"Patient History"**: Queries about past medical events, symptoms, diagnoses, family history, social habits, or summaries of the patient's background.
- **"Discharge"**: Queries about the current hospital stay, treatments given, hospital course, discharge medications, discharge instructions, or discharge diagnosis/condition/disposition.
- **"Ambiguous"**: Use this if the query is unclear, too short, or could apply to both categories.

You MUST call the `route_query` function with your decision.
"""
        print(f"Initialized {self.name} with model {self.model}.")

    def execute(self, query: str) -> str:
        """
        Classifies the user's query by forcing a tool call.
        """
        print(f"\n--- {self.name} EXECUTING (Forced Tool Call) ---")
        print(f"Query: {query}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=MANAGER_TOOL_DECLARATION,
                tool_choice={"type": "function", "function": {"name": "route_query"}},
                temperature=0.0
            )
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "route_query":
                arguments = json.loads(tool_call.function.arguments)
                classification = arguments.get("intent", "Ambiguous")
                print(f"Classification: {classification}")
                return classification
            else:
                print("Error: Manager agent did not call the expected 'route_query' function.")
                return "Ambiguous"
        except Exception as e:
            print(f"Error during {self.name} execution: {e}")
            traceback.print_exc()
            return "Ambiguous"


class PatientHistoryAgent(BaseAgent):
    """
    Agent specialized in summarizing and answering questions about patient history.
    """
    def __init__(self, client: OpenAI, model_name: str):
        # --- MODIFIED: Prompt now mentions dates ---
        self.SYSTEM_PROMPT = """
You are a specialized AI assistant for summarizing patient medical histories.
Your task is to **precisely answer the user's query** about a patient based *only* on the context provided by the `retrieve_patient_information` tool.

- If the user asks for a specific section (like **'HPI'** or **'social history'**), provide *only* that section.
- If the user asks for a **full history summary**, you can provide the four key sections:
    1.  **History of Present Illness (HPI)**
    2.  **Past Medical History**
    3.  **Family History**
    4.  **Social History**

RULES:
- **Answer ONLY what the user asked for.** Do not add extra sections (like 'Family History') if they were not requested.
- **Pay close attention to dates.** If the user asks for information from a specific admission date (e.g., "on 2025-04-09"), you MUST extract this date and pass it to the `retrieve_patient_information` tool in YYYY-MM-DD format.
- **Grounding Failure**: If the retrieved context is insufficient to answer the user's *specific question*, you MUST state: 'Information not available in the provided documents.'
- **DO NOT** invent, infer, or use any outside knowledge.
- You must call the `retrieve_patient_information` tool with a relevant query.
"""
        super().__init__(
            client=client,
            model=model_name,
            name="Patient History Agent (OpenAI)",
            tools=WORKER_TOOL_DECLARATION
        )

    def run(self, query: str, patient_id: Optional[str], patient_name: Optional[str], chat_history: List[Dict]) -> str:
        """Runs the agent's execution loop."""
        return self.execute(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=query,
            patient_id=patient_id,
            patient_name=patient_name,
            chat_history=chat_history
        )

class DischargeAgent(BaseAgent):
    """
    Agent specialized in summarizing hospital course and drafting discharge plans.
    """
    def __init__(self, client: OpenAI, model_name: str):
        # --- MODIFIED: Prompt now mentions dates and all discharge sections ---
        self.SYSTEM_PROMPT = """
You are a specialized AI assistant for summarizing hospital visits and drafting discharge information.
Your task is to **precisely answer the user's query** based *only* on the context provided by the `retrieve_patient_information` tool.

- If the user asks for a specific piece of information (like **'hospital course'**, **'discharge medications'**, **'discharge instructions'**, **'discharge diagnosis'**, **'discharge condition'**, or **'discharge disposition'**), provide *only* that specific information.

- If the user asks for a **full discharge summary**, you MUST structure your answer with these key sections (if available):
    1.  **Brief Hospital Course**: Key events, treatments, and outcomes.
    2.  **Discharge Diagnosis**: The primary and secondary diagnoses.
    3.  **Discharge Condition**: The patient's condition on discharge.
    4.  **Discharge Disposition**: Where the patient is being discharged to (e.g., Home, Rehab).
    5.  **Discharge Medications**: A list or summary of medications.
    6.  **Discharge Instructions**: Specific instructions for the patient.

RULES:
- **Answer ONLY what the user asked for.** Do not add extra sections if they were not requested.
- **Pay close attention to dates.** If the user asks for information from a specific admission or discharge date (e.g., "from the 2025-04-09 admission"), you MUST extract this date and pass it to the `retrieve_patient_information` tool in YYYY-MM-DD format.
- **Grounding Failure**: If the retrieved context is insufficient to answer the user's *specific question*, you MUST state: 'Information not available in the provided documents.'
- **DO NOT** invent, infer, or use any outside knowledge.
- You must call the `retrieve_patient_information` tool with a relevant query.
"""
        super().__init__(
            client=client,
            model=model_name,
            name="Discharge Agent (OpenAI)",
            tools=WORKER_TOOL_DECLARATION
        )

    def run(self, query: str, patient_id: Optional[str], patient_name: Optional[str], chat_history: List[Dict]) -> str:
        """Runs the agent's execution loop."""
        return self.execute(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=query,
            patient_id=patient_id,
            patient_name=patient_name,
            chat_history=chat_history
        )