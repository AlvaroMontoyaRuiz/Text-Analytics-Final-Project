#!/usr/bin/env python3
"""
agents.py
Defines a HYBRID multi-agent system.
- ManagerAgent: Uses Google Gemini Pro for classification.
- Worker Agents: Use OpenAI gpt-4o-mini for task execution.
"""
import os
import json
import traceback
from typing import List, Dict, Optional, Any, Generator

# --- Import both clients ---
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import (
    FunctionDeclaration, 
    GenerationConfig,
    HarmCategory, 
    HarmBlockThreshold
)

# Import the tool functions from tools.py
import tools

# --- OpenAI Tool Definitions (for Worker Agents) ---
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

# Map tool names to their actual Python functions (for worker agents)
TOOL_REGISTRY = {
    "retrieve_patient_information": tools.retrieve_patient_information,
}


# --- Gemini Tool Definition (for Manager Agent) ---
MANAGER_FUNCTION_DECLARATION = FunctionDeclaration(
    name="route_query",
    description="Classifies the user's query and routes it to the correct agent. Use the definitions in the 'intent' parameter to determine the correct route.",
    parameters={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": (
                    "The classified intent of the user's query. "
                    "Use 'Patient History' for queries about past medical events, symptoms, diagnoses, family history, social habits, or background summaries. "
                    "Use 'Discharge' for queries about the current hospital stay, treatments, hospital course, discharge medications, instructions, or diagnosis. "
                    "Use 'Ambiguous' if the query is unclear, too short, or fits both."
                ),
                "enum": ["Patient History", "Discharge", "Ambiguous"]
            }
        },
        "required": ["intent"]
    },
)

# Gemini safety settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
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
        print(f"Initialized OpenAI Agent {self.name} with model {self.model}.")

    def _execute_tool_loop(
        self, 
        system_prompt: str, 
        user_message: str, 
        patient_id: Optional[str],
        patient_name: Optional[str],
        chat_history: List[Dict],
        max_iterations: int = 10
    ) -> List[Dict]:
        print(f"\n--- {self.name} EXECUTING TOOL LOOP ---")
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
                    temperature=0.1,
                    stream=False 
                )
                assistant_message = response.choices[0].message
                
                if assistant_message.tool_calls:
                    messages.append(assistant_message.model_dump(exclude_none=True))
                    for tool_call in assistant_message.tool_calls:
                        
                        # --- THIS IS THE FIX ---
                        function_name = tool_call.function.name
                        # --- END OF FIX ---
                        
                        function_args = json.loads(tool_call.function.arguments)
                        if function_name not in TOOL_REGISTRY:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": f"Error: Unknown tool '{function_name}'"
                            })
                            continue
                        
                        function_args["patient_id"] = patient_id
                        function_args["patient_name"] = patient_name
                        
                        print(f" 🔧 Calling Tool: {function_name}({function_args})")
                        
                        tool_function = TOOL_REGISTRY[function_name]
                        
                        try:
                            tool_result = tool_function(**function_args)
                        except Exception as e:
                            print(f"Error calling tool {function_name}: {e}")
                            tool_result = f"Error executing tool {function_name}: {e}"

                        self.last_context = tool_result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_result
                        })
                else:
                    print(f"✅ {self.name} Tool Loop Complete.")
                    if assistant_message.content:
                        messages.append(assistant_message.model_dump(exclude_none=True))
                    return messages 
            
            print(f"Agent reached max iterations ({max_iterations}).")
            return messages
        
        except Exception as e:
            print(f"Error during {self.name} tool loop: {e}")
            traceback.print_exc()
            messages.append({"role": "assistant", "content": f"An error occurred in the {self.name}: {e}."})
            return messages

    def stream_response(
        self, 
        system_prompt: str, 
        user_message: str, 
        patient_id: Optional[str],
        patient_name: Optional[str],
        chat_history: List[Dict],
        max_iterations: int = 10
    ) -> Generator[str, None, None]:
        
        messages_with_context = self._execute_tool_loop(
            system_prompt,
            user_message,
            patient_id,
            patient_name,
            chat_history,
            max_iterations
        )
        
        if messages_with_context[-1]["role"] == "assistant" and "An error occurred" in messages_with_context[-1].get("content", ""):
            yield messages_with_context[-1]["content"]
            return

        print(f"--- {self.name} STREAMING FINAL ANSWER ---")
        
        try:
            final_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages_with_context,
                temperature=0.1,
                stream=True,
                tools=None,
                tool_choice=None
            )
            
            for chunk in final_stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error during {self.name} streaming: {e}")
            traceback.print_exc()
            yield f"An error occurred while streaming the response: {e}"


# --- Specialized Agents ---

class ManagerAgent:
    """
    Classifies user intent by FORCING a function call.
    This agent uses the Google Gemini Pro API.
    """
    def __init__(self, model_name: str):
        self.name = "Manager Agent (Gemini)"
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=[MANAGER_FUNCTION_DECLARATION], 
            generation_config=GenerationConfig(temperature=0.0),
            safety_settings=SAFETY_SETTINGS
        )
        print(f"Initialized {self.name} with model {model_name}.")

    def execute(self, query: str) -> str:
        """
        Classifies the user's query by forcing a tool call.
        """
        print(f"\n--- {self.name} EXECUTING (Forced Tool Call) ---")
        print(f"Query: {query}")
        
        try:
            tool_config = {"function_calling_config": {"mode": 'any', "allowed_function_names": ["route_query"]}}
            
            response = self.model.generate_content(
                query,
                tool_config=tool_config
            )
            
            if not response.candidates[0].content.parts or not response.candidates[0].content.parts[0].function_call:
                 print("Error: Manager agent did not call a function, it returned text.")
                 return "Ambiguous"

            fc = response.candidates[0].content.parts[0].function_call
            if fc.name == "route_query":
                classification = fc.args.get("intent", "Ambiguous")
                print(f"Classification: {classification}")
                return classification
            else:
                print(f"Error: Manager agent called an unexpected function: {fc.name}")
                return "Ambiguous"

        except Exception as e:
            print(f"Error during {self.name} execution: {e}")
            traceback.print_exc()
            return "Ambiguous" 


class PatientHistoryAgent(BaseAgent):
    """
    Agent specialized in summarizing and answering questions about patient history.
    (Uses OpenAI BaseAgent)
    """
    def __init__(self, client: OpenAI, model_name: str):
        self.SYSTEM_PROMPT = """
You are a specialized AI assistant for summarizing patient medical histories.
Your task is to **precisely answer the user's query** about a patient based *only* on the context provided by the `retrieve_patient_information` tool.

- If the user asks for a specific section (like **'HPI'** or **'social history'**), provide *only* that section.
- If the user asks for a **full history summary**, you can provide these key sections:
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

    def run(self, query: str, patient_id: Optional[str], patient_name: Optional[str], chat_history: List[Dict]):
        """Runs the agent's execution loop and returns a generator."""
        return self.stream_response(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=query,
            patient_id=patient_id,
            patient_name=patient_name,
            chat_history=chat_history
        )

class DischargeAgent(BaseAgent):
    """
    Agent specialized in summarizing hospital course and drafting discharge plans.
    (Uses OpenAI BaseAgent)
    """
    def __init__(self, client: OpenAI, model_name: str):
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

    def run(self, query: str, patient_id: Optional[str], patient_name: Optional[str], chat_history: List[Dict]):
        """Runs the agent's execution loop and returns a generator."""
        return self.stream_response(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=query,
            patient_id=patient_id,
            patient_name=patient_name,
            chat_history=chat_history
        )