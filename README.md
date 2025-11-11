# Text Analytics Final Project

## Project Description

**MedSynth:** *Intelligent Clinical Summarization* is a multi-agent RAG-based application that summarizes clincal histories, discharge notes and follow-up plans. It is designed for hospital-based providers such as hospitalists, Emergency Room doctors etc. The aim is to provide targeted summarization, transparent reasoning and retrieval traceability; so that we can help reduce medical practitioner administrative load.

## Key Features

1) Multi-agent structure
    * Manager Agent(Classifies intent): Google Gemini 2.5 Pro
        * Sub-agent: History Summarization (GPT-4o Mini) - Handles History of Present Illness + Past Medical History
        * Sub-agent: Discharge Generation (GPT-4o Mini) - Handles Hospital Course + Medication Retrieval

2) RAG Layer: Pinecone (Vector Database)
    * Hybrid Retrieval
        * Dense: Llama-text-embed-v2 used to understand the *meaning* of the query (e.g., "what drugs is she on?" is similar to "list medications")
        * Sparse: Pinecone is used for *precision* and to find exact terms like patient names ("Casey Gray") or specific IDs ("10001401").

3) LLM Layer: Google Gemini 2.5 Pro
    * Provides reasoning and guardrails
    * Structures summary to user
    * Stored in Session DB

## UI Screenshots

### Log-in Page

![Log-in page](https://raw.githubusercontent.com/AlvaroMontoyaRuiz/Text-Analytics-Final-Project/refs/heads/main/miscellaneous/Log-in%20page%20UI.png)

### Chat Page

![Chat page part 1](https://raw.githubusercontent.com/AlvaroMontoyaRuiz/Text-Analytics-Final-Project/refs/heads/main/miscellaneous/Chat%20page%20pt1.png)

![Chat page part 2](https://raw.githubusercontent.com/AlvaroMontoyaRuiz/Text-Analytics-Final-Project/refs/heads/main/miscellaneous/Chat%20page%20pt2.png)

## Setup Instructions

1) Clone Repository: [https://github.com/AlvaroMontoyaRuiz/Text-Analytics-Final-Project]

2) `pip install -r requirements.txt`

3) Copy `.env.example` to `secrets.toml` and add API keys

4) Initialize vector database: `python setup_db`

5) Run: `streamlit run app.py`

## How To

1) Go to [https://medsynth.streamlit.app/]
    * If screen unreadable, select `...` in top right hand corner
    * Select `settings`
    * Select 'Light mode' under 'Choose App Theme'

2) Select `Register`

3) Choose `Username` and `Password`, then select `Create Account`

4) Once account created, select `Login`

5) Once logged in, select blue `+ New Chat` button on left hand side of screen

6) New chat populates, enter first query, ex., 'What are all admission dates for Morgan Foster?'

7) Wait for bot response to generate information

8) Upon information generation, select second query, ex. 'Get the discharge medications for Morgan Foster from their 2025-09-17 admission.'

9) Wait for bot response to generate information

10) Upon information retrieval, select final query, e.x., 'Provide brief hospital course for Morgan Foster on 2025-09-17 admission.'

11) Wait for bot response to generate information

## Team member contributions

* Alvaro Montoya Ruiz: developed core architecture and UI of application. Took preliminary model from prior weekly assignment and refined model and UI for demo purposes.

* Hriday Reddy Purma: assisted in refining architecture and documentation. Outlined in-depth, project overview and key points.

* Mauricio Bermudez: Performed dataset ingestion and cleaned and filtered medical documentation. Performed in-depth code review to ensure well-structured codebase. Created presentations for final project.

## Technologies used (LLM, vector DB, tools, etc.)

1) LLMs
    * Google Gemini 2.5 Pro
    * GPT-4o Mini

2) Vector Database
    * Pinecone

3) Tools
    * LangChain
    * `route_query(intent):` Used by Manager Agent as a control
    * `retrieve_patient_information(query, patient_id, patient_name, ...):` Used by SubAgents  for document retrieval

4) Front-end
    * Streamlit

## Known limitations

1) Time spent on document retrieval can be improved in the future. Currently takes 45 seconds ~ 1 minute per query

2) Cannot provide medical advice or diagnoses

3) Queries must be specific based off demo

## Future improvements

1) UI Functionality to search for patients first, then using RAG to help speed process, so agents are not combing through all potential documentation

2) Improve UI for new user friendliness

3) In-depth query definitions to allow for larger variation of requests

## Disclaimer

`.env.example` contains non-existent API keys. The codes were randomly generated and are not tied to any application.
