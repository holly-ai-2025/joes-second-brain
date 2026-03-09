import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


class MessageInput(BaseModel):
    message: str


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    result = supabase.table("tasks").select("*").order("id").execute()
    return result.data


@app.post("/tasks")
def add_task(title: str):
    result = supabase.table("tasks").insert({"title": title}).execute()
    return result.data


@app.post("/agent")
def agent(input_data: MessageInput):
    prompt = f"""
You are a simple task router.

Return only valid JSON.

If the user wants to add a task, return:
{{"action": "add_task", "title": "task text"}}

If the user wants to list tasks, return:
{{"action": "list_tasks"}}

User message: {input_data.message}
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    decision = json.loads(response.output_text)

    if decision["action"] == "add_task":
        result = supabase.table("tasks").insert({"title": decision["title"]}).execute()
        return result.data

    if decision["action"] == "list_tasks":
        result = supabase.table("tasks").select("*").order("id").execute()
        return result.data

    return {"error": "Unknown action"}