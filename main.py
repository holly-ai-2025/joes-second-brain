import os
import json
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError, Field
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from context_builder import (
    build_context_pack,
    select_relevant_context,
    search_global_memory,
    format_context_for_prompt,
    format_global_context_for_prompt,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


class MessageInput(BaseModel):
    message: str = Field(..., min_length=1)


class AgentWithContextInput(BaseModel):
    conversation_id: str
    message: str = Field(..., min_length=1)


class RouteDecisionInput(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


class CompanionInput(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


class AddTaskArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ListTasksArgs(BaseModel):
    pass


class StartConversationInput(BaseModel):
    session_name: str | None = None
    metadata: dict = Field(default_factory=dict)


class SaveSegmentInput(BaseModel):
    conversation_id: str
    segment_index: int
    topic: str | None = None
    summary: str | None = None
    raw_user_text: str | None = None
    raw_assistant_text: str | None = None
    metadata: dict = Field(default_factory=dict)


class StoreFactInput(BaseModel):
    conversation_id: str
    segment_id: str
    fact_text: str
    fact_type: str | None = None
    confidence: float | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    metadata: dict = Field(default_factory=dict)


class StoreMemoryInput(BaseModel):
    conversation_id: str | None = None
    segment_id: str | None = None
    fact_id: str | None = None
    memory_kind: str
    content: str
    status: str = "active"
    metadata: dict = Field(default_factory=dict)


class ContextPackInput(BaseModel):
    conversation_id: str


class ConversationRecord(BaseModel):
    id: str
    session_name: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    metadata: dict = Field(default_factory=dict)


class ConversationsResponse(BaseModel):
    count: int
    conversations: list[ConversationRecord]


class SegmentRecord(BaseModel):
    id: str
    conversation_id: str
    segment_index: int
    topic: str | None = None
    summary: str | None = None
    raw_user_text: str | None = None
    raw_assistant_text: str | None = None
    metadata: dict = Field(default_factory=dict)


class SegmentsResponse(BaseModel):
    count: int
    conversation_id: str
    segments: list[SegmentRecord]


class FactRecord(BaseModel):
    id: str
    conversation_id: str
    segment_id: str
    fact_text: str
    fact_type: str | None = None
    confidence: float | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    metadata: dict = Field(default_factory=dict)


class FactsResponse(BaseModel):
    count: int
    conversation_id: str | None = None
    segment_id: str | None = None
    facts: list[FactRecord]


class MemoryRecord(BaseModel):
    id: str
    conversation_id: str | None = None
    segment_id: str | None = None
    fact_id: str | None = None
    memory_kind: str
    content: str
    status: str
    metadata: dict = Field(default_factory=dict)


class MemoriesResponse(BaseModel):
    count: int
    conversation_id: str
    memories: list[MemoryRecord]


class ContextPackResponse(BaseModel):
    conversation: ConversationRecord
    segments: list[SegmentRecord]
    facts: list[FactRecord]
    memories: list[MemoryRecord]
    counts: dict


TOOLS = [
    {
        "type": "function",
        "name": "add_task",
        "description": "Add a new task to the tasks table.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short task title"
                }
            },
            "required": ["title"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "list_tasks",
        "description": "List all tasks from the tasks table.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
]


def run_add_task(args: dict):
    validated = AddTaskArgs(**args)
    clean_title = validated.title.strip()

    result = supabase.table("tasks").insert({
        "title": clean_title,
        "status": "open"
    }).execute()

    return {
        "success": True,
        "message": f'Task added: "{clean_title}"',
        "data": result.data
    }


def run_list_tasks(args: dict):
    ListTasksArgs(**args)

    result = supabase.table("tasks").select("*").order("id").execute()
    tasks = result.data or []

    return {
        "success": True,
        "count": len(tasks),
        "data": tasks
    }


TOOL_HANDLERS = {
    "add_task": run_add_task,
    "list_tasks": run_list_tasks,
}


def execute_tool(tool_name: str, raw_args: str):
    if tool_name not in TOOL_HANDLERS:
        raise ValueError(f"Unknown tool: {tool_name}")

    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON arguments for tool: {tool_name}")

    logger.info("Tool called: %s | args=%s", tool_name, args)

    result = TOOL_HANDLERS[tool_name](args)

    logger.info("Tool result: %s", result)

    return result


def fetch_conversation_or_404(conversation_id: str):
    result = (
        supabase.table("conversations")
        .select("id, session_name, started_at, ended_at, metadata")
        .eq("id", conversation_id)
        .execute()
    )

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return rows[0]


def normalize_message_for_routing(message: str) -> str:
    return " ".join(message.strip().lower().split())


def contains_any_phrase(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def starts_with_any_phrase(text: str, prefixes: list[str]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def is_task_query(text: str) -> bool:
    task_create_phrases = [
        "add a task",
        "add task",
        "create a task",
        "create task",
        "new task",
        "add this to my tasks",
        "add that to my tasks",
        "add this to my task list",
        "add that to my task list",
        "put this on my task list",
        "put that on my task list",
        "put this on my list",
        "put that on my list",
        "put this on my tasks",
        "put that on my tasks",
        "can you add a task",
        "can you create a task",
    ]

    task_list_phrases = [
        "list tasks",
        "show tasks",
        "show my tasks",
        "list my tasks",
        "my tasks",
        "what are my tasks",
        "what tasks do i have",
        "do i have any tasks",
        "do i have tasks",
        "what is on my task list",
        "what's on my task list",
        "show me my task list",
        "show me my tasks",
        "what is on my list",
        "what's on my list",
        "what do i have on my list",
        "what do i have on my task list",
    ]

    return contains_any_phrase(text, task_create_phrases + task_list_phrases)


def is_memory_query(text: str) -> bool:
    explicit_memory_phrases = [
        "what have i said",
        "what did i say",
        "what did we say",
        "what did i mention",
        "what have we discussed",
        "what did we discuss",
        "what have i said about",
        "what did i say about",
        "what did i mention about",
        "what have we discussed about",
        "what did we discuss about",
        "what do i know about",
        "what do you know about",
        "what do you remember about",
        "what have you stored about",
        "what have you got about",
        "tell me what you know about",
        "tell me what you remember about",
        "tell me what i said about",
        "tell me what i mentioned about",
        "remind me about",
        "remind me what i said about",
        "remind me what i mentioned about",
        "have i mentioned",
        "did i mention",
        "have we discussed",
        "did we discuss",
        "remember",
        "mentioned",
        "earlier",
        "before",
    ]

    followup_memory_starts = [
        "what about ",
        "and what about ",
    ]

    if contains_any_phrase(text, explicit_memory_phrases):
        return True

    if starts_with_any_phrase(text, followup_memory_starts):
        return True

    return False


def route_message_decision(message: str, conversation_id: str | None = None) -> dict:
    text = normalize_message_for_routing(message)

    local_memory_phrases = [
        "in this conversation",
        "in this chat",
        "earlier in this chat",
        "earlier in this conversation",
        "in our current conversation",
        "from this conversation",
        "from this chat",
        "in this thread",
        "from this thread",
    ]

    if is_task_query(text):
        logger.info("Routing decision: task | message=%s", text)
        return {
            "route": "task",
            "reason": "Matched task phrase"
        }

    if contains_any_phrase(text, local_memory_phrases):
        if conversation_id:
            logger.info("Routing decision: conversation_memory | message=%s", text)
            return {
                "route": "conversation_memory",
                "reason": "Matched current-conversation memory phrase and conversation_id was provided"
            }
        logger.info("Routing decision: global_memory | message=%s", text)
        return {
            "route": "global_memory",
            "reason": "Matched current-conversation memory phrase but no conversation_id was provided"
        }

    if is_memory_query(text):
        if conversation_id:
            logger.info("Routing decision: conversation_memory | message=%s", text)
            return {
                "route": "conversation_memory",
                "reason": "Matched memory phrase and conversation_id was provided"
            }
        logger.info("Routing decision: global_memory | message=%s", text)
        return {
            "route": "global_memory",
            "reason": "Matched memory phrase with no conversation_id provided"
        }

    logger.info("Routing decision: normal_chat | message=%s", text)
    return {
        "route": "normal_chat",
        "reason": "No task or memory phrase matched"
    }


def run_task_agent_flow(user_message: str) -> dict:
    first_response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a tiny task assistant prototype. "
                    "You only support two features: add_task and list_tasks. "
                    "Only use the provided tools. "
                    "Do not invent tools or features. "
                    "Do not mention due dates, priorities, descriptions, reminders, projects, tags, deleting, editing, or assigning tasks. "
                    "If the user clearly wants to add a task and provides a title, call add_task. "
                    "If the user wants to see tasks, call list_tasks. "
                    "If the user says something like 'add a task' but does not provide a title, do not call a tool. Reply exactly: Please include a task title. "
                    "If the user asks for something outside these two features, reply exactly: This prototype only supports adding and listing tasks. "
                    "Keep replies short. "
                    "Use plain ASCII only."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        tools=TOOLS
    )

    tool_outputs = []
    tool_used = False

    for item in first_response.output:
        if item.type == "function_call":
            tool_used = True
            result = execute_tool(item.name, item.arguments)

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result)
            })

    if tool_used:
        final_response = client.responses.create(
            model="gpt-5",
            previous_response_id=first_response.id,
            input=tool_outputs
        )
        reply = final_response.output_text.strip()
    else:
        reply = first_response.output_text.strip()

    return {"reply": reply}


def run_conversation_memory_flow(user_message: str, conversation_id: str) -> dict:
    full_context_pack = build_context_pack(conversation_id, supabase)
    selected_context_pack = select_relevant_context(full_context_pack, user_message)
    context_prompt = format_context_for_prompt(selected_context_pack)

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a grounded assistant prototype. "
                    "Use the supplied context pack when it is relevant to the user's question. "
                    "If the answer is supported by the context pack, answer clearly using that context. "
                    "If the context pack does not contain the answer, say that the stored context does not contain that information. "
                    "Do not invent memories, facts, or prior conversation details that are not present in the provided context. "
                    "Keep replies short and clear. "
                    "Use plain ASCII only."
                )
            },
            {
                "role": "system",
                "content": context_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    reply = response.output_text.strip()

    return {
        "reply": reply,
        "context_used": context_prompt
    }


def run_global_memory_flow(user_message: str) -> dict:
    global_search_result = search_global_memory(user_message, supabase)
    context_prompt = format_global_context_for_prompt(global_search_result)

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a grounded assistant prototype. "
                    "Use the supplied global memory context when it is relevant to the user's question. "
                    "If the answer is supported by the global memory context, answer clearly using that context. "
                    "If the global memory context does not contain the answer, say that the stored memory does not contain that information. "
                    "Do not invent memories, facts, or prior conversation details that are not present in the provided context. "
                    "Keep replies short and clear. "
                    "Use plain ASCII only."
                )
            },
            {
                "role": "system",
                "content": context_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    reply = response.output_text.strip()

    return {
        "reply": reply,
        "context_used": context_prompt
    }


def run_normal_chat_flow(user_message: str) -> dict:
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a simple assistant prototype. "
                    "Reply naturally and briefly. "
                    "Do not invent stored memory. "
                    "Use plain ASCII only."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    reply = response.output_text.strip()

    return {"reply": reply}


@app.get("/")
def root():
    return {"message": "Task loop API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    result = supabase.table("tasks").select("*").order("id").execute()
    return result.data


@app.post("/tasks")
def add_task(title: str):
    clean_title = title.strip()

    result = supabase.table("tasks").insert({
        "title": clean_title,
        "status": "open"
    }).execute()

    return result.data


@app.get("/conversations", response_model=ConversationsResponse)
def get_conversations():
    try:
        result = (
            supabase.table("conversations")
            .select("id, session_name, started_at, ended_at, metadata")
            .order("started_at")
            .execute()
        )

        conversations = result.data or []

        return {
            "count": len(conversations),
            "conversations": conversations
        }

    except Exception as e:
        logger.exception("Failed to fetch conversations")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments", response_model=SegmentsResponse)
def get_segments(conversation_id: str = Query(..., min_length=1)):
    try:
        fetch_conversation_or_404(conversation_id)

        result = (
            supabase.table("conversation_segments")
            .select(
                "id, conversation_id, segment_index, topic, summary, raw_user_text, raw_assistant_text, metadata"
            )
            .eq("conversation_id", conversation_id)
            .order("segment_index")
            .execute()
        )

        segments = result.data or []

        return {
            "count": len(segments),
            "conversation_id": conversation_id,
            "segments": segments
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch segments")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/facts", response_model=FactsResponse)
def get_facts(
    conversation_id: str | None = Query(None),
    segment_id: str | None = Query(None)
):
    try:
        if not conversation_id and not segment_id:
            raise HTTPException(
                status_code=400,
                detail="Provide either conversation_id or segment_id."
            )

        if conversation_id and segment_id:
            raise HTTPException(
                status_code=400,
                detail="Provide only one of conversation_id or segment_id."
            )

        query = (
            supabase.table("memory_facts")
            .select(
                "id, conversation_id, segment_id, fact_text, fact_type, confidence, subject, predicate, object, metadata"
            )
        )

        if conversation_id:
            fetch_conversation_or_404(conversation_id)
            query = query.eq("conversation_id", conversation_id)

        if segment_id:
            query = query.eq("segment_id", segment_id)

        result = query.order("id").execute()
        facts = result.data or []

        return {
            "count": len(facts),
            "conversation_id": conversation_id,
            "segment_id": segment_id,
            "facts": facts
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch facts")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", response_model=MemoriesResponse)
def get_memories(conversation_id: str = Query(..., min_length=1)):
    try:
        fetch_conversation_or_404(conversation_id)

        result = (
            supabase.table("memories")
            .select(
                "id, conversation_id, segment_id, fact_id, memory_kind, content, status, metadata"
            )
            .eq("conversation_id", conversation_id)
            .order("id")
            .execute()
        )

        memories = result.data or []

        return {
            "count": len(memories),
            "conversation_id": conversation_id,
            "memories": memories
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch memories")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context_pack", response_model=ContextPackResponse)
def context_pack(input_data: ContextPackInput):
    try:
        return build_context_pack(input_data.conversation_id, supabase)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build context pack")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context_prompt")
def context_prompt(input_data: AgentWithContextInput):
    try:
        full_context_pack = build_context_pack(input_data.conversation_id, supabase)
        selected_context_pack = select_relevant_context(full_context_pack, input_data.message)
        prompt_text = format_context_for_prompt(selected_context_pack)

        return {
            "conversation_id": input_data.conversation_id,
            "message": input_data.message,
            "context_prompt": prompt_text
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to format selected context prompt")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/global_context_prompt")
def global_context_prompt(input_data: MessageInput):
    try:
        user_message = input_data.message.strip()
        logger.info("Global context prompt: %s", user_message)

        global_search_result = search_global_memory(user_message, supabase)
        prompt_text = format_global_context_for_prompt(global_search_result)

        return {
            "message": user_message,
            "context_prompt": prompt_text
        }

    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }
    except Exception as e:
        logger.exception("Failed to format global context prompt")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_memory")
def search_memory(input_data: MessageInput):
    try:
        user_message = input_data.message.strip()
        logger.info("Global memory search: %s", user_message)

        result = search_global_memory(user_message, supabase)

        return result

    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }
    except Exception as e:
        logger.exception("Failed global memory search")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route_message")
def route_message(input_data: RouteDecisionInput):
    try:
        decision = route_message_decision(
            message=input_data.message,
            conversation_id=input_data.conversation_id
        )

        return {
            "message": input_data.message,
            "conversation_id": input_data.conversation_id,
            "route": decision["route"],
            "reason": decision["reason"]
        }

    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }
    except Exception as e:
        logger.exception("Failed to route message")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_conversation")
def start_conversation(input_data: StartConversationInput):
    try:
        result = supabase.table("conversations").insert({
            "session_name": input_data.session_name,
            "metadata": input_data.metadata
        }).execute()

        return {
            "message": "Conversation created",
            "conversation": result.data[0]
        }

    except Exception as e:
        logger.exception("Failed to create conversation")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_segment")
def save_segment(input_data: SaveSegmentInput):
    try:
        result = supabase.table("conversation_segments").insert({
            "conversation_id": input_data.conversation_id,
            "segment_index": input_data.segment_index,
            "topic": input_data.topic,
            "summary": input_data.summary,
            "raw_user_text": input_data.raw_user_text,
            "raw_assistant_text": input_data.raw_assistant_text,
            "metadata": input_data.metadata
        }).execute()

        return {
            "message": "Segment stored",
            "segment": result.data[0]
        }

    except Exception as e:
        logger.exception("Failed to save segment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store_fact")
def store_fact(input_data: StoreFactInput):
    try:
        segment_result = (
            supabase.table("conversation_segments")
            .select("id, conversation_id")
            .eq("id", input_data.segment_id)
            .execute()
        )

        segment_rows = segment_result.data or []
        if not segment_rows:
            raise HTTPException(status_code=404, detail="Segment not found.")

        segment = segment_rows[0]

        if segment["conversation_id"] != input_data.conversation_id:
            raise HTTPException(
                status_code=400,
                detail="Segment does not belong to the provided conversation."
            )

        result = supabase.table("memory_facts").insert({
            "conversation_id": input_data.conversation_id,
            "segment_id": input_data.segment_id,
            "fact_text": input_data.fact_text,
            "fact_type": input_data.fact_type,
            "confidence": input_data.confidence,
            "subject": input_data.subject,
            "predicate": input_data.predicate,
            "object": input_data.object,
            "metadata": input_data.metadata
        }).execute()

        return {
            "message": "Fact stored",
            "fact": result.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to store fact")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store_memory")
def store_memory(input_data: StoreMemoryInput):
    try:
        result = supabase.table("memories").insert({
            "conversation_id": input_data.conversation_id,
            "segment_id": input_data.segment_id,
            "fact_id": input_data.fact_id,
            "memory_kind": input_data.memory_kind,
            "content": input_data.content,
            "status": input_data.status,
            "metadata": input_data.metadata
        }).execute()

        return {
            "message": "Memory stored",
            "memory": result.data[0]
        }

    except Exception as e:
        logger.exception("Failed to store memory")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent")
def agent(input_data: MessageInput):
    try:
        user_message = input_data.message.strip()
        logger.info("User message: %s", user_message)

        result = run_task_agent_flow(user_message)
        logger.info("Final reply: %s", result["reply"])
        return result

    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }

    except Exception as e:
        logger.exception("Server error")
        return {"error": str(e)}


@app.post("/agent_with_context")
def agent_with_context(input_data: AgentWithContextInput):
    try:
        user_message = input_data.message.strip()
        logger.info(
            "User message with context: conversation_id=%s | message=%s",
            input_data.conversation_id,
            user_message
        )

        result = run_conversation_memory_flow(user_message, input_data.conversation_id)
        logger.info("Grounded reply: %s", result["reply"])
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }
    except Exception as e:
        logger.exception("Server error in /agent_with_context")
        return {"error": str(e)}


@app.post("/agent_global_context")
def agent_global_context(input_data: MessageInput):
    try:
        user_message = input_data.message.strip()
        logger.info("User message with global context: %s", user_message)

        result = run_global_memory_flow(user_message)
        logger.info("Grounded global reply: %s", result["reply"])
        return result

    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }
    except Exception as e:
        logger.exception("Server error in /agent_global_context")
        return {"error": str(e)}


@app.post("/companion")
def companion(input_data: CompanionInput):
    try:
        user_message = input_data.message.strip()
        conversation_id = input_data.conversation_id

        decision = route_message_decision(
            message=user_message,
            conversation_id=conversation_id
        )

        route = decision["route"]
        reason = decision["reason"]

        if route == "task":
            result = run_task_agent_flow(user_message)
            return {
                "route_used": route,
                "reason": reason,
                "reply": result["reply"]
            }

        if route == "conversation_memory":
            if not conversation_id:
                return {
                    "route_used": route,
                    "reason": "Conversation memory route selected but no conversation_id was provided",
                    "reply": "A conversation_id is required for current-conversation memory."
                }

            result = run_conversation_memory_flow(user_message, conversation_id)
            return {
                "route_used": route,
                "reason": reason,
                "reply": result["reply"],
                "context_used": result["context_used"]
            }

        if route == "global_memory":
            result = run_global_memory_flow(user_message)
            return {
                "route_used": route,
                "reason": reason,
                "reply": result["reply"],
                "context_used": result["context_used"]
            }

        result = run_normal_chat_flow(user_message)
        return {
            "route_used": route,
            "reason": reason,
            "reply": result["reply"]
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        logger.exception("Validation error")
        return {
            "error": "Validation failed",
            "details": e.errors()
        }
    except Exception as e:
        logger.exception("Server error in /companion")
        return {"error": str(e)}