import os
import json
import logging
import re
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


FACT_TYPE_VALUES = {
    "plan",
    "possibility",
    "preference",
    "decision",
    "status",
    "relationship",
    "fact",
}

UNCERTAINTY_WORDS = {
    "might",
    "maybe",
    "may",
    "possibly",
    "perhaps",
    "could",
    "unclear",
    "unsure",
    "not sure",
    "probably",
    "likely",
}

STRONG_COMMITMENT_WORDS = {
    "will",
    "definitely",
    "decided",
    "i decided",
    "have decided",
    "committed",
    "confirmed",
    "i am going to",
    "i'm going to",
}

MODERATE_COMMITMENT_WORDS = {
    "plan",
    "planning",
    "intend",
    "intending",
    "going to",
}

PREFERENCE_WORDS = {
    "prefer",
    "like",
    "love",
    "hate",
    "dislike",
    "want",
    "would rather",
}

STATUS_WORDS = {
    "am",
    "i am",
    "i'm",
    "currently",
    "working on",
    "live",
    "living",
    "have",
    "has",
    "is",
    "are",
}

RELATIONSHIP_WORDS = {
    "friend",
    "sister",
    "brother",
    "mum",
    "mom",
    "mother",
    "dad",
    "father",
    "partner",
    "husband",
    "wife",
    "colleague",
    "teammate",
    "manager",
    "client",
}

PROJECT_KEYWORDS = {
    "project",
    "backend",
    "frontend",
    "app",
    "build",
    "system",
    "architecture",
    "memory",
    "routing",
    "task",
    "tasks",
    "companion",
    "gpt",
    "openai",
    "supabase",
    "render",
    "api",
    "whisper",
    "phase",
    "model",
    "model-agnostic",
}

FUTURE_PHRASES = {
    "next week",
    "next weekend",
    "next month",
    "next year",
    "tomorrow",
    "this weekend",
    "later",
    "soon",
    "going to",
    "will",
    "might",
    "may",
    "plan to",
    "plans to",
    "planning to",
}

DECISION_WORDS = {
    "decided",
    "decision",
    "choose",
    "chose",
    "chosen",
    "going with",
    "settled on",
    "committed to",
}

NON_PERSON_ENTITIES = {
    "user",
    "the user",
    "friend",
    "friends",
    "frontend",
    "backend",
    "london",
    "render",
    "gpt",
    "openai",
    "supabase",
    "api",
    "task",
    "tasks",
    "voice interaction",
    "typing",
}

PRONOUN_LIKE_NAMES = {
    "I",
    "I'm",
    "Ive",
    "I've",
    "It",
    "This",
    "That",
    "We",
    "You",
    "He",
    "She",
    "They",
    "The",
}


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


class RouteClassificationOutput(BaseModel):
    route: str
    reason: str


class FactCandidate(BaseModel):
    fact_text: str = Field(..., min_length=1)
    fact_type: str | None = None
    confidence: float | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None


class MemoryCaptureOutput(BaseModel):
    topic: str | None = None
    summary: str | None = None
    facts: list[FactCandidate] = Field(default_factory=list)


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


ALLOWED_ROUTES = {
    "task",
    "conversation_memory",
    "global_memory",
    "normal_chat",
}


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


def create_conversation_record(session_name: str | None = None, metadata: dict | None = None) -> dict:
    result = supabase.table("conversations").insert({
        "session_name": session_name,
        "metadata": metadata or {}
    }).execute()

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Failed to create conversation.")

    return rows[0]


def resolve_or_create_conversation(conversation_id: str | None) -> tuple[str, bool]:
    if conversation_id:
        fetch_conversation_or_404(conversation_id)
        return conversation_id, False

    conversation = create_conversation_record(
        session_name=None,
        metadata={"source": "companion_auto_create"}
    )
    return conversation["id"], True


def normalize_message_for_routing(message: str) -> str:
    return " ".join(message.strip().lower().split())


def contains_any_phrase(text: str, phrases: list[str] | set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def starts_with_any_phrase(text: str, prefixes: list[str]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def normalize_fact_type(raw_fact_type: str | None, fact_text: str) -> str:
    fact_type = (raw_fact_type or "").strip().lower()
    text = normalize_message_for_routing(fact_text)

    if fact_type in FACT_TYPE_VALUES:
        return fact_type

    if contains_any_phrase(text, PREFERENCE_WORDS):
        return "preference"

    if contains_any_phrase(text, DECISION_WORDS):
        return "decision"

    if contains_any_phrase(text, UNCERTAINTY_WORDS):
        return "possibility"

    if contains_any_phrase(text, MODERATE_COMMITMENT_WORDS | STRONG_COMMITMENT_WORDS | FUTURE_PHRASES):
        return "plan"

    if contains_any_phrase(text, RELATIONSHIP_WORDS):
        return "relationship"

    if contains_any_phrase(text, STATUS_WORDS):
        return "status"

    return "fact"


def score_confidence_for_fact(fact_text: str, fact_type: str, extracted_confidence: float | None) -> float:
    text = normalize_message_for_routing(fact_text)

    if contains_any_phrase(text, UNCERTAINTY_WORDS):
        base = 0.5
    elif contains_any_phrase(text, STRONG_COMMITMENT_WORDS | DECISION_WORDS):
        base = 0.9
    elif contains_any_phrase(text, MODERATE_COMMITMENT_WORDS):
        base = 0.75
    elif fact_type in {"preference", "relationship", "status"}:
        base = 0.85
    else:
        base = 0.7

    if extracted_confidence is None:
        return round(base, 2)

    bounded_extracted = max(0.0, min(1.0, extracted_confidence))
    final_score = (base * 0.7) + (bounded_extracted * 0.3)
    return round(max(0.0, min(1.0, final_score)), 2)


def is_likely_named_person(candidate: str | None) -> bool:
    if not candidate:
        return False

    cleaned = candidate.strip()
    cleaned_lower = cleaned.lower()

    if not cleaned or cleaned_lower in NON_PERSON_ENTITIES:
        return False

    if cleaned in PRONOUN_LIKE_NAMES:
        return False

    if re.search(r"\b(next|weekend|week|month|year|tomorrow|later|soon)\b", cleaned_lower):
        return False

    if cleaned_lower in PROJECT_KEYWORDS:
        return False

    return bool(re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", cleaned))


def detect_named_person_signal(fact_text: str, subject: str | None, object_value: str | None) -> bool:
    if is_likely_named_person(subject):
        return True

    if is_likely_named_person(object_value):
        return True

    return False


def detect_future_signal(fact_text: str) -> bool:
    text = normalize_message_for_routing(fact_text)
    return contains_any_phrase(text, FUTURE_PHRASES)


def detect_project_signal(fact_text: str, predicate: str | None, object_value: str | None) -> bool:
    combined = " ".join(
        [
            normalize_message_for_routing(fact_text),
            normalize_message_for_routing(predicate or ""),
            normalize_message_for_routing(object_value or ""),
        ]
    ).strip()

    return contains_any_phrase(combined, PROJECT_KEYWORDS)


def score_importance_for_fact(
    fact_text: str,
    fact_type: str,
    subject: str | None,
    predicate: str | None,
    object_value: str | None
) -> tuple[float, list[str]]:
    score = 0.5
    signals = []

    if detect_named_person_signal(fact_text, subject, object_value):
        score += 0.2
        signals.append("person")

    if detect_future_signal(fact_text) or fact_type in {"plan", "possibility", "decision"}:
        score += 0.2
        signals.append("future")

    if detect_project_signal(fact_text, predicate, object_value):
        score += 0.1
        signals.append("project")

    if fact_type in {"preference", "decision", "relationship"}:
        score += 0.1
        signals.append(f"type:{fact_type}")

    return round(min(score, 1.0), 2), signals


def enrich_fact_candidate(fact: FactCandidate) -> tuple[FactCandidate, dict]:
    clean_fact_text = fact.fact_text.strip()
    normalized_fact_type = normalize_fact_type(fact.fact_type, clean_fact_text)
    normalized_confidence = score_confidence_for_fact(
        fact_text=clean_fact_text,
        fact_type=normalized_fact_type,
        extracted_confidence=fact.confidence
    )
    importance, signals = score_importance_for_fact(
        fact_text=clean_fact_text,
        fact_type=normalized_fact_type,
        subject=fact.subject,
        predicate=fact.predicate,
        object_value=fact.object
    )

    enriched_fact = FactCandidate(
        fact_text=clean_fact_text,
        fact_type=normalized_fact_type,
        confidence=normalized_confidence,
        subject=(fact.subject.strip() if fact.subject else None),
        predicate=(fact.predicate.strip() if fact.predicate else None),
        object=(fact.object.strip() if fact.object else None),
    )

    metadata = {
        "importance": importance,
        "signals": signals,
    }

    return enriched_fact, metadata


def normalize_fact_text_for_dedupe(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def dedupe_fact_candidates(facts: list[FactCandidate]) -> list[FactCandidate]:
    deduped = []
    seen = set()

    for fact in facts:
        key = normalize_fact_text_for_dedupe(fact.fact_text)
        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(fact)

    return deduped


def is_high_confidence_task_query(text: str) -> bool:
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


def get_high_confidence_memory_route(
    text: str,
    conversation_id: str | None = None
) -> dict | None:
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

    explicit_memory_phrases = [
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
    ]

    followup_memory_starts = [
        "what about ",
        "and what about ",
    ]

    if contains_any_phrase(text, local_memory_phrases):
        if conversation_id:
            return {
                "route": "conversation_memory",
                "reason": "Matched explicit current-conversation memory phrase"
            }
        return {
            "route": "global_memory",
            "reason": "Matched explicit current-conversation memory phrase but no conversation_id was provided"
        }

    if contains_any_phrase(text, explicit_memory_phrases):
        if conversation_id:
            return {
                "route": "conversation_memory",
                "reason": "Matched explicit memory lookup phrase and conversation_id was provided"
            }
        return {
            "route": "global_memory",
            "reason": "Matched explicit memory lookup phrase with no conversation_id provided"
        }

    if starts_with_any_phrase(text, followup_memory_starts):
        if conversation_id:
            return {
                "route": "conversation_memory",
                "reason": "Matched follow-up memory lookup phrase and conversation_id was provided"
            }
        return {
            "route": "global_memory",
            "reason": "Matched follow-up memory lookup phrase with no conversation_id provided"
        }

    return None


def classify_intent_route(message: str, conversation_id: str | None = None) -> dict:
    system_prompt = (
        "You are a strict backend route classifier for an AI companion system. "
        "Classify the user's message into exactly one of these routes: "
        "task, conversation_memory, global_memory, normal_chat. "
        "Return JSON only with keys route and reason. "
        "Do not include markdown, extra text, or code fences. "
        "Use these rules: "
        "task = the user is asking to add, create, or list tasks. "
        "conversation_memory = the user is asking about something said, stored, or discussed in the current conversation and a conversation_id is available. "
        "global_memory = the user is asking what is known, remembered, or stored across memory more generally, or conversation memory was requested but no conversation_id is available. "
        "normal_chat = everything else. "
        "Be conservative. "
        "If the message is ordinary conversation and not clearly a task or memory lookup, choose normal_chat."
    )

    user_prompt = json.dumps(
        {
            "message": message,
            "conversation_id_provided": bool(conversation_id),
            "allowed_routes": sorted(ALLOWED_ROUTES),
        }
    )

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    raw_output = response.output_text.strip()
    logger.info("Classifier raw output: %s", raw_output)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Classifier returned invalid JSON: {raw_output}") from e

    validated = RouteClassificationOutput(**parsed)

    if validated.route not in ALLOWED_ROUTES:
        raise ValueError(f"Classifier returned invalid route: {validated.route}")

    return {
        "route": validated.route,
        "reason": validated.reason
    }


def route_message_decision(message: str, conversation_id: str | None = None) -> dict:
    text = normalize_message_for_routing(message)

    if is_high_confidence_task_query(text):
        logger.info("Routing decision: task | source=rule | message=%s", text)
        return {
            "route": "task",
            "reason": "Matched high-confidence task phrase",
            "decision_source": "rule"
        }

    memory_rule_result = get_high_confidence_memory_route(
        text=text,
        conversation_id=conversation_id
    )
    if memory_rule_result:
        logger.info(
            "Routing decision: %s | source=rule | message=%s",
            memory_rule_result["route"],
            text
        )
        return {
            "route": memory_rule_result["route"],
            "reason": memory_rule_result["reason"],
            "decision_source": "rule"
        }

    classifier_result = classify_intent_route(
        message=message.strip(),
        conversation_id=conversation_id
    )
    logger.info(
        "Routing decision: %s | source=classifier | message=%s",
        classifier_result["route"],
        text
    )
    return {
        "route": classifier_result["route"],
        "reason": classifier_result["reason"],
        "decision_source": "classifier"
    }


def get_next_segment_index(conversation_id: str) -> int:
    result = (
        supabase.table("conversation_segments")
        .select("segment_index")
        .eq("conversation_id", conversation_id)
        .order("segment_index", desc=True)
        .limit(1)
        .execute()
    )

    rows = result.data or []
    if not rows:
        return 1

    return int(rows[0]["segment_index"]) + 1


def extract_memory_capture(
    user_message: str,
    assistant_reply: str,
    route: str
) -> MemoryCaptureOutput:
    system_prompt = (
        "You are a strict structured-memory extraction helper for an AI companion backend. "
        "Your job is to extract conservative but useful memory candidates from a single user-assistant exchange. "
        "Return JSON only with keys: topic, summary, facts. "
        "facts must be a list of objects with keys: fact_text, fact_type, confidence, subject, predicate, object. "
        "Use fact_type conservatively from this preferred set when possible: plan, possibility, preference, decision, status, relationship, fact. "
        "Do not include markdown or any extra text. "
        "Only extract facts that are grounded in the user's message, not invented by the assistant. "
        "If the user's message contains multiple distinct meaningful facts, extract multiple separate facts rather than collapsing them into one. "
        "Prefer durable or useful information such as preferences, plans, possibilities, decisions, relationships, project details, named people and their roles, collaborator availability, responsibilities, tool or app intentions, app choices or comparisons, switching from one tool to another, routine or habit intentions, recurring scheduling intentions, recurring planning blocks, and notable user-provided facts. "
        "If the user mentions wanting to start, schedule, block out, or regularly do something, treat that as a meaningful plan or routine candidate when it is grounded and specific enough. "
        "If the user mentions considering, comparing, or switching apps or tools, treat that as a meaningful candidate even if they have not made a final decision yet. "
        "Do not store generic chit-chat, assistant advice, filler, or very temporary emotions. "
        "If the route is task, do not store the task itself unless the user message also contains a meaningful non-task fact worth remembering. "
        "You may return zero facts. "
        "Keep topic and summary short and factual. "
        "Use plain ASCII only."
    )

    user_prompt = json.dumps(
        {
            "route": route,
            "user_message": user_message,
            "assistant_reply": assistant_reply,
        }
    )

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    raw_output = response.output_text.strip()
    logger.info("Memory capture raw output: %s", raw_output)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Memory extractor returned invalid JSON: {raw_output}") from e

    validated = MemoryCaptureOutput(**parsed)

    trimmed_facts = []
    for fact in validated.facts[:6]:
        clean_fact_text = fact.fact_text.strip()
        if not clean_fact_text:
            continue

        preliminary_fact = FactCandidate(
            fact_text=clean_fact_text,
            fact_type=(fact.fact_type.strip() if fact.fact_type else None),
            confidence=fact.confidence,
            subject=(fact.subject.strip() if fact.subject else None),
            predicate=(fact.predicate.strip() if fact.predicate else None),
            object=(fact.object.strip() if fact.object else None),
        )

        enriched_fact, _ = enrich_fact_candidate(preliminary_fact)
        trimmed_facts.append(enriched_fact)

    trimmed_facts = dedupe_fact_candidates(trimmed_facts)

    return MemoryCaptureOutput(
        topic=validated.topic.strip() if validated.topic else None,
        summary=validated.summary.strip() if validated.summary else None,
        facts=trimmed_facts,
    )


def save_turn_segment(
    conversation_id: str,
    user_message: str,
    assistant_reply: str,
    route: str,
    decision_source: str
) -> dict:
    segment_index = get_next_segment_index(conversation_id)

    segment_payload = {
        "conversation_id": conversation_id,
        "segment_index": segment_index,
        "topic": None,
        "summary": None,
        "raw_user_text": user_message,
        "raw_assistant_text": assistant_reply,
        "metadata": {
            "source": "companion_turn",
            "route": route,
            "decision_source": decision_source,
            "extraction_status": "pending",
            "extraction_error": None,
        }
    }

    segment_result = supabase.table("conversation_segments").insert(segment_payload).execute()
    rows = segment_result.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Failed to store segment.")

    return rows[0]


def extract_and_store_turn_memory(
    conversation_id: str,
    segment_id: str,
    route: str,
    decision_source: str,
    user_message: str,
    assistant_reply: str
) -> dict:
    extraction = None
    extraction_error = None

    try:
        extraction = extract_memory_capture(
            user_message=user_message,
            assistant_reply=assistant_reply,
            route=route
        )
    except Exception as e:
        extraction_error = str(e)
        logger.exception("Memory extraction failed after segment creation")

    update_payload = {
        "metadata": {
            "source": "companion_turn",
            "route": route,
            "decision_source": decision_source,
            "extraction_status": "ok" if extraction else "failed",
            "extraction_error": extraction_error,
        }
    }

    if extraction:
        update_payload["topic"] = extraction.topic
        update_payload["summary"] = extraction.summary

    (
        supabase.table("conversation_segments")
        .update(update_payload)
        .eq("id", segment_id)
        .execute()
    )

    facts_created = 0
    memories_created = 0

    facts_to_store = extraction.facts if extraction else []

    for fact in facts_to_store:
        enriched_fact, enrichment_metadata = enrich_fact_candidate(fact)

        fact_result = supabase.table("memory_facts").insert({
            "conversation_id": conversation_id,
            "segment_id": segment_id,
            "fact_text": enriched_fact.fact_text,
            "fact_type": enriched_fact.fact_type or "fact",
            "confidence": enriched_fact.confidence,
            "subject": enriched_fact.subject,
            "predicate": enriched_fact.predicate,
            "object": enriched_fact.object,
            "metadata": {
                "source": "companion_turn",
                "route": route,
                "importance": enrichment_metadata["importance"],
                "signals": enrichment_metadata["signals"],
            }
        }).execute()

        fact_rows = fact_result.data or []
        if not fact_rows:
            raise HTTPException(status_code=500, detail="Failed to store fact.")

        fact_row = fact_rows[0]
        facts_created += 1

        memory_result = supabase.table("memories").insert({
            "conversation_id": conversation_id,
            "segment_id": segment_id,
            "fact_id": fact_row["id"],
            "memory_kind": "semantic_scaffold",
            "content": enriched_fact.fact_text,
            "status": "active",
            "metadata": {
                "source": "companion_turn",
                "route": route,
                "importance": enrichment_metadata["importance"],
            }
        }).execute()

        memory_rows = memory_result.data or []
        if not memory_rows:
            raise HTTPException(status_code=500, detail="Failed to store memory.")

        memories_created += 1

    return {
        "status": "stored" if extraction else "partial",
        "reason": "Segment stored and memory extraction completed" if extraction else "Segment stored but memory extraction failed",
        "facts_created": facts_created,
        "memories_created": memories_created,
        "extraction_status": "ok" if extraction else "failed",
        "extraction_error": extraction_error,
        "topic": extraction.topic if extraction else None,
        "summary": extraction.summary if extraction else None,
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
            "reason": decision["reason"],
            "decision_source": decision["decision_source"]
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
        client_conversation_id = input_data.conversation_id

        conversation_id, conversation_created = resolve_or_create_conversation(client_conversation_id)

        decision = route_message_decision(
            message=user_message,
            conversation_id=client_conversation_id
        )

        route = decision["route"]
        reason = decision["reason"]
        decision_source = decision["decision_source"]

        result = None

        if route == "task":
            result = run_task_agent_flow(user_message)

        elif route == "conversation_memory":
            result = run_conversation_memory_flow(user_message, conversation_id)

        elif route == "global_memory":
            result = run_global_memory_flow(user_message)

        else:
            result = run_normal_chat_flow(user_message)

        assistant_reply = result["reply"]

        segment = save_turn_segment(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_reply=assistant_reply,
            route=route,
            decision_source=decision_source
        )

        memory_capture = extract_and_store_turn_memory(
            conversation_id=conversation_id,
            segment_id=segment["id"],
            route=route,
            decision_source=decision_source,
            user_message=user_message,
            assistant_reply=assistant_reply
        )

        response_payload = {
            "conversation_id": conversation_id,
            "conversation_created": conversation_created,
            "route_used": route,
            "reason": reason,
            "decision_source": decision_source,
            "assistant_reply": assistant_reply,
            "reply": assistant_reply,
            "segment_id": segment["id"],
            "segment_index": segment["segment_index"],
            "memory_capture": memory_capture,
        }

        if "context_used" in result:
            response_payload["context_used"] = result["context_used"]

        return response_payload

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