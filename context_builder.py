from typing import Any, Dict


def build_context_pack(conversation_id: str, supabase) -> Dict[str, Any]:
    """
    Retrieve all stored context for a conversation and return it
    as one stable structured object.

    Rules:
    - Raise an error only if the conversation itself is not found
    - Return empty lists safely for missing child records
    - Order segments by segment_index
    - Include counts for debugging / downstream use
    """

    # 1. Fetch conversation
    conversation_response = (
        supabase.table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .limit(1)
        .execute()
    )

    conversation_rows = conversation_response.data or []

    if not conversation_rows:
        raise ValueError(f"Conversation not found: {conversation_id}")

    conversation = conversation_rows[0]

    # 2. Fetch segments for this conversation
    segments_response = (
        supabase.table("conversation_segments")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("segment_index")
        .execute()
    )

    segments = segments_response.data or []

    # 3. Fetch facts for this conversation
    facts_response = (
        supabase.table("memory_facts")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("id")
        .execute()
    )

    facts = facts_response.data or []

    # 4. Fetch memories for this conversation
    memories_response = (
        supabase.table("memories")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("id")
        .execute()
    )

    memories = memories_response.data or []

    # 5. Build stable output shape
    context_pack = {
        "conversation": conversation,
        "segments": segments,
        "facts": facts,
        "memories": memories,
        "counts": {
            "segments": len(segments),
            "facts": len(facts),
            "memories": len(memories),
        },
    }

    return context_pack


def _clip_text(text: str | None, max_chars: int = 220) -> str | None:
    """
    Return a normalized, clipped excerpt.

    Rules:
    - collapse newlines / repeated whitespace
    - trim surrounding whitespace
    - clip to max_chars with trailing '...' if needed
    """
    if not text:
        return None

    normalized = " ".join(text.split()).strip()

    if not normalized:
        return None

    if len(normalized) <= max_chars:
        return normalized

    clipped = normalized[: max_chars - 3].rstrip()
    return f"{clipped}..."


def _extract_keywords(user_message: str) -> set[str]:
    """
    Very simple keyword extraction for deterministic matching.

    Rules:
    - lowercase only
    - remove short tokens
    - ignore a small set of common words
    - no stemming / no fuzzy matching
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "than",
        "to", "of", "in", "on", "at", "for", "from", "with", "about",
        "into", "by", "is", "are", "was", "were", "be", "been", "being",
        "it", "this", "that", "these", "those", "i", "you", "he", "she",
        "they", "we", "me", "my", "your", "our", "their", "what", "which",
        "who", "whom", "when", "where", "why", "how", "did", "do", "does",
        "say", "said", "tell", "told", "mention", "mentioned", "have",
        "has", "had", "will", "would", "could", "should"
    }

    cleaned = []
    current = []

    for char in user_message.lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                cleaned.append("".join(current))
                current = []

    if current:
        cleaned.append("".join(current))

    keywords = {
        token for token in cleaned
        if len(token) >= 3 and token not in stop_words
    }

    return keywords


def _count_keyword_matches(text: str | None, keywords: set[str]) -> int:
    if not text or not keywords:
        return 0

    haystack = text.lower()
    return sum(1 for keyword in keywords if keyword in haystack)


def _dedupe_rows_by_id(rows: list[dict]) -> list[dict]:
    deduped = []
    seen_ids = set()

    for row in rows:
        row_id = row.get("id")
        if row_id and row_id in seen_ids:
            continue
        if row_id:
            seen_ids.add(row_id)
        deduped.append(row)

    return deduped


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_unit_score(value: Any) -> float:
    """
    Normalize a score-like value to the range 0.0 -> 1.0 where possible.

    Supported examples:
    - 0.8 -> 0.8
    - 80 -> 0.8
    - "0.8" -> 0.8
    - None / invalid -> 0.0
    """
    numeric = _safe_float(value, default=0.0)

    if numeric <= 0:
        return 0.0
    if numeric <= 1:
        return numeric
    if numeric <= 100:
        return numeric / 100.0
    return 1.0


def _normalize_importance_score(value: Any) -> float:
    """
    Normalize importance into a simple deterministic 0.0 -> 1.0 score.

    Supported examples:
    - low / medium / high
    - 0.0 -> 1.0
    - 0 -> 100
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {
            "low": 0.25,
            "medium": 0.5,
            "med": 0.5,
            "high": 1.0,
        }
        if lowered in mapping:
            return mapping[lowered]

    return _normalize_unit_score(value)


def _extract_signal_values(raw_signals: Any) -> set[str]:
    """
    Normalize stored signals into a lowercase set of simple strings.

    Supported shapes:
    - ["future", "possibility"]
    - "future, possibility"
    - {"future": true, "possibility": true}
    - {"tags": ["future", "possibility"]}
    """
    values = set()

    if raw_signals is None:
        return values

    if isinstance(raw_signals, list):
        for item in raw_signals:
            if item is None:
                continue
            values.add(str(item).strip().lower())

    elif isinstance(raw_signals, str):
        normalized = raw_signals.replace("|", ",")
        for part in normalized.split(","):
            cleaned = part.strip().lower()
            if cleaned:
                values.add(cleaned)

    elif isinstance(raw_signals, dict):
        for key, value in raw_signals.items():
            if isinstance(value, bool):
                if value:
                    values.add(str(key).strip().lower())
            elif isinstance(value, list):
                for item in value:
                    if item is None:
                        continue
                    values.add(str(item).strip().lower())
            elif value is not None:
                values.add(str(value).strip().lower())

    return {value for value in values if value}


def _detect_query_signals(user_message: str, keywords: set[str]) -> set[str]:
    """
    Infer a very small set of deterministic query-side signals.

    This is intentionally simple and bounded.
    """
    text = (user_message or "").lower()
    query_signals = set()

    future_terms = {
        "plan", "plans", "planned", "planning", "next", "later",
        "future", "upcoming", "soon", "tomorrow", "weekend", "month",
        "eventually", "going", "might", "maybe"
    }
    preference_terms = {
        "prefer", "preference", "like", "likes", "want", "wants",
        "favorite", "favourite", "enjoy", "enjoys"
    }
    possibility_terms = {
        "might", "maybe", "possible", "possibly", "perhaps",
        "could", "may", "uncertain"
    }

    if any(term in text for term in future_terms) or (keywords & future_terms):
        query_signals.add("future")

    if any(term in text for term in preference_terms) or (keywords & preference_terms):
        query_signals.add("preference")

    if any(term in text for term in possibility_terms) or (keywords & possibility_terms):
        query_signals.add("possibility")

    return query_signals


def _score_fact(fact: dict, keywords: set[str], query_signals: set[str]) -> dict[str, Any]:
    """
    Deterministic Phase 9 fact scoring.

    Score components:
    - text match in fact_text / subject / object / predicate / fact_type
    - confidence bonus
    - importance bonus
    - small signal bonus
    """
    fact_text = fact.get("fact_text") or ""
    fact_type = fact.get("fact_type") or ""
    subject = fact.get("subject") or ""
    predicate = fact.get("predicate") or ""
    object_text = fact.get("object") or ""

    metadata = fact.get("metadata") or {}
    raw_signals = metadata.get("signals")
    stored_signals = _extract_signal_values(raw_signals)

    fact_text_matches = _count_keyword_matches(fact_text, keywords)
    subject_matches = _count_keyword_matches(subject, keywords)
    object_matches = _count_keyword_matches(object_text, keywords)
    predicate_matches = _count_keyword_matches(predicate, keywords)
    fact_type_matches = _count_keyword_matches(fact_type, keywords)

    text_score = (
        (fact_text_matches * 3.0)
        + ((subject_matches + object_matches) * 2.0)
        + (predicate_matches * 1.0)
        + (fact_type_matches * 1.0)
    )

    confidence_score = _normalize_unit_score(fact.get("confidence")) * 2.0
    importance_score = _normalize_importance_score(fact.get("importance")) * 2.0

    matched_query_signals = query_signals & stored_signals
    signal_score = min(len(matched_query_signals) * 0.5, 1.0)

    total_score = text_score + confidence_score + importance_score + signal_score

    return {
        "fact": fact,
        "score": total_score,
        "text_score": text_score,
        "confidence_score": confidence_score,
        "importance_score": importance_score,
        "signal_score": signal_score,
        "keyword_matches": fact_text_matches + subject_matches + object_matches + predicate_matches + fact_type_matches,
        "matched_query_signals": sorted(matched_query_signals),
    }


def _score_segment(
    segment: dict,
    keywords: set[str],
    linked_fact_scores: list[float]
) -> dict[str, Any]:
    """
    Deterministic Phase 9 segment scoring.

    Score components:
    - direct text match in topic / summary / raw excerpts
    - strongest linked fact score
    - small recency bonus
    """
    topic = segment.get("topic") or ""
    summary = segment.get("summary") or ""
    raw_user_text = segment.get("raw_user_text") or ""
    raw_assistant_text = segment.get("raw_assistant_text") or ""

    topic_matches = _count_keyword_matches(topic, keywords)
    summary_matches = _count_keyword_matches(summary, keywords)
    raw_user_matches = _count_keyword_matches(raw_user_text, keywords)
    raw_assistant_matches = _count_keyword_matches(raw_assistant_text, keywords)

    text_score = (
        (topic_matches * 2.0)
        + (summary_matches * 1.5)
        + ((raw_user_matches + raw_assistant_matches) * 1.0)
    )

    max_linked_fact_score = max(linked_fact_scores) if linked_fact_scores else 0.0
    recency_bonus = _safe_float(segment.get("segment_index"), default=0.0) * 0.01

    total_score = text_score + max_linked_fact_score + recency_bonus

    return {
        "segment": segment,
        "score": total_score,
        "text_score": text_score,
        "max_linked_fact_score": max_linked_fact_score,
        "recency_bonus": recency_bonus,
        "keyword_matches": topic_matches + summary_matches + raw_user_matches + raw_assistant_matches,
    }


def select_relevant_context(
    context_pack: Dict[str, Any],
    user_message: str,
    max_segments: int = 2,
    max_facts: int = 2,
    max_memories: int = 2
) -> Dict[str, Any]:
    """
    Select a smaller, more relevant subset of the full context pack.

    Phase 9 Step 1 rules:
    - keep conversation metadata unchanged
    - score facts first using keyword match + confidence + importance + signals
    - score segments using direct text match + linked fact score + small recency bonus
    - select top segments first, then top facts within those segments
    - select memories linked to selected facts first, then selected segments
    - only fall back to recency / original order when nothing matches meaningfully
    - keep output deterministic
    - return the same stable shape as build_context_pack()
    """

    conversation = context_pack.get("conversation") or {}
    segments = context_pack.get("segments") or []
    facts = context_pack.get("facts") or []
    memories = context_pack.get("memories") or []

    keywords = _extract_keywords(user_message)
    query_signals = _detect_query_signals(user_message, keywords)

    facts_by_segment: dict[Any, list[dict]] = {}
    memories_by_segment: dict[Any, list[dict]] = {}
    memories_by_fact: dict[Any, list[dict]] = {}

    for fact in facts:
        segment_id = fact.get("segment_id")
        facts_by_segment.setdefault(segment_id, []).append(fact)

    for memory in memories:
        segment_id = memory.get("segment_id")
        fact_id = memory.get("fact_id")

        memories_by_segment.setdefault(segment_id, []).append(memory)
        memories_by_fact.setdefault(fact_id, []).append(memory)

    # ---- FACT SCORING ----
    scored_facts = []
    fact_score_by_id: dict[Any, float] = {}

    for fact in facts:
        scored = _score_fact(fact, keywords, query_signals)
        scored_facts.append(scored)

        fact_id = fact.get("id")
        if fact_id:
            fact_score_by_id[fact_id] = scored["score"]

    scored_facts.sort(
        key=lambda item: (
            item["score"],
            item["keyword_matches"],
            item["fact"].get("id", 0)
        ),
        reverse=True
    )

    # ---- SEGMENT SCORING ----
    scored_segments = []

    for segment in segments:
        segment_id = segment.get("id")
        linked_facts = facts_by_segment.get(segment_id, [])
        linked_fact_scores = []

        for fact in linked_facts:
            fact_id = fact.get("id")
            linked_fact_scores.append(fact_score_by_id.get(fact_id, 0.0))

        scored = _score_segment(segment, keywords, linked_fact_scores)
        scored_segments.append(scored)

    scored_segments.sort(
        key=lambda item: (
            item["score"],
            item["keyword_matches"],
            item["segment"].get("segment_index", 0)
        ),
        reverse=True
    )

    selected_segments = []
    selected_segment_ids = set()

    has_meaningful_segment_match = any(item["score"] > 0 for item in scored_segments)

    if has_meaningful_segment_match:
        for item in scored_segments:
            segment = item["segment"]
            segment_id = segment.get("id")

            if segment_id in selected_segment_ids:
                continue

            selected_segments.append(segment)
            if segment_id:
                selected_segment_ids.add(segment_id)

            if len(selected_segments) >= max_segments:
                break
    else:
        segments_recent_first = sorted(
            segments,
            key=lambda s: s.get("segment_index", 0),
            reverse=True
        )

        for segment in segments_recent_first:
            segment_id = segment.get("id")

            if segment_id in selected_segment_ids:
                continue

            selected_segments.append(segment)
            if segment_id:
                selected_segment_ids.add(segment_id)

            if len(selected_segments) >= max_segments:
                break

    selected_segments.sort(key=lambda s: s.get("segment_index", 0))

    # ---- FACT SELECTION ----
    selected_facts = []
    selected_fact_ids = set()

    scored_facts_in_selected_segments = []
    scored_facts_outside_selected_segments = []

    for item in scored_facts:
        fact = item["fact"]
        if fact.get("segment_id") in selected_segment_ids:
            scored_facts_in_selected_segments.append(item)
        else:
            scored_facts_outside_selected_segments.append(item)

    has_meaningful_fact_match = any(item["score"] > 0 for item in scored_facts)

    if has_meaningful_fact_match:
        for item in scored_facts_in_selected_segments:
            fact = item["fact"]
            fact_id = fact.get("id")

            if fact_id in selected_fact_ids:
                continue

            selected_facts.append(fact)
            if fact_id:
                selected_fact_ids.add(fact_id)

            if len(selected_facts) >= max_facts:
                break

        if len(selected_facts) < max_facts:
            for item in scored_facts_outside_selected_segments:
                fact = item["fact"]
                fact_id = fact.get("id")

                if fact_id in selected_fact_ids:
                    continue

                selected_facts.append(fact)
                if fact_id:
                    selected_fact_ids.add(fact_id)

                if len(selected_facts) >= max_facts:
                    break
    else:
        linked_facts = []

        for segment in selected_segments:
            segment_id = segment.get("id")
            linked_facts.extend(facts_by_segment.get(segment_id, []))

        for fact in linked_facts:
            fact_id = fact.get("id")

            if fact_id in selected_fact_ids:
                continue

            selected_facts.append(fact)
            if fact_id:
                selected_fact_ids.add(fact_id)

            if len(selected_facts) >= max_facts:
                break

        if len(selected_facts) < max_facts:
            for fact in facts:
                fact_id = fact.get("id")

                if fact_id in selected_fact_ids:
                    continue

                selected_facts.append(fact)
                if fact_id:
                    selected_fact_ids.add(fact_id)

                if len(selected_facts) >= max_facts:
                    break

    # ---- MEMORY SELECTION ----
    selected_memories = []
    selected_memory_ids = set()

    linked_to_selected_facts = []
    linked_to_selected_segments = []
    unlinked_fallback = []

    for memory in memories:
        memory_id = memory.get("id")
        fact_id = memory.get("fact_id")
        segment_id = memory.get("segment_id")

        if memory_id in selected_memory_ids:
            continue

        if fact_id in selected_fact_ids:
            linked_to_selected_facts.append(memory)
        elif segment_id in selected_segment_ids:
            linked_to_selected_segments.append(memory)
        else:
            unlinked_fallback.append(memory)

    for memory in linked_to_selected_facts:
        memory_id = memory.get("id")

        if memory_id in selected_memory_ids:
            continue

        selected_memories.append(memory)
        if memory_id:
            selected_memory_ids.add(memory_id)

        if len(selected_memories) >= max_memories:
            break

    if len(selected_memories) < max_memories:
        for memory in linked_to_selected_segments:
            memory_id = memory.get("id")

            if memory_id in selected_memory_ids:
                continue

            selected_memories.append(memory)
            if memory_id:
                selected_memory_ids.add(memory_id)

            if len(selected_memories) >= max_memories:
                break

    if len(selected_memories) < max_memories:
        for memory in unlinked_fallback:
            memory_id = memory.get("id")

            if memory_id in selected_memory_ids:
                continue

            selected_memories.append(memory)
            if memory_id:
                selected_memory_ids.add(memory_id)

            if len(selected_memories) >= max_memories:
                break

    selected_context = {
        "conversation": conversation,
        "segments": selected_segments,
        "facts": selected_facts,
        "memories": selected_memories,
        "counts": {
            "segments": len(selected_segments),
            "facts": len(selected_facts),
            "memories": len(selected_memories),
        },
    }

    return selected_context


def search_global_memory(
    user_message: str,
    supabase,
    max_conversations: int = 3,
    max_segments_per_conversation: int = 2,
    max_facts_per_conversation: int = 2,
    max_memories_per_conversation: int = 2
) -> Dict[str, Any]:
    """
    Deterministic cross-conversation memory search.

    First version rules:
    - search across all stored conversations
    - search conversation_segments, memory_facts, and memories
    - score rows by simple keyword overlap
    - group matches by conversation_id
    - attach conversation metadata
    - keep output small and inspectable
    - no embeddings
    - no model involvement
    """

    keywords = _extract_keywords(user_message)

    conversations_result = (
        supabase.table("conversations")
        .select("*")
        .order("started_at", desc=True)
        .execute()
    )
    conversation_rows = conversations_result.data or []
    conversations_by_id = {
        row.get("id"): row
        for row in conversation_rows
        if row.get("id")
    }

    segments_result = (
        supabase.table("conversation_segments")
        .select("*")
        .order("segment_index")
        .execute()
    )
    all_segments = segments_result.data or []

    facts_result = (
        supabase.table("memory_facts")
        .select("*")
        .order("id")
        .execute()
    )
    all_facts = facts_result.data or []

    memories_result = (
        supabase.table("memories")
        .select("*")
        .order("id")
        .execute()
    )
    all_memories = memories_result.data or []

    facts_by_id = {
        row.get("id"): row
        for row in all_facts
        if row.get("id")
    }

    grouped_matches: dict[str, dict[str, Any]] = {}

    def ensure_group(conversation_id: str) -> dict[str, Any]:
        if conversation_id not in grouped_matches:
            grouped_matches[conversation_id] = {
                "conversation": conversations_by_id.get(conversation_id, {"id": conversation_id}),
                "segments": [],
                "facts": [],
                "memories": [],
                "score": 0,
            }
        return grouped_matches[conversation_id]

    # ---- SEGMENTS ----
    for segment in all_segments:
        conversation_id = segment.get("conversation_id")
        if not conversation_id:
            continue

        text_parts = [
            segment.get("topic") or "",
            segment.get("summary") or "",
            segment.get("raw_user_text") or "",
            segment.get("raw_assistant_text") or "",
        ]
        combined_text = " ".join(text_parts)

        match_count = _count_keyword_matches(combined_text, keywords)
        if match_count <= 0:
            continue

        group = ensure_group(conversation_id)
        segment_copy = dict(segment)
        segment_copy["_match_score"] = match_count
        group["segments"].append(segment_copy)
        group["score"] += match_count * 3

    # ---- FACTS ----
    for fact in all_facts:
        conversation_id = fact.get("conversation_id")
        if not conversation_id:
            continue

        combined_text = " ".join([
            fact.get("fact_text") or "",
            fact.get("fact_type") or "",
            fact.get("subject") or "",
            fact.get("predicate") or "",
            fact.get("object") or "",
        ])

        match_count = _count_keyword_matches(combined_text, keywords)
        if match_count <= 0:
            continue

        group = ensure_group(conversation_id)
        fact_copy = dict(fact)
        fact_copy["_match_score"] = match_count
        group["facts"].append(fact_copy)
        group["score"] += match_count * 2

    # ---- MEMORIES ----
    for memory in all_memories:
        conversation_id = memory.get("conversation_id")
        if not conversation_id:
            continue

        linked_fact = facts_by_id.get(memory.get("fact_id"))
        linked_fact_text = ""
        linked_fact_type = ""

        if linked_fact:
            linked_fact_text = linked_fact.get("fact_text") or ""
            linked_fact_type = linked_fact.get("fact_type") or ""

        combined_text = " ".join([
            memory.get("content") or "",
            memory.get("memory_kind") or "",
            memory.get("status") or "",
            linked_fact_text,
            linked_fact_type,
        ])

        match_count = _count_keyword_matches(combined_text, keywords)
        if match_count <= 0:
            continue

        group = ensure_group(conversation_id)
        memory_copy = dict(memory)
        memory_copy["_match_score"] = match_count
        group["memories"].append(memory_copy)
        group["score"] += match_count

    matches = []

    for conversation_id, group in grouped_matches.items():
        segments = _dedupe_rows_by_id(group["segments"])
        facts = _dedupe_rows_by_id(group["facts"])
        memories = _dedupe_rows_by_id(group["memories"])

        segments.sort(
            key=lambda row: (row.get("_match_score", 0), row.get("segment_index", 0)),
            reverse=True
        )
        facts.sort(key=lambda row: row.get("_match_score", 0), reverse=True)
        memories.sort(key=lambda row: row.get("_match_score", 0), reverse=True)

        selected_segments = segments[:max_segments_per_conversation]
        selected_facts = facts[:max_facts_per_conversation]
        selected_memories = memories[:max_memories_per_conversation]

        selected_segments.sort(key=lambda row: row.get("segment_index", 0))

        match_entry = {
            "conversation": group["conversation"],
            "segments": selected_segments,
            "facts": selected_facts,
            "memories": selected_memories,
            "counts": {
                "segments": len(selected_segments),
                "facts": len(selected_facts),
                "memories": len(selected_memories),
            },
            "score": group["score"],
        }

        matches.append(match_entry)

    matches.sort(
        key=lambda entry: (
            entry.get("score", 0),
            entry.get("conversation", {}).get("started_at") or ""
        ),
        reverse=True
    )

    matches = matches[:max_conversations]

    return {
        "query": user_message,
        "keywords": sorted(keywords),
        "matches": matches,
        "counts": {
            "conversations": len(matches),
            "segments": sum(len(match.get("segments", [])) for match in matches),
            "facts": sum(len(match.get("facts", [])) for match in matches),
            "memories": sum(len(match.get("memories", [])) for match in matches),
        },
    }


def format_context_for_prompt(context_pack: Dict[str, Any]) -> str:
    """
    Convert a structured context pack into a compact, inspectable
    plain-text block suitable for prompt injection.

    This is intentionally simple and deterministic:
    - no semantic ranking
    - no summarization logic
    - no model calls
    - raw excerpts are clipped to keep token usage bounded
    """

    conversation = context_pack.get("conversation") or {}
    segments = context_pack.get("segments") or []
    facts = context_pack.get("facts") or []
    memories = context_pack.get("memories") or []
    counts = context_pack.get("counts") or {}

    session_name = conversation.get("session_name") or "Untitled conversation"
    conversation_id = conversation.get("id") or "unknown"

    lines = [
        "CONTEXT PACK",
        f"Conversation ID: {conversation_id}",
        f"Session Name: {session_name}",
        (
            "Counts: "
            f"segments={counts.get('segments', 0)}, "
            f"facts={counts.get('facts', 0)}, "
            f"memories={counts.get('memories', 0)}"
        ),
        ""
    ]

    lines.append("SEGMENT SUMMARIES:")
    if segments:
        for segment in segments:
            segment_index = segment.get("segment_index", "?")
            topic = segment.get("topic")
            summary = segment.get("summary")

            if topic and summary:
                lines.append(f"- Segment {segment_index} | {topic}: {summary}")
            elif summary:
                lines.append(f"- Segment {segment_index}: {summary}")
            elif topic:
                lines.append(f"- Segment {segment_index} | {topic}")
            else:
                lines.append(f"- Segment {segment_index}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("FACTS:")
    if facts:
        for fact in facts:
            fact_text = fact.get("fact_text")
            if fact_text:
                lines.append(f"- {fact_text}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("MEMORIES:")
    if memories:
        for memory in memories:
            content = memory.get("content")
            if content:
                lines.append(f"- {content}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("RAW CONVERSATION EXCERPTS:")
    excerpt_added = False

    for segment in segments:
        segment_index = segment.get("segment_index", "?")
        raw_user_text = _clip_text(segment.get("raw_user_text"))
        raw_assistant_text = _clip_text(segment.get("raw_assistant_text"))

        if raw_user_text:
            lines.append(f"- Segment {segment_index} User: {raw_user_text}")
            excerpt_added = True

        if raw_assistant_text:
            lines.append(f"- Segment {segment_index} Assistant: {raw_assistant_text}")
            excerpt_added = True

    if not excerpt_added:
        lines.append("- None")

    return "\n".join(lines)


def format_global_context_for_prompt(global_search_result: Dict[str, Any]) -> str:
    """
    Convert grouped cross-conversation search results into a compact,
    inspectable plain-text block suitable for prompt injection.

    Rules:
    - keep output deterministic
    - group by conversation
    - include only matched items already selected by search_global_memory()
    - clip raw excerpts
    - keep section structure clear
    """

    query = global_search_result.get("query") or ""
    keywords = global_search_result.get("keywords") or []
    matches = global_search_result.get("matches") or []
    counts = global_search_result.get("counts") or {}

    lines = [
        "GLOBAL MEMORY CONTEXT",
        f"Query: {query}",
        f"Keywords: {', '.join(keywords) if keywords else 'None'}",
        (
            "Counts: "
            f"conversations={counts.get('conversations', 0)}, "
            f"segments={counts.get('segments', 0)}, "
            f"facts={counts.get('facts', 0)}, "
            f"memories={counts.get('memories', 0)}"
        ),
        ""
    ]

    if not matches:
        lines.append("MATCHES:")
        lines.append("- None")
        return "\n".join(lines)

    lines.append("MATCHES:")

    for index, match in enumerate(matches, start=1):
        conversation = match.get("conversation") or {}
        conversation_id = conversation.get("id") or "unknown"
        session_name = conversation.get("session_name") or "Untitled conversation"
        started_at = conversation.get("started_at") or "unknown"
        segments = match.get("segments") or []
        facts = match.get("facts") or []
        memories = match.get("memories") or []
        match_counts = match.get("counts") or {}
        score = match.get("score", 0)

        lines.append(
            f"Conversation {index}: {session_name} | ID={conversation_id} | "
            f"started_at={started_at} | score={score}"
        )
        lines.append(
            "Counts: "
            f"segments={match_counts.get('segments', 0)}, "
            f"facts={match_counts.get('facts', 0)}, "
            f"memories={match_counts.get('memories', 0)}"
        )

        lines.append("Segment summaries:")
        if segments:
            for segment in segments:
                segment_index = segment.get("segment_index", "?")
                topic = segment.get("topic")
                summary = segment.get("summary")

                if topic and summary:
                    lines.append(f"- Segment {segment_index} | {topic}: {summary}")
                elif summary:
                    lines.append(f"- Segment {segment_index}: {summary}")
                elif topic:
                    lines.append(f"- Segment {segment_index} | {topic}")
                else:
                    lines.append(f"- Segment {segment_index}")
        else:
            lines.append("- None")

        lines.append("Facts:")
        if facts:
            for fact in facts:
                fact_text = fact.get("fact_text")
                if fact_text:
                    lines.append(f"- {fact_text}")
        else:
            lines.append("- None")

        lines.append("Memories:")
        if memories:
            for memory in memories:
                content = memory.get("content")
                if content:
                    lines.append(f"- {content}")
        else:
            lines.append("- None")

        lines.append("Raw conversation excerpts:")
        excerpt_added = False

        for segment in segments:
            segment_index = segment.get("segment_index", "?")
            raw_user_text = _clip_text(segment.get("raw_user_text"))
            raw_assistant_text = _clip_text(segment.get("raw_assistant_text"))

            if raw_user_text:
                lines.append(f"- Segment {segment_index} User: {raw_user_text}")
                excerpt_added = True

            if raw_assistant_text:
                lines.append(f"- Segment {segment_index} Assistant: {raw_assistant_text}")
                excerpt_added = True

        if not excerpt_added:
            lines.append("- None")

        if index < len(matches):
            lines.append("")

    return "\n".join(lines)