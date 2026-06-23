"""
HiveLink Skills — v0.9
======================
A skill is a named, reusable context block that gets injected into the system
prompt before a chat session. Skills can include:

  - A system prompt (instructions, persona, KB Rides context, etc.)
  - An optional list of input fields to collect before the session starts
    (the "elicitation" UI — a modal that pops up asking the user for values
    that are then interpolated into the system prompt)
  - Metadata: name, description, author, version, trigger keywords

Skill format (JSON):
{
  "id":          "kb-rides-analyst",         # auto-generated if absent on import
  "name":        "KB Rides Analyst",
  "description": "Shopify/LTV analyst for KB Rides store",
  "author":      "Enrico",
  "version":     "1.0.0",
  "trigger":     "analyze, shopify, revenue, ltv",   # comma-separated hint words
  "system":      "You are an expert Shopify analyst...",
  "inputs": [                                # optional — triggers elicitation UI
    {
      "id":          "store_name",
      "label":       "Store name",
      "placeholder": "e.g. KB Rides",
      "required":    true
    }
  ]
}

Skills are stored in ~/.hivelink/skills.json as a list.
Remote skills (imported from a URL) can be plain markdown or JSON.
If markdown, the entire content becomes the system prompt (Vercel-style skill
files are plain markdown with YAML frontmatter for metadata).
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

# ── Storage ───────────────────────────────────────────────────────────────────

_SKILLS_DIR  = Path.home() / ".hivelink"
_SKILLS_FILE = _SKILLS_DIR / "skills.json"


def _load() -> list[dict]:
    try:
        if _SKILLS_FILE.exists():
            return json.loads(_SKILLS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save(skills: list[dict]) -> None:
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    _SKILLS_FILE.write_text(json.dumps(skills, indent=2), encoding="utf-8")


# ── CRUD ──────────────────────────────────────────────────────────────────────

def list_skills() -> list[dict]:
    return _load()


def get_skill(skill_id: str) -> dict | None:
    return next((s for s in _load() if s.get("id") == skill_id), None)


def create_skill(data: dict) -> dict:
    skills = _load()
    skill  = _normalise(data)
    # Enforce unique id
    existing_ids = {s["id"] for s in skills}
    if skill["id"] in existing_ids:
        skill["id"] = str(uuid.uuid4())[:8]
    skills.append(skill)
    _save(skills)
    return skill


def update_skill(skill_id: str, data: dict) -> dict | None:
    skills = _load()
    for i, s in enumerate(skills):
        if s.get("id") == skill_id:
            updated = {**s, **data, "id": skill_id}
            skills[i] = _normalise(updated)
            _save(skills)
            return skills[i]
    return None


def delete_skill(skill_id: str) -> bool:
    skills = _load()
    new    = [s for s in skills if s.get("id") != skill_id]
    if len(new) == len(skills):
        return False
    _save(new)
    return True


# ── Import from URL ───────────────────────────────────────────────────────────

async def import_from_url(url: str) -> dict:
    """
    Fetch a skill from a URL and save it.
    Supports two formats:
      1. JSON  — must be a valid skill dict (see module docstring)
      2. Markdown — entire content becomes the system prompt;
                    YAML frontmatter (---) is parsed for metadata fields.
                    This is compatible with Vercel-style skill markdown files.
    """
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.text

    # Try JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict) and ("system" in data or "name" in data):
            data["source_url"] = url
            return create_skill(data)
    except json.JSONDecodeError:
        pass

    # Treat as markdown — parse YAML frontmatter if present
    data = _parse_markdown_skill(content, url)
    return create_skill(data)


def _parse_markdown_skill(content: str, source_url: str) -> dict:
    """
    Parse a Vercel-style markdown skill file.
    Frontmatter (between --- delimiters) is parsed for name, description,
    author, version, trigger. Everything else becomes the system prompt.
    """
    meta: dict[str, Any] = {}
    body = content

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if fm_match:
        fm_text, body = fm_match.group(1), fm_match.group(2)
        for line in fm_text.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip().lower()] = v.strip()

    # Derive a name from the URL if not in frontmatter
    if "name" not in meta:
        slug = source_url.rstrip("/").split("/")[-1]
        slug = re.sub(r"\.(md|txt|json)$", "", slug)
        meta["name"] = slug.replace("-", " ").replace("_", " ").title()

    return {
        "name":        meta.get("name", "Imported skill"),
        "description": meta.get("description", ""),
        "author":      meta.get("author", ""),
        "version":     meta.get("version", "1.0.0"),
        "trigger":     meta.get("trigger", ""),
        "system":      body.strip(),
        "inputs":      [],
        "source_url":  source_url,
    }


# ── Normalise ─────────────────────────────────────────────────────────────────

def _normalise(data: dict) -> dict:
    """Ensure all required fields exist and id is set."""
    return {
        "id":          data.get("id") or str(uuid.uuid4())[:8],
        "name":        data.get("name", "Untitled skill"),
        "description": data.get("description", ""),
        "author":      data.get("author", ""),
        "version":     data.get("version", "1.0.0"),
        "trigger":     data.get("trigger", ""),
        "system":      data.get("system", ""),
        "inputs":      data.get("inputs", []),
        "source_url":  data.get("source_url", ""),
    }


# ── System prompt resolution ───────────────────────────────────────────────────

def resolve_system_prompt(skill_id: str, input_values: dict[str, str]) -> str:
    """
    Return the final system prompt for a skill, with input values interpolated.
    Template variables use {input_id} syntax.
    e.g. "You are analyzing {store_name}" → "You are analyzing KB Rides"
    """
    skill = get_skill(skill_id)
    if not skill:
        return ""
    prompt = skill.get("system", "")
    for k, v in input_values.items():
        prompt = prompt.replace("{" + k + "}", v)
    return prompt
