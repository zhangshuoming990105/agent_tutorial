"""
Dynamic skill routing for tool/schema exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class SkillSpec:
    name: str
    description: str
    tools: list[str]
    triggers: list[str]
    always_on: bool
    instructions: str


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text

    meta_lines = []
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
        meta_lines.append(lines[i])
    if end is None:
        return {}, text

    meta: dict = {}
    key = None
    for raw in meta_lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and key:
            item = line.lstrip()[2:].strip()
            meta.setdefault(key, []).append(item)
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            key = k.strip()
            value = v.strip()
            if value == "":
                meta[key] = []
            else:
                meta[key] = value

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def load_skills(skills_root: Path) -> dict[str, SkillSpec]:
    skills: dict[str, SkillSpec] = {}
    if not skills_root.exists():
        return skills

    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        name = str(meta.get("name", skill_file.parent.name)).strip()
        description = str(meta.get("description", "")).strip()
        tools = meta.get("tools", [])
        triggers = meta.get("triggers", [])
        always_on_raw = str(meta.get("always_on", "false")).lower()
        always_on = always_on_raw in ("1", "true", "yes", "on")

        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",") if t.strip()]
        if isinstance(triggers, str):
            triggers = [t.strip() for t in triggers.split(",") if t.strip()]

        if not name:
            continue
        skills[name] = SkillSpec(
            name=name,
            description=description,
            tools=list(tools),
            triggers=[t.lower() for t in triggers],
            always_on=always_on,
            instructions=body,
        )
    return skills


def select_skills(
    user_input: str,
    all_skills: dict[str, SkillSpec],
    pinned_on: set[str] | None = None,
) -> list[SkillSpec]:
    pinned_on = pinned_on or set()
    text = user_input.lower()
    words = set(re.findall(r"[a-z0-9_./-]+", text))
    path_like = bool(re.search(r"(^|\s)(/|~\/)[^\s]*", text)) or bool(
        re.search(r"\b[a-z0-9_.-]+/[a-z0-9_./-]*", text)
    )

    selected: set[str] = set()
    for name, skill in all_skills.items():
        if skill.always_on:
            selected.add(name)
        if name in pinned_on:
            selected.add(name)
            continue
        if any((tr in text) or (tr in words) for tr in skill.triggers):
            selected.add(name)

    # Heuristic routing for common intents that may not match simple trigger words.
    if "filesystem" in all_skills:
        file_intent = any(
            k in text
            for k in (
                "what's in",
                "what is in",
                "list ",
                "show ",
                "contents",
                "inside ",
                "directory",
                "folder",
                "file ",
                "files ",
                "read ",
                "write ",
                "edit ",
                "search ",
                "grep",
                "find ",
            )
        )
        if path_like or file_intent:
            selected.add("filesystem")

    if "shell" in all_skills:
        shell_intent = any(
            k in text
            for k in (
                "run ",
                "execute ",
                "shell",
                "terminal",
                "command",
                "bash",
                "zsh",
                "pwd",
                "ls ",
                "git ",
                "npm ",
                "pip ",
                "python ",
            )
        ) or bool(re.search(r"[|;&`$()]", text))
        if shell_intent:
            selected.add("shell")

    if "core" in all_skills:
        selected.add("core")

    if not selected and all_skills:
        selected.add(next(iter(all_skills.keys())))

    return [all_skills[name] for name in sorted(selected) if name in all_skills]


def build_skill_prompt(selected_skills: list[SkillSpec]) -> str:
    if not selected_skills:
        return ""
    lines = ["Active skills guidance:"]
    for skill in selected_skills:
        lines.append(f"- [{skill.name}] {skill.description}")
        if skill.instructions:
            lines.append(skill.instructions.strip())
    return "\n".join(lines).strip()
