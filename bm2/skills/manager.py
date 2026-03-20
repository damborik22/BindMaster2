"""Skills system — loads, indexes, and queries domain-expertise documents.

Skills are markdown files with YAML frontmatter that encode expert knowledge
about protein binder design. They are NOT executable code — they are reference
material for users and LLM assistants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """A domain-expertise document."""

    name: str
    description: str
    keywords: list[str]
    content: str
    source: str = "builtin"  # "builtin" or "custom"


class SkillsManager:
    """Loads, indexes, and queries skill documents."""

    def __init__(
        self,
        builtin_dir: Optional[Path] = None,
        custom_dir: Optional[Path] = None,
    ):
        self._skills: dict[str, Skill] = {}
        self._builtin_dir = builtin_dir or (
            Path(__file__).parent / "builtin"
        )
        self._custom_dir = custom_dir
        self._load_builtin()
        if custom_dir:
            self._load_custom(custom_dir)

    def _load_builtin(self) -> None:
        if not self._builtin_dir.exists():
            return
        for md_path in sorted(self._builtin_dir.glob("*.md")):
            skill = self._parse_skill_file(md_path, source="builtin")
            if skill:
                self._skills[skill.name] = skill

    def _load_custom(self, directory: Path) -> None:
        if not directory.exists():
            return
        for md_path in sorted(directory.glob("*.md")):
            skill = self._parse_skill_file(md_path, source="custom")
            if skill:
                self._skills[skill.name] = skill

    def _parse_skill_file(
        self, path: Path, source: str
    ) -> Optional[Skill]:
        """Parse markdown file with optional YAML frontmatter.

        Format:
            ---
            name: strategy-selector
            description: Which tool to use for your target
            keywords: [tool, strategy, recommend]
            ---
            # Content here...
        """
        try:
            text = path.read_text()
        except Exception:
            return None

        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    import yaml

                    meta = yaml.safe_load(parts[1])
                    content = parts[2].strip()
                    return Skill(
                        name=meta.get("name", path.stem),
                        description=meta.get("description", ""),
                        keywords=meta.get("keywords", []),
                        content=content,
                        source=source,
                    )
                except Exception:
                    pass

        # Fallback: use filename as name
        return Skill(
            name=path.stem,
            description="",
            keywords=[path.stem.replace("-", " ")],
            content=text,
            source=source,
        )

    def query(self, question: str, top_n: int = 3) -> list[Skill]:
        """Find most relevant skills by keyword matching.

        Scores each skill by counting keyword hits in the question.
        """
        question_lower = question.lower()
        question_words = set(re.findall(r"\w+", question_lower))

        scored: list[tuple[float, Skill]] = []
        for skill in self._skills.values():
            score = 0.0
            for kw in skill.keywords:
                kw_lower = kw.lower()
                if kw_lower in question_lower:
                    score += 2  # Exact substring match
                elif kw_lower in question_words:
                    score += 1  # Word match
            for word in question_words:
                if word in skill.description.lower():
                    score += 0.5
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_n]]

    def get(self, name: str) -> Skill:
        """Get a skill by name.

        Raises:
            KeyError: If skill not found.
        """
        if name not in self._skills:
            raise KeyError(
                f"Skill not found: {name}. "
                f"Available: {self.list_names()}"
            )
        return self._skills[name]

    def list_names(self) -> list[str]:
        """Return sorted list of all skill names."""
        return sorted(self._skills.keys())

    def list_all(self) -> list[dict[str, str]]:
        """Return all skills with name, description, source."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
            }
            for s in self._skills.values()
        ]
