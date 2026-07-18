"""
Prompt Registry — versioned prompt management system.

Provides a registry for storing, retrieving, and versioning prompt templates.
Supports listing all prompts, fetching by name/version, and selecting the 
latest version of a named prompt.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from functools import lru_cache

from app.prompts.templates import (
    RAG_DEFAULT_V1_SYSTEM, RAG_DEFAULT_V1_HUMAN,
    RAG_CONCISE_V1_SYSTEM, RAG_CONCISE_V1_HUMAN,
    RAG_DETAILED_V1_SYSTEM, RAG_DETAILED_V1_HUMAN,
    RAG_COT_V1_SYSTEM, RAG_COT_V1_HUMAN,
    RAG_CONVERSATIONAL_V1_SYSTEM, RAG_CONVERSATIONAL_V1_HUMAN,
    QUESTION_REFORMULATION_SYSTEM, QUESTION_REFORMULATION_HUMAN,
)


@dataclass
class PromptVersion:
    """A single versioned prompt template."""

    name: str
    version: int
    system_template: str
    human_template: str
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "system_template": self.system_template,
            "human_template": self.human_template,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class PromptRegistry:
    """
    In-memory registry for versioned prompt templates.
    
    Prompts are stored by name, with multiple versions per name.
    The registry is pre-populated with built-in prompt variants.
    """

    def __init__(self):
        # {name: {version: PromptVersion}}
        self._prompts: dict[str, dict[int, PromptVersion]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in prompt templates."""
        builtins = [
            PromptVersion(
                name="rag_default",
                version=1,
                system_template=RAG_DEFAULT_V1_SYSTEM,
                human_template=RAG_DEFAULT_V1_HUMAN,
                description="Standard RAG prompt — balanced, professional answers with source citations",
                metadata={"type": "rag", "style": "balanced"},
            ),
            PromptVersion(
                name="rag_concise",
                version=1,
                system_template=RAG_CONCISE_V1_SYSTEM,
                human_template=RAG_CONCISE_V1_HUMAN,
                description="Concise RAG prompt — brief 2-3 sentence answers",
                metadata={"type": "rag", "style": "concise"},
            ),
            PromptVersion(
                name="rag_detailed",
                version=1,
                system_template=RAG_DETAILED_V1_SYSTEM,
                human_template=RAG_DETAILED_V1_HUMAN,
                description="Detailed RAG prompt — comprehensive, structured answers with full citations",
                metadata={"type": "rag", "style": "detailed"},
            ),
            PromptVersion(
                name="rag_cot",
                version=1,
                system_template=RAG_COT_V1_SYSTEM,
                human_template=RAG_COT_V1_HUMAN,
                description="Chain-of-thought RAG prompt — step-by-step reasoning before answering",
                metadata={"type": "rag", "style": "chain_of_thought"},
            ),
            PromptVersion(
                name="rag_conversational",
                version=1,
                system_template=RAG_CONVERSATIONAL_V1_SYSTEM,
                human_template=RAG_CONVERSATIONAL_V1_HUMAN,
                description="Conversational RAG prompt — multi-turn context-aware responses",
                metadata={"type": "rag", "style": "conversational"},
            ),
            PromptVersion(
                name="question_reformulation",
                version=1,
                system_template=QUESTION_REFORMULATION_SYSTEM,
                human_template=QUESTION_REFORMULATION_HUMAN,
                description="Reformulates follow-up questions into standalone queries for retrieval",
                metadata={"type": "utility", "style": "reformulation"},
            ),
        ]
        for prompt in builtins:
            self.register(prompt)

    def register(self, prompt: PromptVersion) -> None:
        """Register a new prompt version."""
        if prompt.name not in self._prompts:
            self._prompts[prompt.name] = {}
        self._prompts[prompt.name][prompt.version] = prompt

    def get(self, name: str, version: Optional[int] = None) -> PromptVersion:
        """
        Get a prompt by name and optional version.
        
        Args:
            name: The prompt name.
            version: Specific version number. If None, returns latest.
            
        Raises:
            KeyError: If the prompt name or version does not exist.
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found. Available: {list(self._prompts.keys())}")

        versions = self._prompts[name]
        if version is not None:
            if version not in versions:
                raise KeyError(
                    f"Version {version} of prompt '{name}' not found. "
                    f"Available versions: {sorted(versions.keys())}"
                )
            return versions[version]

        # Return the latest version
        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_prompts(self) -> list[dict]:
        """List all registered prompts with their available versions."""
        result = []
        for name, versions in self._prompts.items():
            latest = versions[max(versions.keys())]
            result.append({
                "name": name,
                "latest_version": max(versions.keys()),
                "available_versions": sorted(versions.keys()),
                "description": latest.description,
                "metadata": latest.metadata,
            })
        return result

    def list_versions(self, name: str) -> list[dict]:
        """List all versions of a specific prompt."""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found.")
        return [v.to_dict() for v in sorted(
            self._prompts[name].values(), key=lambda p: p.version
        )]


@lru_cache()
def get_prompt_registry() -> PromptRegistry:
    """Get the singleton prompt registry instance."""
    return PromptRegistry()
