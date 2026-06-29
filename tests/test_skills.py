"""Unit tests for the dynamic skill loading system."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.skills.discovery import discover_all
from app.skills.loader import resolve_skill
from app.skills.registry import SkillRegistry
from app.skills.models import SkillMetadata
from app.skills.slugify import slugify
from app.skills.prompt_builder import build_expert_prompt, build_engineer_prompt


def test_discover_builtin_skills():
    """Verify builtin skills are discovered by default."""
    metas = discover_all()
    slugs = {m.slug for m in metas}
    assert "st-style-guide" in slugs, "Missing builtin: st-style-guide"
    assert "matiec-st-guide" in slugs, "Missing builtin: matiec-st-guide"
    assert len(slugs) == 2, f"Expected 2 builtin skills, got {len(slugs)}"


def test_discover_ponytail_explicit_dir():
    """Explicitly scanning the external skills dir works for tests."""
    metas = discover_all("skills")
    slugs = {m.slug for m in metas}
    expected = {"ponytail", "ponytail-review", "ponytail-audit",
                "ponytail-debt", "ponytail-gain", "ponytail-help"}
    assert expected.issubset(slugs), f"Missing: {expected - slugs}"


def test_resolve_builtin_skill():
    """Resolve st-style-guide and verify prompt body extraction."""
    metas = [m for m in discover_all() if m.slug == "st-style-guide"]
    assert len(metas) == 1
    resolved = resolve_skill(metas[0])
    assert resolved is not None
    assert "ST Style Guide" in resolved.prompt_body
    assert len(resolved.prompt_body) > 100
    assert len(resolved.tool_fns) == 0
    assert not resolved.prompt_body.startswith("---"), "Frontmatter should be stripped"
    assert resolved.meta.name == "st-style-guide"


def test_resolve_matiec_guide():
    """Resolve matiec-st-guide and verify matiec-specific content."""
    metas = [m for m in discover_all() if m.slug == "matiec-st-guide"]
    assert len(metas) == 1
    resolved = resolve_skill(metas[0])
    assert resolved is not None
    assert "BEGIN" in resolved.prompt_body or "matiec" in resolved.prompt_body.lower()


def test_resolve_nonexistent_skill():
    """resolve_skill returns None when path is missing."""
    meta = SkillMetadata(slug="ghost", name="Ghost", description="")
    resolved = resolve_skill(meta)
    assert resolved is None


def test_registry_activate_deactivate():
    """Test activation lifecycle with builtin skills."""
    reg = SkillRegistry()
    reg.discover_all()
    assert reg.activate("st-style-guide"), "Should activate"
    assert reg.is_active("st-style-guide")
    assert "st-style-guide" in reg.active_slugs
    assert reg.activate("st-style-guide"), "Double activate"
    assert len(reg.active_slugs) == 1
    assert reg.deactivate("st-style-guide"), "Should deactivate"
    assert not reg.is_active("st-style-guide")


def test_registry_unknown_skill():
    """Activating unknown skills should fail gracefully."""
    reg = SkillRegistry()
    reg.discover_all()
    assert not reg.activate("nonexistent")
    assert reg.active_slugs == []


def test_registry_active_prompt_bodies():
    """active_prompt_bodies should return bodies of active skills."""
    reg = SkillRegistry()
    reg.discover_all()
    reg.activate("st-style-guide")
    reg.activate("matiec-st-guide")
    bodies = reg.active_prompt_bodies()
    assert len(bodies) == 2
    assert any("ST Style Guide" in b for b in bodies)
    assert any("matiec" in b.lower() for b in bodies)


def test_build_expert_prompt_no_skills():
    """Without skills, the base prompt is returned verbatim."""
    base = "You are an Expert PLC engineering agent."
    assert build_expert_prompt(base, None, None) == base
    reg = SkillRegistry()
    reg.discover_all()
    assert build_expert_prompt(base, reg, []) == base


def test_build_expert_prompt_with_skill():
    """With active skills, their bodies are appended."""
    base = "You are an Expert."
    reg = SkillRegistry()
    reg.discover_all()
    reg.activate("st-style-guide")
    result = build_expert_prompt(base, reg, ["st-style-guide"])
    assert result.startswith(base)
    assert "ST Style Guide" in result


def test_build_engineer_prompt_filters():
    """build_engineer_prompt only injects st-style-guide and matiec-st-guide."""
    base = "Engineer base."
    reg = SkillRegistry()
    reg.discover_all()
    reg.activate("st-style-guide")
    result = build_engineer_prompt(base, reg, ["st-style-guide"])
    assert "ST Style Guide" in result

    # Activate a non-engineer skill, it should NOT appear
    from app.skills.discovery import discover_all as disc
    for m in disc("skills"):
        if m.slug == "ponytail":
            reg._available["ponytail"] = m
    reg.activate("ponytail")
    result2 = build_engineer_prompt(base, reg, ["st-style-guide", "ponytail"])
    assert "ST Style Guide" in result2
    assert "lazy" not in result2.lower(), "ponytail body should be filtered"


def test_slugify_basic():
    assert slugify("My Skill") == "my-skill"
    assert slugify("Hello-World_123") == "hello-world_123"


def test_slugify_cyrillic_fallback():
    result = slugify("Привет мир")
    assert result.startswith("skill-")
    assert len(result) == 18


def test_get_available_and_metadata():
    reg = SkillRegistry()
    metas = reg.discover_all()
    assert len(reg.get_available()) == len(metas)
    meta = reg.get_metadata("st-style-guide")
    assert meta is not None
    assert meta.slug == "st-style-guide"
    assert reg.get_metadata("nope") is None


def test_get_resolved_resolves_lazily():
    reg = SkillRegistry()
    reg.discover_all()
    reg.activate("st-style-guide")
    resolved = reg.get_resolved("st-style-guide")
    assert resolved is not None
    assert len(resolved.prompt_body) > 0


if __name__ == "__main__":
    test_discover_builtin_skills()
    test_discover_ponytail_explicit_dir()
    test_resolve_builtin_skill()
    test_resolve_matiec_guide()
    test_resolve_nonexistent_skill()
    test_registry_activate_deactivate()
    test_registry_unknown_skill()
    test_registry_active_prompt_bodies()
    test_build_expert_prompt_no_skills()
    test_build_expert_prompt_with_skill()
    test_build_engineer_prompt_filters()
    test_slugify_basic()
    test_slugify_cyrillic_fallback()
    test_get_available_and_metadata()
    test_get_resolved_resolves_lazily()
    print("ALL TESTS PASSED")