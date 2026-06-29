"""app/api/routes/skills.py — REST endpoints for dynamic skill management."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.api.schemas import (
    SkillActivateResponse,
    SkillDetail,
    SkillInfo,
    SkillUploadResponse,
)
from app.config import get_settings
from app.skills.discovery import _parse_flat_md, _parse_skill_md
from app.skills.registry import SkillRegistry
from app.skills.slugify import slugify

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/skills", tags=["Skills"])


def _get_registry(request: Request) -> SkillRegistry:
    registry = getattr(request.app.state, "skill_registry", None)
    if registry is None:
        raise HTTPException(503, "Skill registry not available")
    return registry


@router.get("", response_model=list[SkillInfo], summary="Список всех обнаруженных скиллов")
async def list_skills(request: Request):
    registry = _get_registry(request)
    available = registry.get_available()
    result: list[SkillInfo] = []
    for meta in available:
        resolved = registry.get_resolved(meta.slug)
        prompt_body = resolved.prompt_body if resolved else ""
        result.append(
            SkillInfo(
                slug=meta.slug,
                name=meta.name,
                version=meta.version,
                description=meta.description,
                has_tools=meta.has_tools,
                has_nodes=meta.has_nodes,
                active=registry.is_active(meta.slug),
                legacy=meta.legacy,
                prompt_preview=prompt_body[:200] if prompt_body else "",
            )
        )
    return result


@router.get("/active", response_model=list[SkillInfo], summary="Список активных скиллов")
async def list_active_skills(request: Request):
    registry = _get_registry(request)
    result: list[SkillInfo] = []
    for slug in registry.active_slugs:
        resolved = registry.get_resolved(slug)
        if resolved is None:
            continue
        meta = resolved.meta
        result.append(
            SkillInfo(
                slug=slug,
                name=meta.name,
                version=meta.version,
                description=meta.description,
                has_tools=meta.has_tools,
                has_nodes=meta.has_nodes,
                active=True,
                legacy=meta.legacy,
                prompt_preview=resolved.prompt_body[:200] if resolved.prompt_body else "",
            )
        )
    return result


@router.get("/{slug}", response_model=SkillDetail, summary="Детальная информация о скилле")
async def get_skill(slug: str, request: Request):
    registry = _get_registry(request)
    resolved = registry.get_resolved(slug)
    if resolved is None:
        # Try to find it in available but not yet resolved
        meta = registry.get_metadata(slug)
        if meta is None:
            raise HTTPException(404, f"Skill '{slug}' not found")
        # Resolve on-demand
        from app.skills.loader import resolve_skill

        resolved = resolve_skill(meta)
        if resolved is None:
            msg = f"Failed to resolve skill '{slug}'"
            raise HTTPException(422, msg)

    return SkillDetail(
        slug=resolved.meta.slug,
        name=resolved.meta.name,
        version=resolved.meta.version,
        description=resolved.meta.description,
        has_tools=resolved.meta.has_tools,
        has_nodes=resolved.meta.has_nodes,
        active=registry.is_active(slug),
        legacy=resolved.meta.legacy,
        prompt_preview=resolved.prompt_body[:200],
        license=resolved.meta.license,
        depends_on=resolved.meta.depends_on,
        tools=[t.name for t in resolved.tool_fns],
        prompt_body=resolved.prompt_body,
    )


@router.post(
    "/{slug}/activate",
    response_model=SkillActivateResponse,
    summary="Активировать скилл (глобально для всех сессий)",
)
async def activate_skill(slug: str, request: Request):
    registry = _get_registry(request)
    ok = registry.activate(slug)
    if not ok:
        raise HTTPException(400, f"Cannot activate skill '{slug}'")
    return SkillActivateResponse(slug=slug, active=True, message=f"Skill '{slug}' activated")


@router.post(
    "/{slug}/deactivate",
    response_model=SkillActivateResponse,
    summary="Деактивировать скилл",
)
async def deactivate_skill(slug: str, request: Request):
    registry = _get_registry(request)
    ok = registry.deactivate(slug)
    if not ok:
        raise HTTPException(400, f"Cannot deactivate skill '{slug}'")
    return SkillActivateResponse(slug=slug, active=False, message=f"Skill '{slug}' deactivated")


@router.post(
    "/upload",
    response_model=SkillUploadResponse,
    summary="Загрузить .md файл скилла (сохраняется в app/skills/uploads/)",
    status_code=201,
)
async def upload_skill(file: UploadFile, request: Request):
    settings = get_settings()
    registry = _get_registry(request)

    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(400, "Only .md files are accepted")

    # Validate file size
    max_bytes = settings.skills_max_upload_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            413,
            f"File too large (max {settings.skills_max_upload_mb} MB)",
        )

    text = contents.decode("utf-8")

    # Parse frontmatter to extract metadata and determine slug
    has_frontmatter = text.startswith("---")
    if has_frontmatter:
        parsed_slug = None
        parts = text.split("---", 2)
        if len(parts) >= 3:
            import yaml

            try:
                frontmatter = yaml.safe_load(parts[1])
                if isinstance(frontmatter, dict):
                    name = str(frontmatter.get("name", ""))
                    if name:
                        parsed_slug = slugify(name)
            except yaml.YAMLError:
                pass
        slug = parsed_slug or slugify(file.filename.removesuffix(".md"))
    else:
        slug = slugify(file.filename.removesuffix(".md"))

    # Check for collisions
    existing = registry.get_metadata(slug)
    if existing is not None:
        raise HTTPException(409, f"Skill '{slug}' already exists")

    # Save to uploads directory
    upload_dir = Path(settings.skills_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = upload_dir / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(text, encoding="utf-8")

    # Register in registry
    if has_frontmatter:
        meta = _parse_skill_md(skill_path, slug)
    else:
        meta = _parse_flat_md(skill_path)
    if meta is None:
        # Clean up on failure
        skill_path.unlink(missing_ok=True)
        raise HTTPException(422, "Failed to parse uploaded skill file")

    registry._available[slug] = meta
    if slug in registry._resolved:
        del registry._resolved[slug]

    return SkillUploadResponse(
        slug=meta.slug,
        name=meta.name,
        description=meta.description,
        version=meta.version,
        message=f"Skill '{meta.name}' uploaded and registered",
    )