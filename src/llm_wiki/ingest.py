import importlib.resources
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from llm_wiki.llm import LLMClient
from llm_wiki.maintenance import update_index
from llm_wiki.paths import WikiPaths
from llm_wiki.sync import extract_text_from_file


@dataclass(frozen=True)
class WikiPageDraft:
    category: Literal["concepts", "entities", "synthesis"]
    slug: str
    title: str
    body_md: str
    source_refs: tuple[str, ...]


def _load_prompt(name: str) -> str:
    return importlib.resources.files("llm_wiki.prompts").joinpath(name).read_text(encoding="utf-8")


def _parse_outline(raw_json: str) -> list[dict]:
    try:
        data = json.loads(raw_json)
        pages = data.get("pages", [])
        valid = []
        for p in pages:
            if all(k in p for k in ("category", "slug", "title", "one_line")):
                if p["category"] in ("concepts", "entities", "synthesis"):
                    valid.append(p)
        return valid
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for outline: {exc}") from exc


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text)


def ingest_source(
    source_path: Path,
    paths: WikiPaths,
    llm: LLMClient,
) -> list[WikiPageDraft]:
    content = extract_text_from_file(str(source_path))
    if not content.strip():
        return []

    outline_system = _load_prompt("ingest_outline.md")
    outline_raw = llm.complete(outline_system, content, response_format="json")
    page_specs = _parse_outline(outline_raw)

    page_system = _load_prompt("ingest_page.md")
    drafts: list[WikiPageDraft] = []

    for spec in page_specs:
        user_msg = (
            f"Page to write:\n"
            f"- title: {spec['title']}\n"
            f"- category: {spec['category']}\n"
            f"- one_line: {spec['one_line']}\n\n"
            f"Source document:\n{content}"
        )
        body_md = llm.complete(page_system, user_msg, response_format="text")
        drafts.append(
            WikiPageDraft(
                category=spec["category"],
                slug=_slugify(spec["slug"]),
                title=spec["title"],
                body_md=body_md.strip(),
                source_refs=(str(source_path),),
            )
        )

    return drafts


def _target_path(draft: WikiPageDraft, paths: WikiPaths) -> Path:
    category_dir: dict[str, Path] = {
        "concepts": paths.concepts,
        "entities": paths.entities,
        "synthesis": paths.synthesis,
    }
    return category_dir[draft.category] / f"{draft.slug}.md"


def _backup(target: Path, paths: WikiPaths) -> None:
    trash_dir = paths.wiki / ".wiki-trash"
    trash_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    shutil.copy2(target, trash_dir / f"{target.stem}-{ts}.md")


def write_drafts(
    drafts: list[WikiPageDraft],
    paths: WikiPaths,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    written: list[Path] = []
    today = date.today().isoformat()

    for draft in drafts:
        target = _target_path(draft, paths)

        if dry_run:
            action = "CREATE" if not target.exists() else ("OVERWRITE" if overwrite else "MERGE")
            print(f"[dry-run] {action} {target}")
            written.append(target)
            continue

        if not target.exists():
            target.write_text(draft.body_md + "\n", encoding="utf-8")
        elif overwrite:
            _backup(target, paths)
            target.write_text(draft.body_md + "\n", encoding="utf-8")
        else:
            _backup(target, paths)
            existing = target.read_text(encoding="utf-8")
            update_section = f"\n\n## Update {today}\n\n{draft.body_md}\n"
            target.write_text(existing.rstrip() + update_section, encoding="utf-8")

        written.append(target)

    if written and not dry_run:
        update_index(paths)

    return written
