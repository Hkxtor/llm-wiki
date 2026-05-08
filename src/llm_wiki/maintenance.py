import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from llm_wiki.paths import WikiPaths


def append_log(
    paths: WikiPaths,
    event: Literal["ingest", "query", "lint"],
    title: str,
    details: str | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"## [{timestamp}] {event} | {title}\n"
    if details:
        entry += f"\n{details}\n"
    entry += "\n"

    existing = paths.log_md.read_text(encoding="utf-8") if paths.log_md.exists() else ""
    new_content = existing + entry

    # Atomic write — avoids corruption on interrupt
    tmp_fd, tmp_path = tempfile.mkstemp(dir=paths.wiki, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(tmp_path, paths.log_md)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _extract_title(md_file: Path) -> str | None:
    for line in md_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _extract_links(text: str) -> set[str]:
    return set(re.findall(r"\[\[([^\]]+)\]\]", text))


def _all_wiki_pages(paths: WikiPaths) -> list[Path]:
    pages: list[Path] = []
    for category_dir in (paths.concepts, paths.entities, paths.synthesis):
        if category_dir.exists():
            pages.extend(sorted(category_dir.glob("*.md")))
    return pages


def update_index(paths: WikiPaths) -> None:
    """Regenerate wiki/index.md from all pages in concepts/entities/synthesis."""
    sections: dict[str, list[str]] = {"concepts": [], "entities": [], "synthesis": []}

    for category in ("concepts", "entities", "synthesis"):
        category_dir = paths.wiki / category
        if not category_dir.exists():
            continue
        for md_file in sorted(category_dir.glob("*.md")):
            title = _extract_title(md_file) or md_file.stem
            sections[category].append(f"- [[{title}]]")

    lines = ["# Wiki Index\n"]
    for category, entries in sections.items():
        if entries:
            lines.append(f"## {category.capitalize()}\n")
            lines.extend(entries)
            lines.append("")

    content = "\n".join(lines) + "\n"

    tmp_fd, tmp_path = tempfile.mkstemp(dir=paths.wiki, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, paths.index_md)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class LintReport:
    orphans: tuple[str, ...]
    broken_links: tuple[str, ...]
    missing_titles: tuple[str, ...]
    empty_pages: tuple[str, ...]

    @property
    def has_issues(self) -> bool:
        return bool(self.orphans or self.broken_links or self.missing_titles or self.empty_pages)


def lint_wiki(paths: WikiPaths) -> LintReport:
    """Detect orphans, broken links, missing H1 titles, and empty pages."""
    all_pages = _all_wiki_pages(paths)

    title_to_path: dict[str, Path] = {}
    for page in all_pages:
        title = _extract_title(page)
        if title:
            title_to_path[title] = page

    inbound: dict[Path, int] = {p: 0 for p in all_pages}
    broken_links: list[str] = []

    for page in all_pages:
        text = page.read_text(encoding="utf-8")
        for link_title in _extract_links(text):
            if link_title in title_to_path:
                inbound[title_to_path[link_title]] += 1
            else:
                broken_links.append(f"{page.name}: [[{link_title}]]")

    orphans = [
        p.name for p, count in inbound.items()
        if count == 0 and p not in (paths.index_md,)
    ]

    missing_titles = [
        p.name for p in all_pages if _extract_title(p) is None
    ]

    empty_pages = [
        p.name for p in all_pages
        if len([l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]) < 3
    ]

    return LintReport(
        orphans=tuple(sorted(orphans)),
        broken_links=tuple(sorted(broken_links)),
        missing_titles=tuple(sorted(missing_titles)),
        empty_pages=tuple(sorted(empty_pages)),
    )
