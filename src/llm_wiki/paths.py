from dataclasses import dataclass
from pathlib import Path

from llm_wiki.config import Config


@dataclass(frozen=True)
class WikiPaths:
    root: Path
    raw: Path
    wiki: Path
    concepts: Path
    entities: Path
    synthesis: Path
    index_md: Path
    log_md: Path

    @classmethod
    def from_config(cls, cfg: Config, root: Path | None = None) -> "WikiPaths":
        base = root or Path.cwd()
        wiki = base / cfg.wiki_dir
        return cls(
            root=base,
            raw=base / cfg.raw_dir,
            wiki=wiki,
            concepts=wiki / "concepts",
            entities=wiki / "entities",
            synthesis=wiki / "synthesis",
            index_md=wiki / "index.md",
            log_md=wiki / "log.md",
        )

    def ensure_exists(self) -> None:
        for d in (self.raw, self.concepts, self.entities, self.synthesis):
            d.mkdir(parents=True, exist_ok=True)
