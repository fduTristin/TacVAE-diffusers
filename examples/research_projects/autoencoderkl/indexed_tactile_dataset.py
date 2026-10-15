"""Read image paths relative to a tactile training directory."""

from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_indexed_image_paths(index_file: str | Path, max_samples: int | None = None) -> list[Path]:
    """Accept nested relative paths while rejecting paths outside the dataset."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    paths = []
    with Path(index_file).open(encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            parts = name.split("/")
            if (
                "\\" in name
                or any(part in {"", ".", ".."} for part in parts)
                or Path(name).is_absolute()
                or Path(parts[-1]).suffix.lower() not in IMAGE_SUFFIXES
            ):
                raise ValueError(f"Invalid indexed image path: {name}")
            paths.append(Path(*parts))
            if max_samples is not None and len(paths) >= max_samples:
                break
    if not paths:
        raise ValueError(f"Training index is empty: {index_file}")
    return paths
