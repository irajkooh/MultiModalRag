import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
IMAGE_DIR = Path(__file__).parent.parent / "data" / "images"


class ImageStore:
    def __init__(self, image_dir=IMAGE_DIR):
        self.dir = Path(image_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "index.json"
        self._index = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except Exception:
                self._index = {}

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _source_dir(self, source: str) -> Path:
        safe = source.replace("/", "_").replace("\\", "_")
        return self.dir / safe

    def save(self, source: str, images: list):
        """Save list of (page, img_idx, PIL.Image) tuples. Returns count saved."""
        self._remove_files(source)
        self._index[source] = []
        src_dir = self._source_dir(source)
        src_dir.mkdir(parents=True, exist_ok=True)
        for page, img_idx, pil_img in images:
            path = src_dir / f"p{page}_i{img_idx}.png"
            pil_img.save(path, format="PNG")
            self._index[source].append({"path": str(path), "page": page, "image_index": img_idx})
        self._save_index()
        return len(images)

    def list_images(self, source: str) -> list:
        return [e for e in self._index.get(source, []) if Path(e["path"]).exists()]

    def has_images(self, source: str) -> bool:
        return bool(self._index.get(source))

    def was_attempted(self, source: str) -> bool:
        return source in self._index

    def remove(self, source: str):
        self._remove_files(source)
        self._index.pop(source, None)
        self._save_index()

    def clear_all(self):
        for source in list(self._index.keys()):
            self._remove_files(source)
        self._index = {}
        self._save_index()

    def _remove_files(self, source: str):
        for e in self._index.get(source, []):
            Path(e["path"]).unlink(missing_ok=True)
        src_dir = self._source_dir(source)
        if src_dir.exists() and not any(src_dir.iterdir()):
            src_dir.rmdir()
