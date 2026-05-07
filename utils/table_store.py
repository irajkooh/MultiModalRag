import json
import logging
import re
import sqlite3
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
TABLE_DIR = Path(__file__).parent.parent / "data" / "tables"

_num_re = re.compile(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$')


def _safe_name(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", source)[:40]


def _try_clean_numeric(series: pd.Series) -> "pd.Series | None":
    """Extract trailing number from OCR-garbled text (e.g. 'Mobile -833.71'). Returns float series if >60% parse."""
    # Skip date-like columns — datetime strings end in digits and would be mangled
    try:
        parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        parsed_dates = pd.to_datetime(series, errors="coerce")
    if parsed_dates.notna().mean() > 0.5:
        return None

    def extract_last_num(s):
        s_str = str(s).strip()
        is_negative = s_str.startswith('-') and not s_str[1:2].isdigit()
        m = _num_re.search(s_str)
        if m:
            val = float(m.group(1).replace(',', ''))
            if is_negative and val > 0:
                val = -val
            return val
        return None

    cleaned = series.map(extract_last_num)
    if cleaned.notna().mean() > 0.6:
        return cleaned
    return None


class TableStore:
    def __init__(self, table_dir=TABLE_DIR):
        self.dir = Path(table_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "index.json"
        self._index: dict[str, int] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text())
                # Migrate from old parquet-path format (values were lists) to int count format
                self._index = {k: v for k, v in raw.items() if isinstance(v, int)}
            except Exception:
                self._index = {}

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _db_path(self, source: str) -> Path:
        return self.dir / f"{_safe_name(source)}.db"

    def save(self, source: str, dataframes: list):
        db_path = self._db_path(source)
        db_path.unlink(missing_ok=True)
        n = 0
        if dataframes:
            conn = sqlite3.connect(str(db_path))
            for i, df in enumerate(dataframes):
                try:
                    df.to_sql(f"t{i}", conn, if_exists="replace", index=False)
                    n += 1
                except Exception as e:
                    logger.warning(f"SQLite write failed for '{source}' table {i}: {e}")
            conn.close()
        self._index[source] = n
        self._save_index()

    def load(self, source: str) -> list[pd.DataFrame]:
        n = self._index.get(source, -1)
        if n <= 0:
            return []
        db_path = self._db_path(source)
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        tables = []
        for i in range(n):
            try:
                tables.append(pd.read_sql(f"SELECT * FROM t{i}", conn))
            except Exception:
                pass
        conn.close()
        return tables

    def has_tables(self, source: str) -> bool:
        return self._index.get(source, 0) > 0

    def was_attempted(self, source: str) -> bool:
        return source in self._index

    def remove(self, source: str):
        self._db_path(source).unlink(missing_ok=True)
        self._index.pop(source, None)
        self._save_index()

    def clear_all(self):
        for source in list(self._index.keys()):
            self._db_path(source).unlink(missing_ok=True)
        self._index = {}
        self._save_index()

    def load_into_memory(self, sources: list[str]) -> tuple[sqlite3.Connection, list[dict]]:
        """Load tables for `sources` into an in-memory SQLite DB.

        Returns (conn, schema_info) where schema_info is a list of dicts with
        keys: table_name, source, columns, sample_str.
        Only tables with ≥2 rows and ≥2 cols and at least one numeric-ish column
        are included (garbage-table filter).
        """
        conn = sqlite3.connect(":memory:")
        schema_info = []
        for src in sources:
            dfs = self.load(src)
            src_idx = 0
            for df in dfs:
                if not _is_useful(df):
                    continue
                # Clean OCR-garbled text columns that are actually numeric
                df = df.copy()
                for col in list(df.columns):
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        cleaned = _try_clean_numeric(df[col])
                        if cleaned is not None:
                            df[col] = cleaned
                tname = f"{_safe_name(src)}_t{src_idx}"
                src_idx += 1
                try:
                    df.to_sql(tname, conn, if_exists="replace", index=False)
                except Exception as e:
                    logger.warning(f"in-memory load failed for '{src}': {e}")
                    continue
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                sample = df.head(min(len(df), 8)).to_string(index=False)
                schema_info.append({
                    "table_name": tname,
                    "source": src,
                    "numeric_cols": numeric_cols,
                    "text_cols": text_cols,
                    "sample": sample,
                    "nrows": len(df),
                })
        return conn, schema_info


def _is_useful(df: pd.DataFrame) -> bool:
    if len(df) < 2 or len(df.columns) < 2:
        return False
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return True
        parsed = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,\s%]", "", regex=True),
            errors="coerce",
        )
        if parsed.notna().mean() > 0.35:
            return True
    return False
