# pipeline/ingest/statcast.py
from pybaseball import statcast
from datetime import date, timedelta
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/statcast")

def fetch_and_cache_statcast(start: date, end: date) -> pd.DataFrame:
    """
    Fetch Statcast pitch-level data for a date range.
    Cached by date to raw/statcast/YYYY-MM-DD.parquet.
    NEVER called with end >= today (in production).
    """
    assert end < date.today(), \
        f"Attempted to fetch future Statcast data: end={end}. This is lookahead."
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    
    current = start
    while current <= end:
        cache_path = RAW_DIR / f"{current.isoformat()}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
        else:
            logger.info(f"Fetching Statcast for {current}")
            try:
                df = statcast(
                    start_dt=current.isoformat(),
                    end_dt=current.isoformat()
                )
                if not df.empty:
                    df.to_parquet(cache_path, index=False)
                    frames.append(df)
            except Exception as e:
                logger.error(f"Statcast fetch failed for {current}: {e}")
        current += timedelta(days=1)
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()