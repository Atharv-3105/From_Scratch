import zipfile 
from contextlib import contextmanager
from pathlib import Path 
import pandas as pd 

from traces.ingest import stream_invocations, KEYS_FILE
from traces.join import build_function_day_table 

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ZIP_FILE = RAW_DIR / "azure-functions-dataset-2019.zip"
TRAIN_DAYS = range(1, 11)

@contextmanager
def extracted(ref, name, dest):
    path = Path(ref.extract(name, path = dest))
    try:
        yield path 
    finally:
        path.unlink(missing_ok = True)
        
def main(zip = ZIP_FILE, raw_dir = RAW_DIR, proceesed_dir = PROCESSED_DIR):
    proceesed_dir = Path(proceesed_dir)
    
    with zipfile.ZipFile(zip, "r") as z:
        print("Streaming invocations")
        with extracted(z, "invocations_per_function_md.feather", raw_dir) as p:
            stream_invocations(p, proceesed_dir)
            
        print("Loading durations and memory")
        with extracted(z, "function_durations_percentiles.feather", raw_dir) as p:
            dur = pd.read_feather(p)
        with extracted(z, "app_memory_percentiles.feather", raw_dir) as p:
            mem = pd.read_feather(p)
            
        
    keys = pd.read_parquet(proceesed_dir / KEYS_FILE)
    table = build_function_day_table(keys, dur, mem, train_days=TRAIN_DAYS)
    
    output = proceesed_dir / "function_day.parquet"
    table.to_parquet(output, engine = "pyarrow")
    print(f"Saved {len(table):,} function-days to {output}")
    print("\nduration source:\n", table["dur_source"].value_counts().to_string())
    print("\nmemory source:\n", table["mem_source"].value_counts().to_string())
    
    
if __name__ == "__main__":
    main()

