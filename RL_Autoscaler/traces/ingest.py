from pathlib import Path 
import numpy as np 
import pandas as pd 
import pyarrow as pa 
import pyarrow.ipc as ipc
import pyarrow.parquet as pq 

N_MIN = 1440
MINUTE_COLS = [str(i) for i in range(1, N_MIN + 1)]
KEY_COLS = ["HashOwner", "HashApp", "HashFunction", "Trigger", "day"]
MINUTE_DTYPE = np.uint32 
DTYPE_MAX = np.iinfo(MINUTE_DTYPE).max 
MINUTES_FILE = "inv_minutes.bin"
KEYS_FILE = "inv_keys.parquet"

def batch_to_block(batch):
    block = np.empty((batch.num_rows, N_MIN), dtype = MINUTE_DTYPE)
    for j, name in enumerate(MINUTE_COLS):
        col = batch.column(name)
        if col.null_count:
            raise ValueError(f"column {name} has nulls")
        arr = col.to_numpy()
        if arr.min() < 0 or arr.max() > DTYPE_MAX:
            raise ValueError("column does not fit in MINUTE_DTYPE size")
        block[:, j] = arr 
    
    return block 

def stream_invocations(feather_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    
    key_tables = []
    with open(feather_path, "rb") as src, open(out_dir / MINUTES_FILE, "wb") as dst:
        reader = ipc.open_file(src)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            if batch.num_rows == 0:
                continue 
            block = batch_to_block(batch)
            dst.write(memoryview(block))
            
            keys = pa.Table.from_batches([batch.select(KEY_COLS)])
            totals = pa.array(block.sum(axis = 1, dtype = np.int64))
            key_tables.append(keys.append_column("total_invocations", totals))
            
    pq.write_table(pa.concat_tables(key_tables), out_dir / KEYS_FILE)
    
def load_invocations(out_dir):
    out_dir = Path(out_dir)
    keys = pd.read_parquet(out_dir / KEYS_FILE)
    minutes = np.memmap(out_dir / KEYS_FILE, dtype = MINUTE_DTYPE, mode = "r").reshape(-1, N_MIN)
    if len(keys) != minutes.shape[0]:
        raise ValueError("keys and minute matrix are misaligned")
    
    return keys, minutes 
            