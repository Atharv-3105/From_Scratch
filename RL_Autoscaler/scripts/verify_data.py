import zipfile 
import pyarrow.ipc as feather
from pathlib import Path 
import pandas as pd 

RAW_DIR = Path("data/raw")
ZIP_FILE = RAW_DIR / "azure-functions-dataset-2019.zip"
SCHEMA_REPORT = Path("data/schema_report.txt")

def verify_data():
    print(f"Found archive: {ZIP_FILE.name} ({ZIP_FILE.stat().st_size / (1024**2):.2f} MB)")
    
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        file_list=  zip_ref.namelist()
        print(f"\n Archive contains {len(file_list)} files:")
        for f in file_list:
            print(f"    - {f}")
        
        sample_file = next((f for f in file_list if f.endswith(('.csv', '.feather'))), None)
        
        if sample_file:
            print(f"Inspecting Sample File: {sample_file}")
            
            temp_path = zip_ref.extract(sample_file, path = RAW_DIR)
            try:
                if sample_file.endswith('.feather'):
                    df = pd.read_feather(temp_path)
                else:
                    df = pd.read_csv(temp_path, nrows = 2)
                    
                print(f"Successfully read sample. Columns: {list(df.columns)[:5]}...")
                print(f"    Shape:{df.shape}")
            except Exception as e:
                print(f"Couldn't read sample file: {e}")
            
            finally:
                Path(temp_path).unlink(missing_ok=True)
    
def get_all_shapes():
    print("Dataset Shapes:\n" + "-"*40)
    
    with zipfile.ZipFile(ZIP_FILE, 'r') as ref:
        for file_info in ref.infolist():
            filename = file_info.filename 
            if filename.endswith('.feather'):
                temp_path = ref.extract(filename, path = RAW_DIR)
                try:
                    with open(temp_path, 'rb') as f:
                        reader = feather.open_file(f)
                        num_cols = len(reader.schema)
                        num_rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))

                    print(f"{filename:<45} {num_rows:>10,} rows, {num_cols:>2} cols")
                except Exception as e:
                    print(f" Couldn't read metadata for {filename}: {e}")
                finally:
                    Path(temp_path).unlink(missing_ok = True)

def get_schema():
    with open(SCHEMA_REPORT, 'w', encoding='utf-8') as out:
        print("Schema Discovery:\n" + "="*60, file=out)

        with zipfile.ZipFile(ZIP_FILE, 'r') as ref:
            for file_info in ref.infolist():
                filename = file_info.filename
                if filename.endswith('.feather'):
                    temp_path = ref.extract(filename, path = RAW_DIR)
                    reader = None
                    try:
                        with open(temp_path, 'rb') as f:

                            reader = feather.open_file(f)
                            schema = reader.schema

                            print(f"\n{filename}", file=out)
                            print("-"*60, file=out)
                            print(f"Columns ({len(schema)})", file=out)
                            for i, field in enumerate(schema):
                                print(f"    {i:3d}: {field.name:<40} {field.type}", file=out)

                            batch = reader.get_batch(0)
                            print(f"\nFirst Row Sample:", file=out)
                            for i, field in enumerate(schema):
                                if i < 5 or field.name.startswith('Hash'):
                                    val = batch.column(field.name)[0].as_py()
                                    print(f"    {field.name}: {val}", file=out)
                                elif i == 5:
                                    print(f"    ...(showing first 5 columns)", file=out)
                    except Exception as e:
                        print(f"Error reading {filename}: {e}", file=out)
                    finally:
                        Path(temp_path).unlink(missing_ok=True)

    print(f"Schema report written to {SCHEMA_REPORT}")


if __name__ == "__main__":
    # verify_data()
    # get_all_shapes()
    cols_to_show = ["HashFunction", "total_invocations"]
    df = pd.read_parquet("data/processed/function_day.parquet")
    print(df.head())
    
    