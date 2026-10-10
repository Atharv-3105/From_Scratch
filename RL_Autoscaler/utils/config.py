from pathlib import Path 
import yaml 
from typing import Union 

def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding = 'utf-8') as f:
        config = yaml.safe_load(f)
        
    return config if config is not None else {}