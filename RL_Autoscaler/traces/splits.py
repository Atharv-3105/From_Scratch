import hashlib 
from pathlib import Path 
import pandas as pd 

ALL_DAYS = set(range(1, 15))
UNIT_COLUMNS = {"owner": ["HashOwner"], "app": ["HashOwner", "HashApp"]}
FUNC_KEYS = ["HashOwner", "HashApp", "HashFunction"]

def split_of(unit_key, seed, fractions):
    """ 
        Decides the split of 1 Owner from its own name and the seed only
        We hash instead of shuffling a list, so the answer for this owner never changes when other owners are added or removed
    """
    
    digest = hashlib.sha256(f"{seed}|{unit_key}".encode()).hexdigest()
    position = int(digest[:12], 16) / 16 ** 12  #A random number in range [0, 1)
    
    if position < fractions["train"]:
        return "train"
    if position < fractions["train"] + fractions["val"]:
        return "val"
    
    return "test"

def assign_splits(stats, config):
    """ 
        Add a 'split' column, Every function of the same onwer(or app) gets the same split
        inlcuding functions with no train days
    """
    columns = UNIT_COLUMNS[config["split_unit"]]
    stats = stats.drop(columns = "split", errors = "ignore")
    units = stats[columns].drop_duplicates().reset_index(drop = True)
    units["split"] = [split_of("|".join(unit), config["seed"], config["fractions"]) for unit in units[columns].itertuples(index = False, name = None)]
    
    return stats.merge(units, on = columns, how = "left")

def validate_unit_split(stats, config):
    columns = UNIT_COLUMNS[config["split_unit"]]
    if stats["split"].isna().any():
        raise ValueError("some functions have no split")
    
    splits_per_unit = stats.groupby(columns)["split"].nunique()
    if(splits_per_unit > 1).any():
        raise ValueError(f"{int((splits_per_unit > 1).sum())} {config['split_unit']}(s) appear in more than 1 split")
    

def validate_day_split(config):
    """ 
        Function to Check whether the Day being selected from the Dataset are the one defined in the Config for splits
    """
    days = {name: set(values) for name, values in config["days"].items()}
    weekend = set(config["weekend_days"])
    train, val, test = days["train"], days["val"], days["test"]
    
    if len(train) + len(val) + len(test) != len(train | val | test):
        raise ValueError("a day is used in more than one split")
    
    if (train | val | test) != ALL_DAYS:
        raise ValueError("every day 1-14 must belong to exactly one split")
    
    if max(train) >= min(val | test):
        raise ValueError("validation and test splits must have days after train days")
    
    for name in ("val", "test"):
        if not (days[name] & weekend) or not (days[name] - weekend):
            raise ValueError(f"{name} days must contain at least one weekday and one weekend day")
        

def cell_mask(table, cell_name, config):
    """
        Rows of `table` (needs columns 'split' and 'day') that belong to one named cell.
    """
    cell = config["cells"][cell_name]
    return table["split"].eq(cell["functions"]) & table["day"].isin(config["days"][cell["days"]])


def group_shares(stats):
    """
        For each split, what share of its (labelled) functions falls in each group?
    """
    labelled = stats[stats["has_train"]]
    return pd.crosstab(labelled["group"], labelled["split"], normalize="columns")


def fingerprint(stats):
    """
        One short code for the whole assignment. Change any function's split and the code changes.
    """
    lines = sorted("|".join(map(str, row)) for row in stats[FUNC_KEYS + ["split"]].itertuples(index=False, name=None))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_or_freeze(stats, fingerprint_file):
    """
        First run: save the fingerprint (this freezes the split). Later runs: refuse to continue
        if the assignment is different, and never overwrite the saved file.
    """
    path = Path(fingerprint_file)
    new = fingerprint(stats)
    if not path.exists():
        path.write_text(new + "\n")
        return "frozen"
    if path.read_text().strip() != new:
        raise RuntimeError(
            f"The split no longer matches the frozen one in {path}. Do not regenerate it: "
            "find out what changed (seed, fractions, split_unit or the set of functions)."
        )
    return "verified"