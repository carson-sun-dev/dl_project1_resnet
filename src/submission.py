"""Validate the competition CSV without reading or estimating hidden labels."""
import argparse
from .data import read_csv


def validate_submission(path, sample_path):
    rows = read_csv(path, ["id", "label"])
    sample = read_csv(sample_path, ["id", "label"])
    if [r["id"] for r in rows] != [r["id"] for r in sample]:
        raise ValueError("Prediction row count or ID order differs from sample_submission.csv")
    if any(row["label"] not in {str(i) for i in range(10)} for row in rows):
        raise ValueError("Every label must be an integer 0..9")
    return {"rows": len(rows), "columns": ["id", "label"], "valid": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True)
    p.add_argument("--sample", default="data/sample_submission.csv")
    a = p.parse_args()
    print(validate_submission(a.file, a.sample))
