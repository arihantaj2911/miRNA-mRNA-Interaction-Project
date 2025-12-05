"""
feature_extraction.py

Feature extraction functions for miRNA–mRNA interaction prediction.

Assumes a dataset with at least these columns:
- miRNA_sequence
- mRNA_UTR_sequence

If your column names differ, update COLUMN_MIRNA and COLUMN_UTR.
"""

import pandas as pd

# === CHANGE HERE IF YOUR COLUMN NAMES ARE DIFFERENT ===
COLUMN_MIRNA = "miRNA"
COLUMN_UTR = "mRNA_gene"



def calculate_gc_content(seq: str) -> float:
    """
    Calculate GC content of a nucleotide sequence (DNA or RNA).
    Returns value between 0 and 1.
    """
    if not isinstance(seq, str) or len(seq) == 0:
        return 0.0

    seq = seq.upper()
    valid_bases = [b for b in seq if b in ("A", "T", "U", "G", "C")]
    if len(valid_bases) == 0:
        return 0.0

    gc_count = sum(1 for b in valid_bases if b in ("G", "C"))
    return gc_count / len(valid_bases)


def reverse_complement(seq: str) -> str:
    """
    Reverse complement for RNA/DNA sequences.
    """
    if not isinstance(seq, str):
        return ""
    seq = seq.upper()
    comp_map = {
        "A": "U", "U": "A",
        "T": "A",  # allow T as U
        "G": "C", "C": "G"
    }
    return "".join(comp_map.get(b, "N") for b in reversed(seq))


def seed_match_score(mirna_seq: str, utr_seq: str, seed_start: int = 1, seed_len: int = 7) -> int:
    """
    Simple seed match: take positions [seed_start : seed_start+seed_len] of miRNA,
    reverse complement them, and check if they appear in the UTR.
    Returns 1 if present, 0 otherwise.
    """
    if not isinstance(mirna_seq, str) or not isinstance(utr_seq, str):
        return 0

    if len(mirna_seq) < seed_start + seed_len:
        return 0

    # seed region (1-based index -> Python slicing)
    seed = mirna_seq[seed_start: seed_start + seed_len]
    seed_rc = reverse_complement(seed)
    utr_seq = utr_seq.upper()

    return 1 if seed_rc in utr_seq else 0


def utr_length(utr_seq: str) -> int:
    """
    Return the length of the UTR sequence.
    """
    if not isinstance(utr_seq, str):
        return 0
    return len(utr_seq)


def extract_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with miRNA_sequence and mRNA_UTR_sequence,
    return a new DataFrame with numeric features.
    """
    if COLUMN_MIRNA not in df.columns or COLUMN_UTR not in df.columns:
        raise ValueError(f"Expected columns '{COLUMN_MIRNA}' and '{COLUMN_UTR}' in dataframe.")

    features = {}
    features["seed_match"] = [
        seed_match_score(m, u)
        for m, u in zip(df[COLUMN_MIRNA], df[COLUMN_UTR])
    ]
    features["mirna_gc"] = [calculate_gc_content(m) for m in df[COLUMN_MIRNA]]
    features["utr_gc"] = [calculate_gc_content(u) for u in df[COLUMN_UTR]]
    features["utr_length"] = [utr_length(u) for u in df[COLUMN_UTR]]

    features_df = pd.DataFrame(features, index=df.index)
    return features_df


if __name__ == "__main__":
    print("feature_extraction.py loaded. This module is intended to be imported.")
