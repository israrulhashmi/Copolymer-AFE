"""Feature extraction from copolymer sequence strings."""
from collections import Counter
import numpy as np

def sequence_to_counts(seq, alphabet=None):
    """Simple featurizer: counts of each monomer in sequence.

    Args:
        seq (str): polymer sequence, e.g. "AABBC"
        alphabet (list|None): list of monomer symbols to include. If None, uses letters present.

    Returns:
        np.ndarray: counts vector (len = len(alphabet))
    """
    if alphabet is None:
        alphabet = sorted(set(seq))
    c = Counter(seq)
    return np.array([c[s] for s in alphabet], dtype=float)
