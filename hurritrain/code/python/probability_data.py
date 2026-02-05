"""
Load KDE probability data (px, py) from pickle files.
"""

import os
import pickle


def load_probability_data(
    probability_data_dir: str = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data",
    px_filename: str = "eof_kde_h500_modes3.pkl",
    py_filename: str = "kde_pdf_PRESsfc.pkl",
) -> tuple[dict, dict]:
    """
    Load the KDE probability density function files for px and py.

    Args:
        probability_data_dir: Directory containing the .pkl files.
        px_filename: Filename for px (EOF KDE, default: "eof_kde_h500_modes3.pkl").
        py_filename: Filename for py (variable KDE, default: "kde_pdf_PRESsfc.pkl").

    Returns:
        tuple: (px, py)
            - px: Dictionary containing 'kde' (gaussian_kde object) and 'eof_results' (dict)
            - py: Dictionary containing 'kde' (gaussian_kde object) and 'mean_values' (np.ndarray)

    Example:
        px, py = load_probability_data()
        px_kde = px['kde']
        px_eof_results = px['eof_results']
        py_kde = py['kde']
        py_mean_values = py['mean_values']
    """
    px_path = os.path.join(probability_data_dir, px_filename)
    py_path = os.path.join(probability_data_dir, py_filename)

    if not os.path.exists(px_path):
        raise FileNotFoundError(
            f"px file not found: {px_path}\n"
            f"Please run px_x.py to generate this file first."
        )
    if not os.path.exists(py_path):
        raise FileNotFoundError(
            f"py file not found: {py_path}\n"
            f"Please run py_y.py to generate this file first."
        )

    print(f"Loading px from {px_path}...")
    with open(px_path, "rb") as f:
        px = pickle.load(f)

    print(f"Loading py from {py_path}...")
    with open(py_path, "rb") as f:
        py = pickle.load(f)

    print("Probability data loaded successfully!")
    print(f"  px keys: {list(px.keys())}")
    print(f"  py keys: {list(py.keys())}")

    return px, py
