import pandas as pd
from randquantkry.post_processing import (
    _df_to_results,
    _filtered_xy,
    _sort_labels,
)


def test_filtered_xy_drops_none_values():
    n_qubits = [2, 4, 6, 8]
    values = [1.0, None, 3.0, None]
    xs, ys = _filtered_xy(n_qubits, values)
    assert xs == [2, 6]
    assert ys == [1.0, 3.0]


def test_filtered_xy_all_none_returns_empty():
    xs, ys = _filtered_xy([2, 4], [None, None])
    assert xs == [] and ys == []


def test_sort_labels_follows_method_order():
    results = {
        "randomized_krylov": {"_base_method": "randomized_krylov"},
        "naive": {"_base_method": "naive"},
        "krylov": {"_base_method": "krylov"},
    }
    assert _sort_labels(results) == ["naive", "krylov", "randomized_krylov"]


def test_sort_labels_unknown_method_goes_last():
    results = {
        "mystery_method": {"_base_method": "mystery_method"},
        "naive": {"_base_method": "naive"},
    }
    sorted_labels = _sort_labels(results)
    assert sorted_labels[-1] == "mystery_method"


def test_df_to_results_groups_by_method():
    df = pd.DataFrame(
        {
            "n_qubits": [4, 6, 4, 6],
            "method": ["krylov", "krylov", "naive", "naive"],
            "memory_gb": [1.0, 2.0, 0.5, 1.5],
        }
    )
    results = _df_to_results(df, y_col="memory_gb")
    assert set(results.keys()) == {"krylov", "naive"}
    assert results["krylov"]["n_qubits"] == [4, 6]


def test_df_to_results_s_ratio_label_when_present():
    df = pd.DataFrame(
        {
            "n_qubits": [4, 4],
            "method": ["randomized_krylov", "randomized_krylov"],
            "s_ratio": [1.5, 2.0],
            "memory_gb": [1.0, 1.2],
        }
    )
    results = _df_to_results(df, y_col="memory_gb")
    assert any("s_ratio=1.5" in label for label in results)
    assert any("s_ratio=2.0" in label for label in results)
