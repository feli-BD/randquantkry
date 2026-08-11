import pandas as pd


def test_parameter_study_returns_dataframe_with_expected_columns(small_hamiltonian, small_initial_state, tmp_path):
    from randquantkry.benchmark import Benchmark

    b = Benchmark(
        hamiltonian=small_hamiltonian,
        initial_state=small_initial_state,
        time=1.0,
        krylov_dim=8,
        num_random_vectors=None,
        methods=["krylov", "randomized_krylov"],
        seed=42,
        output_dir=str(tmp_path / "results"),
    )
    df, _ = b.parameter_study(
        n_qubits_list=[4],
        krylov_dim_list=[4, 8],
        s_ratio_list=[1.5, 2.0],
        time_list=[1.0],
        methods=("krylov", "randomized_krylov"),
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "n_qubits",
        "krylov_dim",
        "s_ratio",
        "time",
        "method",
        "runtime_sec",
        "memory_gb",
        "rel_error",
        "ground_truth",
    }
    assert expected_cols.issubset(df.columns)


def test_parameter_study_skips_krylov_dim_above_N(small_hamiltonian, small_initial_state, tmp_path):
    """n_qubits=4 -> N=16; krylov_dim=20 should be silently skipped, not crash."""
    from randquantkry.benchmark import Benchmark

    b = Benchmark(
        hamiltonian=small_hamiltonian,
        initial_state=small_initial_state,
        time=1.0,
        krylov_dim=8,
        num_random_vectors=None,
        methods=["krylov"],
        seed=42,
        output_dir=str(tmp_path / "results"),
    )
    df, _ = b.parameter_study(
        n_qubits_list=[4],
        krylov_dim_list=[8, 20],
        s_ratio_list=[1.5],
        time_list=[1.0],
        methods=("krylov",),
    )
    assert 20 not in df["krylov_dim"].values  # confirms the >= N skip logic actually fires


def test_parameter_study_rel_error_present_when_naive_feasible(small_hamiltonian, small_initial_state, tmp_path):
    """n_qubits <= MAX_NAIVE_QUBITS should give exact ground truth, so rel_error should be a real number, not None."""
    from randquantkry.benchmark import Benchmark

    b = Benchmark(
        hamiltonian=small_hamiltonian,
        initial_state=small_initial_state,
        time=1.0,
        krylov_dim=8,
        num_random_vectors=None,
        methods=["krylov"],
        seed=42,
        output_dir=str(tmp_path / "results"),
    )
    df, _ = b.parameter_study(
        n_qubits_list=[4],
        krylov_dim_list=[8],
        s_ratio_list=[1.5],
        time_list=[1.0],
        methods=("krylov",),
    )
    assert df["rel_error"].notna().all()
    assert (df["rel_error"] < 1.0).all()  # sanity: error shouldn't be enormous for a reasonable krylov_dim
