import numpy as np
from randquantkry.evolution import time_evolve_naive
from randquantkry.hamiltonian import build_hamiltonian, build_initial_state
from randquantkry.trotter import time_evolve_trotter

H = build_hamiltonian(3, [("XX", 0, 1.0), ("Z", 1, 0.5)])
psi0 = build_initial_state(3, 0)


def test_trotter_preserves_norm():
    state = time_evolve_trotter(H, psi0, time=1.0, num_steps=10, order=1)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-8)


def test_trotter_converges_to_exact_with_more_steps():
    """The core claim of Trotter: error should shrink as num_steps increases."""
    _, exact = time_evolve_naive(H, psi0, time=1.0)

    err_few_steps = np.linalg.norm(
        time_evolve_trotter(H, psi0, time=1.0, num_steps=2, order=1) - exact
    )
    err_many_steps = np.linalg.norm(
        time_evolve_trotter(H, psi0, time=1.0, num_steps=50, order=1) - exact
    )
    assert err_many_steps < err_few_steps


def test_trotter_is_actually_approximate_not_exact():
    """
    Regression test for the exact bug the docstring warns about: if someone
    ever 'simplifies' this function by removing qc.decompose(), Qiskit's
    Statevector.evolve() shortcut would silently return the EXACT e^{-iHt}
    result regardless of num_steps, defeating the whole point. This test
    catches that regression directly by checking that a LOW-order, FEW-step
    Trotter run has a meaningfully large error -- not near machine precision.
    """
    _, exact = time_evolve_naive(H, psi0, time=1.0)
    approx = time_evolve_trotter(H, psi0, time=1.0, num_steps=1, order=1)
    err = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    # A single first-order Trotter step should have REAL, non-trivial error --
    # if this comes back near 1e-15, the decomposition step got silently skipped.
    assert err > 1e-6, (
        f"Trotter error suspiciously small ({err}); check that qc.decompose() "
        "is still being called -- see function docstring."
    )


def test_trotter_higher_order_more_accurate_at_fixed_steps():
    """4th-order Suzuki should out-perform 1st-order Lie-Trotter at the same step count."""
    _, exact = time_evolve_naive(H, psi0, time=1.0)
    err_order1 = np.linalg.norm(
        time_evolve_trotter(H, psi0, time=1.0, num_steps=5, order=1) - exact
    )
    err_order4 = np.linalg.norm(
        time_evolve_trotter(H, psi0, time=1.0, num_steps=5, order=4) - exact
    )
    assert err_order4 < err_order1
