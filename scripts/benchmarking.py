import numpy as np
from randquantkry.hamiltonian import build_hamiltonian
from randquantkry.benchmark import Benchmark
from randquantkry.post_processing import plot_runtime, plot_memory
from randquantkry.utils.logging_config import configure_logging

configure_logging(console_level='INFO')

n_qubits = 14
rng = np.random.default_rng(0)

terms = []
for i in range(n_qubits):
    terms.append(('X', i, float(rng.uniform(0.5, 1.5))))       # X field, every qubit
for i in range(n_qubits - 1):
    terms.append(('ZZ', i, float(rng.uniform(0.3, 0.8))))      # ZZ coupling, every neighboring pair

hamiltonian = build_hamiltonian(n_qubits, terms)
print(hamiltonian)
n_qubits = hamiltonian.num_qubits
initial_state = np.zeros(2**n_qubits, dtype=complex)
initial_state[0] = 1.0   # |0000...0>
print(f"n_qubits: {n_qubits}, initial_state: {initial_state}")

time = 1.0
krylov_dim=50
num_random_vectors=2*krylov_dim+1
seed = 67

benchmark1 = Benchmark(hamiltonian, initial_state, time, krylov_dim, num_random_vectors, seed = seed)

rt_results = Benchmark.benchmark_runtime(benchmark1, n_qubits_list=[ 20, 22, 24], methods=['randomized_krylov'])

plot_runtime(rt_results)

mem_results = Benchmark.benchmark_memory(benchmark1, n_qubits_list=[ 20, 22, 24], methods=['randomized_krylov'], empirical = True)

plot_memory(
    mem_results,
    ram_limits={'Typical PC RAM: 16GB': 16, 'Typical HPC RAM: 512GB': 512}, empirical = True
)