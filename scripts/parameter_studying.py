import numpy as np
import pandas as pd
from randquantkry.hamiltonian import build_hamiltonian
from randquantkry.benchmark import Benchmark
from randquantkry.post_processing import plot_parameter_study
from randquantkry.utils.logging_config import configure_logging

configure_logging(console_level='INFO')

n_qubits = 8
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

time = 2.0
krylov_dim=50
num_random_vectors=2*krylov_dim+1
seed = 67

benchmark1 = Benchmark(hamiltonian, initial_state, time, krylov_dim, num_random_vectors, seed = seed)

# easy test case

# rows = Benchmark.parameter_studyrows = benchmark1.parameter_study(
#     n_qubits_list=[6, 8],
#     krylov_dim_list=[10, 15],
#     s_ratio_list=[1.5, 2.0, 3.0],
#     time_list=[0.5, 1.5],
#     methods=("krylov", "randomized_krylov"),
#     seed=67,
#     )

# # 1. Runtime vs krylov_dim, comparing methods, at a fixed (n_qubits, time)
# data = rows.copy()
# data["method_label"] = data.apply(
#     lambda r: f"{r['method']}" if r["s_ratio"] is None or (isinstance(r["s_ratio"], float) and pd.isna(r["s_ratio"]))
#               else f"{r['method']} (s_ratio={r['s_ratio']})",
#     axis=1
# )
# plot_parameter_study(data, x="krylov_dim", y="runtime_sec",
#                       fixed={"n_qubits": 8, "time": 0.5}, hue="method_label")

# # 2. Accuracy vs s_ratio, randomized_krylov only, at a fixed (n_qubits, krylov_dim, time)
# randomized_only = rows[rows["method"] == "randomized_krylov"]
# plot_parameter_study(randomized_only, x="s_ratio", y="rel_error",
#                       fixed={"n_qubits": 8, "krylov_dim": 10, "time": 0.5},
#                       hue="krylov_dim")   # only one krylov_dim in the filtered slice -> one line

# # 3. Runtime vs n_qubits, comparing methods, at a fixed (krylov_dim, time)
# plot_parameter_study(data, x="n_qubits", y="runtime_sec",
#                       fixed={"krylov_dim": 10, "time": 0.5})

# print(rows[["n_qubits", "krylov_dim", "s_ratio", "method", "runtime_sec", "rel_error"]])

rows = benchmark1.parameter_study(
    n_qubits_list=[10, 14, 18, 22],
    krylov_dim_list=[20, 40, 80, 160], 
    s_ratio_list=[2.0, 3.0, 5.0],
    time_list=[0.5, 1.0, 2.0, 3.0],
    methods=("krylov", "randomized_krylov"),
    seed=67,
)
# 
# plotting

# runtime vs krylov_dim -- now shows 4 clean lines: krylov, and randomized_krylov at each s_ratio
plot_parameter_study(rows, x="krylov_dim", y="runtime_sec", fixed={"n_qubits": 14, "time": 1.0})

# accuracy vs s_ratio -- now correctly shows ONE clean line (randomized_krylov only;
# "krylov" is automatically dropped since it has no s_ratio, and hue is forced to "method"
# instead of the redundant "s_ratio" you had)
plot_parameter_study(rows, x="s_ratio", y="rel_error", fixed={"n_qubits": 14, "krylov_dim": 40, "time": 1.0})

# runtime vs n_qubits -- same 4-line fix as plot 1
plot_parameter_study(rows, x="n_qubits", y="runtime_sec", fixed={"krylov_dim": 40, "time": 1.0})