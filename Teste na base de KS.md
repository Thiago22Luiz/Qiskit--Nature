import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp

# Integrais de KS
h_ks = np.array([
    [-1.250844, -0.0],
    [-0.0, -0.479684]
])

g_ks = np.array([
    [[[ 2.513857, -1.476086],
      [-1.476086,  1.866952]],

     [[-1.476086,  1.005709],
      [ 1.005709, -1.476086]]],


    [[[-1.476086,  1.005709],
      [ 1.005709, -1.476086]],

     [[ 1.866952, -1.476086],
      [-1.476086,  2.513857]]]
])

energy_nuclear = 0.7135689  # Energia nuclear

n_orb = h_ks.shape[0]
n_spin_orb = 2 * n_orb  # Orbitais espaciais × 2 spins


h1_dict = {}
for p in range(n_spin_orb):
    for q in range(n_spin_orb):
        if (p % 2) == (q % 2):  # conserva spin
            i, j = p // 2, q // 2
            val = h_ks[i, j]
            if abs(val) > 1e-10:
                h1_dict[f"+_{p} -_{q}"] = val

h1 = FermionicOp(h1_dict, num_spin_orbitals=n_spin_orb)


h2_dict = {}
for p in range(n_spin_orb):
    for q in range(n_spin_orb):
        for r in range(n_spin_orb):
            for s in range(n_spin_orb):
                if (p % 2 == r % 2) and (q % 2 == s % 2):  # conserva spin
                    i, j, k, l = p // 2, q // 2, r // 2, s // 2
                    val = g_ks[i, j, k, l]
                    if abs(val) > 1e-10:
                        h2_dict[f"+_{p} +_{q} -_{s} -_{r}"] =  val

h2 = FermionicOp(h2_dict, num_spin_orbitals=n_spin_orb)


fermionic_op = h1 + h2


mapper = JordanWignerMapper()
qubit_op = mapper.map(fermionic_op)  # retorna um SparsePauliOp

 # Solver clássico (Alterar para VQE)
qubit_op_opflow = PauliSumOp(qubit_op)  # para compatibilidade com opflow
solver = NumPyMinimumEigensolver()
result = solver.compute_minimum_eigenvalue(qubit_op_opflow)

electronic_energy = result.eigenvalue.real
total_energy = electronic_energy + energy_nuclear


print("\nEnergia eletrônica (exata):", electronic_energy)
print("Energia nuclear:", energy_nuclear)
print("Energia total (eletrônica + nuclear):", total_energy)

