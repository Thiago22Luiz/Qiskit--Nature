import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.opflow import MatrixOp


h_mo = np.array([
    [-1.250844, -0.0],
    [0.0, -0.479684]
])

g_mo = np.array([
    [
        [[ 0.309527, -0.004598],
         [-0.004598,  0.881317]],
        [[-0.004598,  0.020074],
         [ 0.020074, -0.022702]]
    ],
    [
        [[-0.004598,  0.020074],
         [ 0.020074, -0.022702]],
        [[ 0.881317, -0.022702],
         [-0.022702, 7.545265]]
    ]
])


energy_nuclear = 0.7135689  # Hartree (exemplo típico para H2)


n_orb = 2
n_spin_orb = 2 * n_orb
fer_op_dict = {}


for p in range(n_spin_orb):
    for q in range(n_spin_orb):
        if (p // n_orb) == (q // n_orb):
            i, j = p % n_orb, q % n_orb
            val = h_mo[i, j]
            if abs(val) > 1e-10:
                fer_op_dict[f"+_{p} -_{q}"] = val


for p in range(n_spin_orb):
    for q in range(n_spin_orb):
        for r in range(n_spin_orb):
            for s in range(n_spin_orb):
                if (p // n_orb) == (r // n_orb) and (q // n_orb) == (s // n_orb):
                    i, j, k, l = p % n_orb, q % n_orb, r % n_orb, s % n_orb
                    val = 0.5 * g_mo[i, j, k, l]
                    if abs(val) > 1e-10:
                        key = f"+_{p} +_{q} -_{s} -_{r}"
                        fer_op_dict[key] = fer_op_dict.get(key, 0) + val

fermion_op = FermionicOp(fer_op_dict, num_spin_orbitals=n_spin_orb)

mapper = JordanWignerMapper()
qubit_op = mapper.map(fermion_op)


ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=2, parameter_prefix='y')
optimizer = COBYLA(maxiter=200)
estimator = Estimator()

vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)


vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)


energy_total = vqe_result.eigenvalue.real + energy_nuclear

print("\nEnergia eletrônica (VQE):", vqe_result.eigenvalue.real)
print("Energia nuclear:", energy_nuclear)
print("Energia total (VQE + energia nuclear):", energy_total)
