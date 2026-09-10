"""
Finite-shot, measurement-based LiH energy estimation at R = 1.6 Angstrom.

The ansatz state is simulated ideally, but every Hamiltonian expectation value
is estimated from bit strings produced by finite-shot Pauli measurements.  No
gate, decoherence, or readout noise is included.
"""

import argparse
import json
import math
import os

import numpy as np
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from pyscf import ao2mo, gto, scf


PAULI_MATRICES = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def apply_one_qubit_gate(state, gate, qubit):
    """Apply a one-qubit gate using the little-endian OpenFermion convention."""
    step = 1 << qubit
    block = step << 1
    updated = state.copy()
    for base in range(0, state.size, block):
        for offset in range(step):
            i0 = base + offset
            i1 = i0 + step
            a0, a1 = state[i0], state[i1]
            updated[i0] = gate[0, 0] * a0 + gate[0, 1] * a1
            updated[i1] = gate[1, 0] * a0 + gate[1, 1] * a1
    return updated


def apply_pauli_word(state, word):
    result = state
    for qubit, pauli in word:
        result = apply_one_qubit_gate(result, PAULI_MATRICES[pauli], qubit)
    return result


def hartree_fock_state(n_qubits, n_electrons):
    state = np.zeros(1 << n_qubits, dtype=complex)
    occupied = sum(1 << qubit for qubit in range(n_electrons))
    state[occupied] = 1.0
    return state


def ucc_doubles_state(n_qubits, n_electrons, theta):
    """Apply the project's spin-preserving UCCD excitation ordering.

    Each anti-Hermitian excitation generator G obeys G^3 = -G on the
    determinant subspace, so exp(theta G)|psi> is evaluated exactly as
    |psi> + sin(theta) G|psi> + (1-cos(theta)) G^2|psi>.
    """
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    excitations = []
    for i_pos, i in enumerate(occupied):
        for j in occupied[i_pos + 1 :]:
            for k_pos, k in enumerate(virtual):
                for l in virtual[k_pos + 1 :]:
                    if (i & 1) + (j & 1) == (k & 1) + (l & 1):
                        excitations.append(((k, l), (j, i)))

    if len(theta) != len(excitations):
        raise ValueError(f"Expected {len(excitations)} UCCD parameters, got {len(theta)}")

    state = hartree_fock_state(n_qubits, n_electrons)
    for angle, (creation, annihilation) in zip(theta, excitations):
        forward = fermionic_excitation(state, creation, annihilation)
        backward = fermionic_excitation(state, annihilation, creation)
        generator_state = forward - backward
        generator_squared_state = fermionic_excitation(generator_state, creation, annihilation) - fermionic_excitation(
            generator_state, annihilation, creation
        )
        state = state + math.sin(angle) * generator_state + (1.0 - math.cos(angle)) * generator_squared_state
    return state / np.linalg.norm(state)


def fermionic_excitation(state, creation, annihilation):
    """Apply a†...a to a state vector using Jordan-Wigner occupation signs."""
    output = np.zeros_like(state)
    for basis, amplitude in enumerate(state):
        if amplitude == 0.0:
            continue
        index = basis
        sign = 1.0
        valid = True
        for qubit in annihilation:
            if not (index & (1 << qubit)):
                valid = False
                break
            sign *= -1.0 if bin(index & ((1 << qubit) - 1)).count("1") % 2 else 1.0
            index &= ~(1 << qubit)
        if not valid:
            continue
        for qubit in creation:
            if index & (1 << qubit):
                valid = False
                break
            sign *= -1.0 if bin(index & ((1 << qubit) - 1)).count("1") % 2 else 1.0
            index |= 1 << qubit
        if valid:
            output[index] += sign * amplitude
    return output


def build_qubit_hamiltonian(r_angstrom):
    mol = gto.M(atom=f"Li 0 0 0; H 0 0 {r_angstrom}", basis="sto-3g", spin=0, charge=0, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor("int2e"), mf.mo_coeff), mol.nao)
    interaction = InteractionOperator(mol.energy_nuc(), h1, eri / 2.0)
    return jordan_wigner(get_fermion_operator(interaction)), mol.nelectron, mol.nao * 2


def qwc_groups(qubit_hamiltonian):
    """Greedily group Pauli terms compatible with one tensor-product basis."""
    groups = []
    for word, coefficient in qubit_hamiltonian.terms.items():
        coefficient = float(np.real_if_close(coefficient))
        placed = False
        for basis, terms in groups:
            if all(qubit not in basis or basis[qubit] == pauli for qubit, pauli in word):
                basis.update(dict(word))
                terms.append((word, coefficient))
                placed = True
                break
        if not placed:
            groups.append((dict(word), [(word, coefficient)]))
    return groups


def sample_group(state, basis, shots, rng):
    rotated = state
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
    s_dagger_h = np.array([[1.0, 1.0], [-1.0j, 1.0j]], dtype=complex) / math.sqrt(2.0)
    for qubit, pauli in basis.items():
        if pauli == "X":
            rotated = apply_one_qubit_gate(rotated, hadamard, qubit)
        elif pauli == "Y":
            rotated = apply_one_qubit_gate(rotated, s_dagger_h, qubit)

    outcomes = rng.choice(rotated.size, size=shots, p=np.abs(rotated) ** 2)
    return outcomes


def estimate_energy(state, groups, shots, rng):
    energy = 0.0
    variance = 0.0
    for basis, terms in groups:
        outcomes = sample_group(state, basis, shots, rng)
        for word, coefficient in terms:
            if not word:
                estimate = 1.0
                term_variance = 0.0
            else:
                parity = np.zeros(shots, dtype=np.uint8)
                for qubit, _ in word:
                    parity ^= ((outcomes >> qubit) & 1).astype(np.uint8)
                values = 1.0 - 2.0 * parity
                estimate = float(values.mean())
                term_variance = float(values.var(ddof=1) / shots)
            energy += coefficient * estimate
            variance += coefficient * coefficient * term_variance
    return energy, math.sqrt(variance)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Finite-shot LiH R=1.6 measurement simulator")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "lih_r160_shot_noise.json"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_r160_shot_noise.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    r_angstrom = float(config["chemistry"]["bond_length_angstrom"])
    shots = int(config["measurement"]["shots_per_group"])
    seeds = [int(seed) for seed in config["measurement"]["seeds"]]
    theta = np.asarray(config["ansatz"]["theta"], dtype=float)

    hamiltonian, n_electrons, n_qubits = build_qubit_hamiltonian(r_angstrom)
    state = ucc_doubles_state(n_qubits, n_electrons, theta)
    groups = qwc_groups(hamiltonian)
    exact_energy = sum(
        coefficient.real * np.vdot(state, apply_pauli_word(state, word)).real
        for word, coefficient in hamiltonian.terms.items()
    )

    trials = []
    for seed in seeds:
        estimate, standard_error = estimate_energy(state, groups, shots, np.random.default_rng(seed))
        trials.append({"seed": seed, "energy_ha": estimate, "estimated_standard_error_ha": standard_error})

    result = {
        "description": "Ideal UCCD-like state preparation with finite-shot Pauli measurements; no hardware noise.",
        "chemistry": {"bond_length_angstrom": r_angstrom, "basis": "sto-3g"},
        "measurement": {"shots_per_group": shots, "number_of_qwc_groups": len(groups), "number_of_pauli_terms": len(hamiltonian.terms)},
        "exact_energy_ha": exact_energy,
        "trials": trials,
        "sample_mean_energy_ha": float(np.mean([trial["energy_ha"] for trial in trials])),
        "sample_standard_deviation_ha": float(np.std([trial["energy_ha"] for trial in trials], ddof=1)) if len(trials) > 1 else 0.0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()