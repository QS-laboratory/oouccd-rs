"""Joint physical RDM reconstruction from existing independent-Pauli counts.

No exact expectations enter reconstruction. DQG is a necessary-condition
relaxation; sector_mle is an exponential-size, small-system reference.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse


def array(value):
    return np.asarray(value["data"], dtype=float).reshape(value["dim"])


def plans_matrix(plans, width):
    rows, columns, values = [], [], []
    for row, plan in enumerate(plans):
        for col, coefficient in plan["terms"]:
            rows.append(row); columns.append(col); values.append(coefficient)
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(plans), width))


def dictionaries_matrix(rows, width):
    return plans_matrix([{"terms": list(row.items())} for row in rows], width)


def add(target, source, scale=1):
    for index, coefficient in source.items():
        target[index] = target.get(index, 0.0) + scale * coefficient


def spin_metric_maps(snapshot):
    """Derive Q and G by CAR in the same unordered-pair convention as D."""
    spin = snapshot["spin_plans"]
    n = spin["n_spin"]; pairs = [tuple(pair) for pair in spin["pairs"]]
    lookup = {pair: i for i, pair in enumerate(pairs)}
    d1 = [dict(p["terms"]) for p in spin["one_body"]]
    d2 = [dict(p["terms"]) for p in spin["pair_body"]]
    identity = next(i for i, w in enumerate(snapshot["plan"]["pauli_words"])
                    if not (w["x_mask"] | w["y_mask"] | w["z_mask"]))

    def gamma(i, j): return d1[i*n+j]

    def particle(i, j, k, l):
        if i == j or k == l: return {}
        sign = (1 if i < j else -1) * (1 if k < l else -1)
        index = lookup[tuple(sorted((i, j)))]*len(pairs) + lookup[tuple(sorted((k, l)))]
        return {key: sign*value for key, value in d2[index].items()}

    q_rows = []
    for i, j in pairs:
        for k, l in pairs:
            row = {identity: float((i == k and j == l) - (i == l and j == k))}
            add(row, gamma(k, i), -(j == l)); add(row, gamma(k, j), i == l)
            add(row, gamma(l, i), j == k); add(row, gamma(l, j), -(i == k))
            add(row, particle(k, l, i, j))
            q_rows.append(row)
    g_rows = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    row = {}
                    add(row, gamma(i, k), j == l)
                    add(row, particle(i, l, k, j), -1)
                    g_rows.append(row)
    width = len(snapshot["plan"]["pauli_words"])
    return {"gamma": dictionaries_matrix(d1, width), "D": dictionaries_matrix(d2, width),
            "Q": dictionaries_matrix(q_rows, width), "G": dictionaries_matrix(g_rows, width)}


def sector_map(snapshot):
    """Projected Pauli operators in a real fixed-(Nalpha,Nbeta) Fock basis."""
    n = snapshot["n_spatial"]; electrons = snapshot["n_elec"]
    na, nb = (electrons+1)//2, electrons//2
    basis = [sum(1 << (2*i) for i in up) + sum(1 << (2*j+1) for j in down)
             for up in itertools.combinations(range(n), na)
             for down in itertools.combinations(range(n), nb)]
    lookup = {mask: index for index, mask in enumerate(basis)}
    rows, columns, values = [], [], []
    for index, word in enumerate(snapshot["plan"]["pauli_words"]):
        flip = word["x_mask"] | word["y_mask"]
        phase_mask = word["y_mask"] | word["z_mask"]
        y_count = word["y_mask"].bit_count()
        if y_count % 2:
            raise ValueError("Real-sector reference cannot silently discard imaginary Pauli operators")
        phase = (-1)**(y_count//2)
        for row, mask in enumerate(basis):
            column = lookup.get(mask ^ flip)
            if column is not None:
                rows.append(index); columns.append(row*len(basis)+column)
                values.append(phase * (-1)**((mask & phase_mask).bit_count()))
    return sparse.csr_matrix((values, (rows, columns)),
                             shape=(len(snapshot["plan"]["pauli_words"]), len(basis)**2)), len(basis)


class Reconstruction:
    def __init__(self, snapshot, method, solver="SCS", tolerance=1e-6, max_iterations=20000):
        import cvxpy as cp
        self.cp, self.snapshot, self.method = cp, snapshot, method
        self.solver, self.tolerance, self.max_iterations = solver, tolerance, max_iterations
        self.width = len(snapshot["plan"]["pauli_words"])
        self.spatial = plans_matrix(snapshot["plan"]["one_body"] + snapshot["plan"]["two_body"], self.width)
        self.identity = np.array([not (w["x_mask"] | w["y_mask"] | w["z_mask"])
                                  for w in snapshot["plan"]["pauli_words"]])
        self.observed = np.flatnonzero(~self.identity)
        self.f = np.asarray(snapshot["constraints"]["rows"])
        self.target = np.asarray(snapshot["constraints"]["targets"])
        n = snapshot["n_spatial"]
        self.metrics = None; self.rho = None
        constraints = []
        if method == "sector_mle":
            projection, dimension = sector_map(snapshot)
            self.rho = cp.Variable((dimension, dimension), symmetric=True)
            self.mu = projection @ cp.vec(self.rho, order="C")
            constraints += [self.rho >> 0, cp.trace(self.rho) == 1]
            self.sector_dimension = dimension
        else:
            self.mu = cp.Variable(self.width)
            constraints += [self.mu[self.identity] == 1, self.mu >= -1, self.mu <= 1,
                            self.f @ self.mu == self.target]
            if method == "spatial_psd":
                one = cp.reshape(self.spatial[:n*n] @ self.mu, (n, n), order="C")
                two = cp.reshape(self.spatial[n*n:] @ self.mu, (n*n, n*n), order="C")
                constraints += [one >> 0, 2*np.eye(n)-one >> 0, two >> 0]
            elif method in ("dqg", "mle"):
                self.metrics = spin_metric_maps(snapshot)
                spin_n = 2*n; pair_n = len(snapshot["spin_plans"]["pairs"])
                gamma = cp.reshape(self.metrics["gamma"] @ self.mu, (spin_n, spin_n), order="C")
                particle = cp.reshape(self.metrics["D"] @ self.mu, (pair_n, pair_n), order="C")
                hole = cp.reshape(self.metrics["Q"] @ self.mu, (pair_n, pair_n), order="C")
                ph = cp.reshape(self.metrics["G"] @ self.mu, (spin_n**2, spin_n**2), order="C")
                na, nb = (snapshot["n_elec"]+1)//2, snapshot["n_elec"]//2
                aa = [i for i, (p, q) in enumerate(snapshot["spin_plans"]["pairs"]) if p%2 == q%2 == 0]
                bb = [i for i, (p, q) in enumerate(snapshot["spin_plans"]["pairs"]) if p%2 == q%2 == 1]
                constraints += [gamma >> 0, np.eye(spin_n)-gamma >> 0,
                                particle >> 0, hole >> 0, ph >> 0,
                                cp.sum(cp.diag(gamma)[::2]) == na,
                                cp.sum(cp.diag(particle)[aa]) == na*(na-1)/2,
                                cp.sum(cp.diag(particle)[bb]) == nb*(nb-1)/2]
            else:
                raise ValueError(f"Unknown reconstruction method: {method}")
        self.k = cp.Parameter(len(self.observed), nonneg=True)
        self.failures = cp.Parameter(len(self.observed), nonneg=True)
        if method in ("mle", "sector_mle"):
            probability = (1+self.mu[self.observed])/2
            objective = -cp.sum(cp.multiply(self.k, cp.log(probability))) \
                        -cp.sum(cp.multiply(self.failures, cp.log(1-probability)))
        else:
            self.weights = cp.Parameter(len(self.observed), nonneg=True)
            self.center = cp.Parameter(len(self.observed))
            objective = cp.sum_squares(cp.multiply(self.weights, self.mu[self.observed])-self.center)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve(self, counts):
        start = time.perf_counter()
        shots = counts["shots"]; k = np.asarray(counts["plus_counts"], dtype=float)
        if len(k) != self.width or shots < 1 or np.any(k < 0) or np.any(k > shots):
            raise ValueError("Invalid independent-Pauli counts")
        if np.any(k[self.identity] != shots):
            raise ValueError("Identity counts must be exact")
        raw = 2*k/shots-1
        self.k.value = k[self.observed]/shots
        self.failures.value = (shots-k[self.observed])/shots
        if self.method not in ("mle", "sector_mle"):
            a, b = k[self.observed]+.5, shots-k[self.observed]+.5
            variances = 4*a*b/((a+b)**2*(a+b+1))
            weights = np.sqrt(np.median(variances)/variances)
            self.weights.value = weights; self.center.value = weights*raw[self.observed]
        options = {"verbose": False, "warm_start": True}
        if self.solver == "SCS":
            options.update(eps=self.tolerance, max_iters=self.max_iterations)
        else:
            options.update(tol_gap_abs=self.tolerance, tol_feas=self.tolerance,
                           tol_gap_rel=self.tolerance, max_iter=min(1000, self.max_iterations))
        self.problem.solve(solver=self.solver, **options)
        if self.problem.status not in ("optimal", "optimal_inaccurate") or self.mu.value is None:
            raise RuntimeError(f"Reconstruction failed: {self.problem.status}")
        mu = np.asarray(self.mu.value).ravel()
        diagnostics = self.validate(mu)
        if not np.all(np.isfinite(mu)) or diagnostics["max_constraint_residual"] > 5e-5 \
                or diagnostics["max_pauli_bound_violation"] > 5e-5 \
                or min(diagnostics["minimum_eigenvalues"].values(), default=0) < -5e-5:
            raise RuntimeError(f"Infeasible numerical reconstruction: {diagnostics}")
        return {"method": self.method, "status": self.problem.status,
                "seconds": time.perf_counter()-start, "means": mu.tolist(),
                "validation": diagnostics, "solver": self.solver,
                "solver_iterations": self.problem.solver_stats.num_iters,
                "solver_seconds": self.problem.solver_stats.solve_time}

    def validate(self, mu):
        n = self.snapshot["n_spatial"]
        spatial = np.asarray(self.spatial @ mu).ravel()
        gamma = spatial[:n*n].reshape(n, n); two = spatial[n*n:].reshape(n*n, n*n)
        eig = {"spatial_gamma": float(np.linalg.eigvalsh(gamma).min()),
               "spatial_hole": float(np.linalg.eigvalsh(2*np.eye(n)-gamma).min()),
               "spatial_pair": float(np.linalg.eigvalsh(two).min())}
        if self.metrics:
            for name, matrix in self.metrics.items():
                dim = int(round(np.sqrt(matrix.shape[0])))
                value = np.asarray(matrix @ mu).reshape(dim, dim)
                eig[name] = float(np.linalg.eigvalsh((value+value.T)/2).min())
        if self.rho is not None:
            eig["sector_density"] = float(np.linalg.eigvalsh(self.rho.value).min())
        return {"max_constraint_residual": float(np.max(np.abs(self.f @ mu-self.target))),
                "max_pauli_bound_violation": float(max(0, np.max(np.abs(mu))-1)),
                "minimum_eigenvalues": eig}


def score(snapshot, means):
    width = len(snapshot["plan"]["pauli_words"])
    matrix = plans_matrix(snapshot["plan"]["one_body"] + snapshot["plan"]["two_body"], width)
    values = np.asarray(matrix @ np.asarray(means)).ravel()
    exact = np.asarray(matrix @ np.asarray(snapshot["exact_means"])).ravel()
    h = np.r_[array(snapshot["h1"]).ravel(), array(snapshot["h2"]).ravel()]
    return {"rdm_squared_error": float(np.sum((values-exact)**2)),
            "energy_error_ha": float(h @ (values-exact))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--method", choices=["spatial_psd", "dqg", "mle", "sector_mle"], required=True)
    parser.add_argument("--solver", choices=["SCS", "CLARABEL"], default="SCS")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError("Refusing to overwrite reconstruction result")
    snapshot = json.loads(args.snapshot.read_text()); counts = json.loads(args.counts.read_text())
    counts = counts.get("counts", counts)
    model = Reconstruction(snapshot, args.method, args.solver)
    result = model.solve(counts)
    result["score"] = score(snapshot, result["means"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
