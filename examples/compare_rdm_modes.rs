use oouccd_rs::chemistry::engine::ClassicalEngine;
use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::config::{ChemistryConfig, RdmEvaluationConfig};
use oouccd_rs::quantum::engine::QuantumEngine;
use serde::Serialize;
use std::fs;

const OPTIMIZED_UCCD_THETA: [f64; 76] = [
    -0.0039636219080549928,
    -8.8817841970012523e-16,
    0.0,
    0.00024376620993749043,
    8.8817841970012523e-16,
    2.6645352591003757e-15,
    -0.0002431573455337066,
    -0.0018126049656621568,
    3.5527136788005009e-15,
    -2.6645352591003757e-15,
    -4.4408920985006262e-16,
    -2.2204460492503131e-15,
    -0.0018113330063758148,
    -2.2204460492503131e-15,
    1.3322676295501878e-15,
    -0.0010402711125934161,
    -8.8817841970012523e-16,
    -8.8817841970012523e-15,
    0.00045428130607927386,
    -7.5495165674510645e-15,
    -5.773159728050814e-15,
    -5.3290705182007514e-15,
    5.6645746915773287e-05,
    -4.4408920985006262e-16,
    1.5099033134902129e-14,
    -0.00029911595010245762,
    -5.773159728050814e-15,
    -8.8817841970012523e-16,
    0.00075350567423360104,
    -0.0029307405621494809,
    8.8817841970012523e-15,
    -6.6613381477509392e-15,
    -9.3258734068513149e-15,
    3.1086244689504383e-15,
    -0.0029307106964200358,
    -4.4408920985006262e-16,
    1.7763568394002505e-15,
    -0.0002170196367861088,
    -5.6584611126009321e-05,
    7.1054273576010019e-15,
    3.1086244689504383e-15,
    0.00075357653086749465,
    4.8849813083506888e-15,
    -4.8849813083506888e-15,
    -0.00029904033502781147,
    0.0029307319220621686,
    1.3322676295501878e-15,
    -1.3322676295501878e-15,
    4.4408920985006262e-16,
    -2.2204460492503131e-15,
    0.002930717781649772,
    1.3322676295501878e-15,
    -2.2204460492503131e-15,
    0.0002170161244023916,
    5.3290705182007514e-15,
    -1.3322676295501878e-15,
    0.0004544848463114981,
    8.8817841970012523e-16,
    3.5527136788005009e-15,
    -2.2204460492503131e-15,
    -0.026198748633459168,
    -2.2204460492503131e-15,
    -3.1086244689504383e-15,
    0.051870119628467659,
    -6.2172489379008766e-15,
    -2.0872192862952943e-14,
    -0.051939521941362088,
    -0.026056149721145783,
    -4.8849813083506888e-15,
    1.7763568394002505e-15,
    -6.2172489379008766e-15,
    -1.021405182655144e-14,
    -0.026065136852447601,
    -9.7699626167013776e-15,
    1.0658141036401503e-14,
    -0.11715748361845257,
];

#[derive(Serialize)]
struct Trial {
    seed: u64,
    energy_ha: f64,
    error_from_statevector_ha: f64,
}

#[derive(Serialize)]
struct Comparison {
    bond_length_angstrom: f64,
    basis: &'static str,
    ansatz: &'static str,
    shots_per_pauli_word: usize,
    statevector_energy_ha: f64,
    finite_shot_trials: Vec<Trial>,
    finite_shot_mean_energy_ha: f64,
    finite_shot_standard_deviation_ha: f64,
}

fn main() -> anyhow::Result<()> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "summary/results/lih_r160_rdm_mode_comparison.json".to_string());
    let shots = std::env::var("RDM_SHOTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1000);
    let seeds = [20260910, 20260911, 20260912, 20260913, 20260914];

    let chemistry = ChemistryConfig {
        geometry: "Li 0 0 0; H 0 0 1.60000".to_string(),
        basis: "sto-3g".to_string(),
        charge: 0,
        spin: 0,
        freeze_core: 0,
    };
    let context = PyScfDriver::compute(&chemistry)?;
    let classical = ClassicalEngine::from_context(&context);

    let exact_engine = QuantumEngine::new(
        context.n_orb,
        context.n_elec,
        false,
        RdmEvaluationConfig::Statevector {},
    );
    let (rdm1, rdm2) = exact_engine.get_rdm(&OPTIMIZED_UCCD_THETA);
    let exact_energy = classical.evaluate_energy(&rdm1, &rdm2);

    let mut trials = Vec::new();
    for seed in seeds {
        let sampled_engine = QuantumEngine::new(
            context.n_orb,
            context.n_elec,
            false,
            RdmEvaluationConfig::FiniteShot {
                shots,
                seed: Some(seed),
            },
        );
        let (rdm1, rdm2) = sampled_engine.get_rdm(&OPTIMIZED_UCCD_THETA);
        let energy = classical.evaluate_energy(&rdm1, &rdm2);
        trials.push(Trial {
            seed,
            energy_ha: energy,
            error_from_statevector_ha: energy - exact_energy,
        });
    }

    let mean = trials.iter().map(|trial| trial.energy_ha).sum::<f64>() / trials.len() as f64;
    let variance = trials
        .iter()
        .map(|trial| (trial.energy_ha - mean).powi(2))
        .sum::<f64>()
        / (trials.len() - 1) as f64;
    let comparison = Comparison {
        bond_length_angstrom: 1.6,
        basis: "sto-3g",
        ansatz: "UCCD(seq), optimized state from summary/results/lih_pes/task_3.db",
        shots_per_pauli_word: shots,
        statevector_energy_ha: exact_energy,
        finite_shot_trials: trials,
        finite_shot_mean_energy_ha: mean,
        finite_shot_standard_deviation_ha: variance.sqrt(),
    };

    if let Some(parent) = std::path::Path::new(&output).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &output,
        format!("{}\n", serde_json::to_string_pretty(&comparison)?),
    )?;
    println!("Saved: {}", fs::canonicalize(&output)?.display());
    Ok(())
}
