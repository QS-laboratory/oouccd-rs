use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::optimizer::solver::Optimizer;
use oouccd_rs::TaskContext;
use roqollage::{circuit_to_image, RenderPragmas};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // PyO3 embeds Python, so expose the scientific environment's packages.
    prepare_python(root)?;

    // A fresh directory on every run keeps previous calculation histories intact.
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let output = root
        .join("local_runs")
        .join(format!("manual_h2_{timestamp}"));
    std::fs::create_dir_all(&output)?;

    // Unlike TaskContext::build(), new() uses these paths directly.
    let task = TaskContext::new(
        root.join("examples/config_h2_manual.json"),
        output.join("results.db"),
    );
    let config = task.get_config();
    println!("Geometry: {}", config.chemistry.geometry);
    println!("Result DB: {}", task.db_filepath);

    println!("Chemistry: {:?}", &config.chemistry);

    let molecule = PyScfDriver::compute(&config.chemistry)?;

    println!("Molecule:");
    println!("nelec: {}, norb: {}", molecule.n_elec, molecule.n_orb);
    println!("nuc repulsion: {}", molecule.nuclear_repulsion);
    println!("h1: {:?}", molecule.h1.shape());
    println!("h2: {:?}", molecule.h2.shape());

    let mut optimizer = Optimizer::build(&molecule, &config.method)?;

    let image = circuit_to_image(
        &optimizer.quantum_engine.ansatz_circuit,
        Some(2.0),
        RenderPragmas::None,
        None,
        Some(30),
        Some(3),
    )?;

    image.save("circuit.png")?;
    println!("excitations: {:?}", optimizer.quantum_engine.excitations);

    let dummy_theta = &[1.0, 2.0, 3.0];
    let amp = optimizer.quantum_engine.get_state_vector(dummy_theta);
    println!("Prob. Amp.: {:?}", amp);
    let (h1, h2) = optimizer.quantum_engine.get_rdm(dummy_theta);
    println!("h1: {:?}", h1);

    let mut logger = task.get_logger(&config);
    optimizer.run(&mut logger)?;

    Ok(())
}

fn prepare_python(root: &Path) -> anyhow::Result<()> {
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        std::env::set_var(name, "1");
    }
    let python = root.join(".paper-env/venv/bin/python");
    let output = Command::new(&python)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ])
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "Failed to inspect {}",
        python.display()
    );
    std::env::set_var("PYTHONPATH", String::from_utf8(output.stdout)?.trim());
    Ok(())
}
