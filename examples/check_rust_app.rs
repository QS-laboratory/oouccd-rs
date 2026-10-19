use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::optimizer::solver::Optimizer;
use oouccd_rs::TaskContext;
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
    let mut optimizer = Optimizer::build(&molecule, &config.method);
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
