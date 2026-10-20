use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::optimizer::solver::Optimizer;
use oouccd_rs::TaskContext;
use roqollage::{circuit_to_image, RenderPragmas};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
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

    let molecule = PyScfDriver::compute(&config.chemistry)?;
    let mut optimizer = Optimizer::build(&molecule, &config.method);
    if std::env::args().any(|arg| arg == "--render-circuit") {
        let image = circuit_to_image(
            &optimizer.quantum_engine.ansatz_circuit,
            Some(2.0),
            RenderPragmas::None,
            None,
            Some(30),
            Some(3),
        )?;
        image.save(output.join("circuit.png"))?;
    }

    let mut logger = task.get_logger(&config);
    optimizer.run(&mut logger)?;

    Ok(())
}
