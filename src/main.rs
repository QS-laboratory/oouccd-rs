use oouccd_rs::checkpoint::manager::Logger;
use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::config::TaskConfig;
use oouccd_rs::optimizer::solver::Optimizer;
use oouccd_rs::TaskContext;

fn main() -> anyhow::Result<()> {
    let task_idx = std::env::var("BATCH_TASK_INDEX").unwrap_or_else(|_| "local".to_string());
    eprintln!("[oouccd] task={task_idx} | starting");

    let env_config = TaskContext::build();
    let config = env_config.get_config();
    let mut logger = env_config.get_logger(&config);

    let result = run(&config, &mut logger);
    match &result {
        Ok(()) => eprintln!("[oouccd] task={task_idx} | finished successfully"),
        Err(e) => eprintln!("[oouccd] task={task_idx} | failed: {e}"),
    }
    result
}

fn run(config: &TaskConfig, logger: &mut Logger) -> anyhow::Result<()> {
    eprintln!(
        "[pyscf] computing | geometry='{}' basis='{}'",
        config.chemistry.geometry, config.chemistry.basis,
    );
    let ctx = PyScfDriver::compute(&config.chemistry)
        .map_err(|e| anyhow::anyhow!("PySCF Error: {}", e))?;
    eprintln!("[pyscf] done | nao={} nelec={}", ctx.n_orb, ctx.n_elec);

    let mut optimizer = Optimizer::build(&ctx, &config.method);

    optimizer.run(logger)?;

    Ok(())
}
