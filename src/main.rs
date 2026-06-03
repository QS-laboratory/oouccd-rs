use clap::Parser;
use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::checkpoint::manager::Logger;
use oouccd_rs::config::TaskConfig;
use oouccd_rs::optimizer::solver::Optimizer;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct TaskContext {
    #[arg(short, long, default_value = "config.json")]
    config_filepath: String,

    #[arg(short, long, env = "DB_FILEPATH", default_value = "results.db")]
    pub db_filepath: String,
}

impl TaskContext {
    pub fn build() -> Self {
        let mut config = Self::parse();
        if let Ok(gcs_path) = std::env::var("GCS_PATH") {
            let task_idx = std::env::var("BATCH_TASK_INDEX").unwrap_or_else(|_| "0".to_string());
            let gcs_path = gcs_path.trim_end_matches('/');
            config.db_filepath = format!("{}/task_{}/results.db", gcs_path, task_idx);
        }
        config
    }

    pub fn get_config(&self) -> TaskConfig {
        let contents = fs::read_to_string(&self.config_filepath)
            .unwrap_or_else(|_| panic!("Failed to read config file: {}", self.config_filepath));
        serde_json::from_str(&contents).expect("Failed to parse TaskConfig from JSON")
    }

    pub fn get_logger(&self, config: &TaskConfig) -> Logger {
        Logger::build(config, &self.db_filepath).expect("Failed to build Logger")
    }
}

fn main() -> anyhow::Result<()> {
    let task_idx = std::env::var("BATCH_TASK_INDEX").unwrap_or_else(|_| "local".to_string());
    eprintln!("[oouccd] task={task_idx} | starting");

    let env_config = TaskContext::build();
    let config = env_config.get_config();
    let mut logger = env_config.get_logger(&config);

    let result = run(&config, &mut logger);
    match &result {
        Ok(()) => eprintln!("[oouccd] task={task_idx} | finished successfully"),
        Err(e)  => eprintln!("[oouccd] task={task_idx} | failed: {e}"),
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
