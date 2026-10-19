use crate::checkpoint::manager::Logger;
use crate::config::TaskConfig;
use clap::Parser;
use std::fs;
use std::path::Path;

/// Input/output paths for either a CLI task or a manually constructed task.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct TaskContext {
    #[arg(short, long, default_value = "config.json")]
    config_filepath: String,

    #[arg(short, long, env = "DB_FILEPATH", default_value = "results.db")]
    pub db_filepath: String,
}

impl TaskContext {
    /// Use explicit paths without reading CLI arguments or cloud environment variables.
    pub fn new(config_filepath: impl AsRef<Path>, db_filepath: impl AsRef<Path>) -> Self {
        Self {
            config_filepath: config_filepath.as_ref().to_string_lossy().into_owned(),
            db_filepath: db_filepath.as_ref().to_string_lossy().into_owned(),
        }
    }

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
