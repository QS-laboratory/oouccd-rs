use anyhow::Result;
use ndarray::Array1;

use crate::config::TaskConfig;
use crate::checkpoint::gcs::GcsHandler;
use crate::checkpoint::sqlite::SqliteHandler;

pub enum LoggerMessage {
    RegisterParameters {
        ucc_names: Vec<String>,
        oo_names: Vec<String>,
    },
    SaveIteration {
        macro_idx: usize,
        micro_steps_taken: usize,
        energy: f64,
        converged: bool,
        ucc_params: Array1<f64>,
        is_completed: bool,
    },
}

pub struct Logger {
    sender: Option<std::sync::mpsc::SyncSender<LoggerMessage>>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl Logger {
    pub fn build(
        config: &TaskConfig,
        db_uri: &str,
    ) -> Result<Self> {
        let is_cloud = db_uri.starts_with("gs://");
        let db_filename = if is_cloud {
            "results.db".to_string()
        } else {
            db_uri.to_string()
        };

        let gcs = if is_cloud {

            let path_without_scheme = &db_uri[5..];
            let (bucket_name, target_gcs_path) = match path_without_scheme.find('/') {
                Some(idx) => (&path_without_scheme[..idx], &path_without_scheme[idx + 1..]),
                None => (path_without_scheme, ""),
            };
            Some(GcsHandler::new(bucket_name, &db_filename, target_gcs_path))
        } else {
            None
        };

        let mut sqlite = SqliteHandler::new(&db_filename, config)?;

        let (tx, rx) = std::sync::mpsc::sync_channel::<LoggerMessage>(100);

        let handle = std::thread::spawn(move || {
            for payload in rx {
                match payload {
                    LoggerMessage::RegisterParameters {
                        ucc_names,
                        oo_names,
                    } => {
                        let _ = sqlite.register_parameters(&ucc_names, &oo_names);
                    }
                    LoggerMessage::SaveIteration {
                        macro_idx,
                        micro_steps_taken,
                        energy,
                        converged,
                        ucc_params,
                        is_completed,
                    } => {
                        let _ = sqlite.save_iteration(
                            macro_idx,
                            micro_steps_taken,
                            energy,
                            converged,
                            &ucc_params,
                            is_completed,
                        );

                        if let Some(ref g) = gcs {
                            let _ = g.save_checkpoint(is_completed);
                        }

                    }
                }
            }
        });

        Ok(Self {
            sender: Some(tx),
            thread_handle: Some(handle),
        })
    }


    pub fn register_parameters(&self, ucc_names: Vec<String>, oo_names: Vec<String>) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(LoggerMessage::RegisterParameters { ucc_names, oo_names });
        }
    }

    pub fn save_iteration(
        &self,
        macro_idx: usize,
        micro_steps_taken: usize,
        energy: f64,
        converged: bool,
        ucc_params: &Array1<f64>,
        is_completed: bool,
    ) {
        if let Some(sender) = &self.sender {
            let payload = LoggerMessage::SaveIteration {
                macro_idx,
                micro_steps_taken,
                energy,
                converged,
                ucc_params: ucc_params.clone(),
                is_completed,
            };
            let _ = sender.send(payload);
        }
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}
