use crate::config::TaskConfig;

use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use rusqlite::{params, Connection, OptionalExtension, Result};
use std::io::Cursor;

/// Python uses a Numpy npy blob formatting. This function acts identically to Python's `io.BytesIO() + np.save()` mapping
pub fn array_to_blob(arr: &Array1<f64>) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    arr.write_npy(&mut cursor)
        .expect("Failed to serialize array to NPY format");
    cursor.into_inner()
}

pub struct SqliteHandler {
    conditions_json: String,
    conn: Connection,
}

impl SqliteHandler {
    pub fn new(db_path: &str, task: &TaskConfig) -> Result<Self> {
        let conditions = task.generate_conditions();
        let conn = Connection::open(db_path)?;

        let conditions_json =
            serde_json::to_string(&conditions).expect("Failed to serialize conditions to JSON");

        // Runs Table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conditions_json TEXT,
                final_energy REAL,
                final_theta BLOB,
                final_kappa BLOB,
                is_completed BOOLEAN DEFAULT 0
            )",
            [],
        )?;

        // Migration: Add is_completed for backward compatibility with existing databases
        let _ = conn.execute(
            "ALTER TABLE runs ADD COLUMN is_completed BOOLEAN DEFAULT 0",
            [],
        );

        // Parameter Names Table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS parameter_names (
                run_id INTEGER,
                param_type TEXT,
                param_index INTEGER,
                name TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )",
            [],
        )?;

        // Macro Iterations Table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS macro_iterations (
                run_id INTEGER,
                macro_idx INTEGER,
                micro_steps_taken INTEGER,
                energy REAL,
                converged BOOLEAN,
                ucc_params BLOB,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )",
            [],
        )?;

        // Index
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conditions ON runs(conditions_json)",
            [],
        )?;

        Ok(SqliteHandler {
            conditions_json,
            conn,
        })
    }

    pub fn register_parameters(&mut self, ucc_names: &[String], oo_names: &[String]) -> Result<()> {
        let tx = self.conn.transaction()?;
        let row_id_opt: Option<i64> = tx
            .query_row(
                "SELECT id FROM runs WHERE conditions_json = ?",
                params![self.conditions_json],
                |row| row.get(0),
            )
            .optional()?;

        let run_id = match row_id_opt {
            Some(id) => id,
            None => {
                tx.execute(
                    "INSERT INTO runs (conditions_json, is_completed) VALUES (?, 0)",
                    params![self.conditions_json],
                )?;
                tx.last_insert_rowid()
            }
        };

        // Check if parameters are already registered
        let count: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM parameter_names WHERE run_id = ?",
                params![run_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if count == 0 {
            let mut stmt = tx.prepare("INSERT INTO parameter_names (run_id, param_type, param_index, name) VALUES (?, ?, ?, ?)")?;
            for (i, name) in ucc_names.iter().enumerate() {
                stmt.execute(params![run_id, "ucc", i as i64, name])?;
            }
            for (i, name) in oo_names.iter().enumerate() {
                stmt.execute(params![run_id, "oo", i as i64, name])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    pub fn save_iteration(
        &mut self,
        macro_idx: usize,
        micro_steps_taken: usize,
        energy: f64,
        converged: bool,
        ucc_params: &Array1<f64>,
        is_completed: bool,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;

        let theta_blob = array_to_blob(ucc_params);

        let row_id_opt: Option<i64> = tx
            .query_row(
                "SELECT id FROM runs WHERE conditions_json = ?",
                params![self.conditions_json],
                |row| row.get(0),
            )
            .optional()?;

        let run_id = match row_id_opt {
            Some(id) => {
                tx.execute(
                    "UPDATE runs SET final_energy = ?, final_theta = ?, is_completed = ? WHERE id = ?",
                    params![energy, theta_blob, is_completed, id],
                )?;
                id
            }
            None => {
                tx.execute(
                    "INSERT INTO runs (conditions_json, final_energy, final_theta, is_completed) VALUES (?, ?, ?, ?)",
                    params![self.conditions_json, energy, theta_blob, is_completed],
                )?;
                tx.last_insert_rowid()
            }
        };

        {
            let mut stmt = tx.prepare(
                "INSERT INTO macro_iterations (run_id, macro_idx, micro_steps_taken, energy, converged, ucc_params) VALUES (?, ?, ?, ?, ?, ?)"
            )?;
            stmt.execute(params![
                run_id as i64,
                macro_idx as i64,
                micro_steps_taken as i64,
                energy,
                converged,
                &theta_blob,
            ])?;
        }

        tx.commit()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChemistryConfig, MethodConfig, TaskConfig};
    use ndarray::array;
    use std::fs;

    #[test]
    fn test_sqlite_logger_io() {
        let db_path = "test_logger.sqlite";
        let _ = fs::remove_file(db_path); // Cleanup before

        let method = MethodConfig {
            label: "TEST".to_string(),
            max_macro_iterations: None,
            energy_tolerance: None,
            ucc_optimizer: crate::config::UccOptimizerConfig {
                solver_config: crate::config::SolverConfig::Sequential {},
                ansatz_config: crate::config::AnsatzConfig::Uccd,
            },
            orbital_optimizer: None,
        };

        let task_config = TaskConfig {
            task_name: "test_task".to_string(),
            chemistry: ChemistryConfig {
                geometry: "H 0 0 0; H 0 0 1.5".to_string(),
                basis: "sto-3g".to_string(),
                charge: 0,
                spin: 0,
                freeze_core: 0,
            },
            method: method.clone(),
        };
        let mut logger = SqliteHandler::new(db_path, &task_config).unwrap();

        let ucc_names = vec!["theta_0".to_string(), "theta_1".to_string()];
        let oo_names = vec!["0,1".to_string(), "0,2".to_string()];
        logger.register_parameters(&ucc_names, &oo_names).unwrap();

        let theta_arr = array![0.1, 0.2];
        let final_theta = array![0.5];

        logger
            .save_iteration(1, 10, -1.0, false, &theta_arr, false)
            .unwrap();
        logger
            .save_iteration(2, 20, -1.1, false, &theta_arr, false)
            .unwrap();
        logger
            .save_iteration(3, 5, -1.14, true, &final_theta, true)
            .unwrap();

        // Check parameter mapping persistence
        let row_count: i64 = logger
            .conn
            .query_row("SELECT count(*) FROM parameter_names", [], |r| r.get(0))
            .unwrap();
        assert_eq!(row_count, 4);

        let iter_count: i64 = logger
            .conn
            .query_row("SELECT count(*) FROM macro_iterations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(iter_count, 3);

        // Final cleanup
        let _ = fs::remove_file(db_path);
    }
}
