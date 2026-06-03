use anyhow::Result;
use std::fs;


pub struct GcsHandler {
    pub bucket_name: String,
    pub db_filename: String,
    pub target_gcs_path: String,
}

impl GcsHandler {
    pub fn new(bucket_name: &str, db_filename: &str, target_gcs_path: &str) -> Self {
        Self {
            bucket_name: bucket_name.to_string(),
            db_filename: db_filename.to_string(),
            target_gcs_path: target_gcs_path.to_string(),
        }
    }

    /// Upload the local SQLite database to GCS. Called only when all iterations are complete.
    pub fn save_checkpoint(&self, is_completed: bool) -> Result<()> {

        if !is_completed {
            return Ok(());
        }

        eprintln!("[gcs] upload | starting");

        let access_token = match crate::checkpoint::gcp_auth::fetch_gcp_token() {
            Some(t) => t,
            None => return Ok(()),
        };

        let binary_payload = match fs::read(&self.db_filename) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[gcs] upload | failed to read local db: {e}");
                return Ok(());
            }
        };

        let url = format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}",
            self.bucket_name,
            urlencoding::encode(&self.target_gcs_path)
        );

        let res = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", access_token))
            .set("Content-Type", "application/octet-stream")
            .send_bytes(&binary_payload);

        match res {
            Ok(_) => {
                eprintln!("[gcs] upload succeeded | {} bytes", binary_payload.len());
                Ok(())
            }
            Err(e) => {
                eprintln!("[gcs] upload failed | {e}");
                Err(anyhow::anyhow!("GCS Save Error: {}", e))
            }
        }
    }
}
