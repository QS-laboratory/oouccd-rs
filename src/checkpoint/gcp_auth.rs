use serde_json;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use ureq;

static TOKEN_CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);

pub fn fetch_gcp_token() -> Option<String> {
    if let Ok(cache) = TOKEN_CACHE.lock() {
        if let Some((token, expiry)) = cache.as_ref() {
            if Instant::now() < *expiry {
                return Some(token.clone());
            }
        }
    }

    // Fetch Token (Timeouts bypass if local)
    let token_req = ureq::get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token?scopes=https://www.googleapis.com/auth/cloud-platform")
        .set("Metadata-Flavor", "Google")
        .timeout(Duration::from_millis(600))
        .call();

    let access_token = match token_req {
        Ok(res) => {
            if let Ok(value) = res.into_json::<serde_json::Value>() {
                value
                    .get("access_token")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        }
        Err(_) => None, // Local Dev Mode (No Metadata server found)
    };

    if let Some(ref token) = access_token {
        if let Ok(mut cache) = TOKEN_CACHE.lock() {
            // GCP tokens usually expire in 3600 seconds, so caching for 3000 seconds is safe.
            *cache = Some((token.clone(), Instant::now() + Duration::from_secs(3000)));
        }
    }

    access_token
}
