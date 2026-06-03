# --------- 1. CHEF BASE ENVIRONMENT ---------
FROM lukemathwalker/cargo-chef:0.1.77-rust-slim-bookworm AS chef

ENV DEBIAN_FRONTEND=noninteractive

# Install static C-libraries and Native OS tools
RUN apt-get update && apt-get install -y \
    curl gnupg python3 python3-pip python3-venv python3-dev \
    libssl-dev pkg-config build-essential libfontconfig1-dev \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Pre-compile analytical Python libraries for exact Pyo3 matching
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# --------- 2. RECIPE PLANNER ---------
FROM chef AS planner
COPY . .
# Calculate the exact dependency hash signature natively
RUN cargo chef prepare --recipe-path recipe.json

# --------- 3. DEPENDENCY BUILDER ---------
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Compile only the mathematical and generic crates (This layer skips everything else and caches FOREVER)
RUN cargo chef cook --release --recipe-path recipe.json

# --------- 4. FAST SOURCE CODE COMPILE ---------
# Now if you edit main.rs, Docker physically drops instantly to this line!
COPY . .
RUN cargo build --release

# Seal the Cloud Batch entrypoint securely
RUN chmod +x run_batch.sh
ENTRYPOINT ["/app/run_batch.sh"]
