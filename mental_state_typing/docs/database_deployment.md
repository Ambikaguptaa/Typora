# Database Deployment & Persistence Architecture

**Project**: Mental-State Detection System using Typing Behavior  
**Phase**: Prompt 14 — Real Dataset Integration, Production LSTM & Persistent Database

---

## 1. Overview & Architectural Principle

The Mental-State Detection System uses a dual-tier storage strategy designed for seamless local development and production cloud deployment:

| Environment | Primary Backend | Configuration Source | Storage Location | Durability |
|---|---|---|---|---|
| **Local Development** | SQLite | `settings.database_path` or `.env` | `database/mental_state.db` | Persistent on local host |
| **Streamlit Community Cloud** | External PostgreSQL | `st.secrets["DATABASE_URL"]` or `st.secrets["postgres"]` | Managed Cloud PostgreSQL (e.g. Supabase, Neon, RDS) | Persistent across ephemeral container restarts |

### Core Safety Invariants:
1. **Zero Raw Text**: No character keys, typed text, words, or passwords are ever persisted to the database. All tables store strictly derived mathematical timing features, Z-scores, and metadata.
2. **Privacy Enforcement**: Any attempted persistence of payloads containing forbidden keys (`key`, `char`, `text`, `typed_text`, `password`, `word`, `sentence`, `message`, `user_input`, `content`) triggers an immediate `PrivacyViolationError`.
3. **Secret Protection**: Connection strings, passwords, and tokens are masked in all logs and error messages via `sanitize_database_url_for_logging()`.

---

## 2. Database Schema Specification

The database defines 5 persistent entities:

### 2.1 `sessions`
Tracks active and completed typing sessions:
- `session_id` (TEXT PRIMARY KEY): Unique session identifier.
- `pseudonymized_user_id` (TEXT NOT NULL): Keyed HMAC-SHA256 pseudonymized participant token.
- `session_type` (TEXT): `ANALYSIS` or `CALIBRATION`.
- `created_at` (TIMESTAMP): Creation timestamp.
- `sample_count` (INTEGER): Number of paired timing events.
- `status` (TEXT): `active`, `completed`, or `purged`.

### 2.2 `typing_metrics`
Summary timing statistics per session:
- `metric_id` (PRIMARY KEY AUTOINCREMENT / SERIAL): Unique metric record identifier.
- `session_id` (TEXT FK): Associated session.
- `mean_hold_time_ms` (REAL): Average key depression time.
- `mean_flight_time_ms` (REAL): Average inter-key flight time.
- `pause_rate` (REAL): Proportion of intervals exceeding pause threshold.
- `estimated_strain_score` (REAL): Aggregate behavioral variation estimate.
- `recorded_at` (TIMESTAMP): Record timestamp.

### 2.3 `assessments`
Complete behavioral assessment records (enables report regeneration on ephemeral cloud hosts):
- `assessment_id` (TEXT PRIMARY KEY): Unique assessment identifier (`asmt_<session_id>`).
- `session_id` (TEXT FK): Associated session.
- `pseudonymized_user_id` (TEXT NOT NULL): Pseudonymized subject identifier.
- `session_type` (TEXT): `ANALYSIS` or `CALIBRATION`.
- `created_at` (TIMESTAMP): Assessment generation timestamp.
- `duration_s` (REAL): Active typing duration.
- `event_count` (INTEGER): Total keystroke transitions analyzed.
- `mean_dwell_ms`, `mean_flight_ms`, `mean_pause_ms`, `pause_rate`, `typing_speed_wpm`, `backspace_rate`: Derived timing parameters.
- `tdi_score` (REAL): Typing Deviation Index against personal baseline.
- `tdi_divergence_level` (TEXT): Divergence classification (`NORMAL`, `MODERATE`, `SIGNIFICANT`).
- `model_predicted_class`, `model_confidence`, `model_status`: Gated behavioral model outputs.
- `assessment_json` (TEXT): Audited, non-sensitive JSON payload for complete report regeneration.

### 2.4 `baseline_profiles`
Persistent personal baseline calibration records:
- `user_id` (TEXT PRIMARY KEY): Pseudonymized participant ID.
- `session_count` (INTEGER): Number of completed calibration sessions (0 to $\ge 5$).
- `is_ready` (INTEGER): 1 if $\ge 5$ sessions completed, 0 otherwise.
- `mean_dwell_time_ms`, `std_dwell_time_ms`, `mean_flight_time_ms`, `std_flight_time_ms`, etc.: Longitudinal mean and standard deviation parameters.
- `last_updated` (TIMESTAMP): Timestamp of last calibration.

### 2.5 `audit_logs`
Security and operational audit trail:
- `log_id` (PRIMARY KEY AUTOINCREMENT / SERIAL): Unique log identifier.
- `event_type` (TEXT NOT NULL): E.g., `DATA_PURGE`, `ACCESS_CHECK`, `RETENTION_CYCLE`.
- `user_id` (TEXT): Optional participant token.
- `details` (TEXT): Non-sensitive event metadata.
- `created_at` (TIMESTAMP): Event timestamp.

---

## 3. Streamlit Community Cloud Deployment Configuration

When deploying the application to Streamlit Community Cloud:

1. In the Streamlit Cloud Dashboard, navigate to **App Settings > Secrets**.
2. Add your external PostgreSQL connection credentials:

```toml
# Streamlit Secrets (.streamlit/secrets.toml)
# Example for Supabase (Session Pooler on port 5432):
DATABASE_URL = "postgresql://postgres.yourprojectref:your_password@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"

# Alternatively, configure via the postgres dictionary:
[postgres]
user = "postgres.yourprojectref"
password = "your_password"
host = "aws-0-us-east-1.pooler.supabase.com"
port = 5432
dbname = "postgres"
```

### 3.1 Supabase Configuration Notes (IPv6 vs IPv4 Session Pooler)
> [!IMPORTANT]
> **Direct Supabase endpoints (`db.<project-ref>.supabase.co:5432`) resolve only via IPv6.**  
> Streamlit Community Cloud runs in AWS environments without outbound IPv6 routing. Attempting to use the direct host will result in connection timeouts or `could not translate host name "db.<ref>.supabase.co"` / `Network is unreachable`.
> 
> **Required Configuration (Session Pooler)**:
> 1. In your Supabase Dashboard, go to **Project Settings > Database > Connection Pooling**.
> 2. Select **Session** mode.
> 3. Use port **5432**.
> 4. Ensure username is `postgres.[your-project-ref]`.
> 5. Host is `aws-<index>-[region].pooler.supabase.com`.
> 6. Append `?sslmode=require`.

### 3.2 Canonical Streamlit Secret Format (Step 15)
In the **Streamlit Community Cloud Dashboard** (`App Settings > Secrets`), enter:
```toml
# Canonical Cloud Secret Interface
DATABASE_URL = "postgresql://postgres.[YOUR-PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[YOUR-REGION].pooler.supabase.com:5432/postgres?sslmode=require"
```
- No separate host, port, or database configuration is required when `DATABASE_URL` is configured.
- Real credentials must NEVER be committed to Git or pushed to GitHub (`.gitignore` protects secrets).

### 3.3 Configuration Resolution Precedence (Deterministic Policy)
The application resolves database configuration using a strict deterministic priority order:
1. **Streamlit Cloud Root-Level Secret `st.secrets["DATABASE_URL"]`** (Canonical cloud secret; authoritatively overrides container environment and local `.env`).
2. **Streamlit Cloud Secret `st.secrets["database_url"]`** (Lowercase variant).
3. **Streamlit Cloud Secret `st.secrets["connections"]["postgresql"]["url"]`** (Streamlit native SQL connection format).
4. **Streamlit Cloud Secret `st.secrets["postgres"]["url"]` or table** (`st.secrets["postgres"]`).
5. **Streamlit Cloud Secret `st.secrets["postgresql"]["url"]`**.
6. **Process Environment `DATABASE_URL`** (`os.environ["DATABASE_URL"]`, for Docker/CLI workflows).
7. **Local `.env` `DATABASE_URL`** (For local environment customization).
8. **Local SQLite Fallback** (`database/mental_state.db`, active ONLY in local development).

### 3.4 Strict No-Silent-Fallback Invariant (Cloud Data Integrity)
> [!WARNING]
> **Cloud Deployments Never Silently Fall Back to SQLite.**  
> If external PostgreSQL fails or `DATABASE_URL` is missing in Streamlit Cloud, the application will **NEVER** silently switch to SQLite. Silently falling back to SQLite on Streamlit Cloud creates ephemeral local storage that is wiped on container reboot, falsely presenting the system as functional while dropping participant records.
> 
> Instead, the application:
> - Traps database failure at the error boundary to prevent whole-app crashing.
> - Reports truthful status in the sidebar: `OFFLINE (CONFIG_ERROR)`, `OFFLINE (DNS_ERROR)`, `OFFLINE (AUTH_ERROR)`, or `OFFLINE (CONNECTION_ERROR)`.
> - Displays safe masked diagnostics in the **System Status** dashboard and flushes diagnostic summaries to `sys.stderr`.
> - Preserves local SQLite exclusively for genuine local development workflows.

### 3.5 Database Health Reporting
Health status is evaluated via `SELECT 1;` with immediate connection cleanup:
- `DATABASE_READY`: PostgreSQL connection established and query verified.
- `CONFIG_ERROR`: Missing or malformed `DATABASE_URL` secret.
- `DNS_ERROR`: Hostname resolution failure (e.g. attempting to resolve direct IPv6 Supabase host).
- `AUTH_ERROR`: Bad username or password.
- `CONNECTION_ERROR`: TCP timeout, network unreachable, or SSL negotiation failure.

### 3.6 PostgreSQL Driver & Native Segmentation Fault Prevention
The production deployment uses `pg8000>=1.30.0`, a **100% pure-Python** PostgreSQL driver:
- **No Compiled C Extensions**: Unlike `psycopg2-binary`, `pg8000` contains zero compiled native binaries and zero bundled `libpq.so`/`libssl.so` libraries.
- **Zero Native Symbol Conflicts**: Prevents glibc / OpenSSL thread-local storage collisions when running alongside TensorFlow 2.21.0, PyArrow, and Streamlit in Linux container environments.
- **Safe SSL**: Leverages Python's standard library `ssl.create_default_context()` and `scramp` for SCRAM-SHA-256 authentication over Supabase pooler connections.
- **Cached Lazy Initialization**: Database schema DDL is executed once per process lifetime via `@st.cache_resource`, and connection health is cached via `@st.cache_data(ttl=60)` to eliminate connection storms and port exhaustion on rerun.

