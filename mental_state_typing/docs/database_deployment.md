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
> 5. Host is `aws-0-[region].pooler.supabase.com`.
> 6. Append `?sslmode=require`.

### 3.2 Configuration Resolution Precedence
The application resolves database configuration using a strict deterministic priority order:
1. **Streamlit Secrets `st.secrets["DATABASE_URL"]`** (Authoritative in cloud deployments; cannot be overridden by local `.env` or container environment).
2. **Streamlit Secrets `st.secrets["database_url"]`** (Lowercase variant).
3. **Streamlit Secrets `st.secrets["connections"]["postgresql"]["url"]`** (Streamlit SQL connection format).
4. **Streamlit Secrets `st.secrets["postgres"]["url"]` or table** (`st.secrets["postgres"]`).
5. **Streamlit Secrets `st.secrets["postgresql"]["url"]`**.
6. **Environment Variable `DATABASE_URL`** (`os.environ["DATABASE_URL"]`, used for Docker/CLI workflows).
7. **Local SQLite Fallback** (`database/mental_state.db`, active ONLY when no secrets or env vars are present).

### 3.3 Strict No-Silent-Fallback Invariant
> [!WARNING]
> **Cloud Deployments Never Silently Fall Back to SQLite.**  
> If external PostgreSQL fails (e.g. DNS failure, auth error, network timeout), the application will **NEVER** silently switch to SQLite. Silently falling back to SQLite on Streamlit Cloud would give a false appearance of operation while writing user records to an ephemeral filesystem that is wiped on container reboot.
> 
> Instead, the application:
> - Traps the database failure at the error boundary to prevent whole-app crashing.
> - Reports the truthful status in the sidebar: `OFFLINE (DNS_ERROR)`, `OFFLINE (AUTH_ERROR)`, or `OFFLINE (CONNECTION_ERROR)`.
> - Emits a safe diagnostic block to `stderr` indicating the config source, masked host, masked user, and exact exception reason.
> - Preserves local SQLite exclusively for genuine local development workflows.
