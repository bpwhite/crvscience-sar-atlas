//! Project Workspace Persistence in Portable JSON (`.saratlas.json`) and Normalized
//! Relational `SQLite 3` (`.sqlite` / `.db` via bundled [`rusqlite`](https://docs.rs/rusqlite/latest/rusqlite/)).
//!
//! Preserves exact sanitized molecules, custom 2D atom coordinates (via MDL V2000 molblocks),
//! pre-materialized `QED` / `CNS-MPO` / `ADMET` (`ESOL`, `BOILED-Egg`, `Pfizer 3/75`, `hERG`)
//! relational columns for external SQL analytics, active `SMARTS` & `Free-Wilson` settings,
//! and the in-progress `2D Sketcher` structure!

use crate::discovery_pipeline::DiscoveryPipelineConfig;
use crate::mol::Molecule;
use crate::mol_io::{format_sdf_record, parse_sdf_record, MolIoError};
use crate::sketcher::molecule_to_smiles;
use crate::DatasetAnalysis;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

pub const PROJECT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ProjectStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON workspace error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SQLite database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Molecule structure error: {0}")]
    MolIo(#[from] MolIoError),
    #[error("Workspace contains 0 compounds")]
    EmptyWorkspace,
}

/// Stored compound entry inside a portable `.saratlas.json` workspace file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredCompoundEntry {
    pub id: String,
    pub name: String,
    pub target: String,
    pub metric: String,
    pub raw_nm: f64,
    pub pact: f64,
    pub canonical_smiles: String,
    pub source_url: String,
    pub molblock_v2000: String,
}

/// Complete serializable project session state across `DatasetAnalysis`, `SMARTS`, `Free-Wilson`,
/// and the interactive `2D Sketcher`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SarProjectWorkspace {
    pub schema_version: u32,
    pub app_version: String,
    pub project_name: String,
    pub notes: String,
    pub min_tanimoto: f32,
    pub sali_cliff_threshold: f32,
    pub active_smarts_query: String,
    pub target_filter: String,
    pub fw_core_name: String,
    pub fw_core_smarts: String,
    pub fw_max_slots: usize,
    pub fw_ridge_lambda: f64,
    pub sketcher_molblock_v2000: Option<String>,
    pub sketcher_pinned_parent_id: Option<String>,
    #[serde(default)]
    pub pipeline_config: Option<DiscoveryPipelineConfig>,
    pub compounds: Vec<StoredCompoundEntry>,
}

/// Restored runtime state unpacked from a `.saratlas.json` or `.sqlite` workspace file.
pub struct LoadedProjectSession {
    pub project_name: String,
    pub notes: String,
    pub ds: DatasetAnalysis,
    pub active_smarts_query: String,
    pub target_filter: String,
    pub fw_core_name: String,
    pub fw_core_smarts: String,
    pub fw_max_slots: usize,
    pub fw_ridge_lambda: f64,
    pub sketcher_mol: Option<Molecule>,
    pub sketcher_pinned_parent_id: Option<String>,
    pub pipeline_config: Option<DiscoveryPipelineConfig>,
}

impl SarProjectWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        project_name: impl Into<String>,
        notes: impl Into<String>,
        ds: &DatasetAnalysis,
        active_smarts_query: &str,
        target_filter: &str,
        fw_core_name: &str,
        fw_core_smarts: &str,
        fw_max_slots: usize,
        fw_ridge_lambda: f64,
        sketcher_mol: Option<&Molecule>,
        sketcher_pinned_parent_id: Option<String>,
    ) -> Self {
        let compounds = ds
            .mols
            .iter()
            .map(|m| StoredCompoundEntry {
                id: m.id.clone(),
                name: m.name.clone(),
                target: m.target.clone(),
                metric: m.metric.clone(),
                raw_nm: m.raw_nm,
                pact: m.pact,
                canonical_smiles: molecule_to_smiles(m),
                source_url: m.source_url.clone(),
                molblock_v2000: format_sdf_record(m),
            })
            .collect();

        let sketcher_molblock_v2000 = sketcher_mol
            .filter(|m| !m.atoms.is_empty())
            .map(format_sdf_record);

        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            project_name: project_name.into(),
            notes: notes.into(),
            min_tanimoto: ds.min_tanimoto,
            sali_cliff_threshold: ds.sali_cliff_threshold,
            active_smarts_query: active_smarts_query.to_string(),
            target_filter: target_filter.to_string(),
            fw_core_name: fw_core_name.to_string(),
            fw_core_smarts: fw_core_smarts.to_string(),
            fw_max_slots,
            fw_ridge_lambda,
            sketcher_molblock_v2000,
            sketcher_pinned_parent_id,
            pipeline_config: None,
            compounds,
        }
    }

    /// Attaches an active `DiscoveryPipelineConfig` workflow recipe to the workspace snapshot.
    pub fn with_pipeline_config(mut self, cfg: DiscoveryPipelineConfig) -> Self {
        self.pipeline_config = Some(cfg);
        self
    }

    /// Reconstructs the full `DatasetAnalysis` (preserving exact saved V2000 2D coordinates)
    /// and optional in-progress `2D Sketcher` molecule.
    pub fn into_loaded_session(self) -> Result<LoadedProjectSession, ProjectStoreError> {
        if self.compounds.is_empty() {
            return Err(ProjectStoreError::EmptyWorkspace);
        }

        let mut mols = Vec::with_capacity(self.compounds.len());
        for entry in self.compounds {
            let mut m = parse_sdf_record(&entry.molblock_v2000)?;
            m.id = entry.id;
            m.name = entry.name;
            m.target = entry.target;
            m.metric = entry.metric;
            m.raw_nm = entry.raw_nm;
            m.pact = entry.pact;
            m.smiles = entry.canonical_smiles;
            m.source_url = entry.source_url;
            mols.push(m);
        }

        let ds = DatasetAnalysis::from_molecules(
            mols,
            self.min_tanimoto,
            self.sali_cliff_threshold,
        );

        let sketcher_mol = match self.sketcher_molblock_v2000 {
            Some(ref mb) if !mb.trim().is_empty() => parse_sdf_record(mb).ok(),
            _ => None,
        };

        Ok(LoadedProjectSession {
            project_name: self.project_name,
            notes: self.notes,
            ds,
            active_smarts_query: self.active_smarts_query,
            target_filter: self.target_filter,
            fw_core_name: self.fw_core_name,
            fw_core_smarts: self.fw_core_smarts,
            fw_max_slots: self.fw_max_slots,
            fw_ridge_lambda: self.fw_ridge_lambda,
            sketcher_mol,
            sketcher_pinned_parent_id: self.sketcher_pinned_parent_id,
            pipeline_config: self.pipeline_config,
        })
    }
}

/// Saves `ws` (along with pre-materialized `ADMET` & `SALI` SQL tables when writing SQLite)
/// automatically dispatching by file extension (`.sqlite` / `.sqlite3` / `.db` vs `.json`).
pub fn save_project_auto(
    path: &Path,
    ws: &SarProjectWorkspace,
    ds_for_sql_metrics: &DatasetAnalysis,
) -> Result<(), ProjectStoreError> {
    if is_sqlite_extension(path) {
        save_project_sqlite(path, ws, ds_for_sql_metrics)
    } else {
        save_project_json(path, ws)
    }
}

/// Loads a `LoadedProjectSession` from either a `.sqlite`/`.db` database or `.saratlas.json` file
/// (inspecting both the 16-byte `SQLite format 3\0` magic header and file extension).
pub fn load_project_auto(path: &Path) -> Result<LoadedProjectSession, ProjectStoreError> {
    let bytes = std::fs::read(path)?;
    if bytes.starts_with(b"SQLite format 3\0") || is_sqlite_extension(path) {
        load_project_sqlite(path)
    } else {
        let txt = String::from_utf8_lossy(&bytes);
        let ws: SarProjectWorkspace = serde_json::from_str(&txt)?;
        ws.into_loaded_session()
    }
}

pub fn save_project_json(path: &Path, ws: &SarProjectWorkspace) -> Result<(), ProjectStoreError> {
    let payload = serde_json::to_string_pretty(ws)?;
    std::fs::write(path, payload)?;
    Ok(())
}

/// Writes a normalized, directly SQL-queryable `SQLite 3` database containing `project_meta`,
/// `compounds` (with full `QED`, `CNS-MPO`, `ESOL LogS`, `BOILED-Egg`, `Pfizer 3/75`, `hERG`,
/// and V2000 2D structure molblocks), and `sali_cliffs`.
pub fn save_project_sqlite(
    path: &Path,
    ws: &SarProjectWorkspace,
    ds: &DatasetAnalysis,
) -> Result<(), ProjectStoreError> {
    let mut conn = Connection::open(path)?;
    save_project_to_sqlite_conn(&mut conn, ws, ds)
}

pub fn save_project_to_sqlite_conn(
    conn: &mut Connection,
    ws: &SarProjectWorkspace,
    ds: &DatasetAnalysis,
) -> Result<(), ProjectStoreError> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         CREATE TABLE IF NOT EXISTS project_meta (
             key   TEXT PRIMARY KEY,
             value TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS compounds (
             ord               INTEGER PRIMARY KEY,
             compound_id       TEXT NOT NULL,
             name              TEXT NOT NULL,
             target            TEXT NOT NULL,
             metric            TEXT NOT NULL,
             raw_nm            REAL NOT NULL,
             pact              REAL NOT NULL,
             canonical_smiles  TEXT NOT NULL,
             formula           TEXT NOT NULL,
             mw                REAL NOT NULL,
             clogp             REAL NOT NULL,
             clogd_74          REAL NOT NULL,
             tpsa              REAL NOT NULL,
             qed_mean          REAL NOT NULL,
             cns_mpo           REAL NOT NULL,
             lle               REAL NOT NULL,
             esol_logs         REAL NOT NULL,
             boiled_egg        TEXT NOT NULL,
             pgp_substrate     INTEGER NOT NULL,
             pfizer_3_75_pass  INTEGER NOT NULL,
             gsk_4_400_pass    INTEGER NOT NULL,
             herg_risk         TEXT NOT NULL,
             pains_count       INTEGER NOT NULL,
             pareto_optimal    INTEGER NOT NULL,
             source_url        TEXT NOT NULL,
             molblock_v2000    TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_compounds_target ON compounds(target);
         CREATE INDEX IF NOT EXISTS idx_compounds_pact   ON compounds(pact DESC);
         CREATE TABLE IF NOT EXISTS sali_cliffs (
             id             INTEGER PRIMARY KEY AUTOINCREMENT,
             compound_a_id  TEXT NOT NULL,
             compound_b_id  TEXT NOT NULL,
             target         TEXT NOT NULL,
             tanimoto       REAL NOT NULL,
             delta_pact     REAL NOT NULL,
             sali           REAL NOT NULL,
             is_cliff       INTEGER NOT NULL
         );",
    )?;

    let tx = conn.transaction()?;
    tx.execute("DELETE FROM compounds", [])?;
    tx.execute("DELETE FROM sali_cliffs", [])?;
    {
        let mut meta_stmt =
            tx.prepare("INSERT OR REPLACE INTO project_meta (key, value) VALUES (?1, ?2)")?;
        let meta_pairs = [
            ("schema_version", ws.schema_version.to_string()),
            ("app_version", ws.app_version.clone()),
            ("project_name", ws.project_name.clone()),
            ("notes", ws.notes.clone()),
            ("min_tanimoto", ws.min_tanimoto.to_string()),
            ("sali_cliff_threshold", ws.sali_cliff_threshold.to_string()),
            ("active_smarts_query", ws.active_smarts_query.clone()),
            ("target_filter", ws.target_filter.clone()),
            ("fw_core_name", ws.fw_core_name.clone()),
            ("fw_core_smarts", ws.fw_core_smarts.clone()),
            ("fw_max_slots", ws.fw_max_slots.to_string()),
            ("fw_ridge_lambda", ws.fw_ridge_lambda.to_string()),
            (
                "sketcher_molblock_v2000",
                ws.sketcher_molblock_v2000.clone().unwrap_or_default(),
            ),
            (
                "sketcher_pinned_parent_id",
                ws.sketcher_pinned_parent_id.clone().unwrap_or_default(),
            ),
            (
                "pipeline_config_json",
                ws.pipeline_config
                    .as_ref()
                    .and_then(|c| serde_json::to_string(c).ok())
                    .unwrap_or_default(),
            ),
        ];
        for (k, v) in meta_pairs {
            meta_stmt.execute(params![k, v])?;
        }
    }

    {
        let mut cmp_stmt = tx.prepare(
            "INSERT INTO compounds (
                ord, compound_id, name, target, metric, raw_nm, pact, canonical_smiles,
                formula, mw, clogp, clogd_74, tpsa, qed_mean, cns_mpo, lle,
                esol_logs, boiled_egg, pgp_substrate, pfizer_3_75_pass, gsk_4_400_pass,
                herg_risk, pains_count, pareto_optimal, source_url, molblock_v2000
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21,
                ?22, ?23, ?24, ?25, ?26
             )",
        )?;

        for (i, c) in ws.compounds.iter().enumerate() {
            let p = &ds.profiles[i];
            let ad = &ds.admet[i];
            let pains_cnt = ds.alerts[i].len() as i64;
            cmp_stmt.execute(params![
                i as i64,
                c.id,
                c.name,
                c.target,
                c.metric,
                c.raw_nm,
                c.pact,
                c.canonical_smiles,
                p.formula,
                p.mw,
                p.clogp,
                p.clogd_74,
                p.tpsa,
                p.qed_mean,
                p.cns_mpo,
                p.lle,
                ad.esol_logs,
                ad.boiled_egg.label(),
                ad.pgp_substrate as i64,
                ad.pfizer_3_75_pass as i64,
                ad.gsk_4_400_pass as i64,
                ad.herg_risk.label(),
                pains_cnt,
                p.pareto_optimal as i64,
                c.source_url,
                c.molblock_v2000,
            ])?;
        }
    }

    {
        let mut cliff_stmt = tx.prepare(
            "INSERT INTO sali_cliffs (
                compound_a_id, compound_b_id, target, tanimoto, delta_pact, sali, is_cliff
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for e in ds.edges.iter().filter(|e| e.same_target).take(500) {
            cliff_stmt.execute(params![
                ds.mols[e.mol_a].id,
                ds.mols[e.mol_b].id,
                ds.mols[e.mol_a].target,
                e.tanimoto as f64,
                e.delta_pact as f64,
                e.sali as f64,
                e.is_cliff as i64,
            ])?;
        }
    }

    tx.commit()?;
    Ok(())
}

pub fn load_project_sqlite(path: &Path) -> Result<LoadedProjectSession, ProjectStoreError> {
    let conn = Connection::open(path)?;
    load_project_from_sqlite_conn(&conn)
}

pub fn load_project_from_sqlite_conn(
    conn: &Connection,
) -> Result<LoadedProjectSession, ProjectStoreError> {
    let mut meta = std::collections::HashMap::<String, String>::new();
    {
        let mut stmt = conn.prepare("SELECT key, value FROM project_meta")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (k, v) = row?;
            meta.insert(k, v);
        }
    }

    let mut compounds = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT compound_id, name, target, metric, raw_nm, pact, canonical_smiles, source_url, molblock_v2000
             FROM compounds ORDER BY ord ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(StoredCompoundEntry {
                id: r.get(0)?,
                name: r.get(1)?,
                target: r.get(2)?,
                metric: r.get(3)?,
                raw_nm: r.get(4)?,
                pact: r.get(5)?,
                canonical_smiles: r.get(6)?,
                source_url: r.get(7)?,
                molblock_v2000: r.get(8)?,
            })
        })?;
        for row in rows {
            compounds.push(row?);
        }
    }

    let get_s = |k: &str, def: &str| meta.get(k).cloned().unwrap_or_else(|| def.to_string());
    let sk_mb = get_s("sketcher_molblock_v2000", "");
    let sk_pin = get_s("sketcher_pinned_parent_id", "");

    let ws = SarProjectWorkspace {
        schema_version: get_s("schema_version", "1").parse().unwrap_or(1),
        app_version: get_s("app_version", "0.6.0"),
        project_name: get_s("project_name", "SQLite-SAR-Project"),
        notes: get_s("notes", ""),
        min_tanimoto: get_s("min_tanimoto", "0.30").parse().unwrap_or(0.30),
        sali_cliff_threshold: get_s("sali_cliff_threshold", "6.0").parse().unwrap_or(6.0),
        active_smarts_query: get_s("active_smarts_query", ""),
        target_filter: get_s("target_filter", "All"),
        fw_core_name: get_s("fw_core_name", "Auto Core"),
        fw_core_smarts: get_s("fw_core_smarts", ""),
        fw_max_slots: get_s("fw_max_slots", "3").parse().unwrap_or(3),
        fw_ridge_lambda: get_s("fw_ridge_lambda", "0.15").parse().unwrap_or(0.15),
        sketcher_molblock_v2000: (!sk_mb.trim().is_empty()).then_some(sk_mb),
        sketcher_pinned_parent_id: (!sk_pin.trim().is_empty()).then_some(sk_pin),
        pipeline_config: {
            let p_json = get_s("pipeline_config_json", "");
            if p_json.trim().is_empty() {
                None
            } else {
                serde_json::from_str(&p_json).ok()
            }
        },
        compounds,
    };

    ws.into_loaded_session()
}

fn is_sqlite_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "sqlite" | "sqlite3" | "db"
    )
}
