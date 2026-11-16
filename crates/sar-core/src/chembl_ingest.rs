// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! Live EMBL-EBI `ChEMBL` REST Web Services and NCBI `PubChem` `PUG-REST` bioactivity
//! & compound structure importer, salt stripper, and replicate `pChEMBL` / `IC50` median curator:
//! - ChEMBL Data Web Services (`https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services`,
//!   `https://www.ebi.ac.uk/chembl/`)
//! - PubChem PUG-REST BioAssay & Compound Services (`https://academic.oup.com/nar/article/46/W1/W563/4990016`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC6030920/`,
//!   `https://iupac.github.io/WFChemCookbook/datasources/pubchem_pugrest1.html`)

use crate::mol::{Element, Molecule};
use crate::mol_io::parse_smiles;
use crate::sketcher::molecule_to_smiles;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;
use thiserror::Error;

pub const CHEMBL_API_BASE: &str = "https://www.ebi.ac.uk/chembl/api/data";
pub const PUBCHEM_PUG_BASE: &str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug";

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("JSON response parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("No valid bioactive small-molecule structures matched query: {0}")]
    EmptyResult(String),
}

/// Curated EMBL-EBI ChEMBL protein target preset for one-click medicinal-chemistry series import.
pub struct ChemblTargetPreset {
    pub target_chembl_id: &'static str,
    pub short_name: &'static str,
    pub full_name: &'static str,
    pub target_url: &'static str,
}

pub const CURATED_CHEMBL_TARGETS: [ChemblTargetPreset; 6] = [
    ChemblTargetPreset {
        target_chembl_id: "CHEMBL203",
        short_name: "EGFR",
        full_name: "Epidermal growth factor receptor (Homo sapiens)",
        target_url: "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL203",
    },
    ChemblTargetPreset {
        target_chembl_id: "CHEMBL279",
        short_name: "VEGFR2",
        full_name: "Vascular endothelial growth factor receptor 2",
        target_url: "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL279",
    },
    ChemblTargetPreset {
        target_chembl_id: "CHEMBL5145",
        short_name: "BRAF",
        full_name: "Serine/threonine-protein kinase B-raf",
        target_url: "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL5145",
    },
    ChemblTargetPreset {
        target_chembl_id: "CHEMBL217",
        short_name: "Dopamine D2",
        full_name: "Dopamine D2 receptor (Homo sapiens)",
        target_url: "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL217",
    },
    ChemblTargetPreset {
        target_chembl_id: "CHEMBL230",
        short_name: "COX-2",
        full_name: "Cyclooxygenase-2 (Prostaglandin G/H synthase 2)",
        target_url: "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL230",
    },
    ChemblTargetPreset {
        target_chembl_id: "CHEMBL4822",
        short_name: "BACE1",
        full_name: "Beta-secretase 1 (Homo sapiens)",
        target_url: "https://www.ebi.ac.uk/chembl/explore/target/CHEMBL4822",
    },
];

/// Curated NCBI PubChem BioAssay (`AID`) preset for one-click PUG-REST import.
pub struct PubChemAssayPreset {
    pub aid: u64,
    pub target_label: &'static str,
    pub description: &'static str,
}

pub const CURATED_PUBCHEM_ASSAYS: [PubChemAssayPreset; 4] = [
    PubChemAssayPreset {
        aid: 1433,
        target_label: "EGFR",
        description: "EGFR Kinase Inhibition Confirmatory BioAssay",
    },
    PubChemAssayPreset {
        aid: 624202,
        target_label: "BRAF",
        description: "BRAF V600E Kinase Inhibitor Dose-Response Assay",
    },
    PubChemAssayPreset {
        aid: 485341,
        target_label: "Dopamine D2",
        description: "DRD2 Antagonist Quantitative HTS BioAssay",
    },
    PubChemAssayPreset {
        aid: 1851,
        target_label: "CYP450",
        description: "Cytochrome P450 Isoform Inhibition qHTS Panel",
    },
];

/// Query parameters for fetching a congeneric or target series from EMBL-EBI ChEMBL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChemblQueryConfig {
    pub target_chembl_id: String,
    pub target_label: String,
    pub min_pchembl: f64,
    pub max_compounds: usize,
    /// Assay type (`"B"` = Binding, `"F"` = Functional, `""` = Any).
    pub assay_type: String,
}

impl Default for ChemblQueryConfig {
    fn default() -> Self {
        Self {
            target_chembl_id: "CHEMBL203".to_string(),
            target_label: "EGFR".to_string(),
            min_pchembl: 6.5,
            max_compounds: 80,
            assay_type: "B".to_string(),
        }
    }
}

/// Summary report returned after curating live ChEMBL or PubChem records.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestReport {
    pub source_name: String,
    pub endpoint_url: String,
    pub raw_records_scanned: usize,
    pub desalted_count: usize,
    pub deduplicated_molecules: usize,
}

struct RawBioRecord {
    id: String,
    name: String,
    target: String,
    metric: String,
    smiles: String,
    pact: f64,
    source_url: String,
}

fn build_http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(12))
        .timeout_read(Duration::from_secs(25))
        .user_agent("sar-atlas/0.6.0 (https://github.com/crvscience/sar-atlas)")
        .build()
}

fn http_get_text(agent: &ureq::Agent, url: &str) -> Result<String, IngestError> {
    agent
        .get(url)
        .set("Accept", "application/json, text/csv, text/plain, */*")
        .call()
        .map_err(|e| IngestError::Http(format!("{url} -> {e}")))?
        .into_string()
        .map_err(|e| IngestError::Http(format!("Body read error on {url}: {e}")))
}

/// Fetches live `IC50`/`Ki`/`Kd` bioactivity records for `cfg.target_chembl_id` from EMBL-EBI ChEMBL,
/// strips counterion salts, aggregates replicate `pChEMBL` potencies by median, and returns sanitized `Molecule`s.
pub fn fetch_chembl_target_series(
    cfg: &ChemblQueryConfig,
) -> Result<(Vec<Molecule>, IngestReport), IngestError> {
    let agent = build_http_agent();
    let tid = cfg.target_chembl_id.trim().to_ascii_uppercase();
    let max_mols = cfg.max_compounds.clamp(5, 500);
    let page_limit = (max_mols * 2).clamp(40, 250);

    let mut url = format!(
        "{CHEMBL_API_BASE}/activity.json?target_chembl_id={tid}&pchembl_value__gte={:.2}&limit={page_limit}&offset=0",
        cfg.min_pchembl.clamp(4.0, 11.0)
    );
    if !cfg.assay_type.trim().is_empty() {
        url.push_str(&format!("&assay_type={}", cfg.assay_type.trim()));
    }

    let body = http_get_text(&agent, &url)?;
    let (mols, mut rep) = parse_chembl_activities_json(&body, &cfg.target_label, max_mols)?;
    rep.endpoint_url = url;
    Ok((mols, rep))
}

/// Parses an EMBL-EBI ChEMBL `/api/data/activity.json` payload (offline-testable!).
pub fn parse_chembl_activities_json(
    json_str: &str,
    fallback_target_label: &str,
    max_compounds: usize,
) -> Result<(Vec<Molecule>, IngestReport), IngestError> {
    let root: Value = serde_json::from_str(json_str)?;
    let arr = root
        .get("activities")
        .and_then(|v| v.as_array())
        .ok_or_else(|| IngestError::EmptyResult("Missing 'activities' array in ChEMBL JSON".into()))?;

    let mut raw_rows = Vec::new();
    for item in arr {
        let Some(smi) = item.get("canonical_smiles").and_then(|v| v.as_str()) else {
            continue;
        };
        if smi.trim().is_empty() {
            continue;
        }
        let chembl_id = item
            .get("molecule_chembl_id")
            .and_then(|v| v.as_str())
            .unwrap_or("CHEMBL-UNK")
            .to_string();
        let pref_name = item
            .get("molecule_pref_name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(title_case_drug_name)
            .unwrap_or_else(|| chembl_id.clone());

        let target = if !fallback_target_label.trim().is_empty() {
            fallback_target_label.trim().to_string()
        } else {
            item.get("target_pref_name")
                .and_then(|v| v.as_str())
                .unwrap_or("ChEMBL-Target")
                .to_string()
        };

        let metric = item
            .get("standard_type")
            .and_then(|v| v.as_str())
            .unwrap_or("IC50")
            .to_string();

        // Prefer curated `pchembl_value`; otherwise compute from `standard_value` in nM
        let pact = if let Some(pc) = item.get("pchembl_value").and_then(val_as_f64) {
            pc
        } else if let Some(sv) = item.get("standard_value").and_then(val_as_f64) {
            let units = item
                .get("standard_units")
                .and_then(|v| v.as_str())
                .unwrap_or("nM");
            let nm = match units {
                "uM" | "µM" => sv * 1000.0,
                "pM" => sv * 0.001,
                _ => sv,
            };
            if nm > 0.0 {
                9.0 - nm.log10()
            } else {
                continue;
            }
        } else {
            continue;
        };

        let source_url = format!("https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}");
        raw_rows.push(RawBioRecord {
            id: chembl_id,
            name: pref_name,
            target,
            metric,
            smiles: smi.to_string(),
            pact,
            source_url,
        });
    }

    curate_and_deduplicate(
        raw_rows,
        max_compounds,
        "EMBL-EBI ChEMBL REST",
        "https://www.ebi.ac.uk/chembl/api/data/activity.json",
    )
}

/// Fetches active dose-response compounds from an NCBI PubChem BioAssay `AID` via `PUG-REST`:
/// 1. Downloads `/assay/aid/{aid}/concise/CSV` and filters quantitative `IC50`/`Ki`/`Kd`/`EC50` rows.
/// 2. Resolves unique `CID`s in batch via `/compound/cid/.../property/CanonicalSMILES,ConnectivitySMILES,Title/JSON`.
pub fn fetch_pubchem_bioassay_aid(
    aid: u64,
    target_label: &str,
    max_compounds: usize,
) -> Result<(Vec<Molecule>, IngestReport), IngestError> {
    let agent = build_http_agent();
    let max_mols = max_compounds.clamp(5, 300);
    let csv_url = format!("{PUBCHEM_PUG_BASE}/assay/aid/{aid}/concise/CSV");
    let csv_text = http_get_text(&agent, &csv_url)?;

    let cid_potencies = extract_active_cids_from_pubchem_concise_csv(&csv_text, max_mols * 2);
    if cid_potencies.is_empty() {
        return Err(IngestError::EmptyResult(format!(
            "PubChem AID {aid} returned 0 active quantitative CID rows"
        )));
    }

    let cids: Vec<u64> = cid_potencies.iter().map(|&(cid, _, _)| cid).collect();
    let mut prop_map: HashMap<u64, (String, String)> = HashMap::new();

    for chunk in cids.chunks(60) {
        let cid_list = chunk
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let prop_url = format!(
            "{PUBCHEM_PUG_BASE}/compound/cid/{cid_list}/property/CanonicalSMILES,ConnectivitySMILES,IsomericSMILES,Title/JSON"
        );
        if let Ok(prop_json) = http_get_text(&agent, &prop_url) {
            parse_pubchem_property_table_into(&prop_json, &mut prop_map);
        }
    }

    let mut raw_rows = Vec::new();
    let tgt = if target_label.trim().is_empty() {
        format!("PubChem-AID-{aid}")
    } else {
        target_label.trim().to_string()
    };

    for (cid, metric, pact) in cid_potencies {
        if let Some((smi, title)) = prop_map.get(&cid) {
            raw_rows.push(RawBioRecord {
                id: format!("CID{cid}"),
                name: if title.is_empty() {
                    format!("CID-{cid}")
                } else {
                    title.clone()
                },
                target: tgt.clone(),
                metric,
                smiles: smi.clone(),
                pact,
                source_url: format!("https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"),
            });
        }
    }

    let (mols, mut rep) =
        curate_and_deduplicate(raw_rows, max_mols, "NCBI PubChem PUG-REST", &csv_url)?;
    rep.endpoint_url = csv_url;
    Ok((mols, rep))
}

/// Fetches specific compounds by PubChem CID list or chemical/drug names (comma-separated,
/// e.g. `"erlotinib, gefitinib, osimertinib, lapatinib"`) via NCBI `PUG-REST`.
pub fn fetch_pubchem_compounds_by_identifiers(
    identifiers_csv: &str,
    target_label: &str,
    default_pact: f64,
) -> Result<(Vec<Molecule>, IngestReport), IngestError> {
    let agent = build_http_agent();
    let tokens: Vec<&str> = identifiers_csv
        .split(|c| c == ',' || c == ';' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err(IngestError::EmptyResult("Empty PubChem identifier list".into()));
    }

    let all_numeric = tokens
        .iter()
        .all(|t| t.strip_prefix("CID").unwrap_or(t).chars().all(|c| c.is_ascii_digit()));

    let mut raw_rows = Vec::new();
    let tgt = if target_label.trim().is_empty() {
        "PubChem-Import".to_string()
    } else {
        target_label.trim().to_string()
    };
    let mut first_url = String::new();

    if all_numeric {
        let cid_csv = tokens
            .iter()
            .map(|t| t.strip_prefix("CID").unwrap_or(t))
            .collect::<Vec<_>>()
            .join(",");
        let url = format!(
            "{PUBCHEM_PUG_BASE}/compound/cid/{cid_csv}/property/CanonicalSMILES,ConnectivitySMILES,IsomericSMILES,Title/JSON"
        );
        first_url = url.clone();
        let body = http_get_text(&agent, &url)?;
        let mut map = HashMap::new();
        parse_pubchem_property_table_into(&body, &mut map);
        for (cid, (smi, title)) in map {
            raw_rows.push(RawBioRecord {
                id: format!("CID{cid}"),
                name: if title.is_empty() {
                    format!("CID-{cid}")
                } else {
                    title
                },
                target: tgt.clone(),
                metric: "Assay".to_string(),
                smiles: smi,
                pact: default_pact,
                source_url: format!("https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"),
            });
        }
    } else {
        for tok in tokens.into_iter().take(40) {
            let encoded = percent_encode_path_segment(tok);
            let url = format!(
                "{PUBCHEM_PUG_BASE}/compound/name/{encoded}/property/CanonicalSMILES,ConnectivitySMILES,IsomericSMILES,Title/JSON"
            );
            if first_url.is_empty() {
                first_url = url.clone();
            }
            if let Ok(body) = http_get_text(&agent, &url) {
                let mut map = HashMap::new();
                parse_pubchem_property_table_into(&body, &mut map);
                if let Some((&cid, (smi, title))) = map.iter().next() {
                    raw_rows.push(RawBioRecord {
                        id: format!("CID{cid}"),
                        name: if title.is_empty() {
                            tok.to_string()
                        } else {
                            title.clone()
                        },
                        target: tgt.clone(),
                        metric: "Imported".to_string(),
                        smiles: smi.clone(),
                        pact: default_pact,
                        source_url: format!("https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"),
                    });
                }
            }
        }
    }

    curate_and_deduplicate(raw_rows, 200, "NCBI PubChem Compound Lookup", &first_url)
}

/// Parses a PubChem `PropertyTable` JSON payload (`CanonicalSMILES` / `ConnectivitySMILES` / `IsomericSMILES` + `Title`).
pub fn parse_pubchem_property_table_into(
    json_str: &str,
    out: &mut HashMap<u64, (String, String)>,
) {
    let Ok(root) = serde_json::from_str::<Value>(json_str) else {
        return;
    };
    let Some(props) = root
        .get("PropertyTable")
        .and_then(|pt| pt.get("Properties"))
        .and_then(|p| p.as_array())
    else {
        return;
    };

    for item in props {
        let Some(cid) = item.get("CID").and_then(|v| v.as_u64()) else {
            continue;
        };
        let smi = item
            .get("IsomericSMILES")
            .or_else(|| item.get("CanonicalSMILES"))
            .or_else(|| item.get("ConnectivitySMILES"))
            .or_else(|| item.get("SMILES"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if smi.is_empty() {
            continue;
        }
        let title = item
            .get("Title")
            .or_else(|| item.get("IUPACName"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        out.insert(cid, (smi, title));
    }
}

/// Parses PubChem `/concise/CSV` bioassay rows into `(cid, metric_name, pact)`.
pub fn extract_active_cids_from_pubchem_concise_csv(
    csv_text: &str,
    limit: usize,
) -> Vec<(u64, String, f64)> {
    let mut lines = csv_text.lines().filter(|l| !l.trim().is_empty());
    let Some(header_line) = lines.next() else {
        return Vec::new();
    };
    let headers = split_csv_row(header_line);
    let find_col = |pred: &dyn Fn(&str) -> bool| -> Option<usize> {
        headers.iter().position(|h| pred(&h.to_ascii_lowercase()))
    };

    let Some(cid_col) = find_col(&|h| h == "cid" || h == "pubchem_cid") else {
        return Vec::new();
    };
    let outcome_col = find_col(&|h| h.contains("activity outcome") || h == "pubchem_activity_outcome");
    let val_um_col = find_col(&|h| h.contains("activity value [um]") || h.contains("acvalue"));
    let val_nm_col = find_col(&|h| h.contains("activity value [nm]"));
    let name_col = find_col(&|h| h.contains("activity name"));

    let mut by_cid: BTreeMap<u64, (String, Vec<f64>)> = BTreeMap::new();

    for line in lines {
        let cols = split_csv_row(line);
        let Some(cid_str) = cols.get(cid_col) else {
            continue;
        };
        let Ok(cid) = cid_str.trim().parse::<u64>() else {
            continue;
        };
        if let Some(oc) = outcome_col {
            if let Some(out_s) = cols.get(oc) {
                let low = out_s.to_ascii_lowercase();
                if low.contains("inactive") || low.contains("inconclusive") {
                    continue;
                }
            }
        }

        let raw_nm = if let Some(um_i) = val_um_col.and_then(|c| cols.get(c)) {
            um_i.trim().parse::<f64>().ok().map(|v| v * 1000.0)
        } else if let Some(nm_i) = val_nm_col.and_then(|c| cols.get(c)) {
            nm_i.trim().parse::<f64>().ok()
        } else {
            None
        };

        let Some(nm) = raw_nm.filter(|&v| v > 0.0 && v < 1e7) else {
            continue;
        };
        let pact = (9.0 - nm.log10()).clamp(3.0, 12.5);
        let metric = name_col
            .and_then(|c| cols.get(c))
            .filter(|s| !s.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| "IC50".to_string());

        let entry = by_cid.entry(cid).or_insert_with(|| (metric, Vec::new()));
        entry.1.push(pact);
        if by_cid.len() >= limit {
            break;
        }
    }

    let mut out: Vec<(u64, String, f64)> = by_cid
        .into_iter()
        .map(|(cid, (metric, mut pacts))| {
            let med = median_f64(&mut pacts);
            (cid, metric, med)
        })
        .collect();
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Strips pharmaceutical salt counterions (`.Cl`, `.[Na+]`, `.CS(=O)(=O)O`, etc.) from a
/// multi-component SMILES string, returning `(largest_organic_fragment_smiles, was_desalted)`.
pub fn desalt_largest_organic_smiles(raw_smiles: &str) -> (String, bool) {
    let trimmed = raw_smiles.trim();
    if !trimmed.contains('.') {
        return (trimmed.to_string(), false);
    }
    let parts: Vec<&str> = trimmed.split('.').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 1 {
        return (trimmed.to_string(), false);
    }

    let mut best_part = parts[0];
    let mut best_score = 0usize;

    for &part in &parts {
        if let Ok(m) = parse_smiles(part) {
            let c_cnt = m.atoms.iter().filter(|a| a.elem == Element::C).count();
            let heavy = m.atoms.iter().filter(|a| a.elem != Element::H).count();
            let score = c_cnt * 100 + heavy;
            if score > best_score {
                best_score = score;
                best_part = part;
            }
        }
    }

    (best_part.to_string(), true)
}

fn curate_and_deduplicate(
    raw_rows: Vec<RawBioRecord>,
    max_compounds: usize,
    source_name: &str,
    endpoint_url: &str,
) -> Result<(Vec<Molecule>, IngestReport), IngestError> {
    let raw_count = raw_rows.len();
    let mut desalted_count = 0usize;

    // Group replicates by canonical OpenSMILES so identical structures merge via median pAct
    struct Group {
        mol: Molecule,
        pacts: Vec<f64>,
        base_metric: String,
    }
    let mut grouped: BTreeMap<String, Group> = BTreeMap::new();

    for rec in raw_rows {
        let (clean_smi, was_desalted) = desalt_largest_organic_smiles(&rec.smiles);
        if was_desalted {
            desalted_count += 1;
        }
        let Ok(mut mol) = parse_smiles(&clean_smi) else {
            continue;
        };
        let heavy = mol.atoms.iter().filter(|a| a.elem != Element::H).count();
        if !(4..=95).contains(&heavy) {
            continue;
        }
        let can_smi = molecule_to_smiles(&mol);
        mol.smiles = can_smi.clone();
        mol.id = rec.id;
        mol.name = rec.name;
        mol.target = rec.target;
        mol.source_url = rec.source_url;

        let g = grouped.entry(can_smi).or_insert_with(|| Group {
            mol,
            pacts: Vec::new(),
            base_metric: rec.metric,
        });
        g.pacts.push(rec.pact);
    }

    let mut curated: Vec<Molecule> = grouped
        .into_values()
        .map(|mut g| {
            let med = median_f64(&mut g.pacts);
            g.mol.pact = med;
            g.mol.raw_nm = 10f64.powf(9.0 - med.clamp(2.0, 14.0));
            g.mol.metric = if g.pacts.len() > 1 {
                format!("{} (n={})", g.base_metric, g.pacts.len())
            } else {
                g.base_metric
            };
            g.mol
        })
        .collect();

    curated.sort_by(|a, b| {
        b.pact
            .partial_cmp(&a.pact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    curated.truncate(max_compounds);

    if curated.is_empty() {
        return Err(IngestError::EmptyResult(format!(
            "{source_name} returned 0 parseable drug-like molecules (from {raw_count} raw records)"
        )));
    }

    let report = IngestReport {
        source_name: source_name.to_string(),
        endpoint_url: endpoint_url.to_string(),
        raw_records_scanned: raw_count,
        desalted_count,
        deduplicated_molecules: curated.len(),
    };
    Ok((curated, report))
}

fn val_as_f64(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_f64() {
        Some(n)
    } else if let Some(s) = v.as_str() {
        s.trim().parse::<f64>().ok()
    } else {
        None
    }
}

fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 7.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

fn title_case_drug_name(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.chars().any(|c| c.is_ascii_lowercase()) {
        return trimmed.to_string();
    }
    trimmed
        .split_whitespace()
        .map(|w| {
            let mut ch = w.chars();
            match ch.next() {
                None => String::new(),
                Some(first) => {
                    first.to_ascii_uppercase().to_string()
                        + &ch.as_str().to_ascii_lowercase()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_csv_row(line: &str) -> Vec<String> {
    let mut cols = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch == '"' {
            if in_quotes && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                cur.push('"');
                i += 2;
                continue;
            }
            in_quotes = !in_quotes;
        } else if ch == ',' && !in_quotes {
            cols.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
        i += 1;
    }
    cols.push(cur);
    cols
}

fn percent_encode_path_segment(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for b in input.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
