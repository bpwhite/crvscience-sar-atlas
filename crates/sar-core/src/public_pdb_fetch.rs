//! Public `RCSB PDB` & `AlphaFold Protein Structure Database` (`AFDB`) Live Coordinate, Co-Crystal
//! Ligand `SMILES`, Per-Residue `pLDDT` / `B-Factor` Confidence & `PAE` Domain Matrix Fetcher,
//! Loop-Trimming Receptor Sanitizer, and `Karelina-Dror` Binding-Pocket Docking Readiness Assessor
//! (`public_pdb_fetch.rs`, `v3.7.0`).
//!
//! Architectural Isolation Note:
//! - Connects exclusively to open public scientific repositories (`files.rcsb.org`, `data.rcsb.org`,
//!   and `alphafold.ebi.ac.uk`) with deterministic offline benchmark fallbacks and local `.pdb` /
//!   `PAE .json` file support; contains zero hooks or couplings to external applications.
//!
//! Primary References:
//! - Burley et al. (2023/2025), *J. Mol. Biol.* / *Nucleic Acids Res.* — `RCSB Protein Data Bank`
//!   REST Data, Chemical Component Dictionary (`CCD`), and Computed Structure Model APIs:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC12807596/` &
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11514064/`
//! - Varadi, Anyango, Deshpande, Nair, Natassia, Yordanova, Yuan, Stroe, Wood, Laydon, Zidek,
//!   Green, Tunyasuvunakool, Petersen, Jumper, Clancy, Green, Vora, Lutfi, Figurnov, Cowie,
//!   Hobbs, Kohli, Kleywegt, Birney, Hassabis & Velankar (2022/2025), *Nucleic Acids Res.* 50(D1),
//!   D439-D444 — `AlphaFold Protein Structure Database`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8728224/` &
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC12807749/`
//! - Karelina, Noh & Dror (2023), *Proc. Natl. Acad. Sci. U.S.A.* 120(48):e2303954120 ("How
//!   accurately can one predict drug binding modes using AlphaFold models?" — `1.3 A` median pocket
//!   backbone `RMSD` vs `apo`-like inward side-chain `chi1` occlusion requiring flexible-receptor
//!   repacking): `https://pmc.ncbi.nlm.nih.gov/articles/PMC10746139/`
//! - Jumper et al. (2021), *Nature* 596, 583-589 (`pLDDT` `[90+, 70-90, 50-70, <50]` per-residue
//!   confidence & `PAE` pairwise aligned error): `https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/`.

use crate::allosteric_pocket::curated_allosteric_pdb_text;
use crate::foldit_sculpt::curated_vhl_5nvx_pdb_text;
use crate::mol::{Element, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, ParsedPdbStructure,
    PocketCavityAnalysis,
};
use crate::watermap_solvation::curated_factor_xa_1ezq_pdb_text;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Default `pLDDT` threshold below which disordered terminal tails / loops are flagged for receptor trimming.
pub const DEFAULT_DISORDERED_PLDDT_CUTOFF: f32 = 55.0;

/// Public structural repository or offline benchmark mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicStructureSource {
    /// Live HTTP fetch from `RCSB PDB` (`files.rcsb.org` `.pdb` + `data.rcsb.org` entry & `chemcomp` ligand `SMILES`), with offline fallback.
    RcsbExperimentalPdb,
    /// Live HTTP fetch from `EMBL-EBI AlphaFold DB` (`alphafold.ebi.ac.uk/api/prediction/{UniProt}` `.pdb` + `PAE` JSON), with offline fallback.
    AlphaFoldPublicDb,
    /// Deterministic built-in benchmark comparison (Experimental `Holo` vs `AlphaFold DB` `pLDDT`/`PAE` profile, zero network needed).
    OfflineBenchmarkPreset,
    /// Local `.pdb` file (experimental `B-factor` or `AlphaFold` `pLDDT` in `B-factor` column) + optional local `PAE .json` file.
    LocalPdbAndPaeFiles,
}

impl PublicStructureSource {
    pub const ALL: [Self; 4] = [
        Self::OfflineBenchmarkPreset,
        Self::RcsbExperimentalPdb,
        Self::AlphaFoldPublicDb,
        Self::LocalPdbAndPaeFiles,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::OfflineBenchmarkPreset => "Curated PDB / AFDB Benchmark",
            Self::RcsbExperimentalPdb => "Live RCSB PDB Fetch",
            Self::AlphaFoldPublicDb => "Live AlphaFold DB (UniProt)",
            Self::LocalPdbAndPaeFiles => "Local .PDB + PAE .JSON",
        }
    }
}

/// Five built-in benchmark targets pairing `RCSB` experimental complexes with `AlphaFold DB` `pLDDT` / `PAE` profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicTargetPreset {
    /// `AF-P01116-F1` (`KRAS`, `UniProt P01116` vs `6OIM`): Classic `Karelina-Dror` case — `93` core `pLDDT`, `66` Switch-II loop `pLDDT`, & `His95/Tyr96` apo occlusion!
    KrasAfP01116SwitchIiOcclusion,
    /// `1M17` (`EGFR Kinase`, `RCSB` X-ray `2.60 A` + co-crystal `AQ4` Erlotinib, `UniProt P00533`).
    EgfrRcsb1m17ExperimentalHolo,
    /// `AF-P40337-F1` (`VHL E3 Ligase`, `UniProt P40337` vs `5NVX`): High-confidence `Tyr98/His115` beta-core (`pLDDT 92`) + N-terminal disordered tail (`pLDDT 41`) needing loop trim!
    VhlAfP40337DisorderedTailTrim,
    /// `1EZQ` (`Coagulation Factor Xa`, `RCSB` X-ray `2.20 A` + `RPR` ligand, `UniProt P00742`).
    FactorXaRcsb1ezqExperimentalHolo,
    /// `3LN1` (`COX-2`, `RCSB` X-ray `2.40 A` + `CEL` Celecoxib, `UniProt P35354`).
    Cox2Rcsb3ln1ExperimentalHolo,
}

impl PublicTargetPreset {
    pub const ALL: [Self; 5] = [
        Self::KrasAfP01116SwitchIiOcclusion,
        Self::EgfrRcsb1m17ExperimentalHolo,
        Self::VhlAfP40337DisorderedTailTrim,
        Self::FactorXaRcsb1ezqExperimentalHolo,
        Self::Cox2Rcsb3ln1ExperimentalHolo,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::KrasAfP01116SwitchIiOcclusion => "AF-P01116-KRAS",
            Self::EgfrRcsb1m17ExperimentalHolo => "RCSB-1M17-EGFR",
            Self::VhlAfP40337DisorderedTailTrim => "AF-P40337-VHL",
            Self::FactorXaRcsb1ezqExperimentalHolo => "RCSB-1EZQ-FXA",
            Self::Cox2Rcsb3ln1ExperimentalHolo => "RCSB-3LN1-COX2",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::KrasAfP01116SwitchIiOcclusion => "AFDB KRAS (P01116)",
            Self::EgfrRcsb1m17ExperimentalHolo => "RCSB EGFR (1M17)",
            Self::VhlAfP40337DisorderedTailTrim => "AFDB VHL (P40337)",
            Self::FactorXaRcsb1ezqExperimentalHolo => "RCSB FXa (1EZQ)",
            Self::Cox2Rcsb3ln1ExperimentalHolo => "RCSB COX-2 (3LN1)",
        }
    }

    pub fn default_pdb_id(self) -> &'static str {
        match self {
            Self::KrasAfP01116SwitchIiOcclusion => "6OIM",
            Self::EgfrRcsb1m17ExperimentalHolo => "1M17",
            Self::VhlAfP40337DisorderedTailTrim => "5NVX",
            Self::FactorXaRcsb1ezqExperimentalHolo => "1EZQ",
            Self::Cox2Rcsb3ln1ExperimentalHolo => "3LN1",
        }
    }

    pub fn default_uniprot_id(self) -> &'static str {
        match self {
            Self::KrasAfP01116SwitchIiOcclusion => "P01116",
            Self::EgfrRcsb1m17ExperimentalHolo => "P00533",
            Self::VhlAfP40337DisorderedTailTrim => "P40337",
            Self::FactorXaRcsb1ezqExperimentalHolo => "P00742",
            Self::Cox2Rcsb3ln1ExperimentalHolo => "P35354",
        }
    }
}

/// Canonical 4-band AlphaFold `pLDDT` / experimental precision tier ([Jumper et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidueConfidenceBand {
    /// `pLDDT >= 90` (or `B-factor <= 22 A^2`): atomic-accuracy backbone & reliable `chi1` rotamers.
    VeryHigh90Plus,
    /// `70 <= pLDDT < 90` (or `B-factor 22..38 A^2`): confident backbone fold; verify side-chain rotamers.
    Confident70To90,
    /// `50 <= pLDDT < 70` (or `B-factor 38..58 A^2`): flexible loop / Switch region; use Induced-Fit or 3D Sculpt.
    Low50To70,
    /// `pLDDT < 50` (or `B-factor > 58 A^2`): intrinsically disordered tail; trim if draping across pocket!
    DisorderedBelow50,
}

impl ResidueConfidenceBand {
    pub fn from_plddt(plddt: f32) -> Self {
        if plddt >= 90.0 {
            Self::VeryHigh90Plus
        } else if plddt >= 70.0 {
            Self::Confident70To90
        } else if plddt >= 50.0 {
            Self::Low50To70
        } else {
            Self::DisorderedBelow50
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::VeryHigh90Plus => "Very High (>=90)",
            Self::Confident70To90 => "Confident (70-90)",
            Self::Low50To70 => "Low / Loop (50-70)",
            Self::DisorderedBelow50 => "Disordered (<50)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::VeryHigh90Plus => [59, 130, 246],
            Self::Confident70To90 => [56, 189, 248],
            Self::Low50To70 => [250, 204, 21],
            Self::DisorderedBelow50 => [249, 115, 22],
        }
    }
}

/// Structure-based drug discovery readiness verdict (`Karelina, Noh & Dror 2023 PNAS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PocketReadinessVerdict {
    /// High pocket confidence (`pLDDT >= 86` or `Holo X-ray/EM <= 2.8 A`), low `PAE`, open cavity, zero disordered intrusions.
    RigidDockingAndWaterMapReady,
    /// Confident backbone (`pLDDT >= 72`), but `apo`-like bulky side chains or flexible Switch loop partially occlude the pocket -> run `Induced-Fit` or `Foldit 3D Sculpt` `Shake`!
    RequiresInducedFitOrSculptRepack,
    /// Disordered low-`pLDDT` (`< 55`) terminal tail or loop intrudes into the binding cleft -> export loop-trimmed `.pdb` first!
    TrimDisorderedLoopsBeforeDocking,
    /// Pocket backbone `pLDDT < 68` or inter-domain `PAE > 6.5 A` -> structural uncertainty high; prioritize ligand-based QSAR.
    LowConfidencePreferLigandQsar,
}

impl PocketReadinessVerdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::RigidDockingAndWaterMapReady => {
                "READY: High-Confidence Open Pocket (Direct Pocket / WaterMap / H-Bond Ready)"
            }
            Self::RequiresInducedFitOrSculptRepack => {
                "REPACK NEEDED: Apo Side-Chain / Loop Flexibility (Send to Induced-Fit or 3D Sculpt)"
            }
            Self::TrimDisorderedLoopsBeforeDocking => {
                "TRIM LOOPS FIRST: Low-pLDDT (<55) Intrusion in Cleft (Use Clean Trimmed .PDB)"
            }
            Self::LowConfidencePreferLigandQsar => {
                "CAUTION: Low Pocket Backbone Confidence / High PAE (Prefer Ligand QSAR)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::RigidDockingAndWaterMapReady => [52, 211, 153],
            Self::RequiresInducedFitOrSculptRepack => [250, 204, 21],
            Self::TrimDisorderedLoopsBeforeDocking => [251, 146, 60],
            Self::LowConfidencePreferLigandQsar => [244, 63, 94],
        }
    }
}

/// Per-residue confidence, crystallographic `RMSF`, pocket-lining status, and `apo`-occlusion diagnostic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidueConfidenceRow {
    pub chain_id: char,
    pub res_seq: i32,
    pub res_name: String,
    pub ca_pos: [f32; 3],
    pub raw_b_or_plddt: f32,
    /// Normalized `0..100` confidence (`pLDDT` directly for `AlphaFold`, or equivalent precision score for X-ray/cryo-EM).
    pub effective_plddt: f32,
    /// Estimated positional `RMSF` (`Angstrom`) from `B = 8*pi^2/3 * RMSF^2` or inverse `pLDDT`.
    pub est_rmsf_a: f32,
    pub band: ResidueConfidenceBand,
    pub lines_binding_pocket: bool,
    pub min_dist_to_pocket_center_a: f32,
    pub is_apo_occluding_sidechain: bool,
    pub is_disordered_pocket_intrusion: bool,
}

/// Co-crystallized ligand extracted from `HETATM` + enriched via `RCSB Chemical Component Dictionary` (`CCD`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedPdbLigand {
    pub comp_id: String,
    pub chemical_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub heavy_atom_count: usize,
    pub centroid: [f32; 3],
    pub smiles: String,
    pub mw: f64,
    pub clogp: f64,
    pub tpsa: f64,
    pub parsed_mol: Option<Molecule>,
}

/// Configuration for fetching and auditing a public `RCSB PDB` or `AlphaFold DB` structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicPdbFetchConfig {
    pub source: PublicStructureSource,
    pub preset: PublicTargetPreset,
    pub rcsb_pdb_id: String,
    pub uniprot_accession: String,
    pub local_pdb_path: Option<String>,
    pub local_pae_json_path: Option<String>,
    /// Allow graceful fallback to curated offline structure if live HTTP endpoint is unreachable.
    pub allow_offline_fallback: bool,
    /// Per-residue `pLDDT` threshold below which disordered residues can be stripped when exporting a clean receptor `.pdb`.
    pub trim_low_plddt_cutoff: f32,
}

impl Default for PublicPdbFetchConfig {
    fn default() -> Self {
        Self {
            source: PublicStructureSource::OfflineBenchmarkPreset,
            preset: PublicTargetPreset::KrasAfP01116SwitchIiOcclusion,
            rcsb_pdb_id: "1M17".to_string(),
            uniprot_accession: "P01116".to_string(),
            local_pdb_path: None,
            local_pae_json_path: None,
            allow_offline_fallback: true,
            trim_low_plddt_cutoff: DEFAULT_DISORDERED_PLDDT_CUTOFF,
        }
    }
}

/// Complete `RCSB PDB` / `AlphaFold DB` structural confidence, ligand extraction, `PAE`, and pocket readiness report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicStructureReadinessReport {
    pub accession_code: String,
    pub title: String,
    pub provenance_badge: String,
    pub source_url: String,
    pub is_alphafold_plddt_scale: bool,
    pub used_live_network: bool,
    pub experimental_method: String,
    pub resolution_a: Option<f64>,
    pub uniprot_id: String,
    pub gene_or_target: String,
    pub organism: String,
    pub total_residues: usize,
    pub global_mean_plddt: f64,
    pub pct_very_high_90: f64,
    pub pct_confident_70_90: f64,
    pub pct_low_50_70: f64,
    pub pct_disordered_lt50: f64,
    pub pocket_center: [f32; 3],
    pub pocket_volume_a3: f64,
    pub pocket_druggability_dscore: f64,
    pub pocket_lining_residue_count: usize,
    pub pocket_mean_plddt: f64,
    pub pocket_min_plddt: f64,
    /// Mean pairwise Predicted Aligned Error (`Angstrom`) across pocket-lining residues (`<= 3.5 A` = rigid single-domain cleft).
    pub pocket_mean_pae_a: f64,
    pub apo_occluding_residues: Vec<String>,
    pub disordered_intrusion_residues: Vec<String>,
    /// Composite `[0..100]` Pocket Docking Readiness Score (`Karelina-Dror` + `pLDDT` + `PAE` + druggability).
    pub readiness_score_0_100: f64,
    pub verdict: PocketReadinessVerdict,
    pub recommended_next_module: &'static str,
    pub residue_rows: Vec<ResidueConfidenceRow>,
    /// Downsampled `N x N` (`N <= 28`) `PAE` matrix (`Angstrom`) + axis residue numbers for fast heatmap visualization.
    pub pae_grid_size: usize,
    pub pae_axis_res_seqs: Vec<i32>,
    pub pae_matrix_a: Vec<f32>,
    pub extracted_ligands: Vec<ExtractedPdbLigand>,
    pub display_atoms: Vec<([f32; 3], f32, ResidueConfidenceBand, bool, bool)>,
    pub raw_pdb_text: String,
    pub trimmed_clean_pdb_text: String,
    pub trimmed_removed_atoms: usize,
    pub pocket_summary: PocketCavityAnalysis,
}

fn build_public_http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(8))
        .timeout_read(Duration::from_secs(16))
        .user_agent("sar-atlas/3.7.0 (public-data-only; standalone)")
        .build()
}

fn http_get_str(agent: &ureq::Agent, url: &str) -> Result<String, String> {
    agent
        .get(url)
        .set("Accept", "application/json, chemical/x-pdb, text/plain, */*")
        .call()
        .map_err(|e| format!("{url} -> {e}"))?
        .into_string()
        .map_err(|e| format!("{url} decode -> {e}"))
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Curated known ligand `SMILES` table for offline & fast CCD enrichment of common benchmark `HETATM` codes.
fn curated_ccd_smiles(comp_id: &str) -> Option<(&'static str, &'static str)> {
    match comp_id.trim().to_ascii_uppercase().as_str() {
        "AQ4" | "ERL" => Some((
            "Erlotinib (AQ4)",
            "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        )),
        "RPR" | "FXA" => Some((
            "Factor Xa Benzamidine-Lactam Inhibitor",
            "COc1ccc(C(=O)Nc2ccc(N3CCCCC3=O)cc2)cc1",
        )),
        "CEL" => Some((
            "Celecoxib (CEL)",
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        )),
        "VHL" | "VH2" | "997" => Some((
            "VH032 Hydroxyproline E3 Binder",
            "CC1=C(C2=CC=C(CNC(=O)C3CC(O)CN3C(=O)C(C)(C)C)C=C2)SC=N1",
        )),
        "MOV" | "AMG" | "SOT" => Some((
            "Sotorasib / AMG-510 KRAS G12C Binder",
            "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C)c3cc(F)ccc23)CC1",
        )),
        "8NU" | "RIS" => Some((
            "Risperidone (8NU)",
            "Cc1nc2ccccn2c(=O)c1CCN1CCC(c2noc3cc(F)ccc23)CC1",
        )),
        "STI" | "IMA" => Some((
            "Imatinib (STI)",
            "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        )),
        "ATP" => Some((
            "Adenosine-5'-Triphosphate (ATP)",
            "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O",
        )),
        _ => None,
    }
}

/// Generates realistic AlphaFold DB `.pdb` text (with `pLDDT` in the `B-factor` column, apo-occluding side chains, or disordered N-terminal tail) for offline benchmarks.
fn curated_offline_public_pdb(preset: PublicTargetPreset) -> (String, bool, String, Option<f64>) {
    match preset {
        PublicTargetPreset::KrasAfP01116SwitchIiOcclusion => {
            // Rewrite 6OIM protein lines as an AlphaFold AF-P01116-F1 apo prediction:
            // High core pLDDT (~93.4), Switch-II residues 60..100 have loop pLDDT (~66.8) and His95/Tyr96 shifted inward by 0.9 A!
            let base = curated_allosteric_pdb_text(1);
            let mut out = String::from(
                "HEADER    ALPHAFOLD MONOMER V4.0 PREDICTION       01-NOV-25   AF2K\nTITLE     ALPHAFOLD DB AF-P01116-F1 GTPase KRAS APO PREDICTION WITH SWITCH-II LOOP\n",
            );
            for line in base.lines() {
                if line.starts_with("ATOM  ") && line.len() >= 66 {
                    let res_seq: i32 = line[22..26].trim().parse().unwrap_or(1);
                    let res_nm = line[17..20].trim();
                    let is_sw2 = (58..=74).contains(&res_seq) || (94..=101).contains(&res_seq);
                    let plddt = if is_sw2 {
                        65.4 + ((res_seq % 5) as f32) * 1.6
                    } else {
                        92.8 + ((res_seq % 4) as f32) * 1.1
                    };
                    let mut x: f32 = line[30..38].trim().parse().unwrap_or(0.0);
                    let mut y: f32 = line[38..46].trim().parse().unwrap_or(0.0);
                    if matches!(res_nm, "HIS" | "TYR" | "MET") && is_sw2 {
                        // Apo inward chi1 swing partially occluding Switch-II groove (Karelina-Dror 2023)!
                        x += 0.75;
                        y -= 0.55;
                    }
                    out.push_str(&format!(
                        "{}{:>8.3}{:>8.3}{}{:>6.2}{}\n",
                        &line[..30],
                        x,
                        y,
                        &line[46..60],
                        plddt,
                        &line[66..]
                    ));
                }
            }
            out.push_str("END\n");
            (
                out,
                true,
                "AlphaFold v2/v4 Monomer Prediction (AF-P01116-F1)".to_string(),
                None,
            )
        }
        PublicTargetPreset::VhlAfP40337DisorderedTailTrim => {
            // Rewrite 5NVX protein lines as AF-P40337-F1 + append 4 N-terminal disordered acidic-domain residues (res 50..53, pLDDT 38..46) draping near the cleft!
            let base = curated_vhl_5nvx_pdb_text();
            let mut out = String::from(
                "HEADER    ALPHAFOLD MONOMER V4.0 PREDICTION       01-NOV-25   AF2V\nTITLE     ALPHAFOLD DB AF-P40337-F1 VON HIPPEL-LINDAU E3 LIGASE WITH N-TERM TAIL\n",
            );
            out.push_str(
                "ATOM      1  CA  GLU A  51       7.250   0.850   2.250  1.00 41.20           C\n",
            );
            out.push_str(
                "ATOM      2  CG  GLU A  51       7.680   0.520   1.920  1.00 39.80           C\n",
            );
            out.push_str(
                "ATOM      3  CA  ALA A  52       8.350   1.350   2.050  1.00 44.60           C\n",
            );
            out.push_str(
                "ATOM      4  CA  GLY A  53       6.450   1.550   1.650  1.00 47.90           C\n",
            );
            let mut serial = 5_u32;
            for line in base.lines() {
                if line.starts_with("ATOM  ") && line.len() >= 66 {
                    let res_seq: i32 = line[22..26].trim().parse().unwrap_or(90);
                    let plddt = 91.2 + ((res_seq % 5) as f32) * 1.1;
                    out.push_str(&format!(
                        "ATOM  {:>5}{}{:>6.2}{}\n",
                        serial,
                        &line[11..60],
                        plddt,
                        &line[66..]
                    ));
                    serial += 1;
                }
            }
            out.push_str("END\n");
            (
                out,
                true,
                "AlphaFold v2/v4 Monomer Prediction (AF-P40337-F1)".to_string(),
                None,
            )
        }
        PublicTargetPreset::EgfrRcsb1m17ExperimentalHolo => (
            curated_pocket_pdb_text(0).to_string(),
            false,
            "X-RAY DIFFRACTION (RCSB 1M17)".to_string(),
            Some(2.60),
        ),
        PublicTargetPreset::FactorXaRcsb1ezqExperimentalHolo => (
            curated_factor_xa_1ezq_pdb_text().to_string(),
            false,
            "X-RAY DIFFRACTION (RCSB 1EZQ)".to_string(),
            Some(2.20),
        ),
        PublicTargetPreset::Cox2Rcsb3ln1ExperimentalHolo => (
            curated_pocket_pdb_text(2).to_string(),
            false,
            "X-RAY DIFFRACTION (RCSB 3LN1)".to_string(),
            Some(2.40),
        ),
    }
}

/// Parses an AlphaFold DB `PAE` JSON payload (`[{"predicted_aligned_error": [[...]]}]` or `{"pae": [[...]]}`) into a downsampled `N x N` matrix (`N <= 24`).
fn parse_or_synthesize_pae_matrix(
    pae_json_opt: Option<&str>,
    residue_rows: &[ResidueConfidenceRow],
    is_alphafold: bool,
) -> (usize, Vec<i32>, Vec<f32>) {
    let n_res = residue_rows.len();
    if n_res == 0 {
        return (0, Vec::new(), Vec::new());
    }
    let grid_n = n_res.min(22);
    let mut axis_seqs = Vec::with_capacity(grid_n);
    let mut sample_indices = Vec::with_capacity(grid_n);
    for g in 0..grid_n {
        let idx = if grid_n <= 1 {
            0
        } else {
            (g * (n_res - 1)) / (grid_n - 1)
        };
        sample_indices.push(idx);
        axis_seqs.push(residue_rows[idx].res_seq);
    }

    if let Some(raw_json) = pae_json_opt {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_json) {
            let root = if let Some(arr) = val.as_array() {
                arr.first().unwrap_or(&val)
            } else {
                &val
            };
            let mat_opt = root
                .get("predicted_aligned_error")
                .or_else(|| root.get("pae"))
                .and_then(|v| v.as_array());
            if let Some(rows_arr) = mat_opt {
                let full_len = rows_arr.len().max(1);
                let mut out = vec![0.0_f32; grid_n * grid_n];
                for (gi, &ri) in sample_indices.iter().enumerate() {
                    let src_i = ((ri * full_len) / n_res.max(1)).min(full_len - 1);
                    if let Some(r_arr) = rows_arr.get(src_i).and_then(|v| v.as_array()) {
                        for (gj, &rj) in sample_indices.iter().enumerate() {
                            let src_j = ((rj * r_arr.len().max(1)) / n_res.max(1))
                                .min(r_arr.len().saturating_sub(1));
                            let pae_v = r_arr
                                .get(src_j)
                                .and_then(|x| x.as_f64())
                                .unwrap_or(2.5) as f32;
                            out[gi * grid_n + gj] = pae_v.clamp(0.2, 31.75);
                        }
                    }
                }
                return (grid_n, axis_seqs, out);
            }
        }
    }

    // Synthesize physically consistent pairwise positional uncertainty from residue `pLDDT` / `RMSF` + 3D spatial separation!
    let mut out = vec![0.0_f32; grid_n * grid_n];
    for (gi, &ri) in sample_indices.iter().enumerate() {
        for (gj, &rj) in sample_indices.iter().enumerate() {
            if gi == gj {
                out[gi * grid_n + gj] = if is_alphafold { 0.65 } else { 0.40 };
                continue;
            }
            let a = &residue_rows[ri];
            let b = &residue_rows[rj];
            let d3 = dist3(a.ca_pos, b.ca_pos);
            let unc_a = (100.0 - a.effective_plddt).max(3.0) * 0.16;
            let unc_b = (100.0 - b.effective_plddt).max(3.0) * 0.16;
            let domain_term = (d3 * 0.065).clamp(0.2, 4.5);
            let pae_est = (0.55 + unc_a + unc_b + domain_term).clamp(0.5, 30.0);
            out[gi * grid_n + gj] = pae_est;
        }
    }
    (grid_n, axis_seqs, out)
}

/// Strips disordered residues with `pLDDT < cutoff` (when `is_alphafold` is true) or ultra-high `B-factor > 75 A^2` and formats a clean `.pdb`.
pub fn filter_pdb_removing_disordered_loops(
    raw_pdb: &str,
    is_alphafold: bool,
    plddt_cutoff: f32,
) -> (String, usize) {
    let mut out = String::new();
    let mut removed = 0_usize;
    for line in raw_pdb.lines() {
        if line.starts_with("ATOM  ") && line.len() >= 66 {
            let b_val: f32 = line[60..66].trim().parse().unwrap_or(50.0);
            let drop = if is_alphafold {
                b_val < plddt_cutoff
            } else {
                b_val > 78.0
            };
            if drop {
                removed += 1;
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    (out, removed)
}

/// Fetches (from live `RCSB PDB` / `AlphaFold DB` or local/offline preset) and performs a full `pLDDT` / `PAE` / `Karelina-Dror` pocket readiness audit.
pub fn fetch_and_assess_public_structure(
    cfg: &PublicPdbFetchConfig,
) -> Result<PublicStructureReadinessReport, String> {
    let agent = build_public_http_agent();
    let mut raw_pdb = String::new();
    let mut pae_json: Option<String> = None;
    let mut is_af = false;
    let mut used_live = false;
    let mut method_str = String::new();
    let mut resolution_opt: Option<f64> = None;
    let mut accession = String::new();
    let mut source_url = String::new();
    let mut provenance = String::new();
    let mut uniprot_id = cfg.uniprot_accession.trim().to_ascii_uppercase();
    let mut gene_name = String::from("Target");
    let mut organism = String::from("Homo sapiens");
    let mut rcsb_chemcomp_smiles: BTreeMap<String, (String, String)> = BTreeMap::new();

    match cfg.source {
        PublicStructureSource::RcsbExperimentalPdb => {
            let code = cfg.rcsb_pdb_id.trim().to_ascii_uppercase();
            let pdb_code = if code.len() == 4 {
                code
            } else {
                cfg.preset.default_pdb_id().to_string()
            };
            let pdb_url = format!("https://files.rcsb.org/download/{pdb_code}.pdb");
            match http_get_str(&agent, &pdb_url) {
                Ok(txt) if txt.contains("ATOM  ") => {
                    raw_pdb = txt;
                    used_live = true;
                    accession = pdb_code.clone();
                    source_url = format!("https://www.rcsb.org/structure/{pdb_code}");
                    provenance = format!("Live RCSB PDB ({pdb_code})");
                    method_str = "EXPERIMENTAL PDB".to_string();

                    // Query RCSB Data REST API for resolution, experimental method & co-crystal chemcomp IDs
                    let entry_url = format!("https://data.rcsb.org/rest/v1/core/entry/{pdb_code}");
                    if let Ok(entry_json) = http_get_str(&agent, &entry_url) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&entry_json) {
                            if let Some(m) = v
                                .pointer("/exptl/0/method")
                                .and_then(|x| x.as_str())
                            {
                                method_str = m.to_string();
                            }
                            resolution_opt = v
                                .pointer("/rcsb_entry_info/resolution_combined/0")
                                .and_then(|x| x.as_f64());
                            if let Some(comps) = v
                                .pointer("/rcsb_entry_container_identifiers/non_polymer_entity_ids")
                                .and_then(|x| x.as_array())
                            {
                                let _ = comps;
                            }
                        }
                    }
                    // Parse temporary PDB to discover co-crystal ligand 3-letter codes and query CCD SMILES!
                    if let Ok(tmp_p) = parse_pdb_structure(&raw_pdb) {
                        for lig in &tmp_p.cocrystal_ligands {
                            let cid = lig.res_name.trim().to_ascii_uppercase();
                            if rcsb_chemcomp_smiles.contains_key(&cid) {
                                continue;
                            }
                            let ccd_url =
                                format!("https://data.rcsb.org/rest/v1/core/chemcomp/{cid}");
                            if let Ok(ccd_j) = http_get_str(&agent, &ccd_url) {
                                if let Ok(cv) = serde_json::from_str::<serde_json::Value>(&ccd_j) {
                                    let cname = cv
                                        .pointer("/chem_comp/name")
                                        .and_then(|x| x.as_str())
                                        .unwrap_or(&cid)
                                        .to_string();
                                    let smi = cv
                                        .pointer("/rcsb_chem_comp_descriptor/smilesstereo")
                                        .or_else(|| cv.pointer("/rcsb_chem_comp_descriptor/smiles"))
                                        .and_then(|x| x.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    if !smi.is_empty() {
                                        rcsb_chemcomp_smiles.insert(cid, (cname, smi));
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) if !cfg.allow_offline_fallback => {
                    return Err(format!("RCSB PDB live fetch failed for {pdb_code}: {e}"));
                }
                _ => {
                    let (fb_txt, fb_af, fb_m, fb_res) =
                        curated_offline_public_pdb(PublicTargetPreset::EgfrRcsb1m17ExperimentalHolo);
                    raw_pdb = fb_txt;
                    is_af = fb_af;
                    method_str = fb_m;
                    resolution_opt = fb_res;
                    accession = "1M17".to_string();
                    uniprot_id = "P00533".to_string();
                    gene_name = "EGFR".to_string();
                    source_url = "https://www.rcsb.org/structure/1M17".to_string();
                    provenance = "Offline Fallback (RCSB 1M17 Cached Preset)".to_string();
                }
            }
        }
        PublicStructureSource::AlphaFoldPublicDb => {
            let acc = if uniprot_id.is_empty() {
                "P01116".to_string()
            } else {
                uniprot_id.clone()
            };
            let api_url = format!("https://alphafold.ebi.ac.uk/api/prediction/{acc}");
            let mut pdb_dl_url = format!("https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb");
            let mut pae_dl_url: Option<String> = None;

            if let Ok(meta_txt) = http_get_str(&agent, &api_url) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&meta_txt) {
                    let rec = v.as_array().and_then(|a| a.first()).unwrap_or(&v);
                    if let Some(pu) = rec.get("pdbUrl").and_then(|x| x.as_str()) {
                        pdb_dl_url = pu.to_string();
                    }
                    if let Some(pau) = rec.get("paeDocUrl").and_then(|x| x.as_str()) {
                        pae_dl_url = Some(pau.to_string());
                    }
                    if let Some(g) = rec.get("gene").and_then(|x| x.as_str()) {
                        gene_name = g.to_string();
                    }
                    if let Some(org) = rec.get("organismScientificName").and_then(|x| x.as_str()) {
                        organism = org.to_string();
                    }
                }
            }

            match http_get_str(&agent, &pdb_dl_url) {
                Ok(txt) if txt.contains("ATOM  ") => {
                    raw_pdb = txt;
                    is_af = true;
                    used_live = true;
                    accession = format!("AF-{acc}-F1");
                    uniprot_id = acc.clone();
                    source_url = format!("https://alphafold.ebi.ac.uk/entry/{acc}");
                    provenance = format!("Live AlphaFold DB (AF-{acc}-F1)");
                    method_str = "ALPHAFOLD MONOMER PREDICTION".to_string();
                    if let Some(ref pau) = pae_dl_url {
                        pae_json = http_get_str(&agent, pau).ok();
                    }
                }
                Err(e) if !cfg.allow_offline_fallback => {
                    return Err(format!("AlphaFold DB live fetch failed for {acc}: {e}"));
                }
                _ => {
                    let (fb_txt, fb_af, fb_m, fb_res) =
                        curated_offline_public_pdb(PublicTargetPreset::KrasAfP01116SwitchIiOcclusion);
                    raw_pdb = fb_txt;
                    is_af = fb_af;
                    method_str = fb_m;
                    resolution_opt = fb_res;
                    accession = "AF-P01116-F1".to_string();
                    uniprot_id = "P01116".to_string();
                    gene_name = "KRAS".to_string();
                    source_url = "https://alphafold.ebi.ac.uk/entry/P01116".to_string();
                    provenance = "Offline Fallback (AlphaFold DB AF-P01116-F1 Preset)".to_string();
                }
            }
        }
        PublicStructureSource::LocalPdbAndPaeFiles => {
            if let Some(ref lp) = cfg.local_pdb_path {
                let p_trim = lp.trim();
                if !p_trim.is_empty() {
                    if let Ok(txt) = std::fs::read_to_string(p_trim) {
                        raw_pdb = txt;
                    }
                }
            }
            if raw_pdb.is_empty() {
                if !cfg.allow_offline_fallback {
                    return Err("Local .pdb path is empty or unreadable".to_string());
                }
                let (fb_txt, fb_af, fb_m, fb_res) = curated_offline_public_pdb(cfg.preset);
                raw_pdb = fb_txt;
                is_af = fb_af;
                method_str = fb_m;
                resolution_opt = fb_res;
                accession = cfg.preset.code().to_string();
                provenance = format!("Curated Fallback ({})", cfg.preset.short_label());
                source_url = format!(
                    "https://alphafold.ebi.ac.uk/entry/{}",
                    cfg.preset.default_uniprot_id()
                );
            } else {
                let up_head = raw_pdb.lines().take(12).collect::<Vec<_>>().join(" ").to_ascii_uppercase();
                is_af = up_head.contains("ALPHAFOLD") || up_head.contains("AF-");
                accession = "LOCAL-PDB".to_string();
                provenance = "Local Structure File".to_string();
                method_str = if is_af {
                    "LOCAL ALPHAFOLD MODEL".to_string()
                } else {
                    "LOCAL EXPERIMENTAL PDB".to_string()
                };
                source_url = "local://pdb".to_string();
                if let Some(ref pae_p) = cfg.local_pae_json_path {
                    if !pae_p.trim().is_empty() {
                        pae_json = std::fs::read_to_string(pae_p.trim()).ok();
                    }
                }
            }
        }
        PublicStructureSource::OfflineBenchmarkPreset => {
            let (fb_txt, fb_af, fb_m, fb_res) = curated_offline_public_pdb(cfg.preset);
            raw_pdb = fb_txt;
            is_af = fb_af;
            method_str = fb_m;
            resolution_opt = fb_res;
            accession = cfg.preset.code().to_string();
            uniprot_id = cfg.preset.default_uniprot_id().to_string();
            gene_name = cfg.preset.short_label().to_string();
            provenance = format!("Built-In Public Benchmark ({})", cfg.preset.short_label());
            source_url = if is_af {
                format!("https://alphafold.ebi.ac.uk/entry/{uniprot_id}")
            } else {
                format!(
                    "https://www.rcsb.org/structure/{}",
                    cfg.preset.default_pdb_id()
                )
            };
        }
    }

    let parsed: ParsedPdbStructure = parse_pdb_structure(&raw_pdb)?;
    let center_override = if cfg.source == PublicStructureSource::OfflineBenchmarkPreset
        && cfg.preset == PublicTargetPreset::KrasAfP01116SwitchIiOcclusion
    {
        Some([12.90_f32, 5.10, 8.20])
    } else if cfg.source == PublicStructureSource::OfflineBenchmarkPreset
        && cfg.preset == PublicTargetPreset::VhlAfP40337DisorderedTailTrim
    {
        Some([7.85_f32, 0.95, 2.05])
    } else {
        None
    };
    let pocket = analyze_binding_pocket(&parsed, center_override, 9.8);

    // Group protein atoms by `(chain_id, res_seq)` to compute per-residue pLDDT / B-factor & pocket occlusion!
    let mut res_groups: BTreeMap<(char, i32), (String, Vec<[f32; 3]>, Option<[f32; 3]>, f32)> =
        BTreeMap::new();
    for pa in &parsed.protein_atoms {
        let entry = res_groups
            .entry((pa.chain_id, pa.res_seq))
            .or_insert_with(|| (pa.res_name.clone(), Vec::new(), None, pa.b_factor));
        entry.1.push(pa.pos);
        if pa.atom_name == "CA" {
            entry.2 = Some(pa.pos);
            entry.3 = pa.b_factor;
        }
    }

    let mut residue_rows: Vec<ResidueConfidenceRow> = Vec::with_capacity(res_groups.len());
    let mut n_vh = 0_usize;
    let mut n_cf = 0_usize;
    let mut n_lo = 0_usize;
    let mut n_di = 0_usize;
    let mut sum_global_plddt = 0.0_f64;
    let mut sum_pocket_plddt = 0.0_f64;
    let mut min_pocket_plddt = 100.0_f64;
    let mut pocket_res_cnt = 0_usize;
    let mut apo_occluding: Vec<String> = Vec::new();
    let mut disordered_intrusions: Vec<String> = Vec::new();

    for (&(ch, rseq), (rname, atom_pts, ca_opt, b_or_plddt)) in &res_groups {
        let ca = ca_opt.unwrap_or_else(|| atom_pts[0]);
        let (eff_plddt, rmsf_a) = if is_af {
            let p = b_or_plddt.clamp(0.0, 100.0);
            let r = (0.25 + (100.0 - p) * 0.032).clamp(0.20, 3.50);
            (p, r)
        } else {
            // Convert experimental B-factor (A^2) to RMSF = sqrt(3*B / (8*pi^2)) and map onto an equivalent [25..98] confidence scale
            let b_clamped = b_or_plddt.clamp(4.0, 120.0);
            let rmsf = ((3.0 * b_clamped) / (8.0 * std::f32::consts::PI * std::f32::consts::PI)).sqrt();
            let eq_plddt = (104.0 - b_clamped * 0.72).clamp(28.0, 98.5);
            (eq_plddt, rmsf)
        };

        let band = ResidueConfidenceBand::from_plddt(eff_plddt);
        match band {
            ResidueConfidenceBand::VeryHigh90Plus => n_vh += 1,
            ResidueConfidenceBand::Confident70To90 => n_cf += 1,
            ResidueConfidenceBand::Low50To70 => n_lo += 1,
            ResidueConfidenceBand::DisorderedBelow50 => n_di += 1,
        }
        sum_global_plddt += eff_plddt as f64;

        let min_d = atom_pts
            .iter()
            .map(|&pt| dist3(pt, pocket.center))
            .fold(f32::MAX, f32::min);
        let lines_pocket = min_d <= 6.80;
        let bulky = matches!(
            rname.as_str(),
            "TYR" | "PHE" | "TRP" | "MET" | "HIS" | "ARG" | "LYS" | "GLN" | "LEU"
        );
        let is_occluding = is_af
            && parsed.cocrystal_ligands.is_empty()
            && bulky
            && min_d <= 3.35;
        let is_intrusion = lines_pocket && eff_plddt < cfg.trim_low_plddt_cutoff;

        if lines_pocket {
            pocket_res_cnt += 1;
            sum_pocket_plddt += eff_plddt as f64;
            min_pocket_plddt = min_pocket_plddt.min(eff_plddt as f64);
        }
        if is_occluding {
            apo_occluding.push(format!("{rname}{rseq}"));
        }
        if is_intrusion {
            disordered_intrusions.push(format!("{rname}{rseq} (pLDDT {eff_plddt:.0})"));
        }

        residue_rows.push(ResidueConfidenceRow {
            chain_id: ch,
            res_seq: rseq,
            res_name: rname.clone(),
            ca_pos: ca,
            raw_b_or_plddt: *b_or_plddt,
            effective_plddt: eff_plddt,
            est_rmsf_a: rmsf_a,
            band,
            lines_binding_pocket: lines_pocket,
            min_dist_to_pocket_center_a: min_d,
            is_apo_occluding_sidechain: is_occluding,
            is_disordered_pocket_intrusion: is_intrusion,
        });
    }

    let tot_res = residue_rows.len().max(1);
    let global_mean_plddt = sum_global_plddt / (tot_res as f64);
    let pocket_mean_plddt = if pocket_res_cnt > 0 {
        sum_pocket_plddt / (pocket_res_cnt as f64)
    } else {
        global_mean_plddt
    };
    if pocket_res_cnt == 0 {
        min_pocket_plddt = global_mean_plddt;
    }

    // Parse or compute PAE matrix + mean pocket-wall PAE!
    let (pae_n, pae_axes, pae_mat) =
        parse_or_synthesize_pae_matrix(pae_json.as_deref(), &residue_rows, is_af);
    let mut pae_pocket_sum = 0.0_f64;
    let mut pae_pocket_pairs = 0_usize;
    if pae_n > 0 {
        for i in 0..pae_n {
            for j in 0..pae_n {
                if i != j {
                    let ri_in = residue_rows
                        .iter()
                        .find(|r| r.res_seq == pae_axes[i])
                        .map(|r| r.lines_binding_pocket)
                        .unwrap_or(false);
                    let rj_in = residue_rows
                        .iter()
                        .find(|r| r.res_seq == pae_axes[j])
                        .map(|r| r.lines_binding_pocket)
                        .unwrap_or(false);
                    if ri_in && rj_in {
                        pae_pocket_sum += pae_mat[i * pae_n + j] as f64;
                        pae_pocket_pairs += 1;
                    }
                }
            }
        }
    }
    let pocket_mean_pae_a = if pae_pocket_pairs > 0 {
        pae_pocket_sum / (pae_pocket_pairs as f64)
    } else if is_af {
        2.65
    } else {
        1.15
    };

    // Extract co-crystallized HETATM ligands & enrich with live CCD or curated SMILES!
    let mut extracted_ligs: Vec<ExtractedPdbLigand> = Vec::new();
    for cl in &parsed.cocrystal_ligands {
        let cid = cl.res_name.trim().to_ascii_uppercase();
        let (cname, smi_s) = if let Some((n, s)) = rcsb_chemcomp_smiles.get(&cid) {
            (n.clone(), s.clone())
        } else if let Some((n, s)) = curated_ccd_smiles(&cid) {
            (n.to_string(), s.to_string())
        } else {
            (format!("PDB Ligand {cid}"), String::new())
        };
        let mut parsed_m: Option<Molecule> = None;
        let (mut mw, mut clogp, mut tpsa) = ((cl.atoms.len() as f64) * 13.5, 2.2, 55.0);
        if !smi_s.is_empty() {
            if let Ok(mut m) = parse_smiles(&smi_s) {
                m.id = format!("{}-{}", parsed.pdb_id, cid);
                m.name = cname.clone();
                let ph = evaluate_physchem(&m);
                mw = ph.mw;
                clogp = ph.clogp;
                tpsa = ph.tpsa;
                parsed_m = Some(m);
            }
        }
        extracted_ligs.push(ExtractedPdbLigand {
            comp_id: cid,
            chemical_name: cname,
            chain_id: cl.chain_id,
            res_seq: cl.res_seq,
            heavy_atom_count: cl.atoms.len(),
            centroid: cl.centroid,
            smiles: smi_s,
            mw,
            clogp,
            tpsa,
            parsed_mol: parsed_m,
        });
    }

    // Karelina-Dror Pocket Docking Readiness Verdict & [0..100] Score
    let holo_bonus = if !extracted_ligs.is_empty() { 10.0 } else { 0.0 };
    let occ_pen = (apo_occluding.len() as f64) * 8.5;
    let dis_pen = (disordered_intrusions.len() as f64) * 14.0;
    let pae_pen = (pocket_mean_pae_a - 2.5).max(0.0) * 3.8;
    let readiness_score = (pocket_mean_plddt * 0.72
        + pocket.druggability_dscore * 22.0
        + holo_bonus
        - occ_pen
        - dis_pen
        - pae_pen)
        .clamp(8.0, 99.4);

    let (verdict, next_mod) = if !disordered_intrusions.is_empty() {
        (
            PocketReadinessVerdict::TrimDisorderedLoopsBeforeDocking,
            "Export Loop-Trimmed Clean .PDB -> Pocket / WaterMap",
        )
    } else if pocket_mean_plddt < 68.0 || pocket_mean_pae_a > 6.5 {
        (
            PocketReadinessVerdict::LowConfidencePreferLigandQsar,
            "Active-Learning GP/Ensemble QSAR (Domain 1)",
        )
    } else if !apo_occluding.is_empty() || (is_af && pocket_mean_plddt < 86.0) {
        (
            PocketReadinessVerdict::RequiresInducedFitOrSculptRepack,
            "Induced-Fit Ensemble or Foldit 3D Sculpt (Shake)",
        )
    } else {
        (
            PocketReadinessVerdict::RigidDockingAndWaterMapReady,
            "WaterMap Solvation & His/Asp/Glu H-Bond Network",
        )
    };

    let (trimmed_pdb, removed_atoms) =
        filter_pdb_removing_disordered_loops(&raw_pdb, is_af, cfg.trim_low_plddt_cutoff);

    // Compact 3D display atom buffer (`pos`, `plddt`, `band`, `is_occluding`, `is_intrusion`)
    let mut disp_atoms: Vec<([f32; 3], f32, ResidueConfidenceBand, bool, bool)> = Vec::new();
    for pa in parsed.protein_atoms.iter().take(320) {
        if matches!(pa.elem, Element::H) {
            continue;
        }
        let r_row = residue_rows
            .iter()
            .find(|r| r.chain_id == pa.chain_id && r.res_seq == pa.res_seq);
        let (pl, bd, occ, intr) = r_row
            .map(|r| {
                (
                    r.effective_plddt,
                    r.band,
                    r.is_apo_occluding_sidechain,
                    r.is_disordered_pocket_intrusion,
                )
            })
            .unwrap_or((85.0, ResidueConfidenceBand::Confident70To90, false, false));
        disp_atoms.push((pa.pos, pl, bd, occ, intr));
    }

    Ok(PublicStructureReadinessReport {
        accession_code: if accession.is_empty() {
            parsed.pdb_id.clone()
        } else {
            accession
        },
        title: parsed.title.clone(),
        provenance_badge: provenance,
        source_url,
        is_alphafold_plddt_scale: is_af,
        used_live_network: used_live,
        experimental_method: method_str,
        resolution_a: resolution_opt,
        uniprot_id,
        gene_or_target: gene_name,
        organism,
        total_residues: residue_rows.len(),
        global_mean_plddt,
        pct_very_high_90: (n_vh as f64) * 100.0 / (tot_res as f64),
        pct_confident_70_90: (n_cf as f64) * 100.0 / (tot_res as f64),
        pct_low_50_70: (n_lo as f64) * 100.0 / (tot_res as f64),
        pct_disordered_lt50: (n_di as f64) * 100.0 / (tot_res as f64),
        pocket_center: pocket.center,
        pocket_volume_a3: pocket.volume_angstrom3,
        pocket_druggability_dscore: pocket.druggability_dscore,
        pocket_lining_residue_count: pocket_res_cnt,
        pocket_mean_plddt,
        pocket_min_plddt,
        pocket_mean_pae_a,
        apo_occluding_residues: apo_occluding,
        disordered_intrusion_residues: disordered_intrusions,
        readiness_score_0_100: readiness_score,
        verdict,
        recommended_next_module: next_mod,
        residue_rows,
        pae_grid_size: pae_n,
        pae_axis_res_seqs: pae_axes,
        pae_matrix_a: pae_mat,
        extracted_ligands: extracted_ligs,
        display_atoms: disp_atoms,
        raw_pdb_text: raw_pdb,
        trimmed_clean_pdb_text: trimmed_pdb,
        trimmed_removed_atoms: removed_atoms,
        pocket_summary: pocket,
    })
}

/// Exports per-residue `pLDDT` / `B-factor`, `RMSF`, and binding-pocket readiness diagnostics to `.csv`.
pub fn export_public_structure_confidence_csv(rep: &PublicStructureReadinessReport) -> String {
    let mut out = String::from(
        "accession,chain,res_seq,res_name,raw_b_or_plddt,effective_plddt,est_rmsf_a,confidence_band,lines_pocket,dist_to_pocket_a,apo_occluding,disordered_intrusion\n",
    );
    for r in &rep.residue_rows {
        out.push_str(&format!(
            "{},{},{},{},{:.2},{:.2},{:.3},\"{}\",{},{:.2},{},{}\n",
            rep.accession_code,
            r.chain_id,
            r.res_seq,
            r.res_name,
            r.raw_b_or_plddt,
            r.effective_plddt,
            r.est_rmsf_a,
            r.band.short_label(),
            r.lines_binding_pocket,
            r.min_dist_to_pocket_center_a,
            r.is_apo_occluding_sidechain,
            r.is_disordered_pocket_intrusion
        ));
    }
    out
}
