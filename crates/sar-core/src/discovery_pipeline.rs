//! End-to-End Configurable Multi-Round Closed-Loop `DMTA` Molecule Discovery Funnel,
//! Custom `.pdb` & Modality Branch Orchestrator (`FBDD`, `Macrocycle`, `pKa`, `Covalent GSH`,
//! `Allosteric MWC`), `Bemis-Murcko` / `MaxMin` Diversity Guardrail, Live Async Progress Emitter,
//! and `JSON` Recipe + `SQLite 3` Run-Ledger Engine (`discovery_pipeline.rs`, `v3.8.0`).
//!
//! Primary References:
//! - Kumar & Zhang (2015), *Methods* 71, 26-37 ("Hierarchical virtual screening approaches in small
//!   molecule drug discovery": `https://pmc.ncbi.nlm.nih.gov/articles/PMC7129923/`) & Reproducible
//!   Hierarchical ML + Docking Cascades (`https://pmc.ncbi.nlm.nih.gov/articles/PMC13292220/`)
//! - Closed-Loop Multi-Objective Generative Design-Make-Test-Analyze (`DMTA`) Cycles
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC13538683/`, Plowright et al. 2012:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC7805875/`)
//! - Graff, Shakhnovich & Coley (2021) `MolPAL` Pool Active Learning (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8188596/`)
//! - Cascading On-Target `L-RB-FEP+` & Off-Target Selectivity Funnels (`https://pmc.ncbi.nlm.nih.gov/articles/PMC12381238/`),
//!   `Gu & Shoichet (2021)` `TEU` Strain Filtering (`https://pubmed.ncbi.nlm.nih.gov/34467754/`),
//!   `Bemis & Murcko (1996)` Frameworks (`https://pubmed.ncbi.nlm.nih.gov/8709122/`), & `MaxMin`
//!   Sphere-Exclusion Diversity (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8067665/`).

use crate::active_learning::{
    run_active_learning_campaign, AcquisitionStrategy, ActiveLearningConfig, CandidatePoolSource,
    QsarKernelMode, SurrogateModelKind,
};
use crate::admet::{evaluate_admet, BoiledEggRegion};
use crate::alerts::scan_structural_alerts;
use crate::allosteric_pocket::{load_curated_allosteric_system, score_ligand_allosteric_coupling};
use crate::bioisostere::generate_bioisosteres;
use crate::conformer3d::{
    compute_gasteiger_charges, evaluate_3d_summary_profile, format_3d_conformer_sdf,
};
use crate::covalent_warhead::evaluate_covalent_summary;
use crate::de_novo_gen::{evolve_de_novo_molecules, DeNovoConfig, DeNovoGoalPreset};
use crate::fingerprint::{compute_ecfp, Ecfp2048};
use crate::fragment_grower::{grow_fragment_along_vector, RT_LN10_KCAL};
use crate::fragment_linker::{design_linked_fragments, FragmentLinkerConfig, FragmentPairPreset};
use crate::hbond_network::{
    optimize_hbond_network_with_prebuilt, HBondNetworkConfig, HBondNetworkPreset,
};
use crate::electrostatics_esp::{
    analyze_esp_with_prebuilt, ElectrostaticsEspConfig, EspTargetPreset,
};
use crate::watermap_solvation::{
    analyze_watermap_with_prebuilt, WaterMapConfig, WaterMapPreset,
};
use crate::induced_fit::{
    build_receptor_ensemble, perceive_flexible_pocket_residues,
    score_ligand_with_prebuilt_ensemble, InducedFitConfig, InducedFitPreset,
};
use crate::macrocycle::design_macrocycles;
use crate::mol::Molecule;
use crate::mol_io::parse_smiles;
use crate::parallel_lib::{
    enumerate_parallel_library, LibraryPreFilterMode, LibrarySynthesisMode, ParallelLibConfig,
};
use crate::physchem::evaluate_physchem;
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
};
use crate::rbfe_network::{
    run_rbfe_network_campaign, RbfeNetworkConfig, RbfeSeriesPreset, RbfeTopologyMode,
};
use crate::scaffold::murcko_signatures;
use crate::selectivity_screen::{
    analyze_dual_pocket_selectivity, load_curated_selectivity_pair, score_ligand_dual_selectivity,
    CURATED_SELECTIVITY_PAIRS,
};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::torsion_strain::{analyze_torsion_strain_and_locks, BioactivePoseSource};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Target-aligned campaign preset coupling 3D Pocket, Off-Target Selectivity Pair, DEL Scheme, Allosteric System & RBFE Series.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryFunnelPreset {
    AutoDatasetCampaign,
    EgfrKinasePanMutant,
    Cox2IsoformSelective,
    Drd2CnsBbbSelective,
}

impl DiscoveryFunnelPreset {
    pub const ALL: [Self; 4] = [
        Self::AutoDatasetCampaign,
        Self::EgfrKinasePanMutant,
        Self::Cox2IsoformSelective,
        Self::Drd2CnsBbbSelective,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::AutoDatasetCampaign => "AUTO-CAMPAIGN",
            Self::EgfrKinasePanMutant => "EGFR-1M17-FUNNEL",
            Self::Cox2IsoformSelective => "COX2-3LN1-FUNNEL",
            Self::Drd2CnsBbbSelective => "DRD2-6CM4-FUNNEL",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::AutoDatasetCampaign => "Auto Campaign",
            Self::EgfrKinasePanMutant => "EGFR Kinase (1M17)",
            Self::Cox2IsoformSelective => "COX-2 Selective (3LN1)",
            Self::Drd2CnsBbbSelective => "DRD2 CNS BBB (6CM4)",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AutoDatasetCampaign => {
                "Auto-Detected Dominant Target Campaign from Active Dataset"
            }
            Self::EgfrKinasePanMutant => {
                "EGFR ATP-Hinge Lead Optimization (1M17 Pocket + T790M Gatekeeper + Lomap RBFE)"
            }
            Self::Cox2IsoformSelective => {
                "COX-2 Isoform-Selective Discovery (3LN1 vs COX-1 3KK6 Anti-Target + Lomap RBFE)"
            }
            Self::Drd2CnsBbbSelective => {
                "Dopamine D2 CNS BBB Discovery (6CM4 vs DRD3 3PBL Paralog + CNS-MPO + Lomap RBFE)"
            }
        }
    }

    /// Returns `(pocket_idx, sel_pair_idx, plib_scheme_idx, allosteric_idx, rbfe_series, pose_source)`.
    pub fn resolve_indices(
        self,
        dominant_target: &str,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        RbfeSeriesPreset,
        BioactivePoseSource,
    ) {
        let effective = match self {
            Self::AutoDatasetCampaign => {
                let up = dominant_target.to_ascii_uppercase();
                if up.contains("COX") || up.contains("PTGS") {
                    Self::Cox2IsoformSelective
                } else if up.contains("DRD") || up.contains("D2") || up.contains("DOPAMINE") {
                    Self::Drd2CnsBbbSelective
                } else {
                    Self::EgfrKinasePanMutant
                }
            }
            other => other,
        };
        match effective {
            Self::EgfrKinasePanMutant => (
                0, // 1M17 EGFR
                1, // EGFR-WT-VS-T790M
                0, // Kinase DEL/Parallel scheme 0
                0, // 5D41 EGFR allosteric system
                RbfeSeriesPreset::Egfr1m17Quinazolines,
                BioactivePoseSource::PocketEgfr1m17,
            ),
            Self::Cox2IsoformSelective => (
                2, // 3LN1 COX-2
                0, // COX2-VS-COX1
                1, // Suzuki/Amide diaryl scheme 1
                2, // 3K5V allosteric system fallback
                RbfeSeriesPreset::Cox23ln1Diarylpyrazoles,
                BioactivePoseSource::PocketCox23ln1,
            ),
            Self::Drd2CnsBbbSelective | Self::AutoDatasetCampaign => (
                1, // 6CM4 DRD2
                2, // DRD2-VS-DRD3
                2, // Buchwald/Click amine scheme 2
                1, // 6OIM Switch-II cryptic system fallback
                RbfeSeriesPreset::Drd26cm4Arylpiperazines,
                BioactivePoseSource::PocketDrd26cm4,
            ),
        }
    }
}

/// Origin stream that contributed a candidate molecule into Stage 1 of the discovery funnel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunnelSourceStream {
    DatasetLead,
    DeNovoEvolution,
    ParallelDelLibrary,
    BioisostereSwap,
    RotorConformationLock,
    FbddFragmentGrow,
    FbddFragmentLink,
    MacrocycleClosure,
    CustomUserSmiles,
}

impl FunnelSourceStream {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::DatasetLead => "Dataset",
            Self::DeNovoEvolution => "DeNovo-GA",
            Self::ParallelDelLibrary => "DEL/Plate",
            Self::BioisostereSwap => "Bioisostere",
            Self::RotorConformationLock => "Rotor-Lock",
            Self::FbddFragmentGrow => "FBDD-Grow",
            Self::FbddFragmentLink => "FBDD-Link",
            Self::MacrocycleClosure => "Macrocycle",
            Self::CustomUserSmiles => "Custom-SMI",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::DatasetLead => [148, 163, 184],          // Slate
            Self::DeNovoEvolution => [52, 211, 153],       // Emerald
            Self::ParallelDelLibrary => [192, 132, 252],   // Violet
            Self::BioisostereSwap => [56, 189, 248],       // Sky Blue
            Self::RotorConformationLock => [251, 191, 36], // Amber
            Self::FbddFragmentGrow => [45, 212, 191],      // Teal
            Self::FbddFragmentLink => [34, 211, 238],      // Cyan
            Self::MacrocycleClosure => [251, 146, 60],     // Orange
            Self::CustomUserSmiles => [244, 114, 182],     // Pink
        }
    }
}

/// Live background progress telemetry emitted during multi-round / multi-stage funnel execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineProgressUpdate {
    pub current_round: usize,
    pub total_rounds: usize,
    pub stage_num: u8,
    pub stage_label: String,
    pub fraction_complete: f32,
}

/// Per-round convergence telemetry across closed-loop iterative `DMTA` cycles (`Round 1..=R`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DmtaRoundSummary {
    pub round: usize,
    pub seed_ids: Vec<String>,
    pub pool_size: usize,
    pub stage2_passed: usize,
    pub stage3_passed: usize,
    pub stage4_passed: usize,
    pub rbfe_finalists: usize,
    pub promoted_count: usize,
    pub unique_promoted_scaffolds: usize,
    pub surrogate_loo_q2: f64,
    pub best_rbfe_dg_kcal: f64,
    pub best_predicted_pact: f64,
    pub mean_promoted_pact: f64,
    pub mean_promoted_lle: f64,
    pub mean_promoted_strain_teu: f64,
}

/// Configurable recipe for the `v3.1.0` multi-round, custom-PDB, modality-aware discovery funnel.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscoveryPipelineConfig {
    pub recipe_name: String,
    pub preset: DiscoveryFunnelPreset,
    pub seed_mol_idx: Option<usize>,

    /// Number of autonomous closed-loop `DMTA` rounds (`1..=5`, default `2`).
    /// For `dmta_rounds >= 2`, Round `r` survivors re-seed Round `r+1` generation & `GP+RF` retraining!
    pub dmta_rounds: usize,

    /// Optional filesystem paths to custom target and off-target `.pdb` structures.
    pub custom_target_pdb_path: Option<String>,
    pub custom_offtarget_pdb_path: Option<String>,

    // Stage 1 Intake Stream Toggles (8 streams)
    pub include_dataset_compounds: bool,
    pub include_de_novo_evolution: bool,
    pub include_parallel_del_plate: bool,
    pub include_bioisosteres: bool,
    pub include_rotor_locks: bool,
    pub include_fbdd_fragment_growing: bool,
    pub include_fbdd_fragment_linking: bool,
    pub include_macrocyclization: bool,
    pub custom_input_smiles: Vec<String>,

    // Stage 2: PAINS, QED, ESOL, SAScore & pH 7.4 Ionization / Protomer Gate
    pub require_zero_pains: bool,
    pub min_qed: f64,
    pub max_sa_score: f64,
    pub min_esol_logs: f64,
    pub enforce_ph74_charge_gate: bool,
    pub max_abs_charge_74: i8,

    // Stage 3: 3D PDB Pocket + Shoichet Torsion TEU + Optional Covalent GSH Reactivity Gate
    pub max_pocket_dg_kcal: f64,
    pub enable_induced_fit_ensemble: bool,
    pub optimize_pocket_hbond_network: bool,
    pub enable_watermap_solvation: bool,
    pub enable_esp_multipole_scoring: bool,
    pub max_total_strain_teu: f64,
    pub max_single_bond_teu: f64,
    pub require_covalent_warhead_window: bool,
    pub min_gsh_half_life_h: f64,
    pub max_gsh_half_life_h: f64,

    // Stage 4: Dual-Pocket Off-Target Selectivity + Optional Allosteric GNM/ANM Cooperativity Branch
    pub min_selectivity_delta_pkd: f64,
    pub enable_allosteric_coupling_branch: bool,

    // Stage 5 & Stage 6: Active-Learning Finalist Cap + Murcko / MaxMin Diversity Promotion Guardrail
    pub max_rbfe_finalists: usize,
    pub final_promote_count: usize,
    pub max_per_murcko_scaffold: usize,
    pub min_promoted_ecfp4_dissim: f64,
}

impl Default for DiscoveryPipelineConfig {
    fn default() -> Self {
        Self {
            recipe_name: "EGFR-MultiRound-LeadOpt-v3.1".to_string(),
            preset: DiscoveryFunnelPreset::AutoDatasetCampaign,
            seed_mol_idx: None,
            dmta_rounds: 2,
            custom_target_pdb_path: None,
            custom_offtarget_pdb_path: None,
            include_dataset_compounds: true,
            include_de_novo_evolution: true,
            include_parallel_del_plate: true,
            include_bioisosteres: true,
            include_rotor_locks: true,
            include_fbdd_fragment_growing: true,
            include_fbdd_fragment_linking: true,
            include_macrocyclization: false,
            custom_input_smiles: Vec::new(),
            require_zero_pains: true,
            min_qed: 0.30,
            max_sa_score: 4.95,
            min_esol_logs: -6.60,
            enforce_ph74_charge_gate: true,
            max_abs_charge_74: 1,
            max_pocket_dg_kcal: -5.80,
            enable_induced_fit_ensemble: false,
            optimize_pocket_hbond_network: true,
            enable_watermap_solvation: true,
            enable_esp_multipole_scoring: true,
            max_total_strain_teu: 6.50,
            max_single_bond_teu: 2.10,
            require_covalent_warhead_window: false,
            min_gsh_half_life_h: 0.25,
            max_gsh_half_life_h: 24.0,
            min_selectivity_delta_pkd: -0.55,
            enable_allosteric_coupling_branch: false,
            max_rbfe_finalists: 14,
            final_promote_count: 8,
            max_per_murcko_scaffold: 3,
            min_promoted_ecfp4_dissim: 0.14,
        }
    }
}

/// Stage-level attrition telemetry (`Stage 1` through `Stage 6`) for the final (or single) DMTA round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunnelStageSummary {
    pub stage_num: u8,
    #[serde(skip_deserializing)]
    pub stage_code: &'static str,
    #[serde(skip_deserializing)]
    pub stage_title: &'static str,
    pub gate_criteria: String,
    pub input_count: usize,
    pub passed_count: usize,
    pub dropped_count: usize,
    pub stage_pass_pct: f64,
    pub cumulative_yield_pct: f64,
}

/// Full per-molecule provenance, modality readouts, Murcko diversity audit, and promotion verdict.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunnelCandidateRecord {
    pub rank: usize,
    pub dmta_round_born: usize,
    pub candidate_id: String,
    pub candidate_name: String,
    pub source_stream: FunnelSourceStream,
    pub provenance_detail: String,
    pub canonical_smiles: String,
    pub murcko_scaffold: String,

    pub furthest_stage_reached: u8,
    pub promoted_survivor: bool,
    pub diversity_pick_order: Option<usize>,
    pub nearest_promoted_tanimoto: f32,
    pub status_summary: String,

    // Stage 2: 2D MedChem, ADMET, SAScore & pH 7.4 Ionization
    pub mw: f64,
    pub clogp: f64,
    pub tpsa: f64,
    pub qed: f64,
    pub cns_mpo: f64,
    pub esol_logs: f64,
    pub boiled_egg: BoiledEggRegion,
    pub sa_score: f64,
    pub pains_alerts: usize,
    pub dominant_charge_74: i8,
    pub logd_74: f64,

    // Stage 3: 3D Pocket, Shoichet Torsion TEU & Covalent GSH
    pub pocket_dg_kcal: Option<f64>,
    pub pocket_complementarity: Option<f64>,
    pub displaced_unhappy_waters: Option<usize>,
    pub total_strain_teu: Option<f64>,
    pub max_single_bond_teu: Option<f64>,
    pub covalent_warhead_name: Option<String>,
    pub covalent_gsh_half_life_h: Option<f64>,
    pub bioactive_coords3d: Vec<[f32; 3]>,

    // Stage 4: Dual-Pocket Off-Target Selectivity & Allosteric Cooperativity
    pub offtarget_dg_kcal: Option<f64>,
    pub ddg_selectivity_kcal: Option<f64>,
    pub selectivity_delta_pkd: Option<f64>,
    pub selectivity_fold: Option<f64>,
    pub allosteric_alpha: Option<f64>,
    pub allosteric_ddg_coop_kcal: Option<f64>,

    // Stage 5: Active-Learning GP+RF QSAR
    pub al_predicted_pact: Option<f64>,
    pub al_epistemic_std: Option<f64>,
    pub al_expected_improvement: Option<f64>,

    // Stage 6: Lomap BAR/MLE RBFE Finalist Reconciliation
    pub rbfe_mle_dg_kcal: Option<f64>,
    pub rbfe_ddg_vs_ref_kcal: Option<f64>,
    pub rbfe_predicted_pact: Option<f64>,
    pub rbfe_lomap_degree: Option<usize>,

    pub composite_funnel_score: f64,
    pub mol: Molecule,
}

/// Compact persistent run-ledger row stored in `SQLite 3` (`pipeline_runs` table) or JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineRunLedgerEntry {
    pub run_id: i64,
    pub epoch_seconds: u64,
    pub recipe_name: String,
    pub campaign_code: String,
    pub resolved_target: String,
    pub target_pocket_pdb: String,
    pub dmta_rounds: usize,
    pub initial_pool_size: usize,
    pub promoted_count: usize,
    pub unique_promoted_scaffolds: usize,
    pub best_rbfe_dg_kcal: f64,
    pub mean_promoted_pact: f64,
    pub promoted_smiles_joined: String,
}

/// Complete `v3.1.0` multi-round 6-stage discovery pipeline execution report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveryPipelineReport {
    pub resolved_target: String,
    pub target_pocket_pdb: String,
    pub selectivity_pair_code: String,
    pub pdb_source_note: String,
    pub seed_compound_id: String,
    pub seed_compound_name: String,
    pub config: DiscoveryPipelineConfig,
    pub initial_pool_size: usize,
    pub stage4_survivor_count: usize,
    pub rbfe_finalist_count: usize,
    pub promoted_survivor_count: usize,
    pub unique_promoted_scaffolds: usize,
    pub overall_yield_pct: f64,
    pub round_history: Vec<DmtaRoundSummary>,
    pub stage_summaries: Vec<FunnelStageSummary>,
    pub candidates: Vec<FunnelCandidateRecord>,
}

impl DiscoveryPipelineReport {
    /// Extracts the `Molecule` structs of all promoted synthesis survivors (`promoted_survivor == true`),
    /// annotated with their `RBFE`/`Active-Learning` predicted `pAct`, ready for `ds.merge_imported_molecules(...)`.
    pub fn promoted_molecules(&self) -> Vec<Molecule> {
        self.candidates
            .iter()
            .filter(|c| c.promoted_survivor)
            .map(|c| {
                let mut m = c.mol.clone();
                m.target = self.resolved_target.clone();
                m.metric = "Pred_pAct".to_string();
                let p = c
                    .rbfe_predicted_pact
                    .or(c.al_predicted_pact)
                    .unwrap_or(m.pact.max(7.0));
                m.pact = p;
                m.raw_nm = (10.0_f64.powf(9.0 - p)).clamp(0.01, 100_000.0);
                if m.source_url.is_empty() {
                    m.source_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7129923/".to_string();
                }
                m
            })
            .collect()
    }
}

/// Convenience wrapper executing `run_discovery_pipeline_with_progress` with a no-op progress listener.
pub fn run_discovery_pipeline(
    ds: &DatasetAnalysis,
    cfg: &DiscoveryPipelineConfig,
) -> DiscoveryPipelineReport {
    run_discovery_pipeline_with_progress(ds, cfg, |_| {})
}

/// Executes the `v3.1.0` multi-round closed-loop `DMTA` discovery cascade while streaming live
/// `PipelineProgressUpdate` events for non-blocking GUI progress bars and CLI monitors.
pub fn run_discovery_pipeline_with_progress<F>(
    ds: &DatasetAnalysis,
    cfg: &DiscoveryPipelineConfig,
    mut on_progress: F,
) -> DiscoveryPipelineReport
where
    F: FnMut(PipelineProgressUpdate),
{
    let total_rounds = cfg.dmta_rounds.clamp(1, 5);

    // 0. Resolve dominant target, preset indices, and custom vs built-in .pdb structures
    let dominant_target = {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for m in &ds.mols {
            let t = if m.target.trim().is_empty() {
                "EGFR".to_string()
            } else {
                m.target.trim().to_string()
            };
            *counts.entry(t).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(t, _)| t)
            .unwrap_or_else(|| "EGFR".to_string())
    };

    let (pocket_idx, sel_pair_idx, plib_scheme_idx, allo_idx, rbfe_series, pose_src) =
        cfg.preset.resolve_indices(&dominant_target);

    let mut resolved_target = match pocket_idx {
        0 => "EGFR".to_string(),
        1 => "DRD2".to_string(),
        _ => "COX-2".to_string(),
    };

    // Load target .pdb (custom path if provided and readable, else curated preset)
    let mut pdb_notes = Vec::new();
    let target_pdb_struct = match cfg
        .custom_target_pdb_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(custom_p) => match std::fs::read_to_string(custom_p)
            .map_err(|e| e.to_string())
            .and_then(|txt| parse_pdb_structure(&txt))
        {
            Ok(parsed) => {
                resolved_target = parsed.pdb_id.clone();
                pdb_notes.push(format!("Custom Target PDB '{}' ({})", custom_p, parsed.pdb_id));
                parsed
            }
            Err(err) => {
                pdb_notes.push(format!(
                    "Custom Target PDB '{}' fallback to preset ({})",
                    custom_p, err
                ));
                parse_pdb_structure(curated_pocket_pdb_text(pocket_idx))
                    .expect("Valid curated PDB")
            }
        },
        None => {
            let p = parse_pdb_structure(curated_pocket_pdb_text(pocket_idx))
                .expect("Valid curated PDB");
            pdb_notes.push(format!("Built-in Target PDB {}", p.pdb_id));
            p
        }
    };
    let target_pocket = analyze_binding_pocket(&target_pdb_struct, None, 10.0);
    let ifd_prebuilt = if cfg.enable_induced_fit_ensemble {
        let flex_r = perceive_flexible_pocket_residues(&target_pdb_struct, target_pocket.center, 6);
        let (_, ens_mems) = build_receptor_ensemble(&target_pdb_struct, None, &flex_r, 298.15);
        Some((flex_r, ens_mems))
    } else {
        None
    };

    // Load dual-pocket selectivity comparison (custom off-target .pdb if provided, else curated pair)
    let (dual_selectivity, selectivity_pair_code) = match cfg
        .custom_offtarget_pdb_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(off_p) => match std::fs::read_to_string(off_p)
            .map_err(|e| e.to_string())
            .and_then(|txt| parse_pdb_structure(&txt))
        {
            Ok(off_parsed) => {
                let code = format!("{}-VS-{}", target_pdb_struct.pdb_id, off_parsed.pdb_id);
                pdb_notes.push(format!(
                    "Custom Off-Target PDB '{}' ({})",
                    off_p, off_parsed.pdb_id
                ));
                (
                    analyze_dual_pocket_selectivity(&target_pdb_struct, &off_parsed, 10.0),
                    code,
                )
            }
            Err(err) => {
                pdb_notes.push(format!(
                    "Custom Off-Target PDB '{}' fallback ({})",
                    off_p, err
                ));
                (
                    load_curated_selectivity_pair(sel_pair_idx, false, 10.0)
                        .expect("Valid selectivity pair"),
                    CURATED_SELECTIVITY_PAIRS[sel_pair_idx].code.to_string(),
                )
            }
        },
        None => (
            load_curated_selectivity_pair(sel_pair_idx, false, 10.0)
                .expect("Valid selectivity pair"),
            CURATED_SELECTIVITY_PAIRS[sel_pair_idx].code.to_string(),
        ),
    };

    let allo_network = if cfg.enable_allosteric_coupling_branch {
        load_curated_allosteric_system(allo_idx, 7.5).ok()
    } else {
        None
    };

    // Pick initial Round-1 seed lead
    let seed_idx = cfg.seed_mol_idx.unwrap_or_else(|| {
        let mut best_i = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        let tok = resolved_target
            .chars()
            .take(3)
            .collect::<String>()
            .to_ascii_uppercase();
        for (i, m) in ds.mols.iter().enumerate() {
            let tgt_bonus = if !tok.is_empty() && m.target.to_ascii_uppercase().contains(&tok) {
                2.0
            } else {
                0.0
            };
            let sc = m.pact + tgt_bonus;
            if sc > best_score {
                best_score = sc;
                best_i = i;
            }
        }
        best_i
    });

    let initial_seed_mol = ds
        .mols
        .get(seed_idx)
        .cloned()
        .unwrap_or_else(|| parse_smiles("COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC").expect("Seed SMILES"));
    let seed_compound_id = initial_seed_mol.id.clone();
    let seed_compound_name = initial_seed_mol.name.clone();

    let mut working_ds = ds.clone();
    let mut round_seeds: Vec<Molecule> = vec![initial_seed_mol];
    let mut global_seen_promoted: BTreeSet<String> = BTreeSet::new();
    let mut round_history: Vec<DmtaRoundSummary> = Vec::with_capacity(total_rounds);

    let mut final_stage_summaries: Vec<FunnelStageSummary> = Vec::new();
    let mut final_records: Vec<FunnelCandidateRecord> = Vec::new();
    let mut final_initial_pool = 0usize;
    let mut final_s4_survivors = 0usize;
    let mut final_rbfe_finalists = 0usize;
    let mut final_promoted_count = 0usize;
    let mut final_unique_scaffolds = 0usize;

    // =========================================================================
    // MULTI-ROUND CLOSED-LOOP DMTA ITERATION (Round 1 ..= total_rounds)
    // =========================================================================
    for round_idx in 1..=total_rounds {
        let emit = |on_p: &mut F, stg: u8, lbl: &str| {
            let done_units = ((round_idx - 1) * 6 + (stg as usize).saturating_sub(1)) as f32;
            let tot_units = (total_rounds * 6).max(1) as f32;
            on_p(PipelineProgressUpdate {
                current_round: round_idx,
                total_rounds,
                stage_num: stg,
                stage_label: format!("Round {round_idx}/{total_rounds} — {lbl}"),
                fraction_complete: (done_units / tot_units).clamp(0.0, 0.99),
            });
        };

        // ---------------------------------------------------------------------
        // STAGE 1: Multi-Stream Intake & Modality Expansion
        // ---------------------------------------------------------------------
        emit(&mut on_progress, 1, "Stage 1: Pool Expansion (DeNovo, DEL, FBDD, Locks)");

        let mut raw_intake: Vec<(FunnelSourceStream, String, Molecule)> = Vec::new();
        let mut seen_smiles: BTreeSet<String> = BTreeSet::new();

        fn push_unique_intake(
            raw_intake: &mut Vec<(FunnelSourceStream, String, Molecule)>,
            seen_smiles: &mut BTreeSet<String>,
            stream: FunnelSourceStream,
            prov: String,
            mut m: Molecule,
            id_fallback: &str,
        ) {
            m.sanitize();
            let can = molecule_to_smiles(&m);
            if !can.is_empty() && seen_smiles.insert(can.clone()) {
                m.smiles = can;
                if m.id.trim().is_empty() {
                    m.id = id_fallback.to_string();
                }
                if m.name.trim().is_empty() {
                    m.name = id_fallback.to_string();
                }
                raw_intake.push((stream, prov, m));
            }
        }
        macro_rules! push_unique {
            ($stream:expr, $prov:expr, $m:expr, $id_fallback:expr $(,)?) => {
                push_unique_intake(
                    &mut raw_intake,
                    &mut seen_smiles,
                    $stream,
                    $prov,
                    $m,
                    $id_fallback,
                )
            };
        }

        if cfg.include_dataset_compounds && round_idx == 1 {
            for m in working_ds.mols.iter().take(16) {
                push_unique!(
                    FunnelSourceStream::DatasetLead,
                    format!("Dataset ({})", m.target),
                    m.clone(),
                    &m.id.clone(),
                );
            }
        } else if round_idx > 1 {
            // Carry forward prior round's seed champions so Round R compares strictly >= Round R-1!
            for sm in &round_seeds {
                push_unique!(
                    FunnelSourceStream::DatasetLead,
                    format!("DMTA R{} Carryover Champion", round_idx - 1),
                    sm.clone(),
                    &sm.id.clone(),
                );
            }
        }

        for (si, s_mol) in round_seeds.iter().enumerate() {
            let r_tag = format!("R{round_idx}");
            if cfg.include_de_novo_evolution {
                let dn_rep = evolve_de_novo_molecules(
                    s_mol,
                    Some(&working_ds),
                    DeNovoConfig {
                        goal: if round_idx % 2 == 1 {
                            DeNovoGoalPreset::BalancedLeadDiscovery
                        } else {
                            DeNovoGoalPreset::PotencyAndEfficiency
                        },
                        generations: 3,
                        population_size: 12,
                        max_returned: 10,
                        random_seed: 0xD17A_0000_u64
                            .wrapping_add((round_idx as u64) * 977 + (si as u64) * 31),
                        ..DeNovoConfig::default()
                    },
                );
                for c in dn_rep.candidates.into_iter().take(9) {
                    let mut cm = c.mol;
                    cm.id = format!("{r_tag}-DN{:02}", c.rank + si * 10);
                    cm.name = format!("{} {} ({})", r_tag, c.last_operator.short_label(), s_mol.id);
                    push_unique!(
                        FunnelSourceStream::DeNovoEvolution,
                        format!("{} {} (Seed {})", r_tag, c.last_operator.short_label(), s_mol.id),
                        cm,
                        "DN",
                    );
                }
            }

            if cfg.include_bioisosteres {
                let bios =
                    generate_bioisosteres(s_mol, &working_ds, working_ds.default_sar_matrix.as_ref());
                for (bi, b) in bios.into_iter().take(6).enumerate() {
                    let mut bm = b.mol;
                    bm.id = format!("{r_tag}-BIO{:02}", bi + 1 + si * 6);
                    bm.name = format!("{} Bioisostere {}", r_tag, b.rule_code);
                    push_unique!(
                        FunnelSourceStream::BioisostereSwap,
                        format!("Meanwell {} (Seed {})", b.rule_name, s_mol.id),
                        bm,
                        "BIO",
                    );
                }
            }

            if cfg.include_rotor_locks {
                let tor_rep =
                    analyze_torsion_strain_and_locks(s_mol, Some(&working_ds), pose_src, true);
                for lk in tor_rep.locked_analogs.into_iter().take(6) {
                    let mut lm = lk.mol;
                    lm.id = format!("{r_tag}-LCK{:02}", lk.rank + si * 6);
                    lm.name = format!(
                        "{} {} ({})",
                        r_tag,
                        lk.strategy.short_label(),
                        lk.target_dihedral_label
                    );
                    push_unique!(
                        FunnelSourceStream::RotorConformationLock,
                        format!(
                            "{} (+{:.2} kcal, Seed {})",
                            lk.strategy.short_label(),
                            lk.preorganization_gain_kcal,
                            s_mol.id
                        ),
                        lm,
                        "LCK",
                    );
                }
            }

            if cfg.include_fbdd_fragment_growing {
                let fg_rep = grow_fragment_along_vector(
                    s_mol,
                    si % 3,
                    true,
                    Some((&working_ds.mols, &working_ds.fps)),
                );
                for fg in fg_rep.candidates.into_iter().take(6) {
                    let mut fm = fg.mol;
                    fm.id = format!("{r_tag}-FBDD{:02}", fg.rank + si * 6);
                    fm.name = format!("{} FBDD {} ({})", r_tag, fg.synthon_code, fg.vector_label);
                    push_unique!(
                        FunnelSourceStream::FbddFragmentGrow,
                        format!("Rule-of-3 Grow {} ({})", fg.synthon_name, fg.coupling_reaction),
                        fm,
                        "FBDD",
                    );
                }
            }

            if cfg.include_fbdd_fragment_linking {
                let fl_preset = if si == 0 {
                    match pocket_idx {
                        0 => FragmentPairPreset::Egfr1m17HingeAndBackPocket,
                        1 => FragmentPairPreset::Drd26cm4PiperidineAndBenzisoxazole,
                        _ => FragmentPairPreset::Cox23ln1SulfonamideAndTolyl,
                    }
                } else {
                    FragmentPairPreset::DeconstructSelectedLead
                };
                let fl_rep = design_linked_fragments(
                    &FragmentLinkerConfig {
                        preset: fl_preset,
                        vector_pair_rank: 0,
                        custom_pocket_idx: pocket_idx,
                        ..FragmentLinkerConfig::default()
                    },
                    Some(s_mol),
                    Some(&working_ds),
                );
                for fl in fl_rep.candidates.into_iter().take(5) {
                    let mut lm = fl.mol;
                    lm.id = format!("{r_tag}-FLNK{:02}", fl.rank + si * 5);
                    lm.name = format!("{} Linked {}", r_tag, fl.linker_name);
                    push_unique!(
                        FunnelSourceStream::FbddFragmentLink,
                        format!(
                            "DeLinker {} (ddG_super {:+.2} kcal, RMSD {:.2} A)",
                            fl.linker_code, fl.ddg_superadditivity_kcal, fl.fragment_core_rmsd_a
                        ),
                        lm,
                        "FLNK",
                    );
                }
            }

            if cfg.include_macrocyclization {
                let mc_rep = design_macrocycles(s_mol, 0);
                for mc in mc_rep.candidates.into_iter().take(5) {
                    let mut mm = mc.mol;
                    mm.id = format!("{r_tag}-MAC{:02}", mc.rank + si * 5);
                    mm.name = format!("{} Macrocycle {}", r_tag, mc.tether_code);
                    push_unique!(
                        FunnelSourceStream::MacrocycleClosure,
                        format!("{} ({})", mc.tether_name, mc.clinical_precedent),
                        mm,
                        "MAC",
                    );
                }
            }
        }

        if cfg.include_parallel_del_plate {
            let pl_cfg = ParallelLibConfig {
                scheme_idx: (plib_scheme_idx + round_idx - 1) % 6,
                synthesis_mode: LibrarySynthesisMode::SolutionPhaseParallel,
                prefilter_mode: LibraryPreFilterMode::Rocs3dOverlay,
                reagents_per_cycle: 4,
                target_diverse_plate_size: 10,
                max_3d_scored_pool: 18,
                ..ParallelLibConfig::default()
            };
            let pl_rep = enumerate_parallel_library(
                &pl_cfg,
                round_seeds.first(),
                Some(&target_pocket),
                Some(&working_ds),
            );
            for m_item in pl_rep.members.into_iter().take(10) {
                let mut pm = m_item.mol;
                pm.id = format!("R{round_idx}-LIB{:02}", m_item.rank);
                pm.name = format!("R{round_idx} Plate {}", m_item.library_id);
                push_unique!(
                    FunnelSourceStream::ParallelDelLibrary,
                    m_item.reaction_route,
                    pm,
                    "LIB",
                );
            }
        }

        for (ui, smi) in cfg.custom_input_smiles.iter().enumerate() {
            if let Ok(mut um) = parse_smiles(smi) {
                um.id = format!("USR-{:02}", ui + 1);
                um.name = format!("User Candidate {}", ui + 1);
                push_unique!(
                    FunnelSourceStream::CustomUserSmiles,
                    "User Custom Input".to_string(),
                    um,
                    "USR",
                );
            }
        }

        if raw_intake.is_empty() {
            push_unique!(
                FunnelSourceStream::DatasetLead,
                "Fallback Seed Lead".to_string(),
                round_seeds[0].clone(),
                "SEED-01",
            );
        }

        let initial_pool_size = raw_intake.len();

        // ---------------------------------------------------------------------
        // STAGE 2: PAINS, QED, ESOL, SAScore & pH 7.4 Ionization / Protomer Gate
        // ---------------------------------------------------------------------
        emit(&mut on_progress, 2, "Stage 2: PAINS, ADMET, SAScore & pH 7.4 pKa");

        let mut records: Vec<FunnelCandidateRecord> = raw_intake
            .into_par_iter()
            .enumerate()
            .map(|(idx, (stream, prov, mol))| {
                let alerts = scan_structural_alerts(&mol);
                let prof = evaluate_physchem(&mol, alerts.len());
                let admet = evaluate_admet(&mol, &prof);
                let sa = evaluate_synthetic_accessibility(&mol);
                let pka = evaluate_physiological_pka_74(&mol);
                let (murcko_smi, _, _, _) = murcko_signatures(&mol);
                let core_key = if murcko_smi.is_empty() {
                    "Acyclic".to_string()
                } else {
                    murcko_smi
                };

                let mut fail_reasons = Vec::new();
                if cfg.require_zero_pains && !alerts.is_empty() {
                    fail_reasons.push(format!("PAINS ({})", alerts.len()));
                }
                if prof.qed_mean < cfg.min_qed {
                    fail_reasons.push(format!("QED {:.2} < {:.2}", prof.qed_mean, cfg.min_qed));
                }
                if sa.sa_score > cfg.max_sa_score {
                    fail_reasons.push(format!("SAScore {:.2} > {:.2}", sa.sa_score, cfg.max_sa_score));
                }
                if admet.esol_logs < cfg.min_esol_logs {
                    fail_reasons.push(format!(
                        "ESOL {:.2} < {:.2}",
                        admet.esol_logs, cfg.min_esol_logs
                    ));
                }
                if cfg.enforce_ph74_charge_gate
                    && pka.dominant_net_charge.unsigned_abs() > cfg.max_abs_charge_74.unsigned_abs()
                {
                    fail_reasons.push(format!(
                        "|Q_7.4|={} > {}",
                        pka.dominant_net_charge.abs(),
                        cfg.max_abs_charge_74.abs()
                    ));
                }

                let passed_s2 = fail_reasons.is_empty();
                let furthest = if passed_s2 { 2u8 } else { 1u8 };
                let status = if passed_s2 {
                    "Passed Stage 2 (ADMET/SAScore/pKa)".to_string()
                } else {
                    format!("Dropped S2: {}", fail_reasons.join(", "))
                };
                let base_score = 18.0 * prof.qed_mean + (6.5 - sa.sa_score).max(0.0) * 3.0;

                FunnelCandidateRecord {
                    rank: idx + 1,
                    dmta_round_born: round_idx,
                    candidate_id: mol.id.clone(),
                    candidate_name: mol.name.clone(),
                    source_stream: stream,
                    provenance_detail: prov,
                    canonical_smiles: mol.smiles.clone(),
                    murcko_scaffold: core_key,
                    furthest_stage_reached: furthest,
                    promoted_survivor: false,
                    diversity_pick_order: None,
                    nearest_promoted_tanimoto: 0.0,
                    status_summary: status,
                    mw: prof.mw,
                    clogp: prof.clogp,
                    tpsa: prof.tpsa,
                    qed: prof.qed_mean,
                    cns_mpo: prof.cns_mpo,
                    esol_logs: admet.esol_logs,
                    boiled_egg: admet.boiled_egg,
                    sa_score: sa.sa_score,
                    pains_alerts: alerts.len(),
                    dominant_charge_74: pka.dominant_net_charge,
                    logd_74: pka.microstate_logd_74,
                    pocket_dg_kcal: None,
                    pocket_complementarity: None,
                    displaced_unhappy_waters: None,
                    total_strain_teu: None,
                    max_single_bond_teu: None,
                    covalent_warhead_name: None,
                    covalent_gsh_half_life_h: None,
                    bioactive_coords3d: Vec::new(),
                    offtarget_dg_kcal: None,
                    ddg_selectivity_kcal: None,
                    selectivity_delta_pkd: None,
                    selectivity_fold: None,
                    allosteric_alpha: None,
                    allosteric_ddg_coop_kcal: None,
                    al_predicted_pact: None,
                    al_epistemic_std: None,
                    al_expected_improvement: None,
                    rbfe_mle_dg_kcal: None,
                    rbfe_ddg_vs_ref_kcal: None,
                    rbfe_predicted_pact: None,
                    rbfe_lomap_degree: None,
                    composite_funnel_score: base_score,
                    mol,
                }
            })
            .collect();

        let s2_passed_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.furthest_stage_reached >= 2)
            .map(|(i, _)| i)
            .collect();
        let s2_passed_count = s2_passed_indices.len();

        // ---------------------------------------------------------------------
        // STAGE 3: 3D Pocket + Shoichet Torsion TEU + Optional Covalent GSH Gate
        // ---------------------------------------------------------------------
        emit(&mut on_progress, 3, "Stage 3: 3D Pocket, Shoichet TEU & Covalent GSH");

        #[allow(clippy::type_complexity)]
        let s3_evals: Vec<(
            usize,
            f64,
            f64,
            usize,
            f64,
            f64,
            Option<String>,
            Option<f64>,
            Vec<[f32; 3]>,
            bool,
            String,
        )> = s2_passed_indices
            .par_iter()
            .map(|&ri| {
                let m = &records[ri].mol;
                let p_res = score_ligand_in_pocket(m, None, &target_pocket);
                let effective_pocket_dg = if let Some((ref fl_r, ref ens_m)) = ifd_prebuilt {
                    let ifd_rep = score_ligand_with_prebuilt_ensemble(
                        m,
                        None,
                        &InducedFitConfig {
                            preset: InducedFitPreset::CustomPdbFlexibleEnsemble,
                            ..InducedFitConfig::default()
                        },
                        &target_pdb_struct.pdb_id,
                        &resolved_target,
                        &target_pocket,
                        fl_r,
                        ens_m,
                    );
                    ifd_rep.boltzmann_ensemble_dg_kcal.min(p_res.total_dg_pocket_kcal)
                } else {
                    p_res.total_dg_pocket_kcal
                };
                let effective_pocket_dg = if cfg.optimize_pocket_hbond_network {
                    let hbn_rep = optimize_hbond_network_with_prebuilt(
                        m,
                        None,
                        &HBondNetworkConfig {
                            preset: HBondNetworkPreset::CustomPdbHBondNetwork,
                            target_ph: 7.4,
                            ..HBondNetworkConfig::default()
                        },
                        &target_pdb_struct,
                        &target_pocket,
                        &resolved_target,
                        false,
                    );
                    (effective_pocket_dg + (hbn_rep.ddg_hbn_optimization_gain_kcal * 0.35).clamp(-1.80, 0.0))
                        .min(effective_pocket_dg)
                } else {
                    effective_pocket_dg
                };
                let effective_pocket_dg = if cfg.enable_watermap_solvation {
                    let wm_rep = analyze_watermap_with_prebuilt(
                        m,
                        None,
                        &WaterMapConfig {
                            preset: WaterMapPreset::CustomPdbWaterMap,
                            gist_voxel_spacing_a: 0.95,
                            design_rgroup_vectors: false,
                            ..WaterMapConfig::default()
                        },
                        &target_pdb_struct,
                        &target_pocket,
                        &resolved_target,
                        false,
                    );
                    effective_pocket_dg + (wm_rep.net_watermap_dg_kcal * 0.32).clamp(-1.65, 1.25)
                } else {
                    effective_pocket_dg
                };
                let effective_pocket_dg = if cfg.enable_esp_multipole_scoring {
                    let esp_rep = analyze_esp_with_prebuilt(
                        m,
                        None,
                        &ElectrostaticsEspConfig {
                            preset: EspTargetPreset::CustomPdbEspComplementarity,
                            generate_esp_designs: false,
                            ..ElectrostaticsEspConfig::default()
                        },
                        &target_pdb_struct,
                        &target_pocket,
                        &resolved_target,
                        false,
                    );
                    effective_pocket_dg + (esp_rep.total_multipole_dg_kcal * 0.28).clamp(-1.45, 0.95)
                } else {
                    effective_pocket_dg
                };
                let t_rep = analyze_torsion_strain_and_locks(m, None, pose_src, false);
                let cov = evaluate_covalent_summary(m);

                let mut fails = Vec::new();
                if effective_pocket_dg > cfg.max_pocket_dg_kcal {
                    fails.push(format!(
                        "Pocket dG {:+.2} > {:+.2}",
                        effective_pocket_dg, cfg.max_pocket_dg_kcal
                    ));
                }
                if t_rep.total_strain_teu > cfg.max_total_strain_teu {
                    fails.push(format!(
                        "Total TEU {:.2} > {:.2}",
                        t_rep.total_strain_teu, cfg.max_total_strain_teu
                    ));
                }
                if t_rep.max_single_bond_teu > cfg.max_single_bond_teu {
                    fails.push(format!(
                        "Max Rotor TEU {:.2} > {:.2}",
                        t_rep.max_single_bond_teu, cfg.max_single_bond_teu
                    ));
                }
                if cfg.require_covalent_warhead_window {
                    match cov.gsh_half_life_h {
                        Some(th)
                            if th >= cfg.min_gsh_half_life_h
                                && th <= cfg.max_gsh_half_life_h => {}
                        Some(th) => fails.push(format!(
                            "GSH t1/2 {:.2}h outside [{:.2}..{:.1}h]",
                            th, cfg.min_gsh_half_life_h, cfg.max_gsh_half_life_h
                        )),
                        None => fails.push("Missing tuned covalent warhead".to_string()),
                    }
                }

                let wh_name = if cov.has_warhead {
                    Some(cov.primary_warhead_name.to_string())
                } else {
                    None
                };
                let ok = fails.is_empty();
                let msg = if ok {
                    "Passed Stage 3 (3D Pocket & TEU)".to_string()
                } else {
                    format!("Dropped S3: {}", fails.join(", "))
                };
                (
                    ri,
                    effective_pocket_dg,
                    p_res.complementarity_score,
                    p_res.displaced_unhappy_waters,
                    t_rep.total_strain_teu,
                    t_rep.max_single_bond_teu,
                    wh_name,
                    cov.gsh_half_life_h,
                    p_res.aligned_ligand_coords3d,
                    ok,
                    msg,
                )
            })
            .collect();

        for (ri, pdg, pcomp, dwat, tot_teu, max_teu, wh_n, gsh_h, coords, ok, msg) in s3_evals {
            let r = &mut records[ri];
            r.pocket_dg_kcal = Some(pdg);
            r.pocket_complementarity = Some(pcomp);
            r.displaced_unhappy_waters = Some(dwat);
            r.total_strain_teu = Some(tot_teu);
            r.max_single_bond_teu = Some(max_teu);
            r.covalent_warhead_name = wh_n;
            r.covalent_gsh_half_life_h = gsh_h;
            r.bioactive_coords3d = coords;
            r.status_summary = msg;
            r.composite_funnel_score +=
                ((-pdg - 4.0).max(0.0) * 3.8) + ((7.0 - tot_teu).max(0.0) * 1.4);
            if ok {
                r.furthest_stage_reached = 3;
            }
        }

        let s3_passed_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.furthest_stage_reached >= 3)
            .map(|(i, _)| i)
            .collect();
        let s3_passed_count = s3_passed_indices.len();

        // ---------------------------------------------------------------------
        // STAGE 4: Dual-Pocket Selectivity + Optional Allosteric GNM/ANM Coupling
        // ---------------------------------------------------------------------
        emit(&mut on_progress, 4, "Stage 4: Dual-Pocket Selectivity & Allostery");

        #[allow(clippy::type_complexity)]
        let s4_evals: Vec<(
            usize,
            f64,
            f64,
            f64,
            f64,
            Option<f64>,
            Option<f64>,
            bool,
            String,
        )> = s3_passed_indices
            .par_iter()
            .map(|&ri| {
                let m = &records[ri].mol;
                let sel = score_ligand_dual_selectivity(m, None, &dual_selectivity);
                let (allo_a, allo_ddg) = if let Some(ref net) = allo_network {
                    score_ligand_allosteric_coupling(m, None, net, 0)
                        .map(|ar| (Some(ar.cooperativity_alpha), Some(ar.cooperativity_ddg_kcal)))
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                let ok = sel.delta_pkd >= cfg.min_selectivity_delta_pkd;
                let msg = if ok {
                    "Passed Stage 4 (Selectivity Gate)".to_string()
                } else {
                    format!(
                        "Dropped S4: dpKd {:+.2} < {:+.2}",
                        sel.delta_pkd, cfg.min_selectivity_delta_pkd
                    )
                };
                (
                    ri,
                    sel.offtarget_dg_kcal,
                    sel.ddg_selectivity_kcal,
                    sel.delta_pkd,
                    sel.selectivity_index_fold,
                    allo_a,
                    allo_ddg,
                    ok,
                    msg,
                )
            })
            .collect();

        for (ri, off_dg, ddg_sel, dpkd, si_fold, allo_a, allo_ddg, ok, msg) in s4_evals {
            let r = &mut records[ri];
            r.offtarget_dg_kcal = Some(off_dg);
            r.ddg_selectivity_kcal = Some(ddg_sel);
            r.selectivity_delta_pkd = Some(dpkd);
            r.selectivity_fold = Some(si_fold);
            r.allosteric_alpha = allo_a;
            r.allosteric_ddg_coop_kcal = allo_ddg;
            r.status_summary = msg;
            r.composite_funnel_score += (dpkd + 1.0).clamp(0.0, 3.5) * 3.5;
            if let Some(ddg_c) = allo_ddg {
                r.composite_funnel_score += ddg_c.abs().min(2.0) * 2.0;
            }
            if ok {
                r.furthest_stage_reached = 4;
            }
        }

        let mut s4_passed_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.furthest_stage_reached >= 4)
            .map(|(i, _)| i)
            .collect();

        if s4_passed_indices.is_empty() && !records.is_empty() {
            let mut by_score: Vec<usize> = (0..records.len()).collect();
            by_score.sort_by(|&a, &b| {
                records[b]
                    .composite_funnel_score
                    .partial_cmp(&records[a].composite_funnel_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &ri in by_score.iter().take(3) {
                records[ri].furthest_stage_reached = 4;
                records[ri].status_summary = "Rescued Top Near-Threshold Lead".to_string();
                s4_passed_indices.push(ri);
            }
        }
        let s4_passed_count = s4_passed_indices.len();

        // ---------------------------------------------------------------------
        // STAGE 5: Active-Learning ECFP4 + 3D-USRCAT GP & 24-Tree RF QSAR
        // ---------------------------------------------------------------------
        emit(&mut on_progress, 5, "Stage 5: Active-Learning GP+RF QSAR");

        let survivor_smiles: Vec<String> = s4_passed_indices
            .iter()
            .map(|&ri| records[ri].canonical_smiles.clone())
            .collect();

        let max_finalists = cfg.max_rbfe_finalists.clamp(4, 24);
        let al_rep = run_active_learning_campaign(
            &working_ds,
            &ActiveLearningConfig {
                kernel_mode: QsarKernelMode::HybridEcfp4And3dUsrcat,
                surrogate_kind: SurrogateModelKind::ConsensusGpAndEnsemble,
                acquisition: AcquisitionStrategy::ExpectedImprovement,
                pool_source: CandidatePoolSource::HarvestedAnalogsAndLibrary,
                batch_size: max_finalists.min(16),
                custom_pool_smiles: survivor_smiles,
                ..ActiveLearningConfig::default()
            },
        );
        let loo_q2 = al_rep.validation.loo_q2;

        let mut al_by_smiles: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
        for c in &al_rep.candidates {
            al_by_smiles.insert(
                c.canonical_smiles.clone(),
                (c.predicted_pact, c.epistemic_std, c.expected_improvement),
            );
        }

        for &ri in &s4_passed_indices {
            let r = &mut records[ri];
            let (pred_p, ep_std, ei) = al_by_smiles
                .get(&r.canonical_smiles)
                .copied()
                .unwrap_or_else(|| {
                    let p_est = if r.mol.pact > 0.5 {
                        r.mol.pact
                    } else {
                        (-r.pocket_dg_kcal.unwrap_or(-8.5) / RT_LN10_KCAL).clamp(5.5, 9.5)
                    };
                    (p_est, 0.22, 0.08)
                });
            r.al_predicted_pact = Some(pred_p);
            r.al_epistemic_std = Some(ep_std);
            r.al_expected_improvement = Some(ei);
            r.composite_funnel_score += (pred_p - 5.5).max(0.0) * 6.5 + ei * 18.0;
        }

        let mut ranked_s4 = s4_passed_indices.clone();
        ranked_s4.sort_by(|&a, &b| {
            records[b]
                .composite_funnel_score
                .partial_cmp(&records[a].composite_funnel_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut s5_finalist_indices = Vec::new();
        for (pos, &ri) in ranked_s4.iter().enumerate() {
            if pos < max_finalists {
                records[ri].furthest_stage_reached = 5;
                records[ri].status_summary = "Advanced to Stage 6 Lomap RBFE".to_string();
                s5_finalist_indices.push(ri);
            } else {
                records[ri].status_summary = format!(
                    "Held at Stage 5 AL Cap (#{} > {})",
                    pos + 1,
                    max_finalists
                );
            }
        }
        let s5_passed_count = s5_finalist_indices.len();

        // ---------------------------------------------------------------------
        // STAGE 6: Lomap BAR/MLE RBFE + Bemis-Murcko / MaxMin Diversity Promotion
        // ---------------------------------------------------------------------
        emit(&mut on_progress, 6, "Stage 6: Lomap BAR/MLE RBFE & Murcko Promotion");

        let finalist_smiles: Vec<String> = s5_finalist_indices
            .iter()
            .map(|&ri| records[ri].canonical_smiles.clone())
            .collect();

        let rbfe_rep = run_rbfe_network_campaign(
            &working_ds,
            &RbfeNetworkConfig {
                series_preset: rbfe_series,
                topology_mode: RbfeTopologyMode::CycleClosedLomap,
                lambda_windows: 9,
                max_series_ligands: 10,
                custom_analog_smiles: finalist_smiles,
                ..RbfeNetworkConfig::default()
            },
        );

        let mut rbfe_by_smiles: BTreeMap<String, (f64, f64, f64, usize)> = BTreeMap::new();
        for nd in &rbfe_rep.nodes {
            rbfe_by_smiles.insert(
                nd.canonical_smiles.clone(),
                (
                    nd.mle_dg_bind_kcal,
                    nd.ddg_vs_reference_kcal,
                    nd.mle_predicted_pact,
                    nd.degree,
                ),
            );
        }

        for &ri in &s5_finalist_indices {
            let r = &mut records[ri];
            r.furthest_stage_reached = 6;
            if let Some(&(mle_dg, ddg_ref, mle_pact, deg)) = rbfe_by_smiles.get(&r.canonical_smiles)
            {
                r.rbfe_mle_dg_kcal = Some(mle_dg);
                r.rbfe_ddg_vs_ref_kcal = Some(ddg_ref);
                r.rbfe_predicted_pact = Some(mle_pact);
                r.rbfe_lomap_degree = Some(deg);
                r.composite_funnel_score +=
                    ((-mle_dg - 7.5).max(0.0) * 5.2) + ((deg.min(3)) as f64);
            } else {
                let est_p = r.al_predicted_pact.unwrap_or(7.5);
                let est_dg = -RT_LN10_KCAL * est_p;
                r.rbfe_mle_dg_kcal = Some(est_dg);
                r.rbfe_ddg_vs_ref_kcal = Some(0.0);
                r.rbfe_predicted_pact = Some(est_p);
                r.rbfe_lomap_degree = Some(0);
            }
        }

        let max_raw = records
            .iter()
            .map(|r| r.composite_funnel_score)
            .fold(1.0_f64, f64::max);
        for r in &mut records {
            r.composite_funnel_score = ((r.composite_funnel_score / max_raw) * 98.5).clamp(1.0, 99.9);
        }

        records.sort_by(|a, b| {
            b.furthest_stage_reached
                .cmp(&a.furthest_stage_reached)
                .then_with(|| {
                    b.composite_funnel_score
                        .partial_cmp(&a.composite_funnel_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        for (idx, r) in records.iter_mut().enumerate() {
            r.rank = idx + 1;
        }

        // Bemis-Murcko Core Cap + MaxMin ECFP4 Dissimilarity Diversity Promotion!
        let promote_cap = cfg.final_promote_count.clamp(1, 24);
        let murcko_cap = cfg.max_per_murcko_scaffold.clamp(1, 24);
        let max_tc = (1.0 - cfg.min_promoted_ecfp4_dissim.clamp(0.0, 0.70)) as f32;

        let fps: Vec<Ecfp2048> = records.iter().map(|r| compute_ecfp(&r.mol, 2)).collect();
        let mut selected_indices: Vec<usize> = Vec::new();
        let mut scaffold_counts: BTreeMap<String, usize> = BTreeMap::new();

        // Pass 1: Strict Murcko cap + ECFP4 sphere exclusion
        for (idx, r) in records.iter().enumerate() {
            if selected_indices.len() >= promote_cap || r.furthest_stage_reached < 6 {
                continue;
            }
            let cur_scaf_n = scaffold_counts.get(&r.murcko_scaffold).copied().unwrap_or(0);
            if cur_scaf_n >= murcko_cap {
                continue;
            }
            let max_sim_to_sel = selected_indices
                .iter()
                .map(|&si| fps[idx].tanimoto(&fps[si]))
                .fold(0.0_f32, f32::max);
            if !selected_indices.is_empty() && max_sim_to_sel > max_tc {
                continue;
            }
            selected_indices.push(idx);
            *scaffold_counts.entry(r.murcko_scaffold.clone()).or_insert(0) += 1;
        }

        // Pass 2: Graceful backfill if strict diversity cap left slots open among Stage-6 finalists
        for (idx, r) in records.iter().enumerate() {
            if selected_indices.len() >= promote_cap {
                break;
            }
            if r.furthest_stage_reached == 6 && !selected_indices.contains(&idx) {
                selected_indices.push(idx);
                *scaffold_counts.entry(r.murcko_scaffold.clone()).or_insert(0) += 1;
            }
        }

        for (pick_pos, &si) in selected_indices.iter().enumerate() {
            let max_sim = selected_indices
                .iter()
                .filter(|&&oj| oj != si)
                .map(|&oj| fps[si].tanimoto(&fps[oj]))
                .fold(0.0_f32, f32::max);
            records[si].promoted_survivor = true;
            records[si].diversity_pick_order = Some(pick_pos + 1);
            records[si].nearest_promoted_tanimoto = max_sim;
            records[si].status_summary = format!("PROMOTED #{} (Murcko-Diversified)", pick_pos + 1);
            global_seen_promoted.insert(records[si].canonical_smiles.clone());
        }
        for (idx, r) in records.iter_mut().enumerate() {
            if !r.promoted_survivor && r.furthest_stage_reached == 6 {
                let max_sim = selected_indices
                    .iter()
                    .map(|&oj| fps[idx].tanimoto(&fps[oj]))
                    .fold(0.0_f32, f32::max);
                r.nearest_promoted_tanimoto = max_sim;
                r.status_summary = "Stage 6 Backup (Scaffold/Sphere Cap)".to_string();
            }
        }

        let promoted_survivor_count = selected_indices.len();
        let unique_scaf_count = scaffold_counts.len().max(1);

        // Compute DmtaRoundSummary metrics
        let mut best_dg = 0.0_f64;
        let mut best_pact = 0.0_f64;
        let mut sum_pact = 0.0_f64;
        let mut sum_lle = 0.0_f64;
        let mut sum_teu = 0.0_f64;
        for &si in &selected_indices {
            let r = &records[si];
            let dg = r.rbfe_mle_dg_kcal.unwrap_or(-9.5);
            let pa = r
                .rbfe_predicted_pact
                .or(r.al_predicted_pact)
                .unwrap_or(7.0);
            if best_dg == 0.0 || dg < best_dg {
                best_dg = dg;
            }
            if pa > best_pact {
                best_pact = pa;
            }
            sum_pact += pa;
            sum_lle += pa - r.clogp;
            sum_teu += r.total_strain_teu.unwrap_or(0.0);
        }
        let denom_p = promoted_survivor_count.max(1) as f64;
        round_history.push(DmtaRoundSummary {
            round: round_idx,
            seed_ids: round_seeds.iter().map(|m| m.id.clone()).collect(),
            pool_size: initial_pool_size,
            stage2_passed: s2_passed_count,
            stage3_passed: s3_passed_count,
            stage4_passed: s4_passed_count,
            rbfe_finalists: s5_passed_count,
            promoted_count: promoted_survivor_count,
            unique_promoted_scaffolds: unique_scaf_count,
            surrogate_loo_q2: loo_q2,
            best_rbfe_dg_kcal: best_dg,
            best_predicted_pact: best_pact,
            mean_promoted_pact: sum_pact / denom_p,
            mean_promoted_lle: sum_lle / denom_p,
            mean_promoted_strain_teu: sum_teu / denom_p,
        });

        // Closed-Loop Feedback into Round `r + 1`:
        // 1) Pseudo-label Round `r` promoted survivors with their reconciled RBFE/AL pAct & merge into `working_ds`!
        // 2) Pick top 2 Murcko-distinct promoted survivors as next-round generative/locking seeds!
        if round_idx < total_rounds && !selected_indices.is_empty() {
            let mut next_incoming = Vec::new();
            let mut next_seeds = Vec::new();
            let mut used_seed_cores = BTreeSet::new();
            for &si in &selected_indices {
                let c = &records[si];
                let mut m = c.mol.clone();
                m.target = resolved_target.clone();
                m.pact = c
                    .rbfe_predicted_pact
                    .or(c.al_predicted_pact)
                    .unwrap_or(7.5);
                if next_seeds.len() < 2 && used_seed_cores.insert(c.murcko_scaffold.clone()) {
                    next_seeds.push(m.clone());
                }
                next_incoming.push(m);
            }
            if !next_seeds.is_empty() {
                round_seeds = next_seeds;
            }
            working_ds.merge_imported_molecules(next_incoming);
        }

        // Capture final round's stage summaries & candidate records
        if round_idx == total_rounds {
            let n0 = initial_pool_size.max(1) as f64;
            let mk_stage = |num: u8,
                            code: &'static str,
                            title: &'static str,
                            crit: String,
                            inp: usize,
                            pass: usize|
             -> FunnelStageSummary {
                let drop = inp.saturating_sub(pass);
                let sp = if inp == 0 {
                    0.0
                } else {
                    (pass as f64) / (inp as f64) * 100.0
                };
                let cy = (pass as f64) / n0 * 100.0;
                FunnelStageSummary {
                    stage_num: num,
                    stage_code: code,
                    stage_title: title,
                    gate_criteria: crit,
                    input_count: inp,
                    passed_count: pass,
                    dropped_count: drop,
                    stage_pass_pct: sp,
                    cumulative_yield_pct: cy,
                }
            };

            final_stage_summaries = vec![
                mk_stage(
                    1,
                    "S1-INTAKE",
                    "1. Multi-Stream & Modality Pool",
                    format!(
                        "DMTA Round {round_idx}/{total_rounds} OpenSMILES-deduplicated intake"
                    ),
                    initial_pool_size,
                    initial_pool_size,
                ),
                mk_stage(
                    2,
                    "S2-ADMET-PKA",
                    "2. PAINS, ADMET, SAScore & pKa",
                    format!(
                        "PAINS={}, QED>={:.2}, SASc<={:.1}, |Q7.4|<={}",
                        if cfg.require_zero_pains { "0" } else { "Any" },
                        cfg.min_qed,
                        cfg.max_sa_score,
                        if cfg.enforce_ph74_charge_gate {
                            cfg.max_abs_charge_74.to_string()
                        } else {
                            "Any".to_string()
                        }
                    ),
                    initial_pool_size,
                    s2_passed_count,
                ),
                mk_stage(
                    3,
                    "S3-POCKET-TEU",
                    "3. 3D Pocket, TEU & Covalent GSH",
                    format!(
                        "dG<={:+.1}, TEU<={:.1}{}",
                        cfg.max_pocket_dg_kcal,
                        cfg.max_total_strain_teu,
                        if cfg.require_covalent_warhead_window {
                            ", Cov-GSH"
                        } else {
                            ""
                        }
                    ),
                    s2_passed_count,
                    s3_passed_count,
                ),
                mk_stage(
                    4,
                    "S4-SELECT-ALLO",
                    "4. Selectivity & Allosteric MWC",
                    format!(
                        "{} dpKd>={:+.2}{}",
                        selectivity_pair_code,
                        cfg.min_selectivity_delta_pkd,
                        if cfg.enable_allosteric_coupling_branch {
                            " + GNM/ANM"
                        } else {
                            ""
                        }
                    ),
                    s3_passed_count,
                    s4_passed_count,
                ),
                mk_stage(
                    5,
                    "S5-ACTIVE-QSAR",
                    "5. Closed-Loop GP+RF Active QSAR",
                    format!("Top {} EI + 3D-USRCAT finalists", max_finalists),
                    s4_passed_count,
                    s5_passed_count,
                ),
                mk_stage(
                    6,
                    "S6-RBFE-MURCKO",
                    "6. Lomap RBFE & Murcko Diversity",
                    format!(
                        "Top {} (<= {}/Murcko, Dissim>={:.2})",
                        promote_cap, murcko_cap, cfg.min_promoted_ecfp4_dissim
                    ),
                    s5_passed_count,
                    promoted_survivor_count,
                ),
            ];

            final_initial_pool = initial_pool_size;
            final_s4_survivors = s4_passed_count;
            final_rbfe_finalists = s5_passed_count;
            final_promoted_count = promoted_survivor_count;
            final_unique_scaffolds = unique_scaf_count;
            final_records = records;
        }
    }

    on_progress(PipelineProgressUpdate {
        current_round: total_rounds,
        total_rounds,
        stage_num: 6,
        stage_label: "Completed Multi-Round Discovery Funnel".to_string(),
        fraction_complete: 1.0,
    });

    let overall_yield_pct =
        (final_promoted_count as f64) / (final_initial_pool.max(1) as f64) * 100.0;

    DiscoveryPipelineReport {
        resolved_target,
        target_pocket_pdb: target_pocket.pdb_id,
        selectivity_pair_code,
        pdb_source_note: pdb_notes.join(" | "),
        seed_compound_id,
        seed_compound_name,
        config: cfg.clone(),
        initial_pool_size: final_initial_pool,
        stage4_survivor_count: final_s4_survivors,
        rbfe_finalist_count: final_rbfe_finalists,
        promoted_survivor_count: final_promoted_count,
        unique_promoted_scaffolds: final_unique_scaffolds,
        overall_yield_pct,
        round_history,
        stage_summaries: final_stage_summaries,
        candidates: final_records,
    }
}

/// Saves a `DiscoveryPipelineConfig` workflow recipe to portable pretty-printed `.json`.
pub fn save_pipeline_recipe_json(
    path: &Path,
    cfg: &DiscoveryPipelineConfig,
) -> Result<(), String> {
    let txt = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(path, txt).map_err(|e| e.to_string())
}

/// Loads a `DiscoveryPipelineConfig` workflow recipe from `.json`.
pub fn load_pipeline_recipe_json(path: &Path) -> Result<DiscoveryPipelineConfig, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&txt).map_err(|e| e.to_string())
}

/// Appends a completed `DiscoveryPipelineReport` (recipe JSON, DMTA round summaries, 6-stage attrition,
/// and promoted survivors) into a normalized `SQLite 3` run-ledger database (creating tables if needed).
pub fn append_pipeline_run_sqlite(
    db_path: &Path,
    report: &DiscoveryPipelineReport,
) -> Result<i64, String> {
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         CREATE TABLE IF NOT EXISTS pipeline_runs (
             run_id                     INTEGER PRIMARY KEY AUTOINCREMENT,
             epoch_seconds              INTEGER NOT NULL,
             recipe_name                TEXT NOT NULL,
             campaign_code              TEXT NOT NULL,
             resolved_target            TEXT NOT NULL,
             target_pocket_pdb          TEXT NOT NULL,
             dmta_rounds                INTEGER NOT NULL,
             initial_pool_size          INTEGER NOT NULL,
             promoted_count             INTEGER NOT NULL,
             unique_promoted_scaffolds  INTEGER NOT NULL,
             best_rbfe_dg_kcal          REAL NOT NULL,
             mean_promoted_pact         REAL NOT NULL,
             recipe_json                TEXT NOT NULL,
             promoted_smiles_joined     TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS pipeline_promoted_leads (
             id                INTEGER PRIMARY KEY AUTOINCREMENT,
             run_id            INTEGER NOT NULL,
             pick_order        INTEGER NOT NULL,
             candidate_id      TEXT NOT NULL,
             source_stream     TEXT NOT NULL,
             murcko_scaffold   TEXT NOT NULL,
             rbfe_dg_kcal      REAL NOT NULL,
             predicted_pact    REAL NOT NULL,
             strain_teu        REAL NOT NULL,
             canonical_smiles  TEXT NOT NULL
         );",
    )
    .map_err(|e| e.to_string())?;

    let now_sec = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let last_r = report.round_history.last();
    let best_dg = last_r.map(|r| r.best_rbfe_dg_kcal).unwrap_or(0.0);
    let mean_p = last_r.map(|r| r.mean_promoted_pact).unwrap_or(0.0);
    let recipe_json = serde_json::to_string(&report.config).unwrap_or_default();
    let prom_smiles: Vec<String> = report
        .candidates
        .iter()
        .filter(|c| c.promoted_survivor)
        .map(|c| c.canonical_smiles.clone())
        .collect();
    let prom_joined = prom_smiles.join(";");

    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "INSERT INTO pipeline_runs (
            epoch_seconds, recipe_name, campaign_code, resolved_target, target_pocket_pdb,
            dmta_rounds, initial_pool_size, promoted_count, unique_promoted_scaffolds,
            best_rbfe_dg_kcal, mean_promoted_pact, recipe_json, promoted_smiles_joined
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            now_sec as i64,
            report.config.recipe_name,
            report.config.preset.code(),
            report.resolved_target,
            report.target_pocket_pdb,
            report.round_history.len() as i64,
            report.initial_pool_size as i64,
            report.promoted_survivor_count as i64,
            report.unique_promoted_scaffolds as i64,
            best_dg,
            mean_p,
            recipe_json,
            prom_joined,
        ],
    )
    .map_err(|e| e.to_string())?;
    let run_id = tx.last_insert_rowid();

    {
        let mut lead_stmt = tx
            .prepare(
                "INSERT INTO pipeline_promoted_leads (
                    run_id, pick_order, candidate_id, source_stream, murcko_scaffold,
                    rbfe_dg_kcal, predicted_pact, strain_teu, canonical_smiles
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .map_err(|e| e.to_string())?;
        for c in report.candidates.iter().filter(|c| c.promoted_survivor) {
            lead_stmt
                .execute(params![
                    run_id,
                    c.diversity_pick_order.unwrap_or(c.rank) as i64,
                    c.candidate_id,
                    c.source_stream.short_label(),
                    c.murcko_scaffold,
                    c.rbfe_mle_dg_kcal.unwrap_or(0.0),
                    c.rbfe_predicted_pact
                        .or(c.al_predicted_pact)
                        .unwrap_or(7.0),
                    c.total_strain_teu.unwrap_or(0.0),
                    c.canonical_smiles,
                ])
                .map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(run_id)
}

/// Reads all historical `PipelineRunLedgerEntry` records (newest first) from a `SQLite 3` database.
pub fn load_pipeline_ledger_sqlite(db_path: &Path) -> Result<Vec<PipelineRunLedgerEntry>, String> {
    if !db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    let has_table: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='pipeline_runs'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    if has_table == 0 {
        return Ok(Vec::new());
    }
    let mut stmt = conn
        .prepare(
            "SELECT run_id, epoch_seconds, recipe_name, campaign_code, resolved_target,
                    target_pocket_pdb, dmta_rounds, initial_pool_size, promoted_count,
                    unique_promoted_scaffolds, best_rbfe_dg_kcal, mean_promoted_pact,
                    promoted_smiles_joined
             FROM pipeline_runs ORDER BY run_id DESC LIMIT 64",
        )
        .map_err(|e| e.to_string())?;
    let mapped = stmt
        .query_map([], |row| {
            Ok(PipelineRunLedgerEntry {
                run_id: row.get(0)?,
                epoch_seconds: row.get::<_, i64>(1)?.max(0) as u64,
                recipe_name: row.get(2)?,
                campaign_code: row.get(3)?,
                resolved_target: row.get(4)?,
                target_pocket_pdb: row.get(5)?,
                dmta_rounds: row.get::<_, i64>(6)?.max(1) as usize,
                initial_pool_size: row.get::<_, i64>(7)?.max(0) as usize,
                promoted_count: row.get::<_, i64>(8)?.max(0) as usize,
                unique_promoted_scaffolds: row.get::<_, i64>(9)?.max(0) as usize,
                best_rbfe_dg_kcal: row.get(10)?,
                mean_promoted_pact: row.get(11)?,
                promoted_smiles_joined: row.get(12)?,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for item in mapped {
        out.push(item.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

/// Exports the 6-stage attrition summary + multi-round DMTA telemetry to `.csv`.
pub fn export_funnel_stages_csv(report: &DiscoveryPipelineReport) -> String {
    let mut out = String::from(
        "stage_num,stage_code,stage_title,input_count,passed_count,dropped_count,stage_pass_pct,cumulative_yield_pct,criteria\n",
    );
    for s in &report.stage_summaries {
        out.push_str(&format!(
            "{},{},\"{}\",{},{},{},{:.1},{:.1},\"{}\"\n",
            s.stage_num,
            s.stage_code,
            s.stage_title,
            s.input_count,
            s.passed_count,
            s.dropped_count,
            s.stage_pass_pct,
            s.cumulative_yield_pct,
            s.gate_criteria.replace('"', "'")
        ));
    }
    out
}

/// Exports all funnel candidates (or only promoted synthesis survivors when `promoted_only == true`) to `.csv`.
pub fn export_funnel_candidates_csv(
    report: &DiscoveryPipelineReport,
    promoted_only: bool,
) -> String {
    let mut out = String::from(
        "rank,dmta_round,candidate_id,candidate_name,source_stream,murcko_scaffold,furthest_stage,promoted_survivor,nearest_prom_tc,composite_score,qed,sa_score,esol_logs,charge_74,pocket_dg_kcal,total_strain_teu,selectivity_dpkd,al_pred_pact,al_ei,rbfe_mle_dg_kcal,rbfe_pred_pact,status,smiles\n",
    );
    for c in &report.candidates {
        if promoted_only && !c.promoted_survivor {
            continue;
        }
        let fmt_opt = |v: Option<f64>| v.map(|x| format!("{x:.3}")).unwrap_or_default();
        out.push_str(&format!(
            "{},{},{},\"{}\",{},\"{}\",S{},{},{:.3},{:.2},{:.3},{:.2},{:.2},{},{},{},{},{},{},{},{},\"{}\",{}\n",
            c.rank,
            c.dmta_round_born,
            c.candidate_id,
            c.candidate_name.replace('"', "'"),
            c.source_stream.short_label(),
            c.murcko_scaffold,
            c.furthest_stage_reached,
            c.promoted_survivor,
            c.nearest_promoted_tanimoto,
            c.composite_funnel_score,
            c.qed,
            c.sa_score,
            c.esol_logs,
            c.dominant_charge_74,
            fmt_opt(c.pocket_dg_kcal),
            fmt_opt(c.total_strain_teu),
            fmt_opt(c.selectivity_delta_pkd),
            fmt_opt(c.al_predicted_pact),
            fmt_opt(c.al_expected_improvement),
            fmt_opt(c.rbfe_mle_dg_kcal),
            fmt_opt(c.rbfe_predicted_pact),
            c.status_summary.replace('"', "'"),
            c.canonical_smiles
        ));
    }
    out
}

/// Exports 3D bioactive pocket poses of all promoted synthesis survivors as a multi-record `.sdf`.
pub fn format_promoted_survivors_sdf(report: &DiscoveryPipelineReport) -> String {
    let mut sdf = String::new();
    for c in report.candidates.iter().filter(|c| c.promoted_survivor) {
        let coords = if c.bioactive_coords3d.len() == c.mol.atoms.len() {
            c.bioactive_coords3d.clone()
        } else {
            evaluate_3d_summary_profile(&c.mol, true).coords3d
        };
        let charges = compute_gasteiger_charges(&c.mol);
        let dg = c.rbfe_mle_dg_kcal.or(c.pocket_dg_kcal).unwrap_or(-9.0);
        let rec = format_3d_conformer_sdf(
            &c.mol,
            &coords,
            &charges,
            dg,
            &[
                ("FUNNEL_RANK", c.rank.to_string()),
                ("DMTA_ROUND", c.dmta_round_born.to_string()),
                ("SOURCE_STREAM", c.source_stream.short_label().to_string()),
                ("MURCKO_SCAFFOLD", c.murcko_scaffold.clone()),
                (
                    "COMPOSITE_FUNNEL_SCORE",
                    format!("{:.2}", c.composite_funnel_score),
                ),
                (
                    "RBFE_MLE_DG_KCAL",
                    format!("{:.3}", c.rbfe_mle_dg_kcal.unwrap_or(dg)),
                ),
                (
                    "SHOICHET_TOTAL_TEU",
                    format!("{:.2}", c.total_strain_teu.unwrap_or(0.0)),
                ),
            ],
        );
        sdf.push_str(&rec);
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_benchmark_26;

    #[test]
    fn test_v310_multi_round_dmta_murcko_diversity_and_sqlite_ledger() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = DiscoveryPipelineConfig {
            recipe_name: "UnitTest-2Round-EGFR".to_string(),
            preset: DiscoveryFunnelPreset::EgfrKinasePanMutant,
            dmta_rounds: 2,
            include_fbdd_fragment_growing: true,
            include_macrocyclization: true,
            enable_allosteric_coupling_branch: true,
            max_per_murcko_scaffold: 2,
            final_promote_count: 6,
            ..DiscoveryPipelineConfig::default()
        };
        let mut updates = 0usize;
        let rep = run_discovery_pipeline_with_progress(&ds, &cfg, |_| {
            updates += 1;
        });
        assert!(updates >= 12);
        assert_eq!(rep.round_history.len(), 2);
        assert!(rep.promoted_survivor_count >= 1 && rep.promoted_survivor_count <= 6);
        assert!(rep.unique_promoted_scaffolds >= 1);

        let tmp_dir = std::env::temp_dir();
        let recipe_p = tmp_dir.join("sar_atlas_v310_test_recipe.json");
        let ledger_p = tmp_dir.join("sar_atlas_v310_test_ledger.sqlite");
        let _ = std::fs::remove_file(&ledger_p);

        save_pipeline_recipe_json(&recipe_p, &cfg).expect("Save recipe JSON");
        let loaded_cfg = load_pipeline_recipe_json(&recipe_p).expect("Load recipe JSON");
        assert_eq!(loaded_cfg.dmta_rounds, 2);

        let run_id = append_pipeline_run_sqlite(&ledger_p, &rep).expect("Append SQLite ledger");
        assert!(run_id >= 1);
        let ledger = load_pipeline_ledger_sqlite(&ledger_p).expect("Load SQLite ledger");
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger[0].dmta_rounds, 2);

        let _ = std::fs::remove_file(&recipe_p);
        let _ = std::fs::remove_file(&ledger_p);
    }
}
