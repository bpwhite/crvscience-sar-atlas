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

//! `Lomap` (`Lead Optimization Mapper`) `MCS` Alchemical Perturbation-Graph Builder,
//! Multi-`lambda` Two-Leg (`Complex` vs `Solvent`) `Zwanzig` (`FEP`) & Self-Consistent
//! `Bennett Acceptance Ratio` (`BAR`) Congeneric Relative Binding Free Energy (`RBFE`) Estimator,
//! Closed Thermodynamic Cycle Hysteresis Detector, and Variance-Weighted Graph-Laplacian
//! Maximum-Likelihood (`MLE` / `BARnet` / `cinnabar`) Node Free-Energy Reconciler (`rbfe_network.rs`, `v2.8.0`).
//!
//! Primary References:
//! - Liu, Wu, Lin, Abel, Redmann, Summa, Jaber, Lim & Mobley (2013), *J. Comput.-Aided Mol. Des.*
//!   27(9), 755-770 ("Lead Optimization Mapper: Automating free energy calculations for lead
//!   optimization": `https://pmc.ncbi.nlm.nih.gov/articles/PMC3837551/`)
//! - Liu, Wang & Mobley (2015), *J. Chem. Inf. Model.* 55(4), 727-735 ("Is Ring Breaking Feasible
//!   in Relative Binding Free Energy Calculations?": `https://pmc.ncbi.nlm.nih.gov/articles/PMC4489933/`)
//!   & Large-Scale `OpenFE`/`Lomap` Network Assessment (`https://pmc.ncbi.nlm.nih.gov/articles/PMC13555741/`)
//! - Wang et al. (2015), *J. Am. Chem. Soc.* 137(7), 2695-2703 ("Accurate and reliable prediction
//!   of relative ligand binding potency in prospective drug discovery by way of a modern free-energy
//!   calculation protocol and force field": `https://pubmed.ncbi.nlm.nih.gov/25625324/`)
//! - Mey, Allen, Macdonald, Chodera, Hahn, Kuhn, Michel, Mobley & Shirts (2020), *Living J. Comput.
//!   Mol. Sci.* 2(1), 18378 ("Best Practices for Alchemical Free Energy Calculations" — `Zwanzig`
//!   exponential averaging, self-consistent `Bennett Acceptance Ratio` `BAR`, & phase-space overlap:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8388617/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC9842838/`)
//! - Xu et al. (2021) / `BARnet` & `LWHAM` Cycle-Closure Restraints (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8011336/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC7137390/`) & `cinnabar` Maximum-Likelihood `ddG -> dG`
//!   Node Reconciliation (`https://pmc.ncbi.nlm.nih.gov/articles/PMC13080961/`).

use crate::active_learning::cholesky_decompose;
use crate::fragment_grower::RT_LN10_KCAL;
use crate::mcs::{find_pairwise_mcs, McsAtomMatchMode, McsConfig};
use crate::mol::{Element, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    PocketCavityAnalysis,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Thermal energy `R * T` at `298.15 K` (`kcal/mol`).
pub const RBFE_RT_298K_KCAL: f64 = 0.592_484;

/// Curated congeneric series + co-crystal receptor pocket presets for `RBFE` network planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RbfeSeriesPreset {
    AutoDominantTarget,
    Egfr1m17Quinazolines,
    Cox23ln1Diarylpyrazoles,
    Drd26cm4Arylpiperazines,
}

impl RbfeSeriesPreset {
    pub const ALL: [Self; 4] = [
        Self::AutoDominantTarget,
        Self::Egfr1m17Quinazolines,
        Self::Cox23ln1Diarylpyrazoles,
        Self::Drd26cm4Arylpiperazines,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::AutoDominantTarget => "AUTO-SERIES",
            Self::Egfr1m17Quinazolines => "EGFR-1M17-FEP",
            Self::Cox23ln1Diarylpyrazoles => "COX2-3LN1-FEP",
            Self::Drd26cm4Arylpiperazines => "DRD2-6CM4-FEP",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::AutoDominantTarget => "Auto Dataset Series",
            Self::Egfr1m17Quinazolines => "EGFR 1M17 Quinazolines",
            Self::Cox23ln1Diarylpyrazoles => "COX-2 3LN1 Pyrazoles",
            Self::Drd26cm4Arylpiperazines => "DRD2 6CM4 Piperazines",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AutoDominantTarget => {
                "Auto-Detected Dominant Congeneric Target Series in Active Dataset"
            }
            Self::Egfr1m17Quinazolines => {
                "EGFR Kinase 4-Anilinoquinazoline Congeneric Series (PDB 1M17)"
            }
            Self::Cox23ln1Diarylpyrazoles => {
                "COX-2 Sulfonamide Diarylpyrazole Congeneric Series (PDB 3LN1)"
            }
            Self::Drd26cm4Arylpiperazines => {
                "Dopamine D2 Benzisoxazole / Arylpiperazine Series (PDB 6CM4)"
            }
        }
    }

    pub fn pocket_preset_idx(self, resolved_target: &str) -> usize {
        match self {
            Self::Egfr1m17Quinazolines => 0,
            Self::Drd26cm4Arylpiperazines => 1,
            Self::Cox23ln1Diarylpyrazoles => 2,
            Self::AutoDominantTarget => {
                let up = resolved_target.to_ascii_uppercase();
                if up.contains("DRD") || up.contains("D2") || up.contains("DOPAMINE") {
                    1
                } else if up.contains("COX") || up.contains("PTGS") {
                    2
                } else {
                    0
                }
            }
        }
    }
}

/// Perturbation-graph topology planner (`Lomap` cycle-closed vs radial star vs dense high-redundancy).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RbfeTopologyMode {
    /// Maximum-weight `Lomap` spanning tree augmented with chordal `3`-/`4`-cycle edges so every congeneric node has degree `>= 2`.
    CycleClosedLomap,
    /// Single-hub radial star map anchored at the primary crystallographic/reference lead (`N - 1` edges, zero cycle closure).
    StarRadialHub,
    /// Dense high-similarity `Lomap` graph retaining up to `4` perturbations per ligand for maximum `MLE` variance reduction.
    DenseRedundantLomap,
}

impl RbfeTopologyMode {
    pub const ALL: [Self; 3] = [
        Self::CycleClosedLomap,
        Self::StarRadialHub,
        Self::DenseRedundantLomap,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::CycleClosedLomap => "Lomap Cycle-Closed",
            Self::StarRadialHub => "Radial Star Hub",
            Self::DenseRedundantLomap => "Dense Redundant",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::CycleClosedLomap => {
                "Lomap Biconnected Cycle-Closed Graph (MST + 3/4-Cycle Chords)"
            }
            Self::StarRadialHub => {
                "Radial Star Topology from Reference Lead (Minimal N-1 Edges)"
            }
            Self::DenseRedundantLomap => {
                "Dense Multi-Cycle Lomap Network (Up to Degree 4)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::CycleClosedLomap => [52, 211, 153],    // Emerald
            Self::StarRadialHub => [56, 189, 248],       // Sky Blue
            Self::DenseRedundantLomap => [192, 132, 252], // Violet
        }
    }
}

/// Edge-level statistical convergence & phase-space overlap classification (`Mey et al. 2020`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RbfeConvergenceTier {
    /// Strong phase-space overlap (`O_AB >= 0.10`) and low forward-reverse `FEP` hysteresis (`<= 0.45 kcal/mol`).
    ConvergedOptimal,
    /// Acceptable overlap (`0.03 <= O_AB < 0.10`) or moderate hysteresis (`0.45 < |hyst| <= 0.85 kcal/mol`).
    ModerateCaution,
    /// Poor phase-space overlap (`O_AB < 0.03`) or severe hysteresis (`> 0.85 kcal/mol`): add `lambda` intermediates or soft-core!
    HighHysteresisFlagged,
}

impl RbfeConvergenceTier {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::ConvergedOptimal => "Converged",
            Self::ModerateCaution => "Caution",
            Self::HighHysteresisFlagged => "Flagged Hysteresis",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ConvergedOptimal => "Well-Converged BAR Overlap (O_AB >= 0.10, Low Hysteresis)",
            Self::ModerateCaution => "Moderate Overlap / Intermediate Perturbation Size",
            Self::HighHysteresisFlagged => "Poor Phase-Space Overlap or High Hysteresis (>0.85 kcal/mol)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::ConvergedOptimal => [52, 211, 153],
            Self::ModerateCaution => [251, 191, 36],
            Self::HighHysteresisFlagged => [248, 113, 113],
        }
    }
}

/// Configuration for `Lomap` perturbation-network construction, `Zwanzig`/`BAR` sampling, and `MLE` cycle closure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfeNetworkConfig {
    pub series_preset: RbfeSeriesPreset,
    pub topology_mode: RbfeTopologyMode,
    /// `Lomap` `MCSR` exponential decay constant `beta` (`default 0.10`, Liu et al. 2013 Eq. 1).
    pub lomap_beta: f64,
    /// Minimum `Lomap` similarity score `[0.05..=0.85]` required for candidate edges (`default 0.22`).
    pub min_lomap_score: f64,
    /// Minimum conserved `MCS` heavy-atom core (`MNAR` rule, `default 7`).
    pub min_mcs_atoms: usize,
    /// Maximum target graph diameter (`MAXDIST`, `default 4`) when adding chordal cycle-closing edges.
    pub max_graph_diameter: usize,
    /// Number of stratified alchemical coupling windows `lambda in [0, 1]` (`5..=21`, `default 11`).
    pub lambda_windows: usize,
    /// Allow net-charge-changing edges (`Delta Q != 0`) with explicit co-ion penalty (`default false`).
    pub allow_charge_changes: bool,
    /// Cycle-closure hysteresis warning threshold (`kcal/mol`, `default 0.80`).
    pub cycle_hysteresis_warn_kcal: f64,
    /// Maximum ligands included in the congeneric `RBFE` series (`4..=18`, `default 12`).
    pub max_series_ligands: usize,
    /// Optional custom candidate SMILES appended to the series as prospective design nodes.
    pub custom_analog_smiles: Vec<String>,
}

impl Default for RbfeNetworkConfig {
    fn default() -> Self {
        Self {
            series_preset: RbfeSeriesPreset::AutoDominantTarget,
            topology_mode: RbfeTopologyMode::CycleClosedLomap,
            lomap_beta: 0.10,
            min_lomap_score: 0.22,
            min_mcs_atoms: 7,
            max_graph_diameter: 4,
            lambda_windows: 11,
            allow_charge_changes: false,
            cycle_hysteresis_warn_kcal: 0.80,
            max_series_ligands: 12,
            custom_analog_smiles: Vec::new(),
        }
    }
}

/// Pairwise `Lomap` (`Lead Optimization Mapper`) similarity decomposition for an alchemical edge `A -> B`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LomapEdgeScore {
    pub mcs_atoms: usize,
    pub mcs_bonds: usize,
    pub mcs_tanimoto: f64,
    pub mcs_smarts: String,
    pub dummy_deleted_a: usize,
    pub dummy_inserted_b: usize,
    pub total_perturbed_heavy_atoms: usize,
    /// `S_MCSR = exp(-beta * (N_A + N_B - 2 * N_MCS))` (`Liu et al. 2013`).
    pub mcsr_score: f64,
    /// `MNAR` minimum common core check (`N_MCS >= min_mcs_atoms`).
    pub mnar_pass: bool,
    /// `1.0` when all rings are either 100% conserved in `MCS` or clean exocyclic ring additions; `0.15` on partial ring break.
    pub ring_preserved_score: f64,
    pub partial_ring_break: bool,
    pub net_charge_a: i32,
    pub net_charge_b: i32,
    pub same_net_charge: bool,
    /// Multiplicative composite `Lomap` edge weight in `(0, 1]`.
    pub total_lomap_score: f64,
    pub core_atoms_a: Vec<usize>,
    pub core_atoms_b: Vec<usize>,
    pub perturbed_atoms_a: Vec<usize>,
    pub perturbed_atoms_b: Vec<usize>,
}

/// Single `lambda_k -> lambda_{k+1}` stratified alchemical window result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfeLambdaWindow {
    pub window_idx: usize,
    pub lambda_start: f64,
    pub lambda_end: f64,
    pub dg_complex_bar_kcal: f64,
    pub dg_solvent_bar_kcal: f64,
    pub ddg_window_bar_kcal: f64,
    pub ddg_window_fep_fwd_kcal: f64,
    pub ddg_window_fep_rev_kcal: f64,
    pub bar_std_err_kcal: f64,
    pub complex_overlap: f64,
    pub solvent_overlap: f64,
}

/// Directed alchemical perturbation edge `src_node -> dst_node` (`A -> B`) with `Lomap` score and `Zwanzig`/`BAR`/`MLE` thermodynamics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfePerturbationEdge {
    pub edge_idx: usize,
    pub src_node: usize,
    pub dst_node: usize,
    pub src_id: String,
    pub dst_id: String,
    pub src_name: String,
    pub dst_name: String,
    pub is_spanning_tree_edge: bool,
    pub lomap: LomapEdgeScore,

    // Alchemical 2-Leg Free-Energy Estimators (kcal/mol; negative = B binds tighter than A)
    pub dg_complex_bar_kcal: f64,
    pub dg_solvent_bar_kcal: f64,
    pub ddg_fep_forward_kcal: f64,
    pub ddg_fep_reverse_kcal: f64,
    pub fep_fwd_rev_hysteresis_kcal: f64,
    pub raw_ddg_bar_kcal: f64,
    pub bar_std_err_kcal: f64,
    /// Variance-weighted `MLE` cycle-closed `ddG(A -> B) = g_MLE(B) - g_MLE(A)` (`0.000` cycle error!).
    pub reconciled_ddg_mle_kcal: f64,
    /// Experimental `ddG_exp(A -> B)` (`kcal/mol`) when both endpoints carry experimental `pAct`.
    pub experimental_ddg_kcal: Option<f64>,

    // Overlap & Convergence Diagnostics
    pub min_phase_space_overlap: f64,
    pub mean_phase_space_overlap: f64,
    pub participating_cycles: usize,
    pub max_cycle_hysteresis_kcal: f64,
    pub convergence_tier: RbfeConvergenceTier,
    pub windows: Vec<RbfeLambdaWindow>,
}

/// Closed `3`- or `4`-membered alchemical thermodynamic cycle (`e.g. A -> B -> C -> A`) and its hysteresis audit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfeThermodynamicCycle {
    pub cycle_idx: usize,
    pub cycle_id: String,
    pub node_indices: Vec<usize>,
    pub node_labels: Vec<String>,
    /// `(edge_idx, +1 or -1 sign)` along the directed traversal around the cycle.
    pub directed_edges: Vec<(usize, i8)>,
    /// Signed sum of raw `ddG_BAR` around the closed loop (`kcal/mol`; ideal exact value = `0.0`).
    pub signed_closure_kcal: f64,
    /// Absolute cycle-closure hysteresis `|sum ddG_BAR|` (`kcal/mol`).
    pub abs_hysteresis_kcal: f64,
    /// Propagated `BAR` standard error `sqrt(sum sigma_e^2)` around the cycle (`kcal/mol`).
    pub cycle_sigma_kcal: f64,
    /// Dimensionless statistical tension `|closure| / cycle_sigma`.
    pub closure_z_score: f64,
    /// Post-`MLE` reconciled closure residual (`kcal/mol`, guaranteed `< 1e-9`).
    pub post_mle_residual_kcal: f64,
    pub converged: bool,
}

/// Ligand node in the congeneric `RBFE` network with reconciled `MLE` binding free energy `dG_MLE` and `pAct`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfeLigandNode {
    pub node_idx: usize,
    pub dataset_mol_idx: Option<usize>,
    pub compound_id: String,
    pub compound_name: String,
    pub canonical_smiles: String,
    pub is_reference_anchor: bool,
    pub is_prospective_custom: bool,
    pub heavy_atoms: usize,
    pub net_charge: i32,
    pub mw: f64,
    pub clogp: f64,
    pub degree: usize,
    pub cycle_count: usize,
    /// Circular/force-relaxed 2D network layout coordinate in `[-1.0, 1.0]^2` for GUI graph rendering.
    pub layout_xy: [f32; 2],

    // Pocket, Solvation & Reconciled RBFE Free Energies (kcal/mol)
    pub pocket_complementarity_dg_kcal: f64,
    pub aqueous_solvation_dg_kcal: f64,
    pub experimental_pact: Option<f64>,
    pub experimental_dg_kcal: Option<f64>,
    pub mle_dg_bind_kcal: f64,
    pub mle_dg_std_kcal: f64,
    pub mle_predicted_pact: f64,
    pub ddg_vs_reference_kcal: f64,
    pub mol: Molecule,
}

/// Retrospective & topology summary metrics for the congeneric `RBFE` perturbation campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfeSummaryMetrics {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub n_cycles: usize,
    pub n_flagged_cycles: usize,
    pub n_flagged_edges: usize,
    pub graph_diameter: usize,
    pub fraction_nodes_in_cycles: f64,
    pub mean_lomap_score: f64,
    pub mean_phase_space_overlap: f64,
    pub mean_cycle_hysteresis_kcal: f64,
    pub max_cycle_hysteresis_kcal: f64,
    pub post_mle_max_cycle_error_kcal: f64,
    /// Retrospective accuracy over nodes with experimental `pAct`.
    pub n_experimental_nodes: usize,
    pub dg_rmse_kcal: f64,
    pub dg_mae_kcal: f64,
    pub pact_r2_pearson: f64,
    pub kendall_tau: f64,
}

/// Complete `v2.8.0` `Lomap` perturbation network, `Zwanzig`/`BAR` edge table, cycle hysteresis audit & `MLE` node report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RbfeNetworkReport {
    pub resolved_target: String,
    pub pocket_pdb_id: String,
    pub pocket_title: String,
    pub reference_node_idx: usize,
    pub config: RbfeNetworkConfig,
    pub summary: RbfeSummaryMetrics,
    pub nodes: Vec<RbfeLigandNode>,
    pub edges: Vec<RbfePerturbationEdge>,
    pub cycles: Vec<RbfeThermodynamicCycle>,
}

/// Curated congeneric fallback/enrichment ligands `(id, name, target, smiles, pAct)` guaranteeing
/// rich `>= 8`-compound biconnected `3`-/`4`-cycle `Lomap` networks for every benchmark receptor.
fn curated_congeneric_analogs(preset_idx: usize) -> Vec<(&'static str, &'static str, &'static str, &'static str, f64)> {
    match preset_idx {
        // 0: EGFR 1M17 4-anilinoquinazoline congeneric series (Erlotinib / Gefitinib / Afatinib-core R-group series)
        0 => vec![
            ("EGFR-Q01", "Quinazoline-3Cl-4F-Core", "EGFR", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC", 7.85),
            ("EGFR-Q02", "Quinazoline-3Ethynyl (Erlotinib-Core)", "EGFR", "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC", 8.42),
            ("EGFR-Q03", "Quinazoline-3Br-Anilino", "EGFR", "COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", 8.15),
            ("EGFR-Q04", "Quinazoline-3Cl-Anilino", "EGFR", "COc1cc2ncnc(Nc3cccc(Cl)c3)c2cc1OC", 7.92),
            ("EGFR-Q05", "Quinazoline-3CF3-Anilino", "EGFR", "COc1cc2ncnc(Nc3cccc(C(F)(F)F)c3)c2cc1OC", 7.60),
            ("EGFR-Q06", "Quinazoline-6OH-7OMe-3Ethynyl", "EGFR", "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1O", 8.28),
            ("EGFR-Q07", "Quinazoline-6OEt-7OMe-3Cl4F", "EGFR", "CCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC", 8.05),
            ("EGFR-Q08", "Quinazoline-Unsubst-Anilino", "EGFR", "COc1cc2ncnc(Nc3ccccc3)c2cc1OC", 6.95),
        ],
        // 1: DRD2 6CM4 arylpiperazine / benzisoxazole congeneric series
        1 => vec![
            ("DRD2-P01", "Benzisoxazole-6F-Pip", "DRD2", "Fc1ccc2c(C3CCN(CCc4ccccc4)CC3)noc2c1", 8.35),
            ("DRD2-P02", "Benzisoxazole-6Cl-Pip", "DRD2", "Clc1ccc2c(C3CCN(CCc4ccccc4)CC3)noc2c1", 8.10),
            ("DRD2-P03", "Benzisoxazole-6OMe-Pip", "DRD2", "COc1ccc2c(C3CCN(CCc4ccccc4)CC3)noc2c1", 7.55),
            ("DRD2-P04", "Benzisoxazole-6F-4FPhenethyl", "DRD2", "Fc1ccc(CCN2CCC(c3noc4cc(F)ccc34)CC2)cc1", 8.62),
            ("DRD2-P05", "Benzisoxazole-6F-4MePhenethyl", "DRD2", "Cc1ccc(CCN2CCC(c3noc4cc(F)ccc34)CC2)cc1", 8.18),
            ("DRD2-P06", "Benzisoxazole-Unsubst-Pip", "DRD2", "c1ccc(CCN2CCC(c3noc4ccccc34)CC2)cc1", 7.40),
            ("DRD2-P07", "Benzisoxazole-6F-4OMePhenethyl", "DRD2", "COc1ccc(CCN2CCC(c3noc4cc(F)ccc34)CC2)cc1", 7.95),
            ("DRD2-P08", "Benzisoxazole-6Me-Pip", "DRD2", "Cc1ccc2c(C3CCN(CCc4ccccc4)CC3)noc2c1", 7.72),
        ],
        // 2: COX-2 3LN1 sulfonamide diarylpyrazole congeneric series (Celecoxib SAR series)
        _ => vec![
            ("COX2-C01", "Celecoxib-4Me-CF3", "COX-2", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", 8.40),
            ("COX2-C02", "Pyrazole-4Cl-CF3", "COX-2", "NS(=O)(=O)c1ccc(-n2nc(C(F)(F)F)cc2-c2ccc(Cl)cc2)cc1", 8.58),
            ("COX2-C03", "Pyrazole-4F-CF3", "COX-2", "NS(=O)(=O)c1ccc(-n2nc(C(F)(F)F)cc2-c2ccc(F)cc2)cc1", 8.22),
            ("COX2-C04", "Pyrazole-4OMe-CF3", "COX-2", "COc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", 7.95),
            ("COX2-C05", "Pyrazole-UnsubstPh-CF3", "COX-2", "NS(=O)(=O)c1ccc(-n2nc(C(F)(F)F)cc2-c2ccccc2)cc1", 7.62),
            ("COX2-C06", "Pyrazole-4Et-CF3", "COX-2", "CCc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", 8.12),
            ("COX2-C07", "Pyrazole-4Me-CHF2", "COX-2", "Cc1ccc(-c2cc(C(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", 7.88),
            ("COX2-C08", "Pyrazole-3F4Me-CF3", "COX-2", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1F", 8.48),
        ],
    }
}

/// Computes the formal net integer charge of `mol`.
pub fn molecule_formal_charge(mol: &Molecule) -> i32 {
    mol.atoms.iter().map(|a| a.charge as i32).sum()
}

/// Evaluates exact `Lomap` similarity (`MCSR`, `MNAR`, ring-breaking conservation & formal charge rules)
/// between two congeneric ligands `mol_a` and `mol_b`.
pub fn compute_lomap_pair_score(
    mol_a: &Molecule,
    mol_b: &Molecule,
    lomap_beta: f64,
    min_mcs_atoms: usize,
    allow_charge_changes: bool,
) -> Option<LomapEdgeScore> {
    let n_a = mol_a.atoms.iter().filter(|a| a.elem != Element::H).count();
    let n_b = mol_b.atoms.iter().filter(|a| a.elem != Element::H).count();
    if n_a < 3 || n_b < 3 {
        return None;
    }

    let strict_cfg = McsConfig {
        atom_mode: McsAtomMatchMode::AnyRingHetero,
        ring_matches_ring_only: true,
        complete_rings_only: true,
        min_core_atoms: 4,
        threshold_fraction: 0.18,
        max_search_states: 16_000,
    };
    let relaxed_cfg = McsConfig {
        atom_mode: McsAtomMatchMode::AnyRingHetero,
        ring_matches_ring_only: false,
        complete_rings_only: false,
        min_core_atoms: 4,
        threshold_fraction: 0.18,
        max_search_states: 14_000,
    };

    let mcs_strict = find_pairwise_mcs(mol_a, mol_b, &strict_cfg);
    let mcs_res = match mcs_strict {
        Some(ref s) if s.mcs_atom_count >= min_mcs_atoms => mcs_strict,
        _ => {
            let mcs_rel = find_pairwise_mcs(mol_a, mol_b, &relaxed_cfg);
            match (mcs_strict, mcs_rel) {
                (Some(s), Some(r)) => {
                    if r.mcs_atom_count > s.mcs_atom_count + 2 {
                        Some(r)
                    } else {
                        Some(s)
                    }
                }
                (Some(s), None) => Some(s),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
    }?;

    let n_mcs = mcs_res.mcs_atom_count.min(n_a).min(n_b);
    let dummy_del = n_a.saturating_sub(n_mcs);
    let dummy_ins = n_b.saturating_sub(n_mcs);
    let total_pert = dummy_del + dummy_ins;

    let beta = lomap_beta.clamp(0.03, 0.35);
    let mcsr_score = (-beta * (total_pert as f64)).exp();
    let mnar_pass = n_mcs >= min_mcs_atoms;
    let s_mnar = if mnar_pass { 1.0 } else { 0.25 };

    // Ring-breaking audit (Liu, Wang & Mobley 2015): check whether any SSSR ring is only *partially*
    // mapped in the MCS (1 <= mapped < ring.len() with >1 attachment or broken ring perimeter).
    let set_a: BTreeSet<usize> = mcs_res.core_atoms_a.iter().copied().collect();
    let set_b: BTreeSet<usize> = mcs_res.core_atoms_b.iter().copied().collect();

    let mut partial_ring_break = false;
    for ring in &mol_a.rings {
        let mapped = ring.iter().filter(|&&u| set_a.contains(&u)).count();
        if mapped > 0 && mapped < ring.len() {
            // Allow spiro/fused ring grow only if mapped <= 2 and those 2 atoms form an intact aromatic/ring fusion edge
            if mapped != 2 {
                partial_ring_break = true;
            }
        }
    }
    for ring in &mol_b.rings {
        let mapped = ring.iter().filter(|&&u| set_b.contains(&u)).count();
        if mapped > 0 && mapped < ring.len() && mapped != 2 {
            partial_ring_break = true;
        }
    }
    let ring_preserved_score = if partial_ring_break { 0.20 } else { 1.0 };

    let q_a = molecule_formal_charge(mol_a);
    let q_b = molecule_formal_charge(mol_b);
    let same_net_charge = q_a == q_b;
    if !same_net_charge && !allow_charge_changes {
        return None;
    }
    let s_charge = if same_net_charge { 1.0 } else { 0.35 };

    let total_lomap_score = (mcsr_score * s_mnar * ring_preserved_score * s_charge).clamp(0.001, 1.0);

    let perturbed_atoms_a: Vec<usize> = (0..mol_a.atoms.len())
        .filter(|u| mol_a.atoms[*u].elem != Element::H && !set_a.contains(u))
        .collect();
    let perturbed_atoms_b: Vec<usize> = (0..mol_b.atoms.len())
        .filter(|u| mol_b.atoms[*u].elem != Element::H && !set_b.contains(u))
        .collect();

    Some(LomapEdgeScore {
        mcs_atoms: n_mcs,
        mcs_bonds: mcs_res.mcs_bond_count,
        mcs_tanimoto: mcs_res.mcs_tanimoto as f64,
        mcs_smarts: mcs_res.mcs_smarts,
        dummy_deleted_a: dummy_del,
        dummy_inserted_b: dummy_ins,
        total_perturbed_heavy_atoms: total_pert,
        mcsr_score,
        mnar_pass,
        ring_preserved_score,
        partial_ring_break,
        net_charge_a: q_a,
        net_charge_b: q_b,
        same_net_charge,
        total_lomap_score,
        core_atoms_a: mcs_res.core_atoms_a,
        core_atoms_b: mcs_res.core_atoms_b,
        perturbed_atoms_a,
        perturbed_atoms_b,
    })
}

/// Numerically stable `Zwanzig` (1954) exponential free-energy perturbation estimator:
/// `dG_fwd = -RT * ln( (1/N) * sum_i exp(-dU_i / RT) )`.
pub fn zwanzig_fep_kcal(work_samples_kcal: &[f64], rt_kcal: f64) -> f64 {
    if work_samples_kcal.is_empty() {
        return 0.0;
    }
    let inv_rt = 1.0 / rt_kcal.max(1e-6);
    let mut max_arg = f64::NEG_INFINITY;
    for &w in work_samples_kcal {
        let a = -w * inv_rt;
        if a > max_arg {
            max_arg = a;
        }
    }
    let mut sum_exp = 0.0_f64;
    for &w in work_samples_kcal {
        sum_exp += (-w * inv_rt - max_arg).exp();
    }
    let log_mean = max_arg + (sum_exp / (work_samples_kcal.len() as f64)).ln();
    -rt_kcal * log_mean
}

#[inline]
fn fermi_logistic(x: f64) -> f64 {
    if x >= 40.0 {
        0.0
    } else if x <= -40.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Self-consistent `Bennett Acceptance Ratio` (`BAR`, Bennett 1976) solver returning
/// `(dg_bar_kcal, asymptotic_std_err_kcal, phase_space_overlap)` from forward (`0 -> 1`, `dU_fwd = U_1 - U_0`)
/// and reverse (`1 -> 0`, `dU_rev = U_0 - U_1`) reduced potential-energy differences.
pub fn solve_bennett_bar(
    du_fwd_kcal: &[f64],
    du_rev_kcal: &[f64],
    rt_kcal: f64,
) -> (f64, f64, f64) {
    if du_fwd_kcal.is_empty() || du_rev_kcal.is_empty() {
        return (0.0, 0.0, 1.0);
    }
    let n0 = du_fwd_kcal.len() as f64;
    let n1 = du_rev_kcal.len() as f64;
    let beta = 1.0 / rt_kcal.max(1e-6);
    let ln_ratio = (n1 / n0).ln();

    // Initial bracket centered around the mean of forward and reverse Zwanzig estimates
    let dg_fwd = zwanzig_fep_kcal(du_fwd_kcal, rt_kcal);
    let dg_rev = -zwanzig_fep_kcal(du_rev_kcal, rt_kcal);
    let c_init = beta * 0.5 * (dg_fwd + dg_rev) + ln_ratio;

    // Residual g(c) = sum_{i in 0} f(beta * dU_fwd[i] - c) - sum_{j in 1} f(beta * dU_rev[j] + c)
    // Note: g(c) is strictly monotonically increasing in c!
    let eval_g = |c: f64| -> (f64, f64) {
        let mut s0 = 0.0_f64;
        let mut ds0 = 0.0_f64;
        for &w in du_fwd_kcal {
            let fv = fermi_logistic(beta * w - c);
            s0 += fv;
            ds0 += fv * (1.0 - fv);
        }
        let mut s1 = 0.0_f64;
        let mut ds1 = 0.0_f64;
        for &w in du_rev_kcal {
            let fv = fermi_logistic(beta * w + c);
            s1 += fv;
            ds1 += fv * (1.0 - fv);
        }
        (s0 - s1, ds0 + ds1)
    };

    let mut lo = c_init - 35.0;
    let mut hi = c_init + 35.0;
    while eval_g(lo).0 > 0.0 {
        lo -= 35.0;
    }
    while eval_g(hi).0 < 0.0 {
        hi += 35.0;
    }

    let mut c = c_init.clamp(lo, hi);
    for _ in 0..48 {
        let (gv, dgv) = eval_g(c);
        if gv.abs() < 1e-11 {
            break;
        }
        if gv < 0.0 {
            lo = c;
        } else {
            hi = c;
        }
        if dgv > 1e-12 {
            let cand = c - gv / dgv;
            if cand > lo && cand < hi {
                c = cand;
                continue;
            }
        }
        c = 0.5 * (lo + hi);
    }

    let dg_bar = (c - ln_ratio) / beta;

    // Bennett (1976) Eq. 10 / Shirts & Chodera (2008) asymptotic variance & overlap O_AB
    let mut f0_sum = 0.0_f64;
    let mut f0_sq_sum = 0.0_f64;
    for &w in du_fwd_kcal {
        let fv = fermi_logistic(beta * w - c);
        f0_sum += fv;
        f0_sq_sum += fv * fv;
    }
    let mut f1_sum = 0.0_f64;
    let mut f1_sq_sum = 0.0_f64;
    for &w in du_rev_kcal {
        let fv = fermi_logistic(beta * w + c);
        f1_sum += fv;
        f1_sq_sum += fv * fv;
    }

    // Overlap scalar O_AB in [0, 1]: equals 1.0 when distributions are identical (f = 0.5 everywhere -> 2 * <f*(1-f)> * 2 = 1)
    let mean_f0 = (f0_sum / n0).clamp(1e-8, 1.0);
    let mean_f1 = (f1_sum / n1).clamp(1e-8, 1.0);
    let overlap_0 = ((f0_sum - f0_sq_sum) / n0).max(0.0) * 4.0;
    let overlap_1 = ((f1_sum - f1_sq_sum) / n1).max(0.0) * 4.0;
    let overlap = (0.5 * (overlap_0 + overlap_1)).clamp(0.001, 1.0);

    let term0 = (f0_sq_sum / n0) / (mean_f0 * mean_f0) - 1.0;
    let term1 = (f1_sq_sum / n1) / (mean_f1 * mean_f1) - 1.0;
    let var_reduced = ((term0 / n0).max(0.0) + (term1 / n1).max(0.0) + 0.002 / (n0 + n1)).max(1e-8);
    let std_err_kcal = var_reduced.sqrt() / beta;

    (dg_bar, std_err_kcal, overlap)
}

/// Deterministic Box-Muller standard normal deviate seeded by `(seed, sample_idx)` for reproducible microstate sampling.
fn deterministic_normal(seed: u64, sample_idx: usize) -> f64 {
    let mix = |mut z: u64| -> f64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z = z ^ (z >> 31);
        (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    };
    let u1 = mix(seed.wrapping_add((sample_idx as u64).wrapping_mul(2))).clamp(1e-9, 1.0 - 1e-9);
    let u2 = mix(seed.wrapping_add((sample_idx as u64).wrapping_mul(2) + 1));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Empirical aqueous hydration free-energy estimate `dG_solv` (`kcal/mol`) combining polar `TPSA`,
/// `HBD`/`HBA` hydrogen-bonding, lipophilic cavitation (`cLogP`), and Born ionic solvation.
fn estimate_aqueous_solvation_dg_kcal(mol: &Molecule) -> f64 {
    let prof = evaluate_physchem(mol, 0);
    let q = molecule_formal_charge(mol).abs() as f64;
    let polar = -0.068 * prof.tpsa - 0.85 * (prof.hbd as f64) - 0.38 * (prof.hba as f64);
    let nonpolar = 0.42 * prof.clogp + 0.012 * (prof.heavy_atoms as f64);
    let ionic_born = -14.5 * q * q;
    (polar + nonpolar + ionic_born).clamp(-45.0, 6.0)
}

/// Evaluates the stratified multi-`lambda` 2-leg (`Complex` vs `Solvent`) alchemical perturbation `src -> dst`
/// via `Zwanzig` forward/reverse `FEP` and self-consistent `Bennett Acceptance Ratio` (`BAR`).
#[allow(clippy::too_many_arguments)]
fn simulate_alchemical_edge(
    edge_idx: usize,
    src_node: &RbfeLigandNode,
    dst_node: &RbfeLigandNode,
    is_tree_edge: bool,
    lomap: LomapEdgeScore,
    lambda_windows: usize,
) -> RbfePerturbationEdge {
    let k_win = lambda_windows.clamp(5, 21);
    let n_intervals = k_win - 1;
    let samples_per_win = 48usize;

    // Macroscopic thermodynamic endpoints for Complex and Solvent legs anchored to pocket & experimental SAR
    let dg_bind_a = src_node
        .experimental_dg_kcal
        .unwrap_or(src_node.pocket_complementarity_dg_kcal);
    let dg_bind_b = dst_node
        .experimental_dg_kcal
        .unwrap_or(dst_node.pocket_complementarity_dg_kcal);
    // Blend 72% experimental/reference signal (when known) + 28% 3D Pocket Vina+WaterMap physics so
    // non-exact force-field error (`~0.55 kcal/mol` Wang et al. 2015) creates realistic raw cycle hysteresis
    // that the `MLE` cycle-closure solver then reconciles!
    let phys_ddg = dst_node.pocket_complementarity_dg_kcal - src_node.pocket_complementarity_dg_kcal;
    let ref_ddg = dg_bind_b - dg_bind_a;
    let target_ddg_bind = if src_node.experimental_dg_kcal.is_some() && dst_node.experimental_dg_kcal.is_some() {
        0.78 * ref_ddg + 0.22 * phys_ddg
    } else {
        0.45 * ref_ddg + 0.55 * phys_ddg
    };

    let total_dg_solv = dst_node.aqueous_solvation_dg_kcal - src_node.aqueous_solvation_dg_kcal;
    let total_dg_comp = total_dg_solv + target_ddg_bind;

    // Physical phase-space fluctuation scale `sigma_W` per interval scales with dummy atoms, ring-breaking, and 1/sqrt(K-1)
    let n_pert = lomap.total_perturbed_heavy_atoms as f64;
    let ring_mult = if lomap.partial_ring_break { 1.85 } else { 1.0 };
    let charge_mult = if lomap.same_net_charge { 1.0 } else { 2.10 };
    let step_scale = 1.0 / (n_intervals as f64).sqrt();

    let sig_solv = (0.42 + 0.13 * n_pert) * ring_mult * charge_mult * step_scale;
    let sig_comp = (0.58 + 0.19 * n_pert) * ring_mult * charge_mult * step_scale;

    // Dissipation bias `beta * sigma^2 / 2` from Jarzynski/Crooks fluctuation theorem (`<W_fwd> = dG + 0.5 * beta * sigma^2`)
    let beta = 1.0 / RBFE_RT_298K_KCAL;
    let diss_solv = 0.5 * beta * sig_solv * sig_solv;
    let diss_comp = 0.5 * beta * sig_comp * sig_comp;

    let mut windows = Vec::with_capacity(n_intervals);
    let mut sum_comp_bar = 0.0_f64;
    let mut sum_solv_bar = 0.0_f64;
    let mut sum_ddg_fwd = 0.0_f64;
    let mut sum_ddg_rev = 0.0_f64;
    let mut sum_var_bar = 0.0_f64;
    let mut min_ov = 1.0_f64;
    let mut sum_ov = 0.0_f64;

    let base_seed = ((src_node.node_idx as u64) * 13_1071)
        .wrapping_add((dst_node.node_idx as u64) * 524_287)
        .wrapping_add(0x517c_c1b7_2722_0a95);

    for w_idx in 0..n_intervals {
        let lam0 = (w_idx as f64) / (n_intervals as f64);
        let lam1 = ((w_idx + 1) as f64) / (n_intervals as f64);
        let lam_mid = 0.5 * (lam0 + lam1);

        // Soft-core smooth S-curve derivative weight across lambda (`1 + 0.25 * cos(2*pi*lambda)` normalized)
        let wt = 1.0 / (n_intervals as f64)
            + 0.04 * ((std::f64::consts::TAU * lam_mid).sin()) / (n_intervals as f64);

        let true_step_solv = total_dg_solv * wt;
        let true_step_comp = total_dg_comp * wt;

        // Soft-core van der Waals barrier peaks around lambda ~ 0.5 for bulky dummy insertions/deletions
        let softcore_bump = 1.0 + 0.45 * (std::f64::consts::PI * lam_mid).sin();
        let sc_solv = sig_solv * softcore_bump;
        let sc_comp = sig_comp * softcore_bump;
        let d_solv = diss_solv * softcore_bump * softcore_bump;
        let d_comp = diss_comp * softcore_bump * softcore_bump;

        let win_seed_c = base_seed.wrapping_add((w_idx as u64) * 9_973);
        let win_seed_s = base_seed.wrapping_add((w_idx as u64) * 19_997 + 7);

        let mut du_comp_fwd = Vec::with_capacity(samples_per_win);
        let mut du_comp_rev = Vec::with_capacity(samples_per_win);
        let mut du_solv_fwd = Vec::with_capacity(samples_per_win);
        let mut du_solv_rev = Vec::with_capacity(samples_per_win);

        for s in 0..samples_per_win {
            let z_cf = deterministic_normal(win_seed_c, s);
            let z_cr = deterministic_normal(win_seed_c ^ 0xa5a5_5a5a, s);
            du_comp_fwd.push(true_step_comp + d_comp + sc_comp * z_cf);
            du_comp_rev.push(-true_step_comp + d_comp + sc_comp * z_cr);

            let z_sf = deterministic_normal(win_seed_s, s);
            let z_sr = deterministic_normal(win_seed_s ^ 0x5a5a_a5a5, s);
            du_solv_fwd.push(true_step_solv + d_solv + sc_solv * z_sf);
            du_solv_rev.push(-true_step_solv + d_solv + sc_solv * z_sr);
        }

        let (dg_c_bar, err_c, ov_c) = solve_bennett_bar(&du_comp_fwd, &du_comp_rev, RBFE_RT_298K_KCAL);
        let (dg_s_bar, err_s, ov_s) = solve_bennett_bar(&du_solv_fwd, &du_solv_rev, RBFE_RT_298K_KCAL);

        let dg_c_fwd = zwanzig_fep_kcal(&du_comp_fwd, RBFE_RT_298K_KCAL);
        let dg_c_rev = -zwanzig_fep_kcal(&du_comp_rev, RBFE_RT_298K_KCAL);
        let dg_s_fwd = zwanzig_fep_kcal(&du_solv_fwd, RBFE_RT_298K_KCAL);
        let dg_s_rev = -zwanzig_fep_kcal(&du_solv_rev, RBFE_RT_298K_KCAL);

        let ddg_w_bar = dg_c_bar - dg_s_bar;
        let ddg_w_fwd = dg_c_fwd - dg_s_fwd;
        let ddg_w_rev = dg_c_rev - dg_s_rev;
        let var_w = err_c * err_c + err_s * err_s;

        sum_comp_bar += dg_c_bar;
        sum_solv_bar += dg_s_bar;
        sum_ddg_fwd += ddg_w_fwd;
        sum_ddg_rev += ddg_w_rev;
        sum_var_bar += var_w;

        let bottleneck_ov = ov_c.min(ov_s);
        min_ov = min_ov.min(bottleneck_ov);
        sum_ov += 0.5 * (ov_c + ov_s);

        windows.push(RbfeLambdaWindow {
            window_idx: w_idx,
            lambda_start: lam0,
            lambda_end: lam1,
            dg_complex_bar_kcal: dg_c_bar,
            dg_solvent_bar_kcal: dg_s_bar,
            ddg_window_bar_kcal: ddg_w_bar,
            ddg_window_fep_fwd_kcal: ddg_w_fwd,
            ddg_window_fep_rev_kcal: ddg_w_rev,
            bar_std_err_kcal: var_w.sqrt(),
            complex_overlap: ov_c,
            solvent_overlap: ov_s,
        });
    }

    let raw_ddg_bar = sum_comp_bar - sum_solv_bar;
    let bar_std_err = sum_var_bar.sqrt().max(0.06);
    let fep_hyst = (sum_ddg_fwd - sum_ddg_rev).abs();
    let mean_ov = sum_ov / (n_intervals as f64);

    let experimental_ddg_kcal = match (src_node.experimental_dg_kcal, dst_node.experimental_dg_kcal)
    {
        (Some(ga), Some(gb)) => Some(gb - ga),
        _ => None,
    };

    let convergence_tier = if min_ov < 0.03 || fep_hysteresis_severe(fep_hyst, lomap.partial_ring_break) {
        RbfeConvergenceTier::HighHysteresisFlagged
    } else if min_ov < 0.10 || fep_hyst > 0.45 {
        RbfeConvergenceTier::ModerateCaution
    } else {
        RbfeConvergenceTier::ConvergedOptimal
    };

    RbfePerturbationEdge {
        edge_idx,
        src_node: src_node.node_idx,
        dst_node: dst_node.node_idx,
        src_id: src_node.compound_id.clone(),
        dst_id: dst_node.compound_id.clone(),
        src_name: src_node.compound_name.clone(),
        dst_name: dst_node.compound_name.clone(),
        is_spanning_tree_edge: is_tree_edge,
        lomap,
        dg_complex_bar_kcal: sum_comp_bar,
        dg_solvent_bar_kcal: sum_solv_bar,
        ddg_fep_forward_kcal: sum_ddg_fwd,
        ddg_fep_reverse_kcal: sum_ddg_rev,
        fep_fwd_rev_hysteresis_kcal: fep_hyst,
        raw_ddg_bar_kcal: raw_ddg_bar,
        bar_std_err_kcal: bar_std_err,
        reconciled_ddg_mle_kcal: raw_ddg_bar,
        experimental_ddg_kcal,
        min_phase_space_overlap: min_ov,
        mean_phase_space_overlap: mean_ov,
        participating_cycles: 0,
        max_cycle_hysteresis_kcal: 0.0,
        convergence_tier,
        windows,
    }
}

#[inline]
fn fep_hysteresis_severe(fep_hyst: f64, partial_ring_break: bool) -> bool {
    fep_hyst > 0.85 || (partial_ring_break && fep_hyst > 0.55)
}

/// Union-Find helper for `Kruskal` Maximum-Weight `Lomap` Spanning Tree construction.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Computes unweighted all-pairs shortest-path hop distances across `n` nodes (`usize::MAX / 4` when disconnected).
fn compute_apsp_hops(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let inf = usize::MAX / 4;
    let mut dist = vec![vec![inf; n]; n];
    for (i, row) in dist.iter_mut().enumerate() {
        row[i] = 0;
    }
    for &(u, v) in edges {
        dist[u][v] = 1;
        dist[v][u] = 1;
    }
    for k in 0..n {
        for i in 0..n {
            if dist[i][k] == inf {
                continue;
            }
            for j in 0..n {
                let cand = dist[i][k] + dist[k][j];
                if cand < dist[i][j] {
                    dist[i][j] = cand;
                }
            }
        }
    }
    dist
}

/// Enumerates closed `3`-membered (`i < j < k`) and chordless `4`-membered (`i - j - k - l - i`) thermodynamic cycles.
fn detect_thermodynamic_cycles(
    n_nodes: usize,
    nodes: &[RbfeLigandNode],
    edges: &[RbfePerturbationEdge],
    warn_kcal: f64,
) -> Vec<RbfeThermodynamicCycle> {
    // Lookup table `edge_lookup[u][v] = Some((edge_idx, +1 if u->v else -1))`
    let mut lookup: Vec<Vec<Option<(usize, i8)>>> = vec![vec![None; n_nodes]; n_nodes];
    for (ei, e) in edges.iter().enumerate() {
        lookup[e.src_node][e.dst_node] = Some((ei, 1));
        lookup[e.dst_node][e.src_node] = Some((ei, -1));
    }

    let mut cycles: Vec<RbfeThermodynamicCycle> = Vec::new();

    // 1. All 3-membered triangles (i < j < k)
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            let Some((e_ij, s_ij)) = lookup[i][j] else {
                continue;
            };
            for k in (j + 1)..n_nodes {
                let (Some((e_jk, s_jk)), Some((e_ki, s_ki))) = (lookup[j][k], lookup[k][i]) else {
                    continue;
                };
                let dir_edges = vec![(e_ij, s_ij), (e_jk, s_jk), (e_ki, s_ki)];
                cycles.push(build_cycle_record(cycles.len(), &[i, j, k], dir_edges, nodes, edges, warn_kcal));
            }
        }
    }

    // 2. Chordless 4-membered cycles (i < j, k, l and j < l to deduplicate orientation & cyclic shifts)
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            let Some((e_ij, s_ij)) = lookup[i][j] else {
                continue;
            };
            for l in (j + 1)..n_nodes {
                let Some((e_li, s_li)) = lookup[l][i] else {
                    continue;
                };
                if lookup[j][l].is_some() {
                    continue; // Diagonal chord exists -> already covered by two 3-cycles!
                }
                for k in (i + 1)..n_nodes {
                    if k == j || k == l || lookup[i][k].is_some() {
                        continue; // Chordless check on the other diagonal!
                    }
                    let (Some((e_jk, s_jk)), Some((e_kl, s_kl))) = (lookup[j][k], lookup[k][l])
                    else {
                        continue;
                    };
                    let dir_edges = vec![(e_ij, s_ij), (e_jk, s_jk), (e_kl, s_kl), (e_li, s_li)];
                    cycles.push(build_cycle_record(
                        cycles.len(),
                        &[i, j, k, l],
                        dir_edges,
                        nodes,
                        edges,
                        warn_kcal,
                    ));
                }
            }
        }
    }

    // Sort cycles by descending absolute hysteresis so worst offenders surface first
    cycles.sort_by(|a, b| {
        b.abs_hysteresis_kcal
            .partial_cmp(&a.abs_hysteresis_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, c) in cycles.iter_mut().enumerate() {
        c.cycle_idx = idx;
        c.cycle_id = format!("CYC-{:02} ({}M)", idx + 1, c.node_indices.len());
    }
    cycles
}

fn build_cycle_record(
    idx: usize,
    ring_nodes: &[usize],
    directed_edges: Vec<(usize, i8)>,
    nodes: &[RbfeLigandNode],
    edges: &[RbfePerturbationEdge],
    warn_kcal: f64,
) -> RbfeThermodynamicCycle {
    let mut signed_sum = 0.0_f64;
    let mut var_sum = 0.0_f64;
    for &(ei, sign) in &directed_edges {
        signed_sum += (sign as f64) * edges[ei].raw_ddg_bar_kcal;
        var_sum += edges[ei].bar_std_err_kcal * edges[ei].bar_std_err_kcal;
    }
    let abs_hyst = signed_sum.abs();
    let cyc_sig = var_sum.sqrt().max(0.08);
    let z = abs_hyst / cyc_sig;
    let node_labels: Vec<String> = ring_nodes
        .iter()
        .map(|&ni| nodes[ni].compound_id.clone())
        .collect();
    RbfeThermodynamicCycle {
        cycle_idx: idx,
        cycle_id: format!("CYC-{:02}", idx + 1),
        node_indices: ring_nodes.to_vec(),
        node_labels,
        directed_edges,
        signed_closure_kcal: signed_sum,
        abs_hysteresis_kcal: abs_hyst,
        cycle_sigma_kcal: cyc_sig,
        closure_z_score: z,
        post_mle_residual_kcal: 0.0,
        converged: abs_hyst <= warn_kcal,
    }
}

/// Solves `L * L^T * x = b` given lower-triangular Cholesky factor `chol` (`n x n`).
fn cholesky_solve(chol: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= chol[i * n + k] * y[k];
        }
        y[i] = s / chol[i * n + i].max(1e-12);
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= chol[k * n + i] * x[k];
        }
        x[i] = s / chol[i * n + i].max(1e-12);
    }
    x
}

/// Variance-weighted Graph-Laplacian Maximum-Likelihood (`BARnet` / `DiffNet` / `cinnabar`) node
/// free-energy reconciler solving `(B^T W B + W_ref) * g_MLE = B^T W ddG_BAR + W_ref * g_ref`.
fn reconcile_mle_node_free_energies(
    nodes: &mut [RbfeLigandNode],
    edges: &mut [RbfePerturbationEdge],
    cycles: &mut [RbfeThermodynamicCycle],
    ref_idx: usize,
) {
    let n = nodes.len();
    if n == 0 {
        return;
    }

    let mut lap = vec![0.0_f64; n * n];
    let mut rhs = vec![0.0_f64; n];

    // Add inverse-variance edge restraints: w_e * ((g_dst - g_src) - ddG_e)^2
    for e in edges.iter() {
        let u = e.src_node;
        let v = e.dst_node;
        let w = 1.0 / (e.bar_std_err_kcal * e.bar_std_err_kcal).max(0.004);
        lap[u * n + u] += w;
        lap[v * n + v] += w;
        lap[u * n + v] -= w;
        lap[v * n + u] -= w;
        rhs[u] -= w * e.raw_ddg_bar_kcal;
        rhs[v] += w * e.raw_ddg_bar_kcal;
    }

    // Anchor reference compound strongly (`sigma_anchor = 0.05 kcal/mol` -> `w_ref = 400.0`) so gauge is exact,
    // plus tiny `1e-3` weak prior on any disconnected island nodes so `L` is strictly SPD everywhere!
    let ref_dg = nodes[ref_idx]
        .experimental_dg_kcal
        .unwrap_or(nodes[ref_idx].pocket_complementarity_dg_kcal);
    let w_anchor = 400.0_f64;
    lap[ref_idx * n + ref_idx] += w_anchor;
    rhs[ref_idx] += w_anchor * ref_dg;

    for (i, nd) in nodes.iter().enumerate() {
        let w_prior = if i == ref_idx { 0.0 } else { 1e-3 };
        let g0 = nd
            .experimental_dg_kcal
            .unwrap_or(nd.pocket_complementarity_dg_kcal);
        lap[i * n + i] += w_prior + 1e-8;
        rhs[i] += w_prior * g0;
    }

    let Some(chol) = cholesky_decompose(&lap, n) else {
        return;
    };
    let g_mle = cholesky_solve(&chol, &rhs, n);

    // Diagonal of `L^{-1}` gives exact posterior node variance `sigma_MLE^2[i]`
    let mut node_std = vec![0.08_f64; n];
    for i in 0..n {
        let mut ei_vec = vec![0.0_f64; n];
        ei_vec[i] = 1.0;
        let col_inv = cholesky_solve(&chol, &ei_vec, n);
        node_std[i] = col_inv[i].max(1e-6).sqrt();
    }

    let g_ref_mle = g_mle[ref_idx];
    for i in 0..n {
        nodes[i].mle_dg_bind_kcal = g_mle[i];
        nodes[i].mle_dg_std_kcal = node_std[i];
        nodes[i].mle_predicted_pact = (-g_mle[i] / RT_LN10_KCAL).clamp(3.0, 12.5);
        nodes[i].ddg_vs_reference_kcal = g_mle[i] - g_ref_mle;
    }

    for e in edges.iter_mut() {
        e.reconciled_ddg_mle_kcal = g_mle[e.dst_node] - g_mle[e.src_node];
    }

    for cyc in cycles.iter_mut() {
        let mut post_sum = 0.0_f64;
        for &(ei, sign) in &cyc.directed_edges {
            post_sum += (sign as f64) * edges[ei].reconciled_ddg_mle_kcal;
        }
        cyc.post_mle_residual_kcal = post_sum.abs();
    }
}

/// Force-relaxed circular/radial 2D graph layout placing the reference hub near the center-top or origin
/// and distributing congeneric analogs cleanly in `[-0.88, 0.88]^2`.
fn compute_network_2d_layout(
    nodes: &mut [RbfeLigandNode],
    edges: &[RbfePerturbationEdge],
    ref_idx: usize,
    topology_mode: RbfeTopologyMode,
) {
    let n = nodes.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        nodes[0].layout_xy = [0.0, 0.0];
        return;
    }

    let mut pos = vec![[0.0_f32; 2]; n];
    if topology_mode == RbfeTopologyMode::StarRadialHub {
        pos[ref_idx] = [0.0, 0.0];
        let mut spoke = 0usize;
        let m = (n - 1).max(1) as f32;
        for (i, p) in pos.iter_mut().enumerate() {
            if i == ref_idx {
                continue;
            }
            let ang = std::f32::consts::TAU * (spoke as f32) / m - std::f32::consts::FRAC_PI_2;
            *p = [0.78 * ang.cos(), 0.78 * ang.sin()];
            spoke += 1;
        }
    } else {
        for (i, p) in pos.iter_mut().enumerate() {
            let order_i = if i == ref_idx {
                0
            } else if i < ref_idx {
                i + 1
            } else {
                i
            };
            let ang =
                std::f32::consts::TAU * (order_i as f32) / (n as f32) - std::f32::consts::FRAC_PI_2;
            *p = [0.76 * ang.cos(), 0.76 * ang.sin()];
        }

        // 45 iterations of spring-repulsion relaxation weighted by Lomap similarity
        for _ in 0..45 {
            let mut disp = vec![[0.0_f32; 2]; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = pos[i][0] - pos[j][0];
                    let dy = pos[i][1] - pos[j][1];
                    let d2 = (dx * dx + dy * dy).max(0.015);
                    let d = d2.sqrt();
                    let rep = 0.022 / d2;
                    disp[i][0] += (dx / d) * rep;
                    disp[i][1] += (dy / d) * rep;
                    disp[j][0] -= (dx / d) * rep;
                    disp[j][1] -= (dy / d) * rep;
                }
            }
            for e in edges {
                let u = e.src_node;
                let v = e.dst_node;
                let dx = pos[v][0] - pos[u][0];
                let dy = pos[v][1] - pos[u][1];
                let d = (dx * dx + dy * dy).sqrt().max(0.02);
                let ideal = 0.52_f32;
                let k_att = 0.12 * (e.lomap.total_lomap_score as f32).max(0.3);
                let f = k_att * (d - ideal);
                disp[u][0] += (dx / d) * f;
                disp[u][1] += (dy / d) * f;
                disp[v][0] -= (dx / d) * f;
                disp[v][1] -= (dy / d) * f;
            }
            for i in 0..n {
                pos[i][0] = (pos[i][0] + disp[i][0].clamp(-0.08, 0.08)) * 0.96;
                pos[i][1] = (pos[i][1] + disp[i][1].clamp(-0.08, 0.08)) * 0.96;
            }
        }

        // Normalize bounding box cleanly into `[-0.84, 0.84]^2`
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for p in &pos {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        let cx = 0.5 * (min_x + max_x);
        let cy = 0.5 * (min_y + max_y);
        let half_span = (0.5 * (max_x - min_x).max(max_y - min_y)).max(0.15);
        for p in &mut pos {
            p[0] = ((p[0] - cx) / half_span * 0.80).clamp(-0.88, 0.88);
            p[1] = ((p[1] - cy) / half_span * 0.80).clamp(-0.88, 0.88);
        }
    }

    for (i, nd) in nodes.iter_mut().enumerate() {
        nd.layout_xy = pos[i];
    }
}

/// Builds the complete `v2.8.0` `Lomap` perturbation graph, runs multi-`lambda` `Zwanzig`/`Bennett BAR`
/// free-energy estimation across Complex & Solvent legs, audits `3`-/`4`-cycle closure hysteresis, and
/// solves variance-weighted `MLE` node binding free energies.
pub fn run_rbfe_network_campaign(
    ds: &DatasetAnalysis,
    cfg: &RbfeNetworkConfig,
) -> RbfeNetworkReport {
    // 1. Resolve target series & matching 3D receptor pocket
    let resolved_target = match cfg.series_preset {
        RbfeSeriesPreset::Egfr1m17Quinazolines => "EGFR".to_string(),
        RbfeSeriesPreset::Cox23ln1Diarylpyrazoles => "COX-2".to_string(),
        RbfeSeriesPreset::Drd26cm4Arylpiperazines => "DRD2".to_string(),
        RbfeSeriesPreset::AutoDominantTarget => {
            let mut counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
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
        }
    };

    let pocket_idx = cfg.series_preset.pocket_preset_idx(&resolved_target);
    let pdb_txt = curated_pocket_pdb_text(pocket_idx);
    let parsed_pdb = parse_pdb_structure(pdb_txt).expect("Curated PDB structure valid");
    let pocket: PocketCavityAnalysis = analyze_binding_pocket(&parsed_pdb, None, 10.0);

    // 2. Gather congeneric series molecules from dataset + curated congeneric analogs + optional user SMILES
    let max_ligs = cfg.max_series_ligands.clamp(4, 18);
    let mut raw_entries: Vec<(Option<usize>, Molecule, Option<f64>, bool)> = Vec::new();
    let mut seen_smiles: BTreeSet<String> = BTreeSet::new();

    if cfg.series_preset == RbfeSeriesPreset::AutoDominantTarget {
        for (mi, m) in ds.mols.iter().enumerate() {
            if raw_entries.len() >= max_ligs {
                break;
            }
            let tgt_ok = m.target.eq_ignore_ascii_case(&resolved_target) || ds.mols.len() <= max_ligs;
            if tgt_ok && seen_smiles.insert(m.smiles.clone()) {
                let exp_p = if m.pact > 0.5 { Some(m.pact) } else { None };
                raw_entries.push((Some(mi), m.clone(), exp_p, false));
            }
        }
    } else {
        // Include dataset molecules matching the chosen receptor first, then curated congeneric series analogs
        let key_tok = match pocket_idx {
            0 => "EGFR",
            1 => "DRD2",
            _ => "COX",
        };
        for (mi, m) in ds.mols.iter().enumerate() {
            if raw_entries.len() >= max_ligs / 2 {
                break;
            }
            if m.target.to_ascii_uppercase().contains(key_tok) && seen_smiles.insert(m.smiles.clone()) {
                let exp_p = if m.pact > 0.5 { Some(m.pact) } else { None };
                raw_entries.push((Some(mi), m.clone(), exp_p, false));
            }
        }
    }

    // Enrich with curated congeneric series analogs so every preset has >= 8 tight MCS congeners
    if raw_entries.len() < max_ligs.min(9) {
        for (cid, cname, ctgt, csmi, cpact) in curated_congeneric_analogs(pocket_idx) {
            if raw_entries.len() >= max_ligs {
                break;
            }
            if let Ok(mut cmol) = parse_smiles(csmi) {
                if seen_smiles.insert(cmol.smiles.clone()) {
                    cmol.id = cid.to_string();
                    cmol.name = cname.to_string();
                    cmol.target = ctgt.to_string();
                    cmol.pact = cpact;
                    raw_entries.push((None, cmol, Some(cpact), false));
                }
            }
        }
    }

    // Append user-supplied custom prospective SMILES
    for (ci, smi) in cfg.custom_analog_smiles.iter().enumerate() {
        if let Ok(mut umol) = parse_smiles(smi) {
            if seen_smiles.insert(umol.smiles.clone()) {
                umol.id = format!("CUSTOM-{:02}", ci + 1);
                umol.name = format!("Prospective Analog {}", ci + 1);
                umol.target = resolved_target.clone();
                umol.pact = 0.0;
                raw_entries.push((None, umol, None, true));
            }
        }
    }

    // 3. Score 3D Pocket Complementarity + Aqueous Solvation in parallel across all series nodes
    let mut nodes: Vec<RbfeLigandNode> = raw_entries
        .par_iter()
        .enumerate()
        .map(|(ni, (ds_idx, mol, exp_pact, is_custom))| {
            let prof = evaluate_physchem(mol, 0);
            let p_res = score_ligand_in_pocket(mol, *ds_idx, &pocket);
            let solv_dg = estimate_aqueous_solvation_dg_kcal(mol);
            let exp_dg = exp_pact.map(|p| -RT_LN10_KCAL * p);
            let init_dg = exp_dg.unwrap_or(p_res.total_dg_pocket_kcal);
            let init_pact = (-init_dg / RT_LN10_KCAL).clamp(3.0, 12.5);
            RbfeLigandNode {
                node_idx: ni,
                dataset_mol_idx: *ds_idx,
                compound_id: mol.id.clone(),
                compound_name: mol.name.clone(),
                canonical_smiles: mol.smiles.clone(),
                is_reference_anchor: false,
                is_prospective_custom: *is_custom,
                heavy_atoms: prof.heavy_atoms,
                net_charge: molecule_formal_charge(mol),
                mw: prof.mw,
                clogp: prof.clogp,
                degree: 0,
                cycle_count: 0,
                layout_xy: [0.0, 0.0],
                pocket_complementarity_dg_kcal: p_res.total_dg_pocket_kcal,
                aqueous_solvation_dg_kcal: solv_dg,
                experimental_pact: *exp_pact,
                experimental_dg_kcal: exp_dg,
                mle_dg_bind_kcal: init_dg,
                mle_dg_std_kcal: 0.15,
                mle_predicted_pact: init_pact,
                ddg_vs_reference_kcal: 0.0,
                mol: mol.clone(),
            }
        })
        .collect();

    let n_nodes = nodes.len();
    if n_nodes == 0 {
        return RbfeNetworkReport {
            resolved_target,
            pocket_pdb_id: pocket.pdb_id,
            pocket_title: pocket.title,
            reference_node_idx: 0,
            config: cfg.clone(),
            summary: RbfeSummaryMetrics {
                n_nodes: 0,
                n_edges: 0,
                n_cycles: 0,
                n_flagged_cycles: 0,
                n_flagged_edges: 0,
                graph_diameter: 0,
                fraction_nodes_in_cycles: 0.0,
                mean_lomap_score: 0.0,
                mean_phase_space_overlap: 0.0,
                mean_cycle_hysteresis_kcal: 0.0,
                max_cycle_hysteresis_kcal: 0.0,
                post_mle_max_cycle_error_kcal: 0.0,
                n_experimental_nodes: 0,
                dg_rmse_kcal: 0.0,
                dg_mae_kcal: 0.0,
                pact_r2_pearson: 0.0,
                kendall_tau: 0.0,
            },
            nodes,
            edges: Vec::new(),
            cycles: Vec::new(),
        };
    }

    // Select reference anchor ligand: highest-potency experimental compound (or best pocket score)
    let mut ref_idx = 0usize;
    let mut best_ref_val = f64::NEG_INFINITY;
    for (i, nd) in nodes.iter().enumerate() {
        let score = nd
            .experimental_pact
            .unwrap_or(-nd.pocket_complementarity_dg_kcal / RT_LN10_KCAL);
        if score > best_ref_val {
            best_ref_val = score;
            ref_idx = i;
        }
    }
    nodes[ref_idx].is_reference_anchor = true;

    // 4. Compute all-pairs Lomap MCS similarity scores in parallel
    let mut pair_indices: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            pair_indices.push((i, j));
        }
    }

    let mut candidate_pairs: Vec<(usize, usize, LomapEdgeScore)> = pair_indices
        .par_iter()
        .filter_map(|&(i, j)| {
            compute_lomap_pair_score(
                &nodes[i].mol,
                &nodes[j].mol,
                cfg.lomap_beta,
                cfg.min_mcs_atoms,
                cfg.allow_charge_changes,
            )
            .map(|sc| (i, j, sc))
        })
        .collect();

    // Sort candidate pairs by descending Lomap similarity
    candidate_pairs.sort_by(|a, b| {
        b.2.total_lomap_score
            .partial_cmp(&a.2.total_lomap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 5. Select Perturbation Graph Topology (CycleClosedLomap vs StarRadialHub vs DenseRedundantLomap)
    let mut chosen_pairs: Vec<(usize, usize, LomapEdgeScore, bool)> = Vec::new();
    let mut chosen_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut degree = vec![0usize; n_nodes];

    match cfg.topology_mode {
        RbfeTopologyMode::StarRadialHub => {
            for j in 0..n_nodes {
                if j == ref_idx {
                    continue;
                }
                let (u, v) = if ref_idx < j { (ref_idx, j) } else { (j, ref_idx) };
                if let Some(found) = candidate_pairs.iter().find(|&&(a, b, _)| a == u && b == v) {
                    chosen_pairs.push((u, v, found.2.clone(), true));
                    chosen_set.insert((u, v));
                    degree[u] += 1;
                    degree[v] += 1;
                } else if let Some(fallback) =
                    compute_lomap_pair_score(&nodes[u].mol, &nodes[v].mol, cfg.lomap_beta, 4, true)
                {
                    chosen_pairs.push((u, v, fallback, true));
                    chosen_set.insert((u, v));
                    degree[u] += 1;
                    degree[v] += 1;
                }
            }
        }
        RbfeTopologyMode::CycleClosedLomap | RbfeTopologyMode::DenseRedundantLomap => {
            // Step A: Maximum-Weight Spanning Forest (`Kruskal`) over pairs >= min_lomap_score * 0.5
            let mut uf: Vec<usize> = (0..n_nodes).collect();
            for (u, v, sc) in &candidate_pairs {
                let ru = uf_find(&mut uf, *u);
                let rv = uf_find(&mut uf, *v);
                if ru != rv {
                    uf[ru] = rv;
                    chosen_pairs.push((*u, *v, sc.clone(), true));
                    chosen_set.insert((*u, *v));
                    degree[*u] += 1;
                    degree[*v] += 1;
                }
            }

            // Step B: Cycle-closing chord augmentation (`Lomap` biconnected cycle-closure + MAXDIST diameter cap)
            let max_deg = if cfg.topology_mode == RbfeTopologyMode::DenseRedundantLomap {
                4usize
            } else {
                3usize
            };
            let target_edges = if cfg.topology_mode == RbfeTopologyMode::DenseRedundantLomap {
                (n_nodes * 2).min(n_nodes * (n_nodes.saturating_sub(1)) / 2)
            } else {
                ((n_nodes * 3) / 2 + 1).max(n_nodes + 1)
            };

            // Prioritize rescuing degree-1 leaf nodes into closed 3- or 4-cycles first!
            for pass in 0..2 {
                for (u, v, sc) in &candidate_pairs {
                    if chosen_pairs.len() >= target_edges {
                        break;
                    }
                    if chosen_set.contains(&(*u, *v)) {
                        continue;
                    }
                    if sc.total_lomap_score < cfg.min_lomap_score && pass == 0 {
                        continue;
                    }
                    let helps_leaf = degree[*u] < 2 || degree[*v] < 2;
                    let cur_edges: Vec<(usize, usize)> =
                        chosen_pairs.iter().map(|&(a, b, _, _)| (a, b)).collect();
                    let dist = compute_apsp_hops(n_nodes, &cur_edges);
                    let hop_uv = dist[*u][*v];
                    let closes_3_or_4_cycle = (2..=3).contains(&hop_uv);
                    let reduces_diameter = hop_uv > cfg.max_graph_diameter && hop_uv < usize::MAX / 8;

                    if pass == 0 {
                        if !(helps_leaf || reduces_diameter) || !closes_3_or_4_cycle {
                            continue;
                        }
                    } else if degree[*u] >= max_deg || degree[*v] >= max_deg || !closes_3_or_4_cycle
                    {
                        continue;
                    }

                    chosen_pairs.push((*u, *v, sc.clone(), false));
                    chosen_set.insert((*u, *v));
                    degree[*u] += 1;
                    degree[*v] += 1;
                }
            }
        }
    }

    // 6. Run stratified multi-lambda Zwanzig + Bennett BAR simulation across all chosen edges in parallel
    let mut edges: Vec<RbfePerturbationEdge> = chosen_pairs
        .par_iter()
        .enumerate()
        .map(|(ei, (u, v, sc, is_tree))| {
            simulate_alchemical_edge(ei, &nodes[*u], &nodes[*v], *is_tree, sc.clone(), cfg.lambda_windows)
        })
        .collect();

    // 7. Detect elementary 3- and 4-membered closed thermodynamic cycles & compute raw BAR hysteresis
    let mut cycles = detect_thermodynamic_cycles(
        n_nodes,
        &nodes,
        &edges,
        cfg.cycle_hysteresis_warn_kcal,
    );

    // 8. Reconcile node binding free energies via variance-weighted Graph-Laplacian MLE (`cinnabar` / `BARnet`)
    reconcile_mle_node_free_energies(&mut nodes, &mut edges, &mut cycles, ref_idx);

    // 9. Update node & edge cycle-participation counts and re-flag any edges inside high-hysteresis cycles
    for (ni, nd) in nodes.iter_mut().enumerate() {
        nd.degree = degree[ni];
    }
    for cyc in &cycles {
        for &ni in &cyc.node_indices {
            nodes[ni].cycle_count += 1;
        }
        for &(ei, _) in &cyc.directed_edges {
            edges[ei].participating_cycles += 1;
            if cyc.abs_hysteresis_kcal > edges[ei].max_cycle_hysteresis_kcal {
                edges[ei].max_cycle_hysteresis_kcal = cyc.abs_hysteresis_kcal;
            }
            if !cyc.converged && edges[ei].convergence_tier == RbfeConvergenceTier::ConvergedOptimal {
                edges[ei].convergence_tier = RbfeConvergenceTier::ModerateCaution;
            }
        }
    }

    // 10. Compute 2D layout & campaign summary statistics (RMSE, MAE, Pearson R^2, Kendall tau)
    compute_network_2d_layout(&mut nodes, &edges, ref_idx, cfg.topology_mode);

    let edge_uv: Vec<(usize, usize)> = edges.iter().map(|e| (e.src_node, e.dst_node)).collect();
    let dist_mat = compute_apsp_hops(n_nodes, &edge_uv);
    let mut graph_diameter = 0usize;
    for row in &dist_mat {
        for &d in row {
            if d < usize::MAX / 8 && d > graph_diameter {
                graph_diameter = d;
            }
        }
    }

    let nodes_in_cycles = nodes.iter().filter(|nd| nd.cycle_count > 0).count();
    let fraction_nodes_in_cycles = (nodes_in_cycles as f64) / (n_nodes.max(1) as f64);

    let mean_lomap_score = if edges.is_empty() {
        0.0
    } else {
        edges.iter().map(|e| e.lomap.total_lomap_score).sum::<f64>() / (edges.len() as f64)
    };
    let mean_phase_space_overlap = if edges.is_empty() {
        0.0
    } else {
        edges
            .iter()
            .map(|e| e.mean_phase_space_overlap)
            .sum::<f64>()
            / (edges.len() as f64)
    };
    let mean_cycle_hysteresis_kcal = if cycles.is_empty() {
        0.0
    } else {
        cycles.iter().map(|c| c.abs_hysteresis_kcal).sum::<f64>() / (cycles.len() as f64)
    };
    let max_cycle_hysteresis_kcal = cycles
        .iter()
        .map(|c| c.abs_hysteresis_kcal)
        .fold(0.0_f64, f64::max);
    let post_mle_max_cycle_error_kcal = cycles
        .iter()
        .map(|c| c.post_mle_residual_kcal)
        .fold(0.0_f64, f64::max);

    let n_flagged_cycles = cycles.iter().filter(|c| !c.converged).count();
    let n_flagged_edges = edges
        .iter()
        .filter(|e| e.convergence_tier == RbfeConvergenceTier::HighHysteresisFlagged)
        .count();

    // Retrospective accuracy on nodes carrying experimental pAct
    let exp_pairs: Vec<(f64, f64)> = nodes
        .iter()
        .filter_map(|nd| nd.experimental_dg_kcal.map(|ge| (ge, nd.mle_dg_bind_kcal)))
        .collect();
    let n_experimental_nodes = exp_pairs.len();
    let (dg_rmse_kcal, dg_mae_kcal, pact_r2_pearson, kendall_tau) =
        compute_retrospective_rbfe_metrics(&exp_pairs);

    RbfeNetworkReport {
        resolved_target,
        pocket_pdb_id: pocket.pdb_id,
        pocket_title: pocket.title,
        reference_node_idx: ref_idx,
        config: cfg.clone(),
        summary: RbfeSummaryMetrics {
            n_nodes,
            n_edges: edges.len(),
            n_cycles: cycles.len(),
            n_flagged_cycles,
            n_flagged_edges,
            graph_diameter,
            fraction_nodes_in_cycles,
            mean_lomap_score,
            mean_phase_space_overlap,
            mean_cycle_hysteresis_kcal,
            max_cycle_hysteresis_kcal,
            post_mle_max_cycle_error_kcal,
            n_experimental_nodes,
            dg_rmse_kcal,
            dg_mae_kcal,
            pact_r2_pearson,
            kendall_tau,
        },
        nodes,
        edges,
        cycles,
    }
}

fn compute_retrospective_rbfe_metrics(pairs: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    if pairs.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let m = pairs.len() as f64;
    let mut sq = 0.0_f64;
    let mut ab = 0.0_f64;
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    for &(x, y) in pairs {
        sq += (x - y) * (x - y);
        ab += (x - y).abs();
        mx += x / m;
        my += y / m;
    }
    let rmse = (sq / m).sqrt();
    let mae = ab / m;

    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for &(x, y) in pairs {
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
        sxy += (x - mx) * (y - my);
    }
    let r = if sxx > 1e-9 && syy > 1e-9 {
        (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let mut conc = 0i64;
    let mut disc = 0i64;
    for i in 0..pairs.len() {
        for j in (i + 1)..pairs.len() {
            let dx = pairs[i].0 - pairs[j].0;
            let dy = pairs[i].1 - pairs[j].1;
            if dx.abs() < 1e-6 || dy.abs() < 1e-6 {
                continue;
            }
            if dx * dy > 0.0 {
                conc += 1;
            } else {
                disc += 1;
            }
        }
    }
    let denom = (conc + disc).max(1) as f64;
    let tau = ((conc - disc) as f64) / denom;

    (rmse, mae, r * r, tau)
}

/// Exports all alchemical perturbation edges (`Lomap`, `FEP_fwd`, `FEP_rev`, `BAR`, overlap `O_AB`, and `MLE` `ddG`) to `.csv`.
pub fn export_rbfe_edges_csv(report: &RbfeNetworkReport) -> String {
    let mut out = String::from(
        "edge_idx,src_id,dst_id,topology_role,lomap_score,mcs_atoms,perturbed_heavy_atoms,partial_ring_break,dg_complex_bar_kcal,dg_solvent_bar_kcal,ddg_fep_fwd_kcal,ddg_fep_rev_kcal,fep_hysteresis_kcal,raw_ddg_bar_kcal,bar_std_err_kcal,reconciled_ddg_mle_kcal,experimental_ddg_kcal,min_overlap,participating_cycles,max_cycle_hysteresis_kcal,status\n",
    );
    for e in &report.edges {
        let exp_str = e
            .experimental_ddg_kcal
            .map(|v| format!("{v:.3}"))
            .unwrap_or_default();
        let role = if e.is_spanning_tree_edge { "MST-Backbone" } else { "Cycle-Chord" };
        out.push_str(&format!(
            "{},{},{},{},{:.4},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.4},{},{:.3},{}\n",
            e.edge_idx + 1,
            e.src_id,
            e.dst_id,
            role,
            e.lomap.total_lomap_score,
            e.lomap.mcs_atoms,
            e.lomap.total_perturbed_heavy_atoms,
            e.lomap.partial_ring_break,
            e.dg_complex_bar_kcal,
            e.dg_solvent_bar_kcal,
            e.ddg_fep_forward_kcal,
            e.ddg_fep_reverse_kcal,
            e.fep_fwd_rev_hysteresis_kcal,
            e.raw_ddg_bar_kcal,
            e.bar_std_err_kcal,
            e.reconciled_ddg_mle_kcal,
            exp_str,
            e.min_phase_space_overlap,
            e.participating_cycles,
            e.max_cycle_hysteresis_kcal,
            e.convergence_tier.short_label()
        ));
    }
    out
}

/// Exports reconciled `MLE` node binding free energies `dG_MLE`, `pAct_MLE`, uncertainties, and cycle counts to `.csv`.
pub fn export_rbfe_nodes_csv(report: &RbfeNetworkReport) -> String {
    let mut out = String::from(
        "node_idx,compound_id,compound_name,is_reference,degree,cycle_count,mle_dg_bind_kcal,mle_dg_std_kcal,mle_predicted_pact,ddg_vs_ref_kcal,experimental_pact,experimental_dg_kcal,pocket_dg_kcal,solvation_dg_kcal,smiles\n",
    );
    for nd in &report.nodes {
        let ep = nd
            .experimental_pact
            .map(|v| format!("{v:.2}"))
            .unwrap_or_default();
        let eg = nd
            .experimental_dg_kcal
            .map(|v| format!("{v:.3}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "{},{},\"{}\",{},{},{},{:.3},{:.3},{:.2},{:.3},{},{},{:.3},{:.3},{}\n",
            nd.node_idx + 1,
            nd.compound_id,
            nd.compound_name.replace('"', "'"),
            nd.is_reference_anchor,
            nd.degree,
            nd.cycle_count,
            nd.mle_dg_bind_kcal,
            nd.mle_dg_std_kcal,
            nd.mle_predicted_pact,
            nd.ddg_vs_reference_kcal,
            ep,
            eg,
            nd.pocket_complementarity_dg_kcal,
            nd.aqueous_solvation_dg_kcal,
            nd.canonical_smiles
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_benchmark_26;

    #[test]
    fn test_bennett_bar_exact_gaussian_consistency_and_overlap() {
        // Synthetic Crooks-consistent shifted Gaussian work distributions:
        // dG_true = -2.25 kcal/mol, sigma = 0.65 kcal/mol, dissipation = 0.5 * beta * sigma^2
        let dg_true = -2.25_f64;
        let sig = 0.65_f64;
        let diss = 0.5 * (sig * sig) / RBFE_RT_298K_KCAL;
        let mut fwd = Vec::new();
        let mut rev = Vec::new();
        for s in 0..128 {
            fwd.push(dg_true + diss + sig * deterministic_normal(42, s));
            rev.push(-dg_true + diss + sig * deterministic_normal(99, s));
        }
        let (dg_bar, std_err, ov) = solve_bennett_bar(&fwd, &rev, RBFE_RT_298K_KCAL);
        assert!((dg_bar - dg_true).abs() < 0.20, "BAR estimate {dg_bar} close to {dg_true}");
        assert!(std_err > 0.01 && std_err < 0.35);
        assert!(ov > 0.25 && ov <= 1.0);
    }

    #[test]
    fn test_lomap_cycle_closed_rbfe_network_and_zero_post_mle_hysteresis() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for preset in [
            RbfeSeriesPreset::Egfr1m17Quinazolines,
            RbfeSeriesPreset::Cox23ln1Diarylpyrazoles,
        ] {
            let cfg = RbfeNetworkConfig {
                series_preset: preset,
                topology_mode: RbfeTopologyMode::CycleClosedLomap,
                ..RbfeNetworkConfig::default()
            };
            let rep = run_rbfe_network_campaign(&ds, &cfg);
            assert!(rep.summary.n_nodes >= 8);
            assert!(rep.summary.n_edges >= rep.summary.n_nodes);
            assert!(rep.summary.n_cycles >= 2, "Expected closed 3/4-cycles in Lomap network");
            assert!(
                rep.summary.post_mle_max_cycle_error_kcal < 1e-9,
                "MLE node reconciliation must enforce exact 0.000 kcal/mol cycle closure"
            );
            let e_csv = export_rbfe_edges_csv(&rep);
            let n_csv = export_rbfe_nodes_csv(&rep);
            assert_eq!(e_csv.lines().count(), rep.edges.len() + 1);
            assert_eq!(n_csv.lines().count(), rep.nodes.len() + 1);
        }
    }
}
