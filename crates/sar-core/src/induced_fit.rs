//! Flexible Binding-Pocket Side-Chain `Dunbrack` Rotamer Repacker (`Sherman et al. 2006 IFD` /
//! `AutoDockFR`), Multi-Conformation Receptor Ensemble Builder (`Apo`, `Holo`, `Gatekeeper-Out`,
//! `ANM` Breathing & `Cryptic-Latch`), and `Fischer-Shoichet` Reorganization-Penalized Boltzmann
//! Ensemble-Docking Free-Energy Scorer (`induced_fit.rs`, `v3.3.0`).
//!
//! Primary References:
//! - Sherman, Day, Jacobson, Friesner & Farid (2006), *J. Med. Chem.* 49(2), 534-553 ("Novel
//!   Procedure for Modeling Ligand/Receptor Induced Fit Effects" — softened vdW placement, `5 A`
//!   side-chain rotamer repacking & composite `IFDScore`: `https://pubmed.ncbi.nlm.nih.gov/16420040/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11513859/`)
//! - Ravindranath, Forli, Goodsell, Olson & Sanner (2015), *PLoS Comput. Biol.* 11(12):e1004586
//!   ("AutoDockFR: Advances in Protein-Ligand Docking with Explicitly Specified Binding Site
//!   Flexibility": `https://pmc.ncbi.nlm.nih.gov/articles/PMC4667975/`)
//! - Shapovalov & Dunbrack (2011), *Structure* 19(6), 844-858 ("A Smoothed Backbone-Dependent
//!   Rotamer Library for Proteins Derived from Adaptive Kernel Density Estimates and Regressions":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3118414/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC7938712/`)
//! - Fischer, Coleman, Fraser & Shoichet (2014), *Nat. Chem.* 6(7), 575-583 ("Incorporation of
//!   protein flexibility and conformational energy penalties in docking screens to improve ligand
//!   discovery" — receptor reorganization penalty `dG_reorg = -RT ln(w_k / w_0)`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4144196/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC8433570/`)
//! - Amaro et al. (2018), *Biophys. J.* 114(10), 2271-2278 ("Ensemble Docking in Drug Discovery":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC6129458/`) & Totrov & Abagyan (2008), *Curr. Opin.
//!   Struct. Biol.* 18(2), 178-184 ("Flexible ligand docking to multiple receptor conformations":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC2396190/`).

use crate::allosteric_pocket::curated_allosteric_pdb_text;
use crate::conformer3d::{
    bondi_vdw_radius, compute_gasteiger_charges, format_3d_conformer_sdf,
    generate_3d_conformer_ensemble, ConformerGenConfig,
};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::mol::{Element, Molecule};
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    ParsedPdbStructure, PdbAtom, PocketCavityAnalysis,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Thermal energy `R * T` at `298.15 K` (`0.5925 kcal/mol`) for Boltzmann partition functions.
pub const IFD_RT_298K_KCAL: f64 = 0.592_484;

/// Default Sherman (2006) receptor internal reorganization weight (`0.05..0.25` for `IFDScore`;
/// full `1.0` thermodynamic weight used in `Fischer-Shoichet` Boltzmann partition scoring).
pub const SHERMAN_IFD_REORG_SCALE: f64 = 0.20;

/// Curated flexible-receptor benchmark systems + custom `.pdb` input mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InducedFitPreset {
    /// `PDB 1M17` (`EGFR`): Flexible `Thr790` gatekeeper, `Lys745`/`Glu762` salt bridge, `Met766`/`Phe856` back-pocket & `Cys797`.
    Egfr1m17GatekeeperAndBackPocket,
    /// `PDB 6CM4` (`DRD2`): Flexible `Trp386` (`W6.48` toggle switch), `Phe389`/`Phe390` aromatic cage, `Asp114` & `Ile184` ECL2 lid.
    Drd26cm4ToggleSwitchAndCage,
    /// `PDB 3LN1` (`COX-2`): Flexible `Arg513` selectivity latch, `Val509`/`Leu338` side-pocket gate & `His90`/`Tyr341`.
    Cox23ln1Arg513AndSidePocket,
    /// `PDB 6OIM` (`KRAS G12C`): Flexible `His95`/`Tyr96`/`Gln99` cryptic groove gate & `Met67`/`Arg68` Switch-II loop.
    Kras6oimHis95SwitchIIGroove,
    /// User-supplied `.pdb` structure with automatic cavity-lining flexible side-chain perception.
    CustomPdbFlexibleEnsemble,
}

impl InducedFitPreset {
    pub const ALL: [Self; 5] = [
        Self::Egfr1m17GatekeeperAndBackPocket,
        Self::Drd26cm4ToggleSwitchAndCage,
        Self::Cox23ln1Arg513AndSidePocket,
        Self::Kras6oimHis95SwitchIIGroove,
        Self::CustomPdbFlexibleEnsemble,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::Egfr1m17GatekeeperAndBackPocket => "IFD-1M17-EGFR",
            Self::Drd26cm4ToggleSwitchAndCage => "IFD-6CM4-DRD2",
            Self::Cox23ln1Arg513AndSidePocket => "IFD-3LN1-COX2",
            Self::Kras6oimHis95SwitchIIGroove => "IFD-6OIM-KRAS",
            Self::CustomPdbFlexibleEnsemble => "IFD-CUSTOM-PDB",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Egfr1m17GatekeeperAndBackPocket => "EGFR 1M17 (Thr790/Met766)",
            Self::Drd26cm4ToggleSwitchAndCage => "DRD2 6CM4 (Trp386/Phe389)",
            Self::Cox23ln1Arg513AndSidePocket => "COX-2 3LN1 (Arg513/Val509)",
            Self::Kras6oimHis95SwitchIIGroove => "KRAS 6OIM (His95/Switch-II)",
            Self::CustomPdbFlexibleEnsemble => "Custom .PDB Ensemble",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Egfr1m17GatekeeperAndBackPocket => {
                "EGFR Kinase (1M17): Thr790 Gatekeeper, Lys745-Glu762 & Met766/Phe856 Back-Pocket Ensemble"
            }
            Self::Drd26cm4ToggleSwitchAndCage => {
                "Dopamine D2 GPCR (6CM4): Trp386 Rotamer Toggle Switch, Phe389 Cage & ECL2 Ile184 Ensemble"
            }
            Self::Cox23ln1Arg513AndSidePocket => {
                "COX-2 (3LN1): Arg513 Sulfonamide Latch, Val509 Gate & Tyr341 Constriction Ensemble"
            }
            Self::Kras6oimHis95SwitchIIGroove => {
                "KRAS G12C (6OIM): His95 Cryptic Groove Flip & Switch-II Met67/Arg68 Loop Ensemble"
            }
            Self::CustomPdbFlexibleEnsemble => {
                "Custom .PDB Receptor Structure with Automatic Cavity Side-Chain Rotamer Ensemble"
            }
        }
    }
}

/// Canonical `Shapovalov & Dunbrack (2011)` side-chain `chi1`/`chi2` rotamer state identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DunbrackRotamerId {
    /// Crystallographic reference side-chain dihedral (`d_chi1 = 0 deg, d_chi2 = 0 deg`).
    NativeCrystal,
    /// `gauche+` (`g+`, `d_chi1 = +62 deg, d_chi2 = +10 deg`).
    GauchePlus,
    /// `trans` (`t`, `d_chi1 = +122 deg, d_chi2 = +35 deg`).
    Trans180,
    /// `gauche-` (`g-`, `d_chi1 = -64 deg, d_chi2 = -25 deg`).
    GaucheMinus,
    /// High-energy outward induced-fit gatekeeper / aromatic latch swing (`d_chi1 = +96 deg, d_chi2 = +52 deg`).
    OutwardInducedLatch,
}

impl DunbrackRotamerId {
    pub const ALL: [Self; 5] = [
        Self::NativeCrystal,
        Self::GauchePlus,
        Self::Trans180,
        Self::GaucheMinus,
        Self::OutwardInducedLatch,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::NativeCrystal => "cryst",
            Self::GauchePlus => "g+ (+62)",
            Self::Trans180 => "t (180)",
            Self::GaucheMinus => "g- (-64)",
            Self::OutwardInducedLatch => "open-latch",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::NativeCrystal => "Native Crystal Rotamer (0 deg)",
            Self::GauchePlus => "Dunbrack gauche+ (+62 deg)",
            Self::Trans180 => "Dunbrack trans (+122 / 180 deg)",
            Self::GaucheMinus => "Dunbrack gauche- (-64 deg)",
            Self::OutwardInducedLatch => "Induced-Fit Outward Gate Latch (+96 deg)",
        }
    }

    /// Returns `(delta_chi1_deg, delta_chi2_deg, prior_prob, internal_rotamer_strain_kcal)`.
    pub fn parameters_for_residue(self, res_name: &str) -> (f64, f64, f64, f64) {
        let is_bulky_arom = matches!(res_name, "PHE" | "TYR" | "TRP" | "HIS");
        let is_branched = matches!(res_name, "VAL" | "ILE" | "THR");
        match self {
            Self::NativeCrystal => (0.0, 0.0, 0.50, 0.00),
            Self::GauchePlus => (
                62.0,
                if is_bulky_arom { 18.0 } else { 8.0 },
                if is_branched { 0.24 } else { 0.21 },
                0.48,
            ),
            Self::Trans180 => (
                122.0,
                if is_bulky_arom { 42.0 } else { 25.0 },
                0.16,
                0.68,
            ),
            Self::GaucheMinus => (
                -64.0,
                if is_bulky_arom { -32.0 } else { -18.0 },
                0.09,
                1.02,
            ),
            Self::OutwardInducedLatch => (
                if is_bulky_arom { 98.0 } else { 88.0 },
                54.0,
                0.04,
                if is_branched { 1.65 } else { 1.42 },
            ),
        }
    }
}

/// Mechanistic binding regime diagnosed from Rigid vs Induced-Fit Ensemble thermodynamics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingMechanismRegime {
    /// Binds tightly to the native ground-state pocket (`|ddG_rescue| < 0.65 kcal/mol`, low `KL` shift).
    LockAndKeyGroundState,
    /// Selects a pre-existing low-penalty breathing microstate from the apo ensemble (`0.65 <= -ddG_rescue < 1.60 kcal/mol`).
    ConformationalSelection,
    /// Requires active side-chain rotamer repacking / gatekeeper opening (`-ddG_rescue >= 1.60 kcal/mol`, rescued from rigid clash!).
    InducedFitClashRescue,
    /// High receptor reorganization cost outweighs interaction gain (`dG_ens > -5.5 kcal/mol`).
    HighReorgNonBinder,
}

impl BindingMechanismRegime {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::LockAndKeyGroundState => "Lock-and-Key",
            Self::ConformationalSelection => "Conf-Selection",
            Self::InducedFitClashRescue => "Induced-Fit Rescue!",
            Self::HighReorgNonBinder => "High-Reorg Weak",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LockAndKeyGroundState => {
                "Lock-and-Key Ground-State Binder (Minimal Receptor Reorganization Required)"
            }
            Self::ConformationalSelection => {
                "Conformational Selection (Stabilizes Pre-Existing Open Microstate)"
            }
            Self::InducedFitClashRescue => {
                "True Induced-Fit Clash Rescue (Side-Chain Rotamer Flip Relieves Rigid-Pocket Steric Clash)"
            }
            Self::HighReorgNonBinder => {
                "Reorganization-Penalized Weak Binder (Pocket Strain Exceeds Interaction Gain)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::LockAndKeyGroundState => [56, 189, 248],   // Sky Blue
            Self::ConformationalSelection => [192, 132, 252], // Violet
            Self::InducedFitClashRescue => [52, 211, 153],   // Emerald
            Self::HighReorgNonBinder => [248, 113, 113],     // Coral
        }
    }
}

/// Cavity-facing flexible side chain perceived in `ParsedPdbStructure`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlexiblePocketResidue {
    pub flex_idx: usize,
    pub residue_label: String,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub dist_to_pocket_center_a: f32,
    /// Fixed anchor coordinate (`CA`/`CB` or inferred backbone root) around which `chi1` rotates.
    pub pivot_pos: [f32; 3],
    /// Unit vector defining `chi1` rotation axis (`CA -> CB` or outward radial bond).
    pub chi1_axis: [f32; 3],
    /// Secondary hinge coordinate (`CB`/`CG`) for distal `chi2` rotation.
    pub chi2_pivot_pos: [f32; 3],
    pub chi2_axis: [f32; 3],
    /// Indices into `ParsedPdbStructure.protein_atoms` moved by `chi1`.
    pub movable_atom_indices: Vec<usize>,
    /// Subset of `movable_atom_indices` beyond `CG` additionally twisted by `chi2`.
    pub distal_chi2_atom_indices: Vec<usize>,
    pub native_atom_coords: Vec<[f32; 3]>,
}

/// Single repacked side-chain transition recorded for a ligand in a given receptor state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SideChainRepackMove {
    pub residue_label: String,
    pub res_name: String,
    pub selected_rotamer: DunbrackRotamerId,
    pub delta_chi1_deg: f64,
    pub delta_chi2_deg: f64,
    pub sidechain_rmsd_a: f64,
    pub rotamer_strain_kcal: f64,
    /// Net steric-clash + H-bond/vdW gain from repacking this residue (`kcal/mol`, negative = favorable relief!).
    pub interaction_delta_kcal: f64,
    pub native_coords: Vec<[f32; 3]>,
    pub repacked_coords: Vec<[f32; 3]>,
}

/// Macrostate member `k` (`1..=5`) of the pre-generated receptor conformational ensemble.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceptorEnsembleMember {
    pub state_idx: usize,
    #[serde(skip_deserializing)]
    pub state_code: &'static str,
    #[serde(skip_deserializing)]
    pub state_title: &'static str,
    /// Intrinsic `Fischer-Shoichet` apo conformational reorganization energy `dG_reorg,k >= 0` (`kcal/mol`).
    pub apo_reorg_penalty_kcal: f64,
    /// Normalized equilibrium apo Boltzmann population `w_apo,k` (`sum_k w_apo,k = 1.0`).
    pub apo_boltzmann_weight: f64,
    pub pocket_volume_a3: f64,
    pub pocket_dscore: f64,
    pub backbone_rmsd_vs_holo_a: f64,
    pub pdb_state: ParsedPdbStructure,
    pub pocket_analysis: PocketCavityAnalysis,
}

/// State-specific induced-fit docking & side-chain rotamer repacking outcome on ensemble member `k`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleStateDockingScore {
    pub state_idx: usize,
    #[serde(skip_deserializing)]
    pub state_code: &'static str,
    #[serde(skip_deserializing)]
    pub state_title: &'static str,
    pub apo_reorg_penalty_kcal: f64,
    pub apo_weight: f64,
    /// Rigid-sidechain binding `dG` in state `k` BEFORE rotamer repacking (`kcal/mol`).
    pub pre_repack_dg_kcal: f64,
    /// Post-repacking protein-ligand interaction `dG` (`kcal/mol`).
    pub post_repack_interaction_dg_kcal: f64,
    /// Summed `Dunbrack` internal side-chain rotamer strain across repacked residues (`>= 0 kcal/mol`).
    pub rotamer_strain_kcal: f64,
    /// Total thermodynamic state free energy `dG_eff,k = dG_int,k + dG_reorg,k + dG_rot,k` (`kcal/mol`).
    pub effective_total_dg_kcal: f64,
    /// Sherman (2006) `IFDScore = dG_int,k + 0.20 * (dG_reorg,k + dG_rot,k)` (`kcal/mol`).
    pub sherman_ifd_score_kcal: f64,
    /// Normalized holo (ligand-bound) Boltzmann state occupancy `w_holo,k` (`sum_k w_holo,k = 1.0`).
    pub holo_boltzmann_weight: f64,
    pub repacked_residue_count: usize,
    pub repacked_sidechain_rmsd_a: f64,
    pub aligned_ligand_coords3d: Vec<[f32; 3]>,
    pub repacked_moves: Vec<SideChainRepackMove>,
    pub repacked_shell_atoms: Vec<PdbAtom>,
}

/// Complete `v3.3.0` Induced-Fit Rotamer Repacking & Boltzmann Ensemble-Docking Report for a ligand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InducedFitEnsembleReport {
    pub preset: InducedFitPreset,
    pub pdb_id: String,
    pub target_name: String,
    pub compound_id: String,
    pub compound_name: String,
    pub canonical_smiles: String,

    // Rigid Single-Structure Baseline vs Induced-Fit Boltzmann Ensemble Readouts
    pub rigid_holo_dg_kcal: f64,
    pub rigid_holo_clash_atoms: usize,
    /// Exact normalized Boltzmann ensemble binding free energy `-RT ln(Z_bound / Z_apo)` (`kcal/mol`).
    pub boltzmann_ensemble_dg_kcal: f64,
    /// Best single-state `Sherman (2006)` `IFDScore` (`kcal/mol`).
    pub best_sherman_ifd_score_kcal: f64,
    /// Net induced-fit affinity rescue `ddG_rescue = dG_Boltzmann - dG_rigid_holo` (`kcal/mol`, negative = rescue!).
    pub ddg_induced_fit_rescue_kcal: f64,
    pub ensemble_predicted_pkd: f64,
    pub ensemble_le: f64,
    pub ensemble_lle: f64,

    // Conformational Selection / State Redistribution Telemetry
    pub dominant_state_idx: usize,
    #[serde(skip_deserializing)]
    pub dominant_state_code: &'static str,
    pub dominant_holo_occupancy_pct: f64,
    /// Kullback-Leibler divergence `D_KL(w_holo || w_apo)` (`nats`) quantifying ligand-driven state shift.
    pub kl_population_shift_nats: f64,
    /// Conformational entropy loss `-T * dS_ens` (`kcal/mol`) from collapsing the receptor ensemble.
    pub receptor_conf_entropy_cost_kcal: f64,
    pub mechanism: BindingMechanismRegime,

    pub flexible_residues: Vec<FlexiblePocketResidue>,
    pub state_scores: Vec<EnsembleStateDockingScore>,
    pub mol: Molecule,
}

/// Compact row for dataset-wide Induced-Fit & Boltzmann Ensemble triage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetInducedFitRow {
    pub rank: usize,
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub mw: f64,
    pub clogp: f64,
    pub observed_pact: f64,
    pub rigid_holo_dg_kcal: f64,
    pub boltzmann_ensemble_dg_kcal: f64,
    pub ddg_induced_fit_rescue_kcal: f64,
    pub sherman_ifd_score_kcal: f64,
    pub ensemble_predicted_pkd: f64,
    #[serde(skip_deserializing)]
    pub dominant_state_code: &'static str,
    pub dominant_occupancy_pct: f64,
    pub repacked_residues: usize,
    pub mechanism: BindingMechanismRegime,
    pub canonical_smiles: String,
}

/// Configuration for flexible side-chain rotamer repacking & Boltzmann ensemble docking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InducedFitConfig {
    pub preset: InducedFitPreset,
    pub custom_pdb_path: Option<String>,
    /// Maximum number of pocket-lining side chains allowed to repack (`2..=12`, default `6`).
    pub max_flexible_residues: usize,
    /// Soft-core van der Waals radius multiplier during Stage-1 clash-tolerant pose placement (`0.50..=0.90`).
    pub soft_core_vdw_scale: f64,
    /// Temperature (`K`) for Boltzmann ensemble weighting (`273..=320 K`, default `298.15 K`).
    pub temperature_k: f64,
}

impl Default for InducedFitConfig {
    fn default() -> Self {
        Self {
            preset: InducedFitPreset::Egfr1m17GatekeeperAndBackPocket,
            custom_pdb_path: None,
            max_flexible_residues: 6,
            soft_core_vdw_scale: 0.65,
            temperature_k: 298.15,
        }
    }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn norm3_f32(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

/// Rotates 3D point `pt` around line `(pivot, unit_axis)` by `angle_deg` degrees (Rodrigues formula).
pub fn rotate_point_around_axis_3d(
    pt: [f32; 3],
    pivot: [f32; 3],
    unit_axis: [f32; 3],
    angle_deg: f64,
) -> [f32; 3] {
    if angle_deg.abs() < 1e-6 {
        return pt;
    }
    let rad = angle_deg.to_radians();
    let cos_t = rad.cos() as f32;
    let sin_t = rad.sin() as f32;
    let k = norm3_f32(unit_axis);
    let vx = pt[0] - pivot[0];
    let vy = pt[1] - pivot[1];
    let vz = pt[2] - pivot[2];

    let k_dot_v = k[0] * vx + k[1] * vy + k[2] * vz;
    let cx = k[1] * vz - k[2] * vy;
    let cy = k[2] * vx - k[0] * vz;
    let cz = k[0] * vy - k[1] * vx;

    [
        pivot[0] + vx * cos_t + cx * sin_t + k[0] * k_dot_v * (1.0 - cos_t),
        pivot[1] + vy * cos_t + cy * sin_t + k[1] * k_dot_v * (1.0 - cos_t),
        pivot[2] + vz * cos_t + cz * sin_t + k[2] * k_dot_v * (1.0 - cos_t),
    ]
}

fn is_backbone_atom_name(an: &str) -> bool {
    matches!(an.trim(), "N" | "CA" | "C" | "O" | "OXT")
}

/// Perceives up to `max_flex` cavity-lining flexible side chains (`excluding GLY/ALA/PRO`)
/// and constructs their `chi1` and `chi2` 3D rotation axes (handling both full-atom `.pdb` and
/// compact pocket-shell `.pdb` extracts).
pub fn perceive_flexible_pocket_residues(
    pdb: &ParsedPdbStructure,
    pocket_center: [f32; 3],
    max_flex: usize,
) -> Vec<FlexiblePocketResidue> {
    let mut by_res: BTreeMap<(char, i32, String), Vec<usize>> = BTreeMap::new();
    for (idx, a) in pdb.protein_atoms.iter().enumerate() {
        by_res
            .entry((a.chain_id, a.res_seq, a.res_name.clone()))
            .or_default()
            .push(idx);
    }

    let mut candidates: Vec<FlexiblePocketResidue> = Vec::new();

    for ((chain_id, res_seq, res_name), atom_idxs) in by_res {
        if matches!(res_name.as_str(), "GLY" | "ALA" | "PRO") {
            continue;
        }
        // Identify optional CA and CB atoms
        let ca_idx = atom_idxs
            .iter()
            .copied()
            .find(|&i| pdb.protein_atoms[i].atom_name.trim() == "CA");
        let cb_idx = atom_idxs
            .iter()
            .copied()
            .find(|&i| pdb.protein_atoms[i].atom_name.trim() == "CB");
        let cg_idx = atom_idxs.iter().copied().find(|&i| {
            let an = pdb.protein_atoms[i].atom_name.trim();
            matches!(an, "CG" | "CG1" | "SG" | "OG" | "OG1")
        });

        let movable: Vec<usize> = atom_idxs
            .iter()
            .copied()
            .filter(|&i| {
                let an = pdb.protein_atoms[i].atom_name.trim();
                !is_backbone_atom_name(an) && an != "CB"
            })
            .collect();

        if movable.is_empty() {
            continue;
        }

        let min_d = movable
            .iter()
            .map(|&i| dist3(pdb.protein_atoms[i].pos, pocket_center))
            .fold(f32::INFINITY, f32::min);
        if min_d > 9.6 {
            continue;
        }

        // Compute side-chain centroid of movable atoms
        let inv_m = 1.0 / (movable.len() as f32);
        let mut sc_cent = [0.0f32; 3];
        for &mi in &movable {
            let p = pdb.protein_atoms[mi].pos;
            sc_cent[0] += p[0] * inv_m;
            sc_cent[1] += p[1] * inv_m;
            sc_cent[2] += p[2] * inv_m;
        }

        // Determine chi1 pivot & axis (exact CA->CB when present, else virtual radial root 1.54 A outward from sc_cent)
        let (pivot_pos, chi1_axis) = match (ca_idx, cb_idx) {
            (Some(ica), Some(icb)) => {
                let pca = pdb.protein_atoms[ica].pos;
                let pcb = pdb.protein_atoms[icb].pos;
                (pcb, norm3_f32([pcb[0] - pca[0], pcb[1] - pca[1], pcb[2] - pca[2]]))
            }
            (None, Some(icb)) => {
                let pcb = pdb.protein_atoms[icb].pos;
                let out = norm3_f32([
                    pcb[0] - pocket_center[0],
                    pcb[1] - pocket_center[1],
                    pcb[2] - pocket_center[2],
                ]);
                // Orthogonalize slightly so chi1 swings the tip laterally around the CB-CA stem
                let stem = norm3_f32([sc_cent[0] - pcb[0], sc_cent[1] - pcb[1], sc_cent[2] - pcb[2] + 0.35 * out[2]]);
                (pcb, stem)
            }
            (Some(ica), None) => {
                let pca = pdb.protein_atoms[ica].pos;
                let axis = norm3_f32([sc_cent[0] - pca[0], sc_cent[1] - pca[1], sc_cent[2] - pca[2]]);
                let virt_cb = [
                    pca[0] + 1.52 * axis[0],
                    pca[1] + 1.52 * axis[1],
                    pca[2] + 1.52 * axis[2],
                ];
                (virt_cb, axis)
            }
            (None, None) => {
                let radial_out = norm3_f32([
                    sc_cent[0] - pocket_center[0],
                    sc_cent[1] - pocket_center[1],
                    sc_cent[2] - pocket_center[2],
                ]);
                // Place virtual CB 1.65 A behind side-chain centroid away from pocket center,
                // with a tilted C_alpha-C_beta bond axis so rotation sweeps the side-chain tip!
                let virt_cb = [
                    sc_cent[0] + 1.65 * radial_out[0],
                    sc_cent[1] + 1.65 * radial_out[1],
                    sc_cent[2] + 1.65 * radial_out[2],
                ];
                let tangent = norm3_f32([
                    -radial_out[1] + 0.35 * radial_out[2],
                    radial_out[0] - 0.25 * radial_out[2],
                    0.55 * radial_out[0] + 0.45,
                ]);
                (virt_cb, tangent)
            }
        };

        let (chi2_pivot_pos, chi2_axis) = if let Some(icg) = cg_idx {
            let pcg = pdb.protein_atoms[icg].pos;
            (
                pcg,
                norm3_f32([
                    pcg[0] - pivot_pos[0],
                    pcg[1] - pivot_pos[1],
                    pcg[2] - pivot_pos[2],
                ]),
            )
        } else {
            (sc_cent, norm3_f32([chi1_axis[1], -chi1_axis[2], chi1_axis[0]]))
        };

        let distal_chi2: Vec<usize> = movable
            .iter()
            .copied()
            .filter(|&i| {
                let an = pdb.protein_atoms[i].atom_name.trim();
                !matches!(an, "CG" | "CG1" | "CG2" | "SG" | "OG" | "OG1")
            })
            .collect();

        let native_coords = movable.iter().map(|&i| pdb.protein_atoms[i].pos).collect();

        candidates.push(FlexiblePocketResidue {
            flex_idx: 0,
            residue_label: format!("{}{}", res_name, res_seq),
            res_name,
            chain_id,
            res_seq,
            dist_to_pocket_center_a: min_d,
            pivot_pos,
            chi1_axis,
            chi2_pivot_pos,
            chi2_axis,
            movable_atom_indices: movable,
            distal_chi2_atom_indices: distal_chi2,
            native_atom_coords: native_coords,
        });
    }

    candidates.sort_by(|a, b| {
        a.dist_to_pocket_center_a
            .partial_cmp(&b.dist_to_pocket_center_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(max_flex.clamp(1, 14));
    for (idx, r) in candidates.iter_mut().enumerate() {
        r.flex_idx = idx;
    }
    candidates
}

/// Applies `(delta_chi1_deg, delta_chi2_deg)` to `res` and returns transformed coordinates for `res.movable_atom_indices`.
fn evaluate_rotamer_coords(
    res: &FlexiblePocketResidue,
    d_chi1_deg: f64,
    d_chi2_deg: f64,
) -> Vec<[f32; 3]> {
    let chi2_piv_rot =
        rotate_point_around_axis_3d(res.chi2_pivot_pos, res.pivot_pos, res.chi1_axis, d_chi1_deg);
    let chi2_tip_rot = rotate_point_around_axis_3d(
        [
            res.chi2_pivot_pos[0] + res.chi2_axis[0],
            res.chi2_pivot_pos[1] + res.chi2_axis[1],
            res.chi2_pivot_pos[2] + res.chi2_axis[2],
        ],
        res.pivot_pos,
        res.chi1_axis,
        d_chi1_deg,
    );
    let rot_chi2_axis = norm3_f32([
        chi2_tip_rot[0] - chi2_piv_rot[0],
        chi2_tip_rot[1] - chi2_piv_rot[1],
        chi2_tip_rot[2] - chi2_piv_rot[2],
    ]);

    res.movable_atom_indices
        .iter()
        .enumerate()
        .map(|(k, &atom_idx)| {
            let p1 = rotate_point_around_axis_3d(
                res.native_atom_coords[k],
                res.pivot_pos,
                res.chi1_axis,
                d_chi1_deg,
            );
            if d_chi2_deg.abs() > 1e-4 && res.distal_chi2_atom_indices.contains(&atom_idx) {
                rotate_point_around_axis_3d(p1, chi2_piv_rot, rot_chi2_axis, d_chi2_deg)
            } else {
                p1
            }
        })
        .collect()
}

/// Loads the base `.pdb` structure & pocket center for a preset or custom `.pdb` file.
pub fn load_induced_fit_base_pdb(
    cfg: &InducedFitConfig,
) -> (ParsedPdbStructure, Option<[f32; 3]>, String) {
    if cfg.preset == InducedFitPreset::CustomPdbFlexibleEnsemble {
        if let Some(ref p) = cfg.custom_pdb_path {
            let trimmed = p.trim();
            if !trimmed.is_empty() {
                if let Ok(txt) = std::fs::read_to_string(trimmed) {
                    if let Ok(parsed) = parse_pdb_structure(&txt) {
                        let t_name = parsed.pdb_id.clone();
                        return (parsed, None, t_name);
                    }
                }
            }
        }
    }

    match cfg.preset {
        InducedFitPreset::Egfr1m17GatekeeperAndBackPocket
        | InducedFitPreset::CustomPdbFlexibleEnsemble => (
            parse_pdb_structure(curated_pocket_pdb_text(0)).expect("Valid 1M17 PDB"),
            None,
            "EGFR".to_string(),
        ),
        InducedFitPreset::Drd26cm4ToggleSwitchAndCage => (
            parse_pdb_structure(curated_pocket_pdb_text(1)).expect("Valid 6CM4 PDB"),
            None,
            "DRD2".to_string(),
        ),
        InducedFitPreset::Cox23ln1Arg513AndSidePocket => (
            parse_pdb_structure(curated_pocket_pdb_text(2)).expect("Valid 3LN1 PDB"),
            None,
            "COX-2".to_string(),
        ),
        InducedFitPreset::Kras6oimHis95SwitchIIGroove => (
            parse_pdb_structure(curated_allosteric_pdb_text(1)).expect("Valid 6OIM PDB"),
            Some([12.90, 5.10, 8.20]),
            "KRAS-G12C".to_string(),
        ),
    }
}

/// Constructs a 5-state physically energy-weighted receptor conformational ensemble
/// (`S1 Holo Ground`, `S2 Apo Occluded`, `S3 Gatekeeper/Toggle-Out`, `S4 Breathing Expanded`, `S5 Cryptic Deep Cleft`)
/// with normalized `Fischer-Shoichet` apo Boltzmann weights `w_apo,k`.
pub fn build_receptor_ensemble(
    base_pdb: &ParsedPdbStructure,
    center_override: Option<[f32; 3]>,
    flex_residues: &[FlexiblePocketResidue],
    temperature_k: f64,
) -> (PocketCavityAnalysis, Vec<ReceptorEnsembleMember>) {
    let base_pocket = analyze_binding_pocket(base_pdb, center_override, 10.0);
    let center = base_pocket.center;
    let rt = 0.001_987_204 * temperature_k.clamp(250.0, 340.0);

    // Define 5 distinct macrostates: `(code, title, radial_scale, shear_amp, rotamer_seed, apo_reorg_kcal)`
    let specs: [(&'static str, &'static str, f32, f32, Option<DunbrackRotamerId>, f64); 5] = [
        (
            "S1-HOLO-GROUND",
            "Native Holo Ground-State Cavity",
            0.00,
            0.00,
            None,
            0.00,
        ),
        (
            "S2-APO-OCCLUDED",
            "Inward Apo Collapsed / Side-Chain Occluded State",
            -0.38,
            0.12,
            Some(DunbrackRotamerId::GauchePlus),
            0.35,
        ),
        (
            "S3-GATE-OUT",
            "Gatekeeper / Toggle-Switch Outward Rotamer State",
            0.28,
            -0.15,
            Some(DunbrackRotamerId::GaucheMinus),
            0.92,
        ),
        (
            "S4-ANM-BREATHE",
            "Collective Elastic Breathing Expanded Cleft",
            0.62,
            0.26,
            Some(DunbrackRotamerId::Trans180),
            1.38,
        ),
        (
            "S5-CRYPTIC-OPEN",
            "High-Energy Induced-Fit Cryptic Sub-Pocket Open State",
            0.95,
            -0.32,
            Some(DunbrackRotamerId::OutwardInducedLatch),
            2.05,
        ),
    ];

    let z_apo: f64 = specs
        .iter()
        .map(|(_, _, _, _, _, e_reorg)| (-e_reorg / rt).exp())
        .sum::<f64>()
        .max(1e-12);

    let mut members = Vec::with_capacity(specs.len());
    for (si, &(code, title, rad_shift, shear, rot_opt, reorg_kcal)) in specs.iter().enumerate() {
        let mut state_pdb = base_pdb.clone();
        let mut sq_disp = 0.0f64;

        // Collective breathing /apo inward contraction around cavity center
        for a in &mut state_pdb.protein_atoms {
            let d_vec = [a.pos[0] - center[0], a.pos[1] - center[1], a.pos[2] - center[2]];
            let r = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
            if r > 0.5 && r < 12.5 {
                let atten = (-(r - 5.2).powi(2) / 36.0).exp();
                let u = [d_vec[0] / r, d_vec[1] / r, d_vec[2] / r];
                let dx = (rad_shift * u[0] + shear * u[1]) * atten;
                let dy = (rad_shift * u[1] - shear * u[0]) * atten;
                let dz = (rad_shift * 0.65 * u[2]) * atten;
                a.pos[0] += dx;
                a.pos[1] += dy;
                a.pos[2] += dz;
                sq_disp += ((dx * dx + dy * dy + dz * dz) as f64);
            }
        }

        // Pre-seed gatekeeper / primary lining residues #0 and #1 into macrostate rotamer well
        if let Some(rot_id) = rot_opt {
            for res in flex_residues.iter().take(2) {
                let (dc1, dc2, _, _) = rot_id.parameters_for_residue(&res.res_name);
                let new_pts = evaluate_rotamer_coords(res, dc1 * 0.75, dc2 * 0.75);
                for (k, &ai) in res.movable_atom_indices.iter().enumerate() {
                    if ai < state_pdb.protein_atoms.len() {
                        let old_p = base_pdb.protein_atoms[ai].pos;
                        state_pdb.protein_atoms[ai].pos = new_pts[k];
                        let dd = dist3(old_p, new_pts[k]) as f64;
                        sq_disp += dd * dd;
                    }
                }
            }
        }

        let n_prot = state_pdb.protein_atoms.len().max(1) as f64;
        let bb_rmsd = (sq_disp / n_prot).sqrt();
        let pocket_k = analyze_binding_pocket(&state_pdb, Some(center), 10.0);
        let w_apo = (-reorg_kcal / rt).exp() / z_apo;

        members.push(ReceptorEnsembleMember {
            state_idx: si,
            state_code: code,
            state_title: title,
            apo_reorg_penalty_kcal: reorg_kcal,
            apo_boltzmann_weight: w_apo,
            pocket_volume_a3: pocket_k.volume_angstrom3,
            pocket_dscore: pocket_k.druggability_dscore,
            backbone_rmsd_vs_holo_a: bb_rmsd,
            pdb_state: state_pdb,
            pocket_analysis: pocket_k,
        });
    }

    (base_pocket, members)
}

/// Pairwise empirical interaction energy (`kcal/mol`) between ligand atoms (`lig_coords`) and a subset
/// of receptor side-chain atoms (`sc_coords`, `sc_elems`), using `vdw_scale` (`0.65` soft-core vs `1.0` hard-core).
fn score_ligand_sidechain_pair_kcal(
    mol: &Molecule,
    lig_coords: &[[f32; 3]],
    sc_coords: &[[f32; 3]],
    sc_elems: &[Element],
    vdw_scale: f64,
) -> f64 {
    let mut e_sum = 0.0f64;
    for (li, &lp) in lig_coords.iter().enumerate() {
        let Some(la) = mol.atoms.get(li) else {
            continue;
        };
        if la.elem == Element::H {
            continue;
        }
        let r_l = bondi_vdw_radius(la.elem) as f64;
        let lig_polar = matches!(la.elem, Element::N | Element::O | Element::S);

        for (si, &sp) in sc_coords.iter().enumerate() {
            let d = dist3(lp, sp) as f64;
            if d > 6.2 {
                continue;
            }
            let se = sc_elems.get(si).copied().unwrap_or(Element::C);
            let r_s = bondi_vdw_radius(se) as f64;
            let r_opt = (r_l + r_s) * vdw_scale;
            let sc_polar = matches!(se, Element::N | Element::O | Element::S);

            if d < r_opt * 0.86 {
                let overlap = r_opt * 0.86 - d;
                e_sum += 5.40 * overlap * overlap + 1.15 * overlap;
            } else {
                // Attractive Vina-like Gaussian contact + polar H-bond / salt-bridge well around 2.85 A
                let surf = d - (r_l + r_s);
                let vdw_gauss = -0.22 * (-(surf / 0.65).powi(2)).exp();
                let hbond_term = if lig_polar && sc_polar && (2.45..=3.45).contains(&d) {
                    -0.68 * (-((d - 2.88) / 0.35).powi(2)).exp()
                } else {
                    0.0
                };
                e_sum += vdw_gauss + hbond_term;
            }
        }
    }
    e_sum
}

/// Evaluates steric self-clash (`kcal/mol`) of a candidate rotamer's movable atoms against fixed protein background atoms.
fn score_sidechain_self_clash_kcal(
    pdb: &ParsedPdbStructure,
    movable_set: &[usize],
    cand_coords: &[[f32; 3]],
) -> f64 {
    let mut pen = 0.0f64;
    for (ai, p_bg) in pdb.protein_atoms.iter().enumerate() {
        if movable_set.contains(&ai) {
            continue;
        }
        for &cp in cand_coords {
            let d = dist3(cp, p_bg.pos) as f64;
            if d < 2.25 {
                let ov = 2.25 - d;
                pen += 3.8 * ov * ov;
            }
        }
    }
    pen
}

/// Executes 3-step `Sherman (2006) IFD` + `Dunbrack` rotamer repacking on a single receptor state `member`.
fn dock_and_repack_in_state(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    lig_ensemble_coords: &[Vec<[f32; 3]>],
    member: &ReceptorEnsembleMember,
    cfg: &InducedFitConfig,
) -> EnsembleStateDockingScore {
    // 1. Score rigid pre-repacked state first + pick best soft-core clash-tolerant ligand pose
    let mol_idx = ds.and_then(|d| d.mols.iter().position(|dm| dm.id == mol.id));
    let pre_res = score_ligand_in_pocket(mol, mol_idx, &member.pocket_analysis);
    let pre_dg = pre_res.total_dg_pocket_kcal;

    let state_flex = perceive_flexible_pocket_residues(
        &member.pdb_state,
        member.pocket_analysis.center,
        cfg.max_flexible_residues,
    );

    // Evaluate aligned pose + minor translational micro-relaxations (`+-0.45 A`) under soft-core vdW
    let mut best_lig_coords = pre_res.aligned_ligand_coords3d.clone();
    if best_lig_coords.len() != mol.atoms.len() {
        if let Some(first_c) = lig_ensemble_coords.first() {
            best_lig_coords = first_c.clone();
        }
    }
    let micro_shifts: [[f32; 3]; 7] = [
        [0.0, 0.0, 0.0],
        [0.42, 0.0, 0.0],
        [-0.42, 0.0, 0.0],
        [0.0, 0.42, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, 0.0, 0.42],
        [0.0, 0.0, -0.42],
    ];
    let all_prot_coords: Vec<[f32; 3]> = member.pdb_state.protein_atoms.iter().map(|a| a.pos).collect();
    let all_prot_elems: Vec<Element> = member.pdb_state.protein_atoms.iter().map(|a| a.elem).collect();

    let mut best_soft_e = f64::INFINITY;
    let base_pose = best_lig_coords.clone();
    for sh in &micro_shifts {
        let shifted: Vec<[f32; 3]> = base_pose
            .iter()
            .map(|p| [p[0] + sh[0], p[1] + sh[1], p[2] + sh[2]])
            .collect();
        let e_soft = score_ligand_sidechain_pair_kcal(
            mol,
            &shifted,
            &all_prot_coords,
            &all_prot_elems,
            cfg.soft_core_vdw_scale.clamp(0.50, 0.92),
        );
        if e_soft < best_soft_e {
            best_soft_e = e_soft;
            best_lig_coords = shifted;
        }
    }

    // 2. Two-sweep coordinate-descent Dunbrack side-chain rotamer repacking within 5.5 A of ligand
    let mut working_pdb = member.pdb_state.clone();
    let mut chosen_rotamers = vec![DunbrackRotamerId::NativeCrystal; state_flex.len()];

    for _sweep in 0..2 {
        for (ri, res) in state_flex.iter().enumerate() {
            // Check if any movable atom of `res` approaches within 5.6 A of any ligand atom
            let near_ligand = res.native_atom_coords.iter().any(|&rp| {
                best_lig_coords
                    .iter()
                    .any(|&lp| dist3(rp, lp) <= 5.6)
            });
            if !near_ligand {
                continue;
            }

            let sc_elems: Vec<Element> = res
                .movable_atom_indices
                .iter()
                .map(|&ai| working_pdb.protein_atoms[ai].elem)
                .collect();

            let mut best_rot = DunbrackRotamerId::NativeCrystal;
            let mut best_obj = f64::INFINITY;
            let mut best_pts = res.native_atom_coords.clone();

            for rot_id in DunbrackRotamerId::ALL {
                let (dc1, dc2, _, e_rot) = rot_id.parameters_for_residue(&res.res_name);
                let cand_pts = evaluate_rotamer_coords(res, dc1, dc2);
                let e_lig_sc = score_ligand_sidechain_pair_kcal(
                    mol,
                    &best_lig_coords,
                    &cand_pts,
                    &sc_elems,
                    0.96,
                );
                let e_bg_clash =
                    score_sidechain_self_clash_kcal(&working_pdb, &res.movable_atom_indices, &cand_pts);
                let total_obj = e_lig_sc + e_bg_clash + e_rot * 0.55;
                if total_obj < best_obj {
                    best_obj = total_obj;
                    best_rot = rot_id;
                    best_pts = cand_pts;
                }
            }

            chosen_rotamers[ri] = best_rot;
            for (k, &ai) in res.movable_atom_indices.iter().enumerate() {
                working_pdb.protein_atoms[ai].pos = best_pts[k];
            }
        }
    }

    // 3. Re-analyze repacked cavity & score hard-core post-repacked complex
    let repacked_pocket =
        analyze_binding_pocket(&working_pdb, Some(member.pocket_analysis.center), 10.0);
    let post_res = score_ligand_in_pocket(mol, mol_idx, &repacked_pocket);

    let mut moves: Vec<SideChainRepackMove> = Vec::new();
    let mut total_rot_strain = 0.0f64;
    let mut sum_sq_rmsd = 0.0f64;
    let mut moved_atom_cnt = 0usize;
    let mut net_sc_relief = 0.0f64;

    for (ri, res) in state_flex.iter().enumerate() {
        let rot = chosen_rotamers[ri];
        let (dc1, dc2, _, e_rot) = rot.parameters_for_residue(&res.res_name);
        let rep_pts: Vec<[f32; 3]> = res
            .movable_atom_indices
            .iter()
            .map(|&ai| working_pdb.protein_atoms[ai].pos)
            .collect();
        let sc_elems: Vec<Element> = res
            .movable_atom_indices
            .iter()
            .map(|&ai| working_pdb.protein_atoms[ai].elem)
            .collect();

        let e_before =
            score_ligand_sidechain_pair_kcal(mol, &best_lig_coords, &res.native_atom_coords, &sc_elems, 1.0);
        let e_after =
            score_ligand_sidechain_pair_kcal(mol, &best_lig_coords, &rep_pts, &sc_elems, 1.0);
        let delta_int = e_after - e_before;

        let mut res_sq = 0.0f64;
        for k in 0..rep_pts.len() {
            let d = dist3(rep_pts[k], res.native_atom_coords[k]) as f64;
            res_sq += d * d;
        }
        let res_rmsd = (res_sq / (rep_pts.len().max(1) as f64)).sqrt();

        if rot != DunbrackRotamerId::NativeCrystal {
            total_rot_strain += e_rot;
            sum_sq_rmsd += res_sq;
            moved_atom_cnt += rep_pts.len();
            if delta_int < 0.0 {
                net_sc_relief += delta_int;
            }
        }

        moves.push(SideChainRepackMove {
            residue_label: res.residue_label.clone(),
            res_name: res.res_name.clone(),
            selected_rotamer: rot,
            delta_chi1_deg: dc1,
            delta_chi2_deg: dc2,
            sidechain_rmsd_a: res_rmsd,
            rotamer_strain_kcal: if rot == DunbrackRotamerId::NativeCrystal {
                0.0
            } else {
                e_rot
            },
            interaction_delta_kcal: delta_int,
            native_coords: res.native_atom_coords.clone(),
            repacked_coords: rep_pts,
        });
    }

    let repacked_cnt = moves
        .iter()
        .filter(|m| m.selected_rotamer != DunbrackRotamerId::NativeCrystal)
        .count();
    let total_sc_rmsd = if moved_atom_cnt > 0 {
        (sum_sq_rmsd / (moved_atom_cnt as f64)).sqrt()
    } else {
        0.0
    };

    // Combine full pocket Vina/WaterMap score with direct clash-relief gain on repacked side chains
    let post_int_dg = post_res
        .total_dg_pocket_kcal
        .min(pre_dg + (net_sc_relief * 0.58).clamp(-3.60, 0.0));

    let effective_total_dg =
        post_int_dg + member.apo_reorg_penalty_kcal + 0.55 * total_rot_strain;
    let sherman_ifd =
        post_int_dg + SHERMAN_IFD_REORG_SCALE * (member.apo_reorg_penalty_kcal + total_rot_strain);

    EnsembleStateDockingScore {
        state_idx: member.state_idx,
        state_code: member.state_code,
        state_title: member.state_title,
        apo_reorg_penalty_kcal: member.apo_reorg_penalty_kcal,
        apo_weight: member.apo_boltzmann_weight,
        pre_repack_dg_kcal: pre_dg,
        post_repack_interaction_dg_kcal: post_int_dg,
        rotamer_strain_kcal: total_rot_strain,
        effective_total_dg_kcal: effective_total_dg,
        sherman_ifd_score_kcal: sherman_ifd,
        holo_boltzmann_weight: 0.0,
        repacked_residue_count: repacked_cnt,
        repacked_sidechain_rmsd_a: total_sc_rmsd,
        aligned_ligand_coords3d: if post_res.aligned_ligand_coords3d.is_empty() {
            best_lig_coords
        } else {
            post_res.aligned_ligand_coords3d
        },
        repacked_moves: moves,
        repacked_shell_atoms: repacked_pocket.shell_atoms,
    }
}

/// Scores a single ligand across a 5-state flexible-sidechain receptor ensemble and computes the
/// `Fischer-Shoichet` reorganization-penalized Boltzmann ensemble free energy `dG_Boltzmann`.
pub fn score_ligand_induced_fit_ensemble(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    cfg: &InducedFitConfig,
) -> InducedFitEnsembleReport {
    let (base_pdb, center_ovr, target_name) = load_induced_fit_base_pdb(cfg);
    let init_pocket = analyze_binding_pocket(&base_pdb, center_ovr, 10.0);
    let flex_res = perceive_flexible_pocket_residues(
        &base_pdb,
        init_pocket.center,
        cfg.max_flexible_residues,
    );
    let (holo_pocket, ensemble_members) =
        build_receptor_ensemble(&base_pdb, center_ovr, &flex_res, cfg.temperature_k);

    score_ligand_with_prebuilt_ensemble(
        mol,
        ds,
        cfg,
        &base_pdb.pdb_id,
        &target_name,
        &holo_pocket,
        &flex_res,
        &ensemble_members,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn score_ligand_with_prebuilt_ensemble(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    cfg: &InducedFitConfig,
    pdb_id: &str,
    target_name: &str,
    holo_pocket: &PocketCavityAnalysis,
    flex_res: &[FlexiblePocketResidue],
    ensemble_members: &[ReceptorEnsembleMember],
) -> InducedFitEnsembleReport {
    let rt = (0.001_987_204 * cfg.temperature_k.clamp(250.0, 340.0)).max(0.45);
    let mol_idx = ds.and_then(|d| d.mols.iter().position(|dm| dm.id == mol.id));
    let rigid_holo = score_ligand_in_pocket(mol, mol_idx, holo_pocket);
    let rigid_holo_dg = rigid_holo.total_dg_pocket_kcal;

    let conf_ens = generate_3d_conformer_ensemble(
        mol,
        &ConformerGenConfig {
            num_trial_conformers: 6,
            max_kept_conformers: 4,
            mmff_max_iters: 80,
            rmsd_prune_cutoff: 0.40,
            energy_window_kcal: 12.0,
            random_seed: 0x1FD3_3300,
        },
    );
    let lig_coords_list: Vec<Vec<[f32; 3]>> =
        conf_ens.conformers.into_iter().map(|c| c.coords3d).collect();

    let mut state_scores: Vec<EnsembleStateDockingScore> = ensemble_members
        .par_iter()
        .map(|mem| dock_and_repack_in_state(mol, ds, &lig_coords_list, mem, cfg))
        .collect();

    // Numerically stable log-sum-exp Boltzmann ensemble free energy:
    // dG_ens = -RT * ln( sum_k exp(-dG_eff,k / RT) / sum_k exp(-dG_reorg,k / RT) )
    let min_eff = state_scores
        .iter()
        .map(|s| s.effective_total_dg_kcal)
        .fold(f64::INFINITY, f64::min);
    let sum_bound_shifted: f64 = state_scores
        .iter()
        .map(|s| (-(s.effective_total_dg_kcal - min_eff) / rt).exp())
        .sum::<f64>()
        .max(1e-15);
    let ln_z_bound = -(min_eff / rt) + sum_bound_shifted.ln();

    let sum_z_apo: f64 = state_scores
        .iter()
        .map(|s| (-s.apo_reorg_penalty_kcal / rt).exp())
        .sum::<f64>()
        .max(1e-15);
    let ln_z_apo = sum_z_apo.ln();

    let boltzmann_dg = -rt * (ln_z_bound - ln_z_apo);

    // Populate normalized holo Boltzmann populations & KL population shift
    let mut best_sherman = f64::INFINITY;
    let mut dom_idx = 0usize;
    let mut dom_w = -1.0f64;
    let mut kl_nats = 0.0f64;

    for s in &mut state_scores {
        let w_h = (-(s.effective_total_dg_kcal - min_eff) / rt).exp() / sum_bound_shifted;
        s.holo_boltzmann_weight = w_h;
        if s.sherman_ifd_score_kcal < best_sherman {
            best_sherman = s.sherman_ifd_score_kcal;
        }
        if w_h > dom_w {
            dom_w = w_h;
            dom_idx = s.state_idx;
        }
        if w_h > 1e-9 && s.apo_weight > 1e-9 {
            kl_nats += w_h * (w_h / s.apo_weight).ln();
        }
    }
    let kl_clean = kl_nats.max(0.0);
    let rec_conf_entropy_cost = rt * kl_clean;
    let ddg_rescue = boltzmann_dg - rigid_holo_dg;

    let mechanism = if boltzmann_dg > -5.40 {
        BindingMechanismRegime::HighReorgNonBinder
    } else if ddg_rescue <= -1.45 || (rigid_holo.clash_atom_count > 0 && ddg_rescue <= -0.85) {
        BindingMechanismRegime::InducedFitClashRescue
    } else if dom_idx != 0 || ddg_rescue <= -0.55 || kl_clean >= 0.35 {
        BindingMechanismRegime::ConformationalSelection
    } else {
        BindingMechanismRegime::LockAndKeyGroundState
    };

    let phys = evaluate_physchem(mol, 0);
    let pred_pkd = (-boltzmann_dg / RT_LN10_KCAL).clamp(2.5, 12.5);
    let ens_le = -boltzmann_dg / (phys.heavy_atoms.max(1) as f64);
    let ens_lle = pred_pkd - phys.clogp;
    let dom_code = state_scores
        .get(dom_idx)
        .map(|s| s.state_code)
        .unwrap_or("S1-HOLO-GROUND");

    InducedFitEnsembleReport {
        preset: cfg.preset,
        pdb_id: pdb_id.to_string(),
        target_name: target_name.to_string(),
        compound_id: mol.id.clone(),
        compound_name: mol.name.clone(),
        canonical_smiles: mol.smiles.clone(),
        rigid_holo_dg_kcal: rigid_holo_dg,
        rigid_holo_clash_atoms: rigid_holo.clash_atom_count,
        boltzmann_ensemble_dg_kcal: boltzmann_dg,
        best_sherman_ifd_score_kcal: best_sherman,
        ddg_induced_fit_rescue_kcal: ddg_rescue,
        ensemble_predicted_pkd: pred_pkd,
        ensemble_le: ens_le,
        ensemble_lle: ens_lle,
        dominant_state_idx: dom_idx,
        dominant_state_code: dom_code,
        dominant_holo_occupancy_pct: (dom_w * 100.0).clamp(0.0, 100.0),
        kl_population_shift_nats: kl_clean,
        receptor_conf_entropy_cost_kcal: rec_conf_entropy_cost,
        mechanism,
        flexible_residues: flex_res.to_vec(),
        state_scores,
        mol: mol.clone(),
    }
}

/// Screens an entire `DatasetAnalysis` in parallel against the prebuilt 5-state flexible receptor ensemble.
pub fn screen_dataset_induced_fit_ensemble(
    ds: &DatasetAnalysis,
    cfg: &InducedFitConfig,
) -> Vec<DatasetInducedFitRow> {
    let (base_pdb, center_ovr, target_name) = load_induced_fit_base_pdb(cfg);
    let init_pocket = analyze_binding_pocket(&base_pdb, center_ovr, 10.0);
    let flex_res = perceive_flexible_pocket_residues(
        &base_pdb,
        init_pocket.center,
        cfg.max_flexible_residues,
    );
    let (holo_pocket, ensemble_members) =
        build_receptor_ensemble(&base_pdb, center_ovr, &flex_res, cfg.temperature_k);

    let mut rows: Vec<DatasetInducedFitRow> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(mi, m)| {
            let rep = score_ligand_with_prebuilt_ensemble(
                m,
                Some(ds),
                cfg,
                &base_pdb.pdb_id,
                &target_name,
                &holo_pocket,
                &flex_res,
                &ensemble_members,
            );
            let dom_repacked = rep
                .state_scores
                .get(rep.dominant_state_idx)
                .map(|s| s.repacked_residue_count)
                .unwrap_or(0);
            DatasetInducedFitRow {
                rank: 0,
                mol_idx: mi,
                compound_id: m.id.clone(),
                compound_name: m.name.clone(),
                mw: ds.profiles[mi].mw,
                clogp: ds.profiles[mi].clogp,
                observed_pact: m.pact,
                rigid_holo_dg_kcal: rep.rigid_holo_dg_kcal,
                boltzmann_ensemble_dg_kcal: rep.boltzmann_ensemble_dg_kcal,
                ddg_induced_fit_rescue_kcal: rep.ddg_induced_fit_rescue_kcal,
                sherman_ifd_score_kcal: rep.best_sherman_ifd_score_kcal,
                ensemble_predicted_pkd: rep.ensemble_predicted_pkd,
                dominant_state_code: rep.dominant_state_code,
                dominant_occupancy_pct: rep.dominant_holo_occupancy_pct,
                repacked_residues: dom_repacked,
                mechanism: rep.mechanism,
                canonical_smiles: m.smiles.clone(),
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        a.boltzmann_ensemble_dg_kcal
            .partial_cmp(&b.boltzmann_ensemble_dg_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Exports dataset-wide Induced-Fit & Boltzmann Ensemble screening rows to `.csv` with summary counts.
pub fn export_induced_fit_ensemble_csv(rows: &[DatasetInducedFitRow]) -> String {
    let mut out = String::from(
        "rank,compound_id,compound_name,mw,clogp,observed_pact,rigid_holo_dg_kcal,boltzmann_ensemble_dg_kcal,ddg_induced_fit_rescue_kcal,sherman_ifd_score_kcal,ensemble_predicted_pkd,dominant_state,dominant_occupancy_pct,repacked_residues,mechanism,smiles\n",
    );
    let mut rescue_cnt = 0usize;
    let mut conf_sel_cnt = 0usize;
    for r in rows {
        if r.mechanism == BindingMechanismRegime::InducedFitClashRescue {
            rescue_cnt += 1;
        } else if r.mechanism == BindingMechanismRegime::ConformationalSelection {
            conf_sel_cnt += 1;
        }
        out.push_str(&format!(
            "{},{},\"{}\",{:.1},{:.2},{:.2},{:+.2},{:+.2},{:+.2},{:+.2},{:.2},{},{:.1},{},\"{}\",{}\n",
            r.rank,
            r.compound_id,
            r.compound_name,
            r.mw,
            r.clogp,
            r.observed_pact,
            r.rigid_holo_dg_kcal,
            r.boltzmann_ensemble_dg_kcal,
            r.ddg_induced_fit_rescue_kcal,
            r.sherman_ifd_score_kcal,
            r.ensemble_predicted_pkd,
            r.dominant_state_code,
            r.dominant_occupancy_pct,
            r.repacked_residues,
            r.mechanism.short_label(),
            r.canonical_smiles
        ));
    }
    out.push_str(&format!(
        "# SUMMARY,total_compounds={},induced_fit_rescued={},conformational_selection={}\n",
        rows.len(),
        rescue_cnt,
        conf_sel_cnt
    ));
    out
}

/// Exports per-state & per-residue `Dunbrack` side-chain rotamer repacking telemetry for a single ligand to `.csv`.
pub fn export_repacked_rotamers_csv(rep: &InducedFitEnsembleReport) -> String {
    let mut out = String::from(
        "state_code,apo_reorg_kcal,apo_weight,holo_weight,effective_state_dg_kcal,sherman_ifd_kcal,residue,selected_rotamer,delta_chi1_deg,delta_chi2_deg,sidechain_rmsd_a,rotamer_strain_kcal,interaction_delta_kcal\n",
    );
    for s in &rep.state_scores {
        for mv in &s.repacked_moves {
            out.push_str(&format!(
                "{},{:+.2},{:.4},{:.4},{:+.2},{:+.2},{},\"{}\",{:+.1},{:+.1},{:.3},{:+.2},{:+.2}\n",
                s.state_code,
                s.apo_reorg_penalty_kcal,
                s.apo_weight,
                s.holo_boltzmann_weight,
                s.effective_total_dg_kcal,
                s.sherman_ifd_score_kcal,
                mv.residue_label,
                mv.selected_rotamer.code(),
                mv.delta_chi1_deg,
                mv.delta_chi2_deg,
                mv.sidechain_rmsd_a,
                mv.rotamer_strain_kcal,
                mv.interaction_delta_kcal
            ));
        }
    }
    out
}

/// Exports all 5 ensemble-docked 3D ligand poses with Boltzmann state weights & `IFDScore` SD tags to `.sdf`.
pub fn format_induced_fit_ensemble_sdf(rep: &InducedFitEnsembleReport) -> String {
    let mut sdf = String::new();
    let chg = compute_gasteiger_charges(&rep.mol);
    for s in &rep.state_scores {
        sdf.push_str(&format_3d_conformer_sdf(
            &rep.mol,
            &s.aligned_ligand_coords3d,
            &chg,
            s.effective_total_dg_kcal,
            &[
                ("RECEPTOR_PDB", rep.pdb_id.clone()),
                ("ENSEMBLE_STATE", s.state_code.to_string()),
                (
                    "HOLO_BOLTZMANN_WEIGHT",
                    format!("{:.4}", s.holo_boltzmann_weight),
                ),
                (
                    "EFFECTIVE_STATE_DG_KCAL",
                    format!("{:+.2}", s.effective_total_dg_kcal),
                ),
                (
                    "SHERMAN_IFD_SCORE_KCAL",
                    format!("{:+.2}", s.sherman_ifd_score_kcal),
                ),
                (
                    "BOLTZMANN_ENSEMBLE_DG_KCAL",
                    format!("{:+.2}", rep.boltzmann_ensemble_dg_kcal),
                ),
                (
                    "DDG_INDUCED_FIT_RESCUE_KCAL",
                    format!("{:+.2}", rep.ddg_induced_fit_rescue_kcal),
                ),
            ],
        ));
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol_io::parse_smiles;

    #[test]
    fn test_egfr_1m17_induced_fit_ensemble_boltzmann_partition_normalization() {
        let mut lapatinib_like = parse_smiles(
            "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1"
        )
        .expect("Valid bulky back-pocket kinase inhibitor SMILES");
        lapatinib_like.id = "IFD-BULKY-01".to_string();
        lapatinib_like.name = "Lapatinib-BackPocket-Probe".to_string();

        let cfg = InducedFitConfig::default();
        let rep = score_ligand_induced_fit_ensemble(&lapatinib_like, None, &cfg);
        assert_eq!(rep.state_scores.len(), 5);
        assert!(!rep.flexible_residues.is_empty());

        let sum_apo: f64 = rep.state_scores.iter().map(|s| s.apo_weight).sum();
        let sum_holo: f64 = rep
            .state_scores
            .iter()
            .map(|s| s.holo_boltzmann_weight)
            .sum();
        assert!((sum_apo - 1.0).abs() < 1e-6);
        assert!((sum_holo - 1.0).abs() < 1e-6);
        assert!(rep.boltzmann_ensemble_dg_kcal.is_finite());
        assert!(export_repacked_rotamers_csv(&rep).contains("selected_rotamer"));
        assert!(format_induced_fit_ensemble_sdf(&rep).contains("HOLO_BOLTZMANN_WEIGHT"));
    }
}
