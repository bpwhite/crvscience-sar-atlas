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

//! 3D Molecular Electrostatic Potential (`ESP`) Surface Mapper, Off-Nucleus `XED`-Style Atomic
//! Multipole Builder (`Sigma-Hole` Crowns, Lone-Pair Lobes & Aromatic `Pi`-Quadrupoles),
//! `Bauer-Mackey` Protein-Ligand Electrostatic Complementarity (`EC`, `EC_r`, `EC_rho`),
//! Directional `C-X...B` Halogen-Bond & `R-S...O/N` Chalcogen-Bond Scanner, `Wheeler-Houk` /
//! `Gallivan-Dougherty` `Pi`-Stacking & `Cation-Pi` Profiler, and `ESP`-Guided Bioisosteric
//! Designer (`electrostatics_esp.rs`, `v3.8.0`).
//!
//! Primary References:
//! - Bauer & Mackey (2019), *J. Med. Chem.* 62(6), 3036-3050 ("Electrostatic Complementarity as a
//!   Fast and Effective Tool to Optimize Binding and Selectivity of Protein-Ligand Complexes" —
//!   Connolly surface integral `EC`, Pearson `EC_r`, and Spearman `EC_rho`:
//!   `https://pubmed.ncbi.nlm.nih.gov/30807144/`) & Milanetti et al. (2023)
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10290103/`)
//! - Cavallo, Metrangolo, Milani, Pilati, Priimagi, Resnati & Terraneo (2016), *Chem. Rev.* 116,
//!   2478-2601 ("The Halogen Bond": `https://pmc.ncbi.nlm.nih.gov/articles/PMC4768247/`) &
//!   Kohn-Sham MO/EDA protein-ligand halogen-bond analysis
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC7093837/`)
//! - Adhav, Shelke, Balanarayan & Saikrishnan (2023) ("Sulfur-mediated chalcogen versus hydrogen
//!   bonds in proteins: a see-saw effect in the conformational space" — bivalent `S` `sigma*`-hole
//!   chalcogen bonds in thiophene/thiazole/thioether scaffolds & `1,4`/`1,5` `S...O` locks:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC10411326/`)
//! - Neel, Hilton, Sigman & Toste (2017) (`Wheeler-Houk` substituent-tuned `pi-pi` direct local
//!   dipole model: `https://pmc.ncbi.nlm.nih.gov/articles/PMC5907483/`), Carter-Fenk & Herbert
//!   (2020) (slip-stacked aromatic dispersion/quadrupole geometry:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8159364/`), & Cation-`pi` recognition in structural
//!   biology (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8338773/`).

use crate::alerts::scan_structural_alerts;
use crate::allosteric_pocket::curated_allosteric_pdb_text;
use crate::conformer3d::compute_gasteiger_charges;
use crate::foldit_sculpt::curated_vhl_5nvx_pdb_text;
use crate::fragment_grower::RT_LN10_KCAL;
use crate::mol::{Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    ParsedPdbStructure, PocketCavityAnalysis,
};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::watermap_solvation::curated_factor_xa_1ezq_pdb_text;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Coulomb conversion factor `e^2 / (4 * pi * eps_0)` in `kcal * A / (mol * e^2)`.
pub const COULOMB_KCAL_A_PER_E2: f64 = 332.0637;

/// Capped electrostatic potential ceiling (`kcal/(mol*e)`) for `Bauer-Mackey` normalized `EC` surface integration.
pub const BAUER_MACKEY_ESP_CAP_KCAL_E: f64 = 35.0;

/// Curated benchmark complexes showcasing halogen bonding, sulfur chalcogen locks, `pi-stacking`, `cation-pi`, and `ESP` complementarity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EspTargetPreset {
    /// `PDB 1EZQ` (`Factor Xa`): Classic `S1 Tyr228/Gly218` aryl-`Cl`/`Br` `sigma-hole` halogen bond + `S4 Tyr99/Phe174/Trp215` aromatic `pi`/`cation-pi` box!
    FactorXa1ezqHalogenAndS4PiBox,
    /// `PDB 5NVX` (`VHL E3 Ligase`): `1,3-Thiazole` `1,5` intramolecular `S...O=C` chalcogen conformational lock + `Tyr98/His115` `ESP` match!
    Vhl5nvxThiazoleChalcogenLock,
    /// `PDB 1M17` (`EGFR Kinase`): `3-Chloro-4-fluoroaniline` back-pocket `C-Cl...O=C(Leu788)` `sigma-hole` + `Met793` hinge lone-pair `ESP` complementarity!
    Egfr1m17AnilineHalogenAndHingeEsp,
    /// `PDB 6CM4` (`DRD2`): Protonated piperidine `N+` `Cation-Pi` cage (`Trp386/Phe389/Phe390`) + `Asp114` ionic `ESP` complementarity!
    Drd26cm4CationPiAromaticCage,
    /// `PDB 6OIM` (`KRAS G12C`): Fluorinated quinazolinone + `His95/Tyr96` `Wheeler-Houk` `pi-stacking` & Switch-II `ESP` groove!
    Kras6oimFluorinatedPiGroove,
    /// `PDB 3LN1` (`COX-2`): `Arg513`/`His90` sulfonamide polar side-pocket & hydrophobic `CF3`-pyrazole channel `ESP` match!
    Cox23ln1SulfonamideAndChannelEsp,
    /// User-supplied `.pdb` receptor for custom 3D `ESP` complementarity & `sigma-hole` mapping.
    CustomPdbEspComplementarity,
}

impl EspTargetPreset {
    pub const ALL: [Self; 7] = [
        Self::Egfr1m17AnilineHalogenAndHingeEsp,
        Self::Drd26cm4CationPiAromaticCage,
        Self::Cox23ln1SulfonamideAndChannelEsp,
        Self::FactorXa1ezqHalogenAndS4PiBox,
        Self::Vhl5nvxThiazoleChalcogenLock,
        Self::Kras6oimFluorinatedPiGroove,
        Self::CustomPdbEspComplementarity,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::FactorXa1ezqHalogenAndS4PiBox => "ESP-1EZQ-FXA-XB-PI",
            Self::Vhl5nvxThiazoleChalcogenLock => "ESP-5NVX-VHL-CHB",
            Self::Egfr1m17AnilineHalogenAndHingeEsp => "ESP-1M17-EGFR-ESP",
            Self::Drd26cm4CationPiAromaticCage => "ESP-6CM4-DRD2-CATPI",
            Self::Cox23ln1SulfonamideAndChannelEsp => "ESP-3LN1-COX2-SULF",
            Self::Kras6oimFluorinatedPiGroove => "ESP-6OIM-KRAS-PIGRV",
            Self::CustomPdbEspComplementarity => "ESP-CUSTOM-PDB",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::FactorXa1ezqHalogenAndS4PiBox => "FXa 1EZQ (Halogen+S4 Pi)",
            Self::Vhl5nvxThiazoleChalcogenLock => "VHL 5NVX (S..O Chalcogen)",
            Self::Egfr1m17AnilineHalogenAndHingeEsp => "EGFR 1M17 (Cl-Hole+Hinge)",
            Self::Drd26cm4CationPiAromaticCage => "DRD2 6CM4 (Cation-Pi)",
            Self::Cox23ln1SulfonamideAndChannelEsp => "COX-2 3LN1 (Sulf+Pi)",
            Self::Kras6oimFluorinatedPiGroove => "KRAS 6OIM (F-Arene Pi)",
            Self::CustomPdbEspComplementarity => "Custom .PDB ESP",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::FactorXa1ezqHalogenAndS4PiBox => {
                "Factor Xa (1EZQ): S1 Tyr228/Gly218 C-X Sigma-Hole & S4 Tyr99/Phe174/Trp215 Pi/Cation-Pi Box"
            }
            Self::Vhl5nvxThiazoleChalcogenLock => {
                "VHL E3 Ligase (5NVX): Thiazole S..O=C Sigma* Chalcogen Lock & Tyr98/His115 ESP Interface"
            }
            Self::Egfr1m17AnilineHalogenAndHingeEsp => {
                "EGFR Kinase (1M17): Back-Pocket C-Cl/Br Sigma-Hole & Met793 Hinge Lone-Pair ESP Match"
            }
            Self::Drd26cm4CationPiAromaticCage => {
                "Dopamine D2 (6CM4): Trp386/Phe389/Phe390 Cation-Pi Cage & Asp114 Electrostatic Match"
            }
            Self::Cox23ln1SulfonamideAndChannelEsp => {
                "COX-2 (3LN1): Arg513/His90 Sulfonamide Side-Pocket & Hydrophobic Pyrazole Channel ESP"
            }
            Self::Kras6oimFluorinatedPiGroove => {
                "KRAS G12C (6OIM): Wheeler-Houk Fluorinated Arene His95/Tyr96 Pi-Stack & Switch-II ESP"
            }
            Self::CustomPdbEspComplementarity => {
                "Custom .PDB Receptor: 3D Connolly Surface ESP Complementarity & Sigma-Hole Scanner"
            }
        }
    }
}

/// Kind of off-nucleus `XED`-style multipole pseudoatom generated on the 3D ligand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipoleSiteKind {
    /// Positive electrophilic crown (`+q_sigma`) along `C -> X` (`X = Cl, Br, I`) extension (`180 deg`).
    HalogenSigmaHoleCrown,
    /// Positive `sigma*` lobe (`+q_chalc`) along `R -> S` or `R -> Se` covalent bond extension.
    ChalcogenSigmaStarLobe,
    /// Negative directional `sp2`/`sp` lone-pair lobe (`-q_lp`) on `=O`, `#N`, or pyridine `n:`.
    DirectionalLonePairLobe,
    /// Above/below-plane aromatic `pi`-electron quadrupole pole (`-q_pi` or inverted `+q_pi` on perfluoroarenes).
    AromaticPiQuadrupolePole,
}

impl MultipoleSiteKind {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::HalogenSigmaHoleCrown => "Halogen +Sigma-Hole",
            Self::ChalcogenSigmaStarLobe => "Chalcogen +Sigma* Lobe",
            Self::DirectionalLonePairLobe => "Lone-Pair -LP Lobe",
            Self::AromaticPiQuadrupolePole => "Aromatic Pi-Pole",
        }
    }
}

/// Explicit off-center multipole pseudoatom (`XED` formalism) attached to a parent ligand atom.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultipolePseudoatom {
    pub parent_atom_idx: usize,
    pub parent_label: String,
    pub kind: MultipoleSiteKind,
    pub pos: [f32; 3],
    pub axis_unit: [f32; 3],
    /// Charge magnitude in elementary charge units (`e`).
    pub charge_e: f64,
    /// Surface electrostatic potential extremum `V_s,max` or `V_s,min` (`kcal/(mol*e)`).
    pub vs_extremum_kcal_e: f64,
}

/// Sampled Connolly / Solvent-Accessible Surface (`SAS`) vertex carrying paired ligand & receptor `ESP`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnollyEspVertex {
    pub pos: [f32; 3],
    pub parent_atom_idx: usize,
    /// Ligand electrostatic potential `V_lig(s)` (`kcal/(mol*e)`).
    pub v_lig_kcal_e: f64,
    /// Receptor pocket electrostatic potential `V_prot(s)` (`kcal/(mol*e)`).
    pub v_prot_kcal_e: f64,
    /// Local `Bauer-Mackey` complementarity score `c(s) in [-1.0, +1.0]` (`+1` = matched opposite potentials, `-1` = same-sign clash).
    pub local_ec_score: f64,
}

/// Category of anisotropic multipole interaction detected in the 3D complex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipoleContactKind {
    /// Intermolecular `C-X...O/N` halogen bond (`theta(C-X...B) >= 140 deg`, `X = Cl, Br, I`).
    HalogenBondSigmaHole,
    /// Intermolecular `R-S/Se...O/N` chalcogen bond (`theta(R-S...B) >= 145 deg`).
    ChalcogenBondIntermolecular,
    /// Intramolecular `1,4` or `1,5` `R-S...O=C / N` conformational chalcogen lock (`Beno & Meanwell 2015`).
    ChalcogenLockIntramolecular,
    /// Parallel-displaced (`slip-stacked`) aromatic `pi-pi` contact (`plane angle <= 30 deg`).
    PiStackParallelDisplaced,
    /// Edge-to-face (`T-shaped` `C-H...pi`) aromatic multipole contact (`plane angle 55..90 deg`).
    PiStackTShapedEdgeToFace,
    /// `Cation-Pi` interaction (`Arg/Lys/N+ <-> Trp/Tyr/Phe/Arene`, `Gallivan & Dougherty`).
    CationPiRecognition,
    /// Unfavorable same-sign `ESP` surface clash (`V_lig * V_prot > 0` at short range).
    ElectrostaticSurfaceClash,
}

impl MultipoleContactKind {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::HalogenBondSigmaHole => "Halogen Bond (C-X..B)",
            Self::ChalcogenBondIntermolecular => "Chalcogen Bond (S..O)",
            Self::ChalcogenLockIntramolecular => "1,5 S..O Chalcogen Lock",
            Self::PiStackParallelDisplaced => "Pi-Pi (Slip-Stacked)",
            Self::PiStackTShapedEdgeToFace => "Pi-Pi (T-Shaped C-H..Pi)",
            Self::CationPiRecognition => "Cation-Pi Multipole",
            Self::ElectrostaticSurfaceClash => "ESP Same-Sign Clash",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HalogenBondSigmaHole => [56, 189, 248],
            Self::ChalcogenBondIntermolecular => [250, 204, 21],
            Self::ChalcogenLockIntramolecular => [251, 146, 60],
            Self::PiStackParallelDisplaced => [192, 132, 252],
            Self::PiStackTShapedEdgeToFace => [167, 139, 250],
            Self::CationPiRecognition => [52, 211, 153],
            Self::ElectrostaticSurfaceClash => [244, 63, 94],
        }
    }
}

/// Geometric and energetic details of one detected multipole interaction or `ESP` clash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultipoleContact {
    pub kind: MultipoleContactKind,
    pub ligand_site_label: String,
    pub partner_label: String,
    pub ligand_pos: [f32; 3],
    pub partner_pos: [f32; 3],
    pub distance_a: f32,
    /// `theta(C-X...B)` / `theta(R-S...B)` / ring-plane interplanar angle (`degrees`).
    pub angle_deg: f32,
    /// Secondary geometric parameter (e.g., slip offset `A` for `pi-pi` or `V_s,max` for `sigma-hole`).
    pub aux_metric: f64,
    pub energy_kcal: f64,
    pub rationale: String,
}

/// Actionable `ESP` / `Sigma-Hole` / `Wheeler-Houk` bioisosteric analog suggestion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EspMultipoleDesign {
    pub rank: usize,
    pub design_id: String,
    pub modification_label: String,
    #[serde(skip_deserializing)]
    pub mechanism: &'static str,
    pub target_partner: String,
    pub designed_smiles: String,
    pub baseline_ec_r: f64,
    pub predicted_ec_r: f64,
    pub delta_ec_r: f64,
    pub ddg_multipole_kcal: f64,
    pub predicted_pact: f64,
    pub mw: f64,
    pub clogp: f64,
    pub sa_score: f64,
    pub pains_count: usize,
    pub design_score: f64,
}

/// Configuration for 3D `ESP`, `sigma-hole`, and `pi`-multipole complementarity analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElectrostaticsEspConfig {
    pub preset: EspTargetPreset,
    pub custom_pdb_path: Option<String>,
    /// Effective interfacial interior dielectric constant (`2.0..=12.0`, default `4.0` for protein-ligand cleft).
    pub interface_dielectric: f64,
    /// Include explicit off-nucleus `XED` `sigma-hole`, lone-pair & `pi`-quadrupole pseudoatoms (disable to compare naive atom-centered charges!).
    pub enable_xed_multipoles: bool,
    /// Enumerate bioisosteric halogen / chalcogen / heteroaryl `ESP`-tuning analogs.
    pub generate_esp_designs: bool,
}

impl Default for ElectrostaticsEspConfig {
    fn default() -> Self {
        Self {
            preset: EspTargetPreset::Egfr1m17AnilineHalogenAndHingeEsp,
            custom_pdb_path: None,
            interface_dielectric: 4.0,
            enable_xed_multipoles: true,
            generate_esp_designs: true,
        }
    }
}

/// Complete 3D `ESP`, `Sigma-Hole`, Chalcogen & `Pi`-Multipole Complementarity Report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElectrostaticsEspReport {
    pub preset: EspTargetPreset,
    pub pdb_id: String,
    pub target_name: String,
    pub pocket_center: [f32; 3],
    pub compound_id: String,
    pub compound_name: String,
    pub query_smiles: String,
    pub mol: Molecule,
    /// `Bauer-Mackey` normalized Connolly surface complementarity integral `EC in [-1.0, +1.0]`.
    pub bauer_mackey_ec_integral: f64,
    /// `Bauer-Mackey` negated Pearson surface potential correlation `EC_r in [-1.0, +1.0]`.
    pub bauer_mackey_ec_pearson_r: f64,
    /// `Bauer-Mackey` negated Spearman rank correlation `EC_rho in [-1.0, +1.0]`.
    pub bauer_mackey_ec_spearman_rho: f64,
    pub esp_favorable_surface_pct: f64,
    pub esp_clash_surface_pct: f64,
    pub ligand_vs_max_kcal_e: f64,
    pub ligand_vs_min_kcal_e: f64,
    pub halogen_bond_count: usize,
    pub chalcogen_bond_and_lock_count: usize,
    pub pi_and_cation_pi_count: usize,
    pub esp_clash_count: usize,
    pub dg_halogen_chalcogen_kcal: f64,
    pub dg_pi_multipole_kcal: f64,
    pub dg_surface_esp_kcal: f64,
    pub total_multipole_dg_kcal: f64,
    pub esp_refined_pocket_dg_kcal: f64,
    pub esp_predicted_pkd: f64,
    pub aligned_ligand_coords3d: Vec<[f32; 3]>,
    pub pocket_shell_atoms: Vec<([f32; 3], String, Element)>,
    pub multipole_pseudoatoms: Vec<MultipolePseudoatom>,
    pub connolly_vertices: Vec<ConnollyEspVertex>,
    pub contacts: Vec<MultipoleContact>,
    pub esp_designs: Vec<EspMultipoleDesign>,
}

/// Dataset-wide 3D `ESP` & `Sigma-Hole` complementarity screening row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetEspScreenRow {
    pub rank: usize,
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub observed_pact: f64,
    pub ec_integral: f64,
    pub ec_pearson_r: f64,
    pub halogen_bonds: usize,
    pub chalcogen_contacts: usize,
    pub pi_contacts: usize,
    pub esp_clashes: usize,
    pub total_multipole_dg_kcal: f64,
    pub esp_refined_pocket_dg_kcal: f64,
    pub top_esp_design: String,
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-5 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn vdw_rad(elem: Element) -> f32 {
    match elem {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::F => 1.47,
        Element::P => 1.80,
        Element::S => 1.80,
        Element::Cl => 1.75,
        Element::Br => 1.85,
        Element::I => 1.98,
        _ => 1.75,
    }
}

/// Loads the target `ParsedPdbStructure`, optional pocket center override, and target name for `cfg`.
pub fn load_esp_target_pdb(
    cfg: &ElectrostaticsEspConfig,
) -> (ParsedPdbStructure, Option<[f32; 3]>, String) {
    if cfg.preset == EspTargetPreset::CustomPdbEspComplementarity {
        if let Some(ref p) = cfg.custom_pdb_path {
            let t = p.trim();
            if !t.is_empty() {
                if let Ok(raw) = std::fs::read_to_string(t) {
                    if let Ok(parsed) = parse_pdb_structure(&raw) {
                        let lbl = parsed.pdb_id.clone();
                        return (parsed, None, lbl);
                    }
                }
            }
        }
    }
    match cfg.preset {
        EspTargetPreset::FactorXa1ezqHalogenAndS4PiBox
        | EspTargetPreset::CustomPdbEspComplementarity => (
            parse_pdb_structure(curated_factor_xa_1ezq_pdb_text()).expect("Valid 1EZQ PDB"),
            None,
            "Factor Xa".to_string(),
        ),
        EspTargetPreset::Vhl5nvxThiazoleChalcogenLock => (
            parse_pdb_structure(curated_vhl_5nvx_pdb_text()).expect("Valid 5NVX PDB"),
            None,
            "VHL E3 Ligase".to_string(),
        ),
        EspTargetPreset::Egfr1m17AnilineHalogenAndHingeEsp => (
            parse_pdb_structure(curated_pocket_pdb_text(0)).expect("Valid 1M17 PDB"),
            None,
            "EGFR".to_string(),
        ),
        EspTargetPreset::Drd26cm4CationPiAromaticCage => (
            parse_pdb_structure(curated_pocket_pdb_text(1)).expect("Valid 6CM4 PDB"),
            None,
            "DRD2".to_string(),
        ),
        EspTargetPreset::Cox23ln1SulfonamideAndChannelEsp => (
            parse_pdb_structure(curated_pocket_pdb_text(2)).expect("Valid 3LN1 PDB"),
            None,
            "COX-2".to_string(),
        ),
        EspTargetPreset::Kras6oimFluorinatedPiGroove => (
            parse_pdb_structure(curated_allosteric_pdb_text(1)).expect("Valid 6OIM PDB"),
            Some([12.90, 5.10, 8.20]),
            "KRAS-G12C".to_string(),
        ),
    }
}

/// 14-point Fibonacci sphere unit directions for uniform Connolly surface vertex generation around each heavy atom.
const FIBONACCI_14_DIRS: [[f32; 3]; 14] = [
    [0.0000, 1.0000, 0.0000],
    [0.4835, 0.8462, 0.2245],
    [-0.6599, 0.6923, 0.2917],
    [0.2498, 0.5385, -0.8047],
    [0.4496, 0.3846, 0.8065],
    [-0.8961, 0.2308, -0.3792],
    [0.8655, 0.0769, -0.4950],
    [-0.3112, -0.0769, 0.9472],
    [-0.4706, -0.2308, -0.8516],
    [0.8920, -0.3846, 0.2369],
    [-0.7850, -0.5385, 0.3065],
    [0.2528, -0.6923, -0.6758],
    [0.2569, -0.8462, 0.4669],
    [0.0000, -1.0000, 0.0000],
];

#[derive(Clone)]
struct AromaticRing3d {
    label: String,
    centroid: [f32; 3],
    normal: [f32; 3],
    /// Negative (`-0.45..0.0`) for electron-rich (`Trp`/`Tyr`/`Phe`/anisole), near zero or positive (`+0.15`) for electron-poor (`-CF3`/`-F`/pyrimidine).
    pi_quadrupole_q: f64,
}

fn spearman_rank_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return 0.0;
    }
    let rank_vec = |arr: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0; n];
        for (r, &orig_i) in idx.iter().enumerate() {
            ranks[orig_i] = r as f64;
        }
        ranks
    };
    let rx = rank_vec(&xs[..n]);
    let ry = rank_vec(&ys[..n]);
    pearson_corr(&rx, &ry)
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return 0.0;
    }
    let mx = xs[..n].iter().sum::<f64>() / (n as f64);
    let my = ys[..n].iter().sum::<f64>() / (n as f64);
    let mut cov = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom > 1e-9 {
        (cov / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Core 3D `ESP` & `XED` multipole analyzer operating on a pre-parsed PDB & pocket (used by both GUI/CLI and `discovery_pipeline.rs` Stage 3).
pub fn analyze_esp_with_prebuilt(
    mol: &Molecule,
    _ds: Option<&DatasetAnalysis>,
    cfg: &ElectrostaticsEspConfig,
    pdb: &ParsedPdbStructure,
    pocket: &PocketCavityAnalysis,
    target_name: &str,
    enumerate_designs: bool,
) -> ElectrostaticsEspReport {
    let eps_int = cfg.interface_dielectric.clamp(1.5, 20.0);
    let pocket_res = score_ligand_in_pocket(mol, None, pocket);
    let lig_coords = pocket_res.aligned_ligand_coords3d.clone();
    let n_lig = mol.atoms.len().min(lig_coords.len());

    let mut atom_q: Vec<f64> = compute_gasteiger_charges(mol)
        .into_iter()
        .map(|q| q as f64)
        .collect();
    atom_q.resize(n_lig, 0.0);

    // 1. Build off-nucleus `XED`-style multipole pseudoatoms (`Sigma-Holes`, `Chalcogen Sigma*`, `Lone Pairs`, `Pi-Quadrupoles`)!
    let mut pseudoatoms: Vec<MultipolePseudoatom> = Vec::new();
    let mut lig_rings_3d: Vec<AromaticRing3d> = Vec::new();

    if cfg.enable_xed_multipoles {
        for i in 0..n_lig {
            let at = &mol.atoms[i];
            let p_i = lig_coords[i];
            let nbrs = mol.adj.get(i).cloned().unwrap_or_default();

            // (a) Anisotropic Halogen Sigma-Hole (`Cl`, `Br`, `I`) along `C -> X` bond extension!
            if matches!(at.elem, Element::Cl | Element::Br | Element::I) && nbrs.len() == 1 {
                let nb_idx = nbrs[0].0;
                if nb_idx < n_lig {
                    let u_cx = normalize3([
                        p_i[0] - lig_coords[nb_idx][0],
                        p_i[1] - lig_coords[nb_idx][1],
                        p_i[2] - lig_coords[nb_idx][2],
                    ]);
                    // Electron-withdrawing neighbor boost (e.g., aryl-F,CF3,pyrimidine boosts V_s,max by ~25%!)
                    let ewg_boost = if mol.atoms[nb_idx].aromatic {
                        let ring_ewg = mol
                            .adj
                            .get(nb_idx)
                            .map(|l| {
                                l.iter().any(|&(rn, _)| {
                                    rn != i
                                        && matches!(
                                            mol.atoms[rn].elem,
                                            Element::N | Element::F | Element::O
                                        )
                                })
                            })
                            .unwrap_or(false);
                        if ring_ewg {
                            1.26
                        } else {
                            1.10
                        }
                    } else {
                        1.0
                    };
                    let (q_sig, ext_r, base_vs) = match at.elem {
                        Element::Cl => (0.092 * ewg_boost, 1.42_f32, 15.8 * ewg_boost),
                        Element::Br => (0.142 * ewg_boost, 1.54_f32, 23.6 * ewg_boost),
                        Element::I => (0.205 * ewg_boost, 1.68_f32, 31.4 * ewg_boost),
                        _ => (0.0, 1.40, 0.0),
                    };
                    // Counter-charge on nucleus preserves exact molecular net charge while creating equatorial negative belt!
                    atom_q[i] -= q_sig;
                    pseudoatoms.push(MultipolePseudoatom {
                        parent_atom_idx: i,
                        parent_label: format!("{}{}", at.elem.symbol(), i + 1),
                        kind: MultipoleSiteKind::HalogenSigmaHoleCrown,
                        pos: [
                            p_i[0] + u_cx[0] * ext_r,
                            p_i[1] + u_cx[1] * ext_r,
                            p_i[2] + u_cx[2] * ext_r,
                        ],
                        axis_unit: u_cx,
                        charge_e: q_sig,
                        vs_extremum_kcal_e: base_vs,
                    });
                }
            }

            // (b) Bivalent Sulfur `Sigma*` Chalcogen-Bond Lobes (`R1 -> S` and `R2 -> S` extensions; Beno & Meanwell 2015 / Adhav et al. 2023)!
            if at.elem == Element::S && nbrs.len() == 2 {
                let u1 = nbrs[0].0;
                let u2 = nbrs[1].0;
                if u1 < n_lig && u2 < n_lig {
                    let het_boost = if at.aromatic
                        || matches!(mol.atoms[u1].elem, Element::N)
                        || matches!(mol.atoms[u2].elem, Element::N)
                    {
                        1.28
                    } else {
                        1.0
                    };
                    let q_ch = 0.072 * het_boost;
                    for &nb_u in &[u1, u2] {
                        let u_rs = normalize3([
                            p_i[0] - lig_coords[nb_u][0],
                            p_i[1] - lig_coords[nb_u][1],
                            p_i[2] - lig_coords[nb_u][2],
                        ]);
                        atom_q[i] -= q_ch;
                        pseudoatoms.push(MultipolePseudoatom {
                            parent_atom_idx: i,
                            parent_label: format!("S{}", i + 1),
                            kind: MultipoleSiteKind::ChalcogenSigmaStarLobe,
                            pos: [
                                p_i[0] + u_rs[0] * 1.38,
                                p_i[1] + u_rs[1] * 1.38,
                                p_i[2] + u_rs[2] * 1.38,
                            ],
                            axis_unit: u_rs,
                            charge_e: q_ch,
                            vs_extremum_kcal_e: 14.2 * het_boost,
                        });
                    }
                }
            }

            // (c) Directional Carbonyl `=O`, Nitrile `#N`, & Pyridine `n:` Lone-Pair Lobes (`XED` Vinter 1994)
            if (at.elem == Element::O && nbrs.len() == 1 && matches!(at.hyb, Hybridization::Sp2))
                || (at.elem == Element::N
                    && ((nbrs.len() == 1 && matches!(at.hyb, Hybridization::Sp))
                        || (at.aromatic && nbrs.len() == 2 && at.explicit_h == 0 && at.implicit_h == 0)))
            {
                let mut c_sum = [0.0_f32; 3];
                let mut c_cnt = 0_usize;
                for &(nb, _) in &nbrs {
                    if nb < n_lig {
                        c_sum[0] += lig_coords[nb][0];
                        c_sum[1] += lig_coords[nb][1];
                        c_sum[2] += lig_coords[nb][2];
                        c_cnt += 1;
                    }
                }
                if c_cnt > 0 {
                    let u_lp = normalize3([
                        p_i[0] - c_sum[0] / (c_cnt as f32),
                        p_i[1] - c_sum[1] / (c_cnt as f32),
                        p_i[2] - c_sum[2] / (c_cnt as f32),
                    ]);
                    let q_lp = -0.105_f64;
                    atom_q[i] -= q_lp;
                    pseudoatoms.push(MultipolePseudoatom {
                        parent_atom_idx: i,
                        parent_label: format!("{}{}", at.elem.symbol(), i + 1),
                        kind: MultipoleSiteKind::DirectionalLonePairLobe,
                        pos: [
                            p_i[0] + u_lp[0] * 0.58,
                            p_i[1] + u_lp[1] * 0.58,
                            p_i[2] + u_lp[2] * 0.58,
                        ],
                        axis_unit: u_lp,
                        charge_e: q_lp,
                        vs_extremum_kcal_e: -28.5,
                    });
                }
            }
        }
    }

    // (d) Aromatic SSSR Ring `Pi`-Quadrupole Sandwiches (`Wheeler-Houk` substituent-tuned)
    for (r_idx, cyc) in mol.rings.iter().enumerate() {
        if cyc.len() < 5 || cyc.len() > 7 || !cyc.iter().all(|&u| u < n_lig && mol.atoms[u].aromatic)
        {
            continue;
        }
        let mut cen = [0.0_f32; 3];
        for &u in cyc {
            cen[0] += lig_coords[u][0];
            cen[1] += lig_coords[u][1];
            cen[2] += lig_coords[u][2];
        }
        let inv = 1.0 / (cyc.len() as f32);
        cen = [cen[0] * inv, cen[1] * inv, cen[2] * inv];

        let p0 = lig_coords[cyc[0]];
        let p1 = lig_coords[cyc[1]];
        let p2 = lig_coords[cyc[2]];
        let nrm = normalize3(cross3(
            [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
            [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
        ));

        // Wheeler-Houk substituent sum: ring nitrogens & attached F/CF3/CN/S(=O)2 deplete pi density!
        let mut ewg_count = 0_i32;
        let mut edg_count = 0_i32;
        for &u in cyc {
            if mol.atoms[u].elem == Element::N && mol.atoms[u].implicit_h == 0 && mol.atoms[u].explicit_h == 0 {
                ewg_count += 1;
            }
            if let Some(adj_u) = mol.adj.get(u) {
                for &(exo, _) in adj_u {
                    if !cyc.contains(&exo) && exo < n_lig {
                        match mol.atoms[exo].elem {
                            Element::F | Element::Cl | Element::Br | Element::S => ewg_count += 1,
                            Element::O | Element::N => edg_count += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
        // Standard benzene pi-face q_pi = -0.22 e per face (+0.44 e in sigma frame); heavy EWG shifts toward +0.06 e
        let q_pi_face = (-0.22 + (ewg_count as f64) * 0.055 - (edg_count as f64) * 0.035).clamp(-0.32, 0.10);
        lig_rings_3d.push(AromaticRing3d {
            label: format!("LigRing{}", r_idx + 1),
            centroid: cen,
            normal: nrm,
            pi_quadrupole_q: q_pi_face,
        });

        if cfg.enable_xed_multipoles {
            let per_ring_atom_comp = -(2.0 * q_pi_face) / (cyc.len() as f64);
            for &u in cyc {
                atom_q[u] += per_ring_atom_comp;
            }
            for sign in [-1.0_f32, 1.0_f32] {
                pseudoatoms.push(MultipolePseudoatom {
                    parent_atom_idx: cyc[0],
                    parent_label: format!("Ar{}", r_idx + 1),
                    kind: MultipoleSiteKind::AromaticPiQuadrupolePole,
                    pos: [
                        cen[0] + nrm[0] * 0.68 * sign,
                        cen[1] + nrm[1] * 0.68 * sign,
                        cen[2] + nrm[2] * 0.68 * sign,
                    ],
                    axis_unit: [nrm[0] * sign, nrm[1] * sign, nrm[2] * sign],
                    charge_e: q_pi_face,
                    vs_extremum_kcal_e: q_pi_face * 68.0,
                });
            }
        }
    }

    // 2. Build Receptor Electrostatic Charge + Aromatic Ring / Cation Anchor Model
    let mut rec_charges: Vec<([f32; 3], f64, String, Element)> = Vec::new();
    let mut rec_rings: Vec<AromaticRing3d> = Vec::new();
    let mut rec_cations: Vec<([f32; 3], String)> = Vec::new();
    let mut res_arom_pts: BTreeMap<(char, i32), (String, Vec<[f32; 3]>)> = BTreeMap::new();

    for pa in &pdb.protein_atoms {
        if matches!(pa.elem, Element::H) || dist3(pa.pos, pocket.center) > pocket.pocket_radius_a + 4.5 {
            continue;
        }
        let q_rec: f64 = match (pa.res_name.as_str(), pa.atom_name.as_str()) {
            ("ASP", "OD1" | "OD2") | ("GLU", "OE1" | "OE2") => -0.58,
            ("ARG", "NH1" | "NH2" | "NE") => 0.44,
            ("LYS", "NZ") => 0.72,
            (_, "O") => -0.46,
            (_, "N") => 0.32,
            ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") => -0.38,
            ("ASN", "OD1") | ("GLN", "OE1") => -0.45,
            ("ASN", "ND2") | ("GLN", "NE2") => 0.34,
            ("HIS", "ND1" | "NE2") => -0.26,
            (_, "CA" | "C") => 0.16,
            _ => 0.0,
        };
        let lbl = format!("{}{}:{}", pa.res_name, pa.res_seq, pa.atom_name);
        if q_rec.abs() > 1e-4 {
            rec_charges.push((pa.pos, q_rec, lbl.clone(), pa.elem));
        }
        if matches!(
            (pa.res_name.as_str(), pa.atom_name.as_str()),
            ("ARG", "NH1" | "NH2" | "CZ") | ("LYS", "NZ")
        ) && !rec_cations.iter().any(|(_, l)| l.starts_with(&format!("{}{}", pa.res_name, pa.res_seq)))
        {
            rec_cations.push((pa.pos, format!("{}{}", pa.res_name, pa.res_seq)));
        }
        if matches!(pa.res_name.as_str(), "PHE" | "TYR" | "TRP" | "HIS")
            && !matches!(pa.atom_name.as_str(), "N" | "CA" | "C" | "O" | "CB")
        {
            res_arom_pts
                .entry((pa.chain_id, pa.res_seq))
                .or_insert_with(|| (format!("{}{}", pa.res_name, pa.res_seq), Vec::new()))
                .1
                .push(pa.pos);
        }
    }

    for (_, (rlbl, pts)) in res_arom_pts {
        if pts.is_empty() {
            continue;
        }
        let mut c = [0.0_f32; 3];
        for &p in &pts {
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        let inv = 1.0 / (pts.len() as f32);
        c = [c[0] * inv, c[1] * inv, c[2] * inv];
        let nrm = if pts.len() >= 3 {
            normalize3(cross3(
                [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2]],
                [pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2]],
            ))
        } else {
            normalize3([c[0] - pocket.center[0], c[1] - pocket.center[1], c[2] - pocket.center[2]])
        };
        let q_pi = if rlbl.starts_with("TRP") {
            -0.28
        } else if rlbl.starts_with("TYR") || rlbl.starts_with("PHE") {
            -0.23
        } else {
            -0.16
        };
        rec_rings.push(AromaticRing3d {
            label: rlbl,
            centroid: c,
            normal: nrm,
            pi_quadrupole_q: q_pi,
        });
    }

    // 3. Sample Interfacial Connolly Surface Vertices & Compute Paired `V_lig(s)` and `V_prot(s)` (`Bauer & Mackey 2019`)!
    let mut vertices: Vec<ConnollyEspVertex> = Vec::new();
    let mut vs_max = f64::NEG_INFINITY;
    let mut vs_min = f64::INFINITY;

    for i in 0..n_lig {
        let r_surf = vdw_rad(mol.atoms[i].elem) + 0.42;
        let ci = lig_coords[i];
        for dir in &FIBONACCI_14_DIRS {
            let sp = [
                ci[0] + dir[0] * r_surf,
                ci[1] + dir[1] * r_surf,
                ci[2] + dir[2] * r_surf,
            ];
            // Prune interior buried ligand points
            let buried = (0..n_lig).any(|j| {
                j != i && dist3(sp, lig_coords[j]) < vdw_rad(mol.atoms[j].elem) + 0.24
            });
            if buried {
                continue;
            }
            // Evaluate V_lig(sp) across all ligand nuclei + off-center XED pseudoatoms
            let mut v_lig = 0.0_f64;
            for j in 0..n_lig {
                let r = (dist3(sp, lig_coords[j]) as f64).max(1.15);
                let eps_eff = eps_int * (0.65 + 0.35 * r / 3.0).clamp(0.7, 2.5);
                v_lig += COULOMB_KCAL_A_PER_E2 * atom_q[j] / (eps_eff * r);
            }
            for mp in &pseudoatoms {
                let r = (dist3(sp, mp.pos) as f64).max(0.72);
                let eps_eff = eps_int * (0.60 + 0.30 * r / 2.5).clamp(0.65, 2.2);
                v_lig += COULOMB_KCAL_A_PER_E2 * mp.charge_e / (eps_eff * r);
            }

            // Evaluate V_prot(sp) across receptor partial charges + aromatic pi-quadrupoles
            let mut v_prot = 0.0_f64;
            for &(rp, rq, _, _) in &rec_charges {
                let r = (dist3(sp, rp) as f64).max(1.35);
                if r <= 9.5 {
                    let eps_eff = eps_int * (0.70 + 0.30 * r / 3.5).clamp(0.75, 2.8);
                    v_prot += COULOMB_KCAL_A_PER_E2 * rq / (eps_eff * r);
                }
            }
            for rr in &rec_rings {
                for sign in [-1.0_f32, 1.0_f32] {
                    let pole = [
                        rr.centroid[0] + rr.normal[0] * 0.68 * sign,
                        rr.centroid[1] + rr.normal[1] * 0.68 * sign,
                        rr.centroid[2] + rr.normal[2] * 0.68 * sign,
                    ];
                    let r = (dist3(sp, pole) as f64).max(1.45);
                    if r <= 7.5 {
                        v_prot += COULOMB_KCAL_A_PER_E2 * rr.pi_quadrupole_q / (eps_int * r);
                    }
                }
            }

            vs_max = vs_max.max(v_lig);
            vs_min = vs_min.min(v_lig);

            // Bauer-Mackey (2019) capped local surface complementarity score `c(s) in [-1, +1]`
            let vl_c = v_lig.clamp(-BAUER_MACKEY_ESP_CAP_KCAL_E, BAUER_MACKEY_ESP_CAP_KCAL_E);
            let vp_c = v_prot.clamp(-BAUER_MACKEY_ESP_CAP_KCAL_E, BAUER_MACKEY_ESP_CAP_KCAL_E);
            let max_mag = vl_c.abs().max(vp_c.abs()).max(3.5);
            let local_ec = (1.0 - (vl_c + vp_c).abs() / max_mag).clamp(-1.0, 1.0)
                * if vl_c * vp_c <= 0.0 { 1.0 } else { -0.85 };

            vertices.push(ConnollyEspVertex {
                pos: sp,
                parent_atom_idx: i,
                v_lig_kcal_e: v_lig,
                v_prot_kcal_e: v_prot,
                local_ec_score: local_ec.clamp(-1.0, 1.0),
            });
        }
    }

    if !vs_max.is_finite() {
        vs_max = 0.0;
        vs_min = 0.0;
    }

    let n_v = vertices.len().max(1) as f64;
    let ec_integral = vertices.iter().map(|v| v.local_ec_score).sum::<f64>() / n_v;
    let vl_vec: Vec<f64> = vertices.iter().map(|v| v.v_lig_kcal_e).collect();
    let vp_vec: Vec<f64> = vertices.iter().map(|v| v.v_prot_kcal_e).collect();
    let ec_pearson_r = (-pearson_corr(&vl_vec, &vp_vec)).clamp(-1.0, 1.0);
    let ec_spearman_rho = (-spearman_rank_corr(&vl_vec, &vp_vec)).clamp(-1.0, 1.0);

    let fav_pct =
        (vertices.iter().filter(|v| v.local_ec_score >= 0.20).count() as f64) * 100.0 / n_v;
    let clash_pct =
        (vertices.iter().filter(|v| v.local_ec_score <= -0.25).count() as f64) * 100.0 / n_v;

    // 4. Scan Explicit Anisotropic `Sigma-Hole` Halogen Bonds, Sulfur Chalcogen Bonds/Locks, `Pi-Stacking`, `Cation-Pi`, & `ESP` Clashes!
    let mut contacts: Vec<MultipoleContact> = Vec::new();
    let mut xb_cnt = 0_usize;
    let mut chb_cnt = 0_usize;
    let mut pi_cnt = 0_usize;
    let mut clash_cnt = 0_usize;
    let mut dg_xb_chb = 0.0_f64;
    let mut dg_pi = 0.0_f64;

    // (a) Halogen Bonds & Intermolecular Chalcogen Bonds from XED sigma-hole / sigma* pseudoatoms
    for mp in &pseudoatoms {
        if !matches!(
            mp.kind,
            MultipoleSiteKind::HalogenSigmaHoleCrown | MultipoleSiteKind::ChalcogenSigmaStarLobe
        ) {
            continue;
        }
        let x_pos = lig_coords[mp.parent_atom_idx];
        for &(rp, rq, ref rlbl, relem) in &rec_charges {
            if rq >= -0.15 || !matches!(relem, Element::O | Element::N) {
                continue;
            }
            let d_xb = dist3(x_pos, rp);
            if !(2.55..=3.85).contains(&d_xb) {
                continue;
            }
            let u_xb = normalize3([rp[0] - x_pos[0], rp[1] - x_pos[1], rp[2] - x_pos[2]]);
            // `mp.axis_unit` points from `C -> X`; linear `C-X...B` (`theta = 180 deg`) corresponds to `dot(u_cx, u_xb) = +1.0`!
            let cos_lin = dot3(mp.axis_unit, u_xb).clamp(-1.0, 1.0);
            let theta_deg = 180.0 - cos_lin.acos().to_degrees();
            if theta_deg >= 140.0 {
                let ang_factor = ((theta_deg - 135.0) / 45.0).clamp(0.15, 1.0) as f64;
                let dist_factor = (-(((d_xb as f64) - 3.12) / 0.45).powi(2)).exp();
                let is_xb = mp.kind == MultipoleSiteKind::HalogenSigmaHoleCrown;
                let e_bond = -(if is_xb { 1.85 } else { 1.35 })
                    * (mp.vs_extremum_kcal_e / 20.0).clamp(0.55, 1.55)
                    * ang_factor
                    * dist_factor;
                if e_bond <= -0.22 {
                    if is_xb {
                        xb_cnt += 1;
                    } else {
                        chb_cnt += 1;
                    }
                    dg_xb_chb += e_bond;
                    contacts.push(MultipoleContact {
                        kind: if is_xb {
                            MultipoleContactKind::HalogenBondSigmaHole
                        } else {
                            MultipoleContactKind::ChalcogenBondIntermolecular
                        },
                        ligand_site_label: mp.parent_label.clone(),
                        partner_label: rlbl.clone(),
                        ligand_pos: x_pos,
                        partner_pos: rp,
                        distance_a: d_xb,
                        angle_deg: theta_deg,
                        aux_metric: mp.vs_extremum_kcal_e,
                        energy_kcal: e_bond,
                        rationale: format!(
                            "Linear +sigma-hole (Vs,max={:+.1} kcal/e, theta={:.0} deg) engaging Lewis base {}",
                            mp.vs_extremum_kcal_e, theta_deg, rlbl
                        ),
                    });
                }
            }
        }

        // Intramolecular 1,4 / 1,5 `S...O=C` or `S...N` Chalcogen Conformational Lock (Beno & Meanwell 2015)!
        if mp.kind == MultipoleSiteKind::ChalcogenSigmaStarLobe {
            let s_idx = mp.parent_atom_idx;
            for j in 0..n_lig {
                if j == s_idx || !matches!(mol.atoms[j].elem, Element::O | Element::N) {
                    continue;
                }
                let direct_12 = mol
                    .adj
                    .get(s_idx)
                    .map(|l| l.iter().any(|&(nb, _)| nb == j))
                    .unwrap_or(false);
                if direct_12 {
                    continue;
                }
                let d_so = dist3(x_pos, lig_coords[j]);
                if (2.45..=3.38).contains(&d_so) {
                    let u_so = normalize3([
                        lig_coords[j][0] - x_pos[0],
                        lig_coords[j][1] - x_pos[1],
                        lig_coords[j][2] - x_pos[2],
                    ]);
                    let theta = 180.0 - dot3(mp.axis_unit, u_so).clamp(-1.0, 1.0).acos().to_degrees();
                    if theta >= 144.0 {
                        let e_lock = -0.82 * (((theta - 140.0) / 40.0).clamp(0.25, 1.0) as f64);
                        chb_cnt += 1;
                        dg_xb_chb += e_lock;
                        contacts.push(MultipoleContact {
                            kind: MultipoleContactKind::ChalcogenLockIntramolecular,
                            ligand_site_label: mp.parent_label.clone(),
                            partner_label: format!("Intra-{}{}", mol.atoms[j].elem.symbol(), j + 1),
                            ligand_pos: x_pos,
                            partner_pos: lig_coords[j],
                            distance_a: d_so,
                            angle_deg: theta,
                            aux_metric: mp.vs_extremum_kcal_e,
                            energy_kcal: e_lock,
                            rationale: format!(
                                "Intramolecular S..{} sigma* conformational lock (d={:.2} A, theta={:.0} deg) preorganizing bioactive pose",
                                mol.atoms[j].elem.symbol(),
                                d_so,
                                theta
                            ),
                        });
                        break;
                    }
                }
            }
        }
    }

    // (b) Aromatic `Pi-Pi` (Slip-Stacked & T-Shaped) + `Cation-Pi` Multipole Recognition!
    for lr in &lig_rings_3d {
        for rr in &rec_rings {
            let d_cc = dist3(lr.centroid, rr.centroid);
            if !(3.30..=5.65).contains(&d_cc) {
                continue;
            }
            let cos_n = dot3(lr.normal, rr.normal).abs().clamp(0.0, 1.0);
            let plane_ang = cos_n.acos().to_degrees();
            let disp_vec = [
                rr.centroid[0] - lr.centroid[0],
                rr.centroid[1] - lr.centroid[1],
                rr.centroid[2] - lr.centroid[2],
            ];
            let vert_h = dot3(disp_vec, lr.normal).abs();
            let slip_a = (d_cc * d_cc - vert_h * vert_h).max(0.0).sqrt();

            // Wheeler-Houk electronic complementarity bonus when an electron-deficient ligand ring stacks on electron-rich Tyr/Trp/Phe!
            let wheeler_bonus = ((lr.pi_quadrupole_q - rr.pi_quadrupole_q) * 1.45).clamp(-0.15, 0.55);

            if plane_ang <= 32.0 && d_cc <= 4.55 {
                let e_pd = (-1.15 - wheeler_bonus)
                    * (-(((d_cc as f64) - 3.78) / 0.55).powi(2)).exp();
                pi_cnt += 1;
                dg_pi += e_pd;
                contacts.push(MultipoleContact {
                    kind: MultipoleContactKind::PiStackParallelDisplaced,
                    ligand_site_label: lr.label.clone(),
                    partner_label: rr.label.clone(),
                    ligand_pos: lr.centroid,
                    partner_pos: rr.centroid,
                    distance_a: d_cc,
                    angle_deg: plane_ang,
                    aux_metric: slip_a as f64,
                    energy_kcal: e_pd,
                    rationale: format!(
                        "Slip-stacked pi-pi contact (offset {:.2} A, Wheeler-Houk q_pi={:+.2} vs {:+.2})",
                        slip_a, lr.pi_quadrupole_q, rr.pi_quadrupole_q
                    ),
                });
            } else if plane_ang >= 54.0 && d_cc <= 5.55 {
                let e_t = -0.88 * (-(((d_cc as f64) - 4.88) / 0.62).powi(2)).exp();
                pi_cnt += 1;
                dg_pi += e_t;
                contacts.push(MultipoleContact {
                    kind: MultipoleContactKind::PiStackTShapedEdgeToFace,
                    ligand_site_label: lr.label.clone(),
                    partner_label: rr.label.clone(),
                    ligand_pos: lr.centroid,
                    partner_pos: rr.centroid,
                    distance_a: d_cc,
                    angle_deg: plane_ang,
                    aux_metric: slip_a as f64,
                    energy_kcal: e_t,
                    rationale: format!(
                        "Edge-to-face T-shaped C-H..pi quadrupole contact (plane angle {:.0} deg)",
                        plane_ang
                    ),
                });
            }
        }

        // Receptor Cation (Arg/Lys) -> Ligand Electron-Rich Pi Cloud!
        if lr.pi_quadrupole_q <= -0.10 {
            for &(cpos, ref clbl) in &rec_cations {
                let d_cp = dist3(lr.centroid, cpos);
                if (3.25..=4.85).contains(&d_cp) {
                    let u_cp = normalize3([
                        cpos[0] - lr.centroid[0],
                        cpos[1] - lr.centroid[1],
                        cpos[2] - lr.centroid[2],
                    ]);
                    let ang_norm = dot3(u_cp, lr.normal).abs().clamp(0.0, 1.0).acos().to_degrees();
                    if ang_norm <= 42.0 {
                        let e_cp = -1.55 * (-lr.pi_quadrupole_q / 0.22).clamp(0.5, 1.4);
                        pi_cnt += 1;
                        dg_pi += e_cp;
                        contacts.push(MultipoleContact {
                            kind: MultipoleContactKind::CationPiRecognition,
                            ligand_site_label: lr.label.clone(),
                            partner_label: clbl.clone(),
                            ligand_pos: lr.centroid,
                            partner_pos: cpos,
                            distance_a: d_cp,
                            angle_deg: ang_norm,
                            aux_metric: lr.pi_quadrupole_q,
                            energy_kcal: e_cp,
                            rationale: format!(
                                "Receptor {} cation engaging ligand pi-quadrupole (q_pi={:+.2} e)",
                                clbl, lr.pi_quadrupole_q
                            ),
                        });
                    }
                }
            }
        }
    }

    // Ligand Basic Amine / Cation -> Receptor Aromatic Cage (`Trp`/`Tyr`/`Phe`, e.g. DRD2 Trp386/Phe389)!
    for i in 0..n_lig {
        let at = &mol.atoms[i];
        let basic_or_cat = at.charge > 0
            || (at.elem == Element::N
                && !at.aromatic
                && matches!(at.hyb, Hybridization::Sp3)
                && atom_q[i] > -0.32);
        if !basic_or_cat {
            continue;
        }
        for rr in &rec_rings {
            let d_cp = dist3(lig_coords[i], rr.centroid);
            if (3.25..=4.90).contains(&d_cp) {
                let u_cp = normalize3([
                    lig_coords[i][0] - rr.centroid[0],
                    lig_coords[i][1] - rr.centroid[1],
                    lig_coords[i][2] - rr.centroid[2],
                ]);
                let ang_norm = dot3(u_cp, rr.normal).abs().clamp(0.0, 1.0).acos().to_degrees();
                if ang_norm <= 44.0 {
                    let e_cp = -1.68 * (-rr.pi_quadrupole_q / 0.23).clamp(0.6, 1.35);
                    pi_cnt += 1;
                    dg_pi += e_cp;
                    contacts.push(MultipoleContact {
                        kind: MultipoleContactKind::CationPiRecognition,
                        ligand_site_label: format!("N+{i}"),
                        partner_label: rr.label.clone(),
                        ligand_pos: lig_coords[i],
                        partner_pos: rr.centroid,
                        distance_a: d_cp,
                        angle_deg: ang_norm,
                        aux_metric: rr.pi_quadrupole_q,
                        energy_kcal: e_cp,
                        rationale: format!(
                            "Protonated/basic amine engaging {} aromatic cage (Gallivan-Dougherty cation-pi)",
                            rr.label
                        ),
                    });
                }
            }
        }
    }

    // (c) Flag Worst Same-Sign `ESP` Surface Clash Atoms
    let mut clashing_atoms_seen: Vec<usize> = Vec::new();
    for v in &vertices {
        if v.local_ec_score <= -0.58
            && v.v_lig_kcal_e.abs() >= 10.0
            && v.v_prot_kcal_e.abs() >= 10.0
            && !clashing_atoms_seen.contains(&v.parent_atom_idx)
        {
            clashing_atoms_seen.push(v.parent_atom_idx);
            clash_cnt += 1;
            let ai = v.parent_atom_idx;
            contacts.push(MultipoleContact {
                kind: MultipoleContactKind::ElectrostaticSurfaceClash,
                ligand_site_label: format!("{}{}", mol.atoms[ai].elem.symbol(), ai + 1),
                partner_label: "Pocket ESP Wall".to_string(),
                ligand_pos: lig_coords[ai],
                partner_pos: v.pos,
                distance_a: dist3(lig_coords[ai], v.pos),
                angle_deg: 0.0,
                aux_metric: v.local_ec_score,
                energy_kcal: 0.65,
                rationale: format!(
                    "Same-sign ESP mismatch: V_lig={:+.1} vs V_prot={:+.1} kcal/(mol*e)",
                    v.v_lig_kcal_e, v.v_prot_kcal_e
                ),
            });
            if clash_cnt >= 4 {
                break;
            }
        }
    }

    let dg_surface_esp = (-1.45 * ec_integral + 0.35 * (clash_cnt as f64)).clamp(-2.20, 1.80);
    let total_mult_dg = (dg_xb_chb + dg_pi + dg_surface_esp).clamp(-5.20, 2.40);
    let esp_refined_dg = pocket_res.total_dg_pocket_kcal + total_mult_dg * 0.48;
    let esp_pred_pkd = (-esp_refined_dg / RT_LN10_KCAL).clamp(3.0, 13.0);

    // 5. Enumerate `ESP` & `Sigma-Hole` Bioisosteric Tuning Analogs!
    let mut designs: Vec<EspMultipoleDesign> = Vec::new();
    if enumerate_designs && cfg.generate_esp_designs {
        let base_smi = if mol.smiles.trim().is_empty() {
            molecule_to_smiles(mol)
        } else {
            mol.smiles.trim().to_string()
        };

        let mut cand_transforms: Vec<(&'static str, String, &'static str, String, f64, f64)> =
            Vec::new();

        // Transform 1: Cl -> Br Sigma-Hole Escalation (boosts V_s,max from ~16 to ~24 kcal/(mol*e)!)
        if base_smi.contains("Cl") {
            cand_transforms.push((
                "Cl -> Br Sigma-Hole Boost",
                base_smi.replacen("Cl", "Br", 1),
                "Politzer/Boeckler C-Br +Sigma-Hole Escalation",
                "Backbone O=C Lewis Base".to_string(),
                0.08,
                -0.68,
            ));
        } else {
            cand_transforms.push((
                "+Aryl-Cl Sigma-Hole Probe",
                format!("{base_smi}(Cl)"),
                "Install C-Cl +Sigma-Hole Crown for S1/Back-Pocket O=C",
                "Pocket O=C Acceptor".to_string(),
                0.07,
                -0.55,
            ));
        }

        // Transform 2: F -> Cl Sigma-Hole Switch (F lacks a positive sigma-hole; Cl unlocks linear XB!)
        if base_smi.contains("(F)") {
            cand_transforms.push((
                "F -> Cl Sigma-Hole Unlock",
                base_smi.replacen("(F)", "(Cl)", 1),
                "Replace non-XB Fluoro with sigma-hole-active Chloro",
                "Carbonyl O=C Acceptor".to_string(),
                0.06,
                -0.52,
            ));
        }

        // Transform 3: Oxazole -> 1,3-Thiazole Chalcogen Conformational Lock (Beno & Meanwell 2015)
        if base_smi.contains("o1") || base_smi.contains("oc") {
            let thiaz = if base_smi.contains("o1") {
                base_smi.replacen("o1", "s1", 1)
            } else {
                base_smi.replacen("oc", "sc", 1)
            };
            cand_transforms.push((
                "O -> S Thiazole Chalcogen Lock",
                thiaz,
                "Install 1,5 S..O=C sigma* Chalcogen Conformational Lock",
                "Adjacent Carbonyl / Pocket O=C".to_string(),
                0.05,
                -0.62,
            ));
        } else {
            cand_transforms.push((
                "+2-Thienyl Chalcogen Bioisostere",
                format!("{base_smi}(c1cccs1)"),
                "Bivalent S sigma* Chalcogen + Aromatic Dispersion Probe",
                "Aromatic / Carbonyl Sub-Pocket".to_string(),
                0.04,
                -0.44,
            ));
        }

        // Transform 4: Wheeler-Houk Arene Fluorination / Pyridine N-Walk to tune Pi-Stacking & fix ESP clashes!
        if base_smi.contains("cc1") {
            cand_transforms.push((
                "CH -> N Pyridine ESP Walk",
                base_smi.replacen("cc1", "nc1", 1),
                "Wheeler-Houk Heteroaryl Dipole & Pi-Deficiency Tuning",
                "Pi-Stack & Rim Polar Anchor".to_string(),
                0.11,
                -0.58,
            ));
            cand_transforms.push((
                "+Fluoro Wheeler-Houk Pi-Tuner",
                base_smi.replacen("cc1", "c(F)c1", 1),
                "Deplete Arene Pi-Density to Strengthen Tyr/Trp Slip-Stack",
                "Electron-Rich Receptor Tyr/Trp".to_string(),
                0.09,
                -0.49,
            ));
        }

        for (idx, (lbl, cand_s, mech, part, d_ec_est, ddg_est)) in
            cand_transforms.into_iter().enumerate()
        {
            let Ok(cmol) = parse_smiles(&cand_s) else {
                continue;
            };
            let canon = molecule_to_smiles(&cmol);
            if designs.iter().any(|d| d.designed_smiles == canon) {
                continue;
            }
            let alerts = scan_structural_alerts(&cmol);
            let phys = evaluate_physchem(&cmol, alerts.len());
            let sa = evaluate_synthetic_accessibility(&cmol).sa_score;
            let pains_n = alerts
                .iter()
                .filter(|a| a.family.eq_ignore_ascii_case("PAINS"))
                .count();
            let pred_ecr = (ec_pearson_r + d_ec_est).clamp(-0.95, 0.98);
            let dpact = -ddg_est / RT_LN10_KCAL;
            let pred_pact = mol.pact + dpact;
            let score = (52.0 + (-ddg_est) * 32.0 + d_ec_est * 95.0
                - (sa - 2.8).max(0.0) * 4.5
                - (pains_n as f64) * 25.0)
                .clamp(10.0, 99.2);

            designs.push(EspMultipoleDesign {
                rank: idx + 1,
                design_id: format!("ESP-D{:02}", idx + 1),
                modification_label: lbl.to_string(),
                mechanism: mech,
                target_partner: part,
                designed_smiles: canon,
                baseline_ec_r: ec_pearson_r,
                predicted_ec_r: pred_ecr,
                delta_ec_r: pred_ecr - ec_pearson_r,
                ddg_multipole_kcal: ddg_est,
                predicted_pact: pred_pact,
                mw: phys.mw,
                clogp: phys.clogp,
                sa_score: sa,
                pains_count: pains_n,
                design_score: score,
            });
        }
        designs.sort_by(|a, b| {
            b.design_score
                .partial_cmp(&a.design_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (k, d) in designs.iter_mut().enumerate() {
            d.rank = k + 1;
        }
    }

    let shell_display: Vec<([f32; 3], String, Element)> = pocket
        .shell_atoms
        .iter()
        .take(90)
        .map(|a| (a.pos, format!("{}{}:{}", a.res_name, a.res_seq, a.atom_name), a.elem))
        .collect();

    ElectrostaticsEspReport {
        preset: cfg.preset,
        pdb_id: pdb.pdb_id.clone(),
        target_name: target_name.to_string(),
        pocket_center: pocket.center,
        compound_id: mol.id.clone(),
        compound_name: mol.name.clone(),
        query_smiles: if mol.smiles.is_empty() {
            molecule_to_smiles(mol)
        } else {
            mol.smiles.clone()
        },
        mol: mol.clone(),
        bauer_mackey_ec_integral: ec_integral,
        bauer_mackey_ec_pearson_r: ec_pearson_r,
        bauer_mackey_ec_spearman_rho: ec_spearman_rho,
        esp_favorable_surface_pct: fav_pct,
        esp_clash_surface_pct: clash_pct,
        ligand_vs_max_kcal_e: vs_max,
        ligand_vs_min_kcal_e: vs_min,
        halogen_bond_count: xb_cnt,
        chalcogen_bond_and_lock_count: chb_cnt,
        pi_and_cation_pi_count: pi_cnt,
        esp_clash_count: clash_cnt,
        dg_halogen_chalcogen_kcal: dg_xb_chb,
        dg_pi_multipole_kcal: dg_pi,
        dg_surface_esp_kcal: dg_surface_esp,
        total_multipole_dg_kcal: total_mult_dg,
        esp_refined_pocket_dg_kcal: esp_refined_dg,
        esp_predicted_pkd: esp_pred_pkd,
        aligned_ligand_coords3d: lig_coords,
        pocket_shell_atoms: shell_display,
        multipole_pseudoatoms: pseudoatoms,
        connolly_vertices: vertices,
        contacts,
        esp_designs: designs,
    }
}

/// Public entry point computing 3D `ESP` complementarity, `sigma-hole` halogen/chalcogen bonds, and `pi` multipole contacts for `mol`.
pub fn analyze_electrostatics_esp(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    cfg: &ElectrostaticsEspConfig,
) -> ElectrostaticsEspReport {
    let (pdb, center_override, target_name) = load_esp_target_pdb(cfg);
    let pocket = analyze_binding_pocket(&pdb, center_override, 9.5);
    analyze_esp_with_prebuilt(
        mol,
        ds,
        cfg,
        &pdb,
        &pocket,
        &target_name,
        cfg.generate_esp_designs,
    )
}

/// Parallel Rayon screening of an entire `DatasetAnalysis` for 3D `Bauer-Mackey` `ESP` & `sigma-hole` complementarity.
pub fn screen_dataset_electrostatics_esp(
    ds: &DatasetAnalysis,
    cfg: &ElectrostaticsEspConfig,
) -> Vec<DatasetEspScreenRow> {
    let (pdb, center_override, target_name) = load_esp_target_pdb(cfg);
    let pocket = analyze_binding_pocket(&pdb, center_override, 9.5);

    let mut rows: Vec<DatasetEspScreenRow> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(idx, m)| {
            let rep = analyze_esp_with_prebuilt(m, Some(ds), cfg, &pdb, &pocket, &target_name, true);
            let top_d = rep
                .esp_designs
                .first()
                .map(|d| format!("{} ({:+.2} kcal)", d.modification_label, d.ddg_multipole_kcal))
                .unwrap_or_else(|| "ESP Complementarity Optimal".to_string());
            DatasetEspScreenRow {
                rank: 0,
                mol_idx: idx,
                compound_id: m.id.clone(),
                compound_name: m.name.clone(),
                observed_pact: m.pact,
                ec_integral: rep.bauer_mackey_ec_integral,
                ec_pearson_r: rep.bauer_mackey_ec_pearson_r,
                halogen_bonds: rep.halogen_bond_count,
                chalcogen_contacts: rep.chalcogen_bond_and_lock_count,
                pi_contacts: rep.pi_and_cation_pi_count,
                esp_clashes: rep.esp_clash_count,
                total_multipole_dg_kcal: rep.total_multipole_dg_kcal,
                esp_refined_pocket_dg_kcal: rep.esp_refined_pocket_dg_kcal,
                top_esp_design: top_d,
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        a.esp_refined_pocket_dg_kcal
            .partial_cmp(&b.esp_refined_pocket_dg_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Exports detected `sigma-hole` halogen bonds, chalcogen locks, `pi-stacking`, `cation-pi`, and `ESP` clash rows to `.csv`.
pub fn export_esp_contacts_csv(rep: &ElectrostaticsEspReport) -> String {
    let mut out = String::from(
        "pdb_id,compound_id,interaction_kind,ligand_site,partner_site,distance_a,angle_deg,aux_metric,energy_kcal,rationale\n",
    );
    for c in &rep.contacts {
        out.push_str(&format!(
            "{},{},\"{}\",\"{}\",\"{}\",{:.3},{:.1},{:.3},{:+.3},\"{}\"\n",
            rep.pdb_id,
            rep.compound_id,
            c.kind.short_label(),
            c.ligand_site_label,
            c.partner_label,
            c.distance_a,
            c.angle_deg,
            c.aux_metric,
            c.energy_kcal,
            c.rationale
        ));
    }
    out
}

/// Exports `ESP` & `Sigma-Hole` bioisosteric design suggestions to `.csv`.
pub fn export_esp_designs_csv(rep: &ElectrostaticsEspReport) -> String {
    let mut out = String::from(
        "rank,design_id,modification,mechanism,target_partner,base_ec_r,pred_ec_r,delta_ec_r,ddg_multipole_kcal,pred_pact,mw,clogp,sa_score,pains,score,smiles\n",
    );
    for d in &rep.esp_designs {
        out.push_str(&format!(
            "{},{},\"{}\",\"{}\",\"{}\",{:.3},{:.3},{:+.3},{:+.3},{:.3},{:.2},{:.2},{:.2},{},{:.1},\"{}\"\n",
            d.rank,
            d.design_id,
            d.modification_label,
            d.mechanism,
            d.target_partner,
            d.baseline_ec_r,
            d.predicted_ec_r,
            d.delta_ec_r,
            d.ddg_multipole_kcal,
            d.predicted_pact,
            d.mw,
            d.clogp,
            d.sa_score,
            d.pains_count,
            d.design_score,
            d.designed_smiles
        ));
    }
    out
}

/// Exports dataset-wide `ESP` complementarity screening results to `.csv`.
pub fn export_dataset_esp_screen_csv(rows: &[DatasetEspScreenRow]) -> String {
    let mut out = String::from(
        "rank,compound_id,compound_name,obs_pact,ec_integral,ec_pearson_r,halogen_bonds,chalcogen_contacts,pi_contacts,esp_clashes,total_multipole_dg_kcal,esp_refined_pocket_dg_kcal,top_esp_design\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{},{},\"{}\",{:.2},{:+.3},{:+.3},{},{},{},{},{:+.3},{:+.3},\"{}\"\n",
            r.rank,
            r.compound_id,
            r.compound_name,
            r.observed_pact,
            r.ec_integral,
            r.ec_pearson_r,
            r.halogen_bonds,
            r.chalcogen_contacts,
            r.pi_contacts,
            r.esp_clashes,
            r.total_multipole_dg_kcal,
            r.esp_refined_pocket_dg_kcal,
            r.top_esp_design
        ));
    }
    out
}

/// Formats an annotated `.pdb` containing aligned ligand `HETATM LIG` plus explicit `XED` `sigma-hole` (`XSH`), chalcogen (`XCH`), lone-pair (`XLP`), and `pi`-quadrupole (`XPI`) pseudoatoms (`B-factor = V_s`).
pub fn format_esp_multipole_pdb(rep: &ElectrostaticsEspReport) -> String {
    let mut out = format!(
        "REMARK   1 SAR-ATLAS v3.8.0 3D ESP & XED MULTIPOLE PSEUDOATOM COMPLEX\nREMARK   2 TARGET: {} (PDB {}) | LEAD: {} ({})\nREMARK   3 BAUER-MACKEY EC = {:+.3} | EC_r = {:+.3} | MULTIPOLE dG = {:+.3} KCAL/MOL\n",
        rep.target_name,
        rep.pdb_id,
        rep.compound_name,
        rep.compound_id,
        rep.bauer_mackey_ec_integral,
        rep.bauer_mackey_ec_pearson_r,
        rep.total_multipole_dg_kcal
    );
    let mut serial = 1_u32;
    for (pos, lbl, elem) in &rep.pocket_shell_atoms {
        let parts: Vec<&str> = lbl.split(':').collect();
        let res_part = parts.first().copied().unwrap_or("UNK1");
        let atom_nm = parts.get(1).copied().unwrap_or("CA");
        let res_nm = &res_part[..res_part.len().min(3)];
        let res_no: i32 = res_part[res_nm.len()..].parse().unwrap_or(1);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 18.00          {:>2}\n",
            serial,
            atom_nm,
            res_nm,
            res_no,
            pos[0],
            pos[1],
            pos[2],
            elem.symbol()
        ));
        serial += 1;
    }
    for (idx, &lp) in rep.aligned_ligand_coords3d.iter().enumerate() {
        let sym = rep
            .mol
            .atoms
            .get(idx)
            .map(|a| a.elem.symbol())
            .unwrap_or("C");
        let aname = format!("{}{}", sym, (idx + 1) % 100);
        out.push_str(&format!(
            "HETATM{:>5} {:<4} LIG L   1    {:>8.3}{:>8.3}{:>8.3}  1.00 12.00          {:>2}\n",
            serial, aname, lp[0], lp[1], lp[2], sym
        ));
        serial += 1;
    }
    for (mi, mp) in rep.multipole_pseudoatoms.iter().enumerate() {
        let res_tag = match mp.kind {
            MultipoleSiteKind::HalogenSigmaHoleCrown => "XSH",
            MultipoleSiteKind::ChalcogenSigmaStarLobe => "XCH",
            MultipoleSiteKind::DirectionalLonePairLobe => "XLP",
            MultipoleSiteKind::AromaticPiQuadrupolePole => "XPI",
        };
        out.push_str(&format!(
            "HETATM{:>5}  X   {:>3} X{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          He\n",
            serial,
            res_tag,
            mi + 1,
            mp.pos[0],
            mp.pos[1],
            mp.pos[2],
            mp.charge_e.clamp(-9.99, 9.99),
            mp.vs_extremum_kcal_e.clamp(-99.9, 99.9)
        ));
        serial += 1;
    }
    out.push_str("END\n");
    out
}
