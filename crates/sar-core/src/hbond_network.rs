//! Active-Site `His`/`Asp`/`Glu` Coupled Tautomeric & Protonation State Assigner (`PROPKA 3.1` +
//! `Protoss` / `ProtAssign`), `Asn`/`Gln`/`His` `180 deg` Terminal Flip Resolver (`Reduce` /
//! `NQ-Flipper`), Explicit Tetrahedral `H-O-H` Bridge-Water Orientation vs Displacement Solver,
//! and Wyman Proton-Linked `dG_HBN(pH)` Hydrogen-Bond Network Optimizer (`hbond_network.rs`, `v3.4.0`).
//!
//! Primary References:
//! - Bietz, Urbaczek, Schulz & Rarey (2014), *J. Cheminform.* 6:12 (`Protoss` coupled protein-ligand
//!   tautomeric/protonation modes, directional H-bonds, polar clash penalties & unsaturated polar
//!   minimization: `https://pmc.ncbi.nlm.nih.gov/articles/PMC4019353/`)
//! - Madhavi Sastry, Adzhigirey, Day, Annabhimoju & Sherman (2013), *J. Comput.-Aided Mol. Des.*
//!   27(3), 221-234 ("Protein and ligand preparation: parameters, protocols, and influence on virtual
//!   screening enrichments" — `ProtAssign` H-bond network optimization:
//!   `https://pubmed.ncbi.nlm.nih.gov/23579614/`)
//! - Word, Lovell, Richardson & Richardson (1999), *J. Mol. Biol.* 285(4), 1735-1747 ("Asparagine and
//!   glutamine: using hydrogen atom contacts in the choice of side-chain amide orientation" — `Reduce`
//!   `180 deg` `Asn`/`Gln`/`His` flips: `https://pubmed.ncbi.nlm.nih.gov/9917408/`) & Weichenberger &
//!   Sippl (2006), *Bioinformatics* 22(11), 1397-1398 (`NQ-Flipper`:
//!   `https://academic.oup.com/bioinformatics/article/22/11/1397/239014`)
//! - Søndergaard, Olsson, Rostkowski & Jensen (2011), *J. Chem. Theory Comput.* 7(7), 2284-2295
//!   (`PROPKA 3.1` coupled protein-ligand active-site micro-`pKa` shifts:
//!   `https://pubmed.ncbi.nlm.nih.gov/26606496/`)
//! - Barillari, Taylor, Viner & Essex (2007), *J. Am. Chem. Soc.* 129(9), 2577-2587 ("Classification
//!   of water molecules in protein binding sites": `https://pubmed.ncbi.nlm.nih.gov/17288418/`) &
//!   Ross, Morris & Biggin (2012), *PLoS ONE* 7(3):e32036 (`WaterDock` bridging vs displaced waters:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3291545/`).

use crate::allosteric_pocket::curated_allosteric_pdb_text;
use crate::fragment_grower::RT_LN10_KCAL;
use crate::mol::{Element, Molecule};
use crate::physchem::evaluate_physchem;
use crate::pka_protomer::analyze_pka_and_protomers;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    ParsedPdbStructure, PocketCavityAnalysis, WaterThermodynamicRole,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Standard `O-H` / `N-H` covalent bond length (`Angstrom`) for explicit polar proton placement.
pub const POLAR_H_BOND_LEN_A: f32 = 0.98;

/// Seven-point physiological & endosomal/acidic `pH` scan grid (`5.0..=8.0`) for Wyman proton linkage.
pub const HBN_PH_SCAN_POINTS: [f64; 7] = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0];

/// Curated benchmark complexes showcasing active-site tautomer, protonation, amide-flip & bridge-water networks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HBondNetworkPreset {
    /// `PDB 1M17` (`EGFR`): Conserved `HOH2001` `Thr790`/`Gln791` <-> quinazoline `N3` water bridge + `Met793` hinge + `Asp855`/`Glu762`.
    Egfr1m17Thr790WaterBridge,
    /// `PDB 3LN1` (`COX-2`): `His90` (`HID`/`HIE`/`HIP` + `180 deg` flip) + `Gln178` amide flip + `Arg513` + `HOH2201` sulfonamide network.
    Cox23ln1His90Gln178Sulfonamide,
    /// `PDB 6CM4` (`DRD2`): `Asp114` (`D3.32`) proton-coupled salt bridge + `HOH2102` `Tyr416` bridge + `Ser193`/`Thr119` rotors.
    Drd26cm4Asp114WaterSaltBridge,
    /// `PDB 6OIM` (`KRAS G12C`): `His95` cryptic groove tautomer/flip + `Gln99` `180 deg` amide flip + `Tyr96`/`Lys16` network.
    Kras6oimHis95Gln99PolarGroove,
    /// User-supplied `.pdb` structure with automatic polar residue & crystallographic bridge-water perception.
    CustomPdbHBondNetwork,
}

impl HBondNetworkPreset {
    pub const ALL: [Self; 5] = [
        Self::Egfr1m17Thr790WaterBridge,
        Self::Cox23ln1His90Gln178Sulfonamide,
        Self::Drd26cm4Asp114WaterSaltBridge,
        Self::Kras6oimHis95Gln99PolarGroove,
        Self::CustomPdbHBondNetwork,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::Egfr1m17Thr790WaterBridge => "HBN-1M17-EGFR-HOH",
            Self::Cox23ln1His90Gln178Sulfonamide => "HBN-3LN1-COX2-HIS90",
            Self::Drd26cm4Asp114WaterSaltBridge => "HBN-6CM4-DRD2-ASP114",
            Self::Kras6oimHis95Gln99PolarGroove => "HBN-6OIM-KRAS-HIS95",
            Self::CustomPdbHBondNetwork => "HBN-CUSTOM-PDB",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Egfr1m17Thr790WaterBridge => "EGFR 1M17 (Thr790+HOH)",
            Self::Cox23ln1His90Gln178Sulfonamide => "COX-2 3LN1 (His90+Gln178)",
            Self::Drd26cm4Asp114WaterSaltBridge => "DRD2 6CM4 (Asp114+HOH)",
            Self::Kras6oimHis95Gln99PolarGroove => "KRAS 6OIM (His95+Gln99)",
            Self::CustomPdbHBondNetwork => "Custom .PDB Network",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Egfr1m17Thr790WaterBridge => {
                "EGFR (1M17): Conserved HOH2001 Thr790/Gln791 Water Bridge, Met793 Hinge & Asp855/Glu762"
            }
            Self::Cox23ln1His90Gln178Sulfonamide => {
                "COX-2 (3LN1): His90 Tautomer/Flip, Gln178 180-deg Amide Flip, Arg513 & HOH2201 Sulfonamide Relay"
            }
            Self::Drd26cm4Asp114WaterSaltBridge => {
                "DRD2 (6CM4): Asp114 Proton-Linked Salt Bridge, HOH2102 Tyr416 Bridge & Ser193/Thr119 Rotors"
            }
            Self::Kras6oimHis95Gln99PolarGroove => {
                "KRAS G12C (6OIM): His95 Tautomer/180-deg Flip, Gln99 Amide Flip & Tyr96 Switch-II Polar Groove"
            }
            Self::CustomPdbHBondNetwork => {
                "Custom .PDB Active-Site His/Asp/Glu Tautomer, Amide-Flip & Bridge-Water H-Bond Optimizer"
            }
        }
    }
}

/// Functional category of a variable polar node (`VMR`) in the `Protoss` / `ProtAssign` network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HbnNodeKind {
    /// `HIS` (`HID`, `HIE`, `HIP+`) x `(0 deg Crystal, 180 deg Ring-Flip)` = 6 states.
    HistidineTautomerFlip,
    /// `ASP` / `GLU` (`Anionic COO-`, `Neutral syn-OD1H/OE1H`, `Neutral syn-OD2H/OE2H`) = 3 states.
    CarboxylateProtonation,
    /// `ASN` / `GLN` (`0 deg Crystal Amide`, `180 deg Flipped Amide O<->N`) = 2 states.
    AmideFlip180,
    /// `SER` / `THR` / `TYR` / `CYS` rotatable `-OH`/`-SH` donor + optional thiolate/phenolate = 3..4 states.
    HydroxylOrThiolRotor,
    /// Interfacial crystallographic water (`6` explicit tetrahedral `H-O-H` orientations + `DisplacedToBulk`) = 7 states.
    BridgeWaterSite,
    /// Coupled bound ligand protomer/tautomer state from `pka_protomer.rs`.
    CoupledLigandProtomer,
}

impl HbnNodeKind {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::HistidineTautomerFlip => "His HID/HIE/HIP+Flip",
            Self::CarboxylateProtonation => "Asp/Glu ASH/GLH",
            Self::AmideFlip180 => "Asn/Gln 180-Flip",
            Self::HydroxylOrThiolRotor => "Ser/Thr/Tyr/Cys Rotor",
            Self::BridgeWaterSite => "Bridge Water (H-O-H)",
            Self::CoupledLigandProtomer => "Ligand Taut/Protomer",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HistidineTautomerFlip => [192, 132, 252], // Violet
            Self::CarboxylateProtonation => [248, 113, 113], // Coral
            Self::AmideFlip180 => [251, 191, 36],            // Amber
            Self::HydroxylOrThiolRotor => [148, 163, 184],   // Slate
            Self::BridgeWaterSite => [56, 189, 248],         // Sky Blue
            Self::CoupledLigandProtomer => [52, 211, 153],   // Emerald
        }
    }
}

/// Single directional polar site (`Donor D-H` or `Acceptor :A`) carried by a node microstate or fixed anchor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolarVectorSite {
    pub label: String,
    pub heavy_pos: [f32; 3],
    /// Explicit proton `H` coordinate (`Some` for H-bond donor; `None` for pure H-bond acceptor).
    pub proton_pos: Option<[f32; 3]>,
    pub is_donor: bool,
    pub is_acceptor: bool,
    /// Formal ionic weight (`+1.0` cationic, `-1.0` anionic, `0.0` neutral) scaling salt-bridge strength.
    pub formal_charge: f32,
    pub is_buried: bool,
}

/// Candidate discrete microstate `m` of a variable polar node `i`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HbnNodeMicrostate {
    pub mode_idx: usize,
    pub state_code: String,
    pub description: String,
    /// Residue name emitted into prepared `.pdb` (`HID`, `HIE`, `HIP`, `ASH`, `GLH`, `CYM`, `HOH`, etc.).
    pub pdb_res_name: String,
    pub is_flipped_180: bool,
    pub is_water_displaced: bool,
    /// Protons bound relative to standard uncomplexed `pH 7.4` reference (`+1`, `0`, `-1`) for Wyman linkage.
    pub delta_protons_vs_ref: i8,
    /// Intrinsic `pH`-dependent tautomer / `PROPKA` protonation / `WaterMap` displacement free energy (`kcal/mol`).
    pub self_energy_kcal: f64,
    pub polar_sites: Vec<PolarVectorSite>,
}

/// Variable polar node (`His`, `Asp`, `Glu`, `Asn`, `Gln`, `Ser/Thr/Tyr/Cys`, `HOH`, or `Ligand`) with optimal assignment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HbnVariableNode {
    pub node_idx: usize,
    pub node_id: String,
    pub kind: HbnNodeKind,
    pub centroid: [f32; 3],
    /// `PROPKA 3.1`-style effective active-site micro-`pKa` (`Some` for titratable `His`/`Asp`/`Glu`/`Cys`/`Tyr`).
    pub propka_shifted_pka: Option<f64>,
    pub modes: Vec<HbnNodeMicrostate>,
    /// Unoptimized crystal/default mode index (`0`).
    pub naive_mode_idx: usize,
    /// Globally optimized mode index minimizing `E_total_HBN`.
    pub optimal_mode_idx: usize,
    /// Single-node free-energy gain `E(naive) - E(optimal)` (`kcal/mol`, positive = stabilization from switch!).
    pub node_stabilization_gain_kcal: f64,
}

/// Classification of an explicit polar interaction edge in the solved 3D hydrogen-bond network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HBondEdgeKind {
    ProteinLigandDirectHBond,
    IonicSaltBridge,
    WaterMediatedBridgeLeg,
    IntraPocketStructuralHBond,
    UnfavorablePolarClash,
}

impl HBondEdgeKind {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::ProteinLigandDirectHBond => "Prot-Lig H-Bond",
            Self::IonicSaltBridge => "Salt Bridge",
            Self::WaterMediatedBridgeLeg => "Bridge-Water H-Bond",
            Self::IntraPocketStructuralHBond => "Intra-Pocket H-Bond",
            Self::UnfavorablePolarClash => "Polar D-D/A-A Clash!",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::ProteinLigandDirectHBond => [52, 211, 153],   // Emerald
            Self::IonicSaltBridge => [192, 132, 252],           // Violet
            Self::WaterMediatedBridgeLeg => [56, 189, 248],     // Sky Blue
            Self::IntraPocketStructuralHBond => [148, 163, 184], // Slate
            Self::UnfavorablePolarClash => [248, 113, 113],     // Coral
        }
    }
}

/// Explicit 3D hydrogen-bond or polar-clash edge evaluated on the active network configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HBondNetworkEdge {
    pub donor_node_id: String,
    pub donor_atom_label: String,
    pub acceptor_node_id: String,
    pub acceptor_atom_label: String,
    pub donor_pos: [f32; 3],
    pub proton_pos: Option<[f32; 3]>,
    pub acceptor_pos: [f32; 3],
    pub distance_da_a: f32,
    pub angle_dha_deg: f32,
    pub energy_kcal: f64,
    pub kind: HBondEdgeKind,
}

/// Single point on the Wyman proton-linked binding curve `dG_HBN(pH)` (`pH = 5.0..8.0`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtonLinkedPhPoint {
    pub ph: f64,
    pub optimized_network_dg_kcal: f64,
    pub naive_network_dg_kcal: f64,
    /// Net protons taken up (`> 0`) or released (`< 0`) upon ligand binding at this `pH`.
    pub net_proton_uptake_dn_h: f64,
    pub dominant_his_asp_summary: String,
}

/// Complete `v3.4.0` Active-Site `His`/`Asp`/`Glu` Tautomer & Bridge-Water H-Bond Network Report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HBondNetworkReport {
    pub preset: HBondNetworkPreset,
    pub pdb_id: String,
    pub target_name: String,
    pub target_ph: f64,
    pub compound_id: String,
    pub compound_name: String,
    pub canonical_smiles: String,
    pub selected_ligand_state_label: String,
    #[serde(skip_deserializing)]
    pub solver_method: &'static str,
    pub states_evaluated: usize,

    // Naive Unprepared vs Globally Optimized H-Bond Network Energetics (`kcal/mol`)
    pub naive_network_dg_kcal: f64,
    pub optimized_network_dg_kcal: f64,
    /// Net stabilization from `His`/`Asp`/`Glu` tautomer/protonation, `180 deg` flips & bridge-water orientation (`negative = favorable gain!`).
    pub ddg_hbn_optimization_gain_kcal: f64,
    pub naive_polar_clashes: usize,
    pub optimized_polar_clashes: usize,
    pub direct_protein_ligand_hbonds: usize,
    pub retained_bridge_waters: usize,
    pub displaced_waters: usize,
    pub flipped_180_residues: usize,
    pub nonstandard_protonated_residues: usize,
    pub net_proton_uptake_at_ph: f64,
    pub hbn_adjusted_pocket_dg_kcal: f64,
    pub hbn_predicted_pkd: f64,

    pub nodes: Vec<HbnVariableNode>,
    pub naive_edges: Vec<HBondNetworkEdge>,
    pub optimized_edges: Vec<HBondNetworkEdge>,
    pub ph_linkage_curve: Vec<ProtonLinkedPhPoint>,
    pub aligned_ligand_coords3d: Vec<[f32; 3]>,
    pub pocket: PocketCavityAnalysis,
    pub mol: Molecule,
}

/// Compact row for dataset-wide H-Bond Network & Bridge-Water screening.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetHBondScreenRow {
    pub rank: usize,
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub observed_pact: f64,
    pub naive_network_dg_kcal: f64,
    pub optimized_network_dg_kcal: f64,
    pub ddg_hbn_gain_kcal: f64,
    pub hbn_adjusted_pocket_dg_kcal: f64,
    pub hbn_predicted_pkd: f64,
    pub direct_hbonds: usize,
    pub retained_bridge_waters: usize,
    pub flipped_or_taut_switched: usize,
    pub net_proton_uptake: f64,
    pub selected_ligand_state: String,
    pub canonical_smiles: String,
}

/// Configuration for the active-site H-bond & bridge-water network optimizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HBondNetworkConfig {
    pub preset: HBondNetworkPreset,
    pub custom_pdb_path: Option<String>,
    pub target_ph: f64,
    pub include_crystallographic_waters: bool,
    pub allow_180_flips: bool,
    pub allow_asp_glu_protonation: bool,
}

impl Default for HBondNetworkConfig {
    fn default() -> Self {
        Self {
            preset: HBondNetworkPreset::Egfr1m17Thr790WaterBridge,
            custom_pdb_path: None,
            target_ph: 7.4,
            include_crystallographic_waters: true,
            allow_180_flips: true,
            allow_asp_glu_protonation: true,
        }
    }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn add_scaled(p: [f32; 3], dir: [f32; 3], s: f32) -> [f32; 3] {
    let u = norm3(dir);
    [p[0] + u[0] * s, p[1] + u[1] * s, p[2] + u[2] * s]
}

/// Computes `angle(D - H ... A)` in degrees (`180 deg` = perfectly linear hydrogen bond).
fn angle_dha_deg(d_pos: [f32; 3], h_pos: [f32; 3], a_pos: [f32; 3]) -> f32 {
    let hd = norm3([d_pos[0] - h_pos[0], d_pos[1] - h_pos[1], d_pos[2] - h_pos[2]]);
    let ha = norm3([a_pos[0] - h_pos[0], a_pos[1] - h_pos[1], a_pos[2] - h_pos[2]]);
    let cos_a = (hd[0] * ha[0] + hd[1] * ha[1] + hd[2] * ha[2]).clamp(-1.0, 1.0);
    cos_a.acos().to_degrees()
}

/// Loads the base `.pdb` structure & optional pocket center override for `HBondNetworkConfig`.
pub fn load_hbond_network_pdb(
    cfg: &HBondNetworkConfig,
) -> (ParsedPdbStructure, Option<[f32; 3]>, String) {
    if cfg.preset == HBondNetworkPreset::CustomPdbHBondNetwork {
        if let Some(ref p) = cfg.custom_pdb_path {
            let t = p.trim();
            if !t.is_empty() {
                if let Ok(txt) = std::fs::read_to_string(t) {
                    if let Ok(parsed) = parse_pdb_structure(&txt) {
                        let name = parsed.pdb_id.clone();
                        return (parsed, None, name);
                    }
                }
            }
        }
    }
    match cfg.preset {
        HBondNetworkPreset::Egfr1m17Thr790WaterBridge
        | HBondNetworkPreset::CustomPdbHBondNetwork => (
            parse_pdb_structure(curated_pocket_pdb_text(0)).expect("Valid 1M17 PDB"),
            None,
            "EGFR".to_string(),
        ),
        HBondNetworkPreset::Cox23ln1His90Gln178Sulfonamide => (
            parse_pdb_structure(curated_pocket_pdb_text(2)).expect("Valid 3LN1 PDB"),
            None,
            "COX-2".to_string(),
        ),
        HBondNetworkPreset::Drd26cm4Asp114WaterSaltBridge => (
            parse_pdb_structure(curated_pocket_pdb_text(1)).expect("Valid 6CM4 PDB"),
            None,
            "DRD2".to_string(),
        ),
        HBondNetworkPreset::Kras6oimHis95Gln99PolarGroove => (
            parse_pdb_structure(curated_allosteric_pdb_text(1)).expect("Valid 6OIM PDB"),
            Some([12.90, 5.10, 8.20]),
            "KRAS-G12C".to_string(),
        ),
    }
}

/// Builds fixed background polar anchors (`ARG`, `LYS`, backbone `N`/`O`) plus all variable polar nodes
/// (`HIS`, `ASP`/`GLU`, `ASN`/`GLN`, `SER`/`THR`/`TYR`/`CYS`, interfacial `HOH` waters, & coupled `Ligand`).
#[allow(clippy::too_many_arguments)]
fn build_hbond_network_nodes(
    pdb: &ParsedPdbStructure,
    pocket: &PocketCavityAnalysis,
    mol: &Molecule,
    lig_coords: &[[f32; 3]],
    cfg: &HBondNetworkConfig,
    eval_ph: f64,
) -> (Vec<PolarVectorSite>, Vec<HbnVariableNode>) {
    let mut fixed_anchors: Vec<PolarVectorSite> = Vec::new();
    let mut var_nodes: Vec<HbnVariableNode> = Vec::new();

    // Group protein atoms by residue
    let mut by_res: BTreeMap<(char, i32, String), Vec<usize>> = BTreeMap::new();
    for (idx, a) in pdb.protein_atoms.iter().enumerate() {
        if dist3(a.pos, pocket.center) <= 10.5 {
            by_res
                .entry((a.chain_id, a.res_seq, a.res_name.clone()))
                .or_default()
                .push(idx);
        }
    }

    let count_enclosure = |pt: [f32; 3]| -> usize {
        let prot_n = pdb
            .protein_atoms
            .iter()
            .filter(|a| dist3(a.pos, pt) <= 4.6)
            .count();
        let lig_n = lig_coords
            .iter()
            .filter(|&&lp| dist3(lp, pt) <= 4.4)
            .count();
        prot_n + lig_n
    };

    for ((_chain, res_seq, res_name), idxs) in &by_res {
        let res_id = format!("{}{}", res_name, res_seq);

        // Extract backbone N / O as fixed H-bond anchors first
        for &ai in idxs {
            let a = &pdb.protein_atoms[ai];
            let an = a.atom_name.trim();
            if an == "N" && res_name != "PRO" {
                let h_dir = norm3([
                    pocket.center[0] - a.pos[0],
                    pocket.center[1] - a.pos[1],
                    pocket.center[2] - a.pos[2],
                ]);
                fixed_anchors.push(PolarVectorSite {
                    label: format!("{res_id}:N"),
                    heavy_pos: a.pos,
                    proton_pos: Some(add_scaled(a.pos, h_dir, POLAR_H_BOND_LEN_A)),
                    is_donor: true,
                    is_acceptor: false,
                    formal_charge: 0.0,
                    is_buried: count_enclosure(a.pos) >= 6,
                });
            } else if an == "O" {
                fixed_anchors.push(PolarVectorSite {
                    label: format!("{res_id}:O"),
                    heavy_pos: a.pos,
                    proton_pos: None,
                    is_donor: false,
                    is_acceptor: true,
                    formal_charge: 0.0,
                    is_buried: count_enclosure(a.pos) >= 6,
                });
            }
        }

        // Fixed cationic side chains (`LYS`, `ARG`, `TRP NE1`)
        if res_name == "LYS" {
            for &ai in idxs {
                let a = &pdb.protein_atoms[ai];
                if a.atom_name.trim() == "NZ" {
                    let h_dir = norm3([
                        pocket.center[0] - a.pos[0],
                        pocket.center[1] - a.pos[1],
                        pocket.center[2] - a.pos[2],
                    ]);
                    fixed_anchors.push(PolarVectorSite {
                        label: format!("{res_id}:NZ+"),
                        heavy_pos: a.pos,
                        proton_pos: Some(add_scaled(a.pos, h_dir, POLAR_H_BOND_LEN_A)),
                        is_donor: true,
                        is_acceptor: false,
                        formal_charge: 1.0,
                        is_buried: count_enclosure(a.pos) >= 6,
                    });
                }
            }
            continue;
        }
        if res_name == "ARG" {
            for &ai in idxs {
                let a = &pdb.protein_atoms[ai];
                if matches!(a.atom_name.trim(), "NH1" | "NH2" | "NE") {
                    let h_dir = norm3([
                        pocket.center[0] - a.pos[0],
                        pocket.center[1] - a.pos[1],
                        pocket.center[2] - a.pos[2],
                    ]);
                    fixed_anchors.push(PolarVectorSite {
                        label: format!("{res_id}:{}", a.atom_name.trim()),
                        heavy_pos: a.pos,
                        proton_pos: Some(add_scaled(a.pos, h_dir, POLAR_H_BOND_LEN_A)),
                        is_donor: true,
                        is_acceptor: false,
                        formal_charge: 0.5,
                        is_buried: count_enclosure(a.pos) >= 6,
                    });
                }
            }
            continue;
        }

        // 1. Variable HISTIDINE (`HID`, `HIE`, `HIP+`) x (`0 deg`, `180 deg` Ring Flip)
        if matches!(res_name.as_str(), "HIS" | "HID" | "HIE" | "HIP") {
            let nd1_opt = idxs
                .iter()
                .copied()
                .find(|&i| pdb.protein_atoms[i].atom_name.trim() == "ND1");
            let ne2_opt = idxs
                .iter()
                .copied()
                .find(|&i| pdb.protein_atoms[i].atom_name.trim() == "NE2");
            let any_sc = idxs
                .iter()
                .copied()
                .find(|&i| !matches!(pdb.protein_atoms[i].atom_name.trim(), "N" | "CA" | "C" | "O"));

            let (p_nd1, p_ne2) = match (nd1_opt, ne2_opt, any_sc) {
                (Some(i1), Some(i2), _) => (pdb.protein_atoms[i1].pos, pdb.protein_atoms[i2].pos),
                (Some(i1), None, _) => {
                    let p1 = pdb.protein_atoms[i1].pos;
                    (p1, [p1[0] + 1.35, p1[1] - 1.15, p1[2] + 0.65])
                }
                (None, Some(i2), _) => {
                    let p2 = pdb.protein_atoms[i2].pos;
                    ([p2[0] - 1.35, p2[1] + 1.15, p2[2] - 0.65], p2)
                }
                (None, None, Some(ic)) => {
                    let pc = pdb.protein_atoms[ic].pos;
                    (
                        [pc[0] - 0.72, pc[1] + 0.65, pc[2] - 0.30],
                        [pc[0] + 0.78, pc[1] - 0.62, pc[2] + 0.35],
                    )
                }
                _ => continue,
            };

            let cent = [
                0.5 * (p_nd1[0] + p_ne2[0]),
                0.5 * (p_nd1[1] + p_ne2[1]),
                0.5 * (p_nd1[2] + p_ne2[2]),
            ];
            let enc = count_enclosure(cent);
            let buried = enc >= 6;
            // PROPKA 3.1: hydrophobic burial lowers His+ pKa slightly unless stabilized by nearby Asp/Glu
            let near_anion = pdb.protein_atoms.iter().any(|a| {
                matches!(a.res_name.as_str(), "ASP" | "GLU")
                    && matches!(a.atom_name.trim(), "OD1" | "OD2" | "OE1" | "OE2")
                    && dist3(a.pos, cent) <= 4.5
            });
            let pka_his = 6.50 + (if near_anion { 1.15 } else { 0.0 }) - (if enc >= 9 { 0.45 } else { 0.0 });
            let dg_hip = RT_LN10_KCAL * (eval_ph - pka_his);

            let mut modes = Vec::new();
            let flips: &[(bool, [f32; 3], [f32; 3], f64, &str)] = if cfg.allow_180_flips {
                &[
                    (false, p_nd1, p_ne2, 0.0, ""),
                    (true, p_ne2, p_nd1, 0.10, "-Flip180"),
                ]
            } else {
                &[(false, p_nd1, p_ne2, 0.0, "")]
            };

            for &(flipped, pos_d1, pos_e2, flip_cost, suf) in flips {
                let dir_d1 = norm3([pos_d1[0] - cent[0], pos_d1[1] - cent[1], pos_d1[2] - cent[2]]);
                let dir_e2 = norm3([pos_e2[0] - cent[0], pos_e2[1] - cent[1], pos_e2[2] - cent[2]]);

                // HIE: NE2-H donor + ND1 acceptor (slightly more stable intrinsic tautomer in water)
                modes.push(HbnNodeMicrostate {
                    mode_idx: modes.len(),
                    state_code: format!("HIE{suf}"),
                    description: format!("Neutral Ne2-H tautomer{}", if flipped { " (180-deg ring flip)" } else { "" }),
                    pdb_res_name: "HIE".to_string(),
                    is_flipped_180: flipped,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 0,
                    self_energy_kcal: flip_cost,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{res_id}:ND1"),
                            heavy_pos: pos_d1,
                            proton_pos: None,
                            is_donor: false,
                            is_acceptor: true,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                        PolarVectorSite {
                            label: format!("{res_id}:NE2-H"),
                            heavy_pos: pos_e2,
                            proton_pos: Some(add_scaled(pos_e2, dir_e2, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                    ],
                });

                // HID: ND1-H donor + NE2 acceptor
                modes.push(HbnNodeMicrostate {
                    mode_idx: modes.len(),
                    state_code: format!("HID{suf}"),
                    description: format!("Neutral Nd1-H tautomer{}", if flipped { " (180-deg ring flip)" } else { "" }),
                    pdb_res_name: "HID".to_string(),
                    is_flipped_180: flipped,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 0,
                    self_energy_kcal: 0.22 + flip_cost,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{res_id}:ND1-H"),
                            heavy_pos: pos_d1,
                            proton_pos: Some(add_scaled(pos_d1, dir_d1, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                        PolarVectorSite {
                            label: format!("{res_id}:NE2"),
                            heavy_pos: pos_e2,
                            proton_pos: None,
                            is_donor: false,
                            is_acceptor: true,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                    ],
                });

                // HIP+: Dual ND1-H + NE2-H imidazolium cation
                modes.push(HbnNodeMicrostate {
                    mode_idx: modes.len(),
                    state_code: format!("HIP+{suf}"),
                    description: format!("Diprotic imidazolium cation (+1){}", if flipped { " (180-deg ring flip)" } else { "" }),
                    pdb_res_name: "HIP".to_string(),
                    is_flipped_180: flipped,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 1,
                    self_energy_kcal: dg_hip + flip_cost,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{res_id}:ND1-H+"),
                            heavy_pos: pos_d1,
                            proton_pos: Some(add_scaled(pos_d1, dir_d1, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.5,
                            is_buried: buried,
                        },
                        PolarVectorSite {
                            label: format!("{res_id}:NE2-H+"),
                            heavy_pos: pos_e2,
                            proton_pos: Some(add_scaled(pos_e2, dir_e2, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.5,
                            is_buried: buried,
                        },
                    ],
                });
            }

            var_nodes.push(HbnVariableNode {
                node_idx: var_nodes.len(),
                node_id: res_id,
                kind: HbnNodeKind::HistidineTautomerFlip,
                centroid: cent,
                propka_shifted_pka: Some(pka_his),
                modes,
                naive_mode_idx: 0,
                optimal_mode_idx: 0,
                node_stabilization_gain_kcal: 0.0,
            });
            continue;
        }

        // 2. Variable CARBOXYLATES (`ASP` -> `ASP-`/`ASH`, `GLU` -> `GLU-`/`GLH`)
        if matches!(res_name.as_str(), "ASP" | "GLU") {
            let o1_opt = idxs.iter().copied().find(|&i| {
                matches!(pdb.protein_atoms[i].atom_name.trim(), "OD1" | "OE1")
            });
            let o2_opt = idxs.iter().copied().find(|&i| {
                matches!(pdb.protein_atoms[i].atom_name.trim(), "OD2" | "OE2")
            });
            let (p_o1, p_o2) = match (o1_opt, o2_opt) {
                (Some(i1), Some(i2)) => (pdb.protein_atoms[i1].pos, pdb.protein_atoms[i2].pos),
                (Some(i1), None) => {
                    let p1 = pdb.protein_atoms[i1].pos;
                    (p1, [p1[0] + 1.25, p1[1] + 1.65, p1[2] - 0.35])
                }
                (None, Some(i2)) => {
                    let p2 = pdb.protein_atoms[i2].pos;
                    ([p2[0] - 1.25, p2[1] - 1.65, p2[2] + 0.35], p2)
                }
                _ => continue,
            };

            let cent = [
                0.5 * (p_o1[0] + p_o2[0]),
                0.5 * (p_o1[1] + p_o2[1]),
                0.5 * (p_o1[2] + p_o2[2]),
            ];
            let enc = count_enclosure(cent);
            let buried = enc >= 6;
            let base_pka = if res_name == "ASP" { 3.85 } else { 4.45 };
            // PROPKA 3.1: buried carboxylates or Asp/Glu near electronegative ligand oxygens experience large upward pKa shifts!
            let burial_shift = ((enc.saturating_sub(4) as f64) * 0.42).min(2.60);
            let pka_eff = base_pka + burial_shift;
            let dg_neutral = RT_LN10_KCAL * (eval_ph - pka_eff);
            let prot_res_code = if res_name == "ASP" { "ASH" } else { "GLH" };
            let to_pocket_1 = norm3([
                pocket.center[0] - p_o1[0],
                pocket.center[1] - p_o1[1],
                pocket.center[2] - p_o1[2],
            ]);
            let to_pocket_2 = norm3([
                pocket.center[0] - p_o2[0],
                pocket.center[1] - p_o2[1],
                pocket.center[2] - p_o2[2],
            ]);

            let mut modes = vec![HbnNodeMicrostate {
                mode_idx: 0,
                state_code: format!("{res_name}- (COO-)"),
                description: "Deprotonated anionic carboxylate (-1 dual H-bond acceptor)".to_string(),
                pdb_res_name: res_name.clone(),
                is_flipped_180: false,
                is_water_displaced: false,
                delta_protons_vs_ref: 0,
                self_energy_kcal: 0.0,
                polar_sites: vec![
                    PolarVectorSite {
                        label: format!("{res_id}:O1-"),
                        heavy_pos: p_o1,
                        proton_pos: None,
                        is_donor: false,
                        is_acceptor: true,
                        formal_charge: -0.5,
                        is_buried: buried,
                    },
                    PolarVectorSite {
                        label: format!("{res_id}:O2-"),
                        heavy_pos: p_o2,
                        proton_pos: None,
                        is_donor: false,
                        is_acceptor: true,
                        formal_charge: -0.5,
                        is_buried: buried,
                    },
                ],
            }];

            if cfg.allow_asp_glu_protonation {
                modes.push(HbnNodeMicrostate {
                    mode_idx: 1,
                    state_code: format!("{prot_res_code}-O1H"),
                    description: "Protonated neutral carboxylic acid (O1-H donor + O2 acceptor)".to_string(),
                    pdb_res_name: prot_res_code.to_string(),
                    is_flipped_180: false,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 1,
                    self_energy_kcal: dg_neutral,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{res_id}:O1-H"),
                            heavy_pos: p_o1,
                            proton_pos: Some(add_scaled(p_o1, to_pocket_1, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                        PolarVectorSite {
                            label: format!("{res_id}:O2"),
                            heavy_pos: p_o2,
                            proton_pos: None,
                            is_donor: false,
                            is_acceptor: true,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                    ],
                });
                modes.push(HbnNodeMicrostate {
                    mode_idx: 2,
                    state_code: format!("{prot_res_code}-O2H"),
                    description: "Protonated neutral carboxylic acid (O2-H donor + O1 acceptor)".to_string(),
                    pdb_res_name: prot_res_code.to_string(),
                    is_flipped_180: false,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 1,
                    self_energy_kcal: dg_neutral,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{res_id}:O1"),
                            heavy_pos: p_o1,
                            proton_pos: None,
                            is_donor: false,
                            is_acceptor: true,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                        PolarVectorSite {
                            label: format!("{res_id}:O2-H"),
                            heavy_pos: p_o2,
                            proton_pos: Some(add_scaled(p_o2, to_pocket_2, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                    ],
                });
            }

            var_nodes.push(HbnVariableNode {
                node_idx: var_nodes.len(),
                node_id: res_id,
                kind: HbnNodeKind::CarboxylateProtonation,
                centroid: cent,
                propka_shifted_pka: Some(pka_eff),
                modes,
                naive_mode_idx: 0,
                optimal_mode_idx: 0,
                node_stabilization_gain_kcal: 0.0,
            });
            continue;
        }

        // 3. Variable ASPARAGINE / GLUTAMINE (`ASN`, `GLN`) 180-deg Amide Flip (`Word et al. 1999 Reduce`)
        if matches!(res_name.as_str(), "ASN" | "GLN") {
            let o_opt = idxs.iter().copied().find(|&i| {
                matches!(pdb.protein_atoms[i].atom_name.trim(), "OD1" | "OE1")
            });
            let n_opt = idxs.iter().copied().find(|&i| {
                matches!(pdb.protein_atoms[i].atom_name.trim(), "ND2" | "NE2")
            });
            if o_opt.is_none() && n_opt.is_none() {
                continue;
            }
            let (p_o, p_n) = match (o_opt, n_opt) {
                (Some(io), Some(in2)) => (pdb.protein_atoms[io].pos, pdb.protein_atoms[in2].pos),
                (Some(io), None) => {
                    let po = pdb.protein_atoms[io].pos;
                    (po, [po[0] - 1.65, po[1] + 1.35, po[2] - 0.45])
                }
                (None, Some(in2)) => {
                    let pn = pdb.protein_atoms[in2].pos;
                    ([pn[0] + 1.65, pn[1] - 1.35, pn[2] + 0.45], pn)
                }
                _ => unreachable!(),
            };
            let cent = [
                0.5 * (p_o[0] + p_n[0]),
                0.5 * (p_o[1] + p_n[1]),
                0.5 * (p_o[2] + p_n[2]),
            ];
            let buried = count_enclosure(cent) >= 6;
            let dir_n_cryst = norm3([
                pocket.center[0] - p_n[0],
                pocket.center[1] - p_n[1],
                pocket.center[2] - p_n[2],
            ]);
            let dir_n_flip = norm3([
                pocket.center[0] - p_o[0],
                pocket.center[1] - p_o[1],
                pocket.center[2] - p_o[2],
            ]);

            let mut modes = vec![HbnNodeMicrostate {
                mode_idx: 0,
                state_code: "Crystal-0deg".to_string(),
                description: "Crystallographic amide orientation (O acceptor / NH2 donor)".to_string(),
                pdb_res_name: res_name.clone(),
                is_flipped_180: false,
                is_water_displaced: false,
                delta_protons_vs_ref: 0,
                self_energy_kcal: 0.0,
                polar_sites: vec![
                    PolarVectorSite {
                        label: format!("{res_id}:O"),
                        heavy_pos: p_o,
                        proton_pos: None,
                        is_donor: false,
                        is_acceptor: true,
                        formal_charge: 0.0,
                        is_buried: buried,
                    },
                    PolarVectorSite {
                        label: format!("{res_id}:NH2"),
                        heavy_pos: p_n,
                        proton_pos: Some(add_scaled(p_n, dir_n_cryst, POLAR_H_BOND_LEN_A)),
                        is_donor: true,
                        is_acceptor: false,
                        formal_charge: 0.0,
                        is_buried: buried,
                    },
                ],
            }];

            if cfg.allow_180_flips {
                modes.push(HbnNodeMicrostate {
                    mode_idx: 1,
                    state_code: "Flip-180deg".to_string(),
                    description: "180-deg flipped terminal amide swapping O <-> NH2 (Reduce / NQ-Flipper)".to_string(),
                    pdb_res_name: res_name.clone(),
                    is_flipped_180: true,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 0,
                    self_energy_kcal: 0.06,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{res_id}:O(flip)"),
                            heavy_pos: p_n,
                            proton_pos: None,
                            is_donor: false,
                            is_acceptor: true,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                        PolarVectorSite {
                            label: format!("{res_id}:NH2(flip)"),
                            heavy_pos: p_o,
                            proton_pos: Some(add_scaled(p_o, dir_n_flip, POLAR_H_BOND_LEN_A)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.0,
                            is_buried: buried,
                        },
                    ],
                });
            }

            var_nodes.push(HbnVariableNode {
                node_idx: var_nodes.len(),
                node_id: res_id,
                kind: HbnNodeKind::AmideFlip180,
                centroid: cent,
                propka_shifted_pka: None,
                modes,
                naive_mode_idx: 0,
                optimal_mode_idx: 0,
                node_stabilization_gain_kcal: 0.0,
            });
            continue;
        }

        // 4. Variable HYDROXYL / THIOL ROTORS (`SER`, `THR`, `TYR`, `CYS`)
        if matches!(res_name.as_str(), "SER" | "THR" | "TYR" | "CYS") {
            let pol_opt = idxs.iter().copied().find(|&i| {
                matches!(pdb.protein_atoms[i].atom_name.trim(), "OG" | "OG1" | "OH" | "SG")
            });
            let Some(ip) = pol_opt else {
                continue;
            };
            let pos_x = pdb.protein_atoms[ip].pos;
            let buried = count_enclosure(pos_x) >= 6;
            let base_dir = norm3([
                pocket.center[0] - pos_x[0],
                pocket.center[1] - pos_x[1],
                pocket.center[2] - pos_x[2],
            ]);
            let dirs: [[f32; 3]; 3] = [
                base_dir,
                norm3([-base_dir[1] + 0.35, base_dir[0], base_dir[2] * 0.6]),
                norm3([base_dir[1], -base_dir[0] + 0.35, -base_dir[2] * 0.6]),
            ];
            let rot_names = ["Rotor-g+", "Rotor-t", "Rotor-g-"];
            let mut modes: Vec<HbnNodeMicrostate> = dirs
                .iter()
                .enumerate()
                .map(|(mi, &d)| HbnNodeMicrostate {
                    mode_idx: mi,
                    state_code: rot_names[mi].to_string(),
                    description: format!("Rotatable polar proton vector ({})", rot_names[mi]),
                    pdb_res_name: res_name.clone(),
                    is_flipped_180: false,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 0,
                    self_energy_kcal: if mi == 0 { 0.0 } else { 0.04 },
                    polar_sites: vec![PolarVectorSite {
                        label: format!("{res_id}:{}", pdb.protein_atoms[ip].atom_name.trim()),
                        heavy_pos: pos_x,
                        proton_pos: Some(add_scaled(pos_x, d, POLAR_H_BOND_LEN_A)),
                        is_donor: true,
                        is_acceptor: true,
                        formal_charge: 0.0,
                        is_buried: buried,
                    }],
                })
                .collect();

            let pka_opt = if res_name == "CYS" {
                let pka_cys = 8.30;
                let dg_cym = RT_LN10_KCAL * (pka_cys - eval_ph);
                modes.push(HbnNodeMicrostate {
                    mode_idx: 3,
                    state_code: "CYM- (S-)".to_string(),
                    description: "Deprotonated nucleophilic thiolate anion (-1)".to_string(),
                    pdb_res_name: "CYM".to_string(),
                    is_flipped_180: false,
                    is_water_displaced: false,
                    delta_protons_vs_ref: -1,
                    self_energy_kcal: dg_cym,
                    polar_sites: vec![PolarVectorSite {
                        label: format!("{res_id}:SG-"),
                        heavy_pos: pos_x,
                        proton_pos: None,
                        is_donor: false,
                        is_acceptor: true,
                        formal_charge: -1.0,
                        is_buried: buried,
                    }],
                });
                Some(pka_cys)
            } else {
                None
            };

            var_nodes.push(HbnVariableNode {
                node_idx: var_nodes.len(),
                node_id: res_id,
                kind: HbnNodeKind::HydroxylOrThiolRotor,
                centroid: pos_x,
                propka_shifted_pka: pka_opt,
                modes,
                naive_mode_idx: 0,
                optimal_mode_idx: 0,
                node_stabilization_gain_kcal: 0.0,
            });
        }
    }

    // 5. Variable BRIDGE-WATER NODES (`6` Tetrahedral `H-O-H` Orientations + `DisplacedToBulk`)
    if cfg.include_crystallographic_waters {
        for hs in pocket.hydration_sites.iter().take(5) {
            let min_lig_d = lig_coords
                .iter()
                .map(|&lp| dist3(lp, hs.pos))
                .fold(f32::INFINITY, f32::min);
            if min_lig_d > 5.4 {
                continue;
            }
            let sterically_overlapped = min_lig_d < 2.05;

            // Construct 4 tetrahedral unit spokes around water oxygen `hs.pos`
            let u0 = norm3([
                pocket.center[0] - hs.pos[0],
                pocket.center[1] - hs.pos[1],
                pocket.center[2] - hs.pos[2],
            ]);
            let t1 = norm3([-u0[1] + 0.25, u0[0] + 0.15, 0.45]);
            let t2 = norm3([
                u0[1] * t1[2] - u0[2] * t1[1],
                u0[2] * t1[0] - u0[0] * t1[2],
                u0[0] * t1[1] - u0[1] * t1[0],
            ]);
            let spokes: [[f32; 3]; 4] = [
                u0,
                norm3([-0.33 * u0[0] + 0.94 * t1[0], -0.33 * u0[1] + 0.94 * t1[1], -0.33 * u0[2] + 0.94 * t1[2]]),
                norm3([
                    -0.33 * u0[0] - 0.47 * t1[0] + 0.82 * t2[0],
                    -0.33 * u0[1] - 0.47 * t1[1] + 0.82 * t2[1],
                    -0.33 * u0[2] - 0.47 * t1[2] + 0.82 * t2[2],
                ]),
                norm3([
                    -0.33 * u0[0] - 0.47 * t1[0] - 0.82 * t2[0],
                    -0.33 * u0[1] - 0.47 * t1[1] - 0.82 * t2[1],
                    -0.33 * u0[2] - 0.47 * t1[2] - 0.82 * t2[2],
                ]),
            ];
            let pairs: [(usize, usize, &str); 6] = [
                (0, 1, "HOH-Tet(1,2)"),
                (0, 2, "HOH-Tet(1,3)"),
                (0, 3, "HOH-Tet(1,4)"),
                (1, 2, "HOH-Tet(2,3)"),
                (1, 3, "HOH-Tet(2,4)"),
                (2, 3, "HOH-Tet(3,4)"),
            ];

            let retain_clash_pen = if sterically_overlapped {
                5.50 * ((2.15 - min_lig_d) as f64)
            } else {
                match hs.role {
                    WaterThermodynamicRole::ConservedStructuralBridge => -0.55,
                    WaterThermodynamicRole::UnhappyDisplaceableHotspot => 0.65,
                    WaterThermodynamicRole::ExchangeableBulkRim => 0.15,
                }
            };

            let mut modes: Vec<HbnNodeMicrostate> = pairs
                .iter()
                .enumerate()
                .map(|(mi, &(sa, sb, lbl))| HbnNodeMicrostate {
                    mode_idx: mi,
                    state_code: lbl.to_string(),
                    description: format!(
                        "Retained crystallographic bridge water ({}, B={:.1})",
                        hs.role.short_label(),
                        hs.b_factor
                    ),
                    pdb_res_name: "HOH".to_string(),
                    is_flipped_180: false,
                    is_water_displaced: false,
                    delta_protons_vs_ref: 0,
                    self_energy_kcal: retain_clash_pen,
                    polar_sites: vec![
                        PolarVectorSite {
                            label: format!("{}:H1", hs.site_id),
                            heavy_pos: hs.pos,
                            proton_pos: Some(add_scaled(hs.pos, spokes[sa], 0.96)),
                            is_donor: true,
                            is_acceptor: true,
                            formal_charge: 0.0,
                            is_buried: true,
                        },
                        PolarVectorSite {
                            label: format!("{}:H2", hs.site_id),
                            heavy_pos: hs.pos,
                            proton_pos: Some(add_scaled(hs.pos, spokes[sb], 0.96)),
                            is_donor: true,
                            is_acceptor: false,
                            formal_charge: 0.0,
                            is_buried: false,
                        },
                    ],
                })
                .collect();

            // Mode 6: Displaced into bulk solvent (pays liberation penalty if conserved bridge, gains bonus if unhappy!)
            let disp_e = if sterically_overlapped {
                hs.displacement_ddg_bind_kcal
            } else {
                hs.displacement_ddg_bind_kcal + 0.75
            };
            modes.push(HbnNodeMicrostate {
                mode_idx: 6,
                state_code: "Displaced-Bulk".to_string(),
                description: format!(
                    "Displaced into bulk solvent (WaterMap ddG_disp = {:+.2} kcal/mol)",
                    hs.displacement_ddg_bind_kcal
                ),
                pdb_res_name: "DIS".to_string(),
                is_flipped_180: false,
                is_water_displaced: true,
                delta_protons_vs_ref: 0,
                self_energy_kcal: disp_e,
                polar_sites: Vec::new(),
            });

            var_nodes.push(HbnVariableNode {
                node_idx: var_nodes.len(),
                node_id: hs.site_id.clone(),
                kind: HbnNodeKind::BridgeWaterSite,
                centroid: hs.pos,
                propka_shifted_pka: None,
                modes,
                naive_mode_idx: 0,
                optimal_mode_idx: 0,
                node_stabilization_gain_kcal: 0.0,
            });
        }
    }

    // 6. Coupled Bound LIGAND Protomer & Tautomer Microstates (`pka_protomer.rs`)
    let pka_rep = analyze_pka_and_protomers(mol, eval_ph);
    let ref_charge_74 = analyze_pka_and_protomers(mol, 7.4)
        .protomers
        .first()
        .map(|p| p.net_charge)
        .unwrap_or(0);

    let mut lig_modes: Vec<HbnNodeMicrostate> = Vec::new();
    for prot in pka_rep.protomers.iter().take(3) {
        let frac = prot.fraction_at_target_ph.clamp(1e-4, 1.0);
        let dg_state = -0.592_484 * frac.ln();
        let sites = extract_ligand_3d_polar_sites(&prot.mol, lig_coords, &pocket.center);
        lig_modes.push(HbnNodeMicrostate {
            mode_idx: lig_modes.len(),
            state_code: format!("Prot(q={:+})", prot.net_charge),
            description: format!(
                "{} ({:.1}% in bulk at pH {:.1})",
                prot.state_label,
                frac * 100.0,
                eval_ph
            ),
            pdb_res_name: "LIG".to_string(),
            is_flipped_180: false,
            is_water_displaced: false,
            delta_protons_vs_ref: prot.net_charge.saturating_sub(ref_charge_74),
            self_energy_kcal: dg_state,
            polar_sites: sites,
        });
    }
    for taut in pka_rep.tautomers.iter().skip(1).take(2) {
        if taut.mol.atoms.len() == mol.atoms.len() {
            let sites = extract_ligand_3d_polar_sites(&taut.mol, lig_coords, &pocket.center);
            lig_modes.push(HbnNodeMicrostate {
                mode_idx: lig_modes.len(),
                state_code: format!("Taut#{}", taut.rank),
                description: format!(
                    "MolVS tautomer ({}, +{:+.2} kcal/mol)",
                    taut.rule_name, taut.rel_energy_kcal
                ),
                pdb_res_name: "LIG".to_string(),
                is_flipped_180: false,
                is_water_displaced: false,
                delta_protons_vs_ref: 0,
                self_energy_kcal: taut.rel_energy_kcal.max(0.15),
                polar_sites: sites,
            });
        }
    }
    if lig_modes.is_empty() {
        lig_modes.push(HbnNodeMicrostate {
            mode_idx: 0,
            state_code: "Lig-Neutral".to_string(),
            description: "Input ligand state".to_string(),
            pdb_res_name: "LIG".to_string(),
            is_flipped_180: false,
            is_water_displaced: false,
            delta_protons_vs_ref: 0,
            self_energy_kcal: 0.0,
            polar_sites: extract_ligand_3d_polar_sites(mol, lig_coords, &pocket.center),
        });
    }

    let mut lig_cent = pocket.center;
    if !lig_coords.is_empty() {
        let inv = 1.0 / (lig_coords.len() as f32);
        lig_cent = [0.0, 0.0, 0.0];
        for p in lig_coords {
            lig_cent[0] += p[0] * inv;
            lig_cent[1] += p[1] * inv;
            lig_cent[2] += p[2] * inv;
        }
    }

    var_nodes.push(HbnVariableNode {
        node_idx: var_nodes.len(),
        node_id: "LIGAND".to_string(),
        kind: HbnNodeKind::CoupledLigandProtomer,
        centroid: lig_cent,
        propka_shifted_pka: pka_rep.sites.first().map(|s| s.micro_pka),
        modes: lig_modes,
        naive_mode_idx: 0,
        optimal_mode_idx: 0,
        node_stabilization_gain_kcal: 0.0,
    });

    (fixed_anchors, var_nodes)
}

fn extract_ligand_3d_polar_sites(
    mol: &Molecule,
    coords3d: &[[f32; 3]],
    pocket_center: &[f32; 3],
) -> Vec<PolarVectorSite> {
    let mut out = Vec::new();
    for (ai, a) in mol.atoms.iter().enumerate() {
        if ai >= coords3d.len() || !matches!(a.elem, Element::N | Element::O | Element::S) {
            continue;
        }
        let pos = coords3d[ai];
        let is_donor = a.total_h() > 0 || a.charge > 0;
        let is_acceptor = (a.elem == Element::O && a.charge <= 0)
            || (a.elem == Element::N && a.charge <= 0 && a.total_h() <= 1)
            || (a.elem == Element::S && a.charge < 0);
        if !is_donor && !is_acceptor {
            continue;
        }

        // Compute outward covalent vector away from bonded ligand heavy neighbors
        let mut sum_nb = [0.0f32; 3];
        let mut cnt = 0usize;
        if let Some(nbrs) = mol.adj.get(ai) {
            for &(nb, _) in nbrs {
                if nb < coords3d.len() {
                    let d = norm3([
                        coords3d[nb][0] - pos[0],
                        coords3d[nb][1] - pos[1],
                        coords3d[nb][2] - pos[2],
                    ]);
                    sum_nb[0] += d[0];
                    sum_nb[1] += d[1];
                    sum_nb[2] += d[2];
                    cnt += 1;
                }
            }
        }
        let out_dir = if cnt > 0 {
            norm3([-sum_nb[0], -sum_nb[1], -sum_nb[2]])
        } else {
            norm3([
                pos[0] - pocket_center[0],
                pos[1] - pocket_center[1],
                pos[2] - pocket_center[2],
            ])
        };

        out.push(PolarVectorSite {
            label: format!("LIG:{}{}", a.elem.symbol(), ai + 1),
            heavy_pos: pos,
            proton_pos: is_donor.then(|| add_scaled(pos, out_dir, POLAR_H_BOND_LEN_A)),
            is_donor,
            is_acceptor,
            formal_charge: a.charge as f32,
            is_buried: true,
        });
    }
    out
}

/// Evaluates pairwise polar interaction energy (`kcal/mol`) and optional edge record between two `PolarVectorSite`s.
fn score_polar_site_pair(
    node_a: &str,
    kind_a: Option<HbnNodeKind>,
    sa: &PolarVectorSite,
    node_b: &str,
    kind_b: Option<HbnNodeKind>,
    sb: &PolarVectorSite,
) -> Option<(f64, HBondNetworkEdge)> {
    let d_da = dist3(sa.heavy_pos, sb.heavy_pos);
    if !(2.15..=3.62).contains(&d_da) {
        return None;
    }

    let involves_ligand =
        kind_a == Some(HbnNodeKind::CoupledLigandProtomer) || kind_b == Some(HbnNodeKind::CoupledLigandProtomer);
    let involves_water =
        kind_a == Some(HbnNodeKind::BridgeWaterSite) || kind_b == Some(HbnNodeKind::BridgeWaterSite);

    // Check directional A(donor) -> B(acceptor) and B(donor) -> A(acceptor)
    let mut best_hbond: Option<(f64, bool, f32)> = None;

    if sa.is_donor && sb.is_acceptor {
        if let Some(hp) = sa.proton_pos {
            let ang = angle_dha_deg(sa.heavy_pos, hp, sb.heavy_pos);
            if ang >= 108.0 {
                let lin = ((ang - 105.0) / 75.0).clamp(0.15, 1.0) as f64;
                let dist_w = (-(((d_da as f64) - 2.86) / 0.42).powi(2)).exp();
                let salt_mult = if sa.formal_charge * sb.formal_charge < -0.10 {
                    1.68
                } else {
                    1.0
                };
                let e = -2.05 * lin * dist_w * salt_mult;
                best_hbond = Some((e, true, ang));
            }
        }
    }
    if sb.is_donor && sa.is_acceptor {
        if let Some(hp) = sb.proton_pos {
            let ang = angle_dha_deg(sb.heavy_pos, hp, sa.heavy_pos);
            if ang >= 108.0 {
                let lin = ((ang - 105.0) / 75.0).clamp(0.15, 1.0) as f64;
                let dist_w = (-(((d_da as f64) - 2.86) / 0.42).powi(2)).exp();
                let salt_mult = if sa.formal_charge * sb.formal_charge < -0.10 {
                    1.68
                } else {
                    1.0
                };
                let e = -2.05 * lin * dist_w * salt_mult;
                if best_hbond.map(|(eb, _, _)| e < eb).unwrap_or(true) {
                    best_hbond = Some((e, false, ang));
                }
            }
        }
    }

    if let Some((e_hb, a_is_donor, ang)) = best_hbond {
        let edge_kind = if sa.formal_charge * sb.formal_charge < -0.10 {
            HBondEdgeKind::IonicSaltBridge
        } else if involves_water {
            HBondEdgeKind::WaterMediatedBridgeLeg
        } else if involves_ligand {
            HBondEdgeKind::ProteinLigandDirectHBond
        } else {
            HBondEdgeKind::IntraPocketStructuralHBond
        };
        let (dn_id, dn_lbl, dn_p, pr_p, ac_id, ac_lbl, ac_p) = if a_is_donor {
            (
                node_a.to_string(),
                sa.label.clone(),
                sa.heavy_pos,
                sa.proton_pos,
                node_b.to_string(),
                sb.label.clone(),
                sb.heavy_pos,
            )
        } else {
            (
                node_b.to_string(),
                sb.label.clone(),
                sb.heavy_pos,
                sb.proton_pos,
                node_a.to_string(),
                sa.label.clone(),
                sa.heavy_pos,
            )
        };
        return Some((
            e_hb,
            HBondNetworkEdge {
                donor_node_id: dn_id,
                donor_atom_label: dn_lbl,
                acceptor_node_id: ac_id,
                acceptor_atom_label: ac_lbl,
                donor_pos: dn_p,
                proton_pos: pr_p,
                acceptor_pos: ac_p,
                distance_da_a: d_da,
                angle_dha_deg: ang,
                energy_kcal: e_hb,
                kind: edge_kind,
            },
        ));
    }

    // Otherwise check repulsive polar clashes (`Protoss` / `Reduce` D-D or A-A close contact!)
    if d_da <= 3.12 {
        // Pure acceptor-acceptor electrostatic clash (e.g. ASP- O...O=S ligand sulfonamide or O...O amide)
        if sa.is_acceptor && sb.is_acceptor && !sa.is_donor && !sb.is_donor {
            let e_clash = 2.35 * ((3.18 - d_da) as f64).max(0.25);
            return Some((
                e_clash,
                HBondNetworkEdge {
                    donor_node_id: node_a.to_string(),
                    donor_atom_label: sa.label.clone(),
                    acceptor_node_id: node_b.to_string(),
                    acceptor_atom_label: sb.label.clone(),
                    donor_pos: sa.heavy_pos,
                    proton_pos: None,
                    acceptor_pos: sb.heavy_pos,
                    distance_da_a: d_da,
                    angle_dha_deg: 0.0,
                    energy_kcal: e_clash,
                    kind: HBondEdgeKind::UnfavorablePolarClash,
                },
            ));
        }
        // Donor-donor proton-proton clash
        if let (Some(ha), Some(hb)) = (sa.proton_pos, sb.proton_pos) {
            let d_hh = dist3(ha, hb);
            if d_hh <= 2.15 && (!sa.is_acceptor || !sb.is_acceptor) {
                let e_clash = 2.55 * ((2.25 - d_hh) as f64).max(0.25);
                return Some((
                    e_clash,
                    HBondNetworkEdge {
                        donor_node_id: node_a.to_string(),
                        donor_atom_label: sa.label.clone(),
                        acceptor_node_id: node_b.to_string(),
                        acceptor_atom_label: sb.label.clone(),
                        donor_pos: sa.heavy_pos,
                        proton_pos: Some(ha),
                        acceptor_pos: sb.heavy_pos,
                        distance_da_a: d_da,
                        angle_dha_deg: 0.0,
                        energy_kcal: e_clash,
                        kind: HBondEdgeKind::UnfavorablePolarClash,
                    },
                ));
            }
        }
    }

    None
}

/// Solves the coupled discrete H-bond network (`Goldstein DEE` + Branch-and-Bound exact solver with
/// multi-sweep coordinate refinement) at `eval_ph`.
fn solve_global_hbn_assignment(
    fixed_anchors: &[PolarVectorSite],
    nodes: &mut [HbnVariableNode],
) -> (Vec<usize>, f64, f64, &'static str, usize) {
    let m = nodes.len();
    if m == 0 {
        return (Vec::new(), 0.0, 0.0, "Trivial", 1);
    }

    // 1. Precompute unary energies `u[i][s]` = intrinsic self energy + interactions with fixed_anchors + unsaturated buried penalty
    let mut unary: Vec<Vec<f64>> = Vec::with_capacity(m);
    for node in nodes.iter() {
        let mut u_modes = Vec::with_capacity(node.modes.len());
        for mode in &node.modes {
            let mut e_u = mode.self_energy_kcal;
            for ps in &mode.polar_sites {
                let mut satisfied = false;
                for fa in fixed_anchors {
                    if let Some((ep, _)) =
                        score_polar_site_pair(&node.node_id, Some(node.kind), ps, "BACKBONE", None, fa)
                    {
                        e_u += ep;
                        if ep < -0.45 {
                            satisfied = true;
                        }
                    }
                }
                if ps.is_buried && !satisfied {
                    // Mild baseline burial desolvation offset (relieved if a variable neighbor H-bonds to it!)
                    e_u += 0.32;
                }
            }
            u_modes.push(e_u);
        }
        unary.push(u_modes);
    }

    // 2. Precompute sparse pairwise tables `pair[(i, j)][s_i * len_j + s_j]`
    let mut pair_tables: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    for i in 0..m {
        for j in (i + 1)..m {
            if dist3(nodes[i].centroid, nodes[j].centroid) > 9.5 {
                continue;
            }
            let ni = nodes[i].modes.len();
            let nj = nodes[j].modes.len();
            let mut tab = vec![0.0f64; ni * nj];
            let mut non_zero = false;
            for si in 0..ni {
                for sj in 0..nj {
                    let mut e_ij = 0.0f64;
                    for psa in &nodes[i].modes[si].polar_sites {
                        for psb in &nodes[j].modes[sj].polar_sites {
                            if let Some((ep, _)) = score_polar_site_pair(
                                &nodes[i].node_id,
                                Some(nodes[i].kind),
                                psa,
                                &nodes[j].node_id,
                                Some(nodes[j].kind),
                                psb,
                            ) {
                                e_ij += ep;
                                if ep < -0.45 && (psa.is_buried || psb.is_buried) {
                                    e_ij -= 0.24; // Relieves buried polar desolvation!
                                }
                            }
                        }
                    }
                    if e_ij.abs() > 1e-6 {
                        non_zero = true;
                        tab[si * nj + sj] = e_ij;
                    }
                }
            }
            if non_zero {
                pair_tables.insert((i, j), tab);
            }
        }
    }

    let node_mode_lens: Vec<usize> = nodes.iter().map(|n| n.modes.len()).collect();
    let eval_assignment = |assign: &[usize]| -> f64 {
        let mut tot = 0.0f64;
        for (i, &si) in assign.iter().enumerate() {
            tot += unary[i][si];
        }
        for (&(i, j), tab) in &pair_tables {
            let nj = node_mode_lens[j];
            tot += tab[assign[i] * nj + assign[j]];
        }
        tot
    };

    let naive_assign = vec![0usize; m];
    let naive_e = eval_assignment(&naive_assign);

    // 3. Goldstein Dead-End Elimination (`DEE`) pruning strictly dominated microstates
    let mut alive: Vec<Vec<bool>> = nodes.iter().map(|n| vec![true; n.modes.len()]).collect();
    for i in 0..m {
        let ni = nodes[i].modes.len();
        for a in 0..ni {
            if !alive[i][a] {
                continue;
            }
            for b in 0..ni {
                if a == b || !alive[i][b] {
                    continue;
                }
                // Check if `b` strictly dominates `a`: u[i][a] - u[i][b] + sum_j min_{sj} (E(a,sj) - E(b,sj)) > 0
                let mut worst_diff = unary[i][a] - unary[i][b];
                for j in 0..m {
                    if j == i {
                        continue;
                    }
                    let key = if i < j { (i, j) } else { (j, i) };
                    if let Some(tab) = pair_tables.get(&key) {
                        let nj = nodes[j].modes.len();
                        let mut min_j = f64::INFINITY;
                        for sj in 0..nj {
                            if !alive[j][sj] {
                                continue;
                            }
                            let ea = if i < j {
                                tab[a * nj + sj]
                            } else {
                                tab[sj * ni + a]
                            };
                            let eb = if i < j {
                                tab[b * nj + sj]
                            } else {
                                tab[sj * ni + b]
                            };
                            if ea - eb < min_j {
                                min_j = ea - eb;
                            }
                        }
                        if min_j.is_finite() {
                            worst_diff += min_j;
                        }
                    }
                }
                if worst_diff > 1e-6 {
                    alive[i][a] = false;
                    break;
                }
            }
        }
    }

    // 4. Exact Depth-First Branch-and-Bound (`<= 30,000` visited leaves budget) + Iterative Coordinate Polish
    let mut best_assign = naive_assign.clone();
    let mut best_e = naive_e;
    let mut visited = 0usize;
    let max_visits = 30_000usize;

    // Optimistic lower-bound per unassigned suffix
    let mut min_node_lb = vec![0.0f64; m + 1];
    for i in (0..m).rev() {
        let mut lb_i = f64::INFINITY;
        for s in 0..nodes[i].modes.len() {
            if !alive[i][s] {
                continue;
            }
            let mut e_s = unary[i][s];
            for j in 0..m {
                if j == i {
                    continue;
                }
                let key = if i < j { (i, j) } else { (j, i) };
                if let Some(tab) = pair_tables.get(&key) {
                    let nj = nodes[j].modes.len();
                    let ni = nodes[i].modes.len();
                    let mut min_p = 0.0f64;
                    for sj in 0..nj {
                        let v = if i < j {
                            tab[s * nj + sj]
                        } else {
                            tab[sj * ni + s]
                        };
                        if v < min_p {
                            min_p = v;
                        }
                    }
                    e_s += 0.5 * min_p;
                }
            }
            if e_s < lb_i {
                lb_i = e_s;
            }
        }
        min_node_lb[i] = min_node_lb[i + 1] + lb_i.min(0.0);
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs_bnb(
        idx: usize,
        m: usize,
        cur_e: f64,
        cur: &mut Vec<usize>,
        alive: &[Vec<bool>],
        unary: &[Vec<f64>],
        nodes: &[HbnVariableNode],
        pair_tables: &BTreeMap<(usize, usize), Vec<f64>>,
        min_node_lb: &[f64],
        best_e: &mut f64,
        best_assign: &mut Vec<usize>,
        visited: &mut usize,
        max_visits: usize,
    ) {
        if *visited >= max_visits {
            return;
        }
        if idx == m {
            *visited += 1;
            if cur_e < *best_e {
                *best_e = cur_e;
                *best_assign = cur.clone();
            }
            return;
        }
        if cur_e + min_node_lb[idx] >= *best_e + 1e-6 {
            return;
        }

        // Order alive modes of node `idx` by partial incremental energy
        let mut cands: Vec<(usize, f64)> = Vec::new();
        for s in 0..nodes[idx].modes.len() {
            if !alive[idx][s] {
                continue;
            }
            let mut inc = unary[idx][s];
            for prev in 0..idx {
                if let Some(tab) = pair_tables.get(&(prev, idx)) {
                    let nj = nodes[idx].modes.len();
                    inc += tab[cur[prev] * nj + s];
                }
            }
            cands.push((s, inc));
        }
        cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (s, inc) in cands {
            cur.push(s);
            dfs_bnb(
                idx + 1,
                m,
                cur_e + inc,
                cur,
                alive,
                unary,
                nodes,
                pair_tables,
                min_node_lb,
                best_e,
                best_assign,
                visited,
                max_visits,
            );
            cur.pop();
        }
    }

    let mut cur_stack = Vec::with_capacity(m);
    dfs_bnb(
        0,
        m,
        0.0,
        &mut cur_stack,
        &alive,
        &unary,
        nodes,
        &pair_tables,
        &min_node_lb,
        &mut best_e,
        &mut best_assign,
        &mut visited,
        max_visits,
    );

    // Coordinate-descent polish pass guaranteeing local optimality even if budget capped
    for _pass in 0..3 {
        let mut improved = false;
        for i in 0..m {
            let orig = best_assign[i];
            for s in 0..nodes[i].modes.len() {
                best_assign[i] = s;
                let e_try = eval_assignment(&best_assign);
                if e_try + 1e-7 < best_e {
                    best_e = e_try;
                    improved = true;
                } else {
                    best_assign[i] = orig;
                }
            }
        }
        if !improved {
            break;
        }
    }

    // Record optimal_mode_idx and single-node stabilization delta on each node
    for i in 0..m {
        nodes[i].optimal_mode_idx = best_assign[i];
        let mut alt = best_assign.clone();
        alt[i] = nodes[i].naive_mode_idx;
        let e_without_i = eval_assignment(&alt);
        nodes[i].node_stabilization_gain_kcal = (e_without_i - best_e).max(0.0);
    }

    let method = if visited < max_visits {
        "DEE + Exact Branch-and-Bound"
    } else {
        "DEE + Branch-and-Bound + ICM Polish"
    };
    (best_assign, naive_e, best_e, method, visited.max(1))
}

fn collect_network_edges(
    fixed_anchors: &[PolarVectorSite],
    nodes: &[HbnVariableNode],
    assign: &[usize],
) -> Vec<HBondNetworkEdge> {
    let mut edges = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let mode_i = &node.modes[assign[i]];
        for psa in &mode_i.polar_sites {
            for fa in fixed_anchors {
                if let Some((_, edge)) =
                    score_polar_site_pair(&node.node_id, Some(node.kind), psa, "BACKBONE", None, fa)
                {
                    edges.push(edge);
                }
            }
        }
        for j in (i + 1)..nodes.len() {
            let mode_j = &nodes[j].modes[assign[j]];
            for psa in &mode_i.polar_sites {
                for psb in &mode_j.polar_sites {
                    if let Some((_, edge)) = score_polar_site_pair(
                        &node.node_id,
                        Some(node.kind),
                        psa,
                        &nodes[j].node_id,
                        Some(nodes[j].kind),
                        psb,
                    ) {
                        edges.push(edge);
                    }
                }
            }
        }
    }
    edges.sort_by(|a, b| {
        a.energy_kcal
            .partial_cmp(&b.energy_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges
}

/// Optimizes the coupled `His`/`Asp`/`Glu` tautomeric/protonation, `Asn`/`Gln`/`His` `180 deg` flip,
/// bridge-water `H-O-H` orientation, and ligand protomer/tautomer H-bond network for `mol`.
pub fn optimize_pocket_hbond_network(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    cfg: &HBondNetworkConfig,
) -> HBondNetworkReport {
    let (pdb, center_ovr, target_name) = load_hbond_network_pdb(cfg);
    let pocket = analyze_binding_pocket(&pdb, center_ovr, 10.0);
    optimize_hbond_network_with_prebuilt(mol, ds, cfg, &pdb, &pocket, &target_name, true)
}

pub fn optimize_hbond_network_with_prebuilt(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    cfg: &HBondNetworkConfig,
    pdb: &ParsedPdbStructure,
    pocket: &PocketCavityAnalysis,
    target_name: &str,
    compute_ph_curve: bool,
) -> HBondNetworkReport {
    let mol_idx = ds.and_then(|d| d.mols.iter().position(|dm| dm.id == mol.id));
    let base_pock_score = score_ligand_in_pocket(mol, mol_idx, pocket);
    let lig_coords = base_pock_score.aligned_ligand_coords3d.clone();
    let eval_ph = cfg.target_ph.clamp(4.0, 10.0);

    let (fixed_anchors, mut nodes) =
        build_hbond_network_nodes(pdb, pocket, mol, &lig_coords, cfg, eval_ph);

    let (opt_assign, naive_dg, opt_dg, solver_method, states_eval) =
        solve_global_hbn_assignment(&fixed_anchors, &mut nodes);

    let naive_assign = vec![0usize; nodes.len()];
    let naive_edges = collect_network_edges(&fixed_anchors, &nodes, &naive_assign);
    let optimized_edges = collect_network_edges(&fixed_anchors, &nodes, &opt_assign);

    let naive_clashes = naive_edges
        .iter()
        .filter(|e| e.kind == HBondEdgeKind::UnfavorablePolarClash)
        .count();
    let opt_clashes = optimized_edges
        .iter()
        .filter(|e| e.kind == HBondEdgeKind::UnfavorablePolarClash)
        .count();
    let direct_hbonds = optimized_edges
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                HBondEdgeKind::ProteinLigandDirectHBond | HBondEdgeKind::IonicSaltBridge
            )
        })
        .count();

    let mut retained_waters = 0usize;
    let mut displaced_waters = 0usize;
    let mut flipped_180 = 0usize;
    let mut nonstd_prot = 0usize;
    let mut net_dn_h = 0.0f64;
    let mut lig_state_lbl = "Lig-Neutral".to_string();

    for (ni, node) in nodes.iter().enumerate() {
        let chosen = &node.modes[opt_assign[ni]];
        net_dn_h += chosen.delta_protons_vs_ref as f64;
        if chosen.is_flipped_180 {
            flipped_180 += 1;
        }
        if matches!(chosen.pdb_res_name.as_str(), "HIP" | "ASH" | "GLH" | "CYM" | "HID") {
            nonstd_prot += 1;
        }
        if node.kind == HbnNodeKind::BridgeWaterSite {
            if chosen.is_water_displaced {
                displaced_waters += 1;
            } else {
                retained_waters += 1;
            }
        }
        if node.kind == HbnNodeKind::CoupledLigandProtomer {
            lig_state_lbl = chosen.state_code.clone();
        }
    }

    let ddg_gain = (opt_dg - naive_dg).min(0.0);
    // Cooperative bridge-water + H-bond refinement onto baseline Vina pocket dG
    let hbn_adj_dg = base_pock_score.total_dg_pocket_kcal + (ddg_gain * 0.48).clamp(-3.20, 0.0);
    let hbn_pkd = (-hbn_adj_dg / RT_LN10_KCAL).clamp(2.5, 12.5);

    // Optional 7-point Wyman proton-linked binding curve `dG_HBN(pH)` (`pH = 5.0 .. 8.0`)
    let mut ph_curve = Vec::new();
    if compute_ph_curve {
        for &ph_pt in &HBN_PH_SCAN_POINTS {
            let (fa_p, mut nd_p) =
                build_hbond_network_nodes(pdb, pocket, mol, &lig_coords, cfg, ph_pt);
            let (asg_p, nv_p, op_p, _, _) = solve_global_hbn_assignment(&fa_p, &mut nd_p);
            let mut dn_h_p = 0.0f64;
            let mut tags = Vec::new();
            for (k, nd) in nd_p.iter().enumerate() {
                let ch = &nd.modes[asg_p[k]];
                dn_h_p += ch.delta_protons_vs_ref as f64;
                if matches!(
                    nd.kind,
                    HbnNodeKind::HistidineTautomerFlip | HbnNodeKind::CarboxylateProtonation
                ) {
                    tags.push(format!("{}:{}", nd.node_id, ch.state_code));
                }
            }
            ph_curve.push(ProtonLinkedPhPoint {
                ph: ph_pt,
                optimized_network_dg_kcal: op_p,
                naive_network_dg_kcal: nv_p,
                net_proton_uptake_dn_h: dn_h_p,
                dominant_his_asp_summary: if tags.is_empty() {
                    "Standard ionization".to_string()
                } else {
                    tags.into_iter().take(3).collect::<Vec<_>>().join(", ")
                },
            });
        }
    }

    HBondNetworkReport {
        preset: cfg.preset,
        pdb_id: pdb.pdb_id.clone(),
        target_name: target_name.to_string(),
        target_ph: eval_ph,
        compound_id: mol.id.clone(),
        compound_name: mol.name.clone(),
        canonical_smiles: mol.smiles.clone(),
        selected_ligand_state_label: lig_state_lbl,
        solver_method,
        states_evaluated: states_eval,
        naive_network_dg_kcal: naive_dg,
        optimized_network_dg_kcal: opt_dg,
        ddg_hbn_optimization_gain_kcal: ddg_gain,
        naive_polar_clashes: naive_clashes,
        optimized_polar_clashes: opt_clashes,
        direct_protein_ligand_hbonds: direct_hbonds,
        retained_bridge_waters: retained_waters,
        displaced_waters,
        flipped_180_residues: flipped_180,
        nonstandard_protonated_residues: nonstd_prot,
        net_proton_uptake_at_ph: net_dn_h,
        hbn_adjusted_pocket_dg_kcal: hbn_adj_dg,
        hbn_predicted_pkd: hbn_pkd,
        nodes,
        naive_edges,
        optimized_edges,
        ph_linkage_curve: ph_curve,
        aligned_ligand_coords3d: lig_coords,
        pocket: pocket.clone(),
        mol: mol.clone(),
    }
}

/// Screens an entire `DatasetAnalysis` in parallel for active-site H-bond & bridge-water complementarity.
pub fn screen_dataset_hbond_network(
    ds: &DatasetAnalysis,
    cfg: &HBondNetworkConfig,
) -> Vec<DatasetHBondScreenRow> {
    let (pdb, center_ovr, target_name) = load_hbond_network_pdb(cfg);
    let pocket = analyze_binding_pocket(&pdb, center_ovr, 10.0);

    let mut rows: Vec<DatasetHBondScreenRow> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(mi, m)| {
            let rep = optimize_hbond_network_with_prebuilt(
                m,
                Some(ds),
                cfg,
                &pdb,
                &pocket,
                &target_name,
                false,
            );
            let _ = evaluate_physchem(m, 0);
            DatasetHBondScreenRow {
                rank: 0,
                mol_idx: mi,
                compound_id: m.id.clone(),
                compound_name: m.name.clone(),
                observed_pact: m.pact,
                naive_network_dg_kcal: rep.naive_network_dg_kcal,
                optimized_network_dg_kcal: rep.optimized_network_dg_kcal,
                ddg_hbn_gain_kcal: rep.ddg_hbn_optimization_gain_kcal,
                hbn_adjusted_pocket_dg_kcal: rep.hbn_adjusted_pocket_dg_kcal,
                hbn_predicted_pkd: rep.hbn_predicted_pkd,
                direct_hbonds: rep.direct_protein_ligand_hbonds,
                retained_bridge_waters: rep.retained_bridge_waters,
                flipped_or_taut_switched: rep.flipped_180_residues
                    + rep.nonstandard_protonated_residues,
                net_proton_uptake: rep.net_proton_uptake_at_ph,
                selected_ligand_state: rep.selected_ligand_state_label,
                canonical_smiles: m.smiles.clone(),
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        a.hbn_adjusted_pocket_dg_kcal
            .partial_cmp(&b.hbn_adjusted_pocket_dg_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Exports variable `His`/`Asp`/`Glu`/`Asn`/`Gln` & bridge-water node assignments to `.csv`.
pub fn export_hbond_network_nodes_csv(rep: &HBondNetworkReport) -> String {
    let mut out = String::from(
        "node_id,node_kind,propka_shifted_pka,naive_state,optimal_state,pdb_res_name,is_flipped_180,is_water_displaced,delta_protons,self_energy_kcal,node_stabilization_gain_kcal,description\n",
    );
    for n in &rep.nodes {
        let nv = &n.modes[n.naive_mode_idx];
        let op = &n.modes[n.optimal_mode_idx];
        let pka_str = n
            .propka_shifted_pka
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "{},\"{}\",{},\"{}\",\"{}\",{},{},{},{:+},{:+.2},{:+.2},\"{}\"\n",
            n.node_id,
            n.kind.short_label(),
            pka_str,
            nv.state_code,
            op.state_code,
            op.pdb_res_name,
            op.is_flipped_180,
            op.is_water_displaced,
            op.delta_protons_vs_ref,
            op.self_energy_kcal,
            n.node_stabilization_gain_kcal,
            op.description
        ));
    }
    out
}

/// Exports solved directional H-bond, salt-bridge, and water-relay edges to `.csv`.
pub fn export_hbond_network_edges_csv(rep: &HBondNetworkReport) -> String {
    let mut out = String::from(
        "donor_node,donor_atom,acceptor_node,acceptor_atom,distance_da_a,angle_dha_deg,energy_kcal,edge_kind\n",
    );
    for e in &rep.optimized_edges {
        out.push_str(&format!(
            "{},{},{},{},{:.2},{:.1},{:+.2},\"{}\"\n",
            e.donor_node_id,
            e.donor_atom_label,
            e.acceptor_node_id,
            e.acceptor_atom_label,
            e.distance_da_a,
            e.angle_dha_deg,
            e.energy_kcal,
            e.kind.short_label()
        ));
    }
    out
}

/// Exports dataset-wide H-bond & bridge-water screening table to `.csv` with summary row.
pub fn export_dataset_hbond_screen_csv(rows: &[DatasetHBondScreenRow]) -> String {
    let mut out = String::from(
        "rank,compound_id,compound_name,observed_pact,naive_hbn_dg_kcal,optimized_hbn_dg_kcal,ddg_hbn_gain_kcal,hbn_adjusted_pocket_dg_kcal,hbn_predicted_pkd,direct_hbonds,retained_bridge_waters,flipped_or_taut_switched,net_proton_uptake,ligand_state,smiles\n",
    );
    let mut water_bridged_cnt = 0usize;
    for r in rows {
        if r.retained_bridge_waters > 0 {
            water_bridged_cnt += 1;
        }
        out.push_str(&format!(
            "{},{},\"{}\",{:.2},{:+.2},{:+.2},{:+.2},{:+.2},{:.2},{},{},{},{:+.1},\"{}\",{}\n",
            r.rank,
            r.compound_id,
            r.compound_name,
            r.observed_pact,
            r.naive_network_dg_kcal,
            r.optimized_network_dg_kcal,
            r.ddg_hbn_gain_kcal,
            r.hbn_adjusted_pocket_dg_kcal,
            r.hbn_predicted_pkd,
            r.direct_hbonds,
            r.retained_bridge_waters,
            r.flipped_or_taut_switched,
            r.net_proton_uptake,
            r.selected_ligand_state,
            r.canonical_smiles
        ));
    }
    out.push_str(&format!(
        "# SUMMARY,total_compounds={},water_bridged_compounds={}\n",
        rows.len(),
        water_bridged_cnt
    ));
    out
}

/// Formats a prepared `.pdb` file containing `HID`/`HIE`/`HIP`/`ASH`/`GLH` residue assignments,
/// explicit active-site polar `H` atoms, retained oriented tetrahedral `H-O-H` bridge waters, and
/// the aligned 3D `HETATM` ligand pose.
pub fn format_optimized_hbond_complex_pdb(rep: &HBondNetworkReport) -> String {
    let mut pdb_out = format!(
        "HEADER    HBOND-NETWORK-PREPARED                  14-NOV-26   {}\nTITLE     SAR-ATLAS V3.4.0 PROTOSS/REDUCE PREPARED COMPLEX FOR {} AT PH {:.1}\nREMARK   1 OPTIMIZED HBN DG = {:+.2} KCAL/MOL (GAIN VS NAIVE = {:+.2} KCAL/MOL)\n",
        rep.pdb_id, rep.compound_id, rep.target_ph, rep.optimized_network_dg_kcal, rep.ddg_hbn_optimization_gain_kcal
    );
    let mut serial = 1u32;

    // Emit receptor pocket shell atoms with updated HID/HIE/HIP/ASH/GLH residue names
    let mut res_override: BTreeMap<String, String> = BTreeMap::new();
    for n in &rep.nodes {
        if !matches!(n.kind, HbnNodeKind::BridgeWaterSite | HbnNodeKind::CoupledLigandProtomer) {
            res_override.insert(
                n.node_id.clone(),
                n.modes[n.optimal_mode_idx].pdb_res_name.clone(),
            );
        }
    }

    for a in &rep.pocket.shell_atoms {
        let key = format!("{}{}", a.res_name, a.res_seq);
        let rname = res_override
            .get(&key)
            .map(String::as_str)
            .unwrap_or(a.res_name.as_str());
        pdb_out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00{:>6.2}          {:>2}\n",
            serial,
            a.atom_name,
            rname,
            a.chain_id,
            a.res_seq,
            a.pos[0],
            a.pos[1],
            a.pos[2],
            a.b_factor,
            a.elem.symbol()
        ));
        serial += 1;
    }

    // Emit explicit polar protons & retained oriented tetrahedral H-O-H bridge waters
    let mut wat_seq = 2001i32;
    for n in &rep.nodes {
        let op = &n.modes[n.optimal_mode_idx];
        if n.kind == HbnNodeKind::BridgeWaterSite {
            if op.is_water_displaced {
                continue;
            }
            pdb_out.push_str(&format!(
                "HETATM{:>5}  O   HOH A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 18.50           O\n",
                serial, wat_seq, n.centroid[0], n.centroid[1], n.centroid[2]
            ));
            serial += 1;
            for (hi, ps) in op.polar_sites.iter().enumerate() {
                if let Some(hp) = ps.proton_pos {
                    pdb_out.push_str(&format!(
                        "HETATM{:>5}  H{}  HOH A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 18.50           H\n",
                        serial,
                        hi + 1,
                        wat_seq,
                        hp[0],
                        hp[1],
                        hp[2]
                    ));
                    serial += 1;
                }
            }
            wat_seq += 1;
        } else if n.kind != HbnNodeKind::CoupledLigandProtomer {
            for (hi, ps) in op.polar_sites.iter().enumerate() {
                if let Some(hp) = ps.proton_pos {
                    pdb_out.push_str(&format!(
                        "ATOM  {:>5}  H{}  {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 20.00           H\n",
                        serial,
                        hi + 1,
                        op.pdb_res_name,
                        (hi + 1) as i32,
                        hp[0],
                        hp[1],
                        hp[2]
                    ));
                    serial += 1;
                }
            }
        }
    }

    // Emit aligned 3D ligand heavy atoms
    for (li, a) in rep.mol.atoms.iter().enumerate() {
        if li < rep.aligned_ligand_coords3d.len() {
            let p = rep.aligned_ligand_coords3d[li];
            pdb_out.push_str(&format!(
                "HETATM{:>5} {:<4} LIG L   1    {:>8.3}{:>8.3}{:>8.3}  1.00 20.00          {:>2}\n",
                serial,
                format!("{}{}", a.elem.symbol(), li + 1),
                p[0],
                p[1],
                p[2],
                a.elem.symbol()
            ));
            serial += 1;
        }
    }
    pdb_out.push_str("END\n");
    pdb_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol_io::parse_smiles;

    #[test]
    fn test_cox2_3ln1_his90_gln178_and_bridge_water_hbond_optimizer() {
        let mut celecoxib =
            parse_smiles("Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1")
                .expect("Valid Celecoxib SMILES");
        celecoxib.id = "CEL-3LN1".to_string();
        celecoxib.name = "Celecoxib".to_string();

        let cfg = HBondNetworkConfig {
            preset: HBondNetworkPreset::Cox23ln1His90Gln178Sulfonamide,
            target_ph: 7.4,
            ..HBondNetworkConfig::default()
        };
        let rep = optimize_pocket_hbond_network(&celecoxib, None, &cfg);
        assert_eq!(rep.pdb_id, "3LN1");
        assert!(rep.optimized_network_dg_kcal <= rep.naive_network_dg_kcal + 1e-6);
        assert_eq!(rep.ph_linkage_curve.len(), 7);
        assert!(rep.nodes.iter().any(|n| n.kind == HbnNodeKind::HistidineTautomerFlip));
        let pdb_txt = format_optimized_hbond_complex_pdb(&rep);
        assert!(pdb_txt.contains("HBOND-NETWORK-PREPARED"));
    }
}
