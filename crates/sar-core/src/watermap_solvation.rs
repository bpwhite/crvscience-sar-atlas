//! Inhomogeneous Fluid Solvation Theory (`IFST` / `WaterMap` / `GIST`) `dH` / `-TdS` / `dG`
//! Active-Site Hydration-Site Hotspot Mapper, Capped Ligand Water-Displacement Scorer, and
//! Exit-Vector Matched Displaceable-Water R-Group Designer (`watermap_solvation.rs`, `v3.5.0`).
//!
//! Primary References:
//! - Young, Abel, Kim, Berne & Friesner (2007), *Proc. Natl. Acad. Sci. U.S.A.* 104(3), 808-813
//!   ("Motifs for molecular recognition exploiting hydrophobic enclosure in protein-ligand binding"
//!   — `1.0 A` non-overlapping `QT` hydration spheres, 3-sided hydrophobic + polar orientational
//!   entropy frustration, and `COX-2` channel dewetting:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC1783395/`)
//! - Abel, Young, Farid, Berne & Friesner (2008), *J. Am. Chem. Soc.* 130(9), 2817-2831 ("Role of
//!   the active-site solvent in the thermodynamics of factor Xa ligand binding" — `WaterMap`
//!   hydration-site `dH` / `-TdS_e` / `dG` decomposition, Factor Xa `S1 Tyr228` chloro-displacement
//!   & `S4` aromatic box hotspots, `R^2 = 0.81`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC2761766/`)
//! - Nguyen, Kurtzman Young & Gilson (2012), *J. Chem. Phys.* 137(4):044101 ("Grid inhomogeneous
//!   solvation theory: Hydration structure and thermodynamics of the miniature receptor
//!   cucurbit[7]uril" — `GIST` 3D Cartesian voxel solvation fields:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3416872/`)
//! - Nguyen, Cruz, Gilson & Kurtzman (2014), *J. Chem. Theory Comput.* 10(7), 2769-2780
//!   ("Thermodynamics of Water in an Enzyme Active Site: Grid-Based Hydration Analysis of
//!   Coagulation Factor Xa" — capped single-site ligand overlap `d_i^cap <= 1.0`, `R^2 = 0.84`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4089914/`)
//! - Huggins (2015), *Biophys. J.* 108(4), 928-936 ("Quantifying the Entropy of Binding for Water
//!   Molecules in Protein Cavities by Computing Correlations":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4336375/`) & Barillari, Taylor, Viner & Essex
//!   (2007), *J. Am. Chem. Soc.* 129(9), 2577-2587 (`https://pubmed.ncbi.nlm.nih.gov/17288418/`).

use crate::alerts::scan_structural_alerts;
use crate::allosteric_pocket::curated_allosteric_pdb_text;
use crate::fragment_grower::{perceive_fragment_exit_vectors, RT_LN10_KCAL};
use crate::mol::{Element, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    ParsedPdbStructure, PocketCavityAnalysis,
};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Bulk liquid water interaction energy reference `E_bulk` (`kcal/mol` per water, 298.15 K TIP4P-2005/TIP3P scale; Abel et al. 2008).
pub const WATERMAP_BULK_ENERGY_KCAL: f64 = -9.53;

/// Bulk liquid water coordination count within `3.5 A` oxygen-oxygen shell (`#nbrs_bulk`).
pub const WATERMAP_BULK_NEIGHBORS: f64 = 4.70;

/// Bulk liquid water mean hydrogen-bonded neighbors (`#HBnbrs_bulk`).
pub const WATERMAP_BULK_HBONDS: f64 = 3.55;

/// Default `WaterMap` / `HSA` steric displacement cutoff radius `R_co` (`Angstrom`; Abel et al. 2008 / Nguyen et al. 2014).
pub const WATERMAP_DISPLACEMENT_CUTOFF_A: f32 = 2.24;

/// Curated enzyme & receptor active sites showcasing `IFST` / `WaterMap` hydration-site thermodynamics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterMapPreset {
    /// `PDB 1EZQ` (`Factor Xa`): Classic Abel-Friesner (2008) benchmark — `S1 Tyr228/Asp189` high-`dG` chloro-displaceable `Site 12`, `Cys191-Cys220` disulfide `Site 17`, & `S4 Tyr99/Phe174/Trp215` aromatic box `Sites 13/18/21`.
    FactorXa1ezqS1S4Hotspots,
    /// `PDB 1M17` (`EGFR`): Conserved `Thr790/Gln791` `HOH2001` structural bridge (`dG < -1.8 kcal/mol`) vs enthalpically frustrated `Met766/Leu788` back-pocket hotspots.
    Egfr1m17BackPocketAndHingeBridge,
    /// `PDB 3LN1` (`COX-2`): Young-Friesner (2007) persistent hydrophobic-enclosure channel dewetting (`Val349/Leu352/Trp387`) + `His90/Arg513` polar side-pocket waters.
    Cox23ln1DewettedHydrophobicChannel,
    /// `PDB 6OIM` (`KRAS G12C`): Switch-II cryptic `His95/Tyr96/Gln99` entropically ordered waters + lipophilic `Met72/Val9` sub-pocket hotspots.
    Kras6oimSwitchIiCrypticSolvation,
    /// User-supplied `.pdb` receptor with automatic 3D `GIST` voxel solvation & `1.0 A` `QT` hydration-site clustering.
    CustomPdbWaterMap,
}

impl WaterMapPreset {
    pub const ALL: [Self; 5] = [
        Self::FactorXa1ezqS1S4Hotspots,
        Self::Egfr1m17BackPocketAndHingeBridge,
        Self::Cox23ln1DewettedHydrophobicChannel,
        Self::Kras6oimSwitchIiCrypticSolvation,
        Self::CustomPdbWaterMap,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::FactorXa1ezqS1S4Hotspots => "WM-1EZQ-FXA-S1S4",
            Self::Egfr1m17BackPocketAndHingeBridge => "WM-1M17-EGFR-ATP",
            Self::Cox23ln1DewettedHydrophobicChannel => "WM-3LN1-COX2-DEWET",
            Self::Kras6oimSwitchIiCrypticSolvation => "WM-6OIM-KRAS-SWII",
            Self::CustomPdbWaterMap => "WM-CUSTOM-PDB",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::FactorXa1ezqS1S4Hotspots => "Factor Xa 1EZQ (S1+S4)",
            Self::Egfr1m17BackPocketAndHingeBridge => "EGFR 1M17 (Bridge+Back)",
            Self::Cox23ln1DewettedHydrophobicChannel => "COX-2 3LN1 (Dewetted)",
            Self::Kras6oimSwitchIiCrypticSolvation => "KRAS 6OIM (Switch-II)",
            Self::CustomPdbWaterMap => "Custom .PDB WaterMap",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::FactorXa1ezqS1S4Hotspots => {
                "Coagulation Factor Xa (1EZQ): S1 Tyr228 Chloro-Hotspot & S4 Tyr99/Phe174/Trp215 Aromatic Box"
            }
            Self::Egfr1m17BackPocketAndHingeBridge => {
                "EGFR Kinase (1M17): Conserved Thr790/Gln791 Structural Bridge vs Back-Pocket Unhappy Waters"
            }
            Self::Cox23ln1DewettedHydrophobicChannel => {
                "COX-2 (3LN1): Dewetted Paraffin-Like Channel Hotspots & His90/Arg513 Sulfonamide Pocket"
            }
            Self::Kras6oimSwitchIiCrypticSolvation => {
                "KRAS G12C (6OIM): Switch-II His95/Tyr96/Gln99 Ordered Groove & Met72/Val9 Cryptic Hotspots"
            }
            Self::CustomPdbWaterMap => {
                "Custom .PDB Receptor: 3D GIST Solvation Grid & 1.0 A QT Hydration-Site Mapper"
            }
        }
    }
}

/// Five-class `IFST` / `WaterMap` thermodynamic classification of active-site hydration sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HydrationSiteClass {
    /// `dH >= +1.30 kcal/mol` and `dG >= +1.20 kcal/mol`: broken H-bond shell in apolar/aromatic pocket — prime lipophilic/halide displacement target.
    EnthalpyFrustratedHotspot,
    /// `-TdS >= +1.40 kcal/mol` and `dG >= +1.08 kcal/mol`: 3-sided hydrophobic + 1-polar enclosure — displace with snug polar-capable or aliphatic group.
    EntropyOrderedEnclosure,
    /// `occupancy < 0.38` and `dH >= +3.00 kcal/mol`: Young-Friesner persistent cavity dewetting — high potency gain per heavy atom with near-zero desolvation cost.
    DewettedVacuumCavity,
    /// `|dG| < 1.10 kcal/mol`: solvent-exposed rim water near bulk chemical potential — safe channel for polar solubilizing tails.
    BulkLikeNeutral,
    /// `dG <= -1.10 kcal/mol`: tightly coordinated structural/bridging water — retain as H-bond bridge or replace with directional polar warhead (`-CN`, `-OH`, `-CONH2`).
    ConservedStructuralBridge,
}

impl HydrationSiteClass {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::EnthalpyFrustratedHotspot => "Unhappy (Enthalpy dH+)",
            Self::EntropyOrderedEnclosure => "Unhappy (Entropy -TdS+)",
            Self::DewettedVacuumCavity => "Dewetted Cavity Hotspot",
            Self::BulkLikeNeutral => "Bulk-Like Rim Water",
            Self::ConservedStructuralBridge => "Conserved Bridge (dG-)",
        }
    }

    pub fn design_action(self) -> &'static str {
        match self {
            Self::EnthalpyFrustratedHotspot => "Displace with -Cl, -CF3, -CH3, or Cyclopropyl",
            Self::EntropyOrderedEnclosure => "Displace with -CHF2, -CN, -CH2OH, or -OCH3",
            Self::DewettedVacuumCavity => "Fill vacuum pocket with -CH3, -Et, -Cl, or Aryl",
            Self::BulkLikeNeutral => "Solvent rim: attach morpholine/solubility handle",
            Self::ConservedStructuralBridge => "Retain bridge or mimic with -CN / -OH / -CONH2",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::EnthalpyFrustratedHotspot => [244, 63, 94],
            Self::EntropyOrderedEnclosure => [192, 132, 252],
            Self::DewettedVacuumCavity => [251, 146, 60],
            Self::BulkLikeNeutral => [148, 163, 184],
            Self::ConservedStructuralBridge => [52, 211, 153],
        }
    }
}

/// Ligand interaction fate at an individual hydration site under Nguyen-Gilson capped overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiteLigandStatus {
    /// Sterically displaced (`f_disp >= 0.35`) by nonpolar/halide ligand heavy atom(s).
    DisplacedNonpolar,
    /// Sterically displaced (`f_disp >= 0.35`) by a polar H-bond donor/acceptor (`N`/`O`/`F`) mimicking the water's receptor H-bond.
    ReplacedByPolarMimic,
    /// Retained within H-bonding distance (`2.40..3.55 A`) of a ligand polar atom as an interfacial bridge.
    BridgedByLigand,
    /// Untouched hotspot within growth reach (`<= 5.65 A`) of an outward ligand exit vector.
    ProximalGrowthTarget,
    /// Distal active-site hydration site (`> 5.65 A` from aligned ligand).
    DistalUnoccupied,
}

impl SiteLigandStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::DisplacedNonpolar => "Displaced (Lipophilic)",
            Self::ReplacedByPolarMimic => "Displaced (Polar Mimic)",
            Self::BridgedByLigand => "Retained H-Bond Bridge",
            Self::ProximalGrowthTarget => "Open R-Vector Target",
            Self::DistalUnoccupied => "Distal Unoccupied",
        }
    }
}

/// Compact 3D `GIST` voxel sample retained for high-energy solvation cloud visualization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GistHotspotVoxel {
    pub pos: [f32; 3],
    pub g_sw_density: f64,
    pub dh_norm_kcal: f64,
    pub neg_tds_norm_kcal: f64,
    pub dg_norm_kcal: f64,
    pub is_dewetted_void: bool,
}

/// Complete `IFST` / `WaterMap` thermodynamic profile of one `1.0 A` principal hydration site.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterMapHydrationSite {
    pub site_id: String,
    pub pos: [f32; 3],
    pub is_crystallographic_seed: bool,
    pub class: HydrationSiteClass,
    pub occupancy: f64,
    pub local_density_g_sw: f64,
    pub protein_hbonds: f64,
    pub water_hbonds: f64,
    pub neighbor_waters_3_5a: f64,
    pub percent_bulk_hb: f64,
    pub exposure_fraction: f64,
    pub hydrophobic_enclosure: f64,
    /// Mean solute-water interaction energy `E_sw` (`kcal/mol`).
    pub e_solute_water_kcal: f64,
    /// Mean water-water interaction energy `E_ww` (`kcal/mol`).
    pub e_water_water_kcal: f64,
    /// Excess enthalpy vs bulk liquid water `dH = E_sw + E_ww - E_bulk` (`kcal/mol`; positive = frustrated).
    pub dh_excess_kcal: f64,
    /// First-order translational localization entropy cost `-TdS_trans` (`kcal/mol`; positive = ordered).
    pub neg_tds_trans_kcal: f64,
    /// First-order orientational tetrahedral cone restriction `-TdS_orient` (`kcal/mol`; positive = ordered).
    pub neg_tds_orient_kcal: f64,
    /// Total excess entropic cost `-TdS_e = -TdS_trans + (-TdS_orient)` (`kcal/mol`).
    pub neg_tds_excess_kcal: f64,
    /// Excess hydration free energy `dG_site = dH + (-TdS_e)` (`kcal/mol`; positive = favorable to displace).
    pub dg_excess_kcal: f64,
    pub lining_residues: Vec<String>,
    /// Nguyen-Gilson capped steric overlap `f_disp in [0.0, 1.0]` with the aligned query ligand.
    pub ligand_overlap_f_disp: f64,
    pub min_ligand_dist_a: f32,
    pub status: SiteLigandStatus,
    /// Net contribution to ligand binding free energy `dG_bind` (`kcal/mol`; negative = favorable potency boost).
    pub ligand_dg_contribution_kcal: f64,
}

/// Strategy used by a designed R-group substituent targeting a specific hydration site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RGroupSolvationStrategy {
    /// Nonpolar/halide group expelling an enthalpically frustrated or dewetted cavity water (`-Cl`, `-CF3`, `-CH3`, `-cPr`).
    LipophilicHotspotDisplacer,
    /// Directional H-bond acceptor/donor replacing an ordered/bridging water while preserving receptor H-bonds (`-CN`, `-OH`, `-CONH2`).
    PolarWaterBridgeMimic,
    /// Fluorinated/compact group balancing hydrophobic enclosure fill with dipolar multipolar contact (`-CHF2`, `-F`, `-OCF3`).
    FluorinatedEnclosureFiller,
}

impl RGroupSolvationStrategy {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::LipophilicHotspotDisplacer => "Lipophilic Hotspot Displacer",
            Self::PolarWaterBridgeMimic => "Polar H-Bond Water Mimic",
            Self::FluorinatedEnclosureFiller => "Fluorinated Enclosure Filler",
        }
    }
}

/// Actionable R-group analog designed along a 3D ligand exit vector to displace or mimic a target hydration site.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisplaceableWaterRGroupDesign {
    pub rank: usize,
    pub design_id: String,
    pub target_site_id: String,
    pub target_site_class: HydrationSiteClass,
    pub target_site_dg_kcal: f64,
    pub anchor_atom_idx: usize,
    pub anchor_atom_label: String,
    pub anchor_pos3d: [f32; 3],
    pub target_site_pos3d: [f32; 3],
    pub vector_distance_a: f32,
    pub vector_alignment_deg: f32,
    pub rgroup_code: &'static str,
    pub rgroup_name: &'static str,
    pub rgroup_smiles_suffix: &'static str,
    pub synthetic_reaction: &'static str,
    pub strategy: RGroupSolvationStrategy,
    pub designed_smiles: String,
    pub added_heavy_atoms: usize,
    pub predicted_site_overlap_f: f64,
    /// Net solvation + pocket interaction free-energy improvement `ddG_bind` (`kcal/mol`; negative = tighter binding).
    pub ddg_solvation_gain_kcal: f64,
    /// Predicted potency gain `dpAct = -ddG / 1.364`.
    pub predicted_delta_pact: f64,
    pub predicted_pact: f64,
    /// Verdonk-Rees Group Efficiency `GE = -ddG / dHAC` (`kcal/mol` per added heavy atom).
    pub group_efficiency_ge: f64,
    pub delta_lle: f64,
    pub mw: f64,
    pub clogp: f64,
    pub tpsa: f64,
    pub sa_score: f64,
    pub pains_count: usize,
    /// Multiparameter `[0..100]` WaterMap R-Vector Design Score balancing `-ddG`, `GE`, `dLLE`, vector geometry & `SAScore`.
    pub design_score: f64,
}

/// Configuration for `WaterMap` / `GIST` active-site solvation mapping and R-group vector design.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterMapConfig {
    pub preset: WaterMapPreset,
    pub custom_pdb_path: Option<String>,
    /// 3D `GIST` voxel spacing (`Angstrom`, clamped to `0.50..=1.25`, default `0.75`).
    pub gist_voxel_spacing_a: f32,
    /// Nguyen-Gilson steric displacement cutoff `R_co` (`Angstrom`, default `2.24`).
    pub displacement_cutoff_a: f32,
    /// Minimum excess `dG_site` (`kcal/mol`) classified as a displaceable high-energy hotspot (default `1.20`).
    pub hotspot_dg_threshold_kcal: f64,
    /// Whether to enumerate & score exit-vector matched R-group analogs targeting proximal hydration sites.
    pub design_rgroup_vectors: bool,
}

impl Default for WaterMapConfig {
    fn default() -> Self {
        Self {
            preset: WaterMapPreset::FactorXa1ezqS1S4Hotspots,
            custom_pdb_path: None,
            gist_voxel_spacing_a: 0.75,
            displacement_cutoff_a: WATERMAP_DISPLACEMENT_CUTOFF_A,
            hotspot_dg_threshold_kcal: 1.20,
            design_rgroup_vectors: true,
        }
    }
}

/// Complete `IFST` / `WaterMap` / `GIST` solvation & R-group vector design report for one complex.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterMapSolvationReport {
    pub preset: WaterMapPreset,
    pub pdb_id: String,
    pub target_name: String,
    pub pocket_center: [f32; 3],
    pub compound_id: String,
    pub compound_name: String,
    pub query_smiles: String,
    pub mol: Molecule,
    pub baseline_pact: f64,
    pub baseline_pocket_dg_kcal: f64,
    pub gist_voxel_spacing_a: f32,
    pub total_gist_voxels_evaluated: usize,
    pub high_energy_gist_voxels: usize,
    pub dewetted_gist_voxels: usize,
    pub integrated_cavity_excess_dg_kcal: f64,
    pub total_hydration_sites: usize,
    pub unhappy_hotspot_sites: usize,
    pub dewetted_cavity_sites: usize,
    pub conserved_bridge_sites: usize,
    pub displaced_hotspot_count: usize,
    pub retained_bridge_count: usize,
    pub clashing_displaced_bridges: usize,
    /// Enthalpic contribution to ligand binding from displaced waters (`kcal/mol`, negative = favorable).
    pub dh_displacement_gain_kcal: f64,
    /// Entropic contribution to ligand binding from displaced waters (`kcal/mol`, negative = favorable).
    pub tds_displacement_gain_kcal: f64,
    /// Penalty incurred by sterically expelling conserved structural waters without polar mimicry (`kcal/mol`, positive = unfavorable).
    pub bridge_expulsion_penalty_kcal: f64,
    /// Net `WaterMap` solvation displacement free energy `dG_solv_disp` (`kcal/mol`, negative = potency gain).
    pub net_watermap_dg_kcal: f64,
    /// Solvation-refined binding free energy combining pocket complementarity and `IFST` displacement (`kcal/mol`).
    pub watermap_refined_dg_kcal: f64,
    /// Solvation-refined predicted `pKd = -watermap_refined_dg_kcal / 1.364`.
    pub watermap_predicted_pkd: f64,
    /// Remaining liberation free energy (`kcal/mol`) available from undisplaced hotspots within `5.65 A` of ligand exit vectors.
    pub untapped_proximal_hotspot_dg_kcal: f64,
    pub aligned_ligand_coords3d: Vec<[f32; 3]>,
    pub pocket_shell_display_atoms: Vec<([f32; 3], String, Element)>,
    pub gist_hotspot_cloud: Vec<GistHotspotVoxel>,
    pub hydration_sites: Vec<WaterMapHydrationSite>,
    pub rgroup_designs: Vec<DisplaceableWaterRGroupDesign>,
}

/// Dataset-wide `WaterMap` displacement screening row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetWaterMapScreenRow {
    pub rank: usize,
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub smiles: String,
    pub observed_pact: f64,
    pub mw: f64,
    pub clogp: f64,
    pub displaced_hotspots: usize,
    pub retained_bridges: usize,
    pub clashing_bridges: usize,
    pub dh_disp_gain_kcal: f64,
    pub tds_disp_gain_kcal: f64,
    pub net_watermap_dg_kcal: f64,
    pub watermap_refined_dg_kcal: f64,
    pub untapped_hotspot_dg_kcal: f64,
    pub top_rgroup_suggestion: String,
}

/// Embedded Coagulation Factor Xa (`PDB 1EZQ`) S1 + S4 active site (Abel, Young, Farid, Berne & Friesner 2008).
pub fn curated_factor_xa_1ezq_pdb_text() -> &'static str {
    r#"HEADER    HYDROLASE/INHIBITOR                     11-MAY-00   1EZQ
TITLE     COAGULATION FACTOR XA S1 TYR228 AND S4 AROMATIC BOX ACTIVE SITE
ATOM      1  CA  TYR A  99      12.400   5.120   3.840  1.00 18.20           C
ATOM      2  CG  TYR A  99      11.620   4.180   3.010  1.00 18.90           C
ATOM      3  CE1 TYR A  99      10.810   3.220   3.580  1.00 19.40           C
ATOM      4  OH  TYR A  99       9.980   2.360   2.920  1.00 20.80           O
ATOM      5  CA  PHE A 174      13.950   3.850  -0.420  1.00 19.10           C
ATOM      6  CG  PHE A 174      12.880   3.140   0.320  1.00 19.60           C
ATOM      7  CZ  PHE A 174      11.720   2.380   0.950  1.00 20.10           C
ATOM      8  CA  TRP A 215      10.350   1.180  -0.380  1.00 17.50           C
ATOM      9  CD2 TRP A 215      11.240   1.620   0.640  1.00 17.90           C
ATOM     10  NE1 TRP A 215      11.860   0.640   1.380  1.00 18.40           N
ATOM     11  CZ2 TRP A 215      11.680   2.880   1.020  1.00 18.20           C
ATOM     12  O   GLY A 216       8.920   0.420   1.650  1.00 17.80           O
ATOM     13  N   GLY A 216       9.440  -0.450   2.520  1.00 18.10           N
ATOM     14  O   GLY A 218       7.280  -1.620   3.180  1.00 19.00           O
ATOM     15  SG  CYS A 191       7.850   1.480   4.120  1.00 18.60           S
ATOM     16  SG  CYS A 220       6.920   0.820   5.840  1.00 19.20           S
ATOM     17  CA  ASP A 189       4.120  -2.450   4.860  1.00 16.90           C
ATOM     18  CG  ASP A 189       5.180  -1.820   4.020  1.00 17.20           C
ATOM     19  OD1 ASP A 189       5.920  -0.950   4.500  1.00 17.60           O
ATOM     20  OD2 ASP A 189       5.340  -2.260   2.860  1.00 17.40           O
ATOM     21  CA  SER A 195       8.420  -1.880   0.250  1.00 17.10           C
ATOM     22  OG  SER A 195       7.680  -1.120   1.180  1.00 17.90           O
ATOM     23  ND2 ASN A  98      13.620   5.820   2.180  1.00 21.40           N
ATOM     24  OE1 GLN A 192       8.650   0.820   2.980  1.00 20.50           O
ATOM     25  NE2 GLN A 192       9.820   1.380   3.350  1.00 21.00           N
ATOM     26  CA  TYR A 228       3.850   0.420   4.620  1.00 17.30           C
ATOM     27  CG  TYR A 228       4.860   0.180   3.580  1.00 17.70           C
ATOM     28  OH  TYR A 228       5.780   0.920   2.840  1.00 18.30           O
ATOM     29  CB  ALA A 190       6.180  -0.420   5.180  1.00 18.00           C
ATOM     30  CB  VAL A 213       6.480   1.920   2.140  1.00 18.50           C
ATOM     31  ND1 HIS A  57       9.120  -2.950   1.850  1.00 18.20           N
ATOM     32  NE2 HIS A  57       8.380  -2.420   2.840  1.00 18.50           N
HETATM   33  C1  RPR A 401      11.520   2.820   2.120  1.00 19.20           C
HETATM   34  C2  RPR A 401      10.680   1.920   2.480  1.00 19.00           C
HETATM   35  N3  RPR A 401       9.580   1.120   1.920  1.00 18.80           N
HETATM   36  C4  RPR A 401       8.480   0.380   2.420  1.00 18.70           C
HETATM   37  O5  RPR A 401       8.120  -0.620   1.820  1.00 18.90           O
HETATM   38  C6  RPR A 401       7.420  -0.120   3.380  1.00 18.60           C
HETATM   39  C7  RPR A 401       6.380  -0.880   3.820  1.00 18.40           C
HETATM   40  N8  RPR A 401       5.620  -1.420   3.120  1.00 18.50           N
HETATM   41  O   HOH A2012       5.620   0.380   3.960  1.00 28.40           O
HETATM   42  O   HOH A2013      11.880   3.180   1.620  1.00 34.20           O
HETATM   43  O   HOH A2017       7.380   1.180   4.860  1.00 31.50           O
HETATM   44  O   HOH A2018      12.320   2.420   0.480  1.00 35.80           O
HETATM   45  O   HOH A2021      11.020   2.050   2.880  1.00 33.10           O
HETATM   46  O   HOH A2004       5.480  -1.620   2.180  1.00 16.40           O
HETATM   47  O   HOH A2008       8.180  -1.680   2.420  1.00 17.20           O
END
"#
}

/// Loads the target `ParsedPdbStructure`, optional pocket center override, and target name for `cfg`.
pub fn load_watermap_pdb(
    cfg: &WaterMapConfig,
) -> (ParsedPdbStructure, Option<[f32; 3]>, String) {
    if cfg.preset == WaterMapPreset::CustomPdbWaterMap {
        if let Some(ref path_s) = cfg.custom_pdb_path {
            let trimmed = path_s.trim();
            if !trimmed.is_empty() {
                if let Ok(raw) = std::fs::read_to_string(trimmed) {
                    if let Ok(parsed) = parse_pdb_structure(&raw) {
                        let label = parsed.pdb_id.clone();
                        return (parsed, None, label);
                    }
                }
            }
        }
    }
    match cfg.preset {
        WaterMapPreset::FactorXa1ezqS1S4Hotspots | WaterMapPreset::CustomPdbWaterMap => (
            parse_pdb_structure(curated_factor_xa_1ezq_pdb_text()).expect("Valid 1EZQ Factor Xa PDB"),
            None,
            "Factor Xa".to_string(),
        ),
        WaterMapPreset::Egfr1m17BackPocketAndHingeBridge => (
            parse_pdb_structure(curated_pocket_pdb_text(0)).expect("Valid 1M17 EGFR PDB"),
            None,
            "EGFR".to_string(),
        ),
        WaterMapPreset::Cox23ln1DewettedHydrophobicChannel => (
            parse_pdb_structure(curated_pocket_pdb_text(2)).expect("Valid 3LN1 COX-2 PDB"),
            None,
            "COX-2".to_string(),
        ),
        WaterMapPreset::Kras6oimSwitchIiCrypticSolvation => (
            parse_pdb_structure(curated_allosteric_pdb_text(1)).expect("Valid 6OIM KRAS PDB"),
            Some([12.90, 5.10, 8.20]),
            "KRAS-G12C".to_string(),
        ),
    }
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn is_polar_receptor_atom(elem: Element, atom_name: &str) -> bool {
    matches!(elem, Element::N | Element::O | Element::F)
        || atom_name.starts_with('N')
        || atom_name.starts_with('O')
}

#[derive(Clone)]
struct RawVoxelProbe {
    pos: [f32; 3],
    is_crystal_seed: bool,
    g_sw: f64,
    occupancy: f64,
    protein_hb: f64,
    hydrophobic_contacts: usize,
    polar_contacts: usize,
    e_sw_raw: f64,
    orient_restriction: f64,
    lining: Vec<String>,
}

/// Twelve canonical water dipole orientations on the unit sphere for single-water orientational Boltzmann integration.
const WATER_DIPOLE_DIRS: [[f32; 3]; 12] = [
    [0.0, 0.5257, 0.8507],
    [0.0, -0.5257, 0.8507],
    [0.0, 0.5257, -0.8507],
    [0.0, -0.5257, -0.8507],
    [0.5257, 0.8507, 0.0],
    [-0.5257, 0.8507, 0.0],
    [0.5257, -0.8507, 0.0],
    [-0.5257, -0.8507, 0.0],
    [0.8507, 0.0, 0.5257],
    [-0.8507, 0.0, 0.5257],
    [0.8507, 0.0, -0.5257],
    [-0.8507, 0.0, -0.5257],
];

fn evaluate_water_probe_at_point(
    pos: [f32; 3],
    pdb: &ParsedPdbStructure,
    is_crystal_seed: bool,
) -> Option<RawVoxelProbe> {
    let mut min_prot_d = f32::MAX;
    let mut vdw_sum = 0.0_f64;
    let mut hydrophobic_contacts = 0_usize;
    let mut polar_contacts = 0_usize;
    let mut polar_vectors: Vec<([f32; 3], f64)> = Vec::new();
    let mut lining: Vec<(f32, String)> = Vec::new();

    for pa in &pdb.protein_atoms {
        if matches!(pa.elem, Element::H) {
            continue;
        }
        let d = dist3(pos, pa.pos);
        if d < min_prot_d {
            min_prot_d = d;
        }
        let clash_floor = if is_crystal_seed { 2.15 } else { 2.42 };
        if d < clash_floor {
            return None;
        }
        if d <= 5.20 {
            let polar = is_polar_receptor_atom(pa.elem, &pa.atom_name);
            let r_opt = if polar { 2.85_f64 } else { 3.65_f64 };
            let eps = if polar { 0.32_f64 } else { 0.16_f64 };
            let ratio = r_opt / (d as f64).max(2.15);
            let r6 = ratio.powi(6);
            let lj = eps * (r6 * r6 - 2.0 * r6);
            vdw_sum += lj.clamp(-0.42, 1.80);

            if d <= 4.35 {
                if polar {
                    polar_contacts += 1;
                    if d <= 3.55 {
                        let ux = (pa.pos[0] - pos[0]) / d;
                        let uy = (pa.pos[1] - pos[1]) / d;
                        let uz = (pa.pos[2] - pos[2]) / d;
                        let radial = (-(((d as f64) - 2.84) / 0.42).powi(2)).exp();
                        let charge_boost = if matches!(pa.res_name.as_str(), "ASP" | "GLU" | "ARG" | "LYS") {
                            1.32
                        } else {
                            1.0
                        };
                        polar_vectors.push(([ux, uy, uz], (2.15 * radial * charge_boost).clamp(0.25, 2.85)));
                    }
                } else if matches!(pa.elem, Element::C | Element::S) {
                    hydrophobic_contacts += 1;
                }
                let lbl = format!("{}{}", pa.res_name, pa.res_seq);
                if !lining.iter().any(|(_, l)| l == &lbl) {
                    lining.push((d, lbl));
                }
            }
        }
    }

    if !is_crystal_seed && (min_prot_d > 4.65 || (hydrophobic_contacts + polar_contacts) < 2) {
        return None;
    }

    lining.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let lining_labels: Vec<String> = lining.into_iter().take(4).map(|(_, s)| s).collect();

    let rt = 0.592_f64;
    let mut orient_energies = [0.0_f64; 12];
    let mut orient_hb_counts = [0.0_f64; 12];
    for (k, dir) in WATER_DIPOLE_DIRS.iter().enumerate() {
        let mut e_hb = 0.0_f64;
        let mut hb_c = 0.0_f64;
        for (pvec, strength) in &polar_vectors {
            let dot = (dir[0] * pvec[0] + dir[1] * pvec[1] + dir[2] * pvec[2]) as f64;
            let align = dot.abs().max((1.0 - (dot.abs() - 0.577).abs() * 1.35).clamp(0.0, 1.0));
            if align > 0.52 {
                let contrib = strength * ((align - 0.35) / 0.65).clamp(0.0, 1.0);
                e_hb -= contrib;
                hb_c += (contrib / 1.85).clamp(0.0, 1.15);
            }
        }
        orient_energies[k] = e_hb;
        orient_hb_counts[k] = hb_c.min(3.4);
    }

    let min_e = orient_energies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let mut z_rot = 0.0_f64;
    let mut probs = [0.0_f64; 12];
    for k in 0..12 {
        let w = (-(orient_energies[k] - min_e) / rt).exp();
        probs[k] = w;
        z_rot += w;
    }
    let mut mean_e_hb = 0.0_f64;
    let mut mean_prot_hb = 0.0_f64;
    let mut shannon_orient = 0.0_f64;
    for k in 0..12 {
        let pk = probs[k] / z_rot.max(1e-12);
        mean_e_hb += pk * orient_energies[k];
        mean_prot_hb += pk * orient_hb_counts[k];
        if pk > 1e-12 {
            shannon_orient -= pk * (pk * 12.0).ln();
        }
    }
    let orient_restriction = (-rt * shannon_orient).clamp(0.0, 2.35);
    let e_sw_raw = (vdw_sum + mean_e_hb - 1.15 * (polar_contacts.min(4) as f64) * 0.28).clamp(-10.8, 1.4);

    let enclosure_ratio =
        (hydrophobic_contacts as f64) / ((hydrophobic_contacts + polar_contacts * 2).max(1) as f64);
    let g_sw = if polar_contacts >= 2 {
        (1.85 + 0.65 * mean_prot_hb + 0.14 * (hydrophobic_contacts.min(5) as f64)).clamp(1.6, 5.4)
    } else if hydrophobic_contacts >= 5 && polar_contacts == 0 {
        if is_crystal_seed {
            1.45
        } else {
            (0.24 - 0.02 * ((hydrophobic_contacts.saturating_sub(5)) as f64)).clamp(0.10, 0.36)
        }
    } else {
        (1.15 + 0.48 * mean_prot_hb + 0.18 * (1.0 - enclosure_ratio) * (hydrophobic_contacts as f64))
            .clamp(0.45, 3.80)
    };
    let occupancy = if is_crystal_seed {
        (0.58 + 0.11 * mean_prot_hb - 0.04 * (hydrophobic_contacts.saturating_sub(4) as f64))
            .clamp(0.42, 0.98)
    } else if g_sw < 0.40 {
        (g_sw * 0.85).clamp(0.08, 0.34)
    } else {
        (g_sw / (g_sw + 1.35)).clamp(0.25, 0.96)
    };

    Some(RawVoxelProbe {
        pos,
        is_crystal_seed,
        g_sw,
        occupancy,
        protein_hb: mean_prot_hb,
        hydrophobic_contacts,
        polar_contacts,
        e_sw_raw,
        orient_restriction,
        lining: lining_labels,
    })
}

struct WaterRGroupSpec {
    code: &'static str,
    name: &'static str,
    smiles_branch: &'static str,
    added_hac: usize,
    approx_reach_a: f32,
    is_polar_mimic: bool,
    strategy: RGroupSolvationStrategy,
    reaction: &'static str,
}

const WATER_RGROUP_LIBRARY: [WaterRGroupSpec; 14] = [
    WaterRGroupSpec {
        code: "RG-CL",
        name: "Chloro (-Cl)",
        smiles_branch: "Cl",
        added_hac: 1,
        approx_reach_a: 1.76,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::LipophilicHotspotDisplacer,
        reaction: "Late-Stage Electrophilic / NCS Aryl-H Chlorination",
    },
    WaterRGroupSpec {
        code: "RG-ME",
        name: "Methyl (-CH3)",
        smiles_branch: "C",
        added_hac: 1,
        approx_reach_a: 1.53,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::LipophilicHotspotDisplacer,
        reaction: "Suzuki-Miyaura / Photoredox Minisci Methylation",
    },
    WaterRGroupSpec {
        code: "RG-CN",
        name: "Nitrile (-CN)",
        smiles_branch: "C#N",
        added_hac: 2,
        approx_reach_a: 2.62,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::PolarWaterBridgeMimic,
        reaction: "Pd-Catalyzed Zn(CN)2 Cyanation (Water-Displacing HBA)",
    },
    WaterRGroupSpec {
        code: "RG-CF3",
        name: "Trifluoromethyl (-CF3)",
        smiles_branch: "C(F)(F)F",
        added_hac: 4,
        approx_reach_a: 2.48,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::FluorinatedEnclosureFiller,
        reaction: "Baran Langlois Radical / Ruppert-Prakash CF3 Coupling",
    },
    WaterRGroupSpec {
        code: "RG-CHF2",
        name: "Difluoromethyl (-CHF2)",
        smiles_branch: "C(F)F",
        added_hac: 3,
        approx_reach_a: 2.32,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::FluorinatedEnclosureFiller,
        reaction: "Photoredox Difluoromethylation (Lipophilic C-H HBD Mimic)",
    },
    WaterRGroupSpec {
        code: "RG-OH",
        name: "Hydroxyl (-OH)",
        smiles_branch: "O",
        added_hac: 1,
        approx_reach_a: 1.42,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::PolarWaterBridgeMimic,
        reaction: "Miyaura Borylation-Oxidation (Direct HOH Donor/Acceptor Mimic)",
    },
    WaterRGroupSpec {
        code: "RG-CH2OH",
        name: "Hydroxymethyl (-CH2OH)",
        smiles_branch: "CO",
        added_hac: 2,
        approx_reach_a: 2.45,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::PolarWaterBridgeMimic,
        reaction: "Reductive Formylation / Suzuki Potassiumoxymethyltrifluoroborate",
    },
    WaterRGroupSpec {
        code: "RG-OME",
        name: "Methoxy (-OCH3)",
        smiles_branch: "OC",
        added_hac: 2,
        approx_reach_a: 2.42,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::PolarWaterBridgeMimic,
        reaction: "Buchwald-Hartwig / Ullmann C-O Methoxylation",
    },
    WaterRGroupSpec {
        code: "RG-CPR",
        name: "Cyclopropyl (-cPr)",
        smiles_branch: "C1CC1",
        added_hac: 3,
        approx_reach_a: 2.75,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::LipophilicHotspotDisplacer,
        reaction: "Suzuki Cyclopropylboronic Acid Coupling",
    },
    WaterRGroupSpec {
        code: "RG-ET",
        name: "Ethyl (-CH2CH3)",
        smiles_branch: "CC",
        added_hac: 2,
        approx_reach_a: 2.55,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::LipophilicHotspotDisplacer,
        reaction: "Negishi / Suzuki-Miyaura Alkylation",
    },
    WaterRGroupSpec {
        code: "RG-F",
        name: "Fluoro (-F)",
        smiles_branch: "F",
        added_hac: 1,
        approx_reach_a: 1.38,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::FluorinatedEnclosureFiller,
        reaction: "Selectfluor / Balz-Schiemann Late-Stage Fluorination",
    },
    WaterRGroupSpec {
        code: "RG-OCF3",
        name: "Trifluoromethoxy (-OCF3)",
        smiles_branch: "OC(F)(F)F",
        added_hac: 5,
        approx_reach_a: 3.10,
        is_polar_mimic: false,
        strategy: RGroupSolvationStrategy::FluorinatedEnclosureFiller,
        reaction: "Ag-Mediated Trifluoromethoxylation",
    },
    WaterRGroupSpec {
        code: "RG-CONH2",
        name: "Carboxamide (-CONH2)",
        smiles_branch: "C(=O)N",
        added_hac: 3,
        approx_reach_a: 2.85,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::PolarWaterBridgeMimic,
        reaction: "Pd-Catalyzed Aminocarbonylation (Dual HBD/HBA Bridge Mimic)",
    },
    WaterRGroupSpec {
        code: "RG-PYR",
        name: "1H-Pyrazol-4-yl",
        smiles_branch: "c1c[nH]nc1",
        added_hac: 5,
        approx_reach_a: 3.65,
        is_polar_mimic: true,
        strategy: RGroupSolvationStrategy::PolarWaterBridgeMimic,
        reaction: "Suzuki-Miyaura Pyrazole Boronic Ester Coupling",
    },
];

fn build_rgroup_analog_smiles(
    parent: &Molecule,
    anchor_idx: usize,
    branch: &str,
) -> Option<(String, Molecule)> {
    let base_smi = if !parent.smiles.trim().is_empty() {
        parent.smiles.trim().to_string()
    } else {
        molecule_to_smiles(parent)
    };
    let bytes = base_smi.as_bytes();
    let mut atom_ord = 0_usize;
    let mut insert_byte: Option<usize> = None;
    let mut i = 0_usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'[' {
            if let Some(rel_end) = base_smi[i..].find(']') {
                let end_i = i + rel_end + 1;
                if atom_ord == anchor_idx {
                    insert_byte = Some(end_i);
                    break;
                }
                atom_ord += 1;
                i = end_i;
                continue;
            }
        } else if (b == b'C' && i + 1 < bytes.len() && bytes[i + 1] == b'l')
            || (b == b'B' && i + 1 < bytes.len() && bytes[i + 1] == b'r')
        {
            let mut end_i = i + 2;
            while end_i < bytes.len() && (bytes[end_i].is_ascii_digit() || bytes[end_i] == b'%') {
                end_i += 1;
            }
            if atom_ord == anchor_idx {
                insert_byte = Some(end_i);
                break;
            }
            atom_ord += 1;
            i = end_i;
            continue;
        } else if matches!(
            b,
            b'B' | b'C' | b'N' | b'O' | b'P' | b'S' | b'F' | b'I' | b'b' | b'c' | b'n' | b'o' | b'p' | b's'
        ) {
            let mut end_i = i + 1;
            while end_i < bytes.len() && (bytes[end_i].is_ascii_digit() || bytes[end_i] == b'%') {
                end_i += 1;
            }
            if atom_ord == anchor_idx {
                insert_byte = Some(end_i);
                break;
            }
            atom_ord += 1;
            i = end_i;
            continue;
        }
        i += 1;
    }

    if let Some(pos) = insert_byte {
        let cand = format!("{}({}){}", &base_smi[..pos], branch, &base_smi[pos..]);
        if let Ok(parsed) = parse_smiles(&cand) {
            if parsed.atoms.len() > parent.atoms.len() {
                let canon = molecule_to_smiles(&parsed);
                return Some((canon, parsed));
            }
        }
    }
    let fallback = format!("{}({})", base_smi, branch);
    if let Ok(parsed) = parse_smiles(&fallback) {
        if parsed.atoms.len() > parent.atoms.len() {
            let canon = molecule_to_smiles(&parsed);
            return Some((canon, parsed));
        }
    }
    None
}

/// Core `WaterMap` / `GIST` solvation mapper and R-group designer operating on a pre-parsed PDB & pocket.
pub fn analyze_watermap_with_prebuilt(
    mol: &Molecule,
    _ds: Option<&DatasetAnalysis>,
    cfg: &WaterMapConfig,
    pdb: &ParsedPdbStructure,
    pocket: &PocketCavityAnalysis,
    target_name: &str,
    enumerate_rgroups: bool,
) -> WaterMapSolvationReport {
    let spacing = cfg.gist_voxel_spacing_a.clamp(0.50, 1.25);
    let r_co = cfg.displacement_cutoff_a.clamp(1.60, 3.20);
    let pocket_res = score_ligand_in_pocket(mol, pocket, None);
    let lig_coords = pocket_res.aligned_ligand_coords3d.clone();

    let half_span = 6.75_f32;
    let n_steps = ((2.0 * half_span) / spacing).round().clamp(8.0, 26.0) as i32;
    let half_i = n_steps / 2;

    let mut raw_probes: Vec<RawVoxelProbe> = Vec::new();
    for w in &pdb.waters {
        if dist3(w.pos, pocket.center) <= pocket.pocket_radius_a + 2.5 {
            if let Some(pr) = evaluate_water_probe_at_point(w.pos, pdb, true) {
                raw_probes.push(pr);
            }
        }
    }

    let mut grid_probes: Vec<RawVoxelProbe> = Vec::new();
    for ix in -half_i..=half_i {
        for iy in -half_i..=half_i {
            for iz in -half_i..=half_i {
                let pt = [
                    pocket.center[0] + (ix as f32) * spacing,
                    pocket.center[1] + (iy as f32) * spacing,
                    pocket.center[2] + (iz as f32) * spacing,
                ];
                if dist3(pt, pocket.center) > pocket.pocket_radius_a + 0.8 {
                    continue;
                }
                if let Some(pr) = evaluate_water_probe_at_point(pt, pdb, false) {
                    grid_probes.push(pr);
                }
            }
        }
    }

    let total_gist_voxels = grid_probes.len();
    let voxel_vol = (spacing as f64).powi(3);
    let mut high_energy_voxels = 0_usize;
    let mut dewetted_voxels = 0_usize;
    let mut integrated_dg = 0.0_f64;
    let mut gist_cloud: Vec<GistHotspotVoxel> = Vec::new();

    for gp in &grid_probes {
        let nbr_est = (1.15 + 2.85 * (1.0 - (gp.hydrophobic_contacts as f64 / 9.0).clamp(0.0, 0.85)))
            .clamp(0.8, 4.6);
        let e_ww = -1.28 * nbr_est;
        let dh = (gp.e_sw_raw + e_ww - WATERMAP_BULK_ENERGY_KCAL).clamp(-4.2, 6.8);
        let tds_tr = if gp.g_sw > 1.0 {
            (0.592 * gp.g_sw.ln() * 0.72).clamp(0.0, 1.65)
        } else {
            0.18
        };
        let neg_tds = (tds_tr + gp.orient_restriction * 0.78).clamp(0.15, 3.20);
        let dg_norm = dh + neg_tds;
        let is_dewet = gp.g_sw < 0.38 && dh >= 2.8;
        if is_dewet {
            dewetted_voxels += 1;
        }
        if dg_norm >= cfg.hotspot_dg_threshold_kcal {
            high_energy_voxels += 1;
        }
        integrated_dg += 0.0334 * voxel_vol * gp.g_sw.max(0.25) * dg_norm * 0.28;
        if (dg_norm >= cfg.hotspot_dg_threshold_kcal + 0.45 || is_dewet || dg_norm <= -1.25)
            && gist_cloud.len() < 140
        {
            gist_cloud.push(GistHotspotVoxel {
                pos: gp.pos,
                g_sw_density: gp.g_sw,
                dh_norm_kcal: dh,
                neg_tds_norm_kcal: neg_tds,
                dg_norm_kcal: dg_norm,
                is_dewetted_void: is_dewet,
            });
        }
    }

    let mut candidates = raw_probes;
    let mut filtered_grid: Vec<RawVoxelProbe> = grid_probes
        .into_iter()
        .filter(|p| p.g_sw >= 1.68 || (p.g_sw < 0.38 && p.hydrophobic_contacts >= 5))
        .collect();
    filtered_grid.sort_by(|a, b| {
        let sa = if a.g_sw < 0.38 {
            4.5 + a.hydrophobic_contacts as f64 * 0.2
        } else {
            a.g_sw + a.orient_restriction * 0.6
        };
        let sb = if b.g_sw < 0.38 {
            4.5 + b.hydrophobic_contacts as f64 * 0.2
        } else {
            b.g_sw + b.orient_restriction * 0.6
        };
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.extend(filtered_grid);

    let mut chosen_probes: Vec<RawVoxelProbe> = Vec::new();
    for cand in candidates {
        if chosen_probes.iter().any(|ex| dist3(ex.pos, cand.pos) < 2.25) {
            continue;
        }
        chosen_probes.push(cand);
        if chosen_probes.len() >= 18 {
            break;
        }
    }

    let mut sites: Vec<WaterMapHydrationSite> = Vec::with_capacity(chosen_probes.len());
    for (idx, pr) in chosen_probes.iter().enumerate() {
        let mut explicit_water_nbrs = 0.0_f64;
        for (j, other) in chosen_probes.iter().enumerate() {
            if idx != j {
                let d_oo = dist3(pr.pos, other.pos);
                if d_oo <= 3.65 {
                    explicit_water_nbrs += other.occupancy;
                }
            }
        }
        let open_bulk_nbrs = (2.75
            * (1.0 - (pr.hydrophobic_contacts as f64 / 9.5).clamp(0.0, 0.90))
            * (1.0 - (pr.polar_contacts as f64 / 8.0).clamp(0.0, 0.65)))
        .clamp(0.25, 2.80);
        let nbrs_3_5a = (explicit_water_nbrs + open_bulk_nbrs).clamp(0.40, 4.70);
        let water_hb = (nbrs_3_5a * 0.74).min((4.0 - pr.protein_hb).max(0.25));
        let total_hb = pr.protein_hb + water_hb;
        let pct_bulk_hb = ((total_hb / WATERMAP_BULK_HBONDS) * 100.0).clamp(8.0, 125.0);
        let exposure = (nbrs_3_5a / WATERMAP_BULK_NEIGHBORS).clamp(0.08, 1.0);
        let enclosure = (1.0 - exposure * 0.82 + 0.06 * (pr.hydrophobic_contacts.min(7) as f64))
            .clamp(0.05, 1.0);

        let e_ww = (-1.34 * water_hb - 0.36 * (nbrs_3_5a - water_hb).max(0.0)).clamp(-6.80, -0.35);
        let e_sw = pr.e_sw_raw;
        let dh_excess = (e_sw + e_ww - WATERMAP_BULK_ENERGY_KCAL).clamp(-4.10, 6.40);

        let neg_tds_trans = if pr.g_sw >= 1.0 {
            (0.42 + 0.592 * pr.g_sw.ln() * 0.68 + 0.35 * enclosure).clamp(0.25, 1.85)
        } else {
            (0.22 + 0.25 * enclosure).clamp(0.15, 0.65)
        };
        let neg_tds_orient = (pr.orient_restriction * 0.76
            + if pr.hydrophobic_contacts >= 3 && pr.polar_contacts >= 1 {
                0.58
            } else {
                0.14
            })
        .clamp(0.15, 2.15);
        let neg_tds_excess = if pr.occupancy < 0.38 && dh_excess >= 3.0 {
            (neg_tds_trans * 0.45).clamp(0.10, 0.55)
        } else {
            neg_tds_trans + neg_tds_orient
        };
        let dg_excess = dh_excess + neg_tds_excess;

        let class = if pr.occupancy < 0.38 && dh_excess >= 3.00 {
            HydrationSiteClass::DewettedVacuumCavity
        } else if dg_excess <= -1.10 {
            HydrationSiteClass::ConservedStructuralBridge
        } else if dh_excess >= 1.30 && dg_excess >= cfg.hotspot_dg_threshold_kcal {
            HydrationSiteClass::EnthalpyFrustratedHotspot
        } else if neg_tds_excess >= 1.40 && dg_excess >= cfg.hotspot_dg_threshold_kcal * 0.90 {
            HydrationSiteClass::EntropyOrderedEnclosure
        } else {
            HydrationSiteClass::BulkLikeNeutral
        };

        let mut f_sum = 0.0_f64;
        let mut min_lig_d = f32::MAX;
        let mut polar_displacer = false;
        let mut polar_bridge_contact = false;
        for (a_idx, &lpos) in lig_coords.iter().enumerate() {
            let d = dist3(pr.pos, lpos);
            if d < min_lig_d {
                min_lig_d = d;
            }
            let is_lig_polar = mol
                .atoms
                .get(a_idx)
                .map(|at| matches!(at.elem, Element::N | Element::O | Element::F))
                .unwrap_or(false);
            if d < r_co {
                let contrib = (1.0 - (d / r_co)) as f64;
                f_sum += contrib;
                if is_lig_polar && contrib >= 0.25 {
                    polar_displacer = true;
                }
            } else if is_lig_polar && (2.40..=3.55).contains(&d) {
                polar_bridge_contact = true;
            }
        }
        let f_disp = f_sum.clamp(0.0, 1.0);

        let (status, lig_dg_contrib) = if f_disp >= 0.35 {
            if polar_displacer {
                let gain = if class == HydrationSiteClass::ConservedStructuralBridge {
                    -0.35 * f_disp
                } else {
                    -(dg_excess.max(0.25)) * f_disp * 0.78 - 0.25 * f_disp
                };
                (SiteLigandStatus::ReplacedByPolarMimic, gain)
            } else if class == HydrationSiteClass::ConservedStructuralBridge {
                let pen = (-dg_excess).max(1.10) * f_disp * 0.85;
                (SiteLigandStatus::DisplacedNonpolar, pen)
            } else {
                let gain = -dg_excess * f_disp * 0.74;
                (SiteLigandStatus::DisplacedNonpolar, gain)
            }
        } else if polar_bridge_contact && pr.protein_hb >= 0.8 {
            let bridge_bonus = if class == HydrationSiteClass::ConservedStructuralBridge {
                -0.68
            } else {
                -0.28
            };
            (SiteLigandStatus::BridgedByLigand, bridge_bonus)
        } else if min_lig_d <= 5.65 {
            (SiteLigandStatus::ProximalGrowthTarget, 0.0)
        } else {
            (SiteLigandStatus::DistalUnoccupied, 0.0)
        };

        sites.push(WaterMapHydrationSite {
            site_id: format!("WM-S{:02}", idx + 1),
            pos: pr.pos,
            is_crystallographic_seed: pr.is_crystal_seed,
            class,
            occupancy: pr.occupancy,
            local_density_g_sw: pr.g_sw,
            protein_hbonds: pr.protein_hb,
            water_hbonds: water_hb,
            neighbor_waters_3_5a: nbrs_3_5a,
            percent_bulk_hb: pct_bulk_hb,
            exposure_fraction: exposure,
            hydrophobic_enclosure: enclosure,
            e_solute_water_kcal: e_sw,
            e_water_water_kcal: e_ww,
            dh_excess_kcal: dh_excess,
            neg_tds_trans_kcal: neg_tds_trans,
            neg_tds_orient_kcal: neg_tds_orient,
            neg_tds_excess_kcal: neg_tds_excess,
            dg_excess_kcal: dg_excess,
            lining_residues: pr.lining.clone(),
            ligand_overlap_f_disp: f_disp,
            min_ligand_dist_a: if min_lig_d.is_finite() { min_lig_d } else { 99.9 },
            status,
            ligand_dg_contribution_kcal: lig_dg_contrib,
        });
    }

    sites.sort_by(|a, b| {
        b.dg_excess_kcal
            .partial_cmp(&a.dg_excess_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (k, s) in sites.iter_mut().enumerate() {
        s.site_id = format!("WM-S{:02}", k + 1);
    }

    let unhappy_sites = sites
        .iter()
        .filter(|s| {
            matches!(
                s.class,
                HydrationSiteClass::EnthalpyFrustratedHotspot
                    | HydrationSiteClass::EntropyOrderedEnclosure
            )
        })
        .count();
    let dewetted_sites = sites
        .iter()
        .filter(|s| s.class == HydrationSiteClass::DewettedVacuumCavity)
        .count();
    let bridge_sites = sites
        .iter()
        .filter(|s| s.class == HydrationSiteClass::ConservedStructuralBridge)
        .count();

    let mut displaced_hotspots = 0_usize;
    let mut retained_bridges = 0_usize;
    let mut clashing_bridges = 0_usize;
    let mut dh_disp_gain = 0.0_f64;
    let mut tds_disp_gain = 0.0_f64;
    let mut bridge_penalty = 0.0_f64;
    let mut net_wm_dg = 0.0_f64;
    let mut untapped_dg = 0.0_f64;

    for s in &sites {
        net_wm_dg += s.ligand_dg_contribution_kcal;
        match s.status {
            SiteLigandStatus::DisplacedNonpolar | SiteLigandStatus::ReplacedByPolarMimic => {
                if s.dg_excess_kcal >= cfg.hotspot_dg_threshold_kcal * 0.85
                    || s.class == HydrationSiteClass::DewettedVacuumCavity
                {
                    displaced_hotspots += 1;
                }
                if s.ligand_dg_contribution_kcal > 0.15
                    && s.class == HydrationSiteClass::ConservedStructuralBridge
                {
                    clashing_bridges += 1;
                    bridge_penalty += s.ligand_dg_contribution_kcal;
                } else {
                    dh_disp_gain += -s.dh_excess_kcal.max(0.0) * s.ligand_overlap_f_disp * 0.68;
                    tds_disp_gain += -s.neg_tds_excess_kcal * s.ligand_overlap_f_disp * 0.62;
                }
            }
            SiteLigandStatus::BridgedByLigand => {
                retained_bridges += 1;
            }
            SiteLigandStatus::ProximalGrowthTarget => {
                if s.dg_excess_kcal > 0.85 {
                    untapped_dg += s.dg_excess_kcal * 0.74;
                }
            }
            SiteLigandStatus::DistalUnoccupied => {}
        }
    }

    let watermap_refined_dg =
        pocket_res.total_dg_pocket_kcal - pocket_res.water_displacement_kcal * 0.65 + net_wm_dg;
    let watermap_pred_pkd = (-watermap_refined_dg / RT_LN10_KCAL).clamp(3.0, 12.8);

    let mut designs: Vec<DisplaceableWaterRGroupDesign> = Vec::new();
    if enumerate_rgroups && cfg.design_rgroup_vectors && !lig_coords.is_empty() {
        let parent_physchem = evaluate_physchem(mol);
        let parent_lle = mol.pact - parent_physchem.clogp;
        let exit_vecs = perceive_fragment_exit_vectors(mol);

        for ev in exit_vecs.iter().take(8) {
            let Some(&anchor_pos) = lig_coords.get(ev.anchor_atom) else {
                continue;
            };
            let mut nbr_sum = [0.0_f32; 3];
            let mut nbr_cnt = 0_usize;
            if let Some(adj_list) = mol.adj.get(ev.anchor_atom) {
                for &(nb, _) in adj_list {
                    if let Some(&np) = lig_coords.get(nb) {
                        nbr_sum[0] += np[0];
                        nbr_sum[1] += np[1];
                        nbr_sum[2] += np[2];
                        nbr_cnt += 1;
                    }
                }
            }
            let out_dir = if nbr_cnt > 0 {
                let cx = nbr_sum[0] / (nbr_cnt as f32);
                let cy = nbr_sum[1] / (nbr_cnt as f32);
                let cz = nbr_sum[2] / (nbr_cnt as f32);
                let vx = anchor_pos[0] - cx;
                let vy = anchor_pos[1] - cy;
                let vz = anchor_pos[2] - cz;
                let vn = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-4);
                [vx / vn, vy / vn, vz / vn]
            } else {
                [1.0, 0.0, 0.0]
            };

            for site in &sites {
                if site.ligand_overlap_f_disp >= 0.72 {
                    continue;
                }
                if site.class == HydrationSiteClass::BulkLikeNeutral && site.dg_excess_kcal < 0.75 {
                    continue;
                }
                let vec_d = dist3(anchor_pos, site.pos);
                if !(1.45..=5.35).contains(&vec_d) {
                    continue;
                }
                let sx = (site.pos[0] - anchor_pos[0]) / vec_d;
                let sy = (site.pos[1] - anchor_pos[1]) / vec_d;
                let sz = (site.pos[2] - anchor_pos[2]) / vec_d;
                let cos_th = (out_dir[0] * sx + out_dir[1] * sy + out_dir[2] * sz).clamp(-1.0, 1.0);
                if cos_th < 0.18 {
                    continue;
                }
                let align_deg = cos_th.acos().to_degrees();

                for rg in &WATER_RGROUP_LIBRARY {
                    match site.class {
                        HydrationSiteClass::ConservedStructuralBridge => {
                            if !rg.is_polar_mimic {
                                continue;
                            }
                        }
                        HydrationSiteClass::DewettedVacuumCavity
                        | HydrationSiteClass::EnthalpyFrustratedHotspot => {
                            if rg.code == "RG-CONH2" || rg.code == "RG-PYR" {
                                continue;
                            }
                        }
                        HydrationSiteClass::EntropyOrderedEnclosure
                        | HydrationSiteClass::BulkLikeNeutral => {}
                    }

                    let reach_err = (vec_d - rg.approx_reach_a).abs();
                    if reach_err > 1.55 {
                        continue;
                    }

                    let Some((cand_smi, cand_mol)) =
                        build_rgroup_analog_smiles(mol, ev.anchor_atom, rg.smiles_branch)
                    else {
                        continue;
                    };
                    if designs.iter().any(|d| d.designed_smiles == cand_smi) {
                        continue;
                    }

                    let phys = evaluate_physchem(&cand_mol);
                    let sa = evaluate_synthetic_accessibility(&cand_mol).sa_score;
                    let pains_cnt = scan_structural_alerts(&cand_mol)
                        .alerts
                        .iter()
                        .filter(|a| a.family.eq_ignore_ascii_case("PAINS"))
                        .count();

                    let reach_fit = (1.0 - (reach_err as f64) / 1.75).clamp(0.25, 1.0);
                    let angle_fit = ((cos_th as f64 + 0.15) / 1.15).clamp(0.30, 1.0);
                    let pred_f = (reach_fit * angle_fit * 0.96).clamp(0.32, 0.98);

                    let remaining_frac = (1.0 - site.ligand_overlap_f_disp).clamp(0.25, 1.0);
                    let base_liberation = if site.class == HydrationSiteClass::ConservedStructuralBridge
                    {
                        -(site.neg_tds_excess_kcal * 0.62 + 0.38) * pred_f * remaining_frac
                    } else {
                        -(site.dg_excess_kcal.max(0.80) * 0.78 + 0.28) * pred_f * remaining_frac
                    };
                    let polarity_match_bonus = match (site.class, rg.is_polar_mimic) {
                        (HydrationSiteClass::EntropyOrderedEnclosure, true) => -0.34,
                        (HydrationSiteClass::EnthalpyFrustratedHotspot, false) => -0.32,
                        (HydrationSiteClass::DewettedVacuumCavity, false) => -0.45,
                        (HydrationSiteClass::ConservedStructuralBridge, true) => -0.36,
                        _ => -0.08,
                    };
                    let ddg = (base_liberation + polarity_match_bonus).clamp(-3.45, -0.15);
                    let dpact = -ddg / RT_LN10_KCAL;
                    let pred_pact = mol.pact + dpact;
                    let ge = (-ddg) / (rg.added_hac.max(1) as f64);
                    let cand_lle = pred_pact - phys.clogp;
                    let dlle = cand_lle - parent_lle;

                    let score = (42.0
                        + (-ddg) * 18.5
                        + ge * 22.0
                        + dlle * 8.5
                        - (sa - 2.5).max(0.0) * 4.2
                        - (pains_cnt as f64) * 25.0)
                        .clamp(5.0, 99.6);

                    designs.push(DisplaceableWaterRGroupDesign {
                        rank: 0,
                        design_id: format!("{}-{}", site.site_id, rg.code),
                        target_site_id: site.site_id.clone(),
                        target_site_class: site.class,
                        target_site_dg_kcal: site.dg_excess_kcal,
                        anchor_atom_idx: ev.anchor_atom,
                        anchor_atom_label: ev.atom_label.clone(),
                        anchor_pos3d: anchor_pos,
                        target_site_pos3d: site.pos,
                        vector_distance_a: vec_d,
                        vector_alignment_deg: align_deg,
                        rgroup_code: rg.code,
                        rgroup_name: rg.name,
                        rgroup_smiles_suffix: rg.smiles_branch,
                        synthetic_reaction: rg.reaction,
                        strategy: rg.strategy,
                        designed_smiles: cand_smi,
                        added_heavy_atoms: rg.added_hac,
                        predicted_site_overlap_f: pred_f,
                        ddg_solvation_gain_kcal: ddg,
                        predicted_delta_pact: dpact,
                        predicted_pact: pred_pact,
                        group_efficiency_ge: ge,
                        delta_lle: dlle,
                        mw: phys.mw,
                        clogp: phys.clogp,
                        tpsa: phys.tpsa,
                        sa_score: sa,
                        pains_count: pains_cnt,
                        design_score: score,
                    });
                }
            }
        }

        designs.sort_by(|a, b| {
            b.design_score
                .partial_cmp(&a.design_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        designs.truncate(24);
        for (i, d) in designs.iter_mut().enumerate() {
            d.rank = i + 1;
        }
    }

    let shell_display: Vec<([f32; 3], String, Element)> = pocket
        .shell_atoms
        .iter()
        .take(90)
        .map(|a| (a.pos, format!("{}{}:{}", a.res_name, a.res_seq, a.atom_name), a.elem))
        .collect();

    WaterMapSolvationReport {
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
        baseline_pact: mol.pact,
        baseline_pocket_dg_kcal: pocket_res.total_dg_pocket_kcal,
        gist_voxel_spacing_a: spacing,
        total_gist_voxels_evaluated: total_gist_voxels,
        high_energy_gist_voxels: high_energy_voxels,
        dewetted_gist_voxels: dewetted_voxels,
        integrated_cavity_excess_dg_kcal: integrated_dg,
        total_hydration_sites: sites.len(),
        unhappy_hotspot_sites: unhappy_sites,
        dewetted_cavity_sites: dewetted_sites,
        conserved_bridge_sites: bridge_sites,
        displaced_hotspot_count: displaced_hotspots,
        retained_bridge_count: retained_bridges,
        clashing_displaced_bridges: clashing_bridges,
        dh_displacement_gain_kcal: dh_disp_gain,
        tds_displacement_gain_kcal: tds_disp_gain,
        bridge_expulsion_penalty_kcal: bridge_penalty,
        net_watermap_dg_kcal: net_wm_dg,
        watermap_refined_dg_kcal: watermap_refined_dg,
        watermap_predicted_pkd: watermap_pred_pkd,
        untapped_proximal_hotspot_dg_kcal: untapped_dg,
        aligned_ligand_coords3d: lig_coords,
        pocket_shell_display_atoms: shell_display,
        gist_hotspot_cloud: gist_cloud,
        hydration_sites: sites,
        rgroup_designs: designs,
    }
}

/// Public entry point computing active-site `IFST` / `WaterMap` hydration hotspots and R-group vectors for `mol`.
pub fn analyze_watermap_solvation(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    cfg: &WaterMapConfig,
) -> WaterMapSolvationReport {
    let (pdb, center_override, target_name) = load_watermap_pdb(cfg);
    let pocket = analyze_binding_pocket(&pdb, center_override, 9.5);
    analyze_watermap_with_prebuilt(
        mol,
        ds,
        cfg,
        &pdb,
        &pocket,
        &target_name,
        cfg.design_rgroup_vectors,
    )
}

/// Parallel Rayon screening of an entire `DatasetAnalysis` against the active-site `WaterMap` solvation field.
pub fn screen_dataset_watermap_solvation(
    ds: &DatasetAnalysis,
    cfg: &WaterMapConfig,
) -> Vec<DatasetWaterMapScreenRow> {
    let (pdb, center_override, target_name) = load_watermap_pdb(cfg);
    let pocket = analyze_binding_pocket(&pdb, center_override, 9.5);

    let mut rows: Vec<DatasetWaterMapScreenRow> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(idx, m)| {
            let rep = analyze_watermap_with_prebuilt(m, Some(ds), cfg, &pdb, &pocket, &target_name, true);
            let phys = evaluate_physchem(m);
            let top_rg = rep
                .rgroup_designs
                .first()
                .map(|d| {
                    format!(
                        "{} @ {} -> {} ({:+.2} kcal)",
                        d.rgroup_name, d.anchor_atom_label, d.target_site_id, d.ddg_solvation_gain_kcal
                    )
                })
                .unwrap_or_else(|| "Hotspots Saturated".to_string());
            DatasetWaterMapScreenRow {
                rank: 0,
                mol_idx: idx,
                compound_id: m.id.clone(),
                compound_name: m.name.clone(),
                smiles: rep.query_smiles,
                observed_pact: m.pact,
                mw: phys.mw,
                clogp: phys.clogp,
                displaced_hotspots: rep.displaced_hotspot_count,
                retained_bridges: rep.retained_bridge_count,
                clashing_bridges: rep.clashing_displaced_bridges,
                dh_disp_gain_kcal: rep.dh_displacement_gain_kcal,
                tds_disp_gain_kcal: rep.tds_displacement_gain_kcal,
                net_watermap_dg_kcal: rep.net_watermap_dg_kcal,
                watermap_refined_dg_kcal: rep.watermap_refined_dg_kcal,
                untapped_hotspot_dg_kcal: rep.untapped_proximal_hotspot_dg_kcal,
                top_rgroup_suggestion: top_rg,
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        a.watermap_refined_dg_kcal
            .partial_cmp(&b.watermap_refined_dg_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Exports all active-site `WaterMap` hydration sites and `IFST` thermodynamic components to `.csv`.
pub fn export_watermap_sites_csv(rep: &WaterMapSolvationReport) -> String {
    let mut out = String::from(
        "site_id,pdb_id,x_a,y_a,z_a,crystal_seed,class,occupancy,g_sw_density,protein_hbonds,water_hbonds,nbrs_3_5a,pct_bulk_hb,enclosure,dh_excess_kcal,neg_tds_trans_kcal,neg_tds_orient_kcal,neg_tds_excess_kcal,dg_excess_kcal,f_disp,ligand_status,ligand_dg_contrib_kcal,lining_residues\n",
    );
    for s in &rep.hydration_sites {
        out.push_str(&format!(
            "{},{},{:.3},{:.3},{:.3},{},\"{}\",{:.3},{:.3},{:.2},{:.2},{:.2},{:.1},{:.3},{:+.3},{:+.3},{:+.3},{:+.3},{:+.3},{:.3},\"{}\",{:+.3},\"{}\"\n",
            s.site_id,
            rep.pdb_id,
            s.pos[0],
            s.pos[1],
            s.pos[2],
            s.is_crystallographic_seed,
            s.class.short_label(),
            s.occupancy,
            s.local_density_g_sw,
            s.protein_hbonds,
            s.water_hbonds,
            s.neighbor_waters_3_5a,
            s.percent_bulk_hb,
            s.hydrophobic_enclosure,
            s.dh_excess_kcal,
            s.neg_tds_trans_kcal,
            s.neg_tds_orient_kcal,
            s.neg_tds_excess_kcal,
            s.dg_excess_kcal,
            s.ligand_overlap_f_disp,
            s.status.label(),
            s.ligand_dg_contribution_kcal,
            s.lining_residues.join(";")
        ));
    }
    out
}

/// Exports designed displaceable-water R-group analogs and efficiency metrics to `.csv`.
pub fn export_watermap_rgroup_designs_csv(rep: &WaterMapSolvationReport) -> String {
    let mut out = String::from(
        "rank,design_id,target_site,site_class,site_dg_kcal,anchor_atom,vector_dist_a,vector_align_deg,rgroup_name,strategy,reaction,added_hac,f_overlap,ddg_solv_kcal,delta_pact,pred_pact,ge_kcal_ha,delta_lle,mw,clogp,tpsa,sa_score,pains,design_score,smiles\n",
    );
    for d in &rep.rgroup_designs {
        out.push_str(&format!(
            "{},{},{},\"{}\",{:+.3},{},{:.2},{:.1},\"{}\",\"{}\",\"{}\",{},{:.2},{:+.3},{:+.3},{:.3},{:.3},{:+.3},{:.2},{:.2},{:.1},{:.2},{},{:.1},\"{}\"\n",
            d.rank,
            d.design_id,
            d.target_site_id,
            d.target_site_class.short_label(),
            d.target_site_dg_kcal,
            d.anchor_atom_label,
            d.vector_distance_a,
            d.vector_alignment_deg,
            d.rgroup_name,
            d.strategy.short_label(),
            d.synthetic_reaction,
            d.added_heavy_atoms,
            d.predicted_site_overlap_f,
            d.ddg_solvation_gain_kcal,
            d.predicted_delta_pact,
            d.predicted_pact,
            d.group_efficiency_ge,
            d.delta_lle,
            d.mw,
            d.clogp,
            d.tpsa,
            d.sa_score,
            d.pains_count,
            d.design_score,
            d.designed_smiles
        ));
    }
    out
}

/// Exports dataset-wide `WaterMap` screening rows to `.csv`.
pub fn export_dataset_watermap_screen_csv(rows: &[DatasetWaterMapScreenRow]) -> String {
    let mut out = String::from(
        "rank,compound_id,compound_name,obs_pact,mw,clogp,displaced_hotspots,retained_bridges,clashing_bridges,dh_disp_gain_kcal,tds_disp_gain_kcal,net_watermap_dg_kcal,watermap_refined_dg_kcal,untapped_hotspot_dg_kcal,top_rgroup_suggestion,smiles\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{},{},\"{}\",{:.2},{:.2},{:.2},{},{},{},{:+.3},{:+.3},{:+.3},{:+.3},{:+.3},\"{}\",\"{}\"\n",
            r.rank,
            r.compound_id,
            r.compound_name,
            r.observed_pact,
            r.mw,
            r.clogp,
            r.displaced_hotspots,
            r.retained_bridges,
            r.clashing_bridges,
            r.dh_disp_gain_kcal,
            r.tds_disp_gain_kcal,
            r.net_watermap_dg_kcal,
            r.watermap_refined_dg_kcal,
            r.untapped_hotspot_dg_kcal,
            r.top_rgroup_suggestion,
            r.smiles
        ));
    }
    out
}

/// Formats an annotated `.pdb` containing pocket shell atoms, aligned ligand `HETATM LIG`, and `WaterMap` hydration sites (`HETATM WME/WMS/WMD/WMB/WMN` with `occupancy` & `B-factor = dG_site`).
pub fn format_watermap_annotated_pdb(rep: &WaterMapSolvationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "REMARK   1 SAR-ATLAS v3.5.0 WATERMAP / IFST HYDRATION HOTSPOT & LIGAND COMPLEX\nREMARK   2 TARGET: {} (PDB {}) | LEAD: {} ({})\nREMARK   3 NET WATERMAP dG_SOLV_DISP = {:+.3} KCAL/MOL | REFINED dG = {:+.3} KCAL/MOL\nREMARK   4 WMP HETATM B-FACTOR ENCODES EXCESS HYDRATION FREE ENERGY dG_SITE (KCAL/MOL)\n",
        rep.target_name,
        rep.pdb_id,
        rep.compound_name,
        rep.compound_id,
        rep.net_watermap_dg_kcal,
        rep.watermap_refined_dg_kcal
    ));
    let mut serial = 1_u32;
    for (pos, lbl, elem) in &rep.pocket_shell_display_atoms {
        let parts: Vec<&str> = lbl.split(':').collect();
        let res_part = parts.first().copied().unwrap_or("UNK1");
        let atom_nm = parts.get(1).copied().unwrap_or("CA");
        let res_nm = &res_part[..res_part.len().min(3)];
        let res_num: i32 = res_part[res_nm.len()..].parse().unwrap_or(1);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 20.00          {:>2}\n",
            serial,
            atom_nm,
            res_nm,
            res_num,
            pos[0],
            pos[1],
            pos[2],
            elem.symbol()
        ));
        serial += 1;
    }
    for (idx, &pos) in rep.aligned_ligand_coords3d.iter().enumerate() {
        let sym = rep
            .mol
            .atoms
            .get(idx)
            .map(|a| a.elem.symbol())
            .unwrap_or("C");
        let aname = format!("{}{}", sym, (idx + 1) % 100);
        out.push_str(&format!(
            "HETATM{:>5} {:<4} LIG L   1    {:>8.3}{:>8.3}{:>8.3}  1.00 15.00          {:>2}\n",
            serial, aname, pos[0], pos[1], pos[2], sym
        ));
        serial += 1;
    }
    for (idx, s) in rep.hydration_sites.iter().enumerate() {
        let res_tag = match s.class {
            HydrationSiteClass::EnthalpyFrustratedHotspot => "WME",
            HydrationSiteClass::EntropyOrderedEnclosure => "WMS",
            HydrationSiteClass::DewettedVacuumCavity => "WMD",
            HydrationSiteClass::ConservedStructuralBridge => "WMB",
            HydrationSiteClass::BulkLikeNeutral => "WMN",
        };
        out.push_str(&format!(
            "HETATM{:>5}  O   {:>3} W{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}           O\n",
            serial,
            res_tag,
            idx + 1,
            s.pos[0],
            s.pos[1],
            s.pos[2],
            s.occupancy.clamp(0.01, 1.00),
            s.dg_excess_kcal.clamp(-9.99, 99.99)
        ));
        serial += 1;
    }
    out.push_str("END\n");
    out
}
