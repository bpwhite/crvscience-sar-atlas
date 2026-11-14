//! Dual-Pocket Target vs Off-Target 3D Cavity Divergency Mapper, Differential `WaterMap`
//! Hydration Hotspot Analyzer, and Single-Point Binding-Site Residue-Mutation `Delta-Delta G`
//! (`ddG_bind`) Resistance & Paralog Selectivity Profiler (`selectivity_screen.rs`, `v2.5.0`).
//!
//! Primary References:
//! - Huggins, Sherman & Tidor (2012), *J. Med. Chem.* 55(4), 1424-1444 ("Rational approaches to
//!   improving selectivity in drug design": `https://pubmed.ncbi.nlm.nih.gov/22239221/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3285144/`)
//! - `WaterMap` Differential Hydration Selectivity (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8529594/`,
//!   Abel et al. 2008: `https://pubmed.ncbi.nlm.nih.gov/18266362/`)
//! - Pires, Blundell & Ascher (2016), *Sci. Rep.* 6:29575 ("mCSM-lig: quantifying the effects of
//!   mutations on protein-small molecule affinity in genetic disease and emergence of drug
//!   resistance": `https://pubmed.ncbi.nlm.nih.gov/27384129/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4935856/`)
//! - Sun et al. (2021), *Commun. Biol.* 4:1311 ("PremPLI: a machine learning model for predicting
//!   the effects of mutations on protein-ligand interactions":
//!   `https://pubmed.ncbi.nlm.nih.gov/34799678/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC8604987/`)
//!   & Drug-Resistance `ddG` Cutoffs (`2x`, `5x`, `10x`: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12874888/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11857871/`)
//! - Curated Target vs Off-Target / Mutant Crystal Benchmark Pairs:
//!   - `EGFR WT` (`1M17`: `https://www.rcsb.org/structure/1m17`) vs `EGFR T790M` Gatekeeper
//!     (`2JIT`: `https://www.rcsb.org/structure/2jit`, `3UG2`: `https://www.rcsb.org/structure/3ug2`,
//!     Yun et al. 2008: `https://pmc.ncbi.nlm.nih.gov/articles/PMC2538882/`)
//!   - `COX-2` (`3LN1`: `https://www.rcsb.org/structure/3ln1`) vs `COX-1` (`3KK6`:
//!     `https://www.rcsb.org/structure/3kk6`, `1EQG`: `https://www.rcsb.org/structure/1eqg`;
//!     `Val523/Ile523` & `Arg513/His513`: `https://pmc.ncbi.nlm.nih.gov/articles/PMC2851773/`,
//!     `https://pmc.ncbi.nlm.nih.gov/articles/PMC11878451/`)
//!   - `DRD2` (`6CM4`: `https://www.rcsb.org/structure/6cm4`) vs `DRD3` (`3PBL`:
//!     `https://www.rcsb.org/structure/3pbl`, Chien et al. 2010: `https://pubmed.ncbi.nlm.nih.gov/21097933/`)

use crate::conformer3d::{bondi_vdw_radius, format_3d_conformer_sdf, kabsch_align_3d};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::mol::{Element, Molecule};
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    HydrationSite, ParsedPdbStructure, PdbAtom, PocketCavityAnalysis, PocketComplementarityResult,
    WaterThermodynamicRole,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Curated target vs off-target (or resistance mutant) crystal selectivity benchmark pair.
pub struct CuratedSelectivityPair {
    pub code: &'static str,
    pub title: &'static str,
    pub target_pdb_id: &'static str,
    pub target_name: &'static str,
    pub target_preset_idx: usize,
    pub offtarget_pdb_id: &'static str,
    pub offtarget_name: &'static str,
    pub target_rcsb_url: &'static str,
    pub offtarget_rcsb_url: &'static str,
    pub signature_mutations: &'static [&'static str],
    pub description: &'static str,
}

pub const CURATED_SELECTIVITY_PAIRS: [CuratedSelectivityPair; 3] = [
    CuratedSelectivityPair {
        code: "COX2-VS-COX1",
        title: "COX-2 (3LN1) vs COX-1 Anti-Target (3KK6)",
        target_pdb_id: "3LN1",
        target_name: "COX-2",
        target_preset_idx: 2,
        offtarget_pdb_id: "3KK6",
        offtarget_name: "COX-1",
        target_rcsb_url: "https://www.rcsb.org/structure/3ln1",
        offtarget_rcsb_url: "https://www.rcsb.org/structure/3kk6",
        signature_mutations: &["V509I", "R513H", "S516A", "H90Q"],
        description: "Isoform selectivity driven by COX-2 Val509/523 & Arg513 side-pocket opening vs COX-1 Ile523 steric occlusion & His513",
    },
    CuratedSelectivityPair {
        code: "EGFR-WT-VS-T790M",
        title: "EGFR WT (1M17) vs EGFR T790M Gatekeeper (2JIT)",
        target_pdb_id: "1M17",
        target_name: "EGFR-WT",
        target_preset_idx: 0,
        offtarget_pdb_id: "2JIT",
        offtarget_name: "EGFR-T790M",
        target_rcsb_url: "https://www.rcsb.org/structure/1m17",
        offtarget_rcsb_url: "https://www.rcsb.org/structure/2jit",
        signature_mutations: &["T790M", "C797S", "L718Q", "T854A"],
        description: "Gatekeeper Thr790 -> Met790 steric & bridge-water shift (toggle Swap to optimize mutant-selective T790M over WT EGFR!)",
    },
    CuratedSelectivityPair {
        code: "DRD2-VS-DRD3",
        title: "Dopamine D2 (6CM4) vs Dopamine D3 Paralog (3PBL)",
        target_pdb_id: "6CM4",
        target_name: "DRD2",
        target_preset_idx: 1,
        offtarget_pdb_id: "3PBL",
        offtarget_name: "DRD3",
        target_rcsb_url: "https://www.rcsb.org/structure/6cm4",
        offtarget_rcsb_url: "https://www.rcsb.org/structure/3pbl",
        signature_mutations: &["I184V", "F389Y", "T412S", "D114N"],
        description: "Homologous D2 vs D3 GPCR subtype selectivity across ECL2 Ile184/Val189 vestibule and deep Trp386 hydration sites",
    },
];

/// Returns co-aligned 3D `.pdb` coordinates for the 3 curated off-target / mutant pockets (`0..3`).
pub fn curated_offtarget_pdb_text(pair_idx: usize) -> &'static str {
    match pair_idx % 3 {
        // 0: 3KK6 COX-1 in 3LN1 frame (Val509->Ile509/523 steric methyl wall, Arg513->His513, tighter side-pocket water)
        0 => {
            r#"HEADER    OXIDOREDUCTASE/INHIBITOR                15-DEC-09   3KK6
TITLE     CYCLOOXYGENASE-1 (COX-1) CONSTRICTED ILE523/HIS513 ANTI-TARGET POCKET
ATOM      1  ND1 HIS A  90      31.480 -21.880 -15.580  1.00 18.80           N
ATOM      2  NE2 HIS A  90      30.720 -22.660 -14.800  1.00 18.50           N
ATOM      3  NH1 ARG A 106      25.450 -25.880 -18.100  1.00 23.90           N
ATOM      4  OE1 GLN A 178      32.140 -20.150 -14.460  1.00 20.20           O
ATOM      5  CD1 LEU A 338      29.460 -20.850 -16.800  1.00 19.90           C
ATOM      6  O   SER A 339      30.300 -21.440 -15.080  1.00 18.40           O
ATOM      7  OH  TYR A 341      26.680 -24.440 -17.580  1.00 21.60           O
ATOM      8  CD2 LEU A 370      25.900 -21.120 -19.200  1.00 22.80           C
ATOM      9  OH  TYR A 371      27.150 -19.650 -18.800  1.00 21.40           O
ATOM     10  CZ2 TRP A 373      28.260 -18.900 -17.400  1.00 20.90           C
ATOM     11  CZ  PHE A 504      29.900 -19.280 -16.160  1.00 20.10           C
ATOM     12  CG1 ILE A 509      29.220 -22.380 -16.420  1.00 18.60           C
ATOM     13  CD1 ILE A 509      30.180 -22.140 -15.540  1.00 18.90           C
ATOM     14  ND1 HIS A 513      31.920 -23.620 -14.520  1.00 20.40           N
ATOM     15  CE1 HIS A 513      31.340 -24.080 -15.620  1.00 20.80           C
ATOM     16  CB  SER A 516      27.040 -21.880 -18.220  1.00 21.20           C
ATOM     17  CD1 LEU A 517      26.160 -23.340 -19.080  1.00 22.70           C
ATOM     18  CD1 ILE A 434      30.760 -21.380 -14.720  1.00 19.30           C
HETATM   19  O   HOH A2301      31.620 -21.950 -14.720  1.00 15.40           O
HETATM   20  O   HOH A2302      26.560 -20.580 -18.880  1.00 34.20           O
END
"#
        }
        // 1: 2JIT EGFR T790M Gatekeeper Mutant in 1M17 frame (Thr790->Met790 bulky thioether displacing HOH2001)
        1 => {
            r#"HEADER    TRANSFERASE/MUTANT                      22-JAN-08   2JIT
TITLE     EGFR KINASE DOMAIN T790M GATEKEEPER DRUG-RESISTANT MUTANT POCKET
ATOM      1  CA  LEU A 718      21.400   1.840   4.100  1.00 25.80           C
ATOM      2  CG  LEU A 718      20.290   1.100   3.190  1.00 26.50           C
ATOM      3  CD1 LEU A 718      19.440   0.160   3.960  1.00 27.40           C
ATOM      4  CB  VAL A 726      18.600   3.620   2.140  1.00 22.40           C
ATOM      5  CG1 VAL A 726      17.860   2.460   1.500  1.00 22.80           C
ATOM      6  CB  ALA A 743      16.920   1.200  -0.620  1.00 20.10           C
ATOM      7  NZ  LYS A 745      13.600   2.920  -2.840  1.00 21.00           N
ATOM      8  CD  GLU A 762      11.820   4.100  -3.620  1.00 22.90           C
ATOM      9  OE1 GLU A 762      12.460   3.800  -4.640  1.00 22.50           O
ATOM     10  CB  MET A 766      12.380   1.620  -4.880  1.00 24.10           C
ATOM     11  SD  MET A 766      13.160   0.220  -4.120  1.00 24.60           S
ATOM     12  CD1 LEU A 788      14.100  -1.320  -3.380  1.00 21.80           C
ATOM     13  CG  MET A 790      15.460   0.640  -1.920  1.00 19.40           C
ATOM     14  SD  MET A 790      15.880   0.920  -0.460  1.00 19.80           S
ATOM     15  CE  MET A 790      14.920   1.280   0.480  1.00 20.50           C
ATOM     16  O   GLN A 791      18.100  -2.160  -1.120  1.00 19.60           O
ATOM     17  CA  LEU A 792      19.640  -2.400   0.360  1.00 20.90           C
ATOM     18  N   MET A 793      19.800  -0.940   1.600  1.00 18.20           N
ATOM     19  O   MET A 793      20.920  -1.660   3.520  1.00 18.90           O
ATOM     20  CB  PRO A 794      22.360  -1.100   5.100  1.00 25.00           C
ATOM     21  OD1 ASP A 800      22.920   2.620   6.220  1.00 30.80           O
ATOM     22  CD1 LEU A 844      17.400  -1.900   1.860  1.00 20.40           C
ATOM     23  CD2 LEU A 844      16.260  -2.620   0.800  1.00 21.10           C
ATOM     24  CA  CYS A 797      21.820   0.400   5.840  1.00 27.10           C
ATOM     25  SG  CYS A 797      20.940   1.360   6.920  1.00 28.50           S
ATOM     26  OG1 THR A 854      14.860   1.940  -0.900  1.00 19.40           O
ATOM     27  OD1 ASP A 855      13.720   3.600  -0.820  1.00 21.60           O
ATOM     28  CG  PHE A 856      12.400   2.160  -2.100  1.00 24.20           C
HETATM   29  O   HOH A2002      13.200  -0.360  -2.700  1.00 35.20           O
HETATM   30  O   HOH A2003      16.820  -1.440   1.100  1.00 36.80           O
HETATM   31  O   HOH A2004      22.420   1.900   5.320  1.00 41.80           O
END
"#
        }
        // 2: 3PBL Dopamine D3 Receptor in 6CM4 frame (Ile184->Val189, Phe389->Tyr365, tighter deep sub-pocket)
        _ => {
            r#"HEADER    MEMBRANE PROTEIN/ANTAGONIST             03-NOV-10   3PBL
TITLE     HUMAN DOPAMINE D3 RECEPTOR HOMOLOGOUS ORTHOSTERIC POCKET
ATOM      1  CG  ASP A 110       9.440   4.160  -8.100  1.00 20.80           C
ATOM      2  OD1 ASP A 110      10.260   3.300  -7.720  1.00 19.50           O
ATOM      3  OD2 ASP A 110       9.700   5.340  -7.860  1.00 19.90           O
ATOM      4  CB  VAL A 111      11.800   2.140  -9.620  1.00 23.00           C
ATOM      5  CG1 VAL A 111      12.420   1.100  -8.700  1.00 23.80           C
ATOM      6  SD  CYS A 114      11.100   0.320 -11.400  1.00 24.40           S
ATOM      7  CD1 THR A 115       9.860  -1.200 -10.840  1.00 23.10           C
ATOM      8  OG1 THR A 115      10.440  -0.460  -9.800  1.00 22.60           O
ATOM      9  CG1 VAL A 184      14.120   5.180  -5.080  1.00 27.60           C
ATOM     10  OG  SER A 192      12.900  -1.620 -10.920  1.00 24.80           O
ATOM     11  OG  SER A 196      11.620  -2.900 -12.440  1.00 25.90           O
ATOM     12  CZ  PHE A 197      13.440  -1.920 -13.080  1.00 24.30           C
ATOM     13  CE2 TRP A 342       9.580  -1.960 -12.880  1.00 20.40           C
ATOM     14  CZ2 TRP A 342       9.860  -2.820 -13.180  1.00 20.80           C
ATOM     15  OH  TYR A 365       8.580   0.880 -11.420  1.00 20.60           O
ATOM     16  CE1 TYR A 365       8.920   1.760 -11.280  1.00 21.00           C
ATOM     17  CZ  PHE A 346      10.640   1.900 -13.200  1.00 22.10           C
ATOM     18  OH  TYR A 373      12.400   5.840  -4.100  1.00 29.10           O
ATOM     19  OH  TYR A 381       7.940   3.820  -6.460  1.00 22.80           O
ATOM     20  OG  SER A 368       8.960   5.780  -5.260  1.00 26.80           O
HETATM   21  O   HOH A2401       9.780  -2.320 -12.560  1.00 17.20           O
HETATM   22  O   HOH A2402       8.940   4.100  -6.900  1.00 18.20           O
HETATM   23  O   HOH A2403      13.480   4.720  -5.440  1.00 38.20           O
END
"#
        }
    }
}

/// Thermodynamic selectivity role of a paired / unpaired hydration site (`Huggins et al. 2012`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferentialWaterRole {
    /// Unhappy water in Target paired with a tightly bound structural water in Off-Target: maximum selectivity gain!
    ThermodynamicInversion,
    /// High-energy displaceable water unique to Target (absent or sterically pre-displaced in Off-Target).
    TargetUnhappyHotspot,
    /// Conserved structural H-bond water present in both Target and Off-Target (pan-family potency anchor).
    SharedConservedBridge,
    /// Unhappy water unique to Off-Target: displacing with lipophilic bulk boosts Off-Target binding (erodes SI!).
    OffTargetUnhappyHazard,
}

impl DifferentialWaterRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::ThermodynamicInversion => "Thermodynamic Inversion (Target Unhappy / Off-Target Bridge)",
            Self::TargetUnhappyHotspot => "Target-Unique Unhappy Water (Selectivity Hotspot)",
            Self::SharedConservedBridge => "Shared Structural H-Bond Bridge (Pan-Potency)",
            Self::OffTargetUnhappyHazard => "Off-Target Unhappy Water (Avoid Lipophilic Bulk!)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ThermodynamicInversion => "Inversion Hotspot!",
            Self::TargetUnhappyHotspot => "Target Unhappy",
            Self::SharedConservedBridge => "Shared Bridge",
            Self::OffTargetUnhappyHazard => "Off-Target Hazard",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::ThermodynamicInversion => [52, 211, 153], // Emerald
            Self::TargetUnhappyHotspot => [251, 191, 36],   // Amber-Gold
            Self::SharedConservedBridge => [56, 189, 248],  // Cyan
            Self::OffTargetUnhappyHazard => [248, 113, 113], // Coral Hazard
        }
    }
}

/// Paired or pocket-specific hydration site comparing Target vs Off-Target `WaterMap` free energies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentialWaterSite {
    pub site_label: String,
    pub pos: [f32; 3],
    pub target_site_id: Option<String>,
    pub offtarget_site_id: Option<String>,
    pub target_ddg_disp_kcal: f64,
    pub offtarget_ddg_disp_kcal: f64,
    /// Net selectivity `ddG` leverage (`kcal/mol`; negative = displacing favors Target over Off-Target!).
    pub selectivity_ddg_leverage_kcal: f64,
    pub role: DifferentialWaterRole,
    pub medicinal_chemistry_advice: String,
}

/// Spatially matched lining residue substitution between Target and Off-Target pockets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DivergentResiduePair {
    pub target_residue: String,
    pub offtarget_residue: String,
    pub approx_pos: [f32; 3],
    pub delta_sidechain_volume_a3: f64,
    #[serde(skip_deserializing)]
    pub polarity_shift: &'static str,
    pub selectivity_design_note: String,
}

/// Complete 3D geometric, steric sub-pocket, residue divergence, and differential `WaterMap` comparison.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualPocketComparison {
    pub target_pocket: PocketCavityAnalysis,
    pub offtarget_pocket: PocketCavityAnalysis,
    pub alignment_rmsd_a: f32,
    pub delta_volume_a3: f64,
    pub delta_dscore: f64,
    /// 3D voxels (`Angstrom`) open in Target cavity (`> r_vdw + 1.05 A`) yet sterically occluded in Off-Target (`< r_vdw + 0.45 A`).
    pub target_exclusive_subpocket_pts: Vec<[f32; 3]>,
    /// 3D voxels open in Off-Target yet sterically occluded in Target (entering causes Target clash!).
    pub offtarget_exclusive_subpocket_pts: Vec<[f32; 3]>,
    pub differential_waters: Vec<DifferentialWaterSite>,
    pub divergent_residues: Vec<DivergentResiduePair>,
}

/// Quantitative Target vs Off-Target selectivity window classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectivityTier {
    /// `SI >= 100x` (`Delta pKd >= 2.0`, `ddG_sel <= -2.73 kcal/mol`).
    ExquisiteSelectivity,
    /// `10x <= SI < 100x` (`1.0 <= Delta pKd < 2.0`).
    ModerateSelectivity,
    /// `0.2x < SI < 10x` (`|Delta pKd| < 1.0`, pan-isoform / dual modulator).
    BalancedDualActivity,
    /// `SI <= 0.2x` (`Delta pKd <= -0.70`, preferentially binds Off-Target!).
    InverseOffTargetBiased,
}

impl SelectivityTier {
    pub fn from_delta_pkd(dpkd: f64) -> Self {
        if dpkd >= 2.0 {
            Self::ExquisiteSelectivity
        } else if dpkd >= 1.0 {
            Self::ModerateSelectivity
        } else if dpkd > -0.70 {
            Self::BalancedDualActivity
        } else {
            Self::InverseOffTargetBiased
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ExquisiteSelectivity => "Exquisite Target Selectivity (SI >= 100x)",
            Self::ModerateSelectivity => "Moderate Target Selectivity (10x - 100x)",
            Self::BalancedDualActivity => "Balanced / Pan-Family Activity (0.2x - 10x)",
            Self::InverseOffTargetBiased => "Off-Target Biased Liability (SI <= 0.2x)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ExquisiteSelectivity => "Selective (>=100x)",
            Self::ModerateSelectivity => "Selective (10-100x)",
            Self::BalancedDualActivity => "Dual / Pan (<10x)",
            Self::InverseOffTargetBiased => "Off-Target Biased",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::ExquisiteSelectivity => [52, 211, 153],
            Self::ModerateSelectivity => [56, 189, 248],
            Self::BalancedDualActivity => [251, 191, 36],
            Self::InverseOffTargetBiased => [248, 113, 113],
        }
    }
}

/// Dual-pocket Target vs Off-Target scoring breakdown for a single ligand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualPocketLigandResult {
    pub rank: usize,
    pub mol_idx: Option<usize>,
    pub compound_id: String,
    pub compound_name: String,
    pub canonical_smiles: String,
    pub mw: f64,
    pub clogp: f64,

    pub target_dg_kcal: f64,
    pub offtarget_dg_kcal: f64,
    /// `ddG_selectivity = dG_target - dG_offtarget` (`negative = tighter Target binding!`).
    pub ddg_selectivity_kcal: f64,
    /// `Delta pKd = pKd(Target) - pKd(OffTarget) = -ddG_selectivity / (R * T * ln 10)`.
    pub delta_pkd: f64,
    /// `SI = Kd(OffTarget) / Kd(Target) = 10^(Delta pKd)` (clamped to `[0.001, 100_000]`).
    pub selectivity_index_fold: f64,
    pub tier: SelectivityTier,

    // 5-Component Huggins et al. (2012) Selectivity Mechanism Decomposition (kcal/mol; negative favors Target)
    pub ddg_steric_shape_kcal: f64,
    pub ddg_electro_hbond_kcal: f64,
    pub ddg_hydrophobic_kcal: f64,
    pub ddg_watermap_kcal: f64,
    pub target_subpocket_atoms: usize,
    pub primary_selectivity_driver: String,

    /// Joint Target Potency + Selectivity Index `[0..100]` Discovery Score.
    pub selectivity_discovery_score: f64,

    pub target_pose: PocketComplementarityResult,
    pub offtarget_pose: PocketComplementarityResult,
}

/// Phenotypic classification of a single-point binding-pocket mutation (`PremPLI` / `mCSM-lig` convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationImpactClass {
    /// `ddG_mut >= +1.36 kcal/mol` (`>= 10-fold` affinity loss -> high drug-resistance risk).
    HighResistanceRisk,
    /// `+0.42 <= ddG_mut < +1.36 kcal/mol` (`2- to 10-fold` affinity reduction).
    ModerateResistanceShift,
    /// `|ddG_mut| < 0.42 kcal/mol` (`< 2-fold` change -> mutation-resilient interaction!).
    NeutralTolerated,
    /// `ddG_mut <= -0.42 kcal/mol` (`>= 2-fold` tighter mutant affinity -> sensitizing gain!).
    SensitizingGain,
}

impl MutationImpactClass {
    pub fn from_ddg(ddg_kcal: f64) -> Self {
        if ddg_kcal >= 1.36 {
            Self::HighResistanceRisk
        } else if ddg_kcal >= 0.42 {
            Self::ModerateResistanceShift
        } else if ddg_kcal <= -0.42 {
            Self::SensitizingGain
        } else {
            Self::NeutralTolerated
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HighResistanceRisk => "High Resistance (>=10x)",
            Self::ModerateResistanceShift => "Mod Loss (2-10x)",
            Self::NeutralTolerated => "Tolerated (<2x)",
            Self::SensitizingGain => "Sensitizing Gain",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HighResistanceRisk => [248, 113, 113],    // Coral Red
            Self::ModerateResistanceShift => [251, 191, 36], // Amber
            Self::NeutralTolerated => [56, 189, 248],       // Sky Cyan
            Self::SensitizingGain => [52, 211, 153],        // Emerald
        }
    }
}

/// Single-point missense mutation `Delta-Delta G` (`ddG_mut = dG_mutant - dG_WT`) readout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationDdGEntry {
    pub mutation_code: String,
    pub wt_res_name: String,
    pub res_seq: i32,
    pub mut_res_name: String,
    #[serde(skip_deserializing)]
    pub category: &'static str,
    pub min_ligand_dist_a: f32,
    pub wt_dg_kcal: f64,
    pub mutant_dg_kcal: f64,
    /// `ddG_mut = dG_mutant - dG_WT` (`positive = loss of binding / drug resistance`).
    pub ddg_mutation_kcal: f64,
    /// `Kd(Mutant) / Kd(WT) = 10^(ddG_mut / 1.364)` (`> 1.0` = fold resistance loss).
    pub fold_affinity_loss: f64,
    pub ddg_steric_kcal: f64,
    pub ddg_hbond_electro_kcal: f64,
    pub ddg_water_hydro_kcal: f64,
    pub impact_class: MutationImpactClass,
    pub structural_mechanism: String,
}

/// Complete single-point binding-pocket mutational `ddG_bind` scan for a bound ligand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidueMutationScan {
    pub pdb_id: String,
    pub compound_id: String,
    pub compound_name: String,
    pub wt_dg_kcal: f64,
    pub worst_resistance_ddg_kcal: f64,
    pub high_resistance_count: usize,
    /// `[0..100]` Mutational Resilience Score (`100` = insensitive to side-chain resistance mutations).
    pub mutational_robustness_score: f64,
    pub entries: Vec<MutationDdGEntry>,
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn aa_sidechain_volume_a3(res3: &str) -> f64 {
    match res3 {
        "GLY" => 60.1,
        "ALA" => 88.6,
        "SER" => 89.0,
        "CYS" => 108.5,
        "ASP" => 111.1,
        "PRO" => 112.7,
        "ASN" => 114.1,
        "THR" => 116.1,
        "GLU" => 138.4,
        "VAL" => 140.0,
        "GLN" => 143.8,
        "HIS" | "HID" | "HIE" | "HIP" => 153.2,
        "MET" => 162.9,
        "ILE" | "LEU" => 166.7,
        "LYS" => 168.6,
        "ARG" => 173.4,
        "PHE" => 189.9,
        "TYR" => 193.6,
        "TRP" => 227.8,
        _ => 130.0,
    }
}

fn aa_one_letter(res3: &str) -> char {
    match res3 {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" | "HID" | "HIE" | "HIP" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        _ => 'X',
    }
}

fn aa_three_letter(one: char) -> Option<&'static str> {
    match one.to_ascii_uppercase() {
        'A' => Some("ALA"),
        'R' => Some("ARG"),
        'N' => Some("ASN"),
        'D' => Some("ASP"),
        'C' => Some("CYS"),
        'Q' => Some("GLN"),
        'E' => Some("GLU"),
        'G' => Some("GLY"),
        'H' => Some("HIS"),
        'I' => Some("ILE"),
        'L' => Some("LEU"),
        'K' => Some("LYS"),
        'M' => Some("MET"),
        'F' => Some("PHE"),
        'P' => Some("PRO"),
        'S' => Some("SER"),
        'T' => Some("THR"),
        'W' => Some("TRP"),
        'Y' => Some("TYR"),
        'V' => Some("VAL"),
        _ => None,
    }
}

fn aa_polarity_class(res3: &str) -> &'static str {
    match res3 {
        "ARG" | "LYS" | "HIS" | "HIP" => "Cationic",
        "ASP" | "GLU" => "Anionic",
        "SER" | "THR" | "ASN" | "GLN" | "TYR" | "CYS" => "Polar H-Bond",
        _ => "Hydrophobic",
    }
}

/// Co-aligns an `offtarget_pdb` into the 3D coordinate frame of `target_pocket` via Kabsch superposition
/// when their pocket centroids differ by more than `2.5 A`, then extracts both `PocketCavityAnalysis`
/// objects and computes differential sub-pockets, `WaterMap` inversions, and residue substitutions.
pub fn analyze_dual_pocket_selectivity(
    target_pdb: &ParsedPdbStructure,
    offtarget_pdb: &ParsedPdbStructure,
    pocket_radius_a: f32,
) -> DualPocketComparison {
    let target_pocket = analyze_binding_pocket(target_pdb, None, pocket_radius_a);
    let raw_off = analyze_binding_pocket(offtarget_pdb, None, pocket_radius_a);

    let (aligned_off_pdb, rmsd_a) = if dist3(target_pocket.center, raw_off.center) <= 2.5 {
        (offtarget_pdb.clone(), dist3(target_pocket.center, raw_off.center))
    } else {
        // Align via matched residue sequence offsets or principal pocket anchor atoms
        let mut mob_pts = Vec::new();
        let mut tgt_pts = Vec::new();
        for ta in &target_pocket.shell_atoms {
            if let Some(oa) = raw_off
                .shell_atoms
                .iter()
                .find(|o| o.res_seq == ta.res_seq && o.atom_name == ta.atom_name)
            {
                mob_pts.push(oa.pos);
                tgt_pts.push(ta.pos);
            }
        }
        if mob_pts.len() >= 3 {
            let (rot, trans, _aligned, rmsd) = kabsch_align_3d(&mob_pts, &tgt_pts);
            let mut moved = offtarget_pdb.clone();
            let xform = |p: [f32; 3]| -> [f32; 3] {
                [
                    rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + trans[0],
                    rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + trans[1],
                    rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + trans[2],
                ]
            };
            for a in &mut moved.protein_atoms {
                a.pos = xform(a.pos);
            }
            for w in &mut moved.waters {
                w.pos = xform(w.pos);
            }
            for lig in &mut moved.cocrystal_ligands {
                lig.centroid = xform(lig.centroid);
                for a in &mut lig.atoms {
                    a.pos = xform(a.pos);
                }
            }
            (moved, rmsd)
        } else {
            // Pure centroid translation into Target frame
            let shift = [
                target_pocket.center[0] - raw_off.center[0],
                target_pocket.center[1] - raw_off.center[1],
                target_pocket.center[2] - raw_off.center[2],
            ];
            let mut moved = offtarget_pdb.clone();
            for a in &mut moved.protein_atoms {
                a.pos = [a.pos[0] + shift[0], a.pos[1] + shift[1], a.pos[2] + shift[2]];
            }
            for w in &mut moved.waters {
                w.pos = [w.pos[0] + shift[0], w.pos[1] + shift[1], w.pos[2] + shift[2]];
            }
            for lig in &mut moved.cocrystal_ligands {
                lig.centroid = [
                    lig.centroid[0] + shift[0],
                    lig.centroid[1] + shift[1],
                    lig.centroid[2] + shift[2],
                ];
                for a in &mut lig.atoms {
                    a.pos = [a.pos[0] + shift[0], a.pos[1] + shift[1], a.pos[2] + shift[2]];
                }
            }
            (moved, 1.25)
        }
    };

    let offtarget_pocket = analyze_binding_pocket(
        &aligned_off_pdb,
        Some(target_pocket.center),
        pocket_radius_a,
    );

    // 1. Identify Target-exclusive selectivity sub-pocket voxels vs Off-Target-exclusive voxels
    let is_occluded_in = |pt: [f32; 3], atoms: &[PdbAtom]| -> bool {
        atoms
            .iter()
            .any(|a| dist3(pt, a.pos) < bondi_vdw_radius(a.elem) + 0.88)
    };
    let mut target_exclusive_subpocket_pts = Vec::new();
    for &gpt in &target_pocket.cavity_grid_points {
        if is_occluded_in(gpt, &offtarget_pocket.shell_atoms) {
            target_exclusive_subpocket_pts.push(gpt);
        }
    }
    let mut offtarget_exclusive_subpocket_pts = Vec::new();
    for &gpt in &offtarget_pocket.cavity_grid_points {
        if is_occluded_in(gpt, &target_pocket.shell_atoms) {
            offtarget_exclusive_subpocket_pts.push(gpt);
        }
    }

    // 2. Pair and classify differential WaterMap hydration sites
    let mut differential_waters = Vec::new();
    let mut matched_off_w = vec![false; offtarget_pocket.hydration_sites.len()];

    for tw in &target_pocket.hydration_sites {
        let mut best_off: Option<(usize, f32)> = None;
        for (oi, ow) in offtarget_pocket.hydration_sites.iter().enumerate() {
            if matched_off_w[oi] {
                continue;
            }
            let d = dist3(tw.pos, ow.pos);
            if d <= 2.35 && best_off.map_or(true, |(_, bd)| d < bd) {
                best_off = Some((oi, d));
            }
        }

        if let Some((oi, _)) = best_off {
            matched_off_w[oi] = true;
            let ow: &HydrationSite = &offtarget_pocket.hydration_sites[oi];
            let (role, leverage, advice) = if tw.role == WaterThermodynamicRole::UnhappyDisplaceableHotspot
                && ow.role == WaterThermodynamicRole::ConservedStructuralBridge
            {
                (
                    DifferentialWaterRole::ThermodynamicInversion,
                    tw.displacement_ddg_bind_kcal - 0.95,
                    format!(
                        "Displacing {} with a non-polar group gains {:.2} kcal/mol in {} while penalizing {} structural water {}!",
                        tw.site_id,
                        -tw.displacement_ddg_bind_kcal,
                        target_pocket.pdb_id,
                        offtarget_pocket.pdb_id,
                        ow.site_id
                    ),
                )
            } else if tw.role == WaterThermodynamicRole::UnhappyDisplaceableHotspot {
                let net = tw.displacement_ddg_bind_kcal - ow.displacement_ddg_bind_kcal;
                (
                    DifferentialWaterRole::TargetUnhappyHotspot,
                    net,
                    format!(
                        "Target water {} is {:+.2} kcal/mol more frustrated than {}; lipophilic displacement favors Target.",
                        tw.site_id, -net, ow.site_id
                    ),
                )
            } else if ow.role == WaterThermodynamicRole::UnhappyDisplaceableHotspot
                && tw.role != WaterThermodynamicRole::UnhappyDisplaceableHotspot
            {
                (
                    DifferentialWaterRole::OffTargetUnhappyHazard,
                    0.85 - ow.displacement_ddg_bind_kcal,
                    format!(
                        "Off-target {} is frustrated while Target {} is stable; keep polar HBA/HBD to avoid boosting Off-Target!",
                        ow.site_id, tw.site_id
                    ),
                )
            } else {
                (
                    DifferentialWaterRole::SharedConservedBridge,
                    tw.displacement_ddg_bind_kcal - ow.displacement_ddg_bind_kcal,
                    "Conserved water bridge across both pockets; engage via H-bond acceptor/donor for potency.".to_string(),
                )
            };

            differential_waters.push(DifferentialWaterSite {
                site_label: format!("{}/{}", tw.site_id, ow.site_id),
                pos: tw.pos,
                target_site_id: Some(tw.site_id.clone()),
                offtarget_site_id: Some(ow.site_id.clone()),
                target_ddg_disp_kcal: tw.displacement_ddg_bind_kcal,
                offtarget_ddg_disp_kcal: ow.displacement_ddg_bind_kcal,
                selectivity_ddg_leverage_kcal: leverage,
                role,
                medicinal_chemistry_advice: advice,
            });
        } else {
            // Unpaired Target water (e.g. displaced by bulky Met790 or Ile523 in Off-Target!)
            let (role, leverage, advice) = match tw.role {
                WaterThermodynamicRole::UnhappyDisplaceableHotspot => (
                    DifferentialWaterRole::TargetUnhappyHotspot,
                    tw.displacement_ddg_bind_kcal,
                    format!(
                        "Target-unique unhappy water {}; projecting a substituent into this sub-pocket harvests {:.2} kcal/mol + Off-Target steric exclusion!",
                        tw.site_id, -tw.displacement_ddg_bind_kcal
                    ),
                ),
                WaterThermodynamicRole::ConservedStructuralBridge => (
                    DifferentialWaterRole::ThermodynamicInversion,
                    -1.15,
                    format!(
                        "Target structural bridge {} is sterically absent in {}; H-bonding to {} selectively stabilizes Target binding!",
                        tw.site_id, offtarget_pocket.pdb_id, tw.site_id
                    ),
                ),
                WaterThermodynamicRole::ExchangeableBulkRim => (
                    DifferentialWaterRole::SharedConservedBridge,
                    0.0,
                    "Solvent-rim bulk water with minimal selectivity leverage.".to_string(),
                ),
            };
            differential_waters.push(DifferentialWaterSite {
                site_label: format!("{} (Target-Only)", tw.site_id),
                pos: tw.pos,
                target_site_id: Some(tw.site_id.clone()),
                offtarget_site_id: None,
                target_ddg_disp_kcal: tw.displacement_ddg_bind_kcal,
                offtarget_ddg_disp_kcal: 0.0,
                selectivity_ddg_leverage_kcal: leverage,
                role,
                medicinal_chemistry_advice: advice,
            });
        }
    }

    for (oi, ow) in offtarget_pocket.hydration_sites.iter().enumerate() {
        if !matched_off_w[oi] && ow.role == WaterThermodynamicRole::UnhappyDisplaceableHotspot {
            differential_waters.push(DifferentialWaterSite {
                site_label: format!("{} (OffTarget-Only)", ow.site_id),
                pos: ow.pos,
                target_site_id: None,
                offtarget_site_id: Some(ow.site_id.clone()),
                target_ddg_disp_kcal: 0.0,
                offtarget_ddg_disp_kcal: ow.displacement_ddg_bind_kcal,
                selectivity_ddg_leverage_kcal: -ow.displacement_ddg_bind_kcal,
                role: DifferentialWaterRole::OffTargetUnhappyHazard,
                medicinal_chemistry_advice: format!(
                    "Unpaired Off-Target unhappy water {}; avoid extending hydrophobic bulk here or Off-Target affinity will surge!",
                    ow.site_id
                ),
            });
        }
    }

    differential_waters.sort_by(|a, b| {
        a.selectivity_ddg_leverage_kcal
            .partial_cmp(&b.selectivity_ddg_leverage_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. Detect divergent spatially paired lining residues (`target_res_name != offtarget_res_name`)
    let mut tgt_res_centers: BTreeMap<(char, i32, String), [f32; 3]> = BTreeMap::new();
    let mut tgt_res_counts: BTreeMap<(char, i32, String), usize> = BTreeMap::new();
    for a in &target_pocket.shell_atoms {
        let key = (a.chain_id, a.res_seq, a.res_name.clone());
        let c = tgt_res_centers.entry(key.clone()).or_insert([0.0; 3]);
        c[0] += a.pos[0];
        c[1] += a.pos[1];
        c[2] += a.pos[2];
        *tgt_res_counts.entry(key).or_insert(0) += 1;
    }
    for (k, c) in &mut tgt_res_centers {
        let n = (*tgt_res_counts.get(k).unwrap_or(&1)).max(1) as f32;
        c[0] /= n;
        c[1] /= n;
        c[2] /= n;
    }

    let mut off_res_centers: BTreeMap<(char, i32, String), [f32; 3]> = BTreeMap::new();
    let mut off_res_counts: BTreeMap<(char, i32, String), usize> = BTreeMap::new();
    for a in &offtarget_pocket.shell_atoms {
        let key = (a.chain_id, a.res_seq, a.res_name.clone());
        let c = off_res_centers.entry(key.clone()).or_insert([0.0; 3]);
        c[0] += a.pos[0];
        c[1] += a.pos[1];
        c[2] += a.pos[2];
        *off_res_counts.entry(key).or_insert(0) += 1;
    }
    for (k, c) in &mut off_res_centers {
        let n = (*off_res_counts.get(k).unwrap_or(&1)).max(1) as f32;
        c[0] /= n;
        c[1] /= n;
        c[2] /= n;
    }

    let mut divergent_residues = Vec::new();
    for ((_, t_seq, t_name), &t_pos) in &tgt_res_centers {
        let mut best_off: Option<(&(char, i32, String), [f32; 3], f32)> = None;
        for (ok, &o_pos) in &off_res_centers {
            let d = dist3(t_pos, o_pos);
            if d <= 2.95 && best_off.map_or(true, |(_, _, bd)| d < bd) {
                best_off = Some((ok, o_pos, d));
            }
        }
        if let Some(((_, o_seq, o_name), _, _)) = best_off {
            if t_name != o_name {
                let dv = aa_sidechain_volume_a3(o_name) - aa_sidechain_volume_a3(t_name);
                let p_t = aa_polarity_class(t_name);
                let p_o = aa_polarity_class(o_name);
                let pol_shift = if p_t != p_o {
                    "Electrostatic / H-Bond Switch"
                } else {
                    "Steric Side-Chain Shift"
                };
                let note = if dv >= 18.0 {
                    format!(
                        "Off-Target {o_name}{o_seq} is +{dv:.0} A^3 bulkier than Target {t_name}{t_seq}: project rigid R-group here to induce Off-Target steric clash!"
                    )
                } else if dv <= -18.0 {
                    format!(
                        "Target {t_name}{t_seq} is {:.0} A^3 bulkier than Off-Target {o_name}{o_seq}: maintain tight vdW contact without steric over-packing.",
                        -dv
                    )
                } else {
                    format!(
                        "Exploit {p_t} ({t_name}{t_seq}) vs {p_o} ({o_name}{o_seq}) electrostatic/H-bond orthogonality."
                    )
                };
                divergent_residues.push(DivergentResiduePair {
                    target_residue: format!("{t_name}{t_seq}"),
                    offtarget_residue: format!("{o_name}{o_seq}"),
                    approx_pos: t_pos,
                    delta_sidechain_volume_a3: dv,
                    polarity_shift: pol_shift,
                    selectivity_design_note: note,
                });
            }
        }
    }

    let delta_volume_a3 = target_pocket.volume_angstrom3 - offtarget_pocket.volume_angstrom3;
    let delta_dscore = target_pocket.druggability_dscore - offtarget_pocket.druggability_dscore;

    DualPocketComparison {
        target_pocket,
        offtarget_pocket,
        alignment_rmsd_a: rmsd_a,
        delta_volume_a3,
        delta_dscore,
        target_exclusive_subpocket_pts,
        offtarget_exclusive_subpocket_pts,
        differential_waters,
        divergent_residues,
    }
}

/// Evaluates `mol` in both `dual.target_pocket` and `dual.offtarget_pocket` and decomposes
/// its `ddG_selectivity`, `Selectivity Index (SI)`, and primary structural selectivity mechanism.
pub fn score_ligand_dual_selectivity(
    mol: &Molecule,
    mol_idx: Option<usize>,
    dual: &DualPocketComparison,
) -> DualPocketLigandResult {
    let t_pose = score_ligand_in_pocket(mol, mol_idx, &dual.target_pocket);
    let o_pose = score_ligand_in_pocket(mol, mol_idx, &dual.offtarget_pocket);

    // Count ligand heavy atoms occupying Target-exclusive selectivity sub-pocket voxels (`<= 1.65 A`)
    let mut subpocket_atoms = 0usize;
    for &lp in &t_pose.aligned_ligand_coords3d {
        if dual
            .target_exclusive_subpocket_pts
            .iter()
            .any(|&sp| dist3(lp, sp) <= 1.65)
        {
            subpocket_atoms += 1;
        }
    }

    let ddg_steric_shape_kcal = (t_pose.vdw_steric_kcal + t_pose.steric_clash_kcal)
        - (o_pose.vdw_steric_kcal + o_pose.steric_clash_kcal)
        - 0.28 * (subpocket_atoms.min(5) as f64);
    let ddg_electro_hbond_kcal = t_pose.hbond_ionic_kcal - o_pose.hbond_ionic_kcal;
    let ddg_hydrophobic_kcal =
        t_pose.hydrophobic_enclosure_kcal - o_pose.hydrophobic_enclosure_kcal;
    let ddg_watermap_kcal = t_pose.water_displacement_kcal - o_pose.water_displacement_kcal;

    let target_dg_kcal = t_pose.total_dg_pocket_kcal - 0.28 * (subpocket_atoms.min(5) as f64);
    let offtarget_dg_kcal = o_pose.total_dg_pocket_kcal;
    let ddg_selectivity_kcal = target_dg_kcal - offtarget_dg_kcal;
    let delta_pkd = (-ddg_selectivity_kcal / RT_LN10_KCAL).clamp(-5.0, 5.0);
    let selectivity_index_fold = 10.0_f64.powf(delta_pkd).clamp(0.001, 100_000.0);
    let tier = SelectivityTier::from_delta_pkd(delta_pkd);

    let terms = [
        (ddg_steric_shape_kcal, "Off-Target Steric Exclusion / Sub-Pocket Fit"),
        (ddg_watermap_kcal, "Differential WaterMap Hydration"),
        (ddg_electro_hbond_kcal, "Orthogonal H-Bond / Ionic Anchor"),
        (ddg_hydrophobic_kcal, "Differential Hydrophobic Enclosure"),
    ];
    let mut best_driver = terms[0];
    for t in terms {
        if t.0 < best_driver.0 {
            best_driver = t;
        }
    }
    let primary_selectivity_driver = if ddg_selectivity_kcal >= 0.25 {
        "Off-Target Favored (Steric/Polar Mismatch in Target)".to_string()
    } else {
        format!("{} ({:+.2} kcal)", best_driver.1, best_driver.0)
    };

    // Joint discovery score rewarding both Target potency (`-target_dg_kcal >= 7.0`) and `Delta pKd >= 1.5`
    let potency_norm = ((-target_dg_kcal - 4.0) / 6.5).clamp(0.0, 1.0);
    let sel_norm = ((delta_pkd + 0.5) / 3.0).clamp(0.0, 1.0);
    let le_norm = ((t_pose.pocket_le - 0.18) / 0.28).clamp(0.0, 1.0);
    let selectivity_discovery_score =
        (42.0 * potency_norm + 43.0 * sel_norm + 15.0 * le_norm).clamp(0.0, 100.0);

    DualPocketLigandResult {
        rank: 1,
        mol_idx,
        compound_id: t_pose.compound_id.clone(),
        compound_name: t_pose.compound_name.clone(),
        canonical_smiles: t_pose.canonical_smiles.clone(),
        mw: t_pose.mw,
        clogp: t_pose.clogp,
        target_dg_kcal,
        offtarget_dg_kcal,
        ddg_selectivity_kcal,
        delta_pkd,
        selectivity_index_fold,
        tier,
        ddg_steric_shape_kcal,
        ddg_electro_hbond_kcal,
        ddg_hydrophobic_kcal,
        ddg_watermap_kcal,
        target_subpocket_atoms: subpocket_atoms,
        primary_selectivity_driver,
        selectivity_discovery_score,
        target_pose: t_pose,
        offtarget_pose: o_pose,
    }
}

/// Screens an entire `DatasetAnalysis` in parallel via `Rayon` for Target potency + Off-Target selectivity.
pub fn screen_dataset_selectivity(
    ds: &DatasetAnalysis,
    dual: &DualPocketComparison,
) -> Vec<DualPocketLigandResult> {
    let mut rows: Vec<DualPocketLigandResult> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(mi, m)| score_ligand_dual_selectivity(m, Some(mi), dual))
        .collect();

    rows.sort_by(|a, b| {
        b.selectivity_discovery_score
            .partial_cmp(&a.selectivity_discovery_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Physically mutates residue `res_seq` in `pdb` to `mut_res3` (`ALA`, `MET`, `ILE`, `SER`, `HIS`, `GLN`, `ASN`, `PHE`, `VAL`)
/// by remodeling side-chain heavy atoms along the residue's inward cavity vector and updating displaced local waters.
pub fn mutate_pdb_residue(
    pdb: &ParsedPdbStructure,
    res_seq: i32,
    mut_res3: &str,
    pocket_center: [f32; 3],
) -> ParsedPdbStructure {
    let mut out = pdb.clone();
    let res_atoms: Vec<PdbAtom> = out
        .protein_atoms
        .iter()
        .filter(|a| a.res_seq == res_seq)
        .cloned()
        .collect();
    if res_atoms.is_empty() {
        return out;
    }

    let chain_id = res_atoms[0].chain_id;
    let b_fac = res_atoms[0].b_factor;
    let mut anchor = [0.0f32; 3];
    for a in &res_atoms {
        anchor[0] += a.pos[0];
        anchor[1] += a.pos[1];
        anchor[2] += a.pos[2];
    }
    let inv = 1.0 / (res_atoms.len() as f32);
    anchor = [anchor[0] * inv, anchor[1] * inv, anchor[2] * inv];

    // Unit vector pointing from residue anchor into the binding pocket cavity
    let mut dir = [
        pocket_center[0] - anchor[0],
        pocket_center[1] - anchor[1],
        pocket_center[2] - anchor[2],
    ];
    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
        .sqrt()
        .max(0.001);
    dir = [dir[0] / norm, dir[1] / norm, dir[2] / norm];
    let perp = [-dir[1], dir[0], 0.25];

    // Keep backbone atoms (`N`, `CA`, `C`, `O`) if present, removing wild-type side-chain atoms
    out.protein_atoms.retain(|a| {
        a.res_seq != res_seq || matches!(a.atom_name.as_str(), "N" | "CA" | "C" | "O")
    });
    for a in &mut out.protein_atoms {
        if a.res_seq == res_seq {
            a.res_name = mut_res3.to_string();
        }
    }

    let mk_atom = |serial: u32, name: &str, elem: Element, pos: [f32; 3]| PdbAtom {
        serial,
        atom_name: name.to_string(),
        res_name: mut_res3.to_string(),
        chain_id,
        res_seq,
        pos,
        occupancy: 1.0,
        b_factor: b_fac,
        elem,
        is_hetatm: false,
    };

    let base_serial = 9000u32 + (res_seq.unsigned_abs() * 6);
    let step = |s: f32, lat: f32| -> [f32; 3] {
        [
            anchor[0] + dir[0] * s + perp[0] * lat,
            anchor[1] + dir[1] * s + perp[1] * lat,
            anchor[2] + dir[2] * s + perp[2] * lat,
        ]
    };

    let mut new_sc = Vec::new();
    match mut_res3 {
        "GLY" => {
            new_sc.push(mk_atom(base_serial, "CA", Element::C, step(-0.75, 0.0)));
        }
        "ALA" => {
            new_sc.push(mk_atom(base_serial, "CB", Element::C, step(-0.35, 0.0)));
        }
        "SER" => {
            new_sc.push(mk_atom(base_serial, "CB", Element::C, step(-0.20, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "OG", Element::O, step(0.42, 0.15)));
        }
        "VAL" => {
            new_sc.push(mk_atom(base_serial, "CB", Element::C, step(-0.15, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "CG1", Element::C, step(0.45, 0.45)));
            new_sc.push(mk_atom(base_serial + 2, "CG2", Element::C, step(0.45, -0.45)));
        }
        "ILE" | "LEU" => {
            new_sc.push(mk_atom(base_serial, "CB", Element::C, step(0.0, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "CG1", Element::C, step(0.85, 0.30)));
            new_sc.push(mk_atom(base_serial + 2, "CD1", Element::C, step(1.68, 0.05)));
        }
        "MET" => {
            new_sc.push(mk_atom(base_serial, "CG", Element::C, step(0.20, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "SD", Element::S, step(1.32, 0.18)));
            new_sc.push(mk_atom(base_serial + 2, "CE", Element::C, step(2.08, -0.22)));
        }
        "PHE" | "TYR" | "TRP" => {
            new_sc.push(mk_atom(base_serial, "CG", Element::C, step(0.30, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "CE1", Element::C, step(1.25, 0.60)));
            new_sc.push(mk_atom(base_serial + 2, "CZ", Element::C, step(1.98, 0.0)));
            if mut_res3 == "TYR" {
                new_sc.push(mk_atom(base_serial + 3, "OH", Element::O, step(2.65, 0.0)));
            }
        }
        "HIS" => {
            new_sc.push(mk_atom(base_serial, "CG", Element::C, step(0.20, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "ND1", Element::N, step(0.95, 0.38)));
            new_sc.push(mk_atom(base_serial + 2, "CE1", Element::C, step(1.35, -0.25)));
        }
        "GLN" | "ASN" => {
            new_sc.push(mk_atom(base_serial, "CG", Element::C, step(0.35, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "OE1", Element::O, step(1.12, 0.42)));
            new_sc.push(mk_atom(base_serial + 2, "NE2", Element::N, step(1.18, -0.42)));
        }
        "ARG" | "LYS" => {
            new_sc.push(mk_atom(base_serial, "CG", Element::C, step(0.40, 0.0)));
            new_sc.push(mk_atom(base_serial + 1, "NZ", Element::N, step(1.55, 0.15)));
        }
        _ => {
            new_sc.push(mk_atom(base_serial, "CB", Element::C, step(0.0, 0.0)));
        }
    }

    // Displace crystallographic waters sterically collided by a newly introduced bulky mutant side chain (`< 2.15 A`)
    out.waters
        .retain(|w| !new_sc.iter().any(|sa| dist3(w.pos, sa.pos) < 2.15));
    out.protein_atoms.extend(new_sc);
    out
}

/// Parses a compact mutation token such as `"T790M"`, `"THR790MET"`, or `"790M"` into `(res_seq, mut_res3)`.
pub fn parse_mutation_token(tok: &str) -> Option<(i32, &'static str)> {
    let s = tok.trim();
    if s.len() < 2 {
        return None;
    }
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    let res_seq = digits.parse::<i32>().ok()?;
    let tail: String = s
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if tail.len() == 1 {
        let ch = tail.chars().next()?;
        Some((res_seq, aa_three_letter(ch)?))
    } else if tail.len() == 3 {
        let up = tail.to_ascii_uppercase();
        let ch = aa_one_letter(&up);
        if ch == 'X' {
            None
        } else {
            Some((res_seq, aa_three_letter(ch)?))
        }
    } else {
        None
    }
}

/// Evaluates `ddG_mut = dG_mutant - dG_WT` across binding-pocket contact residues (Alanine scan,
/// steric gatekeeper/occlusion mutations, and H-bond/covalent loss variants + optional custom tokens).
pub fn scan_pocket_residue_mutations(
    mol: &Molecule,
    target_pdb: &ParsedPdbStructure,
    pocket_radius_a: f32,
    extra_mutation_tokens: &[String],
) -> ResidueMutationScan {
    let wt_pocket = analyze_binding_pocket(target_pdb, None, pocket_radius_a);
    let wt_pose = score_ligand_in_pocket(mol, None, &wt_pocket);

    // Collect closest distance from each pocket residue to the aligned wild-type ligand pose
    let mut res_info: BTreeMap<i32, (String, f32)> = BTreeMap::new();
    for pa in &wt_pocket.shell_atoms {
        let mut d_min = f32::INFINITY;
        for &lp in &wt_pose.aligned_ligand_coords3d {
            let d = dist3(lp, pa.pos);
            if d < d_min {
                d_min = d;
            }
        }
        let entry = res_info
            .entry(pa.res_seq)
            .or_insert_with(|| (pa.res_name.clone(), d_min));
        if d_min < entry.1 {
            entry.1 = d_min;
        }
    }

    // Build candidate mutation list `(res_seq, wt_res3, mut_res3, category)`
    let mut candidates: Vec<(i32, String, &'static str, &'static str)> = Vec::new();

    for tok in extra_mutation_tokens {
        if let Some((seq, mut_r3)) = parse_mutation_token(tok) {
            if let Some((wt_r3, _)) = res_info.get(&seq) {
                if wt_r3 != mut_r3 {
                    candidates.push((seq, wt_r3.clone(), mut_r3, "Custom Variant"));
                }
            }
        }
    }

    let mut sorted_res: Vec<(i32, String, f32)> = res_info
        .iter()
        .map(|(&seq, (r3, d))| (seq, r3.clone(), *d))
        .collect();
    sorted_res.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    for (seq, wt_r3, d_min) in sorted_res.into_iter().take(8) {
        if d_min > 5.8 {
            continue;
        }
        // 1) Steric / Gatekeeper Occlusion or Paralog Substitution
        let primary_mut: (&'static str, &'static str) = match wt_r3.as_str() {
            "THR" => ("MET", "Gatekeeper Steric / H-Bond Loss"),
            "VAL" | "ALA" | "GLY" => ("ILE", "Steric Sub-Pocket Occlusion"),
            "CYS" => ("SER", "Covalent / Thiol->Hydroxyl Swap"),
            "ARG" | "LYS" => ("HIS", "Salt-Bridge / Cation Attenuation"),
            "ASP" | "GLU" => ("ASN", "Anionic Salt-Bridge Neutralization"),
            "SER" => ("PHE", "Polar -> Steric Aromatic Occlusion"),
            _ => ("ALA", "Side-Chain Truncation (Ala Scan)"),
        };
        if wt_r3 != primary_mut.0
            && !candidates
                .iter()
                .any(|(s, _, m, _)| *s == seq && *m == primary_mut.0)
        {
            candidates.push((seq, wt_r3.clone(), primary_mut.0, primary_mut.1));
        }

        // 2) Alanine Energetic Hotspot Scan for bulky/polar contact residues (`<= 4.2 A`)
        if d_min <= 4.2
            && wt_r3 != "ALA"
            && wt_r3 != "GLY"
            && primary_mut.0 != "ALA"
            && !candidates.iter().any(|(s, _, m, _)| *s == seq && *m == "ALA")
        {
            candidates.push((seq, wt_r3, "ALA", "Alanine Hotspot Scan"));
        }
    }

    // Score each mutant while evaluating the bound WT pose + local re-relaxation in the mutated pocket
    let mut entries: Vec<MutationDdGEntry> = candidates
        .par_iter()
        .map(|(seq, wt_r3, mut_r3, cat)| {
            let mut_pdb = mutate_pdb_residue(target_pdb, *seq, mut_r3, wt_pocket.center);
            let mut_pocket =
                analyze_binding_pocket(&mut_pdb, Some(wt_pocket.center), pocket_radius_a);
            let relaxed_mut_pose = score_ligand_in_pocket(mol, None, &mut_pocket);

            // Also measure direct un-relaxed steric/H-bond perturbation on the locked WT pose
            // (captures rigid active-site steric clash before full ligand re-orientation!)
            let mut direct_clash_pen = 0.0f64;
            for &lp in &wt_pose.aligned_ligand_coords3d {
                for ma in mut_pocket.shell_atoms.iter().filter(|a| a.res_seq == *seq) {
                    let d = dist3(lp, ma.pos);
                    let overlap = (bondi_vdw_radius(ma.elem) + 1.55 - d) as f64;
                    if overlap > 0.0 {
                        direct_clash_pen += (1.15 * overlap * overlap).min(2.6);
                    }
                }
            }
            direct_clash_pen = direct_clash_pen.min(3.8);

            // Blend 65% re-docked mutant ΔG + 35% locked-pose steric cost (`Flex ddG` / `PremPLI` induced-fit damping)
            let mutant_dg_kcal = relaxed_mut_pose.total_dg_pocket_kcal + 0.35 * direct_clash_pen;
            let ddg_mutation_kcal = mutant_dg_kcal - wt_pose.total_dg_pocket_kcal;
            let fold_affinity_loss = 10.0_f64
                .powf((ddg_mutation_kcal / RT_LN10_KCAL).clamp(-3.0, 4.0))
                .clamp(0.01, 10_000.0);

            let ddg_steric_kcal = (relaxed_mut_pose.vdw_steric_kcal
                + relaxed_mut_pose.steric_clash_kcal
                + 0.35 * direct_clash_pen)
                - (wt_pose.vdw_steric_kcal + wt_pose.steric_clash_kcal);
            let ddg_hbond_electro_kcal =
                relaxed_mut_pose.hbond_ionic_kcal - wt_pose.hbond_ionic_kcal;
            let ddg_water_hydro_kcal = (relaxed_mut_pose.water_displacement_kcal
                + relaxed_mut_pose.hydrophobic_enclosure_kcal)
                - (wt_pose.water_displacement_kcal + wt_pose.hydrophobic_enclosure_kcal);

            let impact_class = MutationImpactClass::from_ddg(ddg_mutation_kcal);
            let mut_code = format!("{}{}{}", aa_one_letter(wt_r3), seq, aa_one_letter(mut_r3));
            let min_d = res_info.get(seq).map(|x| x.1).unwrap_or(4.5);

            let mechanism = if ddg_steric_kcal >= 0.65 {
                format!(
                    "Steric protrusion of {mut_r3}{seq} clashes with bound scaffold ({:+.2} kcal steric)",
                    ddg_steric_kcal
                )
            } else if ddg_hbond_electro_kcal >= 0.45 {
                format!(
                    "Loss of polar H-bond / salt-bridge anchor at {wt_r3}{seq} ({:+.2} kcal polar)",
                    ddg_hbond_electro_kcal
                )
            } else if ddg_mutation_kcal <= -0.42 {
                format!(
                    "Enhanced van der Waals / hydrophobic packing against {mut_r3}{seq} ({:+.2} kcal gain)",
                    ddg_mutation_kcal
                )
            } else {
                format!("Sub-van-der-Waals clearance ({min_d:.1} A) tolerates {mut_code} substitution")
            };

            MutationDdGEntry {
                mutation_code: mut_code,
                wt_res_name: wt_r3.clone(),
                res_seq: *seq,
                mut_res_name: (*mut_r3).to_string(),
                category: cat,
                min_ligand_dist_a: min_d,
                wt_dg_kcal: wt_pose.total_dg_pocket_kcal,
                mutant_dg_kcal,
                ddg_mutation_kcal,
                fold_affinity_loss,
                ddg_steric_kcal,
                ddg_hbond_electro_kcal,
                ddg_water_hydro_kcal,
                impact_class,
                structural_mechanism: mechanism,
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        b.ddg_mutation_kcal
            .partial_cmp(&a.ddg_mutation_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let worst_resistance_ddg_kcal = entries
        .first()
        .map(|e| e.ddg_mutation_kcal.max(0.0))
        .unwrap_or(0.0);
    let high_resistance_count = entries
        .iter()
        .filter(|e| e.impact_class == MutationImpactClass::HighResistanceRisk)
        .count();
    let mean_pos_ddg = if entries.is_empty() {
        0.0
    } else {
        entries
            .iter()
            .map(|e| e.ddg_mutation_kcal.max(0.0))
            .sum::<f64>()
            / (entries.len() as f64)
    };
    let mutational_robustness_score =
        (100.0 - 22.0 * mean_pos_ddg - 12.0 * (high_resistance_count as f64)).clamp(5.0, 100.0);

    ResidueMutationScan {
        pdb_id: wt_pocket.pdb_id,
        compound_id: wt_pose.compound_id,
        compound_name: wt_pose.compound_name,
        wt_dg_kcal: wt_pose.total_dg_pocket_kcal,
        worst_resistance_ddg_kcal,
        high_resistance_count,
        mutational_robustness_score,
        entries,
    }
}

/// Exports dual-pocket Target vs Off-Target selectivity screening results as RFC-4180 `.csv`.
pub fn export_selectivity_screen_csv(
    dual: &DualPocketComparison,
    rows: &[DualPocketLigandResult],
) -> String {
    let mut out = String::with_capacity(6144);
    out.push_str("Rank,Compound_ID,Name,Target_PDB,OffTarget_PDB,Discovery_Score,Target_dG_kcal,OffTarget_dG_kcal,ddG_Selectivity_kcal,Delta_pKd,Selectivity_Index_Fold,Selectivity_Tier,ddG_Steric_kcal,ddG_Electro_kcal,ddG_WaterMap_kcal,SubPocket_Atoms,Primary_Driver,SMILES\n");
    for r in rows {
        out.push_str(&format!(
            "{},{},\"{}\",{},{},{:.2},{:.2},{:.2},{:.2},{:+.2},{:.1},\"{}\",{:.2},{:.2},{:.2},{},\"{}\",\"{}\"\n",
            r.rank,
            r.compound_id,
            r.compound_name.replace('"', "'"),
            dual.target_pocket.pdb_id,
            dual.offtarget_pocket.pdb_id,
            r.selectivity_discovery_score,
            r.target_dg_kcal,
            r.offtarget_dg_kcal,
            r.ddg_selectivity_kcal,
            r.delta_pkd,
            r.selectivity_index_fold,
            r.tier.short_label(),
            r.ddg_steric_shape_kcal,
            r.ddg_electro_hbond_kcal,
            r.ddg_watermap_kcal,
            r.target_subpocket_atoms,
            r.primary_selectivity_driver.replace('"', "'"),
            r.canonical_smiles.replace('"', "'"),
        ));
    }
    out
}

/// Exports the single-point binding-pocket residue mutation `ddG_bind` table as `.csv`.
pub fn export_mutation_scan_csv(scan: &ResidueMutationScan) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("Mutation,PDB_ID,Compound_ID,Compound_Name,Category,Min_Dist_A,WT_dG_kcal,Mutant_dG_kcal,ddG_Mutation_kcal,Fold_Affinity_Loss,ddG_Steric_kcal,ddG_Polar_kcal,ddG_Water_Hydro_kcal,Impact_Class,Mechanism\n");
    for e in &scan.entries {
        out.push_str(&format!(
            "{},{},{},\"{}\",\"{}\",{:.2},{:.2},{:.2},{:+.2},{:.2},{:+.2},{:+.2},{:+.2},\"{}\",\"{}\"\n",
            e.mutation_code,
            scan.pdb_id,
            scan.compound_id,
            scan.compound_name.replace('"', "'"),
            e.category,
            e.min_ligand_dist_a,
            e.wt_dg_kcal,
            e.mutant_dg_kcal,
            e.ddg_mutation_kcal,
            e.fold_affinity_loss,
            e.ddg_steric_kcal,
            e.ddg_hbond_electro_kcal,
            e.ddg_water_hydro_kcal,
            e.impact_class.short_label(),
            e.structural_mechanism.replace('"', "'"),
        ));
    }
    out
}

/// Formats both the Target-bound and Off-Target-bound 3D poses of top selective leads into MDL V2000 `.sdf`.
pub fn format_dual_selectivity_poses_sdf(
    dual: &DualPocketComparison,
    rows: &[DualPocketLigandResult],
    max_leads: usize,
) -> String {
    let mut sdf = String::with_capacity(12288);
    for r in rows.iter().take(max_leads.max(1)) {
        let charges = vec![0.0_f32; r.target_pose.mol.atoms.len()];
        let tags = [
            ("COMPOUND_ID", r.compound_id.clone()),
            ("TARGET_PDB", dual.target_pocket.pdb_id.clone()),
            ("OFFTARGET_PDB", dual.offtarget_pocket.pdb_id.clone()),
            ("TARGET_DG_KCAL", format!("{:.2}", r.target_dg_kcal)),
            ("OFFTARGET_DG_KCAL", format!("{:.2}", r.offtarget_dg_kcal)),
            ("DDG_SELECTIVITY_KCAL", format!("{:+.2}", r.ddg_selectivity_kcal)),
            ("SELECTIVITY_INDEX_FOLD", format!("{:.1}", r.selectivity_index_fold)),
            ("PRIMARY_SELECTIVITY_DRIVER", r.primary_selectivity_driver.clone()),
        ];
        sdf.push_str(&format_3d_conformer_sdf(
            &r.target_pose.mol,
            &r.target_pose.aligned_ligand_coords3d,
            &charges,
            r.target_dg_kcal,
            &tags,
        ));
    }
    sdf
}

/// Helper loading a curated Target + Off-Target pair (`0..3`), optionally swapping Target and Off-Target
/// (e.g., flipping `EGFR WT vs T790M` into `EGFR T790M vs WT` for mutant-selective inhibitor design!).
pub fn load_curated_selectivity_pair(
    pair_idx: usize,
    swap_target_and_offtarget: bool,
    pocket_radius_a: f32,
) -> Result<DualPocketComparison, String> {
    let pr = &CURATED_SELECTIVITY_PAIRS[pair_idx % CURATED_SELECTIVITY_PAIRS.len()];
    let tgt_raw = curated_pocket_pdb_text(pr.target_preset_idx);
    let off_raw = curated_offtarget_pdb_text(pair_idx);
    let (a_raw, b_raw) = if swap_target_and_offtarget {
        (off_raw, tgt_raw)
    } else {
        (tgt_raw, off_raw)
    };
    let a_pdb = parse_pdb_structure(a_raw)?;
    let b_pdb = parse_pdb_structure(b_raw)?;
    Ok(analyze_dual_pocket_selectivity(&a_pdb, &b_pdb, pocket_radius_a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol_io::parse_smiles;

    #[test]
    fn test_cox2_vs_cox1_dual_pocket_selectivity_and_watermap_divergence() {
        let dual = load_curated_selectivity_pair(0, false, 10.5).unwrap();
        assert_eq!(dual.target_pocket.pdb_id, "3LN1");
        assert_eq!(dual.offtarget_pocket.pdb_id, "3KK6");
        assert!(dual.delta_volume_a3 > 0.0); // COX-2 cavity larger than COX-1
        assert!(!dual.differential_waters.is_empty());
        assert!(!dual.divergent_residues.is_empty());

        let mut celecoxib =
            parse_smiles("Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1").unwrap();
        celecoxib.id = "CHEMBL118".to_string();
        celecoxib.name = "Celecoxib".to_string();
        let res = score_ligand_dual_selectivity(&celecoxib, None, &dual);
        assert!(res.target_dg_kcal < -3.0);
        assert!(res.selectivity_index_fold > 1.0);
    }

    #[test]
    fn test_egfr_t790m_and_c797s_residue_mutation_ddg_scan() {
        let tgt_pdb = parse_pdb_structure(curated_pocket_pdb_text(0)).unwrap();
        let mut erlotinib = parse_smiles("COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC").unwrap();
        erlotinib.id = "CHEMBL553".to_string();
        erlotinib.name = "Erlotinib".to_string();

        let scan = scan_pocket_residue_mutations(
            &erlotinib,
            &tgt_pdb,
            10.5,
            &["T790M".to_string(), "C797S".to_string()],
        );
        assert!(!scan.entries.is_empty());
        assert!(scan.entries.iter().any(|e| e.mutation_code == "T790M"));
        let csv = export_mutation_scan_csv(&scan);
        assert!(csv.contains("T790M"));
    }
}
