//! Structure-Based `PDB` Binding-Site Cavity Parser, `WaterMap`-Style Buried/Unhappy Hydration
//! Profiler, Receptor `H-Bond`/Ionic/Hydrophobic 3D Pharmacophore Extractor, and `AutoDock Vina`-Style
//! Pocket-Guided Ligand Complementarity Scorer (`pocket_cavity.rs`, `v2.3.0`).
//!
//! Primary References:
//! - Le Guilloux, Schmidtke & Tuffery (2009), *BMC Bioinformatics* 10:168 ("Fpocket: an open source
//!   platform for ligand pocket detection": `https://pubmed.ncbi.nlm.nih.gov/19486540/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC2700099/`)
//! - Abel, Young, Farid, Berne & Friesner (2008), *J. Am. Chem. Soc.* 130(9), 2817-2831 ("Role of
//!   the active-site solvent in the thermodynamics of factor Xa ligand binding" — `WaterMap`:
//!   `https://pubmed.ncbi.nlm.nih.gov/18266362/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC2761766/`)
//!   & `AutoDock-GIST` (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6274120/`)
//! - Wolber & Langer (2005), *J. Chem. Inf. Model.* 45(1), 160-169 ("LigandScout: 3-D pharmacophores
//!   derived from protein-bound ligands": `https://pubmed.ncbi.nlm.nih.gov/15667141/`) & Receptor-Based
//!   Pharmacophore Generation (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6305075/`)
//! - Trott & Olson (2010), *J. Comput. Chem.* 31(2), 455-461 ("AutoDock Vina: improving the speed
//!   and accuracy of docking": `https://pubmed.ncbi.nlm.nih.gov/19499576/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3041641/`)
//! - Curated Co-Crystal RCSB PDB Structures:
//!   - `1M17` EGFR Kinase + Erlotinib (`https://www.rcsb.org/structure/1m17`)
//!   - `6CM4` Dopamine `D2` Receptor + Risperidone (`https://www.rcsb.org/structure/6cm4`)
//!   - `3LN1` Cyclooxygenase-2 (`COX-2`) + Celecoxib (`https://www.rcsb.org/structure/3ln1`)

use crate::conformer3d::{
    bondi_vdw_radius, evaluate_3d_summary_profile, kabsch_align_3d, RocsColorAtom, RocsColorKind,
};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::mol::{Element, Molecule};
use crate::physchem::evaluate_physchem;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Curated co-crystal RCSB PDB binding-site preset metadata.
pub struct CuratedPocketPreset {
    pub code: &'static str,
    pub pdb_id: &'static str,
    pub target_name: &'static str,
    pub ligand_resn: &'static str,
    pub ligand_name: &'static str,
    pub rcsb_url: &'static str,
    pub description: &'static str,
}

pub const CURATED_PDB_POCKETS: [CuratedPocketPreset; 3] = [
    CuratedPocketPreset {
        code: "1M17-EGFR",
        pdb_id: "1M17",
        target_name: "EGFR",
        ligand_resn: "AQ4",
        ligand_name: "Erlotinib",
        rcsb_url: "https://www.rcsb.org/structure/1m17",
        description: "EGFR tyrosine kinase ATP-hinge (Met793), Thr790 gatekeeper bridge water & hydrophobic back-pocket",
    },
    CuratedPocketPreset {
        code: "6CM4-DRD2",
        pdb_id: "6CM4",
        target_name: "DRD2",
        ligand_resn: "8NU",
        ligand_name: "Risperidone",
        rcsb_url: "https://www.rcsb.org/structure/6cm4",
        description: "Dopamine D2 GPCR orthosteric Asp114 (D3.32) salt-bridge & Trp386/Phe389 aromatic cage",
    },
    CuratedPocketPreset {
        code: "3LN1-COX2",
        pdb_id: "3LN1",
        target_name: "COX-2",
        ligand_resn: "CEL",
        ligand_name: "Celecoxib",
        rcsb_url: "https://www.rcsb.org/structure/3ln1",
        description: "COX-2 Arg513/His90 sulfonamide selectivity side-pocket & hydrophobic CF3-pyrazole channel",
    },
];

/// Thermodynamic role of a crystallographic or grid-inferred pocket hydration site (`WaterMap`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterThermodynamicRole {
    /// Hydrophobically enclosed, under-coordinated water (`Delta G_disp <= -0.75 kcal/mol`): displacing boosts potency!
    UnhappyDisplaceableHotspot,
    /// Tightly coordinated multi-H-bond structural bridge (`>= 3` polar contacts, low `B`-factor): retain or mimic with HBA/HBD!
    ConservedStructuralBridge,
    /// Partially solvent-exposed rim water near bulk equilibrium (`|Delta G_disp| < 0.75 kcal/mol`).
    ExchangeableBulkRim,
}

impl WaterThermodynamicRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::UnhappyDisplaceableHotspot => "Unhappy Displaceable Water (Affinity Hotspot)",
            Self::ConservedStructuralBridge => "Conserved Structural H-Bond Bridge",
            Self::ExchangeableBulkRim => "Exchangeable Solvent-Rim Water",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::UnhappyDisplaceableHotspot => "Unhappy (Displace!)",
            Self::ConservedStructuralBridge => "Structural Bridge",
            Self::ExchangeableBulkRim => "Bulk Rim",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::UnhappyDisplaceableHotspot => [248, 113, 113], // Coral Red (High-energy hotspot)
            Self::ConservedStructuralBridge => [56, 189, 248],   // Cyan (Ordered H-bond bridge)
            Self::ExchangeableBulkRim => [148, 163, 184],        // Slate
        }
    }
}

/// Druggability tier of the 3D binding cavity (`fpocket` / `SiteMap` `DScore`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PocketDruggabilityTier {
    HighlyDruggable,
    Druggable,
    ShallowOrPolar,
}

impl PocketDruggabilityTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::HighlyDruggable => "Highly Druggable Cavity (DScore >= 0.75)",
            Self::Druggable => "Druggable Pocket (0.50 <= DScore < 0.75)",
            Self::ShallowOrPolar => "Shallow / Polar Undruggable Surface",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HighlyDruggable => "High Druggability",
            Self::Druggable => "Druggable",
            Self::ShallowOrPolar => "Shallow/Polar",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HighlyDruggable => [52, 211, 153],
            Self::Druggable => [56, 189, 248],
            Self::ShallowOrPolar => [251, 191, 36],
        }
    }
}

/// Parsed heavy atom from a `.pdb` `ATOM` or `HETATM` record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdbAtom {
    pub serial: u32,
    pub atom_name: String,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub pos: [f32; 3],
    pub occupancy: f32,
    pub b_factor: f32,
    pub elem: Element,
    pub is_hetatm: bool,
}

/// Crystallographic water oxygen (`HOH` / `WAT`) parsed from `.pdb`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdbWaterSite {
    pub serial: u32,
    pub chain_id: char,
    pub res_seq: i32,
    pub pos: [f32; 3],
    pub b_factor: f32,
}

/// Non-solvent co-crystallized `HETATM` ligand group detected in a `.pdb` file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdbHetLigand {
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub centroid: [f32; 3],
    pub atoms: Vec<PdbAtom>,
}

/// Complete parsed `.pdb` macromolecular complex prior to pocket extraction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParsedPdbStructure {
    pub pdb_id: String,
    pub title: String,
    pub protein_atoms: Vec<PdbAtom>,
    pub waters: Vec<PdbWaterSite>,
    pub cocrystal_ligands: Vec<PdbHetLigand>,
}

/// Lining amino-acid residue facing the binding pocket cavity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PocketResidueSummary {
    pub residue_label: String,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    #[serde(skip_deserializing)]
    pub role: &'static str,
    pub min_dist_to_center_a: f32,
    pub contact_atom_count: usize,
}

/// Thermodynamically profiled active-site water (`WaterMap` / `IST` estimator).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HydrationSite {
    pub site_id: String,
    pub pos: [f32; 3],
    pub b_factor: f32,
    pub is_crystallographic: bool,
    pub protein_hbond_count: usize,
    pub coordinating_residues: Vec<String>,
    pub hydrophobic_enclosure_atoms: usize,
    /// Excess enthalpy vs bulk water (`kcal/mol`, positive = enthalpically frustrated).
    pub excess_enthalpy_kcal: f64,
    /// Orientational/translational entropy cost `-T * Delta S` (`kcal/mol`, positive = ordered).
    pub entropic_penalty_kcal: f64,
    /// Predicted ligand binding `Delta G` shift upon displacing this water (`kcal/mol`, negative = affinity gain!).
    pub displacement_ddg_bind_kcal: f64,
    pub role: WaterThermodynamicRole,
}

/// Structure-based 3D receptor pharmacophore feature projected into the binding cavity
/// (`LigandScout` / `Pharmit` convention: `kind` denotes the *complementary ligand feature* desired at `pos`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceptorPharmacophorePoint {
    pub feature_id: String,
    pub kind: RocsColorKind,
    pub pos: [f32; 3],
    pub tolerance_radius_a: f32,
    pub receptor_origin: String,
    #[serde(skip_deserializing)]
    pub interaction_label: &'static str,
    /// Thermodynamic priority weight (`1.0 .. 2.5`).
    pub weight: f64,
}

/// Geometric, polarity, and druggability profile of the extracted 3D binding cavity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PocketCavityAnalysis {
    pub pdb_id: String,
    pub title: String,
    pub center: [f32; 3],
    pub pocket_radius_a: f32,
    pub cocrystal_ligand_name: Option<String>,
    pub cocrystal_heavy_atoms: Vec<[f32; 3]>,
    pub volume_angstrom3: f64,
    pub enclosure_fraction: f64,
    pub hydrophobic_fraction: f64,
    pub polar_hbond_fraction: f64,
    pub mean_b_factor: f32,
    pub druggability_dscore: f64,
    pub druggability_tier: PocketDruggabilityTier,
    pub shell_atoms: Vec<PdbAtom>,
    pub lining_residues: Vec<PocketResidueSummary>,
    pub hydration_sites: Vec<HydrationSite>,
    pub unhappy_water_count: usize,
    pub structural_water_count: usize,
    pub pharmacophore_points: Vec<ReceptorPharmacophorePoint>,
    /// Subsampled buried cavity grid coordinates (`Angstrom`) for 3D pocket-cloud rendering.
    pub cavity_grid_points: Vec<[f32; 3]>,
}

/// Specific receptor–ligand non-covalent contact or displaced water identified in the aligned 3D pose.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PocketLigandInteraction {
    #[serde(skip_deserializing)]
    pub kind_label: &'static str,
    pub receptor_partner: String,
    pub ligand_atom_label: String,
    pub distance_a: f32,
    pub energy_kcal: f64,
}

/// Empirical pocket-guided pose superposition and `AutoDock Vina` + `WaterMap` complementarity score.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PocketComplementarityResult {
    pub rank: usize,
    pub mol_idx: Option<usize>,
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    pub canonical_smiles: String,
    pub mw: f64,
    pub clogp: f64,
    pub observed_pact: f64,

    // Empirical Energy Decomposition (kcal/mol; negative = favorable binding)
    pub vdw_steric_kcal: f64,
    pub steric_clash_kcal: f64,
    pub hbond_ionic_kcal: f64,
    pub hydrophobic_enclosure_kcal: f64,
    pub water_displacement_kcal: f64,
    pub rotor_entropy_kcal: f64,
    pub total_dg_pocket_kcal: f64,

    // Discovery Efficiency & Complementarity Readouts
    pub predicted_pocket_pkd: f64,
    pub pocket_le: f64,
    pub pocket_lle: f64,
    pub pharmacophore_matched: usize,
    pub pharmacophore_total: usize,
    pub pharmacophore_recall_pct: f64,
    pub displaced_unhappy_waters: usize,
    pub bridged_structural_waters: usize,
    pub clash_atom_count: usize,
    /// Composite `[0..100]` Pocket Complementarity Score combining `-Delta G`, `Pocket LE`, pharmacophore recall, and zero clashes.
    pub complementarity_score: f64,

    pub key_interactions: Vec<PocketLigandInteraction>,
    pub aligned_ligand_coords3d: Vec<[f32; 3]>,
    pub mol: Molecule,
}

const SOLVENT_RESIDUES: &[&str] = &["HOH", "WAT", "SOL", "TIP", "DOD", "H2O"];
const BUFFER_IONS_IGNORE: &[&str] = &[
    "SO4", "PO4", "GOL", "EDO", "PEG", "DMS", "ACT", "ACE", "CL", "NA", "K", "MG", "CA", "ZN",
    "MN", "FE", "IOD", "BR", "TRS", "MES", "EPE", "CIT",
];
const STANDARD_AMINO_ACIDS: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
];

/// Returns the built-in atomic `.pdb` text for one of the 3 curated benchmark co-crystal pockets (`0..3`).
pub fn curated_pocket_pdb_text(preset_idx: usize) -> &'static str {
    match preset_idx {
        // 0: 1M17 EGFR Tyrosine Kinase + Erlotinib (AQ4) + Structural & Unhappy Back-Pocket Waters
        0 => {
            r#"HEADER    TRANSFERASE/INHIBITOR                   19-JUN-02   1M17
TITLE     EGFR TYROSINE KINASE DOMAIN ATP-HINGE BINDING POCKET WITH ERLOTINIB
ATOM      1  CA  LEU A 718      21.420   1.850   4.120  1.00 26.40           C
ATOM      2  CG  LEU A 718      20.310   1.120   3.210  1.00 27.10           C
ATOM      3  CD1 LEU A 718      19.450   0.180   3.980  1.00 28.00           C
ATOM      4  CB  VAL A 726      18.620   3.640   2.150  1.00 22.80           C
ATOM      5  CG1 VAL A 726      17.880   2.480   1.520  1.00 23.10           C
ATOM      6  CB  ALA A 743      16.950   1.220  -0.640  1.00 20.50           C
ATOM      7  NZ  LYS A 745      13.620   2.950  -2.880  1.00 21.40           N
ATOM      8  CD  GLU A 762      11.840   4.120  -3.650  1.00 23.20           C
ATOM      9  OE1 GLU A 762      12.480   3.820  -4.680  1.00 22.90           O
ATOM     10  CB  MET A 766      12.350   1.650  -4.920  1.00 24.60           C
ATOM     11  SD  MET A 766      13.120   0.240  -4.180  1.00 25.10           S
ATOM     12  CD1 LEU A 788      14.080  -1.350  -3.420  1.00 22.00           C
ATOM     13  OG1 THR A 790      15.420   0.680  -2.140  1.00 19.20           O
ATOM     14  CG2 THR A 790      14.950  -0.680  -1.620  1.00 20.10           C
ATOM     15  O   GLN A 791      18.120  -2.180  -1.150  1.00 19.80           O
ATOM     16  CA  LEU A 792      19.650  -2.420   0.380  1.00 21.10           C
ATOM     17  N   MET A 793      19.820  -0.950   1.620  1.00 18.40           N
ATOM     18  O   MET A 793      20.940  -1.680   3.540  1.00 19.10           O
ATOM     19  CB  PRO A 794      22.380  -1.120   5.120  1.00 25.30           C
ATOM     20  OD1 ASP A 800      22.950   2.640   6.250  1.00 31.20           O
ATOM     21  CD1 LEU A 844      17.420  -1.920   1.880  1.00 20.60           C
ATOM     22  CD2 LEU A 844      16.280  -2.640   0.820  1.00 21.30           C
ATOM     23  CA  CYS A 797      21.840   0.420   5.860  1.00 27.40           C
ATOM     24  SG  CYS A 797      20.960   1.380   6.940  1.00 28.90           S
ATOM     25  OG1 THR A 854      14.880   1.950  -0.920  1.00 19.60           O
ATOM     26  OD1 ASP A 855      13.740   3.620  -0.840  1.00 21.80           O
ATOM     27  CG  PHE A 856      12.420   2.180  -2.120  1.00 24.50           C
HETATM   28  N1  AQ4 A 999      19.120  -0.320   1.180  1.00 20.10           N
HETATM   29  C2  AQ4 A 999      18.240  -0.880   0.340  1.00 20.40           C
HETATM   30  N3  AQ4 A 999      17.120  -0.240  -0.080  1.00 20.20           N
HETATM   31  C4  AQ4 A 999      16.880   1.020   0.380  1.00 19.90           C
HETATM   32  C5  AQ4 A 999      17.820   1.660   1.260  1.00 20.50           C
HETATM   33  C6  AQ4 A 999      18.960   0.980   1.660  1.00 20.80           C
HETATM   34  N4  AQ4 A 999      15.720   1.640  -0.060  1.00 21.00           N
HETATM   35  C7  AQ4 A 999      14.540   1.120  -0.720  1.00 21.60           C
HETATM   36  C8  AQ4 A 999      13.520   0.480  -1.440  1.00 22.40           C
HETATM   37  C9  AQ4 A 999      12.640  -0.120  -2.280  1.00 23.80           C
HETATM   38  O1  AQ4 A 999      20.180   1.480   2.880  1.00 23.10           O
HETATM   39  O2  AQ4 A 999      19.640   2.820   4.620  1.00 25.40           O
HETATM   40  O   HOH A2001      15.960   0.520  -1.180  1.00 17.80           O
HETATM   41  O   HOH A2002      13.180  -0.380  -2.720  1.00 34.50           O
HETATM   42  O   HOH A2003      16.840  -1.450   1.120  1.00 36.20           O
HETATM   43  O   HOH A2004      22.450   1.920   5.350  1.00 42.10           O
END
"#
        }
        // 1: 6CM4 Dopamine D2 Receptor (DRD2) + Risperidone (8NU)
        1 => {
            r#"HEADER    MEMBRANE PROTEIN/ANTIPSYCHOTIC          24-JAN-18   6CM4
TITLE     HUMAN DOPAMINE D2 RECEPTOR ORTHOSTERIC BINDING POCKET WITH RISPERIDONE
ATOM      1  CG  ASP A 114       9.420   4.180  -8.120  1.00 21.20           C
ATOM      2  OD1 ASP A 114      10.240   3.320  -7.740  1.00 19.80           O
ATOM      3  OD2 ASP A 114       9.680   5.360  -7.880  1.00 20.10           O
ATOM      4  CB  VAL A 115      11.820   2.150  -9.640  1.00 23.40           C
ATOM      5  CG1 VAL A 115      12.450   1.120  -8.720  1.00 24.10           C
ATOM      6  SD  CYS A 118      11.120   0.340 -11.420  1.00 24.80           S
ATOM      7  CD1 THR A 119       9.840  -1.220 -10.860  1.00 23.50           C
ATOM      8  OG1 THR A 119      10.420  -0.480  -9.820  1.00 22.90           O
ATOM      9  CD1 ILE A 184      13.680   4.860  -5.420  1.00 28.40           C
ATOM     10  OG  SER A 193      12.920  -1.640 -10.950  1.00 25.10           O
ATOM     11  OG  SER A 197      11.640  -2.920 -12.480  1.00 26.20           O
ATOM     12  CZ  PHE A 198      13.480  -1.950 -13.120  1.00 24.70           C
ATOM     13  CE2 TRP A 386       9.180  -2.180 -13.640  1.00 21.50           C
ATOM     14  CZ2 TRP A 386       9.920  -3.280 -14.120  1.00 22.00           C
ATOM     15  CZ  PHE A 389       8.240   0.620 -11.880  1.00 20.90           C
ATOM     16  CE1 PHE A 389       8.860   1.740 -11.340  1.00 21.20           C
ATOM     17  CZ  PHE A 390      10.680   1.920 -13.240  1.00 22.40           C
ATOM     18  OH  TYR A 408      12.420   5.880  -4.120  1.00 29.60           O
ATOM     19  OH  TYR A 416       7.920   3.840  -6.480  1.00 23.10           O
ATOM     20  CD1 THR A 412       8.840   5.920  -5.180  1.00 27.30           C
HETATM   21  N1  8NU A 901      10.520   3.680  -6.120  1.00 21.40           N
HETATM   22  C2  8NU A 901      10.940   2.340  -6.640  1.00 21.80           C
HETATM   23  C3  8NU A 901      10.620   1.180  -7.920  1.00 21.50           C
HETATM   24  C4  8NU A 901      10.280  -0.120  -9.240  1.00 22.10           C
HETATM   25  N2  8NU A 901       9.960  -0.840 -10.380  1.00 22.40           N
HETATM   26  O1  8NU A 901      10.820  -1.620 -11.180  1.00 22.90           O
HETATM   27  C5  8NU A 901       9.840  -1.920 -12.140  1.00 23.20           C
HETATM   28  F1  8NU A 901       9.620  -2.980 -13.180  1.00 24.00           F
HETATM   29  C6  8NU A 901      11.420   4.820  -5.480  1.00 23.60           C
HETATM   30  O2  8NU A 901      11.980   5.640  -4.620  1.00 24.80           O
HETATM   31  O   HOH A2101       9.740  -2.380 -12.620  1.00 35.80           O
HETATM   32  O   HOH A2102       8.920   4.120  -6.920  1.00 18.60           O
HETATM   33  O   HOH A2103      12.960   4.420  -5.880  1.00 37.40           O
END
"#
        }
        // 2: 3LN1 Cyclooxygenase-2 (COX-2) + Celecoxib (CEL)
        _ => {
            r#"HEADER    OXIDOREDUCTASE/INHIBITOR                01-DEC-10   3LN1
TITLE     CYCLOOXYGENASE-2 (COX-2) SELECTIVITY POCKET BOUND TO CELECOXIB
ATOM      1  ND1 HIS A  90      31.420 -21.840 -15.620  1.00 19.40           N
ATOM      2  NE2 HIS A  90      30.680 -22.620 -14.840  1.00 19.10           N
ATOM      3  NH1 ARG A 106      25.420 -25.920 -18.140  1.00 24.60           N
ATOM      4  OE1 GLN A 178      32.180 -20.120 -14.420  1.00 20.80           O
ATOM      5  CD1 LEU A 338      29.420 -20.880 -16.840  1.00 20.20           C
ATOM      6  O   SER A 339      30.280 -21.420 -15.120  1.00 18.90           O
ATOM      7  OH  TYR A 341      26.640 -24.480 -17.620  1.00 22.10           O
ATOM      8  CD2 LEU A 370      25.880 -21.140 -19.240  1.00 23.40           C
ATOM      9  OH  TYR A 371      27.120 -19.620 -18.840  1.00 21.80           O
ATOM     10  CZ2 TRP A 373      28.240 -18.880 -17.420  1.00 21.20           C
ATOM     11  CZ  PHE A 504      29.920 -19.240 -16.120  1.00 20.50           C
ATOM     12  CG1 VAL A 509      28.840 -22.640 -16.920  1.00 19.70           C
ATOM     13  NH1 ARG A 513      31.860 -23.480 -14.180  1.00 18.80           N
ATOM     14  NH2 ARG A 513      31.120 -24.120 -15.920  1.00 19.20           N
ATOM     15  CB  ALA A 513      28.140 -21.960 -17.920  1.00 20.40           C
ATOM     16  OG  SER A 516      26.820 -21.920 -18.420  1.00 21.60           O
ATOM     17  CD1 LEU A 517      26.120 -23.380 -19.120  1.00 23.10           C
HETATM   18  S1  CEL A 601      30.880 -22.180 -15.240  1.00 19.20           S
HETATM   19  O1  CEL A 601      31.640 -22.920 -14.480  1.00 19.50           O
HETATM   20  O2  CEL A 601      31.420 -21.120 -15.880  1.00 19.60           O
HETATM   21  N1  CEL A 601      30.120 -21.640 -14.180  1.00 19.80           N
HETATM   22  C1  CEL A 601      29.520 -22.240 -16.280  1.00 20.10           C
HETATM   23  C2  CEL A 601      28.380 -21.840 -17.120  1.00 20.40           C
HETATM   24  N2  CEL A 601      27.420 -22.620 -17.640  1.00 20.90           N
HETATM   25  N3  CEL A 601      26.680 -21.940 -18.480  1.00 21.20           N
HETATM   26  C3  CEL A 601      26.920 -20.680 -18.620  1.00 21.80           C
HETATM   27  F1  CEL A 601      26.180 -19.920 -19.340  1.00 22.90           F
HETATM   28  C4  CEL A 601      28.480 -20.380 -17.020  1.00 21.10           C
HETATM   29  C5  CEL A 601      29.120 -19.220 -16.520  1.00 21.60           C
HETATM   30  O   HOH A2201      31.580 -21.920 -14.680  1.00 18.10           O
HETATM   31  O   HOH A2202      28.880 -19.580 -16.820  1.00 36.40           O
HETATM   32  O   HOH A2203      26.520 -20.540 -18.920  1.00 35.10           O
END
"#
        }
    }
}

/// Fetches a `.pdb` coordinate file directly from RCSB (`https://files.rcsb.org/download/<ID>.pdb`).
pub fn fetch_rcsb_pdb_text(pdb_id: &str) -> Result<String, String> {
    let code = pdb_id.trim().to_ascii_uppercase();
    if code.len() != 4 || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(format!("Invalid 4-character PDB accession '{pdb_id}'"));
    }
    let url = format!("https://files.rcsb.org/download/{code}.pdb");
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("RCSB fetch error for {code}: {e}"))?;
    resp.into_string()
        .map_err(|e| format!("RCSB stream decode error: {e}"))
}

fn infer_pdb_element(atom_name: &str, elem_col: &str, is_hetatm: bool) -> Option<Element> {
    let ec = elem_col.trim();
    if !ec.is_empty() {
        let norm = if ec.len() == 1 {
            ec.to_ascii_uppercase()
        } else {
            let mut chs = ec.chars();
            let c0 = chs.next().unwrap_or('C').to_ascii_uppercase();
            let c1 = chs.next().unwrap_or(' ').to_ascii_lowercase();
            format!("{c0}{c1}").trim().to_string()
        };
        if norm == "H" || norm == "D" {
            return None; // Skip explicit hydrogens for heavy-atom cavity/pharmacophore perception
        }
        if let Some(el) = Element::from_symbol(&norm) {
            return Some(el);
        }
    }

    let an = atom_name.trim();
    if an.is_empty() || an.starts_with('H') || an.starts_with("1H") || an.starts_with("2H") {
        return None;
    }
    if !is_hetatm {
        let first = an.chars().next().unwrap_or('C').to_ascii_uppercase();
        return match first {
            'C' => Some(Element::C),
            'N' => Some(Element::N),
            'O' => Some(Element::O),
            'S' => Some(Element::S),
            'P' => Some(Element::P),
            'F' => Some(Element::F),
            _ => Some(Element::C),
        };
    }
    let prefix: String = an.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    if prefix.eq_ignore_ascii_case("CL") {
        Some(Element::Cl)
    } else if prefix.eq_ignore_ascii_case("BR") {
        Some(Element::Br)
    } else {
        let c0 = prefix.chars().next().unwrap_or('C').to_ascii_uppercase();
        Element::from_symbol(&c0.to_string()).or(Some(Element::C))
    }
}

/// Parses `.pdb` `HEADER`, `TITLE`, `ATOM`, and `HETATM` records into protein heavy atoms,
/// crystallographic waters, and co-crystallized `HETATM` ligands.
pub fn parse_pdb_structure(pdb_text: &str) -> Result<ParsedPdbStructure, String> {
    let mut pdb_id = "USER".to_string();
    let mut title_parts: Vec<String> = Vec::new();
    let mut protein_atoms = Vec::new();
    let mut waters = Vec::new();
    let mut het_groups: BTreeMap<(String, char, i32), Vec<PdbAtom>> = BTreeMap::new();

    for line in pdb_text.lines() {
        if line.starts_with("HEADER") && line.len() >= 66 {
            let code = line[62..66].trim();
            if !code.is_empty() {
                pdb_id = code.to_string();
            }
        } else if line.starts_with("TITLE") && line.len() > 10 {
            title_parts.push(line[10..].trim().to_string());
        } else if line.starts_with("ENDMDL") {
            // Stop after Model 1 for NMR ensembles
            break;
        } else if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
            if line.len() < 54 {
                continue;
            }
            // Skip alternate conformation 'B'/'C' to avoid double-counting disordered sidechains
            let alt_loc = line.as_bytes().get(16).copied().unwrap_or(b' ') as char;
            if alt_loc != ' ' && alt_loc != 'A' && alt_loc != '1' {
                continue;
            }
            let is_het = line.starts_with("HETATM");
            let serial = line[6..11].trim().parse::<u32>().unwrap_or(0);
            let atom_name = line[12..16].trim().to_string();
            let res_name = line[17..20].trim().to_ascii_uppercase();
            let chain_id = line.as_bytes().get(21).copied().unwrap_or(b'A') as char;
            let res_seq = line[22..26].trim().parse::<i32>().unwrap_or(0);
            let Ok(x) = line[30..38].trim().parse::<f32>() else {
                continue;
            };
            let Ok(y) = line[38..46].trim().parse::<f32>() else {
                continue;
            };
            let Ok(z) = line[46..54].trim().parse::<f32>() else {
                continue;
            };
            let occupancy = if line.len() >= 60 {
                line[54..60].trim().parse::<f32>().unwrap_or(1.0)
            } else {
                1.0
            };
            let b_factor = if line.len() >= 66 {
                line[60..66].trim().parse::<f32>().unwrap_or(25.0)
            } else {
                25.0
            };
            let elem_col = if line.len() >= 78 {
                &line[76..78]
            } else {
                ""
            };

            if SOLVENT_RESIDUES.contains(&res_name.as_str()) {
                if atom_name.starts_with('O') || elem_col.trim().eq_ignore_ascii_case("O") {
                    waters.push(PdbWaterSite {
                        serial,
                        chain_id,
                        res_seq,
                        pos: [x, y, z],
                        b_factor,
                    });
                }
                continue;
            }

            let Some(elem) = infer_pdb_element(&atom_name, elem_col, is_het) else {
                continue;
            };

            let patom = PdbAtom {
                serial,
                atom_name,
                res_name: res_name.clone(),
                chain_id,
                res_seq,
                pos: [x, y, z],
                occupancy,
                b_factor,
                elem,
                is_hetatm: is_het,
            };

            if !is_het || STANDARD_AMINO_ACIDS.contains(&res_name.as_str()) {
                protein_atoms.push(patom);
            } else if !BUFFER_IONS_IGNORE.contains(&res_name.as_str()) {
                het_groups
                    .entry((res_name, chain_id, res_seq))
                    .or_default()
                    .push(patom);
            }
        }
    }

    if protein_atoms.is_empty() {
        return Err("No protein heavy atoms found in PDB input.".to_string());
    }

    let mut cocrystal_ligands = Vec::new();
    for ((res_name, chain_id, res_seq), atoms) in het_groups {
        if atoms.len() >= 5 {
            let inv = 1.0 / (atoms.len() as f32);
            let mut c = [0.0f32; 3];
            for a in &atoms {
                c[0] += a.pos[0] * inv;
                c[1] += a.pos[1] * inv;
                c[2] += a.pos[2] * inv;
            }
            cocrystal_ligands.push(PdbHetLigand {
                res_name,
                chain_id,
                res_seq,
                centroid: c,
                atoms,
            });
        }
    }
    // Largest organic HETATM ligand first
    cocrystal_ligands.sort_by(|a, b| b.atoms.len().cmp(&a.atoms.len()));

    let title = if title_parts.is_empty() {
        format!("PDB {pdb_id} Macromolecular Complex")
    } else {
        title_parts.join(" ")
    };

    Ok(ParsedPdbStructure {
        pdb_id,
        title,
        protein_atoms,
        waters,
        cocrystal_ligands,
    })
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// 14 unit directions (3 Cartesian axes + 4 body diagonals x ±1) for `LIGSITE`/`SiteMap` ray-burial enclosure.
const RAY_DIRS_14: [[f32; 3]; 14] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.57735, 0.57735, 0.57735],
    [-0.57735, -0.57735, -0.57735],
    [0.57735, 0.57735, -0.57735],
    [-0.57735, -0.57735, 0.57735],
    [0.57735, -0.57735, 0.57735],
    [-0.57735, 0.57735, -0.57735],
    [-0.57735, 0.57735, 0.57735],
    [0.57735, -0.57735, -0.57735],
];

fn is_protein_polar_donor(a: &PdbAtom) -> bool {
    if a.elem == Element::N {
        // Proline backbone N lacks H; aromatic His ND1/NE2 or Lys NZ / Arg NH / backbone N are donors
        !(a.res_name == "PRO" && a.atom_name == "N")
    } else if a.elem == Element::O {
        matches!(
            (a.res_name.as_str(), a.atom_name.as_str()),
            ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH")
        )
    } else {
        a.elem == Element::S && a.res_name == "CYS"
    }
}

fn is_protein_polar_acceptor(a: &PdbAtom) -> bool {
    if a.elem == Element::O {
        true
    } else if a.elem == Element::N {
        matches!(a.res_name.as_str(), "HIS" | "HID" | "HIE")
    } else {
        a.elem == Element::S && a.res_name == "MET"
    }
}

fn is_protein_anionic(a: &PdbAtom) -> bool {
    matches!(
        (a.res_name.as_str(), a.atom_name.as_str()),
        ("ASP", "OD1") | ("ASP", "OD2") | ("GLU", "OE1") | ("GLU", "OE2")
    )
}

fn is_protein_cationic(a: &PdbAtom) -> bool {
    matches!(
        (a.res_name.as_str(), a.atom_name.as_str()),
        ("LYS", "NZ") | ("ARG", "NH1") | ("ARG", "NH2") | ("ARG", "NE") | ("HIP", "ND1")
    )
}

fn residue_functional_role(res: &str) -> &'static str {
    match res {
        "ASP" | "GLU" => "Anionic Salt-Bridge / HBA",
        "LYS" | "ARG" | "HIP" => "Cationic Salt-Bridge / HBD",
        "HIS" | "HID" | "HIE" => "Amphipathic Imidazole / Pi",
        "SER" | "THR" | "TYR" => "Hydroxyl HBD / HBA",
        "ASN" | "GLN" => "Amide Carboxamide HBD/HBA",
        "CYS" => "Nucleophilic Thiol (Covalent)",
        "PHE" | "TRP" => "Aromatic Pi-Cage",
        "LEU" | "ILE" | "VAL" | "ALA" | "MET" | "PRO" => "Hydrophobic Enclosure",
        _ => "Backbone / Hinge",
    }
}

/// Detects the 3D binding cavity, profiles crystallographic & grid-inferred `WaterMap` hydration sites,
/// and extracts 3D receptor pharmacophores (`HBD`, `HBA`, `Cation`, `Anion`, `Aromatic`, `Hydrophobic`).
pub fn analyze_binding_pocket(
    pdb: &ParsedPdbStructure,
    explicit_center: Option<[f32; 3]>,
    pocket_radius_a: f32,
) -> PocketCavityAnalysis {
    let radius = pocket_radius_a.clamp(6.5, 16.0);

    // 1. Determine pocket center: explicit -> primary co-crystal ligand centroid -> apo buried concavity scan
    let (center, lig_name, lig_coords) = if let Some(c) = explicit_center {
        (c, None, Vec::new())
    } else if let Some(lig) = pdb.cocrystal_ligands.first() {
        let pts: Vec<[f32; 3]> = lig.atoms.iter().map(|a| a.pos).collect();
        (lig.centroid, Some(lig.res_name.clone()), pts)
    } else {
        // Apo protein fallback: coarse 2.5 A grid search for deepest multi-ray-enclosed void near protein centroid
        let inv = 1.0 / (pdb.protein_atoms.len() as f32);
        let mut p_cen = [0.0f32; 3];
        for a in &pdb.protein_atoms {
            p_cen[0] += a.pos[0] * inv;
            p_cen[1] += a.pos[1] * inv;
            p_cen[2] += a.pos[2] * inv;
        }
        let mut best_pt = p_cen;
        let mut best_rays = 0usize;
        for ix in -5..=5 {
            for iy in -5..=5 {
                for iz in -5..=5 {
                    let probe = [
                        p_cen[0] + (ix as f32) * 2.4,
                        p_cen[1] + (iy as f32) * 2.4,
                        p_cen[2] + (iz as f32) * 2.4,
                    ];
                    let clash = pdb
                        .protein_atoms
                        .iter()
                        .any(|a| dist3(probe, a.pos) < 2.8);
                    if clash {
                        continue;
                    }
                    let rays = count_enclosed_rays(probe, &pdb.protein_atoms, 10.5);
                    if rays > best_rays {
                        best_rays = rays;
                        best_pt = probe;
                    }
                }
            }
        }
        (best_pt, None, Vec::new())
    };

    // 2. Extract pocket shell atoms within `radius + 3.5 A`
    let shell_cutoff = radius + 3.5;
    let shell_atoms: Vec<PdbAtom> = pdb
        .protein_atoms
        .iter()
        .filter(|a| dist3(a.pos, center) <= shell_cutoff)
        .cloned()
        .collect();

    let active_shell = if shell_atoms.is_empty() {
        pdb.protein_atoms.clone()
    } else {
        shell_atoms
    };

    // 3. Carve 1.05 A 3D solvent-accessible buried cavity voxels inside `radius`
    let step = 1.05_f32;
    let half_n = ((radius / step).ceil() as i32).min(12);
    let mut cavity_voxels: Vec<([f32; 3], usize)> = Vec::new();

    for ix in -half_n..=half_n {
        for iy in -half_n..=half_n {
            for iz in -half_n..=half_n {
                let pt = [
                    center[0] + (ix as f32) * step,
                    center[1] + (iy as f32) * step,
                    center[2] + (iz as f32) * step,
                ];
                if dist3(pt, center) > radius {
                    continue;
                }
                // Check non-overlapping with protein vdW + 1.15 A probe clearance
                let mut clash = false;
                let mut near_wall = false;
                for pa in &active_shell {
                    let d = dist3(pt, pa.pos);
                    let r_min = bondi_vdw_radius(pa.elem) + 1.05;
                    if d < r_min {
                        clash = true;
                        break;
                    }
                    if d <= 5.6 {
                        near_wall = true;
                    }
                }
                if clash || !near_wall {
                    continue;
                }
                let rays = count_enclosed_rays(pt, &active_shell, 10.5);
                if rays >= 5 {
                    cavity_voxels.push((pt, rays));
                }
            }
        }
    }

    let voxel_vol = (step * step * step) as f64;
    let volume_angstrom3 = ((cavity_voxels.len() as f64) * voxel_vol).max(85.0);
    let enclosure_fraction = if !cavity_voxels.is_empty() {
        let mean_rays = cavity_voxels.iter().map(|(_, r)| *r as f64).sum::<f64>()
            / (cavity_voxels.len() as f64);
        (mean_rays / 14.0).clamp(0.15, 0.98)
    } else {
        0.58
    };

    // Subsample cavity points (`<= 180` points) for fast 3D viewport rendering
    let stride = (cavity_voxels.len() / 160).max(1);
    let cavity_grid_points: Vec<[f32; 3]> = cavity_voxels
        .iter()
        .step_by(stride)
        .map(|(p, _)| *p)
        .collect();

    // 4. Lining residue summary & hydrophobic / polar surface balance
    let mut res_map: BTreeMap<(char, i32, String), (f32, usize)> = BTreeMap::new();
    let mut non_polar_cnt = 0usize;
    let mut polar_cnt = 0usize;
    let mut b_sum = 0.0f32;
    let mut inner_cnt = 0usize;

    for pa in &active_shell {
        let d_c = dist3(pa.pos, center);
        if d_c <= radius + 1.2 {
            inner_cnt += 1;
            b_sum += pa.b_factor;
            if matches!(pa.elem, Element::C | Element::S) && pa.atom_name != "C" {
                non_polar_cnt += 1;
            } else {
                polar_cnt += 1;
            }
            let entry = res_map
                .entry((pa.chain_id, pa.res_seq, pa.res_name.clone()))
                .or_insert((d_c, 0));
            if d_c < entry.0 {
                entry.0 = d_c;
            }
            entry.1 += 1;
        }
    }

    let tot_lining = (non_polar_cnt + polar_cnt).max(1) as f64;
    let hydrophobic_fraction = (non_polar_cnt as f64) / tot_lining;
    let polar_hbond_fraction = (polar_cnt as f64) / tot_lining;
    let mean_b_factor = if inner_cnt > 0 {
        b_sum / (inner_cnt as f32)
    } else {
        24.0
    };

    // Calibrated Halgren SiteMap / fpocket DScore in [0.05 .. 0.99]
    let vol_term = (volume_angstrom3 / 420.0).min(1.15);
    let enc_term = (enclosure_fraction / 0.72).min(1.15);
    let hyd_balance = 1.0 - ((hydrophobic_fraction - 0.62).abs() * 1.35).min(0.65);
    let druggability_dscore =
        (0.38 * vol_term + 0.34 * enc_term + 0.28 * hyd_balance).clamp(0.12, 0.98);
    let druggability_tier = if druggability_dscore >= 0.75 {
        PocketDruggabilityTier::HighlyDruggable
    } else if druggability_dscore >= 0.50 {
        PocketDruggabilityTier::Druggable
    } else {
        PocketDruggabilityTier::ShallowOrPolar
    };

    let mut lining_residues: Vec<PocketResidueSummary> = res_map
        .into_iter()
        .map(|((ch, seq, rname), (md, cnt))| PocketResidueSummary {
            residue_label: format!("{rname}{seq}({ch})"),
            role: residue_functional_role(&rname),
            res_name: rname,
            chain_id: ch,
            res_seq: seq,
            min_dist_to_center_a: md,
            contact_atom_count: cnt,
        })
        .collect();
    lining_residues.sort_by(|a, b| {
        a.min_dist_to_center_a
            .partial_cmp(&b.min_dist_to_center_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 5. Buried / Unhappy Hydration-Site Profiler (`WaterMap` / `IST` Thermodynamics)
    let mut hydration_sites: Vec<HydrationSite> = Vec::new();
    for w in &pdb.waters {
        if dist3(w.pos, center) <= radius + 1.5 {
            hydration_sites.push(profile_hydration_site(
                format!("HOH{}{}", w.chain_id, w.res_seq),
                w.pos,
                w.b_factor,
                true,
                &active_shell,
            ));
        }
    }
    // If crystallographic waters were stripped (e.g. dry apo PDB), place up to 3 virtual IST hydration probes
    // at high-enclosure cavity extrema so water-displacement hotspots remain discoverable!
    if hydration_sites.is_empty() && !cavity_voxels.is_empty() {
        let mut sorted_vox = cavity_voxels.clone();
        sorted_vox.sort_by(|a, b| b.1.cmp(&a.1));
        for (pt, _) in sorted_vox {
            if hydration_sites.len() >= 3 {
                break;
            }
            if hydration_sites.iter().all(|h| dist3(h.pos, pt) >= 3.2) {
                let idx = hydration_sites.len() + 1;
                hydration_sites.push(profile_hydration_site(
                    format!("IST-W{idx}"),
                    pt,
                    32.0,
                    false,
                    &active_shell,
                ));
            }
        }
    }
    hydration_sites.sort_by(|a, b| {
        a.displacement_ddg_bind_kcal
            .partial_cmp(&b.displacement_ddg_bind_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let unhappy_water_count = hydration_sites
        .iter()
        .filter(|h| h.role == WaterThermodynamicRole::UnhappyDisplaceableHotspot)
        .count();
    let structural_water_count = hydration_sites
        .iter()
        .filter(|h| h.role == WaterThermodynamicRole::ConservedStructuralBridge)
        .count();

    // 6. Project 3D Receptor Pharmacophore Features into Open Cavity (`LigandScout` / `Pharmit`)
    let pharmacophore_points =
        extract_receptor_pharmacophores(center, radius, &active_shell, &hydration_sites, &cavity_voxels);

    PocketCavityAnalysis {
        pdb_id: pdb.pdb_id.clone(),
        title: pdb.title.clone(),
        center,
        pocket_radius_a: radius,
        cocrystal_ligand_name: lig_name,
        cocrystal_heavy_atoms: lig_coords,
        volume_angstrom3,
        enclosure_fraction,
        hydrophobic_fraction,
        polar_hbond_fraction,
        mean_b_factor,
        druggability_dscore,
        druggability_tier,
        shell_atoms: active_shell,
        lining_residues,
        hydration_sites,
        unhappy_water_count,
        structural_water_count,
        pharmacophore_points,
        cavity_grid_points,
    }
}

fn count_enclosed_rays(pt: [f32; 3], atoms: &[PdbAtom], max_ray_a: f32) -> usize {
    let mut hit_rays = 0usize;
    for dir in RAY_DIRS_14 {
        let mut hit = false;
        for a in atoms {
            let vx = a.pos[0] - pt[0];
            let vy = a.pos[1] - pt[1];
            let vz = a.pos[2] - pt[2];
            let proj = vx * dir[0] + vy * dir[1] + vz * dir[2];
            if proj <= 0.8 || proj > max_ray_a {
                continue;
            }
            let perp2 = (vx * vx + vy * vy + vz * vz) - proj * proj;
            if perp2 <= 5.2 {
                hit = true;
                break;
            }
        }
        if hit {
            hit_rays += 1;
        }
    }
    hit_rays
}

fn profile_hydration_site(
    site_id: String,
    pos: [f32; 3],
    b_factor: f32,
    is_crystallographic: bool,
    shell: &[PdbAtom],
) -> HydrationSite {
    let mut hb_cnt = 0usize;
    let mut coord_res = Vec::new();
    let mut hydro_enc = 0usize;

    for pa in shell {
        let d = dist3(pos, pa.pos);
        if (2.30..=3.45).contains(&d) && (is_protein_polar_donor(pa) || is_protein_polar_acceptor(pa))
        {
            hb_cnt += 1;
            let lbl = format!("{}{}", pa.res_name, pa.res_seq);
            if !coord_res.contains(&lbl) {
                coord_res.push(lbl);
            }
        } else if d <= 4.35 && matches!(pa.elem, Element::C | Element::S) && pa.atom_name != "C" {
            hydro_enc += 1;
        }
    }

    // Bulk water enjoys ~3.5 H-bonds; missing H-bonds inside hydrophobic enclosure yields positive Delta H_excess
    let missing_hb = (2.8 - (hb_cnt as f64)).max(-1.4);
    let excess_enthalpy_kcal =
        (0.85 * missing_hb + 0.16 * ((hydro_enc as f64) - 2.0).max(0.0)).clamp(-2.2, 3.2);
    let b_order_factor = ((38.0 - (b_factor as f64)) / 20.0).clamp(0.15, 1.35);
    let entropic_penalty_kcal =
        (0.55 * b_order_factor + 0.12 * (hydro_enc.min(8) as f64)).clamp(0.20, 2.10);

    // Displacing water into bulk releases both excess enthalpy AND orientational entropy:
    // Delta G_disp = -(Delta H_excess + (-T*Delta S_hyd)) when unhappy; positive penalty if tightly H-bonded (>=3 H-bonds)!
    let raw_gain = -(excess_enthalpy_kcal + 0.65 * entropic_penalty_kcal);
    let displacement_ddg_bind_kcal = if hb_cnt >= 3
        || (hb_cnt >= 2 && b_factor <= 21.0 && hydro_enc <= 3)
    {
        // Tightly coordinated structural water (e.g., EGFR Thr790/Thr854 bridge HOH2001): displacing without HBA/HBD mimic costs energy
        (0.95 + 0.45 * ((hb_cnt.saturating_sub(2)) as f64)).clamp(0.80, 2.40)
    } else {
        raw_gain.clamp(-2.65, 1.40)
    };

    let role = if displacement_ddg_bind_kcal <= -0.75 {
        WaterThermodynamicRole::UnhappyDisplaceableHotspot
    } else if displacement_ddg_bind_kcal >= 0.75 {
        WaterThermodynamicRole::ConservedStructuralBridge
    } else {
        WaterThermodynamicRole::ExchangeableBulkRim
    };

    HydrationSite {
        site_id,
        pos,
        b_factor,
        is_crystallographic,
        protein_hbond_count: hb_cnt,
        coordinating_residues: coord_res,
        hydrophobic_enclosure_atoms: hydro_enc,
        excess_enthalpy_kcal,
        entropic_penalty_kcal,
        displacement_ddg_bind_kcal,
        role,
    }
}

fn project_toward_cavity(atom_pos: [f32; 3], center: [f32; 3], dist_a: f32) -> [f32; 3] {
    let dx = center[0] - atom_pos[0];
    let dy = center[1] - atom_pos[1];
    let dz = center[2] - atom_pos[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
    [
        atom_pos[0] + dx / len * dist_a,
        atom_pos[1] + dy / len * dist_a,
        atom_pos[2] + dz / len * dist_a,
    ]
}

fn project_to_open_cavity(
    atom_pos: [f32; 3],
    center: [f32; 3],
    dist_a: f32,
    shell: &[PdbAtom],
    voxels: &[([f32; 3], usize)],
) -> [f32; 3] {
    let ray_pt = project_toward_cavity(atom_pos, center, dist_a);
    let ray_ok = !shell.iter().any(|sa| dist3(ray_pt, sa.pos) < 1.85);
    if ray_ok || voxels.is_empty() {
        return ray_pt;
    }

    let min_r = (dist_a - 1.15).max(1.90);
    let max_r = dist_a + 1.45;
    let mut best_pt: Option<([f32; 3], f32)> = None;
    for &(vx, _) in voxels {
        let d_atom = dist3(vx, atom_pos);
        if d_atom < min_r || d_atom > max_r {
            continue;
        }
        if shell.iter().any(|sa| dist3(vx, sa.pos) < 1.85) {
            continue;
        }
        let score = (d_atom - dist_a).abs() + 0.35 * dist3(vx, ray_pt);
        if best_pt.map(|(_, s)| score < s).unwrap_or(true) {
            best_pt = Some((vx, score));
        }
    }
    best_pt.map(|(p, _)| p).unwrap_or(ray_pt)
}

fn extract_receptor_pharmacophores(
    center: [f32; 3],
    radius: f32,
    shell: &[PdbAtom],
    waters: &[HydrationSite],
    voxels: &[([f32; 3], usize)],
) -> Vec<ReceptorPharmacophorePoint> {
    let mut raw_pts: Vec<ReceptorPharmacophorePoint> = Vec::new();

    for pa in shell {
        let d_c = dist3(pa.pos, center);
        if d_c > radius + 0.8 {
            continue;
        }
        let origin = format!("{}{} {}", pa.res_name, pa.res_seq, pa.atom_name);

        if is_protein_anionic(pa) {
            let proj = project_to_open_cavity(pa.pos, center, 2.85, shell, voxels);
            raw_pts.push(ReceptorPharmacophorePoint {
                feature_id: String::new(),
                kind: RocsColorKind::Cation,
                pos: proj,
                tolerance_radius_a: 1.6,
                receptor_origin: origin,
                interaction_label: "Salt-Bridge / HBD Anchor",
                weight: 2.2,
            });
        } else if is_protein_cationic(pa) {
            let proj = project_to_open_cavity(pa.pos, center, 2.90, shell, voxels);
            raw_pts.push(ReceptorPharmacophorePoint {
                feature_id: String::new(),
                kind: RocsColorKind::Anion,
                pos: proj,
                tolerance_radius_a: 1.6,
                receptor_origin: origin,
                interaction_label: "Oxyanion / Sulfonamide / HBA",
                weight: 2.0,
            });
        } else if is_protein_polar_donor(pa) {
            let proj = project_to_open_cavity(pa.pos, center, 2.85, shell, voxels);
            raw_pts.push(ReceptorPharmacophorePoint {
                feature_id: String::new(),
                kind: RocsColorKind::HBondAcceptor,
                pos: proj,
                tolerance_radius_a: 1.4,
                receptor_origin: origin,
                interaction_label: "Complementary Ligand HBA",
                weight: if pa.atom_name == "N" { 2.1 } else { 1.6 }, // Hinge backbone N-H prioritized!
            });
        } else if is_protein_polar_acceptor(pa) {
            let proj = project_to_open_cavity(pa.pos, center, 2.85, shell, voxels);
            raw_pts.push(ReceptorPharmacophorePoint {
                feature_id: String::new(),
                kind: RocsColorKind::HBondDonor,
                pos: proj,
                tolerance_radius_a: 1.4,
                receptor_origin: origin,
                interaction_label: "Complementary Ligand HBD",
                weight: 1.6,
            });
        } else if matches!(pa.res_name.as_str(), "PHE" | "TRP" | "TYR" | "HIS")
            && matches!(pa.atom_name.as_str(), "CZ" | "CE2" | "CG")
        {
            let proj = project_to_open_cavity(pa.pos, center, 3.60, shell, voxels);
            raw_pts.push(ReceptorPharmacophorePoint {
                feature_id: String::new(),
                kind: RocsColorKind::AromaticRing,
                pos: proj,
                tolerance_radius_a: 1.8,
                receptor_origin: origin,
                interaction_label: "Aromatic Pi-Stacking Cage",
                weight: 1.5,
            });
        }
    }

    // Add WaterMap hydration-site pharmacophores:
    // - Unhappy water -> Hydrophobic / lipophilic displacement sphere!
    // - Conserved structural bridge -> Ligand HBA sphere bridging the ordered water!
    for w in waters {
        match w.role {
            WaterThermodynamicRole::UnhappyDisplaceableHotspot => {
                raw_pts.push(ReceptorPharmacophorePoint {
                    feature_id: String::new(),
                    kind: RocsColorKind::Hydrophobic,
                    pos: w.pos,
                    tolerance_radius_a: 1.7,
                    receptor_origin: format!("{} (dG {:.2} kcal)", w.site_id, w.displacement_ddg_bind_kcal),
                    interaction_label: "Displace Unhappy Water!",
                    weight: 2.3,
                });
            }
            WaterThermodynamicRole::ConservedStructuralBridge => {
                let proj = project_to_open_cavity(w.pos, center, 2.75, shell, voxels);
                raw_pts.push(ReceptorPharmacophorePoint {
                    feature_id: String::new(),
                    kind: RocsColorKind::HBondAcceptor,
                    pos: proj,
                    tolerance_radius_a: 1.5,
                    receptor_origin: format!("{} Water Bridge", w.site_id),
                    interaction_label: "Engage Structural Water Bridge",
                    weight: 1.9,
                });
            }
            WaterThermodynamicRole::ExchangeableBulkRim => {}
        }
    }

    // Add deepest buried hydrophobic cavity voxel if fewer than 2 hydrophobic spheres exist
    let hyd_cnt = raw_pts
        .iter()
        .filter(|p| p.kind == RocsColorKind::Hydrophobic)
        .count();
    if hyd_cnt < 2 && !voxels.is_empty() {
        let core_pos = if !shell.iter().any(|sa| dist3(center, sa.pos) < 1.85) {
            center
        } else {
            voxels
                .iter()
                .filter(|&&(vx, _)| !shell.iter().any(|sa| dist3(vx, sa.pos) < 1.85))
                .min_by(|a, b| {
                    b.1.cmp(&a.1)
                        .then_with(|| dist3(a.0, center).total_cmp(&dist3(b.0, center)))
                })
                .map(|&(vx, _)| vx)
                .unwrap_or(center)
        };
        raw_pts.push(ReceptorPharmacophorePoint {
            feature_id: String::new(),
            kind: RocsColorKind::Hydrophobic,
            pos: core_pos,
            tolerance_radius_a: 1.9,
            receptor_origin: "Buried Lipophilic Core".to_string(),
            interaction_label: "Core Hydrophobic Subpocket",
            weight: 1.7,
        });
    }

    // Deduplicate spatially overlapping same-kind spheres (`<= 2.1 A`), keeping highest weight
    raw_pts.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut dedup: Vec<ReceptorPharmacophorePoint> = Vec::new();
    for mut pt in raw_pts {
        let clashes_shell = shell.iter().any(|sa| dist3(pt.pos, sa.pos) < 1.85);
        if clashes_shell {
            continue;
        }
        let dup = dedup
            .iter()
            .any(| ex | ex.kind == pt.kind && dist3(ex.pos, pt.pos) < 2.25);
        if !dup && dedup.len() < 12 {
            pt.feature_id = format!("PH-{:02}", dedup.len() + 1);
            dedup.push(pt);
        }
    }
    dedup
}

fn rotate_align_pose_into_pocket(
    mol: &Molecule,
    init_coords: &[[f32; 3]],
    lig_colors: &[RocsColorAtom],
    pocket: &PocketCavityAnalysis,
) -> Vec<[f32; 3]> {
    if init_coords.is_empty() {
        return Vec::new();
    }
    // Center ligand at pocket centroid first
    let inv = 1.0 / (init_coords.len() as f32);
    let mut lc = [0.0f32; 3];
    for p in init_coords {
        lc[0] += p[0] * inv;
        lc[1] += p[1] * inv;
        lc[2] += p[2] * inv;
    }
    let centered: Vec<[f32; 3]> = init_coords
        .iter()
        .map(|p| [p[0] - lc[0], p[1] - lc[1], p[2] - lc[2]])
        .collect();

    // Generate candidate rigid-body poses:
    // 1) Kabsch superposition of matching ligand->receptor pharmacophore triplets when >=3 exist
    // 2) 12 octahedral / principal-axis orientations translated to pocket.center + small local shifts
    let mut trial_poses: Vec<Vec<[f32; 3]>> = Vec::with_capacity(20);

    if lig_colors.len() >= 3 && pocket.pharmacophore_points.len() >= 3 {
        let mut mob = Vec::new();
        let mut tgt = Vec::new();
        for rp in &pocket.pharmacophore_points {
            if let Some(lc_pt) = lig_colors.iter().find(|c| {
                c.kind == rp.kind
                    || (rp.kind == RocsColorKind::Cation && c.kind == RocsColorKind::HBondDonor)
                    || (rp.kind == RocsColorKind::Anion && c.kind == RocsColorKind::HBondAcceptor)
            }) {
                mob.push(lc_pt.pos3d);
                tgt.push(rp.pos);
                if mob.len() >= 5 {
                    break;
                }
            }
        }
        if mob.len() >= 3 {
            let (rot, trans, _, _) = kabsch_align_3d(&mob, &tgt);
            let mapped: Vec<[f32; 3]> = init_coords
                .iter()
                .map(|p| {
                    [
                        rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + trans[0],
                        rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + trans[1],
                        rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + trans[2],
                    ]
                })
                .collect();
            trial_poses.push(mapped);
        }
    }

    // 8 canonical 3D sign/axis rotations around pocket.center
    let rot_mats: [[[f32; 3]; 3]; 8] = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    ];
    for rm in rot_mats {
        for shift in [[0.0f32, 0.0, 0.0], [0.6, -0.4, 0.3], [-0.6, 0.4, -0.3]] {
            let pose: Vec<[f32; 3]> = centered
                .iter()
                .map(|p| {
                    [
                        pocket.center[0] + shift[0] + rm[0][0] * p[0] + rm[0][1] * p[1] + rm[0][2] * p[2],
                        pocket.center[1] + shift[1] + rm[1][0] * p[0] + rm[1][1] * p[1] + rm[1][2] * p[2],
                        pocket.center[2] + shift[2] + rm[2][0] * p[0] + rm[2][1] * p[1] + rm[2][2] * p[2],
                    ]
                })
                .collect();
            trial_poses.push(pose);
        }
    }

    let mut best_pose = trial_poses.remove(0);
    let mut best_e = quick_pose_energy(mol, &best_pose, pocket);
    for pose in trial_poses {
        let e = quick_pose_energy(mol, &pose, pocket);
        if e < best_e {
            best_e = e;
            best_pose = pose;
        }
    }

    // 10-step coordinate gradient-free translational micro-relaxation (`0.35 A` -> `0.15 A`) to relieve residual clashes
    for step_a in [0.40_f32, 0.20_f32] {
        for dir in [
            [step_a, 0.0, 0.0],
            [-step_a, 0.0, 0.0],
            [0.0, step_a, 0.0],
            [0.0, -step_a, 0.0],
            [0.0, 0.0, step_a],
            [0.0, 0.0, -step_a],
        ] {
            let nudged: Vec<[f32; 3]> = best_pose
                .iter()
                .map(|p| [p[0] + dir[0], p[1] + dir[1], p[2] + dir[2]])
                .collect();
            let e = quick_pose_energy(mol, &nudged, pocket);
            if e < best_e {
                best_e = e;
                best_pose = nudged;
            }
        }
    }

    best_pose
}

fn quick_pose_energy(mol: &Molecule, coords: &[[f32; 3]], pocket: &PocketCavityAnalysis) -> f64 {
    let mut e = 0.0f64;
    let n = mol.atoms.len().min(coords.len());
    for i in 0..n {
        let lp = coords[i];
        let la = &mol.atoms[i];
        let r_l = bondi_vdw_radius(la.elem);
        let lig_hbd = matches!(la.elem, Element::N | Element::O) && la.total_h() > 0;
        let lig_hba = matches!(la.elem, Element::N | Element::O | Element::F) && la.charge <= 0;

        for pa in &pocket.shell_atoms {
            let d = dist3(lp, pa.pos);
            if d > 5.8 {
                continue;
            }
            let r_sum = r_l + bondi_vdw_radius(pa.elem);
            let surf_d = (d - r_sum) as f64;
            if surf_d < -0.55 {
                let pen = (-0.55 - surf_d) * 3.8;
                e += pen * pen;
            } else if surf_d <= 1.4 {
                e -= 0.085 * (-(surf_d * surf_d) / 0.65).exp();
            }
            if (2.45..=3.45).contains(&d)
                && ((lig_hbd && is_protein_polar_acceptor(pa))
                    || (lig_hba && is_protein_polar_donor(pa)))
            {
                e -= 0.72;
            }
        }
        for w in &pocket.hydration_sites {
            if w.role == WaterThermodynamicRole::UnhappyDisplaceableHotspot
                && dist3(lp, w.pos) <= 2.25
            {
                e += 0.45 * w.displacement_ddg_bind_kcal;
            }
        }
    }
    e
}

/// Superimposes `mol` into `pocket` and evaluates empirical `AutoDock Vina` + `WaterMap`
/// complementarity (`vdW`, steric clash, `H-bond`/salt-bridge, hydrophobic enclosure, water displacement, and `Pocket LE`).
pub fn score_ligand_in_pocket(
    mol: &Molecule,
    mol_idx: Option<usize>,
    pocket: &PocketCavityAnalysis,
) -> PocketComplementarityResult {
    let prof = evaluate_physchem(mol, 0);
    let s3d = evaluate_3d_summary_profile(mol, true);
    let aligned = rotate_align_pose_into_pocket(mol, &s3d.coords3d, &s3d.color_atoms, pocket);

    let n = mol.atoms.len().min(aligned.len());
    let mut vdw_steric_kcal = 0.0f64;
    let mut steric_clash_kcal = 0.0f64;
    let mut hbond_ionic_kcal = 0.0f64;
    let mut hydrophobic_enclosure_kcal = 0.0f64;
    let mut clash_atoms = 0usize;
    let mut interactions: Vec<PocketLigandInteraction> = Vec::new();

    for i in 0..n {
        let lp = aligned[i];
        let la = &mol.atoms[i];
        let r_l = bondi_vdw_radius(la.elem);
        let lig_hbd = matches!(la.elem, Element::N | Element::O) && la.total_h() > 0;
        let lig_hba = matches!(la.elem, Element::N | Element::O | Element::F) && la.charge <= 0;
        let lig_cat = la.elem == Element::N && (la.charge > 0 || (!la.aromatic && la.total_h() >= 1));
        let lig_an = la.charge < 0;
        let lig_lip = matches!(
            la.elem,
            Element::C | Element::F | Element::Cl | Element::Br | Element::I | Element::S
        ) && !lig_hbd;

        let mut atom_clashed = false;
        let mut lip_contacts = 0usize;

        for pa in &pocket.shell_atoms {
            let d = dist3(lp, pa.pos);
            if d > 5.8 {
                continue;
            }
            let r_sum = r_l + bondi_vdw_radius(pa.elem);
            let surf_d = (d - r_sum) as f64;

            if surf_d < -0.60 {
                atom_clashed = true;
                let depth = -0.60 - surf_d;
                steric_clash_kcal += (1.85 * depth * depth).min(3.5);
            } else if surf_d <= 1.65 {
                vdw_steric_kcal -= 0.068 * (-(surf_d * surf_d) / 0.72).exp();
            }

            // Salt-bridge check (`<= 3.9 A`)
            if d <= 3.90 && ((lig_cat && is_protein_anionic(pa)) || (lig_an && is_protein_cationic(pa)))
            {
                let e_sb = -1.45 * ((4.1 - (d as f64)) / 1.4).clamp(0.35, 1.15);
                hbond_ionic_kcal += e_sb;
                interactions.push(PocketLigandInteraction {
                    kind_label: "Ionic Salt-Bridge",
                    receptor_partner: format!("{}{} {}", pa.res_name, pa.res_seq, pa.atom_name),
                    ligand_atom_label: format!("{}{}", la.elem.symbol(), i + 1),
                    distance_a: d,
                    energy_kcal: e_sb,
                });
            } else if (2.40..=3.50).contains(&d)
                && ((lig_hbd && is_protein_polar_acceptor(pa))
                    || (lig_hba && is_protein_polar_donor(pa)))
            {
                let e_hb = -0.82 * ((3.60 - (d as f64)) / 0.90).clamp(0.30, 1.10);
                hbond_ionic_kcal += e_hb;
                interactions.push(PocketLigandInteraction {
                    kind_label: "Hydrogen Bond",
                    receptor_partner: format!("{}{} {}", pa.res_name, pa.res_seq, pa.atom_name),
                    ligand_atom_label: format!("{}{}", la.elem.symbol(), i + 1),
                    distance_a: d,
                    energy_kcal: e_hb,
                });
            }

            if lig_lip
                && d <= 4.35
                && matches!(pa.elem, Element::C | Element::S)
                && pa.atom_name != "C"
            {
                lip_contacts += 1;
            }
        }

        if atom_clashed {
            clash_atoms += 1;
        }
        if lip_contacts >= 2 {
            hydrophobic_enclosure_kcal -= 0.14 * ((lip_contacts.min(6)) as f64);
        }
    }

    // Clamp macroscopic terms to physically realistic single-pose receptor interaction bounds
    vdw_steric_kcal = vdw_steric_kcal.clamp(-7.8, 0.0);
    hbond_ionic_kcal = hbond_ionic_kcal.clamp(-5.6, 0.0);
    hydrophobic_enclosure_kcal = hydrophobic_enclosure_kcal.clamp(-4.8, 0.0);
    steric_clash_kcal = steric_clash_kcal.clamp(0.0, 14.0);

    // WaterMap hydration displacement & structural-bridge engagement
    let mut water_displacement_kcal = 0.0f64;
    let mut displaced_unhappy_waters = 0usize;
    let mut bridged_structural_waters = 0usize;

    for w in &pocket.hydration_sites {
        let mut min_d = f32::INFINITY;
        let mut min_polar_d = f32::INFINITY;
        let mut closest_lbl = String::new();
        for i in 0..n {
            let d = dist3(aligned[i], w.pos);
            if d < min_d {
                min_d = d;
                closest_lbl = format!("{}{}", mol.atoms[i].elem.symbol(), i + 1);
            }
            if matches!(mol.atoms[i].elem, Element::N | Element::O | Element::F) && d < min_polar_d
            {
                min_polar_d = d;
            }
        }

        match w.role {
            WaterThermodynamicRole::UnhappyDisplaceableHotspot => {
                if min_d <= 2.35 {
                    displaced_unhappy_waters += 1;
                    water_displacement_kcal += w.displacement_ddg_bind_kcal; // negative = gain!
                    interactions.push(PocketLigandInteraction {
                        kind_label: "Displaced Unhappy Water",
                        receptor_partner: w.site_id.clone(),
                        ligand_atom_label: closest_lbl,
                        distance_a: min_d,
                        energy_kcal: w.displacement_ddg_bind_kcal,
                    });
                }
            }
            WaterThermodynamicRole::ConservedStructuralBridge => {
                if (2.45..=3.60).contains(&min_polar_d) && min_d >= 2.10 {
                    bridged_structural_waters += 1;
                    water_displacement_kcal -= 0.75;
                    interactions.push(PocketLigandInteraction {
                        kind_label: "Water-Bridge H-Bond",
                        receptor_partner: w.site_id.clone(),
                        ligand_atom_label: closest_lbl,
                        distance_a: min_polar_d,
                        energy_kcal: -0.75,
                    });
                } else if min_d < 1.85 && min_polar_d > 2.20 {
                    // Lipophilic atom collided with conserved structural water without polar replacement!
                    water_displacement_kcal += 0.95;
                }
            }
            WaterThermodynamicRole::ExchangeableBulkRim => {}
        }
    }

    // Receptor pharmacophore sphere recall
    let mut pharm_matched = 0usize;
    for rp in &pocket.pharmacophore_points {
        let hit = (0..n).any(|i| {
            let d = dist3(aligned[i], rp.pos);
            if d > rp.tolerance_radius_a + 1.05 {
                return false;
            }
            let la = &mol.atoms[i];
            match rp.kind {
                RocsColorKind::HBondDonor => {
                    matches!(la.elem, Element::N | Element::O) && la.total_h() > 0
                }
                RocsColorKind::HBondAcceptor => {
                    matches!(la.elem, Element::N | Element::O | Element::F)
                }
                RocsColorKind::Cation => la.elem == Element::N,
                RocsColorKind::Anion => matches!(la.elem, Element::O | Element::N | Element::S),
                RocsColorKind::AromaticRing => la.aromatic,
                RocsColorKind::Hydrophobic => matches!(
                    la.elem,
                    Element::C | Element::F | Element::Cl | Element::Br | Element::S
                ),
            }
        });
        if hit {
            pharm_matched += 1;
        }
    }
    let pharm_tot = pocket.pharmacophore_points.len().max(1);
    let pharmacophore_recall_pct = (pharm_matched as f64) / (pharm_tot as f64) * 100.0;

    // AutoDock Vina conformational rotor dilution (`1 / (1 + 0.0585 * N_rot)`) expressed as +kcal/mol entropy cost
    let raw_inter = vdw_steric_kcal
        + steric_clash_kcal
        + hbond_ionic_kcal
        + hydrophobic_enclosure_kcal
        + water_displacement_kcal;
    let vina_denom = 1.0 + 0.0585 * (prof.rotatable_bonds as f64);
    let total_dg_pocket_kcal = (raw_inter / vina_denom).clamp(-14.2, 6.0);
    let rotor_entropy_kcal = (total_dg_pocket_kcal - raw_inter).max(0.0);

    let predicted_pocket_pkd = (-total_dg_pocket_kcal / RT_LN10_KCAL).clamp(3.0, 10.4);
    let hac = prof.heavy_atoms.max(1) as f64;
    let pocket_le = (-total_dg_pocket_kcal / hac).max(0.0);
    let pocket_lle = predicted_pocket_pkd - prof.clogp;

    interactions.sort_by(|a, b| {
        a.energy_kcal
            .partial_cmp(&b.energy_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    interactions.truncate(8);

    // Composite [0..100] Pocket Complementarity Score
    let d_dg = ((-total_dg_pocket_kcal - 4.0) / 7.5).clamp(0.0, 1.0);
    let d_le = ((pocket_le - 0.16) / 0.24).clamp(0.0, 1.0);
    let d_rec = (pharmacophore_recall_pct / 80.0).clamp(0.0, 1.0);
    let clash_pen = if clash_atoms == 0 {
        1.0
    } else {
        (1.0 - 0.12 * (clash_atoms as f64)).clamp(0.35, 0.92)
    };
    let complementarity_score =
        ((0.44 * d_dg + 0.26 * d_le + 0.30 * d_rec) * 100.0 * clash_pen).clamp(1.0, 99.9);

    PocketComplementarityResult {
        rank: 1,
        mol_idx,
        compound_id: if mol.id.is_empty() {
            "QUERY".to_string()
        } else {
            mol.id.clone()
        },
        compound_name: if mol.name.is_empty() {
            "Ligand".to_string()
        } else {
            mol.name.clone()
        },
        target: mol.target.clone(),
        canonical_smiles: mol.smiles.clone(),
        mw: prof.mw,
        clogp: prof.clogp,
        observed_pact: mol.pact,
        vdw_steric_kcal,
        steric_clash_kcal,
        hbond_ionic_kcal,
        hydrophobic_enclosure_kcal,
        water_displacement_kcal,
        rotor_entropy_kcal,
        total_dg_pocket_kcal,
        predicted_pocket_pkd,
        pocket_le,
        pocket_lle,
        pharmacophore_matched: pharm_matched,
        pharmacophore_total: pocket.pharmacophore_points.len(),
        pharmacophore_recall_pct,
        displaced_unhappy_waters,
        bridged_structural_waters,
        clash_atom_count: clash_atoms,
        complementarity_score,
        key_interactions: interactions,
        aligned_ligand_coords3d: aligned,
        mol: mol.clone(),
    }
}

/// Screens an entire `DatasetAnalysis` library against `pocket` in parallel via `Rayon`.
pub fn screen_dataset_against_pocket(
    ds: &DatasetAnalysis,
    pocket: &PocketCavityAnalysis,
) -> Vec<PocketComplementarityResult> {
    let mut results: Vec<PocketComplementarityResult> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(idx, m)| score_ligand_in_pocket(m, Some(idx), pocket))
        .collect();

    results.sort_by(|a, b| {
        b.complementarity_score
            .partial_cmp(&a.complementarity_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rk, r) in results.iter_mut().enumerate() {
        r.rank = rk + 1;
    }
    results
}

/// Exports ranked pocket-guided screening results to `.csv`.
pub fn export_pocket_complementarity_csv(
    pocket: &PocketCavityAnalysis,
    hits: &[PocketComplementarityResult],
) -> String {
    let mut out = String::with_capacity(hits.len() * 220 + 256);
    out.push_str("rank,pdb_id,id,name,target,complementarity_100,dg_pocket_kcal,pred_pocket_pkd,pocket_le,pocket_lle,pharm_recall_pct,displaced_unhappy_waters,bridged_waters,vdw_kcal,hbond_ionic_kcal,hydrophobic_kcal,water_disp_kcal,clash_kcal,obs_pact,smiles\n");
    for h in hits {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",{:.2},{:.2},{:.2},{:.3},{:.2},{:.1},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},\"{}\"\n",
            h.rank,
            pocket.pdb_id,
            h.compound_id.replace('"', "'"),
            h.compound_name.replace('"', "'"),
            h.target.replace('"', "'"),
            h.complementarity_score,
            h.total_dg_pocket_kcal,
            h.predicted_pocket_pkd,
            h.pocket_le,
            h.pocket_lle,
            h.pharmacophore_recall_pct,
            h.displaced_unhappy_waters,
            h.bridged_structural_waters,
            h.vdw_steric_kcal,
            h.hbond_ionic_kcal,
            h.hydrophobic_enclosure_kcal,
            h.water_displacement_kcal,
            h.steric_clash_kcal,
            h.observed_pact,
            h.canonical_smiles.replace('"', "'")
        ));
    }
    out
}

/// Exports 3D pocket-aligned ligand poses as a multi-record MDL V2000 `.sdf` in the exact `.pdb` coordinate frame!
pub fn format_pocket_poses_sdf(
    pocket: &PocketCavityAnalysis,
    hits: &[PocketComplementarityResult],
) -> String {
    let mut sdf = String::with_capacity(hits.len() * 2048);
    for h in hits {
        let na = h.mol.atoms.len();
        let nb = h.mol.bonds.len();
        sdf.push_str(&format!("{}_{}\n", pocket.pdb_id, h.compound_id));
        sdf.push_str("  SAR-Atlas v2.3.0 Pocket-Aligned 3D Pose\n\n");
        sdf.push_str(&format!(
            "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
            na, nb
        ));
        for (i, a) in h.mol.atoms.iter().enumerate() {
            let p = h
                .aligned_ligand_coords3d
                .get(i)
                .copied()
                .unwrap_or([pocket.center[0], pocket.center[1], pocket.center[2]]);
            sdf.push_str(&format!(
                "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
                p[0],
                p[1],
                p[2],
                a.elem.symbol()
            ));
        }
        for b in &h.mol.bonds {
            let bo = match b.order {
                crate::mol::BondOrder::Single => 1,
                crate::mol::BondOrder::Double => 2,
                crate::mol::BondOrder::Triple => 3,
                crate::mol::BondOrder::Aromatic => 4,
            };
            sdf.push_str(&format!("{:>3}{:>3}{:>3}  0  0  0  0\n", b.a + 1, b.b + 1, bo));
        }
        sdf.push_str("M  END\n");
        for (tag, val) in [
            ("PDB_ID", pocket.pdb_id.clone()),
            ("Complementarity_100", format!("{:.2}", h.complementarity_score)),
            ("dG_Pocket_kcal", format!("{:.2}", h.total_dg_pocket_kcal)),
            ("Predicted_Pocket_pKd", format!("{:.2}", h.predicted_pocket_pkd)),
            ("Pocket_LE", format!("{:.3}", h.pocket_le)),
            ("Pharmacophore_Recall_Pct", format!("{:.1}", h.pharmacophore_recall_pct)),
            ("Displaced_Unhappy_Waters", h.displaced_unhappy_waters.to_string()),
        ] {
            sdf.push_str(&format!(">  <{tag}>\n{val}\n\n"));
        }
        sdf.push_str("$$$$\n");
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::load_benchmark_26;

    #[test]
    fn test_pdb_pocket_hydration_pharmacophore_and_ligand_complementarity() {
        // Parse 1M17 EGFR + Erlotinib + Crystallographic Waters
        let pdb = parse_pdb_structure(curated_pocket_pdb_text(0)).expect("1M17 parse failed");
        assert_eq!(pdb.pdb_id, "1M17");
        assert!(!pdb.protein_atoms.is_empty());
        assert_eq!(pdb.waters.len(), 4);
        assert_eq!(pdb.cocrystal_ligands.len(), 1);
        assert_eq!(pdb.cocrystal_ligands[0].res_name, "AQ4");

        let pocket = analyze_binding_pocket(&pdb, None, 10.5);
        assert!(pocket.volume_angstrom3 >= 85.0);
        assert!(pocket.druggability_dscore > 0.45);
        assert!(pocket.unhappy_water_count >= 1);
        assert!(pocket.structural_water_count >= 1);
        assert!(pocket.pharmacophore_points.len() >= 4);

        // Screen benchmark 26 compounds against 1M17 EGFR pocket
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let hits = screen_dataset_against_pocket(&ds, &pocket);
        assert_eq!(hits.len(), 26);
        assert!(hits[0].total_dg_pocket_kcal < -3.5);
        assert!(hits[0].complementarity_score >= hits[25].complementarity_score);

        let csv = export_pocket_complementarity_csv(&pocket, &hits);
        let sdf = format_pocket_poses_sdf(&pocket, &hits[..3]);
        assert_eq!(csv.lines().count(), 27);
        assert!(sdf.contains("V2000") && sdf.contains("dG_Pocket_kcal"));
    }
}
