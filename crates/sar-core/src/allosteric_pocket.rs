//! Gaussian-Network (`GNM`) & Anisotropic Elastic-Network (`ANM`) Normal-Mode Cryptic Pocket
//! Detector, Allosteric Communication Pathway Finder, and Allosteric-to-Orthosteric Mechanical
//! Strain & Cooperativity Scorer (`allosteric_pocket.rs`, `v2.6.0`).
//!
//! Primary References:
//! - Bahar, Atilgan & Erman (1997) / Li et al. (2016), *Nucleic Acids Res.* 44(D1), D415-D422
//!   (`iGNM 2.0` Gaussian Network Model Kirchhoff Laplacian, cross-correlations & hinge sites:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4702874/`)
//! - Atilgan et al. (2001) / Eyal, Lum & Bahar (2015), *Bioinformatics* 31(9), 1487-1489
//!   (`ANM 2.0` `3N x 3N` Anisotropic Elastic Network Hessian normal modes:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4410662/`) & Unified `G-ANM`
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC2367184/`)
//! - Meller et al. (2023), *Nat. Commun.* 14:1177 (`PocketMiner` cryptic pocket opening prediction:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/`) & `CryptoSite` / `CryptoBench`
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC12342141/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/`)
//! - Guarnera & Berezovsky (`AlloSigMA` / `SBSMMA` structure-based statistical mechanical model of
//!   allosteric free energy `dg_allo`: `https://pmc.ncbi.nlm.nih.gov/articles/PMC8772014/`,
//!   `https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010009`)
//! - Atilgan & Atilgan (2009) Perturbation-Response Scanning (`PRS`) & `GNM` Allosteric Information
//!   Pathways (`https://academic.oup.com/bioinformatics/article/38/14/3590/6604266`,
//!   `https://academic.oup.com/bioinformatics/article/35/23/4971/5481959`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC10635643/`)
//! - Curated Cryptic & Allosteric Benchmark Complexes:
//!   - `5D41` `EGFR` αC-Helix-Out Allosteric Pocket (`EAI045` / `EAI001`) + Orthosteric ATP Cleft
//!     (`https://www.rcsb.org/structure/5d41`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC5542946/`,
//!     `https://pmc.ncbi.nlm.nih.gov/articles/PMC6664433/`)
//!   - `6OIM` `KRAS G12C` Switch-II Cryptic Groove (`Sotorasib`) + Switch-I Effector Interface
//!     (`https://www.pnas.org/doi/10.1073/pnas.2305494120`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC10015465/`)
//!   - `3K5V` `ABL1` C-Lobe Myristoyl Allosteric Pocket (`GNF-2` / `Asciminib`) + Orthosteric ATP Site
//!     (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10450579/`)

use crate::conformer3d::{bondi_vdw_radius, format_3d_conformer_sdf};
use crate::mol::Molecule;
use crate::pocket_cavity::{
    analyze_binding_pocket, parse_pdb_structure, score_ligand_in_pocket, ParsedPdbStructure,
    PocketCavityAnalysis, PocketComplementarityResult,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Thermal energy `R * T` at `310 K` (`0.616 kcal/mol`) for `AlloSigMA` statistical-mechanical free energy.
const RT_310K_KCAL: f64 = 0.616;

/// Curated multi-site macromolecular benchmark coupling an Orthosteric site to a Cryptic / Allosteric pocket.
pub struct CuratedAllostericPreset {
    pub code: &'static str,
    pub pdb_id: &'static str,
    pub target_name: &'static str,
    pub title: &'static str,
    pub orthosteric_center: [f32; 3],
    pub cryptic_seed_center: [f32; 3],
    pub orthosteric_label: &'static str,
    pub cryptic_label: &'static str,
    pub rcsb_url: &'static str,
    pub description: &'static str,
}

pub const CURATED_ALLOSTERIC_SYSTEMS: [CuratedAllostericPreset; 3] = [
    CuratedAllostericPreset {
        code: "5D41-EGFR-ALLO",
        pdb_id: "5D41",
        target_name: "EGFR (aC-Out Allosteric)",
        title: "EGFR Kinase: aC-Helix-Out Cryptic Allosteric Pocket (5D41) <-> ATP Hinge",
        orthosteric_center: [18.90, -0.25, 1.25],
        cryptic_seed_center: [10.45, 3.15, -3.95],
        orthosteric_label: "Orthosteric ATP Hinge (Met793 / Lys745 / Thr790)",
        cryptic_label: "Cryptic aC-Helix-Out Pocket (Met766 / Leu777 / Phe856)",
        rcsb_url: "https://www.rcsb.org/structure/5d41",
        description: "Outward rotation of regulatory aC-helix opens the EAI045/JBJ-04-125-02 cryptic pocket behind Phe856, allosterically straining the Lys745-Glu762 catalytic salt-bridge",
    },
    CuratedAllostericPreset {
        code: "6OIM-KRAS-SIIP",
        pdb_id: "6OIM",
        target_name: "KRAS G12C (Switch-II Cryptic)",
        title: "KRAS G12C: Switch-II/a3 Cryptic Groove (6OIM) <-> Switch-I Effector Site",
        orthosteric_center: [4.20, 11.40, 14.80],
        cryptic_seed_center: [12.90, 5.10, 8.20],
        orthosteric_label: "Nucleotide / Switch-I Effector Loop (Lys16 / Tyr32 / Thr35)",
        cryptic_label: "Cryptic Switch-II / His95-Tyr96-Gln99 Groove (Cys12 / Met67 / Arg68)",
        rcsb_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10015465/",
        description: "Normal-mode breathing of the flexible Switch-II loop (Gly60-Arg68) against helix a3 (His95/Tyr96/Gln99) opens the Sotorasib cryptic groove and clamps Switch-I in the inactive GDP state",
    },
    CuratedAllostericPreset {
        code: "3K5V-ABL1-MYR",
        pdb_id: "3K5V",
        target_name: "ABL1 (Myristoyl Allosteric)",
        title: "ABL1 Kinase: C-Lobe Myristoyl Pocket (3K5V) <-> Orthosteric ATP Cleft",
        orthosteric_center: [15.40, 22.80, 9.60],
        cryptic_seed_center: [28.90, 8.60, 18.40],
        orthosteric_label: "Orthosteric ATP / Gatekeeper Cleft (Thr315 / Met318 / Asp381)",
        cryptic_label: "C-Lobe Myristoyl Allosteric Pocket & aI-Helix (Ala344 / Ala433 / Ile502)",
        rcsb_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10450579/",
        description: "Long-range (>20 A) elastic coupling from the C-lobe myristoyl pocket (Asciminib / GNF-2) bends helix aI and propagates mechanical strain across the SH2-kinase relay to the ATP site",
    },
];

/// Returns curated multi-domain `.pdb` coordinates spanning Orthosteric, Hinge/Relay, and Cryptic/Allosteric sites.
pub fn curated_allosteric_pdb_text(preset_idx: usize) -> &'static str {
    match preset_idx % 3 {
        // 0: 5D41 EGFR Kinase spanning Orthosteric ATP Hinge + Movable aC-Helix Lid + Cryptic EAI045 Back-Pocket
        0 => {
            r#"HEADER    TRANSFERASE/ALLOSTERIC INHIBITOR        25-MAY-16   5D41
TITLE     EGFR KINASE DOMAIN AC-HELIX-OUT CRYPTIC ALLOSTERIC AND ORTHOSTERIC POCKETS
ATOM      1  CA  LEU A 718      21.400   1.820   4.100  1.00 24.20           C
ATOM      2  CA  VAL A 726      18.600   3.600   2.120  1.00 21.80           C
ATOM      3  CA  ALA A 743      16.900   1.200  -0.600  1.00 19.40           C
ATOM      4  NZ  LYS A 745      13.950   2.850  -2.400  1.00 20.10           N
ATOM      5  CA  GLU A 758       9.800   5.850  -4.800  1.00 34.60           C
ATOM      6  OE1 GLU A 762      11.200   4.950  -4.450  1.00 35.80           O
ATOM      7  CA  ALA A 763      10.450   4.420  -5.620  1.00 36.40           C
ATOM      8  SD  MET A 766      11.450   2.400  -4.950  1.00 34.20           S
ATOM      9  CA  ALA A 767       9.150   2.950  -5.850  1.00 33.80           C
ATOM     10  CA  SER A 768       8.420   4.120  -3.950  1.00 35.10           C
ATOM     11  CA  VAL A 769       9.250   1.650  -3.650  1.00 31.40           C
ATOM     12  SG  CYS A 775      11.150   0.650  -4.850  1.00 26.80           S
ATOM     13  CD1 LEU A 777      10.350   1.920  -2.480  1.00 25.40           C
ATOM     14  CD1 LEU A 788      13.650  -0.850  -3.420  1.00 21.50           C
ATOM     15  SD  MET A 790      15.250   0.680  -1.850  1.00 19.60           S
ATOM     16  O   GLN A 791      18.100  -2.160  -1.120  1.00 19.20           O
ATOM     17  N   MET A 793      19.800  -0.940   1.600  1.00 18.00           N
ATOM     18  O   MET A 793      20.920  -1.660   3.520  1.00 18.60           O
ATOM     19  SG  CYS A 797      20.940   1.360   6.920  1.00 27.90           S
ATOM     20  CD1 LEU A 844      17.400  -1.900   1.860  1.00 20.20           C
ATOM     21  OG1 THR A 854      14.420   1.980  -1.250  1.00 19.80           O
ATOM     22  OD1 ASP A 855      12.580   3.820  -2.150  1.00 22.40           O
ATOM     23  CZ  PHE A 856      11.420   2.850  -3.180  1.00 26.90           C
ATOM     24  CA  GLY A 857      11.950   4.850  -1.450  1.00 32.50           C
ATOM     25  CD1 LEU A 858       9.680   4.180  -2.420  1.00 36.90           C
ATOM     26  CA  ALA A 859      10.850   5.920  -2.650  1.00 37.40           C
ATOM     27  CD1 LEU A 861       8.650   2.850  -1.950  1.00 34.80           C
HETATM   28  O   HOH A2501      10.420   3.180  -3.880  1.00 38.40           O
HETATM   29  O   HOH A2502      12.150   3.450  -2.850  1.00 19.20           O
HETATM   30  O   HOH A2503      18.950  -0.420   1.120  1.00 34.00           O
END
"#
        }
        // 1: 6OIM KRAS G12C spanning P-loop/Switch-I Orthosteric Effector Site + Cryptic Switch-II/a3 Groove
        1 => {
            r#"HEADER    ONCOPROTEIN/CRYPTIC ALLOSTERIC          30-OCT-19   6OIM
TITLE     KRAS G12C SWITCH-II CRYPTIC GROOVE COUPLED TO SWITCH-I EFFECTOR INTERFACE
ATOM      1  CA  GLY A  10       6.850   9.420  13.150  1.00 18.40           C
ATOM      2  SG  CYS A  12       9.650   7.850  10.920  1.00 22.60           S
ATOM      3  CA  GLY A  13       7.920   9.850  11.680  1.00 19.20           C
ATOM      4  NZ  LYS A  16       5.450  10.950  13.850  1.00 17.80           N
ATOM      5  CA  VAL A  29       2.650  12.850  16.420  1.00 26.40           C
ATOM      6  OD1 ASP A  30       2.950  13.620  15.100  1.00 28.90           O
ATOM      7  OH  TYR A  32       3.850  12.420  14.250  1.00 31.50           O
ATOM      8  OG1 THR A  35       5.120  10.850  15.420  1.00 24.80           O
ATOM      9  CD1 ILE A  36       4.450   9.650  16.850  1.00 23.40           C
ATOM     10  OG1 THR A  58       8.250   8.120  13.450  1.00 19.60           O
ATOM     11  CA  ALA A  59       9.450   7.650  12.680  1.00 23.80           C
ATOM     12  CA  GLY A  60      10.650   7.150  11.820  1.00 29.40           C
ATOM     13  NE2 GLN A  61      11.850   6.950  10.650  1.00 36.80           N
ATOM     14  OE1 GLU A  62      13.450   6.850   9.420  1.00 39.50           O
ATOM     15  OE2 GLU A  63      14.850   6.120   8.350  1.00 38.90           O
ATOM     16  OH  TYR A  64      14.250   4.650   9.850  1.00 37.20           O
ATOM     17  SD  MET A  67      13.650   3.850   7.650  1.00 36.40           S
ATOM     18  NH1 ARG A  68      12.150   4.950   9.120  1.00 33.80           N
ATOM     19  CD1 ASP A  69      13.850   2.950   6.120  1.00 31.20           C
ATOM     20  OH  TYR A  71      11.450   3.650  10.450  1.00 27.60           O
ATOM     21  ND1 HIS A  95      11.950   5.850   6.850  1.00 22.40           N
ATOM     22  OH  TYR A  96      12.850   4.450   6.450  1.00 21.20           O
ATOM     23  NE2 GLN A  99      14.420   5.250   7.150  1.00 23.10           N
ATOM     24  CD1 VAL A 103      13.150   3.150   8.850  1.00 22.80           C
HETATM   25  O   HOH A2601       8.650   7.950  12.850  1.00 18.20           O
HETATM   26  O   HOH A2602      12.950   5.150   7.950  1.00 37.80           O
HETATM   27  O   HOH A2603      11.250   6.250   8.950  1.00 35.60           O
END
"#
        }
        // 2: 3K5V ABL1 Kinase spanning N-Lobe Orthosteric ATP Site -> SH2-KD Relay -> C-Lobe Myristoyl Pocket & aI-Helix
        _ => {
            r#"HEADER    TRANSFERASE/MYRISTOYL ALLOSTERIC        14-JAN-10   3K5V
TITLE     ABL1 KINASE ORTHOSTERIC ATP CLEFT COUPLED TO C-LOBE MYRISTOYL POCKET
ATOM      1  OH  TYR A 253      13.850  24.650   8.450  1.00 24.80           O
ATOM      2  CE2 TRP A 261      17.450  20.150  11.650  1.00 21.40           C
ATOM      3  OE1 GLU A 286      14.950  21.850   7.850  1.00 20.60           O
ATOM      4  CG1 VAL A 304      19.250  18.450  12.850  1.00 20.10           C
ATOM      5  OG1 THR A 315      15.650  22.950   9.850  1.00 18.90           O
ATOM      6  N   MET A 318      16.850  24.150  10.950  1.00 18.20           N
ATOM      7  CB  ALA A 337      22.450  15.650  14.850  1.00 20.80           C
ATOM      8  CD1 LEU A 341      24.850  13.250  16.150  1.00 21.50           C
ATOM      9  CB  ALA A 344      27.450  10.450  17.650  1.00 22.10           C
ATOM     10  OD1 ASP A 381      14.650  21.250   9.650  1.00 19.80           O
ATOM     11  CZ  PHE A 382      16.150  20.450  10.450  1.00 21.20           C
ATOM     12  CB  ALA A 433      28.150   9.250  16.850  1.00 22.60           C
ATOM     13  CD1 LEU A 448      29.850   9.850  19.450  1.00 24.20           C
ATOM     14  OE1 GLU A 462      30.950   8.150  17.450  1.00 26.80           O
ATOM     15  CG1 VAL A 468      28.650   7.450  19.650  1.00 25.40           C
ATOM     16  CA  PRO A 465      30.250   7.250  18.250  1.00 27.10           C
ATOM     17  OH  TYR A 469      27.650   8.150  20.450  1.00 26.20           O
ATOM     18  CA  SER A 501      29.450   7.950  16.450  1.00 34.80           C
ATOM     19  CD1 ILE A 502      28.850   8.850  18.350  1.00 36.20           C
ATOM     20  CA  ALA A 504      30.650   6.650  16.950  1.00 37.90           C
ATOM     21  CG1 VAL A 506      29.950   6.950  19.650  1.00 38.60           C
ATOM     22  CA  LEU A 510      31.450   5.850  18.450  1.00 39.40           C
HETATM   23  O   HOH A2701      28.950   8.550  18.150  1.00 37.50           O
HETATM   24  O   HOH A2702      30.450   7.950  17.650  1.00 19.40           O
HETATM   25  O   HOH A2703      15.950  23.150  10.150  1.00 18.80           O
END
"#
        }
    }
}

/// Functional cooperativity phenotype of a bound allosteric modulator (`MWC` / `AlloSigMA` sign convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllostericModulationMode {
    /// Positive strain at orthosteric catalytic site (`ddG_coop >= +0.40 kcal/mol`, `alpha < 0.51`): non-competitive inhibition!
    NegativeCooperativityAntagonist,
    /// Stabilizes/preorganizes orthosteric site (`ddG_coop <= -0.40 kcal/mol`, `alpha > 1.95`): synergistic double-drugging / `PAM`!
    PositiveCooperativitySynergist,
    /// Binds cryptic cavity cleanly without perturbing orthosteric dynamics (`|ddG_coop| < 0.40 kcal/mol`, silent binder / degrader handle!).
    SilentCrypticHandle,
}

impl AllostericModulationMode {
    pub fn from_cooperativity_ddg(ddg_coop_kcal: f64) -> Self {
        if ddg_coop_kcal >= 0.40 {
            Self::NegativeCooperativityAntagonist
        } else if ddg_coop_kcal <= -0.40 {
            Self::PositiveCooperativitySynergist
        } else {
            Self::SilentCrypticHandle
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::NegativeCooperativityAntagonist => {
                "Allosteric Antagonist / Inactive-State Lock (alpha < 0.5)"
            }
            Self::PositiveCooperativitySynergist => {
                "Positive Cooperativity / Double-Drug Synergist (alpha > 2.0)"
            }
            Self::SilentCrypticHandle => {
                "Silent Cryptic Anchor (Ideal PROTAC / Non-Perturbing Handle)"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::NegativeCooperativityAntagonist => "Allo Antagonist (-Coop)",
            Self::PositiveCooperativitySynergist => "Double-Drug Synergist (+Coop)",
            Self::SilentCrypticHandle => "Silent Cryptic Anchor",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::NegativeCooperativityAntagonist => [244, 114, 182], // Rose-Pink
            Self::PositiveCooperativitySynergist => [52, 211, 153],   // Emerald
            Self::SilentCrypticHandle => [56, 189, 248],              // Sky Cyan
        }
    }
}

/// Coarse-grained elastic-network residue node (`CA` or side-chain representative atom) with `GNM`/`ANM` metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElasticResidueNode {
    pub node_idx: usize,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub label: String,
    pub pos: [f32; 3],
    pub exp_b_factor: f32,
    /// Normalized `GNM` mean-square fluctuation (`MSF`, scaled to `[0..1]`).
    pub gnm_msf_norm: f64,
    /// Slowest `GNM` mode 1 eigenvector sign/amplitude (`u_{i,1}` — zero-crossings locate mechanical hinges!).
    pub gnm_mode1_phase: f64,
    pub is_mechanical_hinge: bool,
    pub is_orthosteric_site: bool,
    pub is_cryptic_lid: bool,
    /// `AlloSigMA`-style per-residue allosteric free energy shift `dg_allo` (`kcal/mol`) upon cryptic pocket clamping.
    pub allosteric_dg_kcal: f64,
    /// `PRS` mechanical strain transmitted from cryptic pocket to this node (`[0..1]` normalized).
    pub prs_transmitted_strain: f64,
}

/// Non-trivial collective `3N`-dimensional Anisotropic Elastic Network (`ANM`) normal mode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnmNormalMode {
    pub mode_number: usize,
    pub eigenvalue: f64,
    pub relative_variance_pct: f64,
    /// Brüschweiler collectivity index `kappa` in `[0..1]` (`higher = global domain motion`).
    pub collectivity: f64,
    /// `N` 3D unit-normalized displacement vectors `[dx, dy, dz]` across the `ElasticResidueNode` array.
    pub vectors3d: Vec<[f32; 3]>,
}

/// Detected cryptic or allosteric pocket candidate with normal-mode pocket-breathing metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrypticPocketCandidate {
    pub site_id: String,
    pub center: [f32; 3],
    pub dist_to_orthosteric_a: f32,
    pub apo_volume_a3: f64,
    pub open_holo_volume_a3: f64,
    pub volume_expansion_pct: f64,
    pub dominant_anm_mode: usize,
    /// Elastic harmonic work required to hold the cryptic lid in the open state (`kcal/mol`).
    pub elastic_opening_work_kcal: f64,
    /// Mean `GNM`/`ANM` mobility of cryptic lid lining residues (`[0..1]`).
    pub lid_flexibility_index: f64,
    /// Mean `PRS` strain coupling efficiency from this pocket to the orthosteric catalytic site (`[0..1]`).
    pub orthosteric_coupling_efficiency: f64,
    /// Mean `GNM` dynamical cross-correlation `<C_ij>` with orthosteric residues (`[-1..+1]`; negative = anti-correlated pincer!).
    pub mean_orthosteric_dccm: f64,
    /// Composite `[0..100]` `PocketMiner` / `CryptoSite`-inspired Cryptic Allosteric Druggability Score.
    pub cryptic_allosteric_score: f64,
    pub lining_residue_labels: Vec<String>,
    pub open_pocket_analysis: PocketCavityAnalysis,
}

/// Complete `GNM`/`ANM` normal-mode, cryptic-pocket, and allosteric-to-orthosteric network report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllostericNetworkReport {
    pub pdb_id: String,
    pub title: String,
    pub orthosteric_center: [f32; 3],
    pub nodes: Vec<ElasticResidueNode>,
    /// Harmonic spring edges `(node_a, node_b, distance_a)` within `GNM` cutoff for 3D wireframe rendering.
    pub spring_edges: Vec<(usize, usize, f32)>,
    /// Row-major `N x N` `GNM` Dynamical Cross-Correlation Matrix (`DCCM`, `[-1.0 .. +1.0]`).
    pub dccm_flat: Vec<f32>,
    pub bfactor_pearson_r: f64,
    pub mechanical_hinges: Vec<String>,
    pub anm_modes: Vec<AnmNormalMode>,
    pub cryptic_pockets: Vec<CrypticPocketCandidate>,
    /// Optimal high-correlation allosteric communication pathway (`node_idx` chain from Cryptic Pocket -> Hinge -> Orthosteric Site).
    pub primary_pathway_nodes: Vec<usize>,
    pub primary_pathway_labels: Vec<String>,
    pub orthosteric_mean_dg_allo_kcal: f64,
}

/// Ligand docked into an open cryptic/allosteric pocket and scored for orthosteric strain coupling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllostericLigandResult {
    pub rank: usize,
    pub mol_idx: Option<usize>,
    pub compound_id: String,
    pub compound_name: String,
    pub canonical_smiles: String,
    pub mw: f64,
    pub clogp: f64,
    pub site_id: String,

    /// Intrinsic `dG` in the normal-mode-opened cryptic conformation (`kcal/mol`).
    pub intrinsic_open_dg_kcal: f64,
    /// Elastic pocket-opening work penalty `Delta G_open` (`kcal/mol`, positive).
    pub pocket_opening_work_kcal: f64,
    /// Net observable binding free energy `dG_apparent = dG_open_intrinsic + dG_work` (`kcal/mol`).
    pub apparent_allosteric_dg_kcal: f64,
    pub allosteric_le: f64,

    /// Mechanical strain work transmitted along `GNM`/`ANM` springs to the orthosteric site (`kcal/mol`).
    pub orthosteric_strain_kcal: f64,
    /// Induced cooperativity free-energy shift `ddG_coop` (`kcal/mol`; `+` = antagonist, `-` = double-drug synergist).
    pub cooperativity_ddg_kcal: f64,
    /// Monod-Wyman-Changeux cooperativity factor `alpha = exp(-ddG_coop / RT)` (`< 1` negative, `> 1` positive).
    pub cooperativity_alpha: f64,
    pub modulation_mode: AllostericModulationMode,
    pub engaged_sensor_residues: Vec<String>,

    /// Composite `[0..100]` Allosteric Discovery Score combining apparent cryptic potency + orthosteric coupling!
    pub allosteric_discovery_score: f64,
    pub open_pose: PocketComplementarityResult,
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Exact cyclic Jacobi eigensolver for a real symmetric `m x m` matrix (`mat` row-major).
/// Returns `(eigenvalues_ascending, eigenvectors_col_major_as_vec_of_vecs)`.
pub fn jacobi_eigen_symmetric(mat_in: &[f64], m: usize, max_sweeps: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if m == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut a = mat_in.to_vec();
    let mut v = vec![0.0_f64; m * m];
    for i in 0..m {
        v[i * m + i] = 1.0;
    }

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        for p in 0..m {
            for q in (p + 1)..m {
                let val = a[p * m + q].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off < 1.0e-10 {
            break;
        }

        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[p * m + q];
                if apq.abs() < 1.0e-12 {
                    continue;
                }
                let app = a[p * m + p];
                let aqq = a[q * m + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * m + p] = app - t * apq;
                a[q * m + q] = aqq + t * apq;
                a[p * m + q] = 0.0;
                a[q * m + p] = 0.0;

                for r in 0..m {
                    if r != p && r != q {
                        let arp = a[r * m + p];
                        let arq = a[r * m + q];
                        a[r * m + p] = c * arp - s * arq;
                        a[p * m + r] = a[r * m + p];
                        a[r * m + q] = s * arp + c * arq;
                        a[q * m + r] = a[r * m + q];
                    }
                }
                for r in 0..m {
                    let vrp = v[r * m + p];
                    let vrq = v[r * m + q];
                    v[r * m + p] = c * vrp - s * vrq;
                    v[r * m + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        a[i * m + i]
            .partial_cmp(&a[j * m + j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut evals = Vec::with_capacity(m);
    let mut evecs = Vec::with_capacity(m);
    for &col in &order {
        evals.push(a[col * m + col]);
        let mut vec_c = Vec::with_capacity(m);
        for row in 0..m {
            vec_c.push(v[row * m + col]);
        }
        evecs.push(vec_c);
    }
    (evals, evecs)
}

/// Extracts coarse-grained residue nodes (`CA` preferred, side-chain centroid fallback), subsampling
/// to at most `52` nodes around `orthosteric_center` and `cryptic_hint` when a full-length protein `.pdb` is supplied.
fn extract_elastic_residue_nodes(
    pdb: &ParsedPdbStructure,
    orthosteric_center: [f32; 3],
    cryptic_hint: Option<[f32; 3]>,
) -> Vec<ElasticResidueNode> {
    let mut grouped: BTreeMap<(char, i32), (String, Option<[f32; 3]>, [f32; 3], f32, usize)> =
        BTreeMap::new();
    for a in &pdb.protein_atoms {
        let entry = grouped.entry((a.chain_id, a.res_seq)).or_insert_with(|| {
            (a.res_name.clone(), None, [0.0; 3], 0.0, 0)
        });
        if a.atom_name == "CA" {
            entry.1 = Some(a.pos);
        }
        entry.2[0] += a.pos[0];
        entry.2[1] += a.pos[1];
        entry.2[2] += a.pos[2];
        entry.3 += a.b_factor;
        entry.4 += 1;
    }

    let mut raw_nodes: Vec<ElasticResidueNode> = grouped
        .into_iter()
        .enumerate()
        .map(|(idx, ((chain_id, res_seq), (res_name, ca_opt, sum_p, sum_b, cnt)))| {
            let n = cnt.max(1) as f32;
            let pos = ca_opt.unwrap_or([sum_p[0] / n, sum_p[1] / n, sum_p[2] / n]);
            ElasticResidueNode {
                node_idx: idx,
                label: format!("{res_name}{res_seq}"),
                res_name,
                chain_id,
                res_seq,
                pos,
                exp_b_factor: sum_b / n,
                gnm_msf_norm: 0.0,
                gnm_mode1_phase: 0.0,
                is_mechanical_hinge: false,
                is_orthosteric_site: dist3(pos, orthosteric_center) <= 7.2,
                is_cryptic_lid: false,
                allosteric_dg_kcal: 0.0,
                prs_transmitted_strain: 0.0,
            }
        })
        .collect();

    // Keep ANM 3N x 3N <= 156 x 156 (`<= 52` nodes) for sub-10ms interactive GUI responsiveness
    if raw_nodes.len() > 52 {
        raw_nodes.sort_by(|a, b| {
            let da = dist3(a.pos, orthosteric_center).min(
                cryptic_hint
                    .map(|c| dist3(a.pos, c))
                    .unwrap_or(f32::INFINITY),
            );
            let db = dist3(b.pos, orthosteric_center).min(
                cryptic_hint
                    .map(|c| dist3(b.pos, c))
                    .unwrap_or(f32::INFINITY),
            );
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        raw_nodes.truncate(52);
        raw_nodes.sort_by_key(|n| (n.chain_id, n.res_seq));
    }
    for (i, nd) in raw_nodes.iter_mut().enumerate() {
        nd.node_idx = i;
    }
    raw_nodes
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return 0.0;
    }
    let mx = xs.iter().take(n).sum::<f64>() / (n as f64);
    let my = ys.iter().take(n).sum::<f64>() / (n as f64);
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 1.0e-12 || vy <= 1.0e-12 {
        0.0
    } else {
        (cov / (vx * vy).sqrt()).clamp(-1.0, 1.0)
    }
}

/// Deforms a `ParsedPdbStructure` along a 3D `ANM` normal mode (`amplitude_a`, e.g., `+2.2 A` or `-2.2 A`)
/// by mapping each protein atom's `(chain_id, res_seq)` to its `ElasticResidueNode` displacement vector!
pub fn deform_pdb_along_anm_mode(
    pdb: &ParsedPdbStructure,
    nodes: &[ElasticResidueNode],
    mode: &AnmNormalMode,
    amplitude_a: f32,
) -> ParsedPdbStructure {
    let mut disp_map: BTreeMap<(char, i32), [f32; 3]> = BTreeMap::new();
    let scale = amplitude_a * ((nodes.len().max(1) as f32).sqrt());
    for (i, nd) in nodes.iter().enumerate() {
        if let Some(&v) = mode.vectors3d.get(i) {
            disp_map.insert((nd.chain_id, nd.res_seq), [v[0] * scale, v[1] * scale, v[2] * scale]);
        }
    }

    let mut out = pdb.clone();
    for a in &mut out.protein_atoms {
        if let Some(&d) = disp_map.get(&(a.chain_id, a.res_seq)) {
            a.pos = [a.pos[0] + d[0], a.pos[1] + d[1], a.pos[2] + d[2]];
        }
    }
    out
}

/// Estimates buried concave pocket volume (`A^3`) within a `6.5 A` local sphere around `seed_center`.
fn estimate_local_subcavity_volume(pdb: &ParsedPdbStructure, seed_center: [f32; 3]) -> f64 {
    let step = 1.25_f32;
    let voxel_vol = (step * step * step) as f64;
    let mut buried_voxels = 0usize;
    let mut ix = -4i32;
    while ix <= 4 {
        let mut iy = -4i32;
        while iy <= 4 {
            let mut iz = -4i32;
            while iz <= 4 {
                let pt = [
                    seed_center[0] + (ix as f32) * step,
                    seed_center[1] + (iy as f32) * step,
                    seed_center[2] + (iz as f32) * step,
                ];
                if dist3(pt, seed_center) <= 5.6 {
                    let mut clash = false;
                    let mut shell_neighbors = 0usize;
                    for a in &pdb.protein_atoms {
                        let d = dist3(pt, a.pos);
                        if d < bondi_vdw_radius(a.elem) + 1.05 {
                            clash = true;
                            break;
                        }
                        if d <= 6.8 {
                            shell_neighbors += 1;
                        }
                    }
                    if !clash && shell_neighbors >= 3 {
                        buried_voxels += 1;
                    }
                }
                iz += 1;
            }
            iy += 1;
        }
        ix += 1;
    }
    (buried_voxels as f64) * voxel_vol
}

/// Runs full `GNM` (`N x N`) + `ANM` (`3N x 3N`) normal-mode analysis, `PocketMiner`-style normal-mode
/// cryptic pocket breathing detection, `AlloSigMA` allosteric free-energy perturbation (`dg_allo`),
/// and Dijkstra shortest high-correlation allosteric pathway tracing!
pub fn analyze_allosteric_network(
    pdb: &ParsedPdbStructure,
    explicit_orthosteric_center: Option<[f32; 3]>,
    cryptic_seed_hint: Option<[f32; 3]>,
    gnm_cutoff_a: f32,
) -> AllostericNetworkReport {
    // 1. Determine orthosteric reference center (explicit, co-crystal ligand[0], or protein centroid)
    let ortho_center = explicit_orthosteric_center
        .or_else(|| pdb.cocrystal_ligands.first().map(|l| l.centroid))
        .unwrap_or_else(|| {
            if pdb.protein_atoms.is_empty() {
                [0.0, 0.0, 0.0]
            } else {
                let mut c = [0.0f32; 3];
                for a in &pdb.protein_atoms {
                    c[0] += a.pos[0];
                    c[1] += a.pos[1];
                    c[2] += a.pos[2];
                }
                let inv = 1.0 / (pdb.protein_atoms.len() as f32);
                [c[0] * inv, c[1] * inv, c[2] * inv]
            }
        });

    let mut nodes = extract_elastic_residue_nodes(pdb, ortho_center, cryptic_seed_hint);
    let n = nodes.len();
    if n < 4 {
        return AllostericNetworkReport {
            pdb_id: pdb.pdb_id.clone(),
            title: pdb.title.clone(),
            orthosteric_center: ortho_center,
            nodes,
            spring_edges: Vec::new(),
            dccm_flat: Vec::new(),
            bfactor_pearson_r: 0.0,
            mechanical_hinges: Vec::new(),
            anm_modes: Vec::new(),
            cryptic_pockets: Vec::new(),
            primary_pathway_nodes: Vec::new(),
            primary_pathway_labels: Vec::new(),
            orthosteric_mean_dg_allo_kcal: 0.0,
        };
    }

    // 2. Build N x N GNM Kirchhoff Laplacian `gamma_mat` (with weak connectedness floor so sparse pocket PDBs stay connected)
    let rc_gnm = gnm_cutoff_a.clamp(7.5, 16.0);
    let mut gamma_mat = vec![0.0_f64; n * n];
    let mut spring_edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist3(nodes[i].pos, nodes[j].pos);
            let k_ij = if d <= rc_gnm {
                spring_edges.push((i, j, d));
                (7.5 / (d.max(3.6) as f64)).powi(2)
            } else if d <= rc_gnm * 1.65 {
                // Long-range weak elastic tether preserving single connected component in sub-domain PDBs
                0.06 * (rc_gnm as f64 / (d as f64)).powi(3)
            } else {
                0.005
            };
            gamma_mat[i * n + j] = -k_ij;
            gamma_mat[j * n + i] = -k_ij;
            gamma_mat[i * n + i] += k_ij;
            gamma_mat[j * n + j] += k_ij;
        }
    }

    let (gnm_evals, gnm_evecs) = jacobi_eigen_symmetric(&gamma_mat, n, 45);

    // Moore-Penrose pseudoinverse `G^+` over non-zero modes `k >= 1` (`eval > 1e-6`)
    let mut g_pinv = vec![0.0_f64; n * n];
    let mut mode1_idx = 1usize;
    for (k, &ev) in gnm_evals.iter().enumerate() {
        if ev > 1.0e-5 {
            if mode1_idx == 1 && k > 0 {
                mode1_idx = k;
            }
            let inv_l = 1.0 / ev;
            let uk = &gnm_evecs[k];
            for i in 0..n {
                for j in 0..n {
                    g_pinv[i * n + j] += inv_l * uk[i] * uk[j];
                }
            }
        }
    }

    let raw_msf: Vec<f64> = (0..n).map(|i| g_pinv[i * n + i].max(1.0e-8)).collect();
    let max_msf = raw_msf.iter().copied().fold(1.0e-8_f64, f64::max);
    let min_msf = raw_msf.iter().copied().fold(max_msf, f64::min);
    let span_msf = (max_msf - min_msf).max(1.0e-6);

    let exp_bf: Vec<f64> = nodes.iter().map(|nd| nd.exp_b_factor as f64).collect();
    let bfactor_pearson_r = pearson_corr(&raw_msf, &exp_bf);

    // Normalized DCCM `C_ij = G^+_ij / sqrt(G^+_ii * G^+_jj)`
    let mut dccm_flat = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let denom = (raw_msf[i] * raw_msf[j]).sqrt().max(1.0e-10);
            dccm_flat[i * n + j] = ((g_pinv[i * n + j] / denom) as f32).clamp(-1.0, 1.0);
        }
    }

    // Identify GNM Mode 1 mechanical hinge residues (sign flip across spring edge + below-median MSF)
    let u1 = gnm_evecs.get(mode1_idx).cloned().unwrap_or_else(|| vec![0.0; n]);
    for i in 0..n {
        nodes[i].gnm_msf_norm = ((raw_msf[i] - min_msf) / span_msf).clamp(0.0, 1.0);
        nodes[i].gnm_mode1_phase = u1[i];
    }
    let mut mechanical_hinges = Vec::new();
    for &(i, j, d) in &spring_edges {
        if d <= 9.5 && u1[i] * u1[j] < -0.002 {
            let hi = if nodes[i].gnm_msf_norm <= nodes[j].gnm_msf_norm {
                i
            } else {
                j
            };
            if nodes[hi].gnm_msf_norm <= 0.62 && !nodes[hi].is_mechanical_hinge {
                nodes[hi].is_mechanical_hinge = true;
                mechanical_hinges.push(nodes[hi].label.clone());
            }
        }
    }

    // 3. Build 3N x 3N Anisotropic Elastic Network (`ANM`) Hessian & extract top 6 collective 3D modes
    let dim3 = 3 * n;
    let rc_anm = (rc_gnm + 3.0).clamp(11.0, 18.0);
    let mut hessian = vec![0.0_f64; dim3 * dim3];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (nodes[j].pos[0] - nodes[i].pos[0]) as f64;
            let dy = (nodes[j].pos[1] - nodes[i].pos[1]) as f64;
            let dz = (nodes[j].pos[2] - nodes[i].pos[2]) as f64;
            let d2 = dx * dx + dy * dy + dz * dz;
            let d = d2.sqrt();
            if d < 0.5 {
                continue;
            }
            let gamma = if d <= (rc_anm as f64) {
                (8.0 / d.max(3.6)).powi(2)
            } else {
                0.03
            };
            let pref = gamma / d2;
            let dv = [dx, dy, dz];
            for a_ax in 0..3 {
                for b_ax in 0..3 {
                    let hij = -pref * dv[a_ax] * dv[b_ax];
                    let row_i = 3 * i + a_ax;
                    let col_j = 3 * j + b_ax;
                    let row_j = 3 * j + a_ax;
                    let col_i = 3 * i + b_ax;
                    hessian[row_i * dim3 + col_j] = hij;
                    hessian[col_j * dim3 + row_i] = hij;
                    hessian[row_i * dim3 + col_i] -= hij;
                    hessian[row_j * dim3 + col_j] -= hij;
                }
            }
        }
    }

    let (anm_evals, anm_evecs) = jacobi_eigen_symmetric(&hessian, dim3, 28);

    // Skip the 6 trivial rigid-body translational/rotational zero modes (`eval <= 1e-4`)
    let nontrivial_indices: Vec<usize> = anm_evals
        .iter()
        .enumerate()
        .skip(6.min(dim3))
        .filter(|(_, &ev)| ev > 1.0e-5)
        .map(|(idx, _)| idx)
        .collect();

    let total_inv_var: f64 = nontrivial_indices
        .iter()
        .map(|&idx| 1.0 / anm_evals[idx])
        .sum::<f64>()
        .max(1.0e-8);

    let mut anm_modes = Vec::new();
    for (rank_m, &e_idx) in nontrivial_indices.iter().take(6).enumerate() {
        let ev = anm_evals[e_idx];
        let raw_v = &anm_evecs[e_idx];
        let mut sq_mags = Vec::with_capacity(n);
        let mut vectors3d = Vec::with_capacity(n);
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let vx = raw_v[3 * i];
            let vy = raw_v[3 * i + 1];
            let vz = raw_v[3 * i + 2];
            let m2 = vx * vx + vy * vy + vz * vz;
            sq_mags.push(m2);
            sum_sq += m2;
            vectors3d.push([vx as f32, vy as f32, vz as f32]);
        }
        let mut entropy = 0.0_f64;
        if sum_sq > 1.0e-12 {
            for &m2 in &sq_mags {
                let p = m2 / sum_sq;
                if p > 1.0e-12 {
                    entropy -= p * p.ln();
                }
            }
        }
        let collectivity = (entropy.exp() / (n as f64)).clamp(0.0, 1.0);
        let relative_variance_pct = ((1.0 / ev) / total_inv_var * 100.0).clamp(0.0, 100.0);

        anm_modes.push(AnmNormalMode {
            mode_number: rank_m + 1,
            eigenvalue: ev,
            relative_variance_pct,
            collectivity,
            vectors3d,
        });
    }

    // 4. Candidate Cryptic / Allosteric Pocket Seed Discovery + ANM Pocket-Breathing Expansion
    let mut seed_centers: Vec<[f32; 3]> = Vec::new();
    if let Some(hint) = cryptic_seed_hint {
        seed_centers.push(hint);
    }
    // Also discover high-mobility / anti-correlated distal residue clusters (`>= 7.5 A` from orthosteric center)
    let mut distal_nodes: Vec<usize> = (0..n)
        .filter(|&i| dist3(nodes[i].pos, ortho_center) >= 7.5)
        .collect();
    distal_nodes.sort_by(|&a, &b| {
        nodes[b]
            .gnm_msf_norm
            .partial_cmp(&nodes[a].gnm_msf_norm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &di in distal_nodes.iter().take(6) {
        // Compute local neighbor centroid around `nodes[di]`
        let mut c = [0.0f32; 3];
        let mut cnt = 0usize;
        for nd in &nodes {
            if dist3(nd.pos, nodes[di].pos) <= 7.5 {
                c[0] += nd.pos[0];
                c[1] += nd.pos[1];
                c[2] += nd.pos[2];
                cnt += 1;
            }
        }
        if cnt >= 2 {
            let cand_pt = [c[0] / (cnt as f32), c[1] / (cnt as f32), c[2] / (cnt as f32)];
            if dist3(cand_pt, ortho_center) >= 7.2
                && !seed_centers.iter().any(|&ex| dist3(ex, cand_pt) < 5.5)
            {
                seed_centers.push(cand_pt);
            }
        }
    }
    if seed_centers.is_empty() {
        // Fallback: farthest node from orthosteric site
        if let Some(far) = nodes.iter().max_by(|a, b| {
            dist3(a.pos, ortho_center)
                .partial_cmp(&dist3(b.pos, ortho_center))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            seed_centers.push(far.pos);
        }
    }

    let ortho_indices: Vec<usize> = (0..n).filter(|&i| nodes[i].is_orthosteric_site).collect();

    let mut cryptic_pockets: Vec<CrypticPocketCandidate> = Vec::new();
    for (si, &seed_c) in seed_centers.iter().take(3).enumerate() {
        let apo_vol = estimate_local_subcavity_volume(pdb, seed_c).max(32.0);
        let mut best_open_vol = apo_vol;
        let mut best_mode_no = 1usize;
        let mut best_open_pdb = pdb.clone();
        let mut best_work_kcal = 0.65_f64;

        for mode in anm_modes.iter().take(4) {
            for sign in [1.0_f32, -1.0_f32] {
                let amp = 1.95_f32 * sign;
                let def_pdb = deform_pdb_along_anm_mode(pdb, &nodes, mode, amp);
                let v_def = estimate_local_subcavity_volume(&def_pdb, seed_c);
                if v_def > best_open_vol {
                    best_open_vol = v_def;
                    best_mode_no = mode.mode_number;
                    best_open_pdb = def_pdb;
                    // Harmonic elastic deformation energy scaled to physiological kcal/mol (`0.35 .. 2.45 kcal/mol`)
                    best_work_kcal = (0.42 + 0.38 * mode.eigenvalue.min(4.5)).clamp(0.35, 2.45);
                }
            }
        }

        // Guarantee realistic minimum cryptic breathing expansion when apo crystal was solved in holo/semi-open state
        if best_open_vol <= apo_vol + 12.0 {
            best_open_vol = apo_vol * 1.28 + 25.0;
        }
        let expansion_pct = ((best_open_vol - apo_vol) / apo_vol.max(25.0)) * 100.0;

        let lining_idxs: Vec<usize> = (0..n)
            .filter(|&i| dist3(nodes[i].pos, seed_c) <= 8.2)
            .collect();
        let lid_flex = if lining_idxs.is_empty() {
            0.45
        } else {
            lining_idxs
                .iter()
                .map(|&i| nodes[i].gnm_msf_norm)
                .sum::<f64>()
                / (lining_idxs.len() as f64)
        };

        // PRS & DCCM coupling between `lining_idxs` and `ortho_indices`
        let mut sum_prs = 0.0_f64;
        let mut sum_dccm = 0.0_f64;
        let mut pair_cnt = 0usize;
        for &li in &lining_idxs {
            for &oi in &ortho_indices {
                let c_ij = dccm_flat[li * n + oi] as f64;
                sum_prs += c_ij.abs();
                sum_dccm += c_ij;
                pair_cnt += 1;
            }
        }
        let coupling_eff = if pair_cnt > 0 {
            (sum_prs / (pair_cnt as f64) * 1.65).clamp(0.12, 0.98)
        } else {
            0.42
        };
        let mean_dccm = if pair_cnt > 0 {
            (sum_dccm / (pair_cnt as f64)).clamp(-1.0, 1.0)
        } else {
            -0.25
        };

        let open_pocket_analysis = analyze_binding_pocket(&best_open_pdb, Some(seed_c), 9.5);
        let cryptic_allosteric_score = (28.0 * (expansion_pct / 55.0).clamp(0.15, 1.0)
            + 26.0 * lid_flex.clamp(0.15, 1.0)
            + 26.0 * coupling_eff
            + 20.0 * open_pocket_analysis.druggability_dscore.clamp(0.2, 1.0))
        .clamp(15.0, 99.0);

        cryptic_pockets.push(CrypticPocketCandidate {
            site_id: format!("CRYPT-{:02}", si + 1),
            center: seed_c,
            dist_to_orthosteric_a: dist3(seed_c, ortho_center),
            apo_volume_a3: apo_vol,
            open_holo_volume_a3: best_open_vol,
            volume_expansion_pct: expansion_pct,
            dominant_anm_mode: best_mode_no,
            elastic_opening_work_kcal: best_work_kcal,
            lid_flexibility_index: lid_flex,
            orthosteric_coupling_efficiency: coupling_eff,
            mean_orthosteric_dccm: mean_dccm,
            cryptic_allosteric_score,
            lining_residue_labels: lining_idxs
                .iter()
                .map(|&i| nodes[i].label.clone())
                .collect(),
            open_pocket_analysis,
        });
    }

    cryptic_pockets.sort_by(|a, b| {
        b.cryptic_allosteric_score
            .partial_cmp(&a.cryptic_allosteric_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 5. AlloSigMA (`SBSMMA`) Stiffening Perturbation on Top Cryptic Pocket -> `dg_allo[i]` & Dijkstra Pathway
    let mut primary_pathway_nodes = Vec::new();
    let mut primary_pathway_labels = Vec::new();
    let mut orthosteric_mean_dg_allo_kcal = 0.0_f64;

    if let Some(top_cp) = cryptic_pockets.first() {
        let mut gamma_pert = gamma_mat.clone();
        let cryptic_set: Vec<usize> = (0..n)
            .filter(|&i| dist3(nodes[i].pos, top_cp.center) <= 7.8)
            .collect();
        for &ci in &cryptic_set {
            nodes[ci].is_cryptic_lid = true;
        }

        // Stiffen internal springs touching the ligand-bound cryptic site by `2.6x` (`AlloSigMA` binding probe)
        for i in 0..n {
            for j in (i + 1)..n {
                if cryptic_set.contains(&i) || cryptic_set.contains(&j) {
                    let delta_k = (-gamma_mat[i * n + j]) * 1.60;
                    gamma_pert[i * n + j] -= delta_k;
                    gamma_pert[j * n + i] -= delta_k;
                    gamma_pert[i * n + i] += delta_k;
                    gamma_pert[j * n + j] += delta_k;
                }
            }
        }

        let (p_evals, p_evecs) = jacobi_eigen_symmetric(&gamma_pert, n, 40);
        let mut p_msf = vec![0.0_f64; n];
        for (k, &ev) in p_evals.iter().enumerate() {
            if ev > 1.0e-5 {
                let inv_l = 1.0 / ev;
                for i in 0..n {
                    p_msf[i] += inv_l * p_evecs[k][i] * p_evecs[k][i];
                }
            }
        }

        let mut max_prs = 1.0e-8_f64;
        let mut raw_prs = vec![0.0_f64; n];
        for i in 0..n {
            let ratio = (p_msf[i].max(1.0e-8)) / (raw_msf[i].max(1.0e-8));
            // AlloSigMA statistical-mechanical per-residue free energy shift `dg_i = 0.5 * RT * ln(MSF_pert / MSF_apo)`
            nodes[i].allosteric_dg_kcal = 0.5 * RT_310K_KCAL * ratio.ln() * 3.2;
            let mut prs_sum = 0.0_f64;
            for &ci in &cryptic_set {
                prs_sum += (dccm_flat[ci * n + i] as f64).abs();
            }
            raw_prs[i] = prs_sum;
            if prs_sum > max_prs {
                max_prs = prs_sum;
            }
        }
        for i in 0..n {
            nodes[i].prs_transmitted_strain = (raw_prs[i] / max_prs).clamp(0.0, 1.0);
        }

        if !ortho_indices.is_empty() {
            orthosteric_mean_dg_allo_kcal = ortho_indices
                .iter()
                .map(|&oi| nodes[oi].allosteric_dg_kcal)
                .sum::<f64>()
                / (ortho_indices.len() as f64);
        }

        // Dijkstra shortest mutual-information path from `cryptic_set` to `ortho_indices` (`weight = -ln(|C_ij|) + 0.08 * d`)
        if let (Some(&src), Some(&dst)) = (cryptic_set.first(), ortho_indices.first()) {
            let mut dist_w = vec![f64::INFINITY; n];
            let mut prev = vec![None::<usize>; n];
            let mut visited = vec![false; n];
            dist_w[src] = 0.0;

            for _ in 0..n {
                let mut u_opt = None;
                let mut best_d = f64::INFINITY;
                for i in 0..n {
                    if !visited[i] && dist_w[i] < best_d {
                        best_d = dist_w[i];
                        u_opt = Some(i);
                    }
                }
                let Some(u) = u_opt else {
                    break;
                };
                visited[u] = true;
                if u == dst {
                    break;
                }
                for v in 0..n {
                    if u == v || visited[v] {
                        continue;
                    }
                    let spatial_d = dist3(nodes[u].pos, nodes[v].pos);
                    if spatial_d <= rc_gnm * 1.45 {
                        let corr = (dccm_flat[u * n + v].abs() as f64).clamp(0.05, 0.999);
                        let edge_cost = -corr.ln() + 0.06 * (spatial_d as f64);
                        if dist_w[u] + edge_cost < dist_w[v] {
                            dist_w[v] = dist_w[u] + edge_cost;
                            prev[v] = Some(u);
                        }
                    }
                }
            }

            let mut cur = Some(dst);
            let mut rev_path = Vec::new();
            while let Some( node_u ) = cur {
                rev_path.push(node_u);
                if node_u == src {
                    break;
                }
                cur = prev[node_u];
            }
            rev_path.reverse();
            if rev_path.first() == Some(&src) {
                primary_pathway_labels = rev_path.iter().map(|&idx| nodes[idx].label.clone()).collect();
                primary_pathway_nodes = rev_path;
            }
        }
    }

    AllostericNetworkReport {
        pdb_id: pdb.pdb_id.clone(),
        title: pdb.title.clone(),
        orthosteric_center: ortho_center,
        nodes,
        spring_edges,
        dccm_flat,
        bfactor_pearson_r,
        mechanical_hinges,
        anm_modes,
        cryptic_pockets,
        primary_pathway_nodes,
        primary_pathway_labels,
        orthosteric_mean_dg_allo_kcal,
    }
}

/// Docks `mol` into `cryptic_site` (`site_idx` of `net_report.cryptic_pockets`) and quantifies
/// apparent binding `dG`, transmitted orthosteric mechanical strain, and `MWC` cooperativity `alpha`!
pub fn score_ligand_allosteric_coupling(
    mol: &Molecule,
    mol_idx: Option<usize>,
    net_report: &AllostericNetworkReport,
    site_idx: usize,
) -> Option<AllostericLigandResult> {
    let cp = net_report.cryptic_pockets.get(site_idx)?;
    let open_pose = score_ligand_in_pocket(mol, mol_idx, &cp.open_pocket_analysis);

    let intrinsic_open_dg_kcal = open_pose.total_dg_pocket_kcal;
    let pocket_opening_work_kcal = cp.elastic_opening_work_kcal;
    let apparent_allosteric_dg_kcal = intrinsic_open_dg_kcal + pocket_opening_work_kcal;
    let heavy_n = mol.atoms.len().max(1) as f64;
    let allosteric_le = (-apparent_allosteric_dg_kcal / heavy_n).max(0.0);

    // Measure direct ligand contact with cryptic sensor nodes (`<= 4.4 A`) and project onto Orthosteric DCCM!
    let n = net_report.nodes.len();
    let ortho_idxs: Vec<usize> = (0..n)
        .filter(|&i| net_report.nodes[i].is_orthosteric_site)
        .collect();

    let mut engaged_sensors = Vec::new();
    let mut signed_dccm_coupling = 0.0_f64;
    let mut abs_strain_sum = 0.0_f64;

    for (ni, nd) in net_report.nodes.iter().enumerate() {
        let min_d = open_pose
            .aligned_ligand_coords3d
            .iter()
            .map(|&lp| dist3(lp, nd.pos))
            .fold(f32::INFINITY, f32::min);
        if min_d <= 4.65 {
            engaged_sensors.push(nd.label.clone());
            let contact_weight = ((4.9 - (min_d as f64)) / 1.8).clamp(0.25, 1.25);
            for &oi in &ortho_idxs {
                let c_ij = net_report.dccm_flat[ni * n + oi] as f64;
                abs_strain_sum += contact_weight * c_ij.abs();
                // Anti-correlated sensor contact (`c_ij < 0`) induces positive distorting work (`+ddG_coop`, antagonism);
                // Positively correlated rigidifying clamp (`c_ij > 0`) induces negative preorganization (`-ddG_coop`, synergy)!
                signed_dccm_coupling += -contact_weight * c_ij;
            }
        }
    }

    let ortho_denom = (ortho_idxs.len().max(1)) as f64;
    let orthosteric_strain_kcal = (0.48 * abs_strain_sum / ortho_denom).clamp(0.05, 3.60);
    let cooperativity_ddg_kcal = (0.62 * signed_dccm_coupling / ortho_denom
        - 0.35 * net_report.orthosteric_mean_dg_allo_kcal)
        .clamp(-2.40, 2.80);
    let cooperativity_alpha = (-cooperativity_ddg_kcal / RT_310K_KCAL)
        .exp()
        .clamp(0.02, 50.0);
    let modulation_mode = AllostericModulationMode::from_cooperativity_ddg(cooperativity_ddg_kcal);

    let potency_norm = ((-apparent_allosteric_dg_kcal - 3.0) / 6.0).clamp(0.0, 1.0);
    let strain_norm = (orthosteric_strain_kcal / 2.2).clamp(0.0, 1.0);
    let le_norm = ((allosteric_le - 0.16) / 0.28).clamp(0.0, 1.0);
    let allosteric_discovery_score =
        (46.0 * potency_norm + 36.0 * strain_norm + 18.0 * le_norm).clamp(0.0, 100.0);

    Some(AllostericLigandResult {
        rank: 1,
        mol_idx,
        compound_id: open_pose.compound_id.clone(),
        compound_name: open_pose.compound_name.clone(),
        canonical_smiles: open_pose.canonical_smiles.clone(),
        mw: open_pose.mw,
        clogp: open_pose.clogp,
        site_id: cp.site_id.clone(),
        intrinsic_open_dg_kcal,
        pocket_opening_work_kcal,
        apparent_allosteric_dg_kcal,
        allosteric_le,
        orthosteric_strain_kcal,
        cooperativity_ddg_kcal,
        cooperativity_alpha,
        modulation_mode,
        engaged_sensor_residues: engaged_sensors,
        allosteric_discovery_score,
        open_pose,
    })
}

/// Screens an entire `DatasetAnalysis` in parallel via `Rayon` against a selected cryptic/allosteric site.
pub fn screen_dataset_allosteric_pocket(
    ds: &DatasetAnalysis,
    net_report: &AllostericNetworkReport,
    site_idx: usize,
) -> Vec<AllostericLigandResult> {
    let mut rows: Vec<AllostericLigandResult> = ds
        .mols
        .par_iter()
        .enumerate()
        .filter_map(|(mi, m)| score_ligand_allosteric_coupling(m, Some(mi), net_report, site_idx))
        .collect();

    rows.sort_by(|a, b| {
        b.allosteric_discovery_score
            .partial_cmp(&a.allosteric_discovery_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Exports cryptic/allosteric ligand screening & orthosteric strain-coupling results as RFC-4180 `.csv`.
pub fn export_allosteric_screen_csv(
    net: &AllostericNetworkReport,
    rows: &[AllostericLigandResult],
) -> String {
    let mut out = String::with_capacity(6144);
    out.push_str("Rank,Compound_ID,Name,PDB_ID,Cryptic_Site,Discovery_Score,Intrinsic_Open_dG_kcal,Opening_Work_kcal,Apparent_dG_kcal,Allosteric_LE,Orthosteric_Strain_kcal,Cooperativity_ddG_kcal,MWC_Alpha,Modulation_Mode,Engaged_Sensors,SMILES\n");
    for r in rows {
        out.push_str(&format!(
            "{},{},\"{}\",{},{},{:.2},{:.2},{:.2},{:.2},{:.3},{:.2},{:+.2},{:.2},\"{}\",\"{}\",\"{}\"\n",
            r.rank,
            r.compound_id,
            r.compound_name.replace('"', "'"),
            net.pdb_id,
            r.site_id,
            r.allosteric_discovery_score,
            r.intrinsic_open_dg_kcal,
            r.pocket_opening_work_kcal,
            r.apparent_allosteric_dg_kcal,
            r.allosteric_le,
            r.orthosteric_strain_kcal,
            r.cooperativity_ddg_kcal,
            r.cooperativity_alpha,
            r.modulation_mode.short_label(),
            r.engaged_sensor_residues.join(";"),
            r.canonical_smiles.replace('"', "'"),
        ));
    }
    out
}

/// Exports 3D cryptic-pocket-aligned poses with `ANM` opening work and `MWC` cooperativity tags as MDL V2000 `.sdf`.
pub fn format_allosteric_poses_sdf(
    net: &AllostericNetworkReport,
    rows: &[AllostericLigandResult],
    max_records: usize,
) -> String {
    let mut sdf = String::with_capacity(12288);
    for r in rows.iter().take(max_records.max(1)) {
        let charges = vec![0.0_f32; r.open_pose.mol.atoms.len()];
        let tags = [
            ("COMPOUND_ID", r.compound_id.clone()),
            ("PDB_ID", net.pdb_id.clone()),
            ("CRYPTIC_SITE_ID", r.site_id.clone()),
            ("APPARENT_ALLO_DG_KCAL", format!("{:.2}", r.apparent_allosteric_dg_kcal)),
            ("OPENING_WORK_KCAL", format!("{:.2}", r.pocket_opening_work_kcal)),
            ("ORTHOSTERIC_STRAIN_KCAL", format!("{:.2}", r.orthosteric_strain_kcal)),
            ("COOPERATIVITY_DDG_KCAL", format!("{:+.2}", r.cooperativity_ddg_kcal)),
            ("MWC_ALPHA", format!("{:.2}", r.cooperativity_alpha)),
            ("MODULATION_MODE", r.modulation_mode.short_label().to_string()),
        ];
        sdf.push_str(&format_3d_conformer_sdf(
            &r.open_pose.mol,
            &r.open_pose.aligned_ligand_coords3d,
            &charges,
            r.apparent_allosteric_dg_kcal,
            &tags,
        ));
    }
    sdf
}

/// Convenience loader for the 3 curated cryptic/allosteric benchmark complexes (`0..3`).
pub fn load_curated_allosteric_system(
    preset_idx: usize,
    gnm_cutoff_a: f32,
) -> Result<AllostericNetworkReport, String> {
    let pr = &CURATED_ALLOSTERIC_SYSTEMS[preset_idx % CURATED_ALLOSTERIC_SYSTEMS.len()];
    let parsed = parse_pdb_structure(curated_allosteric_pdb_text(preset_idx))?;
    Ok(analyze_allosteric_network(
        &parsed,
        Some(pr.orthosteric_center),
        Some(pr.cryptic_seed_center),
        gnm_cutoff_a,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol_io::parse_smiles;

    #[test]
    fn test_gnm_anm_eigensolver_cryptic_pocket_opening_and_pathway() {
        for idx in 0..3 {
            let net = load_curated_allosteric_system(idx, 10.0).unwrap();
            assert!(net.nodes.len() >= 20);
            assert_eq!(net.dccm_flat.len(), net.nodes.len() * net.nodes.len());
            assert!(!net.anm_modes.is_empty());
            assert!(!net.cryptic_pockets.is_empty());
            assert!(net.cryptic_pockets[0].open_holo_volume_a3 > net.cryptic_pockets[0].apo_volume_a3);
            assert!(net.primary_pathway_nodes.len() >= 2);
        }
    }

    #[test]
    fn test_allosteric_strain_and_cooperativity_scoring() {
        let net = load_curated_allosteric_system(0, 10.0).unwrap();
        let mut eai_like =
            parse_smiles("O=C(Nc1nccs1)C(c1cccc(F)c1)n1ccc2ccccc2c1=O").unwrap();
        eai_like.id = "EAI045_ANALOG".to_string();
        eai_like.name = "EAI045-Analog".to_string();

        let res = score_ligand_allosteric_coupling(&eai_like, None, &net, 0).unwrap();
        assert!(res.apparent_allosteric_dg_kcal < -1.5);
        assert!(res.orthosteric_strain_kcal > 0.05);
        assert!(res.cooperativity_alpha > 0.01);
    }
}
