//! DNA-Encoded (`DEL`) & Medicinal-Chemistry Parallel 2-Step and 3-Step Combinatorial Library
//! Enumerator (`Amide`, `Suzuki-Miyaura`, `Buchwald-Hartwig` / `SNAr`, and `CuAAC Click`),
//! On-DNA Aqueous Compatibility Auditor, `MaxMin` `ECFP4` Sphere-Exclusion Diversity Selector,
//! and 3D `PDB Pocket` / `ROCS` Pre-Filtering Engine (`parallel_lib.rs`, `v2.4.0`).
//!
//! Primary References:
//! - Clark et al. (2009), *Nat. Chem. Biol.* 5(9), 647-654 ("Design, synthesis and selection of
//!   DNA-encoded small-molecule libraries": `https://pubmed.ncbi.nlm.nih.gov/19648931/`)
//! - Goodnow, Dumelin & Keefe (2017), *Nat. Rev. Drug Discov.* 16(2), 131-147 ("DNA-encoded
//!   chemistry: enabling the deeper sampling of chemical space":
//!   `https://pubmed.ncbi.nlm.nih.gov/27932801/`)
//! - Gironda-Martínez et al. (2021), *Chem. Rev.* / PMC ("DNA-Encoded Chemical Libraries: A
//!   Comprehensive Review with Succinctly Illustrated Synthetic Cycles & On-DNA Amide, Suzuki,
//!   Buchwald, and CuAAC Click Reactions": `https://pmc.ncbi.nlm.nih.gov/articles/PMC8369695/`)
//!   & 3-Cycle DNA Codon Barcode Design (`https://pmc.ncbi.nlm.nih.gov/articles/PMC7070942/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC10840438/`)
//! - Roughley & Jordan (2011), *J. Med. Chem.* 54(10), 3451-3479 ("The medicinal chemist's
//!   toolbox: an analysis of reactions used in the pursuit of drug candidates":
//!   `https://pubmed.ncbi.nlm.nih.gov/21504168/`) & Parallel `C(sp2)-C(sp3)` Library Synthesis
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC7153271/`)
//! - Ashton et al. (2002) / `MaxMin` `ECFP4` Dissimilarity Diversity Selection & `Enamine REAL`
//!   Make-on-Demand Combinatorial Space Filtering (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8067665/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC10752525/`)

use crate::alerts::scan_structural_alerts;
use crate::conformer3d::{
    align_rocs_3d, evaluate_3d_summary_profile, format_3d_conformer_sdf, usrcat_similarity,
    Summary3dProfile,
};
use crate::fingerprint::{compute_ecfp, Ecfp2048};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    score_ligand_in_pocket, PocketCavityAnalysis, PocketComplementarityResult,
};
use crate::scaffold::murcko_signatures;
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Library synthesis execution format: split-and-pool on-DNA aqueous `DEL` vs plate-based parallel array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LibrarySynthesisMode {
    /// Split-and-pool DNA-Encoded Library (`DEL`) with dsDNA headpiece linker scar and 3-codon barcode tags.
    DnaEncodedOnDna,
    /// 96/384-well solution-phase medicinal-chemistry parallel synthesis plate (`Well A01..H12`).
    SolutionPhaseParallel,
}

impl LibrarySynthesisMode {
    pub const ALL: [Self; 2] = [Self::DnaEncodedOnDna, Self::SolutionPhaseParallel];

    pub fn label(self) -> &'static str {
        match self {
            Self::DnaEncodedOnDna => "DNA-Encoded Split-and-Pool (On-DNA DEL)",
            Self::SolutionPhaseParallel => "Medicinal-Chemistry Parallel Plate (96-Well)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::DnaEncodedOnDna => "On-DNA DEL",
            Self::SolutionPhaseParallel => "MedChem Plate",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::DnaEncodedOnDna => [56, 189, 248],       // Sky Cyan
            Self::SolutionPhaseParallel => [192, 132, 252], // Violet
        }
    }
}

/// Core medicinal-chemistry & `DEL` reaction classes (`Roughley & Jordan 2011`; `Gironda-Martínez 2021`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParallelReactionKind {
    /// `HATU` / `DMTMM` amide bond formation (`R-COOH + HNR1R2 -> R-C(=O)NR1R2`).
    AmideCoupling,
    /// `Pd(dppf)Cl2` / `sSPhos Pd G2` Suzuki-Miyaura coupling (`Ar-X + R-B(OH)2 -> Ar-R`).
    SuzukiMiyaura,
    /// `tBuBrettPhos Pd G3` / `SNAr` C-N amination (`Ar-X + HNR1R2 -> Ar-NR1R2`).
    BuchwaldHartwig,
    /// `CuSO4 / Na-ascorbate / TBTA` regioselective `1,4`-disubstituted `1,2,3`-triazole cycloaddition (`R1-N3 + HC#C-R2`).
    CuAacClick,
}

impl ParallelReactionKind {
    pub const ALL: [Self; 4] = [
        Self::AmideCoupling,
        Self::SuzukiMiyaura,
        Self::BuchwaldHartwig,
        Self::CuAacClick,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::AmideCoupling => "Amide Coupling (DMTMM / HATU)",
            Self::SuzukiMiyaura => "Suzuki-Miyaura (Pd / sSPhos)",
            Self::BuchwaldHartwig => "Buchwald-Hartwig / SNAr (C-N)",
            Self::CuAacClick => "CuAAC Click (1,4-Triazole)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::AmideCoupling => "Amide",
            Self::SuzukiMiyaura => "Suzuki",
            Self::BuchwaldHartwig => "Buchwald",
            Self::CuAacClick => "Click",
        }
    }

    pub fn on_dna_conditions(self) -> &'static str {
        match self {
            Self::AmideCoupling => "DMTMM or EDC/HOAt, pH 8.5 borate buffer / DMA (25 C, 4 h)",
            Self::SuzukiMiyaura => "Pd(OAc)2 / sSPhos (10 mol%), K2CO3, H2O/MeCN (80 C, 2 h)",
            Self::BuchwaldHartwig => "tBuXPhos Pd G3 / CsOH or thermal SNAr, H2O/DMA (65 C, 4 h)",
            Self::CuAacClick => "CuSO4 + Na-ascorbate + TBTA ligand, pH 7.4 PBS/DMSO (25 C, 2 h)",
        }
    }

    pub fn solution_conditions(self) -> &'static str {
        match self {
            Self::AmideCoupling => "HATU (1.2 eq), DIPEA (3.0 eq), DMF (25 C, 2 h)",
            Self::SuzukiMiyaura => "Pd(dppf)Cl2 (5 mol%), K3PO4, 1,4-dioxane/H2O (90 C, 3 h)",
            Self::BuchwaldHartwig => "tBuBrettPhos Pd G3 (5 mol%), NaOtBu or Cs2CO3, dioxane (85 C, 4 h)",
            Self::CuAacClick => "CuSO4 (5 mol%), Na-ascorbate (20 mol%), tBuOH/H2O (25 C, 2 h)",
        }
    }

    /// Baseline single-step aqueous on-DNA conversion & oligonucleotide integrity yield (`0..1`).
    pub fn baseline_on_dna_yield(self) -> f64 {
        match self {
            Self::AmideCoupling => 0.94,
            Self::SuzukiMiyaura => 0.87,
            Self::BuchwaldHartwig => 0.82,
            Self::CuAacClick => 0.95,
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::AmideCoupling => [52, 211, 153],   // Emerald
            Self::SuzukiMiyaura => [56, 189, 248],   // Sky Blue
            Self::BuchwaldHartwig => [251, 191, 36], // Amber
            Self::CuAacClick => [244, 114, 182],     // Rose Pink
        }
    }
}

/// Upstream 3D & diversity pre-filtering / ranking objective for combinatorial cherry-picking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LibraryPreFilterMode {
    /// Joint 3D `PDB Pocket` complementarity + 3D `ROCS` shape-color overlay + `MaxMin` diversity.
    PocketAndRocsHybrid,
    /// Structure-based 3D `PDB Pocket` `Delta G`, unhappy `WaterMap` displacement & receptor pharmacophore recall.
    PdbPocketComplementarity,
    /// Ligand-based 3D `ROCS` Gaussian shape + color Tanimoto (`ComboScore`) & `USRCAT` similarity.
    Rocs3dOverlay,
    /// Pure `Ashton (2002)` `MaxMin` `ECFP4` dissimilarity sphere-exclusion + Murcko scaffold coverage.
    MaxMinDiversityFirst,
}

impl LibraryPreFilterMode {
    pub const ALL: [Self; 4] = [
        Self::PocketAndRocsHybrid,
        Self::PdbPocketComplementarity,
        Self::Rocs3dOverlay,
        Self::MaxMinDiversityFirst,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::PocketAndRocsHybrid => "3D Pocket + ROCS Hybrid + MaxMin",
            Self::PdbPocketComplementarity => "3D PDB Pocket & WaterMap Pre-Filter",
            Self::Rocs3dOverlay => "3D ROCS Shape-Color Pre-Filter",
            Self::MaxMinDiversityFirst => "MaxMin ECFP4 Diversity Plate First",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::PocketAndRocsHybrid => "Pocket + ROCS",
            Self::PdbPocketComplementarity => "PDB Pocket",
            Self::Rocs3dOverlay => "3D ROCS",
            Self::MaxMinDiversityFirst => "MaxMin Diversity",
        }
    }
}

/// Functional synthon class for building-block reagent plates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingBlockClass {
    PrimaryOrSecondaryAmine,
    CarboxylicAcid,
    BoronicAcid,
    OrganicAzide,
    TerminalAlkyne,
}

impl BuildingBlockClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::PrimaryOrSecondaryAmine => "Amine (R1R2NH)",
            Self::CarboxylicAcid => "Carboxylic Acid (R-COOH)",
            Self::BoronicAcid => "Boronic Acid (R-B(OH)2)",
            Self::OrganicAzide => "Organic Azide (R-N3)",
            Self::TerminalAlkyne => "Terminal Alkyne (R-C#CH)",
        }
    }
}

/// Curated reagent synthon entry with pre-assigned 11-bp dsDNA codon sequence (`GC 45-55%`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildingBlockReagent {
    pub code: String,
    pub name: String,
    pub smiles: String,
    pub synthon_class: BuildingBlockClass,
    pub dna_codon_11bp: String,
    pub dna_safe: bool,
    pub sp3_enriched: bool,
}

/// Curated 2-step or 3-step orthogonal combinatorial reaction scheme metadata.
pub struct CuratedLibraryScheme {
    pub code: &'static str,
    pub title: &'static str,
    pub default_mode: LibrarySynthesisMode,
    pub step_count: usize,
    pub steps: &'static [ParallelReactionKind],
    /// Orthogonal central scaffold SMILES (carrying selective `COOH` / `I` / `Br` / `Cl` / `C#CH` handles).
    pub core_smiles: &'static str,
    pub core_name: &'static str,
    pub dna_headpiece_note: &'static str,
    pub description: &'static str,
}

pub const CURATED_LIBRARY_SCHEMES: [CuratedLibraryScheme; 6] = [
    CuratedLibraryScheme {
        code: "DEL-3C-AMIDE-SUZUKI-CLICK",
        title: "3-Cycle DEL: Amide -> Suzuki -> CuAAC Click",
        default_mode: LibrarySynthesisMode::DnaEncodedOnDna,
        step_count: 3,
        steps: &[
            ParallelReactionKind::AmideCoupling,
            ParallelReactionKind::SuzukiMiyaura,
            ParallelReactionKind::CuAacClick,
        ],
        core_smiles: "O=C(O)c1cc(I)cc(C#C)c1",
        core_name: "3-ethynyl-5-iodobenzoic acid",
        dna_headpiece_note: "5'-AOP-HP-NH2 acylated at C1-COOH; orthogonal C3-I Suzuki (Cycle 2) then C5-ethyne CuAAC Click (Cycle 3)",
        description: "Classic split-and-pool 3-cycle DEL cascade combining DMTMM acylation, sSPhos Pd Suzuki coupling, and TBTA-protected CuAAC 1,4-triazole click",
    },
    CuratedLibraryScheme {
        code: "DEL-3C-AMIDE-SUZUKI-BUCHWALD",
        title: "3-Cycle DEL: Amide -> Suzuki -> Buchwald/SNAr",
        default_mode: LibrarySynthesisMode::DnaEncodedOnDna,
        step_count: 3,
        steps: &[
            ParallelReactionKind::AmideCoupling,
            ParallelReactionKind::SuzukiMiyaura,
            ParallelReactionKind::BuchwaldHartwig,
        ],
        core_smiles: "O=C(O)c1cc(Br)cnc1Cl",
        core_name: "5-bromo-2-chloronicotinic acid",
        dna_headpiece_note: "Chemoselective acylation -> C5-Br Pd Suzuki -> C2-Cl Buchwald/SNAr amination on pyridine core",
        description: "Orthogonal dihalonicotinamide 3-cycle DEL generating dense H-bond acceptor/donor kinase & GPCR pharmacophores",
    },
    CuratedLibraryScheme {
        code: "MEDCHEM-2C-QUINAZOLINE",
        title: "2-Step Kinase Plate: Suzuki -> Buchwald (Quinazoline)",
        default_mode: LibrarySynthesisMode::SolutionPhaseParallel,
        step_count: 2,
        steps: &[
            ParallelReactionKind::SuzukiMiyaura,
            ParallelReactionKind::BuchwaldHartwig,
        ],
        core_smiles: "Clc1ncnc2cc(Br)ccc12",
        core_name: "6-bromo-4-chloroquinazoline",
        dna_headpiece_note: "96-well parallel ATP-hinge array (C6-Br Suzuki -> C4-Cl aniline/3D-amine Buchwald/SNAr)",
        description: "Focused 2-step EGFR/kinase ATP-hinge parallel array directly complementary to Met793 & Thr790 gatekeeper pockets",
    },
    CuratedLibraryScheme {
        code: "MEDCHEM-2C-AMIDE-CLICK",
        title: "2-Step sp3-Click Plate: Amide -> CuAAC Triazole",
        default_mode: LibrarySynthesisMode::SolutionPhaseParallel,
        step_count: 2,
        steps: &[
            ParallelReactionKind::AmideCoupling,
            ParallelReactionKind::CuAacClick,
        ],
        core_smiles: "C#Cc1ccc(CC(=O)O)cc1",
        core_name: "2-(4-ethynylphenyl)acetic acid",
        dna_headpiece_note: "96-well modular HATU acylation followed by regioselective CuAAC 1,4-triazole assembly",
        description: "High-reliability 2-step click medicinal-chemistry plate pairing 3D sp3 amines with polar/aromatic azides",
    },
    CuratedLibraryScheme {
        code: "MEDCHEM-3C-PYRIMIDINE",
        title: "3-Step MedChem Array: Amide -> Suzuki -> Buchwald",
        default_mode: LibrarySynthesisMode::SolutionPhaseParallel,
        step_count: 3,
        steps: &[
            ParallelReactionKind::AmideCoupling,
            ParallelReactionKind::SuzukiMiyaura,
            ParallelReactionKind::BuchwaldHartwig,
        ],
        core_smiles: "O=C(O)c1nc(Cl)ncc1Br",
        core_name: "5-bromo-2-chloropyrimidine-4-carboxylic acid",
        dna_headpiece_note: "Trifunctional pyrimidine-4-carboxamide cascade with selective C5-Br Suzuki then C2-Cl amination",
        description: "Three-vector heteroaromatic lead-optimization array exploring exit vectors around a compact 2,4,5-trisubstituted pyrimidine",
    },
    CuratedLibraryScheme {
        code: "DEL-2C-IMIDAZOPYRIDINE",
        title: "2-Cycle Rule-of-4 DEL: Amide -> Suzuki",
        default_mode: LibrarySynthesisMode::DnaEncodedOnDna,
        step_count: 2,
        steps: &[
            ParallelReactionKind::AmideCoupling,
            ParallelReactionKind::SuzukiMiyaura,
        ],
        core_smiles: "O=C(O)c1cn2cc(Br)ccc2n1",
        core_name: "6-bromoimidazo[1,2-a]pyridine-2-carboxylic acid",
        dna_headpiece_note: "Low-MW Rule-of-4 2-cycle DEL avoiding molecular-weight inflation (`MW < 480 Da`)",
        description: "Compact bicyclic 2-cycle DEL tailored for oral/CNS-penetrant chemical space (`MW 310-470 Da`, low TPSA)",
    },
];

/// User-configurable parameters for 2/3-step library enumeration, `MaxMin` diversity, and 3D pre-filtering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParallelLibConfig {
    pub scheme_idx: usize,
    pub synthesis_mode: LibrarySynthesisMode,
    pub prefilter_mode: LibraryPreFilterMode,
    /// Number of reagents per cycle (`4..=10`; e.g., `6` gives `216` in 3-step, `10` gives `100` in 2-step).
    pub reagents_per_cycle: usize,
    /// Target cherry-picked diverse library plate size (`12..=96`, default `24`).
    pub target_diverse_plate_size: usize,
    /// Maximum pool of diversity-seeded property-passing products evaluated in full 3D Pocket + `ROCS` (`24..=120`).
    pub max_3d_scored_pool: usize,
    /// Physicochemical & DNA-compatibility guardrails.
    pub max_mw: f64,
    pub max_clogp: f64,
    pub min_fsp3: f64,
    pub exclude_pains: bool,
    /// Harvest additional primary/secondary amines from the active `DatasetAnalysis` into Cycle 1/3 amine plates.
    pub harvest_dataset_synthons: bool,
    /// Optional custom central core SMILES overriding the preset core.
    pub custom_core_smiles: Option<String>,
}

impl Default for ParallelLibConfig {
    fn default() -> Self {
        Self {
            scheme_idx: 0,
            synthesis_mode: LibrarySynthesisMode::DnaEncodedOnDna,
            prefilter_mode: LibraryPreFilterMode::PocketAndRocsHybrid,
            reagents_per_cycle: 6,
            target_diverse_plate_size: 24,
            max_3d_scored_pool: 64,
            max_mw: 580.0,
            max_clogp: 5.2,
            min_fsp3: 0.12,
            exclude_pains: true,
            harvest_dataset_synthons: false,
            custom_core_smiles: None,
        }
    }
}

/// Individual enumerated combinatorial product with barcode/well tag, `MaxMin` diversity status, and 3D scores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnumeratedLibraryMember {
    pub rank: usize,
    pub library_id: String,
    /// Either `"DEL01-A02-B04-C01 (5'-GTCACGTAGCA:ATCGGCTACGA:TGCATCGACGT-3')"` or `"Plate-1 Well B05"`.
    pub barcode_or_well: String,
    pub cycle_reagent_codes: Vec<String>,
    pub cycle_reagent_names: Vec<String>,
    pub reaction_route: String,
    pub canonical_smiles: String,
    pub murcko_scaffold: String,

    // Physicochemical & Synthetic Tractability Metrics
    pub mw: f64,
    pub clogp: f64,
    pub tpsa: f64,
    pub fsp3: f64,
    pub hbd: usize,
    pub hba: usize,
    pub rotatable_bonds: usize,
    pub qed: f64,
    pub cns_mpo: f64,
    pub sa_score: f64,
    pub pains_alerts: usize,
    /// Cumulative multi-step aqueous on-DNA fidelity or solution-phase conversion (`0..100%`).
    pub synthetic_fidelity_pct: f64,
    pub dna_audit_note: String,

    // MaxMin ECFP4 Diversity Metrics
    pub selected_in_diverse_plate: bool,
    pub diversity_pick_order: Option<usize>,
    /// Maximum `ECFP4` Tanimoto similarity to another member inside the selected diverse plate (`lower = more diverse`).
    pub nearest_plate_tanimoto: f32,

    // 3D ROCS & USRCAT Pre-Filter Readouts
    pub usrcat_sim: f64,
    pub rocs_combo_score: f64,
    pub rocs_shape_tanimoto: f64,
    pub rocs_color_tanimoto: f64,

    // 3D PDB Pocket & WaterMap Pre-Filter Readouts
    pub pocket_complementarity_score: f64,
    pub pocket_dg_kcal: f64,
    pub pocket_le: f64,
    pub pocket_lle: f64,
    pub displaced_unhappy_waters: usize,
    pub pharmacophore_recall_pct: f64,

    /// Overall `[0..100]` multi-objective library triage priority score.
    pub composite_priority_score: f64,

    pub rocs_aligned_coords3d: Vec<[f32; 3]>,
    pub pocket_pose: Option<PocketComplementarityResult>,
    pub mol: Molecule,
}

/// Complete combinatorial enumeration, `MaxMin` diversity, and 3D Pocket/`ROCS` pre-filtering report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParallelLibraryReport {
    pub scheme_code: String,
    pub scheme_title: String,
    pub synthesis_mode: LibrarySynthesisMode,
    pub prefilter_mode: LibraryPreFilterMode,
    pub step_count: usize,
    pub reaction_steps: Vec<ParallelReactionKind>,
    pub core_name: String,
    pub core_smiles: String,
    pub reference_query_name: String,
    pub reference_pocket_pdb: Option<String>,

    // Funnel & Diversity Compression Statistics
    pub theoretical_combinatorial_size: usize,
    pub valid_unique_enumerated: usize,
    pub property_filter_passed: usize,
    pub scored_3d_count: usize,
    pub selected_diverse_plate_count: usize,
    pub unique_murcko_scaffolds_all: usize,
    pub unique_murcko_scaffolds_plate: usize,
    pub mean_pairwise_dissimilarity_all: f64,
    pub mean_pairwise_dissimilarity_plate: f64,
    pub mean_fsp3_plate: f64,
    pub mean_mw_plate: f64,
    pub mean_fidelity_pct_plate: f64,
    pub pocket_hit_count: usize,
    pub rocs_hit_count: usize,

    pub cycle_plates: Vec<Vec<BuildingBlockReagent>>,
    pub members: Vec<EnumeratedLibraryMember>,
}

// ---------------------------------------------------------------------------
// Curated DNA-Compatible & MedChem Building-Block Reagent Plates (10 per class)
// ---------------------------------------------------------------------------

/// Curated primary & secondary amine building blocks (`Amide` & `Buchwald-Hartwig` / `SNAr` synthons).
pub fn curated_amine_plate() -> Vec<BuildingBlockReagent> {
    vec![
        BuildingBlockReagent {
            code: "AM01".to_string(),
            name: "Cyclopropylmethanamine".to_string(),
            smiles: "NCC1CC1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "GTCACGTAGCA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM02".to_string(),
            name: "Morpholine".to_string(),
            smiles: "C1COCCN1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "ATCGGCTACGA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM03".to_string(),
            name: "3-Fluoroazetidine".to_string(),
            smiles: "FC1CNC1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "TGCATCGACGT".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM04".to_string(),
            name: "(Tetrahydrofuran-3-yl)methanamine".to_string(),
            smiles: "NCC1CCOC1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "CGATAGCTGCA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM05".to_string(),
            name: "1-Methylpiperazine".to_string(),
            smiles: "CN1CCNCC1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "GCTACGTATCG".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM06".to_string(),
            name: "3-Ethynylaniline".to_string(),
            smiles: "Nc1cccc(C#C)c1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "TAGCGATCGAC".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "AM07".to_string(),
            name: "4-Fluoro-3-chloroaniline".to_string(),
            smiles: "Nc1ccc(F)c(Cl)c1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "CTGACGTAGCG".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "AM08".to_string(),
            name: "(1-Methyl-1H-pyrazol-4-yl)methanamine".to_string(),
            smiles: "Cn1cc(CN)cn1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "GACGTCATGCA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM09".to_string(),
            name: "Oxetan-3-amine".to_string(),
            smiles: "NC1COC1".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "ACGTAGCTGAC".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AM10".to_string(),
            name: "Bicyclo[1.1.1]pentan-1-amine".to_string(),
            smiles: "NC12CC(C1)C2".to_string(),
            synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
            dna_codon_11bp: "TCAGCGATGCA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
    ]
}

/// Curated boronic acid / trifluoroborate building blocks (`Suzuki-Miyaura` synthons).
pub fn curated_boronic_acid_plate() -> Vec<BuildingBlockReagent> {
    vec![
        BuildingBlockReagent {
            code: "BA01".to_string(),
            name: "(1-Methyl-1H-pyrazol-4-yl)boronic acid".to_string(),
            smiles: "Cn1cc(B(O)O)cn1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "CAGTCGATGCA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "BA02".to_string(),
            name: "Pyridin-3-ylboronic acid".to_string(),
            smiles: "OB(O)c1cccnc1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "GTACGCATGAC".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "BA03".to_string(),
            name: "(4-Fluorophenyl)boronic acid".to_string(),
            smiles: "OB(O)c1ccc(F)cc1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "TGACGCTAGCA".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "BA04".to_string(),
            name: "Cyclopropylboronic acid (sp3)".to_string(),
            smiles: "OB(O)C1CC1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "ATCGCTAGACG".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "BA05".to_string(),
            name: "(2-Methoxypyrimidin-5-yl)boronic acid".to_string(),
            smiles: "COc1ncc(B(O)O)cn1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "GCTAGCATCGA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "BA06".to_string(),
            name: "(3-(Methylsulfonyl)phenyl)boronic acid".to_string(),
            smiles: "CS(=O)(=O)c1cccc(B(O)O)c1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "CGATCGATGAC".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "BA07".to_string(),
            name: "(1H-Indazol-5-yl)boronic acid".to_string(),
            smiles: "OB(O)c1ccc2[nH]ncc2c1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "TACGCATCGAG".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "BA08".to_string(),
            name: "(3-Cyanophenyl)boronic acid".to_string(),
            smiles: "N#Cc1cccc(B(O)O)c1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "GACTAGCATGC".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "BA09".to_string(),
            name: "(6-Methylpyridin-3-yl)boronic acid".to_string(),
            smiles: "Cc1ccc(B(O)O)cn1".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "AGCTGACGTCA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "BA10".to_string(),
            name: "(3,5-Dimethylisoxazol-4-yl)boronic acid".to_string(),
            smiles: "Cc1noc(C)c1B(O)O".to_string(),
            synthon_class: BuildingBlockClass::BoronicAcid,
            dna_codon_11bp: "CTAGCGATCGA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
    ]
}

/// Curated organic azide building blocks (`CuAAC Click` `1,4`-triazole synthons).
pub fn curated_azide_plate() -> Vec<BuildingBlockReagent> {
    vec![
        BuildingBlockReagent {
            code: "AZ01".to_string(),
            name: "2-Azidoethanol".to_string(),
            smiles: "[N-]=[N+]=NCCO".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "GTCATGACGTA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ02".to_string(),
            name: "(Azidomethyl)cyclopropane".to_string(),
            smiles: "[N-]=[N+]=NCC1CC1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "CATGACGTCGA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ03".to_string(),
            name: "3-Azidooxetane".to_string(),
            smiles: "[N-]=[N+]=NC1COC1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "TACGTCAGCTA".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ04".to_string(),
            name: "4-(Azidomethyl)pyridine".to_string(),
            smiles: "[N-]=[N+]=NCc1ccncc1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "GCATGCTAGAC".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ05".to_string(),
            name: "1-(Azidomethyl)-4-fluorobenzene".to_string(),
            smiles: "[N-]=[N+]=NCc1ccc(F)cc1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "ATCGTAGCGAC".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ06".to_string(),
            name: "2-Azidoacetamide".to_string(),
            smiles: "[N-]=[N+]=NCC(=O)N".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "CGACTAGCATG".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ07".to_string(),
            name: "4-(2-Azidoethyl)morpholine".to_string(),
            smiles: "[N-]=[N+]=NCCN1CCOCC1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "TAGCATCGACG".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ08".to_string(),
            name: "4-Azidobenzenesulfonamide".to_string(),
            smiles: "[N-]=[N+]=Nc1ccc(S(=O)(=O)N)cc1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "GCTACGATGCA".to_string(),
            dna_safe: true,
            sp3_enriched: false,
        },
        BuildingBlockReagent {
            code: "AZ09".to_string(),
            name: "3-Azidotetrahydrofuran".to_string(),
            smiles: "[N-]=[N+]=NC1CCOC1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "ACGTCGATAGC".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
        BuildingBlockReagent {
            code: "AZ10".to_string(),
            name: "4-(Azidomethyl)-1-methyl-1H-pyrazole".to_string(),
            smiles: "Cn1cc(CN=[N+]=[N-])cn1".to_string(),
            synthon_class: BuildingBlockClass::OrganicAzide,
            dna_codon_11bp: "TCATGCTAGCG".to_string(),
            dna_safe: true,
            sp3_enriched: true,
        },
    ]
}

// ---------------------------------------------------------------------------
// Molecular Graph Reaction Operators (Amide, Suzuki, Buchwald, CuAAC Click)
// ---------------------------------------------------------------------------

/// Appends all atoms and bonds of `frag` (excluding atoms in `skip_frag_atoms`) into `dst`,
/// returning the mapping from `frag` atom indices to new `dst` atom indices.
fn merge_fragment_into(
    dst: &mut Molecule,
    frag: &Molecule,
    skip_frag_atoms: &BTreeSet<usize>,
) -> BTreeMap<usize, usize> {
    let mut idx_map = BTreeMap::new();
    for (fi, a) in frag.atoms.iter().enumerate() {
        if skip_frag_atoms.contains(&fi) {
            continue;
        }
        let new_idx = dst.atoms.len();
        let mut na = a.clone();
        na.idx = new_idx;
        // Clear aromatic flags on merged raw graph so sanitize + Kekulize/OpenSMILES round-trip stays clean
        dst.atoms.push(na);
        idx_map.insert(fi, new_idx);
    }
    for b in &frag.bonds {
        if let (Some(&na), Some(&nb)) = (idx_map.get(&b.a), idx_map.get(&b.b)) {
            let b_idx = dst.bonds.len();
            let order = if b.aromatic {
                if b.kekule_order == 2 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                }
            } else {
                b.order
            };
            dst.bonds.push(Bond {
                idx: b_idx,
                a: na,
                b: nb,
                order,
                aromatic: false,
                in_ring: false,
                kekule_order: b.kekule_order.clamp(1, 3),
            });
        }
    }
    for a in &mut dst.atoms {
        a.aromatic = false;
    }
    for b in &mut dst.bonds {
        if b.order == BondOrder::Aromatic {
            b.order = if b.kekule_order == 2 {
                BondOrder::Double
            } else {
                BondOrder::Single
            };
        }
        b.aromatic = false;
    }
    idx_map
}

fn finalize_product_graph(mut raw: Molecule) -> Option<Molecule> {
    for (i, a) in raw.atoms.iter_mut().enumerate() {
        a.idx = i;
        // Let OpenSMILES round-trip recompute exact implicit hydrogens unless charged
        if a.charge == 0 {
            a.bracketed = false;
            a.explicit_h = 0;
        }
    }
    for (i, b) in raw.bonds.iter_mut().enumerate() {
        b.idx = i;
    }
    raw.sanitize();
    let smi = molecule_to_smiles(&raw);
    let mut clean = parse_smiles(&smi).ok()?;
    compute_2d_coords(&mut clean);
    clean.smiles = molecule_to_smiles(&clean);
    Some(clean)
}

/// Locates a carboxylic acid `(carbonyl_c_idx, oh_o_idx)` in `mol`.
fn find_carboxylic_acid_site(mol: &Molecule) -> Option<(usize, usize)> {
    for (ci, a) in mol.atoms.iter().enumerate() {
        if a.elem != Element::C || a.in_ring {
            continue;
        }
        let mut has_carbonyl_o = false;
        let mut oh_idx = None;
        for &(nb, bi) in &mol.adj[ci] {
            let b = &mol.bonds[bi];
            let na = &mol.atoms[nb];
            if na.elem == Element::O && b.kekule_order == 2 && mol.adj[nb].len() == 1 {
                has_carbonyl_o = true;
            } else if na.elem == Element::O
                && b.kekule_order == 1
                && mol.adj[nb].len() == 1
                && (na.total_h() >= 1 || na.charge < 0)
            {
                oh_idx = Some(nb);
            }
        }
        if has_carbonyl_o {
            if let Some(oh) = oh_idx {
                return Some((ci, oh));
            }
        }
    }
    None
}

/// Locates a nucleophilic primary or secondary amine nitrogen `(n_idx)` (excluding amides, nitro, azides, nitriles).
fn find_nucleophilic_amine_site(mol: &Molecule) -> Option<usize> {
    let mut best: Option<(u8, usize)> = None;
    for (ni, a) in mol.atoms.iter().enumerate() {
        if a.elem != Element::N || a.charge != 0 || a.aromatic || a.total_h() == 0 {
            continue;
        }
        // Exclude if attached to C=O, S=O, N, or multiple bond
        let invalid = mol.adj[ni].iter().any(|&(nb, bi)| {
            if mol.bonds[bi].kekule_order != 1 {
                return true;
            }
            let na = &mol.atoms[nb];
            if matches!(na.elem, Element::N | Element::O | Element::S) {
                return true;
            }
            // Check if neighbor C is a carbonyl / imine / nitrile
            mol.adj[nb].iter().any(|&(nn, nbi)| {
                nn != ni
                    && mol.bonds[nbi].kekule_order >= 2
                    && matches!(mol.atoms[nn].elem, Element::O | Element::N | Element::S)
            })
        });
        if invalid {
            continue;
        }
        // Prefer aliphatic secondary/primary amines over anilines when multiple N exist (e.g. methylpiperazine / pyrazole-CH2NH2)
        let attached_aromatic = mol.adj[ni].iter().any(|&(nb, _)| mol.atoms[nb].aromatic);
        let priority = if !attached_aromatic && a.total_h() >= 2 {
            3
        } else if !attached_aromatic {
            2
        } else {
            1
        };
        if best.map_or(true, |(p, _)| priority > p) {
            best = Some((priority, ni));
        }
    }
    best.map(|(_, idx)| idx)
}

/// Locates the most reactive aryl/heteroaryl C-X coupling site (`I > Br > Cl`), returning `(c_idx, x_idx, Element)`.
fn find_aryl_halide_site(mol: &Molecule, allow_cl: bool) -> Option<(usize, usize, Element)> {
    let mut best: Option<(u8, usize, usize, Element)> = None;
    for (xi, a) in mol.atoms.iter().enumerate() {
        let rank = match a.elem {
            Element::I => 3,
            Element::Br => 2,
            Element::Cl if allow_cl => 1,
            _ => continue,
        };
        if mol.adj[xi].len() != 1 {
            continue;
        }
        let (ci, bi) = mol.adj[xi][0];
        if mol.bonds[bi].kekule_order != 1 {
            continue;
        }
        let ca = &mol.atoms[ci];
        if ca.elem == Element::C && (ca.aromatic || ca.hyb == Hybridization::Sp2) {
            if best.map_or(true, |(r, _, _, _)| rank > r) {
                best = Some((rank, ci, xi, a.elem));
            }
        }
    }
    best.map(|(_, c, x, el)| (c, x, el))
}

/// Locates a boronic acid/ester `C-B(OR)2` site, returning `(attached_c_idx, skip_boron_group_atoms)`.
fn find_boronic_acid_site(mol: &Molecule) -> Option<(usize, BTreeSet<usize>)> {
    for (bi_atom, a) in mol.atoms.iter().enumerate() {
        if a.elem != Element::B {
            continue;
        }
        let mut carbon_nb = None;
        let mut skip = BTreeSet::new();
        skip.insert(bi_atom);
        for &(nb, _) in &mol.adj[bi_atom] {
            if mol.atoms[nb].elem == Element::C && carbon_nb.is_none() {
                carbon_nb = Some(nb);
            } else if mol.atoms[nb].elem == Element::O && mol.adj[nb].len() == 1 {
                skip.insert(nb);
            }
        }
        if let Some(c_idx) = carbon_nb {
            return Some((c_idx, skip));
        }
    }
    None
}

/// Locates a terminal alkyne `R-C(internal)#CH(terminal)`, returning `(r_anchor_idx, internal_c, terminal_c)`.
fn find_terminal_alkyne_site(mol: &Molecule) -> Option<(usize, usize, usize)> {
    for b in &mol.bonds {
        if b.order != BondOrder::Triple && b.kekule_order != 3 {
            continue;
        }
        let (u, v) = (b.a, b.b);
        if mol.atoms[u].elem != Element::C || mol.atoms[v].elem != Element::C {
            continue;
        }
        if mol.adj[u].len() == 1 && mol.adj[v].len() == 2 {
            let anchor = mol.adj[v]
                .iter()
                .find(|&&(w, _)| w != u)
                .map(|&(w, _)| w)?;
            return Some((anchor, v, u));
        }
        if mol.adj[v].len() == 1 && mol.adj[u].len() == 2 {
            let anchor = mol.adj[u]
                .iter()
                .find(|&&(w, _)| w != v)
                .map(|&(w, _)| w)?;
            return Some((anchor, u, v));
        }
    }
    None
}

/// Locates an organic azide `R-N1=N2=N3` (`[N-]=[N+]=N-R`), returning `(r_anchor_idx, skip_azide_atoms)`.
fn find_organic_azide_site(mol: &Molecule) -> Option<(usize, BTreeSet<usize>)> {
    for (n2, a2) in mol.atoms.iter().enumerate() {
        if a2.elem != Element::N || mol.adj[n2].len() != 2 {
            continue;
        }
        let n_nbs: Vec<usize> = mol.adj[n2]
            .iter()
            .map(|&(w, _)| w)
            .filter(|&w| mol.atoms[w].elem == Element::N)
            .collect();
        if n_nbs.len() != 2 {
            continue;
        }
        let (na, nb) = (n_nbs[0], n_nbs[1]);
        // Terminal N has degree 1 (only bonded to n2); substituted N has degree 2 (bonded to n2 and R)
        let (n_sub, n_term) = if mol.adj[na].len() == 2 && mol.adj[nb].len() == 1 {
            (na, nb)
        } else if mol.adj[nb].len() == 2 && mol.adj[na].len() == 1 {
            (nb, na)
        } else {
            continue;
        };
        let r_anchor = mol.adj[n_sub]
            .iter()
            .find(|&&(w, _)| w != n2)
            .map(|&(w, _)| w)?;
        let mut skip = BTreeSet::new();
        skip.insert(n_sub);
        skip.insert(n2);
        skip.insert(n_term);
        return Some((r_anchor, skip));
    }
    None
}

/// Executes one authentic medicinal-chemistry / `DEL` coupling step (`rxn`) between `scaffold` and `reagent`.
pub fn apply_parallel_reaction_step(
    rxn: ParallelReactionKind,
    scaffold: &Molecule,
    reagent: &Molecule,
) -> Option<Molecule> {
    match rxn {
        ParallelReactionKind::AmideCoupling => {
            // Orientation A: scaffold-COOH + reagent-NHR
            if let (Some((c_acyl, oh_skip)), Some(n_amine)) = (
                find_carboxylic_acid_site(scaffold),
                find_nucleophilic_amine_site(reagent),
            ) {
                let mut prod = Molecule::new("");
                let mut sc_skip = BTreeSet::new();
                sc_skip.insert(oh_skip);
                let sc_map = merge_fragment_into(&mut prod, scaffold, &sc_skip);
                let rg_map = merge_fragment_into(&mut prod, reagent, &BTreeSet::new());
                let u = *sc_map.get(&c_acyl)?;
                let v = *rg_map.get(&n_amine)?;
                let b_idx = prod.bonds.len();
                prod.bonds.push(Bond {
                    idx: b_idx,
                    a: u,
                    b: v,
                    order: BondOrder::Single,
                    aromatic: false,
                    in_ring: false,
                    kekule_order: 1,
                });
                return finalize_product_graph(prod);
            }
            // Orientation B: scaffold-NHR + reagent-COOH
            if let (Some(n_amine), Some((c_acyl, oh_skip))) = (
                find_nucleophilic_amine_site(scaffold),
                find_carboxylic_acid_site(reagent),
            ) {
                let mut prod = Molecule::new("");
                let sc_map = merge_fragment_into(&mut prod, scaffold, &BTreeSet::new());
                let mut rg_skip = BTreeSet::new();
                rg_skip.insert(oh_skip);
                let rg_map = merge_fragment_into(&mut prod, reagent, &rg_skip);
                let u = *sc_map.get(&n_amine)?;
                let v = *rg_map.get(&c_acyl)?;
                let b_idx = prod.bonds.len();
                prod.bonds.push(Bond {
                    idx: b_idx,
                    a: u,
                    b: v,
                    order: BondOrder::Single,
                    aromatic: false,
                    in_ring: false,
                    kekule_order: 1,
                });
                return finalize_product_graph(prod);
            }
            None
        }
        ParallelReactionKind::SuzukiMiyaura => {
            // Chemoselective C-I > C-Br > C-Cl coupling with R-B(OH)2
            let (c_hal, x_hal, _) = find_aryl_halide_site(scaffold, true)?;
            let (c_bor, bor_skip) = find_boronic_acid_site(reagent)?;
            let mut prod = Molecule::new("");
            let mut sc_skip = BTreeSet::new();
            sc_skip.insert(x_hal);
            let sc_map = merge_fragment_into(&mut prod, scaffold, &sc_skip);
            let rg_map = merge_fragment_into(&mut prod, reagent, &bor_skip);
            let u = *sc_map.get(&c_hal)?;
            let v = *rg_map.get(&c_bor)?;
            let b_idx = prod.bonds.len();
            prod.bonds.push(Bond {
                idx: b_idx,
                a: u,
                b: v,
                order: BondOrder::Single,
                aromatic: false,
                in_ring: false,
                kekule_order: 1,
            });
            finalize_product_graph(prod)
        }
        ParallelReactionKind::BuchwaldHartwig => {
            // C-X (I, Br, or Cl) + HNR1R2 -> Ar-NR1R2
            let (c_hal, x_hal, _) = find_aryl_halide_site(scaffold, true)?;
            let n_amine = find_nucleophilic_amine_site(reagent)?;
            let mut prod = Molecule::new("");
            let mut sc_skip = BTreeSet::new();
            sc_skip.insert(x_hal);
            let sc_map = merge_fragment_into(&mut prod, scaffold, &sc_skip);
            let rg_map = merge_fragment_into(&mut prod, reagent, &BTreeSet::new());
            let u = *sc_map.get(&c_hal)?;
            let v = *rg_map.get(&n_amine)?;
            let b_idx = prod.bonds.len();
            prod.bonds.push(Bond {
                idx: b_idx,
                a: u,
                b: v,
                order: BondOrder::Single,
                aromatic: false,
                in_ring: false,
                kekule_order: 1,
            });
            finalize_product_graph(prod)
        }
        ParallelReactionKind::CuAacClick => {
            // Regioselective Cu(I)-catalyzed 1,4-disubstituted 1,2,3-triazole formation:
            // R_alk - C4(=C5H - N1(R_az) - N2 = N3 - 4)
            let (alk_mol, az_mol) = if find_terminal_alkyne_site(scaffold).is_some()
                && find_organic_azide_site(reagent).is_some()
            {
                (scaffold, reagent)
            } else if find_organic_azide_site(scaffold).is_some()
                && find_terminal_alkyne_site(reagent).is_some()
            {
                (reagent, scaffold)
            } else {
                return None;
            };

            let (r_alk_anchor, c_int, c_term) = find_terminal_alkyne_site(alk_mol)?;
            let (r_az_anchor, az_skip) = find_organic_azide_site(az_mol)?;

            let mut prod = Molecule::new("");
            let mut alk_skip = BTreeSet::new();
            alk_skip.insert(c_int);
            alk_skip.insert(c_term);
            let alk_map = merge_fragment_into(&mut prod, alk_mol, &alk_skip);
            let az_map = merge_fragment_into(&mut prod, az_mol, &az_skip);

            let u_alk = *alk_map.get(&r_alk_anchor)?;
            let u_az = *az_map.get(&r_az_anchor)?;

            let mk_ring_atom = |idx: usize, elem: Element| Atom {
                idx,
                elem,
                charge: 0,
                aromatic: false,
                bracketed: false,
                explicit_h: 0,
                implicit_h: 0,
                chiral: 0,
                in_ring: true,
                ring_sizes: vec![5],
                hyb: Hybridization::Sp2,
                pos2d: [0.0, 0.0],
            };

            let c4 = prod.atoms.len();
            prod.atoms.push(mk_ring_atom(c4, Element::C));
            let c5 = prod.atoms.len();
            prod.atoms.push(mk_ring_atom(c5, Element::C));
            let n1 = prod.atoms.len();
            prod.atoms.push(mk_ring_atom(n1, Element::N));
            let n2 = prod.atoms.len();
            prod.atoms.push(mk_ring_atom(n2, Element::N));
            let n3 = prod.atoms.len();
            prod.atoms.push(mk_ring_atom(n3, Element::N));

            let add_bond = |m: &mut Molecule, a: usize, b: usize, ord: BondOrder, k: u8| {
                let idx = m.bonds.len();
                m.bonds.push(Bond {
                    idx,
                    a,
                    b,
                    order: ord,
                    aromatic: false,
                    in_ring: false,
                    kekule_order: k,
                });
            };

            // Substituent attachments: R_alk -> C4, R_az -> N1 (1,4-regioisomer)
            add_bond(&mut prod, u_alk, c4, BondOrder::Single, 1);
            add_bond(&mut prod, u_az, n1, BondOrder::Single, 1);
            // Kekulé 1,2,3-triazole perimeter: C4=C5 - N1 - N2=N3 - C4
            add_bond(&mut prod, c4, c5, BondOrder::Double, 2);
            add_bond(&mut prod, c5, n1, BondOrder::Single, 1);
            add_bond(&mut prod, n1, n2, BondOrder::Single, 1);
            add_bond(&mut prod, n2, n3, BondOrder::Double, 2);
            add_bond(&mut prod, n3, c4, BondOrder::Single, 1);

            finalize_product_graph(prod)
        }
    }
}

/// Returns the default reagent plate corresponding to a reaction step.
pub fn default_plate_for_reaction(rxn: ParallelReactionKind) -> Vec<BuildingBlockReagent> {
    match rxn {
        ParallelReactionKind::AmideCoupling | ParallelReactionKind::BuchwaldHartwig => {
            curated_amine_plate()
        }
        ParallelReactionKind::SuzukiMiyaura => curated_boronic_acid_plate(),
        ParallelReactionKind::CuAacClick => curated_azide_plate(),
    }
}

/// Audits multi-step on-DNA oligonucleotide preservation vs solution-phase conversion (`0..100%`).
fn evaluate_synthetic_fidelity(
    mode: LibrarySynthesisMode,
    steps: &[ParallelReactionKind],
    clogp: f64,
    mw: f64,
    pains_count: usize,
) -> (f64, String) {
    match mode {
        LibrarySynthesisMode::DnaEncodedOnDna => {
            let mut yield_prod: f64 = steps
                .iter()
                .map(|s| s.baseline_on_dna_yield())
                .product();
            let mut notes = Vec::new();
            if clogp > 4.6 {
                yield_prod *= 0.88;
                notes.push("elevated cLogP (>4.6) lowers aqueous borate/DMA micellar solubility");
            }
            if mw > 560.0 {
                yield_prod *= 0.93;
                notes.push("MW >560 Da inflates off-DNA DEL hit validation risk");
            }
            if steps.contains(&ParallelReactionKind::CuAacClick) {
                notes.push("TBTA Cu(I)-ligand shields dsDNA from hydroxyl-radical cleavage");
            }
            if steps.contains(&ParallelReactionKind::SuzukiMiyaura) {
                notes.push("sSPhos Pd scavenged via diethyldithiocarbamate prior to EtOH precipitation");
            }
            if pains_count > 0 {
                yield_prod *= 0.85;
                notes.push("structural alert flagged");
            }
            let pct = (yield_prod * 100.0).clamp(15.0, 99.0);
            let summary = if notes.is_empty() {
                "pH 7.4-9.0 dsDNA-safe (<2% depurination); PCR-amplifiable 3-codon tag".to_string()
            } else {
                notes.join("; ")
            };
            (pct, summary)
        }
        LibrarySynthesisMode::SolutionPhaseParallel => {
            let mut yield_prod: f64 = steps
                .iter()
                .map(|s| (s.baseline_on_dna_yield() + 0.04).min(0.97))
                .product();
            if clogp > 5.2 {
                yield_prod *= 0.90;
            }
            if pains_count > 0 {
                yield_prod *= 0.85;
            }
            let pct = (yield_prod * 100.0).clamp(20.0, 99.0);
            let summary = if clogp <= 4.5 && mw <= 500.0 {
                "RP-HPLC / SFC purification-ready; Lipinski-compliant parallel well".to_string()
            } else {
                "DMSO stock turbidity watch (elevated lipophilicity/MW)".to_string()
            };
            (pct, summary)
        }
    }
}

/// Greedy `Ashton et al. (2002)` `MaxMin` `ECFP4` sphere-exclusion diversity selector.
/// Returns `(selected_indices_in_pick_order, max_tanimoto_to_selected_plate_for_each_item)`.
pub fn select_maxmin_diverse_subset(
    fps: &[Ecfp2048],
    seed_priority_scores: &[f64],
    k_target: usize,
) -> (Vec<usize>, Vec<f32>) {
    let n = fps.len();
    if n == 0 || k_target == 0 {
        return (Vec::new(), Vec::new());
    }
    let k = k_target.min(n);

    // Seed with the highest-scoring candidate (e.g. best 3D Pocket/ROCS or QED lead)
    let mut first_idx = 0usize;
    let mut best_seed = f64::NEG_INFINITY;
    for (i, &sc) in seed_priority_scores.iter().enumerate() {
        if sc > best_seed {
            best_seed = sc;
            first_idx = i;
        }
    }

    let mut selected = Vec::with_capacity(k);
    let mut is_picked = vec![false; n];
    selected.push(first_idx);
    is_picked[first_idx] = true;

    // Track `max_sim[i]` = maximum Tanimoto similarity from `i` to any already-selected molecule
    // (`min_dist[i] = 1.0 - max_sim[i]`; MaxMin picks `i` minimizing `max_sim[i]`, tie-broken by priority score!)
    let mut max_sim: Vec<f32> = (0..n)
        .map(|i| {
            if i == first_idx {
                1.0
            } else {
                fps[i].tanimoto(&fps[first_idx])
            }
        })
        .collect();

    while selected.len() < k {
        let mut next_pick = None;
        let mut best_obj = f64::NEG_INFINITY;
        for i in 0..n {
            if is_picked[i] {
                continue;
            }
            let min_dist = (1.0 - max_sim[i]) as f64;
            // Pure MaxMin distance primary criterion + gentle normalized tie-breaker (`0.08 * score/100`)
            let obj = min_dist + 0.08 * (seed_priority_scores[i] / 100.0).clamp(0.0, 1.0);
            if obj > best_obj {
                best_obj = obj;
                next_pick = Some(i);
            }
        }
        let Some(chosen) = next_pick else {
            break;
        };
        is_picked[chosen] = true;
        selected.push(chosen);
        for i in 0..n {
            if !is_picked[i] {
                let sim = fps[i].tanimoto(&fps[chosen]);
                if sim > max_sim[i] {
                    max_sim[i] = sim;
                }
            }
        }
    }

    // Recompute nearest neighbor Tanimoto inside `selected` (excluding self for selected items)
    let mut nearest_plate_tc = vec![0.0_f32; n];
    for i in 0..n {
        let mut best_tc = 0.0_f32;
        for &sj in &selected {
            if sj == i {
                continue;
            }
            let tc = fps[i].tanimoto(&fps[sj]);
            if tc > best_tc {
                best_tc = tc;
            }
        }
        nearest_plate_tc[i] = best_tc;
    }

    (selected, nearest_plate_tc)
}

fn mean_pairwise_dissimilarity(fps: &[&Ecfp2048]) -> f64 {
    let m = fps.len();
    if m < 2 {
        return 0.0;
    }
    // Cap exact pair sampling at first 140 items so large enumerations stay sub-millisecond
    let cap = m.min(140);
    let mut sum = 0.0_f64;
    let mut pairs = 0usize;
    for i in 0..cap {
        for j in (i + 1)..cap {
            sum += (1.0 - fps[i].tanimoto(fps[j])) as f64;
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        sum / (pairs as f64)
    }
}

fn plate_well_label(zero_based_idx: usize) -> String {
    let plate_no = zero_based_idx / 96 + 1;
    let within = zero_based_idx % 96;
    let row = (b'A' + (within / 12) as u8) as char;
    let col = (within % 12) + 1;
    format!("Plate-{plate_no} Well {row}{col:02}")
}

/// Enumerates a 2-step or 3-step `DEL` or medicinal-chemistry parallel combinatorial library,
/// applies physicochemical & DNA-fidelity filters, runs `MaxMin` `ECFP4` diversity selection,
/// and scores candidates in 3D against an active `PocketCavityAnalysis` and/or `ROCS` query ligand.
pub fn enumerate_parallel_library(
    config: &ParallelLibConfig,
    reference_query_mol: Option<&Molecule>,
    pocket_opt: Option<&PocketCavityAnalysis>,
    dataset_opt: Option<&DatasetAnalysis>,
) -> ParallelLibraryReport {
    let scheme = &CURATED_LIBRARY_SCHEMES[config.scheme_idx % CURATED_LIBRARY_SCHEMES.len()];
    let core_smi = config
        .custom_core_smiles
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(scheme.core_smiles);
    let core_mol = parse_smiles(core_smi)
        .or_else(|_| parse_smiles(scheme.core_smiles))
        .unwrap_or_else(|_| Molecule::new(scheme.core_smiles));

    let per_cycle = config.reagents_per_cycle.clamp(2, 10);

    // Assemble per-cycle reagent plates (optionally harvesting amine synthons from DatasetAnalysis)
    let mut cycle_plates: Vec<Vec<BuildingBlockReagent>> = scheme
        .steps
        .iter()
        .map(|&rxn| {
            let mut plate = default_plate_for_reaction(rxn);
            if config.harvest_dataset_synthons
                && matches!(
                    rxn,
                    ParallelReactionKind::AmideCoupling | ParallelReactionKind::BuchwaldHartwig
                )
            {
                if let Some(ds) = dataset_opt {
                    let mut added = 0usize;
                    for m in &ds.mols {
                        if m.atoms.len() <= 18 && find_nucleophilic_amine_site(m).is_some() {
                            plate.push(BuildingBlockReagent {
                                code: format!("DS{:02}", added + 1),
                                name: m.name.clone(),
                                smiles: m.smiles.clone(),
                                synthon_class: BuildingBlockClass::PrimaryOrSecondaryAmine,
                                dna_codon_11bp: "GCTAGTCAGCA".to_string(),
                                dna_safe: true,
                                sp3_enriched: true,
                            });
                            added += 1;
                            if added >= 3 {
                                break;
                            }
                        }
                    }
                }
            }
            plate.truncate(per_cycle);
            plate
        })
        .collect();

    if cycle_plates.is_empty() {
        cycle_plates.push(curated_amine_plate().into_iter().take(per_cycle).collect());
    }

    let parsed_plates: Vec<Vec<(BuildingBlockReagent, Molecule)>> = cycle_plates
        .iter()
        .map(|pl| {
            pl.iter()
                .filter_map(|bb| parse_smiles(&bb.smiles).ok().map(|m| (bb.clone(), m)))
                .collect()
        })
        .collect();

    let theoretical_combinatorial_size: usize = parsed_plates
        .iter()
        .map(|pl| pl.len().max(1))
        .product();

    // Multi-step combinatorial cascade: Cycle 1 -> Cycle 2 (-> Cycle 3)
    #[derive(Clone)]
    struct PartialIntermediate {
        mol: Molecule,
        codes: Vec<String>,
        names: Vec<String>,
        codons: Vec<String>,
    }

    let mut frontier = vec![PartialIntermediate {
        mol: core_mol.clone(),
        codes: Vec::new(),
        names: Vec::new(),
        codons: Vec::new(),
    }];

    for (step_i, &rxn) in scheme.steps.iter().enumerate() {
        let plate = &parsed_plates[step_i];
        let next_frontier: Vec<PartialIntermediate> = frontier
            .par_iter()
            .flat_map_iter(|inter| {
                plate.iter().filter_map(move |(bb, bb_mol)| {
                    let prod = apply_parallel_reaction_step(rxn, &inter.mol, bb_mol)?;
                    let mut codes = inter.codes.clone();
                    codes.push(bb.code.clone());
                    let mut names = inter.names.clone();
                    names.push(bb.name.clone());
                    let mut codons = inter.codons.clone();
                    codons.push(bb.dna_codon_11bp.clone());
                    Some(PartialIntermediate {
                        mol: prod,
                        codes,
                        names,
                        codons,
                    })
                })
            })
            .collect();
        frontier = next_frontier;
    }

    // Deduplicate canonical OpenSMILES while preserving deterministic reagent-tuple order
    let mut seen_smiles = BTreeSet::new();
    let mut unique_raw = Vec::with_capacity(frontier.len());
    for item in frontier {
        if seen_smiles.insert(item.mol.smiles.clone()) {
            unique_raw.push(item);
        }
    }
    let valid_unique_enumerated = unique_raw.len();

    let route_str = scheme
        .steps
        .iter()
        .map(|s| s.short_label())
        .collect::<Vec<_>>()
        .join(" -> ");

    // Reference 3D ROCS query (user-selected ligand, dataset[0], or Erlotinib fallback)
    let fallback_query = parse_smiles("COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC")
        .unwrap_or_else(|_| core_mol.clone());
    let ref_mol: &Molecule = reference_query_mol
        .or_else(|| dataset_opt.and_then(|ds| ds.mols.first()))
        .unwrap_or(&fallback_query);
    let ref_query_name = if !ref_mol.name.is_empty() {
        ref_mol.name.clone()
    } else if !ref_mol.id.is_empty() {
        ref_mol.id.clone()
    } else {
        "Erlotinib (3D Reference)".to_string()
    };
    let ref_3d: Summary3dProfile = evaluate_3d_summary_profile(ref_mol, false);

    // Stage 1 (Parallel 2D + Fast Physchem + ECFP4 + Murcko + DNA-Fidelity Audit)
    struct Stage1Entry {
        raw_idx: usize,
        inter: PartialIntermediate,
        mw: f64,
        clogp: f64,
        tpsa: f64,
        fsp3: f64,
        hbd: usize,
        hba: usize,
        rot_bonds: usize,
        qed: f64,
        cns_mpo: f64,
        sa_score: f64,
        pains: usize,
        fidelity_pct: f64,
        dna_note: String,
        fp: Ecfp2048,
        murcko: String,
        passes_guardrails: bool,
        stage1_score: f64,
    }

    let stage1_all: Vec<Stage1Entry> = unique_raw
        .into_par_iter()
        .enumerate()
        .map(|(idx, mut inter)| {
            let lib_id = format!("LIB-{:03}", idx + 1);
            inter.mol.id = lib_id;
            inter.mol.name = format!("{} ({})", scheme.code, inter.codes.join("-"));
            inter.mol.target = pocket_opt
                .map(|p| p.pdb_id.clone())
                .unwrap_or_else(|| ref_mol.target.clone());

            let alerts = scan_structural_alerts(&inter.mol);
            let pains = alerts.len();
            let pc = evaluate_physchem(&inter.mol, pains);
            let sa = evaluate_synthetic_accessibility(&inter.mol);
            let fp = compute_ecfp(&inter.mol, 2);
            let (murcko, _, _, _) = murcko_signatures(&inter.mol);
            let (fidelity_pct, dna_note) = evaluate_synthetic_fidelity(
                config.synthesis_mode,
                scheme.steps,
                pc.clogp,
                pc.mw,
                pains,
            );
            let passes = pc.mw <= config.max_mw
                && pc.clogp <= config.max_clogp
                && pc.fsp3 >= config.min_fsp3
                && (!config.exclude_pains || pains == 0);
            let stage1_score = 42.0 * pc.qed_mean
                + 18.0 * pc.fsp3.clamp(0.0, 0.7)
                + 20.0 * (fidelity_pct / 100.0)
                + 20.0 * ((6.5 - sa.sa_score).clamp(0.0, 4.0) / 4.0);

            Stage1Entry {
                raw_idx: idx,
                inter,
                mw: pc.mw,
                clogp: pc.clogp,
                tpsa: pc.tpsa,
                fsp3: pc.fsp3,
                hbd: pc.hbd,
                hba: pc.hba,
                rot_bonds: pc.rotatable_bonds,
                qed: pc.qed_mean,
                cns_mpo: pc.cns_mpo,
                sa_score: sa.sa_score,
                pains,
                fidelity_pct,
                dna_note,
                fp,
                murcko,
                passes_guardrails: passes,
                stage1_score,
            }
        })
        .collect();

    let all_fps_refs: Vec<&Ecfp2048> = stage1_all.iter().map(|e| &e.fp).collect();
    let mean_dissim_all = mean_pairwise_dissimilarity(&all_fps_refs);
    let unique_scaffolds_all: usize = stage1_all
        .iter()
        .map(|e| e.murcko.as_str())
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>()
        .len()
        .max(1);

    let mut passing: Vec<Stage1Entry> = stage1_all
        .iter()
        .filter(|e| e.passes_guardrails)
        .map(|e| Stage1Entry {
            raw_idx: e.raw_idx,
            inter: e.inter.clone(),
            mw: e.mw,
            clogp: e.clogp,
            tpsa: e.tpsa,
            fsp3: e.fsp3,
            hbd: e.hbd,
            hba: e.hba,
            rot_bonds: e.rot_bonds,
            qed: e.qed,
            cns_mpo: e.cns_mpo,
            sa_score: e.sa_score,
            pains: e.pains,
            fidelity_pct: e.fidelity_pct,
            dna_note: e.dna_note.clone(),
            fp: e.fp.clone(),
            murcko: e.murcko.clone(),
            passes_guardrails: e.passes_guardrails,
            stage1_score: e.stage1_score,
        })
        .collect();

    let property_filter_passed = passing.len();
    // Fallback so even ultra-strict user sliders still show top candidates
    if passing.is_empty() {
        passing = stage1_all;
    }

    // Seed a MaxMin-diverse + QED-enriched 3D evaluation pool (`max_3d_scored_pool`) so even 216+ libraries run fast
    let max_3d_pool = config
        .max_3d_scored_pool
        .max(config.target_diverse_plate_size)
        .clamp(16, 140)
        .min(passing.len());

    let passing_fps: Vec<Ecfp2048> = passing.iter().map(|e| e.fp.clone()).collect();
    let stage1_scores: Vec<f64> = passing.iter().map(|e| e.stage1_score).collect();
    let (pool_indices, _) = select_maxmin_diverse_subset(&passing_fps, &stage1_scores, max_3d_pool);
    let in_3d_pool: BTreeSet<usize> = pool_indices.into_iter().collect();

    // Stage 2: Parallel 3D Conformer + ROCS Shape-Color + PDB Pocket Complementarity Scoring
    let mut members: Vec<EnumeratedLibraryMember> = passing
        .into_par_iter()
        .enumerate()
        .map(|(pi, entry)| {
            let barcode_or_well = match config.synthesis_mode {
                LibrarySynthesisMode::DnaEncodedOnDna => format!(
                    "DEL{:02}-{} (5'-{}-3')",
                    config.scheme_idx + 1,
                    entry.inter.codes.join("-"),
                    entry.inter.codons.join(":")
                ),
                LibrarySynthesisMode::SolutionPhaseParallel => plate_well_label(entry.raw_idx),
            };

            let mut usrcat_sim = 0.0_f64;
            let mut rocs_combo = 0.0_f64;
            let mut rocs_shape = 0.0_f64;
            let mut rocs_color = 0.0_f64;
            let mut rocs_coords = Vec::new();
            let mut pock_score = 0.0_f64;
            let mut pock_dg = 0.0_f64;
            let mut pock_le = 0.0_f64;
            let mut pock_lle = 0.0_f64;
            let mut unhappy_h2o = 0usize;
            let mut pharm_recall = 0.0_f64;
            let mut pock_pose = None;

            if in_3d_pool.contains(&pi) {
                let s3d = evaluate_3d_summary_profile(&entry.inter.mol, true);
                usrcat_sim = usrcat_similarity(&ref_3d.usrcat60, &s3d.usrcat60);
                let al = align_rocs_3d(
                    ref_mol,
                    &ref_3d.coords3d,
                    &ref_3d.gasteiger_charges,
                    &ref_3d.color_atoms,
                    &entry.inter.mol,
                    &s3d.coords3d,
                    &s3d.gasteiger_charges,
                    &s3d.color_atoms,
                );
                rocs_combo = al.combo_score;
                rocs_shape = al.shape_tanimoto;
                rocs_color = al.color_tanimoto;
                rocs_coords = al.aligned_hit_coords;

                if let Some(pock) = pocket_opt {
                    let mut pose = score_ligand_in_pocket(&entry.inter.mol, None, pock);
                    pock_score = pose.complementarity_score;
                    pock_dg = pose.total_dg_pocket_kcal;
                    pock_le = pose.pocket_le;
                    let est_pkd = (-pock_dg / RT_LN10_KCAL).clamp(3.0, 11.5);
                    pock_lle = est_pkd - entry.clogp;
                    pose.pocket_lle = pock_lle;
                    unhappy_h2o = pose.displaced_unhappy_waters;
                    pharm_recall = pose.pharmacophore_recall_pct;
                    pock_pose = Some(pose);
                }
            }

            let rocs_pct = ((rocs_combo / 2.0) * 100.0).clamp(0.0, 100.0);
            let medchem_quality = (45.0 * entry.qed
                + 25.0 * (entry.fidelity_pct / 100.0)
                + 15.0 * (entry.fsp3 / 0.45).clamp(0.0, 1.0)
                + 15.0 * ((6.0 - entry.sa_score).clamp(0.0, 3.5) / 3.5))
                .clamp(0.0, 100.0);

            let composite_priority_score = match config.prefilter_mode {
                LibraryPreFilterMode::PocketAndRocsHybrid => {
                    if pocket_opt.is_some() {
                        0.42 * pock_score + 0.30 * rocs_pct + 0.28 * medchem_quality
                    } else {
                        0.55 * rocs_pct + 0.45 * medchem_quality
                    }
                }
                LibraryPreFilterMode::PdbPocketComplementarity => {
                    if pocket_opt.is_some() {
                        0.68 * pock_score + 0.12 * rocs_pct + 0.20 * medchem_quality
                    } else {
                        0.50 * rocs_pct + 0.50 * medchem_quality
                    }
                }
                LibraryPreFilterMode::Rocs3dOverlay => {
                    0.65 * rocs_pct + 0.15 * (usrcat_sim * 100.0) + 0.20 * medchem_quality
                }
                LibraryPreFilterMode::MaxMinDiversityFirst => medchem_quality,
            };

            EnumeratedLibraryMember {
                rank: 0,
                library_id: entry.inter.mol.id.clone(),
                barcode_or_well,
                cycle_reagent_codes: entry.inter.codes,
                cycle_reagent_names: entry.inter.names,
                reaction_route: route_str.clone(),
                canonical_smiles: entry.inter.mol.smiles.clone(),
                murcko_scaffold: entry.murcko,
                mw: entry.mw,
                clogp: entry.clogp,
                tpsa: entry.tpsa,
                fsp3: entry.fsp3,
                hbd: entry.hbd,
                hba: entry.hba,
                rotatable_bonds: entry.rot_bonds,
                qed: entry.qed,
                cns_mpo: entry.cns_mpo,
                sa_score: entry.sa_score,
                pains_alerts: entry.pains,
                synthetic_fidelity_pct: entry.fidelity_pct,
                dna_audit_note: entry.dna_note,
                selected_in_diverse_plate: false,
                diversity_pick_order: None,
                nearest_plate_tanimoto: 0.0,
                usrcat_sim,
                rocs_combo_score: rocs_combo,
                rocs_shape_tanimoto: rocs_shape,
                rocs_color_tanimoto: rocs_color,
                pocket_complementarity_score: pock_score,
                pocket_dg_kcal: pock_dg,
                pocket_le: pock_le,
                pocket_lle: pock_lle,
                displaced_unhappy_waters: unhappy_h2o,
                pharmacophore_recall_pct: pharm_recall,
                composite_priority_score,
                rocs_aligned_coords3d: rocs_coords,
                pocket_pose: pock_pose,
                mol: entry.inter.mol,
            }
        })
        .collect();

    // Final MaxMin diversity plate selection over the 3D-scored candidates
    let member_fps: Vec<Ecfp2048> = members.iter().map(|m| compute_ecfp(&m.mol, 2)).collect();
    let member_priorities: Vec<f64> = members
        .iter()
        .map(|m| m.composite_priority_score)
        .collect();
    let plate_target = config.target_diverse_plate_size.clamp(1, members.len().max(1));
    let (picked_indices, nearest_tc) =
        select_maxmin_diverse_subset(&member_fps, &member_priorities, plate_target);

    for (order_i, &mi) in picked_indices.iter().enumerate() {
        if let Some(mem) = members.get_mut(mi) {
            mem.selected_in_diverse_plate = true;
            mem.diversity_pick_order = Some(order_i + 1);
        }
    }
    for (mi, &tc) in nearest_tc.iter().enumerate() {
        if let Some(mem) = members.get_mut(mi) {
            mem.nearest_plate_tanimoto = tc;
        }
    }

    // Sort: selected diverse plate members first (by priority or pick order), then remaining members
    members.sort_by(|a, b| {
        b.selected_in_diverse_plate
            .cmp(&a.selected_in_diverse_plate)
            .then_with(|| {
                if config.prefilter_mode == LibraryPreFilterMode::MaxMinDiversityFirst {
                    a.diversity_pick_order
                        .unwrap_or(usize::MAX)
                        .cmp(&b.diversity_pick_order.unwrap_or(usize::MAX))
                } else {
                    b.composite_priority_score
                        .partial_cmp(&a.composite_priority_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            })
    });
    for (i, m) in members.iter_mut().enumerate() {
        m.rank = i + 1;
    }

    let plate_fps: Vec<&Ecfp2048> = picked_indices
        .iter()
        .filter_map(|&idx| member_fps.get(idx))
        .collect();
    let mean_dissim_plate = mean_pairwise_dissimilarity(&plate_fps);

    let plate_members: Vec<&EnumeratedLibraryMember> = members
        .iter()
        .filter(|m| m.selected_in_diverse_plate)
        .collect();
    let selected_diverse_plate_count = plate_members.len();
    let unique_murcko_scaffolds_plate = plate_members
        .iter()
        .map(|m| m.murcko_scaffold.as_str())
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>()
        .len()
        .max(usize::from(!plate_members.is_empty()));

    let denom = (selected_diverse_plate_count.max(1)) as f64;
    let mean_fsp3_plate = plate_members.iter().map(|m| m.fsp3).sum::<f64>() / denom;
    let mean_mw_plate = plate_members.iter().map(|m| m.mw).sum::<f64>() / denom;
    let mean_fidelity_pct_plate = plate_members
        .iter()
        .map(|m| m.synthetic_fidelity_pct)
        .sum::<f64>()
        / denom;

    let pocket_hit_count = members
        .iter()
        .filter(|m| m.pocket_dg_kcal <= -6.5)
        .count();
    let rocs_hit_count = members
        .iter()
        .filter(|m| m.rocs_combo_score >= 1.10)
        .count();

    ParallelLibraryReport {
        scheme_code: scheme.code.to_string(),
        scheme_title: scheme.title.to_string(),
        synthesis_mode: config.synthesis_mode,
        prefilter_mode: config.prefilter_mode,
        step_count: scheme.step_count,
        reaction_steps: scheme.steps.to_vec(),
        core_name: scheme.core_name.to_string(),
        core_smiles: core_smi.to_string(),
        reference_query_name: ref_query_name,
        reference_pocket_pdb: pocket_opt.map(|p| p.pdb_id.clone()),
        theoretical_combinatorial_size,
        valid_unique_enumerated,
        property_filter_passed,
        scored_3d_count: in_3d_pool.len(),
        selected_diverse_plate_count,
        unique_murcko_scaffolds_all: unique_scaffolds_all,
        unique_murcko_scaffolds_plate,
        mean_pairwise_dissimilarity_all: mean_dissim_all,
        mean_pairwise_dissimilarity_plate: mean_dissim_plate,
        mean_fsp3_plate,
        mean_mw_plate,
        mean_fidelity_pct_plate,
        pocket_hit_count,
        rocs_hit_count,
        cycle_plates,
        members,
    }
}

/// Exports enumerated combinatorial library members (with DNA barcodes / plate wells, `MaxMin` diversity,
/// and 3D `PDB Pocket` + `ROCS` pre-filter scores) as a RFC-4180 `.csv` table.
pub fn export_parallel_library_csv(report: &ParallelLibraryReport, diverse_plate_only: bool) -> String {
    let mut out = String::with_capacity(8192);
    out.push_str("Rank,Library_ID,Selected_Plate,Diversity_Pick_Order,Mode,Barcode_or_Well,Reaction_Route,Building_Blocks,Priority_Score,Pocket_Complementarity,Pocket_dG_kcal,Pocket_LE,Pocket_LLE,Displaced_Unhappy_H2O,ROCS_Combo,ROCS_Shape,ROCS_Color,USRCAT_Sim,Nearest_Plate_Tc,MW,cLogP,TPSA,Fsp3,QED,SAScore,Fidelity_Pct,SMILES\n");

    for m in &report.members {
        if diverse_plate_only && !m.selected_in_diverse_plate {
            continue;
        }
        let pick_str = m
            .diversity_pick_order
            .map(|k| k.to_string())
            .unwrap_or_default();
        out.push_str(&format!(
            "{},{},{},{},{},\"{}\",\"{}\",\"{}\",{:.2},{:.2},{:.2},{:.3},{:.2},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.2},{:.1},{:.3},{:.3},{:.2},{:.1},\"{}\"\n",
            m.rank,
            m.library_id,
            if m.selected_in_diverse_plate { "YES" } else { "NO" },
            pick_str,
            report.synthesis_mode.short_label(),
            m.barcode_or_well.replace('"', "'"),
            m.reaction_route,
            m.cycle_reagent_codes.join("+"),
            m.composite_priority_score,
            m.pocket_complementarity_score,
            m.pocket_dg_kcal,
            m.pocket_le,
            m.pocket_lle,
            m.displaced_unhappy_waters,
            m.rocs_combo_score,
            m.rocs_shape_tanimoto,
            m.rocs_color_tanimoto,
            m.usrcat_sim,
            m.nearest_plate_tanimoto,
            m.mw,
            m.clogp,
            m.tpsa,
            m.fsp3,
            m.qed,
            m.sa_score,
            m.synthetic_fidelity_pct,
            m.canonical_smiles.replace('"', "'"),
        ));
    }
    out
}

/// Exports the 3D Pocket-aligned (or `ROCS`-aligned) coordinates of shortlisted library members as MDL V2000 `.sdf`.
pub fn format_parallel_library_sdf(report: &ParallelLibraryReport, max_records: usize) -> String {
    let mut sdf = String::with_capacity(16384);
    for m in report
        .members
        .iter()
        .filter(|m| m.selected_in_diverse_plate)
        .take(max_records.max(1))
    {
        let coords = if let Some(ref pose) = m.pocket_pose {
            pose.aligned_ligand_coords3d.clone()
        } else if !m.rocs_aligned_coords3d.is_empty() {
            m.rocs_aligned_coords3d.clone()
        } else {
            evaluate_3d_summary_profile(&m.mol, true).coords3d
        };
        let charges = vec![0.0_f32; m.mol.atoms.len()];
        let tags = [
            ("LIBRARY_ID", m.library_id.clone()),
            ("SYNTHESIS_MODE", report.synthesis_mode.label().to_string()),
            ("BARCODE_OR_WELL", m.barcode_or_well.clone()),
            ("REACTION_ROUTE", m.reaction_route.clone()),
            ("BUILDING_BLOCKS", m.cycle_reagent_codes.join(" + ")),
            ("PRIORITY_SCORE", format!("{:.2}", m.composite_priority_score)),
            ("POCKET_DG_KCAL", format!("{:.2}", m.pocket_dg_kcal)),
            ("ROCS_COMBO_SCORE", format!("{:.3}", m.rocs_combo_score)),
            ("NEAREST_PLATE_TC", format!("{:.3}", m.nearest_plate_tanimoto)),
            ("CANONICAL_SMILES", m.canonical_smiles.clone()),
        ];
        sdf.push_str(&format_3d_conformer_sdf(
            &m.mol,
            &coords,
            &charges,
            m.pocket_dg_kcal,
            &tags,
        ));
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pocket_cavity::{analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure};

    #[test]
    fn test_all_four_parallel_reactions_produce_sanitized_molecules() {
        // 1. Amide Coupling
        let acid = parse_smiles("O=C(O)c1cc(I)cc(C#C)c1").unwrap();
        let amine = parse_smiles("NCC1CC1").unwrap();
        let amide = apply_parallel_reaction_step(ParallelReactionKind::AmideCoupling, &acid, &amine)
            .expect("Amide coupling failed");
        assert!(amide.molecular_weight() > acid.molecular_weight());

        // 2. Suzuki-Miyaura Coupling (chemoselective at C-I)
        let boronic = parse_smiles("Cn1cc(B(O)O)cn1").unwrap();
        let suzuki =
            apply_parallel_reaction_step(ParallelReactionKind::SuzukiMiyaura, &amide, &boronic)
                .expect("Suzuki coupling failed");
        assert!(!suzuki.atoms.iter().any(|a| a.elem == Element::I || a.elem == Element::B));

        // 3. CuAAC Click (1,4-triazole formation at C#CH)
        let azide = parse_smiles("[N-]=[N+]=NCCO").unwrap();
        let triazole =
            apply_parallel_reaction_step(ParallelReactionKind::CuAacClick, &suzuki, &azide)
                .expect("CuAAC click failed");
        assert!(triazole.rings.len() >= 3);

        // 4. Buchwald-Hartwig / SNAr on aryl chloride
        let quinaz = parse_smiles("Clc1ncnc2cc(Br)ccc12").unwrap();
        let morph = parse_smiles("C1COCCN1").unwrap();
        let buch =
            apply_parallel_reaction_step(ParallelReactionKind::BuchwaldHartwig, &quinaz, &morph)
                .expect("Buchwald coupling failed");
        assert!(buch.atoms.iter().any(|a| a.elem == Element::Cl)); // C6-Br reacted first!
    }

    #[test]
    fn test_three_step_del_enumeration_maxmin_and_3d_pocket_rocs_prefilter() {
        let pdb = parse_pdb_structure(curated_pocket_pdb_text(0)).unwrap();
        let pocket = analyze_binding_pocket(&pdb, None, 10.5);
        let cfg = ParallelLibConfig {
            scheme_idx: 0,
            synthesis_mode: LibrarySynthesisMode::DnaEncodedOnDna,
            prefilter_mode: LibraryPreFilterMode::PocketAndRocsHybrid,
            reagents_per_cycle: 4,
            target_diverse_plate_size: 12,
            max_3d_scored_pool: 24,
            ..ParallelLibConfig::default()
        };
        let rep = enumerate_parallel_library(&cfg, None, Some(&pocket), None);
        assert_eq!(rep.theoretical_combinatorial_size, 64);
        assert!(rep.valid_unique_enumerated >= 50);
        assert_eq!(rep.selected_diverse_plate_count, 12);
        assert!(rep.mean_pairwise_dissimilarity_plate >= rep.mean_pairwise_dissimilarity_all * 0.95);
        assert!(rep.members[0].barcode_or_well.starts_with("DEL01-"));
        assert!(rep.members[0].pocket_dg_kcal < -2.0);
        assert!(rep.members[0].rocs_combo_score > 0.4);

        let csv = export_parallel_library_csv(&rep, true);
        assert_eq!(csv.lines().count(), 13);
        let sdf = format_parallel_library_sdf(&rep, 3);
        assert!(sdf.contains("V2000") && sdf.contains("BARCODE_OR_WELL"));
    }
}
