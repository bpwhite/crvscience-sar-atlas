//! Fragment-Based Drug Discovery (`FBDD`) Rule-of-3 (`Ro3`) Exit-Vector Grower &
//! Group-Efficiency (`GE` / `LLE` / `LE` / `SILE`) Optimizer (`v1.7.0`).
//!
//! Implements:
//! 1. Astex `Rule of Three` (`Ro3`: `MW <= 300 Da`, `cLogP <= 3.0`, `HBD <= 3`, `HBA <= 3`,
//!    `N_rot <= 3`, `TPSA <= 60 A^2`) compliance & Astex synthetic **Fragment Sociability**
//!    auditing, plus automatic **Deconstruction-to-Ro3 Core Seed** extraction when given a
//!    full lead-sized molecule (`MW > 300 Da`).
//! 2. 6 curated clinical `FBDD` `Rule-of-3` seeds (`7-Azaindole` Vemurafenib seed,
//!    `4-Aminoquinazoline` EGFR seed, `7H-Pyrrolo[2,3-d]pyrimidin-4-amine` JAK seed,
//!    `1H-Indazole-3-carboxamide` Hsp90 seed, `5-Phenyl-1H-pyrazole` AT13148/H-PGDS seed,
//!    and `2-Aminobenzimidazole` Aurora/Chk1 seed).
//! 3. Directional 2D/topological **Exit-Vector Perception** across `C(sp2)-H` / poised aryl
//!    halides (`-Br`/`-I`/`-Cl`), heteroaryl `N-H`, exocyclic `-NH2`/`-OH`, and `C(sp3)-H`
//!    positions, flagging hinge-critical H-bond donors before elaboration.
//! 4. 16 poised synthetic growth synthons (`ΔHAC = 1..=7`) across 5 subpocket probing roles
//!    (Hydrophobic/`sp3` dimple, Polar H-bond/Sulfonyl `LLE` booster, Heteroaryl pi-stacker,
//!    Aryl cleft, and Solvent-front `sp3` solubilizing amine), scored by:
//!    - Hopkins **Ligand Efficiency** `LE = 1.364 * pAct / HAC` (`kcal/mol/HA`)
//!    - Verdonk-Rees **Group Efficiency** `GE = 1.364 * Delta pAct / Delta HAC` (`kcal/mol/HA`)
//!    - Leeson-Springthorpe **Lipophilic Ligand Efficiency** `LLE = pAct - cLogP` & `LLE_7.4 = pAct - logD_7.4`
//!    - Nissink / Kenny **Size-Independent Ligand Efficiency** `SILE = 1.364 * pAct / HAC^0.3`
//!
//! Primary References:
//! - Congreve, Carr, Murray & Jhoti (2003), *Drug Discov. Today* 8(19), 876-877
//!   ("A 'rule of three' for fragment-based lead discovery?": `https://pubmed.ncbi.nlm.nih.gov/14554012/`)
//! - Jhoti, Williams, Rees & Murray (2013), *Nat. Rev. Drug Discov.* 12(8), 644-645
//!   ("The 'rule of three' for fragment-based drug discovery: where are we now?": `https://pubmed.ncbi.nlm.nih.gov/23845999/`)
//! - Hopkins, Groom & Alex (2004), *Drug Discov. Today* 9(10), 430-431
//!   ("Ligand efficiency: a useful metric for lead selection": `https://pubmed.ncbi.nlm.nih.gov/15109945/`)
//!   & Kenny (2019), *J. Cheminform.* 11, 8 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6755556/`)
//! - Verdonk & Rees (2008), *ChemMedChem* 3(8), 1179-1180
//!   ("Group efficiency: a guideline for hits-to-leads chemistry": `https://pubmed.ncbi.nlm.nih.gov/18651625/`)
//! - Leeson & Springthorpe (2007), *Nat. Rev. Drug Discov.* 6(11), 881-890
//!   ("The influence of drug-like concepts on decision-making in medicinal chemistry": `https://pubmed.ncbi.nlm.nih.gov/17971784/`)
//! - Astex Fragment Sociability, Poised Fragments & Growth Vectors (`RSC Med. Chem.` 2021:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8130625/`) & Deconstruction-Reconstruction (`https://pmc.ncbi.nlm.nih.gov/articles/PMC12169673/`)

use crate::alerts::scan_structural_alerts;
use crate::fingerprint::{compute_ecfp, Ecfp2048};
use crate::layout2d::compute_2d_coords;
use crate::mol::{Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::{format_sdf_record, parse_smiles};
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::scaffold::murcko_framework_mask;
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Thermodynamic conversion factor `RT * ln(10)` at `298.15 K` (`kcal/mol` per `pAct` log unit).
pub const RT_LN10_KCAL: f64 = 1.364;

/// Curated clinical Fragment-Based Drug Discovery (`FBDD`) `Rule-of-3` fragment hit preset.
#[derive(Clone, Copy, Debug)]
pub struct CuratedRo3SeedSpec {
    pub code: &'static str,
    pub name: &'static str,
    pub smiles: &'static str,
    pub clinical_lineage: &'static str,
    pub target_family: &'static str,
    pub fragment_pact: f64,
}

pub const CURATED_RO3_SEEDS: &[CuratedRo3SeedSpec] = &[
    CuratedRo3SeedSpec {
        code: "RO3-7AZA",
        name: "7-Azaindole",
        smiles: "c1cc2cc[nH]c2nc1",
        clinical_lineage: "Plexxikon PLX4032 -> Vemurafenib (BRAF V600E)",
        target_family: "Kinase Hinge Bidentate HBD/HBA",
        fragment_pact: 3.70, // ~200 uM biochemical fragment hit, LE = 0.56
    },
    CuratedRo3SeedSpec {
        code: "RO3-4AQZ",
        name: "4-Aminoquinazoline",
        smiles: "Nc1ncnc2ccccc12",
        clinical_lineage: "Gefitinib / Erlotinib / Afatinib Hinge Core",
        target_family: "EGFR / HER2 Kinase Hinge",
        fragment_pact: 4.30, // ~50 uM, LE = 0.53
    },
    CuratedRo3SeedSpec {
        code: "RO3-7DPP",
        name: "7H-Pyrrolo[2,3-d]pyrimidin-4-amine",
        smiles: "Nc1ncnc2[nH]ccc12",
        clinical_lineage: "Ruxolitinib / Tofacitinib 7-Deazapurine Core",
        target_family: "JAK1/2/3 ATP-Site Hinge",
        fragment_pact: 4.18, // ~66 uM, LE = 0.57
    },
    CuratedRo3SeedSpec {
        code: "RO3-INDZ",
        name: "1H-Indazole-3-carboxamide",
        smiles: "NC(=O)c1c2ccccc2[nH]n1",
        clinical_lineage: "SNX-5422 / Entrectinib Indazole Carboxamide Seed",
        target_family: "Hsp90 / TRK / ALK Polar Anchor",
        fragment_pact: 4.15, // ~71 uM, LE = 0.47
    },
    CuratedRo3SeedSpec {
        code: "RO3-PHPY",
        name: "5-Phenyl-1H-pyrazole",
        smiles: "c1ccc(-c2cc[nH]n2)cc1",
        clinical_lineage: "Astex AT13148 (AKT/ROCK) & H-PGDS Sociable Hit",
        target_family: "AGC Kinase / H-PGDS Catalytic Pocket",
        fragment_pact: 4.10, // ~79 uM, LE = 0.51
    },
    CuratedRo3SeedSpec {
        code: "RO3-BZIM",
        name: "2-Aminobenzimidazole",
        smiles: "Nc1nc2ccccc2[nH]1",
        clinical_lineage: "Astex Aurora / Chk1 / BACE1 Guanidine-Like Seed",
        target_family: "Asp-Dyad / Kinase Hinge HBD2",
        fragment_pact: 3.95, // ~112 uM, LE = 0.54
    },
];

/// Builds a sanitized `Molecule` with 2D coordinates for one of `CURATED_RO3_SEEDS`.
pub fn build_curated_ro3_seed_molecule(seed_idx: usize) -> Molecule {
    let spec = &CURATED_RO3_SEEDS[seed_idx.min(CURATED_RO3_SEEDS.len() - 1)];
    let mut m = parse_smiles(spec.smiles).unwrap_or_else(|_| Molecule::new(spec.code));
    m.id = spec.code.to_string();
    m.name = spec.name.to_string();
    m.target = spec.target_family.to_string();
    m.metric = "IC50".to_string();
    m.pact = spec.fragment_pact;
    m.raw_nm = 10f64.powf(9.0 - spec.fragment_pact);
    m.smiles = molecule_to_smiles(&m);
    compute_2d_coords(&mut m);
    m
}

/// Synthetic C-C / C-N / C-O elaboration chemistry class at a fragment exit vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExitVectorChemistry {
    PoisedArylHalide,
    ArylOrHeteroarylCH,
    HeteroarylNH,
    ExocyclicAmineOrHydroxyl,
    AliphaticSp3CH,
}

impl ExitVectorChemistry {
    pub fn label(self) -> &'static str {
        match self {
            Self::PoisedArylHalide => "Poised Aryl C-X (Suzuki / Buchwald)",
            Self::ArylOrHeteroarylCH => "Aryl C(sp2)-H (C-H Borylation / Suzuki)",
            Self::HeteroarylNH => "Ring N-H (Chan-Lam / Ullmann / N-Alk)",
            Self::ExocyclicAmineOrHydroxyl => "Exocyclic -NH2/-OH (HATU / Sulfonyl / Ether)",
            Self::AliphaticSp3CH => "Aliphatic C(sp3)-H (sp3 Escape Vector)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::PoisedArylHalide => "Aryl C-X",
            Self::ArylOrHeteroarylCH => "Aryl C-H",
            Self::HeteroarylNH => "Ring N-H",
            Self::ExocyclicAmineOrHydroxyl => "-NH / -OH",
            Self::AliphaticSp3CH => "C(sp3)-H",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::PoisedArylHalide => [52, 211, 153],         // Emerald
            Self::ArylOrHeteroarylCH => [56, 189, 248],       // Sky Blue
            Self::HeteroarylNH => [251, 191, 36],             // Amber
            Self::ExocyclicAmineOrHydroxyl => [244, 114, 182], // Pink
            Self::AliphaticSp3CH => [192, 132, 252],          // Violet
        }
    }
}

/// Subpocket probing role of an added fragment-growing synthon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GrowthPocketRole {
    HydrophobicAndSp3Dimple,
    PolarHBondAndLleBooster,
    HeteroarylPiStack,
    LipophilicArylCleft,
    SolventFrontAmine,
}

impl GrowthPocketRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::HydrophobicAndSp3Dimple => "Hydrophobic / Magic-Me / sp3",
            Self::PolarHBondAndLleBooster => "Polar H-Bond / Sulfonyl (LLE+)",
            Self::HeteroarylPiStack => "Heteroaryl HBA Pi-Stack",
            Self::LipophilicArylCleft => "Fluorinated / Benzyl Aryl Cleft",
            Self::SolventFrontAmine => "Solvent-Front sp3 Amine",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HydrophobicAndSp3Dimple => "Sp3 / Me",
            Self::PolarHBondAndLleBooster => "Polar LLE+",
            Self::HeteroarylPiStack => "HetAr Pi",
            Self::LipophilicArylCleft => "Aryl Cleft",
            Self::SolventFrontAmine => "Solvent N",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HydrophobicAndSp3Dimple => [56, 189, 248],  // Sky Blue
            Self::PolarHBondAndLleBooster => [52, 211, 153],  // Emerald
            Self::HeteroarylPiStack => [251, 191, 36],        // Amber
            Self::LipophilicArylCleft => [244, 114, 182],     // Pink
            Self::SolventFrontAmine => [192, 132, 252],       // Violet
        }
    }
}

/// Verdonk-Rees Group Efficiency (`GE`) & Leeson-Springthorpe `LLE` optimization quality tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowthEfficiencyRegime {
    HotspotEngaging,
    ProductiveLeadLike,
    PropertyTuning,
    UnproductiveBulk,
}

impl GrowthEfficiencyRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::HotspotEngaging => "Hotspot-Engaging (GE >= 0.36 & LLE+)",
            Self::ProductiveLeadLike => "Productive F2L Step (GE >= 0.26)",
            Self::PropertyTuning => "Solubilizing / Property-Tuning",
            Self::UnproductiveBulk => "Low-GE Bulk / Hinge Mask Risk",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HotspotEngaging => "Hotspot GE+",
            Self::ProductiveLeadLike => "Productive",
            Self::PropertyTuning => "Prop-Tune",
            Self::UnproductiveBulk => "Low GE",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HotspotEngaging => [52, 211, 153],    // Emerald
            Self::ProductiveLeadLike => [56, 189, 248], // Sky Blue
            Self::PropertyTuning => [251, 191, 36],     // Amber
            Self::UnproductiveBulk => [248, 113, 113],  // Coral
        }
    }
}

/// Astex Rule-of-3 (`Congreve et al. 2003`; `Jhoti et al. 2013`) & Fragment Sociability audit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleOfThreeAudit {
    pub heavy_atoms: usize,
    pub mw: f64,
    pub clogp: f64,
    pub logd_74: f64,
    pub hbd: usize,
    pub hba: usize,
    pub rotatable_bonds: usize,
    pub tpsa: f64,
    pub fsp3: f64,
    pub ro3_violations: usize,
    pub strict_ro3_pass: bool,
    pub astex_core_pass: bool,
    /// Astex synthetic sociability score `[0..100]` reflecting diverse C-H/C-X/N-H exit vectors
    /// and low synthetic complexity (`SAScore`).
    pub sociability_score: f64,
    pub seed_le: f64,
    pub seed_lle: f64,
    pub seed_lle_74: f64,
    pub seed_sile: f64,
}

/// Single elaboratable exit vector on a fragment seed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentExitVector {
    pub vector_idx: usize,
    pub anchor_atom: usize,
    /// Optional poised halide atom (`-Br`, `-I`, `-Cl`) displaced during cross-coupling.
    pub displaced_halide_atom: Option<usize>,
    pub atom_label: String,
    pub chemistry: ExitVectorChemistry,
    /// Unit outward 2D vector `[dx, dy]` in molecule 2D coordinate space for arrow rendering.
    pub dir2d: [f32; 2],
    /// True if growing here caps a potential bidentate hinge H-bond donor (`[nH]`/`-NH2` ortho to ring `N`).
    pub masks_hinge_hbd: bool,
    pub steric_openness: f64,
    pub sociability_weight: f64,
    pub rationale: String,
}

/// Thermodynamic & lipophilic efficiency breakdown for a grown fragment-to-lead candidate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub seed_hac: usize,
    pub grown_hac: usize,
    pub delta_hac: usize,
    pub seed_pact: f64,
    pub predicted_pact: f64,
    pub delta_pact: f64,
    /// Free-energy gain `Delta dG_bind = -1.364 * Delta pAct` (`kcal/mol`, negative = tighter binding).
    pub delta_dg_kcal: f64,
    /// Hopkins Ligand Efficiency `LE = 1.364 * pAct / HAC` (`kcal/mol/HA`).
    pub le: f64,
    pub delta_le: f64,
    /// Verdonk-Rees Group Efficiency `GE = 1.364 * Delta pAct / Delta HAC` (`kcal/mol/added HA`).
    pub ge: f64,
    /// Leeson-Springthorpe Lipophilic Ligand Efficiency `LLE = pAct - cLogP`.
    pub lle: f64,
    pub delta_lle: f64,
    /// Physiological `LLE_7.4 (LipE_D) = pAct - logD_7.4`.
    pub lle_74: f64,
    /// Size-Independent Ligand Efficiency `SILE = 1.364 * pAct / HAC^0.3`.
    pub sile: f64,
    /// Nearest-neighbor `ECFP4` Tanimoto similarity to experimental dataset compounds (`[0..1]`).
    pub nearest_dataset_tanimoto: f32,
    pub nearest_dataset_id: Option<String>,
}

/// Single grown fragment-to-lead candidate produced by elaborating `FragmentExitVector` with a poised synthon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrownFragmentCandidate {
    pub rank: usize,
    pub candidate_id: String,
    #[serde(skip_deserializing)]
    pub synthon_code: &'static str,
    #[serde(skip_deserializing)]
    pub synthon_name: &'static str,
    #[serde(skip_deserializing)]
    pub coupling_reaction: &'static str,
    pub pocket_role: GrowthPocketRole,
    pub vector_idx: usize,
    pub vector_label: String,
    pub used_methylene_spacer: bool,
    pub efficiency: EfficiencyMetrics,
    pub regime: GrowthEfficiencyRegime,
    pub physchem: PhyschemProfile,
    pub logd_74: f64,
    pub sa_score: f64,
    /// Multiparameter Fragment-to-Lead Optimizer Score `[0..100]` balancing `GE`, `LLE`, `LE`,
    /// `QED`/`CNS-MPO`, `SAScore`, and zero `PAINS` alerts.
    pub optimizer_score: f64,
    pub canonical_smiles: String,
    pub seed_atoms: Vec<usize>,
    pub grown_atoms: Vec<usize>,
    pub mol: Molecule,
}

/// Complete Rule-of-3 fragment exit-vector & `GE`/`LLE`/`LE` optimization report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentGrowthReport {
    pub input_name: String,
    pub was_deconstructed_to_ro3: bool,
    pub seed_mol: Molecule,
    pub ro3_audit: RuleOfThreeAudit,
    pub exit_vectors: Vec<FragmentExitVector>,
    pub active_vector_idx: usize,
    pub candidates: Vec<GrownFragmentCandidate>,
}

/// Dataset-level `Rule-of-3` & Ligand/Lipophilic Efficiency summary row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentDatasetRow {
    pub mol_idx: usize,
    pub id: String,
    pub name: String,
    pub hac: usize,
    pub mw: f64,
    pub pact: f64,
    pub le: f64,
    pub lle: f64,
    pub lle_74: f64,
    pub sile: f64,
    pub ro3_violations: usize,
    pub strict_ro3_pass: bool,
    pub exit_vector_count: usize,
    pub sociability_score: f64,
}

struct GrowthSynthonSpec {
    code: &'static str,
    name: &'static str,
    /// OpenSMILES with single-bond `I` dummy marking the attachment point.
    dummy_smiles: &'static str,
    role: GrowthPocketRole,
    aryl_coupling: &'static str,
    hetero_coupling: &'static str,
    /// True if direct attachment atom is `N`, `O`, `F`, or `S(=O)2` (cannot form `N-F`, `O-O`, `N-OR`, etc.).
    attaches_via_heteroatom_or_halide: bool,
    /// True if group is an acyl/sulfonyl electrophile (`-C(=O)NHMe`, `-SO2Me`, `-SO2NH2`) that cleanly acylates/sulfonylates `-NH-` / `-OH`!
    is_acyl_or_sulfonyl_cap: bool,
    /// Intrinsic Astex retrospective subpocket binding free-energy potential (`pAct` gain under well-matched vector).
    intrinsic_delta_pact: f64,
    /// Intrinsic preference for aromatic/sp2 vs exocyclic polar vectors (`-1.0..+1.0`).
    aryl_preference: f64,
}

const GROWTH_SYNTHONS: &[GrowthSynthonSpec] = &[
    GrowthSynthonSpec {
        code: "GR-ME",
        name: "Magic-Methyl (-CH3)",
        dummy_smiles: "IC",
        role: GrowthPocketRole::HydrophobicAndSp3Dimple,
        aryl_coupling: "Suzuki-Miyaura MeB(OH)2 / Negishi Me2Zn",
        hetero_coupling: "Reductive N-Methylation / Williamson MeI",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.42, // ~0.57 kcal/mol/HA when filling hydrophobic dimple or inducing conformational twist
        aryl_preference: 0.25,
    },
    GrowthSynthonSpec {
        code: "GR-F",
        name: "Fluoro-Shield (-F)",
        dummy_smiles: "IF",
        role: GrowthPocketRole::HydrophobicAndSp3Dimple,
        aryl_coupling: "Pd-Catalyzed Aryl Fluorination / Selectfluor",
        hetero_coupling: "Methylene-Fluoro (-CH2F) Bioisostere",
        attaches_via_heteroatom_or_halide: true,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.34, // Multipolar C-F...C=O contact + metabolic block (Delta HAC = 1 -> GE ~ 0.46)
        aryl_preference: 0.85,
    },
    GrowthSynthonSpec {
        code: "GR-CN",
        name: "Cyano-Dipole (-C#N)",
        dummy_smiles: "IC#N",
        role: GrowthPocketRole::PolarHBondAndLleBooster,
        aryl_coupling: "Pd2(dba)3 / Zn(CN)2 Rosenmund-von Braun",
        hetero_coupling: "Cyanomethylation (-CH2CN) Alkylation",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.62, // Displaces conserved structural water with negative delta cLogP -> strong LLE & GE boost!
        aryl_preference: 0.55,
    },
    GrowthSynthonSpec {
        code: "GR-OME",
        name: "Methoxy-HBA (-OCH3)",
        dummy_smiles: "IOC",
        role: GrowthPocketRole::PolarHBondAndLleBooster,
        aryl_coupling: "Cu/Pd Ullmann Methoxylation",
        hetero_coupling: "2-Methoxyethyl / Methoxymethyl Ether",
        attaches_via_heteroatom_or_halide: true,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.50,
        aryl_preference: 0.65,
    },
    GrowthSynthonSpec {
        code: "GR-CPR",
        name: "Cyclopropyl (-cPr)",
        dummy_smiles: "IC1CC1",
        role: GrowthPocketRole::HydrophobicAndSp3Dimple,
        aryl_coupling: "Suzuki Cyclopropylboronic Acid Coupling",
        hetero_coupling: "Buchwald / Reductive Cyclopropylation",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.78, // High-Fsp3 metabolic-stable hydrophobic pocket probe (Delta HAC = 3 -> GE ~ 0.35)
        aryl_preference: 0.35,
    },
    GrowthSynthonSpec {
        code: "GR-CF3",
        name: "Trifluoromethyl (-CF3)",
        dummy_smiles: "IC(F)(F)F",
        role: GrowthPocketRole::HydrophobicAndSp3Dimple,
        aryl_coupling: "Ruppert-Prakash TMSCF3 / CuI Trifluoromethylation",
        hetero_coupling: "2,2,2-Trifluoroethyl (-CH2CF3) Alkylation",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.88,
        aryl_preference: 0.50,
    },
    GrowthSynthonSpec {
        code: "GR-CONHME",
        name: "N-Methylcarbamoyl (-CONHMe)",
        dummy_smiles: "IC(=O)NC",
        role: GrowthPocketRole::PolarHBondAndLleBooster,
        aryl_coupling: "Pd-Catalyzed Aminocarbonylation (CO + MeNH2)",
        hetero_coupling: "Methyl Isocyanate Urea / Carbamate Capping",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: true,
        intrinsic_delta_pact: 1.04, // Bidentate backbone HBD+HBA + lowers cLogP -> strong LLE & GE ~ 0.35!
        aryl_preference: 0.15,
    },
    GrowthSynthonSpec {
        code: "GR-SO2ME",
        name: "Methylsulfonyl (-SO2Me)",
        dummy_smiles: "IS(=O)(=O)C",
        role: GrowthPocketRole::PolarHBondAndLleBooster,
        aryl_coupling: "Cu-Catalyzed NaSO2Me Sulfonylation",
        hetero_coupling: "Mesyl Chloride (MsCl / Et3N) Sulfonamidation",
        attaches_via_heteroatom_or_halide: true,
        is_acyl_or_sulfonyl_cap: true,
        intrinsic_delta_pact: 1.08, // Tetrahedral di-HBA oxyanion probe + strong cLogP drop -> top LLE gain!
        aryl_preference: -0.15,
    },
    GrowthSynthonSpec {
        code: "GR-SO2NH2",
        name: "Sulfonamoyl (-SO2NH2)",
        dummy_smiles: "IS(=O)(=O)N",
        role: GrowthPocketRole::PolarHBondAndLleBooster,
        aryl_coupling: "DABSO / Pd Sulfinate -> NH4OH Sulfonamidation",
        hetero_coupling: "Sulfamide (-NHSO2NH2) Polar Capping",
        attaches_via_heteroatom_or_halide: true,
        is_acyl_or_sulfonyl_cap: true,
        intrinsic_delta_pact: 1.12,
        aryl_preference: 0.10,
    },
    GrowthSynthonSpec {
        code: "GR-OXET",
        name: "Oxetan-3-yl (-Oxetane)",
        dummy_smiles: "IC1COC1",
        role: GrowthPocketRole::PolarHBondAndLleBooster,
        aryl_coupling: "Ni/Photoredox 3-Bromooxetane Cross-Electrophile",
        hetero_coupling: "Oxetan-3-one Reductive Amination / Etherification",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 0.96, // Carreira sp3 polar HBA module: boosts Fsp3 & LLE simultaneously
        aryl_preference: 0.05,
    },
    GrowthSynthonSpec {
        code: "GR-MEPYZ",
        name: "1-Methylpyrazol-4-yl (-MePyz)",
        dummy_smiles: "Ic1cnn(C)c1",
        role: GrowthPocketRole::HeteroarylPiStack,
        aryl_coupling: "Suzuki-Miyaura Pyrazole Pinacol Boronate",
        hetero_coupling: "Chan-Lam / Buchwald Heteroarylation",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 1.48, // Classic Astex sociable heteroaryl HBA + Trp/Phe pi-stacker (Delta HAC = 6 -> GE ~ 0.34)
        aryl_preference: 0.65,
    },
    GrowthSynthonSpec {
        code: "GR-PYR3",
        name: "Pyridin-3-yl (-3-Pyr)",
        dummy_smiles: "Ic1cccnc1",
        role: GrowthPocketRole::HeteroarylPiStack,
        aryl_coupling: "Suzuki-Miyaura 3-Pyridylboronic Acid",
        hetero_coupling: "Buchwald-Hartwig 3-Bromopyridine Coupling",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 1.40,
        aryl_preference: 0.60,
    },
    GrowthSynthonSpec {
        code: "GR-MORPH",
        name: "Morpholin-4-yl (-Morph)",
        dummy_smiles: "IN1CCOCC1",
        role: GrowthPocketRole::SolventFrontAmine,
        aryl_coupling: "Buchwald-Hartwig / SNAr Morpholine Amination",
        hetero_coupling: "2-Morpholinoethyl / Morpholinomethyl Tether",
        attaches_via_heteroatom_or_halide: true,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 1.32, // Solvent-channel HBA + high ESOL & balanced logD
        aryl_preference: 0.55,
    },
    GrowthSynthonSpec {
        code: "GR-4FPH",
        name: "4-Fluorophenyl (-4-F-Ph)",
        dummy_smiles: "Ic1ccc(F)cc1",
        role: GrowthPocketRole::LipophilicArylCleft,
        aryl_coupling: "Suzuki-Miyaura 4-Fluorophenylboronic Acid",
        hetero_coupling: "Chan-Lam / Ullmann 4-Fluoroarylation",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 1.62, // Deep back-pocket / DFG-out hydrophobic cleft engagement (high potency, watch LLE!)
        aryl_preference: 0.45,
    },
    GrowthSynthonSpec {
        code: "GR-BNZ",
        name: "Benzyl (-CH2Ph)",
        dummy_smiles: "ICc1ccccc1",
        role: GrowthPocketRole::LipophilicArylCleft,
        aryl_coupling: "Negishi BnZnBr / Suzuki Benzyltrifluoroborate",
        hetero_coupling: "Benzyl Bromide N/O-Alkylation or Reductive Amination",
        attaches_via_heteroatom_or_halide: false,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 1.54, // Flexible tunnel-pocket aromatic probe (Astex HCV NS3 / kinase back pocket)
        aryl_preference: -0.25,
    },
    GrowthSynthonSpec {
        code: "GR-MEPIP",
        name: "4-Methylpiperazin-1-yl (-MePip)",
        dummy_smiles: "IN1CCN(C)CC1",
        role: GrowthPocketRole::SolventFrontAmine,
        aryl_coupling: "Buchwald-Hartwig RuPhos N-Methylpiperazine",
        hetero_coupling: "Methylpiperazinylmethyl Tether",
        attaches_via_heteroatom_or_halide: true,
        is_acyl_or_sulfonyl_cap: false,
        intrinsic_delta_pact: 1.50, // Salt-bridge Asp/Glu engagement + aqueous solubility rescue (Imatinib/Bosutinib motif)
        aryl_preference: 0.50,
    },
];

/// Evaluates Astex `Rule-of-3` (`Ro3`) compliance, Hopkins `LE`, Leeson-Springthorpe `LLE`/`LLE_7.4`,
/// `SILE`, and Astex synthetic **Fragment Sociability** for any molecule.
pub fn evaluate_rule_of_three_audit(mol: &Molecule) -> RuleOfThreeAudit {
    let alerts = scan_structural_alerts(mol);
    let phys = evaluate_physchem(mol, alerts.len());
    let pka = evaluate_physiological_pka_74(mol);
    let sa = evaluate_synthetic_accessibility(mol);
    let vecs = perceive_fragment_exit_vectors(mol);

    let hac = phys.heavy_atoms.max(1);
    let pact = mol.pact.max(2.50);

    // Congreve et al. (2003) + Jhoti et al. (2013) 6-parameter Rule-of-3 check
    let mut ro3_v = 0usize;
    if phys.mw > 300.0 {
        ro3_v += 1;
    }
    if phys.clogp > 3.0 {
        ro3_v += 1;
    }
    if phys.hbd > 3 {
        ro3_v += 1;
    }
    if phys.hba > 3 {
        ro3_v += 1;
    }
    if phys.rotatable_bonds > 3 {
        ro3_v += 1;
    }
    if phys.tpsa > 60.0 {
        ro3_v += 1;
    }

    // Astex 2013 core 4-parameter practical fragment pass (MW<=300, cLogP<=3, HBD<=3, Nrot<=3)
    let astex_core_pass =
        phys.mw <= 300.0 && phys.clogp <= 3.0 && phys.hbd <= 3 && phys.rotatable_bonds <= 3;

    let seed_le = RT_LN10_KCAL * pact / (hac as f64);
    let seed_lle = pact - phys.clogp;
    let seed_lle_74 = pact - pka.microstate_logd_74;
    let seed_sile = RT_LN10_KCAL * pact / (hac as f64).powf(0.30);

    // Astex Fragment Sociability [0..100]: rewards multiple chemically distinct exit vectors + tractable SAScore
    let mut chem_classes = BTreeSet::new();
    let mut non_hinge_vecs = 0usize;
    for v in &vecs {
        chem_classes.insert(v.chemistry as u8);
        if !v.masks_hinge_hbd {
            non_hinge_vecs += 1;
        }
    }
    let vec_term = ((non_hinge_vecs.min(6) as f64) / 5.0 * 48.0).clamp(0.0, 48.0);
    let div_term = ((chem_classes.len().min(3) as f64) / 3.0 * 24.0).clamp(0.0, 24.0);
    let sa_term = ((5.5 - sa.sa_score).clamp(0.0, 3.5) / 3.5 * 28.0).clamp(0.0, 28.0);
    let sociability_score = (vec_term + div_term + sa_term).clamp(5.0, 100.0);

    RuleOfThreeAudit {
        heavy_atoms: phys.heavy_atoms,
        mw: phys.mw,
        clogp: phys.clogp,
        logd_74: pka.microstate_logd_74,
        hbd: phys.hbd,
        hba: phys.hba,
        rotatable_bonds: phys.rotatable_bonds,
        tpsa: phys.tpsa,
        fsp3: phys.fsp3,
        ro3_violations: ro3_v,
        strict_ro3_pass: ro3_v == 0,
        astex_core_pass,
        sociability_score,
        seed_le,
        seed_lle,
        seed_lle_74,
        seed_sile,
    }
}

/// Deconstructs a larger lead-like molecule (`MW > 300 Da` or `HAC > 20`) down to its central
/// `Rule-of-3` ring/hinge core seed (`HAC <= 18`, `MW <= 295 Da`), preserving direct exocyclic
/// polar H-bond anchors (`-NH2`, `=O`, `-OH` attached to the primary ring system) and calibrating
/// baseline fragment `pAct` to preserve realistic fragment Ligand Efficiency (`LE ~ 0.36..0.52`).
pub fn extract_ro3_seed_core(mol: &Molecule) -> (Molecule, bool) {
    let n = mol.atoms.len();
    let init_hac = mol.atoms.iter().filter(|a| a.elem != Element::H).count();
    let init_mw = mol.molecular_weight();
    if init_hac <= 18 && init_mw <= 300.0 {
        let mut c = mol.clone();
        if c.atoms.iter().all(|a| a.pos2d[0].abs() < 1e-4 && a.pos2d[1].abs() < 1e-4) {
            compute_2d_coords(&mut c);
        }
        return (c, false);
    }

    // 1. Identify connected ring systems in `mol` and score each ring system by heteroaromatic
    //    hinge/pharmacophore richness (N/O/S count + fused bicycle bonus).
    let mut visited = vec![false; n];
    let mut best_ring_sys: Vec<usize> = Vec::new();
    let mut best_score = -1.0f64;

    for start in 0..n {
        if visited[start] || !mol.atoms[start].in_ring || mol.atoms[start].elem == Element::H {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &(v, bi) in &mol.adj[u] {
                if !visited[v] && mol.bonds[bi].in_ring && mol.atoms[v].in_ring {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        if comp.len() > 16 {
            continue;
        }
        let het_cnt = comp
            .iter()
            .filter(|&&u| !matches!(mol.atoms[u].elem, Element::C | Element::H))
            .count();
        let arom_cnt = comp.iter().filter(|&&u| mol.atoms[u].aromatic).count();
        let bicycle_bonus = if (8..=13).contains(&comp.len()) { 6.0 } else { 0.0 };
        let sc = (het_cnt as f64) * 3.5 + (arom_cnt as f64) * 0.8 + bicycle_bonus
            - ((comp.len() as f64 - 9.5).abs() * 0.35);
        if sc > best_score {
            best_score = sc;
            best_ring_sys = comp;
        }
    }

    // Fallback to Murcko framework if no compact ring system <= 16 atoms was found
    let mut keep_set: BTreeSet<usize> = if !best_ring_sys.is_empty() {
        best_ring_sys.into_iter().collect()
    } else {
        let mask = murcko_framework_mask(mol);
        (0..n).filter(|&u| mask[u]).take(14).collect()
    };

    if keep_set.is_empty() {
        return (mol.clone(), false);
    }

    // 2. Preserve 1-atom exocyclic polar H-bond warheads directly bonded to the ring seed:
    //    e.g. carbonyl `=O` (pyridone/quinazolinone), primary `-NH2` / `-OH`, or bridging `-NH-`
    //    converted into an `-NH2` hinge handle if total HAC stays <= 16!
    let ring_snapshot: Vec<usize> = keep_set.iter().copied().collect();
    for u in ring_snapshot {
        for &(v, bi) in &mol.adj[u] {
            if keep_set.contains(&v) {
                continue;
            }
            let av = &mol.atoms[v];
            let b = &mol.bonds[bi];
            let is_exo_carbonyl = b.order == BondOrder::Double
                && matches!(av.elem, Element::O | Element::S)
                && mol.heavy_degree(v) == 1;
            let is_exo_nh_or_oh = b.order == BondOrder::Single
                && matches!(av.elem, Element::N | Element::O)
                && av.charge == 0
                && keep_set.len() < 15;
            if is_exo_carbonyl || is_exo_nh_or_oh {
                keep_set.insert(v);
            }
        }
    }

    // Build induced subgraph of `keep_set`
    let mut old_to_new = vec![None; n];
    let mut seed = Molecule::new(&format!("{}-Ro3Core", mol.id));
    seed.name = format!("{} (Ro3 Core Seed)", mol.name);
    seed.target = mol.target.clone();
    seed.metric = mol.metric.clone();

    for (oi, a) in mol.atoms.iter().enumerate() {
        if !keep_set.contains(&oi) {
            continue;
        }
        let ni = seed.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        na.explicit_h = 0;
        na.bracketed = na.charge != 0;
        seed.atoms.push(na);
        old_to_new[oi] = Some(ni);
    }
    for b in &mol.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = seed.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            seed.bonds.push(nb);
        }
    }
    seed.sanitize();
    let can_smi = molecule_to_smiles(&seed);
    if let Ok(mut reparsed) = parse_smiles(&can_smi) {
        reparsed.id = format!("{}-Ro3", mol.id);
        reparsed.name = format!("{} (Ro3 Seed)", mol.name);
        reparsed.target = mol.target.clone();
        reparsed.metric = mol.metric.clone();
        let seed_hac = reparsed
            .atoms
            .iter()
            .filter(|a| a.elem != Element::H)
            .count()
            .max(5);
        // Scale fragment pAct from parent LE (with slight core-hotspot enrichment typical of hinge cores)
        let parent_le = (RT_LN10_KCAL * mol.pact / (init_hac.max(1) as f64)).clamp(0.28, 0.56);
        let seed_le = (parent_le * 1.12).clamp(0.36, 0.55);
        reparsed.pact = ((seed_le * (seed_hac as f64)) / RT_LN10_KCAL).clamp(3.40, 5.60);
        reparsed.raw_nm = 10f64.powf(9.0 - reparsed.pact);
        reparsed.smiles = can_smi;
        compute_2d_coords(&mut reparsed);
        return (reparsed, true);
    }

    seed.smiles = can_smi;
    compute_2d_coords(&mut seed);
    let seed_hac = seed.atoms.len().max(5);
    seed.pact = ((0.44 * (seed_hac as f64)) / RT_LN10_KCAL).clamp(3.40, 5.50);
    seed.raw_nm = 10f64.powf(9.0 - seed.pact);
    (seed, true)
}

/// Perceives all synthetically & chemically valid peripheral exit vectors on `seed_mol`,
/// computing outward 2D directional arrows `dir2d` and flagging hinge-critical H-bond donors.
pub fn perceive_fragment_exit_vectors(seed_mol: &Molecule) -> Vec<FragmentExitVector> {
    let mut mol = seed_mol.clone();
    if mol.atoms.len() > 1
        && mol
            .atoms
            .iter()
            .all(|a| a.pos2d[0].abs() < 1e-4 && a.pos2d[1].abs() < 1e-4)
    {
        compute_2d_coords(&mut mol);
    }

    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }

    // Molecule 2D centroid for outward vector disambiguation
    let mut mol_cx = 0.0f32;
    let mut mol_cy = 0.0f32;
    let mut cnt = 0.0f32;
    for a in &mol.atoms {
        if a.elem != Element::H {
            mol_cx += a.pos2d[0];
            mol_cy += a.pos2d[1];
            cnt += 1.0;
        }
    }
    if cnt > 0.0 {
        mol_cx /= cnt;
        mol_cy /= cnt;
    }

    let mut raw_vecs: Vec<FragmentExitVector> = Vec::new();

    for u in 0..n {
        let a = &mol.atoms[u];
        if a.elem == Element::H || a.charge != 0 {
            continue;
        }

        // Case A: Poised aryl/heteroaryl halide `C(sp2)-Br`, `C(sp2)-I`, or `C(sp2)-Cl` ->
        //         record on the ipso carbon `u` with `displaced_halide_atom = Some(hal_idx)`!
        if matches!(a.elem, Element::Br | Element::I | Element::Cl) {
            continue;
        }

        let poised_halide = if a.elem == Element::C && (a.aromatic || a.hyb == Hybridization::Sp2) {
            mol.adj[u].iter().find_map(|&(v, bi)| {
                if mol.bonds[bi].order == BondOrder::Single
                    && matches!(mol.atoms[v].elem, Element::I | Element::Br | Element::Cl)
                    && mol.heavy_degree(v) == 1
                {
                    Some(v)
                } else {
                    None
                }
            })
        } else {
            None
        };

        let th = a.total_h();
        if poised_halide.is_none() && th == 0 {
            continue;
        }

        let chemistry = if poised_halide.is_some() {
            ExitVectorChemistry::PoisedArylHalide
        } else {
            match a.elem {
                Element::C => {
                    if a.aromatic || a.hyb == Hybridization::Sp2 {
                        ExitVectorChemistry::ArylOrHeteroarylCH
                    } else if a.hyb == Hybridization::Sp3 {
                        ExitVectorChemistry::AliphaticSp3CH
                    } else {
                        continue;
                    }
                }
                Element::N => {
                    if a.in_ring {
                        ExitVectorChemistry::HeteroarylNH
                    } else {
                        ExitVectorChemistry::ExocyclicAmineOrHydroxyl
                    }
                }
                Element::O => {
                    if !a.in_ring {
                        ExitVectorChemistry::ExocyclicAmineOrHydroxyl
                    } else {
                        continue;
                    }
                }
                _ => continue,
            }
        };

        // Detect whether this exit vector sits on a bidentate kinase-hinge HBD motif:
        // e.g. pyrrolic/pyrazolic `[nH]` adjacent to pyridinic `n`, OR exocyclic `-NH2` ortho to ring `n`!
        let masks_hinge_hbd = match chemistry {
            ExitVectorChemistry::HeteroarylNH => mol.adj[u].iter().any(|&(nb, _)| {
                mol.atoms[nb].elem == Element::N
                    || mol.adj[nb].iter().any(|&(nb2, _)| {
                        nb2 != u
                            && mol.atoms[nb2].elem == Element::N
                            && mol.atoms[nb2].in_ring
                            && mol.atoms[nb2].total_h() == 0
                    })
            }),
            ExitVectorChemistry::ExocyclicAmineOrHydroxyl if a.elem == Element::N => {
                // If `-NH2` (2 H's), monosubstitution leaves 1 `N-H` intact (`masks_hinge_hbd = false`!);
                // only flag if only 1 H remains (`-NHR` -> `-NRR'` eliminates last HBD)
                th == 1
                    && mol.adj[u].iter().any(|&(nb, _)| {
                        mol.atoms[nb].in_ring
                            && mol.adj[nb]
                                .iter()
                                .any(|&(nb2, _)| mol.atoms[nb2].elem == Element::N)
                    })
            }
            _ => false,
        };

        // Compute outward unit 2D vector:
        // If displacing a poised halide, point straight along `u -> halide`!
        // Otherwise point opposite the vector sum of heavy neighbor bonds `sum(pos[nb] - pos[u])`.
        let dir2d = if let Some(hx) = poised_halide {
            let dx = mol.atoms[hx].pos2d[0] - a.pos2d[0];
            let dy = mol.atoms[hx].pos2d[1] - a.pos2d[1];
            let len = (dx * dx + dy * dy).sqrt().max(1e-4);
            [dx / len, dy / len]
        } else {
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            for &(nb, _) in &mol.adj[u] {
                if mol.atoms[nb].elem == Element::H {
                    continue;
                }
                let vx = mol.atoms[nb].pos2d[0] - a.pos2d[0];
                let vy = mol.atoms[nb].pos2d[1] - a.pos2d[1];
                let vl = (vx * vx + vy * vy).sqrt().max(1e-4);
                sx += vx / vl;
                sy += vy / vl;
            }
            let mut ox = -sx;
            let mut oy = -sy;
            let olen = (ox * ox + oy * oy).sqrt();
            if olen < 0.08 {
                ox = a.pos2d[0] - mol_cx;
                oy = a.pos2d[1] - mol_cy;
            }
            let flen = (ox * ox + oy * oy).sqrt().max(1e-4);
            [ox / flen, oy / flen]
        };

        // Steric openness: penalize crowded `ortho`-disubstituted positions (`deg == 2` with both neighbors deg >= 3)
        let crowded_nbrs = mol.adj[u]
            .iter()
            .filter(|&&(nb, _)| Some(nb) != poised_halide && mol.heavy_degree(nb) >= 3)
            .count();
        let steric_openness = (1.0 - (crowded_nbrs as f64) * 0.22).clamp(0.35, 1.0);

        let base_soc = match chemistry {
            ExitVectorChemistry::PoisedArylHalide => 98.0,
            ExitVectorChemistry::ExocyclicAmineOrHydroxyl => 92.0,
            ExitVectorChemistry::ArylOrHeteroarylCH => 85.0,
            ExitVectorChemistry::HeteroarylNH => {
                if masks_hinge_hbd {
                    52.0
                } else {
                    82.0
                }
            }
            ExitVectorChemistry::AliphaticSp3CH => 74.0,
        };
        let sociability_weight =
            (base_soc * (0.72 + 0.28 * steric_openness) - if masks_hinge_hbd { 18.0 } else { 0.0 })
                .clamp(20.0, 99.5);

        let atom_label = if let Some(hx) = poised_halide {
            format!(
                "{}{}(-{}->{})",
                a.elem.symbol(),
                u + 1,
                mol.atoms[hx].elem.symbol(),
                hx + 1
            )
        } else {
            format!("{}{}", a.elem.symbol(), u + 1)
        };

        let rationale = match (chemistry, masks_hinge_hbd) {
            (ExitVectorChemistry::PoisedArylHalide, _) => {
                "Pre-functionalized C(sp2)-X poised handle for direct Suzuki/Buchwald library synthesis".to_string()
            }
            (ExitVectorChemistry::ExocyclicAmineOrHydroxyl, false) => {
                "High-sociability nucleophilic handle (HATU amide, sulfonylation, SNAr, or reductive amination); retains >=1 HBD if -NH2".to_string()
            }
            (ExitVectorChemistry::ArylOrHeteroarylCH, _) => {
                if steric_openness >= 0.78 {
                    "Unhindered aromatic C(sp2)-H exit vector for regioselective halogenation/borylation -> Suzuki/Buchwald".to_string()
                } else {
                    "Peri/ortho-adjacent C(sp2)-H vector — induces conformational dihedral twist (magic-methyl / atropisomeric probe)".to_string()
                }
            }
            (ExitVectorChemistry::HeteroarylNH, true) => {
                "Caution: Caps bidentate hinge-type N-H donor! Useful only if flipping binding mode or probing N-alkyl selectivity".to_string()
            }
            (ExitVectorChemistry::HeteroarylNH, false) => {
                "Azole/lactam N-H exit vector for Chan-Lam N-arylation or N-alkylation".to_string()
            }
            (ExitVectorChemistry::AliphaticSp3CH, _) => {
                "Three-dimensional sp3 exit vector for out-of-plane subpocket exploration".to_string()
            }
            (ExitVectorChemistry::ExocyclicAmineOrHydroxyl, true) => {
                "Secondary polar handle — capping removes remaining HBD; favor HBA/amide compensation".to_string()
            }
        };

        raw_vecs.push(FragmentExitVector {
            vector_idx: 0,
            anchor_atom: u,
            displaced_halide_atom: poised_halide,
            atom_label,
            chemistry,
            dir2d,
            masks_hinge_hbd,
            steric_openness,
            sociability_weight,
            rationale,
        });
    }

    raw_vecs.sort_by(|a, b| {
        b.sociability_weight
            .partial_cmp(&a.sociability_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.anchor_atom.cmp(&b.anchor_atom))
    });
    for (i, v) in raw_vecs.iter_mut().enumerate() {
        v.vector_idx = i;
    }
    raw_vecs
}

fn parse_one_dummy_synthon(dummy_smiles: &str) -> Option<(Molecule, usize)> {
    let raw = parse_smiles(dummy_smiles).ok()?;
    let dummy_idx = raw.atoms.iter().position(|a| a.elem == Element::I)?;
    let attach_old = raw.adj[dummy_idx].first()?.0;

    let mut old_to_new = vec![None; raw.atoms.len()];
    let mut out = Molecule::new("");
    for (oi, a) in raw.atoms.iter().enumerate() {
        if oi == dummy_idx {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == attach_old {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[oi] = Some(ni);
    }
    for b in &raw.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = out.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }
    Some((out, old_to_new[attach_old]?))
}

/// Grafts `synthon` onto `seed` at `exit_vec`, optionally inserting a `-CH2-` methylene spacer
/// when direct heteroatom-heteroatom bonding (`N-F`, `N-OMe`, `N-NR2`, `O-OMe`, `O-F`, `O-SO2R`)
/// would be chemically unstable.
fn assemble_grown_fragment(
    seed: &Molecule,
    exit_vec: &FragmentExitVector,
    synthon: &GrowthSynthonSpec,
) -> Option<(Molecule, Vec<usize>, Vec<usize>, bool)> {
    let anchor_elem = seed.atoms.get(exit_vec.anchor_atom)?.elem;
    let anchor_is_n = anchor_elem == Element::N;
    let anchor_is_o = anchor_elem == Element::O;

    // Determine chemical compatibility or need for a 1-carbon `-CH2-` spacer:
    // - On C (aryl C-H, aryl C-X, sp3 C-H): all 16 synthons attach directly!
    // - On N (`-NH-` / `-NH2`): acyl/sulfonyl caps (`-CONHMe`, `-SO2Me`, `-SO2NH2`) and carbon-attached
    //   synthons (`-Me`, `-cPr`, `-Oxet`, `-MePyz`, `-3-Pyr`, `-4-F-Ph`, `-Bn`, `-CF3` via `-CH2CF3`? `-NCF3` is rare,
    //   route `-CF3`, `-CN`, `-F`, `-OMe`, `-Morph`, `-MePip` on N/O via `-CH2-` spacer so e.g. `-NH-CH2CF3`,
    //   `-NH-CH2CN`, `-NH-CH2CH2...` / `-O-CH2-...` are 100% bench-stable!).
    let needs_ch2_spacer = if anchor_is_n {
        synthon.attaches_via_heteroatom_or_halide && !synthon.is_acyl_or_sulfonyl_cap
            || synthon.code == "GR-CF3"
            || synthon.code == "GR-CN"
    } else if anchor_is_o {
        if synthon.code == "GR-F" || synthon.is_acyl_or_sulfonyl_cap && synthon.code != "GR-CONHME" {
            return None; // Skip aryl/aliphatic sulfonate esters (alkylating alerts!) and hypofluorites
        }
        synthon.attaches_via_heteroatom_or_halide || synthon.code == "GR-CN" || synthon.code == "GR-CF3"
    } else {
        false
    };

    let mut old_to_new = vec![None; seed.atoms.len()];
    let mut out = Molecule::new("");
    let mut seed_kept_new = Vec::new();

    for (oi, a) in seed.atoms.iter().enumerate() {
        if Some(oi) == exit_vec.displaced_halide_atom {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == exit_vec.anchor_atom {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[oi] = Some(ni);
        seed_kept_new.push(ni);
    }

    for b in &seed.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = out.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }

    let anchor_new = old_to_new[exit_vec.anchor_atom]?;
    let (syn_mol, syn_attach) = parse_one_dummy_synthon(synthon.dummy_smiles)?;

    let mut grown_new = Vec::new();
    let link_left = if needs_ch2_spacer {
        let ch2_idx = out.atoms.len();
        let mut ch2 = seed.atoms[exit_vec.anchor_atom].clone();
        ch2.idx = ch2_idx;
        ch2.elem = Element::C;
        ch2.charge = 0;
        ch2.aromatic = false;
        ch2.in_ring = false;
        ch2.explicit_h = 0;
        ch2.implicit_h = 2;
        ch2.bracketed = false;
        ch2.hyb = Hybridization::Sp3;
        out.atoms.push(ch2);
        grown_new.push(ch2_idx);

        let b_sp = out.bonds.len();
        out.bonds.push(Bond {
            idx: b_sp,
            a: anchor_new,
            b: ch2_idx,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
        ch2_idx
    } else {
        anchor_new
    };

    let offset = out.atoms.len();
    for (si, sa) in syn_mol.atoms.iter().enumerate() {
        let mut na = sa.clone();
        na.idx = offset + si;
        out.atoms.push(na);
        grown_new.push(offset + si);
    }
    for sb in &syn_mol.bonds {
        let bi = out.bonds.len();
        let mut nb = sb.clone();
        nb.idx = bi;
        nb.a = offset + sb.a;
        nb.b = offset + sb.b;
        out.bonds.push(nb);
    }

    let b_join = out.bonds.len();
    out.bonds.push(Bond {
        idx: b_join,
        a: link_left,
        b: offset + syn_attach,
        order: BondOrder::Single,
        aromatic: false,
        in_ring: false,
        kekule_order: 1,
    });

    out.sanitize();
    let can_smi = molecule_to_smiles(&out);
    // Verify canonical SMILES roundtrips cleanly with matching heavy-atom count
    let check = parse_smiles(&can_smi).ok()?;
    let exp_hac = out.atoms.iter().filter(|a| a.elem != Element::H).count();
    let chk_hac = check.atoms.iter().filter(|a| a.elem != Element::H).count();
    if exp_hac != chk_hac {
        return None;
    }
    out.smiles = can_smi;
    compute_2d_coords(&mut out);
    Some((out, seed_kept_new, grown_new, needs_ch2_spacer))
}

/// Enumerates and multiparameter-scores (`GE`, `LLE`, `LE`, `SILE`, `QED`, `CNS-MPO`, `SAScore`)
/// all chemically valid fragment-growing elaborations along `vector_choice` on `input_mol`.
///
/// If `deconstruct_to_ro3_if_large` is `true` and `input_mol` exceeds `Rule-of-3` size (`MW > 300 Da`
/// or `HAC > 18`), first extracts the central `Ro3` pharmacophore core seed so fragment-to-lead
/// growing starts from a genuine `Rule-of-3` hit.
pub fn grow_fragment_along_vector(
    input_mol: &Molecule,
    vector_choice: usize,
    deconstruct_to_ro3_if_large: bool,
    dataset_mols_and_fps: Option<(&[Molecule], &[Ecfp2048])>,
) -> FragmentGrowthReport {
    let (seed_mol, was_deconstructed) = if deconstruct_to_ro3_if_large {
        extract_ro3_seed_core(input_mol)
    } else {
        let mut c = input_mol.clone();
        if c.atoms
            .iter()
            .all(|a| a.pos2d[0].abs() < 1e-4 && a.pos2d[1].abs() < 1e-4)
        {
            compute_2d_coords(&mut c);
        }
        (c, false)
    };

    let ro3_audit = evaluate_rule_of_three_audit(&seed_mol);
    let exit_vectors = perceive_fragment_exit_vectors(&seed_mol);
    if exit_vectors.is_empty() {
        return FragmentGrowthReport {
            input_name: input_mol.name.clone(),
            was_deconstructed_to_ro3: was_deconstructed,
            seed_mol,
            ro3_audit,
            exit_vectors,
            active_vector_idx: 0,
            candidates: Vec::new(),
        };
    }

    let active_vector_idx = vector_choice.min(exit_vectors.len() - 1);
    let ev = &exit_vectors[active_vector_idx];
    let seed_fp = compute_ecfp(&seed_mol, 2);

    // Note: if displacing a poised halide (`-Br`/`-I`/`-Cl`), reference heavy-atom count for
    // Verdonk-Rees GE is the dehalogenated fragment core (`seed_hac_effective`), with minimum `delta_hac >= 1`.
    let seed_hac_eff = ro3_audit
        .heavy_atoms
        .saturating_sub(usize::from(ev.displaced_halide_atom.is_some()))
        .max(1);
    let seed_pact = seed_mol.pact.max(2.50);

    let is_aryl_vec = matches!(
        ev.chemistry,
        ExitVectorChemistry::PoisedArylHalide | ExitVectorChemistry::ArylOrHeteroarylCH
    );

    let mut candidates = Vec::with_capacity(GROWTH_SYNTHONS.len());

    for spec in GROWTH_SYNTHONS {
        let Some((mut grown_mol, seed_atoms, grown_atoms, used_spacer)) =
            assemble_grown_fragment(&seed_mol, ev, spec)
        else {
            continue;
        };

        let alerts = scan_structural_alerts(&grown_mol);
        let phys = evaluate_physchem(&grown_mol, alerts.len());
        let pka = evaluate_physiological_pka_74(&grown_mol);
        let sa = evaluate_synthetic_accessibility(&grown_mol);
        let grown_fp = compute_ecfp(&grown_mol, 2);

        let grown_hac = phys.heavy_atoms.max(seed_hac_eff + 1);
        let delta_hac = (grown_hac - seed_hac_eff).max(1);

        // 1. Subpocket biophysical group-contribution model:
        //    - Magic-methyl conformational twist bonus at ortho-substituted aryl positions!
        let ortho_twist_bonus = if spec.code == "GR-ME" && ev.steric_openness < 0.72 && is_aryl_vec {
            0.34
        } else {
            0.0
        };
        let steric_factor = if delta_hac >= 5 && ev.steric_openness < 0.65 {
            0.74
        } else {
            0.88 + 0.14 * ev.steric_openness
        };
        let vec_match = if is_aryl_vec {
            1.0 + 0.16 * spec.aryl_preference
        } else {
            1.0 - 0.12 * spec.aryl_preference
        };
        let hinge_penalty = if ev.masks_hinge_hbd { -0.68 } else { 0.0 };
        let spacer_adj = if used_spacer { 0.08 } else { 0.0 };

        let mut delta_pact =
            spec.intrinsic_delta_pact * steric_factor * vec_match + ortho_twist_bonus + spacer_adj + hinge_penalty;

        // Diminishing marginal efficiency on already large leads (`HAC > 26`) so fragments (`HAC 8..16`)
        // show realistic high GE while oversized molecules plateau toward lead ceilings:
        if seed_hac_eff > 22 {
            let damp = (1.0 - ((seed_hac_eff - 22) as f64) * 0.045).clamp(0.32, 1.0);
            delta_pact *= damp;
        }

        // 2. Empirical Dataset Nearest-Neighbor SAR Transfer (when dataset is supplied):
        let mut nearest_sim = 0.0f32;
        let mut nearest_id: Option<String> = None;
        if let Some((ds_mols, ds_fps)) = dataset_mols_and_fps {
            for (dm, dfp) in ds_mols.iter().zip(ds_fps.iter()) {
                let sim_grown = grown_fp.tanimoto(dfp);
                if sim_grown > nearest_sim {
                    nearest_sim = sim_grown;
                    nearest_id = Some(dm.id.clone());
                }
                let sim_seed = seed_fp.tanimoto(dfp);
                // If grown structure moves closer in ECFP4 space to a known potent experimental analog (`sim >= 0.42`),
                // blend in observed dataset potency signal!
                if sim_grown >= 0.44 && sim_grown > sim_seed + 0.03 {
                    let w = ((sim_grown as f64 - 0.42) * 0.85).clamp(0.0, 0.42);
                    let obs_delta = (dm.pact - seed_pact).clamp(-0.6, 3.2);
                    delta_pact = (1.0 - w) * delta_pact + w * obs_delta;
                }
            }
        }

        let predicted_pact = (seed_pact + delta_pact).clamp(2.80, 10.40);
        let eff_delta_pact = predicted_pact - seed_pact;
        let delta_dg_kcal = -RT_LN10_KCAL * eff_delta_pact;

        let le = RT_LN10_KCAL * predicted_pact / (grown_hac as f64);
        let delta_le = le - ro3_audit.seed_le;
        let ge = RT_LN10_KCAL * eff_delta_pact / (delta_hac as f64);
        let lle = predicted_pact - phys.clogp;
        let delta_lle = lle - ro3_audit.seed_lle;
        let lle_74 = predicted_pact - pka.microstate_logd_74;
        let sile = RT_LN10_KCAL * predicted_pact / (grown_hac as f64).powf(0.30);

        let regime = if ge >= 0.35 && delta_lle >= -0.25 && alerts.is_empty() {
            GrowthEfficiencyRegime::HotspotEngaging
        } else if ge >= 0.25 && le >= 0.28 {
            GrowthEfficiencyRegime::ProductiveLeadLike
        } else if delta_lle >= 0.20 && eff_delta_pact > 0.15 {
            GrowthEfficiencyRegime::PropertyTuning
        } else {
            GrowthEfficiencyRegime::UnproductiveBulk
        };

        // Composite Fragment-to-Lead Optimizer Score [0..100]:
        // Rewards high Verdonk-Rees GE (target >= 0.30), positive Delta LLE / high absolute LLE,
        // preserved LE >= 0.30, high QED, low SAScore, and zero PAINS alerts.
        let ge_term = ((ge / 0.48) * 34.0).clamp(0.0, 38.0);
        let lle_term = (((lle - 2.0) / 4.5) * 18.0 + (delta_lle + 0.6) * 5.5).clamp(0.0, 26.0);
        let le_term = (((le - 0.22) / 0.28) * 16.0).clamp(0.0, 16.0);
        let qed_sa_term =
            (phys.qed_mean * 12.0 + ((5.5 - sa.sa_score).clamp(0.0, 3.5) / 3.5) * 8.0)
                .clamp(0.0, 20.0);
        let alert_pen = (alerts.len() as f64) * 18.0;
        let optimizer_score = (ge_term + lle_term + le_term + qed_sa_term - alert_pen).clamp(2.0, 99.6);

        let cid = format!("F2L-V{}-{}", active_vector_idx + 1, &spec.code[3..]);
        grown_mol.id = cid.clone();
        grown_mol.name = format!("{} + {}", seed_mol.name, spec.name);
        grown_mol.target = seed_mol.target.clone();
        grown_mol.metric = seed_mol.metric.clone();
        grown_mol.pact = predicted_pact;
        grown_mol.raw_nm = 10f64.powf(9.0 - predicted_pact);

        let coupling_reaction = if is_aryl_vec {
            spec.aryl_coupling
        } else {
            spec.hetero_coupling
        };

        candidates.push(GrownFragmentCandidate {
            rank: 0,
            candidate_id: cid,
            synthon_code: spec.code,
            synthon_name: spec.name,
            coupling_reaction,
            pocket_role: spec.role,
            vector_idx: active_vector_idx,
            vector_label: ev.atom_label.clone(),
            used_methylene_spacer: used_spacer,
            efficiency: EfficiencyMetrics {
                seed_hac: seed_hac_eff,
                grown_hac,
                delta_hac,
                seed_pact,
                predicted_pact,
                delta_pact: eff_delta_pact,
                delta_dg_kcal,
                le,
                delta_le,
                ge,
                lle,
                delta_lle,
                lle_74,
                sile,
                nearest_dataset_tanimoto: nearest_sim,
                nearest_dataset_id: nearest_id,
            },
            regime,
            physchem: phys,
            logd_74: pka.microstate_logd_74,
            sa_score: sa.sa_score,
            optimizer_score,
            canonical_smiles: grown_mol.smiles.clone(),
            seed_atoms,
            grown_atoms,
            mol: grown_mol,
        });
    }

    candidates.sort_by(|a, b| {
        b.optimizer_score
            .partial_cmp(&a.optimizer_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.rank = i + 1;
    }

    FragmentGrowthReport {
        input_name: input_mol.name.clone(),
        was_deconstructed_to_ro3: was_deconstructed,
        seed_mol,
        ro3_audit,
        exit_vectors,
        active_vector_idx,
        candidates,
    }
}

/// Evaluates `Rule-of-3`, `LE`, `LLE`, `LLE_7.4`, `SILE`, and exit-vector sociability across an entire dataset.
pub fn summarize_dataset_fragment_efficiency(mols: &[Molecule]) -> Vec<FragmentDatasetRow> {
    mols.iter()
        .enumerate()
        .map(|(idx, m)| {
            let aud = evaluate_rule_of_three_audit(m);
            let vecs = perceive_fragment_exit_vectors(m);
            FragmentDatasetRow {
                mol_idx: idx,
                id: m.id.clone(),
                name: m.name.clone(),
                hac: aud.heavy_atoms,
                mw: aud.mw,
                pact: m.pact,
                le: aud.seed_le,
                lle: aud.seed_lle,
                lle_74: aud.seed_lle_74,
                sile: aud.seed_sile,
                ro3_violations: aud.ro3_violations,
                strict_ro3_pass: aud.strict_ro3_pass,
                exit_vector_count: vecs.len(),
                sociability_score: aud.sociability_score,
            }
        })
        .collect()
}

/// Exports grown fragment-to-lead candidates with `GE`, `LLE`, `LE`, `SILE`, `QED`, and `SAScore` to CSV.
pub fn export_grown_fragments_csv(cands: &[GrownFragmentCandidate]) -> String {
    let mut out = String::from(
        "rank,id,vector,synthon_code,synthon_name,role,coupling,seed_hac,grown_hac,delta_hac,seed_pact,predicted_pact,delta_pact,delta_dg_kcal,ge_kcal_ha,le_kcal_ha,lle,delta_lle,lle_74,sile,mw,clogp,logd_74,tpsa,fsp3,qed,cns_mpo,sa_score,regime,optimizer_score,smiles\n",
    );
    for c in cands {
        let e = &c.efficiency;
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{},{},{},{:.2},{:.2},{:+.2},{:+.2},{:+.3},{:.3},{:.2},{:+.2},{:.2},{:.2},{:.1},{:.2},{:.2},{:.1},{:.2},{:.3},{:.2},{:.2},\"{}\",{:.1},\"{}\"\n",
            c.rank,
            c.candidate_id,
            c.vector_label,
            c.synthon_code,
            c.synthon_name,
            c.pocket_role.short_label(),
            c.coupling_reaction,
            e.seed_hac,
            e.grown_hac,
            e.delta_hac,
            e.seed_pact,
            e.predicted_pact,
            e.delta_pact,
            e.delta_dg_kcal,
            e.ge,
            e.le,
            e.lle,
            e.delta_lle,
            e.lle_74,
            e.sile,
            c.physchem.mw,
            c.physchem.clogp,
            c.logd_74,
            c.physchem.tpsa,
            c.physchem.fsp3,
            c.physchem.qed_mean,
            c.physchem.cns_mpo,
            c.sa_score,
            c.regime.short_label(),
            c.optimizer_score,
            c.canonical_smiles
        ));
    }
    out
}

/// Formats grown fragment-to-lead candidates as a 2D MDL V2000 `.sdf` bundle with efficiency tags.
pub fn format_grown_fragments_sdf(cands: &[GrownFragmentCandidate]) -> String {
    let mut out = String::new();
    for c in cands {
        let e = &c.efficiency;
        let extra = [
            ("EXIT_VECTOR", c.vector_label.clone()),
            ("SYNTHON", c.synthon_name.to_string()),
            ("COUPLING_RXN", c.coupling_reaction.to_string()),
            ("GROUP_EFFICIENCY_GE", format!("{:+.3}", e.ge)),
            ("LIGAND_EFFICIENCY_LE", format!("{:.3}", e.le)),
            ("LLE_CLOGP", format!("{:.2}", e.lle)),
            ("LLE_LOGD74", format!("{:.2}", e.lle_74)),
            ("SILE", format!("{:.2}", e.sile)),
            ("OPTIMIZER_SCORE", format!("{:.1}", c.optimizer_score)),
        ];
        let extra_refs: Vec<(&str, String)> = extra.into_iter().collect();
        let mut rec = format_sdf_record(&c.mol);
        if let Some(pos) = rec.rfind("$$$$") {
            let mut tag_block = String::new();
            for (k, v) in &extra_refs {
                tag_block.push_str(&format!(">  <{k}>\n{v}\n\n"));
            }
            rec.insert_str(pos, &tag_block);
        }
        out.push_str(&rec);
    }
    out
}
