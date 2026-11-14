//! Bioreversible Prodrug Designer: `BCS` Solubility- & Permeability-Masking Promoiety
//! Enumerator (`Ester`, `Phosphate`, `Carbamate`, `Oxazolidinone`, & `hPEPT1` Amino-Acid
//! Carrier) and Tissue-Specific Enzymatic Activation (`CES1` / `CES2` / `ALP` / `BPHL` /
//! `BChE`) + Oral Bioconversion Kinetic Profiler (`v1.8.0`).
//!
//! Primary References:
//! - Rautio et al. (2008), *Nat. Rev. Drug Discov.* 7(3), 255-270
//!   ("Prodrugs: design and clinical applications": `https://pubmed.ncbi.nlm.nih.gov/18219308/`)
//! - Rautio, Meanwell, Di & Hageman (2018), *Nat. Rev. Drug Discov.* 17(8), 559-587
//!   ("The expanding role of prodrugs in contemporary drug design and development":
//!   `https://pubmed.ncbi.nlm.nih.gov/29700501/`)
//! - Wang et al. (2018) & Di (2019), Human Carboxylesterases `CES1` (hepatic; large acyl / small
//!   alcohol) vs `CES2` (intestinal enterocyte; small acyl / bulky alcohol or phenol) Specificity
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6146386/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC6635651/`)
//! - Yuan et al. (2009) & Wiemer (2016), Intestinal Brush-Border Alkaline Phosphatase (`ALP`)
//!   Phosphate & Phosphonooxymethyl Prodrugs (`https://pubmed.ncbi.nlm.nih.gov/19372225/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4774048/`)
//! - Simplício, Clancy & Gilmer (2008), *Molecules* 13(3), 519-547 ("Prodrugs for Amines":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC6245426/`) & Carbamate / Oxazolidinone Prodrugs
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC7968508/`, `https://pubmed.ncbi.nlm.nih.gov/7562932/`)

use crate::admet::{evaluate_admet, BoiledEggRegion};
use crate::alerts::scan_structural_alerts;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Bond, BondOrder, Element, Molecule};
use crate::mol_io::{format_sdf_record, parse_smiles};
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use serde::{Deserialize, Serialize};

/// Number of time-grid points across `0.0..=8.0 hours` for oral prodrug -> active drug PK curves.
pub const PRODRUG_TIME_GRID_LEN: usize = 33;

/// Curated FDA / clinical prodrug parent archetype illustrating textbook `BCS II/III/IV` rescue.
#[derive(Clone, Copy, Debug)]
pub struct CuratedProdrugParentSpec {
    pub code: &'static str,
    pub name: &'static str,
    pub smiles: &'static str,
    pub marketed_prodrug: &'static str,
    pub target_name: &'static str,
    pub parent_pact: f64,
}

pub const CURATED_PRODRUG_PARENTS: &[CuratedProdrugParentSpec] = &[
    CuratedProdrugParentSpec {
        code: "ARC-OSLT",
        name: "Oseltamivir Carboxylate",
        smiles: "CCC(CC)OC1CC(=CC(N)C1NC(=O)C)C(=O)O",
        marketed_prodrug: "Oseltamivir (Tamiflu) — CES1 Hepatic Ethyl Ester (Fa 5% -> 80%)",
        target_name: "Influenza Neuraminidase",
        parent_pact: 8.70,
    },
    CuratedProdrugParentSpec {
        code: "ARC-TEDZ",
        name: "Tedizolid (Oxazolidinone Alcohol)",
        smiles: "Cn1nnnc1-c1ccc(cc1)-c1ccc(F)c(c1)N1CC(CO)OC1=O",
        marketed_prodrug: "Tedizolid Phosphate (Sivextro) — Brush-Border ALP Phosphate",
        target_name: "Bacterial 50S Ribosome",
        parent_pact: 6.60,
    },
    CuratedProdrugParentSpec {
        code: "ARC-ACVR",
        name: "Acyclovir",
        smiles: "NC1=NC(=O)c2ncn(COCCO)c2N1",
        marketed_prodrug: "Valaciclovir (Valtrex) — hPEPT1 + BPHL L-Valinate Ester (F 15% -> 55%)",
        target_name: "HSV DNA Polymerase",
        parent_pact: 6.90,
    },
    CuratedProdrugParentSpec {
        code: "ARC-MYCO",
        name: "Mycophenolic Acid",
        smiles: "COc1c(C)c2COC(=O)c2c(O)c1CC=C(C)CCC(=O)O",
        marketed_prodrug: "Mycophenolate Mofetil (CellCept) — CES1/2 Morpholinoethyl Ester",
        target_name: "IMPDH Type II",
        parent_pact: 7.85,
    },
    CuratedProdrugParentSpec {
        code: "ARC-VALD",
        name: "Valdecoxib",
        smiles: "Cc1c(-c2ccc(S(=O)(=O)N)cc2)c(-c2ccccc2)no1",
        marketed_prodrug: "Parecoxib (Dynastat) — N-Propionyl Sulfonamide Solubility Prodrug",
        target_name: "COX-2",
        parent_pact: 8.30,
    },
    CuratedProdrugParentSpec {
        code: "ARC-DABI",
        name: "Benzamidine-Carboxylate Core",
        smiles: "NC(=N)c1ccc(CNC(=O)CCC(=O)O)cc1",
        marketed_prodrug: "Dabigatran Etexilate Archetype — Carbamate + CES1 Ester Zwitterion Mask",
        target_name: "Thrombin (Factor IIa)",
        parent_pact: 8.15,
    },
];

/// Builds a sanitized `Molecule` with 2D coordinates for one of `CURATED_PRODRUG_PARENTS`.
pub fn build_curated_prodrug_parent(idx: usize) -> Molecule {
    let spec = &CURATED_PRODRUG_PARENTS[idx.min(CURATED_PRODRUG_PARENTS.len() - 1)];
    let mut m = parse_smiles(spec.smiles).unwrap_or_else(|_| Molecule::new(spec.code));
    m.id = spec.code.to_string();
    m.name = spec.name.to_string();
    m.target = spec.target_name.to_string();
    m.metric = "IC50".to_string();
    m.pact = spec.parent_pact;
    m.raw_nm = 10f64.powf(9.0 - spec.parent_pact);
    m.smiles = molecule_to_smiles(&m);
    compute_2d_coords(&mut m);
    m
}

/// Biopharmaceutics Classification System (`BCS`) class driving rational promoiety selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BcsClass {
    /// High solubility, high passive permeability (`F_oral` already favorable; depot/CNS/metabolic-shielding only).
    ClassI,
    /// Low aqueous solubility ("brick-dust"), high passive membrane permeability -> favor polar/ionizable phosphate, succinate, or acyl-sulfonamide promoieties.
    ClassII,
    /// High aqueous solubility, low passive membrane permeability (polar/zwitterionic) -> favor lipophilic ester, carbamate, oxazolidinone, or `PEPT1` carrier promoieties.
    ClassIII,
    /// Low solubility and low passive permeability -> favor amphiphilic mofetil or `PEPT1` L-valyl promoieties.
    ClassIV,
}

impl BcsClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::ClassI => "BCS Class I (Soluble & Permeable — Optional Shield/Depot)",
            Self::ClassII => "BCS Class II (Dissolution-Limited -> Favor Phosphate/Ionizable)",
            Self::ClassIII => "BCS Class III (Permeability-Limited -> Favor Ester/Carbamate/PEPT1)",
            Self::ClassIV => "BCS Class IV (Dual Sol/Perm Barrier -> Favor Mofetil/PEPT1/Self-Immolative)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ClassI => "BCS I",
            Self::ClassII => "BCS II (Low Sol)",
            Self::ClassIII => "BCS III (Low Perm)",
            Self::ClassIV => "BCS IV (Low Sol/Perm)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::ClassI => [52, 211, 153],   // Emerald
            Self::ClassII => [56, 189, 248],  // Sky Blue
            Self::ClassIII => [251, 191, 36], // Amber
            Self::ClassIV => [248, 113, 113], // Coral
        }
    }
}

/// Chemical class of the maskable parent functional group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProdrugHandleKind {
    CarboxylicAcid,
    AliphaticHydroxyl,
    PhenolicHydroxyl,
    PrimaryOrSecondaryAmine,
    AmidineOrGuanidine,
    PrimarySulfonamide,
    AzoleOrLactamNH,
}

impl ProdrugHandleKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::CarboxylicAcid => "Carboxylic Acid (-COOH)",
            Self::AliphaticHydroxyl => "Aliphatic Hydroxyl (R-OH)",
            Self::PhenolicHydroxyl => "Phenolic Hydroxyl (Ar-OH)",
            Self::PrimaryOrSecondaryAmine => "Aliphatic / Aniline Amine (-NH2 / -NHR)",
            Self::AmidineOrGuanidine => "Basic Amidine / Guanidine (-C(=NH)NH2)",
            Self::PrimarySulfonamide => "Primary Sulfonamide (-SO2NH2)",
            Self::AzoleOrLactamNH => "Heteroaryl Azole / Lactam (N-H)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::CarboxylicAcid => "-COOH",
            Self::AliphaticHydroxyl => "R-OH",
            Self::PhenolicHydroxyl => "Ar-OH",
            Self::PrimaryOrSecondaryAmine => "-NHR",
            Self::AmidineOrGuanidine => "Amidine",
            Self::PrimarySulfonamide => "-SO2NH2",
            Self::AzoleOrLactamNH => "Ring N-H",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::CarboxylicAcid => [248, 113, 113],          // Coral
            Self::AliphaticHydroxyl => [56, 189, 248],        // Sky Blue
            Self::PhenolicHydroxyl => [251, 191, 36],         // Amber
            Self::PrimaryOrSecondaryAmine => [52, 211, 153],  // Emerald
            Self::AmidineOrGuanidine => [244, 114, 182],      // Pink
            Self::PrimarySulfonamide => [192, 132, 252],      // Violet
            Self::AzoleOrLactamNH => [34, 211, 238],          // Cyan
        }
    }
}

/// Structural promoiety family (`Ester`, `Phosphate`, `Carbamate`, `Oxazolidinone`, or `PEPT1 Amino Acid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PromoietyFamily {
    Ester,
    Phosphate,
    Carbamate,
    Oxazolidinone,
    Pept1AminoAcid,
}

impl PromoietyFamily {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ester => "Carboxylester / Acyloxyalkyl",
            Self::Phosphate => "Phosphate / Phosphonooxymethyl",
            Self::Carbamate => "Carbamate / Acyl-Sulfonamide",
            Self::Oxazolidinone => "Oxazolidinone / Cyclic Mask",
            Self::Pept1AminoAcid => "hPEPT1 L-Valyl Carrier",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Ester => "Ester",
            Self::Phosphate => "Phosphate",
            Self::Carbamate => "Carbamate",
            Self::Oxazolidinone => "Oxazolidinone",
            Self::Pept1AminoAcid => "PEPT1 Val",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Ester => [52, 211, 153],         // Emerald
            Self::Phosphate => [56, 189, 248],     // Sky Blue
            Self::Carbamate => [251, 191, 36],     // Amber
            Self::Oxazolidinone => [244, 114, 182], // Pink
            Self::Pept1AminoAcid => [192, 132, 252], // Violet
        }
    }
}

/// Primary human bioconversion enzyme & anatomical activation compartment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProdrugActivatingEnzyme {
    /// Hepatic Carboxylesterase 1 (`CES1`: prefers large acyl / small alcohol; minimal gut cleavage -> clean portal/liver release).
    HepaticCes1,
    /// Intestinal Carboxylesterase 2 (`CES2`: prefers small acyl / bulky alcohol or phenol; rapid enterocyte activation).
    IntestinalCes2,
    /// Duodenal/Jejunal Brush-Border Alkaline Phosphatase (`ALP`: cleaves `-OPO3^2-` at membrane surface).
    BrushBorderAlp,
    /// Intestinal `hPEPT1` Uptake + Intracellular Valacyclovirase (`BPHL`) / Aminopeptidase.
    Pept1AndBphl,
    /// Systemic Paraoxonase / Butyrylcholinesterase / Hepatic Amidohydrolase (`PON1` / `BChE` / `FAAH`).
    ParaoxonaseAndBche,
}

impl ProdrugActivatingEnzyme {
    pub fn label(self) -> &'static str {
        match self {
            Self::HepaticCes1 => "Hepatic CES1 (Post-Absorptive Liver)",
            Self::IntestinalCes2 => "Enterocyte CES2 (Intestinal Mucosa)",
            Self::BrushBorderAlp => "Brush-Border ALP (Apical Membrane)",
            Self::Pept1AndBphl => "hPEPT1 Carrier + BPHL Hydrolase",
            Self::ParaoxonaseAndBche => "PON1 / BChE / Hepatic Hydrolase",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HepaticCes1 => "Liver CES1",
            Self::IntestinalCes2 => "Gut CES2",
            Self::BrushBorderAlp => "Apical ALP",
            Self::Pept1AndBphl => "PEPT1/BPHL",
            Self::ParaoxonaseAndBche => "PON1/BChE",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HepaticCes1 => [52, 211, 153],
            Self::IntestinalCes2 => [251, 191, 36],
            Self::BrushBorderAlp => [56, 189, 248],
            Self::Pept1AndBphl => [192, 132, 252],
            Self::ParaoxonaseAndBche => [244, 114, 182],
        }
    }
}

/// Safety classification of the stoichiometric promoiety cleavage byproduct(s).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByproductSafetyTier {
    EndogenousOrGras,
    BenignLowDose,
    FormaldehydeOrPivalateCaution,
}

impl ByproductSafetyTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::EndogenousOrGras => "Endogenous / GRAS Byproduct",
            Self::BenignLowDose => "Benign Metabolite (Standard Dose)",
            Self::FormaldehydeOrPivalateCaution => "Dose-Cap Caution (HCHO / Carnitine)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::EndogenousOrGras => [52, 211, 153],
            Self::BenignLowDose => [56, 189, 248],
            Self::FormaldehydeOrPivalateCaution => [251, 191, 36],
        }
    }
}

/// Parent `BCS` solubility, passive permeability, ionization, and oral bioavailability diagnosis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParentLiabilityDiagnosis {
    pub bcs_class: BcsClass,
    pub mw: f64,
    pub clogp: f64,
    pub logd_74: f64,
    pub net_charge_74: f64,
    pub tpsa: f64,
    /// Intrinsic + `pH 6.8` Henderson-Hasselbalch aqueous solubility (`ug/mL`).
    pub sol_ph68_ug_ml: f64,
    /// Dose number `Do = (100 mg / 250 mL) / Sol_mg_mL` (`Do <= 1.0` = complete luminal dissolution).
    pub dose_number_100mg: f64,
    pub boiled_egg: BoiledEggRegion,
    /// Estimated baseline passive/dissolution-limited oral fraction absorbed `Fa` (`0..100%`).
    pub parent_fa_pct: f64,
    /// Estimated baseline systemic oral bioavailability `F_oral` (`0..100%`).
    pub parent_f_oral_pct: f64,
    pub primary_bottleneck: String,
    #[serde(skip_deserializing)]
    pub recommended_strategy: &'static str,
}

/// Single maskable functional-group handle perceived on the parent drug.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskableHandle {
    pub handle_idx: usize,
    /// Atom index (`O` or `N`) bearing the replaceable hydrogen.
    pub anchor_atom: usize,
    /// Adjacent carbonyl/sulfonyl/imidoyl heavy atom index if part of `-COOH`, `-SO2NH2`, or `-C(=NH)NH2`.
    pub group_center_atom: Option<usize>,
    pub atom_label: String,
    pub kind: ProdrugHandleKind,
    pub steric_accessibility: f64,
    pub rationale: String,
}

/// Enzymatic activation kinetics, chemical stability, and oral PK time-course for a prodrug candidate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnzymaticActivationProfile {
    pub primary_enzyme: ProdrugActivatingEnzyme,
    /// Chemical stability half-life in simulated gastric/intestinal fluid (`pH 1.2 / 6.8`, `hours`).
    pub gi_chemical_t_half_hr: f64,
    /// Fraction surviving luminal chemical degradation prior to absorption/ALP cleavage (`%`).
    pub luminal_stability_pct: f64,
    /// Enzymatic activation half-life in the primary activating compartment (`minutes`).
    pub activation_t_half_min: f64,
    /// Hepatic `CES1` vs Intestinal `CES2` selectivity ratio (`> 1.0` favors post-absorptive liver activation).
    pub ces1_over_ces2_ratio: f64,
    /// Aqueous solubility at physiological `pH 6.8-7.4` (`ug/mL`) and fold-change vs parent.
    pub prodrug_sol_ph68_ug_ml: f64,
    pub solubility_fold_change: f64,
    /// Effective trans-epithelial permeability enhancement factor (`> 1.0` = improved flux).
    pub permeability_fold_change: f64,
    /// Predicted oral fraction absorbed `Fa` (`%`) and systemic active-drug bioavailability `F_oral` (`%`).
    pub predicted_fa_pct: f64,
    pub predicted_active_f_oral_pct: f64,
    pub f_oral_fold_gain: f64,
    #[serde(skip_deserializing)]
    pub released_byproduct: &'static str,
    pub byproduct_safety: ByproductSafetyTier,
    /// Time grid (`hours`, `0.0..=8.0`) and 3 concentration curves (`nM` for a `50 mg`-equivalent oral dose):
    /// `intact_prodrug_nm`, `released_active_nm`, and `direct_parent_oral_nm`.
    #[serde(with = "crate::serde_f32_array")]
    pub time_hr: [f32; PRODRUG_TIME_GRID_LEN],
    #[serde(with = "crate::serde_f32_array")]
    pub intact_prodrug_nm: [f32; PRODRUG_TIME_GRID_LEN],
    #[serde(with = "crate::serde_f32_array")]
    pub released_active_nm: [f32; PRODRUG_TIME_GRID_LEN],
    #[serde(with = "crate::serde_f32_array")]
    pub direct_parent_oral_nm: [f32; PRODRUG_TIME_GRID_LEN],
    pub active_cmax_nm: f64,
    pub active_tmax_hr: f64,
}

/// Single assembled bioreversible prodrug candidate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProdrugCandidate {
    pub rank: usize,
    pub candidate_id: String,
    #[serde(skip_deserializing)]
    pub promoiety_code: &'static str,
    #[serde(skip_deserializing)]
    pub promoiety_name: &'static str,
    pub family: PromoietyFamily,
    #[serde(skip_deserializing)]
    pub clinical_precedent: &'static str,
    pub handle_label: String,
    pub handle_kind: ProdrugHandleKind,
    pub activation: EnzymaticActivationProfile,
    pub physchem: PhyschemProfile,
    pub logd_74: f64,
    pub net_charge_74: f64,
    pub boiled_egg: BoiledEggRegion,
    pub sa_score: f64,
    /// Composite Prodrug Translational Score `[0..100]` balancing predicted active `F_oral %` gain,
    /// rapid quantitative bioconversion, luminal stability, byproduct safety, and `SAScore`.
    pub prodrug_score: f64,
    pub canonical_smiles: String,
    pub parent_atoms: Vec<usize>,
    pub promoiety_atoms: Vec<usize>,
    pub mol: Molecule,
}

/// Complete `BCS` liability diagnosis, maskable-handle map, and ranked promoiety report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProdrugDesignReport {
    pub parent_name: String,
    pub parent_id: String,
    pub parent_pact: f64,
    pub diagnosis: ParentLiabilityDiagnosis,
    pub handles: Vec<MaskableHandle>,
    pub active_handle_idx: usize,
    pub candidates: Vec<ProdrugCandidate>,
}

/// Dataset-level `BCS` & Best-Prodrug Bioavailability Rescue summary row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProdrugDatasetRow {
    pub mol_idx: usize,
    pub id: String,
    pub name: String,
    pub bcs_class: BcsClass,
    pub parent_sol_ug_ml: f64,
    pub parent_f_oral_pct: f64,
    pub handle_count: usize,
    pub best_promoiety: String,
    pub best_family: Option<PromoietyFamily>,
    pub best_enzyme: Option<ProdrugActivatingEnzyme>,
    pub rescued_f_oral_pct: f64,
    pub f_oral_fold_gain: f64,
}

struct PromoietyTemplateSpec {
    code: &'static str,
    name: &'static str,
    family: PromoietyFamily,
    /// OpenSMILES with single-bond `I` dummy replacing `1 H` on the handle's `O` or `N` atom.
    dummy_smiles: &'static str,
    applicable_handles: &'static [ProdrugHandleKind],
    clinical_precedent: &'static str,
    primary_enzyme: ProdrugActivatingEnzyme,
    gi_chem_t_half_hr: f64,
    base_act_t_half_min: f64,
    ces1_over_ces2: f64,
    pept1_active_boost: f64,
    alp_surface_supersat_boost: f64,
    extra_ph68_ionization_sol_factor: f64,
    released_byproduct: &'static str,
    byproduct_safety: ByproductSafetyTier,
}

use ProdrugHandleKind as HK;

const PROMOIETY_SPECS: &[PromoietyTemplateSpec] = &[
    // --- A. Carboxylic Acid (-COOH) Promoieties ---
    PromoietyTemplateSpec {
        code: "PD-COO-ET",
        name: "Ethyl Ester (-COOEt)",
        family: PromoietyFamily::Ester,
        dummy_smiles: "ICC",
        applicable_handles: &[HK::CarboxylicAcid],
        clinical_precedent: "Oseltamivir / Enalapril (Hepatic CES1: Large Acyl + Small EtOH)",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 36.0,
        base_act_t_half_min: 9.5,
        ces1_over_ces2: 14.0,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "Ethanol (1 eq, GRAS)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-COO-IPR",
        name: "Isopropyl Ester (-COOiPr)",
        family: PromoietyFamily::Ester,
        dummy_smiles: "IC(C)C",
        applicable_handles: &[HK::CarboxylicAcid],
        clinical_precedent: "Tenofovir Alafenamide / Fenofibrate (Gut-Stable CES1 Ester)",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 48.0,
        base_act_t_half_min: 14.0,
        ces1_over_ces2: 18.5,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "Isopropanol -> Acetone (Benign)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-COO-MOF",
        name: "Mofetil 2-Morpholinoethyl Ester",
        family: PromoietyFamily::Ester,
        dummy_smiles: "ICCN1CCOCC1",
        applicable_handles: &[HK::CarboxylicAcid],
        clinical_precedent: "Mycophenolate Mofetil (pKa ~5.6 Dual Sol + Perm Ester)",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 28.0,
        base_act_t_half_min: 8.0,
        ces1_over_ces2: 5.2,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 8.5,
        released_byproduct: "2-(4-Morpholinyl)ethanol (Renally Cleared)",
        byproduct_safety: ByproductSafetyTier::BenignLowDose,
    },
    PromoietyTemplateSpec {
        code: "PD-COO-MDX",
        name: "Medoxomil Cyclic Dioxolenone Ester",
        family: PromoietyFamily::Oxazolidinone,
        dummy_smiles: "ICC1=C(C)OC(=O)O1",
        applicable_handles: &[HK::CarboxylicAcid],
        clinical_precedent: "Olmesartan / Azilsartan Medoxomil (PON1/Esterase Cyclic Carbonate)",
        primary_enzyme: ProdrugActivatingEnzyme::ParaoxonaseAndBche,
        gi_chem_t_half_hr: 26.0,
        base_act_t_half_min: 4.8,
        ces1_over_ces2: 1.4,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.4,
        released_byproduct: "Diacetyl / Acetoin + CO2 (Non-Formaldehydogenic)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-COO-PIV",
        name: "Pivoxil (Pivaloyloxymethyl) Ester",
        family: PromoietyFamily::Ester,
        dummy_smiles: "ICOC(=O)C(C)(C)C",
        applicable_handles: &[HK::CarboxylicAcid, HK::AzoleOrLactamNH],
        clinical_precedent: "Cefditoren Pivoxil / Adefovir Dipivoxil (Self-Immolative Acylal)",
        primary_enzyme: ProdrugActivatingEnzyme::IntestinalCes2,
        gi_chem_t_half_hr: 22.0,
        base_act_t_half_min: 5.5,
        ces1_over_ces2: 0.42,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "Pivalic Acid (Carnitine Conjugation) + HCHO (1 eq)",
        byproduct_safety: ByproductSafetyTier::FormaldehydeOrPivalateCaution,
    },
    PromoietyTemplateSpec {
        code: "PD-COO-OXZ",
        name: "Oxazolidin-2-one-5-ylmethyl Ester",
        family: PromoietyFamily::Oxazolidinone,
        dummy_smiles: "ICC1CNC(=O)O1",
        applicable_handles: &[HK::CarboxylicAcid],
        clinical_precedent: "Crystalline Polarity-Balanced 2-Oxazolidinone Ester Prodrug",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 32.0,
        base_act_t_half_min: 11.5,
        ces1_over_ces2: 6.4,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 3.8,
        released_byproduct: "5-(Hydroxymethyl)-1,3-oxazolidin-2-one",
        byproduct_safety: ByproductSafetyTier::BenignLowDose,
    },
    PromoietyTemplateSpec {
        code: "PD-COO-CBM",
        name: "Carbamoyloxymethyl Ester-Carbamate",
        family: PromoietyFamily::Carbamate,
        dummy_smiles: "ICOC(=O)NCC",
        applicable_handles: &[HK::CarboxylicAcid],
        clinical_precedent: "Acyloxymethyl N-Alkyl Carbamate Double Prodrug",
        primary_enzyme: ProdrugActivatingEnzyme::ParaoxonaseAndBche,
        gi_chem_t_half_hr: 30.0,
        base_act_t_half_min: 16.0,
        ces1_over_ces2: 2.1,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.6,
        released_byproduct: "Ethylamine + CO2 + HCHO (1 eq)",
        byproduct_safety: ByproductSafetyTier::FormaldehydeOrPivalateCaution,
    },

    // --- B. Universal Phosphate & Phosphonooxymethyl Promoieties (ALP Solubility Rescue) ---
    PromoietyTemplateSpec {
        code: "PD-PO4-DIR",
        name: "Phosphate Monoester (-OPO3H2)",
        family: PromoietyFamily::Phosphate,
        dummy_smiles: "IP(=O)(O)O",
        applicable_handles: &[HK::AliphaticHydroxyl, HK::PhenolicHydroxyl],
        clinical_precedent: "Fosamprenavir / Tedizolid Phosphate / Fludarabine (Apical ALP)",
        primary_enzyme: ProdrugActivatingEnzyme::BrushBorderAlp,
        gi_chem_t_half_hr: 96.0,
        base_act_t_half_min: 3.2,
        ces1_over_ces2: 1.0,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 2.35,
        extra_ph68_ionization_sol_factor: 650.0,
        released_byproduct: "Inorganic Orthophosphate Pi (Endogenous)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-PO4-POM",
        name: "Phosphonooxymethyl (-OCH2OPO3H2 / -NCH2OPO3H2)",
        family: PromoietyFamily::Phosphate,
        dummy_smiles: "ICOP(=O)(O)O",
        applicable_handles: &[
            HK::AliphaticHydroxyl,
            HK::PhenolicHydroxyl,
            HK::PrimaryOrSecondaryAmine,
            HK::PrimarySulfonamide,
            HK::AzoleOrLactamNH,
            HK::CarboxylicAcid,
        ],
        clinical_precedent: "Fosphenytoin / Fospropofol / Fosfluconazole (Unhindered ALP Spacer)",
        primary_enzyme: ProdrugActivatingEnzyme::BrushBorderAlp,
        gi_chem_t_half_hr: 64.0,
        base_act_t_half_min: 2.6,
        ces1_over_ces2: 1.0,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 2.45,
        extra_ph68_ionization_sol_factor: 820.0,
        released_byproduct: "Inorganic Pi + HCHO (1 eq; Safe <=150 mg Dose)",
        byproduct_safety: ByproductSafetyTier::FormaldehydeOrPivalateCaution,
    },

    // --- C. Hydroxyl & Phenol Promoieties (PEPT1, Ester, Carbamate, Oxazolidinone) ---
    PromoietyTemplateSpec {
        code: "PD-VAL-PEP",
        name: "L-Valyl Carrier Ester / Amide (-CO-CH(NH2)iPr)",
        family: PromoietyFamily::Pept1AminoAcid,
        dummy_smiles: "IC(=O)C(N)C(C)C",
        applicable_handles: &[
            HK::AliphaticHydroxyl,
            HK::PhenolicHydroxyl,
            HK::PrimaryOrSecondaryAmine,
        ],
        clinical_precedent: "Valaciclovir / Valganciclovir / Lisdexamfetamine (hPEPT1 + BPHL)",
        primary_enzyme: ProdrugActivatingEnzyme::Pept1AndBphl,
        gi_chem_t_half_hr: 20.0,
        base_act_t_half_min: 8.5,
        ces1_over_ces2: 1.8,
        pept1_active_boost: 3.40,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 28.0,
        released_byproduct: "Essential Amino Acid L-Valine (GRAS)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-ACYL-PRP",
        name: "Propionyl Ester / Parecoxib Acyl-Sulfonamide",
        family: PromoietyFamily::Ester,
        dummy_smiles: "IC(=O)CC",
        applicable_handles: &[
            HK::AliphaticHydroxyl,
            HK::PhenolicHydroxyl,
            HK::PrimarySulfonamide,
        ],
        clinical_precedent: "Parecoxib (-SO2NHCOEt, pKa ~5.2 Sol Rescue) & Alcohol Propionate",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 30.0,
        base_act_t_half_min: 11.0,
        ces1_over_ces2: 2.8,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 18.0,
        released_byproduct: "Propionic Acid (Endogenous TCA Metabolite)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-OH-PIV",
        name: "Pivalate Sterically Shielded Ester (-OCOtBu)",
        family: PromoietyFamily::Ester,
        dummy_smiles: "IC(=O)C(C)(C)C",
        applicable_handles: &[HK::AliphaticHydroxyl, HK::PhenolicHydroxyl],
        clinical_precedent: "Dipivefrin (Shields Ar-OH from First-Pass UGT/SULT & Gut CES2)",
        primary_enzyme: ProdrugActivatingEnzyme::ParaoxonaseAndBche,
        gi_chem_t_half_hr: 54.0,
        base_act_t_half_min: 22.0,
        ces1_over_ces2: 4.5,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "Pivalic Acid (Monitor Carnitine on Chronic High Dose)",
        byproduct_safety: ByproductSafetyTier::FormaldehydeOrPivalateCaution,
    },
    PromoietyTemplateSpec {
        code: "PD-OH-SUCC",
        name: "Hemisuccinate Ionizable Ester (-OCO(CH2)2COOH)",
        family: PromoietyFamily::Ester,
        dummy_smiles: "IC(=O)CCC(=O)O",
        applicable_handles: &[HK::AliphaticHydroxyl, HK::PhenolicHydroxyl],
        clinical_precedent: "Methylprednisolone / Hydrocortisone Succinate (pH 7.4 Soluble)",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 24.0,
        base_act_t_half_min: 15.0,
        ces1_over_ces2: 7.5,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.15,
        extra_ph68_ionization_sol_factor: 95.0,
        released_byproduct: "Succinic Acid (Krebs Cycle Intermediate, GRAS)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-OH-CBET",
        name: "N-Ethyl Carbamate (-OC(=O)NHEt)",
        family: PromoietyFamily::Carbamate,
        dummy_smiles: "IC(=O)NCC",
        applicable_handles: &[HK::AliphaticHydroxyl, HK::PhenolicHydroxyl],
        clinical_precedent: "Bambuterol / Irinotecan-Class First-Pass Shielding Carbamate",
        primary_enzyme: ProdrugActivatingEnzyme::ParaoxonaseAndBche,
        gi_chem_t_half_hr: 72.0,
        base_act_t_half_min: 38.0,
        ces1_over_ces2: 1.6,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.2,
        released_byproduct: "Ethylamine + CO2",
        byproduct_safety: ByproductSafetyTier::BenignLowDose,
    },
    PromoietyTemplateSpec {
        code: "PD-OH-OXZC",
        name: "(2-Oxo-1,3-oxazolidin-3-yl)carbonyl Cyclic Mask",
        family: PromoietyFamily::Oxazolidinone,
        dummy_smiles: "IC(=O)N1CCOC1=O",
        applicable_handles: &[HK::AliphaticHydroxyl, HK::PhenolicHydroxyl],
        clinical_precedent: "Bioreversible 2-Oxazolidinone Imide-Carbamate Leaving Group",
        primary_enzyme: ProdrugActivatingEnzyme::IntestinalCes2,
        gi_chem_t_half_hr: 28.0,
        base_act_t_half_min: 12.5,
        ces1_over_ces2: 0.85,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 2.6,
        released_byproduct: "1,3-Oxazolidin-2-one + CO2",
        byproduct_safety: ByproductSafetyTier::BenignLowDose,
    },

    // --- D. Amine, Amidine, Sulfonamide & Azole N-H Promoieties (Carbamate & Oxazolidinone) ---
    PromoietyTemplateSpec {
        code: "PD-NH-CBET",
        name: "N-Ethoxycarbonyl Carbamate (-NHC(=O)OEt)",
        family: PromoietyFamily::Carbamate,
        dummy_smiles: "IC(=O)OCC",
        applicable_handles: &[
            HK::PrimaryOrSecondaryAmine,
            HK::AmidineOrGuanidine,
            HK::AzoleOrLactamNH,
        ],
        clinical_precedent: "Loratadine-Style / Amidine Charge-Neutralizing Ethyl Carbamate",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 68.0,
        base_act_t_half_min: 24.0,
        ces1_over_ces2: 8.2,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "Ethanol + CO2 (GRAS)",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-NH-CBHX",
        name: "N-Hexyloxycarbonyl Carbamate (-NHC(=O)OnHex)",
        family: PromoietyFamily::Carbamate,
        dummy_smiles: "IC(=O)OCCCCCC",
        applicable_handles: &[HK::PrimaryOrSecondaryAmine, HK::AmidineOrGuanidine],
        clinical_precedent: "Dabigatran Etexilate / Capecitabine (Masks Cationic Amidine pKa 11.5 -> 4.0)",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 72.0,
        base_act_t_half_min: 13.5,
        ces1_over_ces2: 12.4,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "1-Hexanol (-> Hexanoic Acid, Beta-Oxidized) + CO2",
        byproduct_safety: ByproductSafetyTier::EndogenousOrGras,
    },
    PromoietyTemplateSpec {
        code: "PD-NH-ENAC",
        name: "Isobutyryloxymethyl Carbamate (Enacarbil)",
        family: PromoietyFamily::Carbamate,
        dummy_smiles: "IC(=O)OCOC(=O)C(C)C",
        applicable_handles: &[HK::PrimaryOrSecondaryAmine, HK::AmidineOrGuanidine],
        clinical_precedent: "Gabapentin Enacarbil / Arbaclofen Placarbil (Self-Immolative CES1)",
        primary_enzyme: ProdrugActivatingEnzyme::IntestinalCes2,
        gi_chem_t_half_hr: 30.0,
        base_act_t_half_min: 6.2,
        ces1_over_ces2: 0.75,
        pept1_active_boost: 1.25,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 1.0,
        released_byproduct: "Isobutyric Acid + CO2 + Acetaldehyde/HCHO (1 eq)",
        byproduct_safety: ByproductSafetyTier::FormaldehydeOrPivalateCaution,
    },
    PromoietyTemplateSpec {
        code: "PD-NH-OXZC",
        name: "(2-Oxo-1,3-oxazolidin-5-yl)methyl Carbamate",
        family: PromoietyFamily::Oxazolidinone,
        dummy_smiles: "IC(=O)OCC1CNC(=O)O1",
        applicable_handles: &[
            HK::PrimaryOrSecondaryAmine,
            HK::AmidineOrGuanidine,
            HK::PrimarySulfonamide,
            HK::AzoleOrLactamNH,
        ],
        clinical_precedent: "Bioreversible Cyclic 2-Oxazolidinone Polarity-Tuned Carbamate",
        primary_enzyme: ProdrugActivatingEnzyme::HepaticCes1,
        gi_chem_t_half_hr: 44.0,
        base_act_t_half_min: 17.5,
        ces1_over_ces2: 5.8,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.0,
        extra_ph68_ionization_sol_factor: 4.5,
        released_byproduct: "5-(Hydroxymethyl)-1,3-oxazolidin-2-one + CO2",
        byproduct_safety: ByproductSafetyTier::BenignLowDose,
    },
    PromoietyTemplateSpec {
        code: "PD-MOR-AC",
        name: "Morpholinoacetyl Ionizable Cap",
        family: PromoietyFamily::Ester,
        dummy_smiles: "IC(=O)CN1CCOCC1",
        applicable_handles: &[
            HK::PrimarySulfonamide,
            HK::AliphaticHydroxyl,
            HK::AzoleOrLactamNH,
        ],
        clinical_precedent: "Water-Soluble Morpholinoacetate Ester / Acyl-Sulfonamide (pKa ~6.2)",
        primary_enzyme: ProdrugActivatingEnzyme::IntestinalCes2,
        gi_chem_t_half_hr: 18.0,
        base_act_t_half_min: 7.4,
        ces1_over_ces2: 0.90,
        pept1_active_boost: 1.0,
        alp_surface_supersat_boost: 1.2,
        extra_ph68_ionization_sol_factor: 42.0,
        released_byproduct: "2-(Morpholin-4-yl)acetic Acid (Renally Cleared)",
        byproduct_safety: ByproductSafetyTier::BenignLowDose,
    },
];

/// Estimates `pH 6.8` intestinal luminal solubility (`ug/mL`) by coupling Delaney `ESOL`
/// intrinsic neutral solubility with Henderson-Hasselbalch ionization (`cLogP - logD_7.4`).
fn estimate_ph68_solubility_ug_ml(
    esol_ug_ml: f64,
    clogp: f64,
    logd_74: f64,
    extra_factor: f64,
) -> f64 {
    let ion_delta = (clogp - logd_74).clamp(-0.3, 3.6);
    let hh_factor = 10f64.powf(ion_delta * 0.78);
    (esol_ug_ml * hh_factor * extra_factor).clamp(0.05, 180_000.0)
}

/// Effective formal ionic burden at physiological pH 7.4 for passive membrane permeation:
/// zwitterions carry opposing (+1 / -1) charges whose net sum is ~0 despite high `zwitterion_fraction_74`.
fn effective_ionic_charge_74(pka: &crate::pka_protomer::PhysiologicalChargeSummary) -> f64 {
    let total_ionic = pka.cation_fraction_74 + pka.anion_fraction_74 + 1.85 * pka.zwitterion_fraction_74;
    if pka.charge_class == crate::pka_protomer::ChargeClass74::Zwitterionic
        || pka.zwitterion_fraction_74 > 0.15
    {
        total_ionic.max(pka.mean_net_charge_74.abs())
    } else {
        pka.mean_net_charge_74.abs()
    }
}

/// Estimates passive intestinal effective permeability score (`[0.03..0.98]`) from `TPSA`,
/// `logD_7.4`, ionized net charge, `MW`, and `HBD`.
fn estimate_passive_fa_fraction(
    mw: f64,
    logd_74: f64,
    abs_net_charge: f64,
    tpsa: f64,
    hbd: usize,
) -> f64 {
    let tpsa_pen = ((tpsa - 85.0).max(0.0) / 55.0).powf(1.25) * 0.42;
    let charge_pen = (abs_net_charge.max(0.0) * 0.34).min(0.62);
    let low_logd_pen = ((-0.2 - logd_74).max(0.0) * 0.22).min(0.48);
    let high_mw_pen = ((mw - 520.0).max(0.0) / 260.0) * 0.22;
    let hbd_pen = (hbd.saturating_sub(3) as f64) * 0.08;
    (0.94 - tpsa_pen - charge_pen - low_logd_pen - high_mw_pen - hbd_pen).clamp(0.04, 0.96)
}

/// Diagnoses the parent molecule's `BCS Class I–IV`, dissolution Dose Number (`Do` at `100 mg`),
/// passive membrane permeation, and baseline oral bioavailability `F_oral %`.
pub fn diagnose_parent_bcs_liability(mol: &Molecule) -> ParentLiabilityDiagnosis {
    let alerts = scan_structural_alerts(mol);
    let phys = evaluate_physchem(mol, alerts.len());
    let admet = evaluate_admet(mol, &phys);
    let pka = evaluate_physiological_pka_74(mol);

    let sol_ph68 = estimate_ph68_solubility_ug_ml(
        admet.esol_ug_ml,
        phys.clogp,
        pka.microstate_logd_74,
        1.0,
    );
    // Dose number Do for a 100 mg oral dose in 250 mL fluid (= 400 ug/mL):
    let dose_num = (400.0 / sol_ph68.max(0.05)).clamp(0.002, 5000.0);

    let perm_frac = estimate_passive_fa_fraction(
        phys.mw,
        pka.microstate_logd_74,
        effective_ionic_charge_74(&pka),
        phys.tpsa,
        phys.hbd,
    );
    let high_sol = dose_num <= 1.25; // Sol >= ~320 ug/mL (or >=100 ug/mL for potent <=25 mg drugs)
    let mod_sol = sol_ph68 >= 90.0;
    let high_perm = perm_frac >= 0.62 && admet.boiled_egg != BoiledEggRegion::OutsideLow;

    let bcs_class = match (high_sol || mod_sol, high_perm) {
        (true, true) => BcsClass::ClassI,
        (false, true) => BcsClass::ClassII,
        (true, false) => BcsClass::ClassIII,
        (false, false) => BcsClass::ClassIV,
    };

    // Dissolution-limited absorption factor when Do > 1:
    let diss_factor = if dose_num <= 1.0 {
        1.0
    } else {
        (1.0 / dose_num.powf(0.58)).clamp(0.08, 1.0)
    };
    let parent_fa_pct = (perm_frac * diss_factor * 100.0).clamp(3.0, 96.0);

    // First-pass hepatic + phenolic UGT extraction estimate:
    let has_free_phenol = mol.atoms.iter().enumerate().any(|(u, a)| {
        a.elem == Element::O
            && a.total_h() >= 1
            && mol.adj[u].len() == 1
            && mol.atoms[mol.adj[u][0].0].aromatic
    });
    let fg_fh = if has_free_phenol {
        0.48 // High enterocyte/hepatic UGT1A/SULT1A1 first-pass conjugation of unprotected phenols!
    } else if phys.clogp > 4.6 {
        0.64
    } else {
        0.82
    };
    let parent_f_oral_pct = (parent_fa_pct * fg_fh).clamp(2.0, 92.0);

    let (primary_bottleneck, recommended_strategy) = match bcs_class {
        BcsClass::ClassII => (
            format!(
                "Dissolution-limited 'brick-dust' solubility ({:.1} ug/mL, Do={:.1}x at 100 mg)",
                sol_ph68, dose_num
            ),
            "Attach brush-border ALP Phosphate (-OPO3H2 / -OCH2OPO3H2), Hemisuccinate, or Parecoxib-type Acyl-Sulfonamide",
        ),
        BcsClass::ClassIII => (
            format!(
                "Polar / ionized membrane barrier (TPSA {:.0} A^2, logD7.4 {:.2}, Q7.4 {:+.2})",
                phys.tpsa, pka.microstate_logd_74, pka.mean_net_charge_74
            ),
            "Mask charge/polarity via Hepatic CES1 Alkyl Ester, Lipophilic Carbamate, Oxazolidinone, or hPEPT1 L-Valyl Carrier",
        ),
        BcsClass::ClassIV => (
            format!(
                "Dual solubility ({:.1} ug/mL) & permeability (TPSA {:.0} A^2, Q7.4 {:+.2}) barrier",
                sol_ph68, phys.tpsa, pka.mean_net_charge_74
            ),
            "Use Amphiphilic Mofetil Ester, hPEPT1 L-Valinate Carrier, or ALP Phosphonooxymethyl Spacer",
        ),
        BcsClass::ClassI => {
            if has_free_phenol {
                (
                    "Unprotected phenolic Ar-OH subject to rapid Phase II UGT/SULT first-pass clearance".to_string(),
                    "Shield Ar-OH with Pivalate Ester, N-Ethyl Carbamate, or 2-Oxazolidinone Imide-Carbamate",
                )
            } else {
                (
                    format!(
                        "Favorable baseline solubility ({:.0} ug/mL) & passive HIA (Fa {:.0}%)",
                        sol_ph68, parent_fa_pct
                    ),
                    "Optional long-acting depot ester, CNS-targeted lipophilic ester, or GI-irritation mask",
                )
            }
        }
    };

    ParentLiabilityDiagnosis {
        bcs_class,
        mw: phys.mw,
        clogp: phys.clogp,
        logd_74: pka.microstate_logd_74,
        net_charge_74: pka.mean_net_charge_74,
        tpsa: phys.tpsa,
        sol_ph68_ug_ml: sol_ph68,
        dose_number_100mg: dose_num,
        boiled_egg: admet.boiled_egg,
        parent_fa_pct,
        parent_f_oral_pct,
        primary_bottleneck,
        recommended_strategy,
    }
}

/// Perceives all synthetically & enzymatically maskable functional-group handles (`-COOH`,
/// aliphatic `-OH`, phenolic `Ar-OH`, `-NH2`/`-NHR`, amidine `-C(=NH)NH2`, `-SO2NH2`, and
/// azole/lactam `N-H`) on `mol`.
pub fn perceive_maskable_handles(mol: &Molecule) -> Vec<MaskableHandle> {
    let n = mol.atoms.len();
    let mut handles: Vec<MaskableHandle> = Vec::new();
    let mut claimed_atoms = vec![false; n];

    // Pass 1: Carboxylic acids `-C(=O)OH`, Primary Sulfonamides `-S(=O)2NH2`, and Amidines `-C(=NH)NH2`
    for u in 0..n {
        let a = &mol.atoms[u];
        if a.elem == Element::C && !a.aromatic {
            // Check `-C(=O)OH`
            let mut has_carbonyl_o = false;
            let mut oh_idx = None;
            let mut imine_nh_idx = None;
            let mut amino_nh2_idx = None;
            for &(v, bi) in &mol.adj[u] {
                let b = &mol.bonds[bi];
                let av = &mol.atoms[v];
                if b.order == BondOrder::Double && av.elem == Element::O && mol.heavy_degree(v) == 1 {
                    has_carbonyl_o = true;
                } else if b.order == BondOrder::Single
                    && av.elem == Element::O
                    && av.total_h() >= 1
                    && mol.heavy_degree(v) == 1
                {
                    oh_idx = Some(v);
                } else if b.order == BondOrder::Double
                    && av.elem == Element::N
                    && av.total_h() >= 1
                    && mol.heavy_degree(v) == 1
                {
                    imine_nh_idx = Some(v);
                } else if b.order == BondOrder::Single
                    && av.elem == Element::N
                    && av.total_h() >= 1
                    && mol.heavy_degree(v) == 1
                {
                    amino_nh2_idx = Some(v);
                }
            }
            if has_carbonyl_o {
                if let Some(ox) = oh_idx {
                    claimed_atoms[ox] = true;
                    handles.push(MaskableHandle {
                        handle_idx: 0,
                        anchor_atom: ox,
                        group_center_atom: Some(u),
                        atom_label: format!("O{}(COOH@C{})", ox + 1, u + 1),
                        kind: ProdrugHandleKind::CarboxylicAcid,
                        steric_accessibility: 0.94,
                        rationale: "Ionized carboxylate (-COO-, pKa ~4.2) restricts passive GI permeation; ideal for CES1 ethyl/isopropyl/mofetil or medoxomil esters".to_string(),
                    });
                }
            } else if let (Some(im_n), Some(am_n)) = (imine_nh_idx, amino_nh2_idx) {
                claimed_atoms[im_n] = true;
                claimed_atoms[am_n] = true;
                handles.push(MaskableHandle {
                    handle_idx: 0,
                    anchor_atom: im_n,
                    group_center_atom: Some(u),
                    atom_label: format!("N{}(Amidine@C{})", im_n + 1, u + 1),
                    kind: ProdrugHandleKind::AmidineOrGuanidine,
                    steric_accessibility: 0.92,
                    rationale: "Strongly basic amidine (pKa ~11.4, permanently cationic at pH 7.4); mask as N-alkoxycarbonyl or oxazolidinone carbamate (Dabigatran strategy)".to_string(),
                });
            }
        } else if a.elem == Element::S {
            // Check `-S(=O)(=O)NH2` or `-S(=O)(=O)NHR`
            let oxo_cnt = mol.adj[u]
                .iter()
                .filter(|&&(v, bi)| {
                    mol.bonds[bi].order == BondOrder::Double && mol.atoms[v].elem == Element::O
                })
                .count();
            if oxo_cnt >= 2 {
                for &(v, bi) in &mol.adj[u] {
                    if mol.bonds[bi].order == BondOrder::Single
                        && mol.atoms[v].elem == Element::N
                        && mol.atoms[v].total_h() >= 1
                        && mol.heavy_degree(v) <= 2
                    {
                        claimed_atoms[v] = true;
                        handles.push(MaskableHandle {
                            handle_idx: 0,
                            anchor_atom: v,
                            group_center_atom: Some(u),
                            atom_label: format!("N{}(SO2NH@S{})", v + 1, u + 1),
                            kind: ProdrugHandleKind::PrimarySulfonamide,
                            steric_accessibility: 0.88,
                            rationale: "High-lattice-energy sulfonamide HBD; N-acylation (Parecoxib -SO2NHCOEt, pKa ~5.2) or N-phosphonooxymethyl boosts solubility >100x".to_string(),
                        });
                    }
                }
            }
        }
    }

    // Pass 2: Remaining `-OH` (Phenolic vs Aliphatic) and `-NH2` / `-NHR` / Ring `[nH]`
    for u in 0..n {
        if claimed_atoms[u] {
            continue;
        }
        let a = &mol.atoms[u];
        if a.charge != 0 || a.total_h() == 0 {
            continue;
        }
        if a.elem == Element::O && mol.heavy_degree(u) == 1 {
            let nbr = mol.adj[u][0].0;
            if mol.atoms[nbr].elem == Element::P {
                continue;
            }
            let is_phenol = mol.atoms[nbr].aromatic;
            let nbr_deg = mol.heavy_degree(nbr);
            let steric = if is_phenol {
                0.86
            } else {
                (1.05 - (nbr_deg.saturating_sub(1) as f64) * 0.16).clamp(0.48, 0.98)
            };
            handles.push(MaskableHandle {
                handle_idx: 0,
                anchor_atom: u,
                group_center_atom: Some(nbr),
                atom_label: format!(
                    "O{}({})",
                    u + 1,
                    if is_phenol { "Ar-OH" } else { "R-OH" }
                ),
                kind: if is_phenol {
                    ProdrugHandleKind::PhenolicHydroxyl
                } else {
                    ProdrugHandleKind::AliphaticHydroxyl
                },
                steric_accessibility: steric,
                rationale: if is_phenol {
                    "Phenolic Ar-OH handle: ALP phosphate boosts solubility; pivalate/carbamate/oxazolidinone blocks first-pass UGT/SULT glucuronidation".to_string()
                } else {
                    "Aliphatic R-OH handle: prime site for brush-border ALP phosphate, hPEPT1 L-valinate ester, hemisuccinate, or propionate ester".to_string()
                },
            });
        } else if a.elem == Element::N {
            // Skip simple carboxamide `-C(=O)NH-` unless cyclic lactam `[nH]`
            let adj_carbonyl = mol.adj[u].iter().any(|&(nb, _)| {
                mol.atoms[nb].elem == Element::C
                    && mol.adj[nb].iter().any(|&(o, bi)| {
                        mol.bonds[bi].order == BondOrder::Double && mol.atoms[o].elem == Element::O
                    })
            });
            if a.in_ring {
                handles.push(MaskableHandle {
                    handle_idx: 0,
                    anchor_atom: u,
                    group_center_atom: None,
                    atom_label: format!("N{}(Ring-NH)", u + 1),
                    kind: ProdrugHandleKind::AzoleOrLactamNH,
                    steric_accessibility: 0.80,
                    rationale: "Heteroaromatic azole / lactam N-H: disrupts crystal-packing H-bond network via N-phosphonooxymethyl (Fosphenytoin) or N-alkoxycarbonyl".to_string(),
                });
            } else if !adj_carbonyl {
                handles.push(MaskableHandle {
                    handle_idx: 0,
                    anchor_atom: u,
                    group_center_atom: None,
                    atom_label: format!("N{}(Amine)", u + 1),
                    kind: ProdrugHandleKind::PrimaryOrSecondaryAmine,
                    steric_accessibility: 0.88,
                    rationale: "Ionizable/polar amine (-NH2/-NHR): bioreversible carbamate, Enacarbil acyloxyalkyl carbamate, oxazolidinone carbamate, or L-valyl amide".to_string(),
                });
            }
        }
    }

    // Prioritize `-COOH`, `Amidine`, `Aliphatic/Phenolic -OH`, `-SO2NH2`, `Amine`, then `Ring N-H`
    let kind_prio = |k: ProdrugHandleKind| -> u8 {
        match k {
            HK::CarboxylicAcid => 0,
            HK::AmidineOrGuanidine => 1,
            HK::AliphaticHydroxyl => 2,
            HK::PhenolicHydroxyl => 3,
            HK::PrimarySulfonamide => 4,
            HK::PrimaryOrSecondaryAmine => 5,
            HK::AzoleOrLactamNH => 6,
        }
    };
    handles.sort_by(|a, b| {
        kind_prio(a.kind)
            .cmp(&kind_prio(b.kind))
            .then_with(|| a.anchor_atom.cmp(&b.anchor_atom))
    });
    for (i, h) in handles.iter_mut().enumerate() {
        h.handle_idx = i;
    }
    handles
}

fn parse_one_dummy_promoiety(dummy_smiles: &str) -> Option<(Molecule, usize)> {
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

fn assemble_prodrug_graph(
    parent: &Molecule,
    handle: &MaskableHandle,
    spec: &PromoietyTemplateSpec,
) -> Option<(Molecule, Vec<usize>, Vec<usize>)> {
    if handle.anchor_atom >= parent.atoms.len() || parent.atoms[handle.anchor_atom].total_h() == 0 {
        return None;
    }

    let mut out = Molecule::new("");
    let mut parent_atoms = Vec::with_capacity(parent.atoms.len());
    for (oi, a) in parent.atoms.iter().enumerate() {
        let mut na = a.clone();
        na.idx = oi;
        if oi == handle.anchor_atom {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        parent_atoms.push(oi);
    }
    for (bi, b) in parent.bonds.iter().enumerate() {
        let mut nb = b.clone();
        nb.idx = bi;
        out.bonds.push(nb);
    }

    let (pro_mol, pro_attach) = parse_one_dummy_promoiety(spec.dummy_smiles)?;
    let offset = out.atoms.len();
    let mut pro_atoms = Vec::with_capacity(pro_mol.atoms.len());
    for (pi, pa) in pro_mol.atoms.iter().enumerate() {
        let mut na = pa.clone();
        na.idx = offset + pi;
        out.atoms.push(na);
        pro_atoms.push(offset + pi);
    }
    for pb in &pro_mol.bonds {
        let bi = out.bonds.len();
        let mut nb = pb.clone();
        nb.idx = bi;
        nb.a = offset + pb.a;
        nb.b = offset + pb.b;
        out.bonds.push(nb);
    }

    let b_join = out.bonds.len();
    out.bonds.push(Bond {
        idx: b_join,
        a: handle.anchor_atom,
        b: offset + pro_attach,
        order: BondOrder::Single,
        aromatic: false,
        in_ring: false,
        kekule_order: 1,
    });

    out.sanitize();
    let can_smi = molecule_to_smiles(&out);
    let check = parse_smiles(&can_smi).ok()?;
    let exp_hac = out.atoms.iter().filter(|a| a.elem != Element::H).count();
    let chk_hac = check.atoms.iter().filter(|a| a.elem != Element::H).count();
    if exp_hac != chk_hac {
        return None;
    }
    out.smiles = can_smi;
    compute_2d_coords(&mut out);
    Some((out, parent_atoms, pro_atoms))
}

/// Analytical 3-stage oral absorption + enzymatic bioconversion PK simulator (`0.0..=8.0 hours`)
/// for a standard `50 mg` oral equivalent dose (`V_d ~ 42 L` normalized distribution volume).
#[allow(clippy::too_many_arguments)]
fn simulate_oral_bioconversion_curves(
    parent_mw: f64,
    parent_f_oral_pct: f64,
    prodrug_fa_pct: f64,
    active_f_oral_pct: f64,
    enzyme: ProdrugActivatingEnzyme,
    act_t_half_min: f64,
) -> (
    [f32; PRODRUG_TIME_GRID_LEN],
    [f32; PRODRUG_TIME_GRID_LEN],
    [f32; PRODRUG_TIME_GRID_LEN],
    [f32; PRODRUG_TIME_GRID_LEN],
    f64,
    f64,
) {
    let mut t_grid = [0.0f32; PRODRUG_TIME_GRID_LEN];
    let mut c_pro = [0.0f32; PRODRUG_TIME_GRID_LEN];
    let mut c_act = [0.0f32; PRODRUG_TIME_GRID_LEN];
    let mut c_par_direct = [0.0f32; PRODRUG_TIME_GRID_LEN];

    // Total molar dose for 50 mg active-equivalent in V_d = 42 L -> C_0_max_nM = (0.050 / MW) / 42 * 1e9
    let dose_scale_nm = ((0.050 / parent_mw.max(140.0)) / 42.0) * 1.0e9;

    // Rate constants (hr^-1):
    let ka_pro = match enzyme {
        ProdrugActivatingEnzyme::Pept1AndBphl => 2.10,
        ProdrugActivatingEnzyme::BrushBorderAlp => 1.85,
        _ => 1.45,
    };
    let k_conv = (0.69315 / (act_t_half_min.max(1.5) / 60.0)).clamp(0.45, 18.0);
    let ke_act = 0.24; // Active parent systemic elimination half-life ~ 2.9 hr
    let ka_par_direct = 0.85;

    // Intact systemic prodrug fraction escaping gut/first-pass cleavage:
    let systemic_intact_leak = match enzyme {
        ProdrugActivatingEnzyme::BrushBorderAlp => 0.03, // Cleaved pre-absorptively at apical membrane!
        ProdrugActivatingEnzyme::IntestinalCes2 => 0.14,
        ProdrugActivatingEnzyme::Pept1AndBphl => 0.16,
        ProdrugActivatingEnzyme::HepaticCes1 => 0.34,
        ProdrugActivatingEnzyme::ParaoxonaseAndBche => 0.52,
    };

    let f_act = (active_f_oral_pct / 100.0).clamp(0.02, 0.98);
    let f_par = (parent_f_oral_pct / 100.0).clamp(0.02, 0.96);
    let f_pro_sys = (prodrug_fa_pct / 100.0) * systemic_intact_leak;

    // Effective apparent appearance rate of active drug (combines ka_pro & k_conv):
    let ka_eff_act = ((ka_pro * k_conv) / (ka_pro + k_conv)).max(0.32);

    let mut cmax_act = 0.0f64;
    let mut tmax_act = 0.0f64;

    for i in 0..PRODRUG_TIME_GRID_LEN {
        let t = (i as f64) * (8.0 / ((PRODRUG_TIME_GRID_LEN - 1) as f64));
        t_grid[i] = t as f32;

        // 1. Intact prodrug transient plasma concentration (cleared at rate k_conv):
        let cp = if (ka_pro - k_conv).abs() > 0.05 {
            dose_scale_nm
                * f_pro_sys
                * (ka_pro / (ka_pro - k_conv))
                * ((-k_conv * t).exp() - (-ka_pro * t).exp())
        } else {
            dose_scale_nm * f_pro_sys * ka_pro * t * (-ka_pro * t).exp()
        }
        .max(0.0);

        // 2. Released active parent concentration:
        let ca = if (ka_eff_act - ke_act).abs() > 0.03 {
            dose_scale_nm
                * f_act
                * (ka_eff_act / (ka_eff_act - ke_act))
                * ((-ke_act * t).exp() - (-ka_eff_act * t).exp())
        } else {
            dose_scale_nm * f_act * ka_eff_act * t * (-ke_act * t).exp()
        }
        .max(0.0);

        // 3. Direct oral administration of unmodified parent:
        let cd = if (ka_par_direct - ke_act).abs() > 0.03 {
            dose_scale_nm
                * f_par
                * (ka_par_direct / (ka_par_direct - ke_act))
                * ((-ke_act * t).exp() - (-ka_par_direct * t).exp())
        } else {
            dose_scale_nm * f_par * ka_par_direct * t * (-ke_act * t).exp()
        }
        .max(0.0);

        if ca > cmax_act {
            cmax_act = ca;
            tmax_act = t;
        }

        c_pro[i] = cp as f32;
        c_act[i] = ca as f32;
        c_par_direct[i] = cd as f32;
    }

    (t_grid, c_pro, c_act, c_par_direct, cmax_act, tmax_act)
}

/// Enumerates and enzymatically profiles bioreversible prodrug promoieties (`Ester`, `Phosphate`,
/// `Carbamate`, `Oxazolidinone`, and `hPEPT1` L-Valyl) on `handle_choice` of `parent`.
pub fn design_prodrugs(parent: &Molecule, handle_choice: usize) -> ProdrugDesignReport {
    let mut clean_parent = parent.clone();
    if clean_parent.atoms.len() > 1
        && clean_parent
            .atoms
            .iter()
            .all(|a| a.pos2d[0].abs() < 1e-4 && a.pos2d[1].abs() < 1e-4)
    {
        compute_2d_coords(&mut clean_parent);
    }

    let diagnosis = diagnose_parent_bcs_liability(&clean_parent);
    let handles = perceive_maskable_handles(&clean_parent);
    if handles.is_empty() {
        return ProdrugDesignReport {
            parent_name: clean_parent.name.clone(),
            parent_id: clean_parent.id.clone(),
            parent_pact: clean_parent.pact,
            diagnosis,
            handles,
            active_handle_idx: 0,
            candidates: Vec::new(),
        };
    }

    let active_handle_idx = handle_choice.min(handles.len() - 1);
    let handle = &handles[active_handle_idx];

    let parent_alerts = scan_structural_alerts(&clean_parent);
    let parent_phys = evaluate_physchem(&clean_parent, parent_alerts.len());
    let parent_pka = evaluate_physiological_pka_74(&clean_parent);
    let parent_passive_perm = estimate_passive_fa_fraction(
        diagnosis.mw,
        diagnosis.logd_74,
        effective_ionic_charge_74(&parent_pka),
        diagnosis.tpsa,
        parent_phys.hbd,
    );

    let mut candidates = Vec::new();

    for spec in PROMOIETY_SPECS {
        if !spec.applicable_handles.contains(&handle.kind) {
            continue;
        }
        let Some((mut pd_mol, parent_atoms, promoiety_atoms)) =
            assemble_prodrug_graph(&clean_parent, handle, spec)
        else {
            continue;
        };

        let alerts = scan_structural_alerts(&pd_mol);
        let phys = evaluate_physchem(&pd_mol, alerts.len());
        let admet = evaluate_admet(&pd_mol, &phys);
        let pka = evaluate_physiological_pka_74(&pd_mol);
        let sa = evaluate_synthetic_accessibility(&pd_mol);

        // 1. Prodrug pH 6.8 luminal solubility:
        let pd_sol = estimate_ph68_solubility_ug_ml(
            admet.esol_ug_ml,
            phys.clogp,
            pka.microstate_logd_74,
            spec.extra_ph68_ionization_sol_factor,
        );
        let sol_fold = (pd_sol / diagnosis.sol_ph68_ug_ml.max(0.05)).clamp(0.01, 50_000.0);
        let pd_dose_num = (400.0 / pd_sol.max(0.05)).clamp(0.001, 5000.0);
        let pd_diss_factor = if pd_dose_num <= 1.0 {
            1.0
        } else {
            (1.0 / pd_dose_num.powf(0.58)).clamp(0.08, 1.0)
        };

        // 2. Prodrug effective intestinal flux (passive OR brush-border ALP supersaturation OR hPEPT1):
        let pd_passive = estimate_passive_fa_fraction(
            phys.mw,
            pka.microstate_logd_74,
            effective_ionic_charge_74(&pka),
            phys.tpsa,
            phys.hbd,
        );
        let eff_perm = if spec.primary_enzyme == ProdrugActivatingEnzyme::BrushBorderAlp {
            // Phosphate prodrugs do NOT cross the membrane intact; ALP cleaves them at the brush border
            // to deliver supersaturated parent right at the apical membrane!
            (parent_passive_perm * spec.alp_surface_supersat_boost.min(1.35)).clamp(0.08, 0.96)
        } else {
            (pd_passive * spec.pept1_active_boost).clamp(0.06, 0.97)
        };
        let perm_fold = (eff_perm / parent_passive_perm.max(0.04)).clamp(0.20, 12.0);

        // 3. Luminal stability & enzymatic activation half-life adjusted by handle steric accessibility:
        let luminal_stab_pct =
            (100.0 * (-0.75 / spec.gi_chem_t_half_hr.max(2.0)).exp()).clamp(65.0, 99.6);
        let act_t_half_min =
            (spec.base_act_t_half_min / handle.steric_accessibility.max(0.45)).clamp(1.8, 75.0);

        // Penalty if CES2 cleaves prematurely inside the enterocyte when parent itself is BCS III impermeable!
        let enterocyte_trap_factor = if spec.primary_enzyme == ProdrugActivatingEnzyme::IntestinalCes2
            && diagnosis.bcs_class == BcsClass::ClassIII
        {
            0.76
        } else {
            1.0
        };

        // Conversion completeness factor (very slow carbamates t1/2 > 45 min lose some intact renal excretion):
        let conv_eff = (1.0 - (act_t_half_min - 25.0).max(0.0) * 0.0055).clamp(0.68, 0.99);

        let predicted_fa_pct =
            (eff_perm * pd_diss_factor * (luminal_stab_pct / 100.0) * 100.0).clamp(4.0, 97.5);
        // Prodrug shields phenolic Ar-OH during gut transit if Ar-OH was masked!
        let hepatic_survival = if handle.kind == ProdrugHandleKind::PhenolicHydroxyl
            && spec.primary_enzyme != ProdrugActivatingEnzyme::BrushBorderAlp
        {
            0.84
        } else {
            0.86
        };
        let predicted_active_f_oral_pct =
            (predicted_fa_pct * enterocyte_trap_factor * conv_eff * hepatic_survival)
                .clamp(4.0, 94.0);
        let f_oral_fold_gain =
            (predicted_active_f_oral_pct / diagnosis.parent_f_oral_pct.max(2.0)).clamp(0.25, 25.0);

        let (time_hr, intact_prodrug_nm, released_active_nm, direct_parent_oral_nm, cmax, tmax) =
            simulate_oral_bioconversion_curves(
                diagnosis.mw,
                diagnosis.parent_f_oral_pct,
                predicted_fa_pct,
                predicted_active_f_oral_pct,
                spec.primary_enzyme,
                act_t_half_min,
            );

        // 4. Rational BCS-Matched Translational Score [0..100]:
        let bcs_fit_bonus = match (diagnosis.bcs_class, spec.family) {
            (BcsClass::ClassII, PromoietyFamily::Phosphate) => 16.0,
            (BcsClass::ClassII, PromoietyFamily::Ester) if sol_fold >= 8.0 => 13.0,
            (
                BcsClass::ClassIII,
                PromoietyFamily::Ester
                | PromoietyFamily::Carbamate
                | PromoietyFamily::Oxazolidinone
                | PromoietyFamily::Pept1AminoAcid,
            ) if perm_fold >= 1.25 => 15.0,
            (BcsClass::ClassIV, PromoietyFamily::Pept1AminoAcid) => 16.0,
            (BcsClass::ClassIV, _) if sol_fold >= 2.5 && perm_fold >= 1.15 => 14.0,
            (BcsClass::ClassI, _) => 8.0,
            _ => 3.0,
        };
        let f_term = (predicted_active_f_oral_pct * 0.48
            + (f_oral_fold_gain - 1.0).clamp(-0.5, 4.0) * 6.5)
            .clamp(0.0, 58.0);
        let kin_term = if (2.0..=25.0).contains(&act_t_half_min) {
            14.0
        } else {
            8.5
        };
        let safety_term = match spec.byproduct_safety {
            ByproductSafetyTier::EndogenousOrGras => 10.0,
            ByproductSafetyTier::BenignLowDose => 7.5,
            ByproductSafetyTier::FormaldehydeOrPivalateCaution => 3.0,
        };
        let sa_pen = ((sa.sa_score - 3.8).max(0.0) * 3.2).min(12.0);
        let prodrug_score =
            (f_term + bcs_fit_bonus + kin_term + safety_term - sa_pen).clamp(4.0, 99.4);

        let cid = format!("PD-H{}-{}", active_handle_idx + 1, &spec.code[3..]);
        pd_mol.id = cid.clone();
        pd_mol.name = format!("{} — {}", clean_parent.name, spec.name);
        pd_mol.target = clean_parent.target.clone();
        pd_mol.metric = clean_parent.metric.clone();
        pd_mol.pact = clean_parent.pact; // Active parent potency liberated post-cleavage
        pd_mol.raw_nm = clean_parent.raw_nm;

        candidates.push(ProdrugCandidate {
            rank: 0,
            candidate_id: cid,
            promoiety_code: spec.code,
            promoiety_name: spec.name,
            family: spec.family,
            clinical_precedent: spec.clinical_precedent,
            handle_label: handle.atom_label.clone(),
            handle_kind: handle.kind,
            activation: EnzymaticActivationProfile {
                primary_enzyme: spec.primary_enzyme,
                gi_chemical_t_half_hr: spec.gi_chem_t_half_hr,
                luminal_stability_pct: luminal_stab_pct,
                activation_t_half_min: act_t_half_min,
                ces1_over_ces2_ratio: spec.ces1_over_ces2,
                prodrug_sol_ph68_ug_ml: pd_sol,
                solubility_fold_change: sol_fold,
                permeability_fold_change: perm_fold,
                predicted_fa_pct,
                predicted_active_f_oral_pct,
                f_oral_fold_gain,
                released_byproduct: spec.released_byproduct,
                byproduct_safety: spec.byproduct_safety,
                time_hr,
                intact_prodrug_nm,
                released_active_nm,
                direct_parent_oral_nm,
                active_cmax_nm: cmax,
                active_tmax_hr: tmax,
            },
            physchem: phys,
            logd_74: pka.microstate_logd_74,
            net_charge_74: pka.mean_net_charge_74,
            boiled_egg: admet.boiled_egg,
            sa_score: sa.sa_score,
            prodrug_score,
            canonical_smiles: pd_mol.smiles.clone(),
            parent_atoms,
            promoiety_atoms,
            mol: pd_mol,
        });
    }

    candidates.sort_by(|a, b| {
        b.prodrug_score
            .partial_cmp(&a.prodrug_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.rank = i + 1;
    }

    ProdrugDesignReport {
        parent_name: clean_parent.name.clone(),
        parent_id: clean_parent.id.clone(),
        parent_pact: clean_parent.pact,
        diagnosis,
        handles,
        active_handle_idx,
        candidates,
    }
}

/// Audits `BCS Class I–IV` liabilities and best promoiety oral-bioavailability rescue across an entire dataset.
pub fn summarize_dataset_prodrug_opportunities(mols: &[Molecule]) -> Vec<ProdrugDatasetRow> {
    mols.iter()
        .enumerate()
        .map(|(idx, m)| {
            let rep = design_prodrugs(m, 0);
            if let Some(top) = rep.candidates.first() {
                ProdrugDatasetRow {
                    mol_idx: idx,
                    id: m.id.clone(),
                    name: m.name.clone(),
                    bcs_class: rep.diagnosis.bcs_class,
                    parent_sol_ug_ml: rep.diagnosis.sol_ph68_ug_ml,
                    parent_f_oral_pct: rep.diagnosis.parent_f_oral_pct,
                    handle_count: rep.handles.len(),
                    best_promoiety: top.promoiety_name.to_string(),
                    best_family: Some(top.family),
                    best_enzyme: Some(top.activation.primary_enzyme),
                    rescued_f_oral_pct: top.activation.predicted_active_f_oral_pct,
                    f_oral_fold_gain: top.activation.f_oral_fold_gain,
                }
            } else {
                ProdrugDatasetRow {
                    mol_idx: idx,
                    id: m.id.clone(),
                    name: m.name.clone(),
                    bcs_class: rep.diagnosis.bcs_class,
                    parent_sol_ug_ml: rep.diagnosis.sol_ph68_ug_ml,
                    parent_f_oral_pct: rep.diagnosis.parent_f_oral_pct,
                    handle_count: 0,
                    best_promoiety: "No maskable O-H / N-H / -COOH handle".to_string(),
                    best_family: None,
                    best_enzyme: None,
                    rescued_f_oral_pct: rep.diagnosis.parent_f_oral_pct,
                    f_oral_fold_gain: 1.0,
                }
            }
        })
        .collect()
}

/// Exports enumerated prodrug candidates and enzymatic bioconversion metrics to CSV.
pub fn export_prodrugs_csv(cands: &[ProdrugCandidate]) -> String {
    let mut out = String::from(
        "rank,id,handle,promoiety_code,promoiety_name,family,enzyme,t_half_act_min,ces1_ces2_ratio,luminal_stab_pct,sol_ph68_ug_ml,sol_fold,perm_fold,fa_pct,active_f_oral_pct,f_oral_fold_gain,cmax_nm,tmax_hr,mw,logd_74,tpsa,byproduct,safety,sa_score,prodrug_score,smiles\n",
    );
    for c in cands {
        let ac = &c.activation;
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{:.1},{:.2},{:.1},{:.1},{:.2},{:.2},{:.1},{:.1},{:.2},{:.1},{:.2},{:.1},{:.2},{:.1},\"{}\",\"{}\",{:.2},{:.1},\"{}\"\n",
            c.rank,
            c.candidate_id,
            c.handle_label,
            c.promoiety_code,
            c.promoiety_name,
            c.family.short_label(),
            ac.primary_enzyme.short_label(),
            ac.activation_t_half_min,
            ac.ces1_over_ces2_ratio,
            ac.luminal_stability_pct,
            ac.prodrug_sol_ph68_ug_ml,
            ac.solubility_fold_change,
            ac.permeability_fold_change,
            ac.predicted_fa_pct,
            ac.predicted_active_f_oral_pct,
            ac.f_oral_fold_gain,
            ac.active_cmax_nm,
            ac.active_tmax_hr,
            c.physchem.mw,
            c.logd_74,
            c.physchem.tpsa,
            ac.released_byproduct,
            ac.byproduct_safety.label(),
            c.sa_score,
            c.prodrug_score,
            c.canonical_smiles
        ));
    }
    out
}

/// Formats prodrug candidates as a 2D MDL V2000 `.sdf` bundle with `BCS` & enzymatic activation tags.
pub fn format_prodrugs_sdf(cands: &[ProdrugCandidate]) -> String {
    let mut out = String::new();
    for c in cands {
        let ac = &c.activation;
        let tags = [
            ("PROMOIETY_FAMILY", c.family.label().to_string()),
            ("PROMOIETY_NAME", c.promoiety_name.to_string()),
            ("ACTIVATING_ENZYME", ac.primary_enzyme.label().to_string()),
            ("ACTIVATION_T_HALF_MIN", format!("{:.1}", ac.activation_t_half_min)),
            ("SOL_PH68_UG_ML", format!("{:.1}", ac.prodrug_sol_ph68_ug_ml)),
            ("ACTIVE_F_ORAL_PCT", format!("{:.1}", ac.predicted_active_f_oral_pct)),
            ("RELEASED_BYPRODUCT", ac.released_byproduct.to_string()),
            ("PRODRUG_SCORE", format!("{:.1}", c.prodrug_score)),
        ];
        let mut rec = format_sdf_record(&c.mol);
        if let Some(pos) = rec.rfind("$$$$") {
            let mut block = String::new();
            for (k, v) in &tags {
                block.push_str(&format!(">  <{k}>\n{v}\n\n"));
            }
            rec.insert_str(pos, &block);
        }
        out.push_str(&rec);
    }
    out
}
