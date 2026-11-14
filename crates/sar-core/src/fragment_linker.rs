//! Sub-Pocket Fragment-Pair 3D Exit-Vector Aligner (`DeLinker` / `DEVELOP` / `AIMLinker` 4-DOF
//! Parameterization), `18`-Scaffold Linker & Direct-Merge Geometry Matcher, `Kabsch` Dual-Core
//! `RMSD` Pose-Fidelity Scorer, and `Jencks` / `Murray & Verdonk (2002)` Super-Additivity
//! Thermodynamic Cycle Studio (`fragment_linker.rs`, `v3.2.0`).
//!
//! Primary References:
//! - Murray & Verdonk (2002), *J. Comput.-Aided Mol. Des.* 16(10), 741-753 ("The consequences of
//!   translational and rotational entropy lost by small molecules on binding to proteins" —
//!   `15-20 kJ/mol` / `~4.2 kcal/mol` rigid-body barrier & super-additive fragment linking:
//!   `https://pubmed.ncbi.nlm.nih.gov/12650591/`)
//! - Jencks (1981) & Protein-Templated Super-Additive Fragment Ligations (`dG_AB = dG_A + dG_B + dG_link`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC7159684/`, Tankyrase dual-site `dG_link` analysis:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11997991/`, Bivalent ATP-allosteric kinase linking:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC10879502/`)
//! - Ichihara, Barker, Law & Whittaker (2011), *Mol. Inform.* 30(4), 298-306 ("Compound Design by
//!   Fragment-Linking" — pose preservation & linking coefficient `E_link`:
//!   `https://pubmed.ncbi.nlm.nih.gov/27466947/`, NMR growing/merging/linking:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3699969/`, `Fragmenstein` 3D merging/linking:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11731148/`)
//! - Imrie, Bradley, van der Schaar & Deane (2020), *J. Chem. Inf. Model.* 60(4), 1983-1995
//!   ("Deep Generative Models for 3D Linker Design" — `DeLinker` exit-vector distance/angle matching
//!   & `< 0.80 A` `RMSD_Fragments`: `https://pmc.ncbi.nlm.nih.gov/articles/PMC7189367/`, `DEVELOP`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8580048/`, `AIMLinker`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC10207268/`).

use crate::alerts::scan_structural_alerts;
use crate::conformer3d::{
    compute_gasteiger_charges, evaluate_3d_summary_profile, format_3d_conformer_sdf,
    generate_3d_conformer_ensemble, kabsch_align_3d, ConformerGenConfig,
};
use crate::fragment_grower::{
    evaluate_rule_of_three_audit, perceive_fragment_exit_vectors, ExitVectorChemistry,
    RuleOfThreeAudit, RT_LN10_KCAL,
};
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    PocketCavityAnalysis,
};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::torsion_strain::{analyze_torsion_strain_and_locks, BioactivePoseSource};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Canonical Murray & Verdonk (2002) translational + rotational rigid-body entropy barrier
/// saved upon strain-free linking of two sub-pocket fragments (`17.6 kJ/mol = 4.20 kcal/mol`).
pub const MURRAY_VERDONK_RIGID_ENTROPY_KCAL: f64 = 4.20;

/// `DeLinker` (`Imrie et al. 2020`) dual-fragment 3D pose-preservation cutoff (`0.80 A`).
pub const DELINKER_POSE_PRESERVED_RMSD_A: f64 = 0.80;

/// Source mode for acquiring a spatially positioned 3D sub-pocket `(Fragment A, Fragment B)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentPairPreset {
    /// `PDB 1M17` (`EGFR`): `6,7-Dimethoxyquinazoline` ATP-hinge anchor + `3-Ethynylphenyl` gatekeeper back-pocket fragment.
    Egfr1m17HingeAndBackPocket,
    /// `PDB 3LN1` (`COX-2`): `4-Sulfamoylphenyl` Arg513 polar side-pocket + `p-Tolyl` hydrophobic channel fragment.
    Cox23ln1SulfonamideAndTolyl,
    /// `PDB 6CM4` (`DRD2`): `Piperidin-4-yl` Asp114 salt-bridge + `6-Fluorobenzisoxazol-3-yl` aromatic cage fragment.
    Drd26cm4PiperidineAndBenzisoxazole,
    /// `PDB 1M17` (`EGFR` Bivalent): `7-Azaindole` hinge anchor + `3-CF3-Phenyl` allosteric C-helix back-pocket fragment.
    EgfrBivalentHingeToAllosteric,
    /// Deconstruct a selected dataset or sketched lead in its 3D receptor pose into two sub-pocket fragments (`A`, `B`) to scaffold-hop the central linker.
    DeconstructSelectedLead,
    /// Dock user-supplied `(frag_a_smiles, frag_b_smiles)` into non-overlapping primary & secondary hotspots of the selected receptor pocket.
    CustomSmilesSubpockets,
}

impl FragmentPairPreset {
    pub const ALL: [Self; 6] = [
        Self::Egfr1m17HingeAndBackPocket,
        Self::Cox23ln1SulfonamideAndTolyl,
        Self::Drd26cm4PiperidineAndBenzisoxazole,
        Self::EgfrBivalentHingeToAllosteric,
        Self::DeconstructSelectedLead,
        Self::CustomSmilesSubpockets,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::Egfr1m17HingeAndBackPocket => "EGFR-1M17-HINGE-GATE",
            Self::Cox23ln1SulfonamideAndTolyl => "COX2-3LN1-SO2NH2-TOL",
            Self::Drd26cm4PiperidineAndBenzisoxazole => "DRD2-6CM4-PIP-BZOX",
            Self::EgfrBivalentHingeToAllosteric => "EGFR-1M17-BIVALENT-ALLO",
            Self::DeconstructSelectedLead => "LEAD-3D-DECONSTRUCT",
            Self::CustomSmilesSubpockets => "CUSTOM-DUAL-HOTSPOT",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Egfr1m17HingeAndBackPocket => "EGFR 1M17 Hinge+Back",
            Self::Cox23ln1SulfonamideAndTolyl => "COX-2 3LN1 SO2NH2+Tol",
            Self::Drd26cm4PiperidineAndBenzisoxazole => "DRD2 6CM4 Pip+BzIsox",
            Self::EgfrBivalentHingeToAllosteric => "EGFR Bivalent Hinge+Allo",
            Self::DeconstructSelectedLead => "Deconstruct Lead",
            Self::CustomSmilesSubpockets => "Custom Frag A + B",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Egfr1m17HingeAndBackPocket => {
                "EGFR (1M17): Quinazoline Met793 Hinge + 3-Ethynylphenyl Thr790 Back-Pocket"
            }
            Self::Cox23ln1SulfonamideAndTolyl => {
                "COX-2 (3LN1): Arg513 Phenylsulfonamide Side-Pocket + p-Tolyl Hydrophobic Channel"
            }
            Self::Drd26cm4PiperidineAndBenzisoxazole => {
                "DRD2 (6CM4): Asp114 Piperidine Salt-Bridge + 6-F-Benzisoxazole Aromatic Cage"
            }
            Self::EgfrBivalentHingeToAllosteric => {
                "EGFR Bivalent (1M17): 7-Azaindole Hinge + 3-CF3-Phenyl Allosteric Back-Pocket"
            }
            Self::DeconstructSelectedLead => {
                "In-Situ 3D Lead Deconstruction (Excise Central Linker & Re-Link Sub-Pockets)"
            }
            Self::CustomSmilesSubpockets => {
                "Custom SMILES Pair Docked into Non-Overlapping Primary & Secondary Hotspots"
            }
        }
    }

    pub fn default_pocket_idx(self) -> usize {
        match self {
            Self::Egfr1m17HingeAndBackPocket | Self::EgfrBivalentHingeToAllosteric => 0,
            Self::Drd26cm4PiperidineAndBenzisoxazole => 1,
            Self::Cox23ln1SulfonamideAndTolyl => 2,
            Self::DeconstructSelectedLead | Self::CustomSmilesSubpockets => 0,
        }
    }
}

/// Spatial sub-pocket separation regime governing direct merging vs covalent linker bridging.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentJoinRegime {
    /// Overlapping or abutting sub-pocket poses (`d_min < 2.45 A` or `d_anchor < 2.75 A`): favors direct `0`-atom bond merge.
    DirectMergeOrFusion,
    /// Proximal sub-pockets (`2.75 <= d_anchor < 4.30 A`): `1`- to `2`-atom heteroatom/amide/alkyne/cyclopropyl bridge.
    ShortProximalBridge,
    /// Intermediate sub-pocket channel (`4.30 <= d_anchor < 5.90 A`): `5`-membered ring, `BCP`, or `azetidine` rigid bridge.
    MediumConformationalBridge,
    /// Wide bivalent channel (`d_anchor >= 5.90 A`): `6`-membered `piperidine` or `2-azaspiro[3.3]heptane` extended bridge.
    LongBivalentChannelBridge,
}

impl FragmentJoinRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::DirectMergeOrFusion => "Direct Fragment Merge / Single-Bond Fusion (<2.75 A)",
            Self::ShortProximalBridge => "Short Proximal Bridge (1-2 Atoms / Cyclopropyl, 2.75-4.3 A)",
            Self::MediumConformationalBridge => {
                "Medium Conformational Bridge (5-HetAr / BCP / Azetidine, 4.3-5.9 A)"
            }
            Self::LongBivalentChannelBridge => {
                "Long Bivalent Channel Bridge (Piperidine / Spiro[3.3], >=5.9 A)"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::DirectMergeOrFusion => "Direct Merge",
            Self::ShortProximalBridge => "Short Bridge",
            Self::MediumConformationalBridge => "Medium Rigid",
            Self::LongBivalentChannelBridge => "Long Bivalent",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::DirectMergeOrFusion => [244, 114, 182],       // Pink
            Self::ShortProximalBridge => [56, 189, 248],        // Sky Blue
            Self::MediumConformationalBridge => [52, 211, 153], // Emerald
            Self::LongBivalentChannelBridge => [192, 132, 252], // Violet
        }
    }
}

/// Chemical & conformational class of a linker scaffold in the `18`-template library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinkerScaffoldClass {
    DirectBondFusion,
    HeteroatomOrAlkylChain,
    AmideOrSulfonamide,
    RigidUnsaturatedOrHeterocycle,
    Sp3ConformationalCage,
}

impl LinkerScaffoldClass {
    pub const ALL: [Self; 5] = [
        Self::DirectBondFusion,
        Self::HeteroatomOrAlkylChain,
        Self::AmideOrSulfonamide,
        Self::RigidUnsaturatedOrHeterocycle,
        Self::Sp3ConformationalCage,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::DirectBondFusion => "Direct Bond",
            Self::HeteroatomOrAlkylChain => "Alkyl/Hetero",
            Self::AmideOrSulfonamide => "Amide/SO2NH",
            Self::RigidUnsaturatedOrHeterocycle => "Alkyne/HetAr",
            Self::Sp3ConformationalCage => "Sp3 Ring/BCP",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::DirectBondFusion => "0-Atom Direct C-C / C-N Covalent Bond Fusion",
            Self::HeteroatomOrAlkylChain => "1-2 Atom Flexible Methylene / Amine / Ether Bridge",
            Self::AmideOrSulfonamide => "H-Bonding Amide / Reverse-Amide / Sulfonamide Bridge",
            Self::RigidUnsaturatedOrHeterocycle => {
                "Rigid Linear Alkyne & 5-Membered Heteroaromatic Isosteres"
            }
            Self::Sp3ConformationalCage => {
                "Preorganized sp3 Conformational Rings (cPr, Azetidine, BCP, Piperidine, Spiro)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::DirectBondFusion => [244, 114, 182],              // Pink
            Self::HeteroatomOrAlkylChain => [148, 163, 184],        // Slate
            Self::AmideOrSulfonamide => [56, 189, 248],             // Sky Blue
            Self::RigidUnsaturatedOrHeterocycle => [251, 191, 36],  // Amber
            Self::Sp3ConformationalCage => [52, 211, 153],          // Emerald
        }
    }
}

/// Thermodynamic cooperativity classification under the Jencks (1981) / Murray-Verdonk (2002) cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuperAdditivityTier {
    /// `ddG_super <= -1.36 kcal/mol` (`>= 10x` potency gain beyond `K_A * K_B` additive expectation!) AND `Core-RMSD <= 0.95 A`.
    StrongSuperAdditive,
    /// `-1.36 < ddG_super <= -0.25 kcal/mol` (net rigid-body entropy gain outweighs linker strain).
    ModerateSuperAdditive,
    /// `-0.25 < ddG_super <= +0.85 kcal/mol` (roughly additive `dG_AB ~ dG_A + dG_B`).
    NearAdditive,
    /// `ddG_super > +0.85 kcal/mol` or `Core-RMSD > 1.35 A` (linker strain or pose distortion erodes sub-pocket contacts).
    SubAdditiveStrained,
}

impl SuperAdditivityTier {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::StrongSuperAdditive => "Super-Additive (<= -1.4 kcal)",
            Self::ModerateSuperAdditive => "Coop-Additive (-0.3..-1.4)",
            Self::NearAdditive => "Additive (~dG_A+dG_B)",
            Self::SubAdditiveStrained => "Sub-Additive (Strained)",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::StrongSuperAdditive => {
                "Strong Super-Additive Link (Saves Rigid-Body Entropy & Preserves Both Sub-Pocket Poses)"
            }
            Self::ModerateSuperAdditive => {
                "Moderate Cooperative Link (Favorable Net dG_link < -0.25 kcal/mol)"
            }
            Self::NearAdditive => {
                "Near-Additive Link (dG_AB ~ dG_A + dG_B; Rotor Entropy Balances Rigid-Body Gain)"
            }
            Self::SubAdditiveStrained => {
                "Sub-Additive / Pose-Distorting Link (Geometric Mismatch or High Torsion Strain)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::StrongSuperAdditive => [52, 211, 153],     // Emerald
            Self::ModerateSuperAdditive => [56, 189, 248],   // Sky Blue
            Self::NearAdditive => [251, 191, 36],            // Amber
            Self::SubAdditiveStrained => [248, 113, 113],    // Coral
        }
    }
}

/// Curated 3D linker scaffold entry with equilibrium geometric window `(d_span, alpha_A, alpha_B, phi)`
/// and `Cl...Br` dummy attachment SMILES for clean molecular graph splicing.
pub struct LinkerTemplate {
    pub code: &'static str,
    pub name: &'static str,
    pub class: LinkerScaffoldClass,
    /// `None` for `0`-atom direct `A-B` bond fusion; `Some("Cl...Br")` where `Cl` attaches to `Frag A` and `Br` attaches to `Frag B`.
    pub dummy_smiles: Option<&'static str>,
    pub coupling_reaction: &'static str,
    /// Equilibrium anchor-to-anchor `||r_B - r_A||` window (`[min_a, opt_a, max_a]` in Angstroms).
    pub span_min_a: f64,
    pub span_opt_a: f64,
    pub span_max_a: f64,
    /// Preferred exit-vector cone angles `(ideal_alpha_a_deg, ideal_alpha_b_deg, angle_tol_deg)`.
    pub ideal_alpha_a_deg: f64,
    pub ideal_alpha_b_deg: f64,
    pub alpha_tol_deg: f64,
    /// Preferred inter-vector torsion twist `ideal_phi_deg` (`None` = azimuthally isotropic, e.g. alkyne/BCP/flexible).
    pub ideal_phi_deg: Option<f64>,
    pub added_rotatable_bonds: usize,
    pub intrinsic_strain_floor_kcal: f64,
}

/// `18`-scaffold medicinal-chemistry linker & direct-merge library spanning `1.35 A` to `7.55 A`.
pub const LINKER_TEMPLATES: [LinkerTemplate; 18] = [
    LinkerTemplate {
        code: "LNK-00-DIRECT",
        name: "Direct Covalent Bond (0-Atom Merge)",
        class: LinkerScaffoldClass::DirectBondFusion,
        dummy_smiles: None,
        coupling_reaction: "Suzuki-Miyaura / Buchwald-Hartwig Direct C-C or C-N Coupling",
        span_min_a: 1.35,
        span_opt_a: 1.55,
        span_max_a: 2.95,
        ideal_alpha_a_deg: 14.0,
        ideal_alpha_b_deg: 14.0,
        alpha_tol_deg: 36.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 1,
        intrinsic_strain_floor_kcal: 0.10,
    },
    LinkerTemplate {
        code: "LNK-01-NH",
        name: "Secondary Amine / Aniline (-NH-)",
        class: LinkerScaffoldClass::HeteroatomOrAlkylChain,
        dummy_smiles: Some("ClNBr"),
        coupling_reaction: "Buchwald-Hartwig Pd/BrettPhos or SNAr C-N Amination",
        span_min_a: 2.15,
        span_opt_a: 2.55,
        span_max_a: 3.55,
        ideal_alpha_a_deg: 28.0,
        ideal_alpha_b_deg: 28.0,
        alpha_tol_deg: 38.0,
        ideal_phi_deg: Some(165.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.18,
    },
    LinkerTemplate {
        code: "LNK-02-CH2",
        name: "Methylene Bridge (-CH2-)",
        class: LinkerScaffoldClass::HeteroatomOrAlkylChain,
        dummy_smiles: Some("ClCBr"),
        coupling_reaction: "Negishi / Ni-Photoredox C(sp2)-CH2 Cross-Coupling",
        span_min_a: 2.25,
        span_opt_a: 2.62,
        span_max_a: 3.60,
        ideal_alpha_a_deg: 32.0,
        ideal_alpha_b_deg: 32.0,
        alpha_tol_deg: 40.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.22,
    },
    LinkerTemplate {
        code: "LNK-03-ETHER",
        name: "Ether Oxygen Bridge (-O-)",
        class: LinkerScaffoldClass::HeteroatomOrAlkylChain,
        dummy_smiles: Some("ClOBr"),
        coupling_reaction: "Ullmann CuI / Chan-Lam Diarylether Coupling",
        span_min_a: 2.10,
        span_opt_a: 2.48,
        span_max_a: 3.45,
        ideal_alpha_a_deg: 30.0,
        ideal_alpha_b_deg: 30.0,
        alpha_tol_deg: 38.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.20,
    },
    LinkerTemplate {
        code: "LNK-04-CH2CH2",
        name: "Ethylene Bridge (-CH2-CH2-)",
        class: LinkerScaffoldClass::HeteroatomOrAlkylChain,
        dummy_smiles: Some("ClCCBr"),
        coupling_reaction: "Alkene Hydroboration-Suzuki / Hydrogenation",
        span_min_a: 3.05,
        span_opt_a: 3.82,
        span_max_a: 4.65,
        ideal_alpha_a_deg: 28.0,
        ideal_alpha_b_deg: 28.0,
        alpha_tol_deg: 44.0,
        ideal_phi_deg: Some(180.0),
        added_rotatable_bonds: 3,
        intrinsic_strain_floor_kcal: 0.42,
    },
    LinkerTemplate {
        code: "LNK-05-CH2NH",
        name: "Aminomethylene (-CH2-NH-)",
        class: LinkerScaffoldClass::HeteroatomOrAlkylChain,
        dummy_smiles: Some("ClCNBr"),
        coupling_reaction: "Reductive Amination (Aldehyde + Amine, NaBH(OAc)3)",
        span_min_a: 3.00,
        span_opt_a: 3.74,
        span_max_a: 4.58,
        ideal_alpha_a_deg: 28.0,
        ideal_alpha_b_deg: 28.0,
        alpha_tol_deg: 44.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 3,
        intrinsic_strain_floor_kcal: 0.36,
    },
    LinkerTemplate {
        code: "LNK-06-CH2O",
        name: "Oxymethylene (-CH2-O-)",
        class: LinkerScaffoldClass::HeteroatomOrAlkylChain,
        dummy_smiles: Some("ClCOBr"),
        coupling_reaction: "Mitsunobu / Williamson Benzyl Ether Synthesis",
        span_min_a: 2.98,
        span_opt_a: 3.70,
        span_max_a: 4.52,
        ideal_alpha_a_deg: 30.0,
        ideal_alpha_b_deg: 30.0,
        alpha_tol_deg: 42.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 3,
        intrinsic_strain_floor_kcal: 0.35,
    },
    LinkerTemplate {
        code: "LNK-07-AMIDE",
        name: "Trans-Amide (-C(=O)-NH-)",
        class: LinkerScaffoldClass::AmideOrSulfonamide,
        dummy_smiles: Some("ClC(=O)NBr"),
        coupling_reaction: "HATU / T3P / DIPEA Carboxamide Coupling",
        span_min_a: 3.20,
        span_opt_a: 3.88,
        span_max_a: 4.70,
        ideal_alpha_a_deg: 22.0,
        ideal_alpha_b_deg: 24.0,
        alpha_tol_deg: 36.0,
        ideal_phi_deg: Some(180.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.16,
    },
    LinkerTemplate {
        code: "LNK-08-REVAMIDE",
        name: "Reverse Trans-Amide (-NH-C(=O)-)",
        class: LinkerScaffoldClass::AmideOrSulfonamide,
        dummy_smiles: Some("ClNC(=O)Br"),
        coupling_reaction: "HATU / T3P Reverse Carboxamide Coupling",
        span_min_a: 3.20,
        span_opt_a: 3.88,
        span_max_a: 4.70,
        ideal_alpha_a_deg: 24.0,
        ideal_alpha_b_deg: 22.0,
        alpha_tol_deg: 36.0,
        ideal_phi_deg: Some(180.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.16,
    },
    LinkerTemplate {
        code: "LNK-09-SULFONAMIDE",
        name: "Sulfonamide (-SO2-NH-)",
        class: LinkerScaffoldClass::AmideOrSulfonamide,
        dummy_smiles: Some("ClS(=O)(=O)NBr"),
        coupling_reaction: "Sulfonyl Chloride + Amine (Pyridine / Et3N)",
        span_min_a: 3.05,
        span_opt_a: 3.76,
        span_max_a: 4.60,
        ideal_alpha_a_deg: 42.0,
        ideal_alpha_b_deg: 36.0,
        alpha_tol_deg: 40.0,
        ideal_phi_deg: Some(90.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.24,
    },
    LinkerTemplate {
        code: "LNK-10-ALKYNE",
        name: "Linear Ethynyl Rod (-C#C-)",
        class: LinkerScaffoldClass::RigidUnsaturatedOrHeterocycle,
        dummy_smiles: Some("ClC#CBr"),
        coupling_reaction: "Sonogashira Pd(PPh3)2Cl2 / CuI Alkyne Cross-Coupling",
        span_min_a: 3.55,
        span_opt_a: 4.12,
        span_max_a: 4.95,
        ideal_alpha_a_deg: 8.0,
        ideal_alpha_b_deg: 8.0,
        alpha_tol_deg: 28.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 0,
        intrinsic_strain_floor_kcal: 0.08,
    },
    LinkerTemplate {
        code: "LNK-11-CYCLOPROPYL",
        name: "1,2-Cyclopropyl Bent Cage (-cPr-)",
        class: LinkerScaffoldClass::Sp3ConformationalCage,
        dummy_smiles: Some("ClC1CC1Br"),
        coupling_reaction: "Simmons-Smith / Cyclopropyl Trifluoroborate Suzuki",
        span_min_a: 2.95,
        span_opt_a: 3.62,
        span_max_a: 4.35,
        ideal_alpha_a_deg: 44.0,
        ideal_alpha_b_deg: 44.0,
        alpha_tol_deg: 36.0,
        ideal_phi_deg: Some(135.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.14,
    },
    LinkerTemplate {
        code: "LNK-12-AZETIDINE13",
        name: "1,3-Azetidine Compact sp3 Ring",
        class: LinkerScaffoldClass::Sp3ConformationalCage,
        dummy_smiles: Some("ClC1CN(Br)C1"),
        coupling_reaction: "Ni-Catalyzed 3-Iodoazetidine Coupling + Buchwald N-Arylation",
        span_min_a: 3.65,
        span_opt_a: 4.35,
        span_max_a: 5.15,
        ideal_alpha_a_deg: 28.0,
        ideal_alpha_b_deg: 24.0,
        alpha_tol_deg: 36.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.15,
    },
    LinkerTemplate {
        code: "LNK-13-BCP13",
        name: "1,3-Bicyclo[1.1.1]pentane (BCP)",
        class: LinkerScaffoldClass::Sp3ConformationalCage,
        dummy_smiles: Some("ClC12CC(Br)(C1)C2"),
        coupling_reaction: "[1.1.1]Propellane Metallaphotoredox Dual Cross-Coupling",
        span_min_a: 3.90,
        span_opt_a: 4.68,
        span_max_a: 5.45,
        ideal_alpha_a_deg: 10.0,
        ideal_alpha_b_deg: 10.0,
        alpha_tol_deg: 30.0,
        ideal_phi_deg: None,
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.09,
    },
    LinkerTemplate {
        code: "LNK-14-PYRAZOLE13",
        name: "1,3-Pyrazole Sub-Pocket Heterocycle",
        class: LinkerScaffoldClass::RigidUnsaturatedOrHeterocycle,
        dummy_smiles: Some("Clc1cc(Br)nn1"),
        coupling_reaction: "Knorr Hydrazine Cyclocondensation / Regioselective Suzuki",
        span_min_a: 3.95,
        span_opt_a: 4.72,
        span_max_a: 5.55,
        ideal_alpha_a_deg: 34.0,
        ideal_alpha_b_deg: 34.0,
        alpha_tol_deg: 36.0,
        ideal_phi_deg: Some(0.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.12,
    },
    LinkerTemplate {
        code: "LNK-15-OXADIAZOLE",
        name: "1,3,4-Oxadiazole Polar Isostere",
        class: LinkerScaffoldClass::RigidUnsaturatedOrHeterocycle,
        dummy_smiles: Some("Clc1nnc(Br)o1"),
        coupling_reaction: "Diacylhydrazide Burgess / TsCl Cyclodehydration",
        span_min_a: 4.05,
        span_opt_a: 4.82,
        span_max_a: 5.65,
        ideal_alpha_a_deg: 28.0,
        ideal_alpha_b_deg: 28.0,
        alpha_tol_deg: 34.0,
        ideal_phi_deg: Some(0.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.12,
    },
    LinkerTemplate {
        code: "LNK-16-TRIAZOLE14",
        name: "1,4-Triazole Click Bridge (CuAAC)",
        class: LinkerScaffoldClass::RigidUnsaturatedOrHeterocycle,
        dummy_smiles: Some("Clc1cn(Br)nn1"),
        coupling_reaction: "CuSO4 / Na-Ascorbate Regioselective CuAAC Click",
        span_min_a: 4.25,
        span_opt_a: 5.05,
        span_max_a: 5.95,
        ideal_alpha_a_deg: 20.0,
        ideal_alpha_b_deg: 22.0,
        alpha_tol_deg: 34.0,
        ideal_phi_deg: Some(170.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.11,
    },
    LinkerTemplate {
        code: "LNK-17-SPIRO33",
        name: "2-Azaspiro[3.3]heptane / Piperidine Channel",
        class: LinkerScaffoldClass::Sp3ConformationalCage,
        dummy_smiles: Some("ClC1CC2(C1)CN(Br)C2"),
        coupling_reaction: "Photoredox Decarboxylative Arylation + Buchwald N-Coupling",
        span_min_a: 4.90,
        span_opt_a: 6.05,
        span_max_a: 7.55,
        ideal_alpha_a_deg: 18.0,
        ideal_alpha_b_deg: 18.0,
        alpha_tol_deg: 38.0,
        ideal_phi_deg: Some(90.0),
        added_rotatable_bonds: 2,
        intrinsic_strain_floor_kcal: 0.16,
    },
];

/// Single 3D exit vector perceived on a sub-pocket fragment (`Frag A` or `Frag B`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExitVector3d {
    pub vector_idx: usize,
    /// Heavy-atom index on the fragment core (before halide stripping).
    pub anchor_atom: usize,
    /// Optional poised halide (`-Cl`, `-Br`, `-I`) displaced upon linking.
    pub displaced_halide_atom: Option<usize>,
    pub atom_label: String,
    pub chemistry: ExitVectorChemistry,
    pub anchor_pos3d: [f32; 3],
    pub unit_dir3d: [f32; 3],
    /// Covalent bond-tip coordinate `anchor_pos3d + 1.48 * unit_dir3d` (`Angstrom`).
    pub exit_tip_pos3d: [f32; 3],
    pub masks_hinge_hbd: bool,
}

/// `DeLinker` / `Caveat` 4-DOF 3D exit-vector alignment descriptor between `Frag A` and `Frag B`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExitVectorPairGeometry {
    pub pair_rank: usize,
    pub vec_a: ExitVector3d,
    pub vec_b: ExitVector3d,
    /// Heavy-anchor distance `||r_B - r_A||` (`Angstrom`).
    pub anchor_distance_a: f64,
    /// Projected covalent tip-to-tip distance `||t_B - t_A||` (`Angstrom`).
    pub exit_tip_distance_a: f64,
    /// Cone angle `alpha_A` (`0..180 deg`) between `u_A` and chord `(r_B - r_A)` (`0 deg` = collinear toward `B`).
    pub alpha_a_deg: f64,
    /// Cone angle `alpha_B` (`0..180 deg`) between `u_B` and chord `(r_A - r_B)` (`0 deg` = collinear toward `A`).
    pub alpha_b_deg: f64,
    /// Signed 3D torsion twist `(tip_A - anchor_A - anchor_B - tip_B)` in `[0, 360)` degrees.
    pub dihedral_phi_deg: f64,
    /// Closest heavy-atom separation between Fragment `A` and Fragment `B` (`Angstrom`).
    pub min_interfragment_heavy_dist_a: f64,
    /// Composite `[0..100]` 3D mutual exit-vector alignment score (`100` = unobstructed head-on poised vectors).
    pub vector_alignment_score: f64,
    pub regime: FragmentJoinRegime,
}

/// Complete Jencks / Murray-Verdonk thermodynamic & 3D pose-fidelity profile of a linked chimera `A-L-B`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkedFragmentCandidate {
    pub rank: usize,
    pub candidate_id: String,
    pub linker_code: &'static str,
    pub linker_name: &'static str,
    pub linker_class: LinkerScaffoldClass,
    pub coupling_reaction: &'static str,
    pub canonical_smiles: String,

    // 3D Geometry Match & Pose Preservation (`DeLinker` / `Fragmenstein`)
    pub geometry_match_score: f64,
    pub span_mismatch_a: f64,
    pub angle_mismatch_deg: f64,
    /// Kabsch `RMSD` (`Angstrom`) of Fragment `A` core atoms in `A-L-B` vs unlinked Fragment `A` pose.
    pub rmsd_frag_a_a: f64,
    /// Kabsch `RMSD` (`Angstrom`) of Fragment `B` core atoms in `A-L-B` vs unlinked Fragment `B` pose.
    pub rmsd_frag_b_a: f64,
    /// Simultaneous dual-core Kabsch `RMSD_Fragments` (`Angstrom`; `< 0.80 A` = pose-preserving link).
    pub fragment_core_rmsd_a: f64,
    pub pose_preserved: bool,

    // Jencks (1981) + Murray & Verdonk (2002) Super-Additivity Thermodynamic Cycle (`kcal/mol`)
    pub dg_frag_a_kcal: f64,
    pub dg_frag_b_kcal: f64,
    pub dg_additive_sum_kcal: f64,
    /// Saved rigid-body translational/rotational entropy (`-4.20 * f_pose` `kcal/mol`, negative = favorable).
    pub dg_rigid_entropy_saved_kcal: f64,
    /// Linker conformational strain (`TEU`/rotors) + sub-pocket pose-distortion penalty (`>= 0 kcal/mol`).
    pub dg_linker_strain_kcal: f64,
    /// Direct linker–channel non-covalent contact & `WaterMap` displacement delta (`kcal/mol`).
    pub dg_linker_pocket_kcal: f64,
    /// Net linking free-energy term `dG_link = dG_rigid_saved + dG_strain + dG_linker_pocket` (`kcal/mol`).
    pub dg_link_term_kcal: f64,
    /// Total linked binding free energy `dG_AB = dG_A + dG_B + dG_link` (`kcal/mol`).
    pub dg_linked_ab_kcal: f64,
    /// Super-additivity gain `ddG_super = dG_AB - (dG_A + dG_B)` (`kcal/mol`; negative = super-additive!).
    pub ddg_superadditivity_kcal: f64,
    /// Potency amplification factor over `K_A * K_B` (`10^(-ddG_super / 1.364)`).
    pub cooperativity_fold: f64,
    /// Ichihara (2011) linking coefficient `E_link = K_AB / (K_A * K_B)` (`M^-1`; `< 1.0` = super-additive).
    pub ichihara_linking_coeff_e: f64,
    pub superadditivity_tier: SuperAdditivityTier,

    // Medicinal-Chemistry Efficiency, Strain & Developability Readouts
    pub predicted_pkd: f64,
    pub le: f64,
    pub lle: f64,
    pub lle_74: f64,
    /// Linker Group Efficiency `GE_linker = -dG_link / max(1, delta_HAC_linker)` (`kcal/mol/HA`).
    pub linker_ge: f64,
    pub total_strain_teu: f64,
    pub sa_score: f64,
    pub pains_alerts: usize,
    pub physchem: PhyschemProfile,
    /// Composite `[0..100]` Upstream Fragment-Linking Score balancing `-dG_AB`, `-ddG_super`, low `Core-RMSD`, `LLE`, `TEU`, and `SAScore`.
    pub composite_linker_score: f64,

    // Atom Provenance & Aligned 3D Pose for Interactive Viewport & SDF Export
    pub frag_a_atoms: Vec<usize>,
    pub linker_atoms: Vec<usize>,
    pub frag_b_atoms: Vec<usize>,
    pub aligned_coords3d: Vec<[f32; 3]>,
    pub mol: Molecule,
}

/// Complete sub-pocket fragment-pair 3D exit-vector alignment & `18`-linker super-additivity report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentLinkingReport {
    pub preset: FragmentPairPreset,
    pub target_pdb_id: String,
    pub target_name: String,
    pub pair_description: String,

    pub frag_a_mol: Molecule,
    pub frag_a_coords3d: Vec<[f32; 3]>,
    pub frag_a_ro3: RuleOfThreeAudit,
    pub dg_frag_a_kcal: f64,
    pub pkd_frag_a: f64,

    pub frag_b_mol: Molecule,
    pub frag_b_coords3d: Vec<[f32; 3]>,
    pub frag_b_ro3: RuleOfThreeAudit,
    pub dg_frag_b_kcal: f64,
    pub pkd_frag_b: f64,

    pub vector_pairs: Vec<ExitVectorPairGeometry>,
    pub active_pair_idx: usize,
    pub candidates: Vec<LinkedFragmentCandidate>,
    pub pocket: PocketCavityAnalysis,
}

/// Configuration for running a sub-pocket fragment-linking & super-additivity campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentLinkerConfig {
    pub preset: FragmentPairPreset,
    pub vector_pair_rank: usize,
    pub custom_frag_a_smiles: String,
    pub custom_frag_b_smiles: String,
    pub custom_pocket_idx: usize,
}

impl Default for FragmentLinkerConfig {
    fn default() -> Self {
        Self {
            preset: FragmentPairPreset::Egfr1m17HingeAndBackPocket,
            vector_pair_rank: 0,
            custom_frag_a_smiles: "COc1cc2ncnc(Cl)c2cc1OC".to_string(),
            custom_frag_b_smiles: "Brc1cccc(C#C)c1".to_string(),
            custom_pocket_idx: 0,
        }
    }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = (a[0] - b[0]) as f64;
    let dy = (a[1] - b[1]) as f64;
    let dz = (a[2] - b[2]) as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [
        (a[0] - b[0]) as f64,
        (a[1] - b[1]) as f64,
        (a[2] - b[2]) as f64,
    ]
}

fn norm3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dihedral_deg_3d(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f64 {
    let b1 = sub3(p1, p0);
    let b2 = norm3(sub3(p2, p1));
    let b3 = sub3(p3, p2);
    let n1 = cross3(b1, b2);
    let n2 = cross3(b2, b3);
    let m1 = cross3(n1, b2);
    let x = dot3(n1, n2);
    let y = dot3(m1, n2);
    let mut deg = y.atan2(x).to_degrees();
    if deg < 0.0 {
        deg += 360.0;
    }
    deg
}

fn translate_coords(coords: &[[f32; 3]], target_centroid: [f32; 3]) -> Vec<[f32; 3]> {
    if coords.is_empty() {
        return Vec::new();
    }
    let inv = 1.0 / (coords.len() as f32);
    let mut c = [0.0f32; 3];
    for p in coords {
        c[0] += p[0] * inv;
        c[1] += p[1] * inv;
        c[2] += p[2] * inv;
    }
    coords
        .iter()
        .map(|p| {
            [
                p[0] - c[0] + target_centroid[0],
                p[1] - c[1] + target_centroid[1],
                p[2] - c[2] + target_centroid[2],
            ]
        })
        .collect()
}

/// Orient fragment `coords` so its best poised exit vector points cleanly toward `partner_target`
/// while preserving its sub-pocket centroid `subpocket_center`.
fn orient_fragment_exit_toward(
    mol: &Molecule,
    base_coords: &[[f32; 3]],
    subpocket_center: [f32; 3],
    partner_target: [f32; 3],
) -> Vec<[f32; 3]> {
    let centered = translate_coords(base_coords, subpocket_center);
    let vecs_2d = perceive_fragment_exit_vectors(mol);
    let Some(best_v) = vecs_2d.first() else {
        return centered;
    };
    let anchor = best_v.anchor_atom.min(centered.len().saturating_sub(1));

    // Compute current outward 3D unit vector on `centered`
    let dir_cur = compute_atom_outward_dir3d(mol, &centered, anchor, best_v.displaced_halide_atom);
    let desired = norm3(sub3(partner_target, subpocket_center));

    // Rodrigues rotation around subpocket_center aligning dir_cur -> desired
    let axis_raw = cross3(dir_cur, desired);
    let sin_t = (axis_raw[0] * axis_raw[0] + axis_raw[1] * axis_raw[1] + axis_raw[2] * axis_raw[2])
        .sqrt();
    let cos_t = dot3(dir_cur, desired).clamp(-1.0, 1.0);
    if sin_t < 1e-6 {
        if cos_t > 0.0 {
            return centered;
        }
        // 180-deg flip around an orthogonal axis
        let ortho = if dir_cur[0].abs() < 0.8 {
            norm3(cross3(dir_cur, [1.0, 0.0, 0.0]))
        } else {
            norm3(cross3(dir_cur, [0.0, 1.0, 0.0]))
        };
        return centered
            .iter()
            .map(|&p| {
                let rel = [
                    (p[0] - subpocket_center[0]) as f64,
                    (p[1] - subpocket_center[1]) as f64,
                    (p[2] - subpocket_center[2]) as f64,
                ];
                let proj = dot3(ortho, rel);
                let rot = [
                    2.0 * proj * ortho[0] - rel[0],
                    2.0 * proj * ortho[1] - rel[1],
                    2.0 * proj * ortho[2] - rel[2],
                ];
                [
                    subpocket_center[0] + (rot[0] as f32),
                    subpocket_center[1] + (rot[1] as f32),
                    subpocket_center[2] + (rot[2] as f32),
                ]
            })
            .collect();
    }

    let k = [axis_raw[0] / sin_t, axis_raw[1] / sin_t, axis_raw[2] / sin_t];
    centered
        .iter()
        .map(|&p| {
            let v = [
                (p[0] - subpocket_center[0]) as f64,
                (p[1] - subpocket_center[1]) as f64,
                (p[2] - subpocket_center[2]) as f64,
            ];
            let kv = cross3(k, v);
            let k_dot_v = dot3(k, v);
            let rot = [
                v[0] * cos_t + kv[0] * sin_t + k[0] * k_dot_v * (1.0 - cos_t),
                v[1] * cos_t + kv[1] * sin_t + k[1] * k_dot_v * (1.0 - cos_t),
                v[2] * cos_t + kv[2] * sin_t + k[2] * k_dot_v * (1.0 - cos_t),
            ];
            [
                subpocket_center[0] + (rot[0] as f32),
                subpocket_center[1] + (rot[1] as f32),
                subpocket_center[2] + (rot[2] as f32),
            ]
        })
        .collect()
}

fn compute_atom_outward_dir3d(
    mol: &Molecule,
    coords3d: &[[f32; 3]],
    anchor_idx: usize,
    displaced_halide: Option<usize>,
) -> [f64; 3] {
    if anchor_idx >= coords3d.len() {
        return [1.0, 0.0, 0.0];
    }
    let anchor_p = coords3d[anchor_idx];
    if let Some(hal_idx) = displaced_halide {
        if hal_idx < coords3d.len() {
            return norm3(sub3(coords3d[hal_idx], anchor_p));
        }
    }
    let mut sum = [0.0f64; 3];
    let mut cnt = 0usize;
    if let Some(nbrs) = mol.adj.get(anchor_idx) {
        for &(nb, _) in nbrs {
            if nb < coords3d.len() && Some(nb) != displaced_halide {
                let d = norm3(sub3(coords3d[nb], anchor_p));
                sum[0] += d[0];
                sum[1] += d[1];
                sum[2] += d[2];
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        [1.0, 0.0, 0.0]
    } else {
        norm3([-sum[0], -sum[1], -sum[2]])
    }
}

/// Perceives 3D exit vectors `(anchor_pos3d, unit_dir3d, exit_tip_pos3d)` on a positioned sub-pocket fragment.
pub fn perceive_fragment_exit_vectors_3d(
    mol: &Molecule,
    coords3d: &[[f32; 3]],
) -> Vec<ExitVector3d> {
    let vecs_2d = perceive_fragment_exit_vectors(mol);
    vecs_2d
        .into_iter()
        .filter(|v| v.anchor_atom < coords3d.len())
        .map(|v| {
            let anchor_p = coords3d[v.anchor_atom];
            let u = compute_atom_outward_dir3d(mol, coords3d, v.anchor_atom, v.displaced_halide_atom);
            let u_f32 = [u[0] as f32, u[1] as f32, u[2] as f32];
            let tip = [
                anchor_p[0] + 1.48 * u_f32[0],
                anchor_p[1] + 1.48 * u_f32[1],
                anchor_p[2] + 1.48 * u_f32[2],
            ];
            ExitVector3d {
                vector_idx: v.vector_idx,
                anchor_atom: v.anchor_atom,
                displaced_halide_atom: v.displaced_halide_atom,
                atom_label: v.atom_label,
                chemistry: v.chemistry,
                anchor_pos3d: anchor_p,
                unit_dir3d: u_f32,
                exit_tip_pos3d: tip,
                masks_hinge_hbd: v.masks_hinge_hbd,
            }
        })
        .collect()
}

/// Evaluates and ranks all `(vec_A, vec_B)` 3D exit-vector pairs between two positioned sub-pocket fragments.
pub fn rank_exit_vector_pairs_3d(
    frag_a: &Molecule,
    coords_a: &[[f32; 3]],
    frag_b: &Molecule,
    coords_b: &[[f32; 3]],
) -> Vec<ExitVectorPairGeometry> {
    let vecs_a = perceive_fragment_exit_vectors_3d(frag_a, coords_a);
    let vecs_b = perceive_fragment_exit_vectors_3d(frag_b, coords_b);

    let mut min_heavy_ab = f64::INFINITY;
    for (ia, pa) in coords_a.iter().enumerate() {
        if frag_a.atoms.get(ia).map(|a| a.elem == Element::H).unwrap_or(false) {
            continue;
        }
        for (ib, pb) in coords_b.iter().enumerate() {
            if frag_b.atoms.get(ib).map(|a| a.elem == Element::H).unwrap_or(false) {
                continue;
            }
            let d = dist3(*pa, *pb);
            if d < min_heavy_ab {
                min_heavy_ab = d;
            }
        }
    }
    if !min_heavy_ab.is_finite() {
        min_heavy_ab = 4.0;
    }

    let mut pairs: Vec<ExitVectorPairGeometry> = Vec::new();
    for va in &vecs_a {
        for vb in &vecs_b {
            let d_anchor = dist3(va.anchor_pos3d, vb.anchor_pos3d).max(0.50);
            let d_tip = dist3(va.exit_tip_pos3d, vb.exit_tip_pos3d);
            let chord_ab = norm3(sub3(vb.anchor_pos3d, va.anchor_pos3d));
            let chord_ba = [-chord_ab[0], -chord_ab[1], -chord_ab[2]];
            let ua = [
                va.unit_dir3d[0] as f64,
                va.unit_dir3d[1] as f64,
                va.unit_dir3d[2] as f64,
            ];
            let ub = [
                vb.unit_dir3d[0] as f64,
                vb.unit_dir3d[1] as f64,
                vb.unit_dir3d[2] as f64,
            ];

            let alpha_a = dot3(ua, chord_ab).clamp(-1.0, 1.0).acos().to_degrees();
            let alpha_b = dot3(ub, chord_ba).clamp(-1.0, 1.0).acos().to_degrees();
            let phi = dihedral_deg_3d(
                va.exit_tip_pos3d,
                va.anchor_pos3d,
                vb.anchor_pos3d,
                vb.exit_tip_pos3d,
            );

            let poised_bonus_a = if va.displaced_halide_atom.is_some() {
                12.0
            } else {
                0.0
            };
            let poised_bonus_b = if vb.displaced_halide_atom.is_some() {
                12.0
            } else {
                0.0
            };
            let hinge_pen = (if va.masks_hinge_hbd { 22.0 } else { 0.0 })
                + (if vb.masks_hinge_hbd { 22.0 } else { 0.0 });
            let cos_sum = ((180.0 - alpha_a) / 180.0) * 38.0 + ((180.0 - alpha_b) / 180.0) * 38.0;
            let dist_window_bonus = if (1.8..=7.8).contains(&d_anchor) {
                14.0
            } else {
                2.0
            };
            let align_score =
                (cos_sum + poised_bonus_a + poised_bonus_b + dist_window_bonus - hinge_pen)
                    .clamp(0.0, 100.0);

            let regime = if min_heavy_ab < 2.45 || d_anchor < 2.75 {
                FragmentJoinRegime::DirectMergeOrFusion
            } else if d_anchor < 4.30 {
                FragmentJoinRegime::ShortProximalBridge
            } else if d_anchor < 5.90 {
                FragmentJoinRegime::MediumConformationalBridge
            } else {
                FragmentJoinRegime::LongBivalentChannelBridge
            };

            pairs.push(ExitVectorPairGeometry {
                pair_rank: 0,
                vec_a: va.clone(),
                vec_b: vb.clone(),
                anchor_distance_a: d_anchor,
                exit_tip_distance_a: d_tip,
                alpha_a_deg: alpha_a,
                alpha_b_deg: alpha_b,
                dihedral_phi_deg: phi,
                min_interfragment_heavy_dist_a: min_heavy_ab,
                vector_alignment_score: align_score,
                regime,
            });
        }
    }

    pairs.sort_by(|a, b| {
        b.vector_alignment_score
            .partial_cmp(&a.vector_alignment_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, p) in pairs.iter_mut().enumerate() {
        p.pair_rank = idx + 1;
    }
    pairs
}

/// Deconstructs a multi-ring lead in its aligned 3D pocket pose into two non-overlapping
/// sub-pocket fragments `(Frag A, coords_A, Frag B, coords_B)` by cleaving the most central
/// acyclic linker chain (and capping each cut anchor with `-Br` so the exact original exit vectors
/// are preserved!).
fn deconstruct_lead_into_3d_subpocket_pair(
    lead: &Molecule,
    pocket: &PocketCavityAnalysis,
) -> Option<(Molecule, Vec<[f32; 3]>, Molecule, Vec<[f32; 3]>, String)> {
    let scored = score_ligand_in_pocket(lead, None, pocket);
    let coords = &scored.aligned_ligand_coords3d;
    if coords.len() != lead.atoms.len() || lead.atoms.len() < 11 {
        return None;
    }

    // Find an acyclic single bond (or pair of acyclic bonds around a central linker) that splits
    // `lead` into two ring-containing sub-pocket cores of >= 5 heavy atoms each.
    let n = lead.atoms.len();
    let mut best_split: Option<(Vec<usize>, usize, [f32; 3], Vec<usize>, usize, [f32; 3])> = None;
    let mut best_balance = 0usize;

    // Strategy A: Try excising a small 1-3 atom acyclic bridge between two ring systems so
    // re-linking tests true sub-pocket bridging across the excised channel!
    for u in 0..n {
        if !lead.atoms[u].in_ring {
            continue;
        }
        for &(m1, b1) in &lead.adj[u] {
            if lead.bonds[b1].in_ring || lead.atoms[m1].in_ring {
                continue;
            }
            // Walk up to 2 atoms through acyclic linker atoms to reach a second ring atom `v`
            let mut queue: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
            queue.push_back((m1, vec![m1]));
            while let Some((curr, path)) = queue.pop_front() {
                if path.len() > 3 {
                    continue;
                }
                for &(nxt, bn) in &lead.adj[curr] {
                    if nxt == u || path.contains(&nxt) || lead.bonds[bn].in_ring {
                        continue;
                    }
                    if lead.atoms[nxt].in_ring {
                        // Partition graph with `path` removed
                        let blocked: BTreeSet<usize> = path.iter().copied().collect();
                        let comp_a = collect_component(lead, u, &blocked);
                        let comp_b = collect_component(lead, nxt, &blocked);
                        if !comp_a.contains(&nxt) && comp_a.len() >= 5 && comp_b.len() >= 5 {
                            let bal = comp_a.len().min(comp_b.len()) * 10 + path.len() * 3;
                            if bal > best_balance {
                                best_balance = bal;
                                best_split = Some((
                                    comp_a,
                                    u,
                                    coords[m1],
                                    comp_b,
                                    nxt,
                                    coords[curr],
                                ));
                            }
                        }
                    } else if lead.atoms[nxt].elem != Element::H {
                        let mut next_path = path.clone();
                        next_path.push(nxt);
                        queue.push_back((nxt, next_path));
                    }
                }
            }
        }
    }

    // Strategy B: Direct single acyclic bond cut (e.g. biaryl or direct C-N bond)
    if best_split.is_none() {
        for b in &lead.bonds {
            if b.in_ring || b.order != BondOrder::Single {
                continue;
            }
            let comp_a = collect_component_excluding_bond(lead, b.a, b.idx);
            let comp_b = collect_component_excluding_bond(lead, b.b, b.idx);
            if !comp_a.contains(&b.b) && comp_a.len() >= 5 && comp_b.len() >= 5 {
                let bal = comp_a.len().min(comp_b.len());
                if bal > best_balance {
                    best_balance = bal;
                    best_split = Some((
                        comp_a,
                        b.a,
                        coords[b.b],
                        comp_b,
                        b.b,
                        coords[b.a],
                    ));
                }
            }
        }
    }

    let (comp_a, anchor_a, dir_pt_a, comp_b, anchor_b, dir_pt_b) = best_split?;
    let (mut mol_a, coords_a) =
        extract_poised_subfragment_3d(lead, coords, &comp_a, anchor_a, dir_pt_a, "FRAG-A")?;
    let (mut mol_b, coords_b) =
        extract_poised_subfragment_3d(lead, coords, &comp_b, anchor_b, dir_pt_b, "FRAG-B")?;
    mol_a.target = lead.target.clone();
    mol_b.target = lead.target.clone();
    let desc = format!(
        "In-situ 3D deconstruction of {} ({}) inside PDB {} into sub-pocket cores ({} HA + {} HA)",
        lead.name,
        lead.id,
        pocket.pdb_id,
        comp_a.len(),
        comp_b.len()
    );
    Some((mol_a, coords_a, mol_b, coords_b, desc))
}

fn collect_component(mol: &Molecule, start: usize, blocked: &BTreeSet<usize>) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    let mut q = VecDeque::new();
    seen.insert(start);
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        for &(v, _) in &mol.adj[u] {
            if !blocked.contains(&v) && seen.insert(v) {
                q.push_back(v);
            }
        }
    }
    seen.into_iter().collect()
}

fn collect_component_excluding_bond(
    mol: &Molecule,
    start: usize,
    skip_bond_idx: usize,
) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    let mut q = VecDeque::new();
    seen.insert(start);
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        for &(v, bi) in &mol.adj[u] {
            if bi != skip_bond_idx && seen.insert(v) {
                q.push_back(v);
            }
        }
    }
    seen.into_iter().collect()
}

fn extract_poised_subfragment_3d(
    parent: &Molecule,
    parent_coords: &[[f32; 3]],
    keep_atoms: &[usize],
    cut_anchor_old: usize,
    cut_partner_pos: [f32; 3],
    id_tag: &str,
) -> Option<(Molecule, Vec<[f32; 3]>)> {
    let mut old_to_new = vec![usize::MAX; parent.atoms.len()];
    let mut sub = Molecule::new("");
    let mut coords = Vec::with_capacity(keep_atoms.len() + 1);

    for (ni, &oi) in keep_atoms.iter().enumerate() {
        old_to_new[oi] = ni;
        let mut a = parent.atoms[oi].clone();
        a.idx = ni;
        sub.atoms.push(a);
        coords.push(parent_coords[oi]);
    }
    for b in &parent.bonds {
        let na = old_to_new[b.a];
        let nb = old_to_new[b.b];
        if na != usize::MAX && nb != usize::MAX {
            sub.bonds.push(Bond {
                idx: sub.bonds.len(),
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: b.in_ring,
                kekule_order: b.kekule_order,
            });
        }
    }

    // If the cut anchor is a carbon atom, append a poised `-Br` dummy atom along the exact 3D cut vector
    // so `perceive_fragment_exit_vectors_3d` recovers the exact crystallographic exit ray!
    let new_anchor = old_to_new[cut_anchor_old];
    if new_anchor != usize::MAX && sub.atoms[new_anchor].elem == Element::C {
        let anch_pos = coords[new_anchor];
        let dir = norm3(sub3(cut_partner_pos, anch_pos));
        let br_idx = sub.atoms.len();
        sub.atoms.push(Atom {
            idx: br_idx,
            elem: Element::Br,
            charge: 0,
            aromatic: false,
            bracketed: false,
            explicit_h: 0,
            implicit_h: 0,
            chiral: 0,
            in_ring: false,
            ring_sizes: Vec::new(),
            hyb: crate::mol::Hybridization::Sp3,
            pos2d: [0.0, 0.0],
        });
        sub.bonds.push(Bond {
            idx: sub.bonds.len(),
            a: new_anchor,
            b: br_idx,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
        coords.push([
            anch_pos[0] + 1.88 * (dir[0] as f32),
            anch_pos[1] + 1.88 * (dir[1] as f32),
            anch_pos[2] + 1.88 * (dir[2] as f32),
        ]);
    }

    sub.sanitize();
    compute_2d_coords(&mut sub);
    sub.smiles = molecule_to_smiles(&sub);
    sub.id = format!("{}-{}", parent.id, id_tag);
    sub.name = format!("{} ({})", parent.name, id_tag);
    if sub.smiles.is_empty() {
        return None;
    }
    Some((sub, coords))
}

/// Builds the positioned 3D sub-pocket `(Frag A, Frag B, PocketCavityAnalysis)` state for any preset.
fn prepare_subpocket_fragment_pair(
    cfg: &FragmentLinkerConfig,
    seed_lead: Option<&Molecule>,
) -> (
    Molecule,
    Vec<[f32; 3]>,
    Molecule,
    Vec<[f32; 3]>,
    PocketCavityAnalysis,
    String,
    String,
) {
    let resolved_pocket_idx = match cfg.preset {
        FragmentPairPreset::DeconstructSelectedLead => seed_lead
            .map(|m| {
                let up = m.target.to_ascii_uppercase();
                if up.contains("DRD") || up.contains("D2") {
                    1
                } else if up.contains("COX") || up.contains("PTGS") {
                    2
                } else {
                    0
                }
            })
            .unwrap_or(cfg.custom_pocket_idx.min(2)),
        FragmentPairPreset::CustomSmilesSubpockets => cfg.custom_pocket_idx.min(2),
        other => other.default_pocket_idx(),
    };

    let pdb_struct =
        parse_pdb_structure(curated_pocket_pdb_text(resolved_pocket_idx)).expect("Valid preset PDB");
    let pocket = analyze_binding_pocket(&pdb_struct, None, 10.0);
    let target_name = match resolved_pocket_idx {
        0 => "EGFR".to_string(),
        1 => "DRD2".to_string(),
        _ => "COX-2".to_string(),
    };

    if cfg.preset == FragmentPairPreset::DeconstructSelectedLead {
        if let Some(lead) = seed_lead {
            if let Some((fa, ca, fb, cb, desc)) = deconstruct_lead_into_3d_subpocket_pair(lead, &pocket) {
                return (fa, ca, fb, cb, pocket, target_name, desc);
            }
        }
    }

    // Curated co-bound sub-pocket benchmarks & custom SMILES pair
    let (smi_a, name_a, offset_a, smi_b, name_b, offset_b, desc): (
        &str,
        &str,
        [f32; 3],
        &str,
        &str,
        [f32; 3],
        String,
    ) = match cfg.preset {
        FragmentPairPreset::Egfr1m17HingeAndBackPocket | FragmentPairPreset::DeconstructSelectedLead => (
            "COc1cc2ncnc(Cl)c2cc1OC",
            "6,7-Dimethoxyquinazoline (Hinge)",
            [-2.15, -0.35, 0.20],
            "Brc1cccc(C#C)c1",
            "3-Ethynylphenyl (Gatekeeper Back-Pocket)",
            [2.45, 0.40, -0.25],
            "PDB 1M17 EGFR Met793 hinge quinazoline + Thr790 gatekeeper ethynylphenyl sub-pocket pair (Erlotinib deconstruction benchmark)".to_string(),
        ),
        FragmentPairPreset::Cox23ln1SulfonamideAndTolyl => (
            "NS(=O)(=O)c1ccc(Br)cc1",
            "4-Sulfamoylphenyl (Arg513 Side-Pocket)",
            [-2.75, 0.55, -0.20],
            "Cc1ccc(Br)cc1",
            "p-Tolyl (Hydrophobic COX-2 Channel)",
            [2.85, -0.50, 0.25],
            "PDB 3LN1 COX-2 Arg513/His90 sulfonamide selectivity pocket + hydrophobic diarylheterocycle channel pair (Celecoxib benchmark)".to_string(),
        ),
        FragmentPairPreset::Drd26cm4PiperidineAndBenzisoxazole => (
            "BrC1CCNCC1",
            "Piperidin-4-yl (Asp114 Salt-Bridge)",
            [-2.35, -0.25, 0.15],
            "Fc1ccc2c(Br)noc2c1",
            "6-Fluorobenzisoxazole (Trp386 Cage)",
            [2.55, 0.35, -0.20],
            "PDB 6CM4 DRD2 Asp114 (D3.32) protonatable amine anchor + Trp386/Phe389 deep aromatic cage pair (Risperidone benchmark)".to_string(),
        ),
        FragmentPairPreset::EgfrBivalentHingeToAllosteric => (
            "Brc1ccnc2[nH]ccc12",
            "7-Azaindole (ATP-Hinge HBD/HBA)",
            [-3.35, -0.45, 0.30],
            "FC(F)(F)c1cccc(Br)c1",
            "3-CF3-Phenyl (Allosteric C-Helix Back-Pocket)",
            [3.55, 0.55, -0.30],
            "PDB 1M17 Bivalent ATP-hinge 7-azaindole + extended C-helix allosteric back-pocket CF3-aryl pair (~6.2 A channel)".to_string(),
        ),
        FragmentPairPreset::CustomSmilesSubpockets => (
            cfg.custom_frag_a_smiles.trim(),
            "Custom Fragment A (Primary Hotspot)",
            [-2.40, -0.30, 0.15],
            cfg.custom_frag_b_smiles.trim(),
            "Custom Fragment B (Secondary Hotspot)",
            [2.50, 0.35, -0.15],
            format!(
                "Custom SMILES pair docked into non-overlapping primary & secondary hotspots of PDB {}",
                pocket.pdb_id
            ),
        ),
    };

    let mut mol_a = parse_smiles(smi_a)
        .or_else(|_| parse_smiles("COc1cc2ncnc(Cl)c2cc1OC"))
        .expect("Valid Frag A SMILES");
    mol_a.id = "FRAG-A".to_string();
    mol_a.name = name_a.to_string();
    mol_a.target = target_name.clone();
    compute_2d_coords(&mut mol_a);

    let mut mol_b = parse_smiles(smi_b)
        .or_else(|_| parse_smiles("Brc1cccc(C#C)c1"))
        .expect("Valid Frag B SMILES");
    mol_b.id = "FRAG-B".to_string();
    mol_b.name = name_b.to_string();
    mol_b.target = target_name.clone();
    compute_2d_coords(&mut mol_b);

    let prof_a = evaluate_3d_summary_profile(&mol_a, true);
    let prof_b = evaluate_3d_summary_profile(&mol_b, true);

    // Anchor sub-pockets around `pocket.center` (or furthest pharmacophore hotspot pair when available)
    let center = pocket.center;
    let center_a = [
        center[0] + offset_a[0],
        center[1] + offset_a[1],
        center[2] + offset_a[2],
    ];
    let center_b = [
        center[0] + offset_b[0],
        center[1] + offset_b[1],
        center[2] + offset_b[2],
    ];

    let coords_a = orient_fragment_exit_toward(&mol_a, &prof_a.coords3d, center_a, center_b);
    let coords_b = orient_fragment_exit_toward(&mol_b, &prof_b.coords3d, center_b, center_a);

    (mol_a, coords_a, mol_b, coords_b, pocket, target_name, desc)
}

/// Removes an optional poised halide atom from a fragment, returning `(stripped_mol, stripped_coords3d, new_anchor_idx, core_old_indices)`.
fn strip_poised_halide(
    frag: &Molecule,
    coords: &[[f32; 3]],
    anchor_atom: usize,
    displaced_halide: Option<usize>,
) -> Option<(Molecule, Vec<[f32; 3]>, usize, Vec<usize>)> {
    let mut old_to_new = vec![usize::MAX; frag.atoms.len()];
    let mut core_old = Vec::new();
    let mut out = Molecule::new("");
    let mut out_coords = Vec::new();

    for (oi, atom) in frag.atoms.iter().enumerate() {
        if Some(oi) == displaced_halide {
            continue;
        }
        let ni = out.atoms.len();
        old_to_new[oi] = ni;
        let mut a = atom.clone();
        a.idx = ni;
        out.atoms.push(a);
        if oi < coords.len() {
            out_coords.push(coords[oi]);
            core_old.push(oi);
        }
    }
    let new_anchor = *old_to_new.get(anchor_atom)?;
    if new_anchor == usize::MAX {
        return None;
    }

    for b in &frag.bonds {
        let na = old_to_new[b.a];
        let nb = old_to_new[b.b];
        if na != usize::MAX && nb != usize::MAX {
            out.bonds.push(Bond {
                idx: out.bonds.len(),
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: b.in_ring,
                kekule_order: b.kekule_order,
            });
        }
    }
    out.sanitize();
    // Verify anchor has at least 1 implicit/explicit H available to form the new covalent link
    if out.atoms[new_anchor].total_h() == 0 {
        return None;
    }
    Some((out, out_coords, new_anchor, core_old))
}

/// Assembles `Frag A + LinkerTemplate + Frag B` into a sanitized `Molecule` while tracking exact
/// atom-index provenance `(frag_a_atoms, linker_atoms, frag_b_atoms)`.
fn splice_fragments_with_linker(
    core_a: &Molecule,
    anchor_a: usize,
    core_b: &Molecule,
    anchor_b: usize,
    tmpl: &LinkerTemplate,
) -> Option<(Molecule, Vec<usize>, Vec<usize>, Vec<usize>)> {
    let mut combined = Molecule::new("");
    let mut frag_a_atoms = Vec::with_capacity(core_a.atoms.len());
    let mut linker_atoms = Vec::new();
    let mut frag_b_atoms = Vec::with_capacity(core_b.atoms.len());

    // 1. Copy Core A
    for a in &core_a.atoms {
        let idx = combined.atoms.len();
        let mut ca = a.clone();
        ca.idx = idx;
        combined.atoms.push(ca);
        frag_a_atoms.push(idx);
    }
    for b in &core_a.bonds {
        combined.bonds.push(Bond {
            idx: combined.bonds.len(),
            a: b.a,
            b: b.b,
            order: b.order,
            aromatic: b.aromatic,
            in_ring: b.in_ring,
            kekule_order: b.kekule_order,
        });
    }

    let is_het_a = core_a.atoms[anchor_a].is_hetero();
    let is_het_b = core_b.atoms[anchor_b].is_hetero();

    if let Some(dummy_smi) = tmpl.dummy_smiles {
        let l_mol = parse_smiles(dummy_smi).ok()?;
        let cl_idx = l_mol.atoms.iter().position(|a| a.elem == Element::Cl)?;
        let br_idx = l_mol.atoms.iter().position(|a| a.elem == Element::Br)?;
        let &(left_l_old, _) = l_mol.adj[cl_idx].first()?;
        let &(right_l_old, _) = l_mol.adj[br_idx].first()?;

        // Reject reactive heteroatom-heteroatom single bonds (N-N, N-O, O-O, S-O)
        if is_het_a && l_mol.atoms[left_l_old].is_hetero() {
            return None;
        }
        if is_het_b && l_mol.atoms[right_l_old].is_hetero() {
            return None;
        }

        let mut l_map = vec![usize::MAX; l_mol.atoms.len()];
        for (oi, a) in l_mol.atoms.iter().enumerate() {
            if oi == cl_idx || oi == br_idx {
                continue;
            }
            let ni = combined.atoms.len();
            l_map[oi] = ni;
            let mut ca = a.clone();
            ca.idx = ni;
            combined.atoms.push(ca);
            linker_atoms.push(ni);
        }
        for b in &l_mol.bonds {
            let na = l_map[b.a];
            let nb = l_map[b.b];
            if na != usize::MAX && nb != usize::MAX {
                combined.bonds.push(Bond {
                    idx: combined.bonds.len(),
                    a: na,
                    b: nb,
                    order: b.order,
                    aromatic: b.aromatic,
                    in_ring: b.in_ring,
                    kekule_order: b.kekule_order,
                });
            }
        }

        let left_new = l_map[left_l_old];
        let right_new = l_map[right_l_old];

        // 2. Copy Core B
        let offset_b = combined.atoms.len();
        for a in &core_b.atoms {
            let idx = combined.atoms.len();
            let mut ca = a.clone();
            ca.idx = idx;
            combined.atoms.push(ca);
            frag_b_atoms.push(idx);
        }
        for b in &core_b.bonds {
            combined.bonds.push(Bond {
                idx: combined.bonds.len(),
                a: b.a + offset_b,
                b: b.b + offset_b,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: b.in_ring,
                kekule_order: b.kekule_order,
            });
        }

        // Decrement explicit_h on bracketed hetero NH if needed so sanitize preserves neutral valence
        for &att in &[anchor_a, left_new, right_new, anchor_b + offset_b] {
            if combined.atoms[att].explicit_h > 0 {
                combined.atoms[att].explicit_h -= 1;
            }
        }

        combined.bonds.push(Bond {
            idx: combined.bonds.len(),
            a: anchor_a,
            b: left_new,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
        combined.bonds.push(Bond {
            idx: combined.bonds.len(),
            a: right_new,
            b: anchor_b + offset_b,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
    } else {
        // Direct 0-atom covalent bond A-B
        if is_het_a && is_het_b {
            return None;
        }
        let offset_b = combined.atoms.len();
        for a in &core_b.atoms {
            let idx = combined.atoms.len();
            let mut ca = a.clone();
            ca.idx = idx;
            combined.atoms.push(ca);
            frag_b_atoms.push(idx);
        }
        for b in &core_b.bonds {
            combined.bonds.push(Bond {
                idx: combined.bonds.len(),
                a: b.a + offset_b,
                b: b.b + offset_b,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: b.in_ring,
                kekule_order: b.kekule_order,
            });
        }
        for &att in &[anchor_a, anchor_b + offset_b] {
            if combined.atoms[att].explicit_h > 0 {
                combined.atoms[att].explicit_h -= 1;
            }
        }
        combined.bonds.push(Bond {
            idx: combined.bonds.len(),
            a: anchor_a,
            b: anchor_b + offset_b,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
    }

    combined.sanitize();
    compute_2d_coords(&mut combined);
    let can = molecule_to_smiles(&combined);
    if can.is_empty() || can.contains('.') {
        return None;
    }
    // Validate clean SMILES round-trip
    if parse_smiles(&can).is_err() {
        return None;
    }
    combined.smiles = can;
    Some((combined, frag_a_atoms, linker_atoms, frag_b_atoms))
}

/// Aligns a 3D conformer of `combined` onto the fixed sub-pocket coordinates `(ref_coords_a, ref_coords_b)`
/// using simultaneous dual-core Kabsch superposition, returning `(aligned_all_coords, rmsd_a, rmsd_b, rmsd_ab)`.
fn align_conformer_to_subpocket_cores(
    conf_coords: &[[f32; 3]],
    frag_a_atoms: &[usize],
    ref_coords_a: &[[f32; 3]],
    frag_b_atoms: &[usize],
    ref_coords_b: &[[f32; 3]],
) -> (Vec<[f32; 3]>, f64, f64, f64) {
    let n_a = frag_a_atoms.len().min(ref_coords_a.len());
    let n_b = frag_b_atoms.len().min(ref_coords_b.len());
    if n_a == 0 || n_b == 0 {
        return (conf_coords.to_vec(), 2.5, 2.5, 2.5);
    }

    let mut ref_dual = Vec::with_capacity(n_a + n_b);
    let mut mov_dual = Vec::with_capacity(n_a + n_b);
    for i in 0..n_a {
        ref_dual.push(ref_coords_a[i]);
        mov_dual.push(conf_coords[frag_a_atoms[i]]);
    }
    for i in 0..n_b {
        ref_dual.push(ref_coords_b[i]);
        mov_dual.push(conf_coords[frag_b_atoms[i]]);
    }

    // Compute rigid transform `(R, t)` via Kabsch on `(ref_dual, mov_dual)` and apply to all atoms of `conf_coords`
    let aligned_dual = kabsch_align_3d(&ref_dual, &mov_dual);
    let aligned_all = apply_rigid_transform_from_anchors(&mov_dual, &aligned_dual, conf_coords);

    let mut sq_a = 0.0f64;
    for i in 0..n_a {
        let d = dist3(aligned_all[frag_a_atoms[i]], ref_coords_a[i]);
        sq_a += d * d;
    }
    let mut sq_b = 0.0f64;
    for i in 0..n_b {
        let d = dist3(aligned_all[frag_b_atoms[i]], ref_coords_b[i]);
        sq_b += d * d;
    }
    let rmsd_a = (sq_a / (n_a as f64)).sqrt();
    let rmsd_b = (sq_b / (n_b as f64)).sqrt();
    let rmsd_ab = ((sq_a + sq_b) / ((n_a + n_b) as f64)).sqrt();
    (aligned_all, rmsd_a, rmsd_b, rmsd_ab)
}

/// Recovers the exact orthogonal rotation + translation mapping `src_anchors -> dst_anchors` and applies it to `all_pts`.
fn apply_rigid_transform_from_anchors(
    src_anchors: &[[f32; 3]],
    dst_anchors: &[[f32; 3]],
    all_pts: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let m = src_anchors.len().min(dst_anchors.len());
    if m < 3 {
        return all_pts.to_vec();
    }
    let inv = 1.0 / (m as f64);
    let mut cs = [0.0f64; 3];
    let mut cd = [0.0f64; 3];
    for i in 0..m {
        for k in 0..3 {
            cs[k] += (src_anchors[i][k] as f64) * inv;
            cd[k] += (dst_anchors[i][k] as f64) * inv;
        }
    }
    // Solve 3x3 least-squares rotation R * (s - cs) = (d - cd) via Gram-Schmidt orthonormalization of 2 widest chords
    let mut u1_s = [0.0f64; 3];
    let mut u1_d = [0.0f64; 3];
    let mut best_len = 0.0f64;
    for i in 0..m {
        let vs = [
            (src_anchors[i][0] as f64) - cs[0],
            (src_anchors[i][1] as f64) - cs[1],
            (src_anchors[i][2] as f64) - cs[2],
        ];
        let l = (vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2]).sqrt();
        if l > best_len {
            best_len = l;
            u1_s = [vs[0] / l, vs[1] / l, vs[2] / l];
            let vd = [
                (dst_anchors[i][0] as f64) - cd[0],
                (dst_anchors[i][1] as f64) - cd[1],
                (dst_anchors[i][2] as f64) - cd[2],
            ];
            u1_d = norm3(vd);
        }
    }

    let mut u2_s = [0.0f64; 3];
    let mut u2_d = [0.0f64; 3];
    let mut best_perp = 0.0f64;
    for i in 0..m {
        let vs = [
            (src_anchors[i][0] as f64) - cs[0],
            (src_anchors[i][1] as f64) - cs[1],
            (src_anchors[i][2] as f64) - cs[2],
        ];
        let pr_s = dot3(vs, u1_s);
        let perp_s = [
            vs[0] - pr_s * u1_s[0],
            vs[1] - pr_s * u1_s[1],
            vs[2] - pr_s * u1_s[2],
        ];
        let lp = (perp_s[0] * perp_s[0] + perp_s[1] * perp_s[1] + perp_s[2] * perp_s[2]).sqrt();
        if lp > best_perp {
            best_perp = lp;
            u2_s = [perp_s[0] / lp, perp_s[1] / lp, perp_s[2] / lp];
            let vd = [
                (dst_anchors[i][0] as f64) - cd[0],
                (dst_anchors[i][1] as f64) - cd[1],
                (dst_anchors[i][2] as f64) - cd[2],
            ];
            let pr_d = dot3(vd, u1_d);
            u2_d = norm3([
                vd[0] - pr_d * u1_d[0],
                vd[1] - pr_d * u1_d[1],
                vd[2] - pr_d * u1_d[2],
            ]);
        }
    }

    if best_len < 1e-5 || best_perp < 1e-5 {
        return all_pts
            .iter()
            .map(|p| {
                [
                    ((p[0] as f64) - cs[0] + cd[0]) as f32,
                    ((p[1] as f64) - cs[1] + cd[1]) as f32,
                    ((p[2] as f64) - cs[2] + cd[2]) as f32,
                ]
            })
            .collect();
    }

    let u3_s = cross3(u1_s, u2_s);
    let u3_d = cross3(u1_d, u2_d);

    all_pts
        .iter()
        .map(|p| {
            let rel = [
                (p[0] as f64) - cs[0],
                (p[1] as f64) - cs[1],
                (p[2] as f64) - cs[2],
            ];
            let c1 = dot3(rel, u1_s);
            let c2 = dot3(rel, u2_s);
            let c3 = dot3(rel, u3_s);
            [
                (cd[0] + c1 * u1_d[0] + c2 * u2_d[0] + c3 * u3_d[0]) as f32,
                (cd[1] + c1 * u1_d[1] + c2 * u2_d[1] + c3 * u3_d[1]) as f32,
                (cd[2] + c1 * u1_d[2] + c2 * u2_d[2] + c3 * u3_d[2]) as f32,
            ]
        })
        .collect()
}

/// Runs 3D exit-vector alignment, `18`-scaffold linker geometry matching, `Kabsch` dual-core `RMSD`
/// conformer pose scoring, and `Jencks` / `Murray-Verdonk` super-additivity `Delta-G` ranking.
pub fn design_linked_fragments(
    cfg: &FragmentLinkerConfig,
    seed_lead: Option<&Molecule>,
    ds: Option<&DatasetAnalysis>,
) -> FragmentLinkingReport {
    let (frag_a, coords_a, frag_b, coords_b, pocket, target_name, pair_desc) =
        prepare_subpocket_fragment_pair(cfg, seed_lead);

    // Score unlinked sub-pocket fragments A and B in the 3D receptor cavity
    // (stripping any synthetic halide cap first so intrinsic fragment binding dG_A & dG_B are unperturbed!)
    let vec_pairs = rank_exit_vector_pairs_3d(&frag_a, &coords_a, &frag_b, &coords_b);
    let active_pair_idx = cfg
        .vector_pair_rank
        .min(vec_pairs.len().saturating_sub(1));

    let Some(active_geom) = vec_pairs.get(active_pair_idx).cloned() else {
        let ro3_a = evaluate_rule_of_three_audit(&frag_a);
        let ro3_b = evaluate_rule_of_three_audit(&frag_b);
        return FragmentLinkingReport {
            preset: cfg.preset,
            target_pdb_id: pocket.pdb_id.clone(),
            target_name,
            pair_description: pair_desc,
            frag_a_mol: frag_a,
            frag_a_coords3d: coords_a,
            frag_a_ro3: ro3_a,
            dg_frag_a_kcal: -4.2,
            pkd_frag_a: 3.08,
            frag_b_mol: frag_b,
            frag_b_coords3d: coords_b,
            frag_b_ro3: ro3_b,
            dg_frag_b_kcal: -3.8,
            pkd_frag_b: 2.79,
            vector_pairs: vec_pairs,
            active_pair_idx: 0,
            candidates: Vec::new(),
            pocket,
        };
    };

    let stripped_a = strip_poised_halide(
        &frag_a,
        &coords_a,
        active_geom.vec_a.anchor_atom,
        active_geom.vec_a.displaced_halide_atom,
    );
    let stripped_b = strip_poised_halide(
        &frag_b,
        &coords_b,
        active_geom.vec_b.anchor_atom,
        active_geom.vec_b.displaced_halide_atom,
    );

    let (Some((core_a, core_coords_a, anch_a, _)), Some((core_b, core_coords_b, anch_b, _))) =
        (stripped_a, stripped_b)
    else {
        let ro3_a = evaluate_rule_of_three_audit(&frag_a);
        let ro3_b = evaluate_rule_of_three_audit(&frag_b);
        return FragmentLinkingReport {
            preset: cfg.preset,
            target_pdb_id: pocket.pdb_id.clone(),
            target_name,
            pair_description: pair_desc,
            frag_a_mol: frag_a,
            frag_a_coords3d: coords_a,
            frag_a_ro3: ro3_a,
            dg_frag_a_kcal: -4.2,
            pkd_frag_a: 3.08,
            frag_b_mol: frag_b,
            frag_b_coords3d: coords_b,
            frag_b_ro3: ro3_b,
            dg_frag_b_kcal: -3.8,
            pkd_frag_b: 2.79,
            vector_pairs: vec_pairs,
            active_pair_idx,
            candidates: Vec::new(),
            pocket,
        };
    };

    // Intrinsic sub-pocket binding affinities dG_A and dG_B (typical FBDD fragments bind at -3.4 .. -6.2 kcal/mol, Kd ~ 30 uM .. 3 mM)
    let p_score_a = score_ligand_in_pocket(&core_a, None, &pocket);
    let p_score_b = score_ligand_in_pocket(&core_b, None, &pocket);
    let dg_frag_a_kcal = (p_score_a.total_dg_pocket_kcal * 0.68).clamp(-6.40, -3.10);
    let dg_frag_b_kcal = (p_score_b.total_dg_pocket_kcal * 0.68).clamp(-6.10, -2.90);
    let pkd_frag_a = (-dg_frag_a_kcal / RT_LN10_KCAL).max(2.0);
    let pkd_frag_b = (-dg_frag_b_kcal / RT_LN10_KCAL).max(2.0);

    let mut core_a_eval = core_a.clone();
    core_a_eval.pact = pkd_frag_a;
    let mut core_b_eval = core_b.clone();
    core_b_eval.pact = pkd_frag_b;
    let ro3_a = evaluate_rule_of_three_audit(&core_a_eval);
    let ro3_b = evaluate_rule_of_three_audit(&core_b_eval);

    let pose_src = if target_name.contains("DRD") {
        BioactivePoseSource::PocketDrd26cm4
    } else if target_name.contains("COX") {
        BioactivePoseSource::PocketCox23ln1
    } else {
        BioactivePoseSource::PocketEgfr1m17
    };

    // Parallel evaluation across all 18 linker templates
    let mut cands: Vec<LinkedFragmentCandidate> = LINKER_TEMPLATES
        .par_iter()
        .enumerate()
        .filter_map(|(ti, tmpl)| {
            let (mut linked_mol, frag_a_atoms, linker_atoms, frag_b_atoms) =
                splice_fragments_with_linker(&core_a, anch_a, &core_b, anch_b, tmpl)?;
            linked_mol.id = format!("FLNK-{:02}", ti + 1);
            linked_mol.name = format!("Linked {}", tmpl.name);
            linked_mol.target = target_name.clone();

            // 1. DeLinker 4-DOF Geometric Match Score
            let d_obs = active_geom.anchor_distance_a;
            let span_mismatch = if d_obs < tmpl.span_min_a {
                tmpl.span_min_a - d_obs
            } else if d_obs > tmpl.span_max_a {
                d_obs - tmpl.span_max_a
            } else {
                (d_obs - tmpl.span_opt_a).abs() * 0.32
            };
            let ang_err_a = (active_geom.alpha_a_deg - tmpl.ideal_alpha_a_deg).abs();
            let ang_err_b = (active_geom.alpha_b_deg - tmpl.ideal_alpha_b_deg).abs();
            let angle_mismatch = 0.5 * (ang_err_a + ang_err_b);
            let phi_pen = tmpl
                .ideal_phi_deg
                .map(|ip| {
                    let raw = (active_geom.dihedral_phi_deg - ip).abs() % 360.0;
                    let d_phi = raw.min(360.0 - raw).min((raw - 180.0).abs());
                    (d_phi / 90.0) * 6.0
                })
                .unwrap_or(0.0);

            let geom_score = (100.0
                - span_mismatch * 26.0
                - (angle_mismatch / tmpl.alpha_tol_deg.max(15.0)) * 22.0
                - phi_pen)
                .clamp(0.0, 100.0);

            // 2. Multi-conformer 3D ensemble -> select conformer minimizing Dual-Core Kabsch RMSD_Fragments
            let ens = generate_3d_conformer_ensemble(
                &linked_mol,
                ConformerGenConfig {
                    num_trial_conformers: 8,
                    max_kept_conformers: 5,
                    mmff_max_iters: 90,
                    rmsd_prune_cutoff: 0.35,
                    energy_window_kcal: 14.0,
                    random_seed: 0xF2A6_3200_u64.wrapping_add(ti as u64 * 131),
                },
            );

            let mut best_aligned: Vec<[f32; 3]> = Vec::new();
            let mut best_rmsd_a = 2.50f64;
            let mut best_rmsd_b = 2.50f64;
            let mut best_rmsd_ab = 2.50f64;
            let mut chosen_rel_e = 0.0f64;

            for conf in &ens.conformers {
                let (aligned, ra, rb, rab) = align_conformer_to_subpocket_cores(
                    &conf.coords3d,
                    &frag_a_atoms,
                    &core_coords_a,
                    &frag_b_atoms,
                    &core_coords_b,
                );
                // Penalize conformers whose span forces excessive internal energy or core shift
                let effective_rab = rab + span_mismatch * 0.24 + conf.rel_energy_kcal * 0.04;
                if best_aligned.is_empty() || effective_rab < best_rmsd_ab {
                    best_aligned = aligned;
                    best_rmsd_a = ra + span_mismatch * 0.18;
                    best_rmsd_b = rb + span_mismatch * 0.18;
                    best_rmsd_ab = effective_rab;
                    chosen_rel_e = conf.rel_energy_kcal;
                }
            }
            if best_aligned.is_empty() {
                let prof3 = evaluate_3d_summary_profile(&linked_mol, true);
                let (aligned, ra, rb, rab) = align_conformer_to_subpocket_cores(
                    &prof3.coords3d,
                    &frag_a_atoms,
                    &core_coords_a,
                    &frag_b_atoms,
                    &core_coords_b,
                );
                best_aligned = aligned;
                best_rmsd_a = ra + span_mismatch * 0.20;
                best_rmsd_b = rb + span_mismatch * 0.20;
                best_rmsd_ab = rab + span_mismatch * 0.22;
            }

            let pose_preserved = best_rmsd_ab <= DELINKER_POSE_PRESERVED_RMSD_A;

            // 3. Pocket scoring & CSD/SMARTS Torsion Strain on linked molecule
            let p_linked = score_ligand_in_pocket(&linked_mol, ds, &pocket);
            let tor_rep = analyze_torsion_strain_and_locks(&linked_mol, ds, pose_src, false);

            // 4. Jencks (1981) + Murray & Verdonk (2002) Super-Additivity Free-Energy Cycle
            let dg_additive_sum = dg_frag_a_kcal + dg_frag_b_kcal;

            // Pose-preservation factor attenuates saved rigid-body entropy when linker pulls cores out of hotspots
            let f_pose = (-(best_rmsd_ab / 1.15).powi(2)).exp().clamp(0.05, 1.0);
            let dg_rigid_saved = -MURRAY_VERDONK_RIGID_ENTROPY_KCAL * f_pose;

            // Linker conformational strain + rotor freezing + span/angle pose-distortion penalty (>= 0 kcal/mol)
            let rotor_cost = (tmpl.added_rotatable_bonds as f64) * 0.38;
            let teu_cost = (tor_rep.total_strain_teu * 0.22).min(2.60);
            let pose_distort_cost =
                span_mismatch * 1.15 + (best_rmsd_ab - 0.35).max(0.0).powi(2) * 1.35 + chosen_rel_e * 0.20;
            let dg_linker_strain =
                tmpl.intrinsic_strain_floor_kcal + rotor_cost + teu_cost + pose_distort_cost;

            // Direct linker–pocket contact & WaterMap displacement term in the bridging channel
            let raw_channel_delta =
                (p_linked.total_dg_pocket_kcal - (p_score_a.total_dg_pocket_kcal + p_score_b.total_dg_pocket_kcal) * 0.58)
                    * 0.28;
            let hbond_channel_bonus = match tmpl.class {
                LinkerScaffoldClass::AmideOrSulfonamide => -0.45,
                LinkerScaffoldClass::RigidUnsaturatedOrHeterocycle => -0.35,
                LinkerScaffoldClass::Sp3ConformationalCage => -0.30,
                LinkerScaffoldClass::HeteroatomOrAlkylChain => -0.18,
                LinkerScaffoldClass::DirectBondFusion => -0.10,
            };
            let clash_pen = (p_linked.clash_atom_count as f64) * 0.55;
            let dg_linker_pocket =
                (raw_channel_delta + hbond_channel_bonus + clash_pen).clamp(-1.85, 2.40);

            let dg_link_term = dg_rigid_saved + dg_linker_strain + dg_linker_pocket;
            let dg_linked_ab = dg_additive_sum + dg_link_term;
            let ddg_super = dg_link_term; // = dg_linked_ab - (dg_frag_a_kcal + dg_frag_b_kcal)

            let predicted_pkd = (-dg_linked_ab / RT_LN10_KCAL).clamp(3.0, 12.5);
            linked_mol.pact = predicted_pkd;
            linked_mol.raw_nm = 10f64.powf(9.0 - predicted_pkd);

            let cooperativity_fold = 10f64.powf((-ddg_super / RT_LN10_KCAL).clamp(-4.0, 5.0));
            // Ichihara (2011) linking coefficient E = K_AB / (K_A * K_B) = 10^(pKd_A + pKd_B - pKd_AB) M^-1
            let ichihara_e = 10f64.powf((pkd_frag_a + pkd_frag_b - predicted_pkd).clamp(-5.0, 5.0));

            let super_tier = if ddg_super <= -1.36 && best_rmsd_ab <= 0.95 {
                SuperAdditivityTier::StrongSuperAdditive
            } else if ddg_super <= -0.25 {
                SuperAdditivityTier::ModerateSuperAdditive
            } else if ddg_super <= 0.85 && best_rmsd_ab <= 1.35 {
                SuperAdditivityTier::NearAdditive
            } else {
                SuperAdditivityTier::SubAdditiveStrained
            };

            let alerts = scan_structural_alerts(&linked_mol);
            let physchem = evaluate_physchem(&linked_mol, alerts.len());
            let pka = evaluate_physiological_pka_74(&linked_mol);
            let sa = evaluate_synthetic_accessibility(&linked_mol);

            let hac = physchem.heavy_atoms.max(1) as f64;
            let le = -dg_linked_ab / hac;
            let lle = predicted_pkd - physchem.clogp;
            let lle_74 = predicted_pkd - pka.logd_74;
            let linker_hac = linker_atoms.len().max(1) as f64;
            let linker_ge = (-dg_link_term) / linker_hac;

            let composite_score = (geom_score * 0.28
                + (-ddg_super + 1.5).clamp(0.0, 4.5) * 9.5
                + (1.5 - best_rmsd_ab).max(0.0) * 14.0
                + lle.clamp(0.0, 7.0) * 2.4
                + physchem.qed_mean * 10.0
                - (sa.sa_score - 3.2).max(0.0) * 3.5
                - (alerts.len() as f64) * 12.0)
                .clamp(0.0, 100.0);

            Some(LinkedFragmentCandidate {
                rank: 0,
                candidate_id: linked_mol.id.clone(),
                linker_code: tmpl.code,
                linker_name: tmpl.name,
                linker_class: tmpl.class,
                coupling_reaction: tmpl.coupling_reaction,
                canonical_smiles: linked_mol.smiles.clone(),
                geometry_match_score: geom_score,
                span_mismatch_a: span_mismatch,
                angle_mismatch_deg: angle_mismatch,
                rmsd_frag_a_a: best_rmsd_a,
                rmsd_frag_b_a: best_rmsd_b,
                fragment_core_rmsd_a: best_rmsd_ab,
                pose_preserved,
                dg_frag_a_kcal,
                dg_frag_b_kcal,
                dg_additive_sum_kcal: dg_additive_sum,
                dg_rigid_entropy_saved_kcal: dg_rigid_saved,
                dg_linker_strain_kcal: dg_linker_strain,
                dg_linker_pocket_kcal: dg_linker_pocket,
                dg_link_term_kcal: dg_link_term,
                dg_linked_ab_kcal: dg_linked_ab,
                ddg_superadditivity_kcal: ddg_super,
                cooperativity_fold,
                ichihara_linking_coeff_e: ichihara_e,
                superadditivity_tier: super_tier,
                predicted_pkd,
                le,
                lle,
                lle_74,
                linker_ge,
                total_strain_teu: tor_rep.total_strain_teu,
                sa_score: sa.sa_score,
                pains_alerts: alerts.len(),
                physchem,
                composite_linker_score: composite_score,
                frag_a_atoms,
                linker_atoms,
                frag_b_atoms,
                aligned_coords3d: best_aligned,
                mol: linked_mol,
            })
        })
        .collect();

    cands.sort_by(|a, b| {
        b.composite_linker_score
            .partial_cmp(&a.composite_linker_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, c) in cands.iter_mut().enumerate() {
        c.rank = idx + 1;
        c.candidate_id = format!("FLNK-{:02}", idx + 1);
        c.mol.id = c.candidate_id.clone();
    }

    FragmentLinkingReport {
        preset: cfg.preset,
        target_pdb_id: pocket.pdb_id.clone(),
        target_name,
        pair_description: pair_desc,
        frag_a_mol: core_a_eval,
        frag_a_coords3d: core_coords_a,
        frag_a_ro3: ro3_a,
        dg_frag_a_kcal,
        pkd_frag_a,
        frag_b_mol: core_b_eval,
        frag_b_coords3d: core_coords_b,
        frag_b_ro3: ro3_b,
        dg_frag_b_kcal,
        pkd_frag_b,
        vector_pairs: vec_pairs,
        active_pair_idx,
        candidates: cands,
        pocket,
    }
}

/// Exports the ranked `18`-linker super-additivity leaderboard as a `.csv` table with summary statistics.
pub fn export_linked_fragments_csv(rep: &FragmentLinkingReport) -> String {
    let mut out = String::from(
        "rank,candidate_id,linker_code,linker_name,linker_class,coupling_reaction,geom_match_score,span_mismatch_a,angle_mismatch_deg,core_rmsd_a,pose_preserved,dg_frag_a_kcal,dg_frag_b_kcal,dg_additive_sum_kcal,dg_rigid_saved_kcal,dg_linker_strain_kcal,dg_linker_pocket_kcal,dg_linked_ab_kcal,ddg_superadditivity_kcal,cooperativity_fold,ichihara_e_m_inv,superadditivity_tier,predicted_pkd,le,lle,linker_ge,total_strain_teu,mw,clogp,fsp3,qed,sa_score,composite_score,smiles\n",
    );
    let mut super_cnt = 0usize;
    let mut pose_cnt = 0usize;
    for c in &rep.candidates {
        if c.ddg_superadditivity_kcal <= -0.25 {
            super_cnt += 1;
        }
        if c.pose_preserved {
            pose_cnt += 1;
        }
        out.push_str(&format!(
            "{},{},{},\"{}\",\"{}\",\"{}\",{:.1},{:.2},{:.1},{:.3},{},{:+.2},{:+.2},{:+.2},{:+.2},{:+.2},{:+.2},{:+.2},{:+.2},{:.2},{:.3e},\"{}\",{:.2},{:.3},{:.2},{:+.3},{:.2},{:.1},{:.2},{:.2},{:.3},{:.2},{:.1},{}\n",
            c.rank,
            c.candidate_id,
            c.linker_code,
            c.linker_name,
            c.linker_class.short_label(),
            c.coupling_reaction,
            c.geometry_match_score,
            c.span_mismatch_a,
            c.angle_mismatch_deg,
            c.fragment_core_rmsd_a,
            c.pose_preserved,
            c.dg_frag_a_kcal,
            c.dg_frag_b_kcal,
            c.dg_additive_sum_kcal,
            c.dg_rigid_entropy_saved_kcal,
            c.dg_linker_strain_kcal,
            c.dg_linker_pocket_kcal,
            c.dg_linked_ab_kcal,
            c.ddg_superadditivity_kcal,
            c.cooperativity_fold,
            c.ichihara_linking_coeff_e,
            c.superadditivity_tier.short_label(),
            c.predicted_pkd,
            c.le,
            c.lle,
            c.linker_ge,
            c.total_strain_teu,
            c.physchem.mw,
            c.physchem.clogp,
            c.physchem.fsp3,
            c.physchem.qed_mean,
            c.sa_score,
            c.composite_linker_score,
            c.canonical_smiles
        ));
    }
    out.push_str(&format!(
        "# SUMMARY,total_evaluated={},super_additive_count={},pose_preserved_lt_0_80a={},target_pdb={}\n",
        rep.candidates.len(),
        super_cnt,
        pose_cnt,
        rep.target_pdb_id
    ));
    out
}

/// Exports 3D receptor-aligned poses of unlinked fragments `A`, `B`, and all linked `A-L-B` chimeras to `V2000` `.sdf`.
pub fn format_linked_fragments_sdf(rep: &FragmentLinkingReport) -> String {
    let mut sdf = String::new();
    let chg_a = compute_gasteiger_charges(&rep.frag_a_mol);
    sdf.push_str(&format_3d_conformer_sdf(
        &rep.frag_a_mol,
        &rep.frag_a_coords3d,
        &chg_a,
        rep.dg_frag_a_kcal,
        &[
            ("ROLE", "SUBPOCKET_FRAGMENT_A".to_string()),
            ("DG_FRAG_KCAL", format!("{:+.2}", rep.dg_frag_a_kcal)),
            ("PKD_FRAG", format!("{:.2}", rep.pkd_frag_a)),
        ],
    ));
    let chg_b = compute_gasteiger_charges(&rep.frag_b_mol);
    sdf.push_str(&format_3d_conformer_sdf(
        &rep.frag_b_mol,
        &rep.frag_b_coords3d,
        &chg_b,
        rep.dg_frag_b_kcal,
        &[
            ("ROLE", "SUBPOCKET_FRAGMENT_B".to_string()),
            ("DG_FRAG_KCAL", format!("{:+.2}", rep.dg_frag_b_kcal)),
            ("PKD_FRAG", format!("{:.2}", rep.pkd_frag_b)),
        ],
    ));

    for c in &rep.candidates {
        let chg = compute_gasteiger_charges(&c.mol);
        sdf.push_str(&format_3d_conformer_sdf(
            &c.mol,
            &c.aligned_coords3d,
            &chg,
            c.dg_linked_ab_kcal,
            &[
                ("ROLE", "LINKED_CHIMERA_AB".to_string()),
                ("LINKER_CODE", c.linker_code.to_string()),
                ("LINKER_NAME", c.linker_name.to_string()),
                ("CORE_RMSD_A", format!("{:.3}", c.fragment_core_rmsd_a)),
                ("DG_LINKED_AB_KCAL", format!("{:+.2}", c.dg_linked_ab_kcal)),
                (
                    "DDG_SUPERADDITIVITY_KCAL",
                    format!("{:+.2}", c.ddg_superadditivity_kcal),
                ),
                ("PREDICTED_PKD", format!("{:.2}", c.predicted_pkd)),
                ("ICHIHARA_E_M_INV", format!("{:.3e}", c.ichihara_linking_coeff_e)),
            ],
        ));
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egfr_1m17_fragment_linking_rediscovers_superadditive_bridges() {
        let cfg = FragmentLinkerConfig {
            preset: FragmentPairPreset::Egfr1m17HingeAndBackPocket,
            ..FragmentLinkerConfig::default()
        };
        let rep = design_linked_fragments(&cfg, None, None);
        assert_eq!(rep.target_pdb_id, "1M17");
        assert!(!rep.vector_pairs.is_empty());
        assert!(rep.candidates.len() >= 14);
        let best = &rep.candidates[0];
        assert!(best.ddg_superadditivity_kcal < 0.0);
        assert!(best.dg_linked_ab_kcal < rep.dg_frag_a_kcal + rep.dg_frag_b_kcal);
        assert!(best.predicted_pkd > rep.pkd_frag_a && best.predicted_pkd > rep.pkd_frag_b);
    }

    #[test]
    fn test_in_situ_3d_lead_deconstruction_and_relinking() {
        let erlotinib =
            parse_smiles("COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC").expect("Valid Erlotinib SMILES");
        let cfg = FragmentLinkerConfig {
            preset: FragmentPairPreset::DeconstructSelectedLead,
            ..FragmentLinkerConfig::default()
        };
        let rep = design_linked_fragments(&cfg, Some(&erlotinib), None);
        assert!(!rep.candidates.is_empty());
        let csv = export_linked_fragments_csv(&rep);
        assert!(csv.contains("# SUMMARY,total_evaluated="));
        let sdf = format_linked_fragments_sdf(&rep);
        assert!(sdf.contains("LINKED_CHIMERA_AB"));
    }
}
