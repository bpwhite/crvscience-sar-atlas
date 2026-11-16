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

//! Hierarchical `CSD`/`SMARTS` Dihedral Energy-Profile Scanner, Bioactive 3D Conformer Strain
//! Penalty Estimator (`Gu & Shoichet` `TEU` + `Perola & Charifson` `kcal/mol`), and Automated
//! Rotatable-Bond Locking Studio (`ortho`-Methyl Steric Gearing, `1,5`/`1,6` Intramolecular H-Bond
//! Pseudo-Rings, `alpha`-Alkyl `A(1,3)` Restriction & Small-Ring Cyclization; `torsion_strain.rs`, `v2.9.0`).
//!
//! Primary References:
//! - Schärfer, Schulz-Gasch, Ehrlich, Guba, Rarey & Stahl (2013) / Guba, Meyder, Rarey & Hert (2016),
//!   *J. Chem. Inf. Model.* 56(1), 1-5 (`Torsion Library Reloaded` hierarchical `CSD`/`SMARTS` dihedral
//!   potentials: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12728936/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11523068/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC9632257/`)
//! - Gu, Smith, Yang, Irwin & Shoichet (2021), *J. Chem. Inf. Model.* 61(9), 4331-4341 ("Ligand Strain
//!   Energy in Large Library Docking" — Boltzmann `CSD` `Torsion Energy Units` `TEU`, `1.8 TEU` single-bond
//!   & `6.0 TEU` total-strain thresholds: `https://pubmed.ncbi.nlm.nih.gov/34467754/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8501943/`)
//! - Perola & Charifson (2004), *J. Med. Chem.* 47(10), 2499-2510 ("Conformational analysis of drug-like
//!   molecules bound to proteins: an extensive study of ligand reorganization upon binding":
//!   `https://pubmed.ncbi.nlm.nih.gov/15115393/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC5880911/`)
//! - Kuhn, Mohr & Stahl (2010), *J. Med. Chem.* 53(6), 2601-2611 ("Intramolecular Hydrogen Bonding in
//!   Medicinal Chemistry" — `5`- to `8`-membered `IMHB` conformational locks:
//!   `https://pubmed.ncbi.nlm.nih.gov/20175530/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC5304308/`)
//! - Schönherr & Cernak (2013) / "The Magic Methyl and Its Tricks in Drug Discovery" (`ortho`-methyl
//!   biaryl/amide conformational gearing: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10457765/`).

use crate::conformer3d::{
    align_rocs_3d, bondi_vdw_radius, compute_gasteiger_charges, evaluate_3d_summary_profile,
    format_3d_conformer_sdf, generate_3d_conformer_ensemble, ConformerGenConfig,
};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    CURATED_PDB_POCKETS,
};
use crate::sketcher::molecule_to_smiles;
use crate::smarts::{compile_smarts, match_smarts};
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Number of `10 deg` dihedral bins spanning `[0, 360)` in each `1D` torsion scan (`36 * 10 = 360 deg`).
pub const TORSION_SCAN_BINS: usize = 36;

/// Thermal energy `R * T` at `298.15 K` (`kcal/mol`).
pub const TORSION_RT_298K_KCAL: f64 = 0.592_484;

/// Gu & Shoichet (2021) single-torsion strain warning cutoff (`1.80 TEU`).
pub const SHOICHET_SINGLE_TEU_CUTOFF: f64 = 1.80;

/// Gu & Shoichet (2021) whole-molecule total torsional strain cutoff (`6.00 TEU`).
pub const SHOICHET_TOTAL_TEU_CUTOFF: f64 = 6.00;

/// Reference 3D bioactive pose environment used to measure bound vs free conformational strain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BioactivePoseSource {
    /// Align & relax ligand inside `1M17` EGFR ATP-hinge pocket.
    PocketEgfr1m17,
    /// Align & relax ligand inside `6CM4` Dopamine `D2` orthosteric pocket.
    PocketDrd26cm4,
    /// Align & relax ligand inside `3LN1` `COX-2` diarylpyrazole pocket.
    PocketCox23ln1,
    /// Auto-match receptor pocket from `mol.target` (`EGFR` / `DRD2` / `COX-2`).
    AutoTargetPocket,
    /// Superimpose onto the series' highest-potency reference lead via 3D `ROCS` Shape + Color.
    RocsReferenceOverlay,
}

impl BioactivePoseSource {
    pub const ALL: [Self; 5] = [
        Self::AutoTargetPocket,
        Self::PocketEgfr1m17,
        Self::PocketDrd26cm4,
        Self::PocketCox23ln1,
        Self::RocsReferenceOverlay,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::AutoTargetPocket => "Auto PDB Pocket",
            Self::PocketEgfr1m17 => "1M17 EGFR",
            Self::PocketDrd26cm4 => "6CM4 DRD2",
            Self::PocketCox23ln1 => "3LN1 COX-2",
            Self::RocsReferenceOverlay => "3D ROCS Lead",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AutoTargetPocket => "Auto-Matched Co-Crystal PDB Binding Pocket (Target-Aware)",
            Self::PocketEgfr1m17 => "EGFR Kinase ATP-Hinge Co-Crystal Pocket (PDB 1M17)",
            Self::PocketDrd26cm4 => "Dopamine D2 GPCR Orthosteric Pocket (PDB 6CM4)",
            Self::PocketCox23ln1 => "COX-2 Sulfonamide Side-Pocket (PDB 3LN1)",
            Self::RocsReferenceOverlay => "3D ROCS Gaussian Shape+Color Reference Lead Overlay",
        }
    }

    pub fn resolve_pocket_idx(self, target_hint: &str) -> Option<usize> {
        match self {
            Self::PocketEgfr1m17 => Some(0),
            Self::PocketDrd26cm4 => Some(1),
            Self::PocketCox23ln1 => Some(2),
            Self::AutoTargetPocket => {
                let up = target_hint.to_ascii_uppercase();
                if up.contains("DRD") || up.contains("D2") || up.contains("DOPAMINE") {
                    Some(1)
                } else if up.contains("COX") || up.contains("PTGS") {
                    Some(2)
                } else {
                    Some(0)
                }
            }
            Self::RocsReferenceOverlay => None,
        }
    }
}

/// Per-bond `TorsionAnalyzer` / `Shoichet` dihedral quality band (`Green` / `Amber` / `Red`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TorsionQualityBand {
    /// Within first `CSD` tolerance window (`<= tol_1`) and `TEU <= 1.0`.
    RelaxedGreen,
    /// Between first and second `CSD` tolerance windows (`tol_1 < dev <= tol_2`) or `1.0 < TEU <= 1.8`.
    ToleratedAmber,
    /// Outside second `CSD` tolerance window (`dev > tol_2`) or `TEU > 1.80` (strained!).
    StrainedRed,
}

impl TorsionQualityBand {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::RelaxedGreen => "Relaxed (CSD Peak)",
            Self::ToleratedAmber => "Tolerated Shoulder",
            Self::StrainedRed => "Strained (>1.8 TEU)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::RelaxedGreen => [52, 211, 153],  // Emerald
            Self::ToleratedAmber => [251, 191, 36], // Amber
            Self::StrainedRed => [248, 113, 113],   // Coral Red
        }
    }
}

/// Whole-molecule bioactive conformational strain classification (`Gu & Shoichet 2021` + `Perola & Charifson 2004`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BioactiveStrainTier {
    /// Passes both `Shoichet` filters (`Total <= 6.0 TEU` and `Max Single <= 1.8 TEU`) and `dE_strain <= 3.0 kcal/mol`.
    LowStrainBioactive,
    /// Moderate induced-fit reorganization (`Total <= 8.5 TEU`, `dE_strain <= 5.0 kcal/mol`; tolerated by strong enthalpic contacts).
    ModerateInducedStrain,
    /// Exceeds `Shoichet` (`Max > 1.8 TEU` or `Total > 6.0 TEU`) or `Perola-Charifson` (`> 5.0 kcal/mol`) strain ceiling: high decoy risk!
    HighStrainDecoyRisk,
}

impl BioactiveStrainTier {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::LowStrainBioactive => "Low-Strain Bioactive",
            Self::ModerateInducedStrain => "Moderate Induced Strain",
            Self::HighStrainDecoyRisk => "High-Strain Decoy Alert",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LowStrainBioactive => {
                "Low-Strain Bioactive Pose (Passes Shoichet <=6.0 TEU & <=1.8 Single TEU)"
            }
            Self::ModerateInducedStrain => {
                "Moderate Reorganization Strain (<=5.0 kcal/mol Perola-Charifson Band)"
            }
            Self::HighStrainDecoyRisk => {
                "High Conformational Strain (>6.0 TEU or >1.8 Single TEU — Lock or Filter!)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::LowStrainBioactive => [52, 211, 153],
            Self::ModerateInducedStrain => [251, 191, 36],
            Self::HighStrainDecoyRisk => [248, 113, 113],
        }
    }
}

/// Medicinal-chemistry tactical strategy used to lock or bias a rotatable dihedral toward its bioactive angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotorLockStrategy {
    /// `ortho`-Methyl ("Magic Methyl", `Schönherr & Cernak 2013`) steric gearing twisting `Ar-Ar` / `Ar-C(=O)` / `Ar-NH` to `65..110 deg`.
    OrthoMethylStericGear,
    /// `ortho`-Fluoro or ring-`N` swap forming a `5`- or `6`-membered intramolecular H-bond / dipolar coplanar lock (`Kuhn et al. 2010`).
    IntramolecularHBondOrFluoroLock,
    /// `alpha`-Methyl or `alpha`-fluoro `1,3`-allylic / `gauche` acyclic `sp3` rotor restriction.
    AlphaSubstituentA13OrGaucheLock,
    /// Direct covalent tether cyclizing the rotor into a `3`-, `4`-, `5`-, or `6`-membered ring (`Cyclopropyl`/`Pyrrolidine`/`Benzoxazole`/`Lactam`).
    SmallRingCovalentCyclization,
}

impl RotorLockStrategy {
    pub const ALL: [Self; 4] = [
        Self::OrthoMethylStericGear,
        Self::IntramolecularHBondOrFluoroLock,
        Self::AlphaSubstituentA13OrGaucheLock,
        Self::SmallRingCovalentCyclization,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::OrthoMethylStericGear => "Ortho-Me Steric Gear",
            Self::IntramolecularHBondOrFluoroLock => "1,5/1,6 IMHB / F-Lock",
            Self::AlphaSubstituentA13OrGaucheLock => "Alpha-Me / F-Gauche",
            Self::SmallRingCovalentCyclization => "Small-Ring Cyclization",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OrthoMethylStericGear => {
                "Ortho-Methyl Steric Gearing ('Magic Methyl' Orthogonal Twist)"
            }
            Self::IntramolecularHBondOrFluoroLock => {
                "Intramolecular H-Bond (IMHB) / Ortho-Fluoro Coplanar Pseudo-Ring"
            }
            Self::AlphaSubstituentA13OrGaucheLock => {
                "Alpha-Methyl A(1,3) or Beta-Fluoro Gauche Dipolar Bias"
            }
            Self::SmallRingCovalentCyclization => {
                "Small-Ring Covalent Cyclization (Eliminates Rotatable Bond)"
            }
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::OrthoMethylStericGear => [251, 191, 36],           // Amber
            Self::IntramolecularHBondOrFluoroLock => [56, 189, 248], // Sky Blue
            Self::AlphaSubstituentA13OrGaucheLock => [52, 211, 153], // Emerald
            Self::SmallRingCovalentCyclization => [192, 132, 252],   // Violet
        }
    }
}

/// Curated hierarchical `CSD` / `SMARTS` 4-atom dihedral rule (`Schärfer et al. 2013` / `Guba et al. 2016`).
pub struct CsdTorsionRule {
    pub rule_id: &'static str,
    pub hierarchy_class: &'static str,
    pub title: &'static str,
    /// 4-atom Daylight `SMARTS` pattern matching `(atom_0 - atom_1 - atom_2 - atom_3)` across central bond `(1, 2)`.
    pub smarts_4atom: &'static str,
    /// Empirically observed `CSD` crystal peak centers in `[0, 360)` degrees.
    pub preferred_peaks_deg: &'static [f64],
    /// First `CSD` angular tolerance (`deg`, green band).
    pub tol_1_deg: f64,
    /// Second `CSD` angular tolerance (`deg`, amber shoulder; beyond = red strained).
    pub tol_2_deg: f64,
    /// Cosine Fourier amplitudes `(V1, V2, V3, V4)` (`kcal/mol`) and phases `(gamma1..gamma4)` (`deg`)
    /// where `E_tor(phi) = sum_n 0.5 * V_n * (1 + cos(n * phi - gamma_n))`.
    pub fourier_v_kcal: [f64; 4],
    pub fourier_gamma_deg: [f64; 4],
}

/// Ordered specific-to-general `CSD`/`SMARTS` Torsion Library (`22` rules spanning all 7 `Rarey`/`Guba` classes).
pub const CSD_SMARTS_TORSION_LIBRARY: [CsdTorsionRule; 22] = [
    // --- Specific Biaryl & Heterobiaryl Rules ---
    CsdTorsionRule {
        rule_id: "CSD-BIARYL-ORTHO2",
        hierarchy_class: "Biaryl C(ar)-C(ar)",
        title: "Di-ortho-substituted biaryl (orthogonal ~90/270 deg atropisomeric twist)",
        smarts_4atom: "[cD3]c!@c[cD3]",
        preferred_peaks_deg: &[90.0, 270.0],
        tol_1_deg: 24.0,
        tol_2_deg: 42.0,
        fourier_v_kcal: [0.0, 4.60, 0.0, 0.35],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-BIARYL-HETN",
        hierarchy_class: "Biaryl C(ar)-C(ar)",
        title: "2-Pyridyl / pyrimidinyl / pyrazolyl heterobiaryl (reduced ortho CH clash, ~0/180 or ~25 deg)",
        smarts_4atom: "[n]c!@cc",
        preferred_peaks_deg: &[0.0, 180.0],
        tol_1_deg: 26.0,
        tol_2_deg: 48.0,
        fourier_v_kcal: [0.0, 2.35, 0.0, 0.45],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-BIARYL-ORTHO1",
        hierarchy_class: "Biaryl C(ar)-C(ar)",
        title: "Mono-ortho-substituted biaryl (gauche-twisted ~55/125/235/305 deg)",
        smarts_4atom: "[cD3]c!@c[cH1]",
        preferred_peaks_deg: &[55.0, 125.0, 235.0, 305.0],
        tol_1_deg: 25.0,
        tol_2_deg: 45.0,
        fourier_v_kcal: [0.0, 1.85, 0.0, 0.95],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 180.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-BIARYL-UNSUB",
        hierarchy_class: "Biaryl C(ar)-C(ar)",
        title: "Unsubstituted biphenyl-like C(ar)-C(ar) (~35/145/215/325 deg balanced conjugation/sterics)",
        smarts_4atom: "cc!@cc",
        preferred_peaks_deg: &[36.0, 144.0, 216.0, 324.0],
        tol_1_deg: 25.0,
        tol_2_deg: 48.0,
        fourier_v_kcal: [0.0, 1.95, 0.0, 1.15],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    // --- Aryl-Carbonyl, Benzamide & Amide C(=O)-N Rules ---
    CsdTorsionRule {
        rule_id: "CSD-ARAMIDE-F-IMHB",
        hierarchy_class: "Aryl-Carbonyl & IMHB",
        title: "2-Fluoro/alkoxy-benzamide (1,5 F...HN / O...HN IMHB pseudo-ring at 180 deg)",
        smarts_4atom: "[cD3]c!@[CX3]=[OX1]",
        preferred_peaks_deg: &[180.0, 0.0],
        tol_1_deg: 22.0,
        tol_2_deg: 40.0,
        fourier_v_kcal: [1.10, 3.10, 0.0, 0.0],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-AR-CARBONYL",
        hierarchy_class: "Aryl-Carbonyl & IMHB",
        title: "Aryl-carbonyl / benzamide Ar-C(=O) (conjugated coplanar ~0/180 deg, +/-25 deg)",
        smarts_4atom: "cc!@[CX3]=[OX1]",
        preferred_peaks_deg: &[0.0, 180.0],
        tol_1_deg: 26.0,
        tol_2_deg: 46.0,
        fourier_v_kcal: [0.0, 2.85, 0.0, 0.25],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-AMIDE-TRANS",
        hierarchy_class: "Amide C(=O)-N",
        title: "Secondary/tertiary amide O=C-N-C (strong partial double-bond 0/180 deg planarity)",
        smarts_4atom: "[OX1]=[CX3]!@[NX3][#6]",
        preferred_peaks_deg: &[0.0, 180.0],
        tol_1_deg: 18.0,
        tol_2_deg: 32.0,
        fourier_v_kcal: [1.45, 5.80, 0.0, 0.0],
        fourier_gamma_deg: [180.0, 180.0, 0.0, 0.0],
    },
    // --- Heteroaryl-NH & Anilino (4-Anilinoquinazoline / Kinase Hinge) Rules ---
    CsdTorsionRule {
        rule_id: "CSD-HETAR-NH-IMHB",
        hierarchy_class: "Heteroaryl-NH & Anilino",
        title: "2-Amino-azaheterocycle / 4-anilinoquinazoline n:c-NH-Ar (coplanar 0/180 deg C-H...N lock)",
        smarts_4atom: "nc!@[NX3H1][#6]",
        preferred_peaks_deg: &[0.0, 180.0],
        tol_1_deg: 24.0,
        tol_2_deg: 44.0,
        fourier_v_kcal: [0.85, 3.35, 0.0, 0.0],
        fourier_gamma_deg: [180.0, 180.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-ANILINO-AR",
        hierarchy_class: "Heteroaryl-NH & Anilino",
        title: "Anilino Ar-NH-R lone-pair conjugation (~0/180 or ~30/150 deg)",
        smarts_4atom: "cc!@[NX3][#6]",
        preferred_peaks_deg: &[22.0, 158.0, 202.0, 338.0],
        tol_1_deg: 28.0,
        tol_2_deg: 50.0,
        fourier_v_kcal: [0.0, 2.40, 0.0, 0.45],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    // --- Aryl-Ether & Alkoxy C(ar)-O Rules ---
    CsdTorsionRule {
        rule_id: "CSD-AR-OME-ORTHO",
        hierarchy_class: "Aryl-Ether C(ar)-O",
        title: "Ortho-substituted anisole / 6,7-dialkoxyquinazoline c(cX)-O-C (in-plane 0 deg away from ortho group)",
        smarts_4atom: "[cD3]c!@[OX2][CX4]",
        preferred_peaks_deg: &[180.0, 90.0, 270.0],
        tol_1_deg: 25.0,
        tol_2_deg: 48.0,
        fourier_v_kcal: [1.35, 2.25, 0.0, 0.0],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-AR-ETHER",
        hierarchy_class: "Aryl-Ether C(ar)-O",
        title: "Unencumbered aryl-alkyl / diaryl ether c-O-C (in-plane p-pi conjugation 0/180 deg)",
        smarts_4atom: "cc!@[OX2][#6]",
        preferred_peaks_deg: &[0.0, 180.0],
        tol_1_deg: 26.0,
        tol_2_deg: 50.0,
        fourier_v_kcal: [0.0, 2.65, 0.0, 0.20],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    // --- Sulfonamide & Sulfone S-C / S-N Rules ---
    CsdTorsionRule {
        rule_id: "CSD-AR-SO2",
        hierarchy_class: "Sulfonamide & Sulfone",
        title: "Aryl-sulfonamide / aryl-sulfone c-S(=O)2 (bisecting ~90/270 deg CSD peak!)",
        smarts_4atom: "cc!@[SX4;D4]=[OX1]",
        preferred_peaks_deg: &[90.0, 270.0],
        tol_1_deg: 25.0,
        tol_2_deg: 45.0,
        fourier_v_kcal: [0.0, 2.95, 0.0, 0.0],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-SO2-NH",
        hierarchy_class: "Sulfonamide & Sulfone",
        title: "Sulfonamide O=S-N-C (~65/115/245/295 deg nitrogen lone-pair eclipsing S=O)",
        smarts_4atom: "[OX1]=[SX4]!@[NX3][#6]",
        preferred_peaks_deg: &[68.0, 292.0, 180.0],
        tol_1_deg: 26.0,
        tol_2_deg: 48.0,
        fourier_v_kcal: [0.45, 1.55, 1.35, 0.0],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    // --- Fluorine & Vicinal Heteroatom Gauche Stereoelectronic Rules ---
    CsdTorsionRule {
        rule_id: "CSD-FLUORO-GAUCHE",
        hierarchy_class: "Stereoelectronic Gauche",
        title: "Beta-fluoroethyl F-C(sp3)-C(sp3)-[N,O,F] (sigma_CH -> sigma*_CF hyperconjugative +/-65 deg gauche!)",
        smarts_4atom: "F[CX4]!@[CX4][#7,#8,F]",
        preferred_peaks_deg: &[65.0, 295.0],
        tol_1_deg: 22.0,
        tol_2_deg: 40.0,
        fourier_v_kcal: [1.15, 1.40, 2.55, 0.0],
        fourier_gamma_deg: [180.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-OCCO-GAUCHE",
        hierarchy_class: "Stereoelectronic Gauche",
        title: "Alkoxyethyl / PEG O-C(sp3)-C(sp3)-[O,N] (+/-68 deg gauche & 180 deg anti)",
        smarts_4atom: "[#8][CX4]!@[CX4][#7,#8]",
        preferred_peaks_deg: &[68.0, 180.0, 292.0],
        tol_1_deg: 24.0,
        tol_2_deg: 42.0,
        fourier_v_kcal: [0.55, 0.95, 2.65, 0.0],
        fourier_gamma_deg: [180.0, 0.0, 0.0, 0.0],
    },
    // --- Benzylic, Propargylic & C(sp2)-C(sp3) Rules ---
    CsdTorsionRule {
        rule_id: "CSD-BENZYLIC-SP3",
        hierarchy_class: "Benzylic C(ar)-C(sp3)",
        title: "Benzylic c-CH2-X (~90/270 deg perpendicular avoiding ortho A(1,3) strain)",
        smarts_4atom: "cc!@[CX4][!#1]",
        preferred_peaks_deg: &[90.0, 270.0],
        tol_1_deg: 30.0,
        tol_2_deg: 55.0,
        fourier_v_kcal: [0.0, 1.65, 0.0, 0.15],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-CARBONYL-ALPHA",
        hierarchy_class: "Carbonyl-Alpha C(sp2)-C(sp3)",
        title: "Alpha-carbonyl O=C-C(sp3)-X (eclipsed 0 deg or +/-120 deg staggered)",
        smarts_4atom: "[OX1]=[CX3]!@[CX4][!#1]",
        preferred_peaks_deg: &[0.0, 120.0, 240.0],
        tol_1_deg: 26.0,
        tol_2_deg: 46.0,
        fourier_v_kcal: [0.40, 0.55, 1.65, 0.0],
        fourier_gamma_deg: [180.0, 180.0, 180.0, 0.0],
    },
    // --- Aliphatic C-N, C-O, C-S & C(sp3)-C(sp3) Rules ---
    CsdTorsionRule {
        rule_id: "CSD-ETHER-ALIPH",
        hierarchy_class: "Aliphatic C-O / C-N",
        title: "Aliphatic ether C-O-C(sp3)-C (180 deg anti dominant, +/-75 deg gauche minor)",
        smarts_4atom: "[#6][OX2]!@[CX4][!#1]",
        preferred_peaks_deg: &[180.0, 75.0, 285.0],
        tol_1_deg: 24.0,
        tol_2_deg: 44.0,
        fourier_v_kcal: [0.85, 0.45, 2.35, 0.0],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-AMINE-ALIPH",
        hierarchy_class: "Aliphatic C-O / C-N",
        title: "Aliphatic amine / piperidine-tail C-N(sp3)-C(sp3)-C (60/180/300 deg staggered)",
        smarts_4atom: "[#6][NX3]!@[CX4][!#1]",
        preferred_peaks_deg: &[180.0, 62.0, 298.0],
        tol_1_deg: 24.0,
        tol_2_deg: 44.0,
        fourier_v_kcal: [0.65, 0.25, 2.45, 0.0],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-ALKANE-SP3",
        hierarchy_class: "Aliphatic C(sp3)-C(sp3)",
        title: "Aliphatic C(sp3)-C(sp3) butane-type rotor (180 deg anti global min, +/-60 deg gauche)",
        smarts_4atom: "[!#1][CX4]!@[CX4][!#1]",
        preferred_peaks_deg: &[180.0, 60.0, 300.0],
        tol_1_deg: 22.0,
        tol_2_deg: 38.0,
        fourier_v_kcal: [0.75, 0.20, 2.95, 0.0],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-SP2-SP2-GEN",
        hierarchy_class: "General Conjugated G-G",
        title: "General conjugated sp2-sp2 single bond (~0/180 deg resonance preference)",
        smarts_4atom: "[*;^2][*;^2]!@[*;^2][*]",
        preferred_peaks_deg: &[0.0, 180.0],
        tol_1_deg: 28.0,
        tol_2_deg: 50.0,
        fourier_v_kcal: [0.0, 2.20, 0.0, 0.0],
        fourier_gamma_deg: [0.0, 180.0, 0.0, 0.0],
    },
    CsdTorsionRule {
        rule_id: "CSD-FALLBACK-GG",
        hierarchy_class: "General Fallback G-G",
        title: "General rotatable single bond fallback (6-fold / 3-fold soft torsion)",
        smarts_4atom: "[*][*]!@[*][*]",
        preferred_peaks_deg: &[60.0, 180.0, 300.0],
        tol_1_deg: 30.0,
        tol_2_deg: 55.0,
        fourier_v_kcal: [0.0, 0.0, 1.60, 0.0],
        fourier_gamma_deg: [0.0, 0.0, 0.0, 0.0],
    },
];

/// Complete `36`-point (`0..350 deg`) `1D` dihedral scan and bioactive strain readout for a single rotatable bond.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotatableBondProfile {
    pub rotor_idx: usize,
    pub bond_idx: usize,
    /// Ordered 4-atom dihedral `(i, u, v, j)` in `[0..mol.atoms.len())`.
    pub dihedral_atoms: [usize; 4],
    pub dihedral_label: String,
    pub matched_rule_id: String,
    pub hierarchy_class: String,
    pub rule_title: String,
    pub matched_smarts: String,
    pub preferred_peaks_deg: Vec<f64>,
    pub tol_1_deg: f64,
    pub tol_2_deg: f64,

    /// `36` scan angles `0, 10, ..., 350` degrees.
    pub scan_angles_deg: Vec<f64>,
    /// `CSD` Boltzmann statistical torsional potential (`TEU`) across `[0..360)`.
    pub csd_teu_curve: Vec<f64>,
    /// Combined `CSD` + intramolecular `1,4+` non-bonded steric/electrostatic/`IMHB` `1D` energy (`kcal/mol`, min = `0.0`).
    pub total_energy_kcal_curve: Vec<f64>,
    /// True if an intramolecular H-bond (`d(D..A) <= 3.20 A`) forms at each scan angle.
    pub imhb_formed_at_bin: Vec<bool>,

    // Key Scan & Bioactive Pose Readouts
    pub global_min_angle_deg: f64,
    pub rotational_barrier_kcal: f64,
    pub accessible_wells_count: usize,
    /// Conformational entropy cost `-T * Delta S_tor` (`kcal/mol` at `298.15 K`) of freezing this rotor into 1 well.
    pub rotor_entropy_penalty_kcal: f64,
    pub free_pose_angle_deg: f64,
    pub bioactive_pose_angle_deg: f64,
    pub nearest_csd_peak_deg: f64,
    pub angular_deviation_from_csd_deg: f64,
    pub bioactive_strain_teu: f64,
    pub bioactive_strain_kcal: f64,
    pub has_bioactive_imhb: bool,
    pub quality_band: TorsionQualityBand,
}

/// Conformationally locked medicinal-chemistry analog designed to preorganize a rotatable dihedral.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockedAnalogCandidate {
    pub rank: usize,
    pub candidate_id: String,
    pub target_rotor_idx: usize,
    pub target_dihedral_label: String,
    pub strategy: RotorLockStrategy,
    pub modification_summary: String,
    pub canonical_smiles: String,
    pub parent_rotatable_bonds: usize,
    pub analog_rotatable_bonds: usize,
    /// Target dihedral rotational barrier (`kcal/mol`; `12.0` if covalently locked into a ring!).
    pub locked_barrier_kcal: f64,
    pub barrier_gain_kcal: f64,
    pub analog_total_strain_teu: f64,
    /// Net preorganization free-energy boost `-Delta(Delta G_preorg)` (`kcal/mol`, positive = potency gain!).
    pub preorganization_gain_kcal: f64,
    pub rocs_bioactive_combo: f64,
    pub predicted_locked_pact: f64,
    pub delta_pact_vs_parent: f64,
    pub mw: f64,
    pub clogp: f64,
    pub lle: f64,
    pub sascore: f64,
    pub mol: Molecule,
}

/// Complete `v2.9.0` `CSD`/`SMARTS` dihedral profile, bioactive 3D strain audit & rotor-locking report for a ligand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TorsionStrainReport {
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    pub canonical_smiles: String,
    pub pose_source: BioactivePoseSource,
    pub resolved_pose_label: String,
    pub rotatable_bond_count: usize,
    pub strained_bond_count: usize,
    pub imhb_stabilized_bond_count: usize,

    // Whole-Molecule Strain & Reorganization Readouts
    pub total_strain_teu: f64,
    pub max_single_bond_teu: f64,
    pub shoichet_filter_pass: bool,
    pub bioactive_strain_penalty_kcal: f64,
    pub total_conformational_entropy_kcal: f64,
    pub raw_pocket_dg_kcal: f64,
    pub strain_adjusted_dg_bind_kcal: f64,
    pub free_vs_bioactive_rmsd_a: f64,
    pub strain_tier: BioactiveStrainTier,

    pub free_min_coords3d: Vec<[f32; 3]>,
    pub bioactive_coords3d: Vec<[f32; 3]>,
    pub gasteiger_charges: Vec<f32>,
    pub bond_profiles: Vec<RotatableBondProfile>,
    pub locked_analogs: Vec<LockedAnalogCandidate>,
    pub mol: Molecule,
}

/// Compact dataset-wide torsion strain screening row for rapid false-positive docking decoy triage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetTorsionScreenRow {
    pub rank: usize,
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    pub observed_pact: f64,
    pub rotatable_bonds: usize,
    pub strained_bonds: usize,
    pub total_strain_teu: f64,
    pub max_single_teu: f64,
    pub shoichet_pass: bool,
    pub bioactive_strain_kcal: f64,
    pub strain_adjusted_dg_kcal: f64,
    pub strain_tier: BioactiveStrainTier,
    pub top_lock_strategy: String,
    pub top_lock_gain_kcal: f64,
}

#[inline]
fn wrap_deg_360(deg: f64) -> f64 {
    let mut r = deg % 360.0;
    if r < 0.0 {
        r += 360.0;
    }
    r
}

#[inline]
fn min_circular_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    let d = (wrap_deg_360(a_deg) - wrap_deg_360(b_deg)).abs();
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

/// Computes the IUPAC signed dihedral angle `phi(p0, p1, p2, p3)` wrapped into `[0, 360)` degrees.
pub fn compute_dihedral_deg_3d(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f64 {
    let b1 = [
        (p1[0] - p0[0]) as f64,
        (p1[1] - p0[1]) as f64,
        (p1[2] - p0[2]) as f64,
    ];
    let b2 = [
        (p2[0] - p1[0]) as f64,
        (p2[1] - p1[1]) as f64,
        (p2[2] - p1[2]) as f64,
    ];
    let b3 = [
        (p3[0] - p2[0]) as f64,
        (p3[1] - p2[1]) as f64,
        (p3[2] - p2[2]) as f64,
    ];

    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let dot = |u: [f64; 3], v: [f64; 3]| -> f64 { u[0] * v[0] + u[1] * v[1] + u[2] * v[2] };

    let b2_norm = dot(b2, b2).sqrt().max(1e-9);
    let b2_u = [b2[0] / b2_norm, b2[1] / b2_norm, b2[2] / b2_norm];
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let m1 = cross(n1, b2_u);

    let x = dot(n1, n2);
    let y = dot(m1, n2);
    if x.abs() < 1e-12 && y.abs() < 1e-12 {
        return 180.0;
    }
    wrap_deg_360(y.atan2(x).to_degrees())
}

/// Evaluates the Fourier torsional potential `E_fourier(phi)` (`kcal/mol`) and `CSD` `TEU` at `phi_deg`.
fn eval_rule_fourier_and_teu(rule: &CsdTorsionRule, phi_deg: f64) -> (f64, f64) {
    let rad = phi_deg.to_radians();
    let mut e_f = 0.0_f64;
    for k in 0..4 {
        let vk = rule.fourier_v_kcal[k];
        if vk.abs() > 1e-6 {
            let n = (k + 1) as f64;
            let gam = rule.fourier_gamma_deg[k].to_radians();
            e_f += 0.5 * vk * (1.0 + (n * rad - gam).cos());
        }
    }

    // Distance to nearest empirical CSD peak for Boltzmann TEU conversion (Gu & Shoichet 2021)
    let mut min_dev = 180.0_f64;
    for &pk in rule.preferred_peaks_deg {
        let d = min_circular_diff_deg(phi_deg, pk);
        if d < min_dev {
            min_dev = d;
        }
    }
    let sigma_deg = (rule.tol_1_deg * 0.82).max(10.0);
    let harmonic_teu = 0.55 * (min_dev / sigma_deg).powi(2);
    let teu = (0.52 * e_f + 0.48 * harmonic_teu).clamp(0.0, 6.5);
    (e_f, teu)
}

/// Perceives rotatable single non-ring bonds `(bond_idx, [i, u, v, j])` (including secondary/tertiary amides
/// & aryl-carbonyls whose planar torsions are scanned by `TorsionAnalyzer`, excluding terminal `-CH3`/`-NH2`/`-OH`/`-CF3` rotors).
pub fn perceive_scannable_dihedrals(mol: &Molecule) -> Vec<(usize, [usize; 4])> {
    let mut out = Vec::new();
    for (bi, b) in mol.bonds.iter().enumerate() {
        if b.in_ring || b.order != BondOrder::Single {
            continue;
        }
        let u = b.a;
        let v = b.b;
        if mol.atoms[u].elem == Element::H || mol.atoms[v].elem == Element::H {
            continue;
        }
        // Choose highest-atomic-number heavy neighbor i != v of u, and j != u of v
        let nb_u: Vec<usize> = mol.adj[u]
            .iter()
            .map(|&(nb, _)| nb)
            .filter(|&nb| nb != v && mol.atoms[nb].elem != Element::H)
            .collect();
        let nb_v: Vec<usize> = mol.adj[v]
            .iter()
            .map(|&(nb, _)| nb)
            .filter(|&nb| nb != u && mol.atoms[nb].elem != Element::H)
            .collect();
        if nb_u.is_empty() || nb_v.is_empty() {
            continue;
        }
        // Exclude trivial symmetric terminal CX3 rotors (-CF3, -CCl3, -tBu) and linear sp nitrile/alkyne bonds
        if mol.atoms[u].hyb == Hybridization::Sp || mol.atoms[v].hyb == Hybridization::Sp {
            continue;
        }
        let is_sym_trihalo = |center: usize, nbs: &[usize]| -> bool {
            mol.atoms[center].hyb == Hybridization::Sp3
                && nbs.len() == 3
                && nbs.iter().all(|&x| {
                    mol.adj[x].iter().filter(|&&(y, _)| mol.atoms[y].elem != Element::H).count() == 1
                        && mol.atoms[x].elem == mol.atoms[nbs[0]].elem
                })
        };
        if is_sym_trihalo(u, &nb_u) || is_sym_trihalo(v, &nb_v) {
            continue;
        }

        let pick_ref = |nbs: &[usize]| -> usize {
            *nbs.iter()
                .max_by_key(|&&x| {
                    let a = &mol.atoms[x];
                    (
                        a.elem.atomic_number() as usize * 10 + mol.adj[x].len(),
                        usize::MAX - x,
                    )
                })
                .unwrap_or(&nbs[0])
        };
        let i = pick_ref(&nb_u);
        let j = pick_ref(&nb_v);
        out.push((bi, [i, u, v, j]));
    }
    out
}

/// Matches a 4-atom rotatable bond `(i, u, v, j)` against the hierarchical `CSD_SMARTS_TORSION_LIBRARY`,
/// returning `(rule_ref, canonical_4atom_mapping)`.
pub fn match_csd_torsion_rule(
    mol: &Molecule,
    default_quad: [usize; 4],
) -> (&'static CsdTorsionRule, [usize; 4]) {
    let u = default_quad[1];
    let v = default_quad[2];

    for rule in &CSD_SMARTS_TORSION_LIBRARY {
        let Ok(q) = compile_smarts(rule.smarts_4atom) else {
            continue;
        };
        if q.atoms.len() != 4 {
            continue;
        }
        let res = match_smarts(mol, &q);
        if !res.matched {
            continue;
        }
        for map in &res.mappings {
            if map.len() == 4 {
                if map[1] == u && map[2] == v {
                    return (rule, [map[0], map[1], map[2], map[3]]);
                }
                if map[1] == v && map[2] == u {
                    return (rule, [map[3], map[2], map[1], map[0]]);
                }
            }
        }
    }

    (&CSD_SMARTS_TORSION_LIBRARY[CSD_SMARTS_TORSION_LIBRARY.len() - 1], default_quad)
}

/// Collects the atom indices on the `v` side of cut bond `(u, v)` via BFS.
fn collect_rotating_fragment_atoms(mol: &Molecule, u: usize, v: usize) -> Vec<usize> {
    let mut vis = vec![false; mol.atoms.len()];
    vis[u] = true;
    vis[v] = true;
    let mut q = VecDeque::new();
    q.push_back(v);
    let mut side_v = vec![v];
    while let Some(cur) = q.pop_front() {
        for &(nb, _) in &mol.adj[cur] {
            if !vis[nb] {
                vis[nb] = true;
                if nb != v {
                    side_v.push(nb);
                }
                q.push_back(nb);
            }
        }
    }
    side_v
}

/// Rotates `side_v_atoms` in `coords` around axis `u -> v` by `delta_rad` (`Rodrigues` `SO(3)` formula).
fn rotate_dihedral_in_place(
    coords: &mut [[f32; 3]],
    u: usize,
    v: usize,
    side_v_atoms: &[usize],
    delta_rad: f64,
) {
    let pu = [coords[u][0] as f64, coords[u][1] as f64, coords[u][2] as f64];
    let pv = [coords[v][0] as f64, coords[v][1] as f64, coords[v][2] as f64];
    let ax = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
    let len = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt();
    if len < 1e-8 {
        return;
    }
    let k = [ax[0] / len, ax[1] / len, ax[2] / len];
    let c = delta_rad.cos();
    let s = delta_rad.sin();
    let one_c = 1.0 - c;

    for &idx in side_v_atoms {
        if idx == v || idx >= coords.len() {
            continue;
        }
        let r = [
            (coords[idx][0] as f64) - pv[0],
            (coords[idx][1] as f64) - pv[1],
            (coords[idx][2] as f64) - pv[2],
        ];
        let k_cross_r = [
            k[1] * r[2] - k[2] * r[1],
            k[2] * r[0] - k[0] * r[2],
            k[0] * r[1] - k[1] * r[0],
        ];
        let k_dot_r = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
        let rot = [
            pv[0] + r[0] * c + k_cross_r[0] * s + k[0] * k_dot_r * one_c,
            pv[1] + r[1] * c + k_cross_r[1] * s + k[1] * k_dot_r * one_c,
            pv[2] + r[2] * c + k_cross_r[2] * s + k[2] * k_dot_r * one_c,
        ];
        coords[idx] = [rot[0] as f32, rot[1] as f32, rot[2] as f32];
    }
}

/// Evaluates cross-rotor non-bonded `1,4+` steric clash, electrostatic, and `1,5`/`1,6` `IMHB` energy (`kcal/mol`)
/// between fixed fragment `side_u` and rotating fragment `side_v`.
fn eval_cross_rotor_nonbonded_and_imhb(
    mol: &Molecule,
    coords: &[[f32; 3]],
    charges: &[f32],
    side_u: &[usize],
    side_v: &[usize],
    bonded_or_13: &BTreeSet<(usize, usize)>,
) -> (f64, bool) {
    let mut e_nb = 0.0_f64;
    let mut imhb = false;

    for &a in side_u {
        let pa = coords[a];
        let ea = mol.atoms[a].elem;
        let ra = bondi_vdw_radius(ea) as f64;
        let qa = charges.get(a).copied().unwrap_or(0.0) as f64;
        let a_hbd = matches!(ea, Element::N | Element::O) && mol.atoms[a].total_h() > 0;
        let a_hba = matches!(ea, Element::N | Element::O | Element::F) && mol.atoms[a].charge <= 0;

        for &b in side_v {
            let key = if a < b { (a, b) } else { (b, a) };
            if bonded_or_13.contains(&key) {
                continue;
            }
            let pb = coords[b];
            let dx = (pa[0] - pb[0]) as f64;
            let dy = (pa[1] - pb[1]) as f64;
            let dz = (pa[2] - pb[2]) as f64;
            let d = (dx * dx + dy * dy + dz * dz).sqrt().max(0.65);
            if d > 6.5 {
                continue;
            }

            let eb = mol.atoms[b].elem;
            let b_hbd = matches!(eb, Element::N | Element::O) && mol.atoms[b].total_h() > 0;
            let b_hba = matches!(eb, Element::N | Element::O | Element::F) && mol.atoms[b].charge <= 0;

            // Check 1,5 / 1,6 intramolecular H-bond (Kuhn, Mohr & Stahl 2010: heavy-atom D..A in [2.35, 3.25] A)
            if ((a_hbd && b_hba) || (b_hbd && a_hba)) && (2.30..=3.25).contains(&d) {
                imhb = true;
                let e_hb = -1.55 * ((3.35 - d) / 0.85).clamp(0.25, 1.05);
                e_nb += e_hb;
                continue;
            }

            // Softened 1,4+ steric overlap + dispersion
            let r_sum = 0.86 * (ra + (bondi_vdw_radius(eb) as f64));
            if d < r_sum {
                let pen = r_sum - d;
                e_nb += (3.8 * pen * pen).min(6.5);
            } else {
                let ratio = r_sum / d;
                let r6 = ratio.powi(6);
                e_nb += -0.06 * r6;
            }

            // Screened Coulombic dipole-dipole term (`eps_r = 12.0` effective internal dielectric)
            let qb = charges.get(b).copied().unwrap_or(0.0) as f64;
            e_nb += (27.6 * qa * qb / d).clamp(-1.4, 1.8);
        }
    }

    (e_nb, imhb)
}

/// Builds a `1,2` + `1,3` exclusion set so `1,4+` non-bonded scans skip covalent bonds and geminal valence angles.
fn build_12_13_exclusion_set(mol: &Molecule) -> BTreeSet<(usize, usize)> {
    let mut excl = BTreeSet::new();
    for b in &mol.bonds {
        let p = if b.a < b.b { (b.a, b.b) } else { (b.b, b.a) };
        excl.insert(p);
    }
    for center in 0..mol.atoms.len() {
        let nbs: Vec<usize> = mol.adj[center].iter().map(|&(nb, _)| nb).collect();
        for x in 0..nbs.len() {
            for y in (x + 1)..nbs.len() {
                let a = nbs[x];
                let b = nbs[y];
                let p = if a < b { (a, b) } else { (b, a) };
                excl.insert(p);
            }
        }
    }
    excl
}

/// Scans a single rotatable dihedral `(i, u, v, j)` across `36` bins (`0..350 deg`) and evaluates its bioactive strain.
#[allow(clippy::too_many_arguments)]
fn scan_single_rotatable_bond(
    rotor_idx: usize,
    bond_idx: usize,
    raw_quad: [usize; 4],
    mol: &Molecule,
    free_coords: &[[f32; 3]],
    bioactive_coords: &[[f32; 3]],
    charges: &[f32],
    excl_12_13: &BTreeSet<(usize, usize)>,
) -> RotatableBondProfile {
    let (rule, quad) = match_csd_torsion_rule(mol, raw_quad);
    let [i, u, v, j] = quad;

    let side_v = collect_rotating_fragment_atoms(mol, u, v);
    let set_v: BTreeSet<usize> = side_v.iter().copied().collect();
    let side_u: Vec<usize> = (0..mol.atoms.len())
        .filter(|a| *a != u && *a != v && !set_v.contains(a))
        .collect();

    let base_phi = compute_dihedral_deg_3d(free_coords[i], free_coords[u], free_coords[v], free_coords[j]);
    let bio_phi = compute_dihedral_deg_3d(
        bioactive_coords[i],
        bioactive_coords[u],
        bioactive_coords[v],
        bioactive_coords[j],
    );

    let mut scan_angles_deg = Vec::with_capacity(TORSION_SCAN_BINS);
    let mut csd_teu_curve = Vec::with_capacity(TORSION_SCAN_BINS);
    let mut raw_kcal_curve = Vec::with_capacity(TORSION_SCAN_BINS);
    let mut imhb_formed_at_bin = Vec::with_capacity(TORSION_SCAN_BINS);

    for bin in 0..TORSION_SCAN_BINS {
        let target_deg = (bin as f64) * 10.0;
        let delta_rad = (target_deg - base_phi).to_radians();
        let mut trial_coords = free_coords.to_vec();
        rotate_dihedral_in_place(&mut trial_coords, u, v, &side_v, delta_rad);

        let (e_fourier, teu) = eval_rule_fourier_and_teu(rule, target_deg);
        let (e_nb, has_imhb) = eval_cross_rotor_nonbonded_and_imhb(
            mol,
            &trial_coords,
            charges,
            &side_u,
            &side_v,
            excl_12_13,
        );
        scan_angles_deg.push(target_deg);
        csd_teu_curve.push(teu);
        raw_kcal_curve.push(e_fourier + e_nb);
        imhb_formed_at_bin.push(has_imhb);
    }

    // Zero-reference 1D energy curve to global minimum
    let mut min_e = f64::INFINITY;
    let mut max_e = f64::NEG_INFINITY;
    let mut min_bin = 0usize;
    for (b_idx, &e) in raw_kcal_curve.iter().enumerate() {
        if e < min_e {
            min_e = e;
            min_bin = b_idx;
        }
        if e > max_e {
            max_e = e;
        }
    }
    let total_energy_kcal_curve: Vec<f64> = raw_kcal_curve.iter().map(|&e| (e - min_e).max(0.0)).collect();
    let global_min_angle_deg = scan_angles_deg[min_bin];
    let rotational_barrier_kcal = (max_e - min_e).clamp(0.25, 14.0);

    // Count distinct local-minimum wells within 1.6 kcal/mol of global minimum & compute Boltzmann torsional entropy
    let mut wells = 0usize;
    let mut z_part = 0.0_f64;
    for b_idx in 0..TORSION_SCAN_BINS {
        let prev = total_energy_kcal_curve[(b_idx + TORSION_SCAN_BINS - 1) % TORSION_SCAN_BINS];
        let cur = total_energy_kcal_curve[b_idx];
        let next = total_energy_kcal_curve[(b_idx + 1) % TORSION_SCAN_BINS];
        if cur <= prev && cur < next && cur <= 1.65 {
            wells += 1;
        }
        z_part += (-cur / TORSION_RT_298K_KCAL).exp();
    }
    let accessible_wells_count = wells.max(1);
    let mut shannon = 0.0_f64;
    for &cur in &total_energy_kcal_curve {
        let p = (-cur / TORSION_RT_298K_KCAL).exp() / z_part.max(1e-12);
        if p > 1e-12 {
            shannon -= p * p.ln();
        }
    }
    // Subtract single-well thermal liberation baseline (`ln(3.2)` ~ +/-15 deg libration) so a locked 1-well rotor ~ 0.05 kcal/mol
    let excess_entropy_nats = (shannon - 3.2_f64.ln()).max(0.05);
    let rotor_entropy_penalty_kcal = (TORSION_RT_298K_KCAL * excess_entropy_nats).clamp(0.03, 1.35);

    // Bioactive pose readout interpolated along the 36-bin scan + exact CSD peak check
    let (_, bio_teu_raw) = eval_rule_fourier_and_teu(rule, bio_phi);
    let (_, bio_imhb) = eval_cross_rotor_nonbonded_and_imhb(
        mol,
        bioactive_coords,
        charges,
        &side_u,
        &side_v,
        excl_12_13,
    );
    let bio_bin_f = wrap_deg_360(bio_phi) / 10.0;
    let b0 = (bio_bin_f.floor() as usize) % TORSION_SCAN_BINS;
    let b1 = (b0 + 1) % TORSION_SCAN_BINS;
    let frac = bio_bin_f - (b0 as f64);
    let bioactive_strain_kcal =
        ((1.0 - frac) * total_energy_kcal_curve[b0] + frac * total_energy_kcal_curve[b1]).clamp(0.0, 8.5);
    let bioactive_strain_teu = if bio_imhb {
        (bio_teu_raw * 0.65).max(0.0)
    } else {
        (0.65 * bio_teu_raw + 0.35 * bioactive_strain_kcal).clamp(0.0, 6.5)
    };

    let mut nearest_pk = rule.preferred_peaks_deg[0];
    let mut min_dev = 180.0_f64;
    for &pk in rule.preferred_peaks_deg {
        let d = min_circular_diff_deg(bio_phi, pk);
        if d < min_dev {
            min_dev = d;
            nearest_pk = pk;
        }
    }

    let quality_band = if bioactive_strain_teu > SHOICHET_SINGLE_TEU_CUTOFF || min_dev > rule.tol_2_deg {
        TorsionQualityBand::StrainedRed
    } else if bioactive_strain_teu > 1.00 || min_dev > rule.tol_1_deg {
        TorsionQualityBand::ToleratedAmber
    } else {
        TorsionQualityBand::RelaxedGreen
    };

    let dihedral_label = format!(
        "{}{}-{}{}-{}{}-{}{}",
        mol.atoms[i].elem.symbol(),
        i + 1,
        mol.atoms[u].elem.symbol(),
        u + 1,
        mol.atoms[v].elem.symbol(),
        v + 1,
        mol.atoms[j].elem.symbol(),
        j + 1
    );

    RotatableBondProfile {
        rotor_idx,
        bond_idx,
        dihedral_atoms: quad,
        dihedral_label,
        matched_rule_id: rule.rule_id.to_string(),
        hierarchy_class: rule.hierarchy_class.to_string(),
        rule_title: rule.title.to_string(),
        matched_smarts: rule.smarts_4atom.to_string(),
        preferred_peaks_deg: rule.preferred_peaks_deg.to_vec(),
        tol_1_deg: rule.tol_1_deg,
        tol_2_deg: rule.tol_2_deg,
        scan_angles_deg,
        csd_teu_curve,
        total_energy_kcal_curve,
        imhb_formed_at_bin,
        global_min_angle_deg,
        rotational_barrier_kcal,
        accessible_wells_count,
        rotor_entropy_penalty_kcal,
        free_pose_angle_deg: base_phi,
        bioactive_pose_angle_deg: bio_phi,
        nearest_csd_peak_deg: nearest_pk,
        angular_deviation_from_csd_deg: min_dev,
        bioactive_strain_teu,
        bioactive_strain_kcal,
        has_bioactive_imhb: bio_imhb,
        quality_band,
    }
}

/// Appends a single monovalent substituent (`elem`, single bond) onto `anchor_atom` if `anchor_atom` has `>= 1` implicit/explicit H,
/// returning a freshly parsed, valence-sanitized `Molecule`.
fn try_attach_monovalent_substituent(
    parent: &Molecule,
    anchor_atom: usize,
    sub_elem: Element,
) -> Option<Molecule> {
    if anchor_atom >= parent.atoms.len() || parent.atoms[anchor_atom].total_h() == 0 {
        return None;
    }
    let mut m = parent.clone();
    if m.atoms[anchor_atom].explicit_h > 0 {
        m.atoms[anchor_atom].explicit_h -= 1;
    }
    let new_idx = m.atoms.len();
    let p2d = m.atoms[anchor_atom].pos2d;
    m.atoms.push(Atom {
        idx: new_idx,
        elem: sub_elem,
        charge: 0,
        aromatic: false,
        bracketed: false,
        explicit_h: 0,
        implicit_h: 0,
        chiral: 0,
        in_ring: false,
        ring_sizes: Vec::new(),
        hyb: Hybridization::Sp3,
        pos2d: [p2d[0] + 0.9, p2d[1] + 0.6],
    });
    let b_idx = m.bonds.len();
    m.bonds.push(Bond {
        idx: b_idx,
        a: anchor_atom,
        b: new_idx,
        order: BondOrder::Single,
        aromatic: false,
        in_ring: false,
        kekule_order: 1,
    });
    m.sanitize();
    let smi = molecule_to_smiles(&m);
    parse_smiles(&smi).ok()
}

/// Converts an aromatic `cH` atom (`C` in a 6-membered aromatic ring with 1 H) into a pyridine-type ring `n`
/// to introduce a `1,5` intramolecular H-bond acceptor without adding molecular weight!
fn try_aza_ring_imhb_swap(parent: &Molecule, ch_atom: usize) -> Option<Molecule> {
    if ch_atom >= parent.atoms.len() {
        return None;
    }
    let a = &parent.atoms[ch_atom];
    if a.elem != Element::C || !a.aromatic || a.total_h() == 0 || a.charge != 0 {
        return None;
    }
    let mut m = parent.clone();
    m.atoms[ch_atom].elem = Element::N;
    m.atoms[ch_atom].explicit_h = 0;
    m.atoms[ch_atom].implicit_h = 0;
    m.sanitize();
    let smi = molecule_to_smiles(&m);
    parse_smiles(&smi).ok()
}

/// Closes a direct covalent single bond (or `-CH2-` methylene bridge) between two open-valence heavy atoms `a` and `b`
/// (`3`- to `6`-membered ring cyclization lock) when chemically reasonable.
fn try_small_ring_cyclization(
    parent: &Molecule,
    atom_a: usize,
    atom_b: usize,
    insert_methylene: bool,
) -> Option<Molecule> {
    if atom_a == atom_b
        || atom_a >= parent.atoms.len()
        || atom_b >= parent.atoms.len()
        || parent.atoms[atom_a].total_h() == 0
        || parent.atoms[atom_b].total_h() == 0
    {
        return None;
    }
    if parent
        .bonds
        .iter()
        .any(|eb| (eb.a == atom_a && eb.b == atom_b) || (eb.a == atom_b && eb.b == atom_a))
    {
        return None;
    }

    let mut m = parent.clone();
    if m.atoms[atom_a].explicit_h > 0 {
        m.atoms[atom_a].explicit_h -= 1;
    }
    if m.atoms[atom_b].explicit_h > 0 {
        m.atoms[atom_b].explicit_h -= 1;
    }

    if insert_methylene {
        let ch2_idx = m.atoms.len();
        let mid = [
            0.5 * (m.atoms[atom_a].pos2d[0] + m.atoms[atom_b].pos2d[0]),
            0.5 * (m.atoms[atom_a].pos2d[1] + m.atoms[atom_b].pos2d[1]) + 0.6,
        ];
        m.atoms.push(Atom {
            idx: ch2_idx,
            elem: Element::C,
            charge: 0,
            aromatic: false,
            bracketed: false,
            explicit_h: 0,
            implicit_h: 0,
            chiral: 0,
            in_ring: false,
            ring_sizes: Vec::new(),
            hyb: Hybridization::Sp3,
            pos2d: mid,
        });
        let b1 = m.bonds.len();
        m.bonds.push(Bond {
            idx: b1,
            a: atom_a,
            b: ch2_idx,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
        let b2 = m.bonds.len();
        m.bonds.push(Bond {
            idx: b2,
            a: ch2_idx,
            b: atom_b,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
    } else {
        let bi = m.bonds.len();
        m.bonds.push(Bond {
            idx: bi,
            a: atom_a,
            b: atom_b,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });
    }

    m.sanitize();
    let smi = molecule_to_smiles(&m);
    let mut parsed = parse_smiles(&smi).ok()?;
    compute_2d_coords(&mut parsed);
    Some(parsed)
}

/// Enumerates and ranks conformationally locked analogs across all rotatable bonds of `mol`,
/// matching each locking motif (`ortho`-Me twist, `IMHB` / `ortho`-F planar lock, `alpha`-Me/F, or small-ring cyclization)
/// to the bioactive 3D dihedral geometry!
fn design_rotatable_bond_locks(
    mol: &Molecule,
    bioactive_coords: &[[f32; 3]],
    bond_profiles: &[RotatableBondProfile],
) -> Vec<LockedAnalogCandidate> {
    let parent_prof = evaluate_physchem(mol, 0);
    let parent_s3d = evaluate_3d_summary_profile(mol, true);
    let parent_pact = if mol.pact > 0.5 { mol.pact } else { 7.0 };

    let mut raw_specs: Vec<(usize, RotorLockStrategy, String, Molecule)> = Vec::new();
    let mut seen_smiles: BTreeSet<String> = BTreeSet::new();
    seen_smiles.insert(mol.smiles.clone());

    for bp in bond_profiles {
        let [_, u, v, _] = bp.dihedral_atoms;
        let bio_phi = bp.bioactive_pose_angle_deg;
        let near_planar = min_circular_diff_deg(bio_phi, 0.0) <= 38.0
            || min_circular_diff_deg(bio_phi, 180.0) <= 38.0;
        let twisted_orthogonal = !near_planar;

        // Find aromatic ortho CH neighbors adjacent to u or v
        let mut ortho_ch_sites = Vec::new();
        for &pivot in &[u, v] {
            if mol.atoms[pivot].aromatic {
                for &(nb, _) in &mol.adj[pivot] {
                    if nb != u
                        && nb != v
                        && mol.atoms[nb].aromatic
                        && mol.atoms[nb].elem == Element::C
                        && mol.atoms[nb].total_h() > 0
                    {
                        ortho_ch_sites.push((pivot, nb));
                    }
                }
            }
        }

        // 1. Ortho-Methyl Steric Gear (favored when bioactive pose is twisted, or as exploratory twist test)
        for &(_, ch_u) in ortho_ch_sites.iter().take(2) {
            if let Some(cand_m) = try_attach_monovalent_substituent(mol, ch_u, Element::C) {
                if seen_smiles.insert(cand_m.smiles.clone()) {
                    let note = if twisted_orthogonal {
                        format!(
                            "Ortho-CH3 on C{} gears {} toward twisted {:.0} deg bioactive pose",
                            ch_u + 1,
                            bp.dihedral_label,
                            bio_phi
                        )
                    } else {
                        format!(
                            "Ortho-CH3 steric probe on C{} testing atropisomeric twist on {}",
                            ch_u + 1,
                            bp.dihedral_label
                        )
                    };
                    raw_specs.push((bp.rotor_idx, RotorLockStrategy::OrthoMethylStericGear, note, cand_m));
                }
            }
        }

        // 2. Intramolecular H-Bond (IMHB) / Ortho-Fluoro Coplanar Pseudo-Ring Lock
        let has_nh_or_co = matches!(mol.atoms[u].elem, Element::N | Element::O)
            || matches!(mol.atoms[v].elem, Element::N | Element::O)
            || mol.adj[u].iter().any(|&(x, _)| matches!(mol.atoms[x].elem, Element::N | Element::O))
            || mol.adj[v].iter().any(|&(x, _)| matches!(mol.atoms[x].elem, Element::N | Element::O));

        for &(_, ch_u) in ortho_ch_sites.iter().take(2) {
            if let Some(cand_f) = try_attach_monovalent_substituent(mol, ch_u, Element::F) {
                if seen_smiles.insert(cand_f.smiles.clone()) {
                    let note = format!(
                        "Ortho-F on C{} locks {} via dipolar / 1,5 C-F...H-N pseudo-ring ({:.0} deg)",
                        ch_u + 1,
                        bp.dihedral_label,
                        bio_phi
                    );
                    raw_specs.push((
                        bp.rotor_idx,
                        RotorLockStrategy::IntramolecularHBondOrFluoroLock,
                        note,
                        cand_f,
                    ));
                }
            }
            if has_nh_or_co {
                if let Some(cand_n) = try_aza_ring_imhb_swap(mol, ch_u) {
                    if seen_smiles.insert(cand_n.smiles.clone()) {
                        let note = format!(
                            "Aza CH{}->N swap creates 5/6-membered IMHB coplanar lock on {}",
                            ch_u + 1,
                            bp.dihedral_label
                        );
                        raw_specs.push((
                            bp.rotor_idx,
                            RotorLockStrategy::IntramolecularHBondOrFluoroLock,
                            note,
                            cand_n,
                        ));
                    }
                }
            }
        }

        // 3. Alpha-Methyl A(1,3) or Beta-Fluoro Gauche Lock on sp3 carbons
        for &sp3_atom in &[u, v] {
            if mol.atoms[sp3_atom].elem == Element::C
                && !mol.atoms[sp3_atom].aromatic
                && mol.atoms[sp3_atom].hyb == Hybridization::Sp3
                && mol.atoms[sp3_atom].total_h() > 0
            {
                if let Some(cand_me) = try_attach_monovalent_substituent(mol, sp3_atom, Element::C)
                {
                    if seen_smiles.insert(cand_me.smiles.clone()) {
                        raw_specs.push((
                            bp.rotor_idx,
                            RotorLockStrategy::AlphaSubstituentA13OrGaucheLock,
                            format!(
                                "Alpha-CH3 on C{} restricts sp3 rotor {} via 1,3-allylic / geminal bias",
                                sp3_atom + 1,
                                bp.dihedral_label
                            ),
                            cand_me,
                        ));
                    }
                }
                if let Some(cand_fl) = try_attach_monovalent_substituent(mol, sp3_atom, Element::F)
                {
                    if seen_smiles.insert(cand_fl.smiles.clone()) {
                        raw_specs.push((
                            bp.rotor_idx,
                            RotorLockStrategy::AlphaSubstituentA13OrGaucheLock,
                            format!(
                                "Alpha-F on C{} biases {} into +/-65 deg hyperconjugative gauche well",
                                sp3_atom + 1,
                                bp.dihedral_label
                            ),
                            cand_fl,
                        ));
                    }
                }
            }
        }

        // 4. Small-Ring Covalent Cyclization (geminalcyclopropane / 5- or 6-membered ortho-linker ring closure!)
        let i_atom = bp.dihedral_atoms[0];
        let j_atom = bp.dihedral_atoms[3];
        // Try closing i_atom -- j_atom (forms 4-membered ring, or 5-membered with -CH2- bridge!)
        if let Some(cyc_5) = try_small_ring_cyclization(mol, i_atom, j_atom, true) {
            if seen_smiles.insert(cyc_5.smiles.clone()) {
                raw_specs.push((
                    bp.rotor_idx,
                    RotorLockStrategy::SmallRingCovalentCyclization,
                    format!(
                        "5-Membered -CH2- ring closure ({}{}..{}{}) covalently locks {}",
                        mol.atoms[i_atom].elem.symbol(),
                        i_atom + 1,
                        mol.atoms[j_atom].elem.symbol(),
                        j_atom + 1,
                        bp.dihedral_label
                    ),
                    cyc_5,
                ));
            }
        }
        if let Some(cyc_4) = try_small_ring_cyclization(mol, i_atom, j_atom, false) {
            if seen_smiles.insert(cyc_4.smiles.clone()) {
                raw_specs.push((
                    bp.rotor_idx,
                    RotorLockStrategy::SmallRingCovalentCyclization,
                    format!(
                        "Direct 4-membered ring fusion ({}{}-{}{}) eliminates rotor {}",
                        mol.atoms[i_atom].elem.symbol(),
                        i_atom + 1,
                        mol.atoms[j_atom].elem.symbol(),
                        j_atom + 1,
                        bp.dihedral_label
                    ),
                    cyc_4,
                ));
            }
        }
        // Cyclopropyl lock across sp3-sp3 (u, v) via -CH2- bridge!
        if !mol.atoms[u].aromatic
            && !mol.atoms[v].aromatic
            && mol.atoms[u].elem == Element::C
            && mol.atoms[v].elem == Element::C
        {
            if let Some(cyc_3) = try_small_ring_cyclization(mol, u, v, true) {
                if seen_smiles.insert(cyc_3.smiles.clone()) {
                    raw_specs.push((
                        bp.rotor_idx,
                        RotorLockStrategy::SmallRingCovalentCyclization,
                        format!(
                            "Cyclopropyl C{}-CH2-C{} fusion locks {} rigid",
                            u + 1,
                            v + 1,
                            bp.dihedral_label
                        ),
                        cyc_3,
                    ));
                }
            }
        }
    }

    // Evaluate all candidate locked analogs in parallel: 3D ROCS overlay onto bioactive pose + barrier & strain audit
    let mut cands: Vec<LockedAnalogCandidate> = raw_specs
        .into_par_iter()
        .take(20)
        .enumerate()
        .map(|(ci, (rot_idx, strat, note, mut cmol))| {
            cmol.id = format!("LOCK-{:02}", ci + 1);
            cmol.name = format!("{} ({})", strat.short_label(), bond_profiles[rot_idx].dihedral_label);
            cmol.target = mol.target.clone();

            let c_prof = evaluate_physchem(&cmol, 0);
            let c_sa = evaluate_synthetic_accessibility(&cmol);
            let c_s3d = evaluate_3d_summary_profile(&cmol, true);

            // Align locked analog 3D conformer onto parent bioactive 3D coordinates!
            let rocs = align_rocs_3d(
                mol,
                bioactive_coords,
                &parent_s3d.gasteiger_charges,
                &parent_s3d.color_atoms,
                &cmol,
                &c_s3d.coords3d,
                &c_s3d.gasteiger_charges,
                &c_s3d.color_atoms,
            );

            // Evaluate locked analog's own rotatable dihedrals and total TEU in the aligned bioactive pose
            let c_dihedrals = perceive_scannable_dihedrals(&cmol);
            let c_excl = build_12_13_exclusion_set(&cmol);
            let mut analog_teu_sum = 0.0_f64;
            let mut max_analog_barrier = 0.0_f64;
            for (r_i, (b_i, q4)) in c_dihedrals.iter().enumerate() {
                let sub_bp = scan_single_rotatable_bond(
                    r_i,
                    *b_i,
                    *q4,
                    &cmol,
                    &c_s3d.coords3d,
                    &rocs.aligned_hit_coords,
                    &c_s3d.gasteiger_charges,
                    &c_excl,
                );
                analog_teu_sum += sub_bp.bioactive_strain_teu;
                if sub_bp.rotational_barrier_kcal > max_analog_barrier {
                    max_analog_barrier = sub_bp.rotational_barrier_kcal;
                }
            }

            let parent_bp = &bond_profiles[rot_idx];
            let locked_barrier = if strat == RotorLockStrategy::SmallRingCovalentCyclization
                && c_prof.rotatable_bonds < parent_prof.rotatable_bonds
            {
                12.0
            } else {
                max_analog_barrier.max(parent_bp.rotational_barrier_kcal + 0.85)
            };
            let barrier_gain = (locked_barrier - parent_bp.rotational_barrier_kcal).max(0.15);

            // Preorganization gain combines:
            // + entropy saved from locking parent rotor (`0.65 * rotor_entropy_penalty_kcal`)
            // + relief of parent bioactive dihedral strain when locked near bioactive pose
            // + 3D ROCS bioactive shape/color fidelity bonus (`combo_tanimoto` in `[0, 1]`)
            let bio_phi = parent_bp.bioactive_pose_angle_deg;
            let near_planar = min_circular_diff_deg(bio_phi, 0.0) <= 38.0
                || min_circular_diff_deg(bio_phi, 180.0) <= 38.0;
            let angle_match_mult = match strat {
                RotorLockStrategy::OrthoMethylStericGear => {
                    if !near_planar { 1.15 } else { 0.35 }
                }
                RotorLockStrategy::IntramolecularHBondOrFluoroLock => {
                    if near_planar { 1.18 } else { 0.45 }
                }
                RotorLockStrategy::AlphaSubstituentA13OrGaucheLock => 0.92,
                RotorLockStrategy::SmallRingCovalentCyclization => 1.10,
            };

            let entropy_saved = angle_match_mult
                * (0.85 * parent_bp.rotor_entropy_penalty_kcal
                    + 0.38 * parent_bp.bioactive_strain_kcal.min(2.4));
            let shape_penalty = ((0.76 - rocs.combo_tanimoto).max(0.0)) * 2.2;
            let preorg_gain_kcal = (entropy_saved - shape_penalty).clamp(-1.20, 2.35);

            let delta_pact = preorg_gain_kcal / RT_LN10_KCAL;
            let pred_pact = (parent_pact + delta_pact).clamp(4.0, 11.5);
            cmol.pact = pred_pact;
            let lle = pred_pact - c_prof.clogp;

            LockedAnalogCandidate {
                rank: ci + 1,
                candidate_id: cmol.id.clone(),
                target_rotor_idx: rot_idx,
                target_dihedral_label: parent_bp.dihedral_label.clone(),
                strategy: strat,
                modification_summary: note,
                canonical_smiles: cmol.smiles.clone(),
                parent_rotatable_bonds: parent_prof.rotatable_bonds,
                analog_rotatable_bonds: c_prof.rotatable_bonds,
                locked_barrier_kcal: locked_barrier,
                barrier_gain_kcal: barrier_gain,
                analog_total_strain_teu: analog_teu_sum,
                preorganization_gain_kcal: preorg_gain_kcal,
                rocs_bioactive_combo: rocs.combo_tanimoto * 2.0,
                predicted_locked_pact: pred_pact,
                delta_pact_vs_parent: delta_pact,
                mw: c_prof.mw,
                clogp: c_prof.clogp,
                lle,
                sascore: c_sa.sa_score,
                mol: cmol,
            }
        })
        .collect();

    cands.sort_by(|a, b| {
        b.preorganization_gain_kcal
            .partial_cmp(&a.preorganization_gain_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in cands.iter_mut().enumerate() {
        c.rank = i + 1;
    }
    cands
}

/// Computes heavy-atom 3D coordinate RMSD (`Angstrom`) after centering (`Kabsch` superposition).
fn compute_centered_heavy_rmsd(mol: &Molecule, a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let n = mol.atoms.len().min(a.len()).min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (_, _, aligned_b, _) = crate::conformer3d::kabsch_align_3d(a, b);
    let mut sq = 0.0_f64;
    let mut cnt = 0usize;
    for idx in 0..n {
        if mol.atoms[idx].elem == Element::H {
            continue;
        }
        let dx = (a[idx][0] - aligned_b[idx][0]) as f64;
        let dy = (a[idx][1] - aligned_b[idx][1]) as f64;
        let dz = (a[idx][2] - aligned_b[idx][2]) as f64;
        sq += dx * dx + dy * dy + dz * dz;
        cnt += 1;
    }
    if cnt == 0 {
        0.0
    } else {
        (sq / (cnt as f64)).sqrt()
    }
}

/// Runs the complete `v2.9.0` `CSD`/`SMARTS` `1D` dihedral scan, bioactive 3D conformer strain
/// estimation, and automated rotatable-bond locking design for `mol`.
pub fn analyze_torsion_strain_and_locks(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    pose_source: BioactivePoseSource,
    enumerate_locks: bool,
) -> TorsionStrainReport {
    let charges = compute_gasteiger_charges(mol);
    let ens = generate_3d_conformer_ensemble(
        mol,
        &ConformerGenConfig {
            num_trial_conformers: 6,
            max_kept_conformers: 3,
            mmff_max_iters: 90,
            ..ConformerGenConfig::default()
        },
    );
    let free_min_coords3d = ens
        .conformers
        .first()
        .map(|c| c.coords3d.clone())
        .unwrap_or_else(|| evaluate_3d_summary_profile(mol, true).coords3d);

    // Determine bioactive 3D pose (PDB pocket-aligned vs 3D ROCS reference lead overlay)
    let (bioactive_coords3d, raw_pocket_dg_kcal, resolved_pose_label) =
        if let Some(p_idx) = pose_source.resolve_pocket_idx(&mol.target) {
            let preset = &CURATED_PDB_POCKETS[p_idx];
            let pdb = parse_pdb_structure(curated_pocket_pdb_text(p_idx))
                .expect("Curated PDB pocket valid");
            let pocket = analyze_binding_pocket(&pdb, None, 10.0);
            let p_res = score_ligand_in_pocket(mol, None, &pocket);
            (
                p_res.aligned_ligand_coords3d,
                p_res.total_dg_pocket_kcal,
                format!("PDB {} ({})", preset.pdb_id, preset.target_name),
            )
        } else if let Some(d_ref) = ds {
            // Find highest-pAct reference molecule in dataset
            let mut best_i = 0usize;
            let mut best_p = f64::NEG_INFINITY;
            for (mi, dm) in d_ref.mols.iter().enumerate() {
                if dm.pact > best_p {
                    best_p = dm.pact;
                    best_i = mi;
                }
            }
            if !d_ref.mols.is_empty() {
                let q_mol = &d_ref.mols[best_i];
                let q_s3d = &d_ref.shape3d[best_i];
                let h_s3d = evaluate_3d_summary_profile(mol, true);
                let rocs = align_rocs_3d(
                    q_mol,
                    &q_s3d.coords3d,
                    &q_s3d.gasteiger_charges,
                    &q_s3d.color_atoms,
                    mol,
                    &free_min_coords3d,
                    &charges,
                    &h_s3d.color_atoms,
                );
                let est_dg = -RT_LN10_KCAL * (if mol.pact > 0.5 { mol.pact } else { 7.0 });
                (
                    rocs.aligned_hit_coords,
                    est_dg,
                    format!("3D ROCS Overlay -> {}", q_mol.id),
                )
            } else {
                (free_min_coords3d.clone(), -8.5, "Free 3D Minimum".to_string())
            }
        } else {
            (free_min_coords3d.clone(), -8.5, "Free 3D Minimum".to_string())
        };

    let dihedrals = perceive_scannable_dihedrals(mol);
    let excl_12_13 = build_12_13_exclusion_set(mol);

    let bond_profiles: Vec<RotatableBondProfile> = dihedrals
        .iter()
        .enumerate()
        .map(|(ri, &(bi, q4))| {
            scan_single_rotatable_bond(
                ri,
                bi,
                q4,
                mol,
                &free_min_coords3d,
                &bioactive_coords3d,
                &charges,
                &excl_12_13,
            )
        })
        .collect();

    let rotatable_bond_count = bond_profiles.len();
    let strained_bond_count = bond_profiles
        .iter()
        .filter(|b| b.quality_band == TorsionQualityBand::StrainedRed)
        .count();
    let imhb_stabilized_bond_count = bond_profiles
        .iter()
        .filter(|b| b.has_bioactive_imhb)
        .count();

    let total_strain_teu: f64 = bond_profiles.iter().map(|b| b.bioactive_strain_teu).sum();
    let max_single_bond_teu: f64 = bond_profiles
        .iter()
        .map(|b| b.bioactive_strain_teu)
        .fold(0.0_f64, f64::max);
    let shoichet_filter_pass = total_strain_teu <= SHOICHET_TOTAL_TEU_CUTOFF
        && max_single_bond_teu <= SHOICHET_SINGLE_TEU_CUTOFF;

    let bioactive_strain_penalty_kcal: f64 = bond_profiles
        .iter()
        .map(|b| b.bioactive_strain_kcal)
        .sum::<f64>()
        .clamp(0.0, 14.0);
    let total_conformational_entropy_kcal: f64 = bond_profiles
        .iter()
        .map(|b| b.rotor_entropy_penalty_kcal)
        .sum();
    let strain_adjusted_dg_bind_kcal = raw_pocket_dg_kcal + 0.55 * bioactive_strain_penalty_kcal;
    let free_vs_bioactive_rmsd_a =
        compute_centered_heavy_rmsd(mol, &free_min_coords3d, &bioactive_coords3d);

    let strain_tier = if shoichet_filter_pass && bioactive_strain_penalty_kcal <= 3.0 {
        BioactiveStrainTier::LowStrainBioactive
    } else if total_strain_teu <= 8.5 && bioactive_strain_penalty_kcal <= 5.0 && max_single_bond_teu <= 2.4 {
        BioactiveStrainTier::ModerateInducedStrain
    } else {
        BioactiveStrainTier::HighStrainDecoyRisk
    };

    let locked_analogs = if enumerate_locks && !bond_profiles.is_empty() {
        design_rotatable_bond_locks(mol, &bioactive_coords3d, &bond_profiles)
    } else {
        Vec::new()
    };

    TorsionStrainReport {
        compound_id: mol.id.clone(),
        compound_name: mol.name.clone(),
        target: mol.target.clone(),
        canonical_smiles: mol.smiles.clone(),
        pose_source,
        resolved_pose_label,
        rotatable_bond_count,
        strained_bond_count,
        imhb_stabilized_bond_count,
        total_strain_teu,
        max_single_bond_teu,
        shoichet_filter_pass,
        bioactive_strain_penalty_kcal,
        total_conformational_entropy_kcal,
        raw_pocket_dg_kcal,
        strain_adjusted_dg_bind_kcal,
        free_vs_bioactive_rmsd_a,
        strain_tier,
        free_min_coords3d,
        bioactive_coords3d,
        gasteiger_charges: charges,
        bond_profiles,
        locked_analogs,
        mol: mol.clone(),
    }
}

/// Screens an entire `DatasetAnalysis` in parallel for `Shoichet` `TEU` & `Perola-Charifson` bioactive strain.
pub fn screen_dataset_torsion_strain(
    ds: &DatasetAnalysis,
    pose_source: BioactivePoseSource,
) -> Vec<DatasetTorsionScreenRow> {
    let mut rows: Vec<DatasetTorsionScreenRow> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(mi, m)| {
            let rep = analyze_torsion_strain_and_locks(m, Some(ds), pose_source, true);
            let (top_strat, top_gain) = rep
                .locked_analogs
                .first()
                .map(|c| (c.strategy.short_label().to_string(), c.preorganization_gain_kcal))
                .unwrap_or_else(|| ("Rigid Core".to_string(), 0.0));
            DatasetTorsionScreenRow {
                rank: mi + 1,
                mol_idx: mi,
                compound_id: m.id.clone(),
                compound_name: m.name.clone(),
                target: m.target.clone(),
                observed_pact: m.pact,
                rotatable_bonds: rep.rotatable_bond_count,
                strained_bonds: rep.strained_bond_count,
                total_strain_teu: rep.total_strain_teu,
                max_single_teu: rep.max_single_bond_teu,
                shoichet_pass: rep.shoichet_filter_pass,
                bioactive_strain_kcal: rep.bioactive_strain_penalty_kcal,
                strain_adjusted_dg_kcal: rep.strain_adjusted_dg_bind_kcal,
                strain_tier: rep.strain_tier,
                top_lock_strategy: top_strat,
                top_lock_gain_kcal: top_gain,
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        a.strain_adjusted_dg_kcal
            .partial_cmp(&b.strain_adjusted_dg_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    rows
}

/// Exports per-bond `CSD`/`SMARTS` dihedral scan & bioactive strain metrics + `36`-point curves to `.csv`.
pub fn export_torsion_profiles_csv(report: &TorsionStrainReport) -> String {
    let mut out = String::from(
        "rotor_idx,dihedral_label,rule_id,hierarchy_class,free_angle_deg,bioactive_angle_deg,nearest_csd_peak_deg,csd_dev_deg,rotational_barrier_kcal,accessible_wells,entropy_penalty_kcal,bioactive_strain_teu,bioactive_strain_kcal,bioactive_imhb,quality_band\n",
    );
    for b in &report.bond_profiles {
        out.push_str(&format!(
            "{},{},{},\"{}\",{:.1},{:.1},{:.1},{:.1},{:.3},{},{:.3},{:.3},{:.3},{},{}\n",
            b.rotor_idx + 1,
            b.dihedral_label,
            b.matched_rule_id,
            b.hierarchy_class,
            b.free_pose_angle_deg,
            b.bioactive_pose_angle_deg,
            b.nearest_csd_peak_deg,
            b.angular_deviation_from_csd_deg,
            b.rotational_barrier_kcal,
            b.accessible_wells_count,
            b.rotor_entropy_penalty_kcal,
            b.bioactive_strain_teu,
            b.bioactive_strain_kcal,
            b.has_bioactive_imhb,
            b.quality_band.short_label()
        ));
    }
    out
}

/// Exports conformationally locked rotatable-bond design candidates to `.csv`.
pub fn export_locked_analogs_csv(report: &TorsionStrainReport) -> String {
    let mut out = String::from(
        "rank,candidate_id,target_dihedral,strategy,parent_nrot,analog_nrot,locked_barrier_kcal,barrier_gain_kcal,analog_strain_teu,preorganization_gain_kcal,rocs_combo,predicted_pact,delta_pact,mw,clogp,lle,sascore,modification,smiles\n",
    );
    for c in &report.locked_analogs {
        out.push_str(&format!(
            "{},{},{},\"{}\",{},{},{:.2},{:.2},{:.2},{:.3},{:.3},{:.2},{:+.2},{:.1},{:.2},{:.2},{:.2},\"{}\",{}\n",
            c.rank,
            c.candidate_id,
            c.target_dihedral_label,
            c.strategy.short_label(),
            c.parent_rotatable_bonds,
            c.analog_rotatable_bonds,
            c.locked_barrier_kcal,
            c.barrier_gain_kcal,
            c.analog_total_strain_teu,
            c.preorganization_gain_kcal,
            c.rocs_bioactive_combo,
            c.predicted_locked_pact,
            c.delta_pact_vs_parent,
            c.mw,
            c.clogp,
            c.lle,
            c.sascore,
            c.modification_summary.replace('"', "'"),
            c.canonical_smiles
        ));
    }
    out
}

/// Exports both the free-solution global-minimum 3D conformer and the receptor-bound bioactive 3D pose as a 2-record `.sdf`.
pub fn format_torsion_poses_sdf(report: &TorsionStrainReport) -> String {
    let mut free_m = report.mol.clone();
    free_m.id = format!("{}_FREE_MIN", report.compound_id);
    let mut bio_m = report.mol.clone();
    bio_m.id = format!("{}_BIOACTIVE", report.compound_id);

    let free_sdf = format_3d_conformer_sdf(
        &free_m,
        &report.free_min_coords3d,
        &report.gasteiger_charges,
        0.0,
        &[("POSE_ROLE", "Free_Solution_Global_Minimum".to_string())],
    );
    let bio_sdf = format_3d_conformer_sdf(
        &bio_m,
        &report.bioactive_coords3d,
        &report.gasteiger_charges,
        report.bioactive_strain_penalty_kcal,
        &[
            ("POSE_ROLE", report.resolved_pose_label.clone()),
            ("TOTAL_STRAIN_TEU", format!("{:.3}", report.total_strain_teu)),
            ("MAX_SINGLE_TEU", format!("{:.3}", report.max_single_bond_teu)),
            ("SHOICHET_PASS", report.shoichet_filter_pass.to_string()),
        ],
    );
    format!("{free_sdf}{bio_sdf}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_benchmark_26;

    #[test]
    fn test_csd_smarts_hierarchy_and_36bin_dihedral_scanner() {
        for rule in &CSD_SMARTS_TORSION_LIBRARY {
            let q = compile_smarts(rule.smarts_4atom).expect(rule.rule_id);
            assert_eq!(q.atoms.len(), 4, "Rule {} must be a 4-atom dihedral SMARTS", rule.rule_id);
        }
        // Celecoxib + Erlotinib test: biaryl, sulfonamide, aryl-ether, and anilinoquinazoline CSD rules
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let rep = analyze_torsion_strain_and_locks(
            &ds.mols[0],
            Some(&ds),
            BioactivePoseSource::AutoTargetPocket,
            true,
        );
        assert!(rep.rotatable_bond_count >= 2);
        for bp in &rep.bond_profiles {
            assert_eq!(bp.scan_angles_deg.len(), TORSION_SCAN_BINS);
            assert_eq!(bp.total_energy_kcal_curve.len(), TORSION_SCAN_BINS);
            assert!(bp.rotational_barrier_kcal >= 0.25);
        }
        assert!(!rep.locked_analogs.is_empty());
        let p_csv = export_torsion_profiles_csv(&rep);
        let l_csv = export_locked_analogs_csv(&rep);
        let sdf = format_torsion_poses_sdf(&rep);
        assert_eq!(p_csv.lines().count(), rep.bond_profiles.len() + 1);
        assert_eq!(l_csv.lines().count(), rep.locked_analogs.len() + 1);
        assert!(sdf.contains("$$$$"));
    }
}
