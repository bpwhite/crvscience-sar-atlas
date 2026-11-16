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

//! Quantitative ADMET, Aqueous Solubility (`ESOL`), `BOILED-Egg` GI/BBB Permeation,
//! Preclinical Toxicology Rules (`Pfizer 3/75`, `GSK 4/400`, `Golden Triangle`), `hERG`,
//! and Cytochrome P450 (`CYP`) Soft-Spot Profiling:
//! - Delaney (2004) `ESOL`, *J. Chem. Inf. Comput. Sci.* 44(3), 1000-1005 (`https://pubmed.ncbi.nlm.nih.gov/15154768/`)
//! - Daina & Zoete (2016) `BOILED-Egg`, *ChemMedChem* 11(11), 1117-1121 (`https://pubmed.ncbi.nlm.nih.gov/27218427/`)
//! - Pfizer `3/75`, GSK `4/400` & Golden Triangle safety thresholds (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8622797/`,
//!   `https://www.kohulanr.com/ChemAudit/docs/user-guide/scoring/admet`)

use crate::mol::{Element, Molecule};
use crate::physchem::PhyschemProfile;
use crate::smarts::{compile_smarts, match_smarts};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Delaney (2004) `ESOL` aqueous solubility category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolubilityClass {
    HighlySoluble,
    Soluble,
    ModeratelySoluble,
    PoorlySoluble,
}

impl SolubilityClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::HighlySoluble => "Highly Soluble (>= -2)",
            Self::Soluble => "Soluble (-4..-2)",
            Self::ModeratelySoluble => "Moderate (-6..-4)",
            Self::PoorlySoluble => "Poorly Soluble (< -6)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HighlySoluble => "High",
            Self::Soluble => "Soluble",
            Self::ModeratelySoluble => "Moderate",
            Self::PoorlySoluble => "Poor",
        }
    }
}

/// Daina & Zoete (2016) `BOILED-Egg` passive gastrointestinal (`HIA`) and brain (`BBB`) region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoiledEggRegion {
    /// Yellow yolk: predicted passive blood-brain barrier (`BBB`) & gastrointestinal (`HIA`) permeant.
    YolkBbb,
    /// Egg white: predicted high passive gastrointestinal absorption (`HIA`, `Fa >= 0.30`), non-BBB.
    WhiteHia,
    /// Outside gray zone: low passive GI absorption and limited BBB access.
    OutsideLow,
}

impl BoiledEggRegion {
    pub fn label(self) -> &'static str {
        match self {
            Self::YolkBbb => "Yolk (BBB + HIA)",
            Self::WhiteHia => "White (HIA Passive)",
            Self::OutsideLow => "Outside (Low GI)",
        }
    }
}

/// Estimated `hERG` potassium channel inhibition cardiotoxicity risk tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HergRisk {
    Low,
    Moderate,
    High,
}

impl HergRisk {
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Moderate => "Moderate",
            Self::High => "High",
        }
    }
}

/// Complete quantitative ADMET and preclinical safety scorecard for a single molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdmetProfile {
    /// Aromatic heavy atom proportion `AP = aromatic_heavy / heavy_atoms` used in Delaney `ESOL`.
    pub aromatic_proportion: f64,
    /// Delaney (2004) `ESOL` `log10(S)` in `mol/L`.
    pub esol_logs: f64,
    /// Estimated aqueous solubility in `ug/mL` (`10^LogS * MW * 1000`).
    pub esol_ug_ml: f64,
    pub solubility_class: SolubilityClass,
    /// Daina & Zoete (2016) `BOILED-Egg` permeation region (`YolkBbb`, `WhiteHia`, or `OutsideLow`).
    pub boiled_egg: BoiledEggRegion,
    /// Predicted P-glycoprotein active efflux transporter substrate (`PGP+` blue dot vs `PGP-` red dot).
    pub pgp_substrate: bool,
    /// Pfizer `3/75` safety rule (`true` = PASS; `false` = elevated in vivo toxicity risk when `cLogP > 3 && TPSA < 75`).
    pub pfizer_3_75_pass: bool,
    /// GSK `4/400` ADMET rule (`true` = PASS when `MW <= 400 && cLogP <= 4`).
    pub gsk_4_400_pass: bool,
    /// Johnson `Golden Triangle` permeability/clearance window (`200 <= MW <= 450 && -0.5 <= cLogD7.4 <= 5.0`).
    pub golden_triangle: bool,
    /// Estimated `hERG` K+ channel inhibition risk based on basic amine `pKa` + lipophilicity.
    pub herg_risk: HergRisk,
    /// Count of hepatic `CYP450` metabolically labile sites (`O-alkyl`, `N-Me`, benzylic `CH2`, thioether, unsubstituted `para-aryl`).
    pub cyp_soft_spots: usize,
    /// Atom indices of `CYP450` metabolic soft spots for optional 2D inspection.
    pub cyp_soft_spot_atoms: Vec<usize>,
}

/// Tests whether `(tpsa, wlogp)` lies inside the Daina & Zoete (2016) `BOILED-Egg` White (`HIA`) ellipse.
pub fn in_boiled_egg_white(tpsa: f64, wlogp: f64) -> bool {
    point_in_tilted_ellipse(tpsa, wlogp, 71.05, 2.29, 71.05, 4.40, -10.0_f64.to_radians())
}

/// Tests whether `(tpsa, wlogp)` lies inside the Daina & Zoete (2016) `BOILED-Egg` Yolk (`BBB`) ellipse.
pub fn in_boiled_egg_yolk(tpsa: f64, wlogp: f64) -> bool {
    point_in_tilted_ellipse(tpsa, wlogp, 38.12, 3.07, 41.00, 2.70, -6.0_f64.to_radians())
}

fn point_in_tilted_ellipse(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
    theta_rad: f64,
) -> bool {
    let dx = x - cx;
    let dy = y - cy;
    let (s, c) = theta_rad.sin_cos();
    let ux = dx * c + dy * s;
    let uy = -dx * s + dy * c;
    (ux / rx).powi(2) + (uy / ry).powi(2) <= 1.0
}

/// Evaluates Delaney `ESOL` solubility, `BOILED-Egg` GI/BBB permeation, `P-gp` efflux,
/// `Pfizer 3/75`, `GSK 4/400`, `Golden Triangle`, `hERG` risk, and `CYP450` soft spots.
pub fn evaluate_admet(mol: &Molecule, prof: &PhyschemProfile) -> AdmetProfile {
    let heavy = prof.heavy_atoms.max(1) as f64;
    let arom_heavy = mol
        .atoms
        .iter()
        .filter(|a| a.elem != Element::H && a.aromatic)
        .count() as f64;
    let ap = (arom_heavy / heavy).clamp(0.0, 1.0);

    // Delaney (2004) ESOL equation:
    // LogS = 0.16 - 0.63 * cLogP - 0.0062 * MW + 0.066 * RB - 0.74 * AP
    let esol_logs = 0.16 - 0.63 * prof.clogp - 0.0062 * prof.mw
        + 0.066 * (prof.rotatable_bonds as f64)
        - 0.74 * ap;
    let esol_ug_ml = (10f64.powf(esol_logs.clamp(-12.0, 2.0)) * prof.mw * 1000.0).max(1e-4);

    let solubility_class = if esol_logs >= -2.0 {
        SolubilityClass::HighlySoluble
    } else if esol_logs >= -4.0 {
        SolubilityClass::Soluble
    } else if esol_logs >= -6.0 {
        SolubilityClass::ModeratelySoluble
    } else {
        SolubilityClass::PoorlySoluble
    };

    let boiled_egg = if in_boiled_egg_yolk(prof.tpsa, prof.clogp) {
        BoiledEggRegion::YolkBbb
    } else if in_boiled_egg_white(prof.tpsa, prof.clogp) {
        BoiledEggRegion::WhiteHia
    } else {
        BoiledEggRegion::OutsideLow
    };

    // P-glycoprotein efflux substrate heuristic (MW >= 400 with polar HBA rich or cationic basic center)
    let pgp_substrate = prof.mw >= 400.0
        && (prof.hba >= 6 || prof.est_basic_pka >= 8.0 || prof.tpsa >= 78.0);

    // Preclinical safety rules
    let pfizer_3_75_pass = !(prof.clogp > 3.0 && prof.tpsa < 75.0);
    let gsk_4_400_pass = prof.mw <= 400.0 && prof.clogp <= 4.0;
    let golden_triangle =
        (200.0..=450.0).contains(&prof.mw) && (-0.5..=5.0).contains(&prof.clogd_74);

    // hERG K+ channel liability (protonated basic nitrogen + aromatic lipophilicity)
    let herg_risk = if prof.est_basic_pka >= 8.0 && prof.clogp >= 4.0 && prof.aromatic_rings >= 2 {
        HergRisk::High
    } else if (prof.est_basic_pka >= 7.5 && prof.clogp >= 3.2)
        || (prof.clogp >= 4.8 && prof.tpsa < 65.0)
    {
        HergRisk::Moderate
    } else {
        HergRisk::Low
    };

    // CYP450 metabolic soft-spot SMARTS scan
    let soft_spot_patterns = [
        "[#6;a]-[#8]-[CH3,CH2]",        // Aryl O-dealkylation (methoxy/ethoxy)
        "[#7;X3]-[CH3]",                // N-demethylation
        "[CH2;!R;$(*c1ccccc1)]",        // Benzylic CH2 hydroxylation
        "[#16;X2]",                     // Thioether S-oxidation
    ];
    let mut soft_atoms = BTreeSet::new();
    let mut soft_sites = 0usize;
    for pat in soft_spot_patterns {
        if let Ok(q) = compile_smarts(pat) {
            let r = match_smarts(mol, &q);
            if r.matched {
                soft_sites += r.match_count;
                for a in r.union_atoms {
                    soft_atoms.insert(a);
                }
            }
        }
    }

    AdmetProfile {
        aromatic_proportion: ap,
        esol_logs,
        esol_ug_ml,
        solubility_class,
        boiled_egg,
        pgp_substrate,
        pfizer_3_75_pass,
        gsk_4_400_pass,
        golden_triangle,
        herg_risk,
        cyp_soft_spots: soft_sites,
        cyp_soft_spot_atoms: soft_atoms.into_iter().collect(),
    }
}
