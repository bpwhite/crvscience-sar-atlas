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

//! Physiological `pH 7.4` Site-Specific Micro-`pKa` Predictor, Coupled Henderson-Hasselbalch
//! Microstate Partition Function, Charge-State Protomer Enumerator, and `MolVS` Prototropic
//! Tautomer Scorer.
//!
//! Primary Literature Grounding:
//! - Ropp, Kaminsky, Yablonski & Durrant (2019), *J. Cheminform.* 11:14
//!   (`https://www.semanticscholar.org/paper/Dimorphite-DL%3A-an-open-source-program-for-the-of-Ropp-Kaminsky/f19c60c3ce9c286610e99ac367c4f6fc496b1181`):
//!   `Dimorphite-DL` drug-like small-molecule moiety ionization enumeration across target `pH` windows.
//! - Ropp et al. (2019), *J. Cheminform.* 11:34 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6534830/`):
//!   `Gypsum-DL` physiological ionization & `MolVS` prototropic tautomer preparation.
//! - Milletti et al. (2007), *J. Chem. Inf. Model.* 47(6), 2172–2181
//!   (`https://pubmed.ncbi.nlm.nih.gov/17910431/`): Micro-`pKa` electronic & heterocyclic benchmarks.
//! - Sitzmann, Ihlenfeldt & Nicklaus (2010), *J. Comput.-Aided Mol. Des.* 24, 521–551
//!   (`https://pubmed.ncbi.nlm.nih.gov/20512400/`): Prototropic tautomerism rules & canonicalization.
//! - Dolciami et al. (2022), *J. Cheminform.* 14:28 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC9148294/`):
//!   `MolVS` / RDKit `TautomerEnumerator` aromaticity & carbonyl stability scoring.

use crate::layout2d::compute_2d_coords;
use crate::mol::{BondOrder, Element, Hybridization, Molecule};
use crate::pharmacophore::compute_topological_distances;
use crate::physchem::evaluate_physchem;
use crate::sketcher::molecule_to_smiles;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Number of `pH` grid points (`0.0, 0.5, ..., 14.0`) in full titration & `logD(pH)` curves.
pub const PH_GRID_LEN: usize = 29;

/// Ionizable site acid/base transition direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IonSiteKind {
    /// `HA (charge 0) <-> A- (charge -1) + H+`
    Acidic,
    /// `BH+ (charge +1) <-> B (charge 0) + H+`
    Basic,
}

impl IonSiteKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Acidic => "Acid (HA -> A-)",
            Self::Basic => "Base (BH+ -> B)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Acidic => [34, 211, 238],  // Cyan (anionic when ionized)
            Self::Basic => [192, 132, 252],  // Violet (cationic when ionized)
        }
    }
}

/// Medicinal-chemistry physiological ionization state class at `pH 7.4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChargeClass74 {
    Neutral,
    Cationic,
    Anionic,
    Zwitterionic,
}

impl ChargeClass74 {
    pub fn label(self) -> &'static str {
        match self {
            Self::Neutral => "Neutral (0)",
            Self::Cationic => "Cationic (+)",
            Self::Anionic => "Anionic (-)",
            Self::Zwitterionic => "Zwitterion (+/-)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Neutral => [52, 211, 153],      // Emerald
            Self::Cationic => [192, 132, 252],    // Violet
            Self::Anionic => [34, 211, 238],      // Cyan
            Self::Zwitterionic => [251, 191, 36], // Amber
        }
    }
}

/// Perceived site-specific micro-`pKa` locus on a small molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IonizableSite {
    /// Atom index in `Molecule::atoms` where formal `+1` or `-1` charge resides upon ionization.
    pub atom_idx: usize,
    pub kind: IonSiteKind,
    #[serde(skip_deserializing)]
    pub moiety_name: &'static str,
    /// Intrinsic reference `pKa` before through-bond inductive attenuation.
    pub base_pka: f64,
    /// Electronic through-bond Hammett/Taft inductive shift (`Delta pKa`).
    pub inductive_shift: f64,
    /// Predicted site micro-`pKa` at `298 K` in aqueous solution.
    pub micro_pka: f64,
    /// Fractional ionization (`% ionized`) of this site at the evaluated target `pH`.
    pub ionized_fraction_at_ph: f64,
}

/// Enumerated charge-state protomer microspecies with Boltzmann/Henderson-Hasselbalch abundance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtomerSpecies {
    pub rank: usize,
    pub state_mask: u32,
    pub net_charge: i8,
    pub pos_centers: usize,
    pub neg_centers: usize,
    pub is_zwitterion: bool,
    pub state_label: String,
    /// Exact fractional population in `[0.0, 1.0]` at `target_ph`.
    pub fraction_at_target_ph: f64,
    /// Population across `pH = 0.0, 0.5, ..., 14.0` (`29` points) for interactive speciation curves.
    pub curve_fractions: Vec<f32>,
    pub canonical_smiles: String,
    /// Standalone 2D-depicted `Molecule` carrying explicit `+1` / `-1` formal charges on ionized atoms.
    pub mol: Molecule,
}

/// Enumerated prototropic tautomer scored by `MolVS` / Sitzmann thermodynamic rules.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TautomerSpecies {
    pub rank: usize,
    #[serde(skip_deserializing)]
    pub rule_name: &'static str,
    pub molvs_score: f64,
    /// Estimated free-energy offset relative to the canonical lowest-energy tautomer (`kcal/mol`, `298 K`).
    pub rel_energy_kcal: f64,
    /// Boltzmann equilibrium mole fraction (`[0.0, 1.0]`).
    pub boltzmann_fraction: f64,
    pub canonical_smiles: String,
    pub mol: Molecule,
}

/// Compact `Rayon`-precomputed physiological `pH 7.4` ionization summary stored on `DatasetAnalysis`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysiologicalChargeSummary {
    pub charge_class: ChargeClass74,
    pub dominant_net_charge: i8,
    pub mean_net_charge_74: f64,
    pub neutral_fraction_74: f64,
    pub cation_fraction_74: f64,
    pub anion_fraction_74: f64,
    pub zwitterion_fraction_74: f64,
    pub strongest_acid_pka: Option<f64>,
    pub strongest_base_pka: Option<f64>,
    pub isoelectric_pi: Option<f64>,
    pub microstate_logd_74: f64,
    pub gastric_logd_20: f64,
    pub major_protomers_74: usize,
    pub tautomer_count: usize,
    pub dominant_protomer_smiles: String,
}

/// Full interactive `pH`-dependent micro-`pKa`, protomer speciation, `logD(pH)`, and tautomer report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PkaProtomerReport {
    pub target_ph: f64,
    pub ph_grid: Vec<f32>,
    pub sites: Vec<IonizableSite>,
    pub protomers: Vec<ProtomerSpecies>,
    pub tautomers: Vec<TautomerSpecies>,
    pub mean_charge_curve: Vec<f32>,
    pub logd_curve: Vec<f32>,
    pub neutral_fraction_at_ph: f64,
    pub mean_charge_at_ph: f64,
    pub logd_at_ph: f64,
    pub gastric_logd_20: f64,
    pub jejunal_logd_65: f64,
    pub plasma_logd_74: f64,
    pub isoelectric_pi: Option<f64>,
    pub charge_class_at_ph: ChargeClass74,
}

fn is_carbonyl_c(mol: &Molecule, u: usize) -> bool {
    mol.atoms[u].elem == Element::C
        && !mol.atoms[u].aromatic
        && mol.adj[u].iter().any(|&(v, bi)| {
            mol.atoms[v].elem == Element::O
                && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
        })
}

fn is_sulfonyl_s(mol: &Molecule, u: usize) -> bool {
    if mol.atoms[u].elem != Element::S || mol.atoms[u].aromatic {
        return false;
    }
    mol.adj[u]
        .iter()
        .filter(|&&(v, bi)| {
            mol.atoms[v].elem == Element::O
                && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
        })
        .count()
        >= 2
}

fn is_nitro_n(mol: &Molecule, u: usize) -> bool {
    if mol.atoms[u].elem != Element::N {
        return false;
    }
    mol.adj[u]
        .iter()
        .filter(|&&(v, bi)| {
            mol.atoms[v].elem == Element::O
                && (mol.bonds[bi].order == BondOrder::Double
                    || mol.atoms[v].charge < 0
                    || mol.bonds[bi].kekule_order == 2)
        })
        .count()
        >= 2
}

/// Returns a standardized neutral copy of `mol` (protonating `[O-]`/`[S-]` anions not part of nitro
/// and deprotonating `[NH+]`/`[nH+]` non-quaternary cations) so micro-`pKa` perception is invariant
/// to whether the input SMILES was drawn charged or neutral.
pub fn standardize_neutral_parent(mol: &Molecule) -> Molecule {
    let mut neutral = mol.clone();
    for u in 0..neutral.atoms.len() {
        let is_nitro_o = neutral.atoms[u].elem == Element::O
            && neutral.adj[u].iter().any(|&(v, _)| is_nitro_n(&neutral, v));
        if is_nitro_o {
            continue;
        }
        let c = neutral.atoms[u].charge;
        if c < 0 && matches!(neutral.atoms[u].elem, Element::O | Element::S | Element::N) {
            neutral.atoms[u].charge = 0;
            neutral.atoms[u].explicit_h = 0;
            neutral.atoms[u].bracketed = false;
        } else if c > 0 && neutral.atoms[u].elem == Element::N && !is_nitro_n(&neutral, u) {
            // Only neutralize if heavy bond order sum <= 3 (not a permanent quaternary R4N+!)
            let heavy_bo: u8 = neutral.adj[u]
                .iter()
                .filter(|&&(v, _)| neutral.atoms[v].elem != Element::H)
                .map(|&(_, bi)| neutral.bonds[bi].kekule_order)
                .sum();
            if heavy_bo <= 3 {
                neutral.atoms[u].charge = 0;
                neutral.atoms[u].explicit_h = 0;
                neutral.atoms[u].bracketed = false;
            }
        }
    }
    neutral.sanitize();
    neutral.smiles = molecule_to_smiles(&neutral);
    neutral
}

/// Computes the through-bond Taft/Hammett electron-withdrawing inductive parameter (`sigma_sum >= 0`)
/// experienced at atom `site_u` from neighboring electronegative groups (`d = 2..=4` bonds away),
/// excluding atoms `exclude_set` belonging to the ionizable functional group itself.
fn through_bond_ewg_shift(
    mol: &Molecule,
    apsp: &[Vec<u8>],
    site_u: usize,
    exclude_set: &[usize],
) -> f64 {
    let mut ewg_sum = 0.0f64;
    for v in 0..mol.atoms.len() {
        if v == site_u || exclude_set.contains(&v) {
            continue;
        }
        let d = apsp[site_u][v];
        if !(2..=4).contains(&d) {
            continue;
        }
        // Attenuation across sigma bonds: d=2 (beta) -> 1.0, d=3 (gamma) -> 0.42, d=4 (delta) -> 0.16
        let atten = match d {
            2 => 1.00,
            3 => 0.42,
            4 => 0.16,
            _ => 0.0,
        };

        let a = &mol.atoms[v];
        let group_ewg = if is_sulfonyl_s(mol, v) {
            3.25 // Beta -SO2R drops amine pKa by ~3.2 units (e.g. Lapatinib NH-CH2-CH2-SO2Me -> pKa ~7.3!)
        } else if is_nitro_n(mol, v) {
            3.40
        } else if is_carbonyl_c(mol, v) {
            1.75
        } else {
            match a.elem {
                Element::F => 1.05, // -CF3 (3 F at d=3 -> 3 * 1.05 * 0.42 = 1.32)
                Element::Cl => 0.95,
                Element::Br => 0.90,
                Element::I => 0.75,
                Element::O => {
                    if a.in_ring && !a.aromatic {
                        // Saturated cyclic ether (`-O-` in morpholine sits at gamma distance d=3 where atten=0.42)
                        if d == 3 {
                            6.07
                        } else {
                            2.55
                        }
                    } else if !is_carbonyl_oxygen(mol, v) && !is_sulfonyl_oxygen(mol, v) {
                        1.65 // Beta -OH / -OR ether
                    } else {
                        0.0
                    }
                }
                Element::N => {
                    if !is_nitro_n(mol, v) {
                        // Piperazine second nitrogen (`d = 3` bonds across CH2-CH2 -> wait: in 6-ring piperazine
                        // N1 and N4 are `d = 3` bonds apart along BOTH sides!). Let's calibrate so N-arylpiperazine
                        // distal N shifts by ~ -1.95 units (10.15 -> 8.20, matching Clozapine/Loxapine/Fluphenazine!)
                        let bonded_to_aryl = mol.adj[v].iter().any(|&(w, _)| mol.atoms[w].aromatic);
                        if bonded_to_aryl {
                            4.50 // * 0.42 at d=3 = 1.89!
                        } else {
                            3.10 // * 0.42 at d=3 = 1.30
                        }
                    } else {
                        0.0
                    }
                }
                Element::C => {
                    if a.aromatic && d == 2 {
                        0.55 // Benzylic substituent inductive pull
                    } else if a.hyb == Hybridization::Sp && d == 2 {
                        1.45 // Propargylic / alpha-nitrile pull
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            }
        };
        ewg_sum += group_ewg * atten;
    }
    ewg_sum
}

fn is_carbonyl_oxygen(mol: &Molecule, o_idx: usize) -> bool {
    mol.atoms[o_idx].elem == Element::O
        && mol.adj[o_idx].iter().any(|&(c, bi)| {
            mol.atoms[c].elem == Element::C
                && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
        })
}

fn is_sulfonyl_oxygen(mol: &Molecule, o_idx: usize) -> bool {
    mol.atoms[o_idx].elem == Element::O
        && mol.adj[o_idx].iter().any(|&(s, bi)| {
            mol.atoms[s].elem == Element::S
                && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
        })
}

/// Perceives all physiologically relevant acidic and basic micro-`pKa` centers (`0.5 <= pKa <= 13.8`)
/// on a standardized neutral parent `mol`.
pub fn perceive_ionizable_sites(mol: &Molecule) -> Vec<IonizableSite> {
    let n = mol.atoms.len();
    let apsp = compute_topological_distances(mol);
    let mut claimed = vec![false; n];
    let mut sites: Vec<IonizableSite> = Vec::new();

    // -------------------------------------------------------------------------
    // A. Acidic Centers (`HA -> A- + H+`, EWG LOWERS micro_pka)
    // -------------------------------------------------------------------------
    // A1. 1H-Tetrazole ring (`c1nnnn1H`, base pKa = 4.90)
    for r in &mol.rings {
        if r.len() == 5 && r.iter().all(|&u| mol.atoms[u].aromatic) {
            let n_cnt = r.iter().filter(|&&u| mol.atoms[u].elem == Element::N).count();
            let c_cnt = r.iter().filter(|&&u| mol.atoms[u].elem == Element::C).count();
            if n_cnt == 4 && c_cnt == 1 {
                if let Some(&nh_u) = r
                    .iter()
                    .find(|&&u| mol.atoms[u].elem == Element::N && mol.atoms[u].total_h() > 0)
                {
                    for &ru in r {
                        claimed[ru] = true;
                    }
                    let shift = -0.35 * through_bond_ewg_shift(mol, &apsp, nh_u, r);
                    let mpka = (4.90 + shift).clamp(2.5, 6.8);
                    sites.push(IonizableSite {
                        atom_idx: nh_u,
                        kind: IonSiteKind::Acidic,
                        moiety_name: "1H-Tetrazole",
                        base_pka: 4.90,
                        inductive_shift: shift,
                        micro_pka: mpka,
                        ionized_fraction_at_ph: 0.0,
                    });
                }
            }
        }
    }

    // A2. Carboxylic, Sulfonic, and Phosphonic Acids (`C/S/P(=O)OH`)
    for u in 0..n {
        if claimed[u] || !matches!(mol.atoms[u].elem, Element::C | Element::S | Element::P) {
            continue;
        }
        let mut oxo = Vec::new();
        let mut oh_list = Vec::new();
        for &(v, bi) in &mol.adj[u] {
            if mol.atoms[v].elem == Element::O {
                if mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2 {
                    oxo.push(v);
                } else if mol.atoms[v].total_h() > 0 && mol.heavy_degree(v) == 1 {
                    oh_list.push(v);
                }
            }
        }
        if !oxo.is_empty() && !oh_list.is_empty() {
            let mut grp = vec![u];
            grp.extend(&oxo);
            grp.extend(&oh_list);
            for &g in &grp {
                claimed[g] = true;
            }
            match mol.atoms[u].elem {
                Element::C => {
                    let aryl_attached = mol
                        .adj[u]
                        .iter()
                        .any(|&(v, _)| !grp.contains(&v) && mol.atoms[v].aromatic);
                    let base = if aryl_attached { 4.20 } else { 4.75 };
                    let shift = (-0.30 * through_bond_ewg_shift(mol, &apsp, u, &grp)).clamp(-2.4, 0.5);
                    sites.push(IonizableSite {
                        atom_idx: oh_list[0],
                        kind: IonSiteKind::Acidic,
                        moiety_name: if aryl_attached {
                            "Benzoic / Heteroaryl Acid"
                        } else {
                            "Aliphatic Carboxylic Acid"
                        },
                        base_pka: base,
                        inductive_shift: shift,
                        micro_pka: (base + shift).clamp(1.2, 5.5),
                        ionized_fraction_at_ph: 0.0,
                    });
                }
                Element::S => {
                    sites.push(IonizableSite {
                        atom_idx: oh_list[0],
                        kind: IonSiteKind::Acidic,
                        moiety_name: "Sulfonic Acid",
                        base_pka: -1.20,
                        inductive_shift: 0.0,
                        micro_pka: -1.20,
                        ionized_fraction_at_ph: 0.0,
                    });
                }
                Element::P => {
                    sites.push(IonizableSite {
                        atom_idx: oh_list[0],
                        kind: IonSiteKind::Acidic,
                        moiety_name: "Phosphonic Acid (pKa1)",
                        base_pka: 2.15,
                        inductive_shift: 0.0,
                        micro_pka: 2.15,
                        ionized_fraction_at_ph: 0.0,
                    });
                    if oh_list.len() >= 2 {
                        sites.push(IonizableSite {
                            atom_idx: oh_list[1],
                            kind: IonSiteKind::Acidic,
                            moiety_name: "Phosphonic Acid (pKa2)",
                            base_pka: 7.20,
                            inductive_shift: 0.0,
                            micro_pka: 7.20,
                            ionized_fraction_at_ph: 0.0,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // A3. Acylsulfonamides, Cyclic Imides / Thiazolidinediones / Rhodanines, and Sulfonamides (`NH`)
    for u in 0..n {
        if claimed[u] || mol.atoms[u].elem != Element::N || mol.atoms[u].total_h() == 0 {
            continue;
        }
        let mut carbonyl_nbrs = 0usize;
        let mut thiocarbonyl_nbrs = 0usize;
        let mut sulfonyl_nbrs = 0usize;
        let mut grp = vec![u];
        for &(v, _) in &mol.adj[u] {
            if is_carbonyl_c(mol, v) {
                carbonyl_nbrs += 1;
                grp.push(v);
            } else if is_sulfonyl_s(mol, v) {
                sulfonyl_nbrs += 1;
                grp.push(v);
            } else if mol.atoms[v].elem == Element::C
                && mol.adj[v].iter().any(|&(w, bi)| {
                    mol.atoms[w].elem == Element::S
                        && (mol.bonds[bi].order == BondOrder::Double
                            || mol.bonds[bi].kekule_order == 2)
                })
            {
                thiocarbonyl_nbrs += 1;
                grp.push(v);
            }
        }

        if carbonyl_nbrs >= 1 && sulfonyl_nbrs >= 1 {
            claimed[u] = true;
            sites.push(IonizableSite {
                atom_idx: u,
                kind: IonSiteKind::Acidic,
                moiety_name: "Acylsulfonamide",
                base_pka: 4.45,
                inductive_shift: 0.0,
                micro_pka: 4.45,
                ionized_fraction_at_ph: 0.0,
            });
        } else if carbonyl_nbrs + thiocarbonyl_nbrs >= 2 {
            claimed[u] = true;
            let base = if thiocarbonyl_nbrs >= 1 { 6.45 } else { 8.30 };
            sites.push(IonizableSite {
                atom_idx: u,
                kind: IonSiteKind::Acidic,
                moiety_name: if thiocarbonyl_nbrs >= 1 {
                    "Thiazolidinedione / Rhodanine Imide"
                } else {
                    "Cyclic / Acyclic Imide"
                },
                base_pka: base,
                inductive_shift: 0.0,
                micro_pka: base,
                ionized_fraction_at_ph: 0.0,
            });
        } else if sulfonyl_nbrs >= 1 && carbonyl_nbrs == 0 {
            claimed[u] = true;
            let shift = (-0.32 * through_bond_ewg_shift(mol, &apsp, u, &grp)).clamp(-2.2, 0.2);
            sites.push(IonizableSite {
                atom_idx: u,
                kind: IonSiteKind::Acidic,
                moiety_name: "Sulfonamide (-SO2NHR)",
                base_pka: 10.10,
                inductive_shift: shift,
                micro_pka: (10.10 + shift).clamp(6.8, 11.2),
                ionized_fraction_at_ph: 0.0,
            });
        }
    }

    // A4. Phenols / Enols (`Ar-OH` or `C=C-OH`) and Thiols (`-SH`)
    for u in 0..n {
        if claimed[u] {
            continue;
        }
        let a = &mol.atoms[u];
        if a.elem == Element::O && a.total_h() > 0 && mol.heavy_degree(u) == 1 {
            let Some(&(c_anc, _)) = mol.adj[u].iter().find(|&&(v, _)| mol.atoms[v].elem != Element::H) else {
                continue;
            };
            if mol.atoms[c_anc].aromatic {
                claimed[u] = true;
                let shift = (-0.45 * through_bond_ewg_shift(mol, &apsp, u, &[u, c_anc])).clamp(-3.2, 0.4);
                sites.push(IonizableSite {
                    atom_idx: u,
                    kind: IonSiteKind::Acidic,
                    moiety_name: "Phenolic / Heteroaryl -OH",
                    base_pka: 9.95,
                    inductive_shift: shift,
                    micro_pka: (9.95 + shift).clamp(5.8, 11.2),
                    ionized_fraction_at_ph: 0.0,
                });
            } else if mol.atoms[c_anc].hyb == Hybridization::Sp2 && !is_carbonyl_c(mol, c_anc) {
                claimed[u] = true;
                sites.push(IonizableSite {
                    atom_idx: u,
                    kind: IonSiteKind::Acidic,
                    moiety_name: "Conjugated Enol",
                    base_pka: 8.60,
                    inductive_shift: 0.0,
                    micro_pka: 8.60,
                    ionized_fraction_at_ph: 0.0,
                });
            }
        } else if a.elem == Element::S && a.total_h() > 0 && mol.heavy_degree(u) == 1 {
            let Some(&(c_anc, _)) = mol.adj[u].iter().find(|&&(v, _)| mol.atoms[v].elem != Element::H) else {
                continue;
            };
            claimed[u] = true;
            let is_ar = mol.atoms[c_anc].aromatic;
            let base = if is_ar { 6.60 } else { 10.30 };
            sites.push(IonizableSite {
                atom_idx: u,
                kind: IonSiteKind::Acidic,
                moiety_name: if is_ar { "Thiophenol" } else { "Aliphatic Thiol" },
                base_pka: base,
                inductive_shift: 0.0,
                micro_pka: base,
                ionized_fraction_at_ph: 0.0,
            });
        }
    }

    // -------------------------------------------------------------------------
    // B. Basic Centers (`BH+ -> B + H+`, EWG ALSO LOWERS conjugate-acid micro_pka!)
    // -------------------------------------------------------------------------
    // B1. Amidines & Guanidines (`N=C-N` non-aromatic sp2 carbon with zero =O/=S)
    for u in 0..n {
        if mol.atoms[u].elem == Element::C && !mol.atoms[u].aromatic {
            let has_oxo = mol.adj[u].iter().any(|&(v, bi)| {
                matches!(mol.atoms[v].elem, Element::O | Element::S)
                    && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
            });
            if has_oxo {
                continue;
            }
            let mut imine_n = Vec::new();
            let mut single_n = Vec::new();
            for &(v, bi) in &mol.adj[u] {
                if mol.atoms[v].elem == Element::N && !mol.atoms[v].aromatic {
                    if mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2 {
                        imine_n.push(v);
                    } else {
                        single_n.push(v);
                    }
                }
            }
            if !imine_n.is_empty() && !single_n.is_empty() {
                let target_n = imine_n[0];
                if !claimed[target_n] {
                    claimed[target_n] = true;
                    for &sn in &single_n {
                        // Keep distal piperazine N unclaimed if `sn` is a bridgehead N in a 6-ring!
                        if !mol.atoms[sn].in_ring || mol.rings.iter().any(|r| r.contains(&u) && r.contains(&sn)) {
                            claimed[sn] = true;
                        } else {
                            claimed[sn] = true;
                        }
                    }
                    let cyclic_amidine = mol.atoms[u].in_ring && mol.atoms[target_n].in_ring;
                    let base = if single_n.len() >= 2 {
                        12.80
                    } else if cyclic_amidine {
                        // Cyclic diaryl-fused amidines (e.g. Clozapine / Loxapine C=N in 7-ring) have lower basicity (~5.5)
                        5.40
                    } else {
                        11.40
                    };
                    sites.push(IonizableSite {
                        atom_idx: target_n,
                        kind: IonSiteKind::Basic,
                        moiety_name: if single_n.len() >= 2 {
                            "Guanidine"
                        } else if cyclic_amidine {
                            "Cyclic Amidine"
                        } else {
                            "Acyclic Amidine"
                        },
                        base_pka: base,
                        inductive_shift: 0.0,
                        micro_pka: base,
                        ionized_fraction_at_ph: 0.0,
                    });
                }
            }
        }
    }

    // B2. Aliphatic Amines, Anilines, and Basic Aromatic N-Heterocycles (Quinazoline, Pyridine, Imidazole)
    let mut quinazoline_rings_claimed: BTreeSet<Vec<usize>> = BTreeSet::new();

    for u in 0..n {
        if claimed[u] || mol.atoms[u].elem != Element::N || is_nitro_n(mol, u) {
            continue;
        }
        let a = &mol.atoms[u];
        if a.aromatic {
            // Only lone-pair sp2 aromatic `n` (`total_h == 0 && heavy_degree == 2`) is basic
            // (pyrrolic `[nH]` lone pair is part of the aromatic sextet and non-basic!)
            if a.total_h() == 0 && mol.heavy_degree(u) == 2 {
                // Check smallest ring containing `u`
                let ring_opt = mol
                    .rings
                    .iter()
                    .filter(|r| r.contains(&u) && r.iter().all(|&rv| mol.atoms[rv].aromatic))
                    .min_by_key(|r| r.len());
                let Some(ring) = ring_opt else {
                    continue;
                };
                let mut sorted_r = ring.clone();
                sorted_r.sort_unstable();

                let n_in_ring = ring
                    .iter()
                    .filter(|&&rv| mol.atoms[rv].elem == Element::N)
                    .count();
                let o_s_in_ring = ring
                    .iter()
                    .filter(|&&rv| matches!(mol.atoms[rv].elem, Element::O | Element::S))
                    .count();

                if ring.len() == 5 {
                    let has_pyrrolic_nh = ring
                        .iter()
                        .any(|&rv| mol.atoms[rv].elem == Element::N && mol.atoms[rv].total_h() > 0);
                    if n_in_ring == 2 && o_s_in_ring == 0 && has_pyrrolic_nh {
                        // Imidazole / Benzimidazole (`pKa ~ 6.95` or `5.60` when benzo-fused)
                        let benzo = ring
                            .iter()
                            .filter(|&&rv| mol.atoms[rv].ring_sizes.len() >= 2)
                            .count()
                            >= 2;
                        let base = if benzo { 5.60 } else { 6.95 };
                        let shift = (-0.45 * through_bond_ewg_shift(mol, &apsp, u, ring)).clamp(-2.5, 0.5);
                        claimed[u] = true;
                        sites.push(IonizableSite {
                            atom_idx: u,
                            kind: IonSiteKind::Basic,
                            moiety_name: if benzo { "Benzimidazole N3" } else { "Imidazole N3" },
                            base_pka: base,
                            inductive_shift: shift,
                            micro_pka: (base + shift).clamp(2.0, 8.2),
                            ionized_fraction_at_ph: 0.0,
                        });
                    } else if n_in_ring == 2 && o_s_in_ring == 0 {
                        // N-substituted pyrazole (`pKa ~ 2.5`)
                        claimed[u] = true;
                        sites.push(IonizableSite {
                            atom_idx: u,
                            kind: IonSiteKind::Basic,
                            moiety_name: "Pyrazole N2",
                            base_pka: 2.50,
                            inductive_shift: 0.0,
                            micro_pka: 2.50,
                            ionized_fraction_at_ph: 0.0,
                        });
                    }
                    // Isoxazole / Oxazole / Thiazole (`pKa < 2.5`): skip oxazole/isoxazole (< 1.3), include thiazole (2.5)
                } else if ring.len() == 6 {
                    // For pyrimidine / quinazoline (2 nitrogens in the same 6-ring), only ONE nitrogen
                    // protonates in the physiological `pH 1..14` window (`N1` of 4-aminoquinazoline, `pKa ~ 5.35`!)
                    if n_in_ring >= 2 {
                        if !quinazoline_rings_claimed.insert(sorted_r) {
                            continue;
                        }
                        let has_exocyclic_nh = ring.iter().any(|&rv| {
                            mol.adj[rv].iter().any(|&(ext, _)| {
                                !ring.contains(&ext) && mol.atoms[ext].elem == Element::N
                            })
                        });
                        let base = if has_exocyclic_nh { 5.35 } else { 2.10 };
                        claimed[u] = true;
                        sites.push(IonizableSite {
                            atom_idx: u,
                            kind: IonSiteKind::Basic,
                            moiety_name: if has_exocyclic_nh {
                                "4-Aminoquinazoline / Aminopyrimidine N1"
                            } else {
                                "Pyrimidine / Diazine N"
                            },
                            base_pka: base,
                            inductive_shift: 0.0,
                            micro_pka: base,
                            ionized_fraction_at_ph: 0.0,
                        });
                    } else if n_in_ring == 1 {
                        // Pyridine / Quinoline (`pKa ~ 5.20`)
                        let shift = (-0.50 * through_bond_ewg_shift(mol, &apsp, u, ring)).clamp(-3.2, 0.8);
                        claimed[u] = true;
                        sites.push(IonizableSite {
                            atom_idx: u,
                            kind: IonSiteKind::Basic,
                            moiety_name: "Pyridine / Quinoline N",
                            base_pka: 5.20,
                            inductive_shift: shift,
                            micro_pka: (5.20 + shift).clamp(1.2, 6.8),
                            ionized_fraction_at_ph: 0.0,
                        });
                    }
                }
            }
        } else {
            // Non-aromatic nitrogen: skip amides, ureas, sulfonamides, nitriles (`sp`), imines (`sp2` unless handled)
            if a.hyb == Hybridization::Sp {
                continue;
            }
            let adj_ewg_acyl = mol
                .adj[u]
                .iter()
                .any(|&(v, _)| is_carbonyl_c(mol, v) || is_sulfonyl_s(mol, v));
            if adj_ewg_acyl {
                continue;
            }
            let has_double_bond = mol.adj[u].iter().any(|&(_, bi)| {
                mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2
            });
            if has_double_bond {
                continue;
            }

            let aryl_nbrs = mol
                .adj[u]
                .iter()
                .filter(|&&(v, _)| mol.atoms[v].aromatic)
                .count();
            let sp2_non_arom_nbrs = mol
                .adj[u]
                .iter()
                .filter(|&&(v, _)| !mol.atoms[v].aromatic && mol.atoms[v].hyb == Hybridization::Sp2)
                .count();

            if aryl_nbrs >= 2 || sp2_non_arom_nbrs >= 1 {
                // Diarylamine (`Ar-NH-Ar` like 4-anilinoquinazoline `NH` or phenothiazine `N10`, or amidine-bridgehead piperazine `N`!): non-basic (`pKa < 1.5`)
                continue;
            } else if aryl_nbrs == 1 {
                // Mono-aryl aniline / N-alkylaniline (`pKa ~ 4.60`)
                // Note: in N-arylpiperazine (`Ar-N1CCN4(R)CC1`), the Ar-attached N1 has `pKa ~ 1.5` when distal N4 is protonated;
                // check if in piperazine with a more basic aliphatic N!
                let in_piperazine = mol.rings.iter().any(|r| {
                    r.len() == 6
                        && r.contains(&u)
                        && r.iter().filter(|&&rv| mol.atoms[rv].elem == Element::N).count() == 2
                });
                if in_piperazine {
                    continue;
                }
                claimed[u] = true;
                let shift = (-0.55 * through_bond_ewg_shift(mol, &apsp, u, &[u])).clamp(-3.0, 0.5);
                sites.push(IonizableSite {
                    atom_idx: u,
                    kind: IonSiteKind::Basic,
                    moiety_name: "Aniline (-NHAr)",
                    base_pka: 4.60,
                    inductive_shift: shift,
                    micro_pka: (4.60 + shift).clamp(1.0, 5.5),
                    ionized_fraction_at_ph: 0.0,
                });
            } else {
                // Aliphatic Primary / Secondary / Tertiary Amine (`RNH2`, `R2NH`, `R3N`)!
                let hdeg = mol.heavy_degree(u);
                let base = match hdeg {
                    1 => 10.55,
                    2 => 10.65,
                    _ => 10.15,
                };
                let shift = (-1.0 * through_bond_ewg_shift(mol, &apsp, u, &[u])).clamp(-4.6, 0.2);
                let mpka = (base + shift).clamp(4.2, 11.2);
                let m_name = if a.in_ring {
                    let in_morph = mol.rings.iter().any(|r| {
                        r.contains(&u) && r.iter().any(|&rv| mol.atoms[rv].elem == Element::O)
                    });
                    let in_pipz = mol.rings.iter().any(|r| {
                        r.contains(&u)
                            && r.iter().filter(|&&rv| mol.atoms[rv].elem == Element::N).count() >= 2
                    });
                    if in_morph {
                        "Morpholine Basic N"
                    } else if in_pipz {
                        "Piperazine Basic N"
                    } else {
                        "Piperidine / Pyrrolidine Basic N"
                    }
                } else {
                    match hdeg {
                        1 => "Primary Aliphatic Amine",
                        2 => "Secondary Aliphatic Amine",
                        _ => "Tertiary Aliphatic Amine",
                    }
                };
                claimed[u] = true;
                sites.push(IonizableSite {
                    atom_idx: u,
                    kind: IonSiteKind::Basic,
                    moiety_name: m_name,
                    base_pka: base,
                    inductive_shift: shift,
                    micro_pka: mpka,
                    ionized_fraction_at_ph: 0.0,
                });
            }
        }
    }

    // Retain up to 6 sites closest to physiological pH `7.4` (`|pKa - 7.4|` ascending) so `2^K <= 64`
    sites.sort_by(|a, b| {
        (a.micro_pka - 7.4)
            .abs()
            .partial_cmp(&(b.micro_pka - 7.4).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sites.truncate(6);
    sites
}

fn build_protomer_molecule(
    neutral_mol: &Molecule,
    sites: &[IonizableSite],
    mask: u32,
    rank: usize,
) -> Molecule {
    let mut pm = neutral_mol.clone();
    pm.id = format!("{}-PROT{}", neutral_mol.id, rank + 1);

    for (si, site) in sites.iter().enumerate() {
        let is_protonated = ((mask >> si) & 1) == 1;
        let u = site.atom_idx;
        match site.kind {
            IonSiteKind::Acidic => {
                if !is_protonated {
                    // Ionized anion: `A-` (`charge = -1`)
                    pm.atoms[u].charge = -1;
                    pm.atoms[u].bracketed = true;
                    pm.atoms[u].explicit_h = 0;
                }
            }
            IonSiteKind::Basic => {
                if is_protonated {
                    // Ionized conjugate acid cation: `BH+` (`charge = +1`)
                    pm.atoms[u].charge = 1;
                    pm.atoms[u].bracketed = true;
                    if pm.atoms[u].aromatic {
                        // Pyridinium / quinazolinium `[nH+]`: explicitly carry 1 H on sp2 ring N+
                        pm.atoms[u].explicit_h = 1;
                    } else {
                        pm.atoms[u].explicit_h = 0;
                    }
                }
            }
        }
    }
    pm.sanitize();
    pm.smiles = molecule_to_smiles(&pm);
    pm
}

/// Enumerates prototropic tautomers via `1,3`-triad (`HX - Y = Z  <->  X = Y - ZH`) and `1,5`
/// keto-enol / lactam-lactim / thione-thiol / annular-azole shifts, scoring by `MolVS` rules.
pub fn enumerate_tautomers(neutral_mol: &Molecule) -> Vec<TautomerSpecies> {
    let mut candidates: Vec<(&'static str, Molecule)> =
        vec![("Input / Canonical Tautomer", neutral_mol.clone())];
    let mut seen_smiles: BTreeSet<String> = BTreeSet::new();
    let base_smi = molecule_to_smiles(neutral_mol);
    if !base_smi.is_empty() {
        seen_smiles.insert(base_smi);
    }

    let n = neutral_mol.atoms.len();

    // 1. General 1,3-Prototropic Triad Scan: `X(with H) --single-- Y ==double== Z`
    //    (covers amide/pyridone lactam<->lactim, thione<->thiol, imine<->enamine, keto<->enol!)
    for y in 0..n {
        if candidates.len() >= 12 {
            break;
        }
        // Avoid breaking intact 6-membered all-carbon benzene rings unless ketone/phenol tautomers
        for &(x, bx) in &neutral_mol.adj[y] {
            if neutral_mol.atoms[x].total_h() == 0 || neutral_mol.atoms[x].charge != 0 {
                continue;
            }
            if neutral_mol.bonds[bx].kekule_order != 1 || neutral_mol.bonds[bx].aromatic {
                continue;
            }
            // Only mobile protons on O, N, S, or alpha-CH adjacent to two carbonyls (`1,3-diketone`!)
            let x_el = neutral_mol.atoms[x].elem;
            let x_is_active_ch = x_el == Element::C
                && !neutral_mol.atoms[x].aromatic
                && neutral_mol
                    .adj[x]
                    .iter()
                    .filter(|&&(w, _)| is_carbonyl_c(neutral_mol, w))
                    .count()
                    >= 2;
            if !matches!(x_el, Element::N | Element::O | Element::S) && !x_is_active_ch {
                continue;
            }

            for &(z, bz) in &neutral_mol.adj[y] {
                if z == x || neutral_mol.atoms[z].charge != 0 {
                    continue;
                }
                if neutral_mol.bonds[bz].kekule_order != 2 || neutral_mol.bonds[bz].aromatic {
                    continue;
                }
                let z_el = neutral_mol.atoms[z].elem;
                if !matches!(z_el, Element::O | Element::N | Element::S | Element::C) {
                    continue;
                }
                // Skip simple unactivated aliphatic ketone -> unactivated vinyl alcohol (`C-C=O`) unless `x` is active `CH2` or heteroatom
                if x_el == Element::C && !x_is_active_ch {
                    continue;
                }
                if z_el == Element::C && !matches!(x_el, Element::O | Element::N) {
                    continue;
                }

                let mut t_mol = neutral_mol.clone();
                t_mol.bonds[bx].order = BondOrder::Double;
                t_mol.bonds[bx].kekule_order = 2;
                t_mol.bonds[bx].aromatic = false;
                t_mol.bonds[bz].order = BondOrder::Single;
                t_mol.bonds[bz].kekule_order = 1;
                t_mol.bonds[bz].aromatic = false;
                for &idx in &[x, y, z] {
                    t_mol.atoms[idx].explicit_h = 0;
                    t_mol.atoms[idx].implicit_h = 0;
                    t_mol.atoms[idx].bracketed = false;
                }
                t_mol.sanitize();
                compute_2d_coords(&mut t_mol);
                let can = molecule_to_smiles(&t_mol);
                if !can.is_empty() && seen_smiles.insert(can.clone()) {
                    t_mol.smiles = can;
                    let rule_lbl = match (x_el, z_el) {
                        (Element::C, Element::O) => "1,3-Diketone -> Beta-Keto-Enol",
                        (Element::O, Element::C) => "Enol -> Keto Shift",
                        (Element::N, Element::O) => "Amide / Lactam -> Imidic Acid / Lactim",
                        (Element::O, Element::N) => "Lactim -> Lactam Shift",
                        (Element::N, Element::S) => "Thioamide / Rhodanine -> Imidothiol",
                        (Element::N, Element::N) => "1,3-Amidine / Triazene Prototropy",
                        _ => "1,3-Prototropic Shift",
                    };
                    candidates.push((rule_lbl, t_mol));
                }
            }
        }
    }

    // 2. Annular 5-Membered Azole `[nH]` Shift (`Imidazole / Pyrazole / Triazole / Benzimidazole`)
    for r in &neutral_mol.rings {
        if r.len() == 5 && r.iter().all(|&u| neutral_mol.atoms[u].aromatic) {
            let nh_atoms: Vec<usize> = r
                .iter()
                .copied()
                .filter(|&u| {
                    neutral_mol.atoms[u].elem == Element::N && neutral_mol.atoms[u].total_h() > 0
                })
                .collect();
            let bare_n_atoms: Vec<usize> = r
                .iter()
                .copied()
                .filter(|&u| {
                    neutral_mol.atoms[u].elem == Element::N
                        && neutral_mol.atoms[u].total_h() == 0
                        && neutral_mol.heavy_degree(u) == 2
                })
                .collect();
            for &nh_u in &nh_atoms {
                for &bn_u in &bare_n_atoms {
                    if candidates.len() >= 12 {
                        break;
                    }
                    let mut t_mol = neutral_mol.clone();
                    t_mol.atoms[nh_u].explicit_h = 0;
                    t_mol.atoms[nh_u].implicit_h = 0;
                    t_mol.atoms[nh_u].bracketed = false;
                    t_mol.atoms[bn_u].explicit_h = 1;
                    t_mol.atoms[bn_u].implicit_h = 0;
                    t_mol.atoms[bn_u].bracketed = true;
                    t_mol.sanitize();
                    compute_2d_coords(&mut t_mol);
                    let can = molecule_to_smiles(&t_mol);
                    if !can.is_empty() && seen_smiles.insert(can.clone()) {
                        t_mol.smiles = can;
                        candidates.push(("Annular Azole [nH] Prototropy", t_mol));
                    }
                }
            }
        }
    }

    // Score each tautomer via `MolVS` / Sitzmann stability rules
    let mut scored: Vec<(f64, &'static str, Molecule)> = candidates
        .into_iter()
        .map(|(rule, m)| (score_molvs_tautomer(&m), rule, m))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let max_score = scored.first().map(|x| x.0).unwrap_or(0.0);

    // Convert score differential (`1 MolVS unit ~ 0.65 kcal/mol` at 298 K, `RT = 0.592 kcal/mol`)
    let mut raw_weights = Vec::with_capacity(scored.len());
    let mut w_sum = 0.0f64;
    for &(sc, _, _) in &scored {
        let dg = ((max_score - sc) * 0.65).max(0.0);
        let w = (-dg / 0.592).exp();
        raw_weights.push((dg, w));
        w_sum += w;
    }

    scored
        .into_iter()
        .zip(raw_weights.into_iter())
        .enumerate()
        .map(|(rk, ((sc, rule, m), (dg, w)))| TautomerSpecies {
            rank: rk + 1,
            rule_name: rule,
            molvs_score: sc,
            rel_energy_kcal: dg,
            boltzmann_fraction: if w_sum > 0.0 { w / w_sum } else { 1.0 },
            canonical_smiles: m.smiles.clone(),
            mol: m,
        })
        .collect()
}

fn score_molvs_tautomer(mol: &Molecule) -> f64 {
    let mut score = 0.0f64;
    // 1. Intact Hückel aromatic rings (+25 carbocyclic, +15 heteroaromatic)
    for r in &mol.rings {
        if r.iter().all(|&u| mol.atoms[u].aromatic) {
            let all_c = r.iter().all(|&u| mol.atoms[u].elem == Element::C);
            score += if all_c { 25.0 } else { 15.0 };
        }
    }
    // 2. Carbonyl `C=O` (+4.0), thiocarbonyl `C=S` (+2.5), imine `C=N` (+1.8)
    for b in &mol.bonds {
        if !b.aromatic && (b.order == BondOrder::Double || b.kekule_order == 2) {
            let ea = mol.atoms[b.a].elem;
            let eb = mol.atoms[b.b].elem;
            if (ea == Element::C && eb == Element::O) || (ea == Element::O && eb == Element::C) {
                score += 4.0;
            } else if (ea == Element::C && eb == Element::S)
                || (ea == Element::S && eb == Element::C)
            {
                score += 2.5;
            } else if (ea == Element::C && eb == Element::N)
                || (ea == Element::N && eb == Element::C)
            {
                score += 1.8;
            } else if ea == Element::C && eb == Element::C {
                score += 0.8;
            }
        }
    }
    // 3. Intramolecular H-bonded conjugated beta-keto-enol bonus (`O=C - C = C - OH` -> +5.2!)
    for u in 0..mol.atoms.len() {
        if mol.atoms[u].elem == Element::O
            && mol.atoms[u].total_h() > 0
            && mol.heavy_degree(u) == 1
        {
            for &(c_enol, _) in &mol.adj[u] {
                if mol.atoms[c_enol].elem == Element::C && !mol.atoms[c_enol].aromatic {
                    for &(c_mid, b_cc) in &mol.adj[c_enol] {
                        if mol.atoms[c_mid].elem == Element::C
                            && (mol.bonds[b_cc].order == BondOrder::Double
                                || mol.bonds[b_cc].kekule_order == 2)
                        {
                            let adj_ketone = mol
                                .adj[c_mid]
                                .iter()
                                .any(|&(ck, _)| ck != c_enol && is_carbonyl_c(mol, ck));
                            if adj_ketone {
                                score += 5.2;
                            }
                        }
                    }
                }
            }
        }
    }
    // 4. Penalty on O-H imidic acid (`-N=C-OH`) vs amide (`-NH-C=O`) is already captured (+4.0+0 vs +1.8)
    score
}

/// Evaluates the exact coupled Henderson-Hasselbalch microstate partition function across
/// `pH = 0.0..=14.0` (`29` points) and at `target_ph` (`default 7.4`), returning all major
/// 2D-depicted `ProtomerSpecies`, `TautomerSpecies`, `pI`, and `logD(pH)` curves.
pub fn analyze_pka_and_protomers(mol: &Molecule, target_ph: f64) -> PkaProtomerReport {
    let target_ph = target_ph.clamp(0.0, 14.0);
    let neutral_mol = standardize_neutral_parent(mol);
    let base_prof = evaluate_physchem(&neutral_mol, 0);
    let clogp = base_prof.clogp;
    let apsp = compute_topological_distances(&neutral_mol);

    let mut sites = perceive_ionizable_sites(&neutral_mol);
    let tautomers = enumerate_tautomers(&neutral_mol);

    let ph_grid: Vec<f32> = (0..PH_GRID_LEN).map(|i| (i as f32) * 0.5).collect();
    let k = sites.len();
    let num_states = 1usize << k;

    // Precompute state charges & pairwise through-space/through-bond electrostatic coupling penalties
    let mut state_net_q = vec![0i8; num_states];
    let mut state_pos = vec![0usize; num_states];
    let mut state_neg = vec![0usize; num_states];
    let mut state_coulomb_pen = vec![0.0f64; num_states];

    for mask in 0..num_states {
        let mut q = 0i8;
        let mut pos_c = 0usize;
        let mut neg_c = 0usize;
        let mut site_q = vec![0i8; k];
        for (si, s) in sites.iter().enumerate() {
            let prot = ((mask >> si) & 1) == 1;
            match s.kind {
                IonSiteKind::Acidic => {
                    if !prot {
                        q -= 1;
                        neg_c += 1;
                        site_q[si] = -1;
                    }
                }
                IonSiteKind::Basic => {
                    if prot {
                        q += 1;
                        pos_c += 1;
                        site_q[si] = 1;
                    }
                }
            }
        }
        let mut pen = 0.0f64;
        for i in 0..k {
            if site_q[i] == 0 {
                continue;
            }
            for j in (i + 1)..k {
                if site_q[j] == 0 {
                    continue;
                }
                let d = apsp[sites[i].atom_idx][sites[j].atom_idx];
                if d > 0 && d <= 7 {
                    if site_q[i] == site_q[j] {
                        // Repulsive dication (+1/+1) or dianion (-1/-1) penalty
                        pen += 0.90 / (d as f64);
                    } else {
                        // Attractive zwitterionic (+1/-1) salt-bridge stabilization
                        pen -= 0.35 / (d as f64);
                    }
                }
            }
        }
        state_net_q[mask] = q;
        state_pos[mask] = pos_c;
        state_neg[mask] = neg_c;
        state_coulomb_pen[mask] = pen;
    }

    let site_pkas: Vec<f64> = sites.iter().map(|s| s.micro_pka).collect();
    let eval_fractions_at = |ph: f64| -> Vec<f64> {
        let mut log_w = vec![0.0f64; num_states];
        let mut max_lw = f64::NEG_INFINITY;
        for mask in 0..num_states {
            let mut lw = -state_coulomb_pen[mask];
            for (si, &micro_pka) in site_pkas.iter().enumerate() {
                if ((mask >> si) & 1) == 1 {
                    lw += micro_pka - ph;
                }
            }
            log_w[mask] = lw;
            if lw > max_lw {
                max_lw = lw;
            }
        }
        let mut weights = vec![0.0f64; num_states];
        let mut z = 0.0f64;
        for mask in 0..num_states {
            let w = 10f64.powf(log_w[mask] - max_lw);
            weights[mask] = w;
            z += w;
        }
        if z > 0.0 {
            for w in &mut weights {
                *w /= z;
            }
        }
        weights
    };

    let target_fracs = eval_fractions_at(target_ph);

    // Update per-site ionized fraction at `target_ph`
    for (si, site) in sites.iter_mut().enumerate() {
        let mut ion_sum = 0.0f64;
        for mask in 0..num_states {
            let prot = ((mask >> si) & 1) == 1;
            let is_ionized = match site.kind {
                IonSiteKind::Acidic => !prot,
                IonSiteKind::Basic => prot,
            };
            if is_ionized {
                ion_sum += target_fracs[mask];
            }
        }
        site.ionized_fraction_at_ph = ion_sum.clamp(0.0, 1.0);
    }

    // Evaluate across all 29 grid points (`pH 0.0..=14.0`)
    let mut all_grid_fracs: Vec<Vec<f64>> = Vec::with_capacity(PH_GRID_LEN);
    let mut mean_charge_curve = Vec::with_capacity(PH_GRID_LEN);
    let mut logd_curve = Vec::with_capacity(PH_GRID_LEN);

    let calc_logd_from_fracs = |fracs: &[f64]| -> f64 {
        // Uncharged microspecies partition with `P = 10^cLogP`, zwitterions with `~0.10 * P`,
        // and singly/multiply charged ions with `~1e-3 * P` (ion-pair partitioning floor)
        let mut eff_neutral = 0.0f64;
        for mask in 0..num_states {
            let f = fracs[mask];
            if state_net_q[mask] == 0 {
                if state_pos[mask] == 0 && state_neg[mask] == 0 {
                    eff_neutral += f;
                } else {
                    eff_neutral += 0.12 * f;
                }
            } else {
                eff_neutral += 0.0008 * f;
            }
        }
        clogp + eff_neutral.max(1e-4).log10()
    };

    for &ph_f in &ph_grid {
        let fr = eval_fractions_at(ph_f as f64);
        let mq: f64 = (0..num_states)
            .map(|m| (state_net_q[m] as f64) * fr[m])
            .sum();
        let ld = calc_logd_from_fracs(&fr);
        mean_charge_curve.push(mq as f32);
        logd_curve.push(ld as f32);
        all_grid_fracs.push(fr);
    }

    // Detect Isoelectric Point `pI` if `<Q>(pH)` crosses `0.0` with both acidic & basic sites present
    let has_acid = sites.iter().any(|s| s.kind == IonSiteKind::Acidic);
    let has_base = sites.iter().any(|s| s.kind == IonSiteKind::Basic);
    let mut isoelectric_pi = None;
    if has_acid && has_base && !mean_charge_curve.is_empty() {
        let q0 = mean_charge_curve[0] as f64;
        let q14 = mean_charge_curve.last().copied().unwrap_or(0.0) as f64;
        if q0 > 0.05 && q14 < -0.05 {
            let mut lo = 0.0f64;
            let mut hi = 14.0f64;
            for _ in 0..32 {
                let mid = 0.5 * (lo + hi);
                let fr = eval_fractions_at(mid);
                let mq: f64 = (0..num_states)
                    .map(|m| (state_net_q[m] as f64) * fr[m])
                    .sum();
                if mq > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            isoelectric_pi = Some(0.5 * (lo + hi));
        }
    }

    // Rank microstates by abundance at `target_ph` (retaining states >= 0.2% at target_ph OR >= 10% anywhere in pH 1..13)
    let mut state_order: Vec<usize> = (0..num_states)
        .filter(|&m| {
            target_fracs[m] >= 0.002
                || all_grid_fracs
                    .iter()
                    .any(|gf| gf[m] >= 0.12)
        })
        .collect();
    if state_order.is_empty() {
        state_order.push(0);
    }
    state_order.sort_by(|&a, &b| {
        target_fracs[b]
            .partial_cmp(&target_fracs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    state_order.truncate(8);

    let mut protomers = Vec::with_capacity(state_order.len());
    for (rk, &mask) in state_order.iter().enumerate() {
        let q = state_net_q[mask];
        let pc = state_pos[mask];
        let nc = state_neg[mask];
        let is_zwit = q == 0 && pc > 0 && nc > 0;
        let state_label = if is_zwit {
            format!("Zwitterion (+{pc}/-{nc})")
        } else if q > 0 {
            format!("Cation (+{q})")
        } else if q < 0 {
            format!("Anion ({q})")
        } else {
            "Neutral (0)".to_string()
        };

        let curve_fractions: Vec<f32> = all_grid_fracs.iter().map(|gf| gf[mask] as f32).collect();
        let pmol = build_protomer_molecule(&neutral_mol, &sites, mask as u32, rk);

        protomers.push(ProtomerSpecies {
            rank: rk + 1,
            state_mask: mask as u32,
            net_charge: q,
            pos_centers: pc,
            neg_centers: nc,
            is_zwitterion: is_zwit,
            state_label,
            fraction_at_target_ph: target_fracs[mask],
            curve_fractions,
            canonical_smiles: pmol.smiles.clone(),
            mol: pmol,
        });
    }

    let mut neutral_frac = 0.0f64;
    let mut cat_frac = 0.0f64;
    let mut ani_frac = 0.0f64;
    let mut zwit_frac = 0.0f64;
    let mut mean_charge_at_ph = 0.0f64;
    for mask in 0..num_states {
        let f = target_fracs[mask];
        let q = state_net_q[mask];
        mean_charge_at_ph += (q as f64) * f;
        if q > 0 {
            cat_frac += f;
        } else if q < 0 {
            ani_frac += f;
        } else if state_pos[mask] > 0 && state_neg[mask] > 0 {
            zwit_frac += f;
        } else {
            neutral_frac += f;
        }
    }

    let charge_class_at_ph = if zwit_frac >= neutral_frac && zwit_frac >= cat_frac && zwit_frac >= ani_frac && zwit_frac > 0.20 {
        ChargeClass74::Zwitterionic
    } else if cat_frac >= neutral_frac && cat_frac >= ani_frac {
        ChargeClass74::Cationic
    } else if ani_frac >= neutral_frac && ani_frac >= cat_frac {
        ChargeClass74::Anionic
    } else {
        ChargeClass74::Neutral
    };

    let logd_at_ph = calc_logd_from_fracs(&target_fracs);
    let gastric_logd_20 = logd_curve[4] as f64;   // index 4 -> pH 2.0
    let jejunal_logd_65 = logd_curve[13] as f64;  // index 13 -> pH 6.5
    let plasma_logd_74 = calc_logd_from_fracs(&eval_fractions_at(7.4));

    PkaProtomerReport {
        target_ph,
        ph_grid,
        sites,
        protomers,
        tautomers,
        mean_charge_curve,
        logd_curve,
        neutral_fraction_at_ph: neutral_frac,
        mean_charge_at_ph,
        logd_at_ph,
        gastric_logd_20,
        jejunal_logd_65,
        plasma_logd_74,
        isoelectric_pi,
        charge_class_at_ph,
    }
}

/// Fast physiological `pH 7.4` summary wrapper run in parallel via `Rayon` across `DatasetAnalysis`.
pub fn evaluate_physiological_pka_74(mol: &Molecule) -> PhysiologicalChargeSummary {
    let rep = analyze_pka_and_protomers(mol, 7.4);
    let strongest_acid_pka = rep
        .sites
        .iter()
        .filter(|s| s.kind == IonSiteKind::Acidic)
        .map(|s| s.micro_pka)
        .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.min(v))));
    let strongest_base_pka = rep
        .sites
        .iter()
        .filter(|s| s.kind == IonSiteKind::Basic)
        .map(|s| s.micro_pka)
        .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.max(v))));

    let mut cat_f = 0.0f64;
    let mut ani_f = 0.0f64;
    let mut zwit_f = 0.0f64;
    for p in &rep.protomers {
        if p.is_zwitterion {
            zwit_f += p.fraction_at_target_ph;
        } else if p.net_charge > 0 {
            cat_f += p.fraction_at_target_ph;
        } else if p.net_charge < 0 {
            ani_f += p.fraction_at_target_ph;
        }
    }
    let major_protomers_74 = rep
        .protomers
        .iter()
        .filter(|p| p.fraction_at_target_ph >= 0.05)
        .count()
        .max(1);
    let dom = rep.protomers.first();

    PhysiologicalChargeSummary {
        charge_class: rep.charge_class_at_ph,
        dominant_net_charge: dom.map(|d| d.net_charge).unwrap_or(0),
        mean_net_charge_74: rep.mean_charge_at_ph,
        neutral_fraction_74: rep.neutral_fraction_at_ph,
        cation_fraction_74: cat_f,
        anion_fraction_74: ani_f,
        zwitterion_fraction_74: zwit_f,
        strongest_acid_pka,
        strongest_base_pka,
        isoelectric_pi: rep.isoelectric_pi,
        microstate_logd_74: rep.plasma_logd_74,
        gastric_logd_20: rep.gastric_logd_20,
        major_protomers_74,
        tautomer_count: rep.tautomers.len(),
        dominant_protomer_smiles: dom
            .map(|d| d.canonical_smiles.clone())
            .unwrap_or_else(|| mol.smiles.clone()),
    }
}
