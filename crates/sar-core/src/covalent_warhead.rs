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

//! Tuned Covalent Electrophile Reactivity Profiler, Glutathione (`GSH`) Half-Life Estimator,
//! Two-Step `k_inact / K_I` Kinetic Modeler, and Reversible / Irreversible Covalent Warhead Swapper.
//!
//! Primary References:
//! - Flanagan et al. (2014), *J. Med. Chem.* 57(23), 10072-10079 ("Chemical and computational
//!   methods for the characterization of covalent reactive groups for the prospective design of
//!   irreversible inhibitors" — `GSH t_1/2` benchmark: `https://pubmed.ncbi.nlm.nih.gov/25375838/`)
//! - Lonsdale et al. (2019) ("Characterising covalent warhead reactivity" — two-step `k_inact / K_I`
//!   inactivation kinetics: `https://pmc.ncbi.nlm.nih.gov/articles/PMC6538824/`)
//! - Boike, Henning & Nomura (2022), *Nat. Rev. Drug Discov.* 21, 881-898 ("Advances in covalent
//!   drug discovery" — Cys/Lys/Ser/Tyr chemoproteomics: `https://pmc.ncbi.nlm.nih.gov/articles/PMC9403961/`)
//! - Serafimova et al. (2012), *Nat. Chem. Biol.* 8(5), 471-476 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3657615/`)
//!   & Miller, Paavilainen, Krishnan, Serafimova & Taunton (2013), *J. Am. Chem. Soc.* 135(14),
//!   5298-5301 (`https://pubmed.ncbi.nlm.nih.gov/23540679/`) — reversible-covalent cyanoacrylamides
//! - Jackson et al. (2017) & Cee et al. (2020) ("Covalent Modifiers: Reactivity of alpha,beta-Unsaturated
//!   Carbonyls" & N-arylacrylamide inductive tuning: `https://pmc.ncbi.nlm.nih.gov/articles/PMC5308545/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC9694498/`)

use crate::alerts::scan_structural_alerts;
use crate::conformer3d::compute_gasteiger_charges;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::physchem::evaluate_physchem;
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::rgroup_table::{predict_best_free_wilson, FreeWilsonMatrix};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::DatasetAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Number of timepoints in the `0.0 ..= 24.0 h` target-occupancy vs off-target `GSH` kinetic trace.
pub const KINETIC_GRID_LEN: usize = 25;

/// Primary nucleophilic amino-acid side chain targeted by an electrophilic warhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetResidue {
    Cysteine,
    SerineThreonine,
    LysineTyrosine,
    NonCovalent,
}

impl TargetResidue {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cysteine => "Cys (-SH)",
            Self::SerineThreonine => "Ser / Thr (-OH)",
            Self::LysineTyrosine => "Lys / Tyr / His",
            Self::NonCovalent => "Non-Covalent",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Cysteine => [250, 204, 21],        // Gold (thiol)
            Self::SerineThreonine => [56, 189, 248], // Sky Blue
            Self::LysineTyrosine => [192, 132, 252], // Violet
            Self::NonCovalent => [148, 163, 184],    // Slate
        }
    }
}

/// Thermodynamic adduct reversibility regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarheadReversibility {
    Irreversible,
    ReversibleCovalent,
    NonCovalentControl,
}

impl WarheadReversibility {
    pub fn label(self) -> &'static str {
        match self {
            Self::Irreversible => "Irreversible Covalent",
            Self::ReversibleCovalent => "Reversible Covalent",
            Self::NonCovalentControl => "Non-Covalent Control",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Irreversible => "Irreversible",
            Self::ReversibleCovalent => "Reversible-Cov",
            Self::NonCovalentControl => "Non-Covalent",
        }
    }
}

/// Intrinsic glutathione (`GSH`, `5 mM`, `pH 7.4`, `37 °C`) reactivity window classification
/// calibrated to Flanagan et al. (2014) and Cee et al. (2020).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReactivityWindow {
    HyperReactivePromiscuous,
    FastIrreversible,
    TunedClinicalSweetSpot,
    MildSelectiveIrreversible,
    ReversibleCovalent,
    NonCovalent,
}

impl ReactivityWindow {
    pub fn label(self) -> &'static str {
        match self {
            Self::HyperReactivePromiscuous => "Hyper-Reactive (<0.25 h GSH)",
            Self::FastIrreversible => "Fast Irreversible (0.25-1.5 h)",
            Self::TunedClinicalSweetSpot => "Tuned TCI Sweet Spot (1.5-36 h)",
            Self::MildSelectiveIrreversible => "Attenuated / Selective (>36 h)",
            Self::ReversibleCovalent => "Reversible Covalent (Tunable tau)",
            Self::NonCovalent => "Non-Covalent (Reversible)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HyperReactivePromiscuous => "Hyper-Reactive",
            Self::FastIrreversible => "Fast Irrev",
            Self::TunedClinicalSweetSpot => "Tuned Sweet Spot",
            Self::MildSelectiveIrreversible => "Mild / Selective",
            Self::ReversibleCovalent => "Reversible-Cov",
            Self::NonCovalent => "Non-Covalent",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HyperReactivePromiscuous => [248, 113, 113], // Coral Red
            Self::FastIrreversible => [251, 191, 36],          // Amber
            Self::TunedClinicalSweetSpot => [52, 211, 153],    // Emerald
            Self::MildSelectiveIrreversible => [56, 189, 248], // Sky Blue
            Self::ReversibleCovalent => [192, 132, 252],       // Violet
            Self::NonCovalent => [148, 163, 184],              // Slate
        }
    }
}

/// Perceived electrophilic covalent warhead center on a `Molecule` with Hammett/Gasteiger-tuned
/// `GSH` half-life (`t_1/2`) and two-step `k_inact / K_I` inactivation kinetics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectedWarhead {
    #[serde(skip_deserializing)]
    pub warhead_code: &'static str,
    #[serde(skip_deserializing)]
    pub warhead_name: &'static str,
    #[serde(skip_deserializing)]
    pub mechanism: &'static str,
    pub target_residue: TargetResidue,
    pub reversibility: WarheadReversibility,
    pub window: ReactivityWindow,
    pub electrophile_atom: usize,
    pub attachment_atom: Option<usize>,
    pub warhead_atoms: Vec<usize>,
    pub base_gsh_half_life_h: f64,
    /// Inductive electronic shift `log10(t_1/2 / t_1/2,0)` (negative = electron-withdrawing scaffold accelerates GSH reaction).
    pub inductive_shift_log10: f64,
    pub gsh_half_life_h: f64,
    pub k_gsh_per_h: f64,
    pub k_inact_per_min: f64,
    pub ki_nm: f64,
    pub kinact_over_ki_m_inv_s_inv: f64,
    pub residence_time_h: f64,
}

/// Compact per-molecule covalent reactivity summary stored on `DatasetAnalysis::covalent`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovalentDatasetSummary {
    pub has_warhead: bool,
    pub warhead_count: usize,
    #[serde(skip_deserializing)]
    pub primary_warhead_name: &'static str,
    pub target_residue: TargetResidue,
    pub reversibility: WarheadReversibility,
    pub window: ReactivityWindow,
    pub gsh_half_life_h: Option<f64>,
    pub kinact_over_ki: Option<f64>,
    pub swap_handle_available: bool,
}

/// Enumerated reversible or irreversible covalent analog generated by the Warhead Swapper.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WarheadSwapCandidate {
    pub rank: usize,
    #[serde(skip_deserializing)]
    pub template_code: &'static str,
    #[serde(skip_deserializing)]
    pub template_name: &'static str,
    #[serde(skip_deserializing)]
    pub clinical_precedent: &'static str,
    pub swap_mode_label: String,
    pub reversibility: WarheadReversibility,
    pub target_residue: TargetResidue,
    pub window: ReactivityWindow,
    pub gsh_half_life_h: Option<f64>,
    pub k_inact_per_min: f64,
    pub ki_nm: f64,
    pub kinact_over_ki: f64,
    pub residence_time_h: f64,
    pub predicted_pact: f64,
    pub qed: f64,
    pub sa_score: f64,
    pub logd_74: f64,
    /// Multi-objective Targeted Covalent Inhibitor (`TCI`) desirability score `[0..100]`.
    pub tci_selectivity_score: f64,
    pub canonical_smiles: String,
    pub warhead_atoms: Vec<usize>,
    pub electrophile_atom: Option<usize>,
    pub mol: Molecule,
}

/// Full interactive covalent profiling & warhead-swapping report for a target compound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovalentWarheadReport {
    pub detected: Vec<DetectedWarhead>,
    pub primary_window: ReactivityWindow,
    pub inhibitor_conc_nm: f64,
    pub time_grid_h: Vec<f32>,
    /// Target covalent occupancy `%` (`0..100`) at `inhibitor_conc_nm` over `0..24 h`.
    pub target_occupancy_curve: Vec<f32>,
    /// Off-target physiological `5 mM` `GSH` consumption `%` (`0..100`) over `0..24 h`.
    pub gsh_adduct_curve: Vec<f32>,
    pub swaps: Vec<WarheadSwapCandidate>,
}

fn classify_irrev_gsh_window(t_half_h: f64) -> ReactivityWindow {
    if t_half_h < 0.25 {
        ReactivityWindow::HyperReactivePromiscuous
    } else if t_half_h < 1.5 {
        ReactivityWindow::FastIrreversible
    } else if t_half_h <= 36.0 {
        ReactivityWindow::TunedClinicalSweetSpot
    } else {
        ReactivityWindow::MildSelectiveIrreversible
    }
}

/// Computes the scaffold inductive electronic shift `log10(t_1/2 / t_1/2,0)` at `attach_atom`
/// excluding `warhead_atoms`. Electron-withdrawing N-aryl / haloaryl attachments yield negative
/// shifts (faster `GSH` addition, shorter `t_1/2`); aliphatic N-alkyl attachments yield positive shifts.
fn compute_scaffold_inductive_shift(
    mol: &Molecule,
    charges: &[f32],
    attach_atom: Option<usize>,
    warhead_atoms: &[usize],
) -> f64 {
    let Some(att) = attach_atom else {
        return 0.0;
    };
    if att >= mol.atoms.len() {
        return 0.0;
    }

    let mut shift = 0.0f64;
    let a_att = &mol.atoms[att];

    // N-aryl vs N-alkyl baseline difference (Flanagan 2014: N-Ph acrylamide ~4.2 h vs N-Me/piperidyl ~26 h)
    let attached_to_aryl = a_att.aromatic
        || mol.adj[att]
            .iter()
            .any(|&(nb, _)| !warhead_atoms.contains(&nb) && mol.atoms[nb].aromatic);

    if a_att.elem == Element::N {
        if attached_to_aryl {
            shift -= 0.18;
        } else {
            shift += 0.52;
        }
    } else if a_att.aromatic {
        shift -= 0.12;
    }

    // BFS up to 4 bonds into the non-warhead scaffold to capture EWG (-F, -Cl, -CF3, -CN, -SO2-, ring N)
    let mut dist = vec![usize::MAX; mol.atoms.len()];
    let mut q = VecDeque::new();
    dist[att] = 0;
    q.push_back(att);
    while let Some(u) = q.pop_front() {
        let d = dist[u];
        if d >= 4 {
            continue;
        }
        for &(nb, _) in &mol.adj[u] {
            if warhead_atoms.contains(&nb) {
                continue;
            }
            if dist[nb] == usize::MAX {
                dist[nb] = d + 1;
                q.push_back(nb);
            }
        }
    }

    for (u, &d) in dist.iter().enumerate() {
        if d == 0 || d == usize::MAX || warhead_atoms.contains(&u) {
            continue;
        }
        let atten = 0.52f64.powi((d as i32) - 1);
        let el = mol.atoms[u].elem;
        let ewg: f64 = match el {
            Element::F => -0.14,
            Element::Cl | Element::Br => -0.16,
            Element::N if mol.atoms[u].aromatic => -0.18,
            Element::N if mol.atoms[u].hyb == Hybridization::Sp => -0.26,
            Element::S => -0.22,
            Element::O if !mol.atoms[u].in_ring && mol.adj[u].len() == 2 => 0.08, // Electron-donating alkoxy
            _ => 0.0,
        };
        shift += ewg * atten;
    }

    // Fine Gasteiger charge correction at attachment atom
    let q_att = charges.get(att).copied().unwrap_or(0.0) as f64;
    shift -= (q_att + 0.25).clamp(-0.25, 0.25) * 0.25;

    shift.clamp(-0.85, 0.95)
}

/// Estimates two-step covalent inactivation parameters `(k_inact_min^-1, K_I_nM, k_inact/K_I_M^-1s^-1, tau_h)`
/// from a compound's reversible scaffold `pAct`, tuned `GSH t_1/2`, and warhead reversibility.
fn compute_two_step_kinetics(
    pact: f64,
    gsh_half_life_h: f64,
    reversibility: WarheadReversibility,
    internal_base_boost: f64,
) -> (f64, f64, f64, f64) {
    // Reversible binding constant K_I (nM) derived from pAct (capped to realistic 0.05 nM .. 50 uM)
    let ki_nm = (10.0f64.powf(9.0 - pact.clamp(4.3, 10.5))).clamp(0.05, 50_000.0);
    let ki_molar = ki_nm * 1e-9;

    match reversibility {
        WarheadReversibility::NonCovalentControl => (0.0, ki_nm, 0.0, 0.15),
        WarheadReversibility::ReversibleCovalent => {
            // Pocket-stabilized reversible adduct: fast on-pocket bond formation, finite off-rate k_off
            let k_inact_min = (0.18 / gsh_half_life_h.clamp(0.4, 120.0).sqrt()).clamp(0.02, 0.45);
            let k_inact_s = k_inact_min / 60.0;
            let eff = k_inact_s / ki_molar.max(1e-12);
            // Residence time scales with non-covalent stabilization + thioimidate/cyanoacrylamide adduct barrier
            let tau_h = (1.5 + (pact - 6.5).max(0.0) * 3.2).clamp(0.8, 24.0);
            (k_inact_min, ki_nm, eff, tau_h)
        }
        WarheadReversibility::Irreversible => {
            // Empirical Bronsted/Brdička scaling: pocket k_inact correlates inversely with GSH t_1/2
            // Typical clinical acrylamides (t_1/2 ~ 4 h) exhibit k_inact ~ 0.06 - 0.18 min^-1
            let k_inact_min =
                (0.32 * internal_base_boost / gsh_half_life_h.max(0.05).powf(0.62)).clamp(0.002, 2.4);
            let k_inact_s = k_inact_min / 60.0;
            let eff = k_inact_s / ki_molar.max(1e-12);
            // Irreversible target silencing persists until cellular protein resynthesis (~18 - 36 h)
            (k_inact_min, ki_nm, eff, 24.0)
        }
    }
}

/// Perceives all targeted covalent warheads on `mol`, identifying the electrophilic carbon/sulfur/boron
/// atom, scaffold attachment atom, warhead subgraph, tuned `GSH t_1/2`, and `k_inact / K_I` kinetics.
pub fn detect_covalent_warheads(mol: &Molecule) -> Vec<DetectedWarhead> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }
    let charges = compute_gasteiger_charges(mol);
    let mut results: Vec<DetectedWarhead> = Vec::new();
    let mut claimed_electrophiles: BTreeSet<usize> = BTreeSet::new();

    // Helper: check if carbon `c` is a carbonyl `C=O` (returns oxygen index)
    let carbonyl_oxygen = |c: usize| -> Option<usize> {
        if mol.atoms[c].elem != Element::C {
            return None;
        }
        for &(nb, bi) in &mol.adj[c] {
            if mol.atoms[nb].elem == Element::O
                && mol.bonds[bi].order == BondOrder::Double
                && mol.adj[nb].len() == 1
            {
                return Some(nb);
            }
        }
        None
    };

    // Helper: check if atom `u` is a sulfonyl `-SO2-` sulfur (returns the two `=O` indices)
    let sulfonyl_oxygens = |s: usize| -> Option<[usize; 2]> {
        if mol.atoms[s].elem != Element::S {
            return None;
        }
        let mut ox = Vec::new();
        for &(nb, bi) in &mol.adj[s] {
            if mol.atoms[nb].elem == Element::O && mol.bonds[bi].order == BondOrder::Double {
                ox.push(nb);
            }
        }
        if ox.len() >= 2 {
            Some([ox[0], ox[1]])
        } else {
            None
        }
    };

    // 1. Alpha,Beta-Unsaturated Carbonyls & Sulfonamides (Acrylamides, DMAMB, Cyanoacrylamides, Butynamides, Maleimides, Vinyl Sulfonamides)
    for b in &mol.bonds {
        if b.aromatic || b.in_ring && !matches!(b.order, BondOrder::Double) {
            continue;
        }
        if !matches!(b.order, BondOrder::Double | BondOrder::Triple) {
            continue;
        }
        let u = b.a;
        let v = b.b;
        if mol.atoms[u].elem != Element::C || mol.atoms[v].elem != Element::C {
            continue;
        }

        // Try both orientations: `alpha` bonded to `C=O` or `-SO2-`, `beta` is nucleophilic attack site
        for &(alpha, beta) in &[(u, v), (v, u)] {
            if claimed_electrophiles.contains(&beta) {
                continue;
            }

            // Look for activating C=O or SO2 adjacent to `alpha`
            let mut activator_atoms: Vec<usize> = Vec::new();
            let mut attach_atom: Option<usize> = None;
            let mut is_amide = false;
            let mut is_ester = false;
            let mut is_ketone = false;
            let mut is_vinyl_sulfone = false;

            for &(act, act_bi) in &mol.adj[alpha] {
                if act == beta || mol.bonds[act_bi].order != BondOrder::Single {
                    continue;
                }
                if let Some(ox) = carbonyl_oxygen(act) {
                    activator_atoms.push(act);
                    activator_atoms.push(ox);
                    for &(het, _) in &mol.adj[act] {
                        if het != alpha && het != ox {
                            attach_atom = Some(het);
                            if mol.atoms[het].elem == Element::N {
                                is_amide = true;
                            } else if mol.atoms[het].elem == Element::O {
                                is_ester = true;
                            } else {
                                is_ketone = true;
                            }
                        }
                    }
                    break;
                } else if let Some(oxs) = sulfonyl_oxygens(act) {
                    is_vinyl_sulfone = true;
                    activator_atoms.push(act);
                    activator_atoms.extend_from_slice(&oxs);
                    for &(het, _) in &mol.adj[act] {
                        if het != alpha && het != oxs[0] && het != oxs[1] {
                            attach_atom = Some(het);
                        }
                    }
                    break;
                }
            }

            if activator_atoms.is_empty() {
                continue;
            }
            // Exclude diaryl chalcones / ferulates (like Curcumin PAINS) unless amide/sulfonamide or cyclic maleimide
            if is_ketone && !b.in_ring {
                let beta_has_aryl = mol.adj[beta]
                    .iter()
                    .any(|&(nb, _)| nb != alpha && mol.atoms[nb].aromatic);
                if beta_has_aryl {
                    continue;
                }
            }

            let mut wh_atoms = vec![alpha, beta];
            wh_atoms.extend_from_slice(&activator_atoms);

            // Inspect alpha and beta substituents
            let mut alpha_has_cn = false;
            let mut alpha_has_alkyl = false;
            for &(nb, _) in &mol.adj[alpha] {
                if nb == beta || activator_atoms.contains(&nb) {
                    continue;
                }
                // Check -C#N on alpha (Serafimova/Taunton reversible cyanoacrylamide!)
                if mol.atoms[nb].elem == Element::C {
                    if let Some(&(n_cn, bi_cn)) = mol.adj[nb].iter().find(|&&(x, bi)| {
                        mol.atoms[x].elem == Element::N && mol.bonds[bi].order == BondOrder::Triple
                    }) {
                        alpha_has_cn = true;
                        wh_atoms.push(nb);
                        wh_atoms.push(n_cn);
                        continue;
                    }
                }
                alpha_has_alkyl = true;
                if mol.adj[nb].len() == 1 {
                    wh_atoms.push(nb);
                }
            }

            let mut beta_subs = 0usize;
            let mut beta_has_aminome = false;
            for &(nb, _) in &mol.adj[beta] {
                if nb == alpha {
                    continue;
                }
                beta_subs += 1;
                wh_atoms.push(nb);
                // Check -CH2-NR2 (Afatinib / Neratinib 4-dimethylaminobut-2-enamide!)
                if mol.atoms[nb].elem == Element::C && mol.atoms[nb].hyb == Hybridization::Sp3 {
                    for &(n2, _) in &mol.adj[nb] {
                        if n2 != beta
                            && mol.atoms[n2].elem == Element::N
                            && !mol.atoms[n2].aromatic
                        {
                            beta_has_aminome = true;
                            wh_atoms.push(n2);
                            for &(me, _) in &mol.adj[n2] {
                                if me != nb && mol.adj[me].len() == 1 {
                                    wh_atoms.push(me);
                                }
                            }
                        }
                    }
                }
            }

            let (code, name, mech, rev, base_t12, base_boost) = if b.order == BondOrder::Triple {
                if beta_subs == 0 {
                    (
                        "propiolamide",
                        "Terminal Propiolamide (-C(=O)C#CH)",
                        "Fast conjugate thiol addition to terminal ynamide",
                        WarheadReversibility::Irreversible,
                        0.68,
                        1.25,
                    )
                } else {
                    (
                        "butynamide",
                        "Internal But-2-ynamide (-C(=O)C#C-R)",
                        "Acalabrutinib-class tuned internal alkyne Cys Michael addition",
                        WarheadReversibility::Irreversible,
                        18.5,
                        1.05,
                    )
                }
            } else if alpha_has_cn {
                (
                    "cyanoacrylamide_rev",
                    "Alpha-Cyanoacrylamide (Reversible Covalent)",
                    "Serafimova-Taunton reversible Cys Michael addition / beta-elimination",
                    WarheadReversibility::ReversibleCovalent,
                    3.2,
                    1.15,
                )
            } else if is_vinyl_sulfone {
                let is_sulfonamide = attach_atom
                    .map(|a| mol.atoms[a].elem == Element::N)
                    .unwrap_or(false);
                if is_sulfonamide {
                    (
                        "vinyl_sulfonamide",
                        "Vinyl Sulfonamide (-SO2CH=CH2)",
                        "Conjugate Cys Michael addition to vinyl sulfonamide",
                        WarheadReversibility::Irreversible,
                        0.95,
                        1.20,
                    )
                } else {
                    (
                        "vinyl_sulfone",
                        "Vinyl Sulfone (-SO2CH=CH2)",
                        "Fast irreversible Cys Michael addition to vinyl sulfone",
                        WarheadReversibility::Irreversible,
                        0.12,
                        1.50,
                    )
                }
            } else if b.in_ring && is_amide {
                (
                    "maleimide_endo",
                    "Cyclic Maleimide / Endo-Enamide",
                    "High-reactivity cyclic Michael thiol capture",
                    WarheadReversibility::Irreversible,
                    0.05,
                    1.80,
                )
            } else if beta_has_aminome && is_amide {
                (
                    "dmamb_enamide",
                    "(E)-4-(Dialkylamino)but-2-enamide",
                    "Afatinib-class intramolecular amine base-catalyzed Cys Michael addition",
                    WarheadReversibility::Irreversible,
                    3.4,
                    1.65,
                )
            } else if is_amide && alpha_has_alkyl && beta_subs == 0 {
                (
                    "methacrylamide",
                    "Alpha-Substituted Methacrylamide",
                    "Sterically & electronically attenuated Cys Michael addition",
                    WarheadReversibility::Irreversible,
                    58.0,
                    0.85,
                )
            } else if is_amide && beta_subs > 0 {
                (
                    "crotonamide",
                    "Beta-Substituted Enamide (Crotonamide)",
                    "Sterically tuned beta-substituted Cys Michael addition",
                    WarheadReversibility::Irreversible,
                    44.0,
                    0.90,
                )
            } else if is_amide {
                (
                    "propenamide",
                    "Unsubstituted Acrylamide (-C(=O)CH=CH2)",
                    "Ibrutinib/Osimertinib-class Cys hetero-Michael addition",
                    WarheadReversibility::Irreversible,
                    5.6,
                    1.15,
                )
            } else if is_ester {
                (
                    "acrylate_ester",
                    "Alpha,Beta-Unsaturated Ester / Fumarate",
                    "Activated ester Michael thiol addition",
                    WarheadReversibility::Irreversible,
                    0.85,
                    1.10,
                )
            } else {
                (
                    "enone_michael",
                    "Alpha,Beta-Unsaturated Ketone (Enone)",
                    "Fast enone Michael thiol addition",
                    WarheadReversibility::Irreversible,
                    0.18,
                    1.35,
                )
            };

            let ind_shift = compute_scaffold_inductive_shift(mol, &charges, attach_atom, &wh_atoms);
            let tuned_t12 = (base_t12 * 10.0f64.powf(ind_shift)).clamp(0.01, 500.0);
            let k_gsh = std::f64::consts::LN_2 / tuned_t12;
            let window = match rev {
                WarheadReversibility::ReversibleCovalent => ReactivityWindow::ReversibleCovalent,
                WarheadReversibility::Irreversible => classify_irrev_gsh_window(tuned_t12),
                WarheadReversibility::NonCovalentControl => ReactivityWindow::NonCovalent,
            };
            let (k_inact, ki_nm, eff, tau_h) =
                compute_two_step_kinetics(mol.pact, tuned_t12, rev, base_boost);

            claimed_electrophiles.insert(beta);
            results.push(DetectedWarhead {
                warhead_code: code,
                warhead_name: name,
                mechanism: mech,
                target_residue: TargetResidue::Cysteine,
                reversibility: rev,
                window,
                electrophile_atom: beta,
                attachment_atom: attach_atom,
                warhead_atoms: wh_atoms,
                base_gsh_half_life_h: base_t12,
                inductive_shift_log10: ind_shift,
                gsh_half_life_h: tuned_t12,
                k_gsh_per_h: k_gsh,
                k_inact_per_min: k_inact,
                ki_nm,
                kinact_over_ki_m_inv_s_inv: eff,
                residence_time_h: tau_h,
            });
        }
    }

    // 2. Alpha-Haloacetamides (`-NH-C(=O)-CH2-Cl/Br/F`) & Alpha-Ketoamides (`-C(=O)-C(=O)-N`)
    for c_carb in 0..n {
        let Some(ox) = carbonyl_oxygen(c_carb) else {
            continue;
        };
        let mut attach_n: Option<usize> = None;
        let mut alpha_c: Option<usize> = None;
        for &(nb, _) in &mol.adj[c_carb] {
            if nb == ox {
                continue;
            }
            if mol.atoms[nb].elem == Element::N {
                attach_n = Some(nb);
            } else if mol.atoms[nb].elem == Element::C {
                alpha_c = Some(nb);
            }
        }

        if let (Some(att), Some(ac)) = (attach_n, alpha_c) {
            if claimed_electrophiles.contains(&ac) {
                continue;
            }
            // Check if `ac` carries a leaving halogen Cl/Br/I/F on sp3 carbon
            if mol.atoms[ac].hyb == Hybridization::Sp3 && !mol.atoms[ac].in_ring {
                if let Some(&(hal, _)) = mol.adj[ac].iter().find(|&&(x, _)| {
                    matches!(
                        mol.atoms[x].elem,
                        Element::Cl | Element::Br | Element::I | Element::F
                    )
                }) {
                    let wh_atoms = vec![c_carb, ox, ac, hal];
                    let (code, name, base_t12) = match mol.atoms[hal].elem {
                        Element::F => (
                            "fluoroacetamide",
                            "Alpha-Fluoroacetamide (-C(=O)CH2F)",
                            125.0,
                        ),
                        Element::Cl => (
                            "chloroacetamide",
                            "Alpha-Chloroacetamide (-C(=O)CH2Cl)",
                            1.8,
                        ),
                        _ => (
                            "bromoacetamide",
                            "Alpha-Bromo/Iodoacetamide (-C(=O)CH2Br)",
                            0.08,
                        ),
                    };
                    let ind_shift =
                        compute_scaffold_inductive_shift(mol, &charges, Some(att), &wh_atoms);
                    let tuned_t12 = (base_t12 * 10.0f64.powf(ind_shift)).clamp(0.01, 500.0);
                    let k_gsh = std::f64::consts::LN_2 / tuned_t12;
                    let window = classify_irrev_gsh_window(tuned_t12);
                    let (k_inact, ki_nm, eff, tau_h) = compute_two_step_kinetics(
                        mol.pact,
                        tuned_t12,
                        WarheadReversibility::Irreversible,
                        1.2,
                    );
                    claimed_electrophiles.insert(ac);
                    results.push(DetectedWarhead {
                        warhead_code: code,
                        warhead_name: name,
                        mechanism: "SN2 nucleophilic Cys thiolate displacement of alpha-halide",
                        target_residue: TargetResidue::Cysteine,
                        reversibility: WarheadReversibility::Irreversible,
                        window,
                        electrophile_atom: ac,
                        attachment_atom: Some(att),
                        warhead_atoms: wh_atoms,
                        base_gsh_half_life_h: base_t12,
                        inductive_shift_log10: ind_shift,
                        gsh_half_life_h: tuned_t12,
                        k_gsh_per_h: k_gsh,
                        k_inact_per_min: k_inact,
                        ki_nm,
                        kinact_over_ki_m_inv_s_inv: eff,
                        residence_time_h: tau_h,
                    });
                    continue;
                }
            }

            // Check Alpha-Ketoamide (`R-C(=O)-C(=O)-NHR'`, where `ac` is also a carbonyl!)
            if let Some(ox2) = carbonyl_oxygen(ac) {
                if !claimed_electrophiles.contains(&ac) {
                    let wh_atoms = vec![c_carb, ox, ac, ox2];
                    let (k_inact, ki_nm, eff, tau_h) = compute_two_step_kinetics(
                        mol.pact,
                        45.0,
                        WarheadReversibility::ReversibleCovalent,
                        1.0,
                    );
                    claimed_electrophiles.insert(ac);
                    results.push(DetectedWarhead {
                        warhead_code: "ketoamide_rev",
                        warhead_name: "Alpha-Ketoamide (Reversible Covalent)",
                        mechanism: "Reversible tetrahedral hemiketal / hemithioacetal adduct",
                        target_residue: TargetResidue::SerineThreonine,
                        reversibility: WarheadReversibility::ReversibleCovalent,
                        window: ReactivityWindow::ReversibleCovalent,
                        electrophile_atom: ac,
                        attachment_atom: Some(att),
                        warhead_atoms: wh_atoms,
                        base_gsh_half_life_h: 60.0,
                        inductive_shift_log10: 0.0,
                        gsh_half_life_h: 60.0,
                        k_gsh_per_h: std::f64::consts::LN_2 / 60.0,
                        k_inact_per_min: k_inact,
                        ki_nm,
                        kinact_over_ki_m_inv_s_inv: eff,
                        residence_time_h: tau_h,
                    });
                }
            }
        }
    }

    // 3. Sulfonyl Fluorides (`SuFEx`: `-SO2-F`), Cyanamides / Aliphatic Amino-Nitriles (`-N-C#N` / `-CH(R)-C#N`), & Boronic Acids (`-B(OH)2`)
    for u in 0..n {
        if claimed_electrophiles.contains(&u) {
            continue;
        }
        // SuFEx: S(=O)(=O)-F
        if let Some(oxs) = sulfonyl_oxygens(u) {
            if let Some(&(f_idx, _)) = mol.adj[u]
                .iter()
                .find(|&&(nb, _)| mol.atoms[nb].elem == Element::F)
            {
                let att = mol.adj[u]
                    .iter()
                    .map(|&(nb, _)| nb)
                    .find(|&nb| nb != f_idx && nb != oxs[0] && nb != oxs[1]);
                let wh_atoms = vec![u, oxs[0], oxs[1], f_idx];
                let ind_shift = compute_scaffold_inductive_shift(mol, &charges, att, &wh_atoms);
                let tuned_t12 = (18.0 * 10.0f64.powf(ind_shift)).clamp(1.0, 200.0);
                let (k_inact, ki_nm, eff, tau_h) = compute_two_step_kinetics(
                    mol.pact,
                    tuned_t12,
                    WarheadReversibility::Irreversible,
                    1.0,
                );
                claimed_electrophiles.insert(u);
                results.push(DetectedWarhead {
                    warhead_code: "sufex_so2f",
                    warhead_name: "Sulfonyl Fluoride (SuFEx -SO2F)",
                    mechanism: "Context-dependent Lys/Tyr/His sulfur(VI) fluoride exchange",
                    target_residue: TargetResidue::LysineTyrosine,
                    reversibility: WarheadReversibility::Irreversible,
                    window: classify_irrev_gsh_window(tuned_t12),
                    electrophile_atom: u,
                    attachment_atom: att,
                    warhead_atoms: wh_atoms,
                    base_gsh_half_life_h: 18.0,
                    inductive_shift_log10: ind_shift,
                    gsh_half_life_h: tuned_t12,
                    k_gsh_per_h: std::f64::consts::LN_2 / tuned_t12,
                    k_inact_per_min: k_inact,
                    ki_nm,
                    kinact_over_ki_m_inv_s_inv: eff,
                    residence_time_h: tau_h,
                });
                continue;
            }
        }

        // Boronic Acid: -B(OH)2 / boronate ester
        if mol.atoms[u].elem == Element::B {
            let oxy: Vec<usize> = mol.adj[u]
                .iter()
                .map(|&(nb, _)| nb)
                .filter(|&nb| mol.atoms[nb].elem == Element::O)
                .collect();
            if oxy.len() >= 2 {
                let att = mol.adj[u]
                    .iter()
                    .map(|&(nb, _)| nb)
                    .find(|&nb| !oxy.contains(&nb));
                let mut wh_atoms = vec![u];
                wh_atoms.extend_from_slice(&oxy);
                let (k_inact, ki_nm, eff, tau_h) = compute_two_step_kinetics(
                    mol.pact,
                    90.0,
                    WarheadReversibility::ReversibleCovalent,
                    1.1,
                );
                claimed_electrophiles.insert(u);
                results.push(DetectedWarhead {
                    warhead_code: "boronic_acid_rev",
                    warhead_name: "Boronic Acid (-B(OH)2, Reversible Covalent)",
                    mechanism: "Bortezomib-class reversible tetrahedral Ser/Thr boronate adduct",
                    target_residue: TargetResidue::SerineThreonine,
                    reversibility: WarheadReversibility::ReversibleCovalent,
                    window: ReactivityWindow::ReversibleCovalent,
                    electrophile_atom: u,
                    attachment_atom: att,
                    warhead_atoms: wh_atoms,
                    base_gsh_half_life_h: 120.0,
                    inductive_shift_log10: 0.0,
                    gsh_half_life_h: 120.0,
                    k_gsh_per_h: std::f64::consts::LN_2 / 120.0,
                    k_inact_per_min: k_inact,
                    ki_nm,
                    kinact_over_ki_m_inv_s_inv: eff,
                    residence_time_h: tau_h,
                });
                continue;
            }
        }

        // Cyanamide (`R2N-C#N`) or alpha-amino aliphatic nitrile (`Nirmatrelvir` / `Saxagliptin` class)
        if mol.atoms[u].elem == Element::C && mol.atoms[u].hyb == Hybridization::Sp {
            if let Some(&(n_term, bi)) = mol.adj[u].iter().find(|&&(nb, bi)| {
                mol.atoms[nb].elem == Element::N && mol.bonds[bi].order == BondOrder::Triple
            }) {
                if let Some(&(att, _)) = mol.adj[u].iter().find(|&&(nb, _)| nb != n_term) {
                    let is_cyanamide = mol.atoms[att].elem == Element::N;
                    let is_alpha_amino_nitrile = mol.atoms[att].elem == Element::C
                        && !mol.atoms[att].aromatic
                        && mol.atoms[att].hyb == Hybridization::Sp3
                        && mol.adj[att]
                            .iter()
                            .any(|&(nb, _)| mol.atoms[nb].elem == Element::N);
                    if is_cyanamide || is_alpha_amino_nitrile {
                        let wh_atoms = vec![u, n_term];
                        let (k_inact, ki_nm, eff, tau_h) = compute_two_step_kinetics(
                            mol.pact,
                            85.0,
                            WarheadReversibility::ReversibleCovalent,
                            1.0,
                        );
                        claimed_electrophiles.insert(u);
                        results.push(DetectedWarhead {
                            warhead_code: "cyanamide_nitrile_rev",
                            warhead_name: if is_cyanamide {
                                "Cyanamide (-N-C#N, Reversible Covalent)"
                            } else {
                                "Peptidomimetic Nitrile (-CH-C#N, Reversible Covalent)"
                            },
                            mechanism: "Nirmatrelvir-class reversible Cys thioimidate / Ser imidate adduct",
                            target_residue: TargetResidue::Cysteine,
                            reversibility: WarheadReversibility::ReversibleCovalent,
                            window: ReactivityWindow::ReversibleCovalent,
                            electrophile_atom: u,
                            attachment_atom: Some(att),
                            warhead_atoms: wh_atoms,
                            base_gsh_half_life_h: 95.0,
                            inductive_shift_log10: 0.0,
                            gsh_half_life_h: 95.0,
                            k_gsh_per_h: std::f64::consts::LN_2 / 95.0,
                            k_inact_per_min: k_inact,
                            ki_nm,
                            kinact_over_ki_m_inv_s_inv: eff,
                            residence_time_h: tau_h,
                        });
                    }
                }
            }
        }
    }

    // 4. Epoxides & Aziridines (3-membered strained heterocyclic electrophiles)
    for ring in &mol.rings {
        if ring.len() == 3 {
            if let Some(&het) = ring
                .iter()
                .find(|&&a| matches!(mol.atoms[a].elem, Element::O | Element::N))
            {
                if let Some(&c_elec) = ring
                    .iter()
                    .find(|&&a| a != het && mol.atoms[a].elem == Element::C)
                {
                    if !claimed_electrophiles.contains(&c_elec) {
                        let att = ring
                            .iter()
                            .flat_map(|&ra| mol.adj[ra].iter().map(|&(nb, _)| nb))
                            .find(|nb| !ring.contains(nb));
                        let base_t12 = if mol.atoms[het].elem == Element::O {
                            4.5
                        } else {
                            12.0
                        };
                        let (k_inact, ki_nm, eff, tau_h) = compute_two_step_kinetics(
                            mol.pact,
                            base_t12,
                            WarheadReversibility::Irreversible,
                            1.15,
                        );
                        claimed_electrophiles.insert(c_elec);
                        results.push(DetectedWarhead {
                            warhead_code: "epoxide_aziridine",
                            warhead_name: if mol.atoms[het].elem == Element::O {
                                "Strained Epoxide (Oxirane)"
                            } else {
                                "Strained Aziridine"
                            },
                            mechanism: "Carfilzomib-class strain-release ring opening",
                            target_residue: TargetResidue::Cysteine,
                            reversibility: WarheadReversibility::Irreversible,
                            window: classify_irrev_gsh_window(base_t12),
                            electrophile_atom: c_elec,
                            attachment_atom: att,
                            warhead_atoms: ring.clone(),
                            base_gsh_half_life_h: base_t12,
                            inductive_shift_log10: 0.0,
                            gsh_half_life_h: base_t12,
                            k_gsh_per_h: std::f64::consts::LN_2 / base_t12,
                            k_inact_per_min: k_inact,
                            ki_nm,
                            kinact_over_ki_m_inv_s_inv: eff,
                            residence_time_h: tau_h,
                        });
                    }
                }
            }
        }
    }

    results
}

/// Identifies a scaffold attachment point `(anchor_atom, atoms_to_delete, needs_nh_linker, mode_label)`
/// for either replacing an existing warhead OR covalentizing a non-covalent reversible inhibitor.
fn find_swap_attachment_site(
    mol: &Molecule,
    detected: &[DetectedWarhead],
) -> Option<(usize, BTreeSet<usize>, bool, String)> {
    // Case A: Molecule already carries a detected exocyclic warhead with a clean single-bond attachment
    for dw in detected {
        if let Some(att) = dw.attachment_atom {
            if !mol.atoms[att].in_ring || mol.atoms[att].elem == Element::N || mol.atoms[att].aromatic {
                // Collect connected component on the warhead side of `att`
                let mut to_remove: BTreeSet<usize> = BTreeSet::new();
                let mut q = VecDeque::new();
                for &wa in &dw.warhead_atoms {
                    if wa != att {
                        to_remove.insert(wa);
                        q.push_back(wa);
                    }
                }
                while let Some(u) = q.pop_front() {
                    for &(nb, _) in &mol.adj[u] {
                        if nb != att && !to_remove.contains(&nb) {
                            to_remove.insert(nb);
                            q.push_back(nb);
                        }
                    }
                }
                if !to_remove.is_empty() && to_remove.len() < mol.atoms.len() {
                    let needs_nh = mol.atoms[att].elem != Element::N;
                    return Some((
                        att,
                        to_remove,
                        needs_nh,
                        format!("Direct Swap at {}#{}", mol.atoms[att].elem.symbol(), att + 1),
                    ));
                }
            }
        }
    }

    // Case B: Non-covalent lead covalentization — look for:
    // 1. Terminal alkyne `-C#CH` on aryl ring (e.g. Erlotinib C#C -> meta-acrylamido handle!)
    for b in &mol.bonds {
        if b.order == BondOrder::Triple
            && mol.atoms[b.a].elem == Element::C
            && mol.atoms[b.b].elem == Element::C
        {
            for &(c_in, c_term) in &[(b.a, b.b), (b.b, b.a)] {
                if mol.adj[c_term].len() == 1 && mol.adj[c_in].len() == 2 {
                    if let Some(&(ar, _)) = mol.adj[c_in].iter().find(|&&(x, _)| x != c_term) {
                        if mol.atoms[ar].aromatic {
                            let mut rem = BTreeSet::new();
                            rem.insert(c_in);
                            rem.insert(c_term);
                            return Some((
                                ar,
                                rem,
                                true,
                                format!("Alkyne -> Aryl-NH Swap at C#{}", ar + 1),
                            ));
                        }
                    }
                }
            }
        }
    }

    // 2. Primary or secondary aliphatic/aromatic amine (`-NH2` or `-NH-`) with free valence
    for (u, a) in mol.atoms.iter().enumerate() {
        if a.elem == Element::N && !a.aromatic && a.total_h() >= 1 && mol.adj[u].len() <= 2 {
            // Prefer non-core linker NH if secondary, or terminal NH2
            if mol.adj[u].len() == 1 || !a.in_ring {
                return Some((
                    u,
                    BTreeSet::new(),
                    false,
                    format!("N-Acylation at N#{}", u + 1),
                ));
            }
        }
        if a.elem == Element::N && a.in_ring && !a.aromatic && a.total_h() >= 1 {
            return Some((
                u,
                BTreeSet::new(),
                false,
                format!("Ring N-Acylation at N#{}", u + 1),
            ));
        }
    }

    // 3. Aryl `-Br`, `-I`, or `-Cl` exit vector -> Buchwald/amide `-NH-Warhead` installation
    for el_target in [Element::Br, Element::I, Element::Cl] {
        for (u, a) in mol.atoms.iter().enumerate() {
            if a.elem == el_target && mol.adj[u].len() == 1 {
                let ar = mol.adj[u][0].0;
                if mol.atoms[ar].aromatic {
                    let mut rem = BTreeSet::new();
                    rem.insert(u);
                    return Some((
                        ar,
                        rem,
                        true,
                        format!("Aryl-{} -> -NH-Warhead at C#{}", a.elem.symbol(), ar + 1),
                    ));
                }
            }
        }
    }

    // 4. Terminal aryl methoxy `-O-CH3` -> `-NH-Warhead` (e.g. 6-methoxyquinazolines -> 6-acrylamidoquinazolines like Canertinib/Afatinib!)
    for (u, a) in mol.atoms.iter().enumerate() {
        if a.elem == Element::O && !a.in_ring && mol.adj[u].len() == 2 {
            let n0 = mol.adj[u][0].0;
            let n1 = mol.adj[u][1].0;
            for &(ar, me) in &[(n0, n1), (n1, n0)] {
                if mol.atoms[ar].aromatic
                    && mol.atoms[me].elem == Element::C
                    && mol.adj[me].len() == 1
                {
                    let mut rem = BTreeSet::new();
                    rem.insert(u);
                    rem.insert(me);
                    return Some((
                        ar,
                        rem,
                        true,
                        format!("Aryl-OMe -> -NH-Warhead at C#{}", ar + 1),
                    ));
                }
            }
        }
    }

    None
}

fn push_new_atom(mol: &mut Molecule, elem: Element) -> usize {
    let idx = mol.atoms.len();
    mol.atoms.push(Atom {
        idx,
        elem,
        charge: 0,
        aromatic: false,
        bracketed: false,
        explicit_h: 0,
        implicit_h: 0,
        chiral: 0,
        in_ring: false,
        ring_sizes: Vec::new(),
        hyb: Hybridization::Sp3,
        pos2d: [0.0, 0.0],
    });
    idx
}

fn push_new_bond(mol: &mut Molecule, a: usize, b: usize, order: BondOrder) {
    let idx = mol.bonds.len();
    mol.bonds.push(Bond {
        idx,
        a,
        b,
        order,
        aromatic: false,
        in_ring: false,
        kekule_order: match order {
            BondOrder::Single | BondOrder::Aromatic => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
        },
    });
}

/// Prunes `remove_set` from `parent` and returns `(pruned_mol, mapped_anchor_idx)`.
fn prune_warhead_subgraph(
    parent: &Molecule,
    anchor: usize,
    remove_set: &BTreeSet<usize>,
) -> Option<(Molecule, usize)> {
    if remove_set.contains(&anchor) {
        return None;
    }
    let mut old_to_new = vec![None; parent.atoms.len()];
    let mut out = Molecule::new("");
    out.id = parent.id.clone();
    out.name = parent.name.clone();
    out.target = parent.target.clone();
    out.metric = parent.metric.clone();
    out.raw_nm = parent.raw_nm;
    out.pact = parent.pact;
    out.source_url = parent.source_url.clone();

    for (old_i, a) in parent.atoms.iter().enumerate() {
        if remove_set.contains(&old_i) {
            continue;
        }
        let new_i = out.atoms.len();
        let mut na = a.clone();
        na.idx = new_i;
        // Clear explicit_h on anchor so implicit valence recalculates cleanly after grafting
        if old_i == anchor {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[old_i] = Some(new_i);
    }

    for b in &parent.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = out.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }

    let new_anchor = old_to_new[anchor]?;
    Some((out, new_anchor))
}

struct WarheadTemplateSpec {
    code: &'static str,
    name: &'static str,
    precedent: &'static str,
    rev: WarheadReversibility,
    res: TargetResidue,
}

const WARHEAD_SWAP_TEMPLATES: [WarheadTemplateSpec; 12] = [
    WarheadTemplateSpec {
        code: "propenamide",
        name: "Acrylamide (-NHC(=O)CH=CH2)",
        precedent: "Osimertinib / Ibrutinib",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "dmamb_enamide",
        name: "(E)-4-(Dimethylamino)but-2-enamide",
        precedent: "Afatinib / Neratinib",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "butynamide",
        name: "But-2-ynamide (-NHC(=O)C#CCH3)",
        precedent: "Acalabrutinib",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "cyanoacrylamide_rev",
        name: "Alpha-Cyanoacrylamide (Reversible-Cov)",
        precedent: "Rilzabrutinib (Serafimova-Taunton)",
        rev: WarheadReversibility::ReversibleCovalent,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "methacrylamide",
        name: "Methacrylamide (-NHC(=O)C(Me)=CH2)",
        precedent: "Attenuated Selective Acrylamide",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "crotonamide",
        name: "(E)-Crotonamide (-NHC(=O)CH=CHMe)",
        precedent: "Beta-Methyl Tuned Acrylamide",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "propiolamide",
        name: "Terminal Propiolamide (-NHC(=O)C#CH)",
        precedent: "Fast Ynamide Probe",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "chloroacetamide",
        name: "Alpha-Chloroacetamide (-NHC(=O)CH2Cl)",
        precedent: "ABPP Chemoproteomic Cys Probe",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "fluoroacetamide",
        name: "Alpha-Fluoroacetamide (-NHC(=O)CH2F)",
        precedent: "Low-Reactivity Pocket-Templated SN2",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "cyanamide_nitrile_rev",
        name: "Cyanamide (-NH-C#N, Reversible-Cov)",
        precedent: "Nirmatrelvir / Pyrrolidine-Cyanamide",
        rev: WarheadReversibility::ReversibleCovalent,
        res: TargetResidue::Cysteine,
    },
    WarheadTemplateSpec {
        code: "sufex_so2f",
        name: "Sulfonyl Fluoride (SuFEx -NHSO2F)",
        precedent: "Lys/Tyr/His Chemoproteomic Hub",
        rev: WarheadReversibility::Irreversible,
        res: TargetResidue::LysineTyrosine,
    },
    WarheadTemplateSpec {
        code: "propionamide_ctrl",
        name: "Propionamide Saturated Control (Non-Cov)",
        precedent: "Matched Reversible K_I Baseline",
        rev: WarheadReversibility::NonCovalentControl,
        res: TargetResidue::NonCovalent,
    },
];

/// Grafts template `code` onto `(base_mol, anchor_idx)` (inserting an `-NH-` linker if `needs_nh` is true),
/// returning `(built_mol, warhead_atom_indices, optional_electrophile_atom)`.
fn graft_warhead_template(
    base_mol: &Molecule,
    anchor_idx: usize,
    needs_nh: bool,
    code: &str,
) -> (Molecule, Vec<usize>, Option<usize>) {
    let mut m = base_mol.clone();
    let mut wh_atoms = Vec::new();

    let n_att = if needs_nh {
        let ni = push_new_atom(&mut m, Element::N);
        push_new_bond(&mut m, anchor_idx, ni, BondOrder::Single);
        wh_atoms.push(ni);
        ni
    } else {
        anchor_idx
    };

    let elec_atom = match code {
        "propenamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let c_b = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Double);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, c_b]);
            Some(c_b)
        }
        "dmamb_enamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let c_b = push_new_atom(&mut m, Element::C);
            let ch2 = push_new_atom(&mut m, Element::C);
            let n_me = push_new_atom(&mut m, Element::N);
            let me1 = push_new_atom(&mut m, Element::C);
            let me2 = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Double);
            push_new_bond(&mut m, c_b, ch2, BondOrder::Single);
            push_new_bond(&mut m, ch2, n_me, BondOrder::Single);
            push_new_bond(&mut m, n_me, me1, BondOrder::Single);
            push_new_bond(&mut m, n_me, me2, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, c_b, ch2, n_me, me1, me2]);
            Some(c_b)
        }
        "butynamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let c_b = push_new_atom(&mut m, Element::C);
            let me = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Triple);
            push_new_bond(&mut m, c_b, me, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, c_b, me]);
            Some(c_b)
        }
        "cyanoacrylamide_rev" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let c_cn = push_new_atom(&mut m, Element::C);
            let n_cn = push_new_atom(&mut m, Element::N);
            let c_b = push_new_atom(&mut m, Element::C);
            let ch_ipr = push_new_atom(&mut m, Element::C);
            let me1 = push_new_atom(&mut m, Element::C);
            let me2 = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_cn, BondOrder::Single);
            push_new_bond(&mut m, c_cn, n_cn, BondOrder::Triple);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Double);
            push_new_bond(&mut m, c_b, ch_ipr, BondOrder::Single);
            push_new_bond(&mut m, ch_ipr, me1, BondOrder::Single);
            push_new_bond(&mut m, ch_ipr, me2, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, c_cn, n_cn, c_b, ch_ipr, me1, me2]);
            Some(c_b)
        }
        "methacrylamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let me = push_new_atom(&mut m, Element::C);
            let c_b = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, me, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Double);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, me, c_b]);
            Some(c_b)
        }
        "crotonamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let c_b = push_new_atom(&mut m, Element::C);
            let me = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Double);
            push_new_bond(&mut m, c_b, me, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, c_b, me]);
            Some(c_b)
        }
        "propiolamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let c_a = push_new_atom(&mut m, Element::C);
            let c_b = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, c_a, BondOrder::Single);
            push_new_bond(&mut m, c_a, c_b, BondOrder::Triple);
            wh_atoms.extend_from_slice(&[c_co, o_co, c_a, c_b]);
            Some(c_b)
        }
        "chloroacetamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let ch2 = push_new_atom(&mut m, Element::C);
            let cl = push_new_atom(&mut m, Element::Cl);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, ch2, BondOrder::Single);
            push_new_bond(&mut m, ch2, cl, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, ch2, cl]);
            Some(ch2)
        }
        "fluoroacetamide" => {
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let ch2 = push_new_atom(&mut m, Element::C);
            let fl = push_new_atom(&mut m, Element::F);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, ch2, BondOrder::Single);
            push_new_bond(&mut m, ch2, fl, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, ch2, fl]);
            Some(ch2)
        }
        "cyanamide_nitrile_rev" => {
            let c_cn = push_new_atom(&mut m, Element::C);
            let n_cn = push_new_atom(&mut m, Element::N);
            push_new_bond(&mut m, n_att, c_cn, BondOrder::Single);
            push_new_bond(&mut m, c_cn, n_cn, BondOrder::Triple);
            wh_atoms.extend_from_slice(&[c_cn, n_cn]);
            Some(c_cn)
        }
        "sufex_so2f" => {
            let s = push_new_atom(&mut m, Element::S);
            let o1 = push_new_atom(&mut m, Element::O);
            let o2 = push_new_atom(&mut m, Element::O);
            let fl = push_new_atom(&mut m, Element::F);
            push_new_bond(&mut m, n_att, s, BondOrder::Single);
            push_new_bond(&mut m, s, o1, BondOrder::Double);
            push_new_bond(&mut m, s, o2, BondOrder::Double);
            push_new_bond(&mut m, s, fl, BondOrder::Single);
            wh_atoms.extend_from_slice(&[s, o1, o2, fl]);
            Some(s)
        }
        _ => {
            // Saturated propionamide non-covalent control (-NH-C(=O)-CH2-CH3)
            let c_co = push_new_atom(&mut m, Element::C);
            let o_co = push_new_atom(&mut m, Element::O);
            let ch2 = push_new_atom(&mut m, Element::C);
            let me = push_new_atom(&mut m, Element::C);
            push_new_bond(&mut m, n_att, c_co, BondOrder::Single);
            push_new_bond(&mut m, c_co, o_co, BondOrder::Double);
            push_new_bond(&mut m, c_co, ch2, BondOrder::Single);
            push_new_bond(&mut m, ch2, me, BondOrder::Single);
            wh_atoms.extend_from_slice(&[c_co, o_co, ch2, me]);
            None
        }
    };

    m.sanitize();
    m.smiles = molecule_to_smiles(&m);
    compute_2d_coords(&mut m);
    (m, wh_atoms, elec_atom)
}

/// Evaluates dataset-wide fast covalent reactivity summary for `mol`.
pub fn evaluate_covalent_summary(mol: &Molecule) -> CovalentDatasetSummary {
    let detected = detect_covalent_warheads(mol);
    let swap_ok = find_swap_attachment_site(mol, &detected).is_some();
    if let Some(d0) = detected.first() {
        CovalentDatasetSummary {
            has_warhead: true,
            warhead_count: detected.len(),
            primary_warhead_name: d0.warhead_name,
            target_residue: d0.target_residue,
            reversibility: d0.reversibility,
            window: d0.window,
            gsh_half_life_h: Some(d0.gsh_half_life_h),
            kinact_over_ki: Some(d0.kinact_over_ki_m_inv_s_inv),
            swap_handle_available: swap_ok,
        }
    } else {
        CovalentDatasetSummary {
            has_warhead: false,
            warhead_count: 0,
            primary_warhead_name: "None (Non-Covalent)",
            target_residue: TargetResidue::NonCovalent,
            reversibility: WarheadReversibility::NonCovalentControl,
            window: ReactivityWindow::NonCovalent,
            gsh_half_life_h: None,
            kinact_over_ki: None,
            swap_handle_available: swap_ok,
        }
    }
}

/// Runs full covalent electrophile perception, `0..24 h` target-occupancy vs `GSH`-depletion
/// kinetic curve simulation (at `inhibitor_conc_nm`), and reversible/irreversible warhead swapping.
pub fn analyze_covalent_warheads_and_swaps(
    mol: &Molecule,
    ds: &DatasetAnalysis,
    active_sar_matrix: Option<&FreeWilsonMatrix>,
    inhibitor_conc_nm: f64,
) -> CovalentWarheadReport {
    let detected = detect_covalent_warheads(mol);
    let primary_window = detected
        .first()
        .map(|d| d.window)
        .unwrap_or(ReactivityWindow::NonCovalent);

    let conc_nm = inhibitor_conc_nm.clamp(0.5, 10_000.0);
    let mut time_grid_h = Vec::with_capacity(KINETIC_GRID_LEN);
    let mut target_occupancy_curve = Vec::with_capacity(KINETIC_GRID_LEN);
    let mut gsh_adduct_curve = Vec::with_capacity(KINETIC_GRID_LEN);

    for step in 0..KINETIC_GRID_LEN {
        let t_h = step as f64;
        time_grid_h.push(t_h as f32);

        if let Some(dw) = detected.first() {
            // Off-target 5 mM GSH pseudo-first-order consumption: 1 - exp(-k_gsh * t_h)
            let gsh_pct = (1.0 - (-dw.k_gsh_per_h * t_h).exp()) * 100.0;
            gsh_adduct_curve.push(gsh_pct.clamp(0.0, 100.0) as f32);

            // Target occupancy at [I] = conc_nm
            // Reversible pre-equilibrium fraction f_bound = [I] / ([I] + K_I)
            let f_bound = conc_nm / (conc_nm + dw.ki_nm.max(0.01));
            let k_obs_per_h = dw.k_inact_per_min * 60.0 * f_bound;
            let occ_pct = match dw.reversibility {
                WarheadReversibility::Irreversible => (1.0 - (-k_obs_per_h * t_h).exp()) * 100.0,
                WarheadReversibility::ReversibleCovalent => {
                    let k_off_per_h = 1.0 / dw.residence_time_h.max(0.2);
                    let k_sum = k_obs_per_h + k_off_per_h;
                    let eq_cov = k_obs_per_h / k_sum.max(1e-9);
                    let total_occ = f_bound + (1.0 - f_bound) * eq_cov * (1.0 - (-k_sum * t_h).exp());
                    total_occ * 100.0
                }
                WarheadReversibility::NonCovalentControl => f_bound * 100.0,
            };
            target_occupancy_curve.push(occ_pct.clamp(0.0, 100.0) as f32);
        } else {
            // Pure reversible equilibrium occupancy from mol.pact
            let ki_nm = (10.0f64.powf(9.0 - mol.pact.clamp(4.3, 10.5))).clamp(0.05, 50_000.0);
            let occ_pct = (conc_nm / (conc_nm + ki_nm)) * 100.0;
            target_occupancy_curve.push(occ_pct as f32);
            gsh_adduct_curve.push(0.0);
        }
    }

    // Enumerate Reversible & Irreversible Warhead Swap Panel
    let mut swaps: Vec<WarheadSwapCandidate> = Vec::new();
    if let Some((anchor, remove_set, needs_nh, mode_lbl)) = find_swap_attachment_site(mol, &detected)
    {
        if let Some((pruned_base, mapped_anchor)) =
            prune_warhead_subgraph(mol, anchor, &remove_set)
        {
            for spec in &WARHEAD_SWAP_TEMPLATES {
                let (mut cand_mol, wh_atoms, elec_opt) =
                    graft_warhead_template(&pruned_base, mapped_anchor, needs_nh, spec.code);
                cand_mol.name = format!("{} [{}]", mol.name, spec.code);

                // Predict reversible scaffold recognition pAct via Free-Wilson or parent baseline
                let fw_pred = predict_best_free_wilson(
                    &cand_mol,
                    active_sar_matrix,
                    ds.default_sar_matrix.as_ref(),
                    &ds.mols,
                    &ds.profiles,
                );
                let base_pact = fw_pred
                    .as_ref()
                    .map(|p| p.predicted_pact)
                    .unwrap_or(mol.pact)
                    .clamp(5.0, 10.8);
                cand_mol.pact = base_pact;

                let alerts = scan_structural_alerts(&cand_mol);
                let prof = evaluate_physchem(&cand_mol, alerts.len());
                let sa = evaluate_synthetic_accessibility(&cand_mol);
                let pka = evaluate_physiological_pka_74(&cand_mol);
                let cand_det = detect_covalent_warheads(&cand_mol);

                let (win, gsh_opt, k_inact, ki_nm, eff, tau_h) =
                    if spec.rev == WarheadReversibility::NonCovalentControl {
                        let ki = (10.0f64.powf(9.0 - base_pact)).clamp(0.05, 50_000.0);
                        (ReactivityWindow::NonCovalent, None, 0.0, ki, 0.0, 0.15)
                    } else if let Some(dw) = cand_det.first() {
                        (
                            dw.window,
                            Some(dw.gsh_half_life_h),
                            dw.k_inact_per_min,
                            dw.ki_nm,
                            dw.kinact_over_ki_m_inv_s_inv,
                            dw.residence_time_h,
                        )
                    } else {
                        let ki = (10.0f64.powf(9.0 - base_pact)).clamp(0.05, 50_000.0);
                        (spec.rev_to_window(), Some(24.0), 0.05, ki, 8.0e4, 6.0)
                    };

                // Multi-objective TCI Selectivity Score [0..100]:
                // Rewards Tuned Sweet Spot (1.5-36 h GSH) & Reversible-Covalent warheads, high k_inact/K_I, QED, and tractable SAScore
                let window_bonus = match win {
                    ReactivityWindow::TunedClinicalSweetSpot => 38.0,
                    ReactivityWindow::ReversibleCovalent => 36.0,
                    ReactivityWindow::MildSelectiveIrreversible => 28.0,
                    ReactivityWindow::FastIrreversible => 18.0,
                    ReactivityWindow::NonCovalent => 16.0,
                    ReactivityWindow::HyperReactivePromiscuous => 2.0,
                };
                let eff_bonus = if eff > 0.0 {
                    ((eff.log10() - 2.0) * 6.5).clamp(0.0, 26.0)
                } else {
                    (base_pact - 5.5).clamp(0.0, 14.0)
                };
                let drug_bonus = prof.qed_mean * 22.0 + ((6.5 - sa.sa_score).clamp(0.0, 4.5)) * 3.0;
                let tci_score = (window_bonus + eff_bonus + drug_bonus).clamp(1.0, 99.5);

                swaps.push(WarheadSwapCandidate {
                    rank: 0,
                    template_code: spec.code,
                    template_name: spec.name,
                    clinical_precedent: spec.precedent,
                    swap_mode_label: mode_lbl.clone(),
                    reversibility: spec.rev,
                    target_residue: spec.res,
                    window: win,
                    gsh_half_life_h: gsh_opt,
                    k_inact_per_min: k_inact,
                    ki_nm,
                    kinact_over_ki: eff,
                    residence_time_h: tau_h,
                    predicted_pact: base_pact,
                    qed: prof.qed_mean,
                    sa_score: sa.sa_score,
                    logd_74: pka.microstate_logd_74,
                    tci_selectivity_score: tci_score,
                    canonical_smiles: cand_mol.smiles.clone(),
                    warhead_atoms: wh_atoms,
                    electrophile_atom: elec_opt,
                    mol: cand_mol,
                });
            }
        }
    }

    swaps.sort_by(|a, b| {
        b.tci_selectivity_score
            .partial_cmp(&a.tci_selectivity_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, s) in swaps.iter_mut().enumerate() {
        s.rank = i + 1;
    }

    CovalentWarheadReport {
        detected,
        primary_window,
        inhibitor_conc_nm: conc_nm,
        time_grid_h,
        target_occupancy_curve,
        gsh_adduct_curve,
        swaps,
    }
}

impl WarheadTemplateSpec {
    fn rev_to_window(&self) -> ReactivityWindow {
        match self.rev {
            WarheadReversibility::ReversibleCovalent => ReactivityWindow::ReversibleCovalent,
            WarheadReversibility::Irreversible => ReactivityWindow::TunedClinicalSweetSpot,
            WarheadReversibility::NonCovalentControl => ReactivityWindow::NonCovalent,
        }
    }
}

/// Exports enumerated covalent warhead swap analogs to CSV.
pub fn export_warhead_swaps_csv(swaps: &[WarheadSwapCandidate]) -> String {
    let mut out = String::from(
        "rank,template_code,warhead_name,clinical_precedent,reversibility,target_residue,reactivity_window,gsh_half_life_h,k_inact_min,ki_nm,kinact_over_ki_m_inv_s_inv,residence_time_h,predicted_pact,qed,sa_score,logd_74,tci_selectivity_score,smiles\n",
    );
    for s in swaps {
        let gsh_str = s
            .gsh_half_life_h
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{},{:.4},{:.3},{:.1},{:.2},{:.2},{:.3},{:.2},{:.2},{:.1},\"{}\"\n",
            s.rank,
            s.template_code,
            s.template_name,
            s.clinical_precedent,
            s.reversibility.short_label(),
            s.target_residue.label(),
            s.window.short_label(),
            gsh_str,
            s.k_inact_per_min,
            s.ki_nm,
            s.kinact_over_ki,
            s.residence_time_h,
            s.predicted_pact,
            s.qed,
            s.sa_score,
            s.logd_74,
            s.tci_selectivity_score,
            s.canonical_smiles
        ));
    }
    out
}
