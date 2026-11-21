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

//! Medicinal-Chemistry Acyclic-to-Macrocycle Designer: 3D-Guided Terminal-Pair Finder,
//! Ring-Closing Metathesis (`RCM` Grubbs olefin / hydrogenated alkane) & Macrolactamization
//! (`HATU` amide / `N`-Me amide / glycolamide / diether / `CuAAC` click) Enumerator, Kier `Phi`
//! Flexibility Reducer, and 3D Conformational Preorganization Entropy (`-T * Delta S_conf` &
//! Kabsch Core-RMSD Strain) Scorer.
//!
//! Primary References:
//! - Giordanetto & Kihlberg (2014), *J. Med. Chem.* 57(2), 278-295 ("Macrocyclic drugs and
//!   clinical candidates: what can medicinal chemists learn from their properties?":
//!   `https://pubmed.ncbi.nlm.nih.gov/24044773/`) & Kihlberg (2023) (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10150360/`)
//! - Johnson et al. (2014), *J. Med. Chem.* 57(11), 4720-4744 ("Discovery of PF-06463922
//!   (Lorlatinib), a potent brain-penetrant macrocyclic ALK/ROS1 inhibitor":
//!   `https://pubmed.ncbi.nlm.nih.gov/24819116/`)
//! - Kinase Inhibitor Macrocycles & Conformational Entropy Restriction (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10880908/`)
//! - Kier Flexibility Index `Phi` in Macrocyclic Kinase Inhibitors (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10425998/`)
//! - Synthetic Methodologies Toward Macrocyclic Drugs (`RCM`, Macrolactamization, `CuAAC`:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4374646/`)

use crate::alerts::scan_structural_alerts;
use crate::conformer3d::{
    evaluate_3d_summary_profile, format_3d_conformer_sdf, generate_3d_conformer_ensemble,
    kabsch_align_3d, ConformerGenConfig,
};
use crate::layout2d::compute_2d_coords;
use crate::mol::{Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Synthetic macrocyclization ring-closing reaction family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MacroClosureReaction {
    RingClosingMetathesis,
    RcmHydrogenated,
    Macrolactamization,
    NMethylMacrolactam,
    Macroetherification,
    ClickTriazole,
}

impl MacroClosureReaction {
    pub fn label(self) -> &'static str {
        match self {
            Self::RingClosingMetathesis => "Grubbs RCM Alkene",
            Self::RcmHydrogenated => "RCM + H2 Alkane",
            Self::Macrolactamization => "Macrolactam (-CONH-)",
            Self::NMethylMacrolactam => "N-Me Macrolactam (-CONMe-)",
            Self::Macroetherification => "SNAr / Williamson Diether",
            Self::ClickTriazole => "CuAAC 1,4-Triazole",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::RingClosingMetathesis => [52, 211, 153], // Emerald
            Self::RcmHydrogenated => [56, 189, 248],       // Sky Blue
            Self::Macrolactamization => [251, 191, 36],    // Amber
            Self::NMethylMacrolactam => [244, 114, 182],   // Pink (CNS Lorlatinib-class)
            Self::Macroetherification => [34, 211, 238],   // Cyan
            Self::ClickTriazole => [192, 132, 252],        // Violet
        }
    }
}

/// Giordanetto-Kihlberg (2014 / 2023) oral & CNS macrocycle druggability regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroDruggabilityClass {
    CnsPenetrantMacrocycle,
    OralRo5Macrocycle,
    OralBro5Macrocycle,
    ParenteralMacrocycle,
}

impl MacroDruggabilityClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::CnsPenetrantMacrocycle => "CNS-Penetrant Oral (Lorlatinib Space)",
            Self::OralRo5Macrocycle => "Compact Oral Macrocycle (MW<=550)",
            Self::OralBro5Macrocycle => "Oral bRo5 Macrocycle (Giordanetto Window)",
            Self::ParenteralMacrocycle => "High-Polarity / Parenteral Macrocycle",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::CnsPenetrantMacrocycle => "CNS Oral",
            Self::OralRo5Macrocycle => "Compact Oral",
            Self::OralBro5Macrocycle => "Oral bRo5",
            Self::ParenteralMacrocycle => "Parenteral",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::CnsPenetrantMacrocycle => [244, 114, 182], // Pink
            Self::OralRo5Macrocycle => [52, 211, 153],       // Emerald
            Self::OralBro5Macrocycle => [56, 189, 248],      // Sky Blue
            Self::ParenteralMacrocycle => [251, 191, 36],    // Amber
        }
    }
}

/// Candidate pair of peripheral attachment atoms `(anchor_a, anchor_b)` on an acyclic parent lead
/// suitable for intramolecular macrocyclization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CyclizableTerminalPair {
    pub pair_idx: usize,
    pub anchor_a: usize,
    pub anchor_b: usize,
    pub prune_atoms: Vec<usize>,
    pub backbone_bond_span: usize,
    /// Through-space 3D Euclidean distance (`A`) between `anchor_a` and `anchor_b` in the parent's
    /// `MMFF-lite` low-energy 3D conformer.
    pub spatial_dist_3d_a: f32,
    pub label: String,
    #[serde(skip_deserializing)]
    pub rationale: &'static str,
    pub suitability_score: f64,
}

/// Thermodynamic 3D preorganization & Kier `Phi` conformational restriction metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreorganizationMetrics {
    pub macro_ring_size: usize,
    pub parent_kier_phi: f64,
    pub macro_kier_phi: f64,
    pub delta_kier_phi: f64,
    pub effective_rotors_frozen: f64,
    /// Entropic preorganization gain `-T * Delta S_conf` (`kcal/mol`, negative = favorable!).
    pub entropy_ddg_kcal: f64,
    /// Conformational core-distortion strain penalty (`kcal/mol`, positive = unfavorable strain).
    pub core_strain_penalty_kcal: f64,
    /// Net binding free-energy improvement `ddG_bind = entropy_ddg + core_strain_penalty` (`kcal/mol`).
    pub net_ddg_bind_kcal: f64,
    /// Predicted potency fold-improvement `10^(-net_ddg_bind / 1.364)` over the acyclic parent.
    pub potency_fold_boost: f64,
    pub predicted_macro_pact: f64,
    /// 3D Kabsch heavy-atom RMSD (`A`) between the conserved pharmacophore core in the macrocycle
    /// vs the acyclic parent conformer.
    pub core_rmsd_3d_a: f32,
    pub active_conformer_boltzmann_pct: f64,
}

/// Single assembled macrocyclic analog (`RCM`, macrolactam, `N`-Me lactam, diether, or triazole).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MacrocycleCandidate {
    pub rank: usize,
    pub candidate_id: String,
    #[serde(skip_deserializing)]
    pub tether_code: &'static str,
    #[serde(skip_deserializing)]
    pub tether_name: &'static str,
    pub reaction: MacroClosureReaction,
    #[serde(skip_deserializing)]
    pub clinical_precedent: &'static str,
    pub tether_chain_atoms: usize,
    pub pair_label: String,
    pub preorg: PreorganizationMetrics,
    pub druggability: MacroDruggabilityClass,
    pub physchem: PhyschemProfile,
    pub logd_74: f64,
    pub sa_score: f64,
    /// Composite Macrocycle Design Score `[0..100]` balancing `-ddG_bind` preorganization, low 3D
    /// core RMSD, `CNS-MPO`/`QED`, and synthetic tractability.
    pub macrocycle_score: f64,
    pub canonical_smiles: String,
    pub core_atoms: Vec<usize>,
    pub tether_atoms: Vec<usize>,
    pub macro_ring_atoms: Vec<usize>,
    pub coords3d: Vec<[f32; 3]>,
    pub aligned_parent_coords3d: Vec<[f32; 3]>,
    pub gasteiger_charges: Vec<f32>,
    pub mmff_energy_kcal: f64,
    pub mol: Molecule,
}

/// Complete acyclic-to-macrocycle design & 3D preorganization report for a parent molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MacrocycleDesignReport {
    pub parent_name: String,
    pub parent_pact: f64,
    pub parent_kier_phi: f64,
    pub parent_coords3d: Vec<[f32; 3]>,
    pub terminal_pairs: Vec<CyclizableTerminalPair>,
    pub active_pair_idx: usize,
    pub candidates: Vec<MacrocycleCandidate>,
}

struct TetherTemplateSpec {
    code: &'static str,
    name: &'static str,
    reaction: MacroClosureReaction,
    precedent: &'static str,
    /// SMILES with `Cl` marking `anchor_a` end and `I` marking `anchor_b` end!
    dummy_smiles: &'static str,
    chain_len: usize,
    ideal_span_a: f64,
}

const MACRO_TETHER_LIBRARY: [TetherTemplateSpec; 10] = [
    TetherTemplateSpec {
        code: "rcm_c4_ene",
        name: "RCM (E)-But-2-ene (-CH2CH=CHCH2-)",
        reaction: MacroClosureReaction::RingClosingMetathesis,
        precedent: "Simeprevir / Grubbs II RCM",
        dummy_smiles: "ClCC=CCI",
        chain_len: 4,
        ideal_span_a: 4.5,
    },
    TetherTemplateSpec {
        code: "rcm_o_c4_ene",
        name: "RCM Oxa-Pentene (-OCH2CH=CHCH2-)",
        reaction: MacroClosureReaction::RingClosingMetathesis,
        precedent: "Pacritinib Oxa-RCM Bridge",
        dummy_smiles: "ClOCC=CCI",
        chain_len: 5,
        ideal_span_a: 5.4,
    },
    TetherTemplateSpec {
        code: "rcm_c4_sat",
        name: "RCM + H2 Butane (-CH2CH2CH2CH2-)",
        reaction: MacroClosureReaction::RcmHydrogenated,
        precedent: "Hydrogenated RCM Alkane",
        dummy_smiles: "ClCCCCI",
        chain_len: 4,
        ideal_span_a: 4.7,
    },
    TetherTemplateSpec {
        code: "rcm_o_c5_sat",
        name: "RCM + H2 Pentyloxy (-OCH2CH2CH2CH2-)",
        reaction: MacroClosureReaction::RcmHydrogenated,
        precedent: "Saturated Alkoxy Bridge",
        dummy_smiles: "ClOCCCCI",
        chain_len: 5,
        ideal_span_a: 5.6,
    },
    TetherTemplateSpec {
        code: "lactam_c4",
        name: "Macrolactam C4 (-CH2C(=O)NHCH2-)",
        reaction: MacroClosureReaction::Macrolactamization,
        precedent: "HATU / PyBOP Macrolactam",
        dummy_smiles: "ClCC(=O)NCCI",
        chain_len: 4,
        ideal_span_a: 4.6,
    },
    TetherTemplateSpec {
        code: "lactam_nme_c4",
        name: "N-Me Macrolactam (-CH2C(=O)N(Me)CH2-)",
        reaction: MacroClosureReaction::NMethylMacrolactam,
        precedent: "Lorlatinib (PF-06463922) CNS Bridge",
        dummy_smiles: "ClCC(=O)N(C)CCI",
        chain_len: 4,
        ideal_span_a: 4.6,
    },
    TetherTemplateSpec {
        code: "lactam_oxa_c5",
        name: "Glycolamide Macrolactam (-OCH2C(=O)NHCH2-)",
        reaction: MacroClosureReaction::Macrolactamization,
        precedent: "Repotrectinib Ether-Lactam",
        dummy_smiles: "ClOCC(=O)NCCI",
        chain_len: 5,
        ideal_span_a: 5.5,
    },
    TetherTemplateSpec {
        code: "lactam_nme_c5",
        name: "N-Me Glycolamide (-OCH2C(=O)N(Me)CH2-)",
        reaction: MacroClosureReaction::NMethylMacrolactam,
        precedent: "Zero-HBD CNS Ether-Lactam",
        dummy_smiles: "ClOCC(=O)N(C)CCI",
        chain_len: 5,
        ideal_span_a: 5.5,
    },
    TetherTemplateSpec {
        code: "diether_c5",
        name: "Dioxa-Pentylene (-OCH2CH2OCH2-)",
        reaction: MacroClosureReaction::Macroetherification,
        precedent: "Zotiraciclib / Diether Crown",
        dummy_smiles: "ClOCCOCCI",
        chain_len: 5,
        ideal_span_a: 5.4,
    },
    TetherTemplateSpec {
        code: "click_triazole_c5",
        name: "CuAAC 1,4-Triazole (-CH2-Triazole-CH2-)",
        reaction: MacroClosureReaction::ClickTriazole,
        precedent: "Intramolecular Click Macrocycle",
        dummy_smiles: "ClCc1cn(CCI)nn1",
        chain_len: 5,
        ideal_span_a: 5.8,
    },
];

/// Computes the Kier (1989) Molecular Flexibility Index `Phi = (kappa_1 * kappa_2) / A`
/// incorporating covalent radius/hybridization modifier `alpha`.
pub fn compute_kier_flexibility_phi(mol: &Molecule) -> f64 {
    let a = mol.atoms.len();
    if a <= 3 {
        return 0.5;
    }
    let p1 = mol.bonds.len() as f64;
    // Count length-2 bond paths P2 = sum_u deg(u)*(deg(u) - 1) / 2
    let p2: f64 = mol
        .adj
        .iter()
        .map(|nbrs| {
            let d = nbrs.len();
            if d >= 2 {
                ((d * (d - 1)) / 2) as f64
            } else {
                0.0
            }
        })
        .sum();

    // Kier alpha covalent-radius/hybridization correction relative to C(sp3) (r_Csp3 = 0.77 A)
    let mut alpha = 0.0f64;
    for atom in &mol.atoms {
        let r_cov = match (atom.elem, atom.hyb, atom.aromatic) {
            (Element::C, _, true) | (Element::C, Hybridization::Sp2, _) => 0.67,
            (Element::C, Hybridization::Sp, _) => 0.60,
            (Element::C, _, _) => 0.77,
            (Element::N, _, true) | (Element::N, Hybridization::Sp2, _) => 0.61,
            (Element::N, Hybridization::Sp, _) => 0.55,
            (Element::N, _, _) => 0.70,
            (Element::O, Hybridization::Sp2, _) => 0.55,
            (Element::O, _, _) => 0.66,
            (Element::F, _, _) => 0.64,
            (Element::S, _, _) => 1.04,
            (Element::P, _, _) => 1.10,
            (Element::Cl, _, _) => 0.99,
            (Element::Br, _, _) => 1.14,
            (Element::I, _, _) => 1.33,
            _ => 0.77,
        };
        alpha += (r_cov / 0.77) - 1.0;
    }

    let af = a as f64;
    let d1 = (p1 + alpha).max(0.5);
    let d2 = (p2 + alpha).max(0.5);
    let kappa1 = (af + alpha) * (af + alpha - 1.0).powi(2) / (d1 * d1);
    let kappa2 = (af + alpha - 1.0) * (af + alpha - 2.0).max(0.5).powi(2) / (d2 * d2);
    ((kappa1 * kappa2) / af).clamp(0.15, 25.0)
}

fn shortest_bond_paths(mol: &Molecule) -> Vec<Vec<usize>> {
    let n = mol.atoms.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for s in 0..n {
        dist[s][s] = 0;
        let mut q = VecDeque::new();
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            let du = dist[s][u];
            for &(nb, _) in &mol.adj[u] {
                if dist[s][nb] == usize::MAX {
                    dist[s][nb] = du + 1;
                    q.push_back(nb);
                }
            }
        }
    }
    dist
}

/// Identifies peripheral/ring-wing attachment site candidates `(anchor_atom, atoms_to_prune_if_used)`
/// on an acyclic lead (aryl CH, aryl-halide, aryl-alkyne, aryl-alkoxy core carbon, or aliphatic rim).
fn collect_attachable_handles(mol: &Molecule) -> Vec<(usize, Vec<usize>, &'static str)> {
    let mut handles: Vec<(usize, Vec<usize>, &'static str)> = Vec::new();
    let mut seen = BTreeSet::new();

    // 1. Aromatic carbons carrying exocyclic -OCH3 / -OCH2R / -Br / -Cl / -C#CH (prune substituent, attach tether directly to aryl C!)
    for (u, a) in mol.atoms.iter().enumerate() {
        if !a.aromatic || a.elem != Element::C {
            continue;
        }
        for &(exo, _) in &mol.adj[u] {
            if mol.atoms[exo].in_ring {
                continue;
            }
            // Collect acyclic tail starting at `exo` without crossing back through `u`
            let mut tail = BTreeSet::new();
            let mut q = VecDeque::new();
            tail.insert(exo);
            q.push_back(exo);
            let mut hits_other_ring = false;
            while let Some(cur) = q.pop_front() {
                if cur != exo && mol.atoms[cur].in_ring {
                    hits_other_ring = true;
                    break;
                }
                for &(nb, _) in &mol.adj[cur] {
                    if nb != u && !tail.contains(&nb) {
                        tail.insert(nb);
                        q.push_back(nb);
                    }
                }
            }
            if !hits_other_ring && tail.len() <= 6 && !mol.atoms[exo].in_ring && seen.insert(u) {
                handles.push((u, tail.into_iter().collect(), "Substituted Aryl Vector"));
                break;
            }
        }
    }

    // 2. Unsubstituted aromatic C-H (`ortho`/`meta` positions on distinct aryl/heteroaryl wings!)
    for (u, a) in mol.atoms.iter().enumerate() {
        if a.aromatic && a.elem == Element::C && a.total_h() >= 1 && seen.insert(u) {
            handles.push((u, Vec::new(), "Aryl C-H Vector"));
        }
    }

    // 3. Terminal sp3 C / N / O handles
    for (u, a) in mol.atoms.iter().enumerate() {
        if !a.in_ring
            && matches!(a.elem, Element::C | Element::N | Element::O)
            && a.total_h() >= 1
            && mol.adj[u].len() == 1
            && seen.insert(u)
        {
            handles.push((u, Vec::new(), "Terminal Chain Handle"));
        }
    }

    handles
}

/// Discovers and ranks 3D-compatible `(anchor_a, anchor_b)` terminal pairs for macrocyclization.
pub fn detect_cyclizable_terminal_pairs(
    mol: &Molecule,
    coords3d: &[[f32; 3]],
) -> Vec<CyclizableTerminalPair> {
    let n = mol.atoms.len().min(coords3d.len());
    if n < 6 {
        return Vec::new();
    }

    let topo = shortest_bond_paths(mol);
    let handles = collect_attachable_handles(mol);
    let mut pairs: Vec<CyclizableTerminalPair> = Vec::new();

    for i in 0..handles.len() {
        let (ua, ref prune_a, kind_a) = handles[i];
        if ua >= n {
            continue;
        }
        for j in (i + 1)..handles.len() {
            let (ub, ref prune_b, kind_b) = handles[j];
            if ub >= n {
                continue;
            }
            // Reject if anchors belong to the exact same elementary SSSR ring (that would only fuse a small ring, not bridge wings!)
            let same_small_ring = mol
                .rings
                .iter()
                .any(|r| r.len() <= 10 && r.contains(&ua) && r.contains(&ub));
            if same_small_ring {
                continue;
            }
            // Reject overlapping prune sets that touch the partner anchor
            if prune_a.contains(&ub) || prune_b.contains(&ua) {
                continue;
            }

            let span = topo[ua][ub];
            if !(5..=12).contains(&span) {
                continue;
            }

            let dx = coords3d[ua][0] - coords3d[ub][0];
            let dy = coords3d[ua][1] - coords3d[ub][1];
            let dz = coords3d[ua][2] - coords3d[ub][2];
            let d3 = (dx * dx + dy * dy + dz * dz).sqrt();
            if !(2.8..=9.8).contains(&d3) {
                continue;
            }

            let mut combined_prune = prune_a.clone();
            for &pb in prune_b {
                if !combined_prune.contains(&pb) {
                    combined_prune.push(pb);
                }
            }
            if combined_prune.len() * 2 >= mol.atoms.len() {
                continue;
            }

            // Ideal Lorlatinib/Pacritinib wing-to-wing 3D span ~ 4.2 - 6.2 A across 6 - 9 backbone bonds!
            let dist_opt = 1.0 - (((d3 as f64) - 5.1).abs() / 5.0).min(0.85);
            let span_opt = 1.0 - (((span as f64) - 7.5).abs() / 6.0).min(0.75);
            let sub_bonus = if !prune_a.is_empty() || !prune_b.is_empty() {
                8.0
            } else {
                0.0
            };
            let score = 60.0 * dist_opt + 28.0 * span_opt + sub_bonus;

            pairs.push(CyclizableTerminalPair {
                pair_idx: 0,
                anchor_a: ua,
                anchor_b: ub,
                prune_atoms: combined_prune,
                backbone_bond_span: span,
                spatial_dist_3d_a: d3,
                label: format!(
                    "{}#{} <-> {}#{} ({} bonds, {:.1} A)",
                    mol.atoms[ua].elem.symbol(),
                    ua + 1,
                    mol.atoms[ub].elem.symbol(),
                    ub + 1,
                    span,
                    d3
                ),
                rationale: if kind_a.starts_with("Substituted") || kind_b.starts_with("Substituted")
                {
                    "Replaces open-chain wing substituent(s) with strain-matched macrocyclic tether (Lorlatinib Crizotinib->PF-06463922 strategy)"
                } else {
                    "Cross-wing aryl/aliphatic C-H macrocyclization locking hinge-binding dihedral angles"
                },
                suitability_score: score,
            });
        }
    }

    pairs.sort_by(|a, b| {
        b.suitability_score
            .partial_cmp(&a.suitability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate near-identical neighbor pairs so user sees diverse wing geometries
    let mut diverse: Vec<CyclizableTerminalPair> = Vec::new();
    for mut p in pairs {
        let redundant = diverse.iter().any(|d| {
            (d.anchor_a == p.anchor_a && topo[d.anchor_b][p.anchor_b] <= 1)
                || (d.anchor_b == p.anchor_b && topo[d.anchor_a][p.anchor_a] <= 1)
        });
        if !redundant {
            p.pair_idx = diverse.len();
            diverse.push(p);
            if diverse.len() >= 4 {
                break;
            }
        }
    }
    diverse
}

fn parse_tether_two_dummies(smiles: &str) -> Option<(Molecule, usize, usize)> {
    let raw = parse_smiles(smiles).ok()?;
    let cl_idx = raw.atoms.iter().position(|a| a.elem == Element::Cl)?;
    let i_idx = raw.atoms.iter().position(|a| a.elem == Element::I)?;
    let left_old = raw.adj[cl_idx].first()?.0;
    let right_old = raw.adj[i_idx].first()?.0;

    let mut old_to_new = vec![None; raw.atoms.len()];
    let mut out = Molecule::new("");
    for (oi, a) in raw.atoms.iter().enumerate() {
        if oi == cl_idx || oi == i_idx {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == left_old || oi == right_old {
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
    Some((out, old_to_new[left_old]?, old_to_new[right_old]?))
}

/// Excises `pair.prune_atoms` from `parent`, grafts `tether` across `(anchor_a, anchor_b)` to close
/// the macrocycle, and returns `(macro_mol, parent_kept_old_indices, core_new_indices, tether_new_indices)`.
fn assemble_macrocycle_graph(
    parent: &Molecule,
    pair: &CyclizableTerminalPair,
    tether: &TetherTemplateSpec,
) -> Option<(Molecule, Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rem: BTreeSet<usize> = pair.prune_atoms.iter().copied().collect();
    if rem.contains(&pair.anchor_a) || rem.contains(&pair.anchor_b) {
        return None;
    }

    let mut old_to_new = vec![None; parent.atoms.len()];
    let mut parent_kept_old = Vec::new();
    let mut core_new = Vec::new();
    let mut out = Molecule::new("");

    for (oi, a) in parent.atoms.iter().enumerate() {
        if rem.contains(&oi) {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == pair.anchor_a || oi == pair.anchor_b {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[oi] = Some(ni);
        parent_kept_old.push(oi);
        core_new.push(ni);
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

    let na_idx = old_to_new[pair.anchor_a]?;
    let nb_idx = old_to_new[pair.anchor_b]?;

    let (teth_mol, t_left, t_right) = parse_tether_two_dummies(tether.dummy_smiles)?;
    let offset = out.atoms.len();
    let mut tether_new = Vec::with_capacity(teth_mol.atoms.len());
    for (ti, ta) in teth_mol.atoms.iter().enumerate() {
        let mut na = ta.clone();
        na.idx = offset + ti;
        out.atoms.push(na);
        tether_new.push(offset + ti);
    }
    for tb in &teth_mol.bonds {
        let bi = out.bonds.len();
        let mut nb = tb.clone();
        nb.idx = bi;
        nb.a = offset + tb.a;
        nb.b = offset + tb.b;
        out.bonds.push(nb);
    }

    // Ring-closing bonds: anchor_a -- t_left AND t_right -- anchor_b!
    let b1 = out.bonds.len();
    out.bonds.push(Bond {
        idx: b1,
        a: na_idx,
        b: offset + t_left,
        order: BondOrder::Single,
        aromatic: false,
        in_ring: true,
        kekule_order: 1,
    });
    let b2 = out.bonds.len();
    out.bonds.push(Bond {
        idx: b2,
        a: offset + t_right,
        b: nb_idx,
        order: BondOrder::Single,
        aromatic: false,
        in_ring: true,
        kekule_order: 1,
    });

    out.sanitize();
    out.smiles = molecule_to_smiles(&out);
    compute_2d_coords(&mut out);

    Some((out, parent_kept_old, core_new, tether_new))
}

/// Enumerates, 3D-minimizes, Kabsch-superimposes, and thermodynamically scores `RCM`, macrolactam,
/// `N`-Me lactam, diether, and `CuAAC` click macrocycles for `parent`.
pub fn design_macrocycles(parent: &Molecule, pair_choice: usize) -> MacrocycleDesignReport {
    let parent_3d = evaluate_3d_summary_profile(parent, false);
    let parent_phi = compute_kier_flexibility_phi(parent);
    let parent_alerts = scan_structural_alerts(parent);
    let parent_prof = evaluate_physchem(parent, parent_alerts.len());

    let terminal_pairs = detect_cyclizable_terminal_pairs(parent, &parent_3d.coords3d);
    let active_pair_idx = pair_choice.min(terminal_pairs.len().saturating_sub(1));

    let mut candidates: Vec<MacrocycleCandidate> = Vec::new();
    let Some(pair) = terminal_pairs.get(active_pair_idx) else {
        return MacrocycleDesignReport {
            parent_name: parent.name.clone(),
            parent_pact: parent.pact,
            parent_kier_phi: parent_phi,
            parent_coords3d: parent_3d.coords3d,
            terminal_pairs,
            active_pair_idx: 0,
            candidates,
        };
    };

    let ens_cfg = ConformerGenConfig {
        num_trial_conformers: 4,
        max_kept_conformers: 3,
        mmff_max_iters: 85,
        rmsd_prune_cutoff: 0.40,
        energy_window_kcal: 12.0,
        random_seed: 0x6A09_E667,
    };

    for teth in &MACRO_TETHER_LIBRARY {
        let Some((mut m_mac, parent_kept_old, core_new, tether_new)) =
            assemble_macrocycle_graph(parent, pair, teth)
        else {
            continue;
        };

        let cand_id = format!("{}-MC-{}", parent.id, teth.code);
        m_mac.id = cand_id.clone();
        m_mac.name = format!("{} [{}]", parent.name, teth.name);
        m_mac.target = parent.target.clone();
        m_mac.metric = parent.metric.clone();

        // Detect largest newly formed macrocyclic SSSR ring (`>= 10` atoms)
        let macro_ring_atoms: Vec<usize> = m_mac
            .rings
            .iter()
            .filter(|r| r.len() >= 10 && r.iter().any(|u| tether_new.contains(u)))
            .max_by_key(|r| r.len())
            .cloned()
            .unwrap_or_else(|| {
                let mut v = tether_new.clone();
                v.push(core_new[0]);
                v
            });
        let macro_ring_size = macro_ring_atoms
            .len()
            .max(pair.backbone_bond_span + teth.chain_len + 1);

        // Generate 3D conformer ensemble & align conserved pharmacophore core onto acyclic parent 3D pose!
        let ens = generate_3d_conformer_ensemble(&m_mac, &ens_cfg);
        let Some(c0) = ens.conformers.first() else {
            continue;
        };

        let ref_sub: Vec<[f32; 3]> = core_new
            .iter()
            .map(|&ni| c0.coords3d[ni])
            .collect();
        let mob_parent_sub: Vec<[f32; 3]> = parent_kept_old
            .iter()
            .map(|&oi| parent_3d.coords3d[oi])
            .collect();

        // Align parent's full coordinates onto macrocycle's core frame so both overlay directly in 3D!
        let (rot, trans, _, core_rmsd) = kabsch_align_3d(&mob_parent_sub, &ref_sub);
        let aligned_parent_coords3d: Vec<[f32; 3]> = parent_3d
            .coords3d
            .iter()
            .map(|p| {
                let rx = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + trans[0];
                let ry = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + trans[1];
                let rz = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + trans[2];
                [rx, ry, rz]
            })
            .collect();

        let macro_phi = compute_kier_flexibility_phi(&m_mac);
        let alerts = scan_structural_alerts(&m_mac);
        let prof = evaluate_physchem(&m_mac, alerts.len());
        let pka = evaluate_physiological_pka_74(&m_mac);
        let sa = evaluate_synthetic_accessibility(&m_mac);

        // Thermodynamic 3D Preorganization Entropy (-T * dS_conf) vs Core-Distortion Strain:
        let rot_diff = (parent_prof.rotatable_bonds as f64 - prof.rotatable_bonds as f64).max(0.0);
        let phi_diff = (parent_phi - macro_phi).max(0.0);
        let eff_frozen = (rot_diff + 0.65 * phi_diff + 1.8).clamp(0.8, 6.5);
        let entropy_ddg = -0.52 * eff_frozen; // Favorable negative kcal/mol (-0.4 .. -3.3 kcal/mol)

        // Core RMSD + span mismatch strain penalty
        let span_err = ((pair.spatial_dist_3d_a as f64) - teth.ideal_span_a).abs();
        let core_strain = 0.85 * (core_rmsd as f64).powi(2) + 0.22 * span_err * span_err;
        let net_ddg = (entropy_ddg + core_strain).clamp(-2.45, 2.20);
        let fold_boost = 10.0f64.powf(-net_ddg / 1.364);
        let pred_pact = (parent.pact - net_ddg / 1.364).clamp(4.5, 11.8);
        m_mac.pact = pred_pact;
        m_mac.raw_nm = 10.0f64.powf(9.0 - pred_pact);

        let druggability = if prof.cns_mpo >= 3.8 && prof.hbd <= 2 && prof.tpsa <= 95.0 && prof.mw <= 520.0 {
            MacroDruggabilityClass::CnsPenetrantMacrocycle
        } else if prof.mw <= 560.0 && prof.hbd <= 4 && prof.tpsa <= 130.0 {
            MacroDruggabilityClass::OralRo5Macrocycle
        } else if prof.mw <= 950.0 && prof.hbd <= 6 && prof.tpsa <= 210.0 {
            MacroDruggabilityClass::OralBro5Macrocycle
        } else {
            MacroDruggabilityClass::ParenteralMacrocycle
        };

        // Composite Macrocycle Score [0..100]
        let preorg_pts = ((-net_ddg + 0.6) * 18.5).clamp(0.0, 44.0);
        let rmsd_pts = ((1.35 - core_rmsd as f64) * 18.0).clamp(0.0, 22.0);
        let drug_pts = prof.qed_mean * 18.0 + (prof.cns_mpo / 6.0) * 8.0;
        let synth_pts = ((7.0 - sa.sa_score) * 2.2).clamp(0.0, 8.0);
        let macro_score = (preorg_pts + rmsd_pts + drug_pts + synth_pts).clamp(2.0, 99.5);

        let preorg = PreorganizationMetrics {
            macro_ring_size,
            parent_kier_phi: parent_phi,
            macro_kier_phi: macro_phi,
            delta_kier_phi: macro_phi - parent_phi,
            effective_rotors_frozen: eff_frozen,
            entropy_ddg_kcal: entropy_ddg,
            core_strain_penalty_kcal: core_strain,
            net_ddg_bind_kcal: net_ddg,
            potency_fold_boost: fold_boost,
            predicted_macro_pact: pred_pact,
            core_rmsd_3d_a: core_rmsd,
            active_conformer_boltzmann_pct: c0.boltzmann_weight * 100.0,
        };

        candidates.push(MacrocycleCandidate {
            rank: 0,
            candidate_id: cand_id,
            tether_code: teth.code,
            tether_name: teth.name,
            reaction: teth.reaction,
            clinical_precedent: teth.precedent,
            tether_chain_atoms: teth.chain_len,
            pair_label: pair.label.clone(),
            preorg,
            druggability,
            physchem: prof,
            logd_74: pka.microstate_logd_74,
            sa_score: sa.sa_score,
            macrocycle_score: macro_score,
            canonical_smiles: m_mac.smiles.clone(),
            core_atoms: core_new,
            tether_atoms: tether_new,
            macro_ring_atoms,
            coords3d: c0.coords3d.clone(),
            aligned_parent_coords3d,
            gasteiger_charges: ens.gasteiger_charges,
            mmff_energy_kcal: c0.energy.total_kcal,
            mol: m_mac,
        });
    }

    candidates.sort_by(|a, b| {
        b.macrocycle_score
            .partial_cmp(&a.macrocycle_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.rank = i + 1;
    }

    MacrocycleDesignReport {
        parent_name: parent.name.clone(),
        parent_pact: parent.pact,
        parent_kier_phi: parent_phi,
        parent_coords3d: parent_3d.coords3d,
        terminal_pairs,
        active_pair_idx,
        candidates,
    }
}

/// Formats a 3D MDL V2000 `.sdf` bundle of the ranked macrocyclic conformers with thermodynamic tags.
pub fn format_macrocycles_3d_sdf(cands: &[MacrocycleCandidate]) -> String {
    let mut out = String::new();
    for c in cands {
        let tags = [
            ("MACRO_RING_SIZE", format!("{}", c.preorg.macro_ring_size)),
            ("CLOSURE_REACTION", c.reaction.label().to_string()),
            ("KIER_PHI", format!("{:.3}", c.preorg.macro_kier_phi)),
            ("NET_DDG_BIND_KCAL", format!("{:+.3}", c.preorg.net_ddg_bind_kcal)),
            ("POTENCY_FOLD_BOOST", format!("{:.1}x", c.preorg.potency_fold_boost)),
            ("CORE_RMSD_3D_A", format!("{:.3}", c.preorg.core_rmsd_3d_a)),
        ];
        out.push_str(&format_3d_conformer_sdf(
            &c.mol,
            &c.coords3d,
            &c.gasteiger_charges,
            c.mmff_energy_kcal,
            &tags,
        ));
    }
    out
}

/// Exports enumerated RCM & lactam macrocycles with 3D preorganization entropy metrics to CSV.
pub fn export_macrocycles_csv(cands: &[MacrocycleCandidate]) -> String {
    let mut out = String::from(
        "rank,id,tether_code,tether_name,reaction,precedent,ring_size,kier_phi_parent,kier_phi_macro,rotors_frozen,entropy_ddg_kcal,strain_kcal,net_ddg_kcal,fold_boost,predicted_pact,core_rmsd_3d_a,mw,logd_74,tpsa,hbd,cns_mpo,druggability,sa_score,macrocycle_score,smiles\n",
    );
    for c in cands {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{},{:.2},{:.2},{:.1},{:+.2},{:+.2},{:+.2},{:.1},{:.2},{:.3},{:.1},{:.2},{:.1},{},{:.2},\"{}\",{:.2},{:.1},\"{}\"\n",
            c.rank,
            c.candidate_id,
            c.tether_code,
            c.tether_name,
            c.reaction.label(),
            c.clinical_precedent,
            c.preorg.macro_ring_size,
            c.preorg.parent_kier_phi,
            c.preorg.macro_kier_phi,
            c.preorg.effective_rotors_frozen,
            c.preorg.entropy_ddg_kcal,
            c.preorg.core_strain_penalty_kcal,
            c.preorg.net_ddg_bind_kcal,
            c.preorg.potency_fold_boost,
            c.preorg.predicted_macro_pact,
            c.preorg.core_rmsd_3d_a,
            c.physchem.mw,
            c.logd_74,
            c.physchem.tpsa,
            c.physchem.hbd,
            c.physchem.cns_mpo,
            c.druggability.short_label(),
            c.sa_score,
            c.macrocycle_score,
            c.canonical_smiles
        ));
    }
    out
}
