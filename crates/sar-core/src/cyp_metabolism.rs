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

//! `SMARTCyp` Regioselective Cytochrome P450 (`CYP3A4`, `CYP2D6`, `CYP2C9`) Site-of-Metabolism
//! (`SoM`) Predictor, Hepatic Microsomal Intrinsic Clearance (`CL_int,u`) Estimator, Medicinal-Chemistry
//! Soft-Spot Blocking Advisor, and `BioTransformer`-Style Phase I / Phase II Metabolite Enumerator.
//!
//! Primary References:
//! - Rydberg, Gloriam, Zaretzki, Breneman & Olsen (2010), *ACS Med. Chem. Lett.* 1(3), 96-100
//!   ("SMARTCyp: A 2D Method for Prediction of Cytochrome P450-Mediated Drug Metabolism":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4055970/`, `https://pubmed.ncbi.nlm.nih.gov/24936230/`)
//! - Rydberg, Gloriam & Olsen (2010), *Bioinformatics* 26(23), 2988-2989 ("The SMARTCyp cytochrome
//!   P450 metabolism prediction server": `https://pubmed.ncbi.nlm.nih.gov/20947523/`)
//! - CYP2D6 Basic-Amine Pharmacophore Span Regioselectivity (`https://pmc.ncbi.nlm.nih.gov/articles/PMC4025842/`)
//! - Djoumbou-Feunang et al. (2019 & 2022), *J. Cheminform.* 11:2 & *Nucleic Acids Res.* 50(W1),
//!   W115-W122 ("BioTransformer / BioTransformer 3.0: small molecule Phase I & Phase II metabolism
//!   prediction": `https://pmc.ncbi.nlm.nih.gov/articles/PMC6689873/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC9252798/`)
//! - GLORY / Biotransformation & Reactive Metabolite Bioactivation (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6582643/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC8826528/`)

use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::pka_protomer::{evaluate_physiological_pka_74, perceive_ionizable_sites, IonSiteKind};
use crate::sketcher::molecule_to_smiles;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Major human hepatic Cytochrome P450 isoforms modeled by `SMARTCyp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CypIsoform {
    Cyp3a4,
    Cyp2d6,
    Cyp2c9,
}

impl CypIsoform {
    pub const ALL: [Self; 3] = [Self::Cyp3a4, Self::Cyp2d6, Self::Cyp2c9];

    pub fn label(self) -> &'static str {
        match self {
            Self::Cyp3a4 => "CYP3A4",
            Self::Cyp2d6 => "CYP2D6",
            Self::Cyp2c9 => "CYP2C9",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Cyp3a4 => [251, 191, 36],  // Amber
            Self::Cyp2d6 => [56, 189, 248],  // Sky Blue (cationic N+ anchor)
            Self::Cyp2c9 => [192, 132, 252], // Violet (anionic acid/sulfonamide anchor)
        }
    }
}

/// Hepatic microsomal metabolic stability tier based on predicted unbound `CL_int,u` (`mL/min/kg`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolicStabilityClass {
    HighStability,
    ModerateTurnover,
    HighClearanceLabile,
}

impl MetabolicStabilityClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::HighStability => "High Stability (<15 mL/min/kg)",
            Self::ModerateTurnover => "Moderate Turnover (15-45)",
            Self::HighClearanceLabile => "Labile / High CL (>45 mL/min/kg)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HighStability => "Stable",
            Self::ModerateTurnover => "Moderate",
            Self::HighClearanceLabile => "Labile",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HighStability => [52, 211, 153],       // Emerald
            Self::ModerateTurnover => [251, 191, 36],    // Amber
            Self::HighClearanceLabile => [248, 113, 113], // Coral Red
        }
    }
}

/// Biotransformation phase classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolismPhase {
    PhaseICyp,
    PhaseIIConjugate,
    SequentialPhaseIToII,
}

impl MetabolismPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::PhaseICyp => "Phase I (CYP / Oxidative)",
            Self::PhaseIIConjugate => "Phase II (Direct Conjugate)",
            Self::SequentialPhaseIToII => "Sequential Phase I -> II",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::PhaseICyp => "Phase I",
            Self::PhaseIIConjugate => "Phase II",
            Self::SequentialPhaseIToII => "Phase I->II",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::PhaseICyp => [251, 191, 36],           // Amber
            Self::PhaseIIConjugate => [56, 189, 248],    // Sky Blue
            Self::SequentialPhaseIToII => [192, 132, 252], // Violet
        }
    }
}

/// Mechanistic Phase Ireaction class assigned to a `SMARTCyp` site of metabolism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SomReactionKind {
    ODealkylation,
    NDealkylation,
    SOxidation,
    NOxidation,
    BenzylicHydroxylation,
    AllylicPropargylicOxidation,
    HeterocyclicRingOxidation,
    AromaticHydroxylation,
    AliphaticHydroxylation,
    Epoxidation,
}

impl SomReactionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::ODealkylation => "O-Dealkylation",
            Self::NDealkylation => "N-Dealkylation",
            Self::SOxidation => "S-Oxidation (Sulfoxide)",
            Self::NOxidation => "N-Oxidation (Amine N-Oxide)",
            Self::BenzylicHydroxylation => "Benzylic C-H Hydroxylation",
            Self::AllylicPropargylicOxidation => "Allylic / Alkyne Oxidation",
            Self::HeterocyclicRingOxidation => "Alpha/Beta-Heterocycle Oxidation",
            Self::AromaticHydroxylation => "Aromatic C-H Hydroxylation",
            Self::AliphaticHydroxylation => "Aliphatic sp3 Hydroxylation",
            Self::Epoxidation => "Alkene Epoxidation",
        }
    }
}

/// Atom-level `SMARTCyp` regioselectivity prediction across `CYP3A4`, `CYP2D6`, and `CYP2C9`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmartCypSite {
    pub rank_3a4: usize,
    pub rank_2d6: usize,
    pub rank_2c9: usize,
    pub atom_idx: usize,
    pub atom_label: String,
    pub reaction_kind: SomReactionKind,
    /// DFT transition-state H-abstraction / oxidation activation energy `E_a` (`kJ/mol`).
    pub dft_ea_kj_mol: f64,
    /// Rydberg topological accessibility `A_i` in `[0.50 .. 1.00]`.
    pub accessibility_a: f64,
    /// Bond-path distance to strongest protonated basic amine `N+` (`CYP2D6` Asp301 anchor).
    pub span_to_basic_n: Option<usize>,
    /// Bond-path distance to strongest anionic / sulfonamide anchor (`CYP2C9` Arg108 anchor).
    pub span_to_acidic_anchor: Option<usize>,
    /// Lower `SMARTCyp` score (`kJ/mol`) = higher regioselective lability.
    pub score_3a4: f64,
    pub score_2d6: f64,
    pub score_2c9: f64,
    pub dominant_isoform: CypIsoform,
    /// Concrete medicinal-chemistry structural modification to block this soft spot.
    #[serde(skip_deserializing)]
    pub blocking_strategy: &'static str,
}

/// Enumerated Phase I, Phase II, or sequential `Phase I -> II` metabolite with exact 2D structure,
/// LC-HRMS mass shift (`Delta Da`), `logD_7.4` clearance shift, and reactive bioactivation alert.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaboliteSpecies {
    pub metabolite_id: String,
    pub rank: usize,
    pub phase: MetabolismPhase,
    #[serde(skip_deserializing)]
    pub enzyme: &'static str,
    pub reaction_name: String,
    pub parent_som_atom: Option<usize>,
    pub mass_shift_da: f64,
    pub exact_mw: f64,
    pub logd_74: f64,
    pub delta_logd_74: f64,
    pub predicted_abundance_pct: f64,
    /// `true` for reactive epoxides, *ortho*/*para*-quinone-imine precursors, ketene/furan intermediates, or acyl glucuronides.
    pub is_reactive_bioactivation: bool,
    #[serde(skip_deserializing)]
    pub bioactivation_note: &'static str,
    pub canonical_smiles: String,
    pub modified_atoms: Vec<usize>,
    pub mol: Molecule,
}

/// Dataset-level fast `SMARTCyp` hepatic clearance & bioactivation summary stored on `DatasetAnalysis`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolicDatasetSummary {
    pub top_som_atom: Option<usize>,
    pub top_som_label: String,
    #[serde(skip_deserializing)]
    pub top_reaction: &'static str,
    pub primary_isoform: CypIsoform,
    pub best_smartcyp_score: f64,
    pub cl_int_ml_min_kg: f64,
    pub stability_class: MetabolicStabilityClass,
    pub phase2_direct_handles: usize,
    pub reactive_bioactivation_risk: bool,
}

/// Full `SMARTCyp` + `BioTransformer` Phase I/II metabolic tree report for a target molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CypMetabolismReport {
    pub parent_mw: f64,
    pub parent_logd_74: f64,
    pub cl_int_ml_min_kg: f64,
    pub predicted_t_half_min: f64,
    pub stability_class: MetabolicStabilityClass,
    pub isoform_fraction_fm: [f64; 3], // [fm_3A4, fm_2D6, fm_2C9] summing to 1.0
    pub sites: Vec<SmartCypSite>,
    pub metabolites: Vec<MetaboliteSpecies>,
}

/// Computes all-pairs shortest topological bond distances for `mol`.
fn all_pairs_bond_distances(mol: &Molecule) -> Vec<Vec<usize>> {
    let n = mol.atoms.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for s in 0..n {
        dist[s][s] = 0;
        let mut q = VecDeque::new();
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            let d = dist[s][u];
            for &(nb, _) in &mol.adj[u] {
                if dist[s][nb] == usize::MAX {
                    dist[s][nb] = d + 1;
                    q.push_back(nb);
                }
            }
        }
    }
    dist
}

/// Evaluates intrinsic `SMARTCyp` DFT barrier `E_a (kJ/mol)`, reaction class, and medicinal-chemistry
/// blocking tactic for heavy atom `u` (returns `None` if `u` is non-oxidizable, e.g. `F`, `Cl`, `C=O`, `SO2`).
fn classify_atom_smartcyp_ea(
    mol: &Molecule,
    u: usize,
) -> Option<(f64, SomReactionKind, &'static str)> {
    let a = &mol.atoms[u];
    match a.elem {
        Element::S => {
            // Thioether (-S-, deg 2, no =O) vs sulfoxide (-S(=O)-) vs sulfone/sulfonamide (inert)
            let ox_double = mol.adj[u]
                .iter()
                .filter(|&&(nb, bi)| {
                    mol.atoms[nb].elem == Element::O && mol.bonds[bi].order == BondOrder::Double
                })
                .count();
            if ox_double == 0 && !a.aromatic && mol.adj[u].len() == 2 {
                return Some((
                    53.8,
                    SomReactionKind::SOxidation,
                    "Pre-oxidize thioether to sulfone (-SO2-) or swap -S- -> -O- / -CF2-",
                ));
            } else if ox_double == 1 && mol.adj[u].len() == 3 {
                return Some((
                    67.5,
                    SomReactionKind::SOxidation,
                    "Convert sulfoxide to metabolically inert sulfone (-SO2-) or sulfonimidamide",
                ));
            }
            None
        }
        Element::N => {
            // Tertiary aliphatic amine (sp3, 0 H, non-amide) -> N-oxide
            if !a.aromatic && a.hyb == Hybridization::Sp3 && a.total_h() == 0 && mol.adj[u].len() == 3
            {
                let adj_carbonyl_or_sulfonyl = mol.adj[u].iter().any(|&(nb, _)| {
                    mol.atoms[nb].hyb == Hybridization::Sp2 || mol.atoms[nb].elem == Element::S
                });
                if !adj_carbonyl_or_sulfonyl {
                    return Some((
                        63.8,
                        SomReactionKind::NOxidation,
                        "Attenuate amine basicity with beta-fluorination, morpholine, or bridged bicyclic amine",
                    ));
                }
            }
            None
        }
        Element::C => {
            let th = a.total_h();
            // Aromatic C-H oxidation
            if a.aromatic {
                if th == 0 {
                    return None;
                }
                // Count electron-donating (-OH, -OR, -NR2) vs electron-withdrawing (-F, -Cl, -CF3, -CN, -SO2-, ring =N-) effects
                let mut electronic_delta = 0.0f64;
                if let Some(ring) = mol.rings.iter().find(|r| r.contains(&u) && r.len() <= 6) {
                    let ring_n = ring
                        .iter()
                        .filter(|&&ra| mol.atoms[ra].elem == Element::N && mol.atoms[ra].total_h() == 0)
                        .count();
                    electronic_delta += (ring_n as f64) * 6.5; // Pyridine/pyrimidine/quinazoline ring deactivation!

                    for &ra in ring {
                        if ra == u {
                            continue;
                        }
                        for &(exo, _) in &mol.adj[ra] {
                            if ring.contains(&exo) {
                                continue;
                            }
                            match mol.atoms[exo].elem {
                                Element::O | Element::N if mol.atoms[exo].charge <= 0 => {
                                    electronic_delta -= 4.2; // Ortho/para EDG activation
                                }
                                Element::F | Element::Cl | Element::Br => {
                                    electronic_delta += 2.2;
                                }
                                Element::S => {
                                    electronic_delta += 5.0;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                let ea = (79.5 + electronic_delta).clamp(69.5, 94.0);
                return Some((
                    ea,
                    SomReactionKind::AromaticHydroxylation,
                    "Introduce para/ortho -F, -CN, or ring aza-nitrogen (C-H -> =N-) to raise E_a",
                ));
            }

            // Non-aromatic sp2/sp carbon
            if a.hyb == Hybridization::Sp2 {
                // Exclude carbonyl/imine/carboxyl C=O / C=N
                let has_het_double = mol.adj[u].iter().any(|&(nb, bi)| {
                    mol.atoms[nb].elem != Element::C && mol.bonds[bi].order == BondOrder::Double
                });
                if has_het_double {
                    return None;
                }
                if th > 0 {
                    return Some((
                        70.2,
                        SomReactionKind::Epoxidation,
                        "Replace labile alkene with cyclopropane bioisostere or fluoroalkene (-CF=CH-)",
                    ));
                }
                return None;
            }
            if a.hyb == Hybridization::Sp {
                if th > 0 {
                    // Terminal acetylene C#CH (e.g. Erlotinib!) -> ketene/oxirene CYP bioactivation
                    return Some((
                        71.5,
                        SomReactionKind::AllylicPropargylicOxidation,
                        "Cap terminal -C#CH as internal -C#C-CD3 or replace with -CN / -Cl to prevent ketene MBI",
                    ));
                }
                return None;
            }

            // Saturated sp3 C-H must carry at least 1 H to undergo H-atom abstraction!
            if th == 0 {
                return None;
            }

            // Inspect alpha neighbors of sp3 C-H
            let mut adj_amine_n = false;
            let mut adj_amide_n = false;
            let mut adj_ether_o = false;
            let mut adj_aryl = false;
            let mut adj_alkene_alkyne = false;

            for &(nb, _) in &mol.adj[u] {
                let anb = &mol.atoms[nb];
                if anb.elem == Element::N {
                    let is_amide = mol.adj[nb].iter().any(|&(x, bi)| {
                        x != u
                            && ((mol.atoms[x].hyb == Hybridization::Sp2
                                && mol.bonds[bi].order == BondOrder::Single)
                                || mol.atoms[x].elem == Element::S)
                    });
                    if is_amide || anb.aromatic {
                        adj_amide_n = true;
                    } else {
                        adj_amine_n = true;
                    }
                } else if anb.elem == Element::O && mol.adj[nb].len() == 2 {
                    let is_ester = mol.adj[nb].iter().any(|&(x, _)| {
                        x != u && mol.atoms[x].hyb == Hybridization::Sp2 && !mol.atoms[x].aromatic
                    });
                    if !is_ester {
                        adj_ether_o = true;
                    }
                } else if anb.aromatic {
                    adj_aryl = true;
                } else if anb.elem == Element::C
                    && matches!(anb.hyb, Hybridization::Sp2 | Hybridization::Sp)
                {
                    let is_carbonyl = mol.adj[nb].iter().any(|&(x, bi)| {
                        mol.atoms[x].elem == Element::O && mol.bonds[bi].order == BondOrder::Double
                    });
                    if !is_carbonyl {
                        adj_alkene_alkyne = true;
                    }
                }
            }

            if adj_amine_n {
                let ea = if a.in_ring {
                    61.2 // Alpha-CH2 of morpholine/piperazine/piperidine
                } else if th == 3 {
                    60.4 // N-CH3
                } else {
                    59.2 // N-CH2-R
                };
                let kind = if a.in_ring {
                    SomReactionKind::HeterocyclicRingOxidation
                } else {
                    SomReactionKind::NDealkylation
                };
                return Some((
                    ea,
                    kind,
                    "Deuterate N-alkyl (-NCD3), cyclize to spiro-azetidine/oxetane, or add beta-F to raise C-H E_a",
                ));
            }

            if adj_ether_o {
                let ea = if a.in_ring {
                    67.2 // Morpholine / THF alpha-O CH2
                } else if th == 3 {
                    65.6 // Ar-OCH3 / R-OCH3
                } else {
                    66.4 // Ar-OCH2-R
                };
                let kind = if a.in_ring {
                    SomReactionKind::HeterocyclicRingOxidation
                } else {
                    SomReactionKind::ODealkylation
                };
                return Some((
                    ea,
                    kind,
                    "Swap labile -OCH3 for -OCF3, -OCHF2, or -OCD3 to block CYP O-dealkylation",
                ));
            }

            if adj_amide_n {
                return Some((
                    68.8,
                    SomReactionKind::NDealkylation,
                    "Replace N-alkyl amide substituent with cyclopropyl or trifluoroethyl (-CH2CF3)",
                ));
            }

            if adj_alkene_alkyne {
                return Some((
                    64.2,
                    SomReactionKind::AllylicPropargylicOxidation,
                    "Gem-difluorinate (-CF2-) or sterically shield the allylic/propargylic CH2 position",
                ));
            }

            if adj_aryl {
                let ea = match th {
                    3 => 68.4, // Ar-CH3 (e.g. Celecoxib 4-methylphenyl!)
                    2 => 66.2, // Ar-CH2-R
                    _ => 65.5, // Ar-CHR2
                };
                return Some((
                    ea,
                    SomReactionKind::BenzylicHydroxylation,
                    "Replace benzylic -CH3/-CH2- with -CF3, gem-difluoro (-CF2-), cyclopropyl, or -Cl",
                ));
            }

            // Unactivated aliphatic C-H
            let ea = match th {
                3 => 82.6, // Primary -CH3
                2 => 75.8, // Secondary -CH2-
                _ => 72.5, // Tertiary -CH<
            };
            Some((
                ea,
                SomReactionKind::AliphaticHydroxylation,
                "Fluorinate omega/omega-1 alkyl chain (-CF3 / -CF2-) or truncate lipophilic tail",
            ))
        }
        _ => None,
    }
}

/// Predicts atom-resolved `SMARTCyp` regioselectivity scores across `CYP3A4`, `CYP2D6`, and `CYP2C9`.
pub fn predict_smartcyp_sites(mol: &Molecule) -> Vec<SmartCypSite> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }

    let dist = all_pairs_bond_distances(mol);

    // Rydberg Accessibility A_i = max_d(i) / max_j(max_d(j))
    let max_d_per_atom: Vec<usize> = (0..n)
        .map(|i| {
            dist[i]
                .iter()
                .copied()
                .filter(|&d| d != usize::MAX)
                .max()
                .unwrap_or(1)
                .max(1)
        })
        .collect();
    let global_diam = (*max_d_per_atom.iter().max().unwrap_or(&1)).max(1) as f64;

    // Identify CYP2D6 protonated basic nitrogen anchors & CYP2C9 anionic/sulfonamide anchors
    let ion_sites = perceive_ionizable_sites(mol);
    let mut basic_anchors: Vec<usize> = ion_sites
        .iter()
        .filter(|s| s.kind == IonSiteKind::Basic && s.micro_pka >= 6.4)
        .map(|s| s.atom_idx)
        .collect();
    if basic_anchors.is_empty() {
        // Check uncharged tertiary/secondary aliphatic amines with pKa near threshold
        for (u, a) in mol.atoms.iter().enumerate() {
            if a.elem == Element::N && !a.aromatic && a.hyb == Hybridization::Sp3 {
                basic_anchors.push(u);
            }
        }
    }

    let mut acidic_anchors: Vec<usize> = ion_sites
        .iter()
        .filter(|s| s.kind == IonSiteKind::Acidic && s.micro_pka <= 10.2)
        .map(|s| s.atom_idx)
        .collect();
    // Primary sulfonamides (-SO2NH2, e.g. Celecoxib & Valdecoxib) coordinate Arg108 in CYP2C9!
    for (u, a) in mol.atoms.iter().enumerate() {
        if a.elem == Element::N
            && a.total_h() >= 1
            && mol.adj[u].iter().any(|&(nb, _)| mol.atoms[nb].elem == Element::S)
            && !acidic_anchors.contains(&u)
        {
            acidic_anchors.push(u);
        }
    }

    let mut sites: Vec<SmartCypSite> = Vec::new();
    for u in 0..n {
        let Some((ea, rxn_kind, block_tip)) = classify_atom_smartcyp_ea(mol, u) else {
            continue;
        };

        let acc_a = ((max_d_per_atom[u] as f64) / global_diam).clamp(0.45, 1.0);
        let steric_open = (4usize.saturating_sub(mol.adj[u].len())) as f64 * 0.55;

        // 1. CYP3A4 SMARTCyp Score (kJ/mol): lower = more reactive + sterically exposed
        let score_3a4 = ea - 8.0 * acc_a - steric_open;

        // 2. CYP2D6 SMARTCyp Score: Asp301 salt-bridge holds basic N+ ~5.8 bonds from heme Fe=O
        let span_b = basic_anchors
            .iter()
            .filter_map(|&bn| {
                let d = dist[u][bn];
                if d == usize::MAX {
                    None
                } else {
                    Some(d)
                }
            })
            .min();
        let score_2d6 = if let Some(d_n) = span_b {
            ea - 8.0 * acc_a + 6.7 * ((d_n as f64) - 5.8).abs()
        } else {
            score_3a4 + 14.0
        };

        // 3. CYP2C9 SMARTCyp Score: Arg108 polar/anionic pocket holds acid/sulfonamide ~6.8 bonds from heme Fe=O
        let span_ac = acidic_anchors
            .iter()
            .filter_map(|&an| {
                let d = dist[u][an];
                if d == usize::MAX {
                    None
                } else {
                    Some(d)
                }
            })
            .min();
        let score_2c9 = if let Some(d_a) = span_ac {
            ea - 8.0 * acc_a + 5.8 * ((d_a as f64) - 6.8).abs()
        } else {
            score_3a4 + 12.0
        };

        let dominant_isoform = if score_2d6 + 1.5 < score_3a4 && score_2d6 <= score_2c9 {
            CypIsoform::Cyp2d6
        } else if score_2c9 + 1.5 < score_3a4 && score_2c9 < score_2d6 {
            CypIsoform::Cyp2c9
        } else {
            CypIsoform::Cyp3a4
        };

        sites.push(SmartCypSite {
            rank_3a4: 0,
            rank_2d6: 0,
            rank_2c9: 0,
            atom_idx: u,
            atom_label: format!("{}#{}", mol.atoms[u].elem.symbol(), u + 1),
            reaction_kind: rxn_kind,
            dft_ea_kj_mol: ea,
            accessibility_a: acc_a,
            span_to_basic_n: span_b,
            span_to_acidic_anchor: span_ac,
            score_3a4,
            score_2d6,
            score_2c9,
            dominant_isoform,
            blocking_strategy: block_tip,
        });
    }

    // Assign isoform-specific ranks
    let mut idx_2d6: Vec<usize> = (0..sites.len()).collect();
    idx_2d6.sort_by(|&a, &b| {
        sites[a]
            .score_2d6
            .partial_cmp(&sites[b].score_2d6)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rk, &si) in idx_2d6.iter().enumerate() {
        sites[si].rank_2d6 = rk + 1;
    }

    let mut idx_2c9: Vec<usize> = (0..sites.len()).collect();
    idx_2c9.sort_by(|&a, &b| {
        sites[a]
            .score_2c9
            .partial_cmp(&sites[b].score_2c9)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rk, &si) in idx_2c9.iter().enumerate() {
        sites[si].rank_2c9 = rk + 1;
    }

    // Sort primary vector by best (lowest) across active isoforms, breaking ties with CYP3A4
    sites.sort_by(|a, b| {
        let ma = a.score_3a4.min(a.score_2d6).min(a.score_2c9);
        let mb = b.score_3a4.min(b.score_2d6).min(b.score_2c9);
        ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute rank_3a4 cleanly
    let mut idx_3a4: Vec<usize> = (0..sites.len()).collect();
    idx_3a4.sort_by(|&a, &b| {
        sites[a]
            .score_3a4
            .partial_cmp(&sites[b].score_3a4)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rk, &si) in idx_3a4.iter().enumerate() {
        sites[si].rank_3a4 = rk + 1;
    }

    sites
}

fn push_atom_simple(mol: &mut Molecule, elem: Element, charge: i8) -> usize {
    let idx = mol.atoms.len();
    mol.atoms.push(Atom {
        idx,
        elem,
        charge,
        aromatic: false,
        bracketed: charge != 0,
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

fn push_bond_simple(mol: &mut Molecule, a: usize, b: usize, order: BondOrder) {
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

/// Excises an acyclic alkyl group attached to heteroatom `het_keep` via `c_cleave` (for `O`- or `N`-dealkylation).
fn cleave_acyclic_hetero_alkyl(mol: &Molecule, c_cleave: usize) -> Option<(Molecule, usize)> {
    if mol.atoms[c_cleave].in_ring {
        return None;
    }
    let het_keep = mol.adj[c_cleave]
        .iter()
        .map(|&(nb, _)| nb)
        .find(|&nb| matches!(mol.atoms[nb].elem, Element::O | Element::N))?;

    // Collect connected component on the `c_cleave` side without crossing `het_keep`
    let mut rem = BTreeSet::new();
    let mut q = VecDeque::new();
    rem.insert(c_cleave);
    q.push_back(c_cleave);
    while let Some(cur) = q.pop_front() {
        for &(nb, _) in &mol.adj[cur] {
            if nb != het_keep && !rem.contains(&nb) {
                rem.insert(nb);
                q.push_back(nb);
            }
        }
    }
    // Keep the larger drug-core half!
    if rem.len() * 2 >= mol.atoms.len() {
        return None;
    }

    let mut old_to_new = vec![None; mol.atoms.len()];
    let mut out = Molecule::new("");
    for (oi, a) in mol.atoms.iter().enumerate() {
        if rem.contains(&oi) {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == het_keep {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[oi] = Some(ni);
    }
    for b in &mol.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = out.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }
    let mapped_het = old_to_new[het_keep]?;
    out.sanitize();
    out.smiles = molecule_to_smiles(&out);
    compute_2d_coords(&mut out);
    Some((out, mapped_het))
}

/// Checks if a hydroxylated aromatic metabolite forms an *ortho*- or *para*-aminophenol / catechol
/// quinone-imine / ortho-quinone reactive bioactivation alert!
fn check_quinone_imine_bioactivation(mol: &Molecule, new_oh_o: usize) -> (bool, &'static str) {
    let Some(&(c_ar, _)) = mol.adj.get(new_oh_o).and_then(|v| v.first()) else {
        return (false, "Clean Phase I oxidative metabolite");
    };
    if !mol.atoms[c_ar].aromatic {
        return (false, "Clean Phase I oxidative metabolite");
    }
    if let Some(ring) = mol.rings.iter().find(|r| r.contains(&c_ar) && r.len() == 6) {
        let pos = ring.iter().position(|&x| x == c_ar).unwrap_or(0);
        for step in [1usize, 3, 5] {
            // Ortho (step 1, 5) or Para (step 3) on 6-membered aromatic ring!
            let opp = ring[(pos + step) % 6];
            for &(exo, _) in &mol.adj[opp] {
                if ring.contains(&exo) {
                    continue;
                }
                if mol.atoms[exo].elem == Element::N && mol.atoms[exo].charge == 0 {
                    return (
                        true,
                        "Bioactivation Alert: ortho/para-aminophenol oxidizes to electrophilic quinone-imine (acetaminophen/gefitinib GSH-trapping pathway)",
                    );
                }
                if mol.atoms[exo].elem == Element::O && mol.atoms[exo].total_h() > 0 {
                    return (
                        true,
                        "Bioactivation Alert: catechol oxidizes to reactive ortho-quinone protein-modifying electrophile",
                    );
                }
            }
        }
    }
    (false, "Clean phenolic metabolite (substrate for Phase II UGT/SULT)")
}

/// Grafts a beta-D-glucuronide ring (`+176.03 Da`) onto nucleophilic `-OH` or `-COOH` oxygen `oxy_idx`.
fn build_o_glucuronide_conjugate(parent: &Molecule, oxy_idx: usize) -> Option<(Molecule, Vec<usize>)> {
    let gluc = parse_smiles("BrC1OC(C(=O)O)C(O)C(O)C1O").ok()?;
    let br_pos = gluc.atoms.iter().position(|a| a.elem == Element::Br)?;
    let c1_old = gluc.adj[br_pos].first()?.0;

    let mut out = parent.clone();
    out.atoms[oxy_idx].explicit_h = 0;
    out.atoms[oxy_idx].bracketed = out.atoms[oxy_idx].charge != 0;

    let offset = out.atoms.len();
    let mut old_to_new = vec![None; gluc.atoms.len()];
    let mut added = vec![oxy_idx];

    for (gi, ga) in gluc.atoms.iter().enumerate() {
        if gi == br_pos {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = ga.clone();
        na.idx = ni;
        if gi == c1_old {
            na.explicit_h = 0;
            na.bracketed = false;
        }
        out.atoms.push(na);
        old_to_new[gi] = Some(ni);
        added.push(ni);
    }
    for gb in &gluc.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[gb.a], old_to_new[gb.b]) {
            let bi = out.bonds.len();
            let mut nb = gb.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }
    let _ = offset;
    let c1_new = old_to_new[c1_old]?;
    push_bond_simple(&mut out, oxy_idx, c1_new, BondOrder::Single);

    out.sanitize();
    out.smiles = molecule_to_smiles(&out);
    compute_2d_coords(&mut out);
    Some((out, added))
}

/// Grafts a sulfate ester `-O-SO3H` (`+79.96 Da`) onto phenolic or aliphatic `-OH` oxygen `oxy_idx`.
fn build_o_sulfate_conjugate(parent: &Molecule, oxy_idx: usize) -> (Molecule, Vec<usize>) {
    let mut out = parent.clone();
    out.atoms[oxy_idx].explicit_h = 0;
    out.atoms[oxy_idx].bracketed = false;

    let s = push_atom_simple(&mut out, Element::S, 0);
    let o1 = push_atom_simple(&mut out, Element::O, 0);
    let o2 = push_atom_simple(&mut out, Element::O, 0);
    let oh = push_atom_simple(&mut out, Element::O, 0);
    push_bond_simple(&mut out, oxy_idx, s, BondOrder::Single);
    push_bond_simple(&mut out, s, o1, BondOrder::Double);
    push_bond_simple(&mut out, s, o2, BondOrder::Double);
    push_bond_simple(&mut out, s, oh, BondOrder::Single);

    out.sanitize();
    out.smiles = molecule_to_smiles(&out);
    compute_2d_coords(&mut out);
    (out, vec![oxy_idx, s, o1, o2, oh])
}

/// Estimates unbound human hepatic microsomal intrinsic clearance `CL_int,u (mL/min/kg)` and
/// isoform fractional contributions `[fm_3A4, fm_2D6, fm_2C9]` from `SMARTCyp` barriers and `logD_7.4`.
fn estimate_clearance_and_fm(
    sites: &[SmartCypSite],
    logd_74: f64,
) -> (f64, f64, MetabolicStabilityClass, [f64; 3]) {
    if sites.is_empty() {
        return (4.5, 165.0, MetabolicStabilityClass::HighStability, [0.70, 0.15, 0.15]);
    }

    // Boltzmann rate sum across top 5 reactive sites at 310 K (RT = 2.577 kJ/mol)
    let rt = 2.577f64;
    let mut k_3a4 = 0.0f64;
    let mut k_2d6 = 0.0f64;
    let mut k_2c9 = 0.0f64;

    for s in sites.iter().take(6) {
        k_3a4 += (-(s.score_3a4 - 62.0) / (3.2 * rt)).exp();
        k_2d6 += 0.42 * (-(s.score_2d6 - 62.0) / (3.2 * rt)).exp();
        k_2c9 += 0.38 * (-(s.score_2c9 - 62.0) / (3.2 * rt)).exp();
    }

    let k_tot = (k_3a4 + k_2d6 + k_2c9).max(1e-6);
    let fm = [k_3a4 / k_tot, k_2d6 / k_tot, k_2c9 / k_tot];

    // Lipophilicity affinity boost into hydrophobic CYP active sites
    let lipo_factor = 10.0f64.powf(0.18 * (logd_74 - 2.0).clamp(-2.5, 3.0));
    let cl_int = (9.5 * k_tot.powf(0.55) * lipo_factor).clamp(2.5, 185.0);
    let t_half_min = (693.0 / cl_int).clamp(4.0, 240.0);

    let stab = if cl_int < 15.0 {
        MetabolicStabilityClass::HighStability
    } else if cl_int <= 45.0 {
        MetabolicStabilityClass::ModerateTurnover
    } else {
        MetabolicStabilityClass::HighClearanceLabile
    };

    (cl_int, t_half_min, stab, fm)
}

/// Evaluates fast dataset-level `SMARTCyp` clearance & bioactivation summary for `mol`.
pub fn evaluate_metabolic_summary(mol: &Molecule) -> MetabolicDatasetSummary {
    let sites = predict_smartcyp_sites(mol);
    let pka = evaluate_physiological_pka_74(mol);
    let (cl_int, _, stab, _) = estimate_clearance_and_fm(&sites, pka.microstate_logd_74);

    let phase2_direct_handles = mol
        .atoms
        .iter()
        .filter(|a| matches!(a.elem, Element::O | Element::N) && a.total_h() > 0 && a.charge == 0)
        .count();

    let reactive_risk = sites.iter().take(3).any(|s| {
        matches!(
            s.reaction_kind,
            SomReactionKind::Epoxidation | SomReactionKind::AllylicPropargylicOxidation
        ) || (s.reaction_kind == SomReactionKind::AromaticHydroxylation && s.dft_ea_kj_mol <= 74.5)
    });

    if let Some(s0) = sites.first() {
        let best_sc = s0.score_3a4.min(s0.score_2d6).min(s0.score_2c9);
        MetabolicDatasetSummary {
            top_som_atom: Some(s0.atom_idx),
            top_som_label: s0.atom_label.clone(),
            top_reaction: s0.reaction_kind.label(),
            primary_isoform: s0.dominant_isoform,
            best_smartcyp_score: best_sc,
            cl_int_ml_min_kg: cl_int,
            stability_class: stab,
            phase2_direct_handles,
            reactive_bioactivation_risk: reactive_risk,
        }
    } else {
        MetabolicDatasetSummary {
            top_som_atom: None,
            top_som_label: "-".to_string(),
            top_reaction: "Metabolically Inert",
            primary_isoform: CypIsoform::Cyp3a4,
            best_smartcyp_score: 99.0,
            cl_int_ml_min_kg: cl_int,
            stability_class: stab,
            phase2_direct_handles,
            reactive_bioactivation_risk: false,
        }
    }
}

/// Runs full `SMARTCyp` (`CYP3A4`/`2D6`/`2C9`) regioselective `SoM` ranking and enumerates
/// Phase I, direct Phase II, and sequential `Phase I -> II` metabolites with 2D coordinates.
pub fn analyze_cyp_metabolism(mol: &Molecule, include_sequential_phase2: bool) -> CypMetabolismReport {
    let parent_mw = mol.molecular_weight();
    let parent_pka = evaluate_physiological_pka_74(mol);
    let parent_logd = parent_pka.microstate_logd_74;

    let sites = predict_smartcyp_sites(mol);
    let (cl_int, t_half_min, stab, fm) = estimate_clearance_and_fm(&sites, parent_logd);

    let mut raw_mets: Vec<(
        MetabolismPhase,
        &'static str,
        String,
        Option<usize>,
        f64,
        bool,
        &'static str,
        Vec<usize>,
        Molecule,
        Option<usize>, // optional newly created -OH oxygen index for sequential Phase II!
    )> = Vec::new();
    let mut seen_smiles: BTreeSet<String> = BTreeSet::new();
    seen_smiles.insert(molecule_to_smiles(mol));

    // 1. Enumerate Phase I Metabolites from Top SMARTCyp Sites
    for s in sites.iter().take(5) {
        let u = s.atom_idx;
        let iso_str = s.dominant_isoform.label();
        let weight = (-(s.score_3a4.min(s.score_2d6).min(s.score_2c9) - 55.0) / 8.5).exp();

        match s.reaction_kind {
            SomReactionKind::ODealkylation | SomReactionKind::NDealkylation => {
                if let Some((m_dealk, het_idx)) = cleave_acyclic_hetero_alkyl(mol, u) {
                    if seen_smiles.insert(m_dealk.smiles.clone()) {
                        let new_oh = if mol.atoms[het_idx].elem == Element::O {
                            Some(het_idx)
                        } else {
                            None
                        };
                        let (bio_flag, bio_note) = if let Some(oh) = new_oh {
                            check_quinone_imine_bioactivation(&m_dealk, oh)
                        } else {
                            (false, "Active or polar N-dealkylated Phase I metabolite")
                        };
                        raw_mets.push((
                            MetabolismPhase::PhaseICyp,
                            iso_str,
                            format!("{} at {}", s.reaction_kind.label(), s.atom_label),
                            Some(u),
                            weight * 1.15,
                            bio_flag,
                            bio_note,
                            vec![het_idx],
                            m_dealk,
                            new_oh,
                        ));
                    }
                }
            }
            SomReactionKind::SOxidation => {
                let mut m_so = mol.clone();
                let ox = push_atom_simple(&mut m_so, Element::O, 0);
                push_bond_simple(&mut m_so, u, ox, BondOrder::Double);
                m_so.sanitize();
                m_so.smiles = molecule_to_smiles(&m_so);
                compute_2d_coords(&mut m_so);
                if seen_smiles.insert(m_so.smiles.clone()) {
                    raw_mets.push((
                        MetabolismPhase::PhaseICyp,
                        "CYP3A4 / FMO3",
                        format!("S-Oxidation at {}", s.atom_label),
                        Some(u),
                        weight * 1.25,
                        false,
                        "Polar sulfoxide / sulfone metabolite (major hepatic CYP/FMO pathway)",
                        vec![u, ox],
                        m_so,
                        None,
                    ));
                }
            }
            SomReactionKind::NOxidation => {
                let mut m_no = mol.clone();
                m_no.atoms[u].charge = 1;
                m_no.atoms[u].bracketed = true;
                let ox = push_atom_simple(&mut m_no, Element::O, -1);
                push_bond_simple(&mut m_no, u, ox, BondOrder::Single);
                m_no.sanitize();
                m_no.smiles = molecule_to_smiles(&m_no);
                compute_2d_coords(&mut m_no);
                if seen_smiles.insert(m_no.smiles.clone()) {
                    raw_mets.push((
                        MetabolismPhase::PhaseICyp,
                        "CYP3A4 / FMO3",
                        format!("Tertiary Amine N-Oxide at {}", s.atom_label),
                        Some(u),
                        weight,
                        false,
                        "Zwitterionic amine N-oxide renal clearance metabolite",
                        vec![u, ox],
                        m_no,
                        None,
                    ));
                }
            }
            SomReactionKind::Epoxidation => {
                // Convert C=C bond at `u` into 3-membered oxirane ring!
                if let Some(&(v_partner, bi)) = mol.adj[u].iter().find(|&&(nb, bi)| {
                    mol.atoms[nb].elem == Element::C && mol.bonds[bi].order == BondOrder::Double
                }) {
                    let mut m_ep = mol.clone();
                    m_ep.bonds[bi].order = BondOrder::Single;
                    m_ep.bonds[bi].kekule_order = 1;
                    let ox = push_atom_simple(&mut m_ep, Element::O, 0);
                    push_bond_simple(&mut m_ep, u, ox, BondOrder::Single);
                    push_bond_simple(&mut m_ep, v_partner, ox, BondOrder::Single);
                    m_ep.sanitize();
                    m_ep.smiles = molecule_to_smiles(&m_ep);
                    compute_2d_coords(&mut m_ep);
                    if seen_smiles.insert(m_ep.smiles.clone()) {
                        raw_mets.push((
                            MetabolismPhase::PhaseICyp,
                            iso_str,
                            format!("Alkene Epoxidation at {}", s.atom_label),
                            Some(u),
                            weight,
                            true,
                            "Bioactivation Alert: electrophilic epoxide intermediate (substrate for mEH / GST GSH trapping)",
                            vec![u, v_partner, ox],
                            m_ep,
                            None,
                        ));
                    }
                }
            }
            SomReactionKind::AllylicPropargylicOxidation if mol.atoms[u].hyb == Hybridization::Sp => {
                // Terminal acetylene C#CH -> aryl/alkyl acetic acid via ketene hydrolysis (exact Erlotinib M14 pathway!)
                if let Some(&(c_in, bi)) = mol.adj[u].first() {
                    let mut m_ac = mol.clone();
                    m_ac.bonds[bi].order = BondOrder::Single;
                    m_ac.bonds[bi].kekule_order = 1;
                    let o_co = push_atom_simple(&mut m_ac, Element::O, 0);
                    let o_oh = push_atom_simple(&mut m_ac, Element::O, 0);
                    push_bond_simple(&mut m_ac, u, o_co, BondOrder::Double);
                    push_bond_simple(&mut m_ac, u, o_oh, BondOrder::Single);
                    m_ac.sanitize();
                    m_ac.smiles = molecule_to_smiles(&m_ac);
                    compute_2d_coords(&mut m_ac);
                    if seen_smiles.insert(m_ac.smiles.clone()) {
                        raw_mets.push((
                            MetabolismPhase::PhaseICyp,
                            "CYP3A4",
                            format!("Acetylene -> Ketene -> Carboxylic Acid at {}", s.atom_label),
                            Some(u),
                            weight,
                            true,
                            "Bioactivation Alert: CYP oxidation of terminal alkyne proceeds via reactive ketene/oxirene MBI intermediate (Erlotinib CYP3A4 pathway)",
                            vec![c_in, u, o_co, o_oh],
                            m_ac,
                            Some(o_oh),
                        ));
                    }
                }
            }
            _ => {
                // C-H Hydroxylation (Aromatic, Benzylic, Heterocyclic ring, Allylic, Aliphatic) -> C-OH!
                let mut m_oh = mol.clone();
                m_oh.atoms[u].explicit_h = m_oh.atoms[u].explicit_h.saturating_sub(1);
                let ox = push_atom_simple(&mut m_oh, Element::O, 0);
                push_bond_simple(&mut m_oh, u, ox, BondOrder::Single);
                m_oh.sanitize();
                m_oh.smiles = molecule_to_smiles(&m_oh);
                compute_2d_coords(&mut m_oh);
                if seen_smiles.insert(m_oh.smiles.clone()) {
                    let (bio_flag, bio_note) = if m_oh.atoms[u].aromatic {
                        check_quinone_imine_bioactivation(&m_oh, ox)
                    } else {
                        (false, "Primary hydroxylated Phase I metabolite")
                    };
                    raw_mets.push((
                        MetabolismPhase::PhaseICyp,
                        iso_str,
                        format!("{} at {}", s.reaction_kind.label(), s.atom_label),
                        Some(u),
                        weight,
                        bio_flag,
                        bio_note,
                        vec![u, ox],
                        m_oh,
                        Some(ox),
                    ));
                }
            }
        }
    }

    // 2. Direct Phase II Conjugation on Parent Nucleophilic Handles (-OH -> UGT Glucuronide & SULT Sulfate; Ar-NH2 -> NAT2 Acetamide)
    for (u, a) in mol.atoms.iter().enumerate() {
        if a.elem == Element::O && a.total_h() > 0 && a.charge == 0 && mol.adj[u].len() == 1 {
            let att = mol.adj[u][0].0;
            let is_carboxyl = mol.atoms[att].elem == Element::C
                && mol.adj[att].iter().any(|&(nb, bi)| {
                    nb != u
                        && mol.atoms[nb].elem == Element::O
                        && mol.bonds[bi].order == BondOrder::Double
                });

            if let Some((m_gluc, mod_atoms)) = build_o_glucuronide_conjugate(mol, u) {
                if seen_smiles.insert(m_gluc.smiles.clone()) {
                    raw_mets.push((
                        MetabolismPhase::PhaseIIConjugate,
                        "UGT1A1 / UGT2B7",
                        format!(
                            "Direct {}-Glucuronidation at O#{}",
                            if is_carboxyl { "Acyl" } else { "O" },
                            u + 1
                        ),
                        Some(u),
                        0.55,
                        is_carboxyl,
                        if is_carboxyl {
                            "Acyl glucuronide Alert: can undergo acyl migration & covalent plasma protein transacylation (Indomethacin/Diclofenac MIST class)"
                        } else {
                            "High-solubility beta-D-glucuronide biliary/renal clearance conjugate (+176.03 Da)"
                        },
                        mod_atoms,
                        m_gluc,
                        None,
                    ));
                }
            }

            if !is_carboxyl {
                let (m_sulf, mod_atoms) = build_o_sulfate_conjugate(mol, u);
                if seen_smiles.insert(m_sulf.smiles.clone()) {
                    raw_mets.push((
                        MetabolismPhase::PhaseIIConjugate,
                        "SULT1A1",
                        format!("Direct O-Sulfation at O#{}", u + 1),
                        Some(u),
                        0.40,
                        false,
                        "Anionic sulfate conjugate (+79.96 Da) cleared rapidly via OAT transporters",
                        mod_atoms,
                        m_sulf,
                        None,
                    ));
                }
            }
            break;
        }
    }

    // 3. Sequential Phase I -> Phase II Cascade (Glucuronidation & Sulfation of Top Newly Formed Phase I -OH Metabolite!)
    if include_sequential_phase2 {
        let phase1_with_oh: Vec<(Molecule, usize, String, Option<usize>, f64)> = raw_mets
            .iter()
            .filter_map(|(ph, _, rname, p_som, w, _, _, _, m_p1, oh_opt)| {
                if *ph == MetabolismPhase::PhaseICyp {
                    oh_opt.map(|oh| (m_p1.clone(), oh, rname.clone(), *p_som, *w))
                } else {
                    None
                }
            })
            .take(2)
            .collect();

        for (m_p1, oh_idx, p1_name, p_som, p1_w) in phase1_with_oh {
            if let Some((m_seq_gluc, mod_atoms)) = build_o_glucuronide_conjugate(&m_p1, oh_idx) {
                if seen_smiles.insert(m_seq_gluc.smiles.clone()) {
                    raw_mets.push((
                        MetabolismPhase::SequentialPhaseIToII,
                        "CYP -> UGT1A9",
                        format!("{p1_name} -> O-Glucuronide"),
                        p_som,
                        p1_w * 0.62,
                        false,
                        "Sequential Phase I oxidation + Phase II O-glucuronidation (+192.03 Da / +162.02 Da net)",
                        mod_atoms,
                        m_seq_gluc,
                        None,
                    ));
                }
            }
            let (m_seq_sulf, mod_atoms) = build_o_sulfate_conjugate(&m_p1, oh_idx);
            if seen_smiles.insert(m_seq_sulf.smiles.clone()) {
                raw_mets.push((
                    MetabolismPhase::SequentialPhaseIToII,
                    "CYP -> SULT1A1",
                    format!("{p1_name} -> O-Sulfate"),
                    p_som,
                    p1_w * 0.42,
                    false,
                    "Sequential Phase I oxidation + Phase II O-sulfation",
                    mod_atoms,
                    m_seq_sulf,
                    None,
                ));
            }
        }
    }

    // Normalize relative abundances & populate MetaboliteSpecies
    let w_sum: f64 = raw_mets.iter().map(|r| r.4).sum::<f64>().max(1e-9);
    let mut metabolites: Vec<MetaboliteSpecies> = raw_mets
        .into_iter()
        .map(
            |(ph, enz, rxn_name, p_som, w, bio_flag, bio_note, mod_atoms, mut m, _)| {
                let mw = m.molecular_weight();
                let mass_shift = mw - parent_mw;
                let logd = evaluate_physiological_pka_74(&m).microstate_logd_74;
                let abund = (w / w_sum) * 100.0;
                m.target = mol.target.clone();
                MetaboliteSpecies {
                    metabolite_id: String::new(),
                    rank: 0,
                    phase: ph,
                    enzyme: enz,
                    reaction_name: rxn_name,
                    parent_som_atom: p_som,
                    mass_shift_da: mass_shift,
                    exact_mw: mw,
                    logd_74: logd,
                    delta_logd_74: logd - parent_logd,
                    predicted_abundance_pct: abund,
                    is_reactive_bioactivation: bio_flag,
                    bioactivation_note: bio_note,
                    canonical_smiles: m.smiles.clone(),
                    modified_atoms: mod_atoms,
                    mol: m,
                }
            },
        )
        .collect();

    metabolites.sort_by(|a, b| {
        b.predicted_abundance_pct
            .partial_cmp(&a.predicted_abundance_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, met) in metabolites.iter_mut().enumerate() {
        met.rank = i + 1;
        met.metabolite_id = format!("M{}", i + 1);
        met.mol.id = format!("{}-M{}", mol.id, i + 1);
        met.mol.name = format!("{} {} ({})", mol.name, met.metabolite_id, met.enzyme);
    }

    CypMetabolismReport {
        parent_mw,
        parent_logd_74: parent_logd,
        cl_int_ml_min_kg: cl_int,
        predicted_t_half_min: t_half_min,
        stability_class: stab,
        isoform_fraction_fm: fm,
        sites,
        metabolites,
    }
}

/// Exports enumerated Phase I & Phase II metabolites (including LC-HRMS `mass_shift_da`) to CSV.
pub fn export_metabolites_csv(rep: &CypMetabolismReport) -> String {
    let mut out = String::from(
        "metabolite_id,rank,phase,enzyme,reaction,parent_som_atom,mass_shift_da,mw,logd_74,delta_logd_74,abundance_pct,reactive_bioactivation,note,smiles\n",
    );
    for m in &rep.metabolites {
        let som_txt = m
            .parent_som_atom
            .map(|u| format!("{}", u + 1))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "\"{}\",{},\"{}\",\"{}\",\"{}\",{},{:+.3},{:.2},{:.2},{:+.2},{:.1},{},\"{}\",\"{}\"\n",
            m.metabolite_id,
            m.rank,
            m.phase.short_label(),
            m.enzyme,
            m.reaction_name,
            som_txt,
            m.mass_shift_da,
            m.exact_mw,
            m.logd_74,
            m.delta_logd_74,
            m.predicted_abundance_pct,
            m.is_reactive_bioactivation,
            m.bioactivation_note,
            m.canonical_smiles
        ));
    }
    out
}
