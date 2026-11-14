//! Baell & Holloway (2010) (`https://pubmed.ncbi.nlm.nih.gov/20131845/`) Pan-Assay Interference
//! Compounds (`PAINS`), covalent warheads, and metabolic structural alert detectors.

use crate::mol::{Element, Hybridization, Molecule};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    PainsAssayInterference,
    CovalentWarhead,
    MetabolicLiability,
}

impl AlertSeverity {
    pub fn label(self) -> &'static str {
        match self {
            Self::PainsAssayInterference => "PAINS Interference",
            Self::CovalentWarhead => "Covalent Warhead",
            Self::MetabolicLiability => "Metabolic Alert",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralAlertMatch {
    #[serde(skip_deserializing)]
    pub code: &'static str,
    #[serde(skip_deserializing)]
    pub family: &'static str,
    pub severity: AlertSeverity,
    #[serde(skip_deserializing)]
    pub description: &'static str,
    pub atoms: Vec<usize>,
}

/// Scans `mol` for Baell-Holloway `PAINS` chemotypes, targeted covalent warheads, and reactive liabilities.
pub fn scan_structural_alerts(mol: &Molecule) -> Vec<StructuralAlertMatch> {
    let mut alerts = Vec::new();
    let n = mol.atoms.len();

    // 1. `ene_rhod_A`: 5-alkylidene rhodanine / thiazolidinone (e.g. Epalrestat)
    for ring in &mol.rings {
        if ring.len() == 5 {
            let has_s = ring.iter().any(|&u| mol.atoms[u].elem == Element::S);
            let has_n = ring.iter().any(|&u| mol.atoms[u].elem == Element::N);
            let mut exo_ene_atoms = Vec::new();
            let mut has_carbonyl_or_thiocarbonyl = false;

            for &u in ring {
                if mol.atoms[u].elem == Element::C {
                    for &(v, bi) in &mol.adj[u] {
                        if !ring.contains(&v) && mol.bonds[bi].kekule_order == 2 {
                            if matches!(mol.atoms[v].elem, Element::O | Element::S) {
                                has_carbonyl_or_thiocarbonyl = true;
                            } else if mol.atoms[v].elem == Element::C {
                                exo_ene_atoms.push(u);
                                exo_ene_atoms.push(v);
                            }
                        }
                    }
                }
            }
            if has_s && has_n && has_carbonyl_or_thiocarbonyl && !exo_ene_atoms.is_empty() {
                let mut matched = ring.clone();
                for a in exo_ene_atoms {
                    if !matched.contains(&a) {
                        matched.push(a);
                    }
                }
                alerts.push(StructuralAlertMatch {
                    code: "ene_rhod_A",
                    family: "5-Alkylidene Rhodanine",
                    severity: AlertSeverity::PainsAssayInterference,
                    description: "Photoreactive / promiscuous thiol-trapping Michael acceptor (Baell & Holloway 2010)",
                    atoms: matched,
                });
            }
        }
    }

    // 2. `catechol_A`: ortho-dihydroxyaryl redox cycler / metal chelator (e.g. Quercetin)
    let mut catechol_atoms = Vec::new();
    for b in &mol.bonds {
        if mol.atoms[b.a].aromatic
            && mol.atoms[b.b].aromatic
            && mol.atoms[b.a].elem == Element::C
            && mol.atoms[b.b].elem == Element::C
        {
            if let (Some(oh_a), Some(oh_b)) = (phenolic_oh(mol, b.a), phenolic_oh(mol, b.b)) {
                for idx in [b.a, b.b, oh_a, oh_b] {
                    if !catechol_atoms.contains(&idx) {
                        catechol_atoms.push(idx);
                    }
                }
            }
        }
    }
    if !catechol_atoms.is_empty() {
        alerts.push(StructuralAlertMatch {
            code: "catechol_A",
            family: "Ortho-Catechol",
            severity: AlertSeverity::PainsAssayInterference,
            description: "Redox-cycling ortho-quinone precursor and divalent metal chelator (Baell & Holloway 2010)",
            atoms: catechol_atoms,
        });
    }

    // 3. `quinone_A`: para- or ortho-benzoquinone (6-ring with >=2 exocyclic C=O and non-aromatic ring C=C)
    for ring in &mol.rings {
        if ring.len() == 6 && ring.iter().all(|&u| mol.atoms[u].elem == Element::C) {
            let mut carbonyl_pairs = Vec::new();
            for &u in ring {
                for &(v, bi) in &mol.adj[u] {
                    if !ring.contains(&v)
                        && mol.bonds[bi].kekule_order == 2
                        && mol.atoms[v].elem == Element::O
                    {
                        carbonyl_pairs.push((u, v));
                    }
                }
            }
            if carbonyl_pairs.len() >= 2 {
                let mut matched = ring.clone();
                for (_, o_idx) in carbonyl_pairs {
                    matched.push(o_idx);
                }
                alerts.push(StructuralAlertMatch {
                    code: "quinone_A",
                    family: "Quinone Redox Cycler",
                    severity: AlertSeverity::PainsAssayInterference,
                    description: "Electrophilic & ROS-generating quinone core (Baell & Holloway 2010)",
                    atoms: matched,
                });
            }
        }
    }

    // 4. Acyclic `C=C-C(=O)`: separate targeted `covalent_acrylamide` (`N-C(=O)-C=C`, e.g. Afatinib)
    // from promiscuous `michael_enone` / `curcumin_diketone` (`C-C(=O)-C=C`, e.g. Curcumin)
    let mut enone_atoms = Vec::new();
    let mut acrylamide_atoms = Vec::new();
    for b in &mol.bonds {
        if !b.in_ring
            && !b.aromatic
            && b.kekule_order == 2
            && mol.atoms[b.a].elem == Element::C
            && mol.atoms[b.b].elem == Element::C
        {
            for &(vinyl_c, other_vinyl) in &[(b.a, b.b), (b.b, b.a)] {
                for &(carb, bi_cc) in &mol.adj[vinyl_c] {
                    if carb == other_vinyl || mol.bonds[bi_cc].in_ring || mol.atoms[carb].elem != Element::C {
                        continue;
                    }
                    let oxo_opt = mol.adj[carb].iter().find_map(|&(o, bi_co)| {
                        if mol.bonds[bi_co].kekule_order == 2
                            && !mol.bonds[bi_co].in_ring
                            && mol.atoms[o].elem == Element::O
                        {
                            Some(o)
                        } else {
                            None
                        }
                    });
                    if let Some(oxo) = oxo_opt {
                        let amide_n = mol.adj[carb].iter().find_map(|&(nb, _)| {
                            if nb != vinyl_c && nb != oxo && mol.atoms[nb].elem == Element::N {
                                Some(nb)
                            } else {
                                None
                            }
                        });
                        if let Some(n_idx) = amide_n {
                            for idx in [other_vinyl, vinyl_c, carb, oxo, n_idx] {
                                if !acrylamide_atoms.contains(&idx) {
                                    acrylamide_atoms.push(idx);
                                }
                            }
                        } else {
                            let is_ester_or_acid = mol.adj[carb].iter().any(|&(nb, _)| {
                                nb != vinyl_c && nb != oxo && mol.atoms[nb].elem == Element::O
                            });
                            if !is_ester_or_acid {
                                for idx in [other_vinyl, vinyl_c, carb, oxo] {
                                    if !enone_atoms.contains(&idx) {
                                        enone_atoms.push(idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if !enone_atoms.is_empty() {
        alerts.push(StructuralAlertMatch {
            code: "curcumin_enone_A",
            family: "Acyclic Enone / Curcuminoid",
            severity: AlertSeverity::PainsAssayInterference,
            description: "Conjugated Michael acceptor / membrane-perturbing enone-diketone (Baell & Holloway 2010)",
            atoms: enone_atoms,
        });
    }

    if !acrylamide_atoms.is_empty() {
        alerts.push(StructuralAlertMatch {
            code: "covalent_acrylamide",
            family: "Acrylamide Warhead",
            severity: AlertSeverity::CovalentWarhead,
            description: "Targeted alpha,beta-unsaturated acrylamide covalent Cys modifier (e.g. EGFR Cys797)",
            atoms: acrylamide_atoms,
        });
    }

    // 5. `terminal_alkyne`: `-C#CH` CYP450 time-dependent inactivation alert (e.g. Erlotinib)
    let mut alkyne_atoms = Vec::new();
    for b in &mol.bonds {
        if b.kekule_order == 3
            && mol.atoms[b.a].elem == Element::C
            && mol.atoms[b.b].elem == Element::C
            && (mol.atoms[b.a].total_h() >= 1 || mol.atoms[b.b].total_h() >= 1)
        {
            alkyne_atoms.push(b.a);
            alkyne_atoms.push(b.b);
        }
    }
    if !alkyne_atoms.is_empty() {
        alerts.push(StructuralAlertMatch {
            code: "terminal_alkyne",
            family: "Terminal Acetylene",
            severity: AlertSeverity::MetabolicLiability,
            description: "Potential CYP450 ketene-intermediate mechanism-based inactivation liability",
            atoms: alkyne_atoms,
        });
    }

    // 6. `aromatic_nitro`: `Ar-NO2` mutagenic nitroreduction alert
    let mut nitro_atoms = Vec::new();
    for u in 0..n {
        if mol.atoms[u].elem == Element::N && mol.atoms[u].hyb != Hybridization::Sp3 {
            let oxo_nbrs: Vec<usize> = mol.adj[u]
                .iter()
                .filter_map(|&(v, _)| (mol.atoms[v].elem == Element::O).then_some(v))
                .collect();
            let on_arom = mol.adj[u].iter().any(|&(v, _)| mol.atoms[v].aromatic);
            if oxo_nbrs.len() >= 2 && on_arom {
                nitro_atoms.push(u);
                nitro_atoms.extend(oxo_nbrs);
            }
        }
    }
    if !nitro_atoms.is_empty() {
        alerts.push(StructuralAlertMatch {
            code: "arom_nitro",
            family: "Aromatic Nitro",
            severity: AlertSeverity::MetabolicLiability,
            description: "Nitroreductase reactive nitroso/hydroxylamine structural alert",
            atoms: nitro_atoms,
        });
    }

    alerts
}

fn phenolic_oh(mol: &Molecule, c_idx: usize) -> Option<usize> {
    mol.adj[c_idx].iter().find_map(|&(v, bi)| {
        if !mol.bonds[bi].in_ring
            && mol.bonds[bi].kekule_order == 1
            && mol.atoms[v].elem == Element::O
            && mol.atoms[v].total_h() >= 1
        {
            Some(v)
        } else {
            None
        }
    })
}
