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

//! Hussain & Rea (2010) (`https://pubmed.ncbi.nlm.nih.gov/20121245/`) Matched Molecular Pair (`MMP`)
//! R-group fragmentation & SAR transform engine:
//! 1. Single-cut acyclic heavy-bond fragmentation (`Core[*] + *[R]`, `|R| <= 12` heavy atoms).
//! 2. Implicit hydrogen `*[H]` ring/amine substituent sites (e.g. `Amoxapine *[H] >> Loxapine *[CH3]`).
//! 3. Symmetric twin-cut identical substituent exchanges (e.g. `PD-153035 2x*[OCH3] >> Compound 56 2x*[OCH2CH3]`).
//! 4. Generic cyclic skeleton core bioisosteres (e.g. `Rofecoxib >> Celecoxib >> Valdecoxib` & `Loxapine >> Clozapine`).

use crate::mol::{Element, Molecule};
use crate::physchem::PhyschemProfile;
use crate::scaffold::{murcko_signatures, wl_subgraph_hash};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmpKind {
    SingleCutRGroup,
    SymmetricTwinCut,
    CoreBioisostere,
}

impl MmpKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SingleCutRGroup => "1-Cut R-Group",
            Self::SymmetricTwinCut => "Symmetric 2-Cut",
            Self::CoreBioisostere => "Core Bioisostere",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchedMolecularPair {
    pub mol_from: usize,
    pub mol_to: usize,
    pub target: String,
    pub kind: MmpKind,
    pub transform: String,
    pub core_label: String,
    pub delta_pact: f64,
    pub delta_clogp: f64,
    pub delta_tpsa: f64,
    pub delta_lle: f64,
}

#[derive(Clone, Debug)]
struct CutRecord {
    mol_idx: usize,
    core_formula: String,
    rgroup_label: String,
    kind: MmpKind,
}

/// Discovers intra-target Matched Molecular Pairs (`MMPs`) and core bioisosteres across `mols`.
pub fn discover_mmps(mols: &[Molecule], profiles: &[PhyschemProfile]) -> Vec<MatchedMolecularPair> {
    if mols.len() < 2 {
        return Vec::new();
    }

    // Limit MMP combinatorics cleanly if a synthetic 50,000-compound benchmark is loaded
    let active_len = mols.len().min(600);
    let mut cuts_by_core: BTreeMap<(String, u64), Vec<CutRecord>> = BTreeMap::new();

    for mi in 0..active_len {
        let mol = &mols[mi];
        let n = mol.atoms.len();
        if n < 3 {
            continue;
        }

        // 1. Single-cut acyclic bonds splitting molecule into Core (>= 55% heavy atoms) + R-group (<= 12 heavy atoms)
        let mut single_cuts_for_mol: Vec<(usize, Vec<bool>, String)> = Vec::new();

        for (bi, b) in mol.bonds.iter().enumerate() {
            if b.in_ring || b.aromatic || b.kekule_order != 1 {
                continue;
            }
            let comp_a = component_without_bond(mol, b.a, bi);
            let count_a = comp_a.iter().filter(|&&x| x).count();
            if count_a == 0 || count_a == n {
                continue; // Bridge check failed (still connected)
            }
            let count_b = n - count_a;

            // Test both directions if valid, though normally the larger side is Core
            for &(core_anchor, r_root, is_a_core) in &[(b.a, b.b, true), (b.b, b.a, false)] {
                let (core_cnt, r_cnt) = if is_a_core {
                    (count_a, count_b)
                } else {
                    (count_b, count_a)
                };
                if r_cnt > 12 || core_cnt < 6 || core_cnt < r_cnt {
                    continue;
                }
                let mut core_mask = vec![false; n];
                let mut r_mask = vec![false; n];
                for u in 0..n {
                    let in_a = comp_a[u];
                    if in_a == is_a_core {
                        core_mask[u] = true;
                    } else {
                        r_mask[u] = true;
                    }
                }
                let chash = rooted_core_hash(mol, &core_mask, &[core_anchor]);
                let cform = mask_formula(mol, &core_mask);
                let rlabel = format_rgroup_label(mol, &r_mask, r_root);

                cuts_by_core
                    .entry((mol.target.clone(), chash))
                    .or_default()
                    .push(CutRecord {
                        mol_idx: mi,
                        core_formula: cform,
                        rgroup_label: rlabel.clone(),
                        kind: MmpKind::SingleCutRGroup,
                    });
                single_cuts_for_mol.push((core_anchor, r_mask, rlabel));
            }
        }

        // 2. Implicit `*[H]` substituent sites on ring C-H or basic/secondary N-H atoms
        let full_mask = vec![true; n];
        let full_form = mask_formula(mol, &full_mask);
        for u in 0..n {
            if mol.atoms[u].total_h() > 0 && (mol.atoms[u].in_ring || mol.atoms[u].elem == Element::N) {
                let chash = rooted_core_hash(mol, &full_mask, &[u]);
                cuts_by_core
                    .entry((mol.target.clone(), chash))
                    .or_default()
                    .push(CutRecord {
                        mol_idx: mi,
                        core_formula: full_form.clone(),
                        rgroup_label: "*[H]".to_string(),
                        kind: MmpKind::SingleCutRGroup,
                    });
            }
        }

        // 3. Symmetric twin-cut identical R-groups (e.g. 6,7-dimethoxy vs 6,7-diethoxy quinazolines!)
        for p in 0..single_cuts_for_mol.len() {
            for q in (p + 1)..single_cuts_for_mol.len() {
                let (anc1, ref rmask1, ref rlab1) = single_cuts_for_mol[p];
                let (anc2, ref rmask2, ref rlab2) = single_cuts_for_mol[q];
                if anc1 == anc2 || rlab1 != rlab2 {
                    continue;
                }
                if (0..n).any(|u| rmask1[u] && rmask2[u]) {
                    continue;
                }
                let mut twin_core = vec![true; n];
                for u in 0..n {
                    if rmask1[u] || rmask2[u] {
                        twin_core[u] = false;
                    }
                }
                if twin_core.iter().filter(|&&x| x).count() < 6 {
                    continue;
                }
                let chash = rooted_core_hash(mol, &twin_core, &[anc1, anc2]);
                let cform = mask_formula(mol, &twin_core);
                cuts_by_core
                    .entry((mol.target.clone(), chash))
                    .or_default()
                    .push(CutRecord {
                        mol_idx: mi,
                        core_formula: cform,
                        rgroup_label: format!("2x {}", rlab1),
                        kind: MmpKind::SymmetricTwinCut,
                    });
            }
        }
    }

    let mut pairs_map: BTreeMap<(usize, usize), MatchedMolecularPair> = BTreeMap::new();

    for ((target, _), group) in &cuts_by_core {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = &group[i];
                let b = &group[j];
                if a.mol_idx == b.mol_idx || a.rgroup_label == b.rgroup_label {
                    continue;
                }
                // Orient pair from lower potency -> higher potency so `delta_pact >= 0` shows potency-boosting transforms cleanly
                let (from_rec, to_rec) = if mols[a.mol_idx].pact <= mols[b.mol_idx].pact {
                    (a, b)
                } else {
                    (b, a)
                };
                let key = (from_rec.mol_idx, to_rec.mol_idx);
                let dp = mols[to_rec.mol_idx].pact - mols[from_rec.mol_idx].pact;
                let dlogp = profiles[to_rec.mol_idx].clogp - profiles[from_rec.mol_idx].clogp;
                let dtpsa = profiles[to_rec.mol_idx].tpsa - profiles[from_rec.mol_idx].tpsa;
                let dlle = profiles[to_rec.mol_idx].lle - profiles[from_rec.mol_idx].lle;

                let candidate = MatchedMolecularPair {
                    mol_from: from_rec.mol_idx,
                    mol_to: to_rec.mol_idx,
                    target: target.clone(),
                    kind: if a.kind == MmpKind::SymmetricTwinCut || b.kind == MmpKind::SymmetricTwinCut {
                        MmpKind::SymmetricTwinCut
                    } else {
                        MmpKind::SingleCutRGroup
                    },
                    transform: format!("{} >> {}", from_rec.rgroup_label, to_rec.rgroup_label),
                    core_label: format!("Core [{}]", from_rec.core_formula),
                    delta_pact: dp,
                    delta_clogp: dlogp,
                    delta_tpsa: dtpsa,
                    delta_lle: dlle,
                };

                let replace = match pairs_map.get(&key) {
                    None => true,
                    Some(existing) => {
                        // Prefer larger conserved core (smaller R-group label)
                        candidate.transform.len() < existing.transform.len()
                    }
                };
                if replace {
                    pairs_map.insert(key, candidate);
                }
            }
        }
    }

    // 4. Generic Cyclic Skeleton Bioisosteres (same target, identical carbon-normalized cyclic skeleton, different hetero-core!)
    let mut skel_groups: BTreeMap<(String, String), Vec<(usize, String, String)>> = BTreeMap::new();
    for mi in 0..active_len {
        let (fw_key, skel_key, fw_label, _) = murcko_signatures(&mols[mi]);
        if skel_key != "ACYCLIC" {
            skel_groups
                .entry((mols[mi].target.clone(), skel_key))
                .or_default()
                .push((mi, fw_key, fw_label));
        }
    }

    for ((target, _), members) in skel_groups {
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let (m1, ref fw1, ref lab1) = members[i];
                let (m2, ref fw2, ref lab2) = members[j];
                if fw1 == fw2 {
                    continue;
                }
                let (from_i, to_i, from_lab, to_lab) = if mols[m1].pact <= mols[m2].pact {
                    (m1, m2, lab1, lab2)
                } else {
                    (m2, m1, lab2, lab1)
                };
                let key = (from_i, to_i);
                pairs_map.entry(key).or_insert_with(|| MatchedMolecularPair {
                    mol_from: from_i,
                    mol_to: to_i,
                    target: target.clone(),
                    kind: MmpKind::CoreBioisostere,
                    transform: format!("{} >> {}", short_core_tag(from_lab), short_core_tag(to_lab)),
                    core_label: "Isosteric Cyclic Skeleton".to_string(),
                    delta_pact: mols[to_i].pact - mols[from_i].pact,
                    delta_clogp: profiles[to_i].clogp - profiles[from_i].clogp,
                    delta_tpsa: profiles[to_i].tpsa - profiles[from_i].tpsa,
                    delta_lle: profiles[to_i].lle - profiles[from_i].lle,
                });
            }
        }
    }

    let mut out: Vec<MatchedMolecularPair> = pairs_map.into_values().collect();
    out.sort_by(|a, b| {
        b.delta_pact
            .abs()
            .partial_cmp(&a.delta_pact.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn short_core_tag(label: &str) -> String {
    label
        .split_whitespace()
        .last()
        .map(|f| format!("Core[{}]", f))
        .unwrap_or_else(|| label.to_string())
}

fn component_without_bond(mol: &Molecule, start: usize, skip_bi: usize) -> Vec<bool> {
    let mut vis = vec![false; mol.atoms.len()];
    let mut q = VecDeque::new();
    vis[start] = true;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        for &(v, bi) in &mol.adj[u] {
            if bi != skip_bi && !vis[v] {
                vis[v] = true;
                q.push_back(v);
            }
        }
    }
    vis
}

/// Canonical Weisfeiler-Lehman hash of `core_mask` distinguishing the attachment anchor atom(s).
fn rooted_core_hash(mol: &Molecule, mask: &[bool], anchors: &[usize]) -> u64 {
    let base = wl_subgraph_hash(mol, mask, false);
    // Refine with 4-hop WL centered on attachment points so ortho/meta/para regioisomers never collide!
    let n = mol.atoms.len();
    let mut colors = vec![0u64; n];
    for u in 0..n {
        if !mask[u] {
            continue;
        }
        let is_anc = anchors.contains(&u) as u64;
        let deg = mol.adj[u].iter().filter(|&&(v, _)| mask[v]).count() as u64;
        colors[u] = (mol.atoms[u].elem.atomic_number() as u64)
            ^ (deg << 8)
            ^ ((mol.atoms[u].aromatic as u64) << 16)
            ^ (is_anc * 0x517cc1b727220a95);
    }
    for _ in 0..5 {
        let mut next = colors.clone();
        for u in 0..n {
            if !mask[u] {
                continue;
            }
            let mut nbrs: Vec<u64> = mol.adj[u]
                .iter()
                .filter(|&&(v, _)| mask[v])
                .map(|&(v, bi)| colors[v].wrapping_mul(mol.bonds[bi].kekule_order as u64 + 3))
                .collect();
            nbrs.sort_unstable();
            let mut h = colors[u].wrapping_mul(0xbf58476d1ce4e5b9);
            for nc in nbrs {
                h = (h ^ nc).wrapping_mul(0x94d049bb133111eb);
            }
            next[u] = h ^ (h >> 29);
        }
        colors = next;
    }
    let mut anc_sigs: Vec<u64> = anchors.iter().map(|&a| colors[a]).collect();
    anc_sigs.sort_unstable();
    let mut out = base;
    for s in anc_sigs {
        out = (out ^ s).wrapping_mul(0x9e3779b97f4a7c15);
    }
    out ^ (out >> 31)
}

fn mask_formula(mol: &Molecule, mask: &[bool]) -> String {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (u, &on) in mask.iter().enumerate() {
        if on && mol.atoms[u].elem != Element::H {
            *counts.entry(mol.atoms[u].elem.symbol()).or_insert(0) += 1;
        }
    }
    let mut s = String::new();
    if let Some(&c) = counts.get("C") {
        s.push('C');
        if c > 1 {
            s.push_str(&c.to_string());
        }
        counts.remove("C");
    }
    for (sym, cnt) in counts {
        s.push_str(sym);
        if cnt > 1 {
            s.push_str(&cnt.to_string());
        }
    }
    s
}

/// Formats readable medicinal-chemistry R-group shorthand (`*[Cl]`, `*[Br]`, `*[CF3]`, `*[OCH3]`, `*[OCH2CH3]`, `*[C#CH]`).
fn format_rgroup_label(mol: &Molecule, mask: &[bool], root: usize) -> String {
    let heavy_cnt = mask.iter().filter(|&&x| x).count();
    let r_elem = mol.atoms[root].elem;
    let r_h = mol.atoms[root].total_h();

    if heavy_cnt == 1 {
        let sym = r_elem.symbol();
        return match (r_elem, r_h) {
            (Element::C, 3) => "*[CH3]".to_string(),
            (Element::N, 2) => "*[NH2]".to_string(),
            (Element::O, 1) => "*[OH]".to_string(),
            (Element::S, 1) => "*[SH]".to_string(),
            _ => format!("*[{}]", sym),
        };
    }

    if heavy_cnt == 2 {
        let other = (0..mol.atoms.len())
            .find(|&u| mask[u] && u != root)
            .unwrap_or(root);
        let bi = mol.bond_between(root, other);
        let bsym = match bi.map(|b| mol.bonds[b].kekule_order).unwrap_or(1) {
            3 => "#",
            2 => "=",
            _ => "",
        };
        return format!(
            "*[{}{}{}]",
            atom_with_h(mol, root),
            bsym,
            atom_with_h(mol, other)
        );
    }

    if heavy_cnt == 3 {
        // Common 3-heavy groups: OCH2CH3, CF3 (4 heavy), C(=O)O, S(=O)2CH3, etc.
        let root_nbrs: Vec<usize> = mol.adj[root]
            .iter()
            .map(|&(v, _)| v)
            .filter(|&v| mask[v])
            .collect();
        if root_nbrs.len() == 1 {
            let mid = root_nbrs[0];
            if let Some(leaf) = (0..mol.atoms.len()).find(|&u| mask[u] && u != root && u != mid) {
                return format!(
                    "*[{}{}{}]",
                    atom_with_h(mol, root),
                    atom_with_h(mol, mid),
                    atom_with_h(mol, leaf)
                );
            }
        }
    }

    if heavy_cnt == 4 && r_elem == Element::C {
        let f_cnt = mol.adj[root]
            .iter()
            .filter(|&&(v, _)| mask[v] && mol.atoms[v].elem == Element::F)
            .count();
        if f_cnt == 3 {
            return "*[CF3]".to_string();
        }
    }

    let canon = wl_subgraph_hash(mol, mask, false) & 0xffff;
    format!("*[{}#{:04x}]", mask_formula(mol, mask), canon)
}

fn atom_with_h(mol: &Molecule, u: usize) -> String {
    let sym = mol.atoms[u].elem.symbol();
    let h = mol.atoms[u].total_h();
    if h == 0 {
        sym.to_string()
    } else if h == 1 {
        format!("{}H", sym)
    } else {
        format!("{}H{}", sym, h)
    }
}
