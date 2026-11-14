//! Bemis & Murcko (1996) (`https://pubmed.ncbi.nlm.nih.gov/8709122/`) molecular framework
//! decomposition: extracts ring + linker Murcko scaffolds and carbon-normalized generic cyclic
//! skeletons via Weisfeiler-Lehman canonical graph invariants.

use crate::mol::{Element, Molecule};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MurckoScaffold {
    /// Canonical Morgan/WL topological signature of the Murcko framework (rings + linkers).
    pub framework_key: String,
    /// Human-readable topological descriptor (e.g. `Bicyclic [6,6] C8N2 (Quinazoline-like)`).
    pub framework_label: String,
    /// Carbon-normalized single-bond generic cyclic skeleton signature for bioisostere grouping.
    pub generic_skeleton_key: String,
    /// Representative molecule index for 2D scaffold depiction.
    pub exemplar_mol: usize,
    /// Atom indices in `exemplar_mol` belonging to the Murcko framework.
    pub exemplar_atoms: Vec<usize>,
    /// Indices of all dataset molecules sharing this Murcko framework.
    pub member_mols: Vec<usize>,
    pub ring_count: usize,
    pub mean_pact: f64,
    pub min_pact: f64,
    pub max_pact: f64,
    pub pact_spread: f64,
}

/// Extracts the atom-index mask belonging to the Bemis-Murcko framework of `mol`
/// (iteratively pruning exocyclic heavy atoms of degree `<= 1` while retaining ring systems
/// and the inter-ring acyclic linker paths connecting them).
pub fn murcko_framework_mask(mol: &Molecule) -> Vec<bool> {
    let n = mol.atoms.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        if mol.atoms[i].elem == Element::H {
            keep[i] = false;
        }
    }

    let mut deg = vec![0usize; n];
    for i in 0..n {
        if keep[i] {
            deg[i] = mol.adj[i].iter().filter(|&&(v, _)| keep[v]).count();
        }
    }

    // Iteratively peel all leaf (`deg <= 1`) atoms until only rings + inter-ring linkers remain
    let mut changed = true;
    while changed {
        changed = false;
        for u in 0..n {
            if keep[u] && deg[u] <= 1 {
                keep[u] = false;
                deg[u] = 0;
                changed = true;
                for &(v, _) in &mol.adj[u] {
                    if keep[v] {
                        deg[v] = deg[v].saturating_sub(1);
                    }
                }
            }
        }
    }

    // Also retain exocyclic carbonyl `=O` / `=S` directly double-bonded to a framework ring atom
    // ONLY for display if desired? Strict Bemis & Murcko (1996) removes exocyclic `=O` substituents
    // unless specified; keeping strict ring+linker union ensures clean series clustering!
    keep
}

/// Computes a canonical Weisfeiler-Lehman graph signature `(framework_key, generic_skeleton_key, label)`
/// for the Bemis-Murcko subgraph of `mol`.
pub fn murcko_signatures(mol: &Molecule) -> (String, String, String, Vec<usize>) {
    let keep = murcko_framework_mask(mol);
    let kept_atoms: Vec<usize> = (0..mol.atoms.len()).filter(|&u| keep[u]).collect();
    if kept_atoms.is_empty() {
        return (
            "ACYCLIC".to_string(),
            "ACYCLIC".to_string(),
            "Acyclic Chain".to_string(),
            Vec::new(),
        );
    }

    let fw_hash = wl_subgraph_hash(mol, &keep, false);
    let skel_hash = wl_subgraph_hash(mol, &keep, true);

    // Build human-readable formula + ring-system topology summary for the Murcko core
    let mut elem_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for &u in &kept_atoms {
        *elem_counts.entry(mol.atoms[u].elem.symbol()).or_insert(0) += 1;
    }
    let mut core_formula = String::new();
    if let Some(&c) = elem_counts.get("C") {
        core_formula.push_str(&format!("C{}", if c > 1 { c.to_string() } else { String::new() }));
        elem_counts.remove("C");
    }
    for (sym, cnt) in elem_counts {
        core_formula.push_str(sym);
        if cnt > 1 {
            core_formula.push_str(&cnt.to_string());
        }
    }

    let mut r_sizes: Vec<usize> = mol
        .rings
        .iter()
        .filter(|r| r.iter().all(|&u| keep[u]))
        .map(|r| r.len())
        .collect();
    r_sizes.sort_unstable();
    let ring_str = r_sizes
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let prefix = match r_sizes.len() {
        0 => "Acyclic",
        1 => "Monocyclic",
        2 => "Bicyclic",
        3 => "Tricyclic",
        4 => "Tetracyclic",
        _ => "Polycyclic",
    };

    let fw_key = format!("BM-{}-{:016x}", core_formula, fw_hash);
    let skel_key = format!("SKEL-N{}-R[{}]-{:016x}", kept_atoms.len(), ring_str, skel_hash);
    let label = format!("{} [{}] {}", prefix, ring_str, core_formula);

    (fw_key, skel_key, label, kept_atoms)
}

/// 6-round Weisfeiler-Lehman canonical graph hash over the induced subgraph `keep`.
/// When `generic_skeleton` is `true`, all elements collapse to carbon (`6`) and all bonds to `1`.
pub fn wl_subgraph_hash(mol: &Molecule, keep: &[bool], generic_skeleton: bool) -> u64 {
    let n = mol.atoms.len();
    let mut colors = vec![0u64; n];

    for u in 0..n {
        if !keep[u] {
            continue;
        }
        let sub_deg = mol.adj[u].iter().filter(|&&(v, _)| keep[v]).count() as u64;
        if generic_skeleton {
            colors[u] = mix64(0x636f7265_00000006 ^ (sub_deg << 8) ^ (mol.atoms[u].in_ring as u64));
        } else {
            let a = &mol.atoms[u];
            colors[u] = mix64(
                (a.elem.atomic_number() as u64)
                    ^ (sub_deg << 8)
                    ^ ((a.aromatic as u64) << 16)
                    ^ ((a.in_ring as u64) << 17),
            );
        }
    }

    for round in 0..6u64 {
        let mut next = colors.clone();
        for u in 0..n {
            if !keep[u] {
                continue;
            }
            let mut nbrs: Vec<(u8, u64)> = mol.adj[u]
                .iter()
                .filter(|&&(v, _)| keep[v])
                .map(|&(v, bi)| {
                    let bo = if generic_skeleton {
                        1u8
                    } else if mol.bonds[bi].aromatic {
                        4u8
                    } else {
                        mol.bonds[bi].kekule_order
                    };
                    (bo, colors[v])
                })
                .collect();
            nbrs.sort_unstable();
            let mut h = mix64(colors[u] ^ (round.wrapping_mul(0x9e3779b97f4a7c15)));
            for (bo, nc) in nbrs {
                h = mix64(h ^ ((bo as u64) << 40) ^ nc);
            }
            next[u] = h;
        }
        colors = next;
    }

    let mut multiset: Vec<u64> = (0..n).filter(|&u| keep[u]).map(|u| colors[u]).collect();
    multiset.sort_unstable();
    let mut final_h = 0xcbf29ce484222325u64 ^ (multiset.len() as u64);
    for c in multiset {
        final_h = mix64(final_h ^ c);
    }
    final_h
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Aggregates Bemis-Murcko scaffolds across an entire compound dataset, sorted by member count then max potency.
pub fn extract_murcko_scaffolds(mols: &[Molecule]) -> Vec<MurckoScaffold> {
    let mut groups: BTreeMap<String, MurckoScaffold> = BTreeMap::new();

    for (mi, mol) in mols.iter().enumerate() {
        let (fw_key, skel_key, label, kept_atoms) = murcko_signatures(mol);
        let p = mol.pact;
        let entry = groups.entry(fw_key.clone()).or_insert_with(|| MurckoScaffold {
            framework_key: fw_key,
            framework_label: label,
            generic_skeleton_key: skel_key,
            exemplar_mol: mi,
            exemplar_atoms: kept_atoms.clone(),
            member_mols: Vec::new(),
            ring_count: mol
                .rings
                .iter()
                .filter(|r| r.iter().all(|a| kept_atoms.contains(a)))
                .count(),
            mean_pact: 0.0,
            min_pact: p,
            max_pact: p,
            pact_spread: 0.0,
        });
        if p > entry.max_pact {
            entry.max_pact = p;
            entry.exemplar_mol = mi;
            entry.exemplar_atoms = kept_atoms;
        }
        if p < entry.min_pact {
            entry.min_pact = p;
        }
        entry.member_mols.push(mi);
    }

    let mut out: Vec<MurckoScaffold> = groups
        .into_values()
        .map(|mut g| {
            let sum: f64 = g.member_mols.iter().map(|&mi| mols[mi].pact).sum();
            g.mean_pact = sum / (g.member_mols.len().max(1) as f64);
            g.pact_spread = g.max_pact - g.min_pact;
            g
        })
        .collect();

    out.sort_by(|a, b| {
        b.member_mols
            .len()
            .cmp(&a.member_mols.len())
            .then_with(|| {
                b.max_pact
                    .partial_cmp(&a.max_pact)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    out
}
