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

//! McGregor (1982) + `VF2` Maximum Common Substructure (`MCS`) Core Detector and
//! 2D Orthogonal-Procrustes Congeneric Series R-Group Aligner.
//!
//! Primary Literature Grounding:
//! - McGregor (1982), *Softw. Pract. Exper.* 12(1), 23–34
//!   (`https://www.semanticscholar.org/paper/Backtrack-search-algorithms-and-the-maximal-common-McGregor/ed10c6f788922cd5e7cb26c3d55f676979958852`):
//!   Branch-and-bound backtrack search with lookahead upper bounds for maximal common subgraphs.
//! - Raymond & Willett (2002), *J. Comput.-Aided Mol. Des.* 16(7), 521–533
//!   (`https://pubmed.ncbi.nlm.nih.gov/12510884/`): Cheminformatics MCS taxonomy & `MCS Tanimoto` metric.
//! - Bonnici & Giugno (2013), *BMC Bioinformatics* 14(Suppl 7):S13
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3633016/`): `VF2` state-space subgraph isomorphism feasibility.
//! - Hariharan et al. / MultiMCS & FMCS (2013), *J. Cheminform.*
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3606201/`): Multi-molecule consensus MCS across congeneric series.

use crate::mol::{Bond, Element, Molecule};
use crate::rgroup_table::scaffold_subgraph_to_smarts;
use crate::smarts::{compile_smarts, match_smarts, SmartsQuery};
use crate::DatasetAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Atom-matching policy during McGregor / `VF2` Maximum Common Substructure search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum McsAtomMatchMode {
    /// Strict element + aromaticity equality (`C == C`, `c == c`, `N == N`, `n == n`).
    StrictElement,
    /// Bioisosteric ring-heteroatom tolerance: ring atoms (`C, N, O, S`) match across identical
    /// ring-membership classes (enables unified MCS across pyrazole/furanone/isoxazole coxibs
    /// and dibenzoxazepine/dibenzodiazepine/dibenzothiazepine tricyclics).
    AnyRingHetero,
}

impl McsAtomMatchMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::StrictElement => "Strict Element & Aromaticity",
            Self::AnyRingHetero => "Bioisosteric Ring Heteroatoms (C/N/O/S)",
        }
    }
}

/// Configuration for pairwise and multi-molecule series Maximum Common Substructure discovery.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McsConfig {
    pub atom_mode: McsAtomMatchMode,
    /// Enforce `bond_a.in_ring == bond_b.in_ring` (prevents alkyl chains matching partial phenyl rings).
    pub ring_matches_ring_only: bool,
    /// Prune dangling incomplete ring stubs from the final MCS core when intact rings exist.
    pub complete_rings_only: bool,
    /// Minimum heavy-atom count required for a valid common scaffold core.
    pub min_core_atoms: usize,
    /// Fraction of series compounds (`0.40..=1.00`) that must contain the multi-molecule MCS core.
    pub threshold_fraction: f32,
    /// Maximum backtracking states explored per pairwise McGregor call (guarantees sub-ms responsiveness).
    pub max_search_states: usize,
}

impl Default for McsConfig {
    fn default() -> Self {
        Self {
            atom_mode: McsAtomMatchMode::StrictElement,
            ring_matches_ring_only: true,
            complete_rings_only: true,
            min_core_atoms: 6,
            threshold_fraction: 0.70,
            max_search_states: 32_000,
        }
    }
}

/// Result of a pairwise McGregor / `VF2` Maximum Common Substructure comparison between `mol_a` and `mol_b`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairwiseMcsResult {
    /// Ordered `(atom_in_a, atom_in_b)` vertex correspondence of the connected MCS.
    pub atom_mapping: Vec<(usize, usize)>,
    pub core_atoms_a: Vec<usize>,
    pub core_atoms_b: Vec<usize>,
    pub core_bonds_a: Vec<usize>,
    pub core_bonds_b: Vec<usize>,
    pub mcs_atom_count: usize,
    pub mcs_bond_count: usize,
    /// Raymond & Willett (2002) `MCS Tanimoto = |MCS| / (|A| + |B| - |MCS|)`.
    pub mcs_tanimoto: f32,
    /// Daylight SMARTS pattern encoding the discovered common substructure.
    pub mcs_smarts: String,
    /// 2D Orthogonal-Procrustes superposition RMSD (Å) of `mol_b` core onto `mol_a` core.
    pub core_rmsd_2d: f32,
    pub states_visited: usize,
}

/// A single series member rigidly aligned in 2D onto the consensus MCS core with extracted R-groups.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignedSeriesMember {
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    pub pact: f64,
    /// Copy of `Molecule` whose `pos2d` coordinates have been rigidly transformed (2D Orthogonal Procrustes:
    /// optimal rotation + optional mirror reflection + translation) onto the reference MCS core!
    pub aligned_mol: Molecule,
    /// Atom indices in `aligned_mol` belonging to the conserved MCS core (ordered `0..core_size`).
    pub core_atoms: Vec<usize>,
    /// Bond indices in `aligned_mol` belonging to the conserved MCS core.
    pub core_bonds: Vec<usize>,
    /// Exocyclic substituent atom indices (`R-groups`) outside the MCS core.
    pub rgroup_atoms: Vec<usize>,
    /// Substituent formula/group strings at each variable core attachment slot `R1..Rk`.
    pub r_groups: Vec<String>,
    /// 2D RMSD (Å) of `core_atoms` against the reference template core after Procrustes alignment.
    pub core_rmsd_2d: f32,
    /// Pairwise `MCS Tanimoto` against the reference template molecule.
    pub mcs_tanimoto_to_ref: f32,
    /// `true` if a mirror reflection (`det(R) = -1`) was applied to align 2D ring orientation.
    pub reflected_2d: bool,
}

/// Complete multi-molecule Series MCS Core & 2D R-Group Alignment solution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeriesMcsAlignment {
    pub target_scope: String,
    pub total_input_molecules: usize,
    pub matched_member_count: usize,
    pub coverage_fraction: f32,
    pub reference_mol_idx: usize,
    pub reference_name: String,
    pub mcs_atom_count: usize,
    pub mcs_bond_count: usize,
    /// Daylight SMARTS string ready for one-click handoff into the Free-Wilson SAR Matrix solver!
    pub mcs_smarts: String,
    /// Variable core attachment slot labels (`["R1", "R2", ...]`) ordered by substituent diversity.
    pub slot_labels: Vec<String>,
    /// 2D coordinates of each `R1..Rk` attachment port on the reference core for port-badge rendering.
    pub slot_anchor_pos2d: Vec<[f32; 2]>,
    pub mean_core_rmsd_2d: f32,
    pub members: Vec<AlignedSeriesMember>,
}

fn is_ring_bioisostere_elem(el: Element) -> bool {
    matches!(el, Element::C | Element::N | Element::O | Element::S)
}

#[inline]
fn atoms_compatible(mol_a: &Molecule, u: usize, mol_b: &Molecule, v: usize, cfg: &McsConfig) -> bool {
    let aa = &mol_a.atoms[u];
    let ab = &mol_b.atoms[v];
    if aa.elem == Element::H || ab.elem == Element::H {
        return false;
    }
    if cfg.ring_matches_ring_only && aa.in_ring != ab.in_ring {
        return false;
    }
    match cfg.atom_mode {
        McsAtomMatchMode::StrictElement => aa.elem == ab.elem && aa.aromatic == ab.aromatic,
        McsAtomMatchMode::AnyRingHetero => {
            if aa.in_ring && ab.in_ring && is_ring_bioisostere_elem(aa.elem) && is_ring_bioisostere_elem(ab.elem) {
                // Match ring atoms across C/N/O/S when they share a common smallest ring size (e.g. 5-ring or 6-ring)
                let min_ra = aa.ring_sizes.iter().copied().min().unwrap_or(0);
                let min_rb = ab.ring_sizes.iter().copied().min().unwrap_or(0);
                min_ra == min_rb || (aa.elem == ab.elem && aa.aromatic == ab.aromatic)
            } else {
                aa.elem == ab.elem && aa.aromatic == ab.aromatic
            }
        }
    }
}

#[inline]
fn bonds_compatible(ba: &Bond, bb: &Bond, cfg: &McsConfig) -> bool {
    if cfg.ring_matches_ring_only && ba.in_ring != bb.in_ring {
        return false;
    }
    match cfg.atom_mode {
        McsAtomMatchMode::StrictElement => ba.aromatic == bb.aromatic && ba.order == bb.order,
        McsAtomMatchMode::AnyRingHetero => {
            if ba.in_ring && bb.in_ring {
                true
            } else {
                ba.aromatic == bb.aromatic && ba.order == bb.order
            }
        }
    }
}

/// Prunes dangling incomplete ring stubs from a candidate atom mapping `(u, v)` when `complete_rings_only`
/// is enabled, retaining the largest connected core component.
fn prune_incomplete_rings(
    mol_a: &Molecule,
    raw_map: &[(usize, usize)],
    min_atoms: usize,
) -> Vec<(usize, usize)> {
    if raw_map.len() <= min_atoms {
        return raw_map.to_vec();
    }
    let mut keep_a: BTreeSet<usize> = raw_map.iter().map(|&(u, _)| u).collect();

    // Iteratively drop any ring atom `u` that belongs to a ring in `mol_a` yet NO complete ring
    // containing `u` is fully present in `keep_a` (unless `u` is an exocyclic substituent anchor
    // whose ring was not partially entered — i.e., only prune if >= 2 atoms of a ring were entered
    // without completing that ring!)
    let mut changed = true;
    while changed {
        changed = false;
        let mut to_remove = Vec::new();
        for &u in &keep_a {
            if !mol_a.atoms[u].in_ring {
                continue;
            }
            let rings_with_u: Vec<&Vec<usize>> =
                mol_a.rings.iter().filter(|r| r.contains(&u)).collect();
            let any_complete = rings_with_u
                .iter()
                .any(|r| r.iter().all(|ru| keep_a.contains(ru)));
            let any_partial_multi = rings_with_u.iter().any(|r| {
                let cnt = r.iter().filter(|ru| keep_a.contains(ru)).count();
                cnt >= 2 && cnt < r.len()
            });
            if !any_complete && any_partial_multi {
                to_remove.push(u);
            }
        }
        if !to_remove.is_empty() {
            for u in to_remove {
                keep_a.remove(&u);
            }
            changed = true;
        }
    }

    if keep_a.len() < min_atoms {
        return raw_map.to_vec();
    }

    // Retain the largest connected component inside `keep_a`
    let mut best_comp: Vec<usize> = Vec::new();
    let mut visited: BTreeSet<usize> = BTreeSet::new();
    for &seed in &keep_a {
        if visited.contains(&seed) {
            continue;
        }
        let mut comp = Vec::new();
        let mut q = VecDeque::new();
        visited.insert(seed);
        q.push_back(seed);
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &(nb, _) in &mol_a.adj[u] {
                if keep_a.contains(&nb) && visited.insert(nb) {
                    q.push_back(nb);
                }
            }
        }
        if comp.len() > best_comp.len() {
            best_comp = comp;
        }
    }

    if best_comp.len() < min_atoms {
        return raw_map.to_vec();
    }
    let comp_set: BTreeSet<usize> = best_comp.into_iter().collect();
    raw_map
        .iter()
        .copied()
        .filter(|&(u, _)| comp_set.contains(&u))
        .collect()
}

/// Builds a generalize-aware Daylight SMARTS string from `atom_mapping` on `(mol_a, mol_b)`.
/// When `AnyRingHetero` matched differing ring elements (e.g. `N` vs `O` or `C` vs `N`), emits
/// `[#6,#7,#8,#16;r]` at those generalized positions so the SMARTS matches every bioisosteric core!
fn mapping_to_consensus_smarts(
    mol_a: &Molecule,
    mol_b: &Molecule,
    mapping: &[(usize, usize)],
    mode: McsAtomMatchMode,
) -> String {
    if mapping.is_empty() {
        return String::new();
    }
    if mode == McsAtomMatchMode::StrictElement {
        let atoms_a: Vec<usize> = mapping.iter().map(|&(u, _)| u).collect();
        if let Some(sm) = scaffold_subgraph_to_smarts(mol_a, &atoms_a) {
            return sm;
        }
    }

    let n = mol_a.atoms.len();
    let mut map_lookup = vec![None::<usize>; n];
    for &(u, v) in mapping {
        if u < n {
            map_lookup[u] = Some(v);
        }
    }

    let start = mapping[0].0;
    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    let mut children: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut back_edges: Vec<(usize, usize, usize)> = Vec::new();

    fn dfs_tree(
        u: usize,
        mol: &Molecule,
        map_lookup: &[Option<usize>],
        vis: &mut [bool],
        par: &mut [usize],
        ch: &mut [Vec<(usize, usize)>],
        backs: &mut Vec<(usize, usize, usize)>,
    ) {
        vis[u] = true;
        for &(nb, bi) in &mol.adj[u] {
            if map_lookup[nb].is_none() {
                continue;
            }
            if !vis[nb] {
                par[nb] = u;
                ch[u].push((nb, bi));
                dfs_tree(nb, mol, map_lookup, vis, par, ch, backs);
            } else if nb != par[u] && u < nb {
                backs.push((u, nb, bi));
            }
        }
    }

    dfs_tree(
        start,
        mol_a,
        &map_lookup,
        &mut visited,
        &mut parent,
        &mut children,
        &mut back_edges,
    );

    let bond_tok = |bi: usize, v_u: usize, v_nb: usize| -> &'static str {
        let ba = &mol_a.bonds[bi];
        if let Some(bb_idx) = mol_b.bond_between(v_u, v_nb) {
            let bb = &mol_b.bonds[bb_idx];
            if ba.in_ring && bb.in_ring && (ba.aromatic != bb.aromatic || ba.order != bb.order) {
                return "@";
            }
        }
        if ba.aromatic {
            ""
        } else {
            match ba.kekule_order {
                3 => "#",
                2 => "=",
                _ => "-",
            }
        }
    };

    let mut rtags: Vec<Vec<(u16, &'static str)>> = vec![Vec::new(); n];
    for (ri, &(u, nb, bi)) in back_edges.iter().enumerate() {
        let rnum = (ri + 1) as u16;
        let bt = bond_tok(bi, map_lookup[u].unwrap(), map_lookup[nb].unwrap());
        rtags[u].push((rnum, bt));
        rtags[nb].push((rnum, ""));
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_smarts(
        u: usize,
        in_edge: Option<(usize, usize)>,
        mol_a: &Molecule,
        mol_b: &Molecule,
        map_lookup: &[Option<usize>],
        ch: &[Vec<(usize, usize)>],
        rtags: &[Vec<(u16, &'static str)>],
        bond_tok: &dyn Fn(usize, usize, usize) -> &'static str,
        out: &mut String,
    ) {
        let v = map_lookup[u].unwrap();
        if let Some((par_u, bi)) = in_edge {
            let par_v = map_lookup[par_u].unwrap();
            out.push_str(bond_tok(bi, par_v, v));
        }
        let aa = &mol_a.atoms[u];
        let ab = &mol_b.atoms[v];
        let tok = if aa.elem == ab.elem && aa.aromatic == ab.aromatic {
            match (aa.elem, aa.aromatic) {
                (Element::C, true) => "c".to_string(),
                (Element::C, false) => "C".to_string(),
                (Element::N, true) => "n".to_string(),
                (Element::N, false) => "N".to_string(),
                (Element::O, true) => "o".to_string(),
                (Element::O, false) => "O".to_string(),
                (Element::S, true) => "s".to_string(),
                (Element::S, false) => "S".to_string(),
                _ => format!("[#{}]", aa.elem.atomic_number()),
            }
        } else if aa.in_ring && ab.in_ring {
            "[#6,#7,#8,#16;r]".to_string()
        } else {
            format!("[#{},#{}]", aa.elem.atomic_number(), ab.elem.atomic_number())
        };
        out.push_str(&tok);

        for &(rnum, b_str) in &rtags[u] {
            out.push_str(b_str);
            if rnum < 10 {
                out.push_str(&rnum.to_string());
            } else {
                out.push_str(&format!("%{rnum:02}"));
            }
        }

        let kids = &ch[u];
        for (i, &(nb, bi)) in kids.iter().enumerate() {
            let branch = i + 1 < kids.len();
            if branch {
                out.push('(');
            }
            emit_smarts(
                nb,
                Some((u, bi)),
                mol_a,
                mol_b,
                map_lookup,
                ch,
                rtags,
                bond_tok,
                out,
            );
            if branch {
                out.push(')');
            }
        }
    }

    let mut out = String::new();
    emit_smarts(
        start,
        None,
        mol_a,
        mol_b,
        &map_lookup,
        &children,
        &rtags,
        &bond_tok,
        &mut out,
    );
    out
}

/// Computes the optimal 2D Orthogonal-Procrustes rigid transformation (`rotation` + optional
/// `reflection` + `translation`) mapping `mobile_pts` onto `ref_pts` (`|ref_pts| == |mobile_pts|`),
/// returning `(rot_2x2, ref_centroid, mob_centroid, rmsd_2d, reflected)`.
pub fn solve_2d_procrustes(
    ref_pts: &[[f32; 2]],
    mob_pts: &[[f32; 2]],
) -> ([[f32; 2]; 2], [f32; 2], [f32; 2], f32, bool) {
    let k = ref_pts.len().min(mob_pts.len());
    if k == 0 {
        return ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [0.0, 0.0], 0.0, false);
    }

    let mut rc = [0.0f64; 2];
    let mut mc = [0.0f64; 2];
    for i in 0..k {
        rc[0] += ref_pts[i][0] as f64;
        rc[1] += ref_pts[i][1] as f64;
        mc[0] += mob_pts[i][0] as f64;
        mc[1] += mob_pts[i][1] as f64;
    }
    let inv = 1.0 / (k as f64);
    rc[0] *= inv;
    rc[1] *= inv;
    mc[0] *= inv;
    mc[1] *= inv;

    let mut best_rmsd = f64::INFINITY;
    let mut best_mat = [[1.0f32, 0.0], [0.0, 1.0]];
    let mut best_refl = false;

    // Evaluate both proper rotation (flip_x = 1.0) and mirror reflection + rotation (flip_x = -1.0)
    for &(flip_x, is_refl) in &[(1.0f64, false), (-1.0f64, true)] {
        let mut dot_sum = 0.0f64;
        let mut cross_sum = 0.0f64;
        for i in 0..k {
            let px = (ref_pts[i][0] as f64) - rc[0];
            let py = (ref_pts[i][1] as f64) - rc[1];
            let qx = ((mob_pts[i][0] as f64) - mc[0]) * flip_x;
            let qy = (mob_pts[i][1] as f64) - mc[1];
            dot_sum += qx * px + qy * py;
            cross_sum += qx * py - qy * px;
        }
        let theta = cross_sum.atan2(dot_sum);
        let c = theta.cos();
        let s = theta.sin();
        // Combined matrix applied to `(mob - mc)`: first multiply x by `flip_x`, then rotate by `theta`
        let m00 = c * flip_x;
        let m01 = -s;
        let m10 = s * flip_x;
        let m11 = c;

        let mut sq_err = 0.0f64;
        for i in 0..k {
            let px = (ref_pts[i][0] as f64) - rc[0];
            let py = (ref_pts[i][1] as f64) - rc[1];
            let dx_in = (mob_pts[i][0] as f64) - mc[0];
            let dy_in = (mob_pts[i][1] as f64) - mc[1];
            let rx = m00 * dx_in + m01 * dy_in;
            let ry = m10 * dx_in + m11 * dy_in;
            sq_err += (rx - px).powi(2) + (ry - py).powi(2);
        }
        let rmsd = (sq_err * inv).sqrt();
        if rmsd < best_rmsd - 1e-6 {
            best_rmsd = rmsd;
            best_mat = [[m00 as f32, m01 as f32], [m10 as f32, m11 as f32]];
            best_refl = is_refl;
        }
    }

    (
        best_mat,
        [rc[0] as f32, rc[1] as f32],
        [mc[0] as f32, mc[1] as f32],
        best_rmsd as f32,
        best_refl,
    )
}

fn apply_2d_transform(mol: &mut Molecule, mat: [[f32; 2]; 2], ref_c: [f32; 2], mob_c: [f32; 2]) {
    for a in &mut mol.atoms {
        let dx = a.pos2d[0] - mob_c[0];
        let dy = a.pos2d[1] - mob_c[1];
        a.pos2d[0] = ref_c[0] + mat[0][0] * dx + mat[0][1] * dy;
        a.pos2d[1] = ref_c[1] + mat[1][0] * dx + mat[1][1] * dy;
    }
}

/// Solves the connected induced Maximum Common Substructure (`MCS`) between `mol_a` and `mol_b`
/// via McGregor (1982) branch-and-bound backtracking with `VF2` frontier feasibility.
pub fn find_pairwise_mcs(
    mol_a: &Molecule,
    mol_b: &Molecule,
    cfg: &McsConfig,
) -> Option<PairwiseMcsResult> {
    let na = mol_a.atoms.len();
    let nb = mol_b.atoms.len();
    if na == 0 || nb == 0 {
        return None;
    }

    let heavy_a: Vec<usize> = (0..na)
        .filter(|&u| mol_a.atoms[u].elem != Element::H)
        .collect();
    let heavy_b: Vec<usize> = (0..nb)
        .filter(|&v| mol_b.atoms[v].elem != Element::H)
        .collect();
    if heavy_a.len() < cfg.min_core_atoms || heavy_b.len() < cfg.min_core_atoms {
        return None;
    }

    let mut compat = vec![vec![false; nb]; na];
    let mut compat_b_any = vec![false; nb];
    let mut compat_a_deg = vec![0usize; na];
    for &u in &heavy_a {
        for &v in &heavy_b {
            if atoms_compatible(mol_a, u, mol_b, v, cfg) {
                compat[u][v] = true;
                compat_b_any[v] = true;
                compat_a_deg[u] += 1;
            }
        }
    }

    // Order seed vertices in G_a by (in_ring DESC, heavy_degree DESC, fewer compat_b ASC)
    let mut ordered_seeds_a = heavy_a.clone();
    ordered_seeds_a.sort_by(|&u1, &u2| {
        let k1 = (
            mol_a.atoms[u1].in_ring as u8,
            mol_a.heavy_degree(u1),
            usize::MAX - compat_a_deg[u1],
        );
        let k2 = (
            mol_a.atoms[u2].in_ring as u8,
            mol_a.heavy_degree(u2),
            usize::MAX - compat_a_deg[u2],
        );
        k2.cmp(&k1)
    });

    struct SearchCtx<'a> {
        mol_a: &'a Molecule,
        mol_b: &'a Molecule,
        cfg: &'a McsConfig,
        compat: &'a [Vec<bool>],
        forbidden_seeds_a: Vec<bool>,
        map_a: Vec<Option<usize>>,
        map_b: Vec<Option<usize>>,
        cur_pairs: Vec<(usize, usize)>,
        best_pairs: Vec<(usize, usize)>,
        states: usize,
    }

    fn reachable_upper_bound(ctx: &SearchCtx<'_>) -> usize {
        // BFS from currently mapped vertices in G_a through unmapped, non-forbidden, compatible atoms
        let na = ctx.mol_a.atoms.len();
        let mut vis_a = vec![false; na];
        let mut q = VecDeque::new();
        for &(u, _) in &ctx.cur_pairs {
            vis_a[u] = true;
            q.push_back(u);
        }
        let mut extra_a = 0usize;
        while let Some(u) = q.pop_front() {
            for &(nb_a, _) in &ctx.mol_a.adj[u] {
                if vis_a[nb_a]
                    || ctx.forbidden_seeds_a[nb_a]
                    || ctx.mol_a.atoms[nb_a].elem == Element::H
                {
                    continue;
                }
                let has_free_b = ctx.compat[nb_a]
                    .iter()
                    .enumerate()
                    .any(|(vb, &ok)| ok && ctx.map_b[vb].is_none());
                if has_free_b {
                    vis_a[nb_a] = true;
                    extra_a += 1;
                    q.push_back(nb_a);
                }
            }
        }
        let free_b = (0..ctx.mol_b.atoms.len())
            .filter(|&vb| ctx.map_b[vb].is_none() && ctx.mol_b.atoms[vb].elem != Element::H)
            .count();
        ctx.cur_pairs.len() + extra_a.min(free_b)
    }

    fn is_vf2_feasible(ctx: &SearchCtx<'_>, u: usize, v: usize) -> bool {
        if !ctx.compat[u][v] || ctx.map_a[u].is_some() || ctx.map_b[v].is_some() {
            return false;
        }
        // Every already-mapped neighbor `u_m` of `u` must correspond to a compatible neighbor `v_m` of `v`
        for &(u_nb, bi_a) in &ctx.mol_a.adj[u] {
            if let Some(v_nb) = ctx.map_a[u_nb] {
                let Some(bi_b) = ctx.mol_b.bond_between(v, v_nb) else {
                    return false;
                };
                if !bonds_compatible(&ctx.mol_a.bonds[bi_a], &ctx.mol_b.bonds[bi_b], ctx.cfg) {
                    return false;
                }
            }
        }
        // Induced constraint: every already-mapped neighbor `v_m` of `v` must also be bonded to `u`
        for &(v_nb, bi_b) in &ctx.mol_b.adj[v] {
            if let Some(u_nb) = ctx.map_b[v_nb] {
                let Some(bi_a) = ctx.mol_a.bond_between(u, u_nb) else {
                    return false;
                };
                if !bonds_compatible(&ctx.mol_a.bonds[bi_a], &ctx.mol_b.bonds[bi_b], ctx.cfg) {
                    return false;
                }
            }
        }
        true
    }

    fn expand_mcgregor(ctx: &mut SearchCtx<'_>, skipped_frontier_a: &mut Vec<bool>) {
        if ctx.states >= ctx.cfg.max_search_states {
            return;
        }
        ctx.states += 1;

        if ctx.cur_pairs.len() > ctx.best_pairs.len() {
            ctx.best_pairs = ctx.cur_pairs.clone();
        }

        if reachable_upper_bound(ctx) <= ctx.best_pairs.len() {
            return;
        }

        // Collect connected frontier candidates `u` in G_a adjacent to `cur_pairs`
        // Prioritize `u` with the MOST already-mapped neighbors (closes rings immediately -> strong VF2 pruning!)
        let mut best_u: Option<(usize, usize, usize)> = None;
        for &(mapped_u, _) in &ctx.cur_pairs {
            for &(cand_u, _) in &ctx.mol_a.adj[mapped_u] {
                if ctx.map_a[cand_u].is_some()
                    || ctx.forbidden_seeds_a[cand_u]
                    || skipped_frontier_a[cand_u]
                    || ctx.mol_a.atoms[cand_u].elem == Element::H
                {
                    continue;
                }
                let mapped_nbr_cnt = ctx.mol_a.adj[cand_u]
                    .iter()
                    .filter(|&&(w, _)| ctx.map_a[w].is_some())
                    .count();
                let ring_prio = ctx.mol_a.atoms[cand_u].in_ring as usize;
                let key = (mapped_nbr_cnt, ring_prio);
                if best_u.map_or(true, |(_, bm, br)| key > (bm, br)) {
                    best_u = Some((cand_u, mapped_nbr_cnt, ring_prio));
                }
            }
        }

        let Some((u, _, _)) = best_u else {
            return;
        };

        // Restrict candidate `v` in G_b to neighbors of `map_a[u_mapped_nbr]`!
        let anchor_u_m = ctx.mol_a.adj[u]
            .iter()
            .find_map(|&(w, _)| ctx.map_a[w].map(|vw| (w, vw)));
        let mut cand_vs: Vec<usize> = Vec::new();
        if let Some((_, anchor_v)) = anchor_u_m {
            for &(vb, _) in &ctx.mol_b.adj[anchor_v] {
                if is_vf2_feasible(ctx, u, vb) && !cand_vs.contains(&vb) {
                    cand_vs.push(vb);
                }
            }
        }

        // Sort candidate `vb` by degree similarity
        let deg_u = ctx.mol_a.heavy_degree(u);
        cand_vs.sort_by_key(|&vb| ctx.mol_b.heavy_degree(vb).abs_diff(deg_u));

        // Branch 1..k: map `u -> vb`
        for vb in cand_vs {
            ctx.map_a[u] = Some(vb);
            ctx.map_b[vb] = Some(u);
            ctx.cur_pairs.push((u, vb));
            expand_mcgregor(ctx, skipped_frontier_a);
            ctx.cur_pairs.pop();
            ctx.map_a[u] = None;
            ctx.map_b[vb] = None;
            if ctx.states >= ctx.cfg.max_search_states {
                return;
            }
        }

        // Branch k+1: exclude `u` from the current connected subgraph and explore remaining frontier
        skipped_frontier_a[u] = true;
        if reachable_upper_bound(ctx) > ctx.best_pairs.len() {
            // Subtract 1 from upper bound loosely via skipped mask check
            let mut saved_forbid = ctx.forbidden_seeds_a[u];
            ctx.forbidden_seeds_a[u] = true;
            if reachable_upper_bound(ctx) > ctx.best_pairs.len() {
                expand_mcgregor(ctx, skipped_frontier_a);
            }
            ctx.forbidden_seeds_a[u] = saved_forbid;
            let _ = &mut saved_forbid;
        }
        skipped_frontier_a[u] = false;
    }

    let mut ctx = SearchCtx {
        mol_a,
        mol_b,
        cfg,
        compat: &compat,
        forbidden_seeds_a: vec![false; na],
        map_a: vec![None; na],
        map_b: vec![None; nb],
        cur_pairs: Vec::with_capacity(na.min(nb)),
        best_pairs: Vec::new(),
        states: 0,
    };

    let mut skipped_frontier = vec![false; na];
    for &u0 in &ordered_seeds_a {
        if ctx.states >= cfg.max_search_states {
            break;
        }
        let rem_a = heavy_a
            .iter()
            .filter(|&&u| !ctx.forbidden_seeds_a[u])
            .count();
        if rem_a <= ctx.best_pairs.len() || rem_a < cfg.min_core_atoms {
            break;
        }

        for &v0 in &heavy_b {
            if !compat[u0][v0] {
                continue;
            }
            ctx.map_a[u0] = Some(v0);
            ctx.map_b[v0] = Some(u0);
            ctx.cur_pairs.push((u0, v0));
            skipped_frontier.fill(false);
            expand_mcgregor(&mut ctx, &mut skipped_frontier);
            ctx.cur_pairs.pop();
            ctx.map_a[u0] = None;
            ctx.map_b[v0] = None;
            if ctx.states >= cfg.max_search_states {
                break;
            }
        }
        // Mark `u0` permanently explored as a root seed (RIMACS / McGregor finished-vertex pruning!)
        ctx.forbidden_seeds_a[u0] = true;
    }

    let final_pairs = if cfg.complete_rings_only {
        prune_incomplete_rings(mol_a, &ctx.best_pairs, cfg.min_core_atoms)
    } else {
        ctx.best_pairs
    };

    if final_pairs.len() < cfg.min_core_atoms {
        return None;
    }

    let core_atoms_a: Vec<usize> = final_pairs.iter().map(|&(u, _)| u).collect();
    let core_atoms_b: Vec<usize> = final_pairs.iter().map(|&(_, v)| v).collect();
    let set_a: BTreeSet<usize> = core_atoms_a.iter().copied().collect();
    let set_b: BTreeSet<usize> = core_atoms_b.iter().copied().collect();

    let core_bonds_a: Vec<usize> = mol_a
        .bonds
        .iter()
        .enumerate()
        .filter_map(|(bi, b)| (set_a.contains(&b.a) && set_a.contains(&b.b)).then_some(bi))
        .collect();
    let core_bonds_b: Vec<usize> = mol_b
        .bonds
        .iter()
        .enumerate()
        .filter_map(|(bi, b)| (set_b.contains(&b.a) && set_b.contains(&b.b)).then_some(bi))
        .collect();

    let mcs_atom_count = final_pairs.len();
    let denom = heavy_a.len() + heavy_b.len() - mcs_atom_count;
    let mcs_tanimoto = if denom == 0 {
        1.0
    } else {
        (mcs_atom_count as f32) / (denom as f32)
    };

    let is_rigid_scaffold_a = |u: usize| -> bool {
        mol_a.atoms[u].in_ring
            || mol_a.adj[u]
                .iter()
                .filter(|&&(nbr, _)| mol_a.atoms[nbr].in_ring)
                .count()
                >= 2
    };
    let is_rigid_scaffold_b = |v: usize| -> bool {
        mol_b.atoms[v].in_ring
            || mol_b.adj[v]
                .iter()
                .filter(|&&(nbr, _)| mol_b.atoms[nbr].in_ring)
                .count()
                >= 2
    };
    let rigid_pairs: Vec<(usize, usize)> = core_atoms_a
        .iter()
        .copied()
        .zip(core_atoms_b.iter().copied())
        .filter(|&(u, v)| is_rigid_scaffold_a(u) && is_rigid_scaffold_b(v))
        .collect();
    let (ref_pts, mob_pts): (Vec<[f32; 2]>, Vec<[f32; 2]>) = if rigid_pairs.len() >= 4 {
        rigid_pairs
            .iter()
            .map(|&(u, v)| (mol_a.atoms[u].pos2d, mol_b.atoms[v].pos2d))
            .unzip()
    } else {
        (
            core_atoms_a.iter().map(|&u| mol_a.atoms[u].pos2d).collect(),
            core_atoms_b.iter().map(|&v| mol_b.atoms[v].pos2d).collect(),
        )
    };
    let (_, _, _, core_rmsd_2d, _) = solve_2d_procrustes(&ref_pts, &mob_pts);

    let mcs_smarts = mapping_to_consensus_smarts(mol_a, mol_b, &final_pairs, cfg.atom_mode);
    let mcs_bond_count = core_bonds_a.len();

    Some(PairwiseMcsResult {
        atom_mapping: final_pairs,
        core_atoms_a,
        core_atoms_b,
        core_bonds_a,
        core_bonds_b,
        mcs_atom_count,
        mcs_bond_count,
        mcs_tanimoto,
        mcs_smarts,
        core_rmsd_2d,
        states_visited: ctx.states,
    })
}

/// Extracts concise substituent formula tags (`"-[H]"`, `"-[Cl]"`, `"-[C2H5O]"`) at each query
/// core atom `0..core_map.len()` for an aligned molecule.
fn extract_port_substituents(mol: &Molecule, core_map: &[usize]) -> Vec<String> {
    let qn = core_map.len();
    let tn = mol.atoms.len();
    let mut in_core = vec![false; tn];
    for &ti in core_map {
        if ti < tn {
            in_core[ti] = true;
        }
    }

    let mut out = Vec::with_capacity(qn);
    for &ti in core_map {
        let mut exocyclic_branches = Vec::new();
        for &(nb, _) in &mol.adj[ti] {
            if in_core[nb] || mol.atoms[nb].elem == Element::H {
                continue;
            }
            // BFS branch outside core
            let mut visited = vec![false; tn];
            let mut q = VecDeque::new();
            visited[nb] = true;
            q.push_back(nb);
            let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
            let mut h_sum = 0usize;
            while let Some(cur) = q.pop_front() {
                let a = &mol.atoms[cur];
                *counts.entry(a.elem.symbol()).or_insert(0) += 1;
                h_sum += a.total_h() as usize;
                for &(w, _) in &mol.adj[cur] {
                    if !in_core[w] && !visited[w] && mol.atoms[w].elem != Element::H {
                        visited[w] = true;
                        q.push_back(w);
                    }
                }
            }
            if h_sum > 0 {
                *counts.entry("H").or_insert(0) += h_sum;
            }
            let mut f = String::from("-[");
            if let Some(&c) = counts.get("C") {
                f.push('C');
                if c > 1 {
                    f.push_str(&c.to_string());
                }
                counts.remove("C");
                if let Some(&h) = counts.get("H") {
                    f.push('H');
                    if h > 1 {
                        f.push_str(&h.to_string());
                    }
                    counts.remove("H");
                }
            }
            for (sym, cnt) in counts {
                f.push_str(sym);
                if cnt > 1 {
                    f.push_str(&cnt.to_string());
                }
            }
            f.push(']');
            exocyclic_branches.push(f);
        }
        exocyclic_branches.sort();
        if exocyclic_branches.is_empty() {
            out.push("-[H]".to_string());
        } else {
            out.push(exocyclic_branches.join("+"));
        }
    }
    out
}

/// Discovers the consensus Maximum Common Substructure (`MCS`) core across a target series (or
/// active subset in `ds`) and rigidly aligns all matching series members onto a unified 2D
/// coordinate frame via Orthogonal Procrustes superposition (`solve_2d_procrustes`).
pub fn detect_and_align_series_mcs(
    ds: &DatasetAnalysis,
    member_indices: &[usize],
    target_scope: impl Into<String>,
    cfg: &McsConfig,
) -> Option<SeriesMcsAlignment> {
    let scope_str = target_scope.into();
    if member_indices.len() < 2 {
        return None;
    }

    let min_required = ((member_indices.len() as f32) * cfg.threshold_fraction.clamp(0.35, 1.0))
        .ceil()
        .max(2.0) as usize;

    // Order candidate seeds by potency descending so high-value leads anchor the consensus core
    let mut ordered_members = member_indices.to_vec();
    ordered_members.sort_by(|&a, &b| {
        ds.mols[b]
            .pact
            .partial_cmp(&ds.mols[a].pact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Harvest candidate MCS SMARTS patterns from pairwise McGregor comparisons among diverse series seeds
    let mut candidate_smarts: BTreeSet<String> = BTreeSet::new();
    let seed_limit = ordered_members.len().min(6);
    for i in 0..seed_limit {
        for j in (i + 1)..ordered_members.len().min(12) {
            if let Some(pmcs) =
                find_pairwise_mcs(&ds.mols[ordered_members[i]], &ds.mols[ordered_members[j]], cfg)
            {
                if !pmcs.mcs_smarts.is_empty() {
                    candidate_smarts.insert(pmcs.mcs_smarts);
                }
            }
        }
    }

    // Also include individual Murcko frameworks of the top seeds as clean fallback core candidates
    for &mi in ordered_members.iter().take(4) {
        let (_, _, _, fw_atoms) = crate::scaffold::murcko_signatures(&ds.mols[mi]);
        if fw_atoms.len() >= cfg.min_core_atoms {
            if let Some(sm) = scaffold_subgraph_to_smarts(&ds.mols[mi], &fw_atoms) {
                candidate_smarts.insert(sm);
            }
        }
    }

    // Score every candidate SMARTS across all `ordered_members` via our exact VF2 `match_smarts` engine!
    let mut best_choice: Option<(usize, usize, String, SmartsQuery, Vec<(usize, Vec<usize>)>)> =
        None;

    for sm in candidate_smarts {
        let Ok(query) = compile_smarts(&sm) else {
            continue;
        };
        let q_len = query.atoms.len();
        if q_len < cfg.min_core_atoms {
            continue;
        }
        let mut matched_list: Vec<(usize, Vec<usize>)> = Vec::new();
        for &mi in &ordered_members {
            let res = match_smarts(&ds.mols[mi], &query);
            if res.matched && !res.mappings.is_empty() {
                matched_list.push((mi, res.mappings[0].clone()));
            }
        }
        if matched_list.len() < 2 {
            continue;
        }

        // Prefer candidates meeting `min_required`, maximizing `q_len` first then `matched_list.len()`
        let meets = matched_list.len() >= min_required;
        let replace = match &best_choice {
            None => true,
            Some((best_qlen, best_mcnt, _, _, _)) => {
                let best_meets = *best_mcnt >= min_required;
                if meets != best_meets {
                    meets
                } else {
                    let score_new = matched_list.len() * matched_list.len() * q_len;
                    let score_best = (*best_mcnt) * (*best_mcnt) * (*best_qlen);
                    if score_new != score_best {
                        score_new > score_best
                    } else if matched_list.len() != *best_mcnt {
                        matched_list.len() > *best_mcnt
                    } else {
                        q_len > *best_qlen
                    }
                }
            }
        };
        if replace {
            best_choice = Some((q_len, matched_list.len(), sm, query, matched_list));
        }
    }

    let (mcs_atom_count, _, mcs_smarts, query, matched_raw) = best_choice?;
    let mcs_bond_count = query.bonds.len();

    // Reference template = first (`highest pAct`) matched compound, centered at `[0.0, 0.0]`
    let (ref_mol_idx, ref_core_map) = matched_raw[0].clone();
    let mut ref_mol_aligned = ds.mols[ref_mol_idx].clone();
    let mut ref_cx = 0.0f32;
    let mut ref_cy = 0.0f32;
    for &u in &ref_core_map {
        ref_cx += ref_mol_aligned.atoms[u].pos2d[0];
        ref_cy += ref_mol_aligned.atoms[u].pos2d[1];
    }
    ref_cx /= mcs_atom_count as f32;
    ref_cy /= mcs_atom_count as f32;
    for a in &mut ref_mol_aligned.atoms {
        a.pos2d[0] -= ref_cx;
        a.pos2d[1] -= ref_cy;
    }
    let ref_pts: Vec<[f32; 2]> = ref_core_map
        .iter()
        .map(|&u| ref_mol_aligned.atoms[u].pos2d)
        .collect();
    let ref_heavy = ref_mol_aligned
        .atoms
        .iter()
        .filter(|a| a.elem != Element::H)
        .count();

    // For every matched member, test ALL automorphic VF2 embeddings `res.mappings` against `ref_pts`
    // and choose the symmetry-equivalent mapping that minimizes 2D Procrustes core RMSD (and maximizes
    // substituent alignment consistency)!
    let mut raw_member_subst: Vec<Vec<String>> = Vec::with_capacity(matched_raw.len());
    let mut aligned_tuples: Vec<(usize, Molecule, Vec<usize>, f32, bool)> =
        Vec::with_capacity(matched_raw.len());

    for (pos_i, (mi, _)) in matched_raw.iter().enumerate() {
        let orig = &ds.mols[*mi];
        if pos_i == 0 {
            let subst = extract_port_substituents(&ref_mol_aligned, &ref_core_map);
            raw_member_subst.push(subst);
            aligned_tuples.push((*mi, ref_mol_aligned.clone(), ref_core_map.clone(), 0.0, false));
            continue;
        }
        let mres = match_smarts(orig, &query);
        let mut best_emb = mres.mappings[0].clone();
        let mut best_xform = ([[1.0f32, 0.0], [0.0, 1.0]], [0.0f32, 0.0], [0.0f32, 0.0], f32::INFINITY, false);

        for map_cand in &mres.mappings {
            let mut rigid_ref = Vec::new();
            let mut rigid_mob = Vec::new();
            for (idx, &u) in map_cand.iter().enumerate() {
                if idx >= ref_pts.len() {
                    break;
                }
                let is_rigid = orig.atoms[u].in_ring
                    || orig.adj[u]
                        .iter()
                        .filter(|&&(nbr, _)| orig.atoms[nbr].in_ring)
                        .count()
                        >= 2;
                if is_rigid {
                    rigid_ref.push(ref_pts[idx]);
                    rigid_mob.push(orig.atoms[u].pos2d);
                }
            }
            let xf = if rigid_ref.len() >= 4 {
                let (c, s, t, rigid_rmsd, n) = solve_2d_procrustes(&rigid_ref, &rigid_mob);
                (c, s, t, rigid_rmsd, n)
            } else {
                let mob_pts: Vec<[f32; 2]> = map_cand.iter().map(|&u| orig.atoms[u].pos2d).collect();
                solve_2d_procrustes(&ref_pts, &mob_pts)
            };
            if xf.3 < best_xform.3 - 1e-4 {
                best_xform = xf;
                best_emb = map_cand.clone();
            }
        }

        let mut al_mol = orig.clone();
        if best_xform.3 > 0.25 {
            // Try 180-deg single-bond rotamer flips across acyclic MCS bridge bonds (e.g. 4-anilino C-N)
            let core_set_cand: BTreeSet<usize> = best_emb.iter().copied().collect();
            for b in &orig.bonds {
                if b.in_ring
                    || b.aromatic
                    || b.order != crate::mol::BondOrder::Single
                    || !core_set_cand.contains(&b.a)
                    || !core_set_cand.contains(&b.b)
                {
                    continue;
                }
                let mut side_a = Vec::new();
                let mut visited = BTreeSet::new();
                let mut q = std::collections::VecDeque::new();
                visited.insert(b.a);
                q.push_back(b.a);
                let mut hits_b = false;
                while let Some(cur) = q.pop_front() {
                    if cur == b.b {
                        hits_b = true;
                        break;
                    }
                    side_a.push(cur);
                    for &(nbr, bi) in &orig.adj[cur] {
                        if bi != b.idx && visited.insert(nbr) {
                            q.push_back(nbr);
                        }
                    }
                }
                if hits_b || side_a.is_empty() {
                    continue;
                }
                let side_b_len = orig.atoms.len().saturating_sub(side_a.len());
                let rot_side: Vec<usize> = if side_a.len() <= side_b_len {
                    side_a
                } else {
                    (0..orig.atoms.len()).filter(|i| !visited.contains(i)).collect()
                };
                let p = orig.atoms[b.a].pos2d;
                let v = orig.atoms[b.b].pos2d;
                let dx = (v[0] - p[0]) as f64;
                let dy = (v[1] - p[1]) as f64;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 {
                    continue;
                }
                let ux = dx / len;
                let uy = dy / len;
                let mut trial_mol = orig.clone();
                for &w in &rot_side {
                    let rx = (trial_mol.atoms[w].pos2d[0] - p[0]) as f64;
                    let ry = (trial_mol.atoms[w].pos2d[1] - p[1]) as f64;
                    let proj = rx * ux + ry * uy;
                    trial_mol.atoms[w].pos2d[0] = (p[0] as f64 + 2.0 * proj * ux - rx) as f32;
                    trial_mol.atoms[w].pos2d[1] = (p[1] as f64 + 2.0 * proj * uy - ry) as f32;
                }
                let t_xf = {
                    let mut best_cand_xf = ([[1.0f32, 0.0], [0.0, 1.0]], [0.0f32, 0.0], [0.0f32, 0.0], f32::INFINITY, false);
                    let mut best_cand_emb = best_emb.clone();
                    for map_cand in &mres.mappings {
                        let mut t_ref = Vec::new();
                        let mut t_mob = Vec::new();
                        for (idx, &u) in map_cand.iter().enumerate() {
                            if idx >= ref_pts.len() {
                                break;
                            }
                            let is_rigid = trial_mol.atoms[u].in_ring
                                || trial_mol.adj[u]
                                    .iter()
                                    .filter(|&&(nbr, _)| trial_mol.atoms[nbr].in_ring)
                                    .count()
                                    >= 2;
                            if is_rigid {
                                t_ref.push(ref_pts[idx]);
                                t_mob.push(trial_mol.atoms[u].pos2d);
                            }
                        }
                        let xf = if t_ref.len() >= 4 {
                            solve_2d_procrustes(&t_ref, &t_mob)
                        } else {
                            let all_mob: Vec<[f32; 2]> = map_cand.iter().map(|&u| trial_mol.atoms[u].pos2d).collect();
                            solve_2d_procrustes(&ref_pts, &all_mob)
                        };
                        if xf.3 < best_cand_xf.3 - 1e-4 {
                            best_cand_xf = xf;
                            best_cand_emb = map_cand.clone();
                        }
                    }
                    (best_cand_xf, best_cand_emb)
                };
                if t_xf.0.3 < best_xform.3 - 1e-4 {
                    best_xform = t_xf.0;
                    best_emb = t_xf.1;
                    al_mol = trial_mol;
                }
            }
        }
        apply_2d_transform(&mut al_mol, best_xform.0, best_xform.1, best_xform.2);
        let subst = extract_port_substituents(&al_mol, &best_emb);
        raw_member_subst.push(subst);
        aligned_tuples.push((*mi, al_mol, best_emb, best_xform.3, best_xform.4));
    }

    // Identify variable attachment ports `qp in 0..mcs_atom_count` where at least one compound has `!= "-[H]"`
    let mut port_diversity: Vec<(usize, usize)> = (0..mcs_atom_count)
        .filter_map(|qp| {
            let distinct: BTreeSet<&str> = raw_member_subst
                .iter()
                .map(|v| v[qp].as_str())
                .collect();
            let any_non_h = distinct.iter().any(|&s| s != "-[H]");
            any_non_h.then_some((qp, distinct.len()))
        })
        .collect();
    port_diversity.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    port_diversity.truncate(4);

    let slot_labels: Vec<String> = (0..port_diversity.len())
        .map(|i| format!("R{}", i + 1))
        .collect();
    let slot_anchor_pos2d: Vec<[f32; 2]> = port_diversity
        .iter()
        .map(|&(qp, _)| ref_pts[qp])
        .collect();

    let mut members = Vec::with_capacity(aligned_tuples.len());
    let mut rmsd_sum = 0.0f32;
    for (idx_in_list, (mi, aligned_mol, core_atoms, core_rmsd_2d, reflected_2d)) in
        aligned_tuples.into_iter().enumerate()
    {
        rmsd_sum += core_rmsd_2d;
        let core_set: BTreeSet<usize> = core_atoms.iter().copied().collect();
        let core_bonds: Vec<usize> = aligned_mol
            .bonds
            .iter()
            .enumerate()
            .filter_map(|(bi, b)| (core_set.contains(&b.a) && core_set.contains(&b.b)).then_some(bi))
            .collect();
        let rgroup_atoms: Vec<usize> = (0..aligned_mol.atoms.len())
            .filter(|&u| aligned_mol.atoms[u].elem != Element::H && !core_set.contains(&u))
            .collect();
        let r_groups: Vec<String> = port_diversity
            .iter()
            .map(|&(qp, _)| raw_member_subst[idx_in_list][qp].clone())
            .collect();

        let mem_heavy = aligned_mol
            .atoms
            .iter()
            .filter(|a| a.elem != Element::H)
            .count();
        let denom = (ref_heavy + mem_heavy).saturating_sub(mcs_atom_count).max(1);
        let mcs_tanimoto_to_ref = (mcs_atom_count as f32) / (denom as f32);

        members.push(AlignedSeriesMember {
            mol_idx: mi,
            compound_id: aligned_mol.id.clone(),
            compound_name: aligned_mol.name.clone(),
            target: aligned_mol.target.clone(),
            pact: aligned_mol.pact,
            aligned_mol,
            core_atoms,
            core_bonds,
            rgroup_atoms,
            r_groups,
            core_rmsd_2d,
            mcs_tanimoto_to_ref,
            reflected_2d,
        });
    }

    let matched_member_count = members.len();
    let mean_core_rmsd_2d = rmsd_sum / (matched_member_count.max(1) as f32);
    let coverage_fraction = (matched_member_count as f32) / (member_indices.len() as f32);

    Some(SeriesMcsAlignment {
        target_scope: scope_str,
        total_input_molecules: member_indices.len(),
        matched_member_count,
        coverage_fraction,
        reference_mol_idx: ref_mol_idx,
        reference_name: ds.mols[ref_mol_idx].name.clone(),
        mcs_atom_count,
        mcs_bond_count,
        mcs_smarts,
        slot_labels,
        slot_anchor_pos2d,
        mean_core_rmsd_2d,
        members,
    })
}

/// Exports the 2D-aligned Series MCS R-group table & core superposition metrics as CSV.
pub fn export_series_mcs_csv(series: &SeriesMcsAlignment) -> String {
    let mut hdr = String::from("rank,id,name,target,pact,mcs_tanimoto_to_ref,core_rmsd_2d_ang,reflected_2d");
    for sl in &series.slot_labels {
        hdr.push(',');
        hdr.push_str(sl);
    }
    hdr.push_str(",mcs_smarts,smiles,source_url\n");

    let mut out = hdr;
    for (rk, m) in series.members.iter().enumerate() {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",{:.2},{:.3},{:.4},{}",
            rk + 1,
            m.compound_id,
            m.compound_name,
            m.target,
            m.pact,
            m.mcs_tanimoto_to_ref,
            m.core_rmsd_2d,
            m.reflected_2d
        ));
        for rg in &m.r_groups {
            out.push_str(&format!(",\"{}\"", rg));
        }
        out.push_str(&format!(
            ",\"{}\",{},\"{}\"\n",
            series.mcs_smarts, m.aligned_mol.smiles, m.aligned_mol.source_url
        ));
    }
    out
}
