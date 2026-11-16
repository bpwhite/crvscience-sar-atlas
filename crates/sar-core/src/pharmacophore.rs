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

//! 2D Topological Pharmacophore Feature Mapper, Canonical 3-Point Distance Triangles,
//! `CATS2` Correlation Vectors, and Scaffold-Hopping Alignment Similarity Engine.
//!
//! Primary Literature Grounding:
//! - Gobbi & Poppinger (1998), *Biotechnol. Bioeng.* 61(1), 47–54 (`https://pubmed.ncbi.nlm.nih.gov/10099495/`):
//!   2D topological pharmacophore feature typing & canonical 3-point distance-triangle fingerprints.
//! - Schneider, Neidhart, Giller & Schmid (1999), *Angew. Chem. Int. Ed.* 38, 2894–2896
//!   (`https://pubmed.ncbi.nlm.nih.gov/10540384/`): Chemically Advanced Template Search (`CATS`)
//!   topological pharmacophore correlation vectors for scaffold hopping.
//! - Reutlinger et al. (2013), *Mol. Inform.* 32(2), 133–138 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3743170/`):
//!   `CATS2` separation of aromatic (`AROM`) and lipophilic/hydrophobic (`HYD`) feature classes across
//!   `0..=9` topological bond lags.
//! - Bragina et al. / SwissSimilarity (2022), *Int. J. Mol. Sci.* 23, 811
//!   (`https://pdfs.semanticscholar.org/67ea/4a92a16da11d44125e7ac25d291de87dd5d0.pdf`):
//!   Benchmarking 2D Gobbi pharmacophore triangle fingerprints against `ECFP4` for scaffold hopping.

use crate::fingerprint::{compute_ecfp, Ecfp2048};
use crate::mol::{BondOrder, Element, Hybridization, Molecule};
use crate::scaffold::murcko_signatures;
use crate::DatasetAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Number of `u64` words in the 1024-bit Gobbi-Poppinger 2D pharmacophore triangle fingerprint.
pub const PHARM_FP_WORDS: usize = 16;

/// Number of topological bond-lag bins (`0..=9`) in the `CATS2` correlation vector.
pub const CATS2_MAX_LAG: usize = 10;

/// Total dimensionality of `CATS2`: `21` unordered feature-type pairs x `10` bond lags = `210`.
pub const CATS2_DIM: usize = 21 * CATS2_MAX_LAG;

/// Canonical medicinal-chemistry pharmacophore feature classes (`Gobbi & Poppinger 1998` + `CATS2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PharmFeatureKind {
    HBondDonor = 0,
    HBondAcceptor = 1,
    Aromatic = 2,
    Hydrophobic = 3,
    PosIonizable = 4,
    NegIonizable = 5,
}

impl PharmFeatureKind {
    pub const ALL: [Self; 6] = [
        Self::HBondDonor,
        Self::HBondAcceptor,
        Self::Aromatic,
        Self::Hydrophobic,
        Self::PosIonizable,
        Self::NegIonizable,
    ];

    #[inline]
    pub fn type_index(self) -> usize {
        self as usize
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::HBondDonor => "HBD",
            Self::HBondAcceptor => "HBA",
            Self::Aromatic => "AROM",
            Self::Hydrophobic => "HYD",
            Self::PosIonizable => "POS",
            Self::NegIonizable => "NEG",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::HBondDonor => "H-Bond Donor",
            Self::HBondAcceptor => "H-Bond Acceptor",
            Self::Aromatic => "Aromatic Ring",
            Self::Hydrophobic => "Hydrophobic",
            Self::PosIonizable => "Pos. Ionizable",
            Self::NegIonizable => "Neg. Ionizable",
        }
    }

    /// High-contrast dark-theme RGB color `[u8; 3]` for 2D pharmacophore halo & triangle overlays.
    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HBondDonor => [56, 189, 248],    // Sky Blue
            Self::HBondAcceptor => [248, 113, 113], // Coral Red
            Self::Aromatic => [251, 191, 36],      // Amber Gold
            Self::Hydrophobic => [52, 211, 153],   // Emerald Green
            Self::PosIonizable => [192, 132, 252], // Violet
            Self::NegIonizable => [34, 211, 238],  // Cyan
        }
    }

    /// Relative pharmacophoric specificity weight used to rank consensus 3-point triangles
    /// (polar/ionizable recognition anchors carry higher visual priority than abundant alkyls).
    pub fn specificity_weight(self) -> f32 {
        match self {
            Self::PosIonizable | Self::NegIonizable => 1.65,
            Self::HBondDonor => 1.40,
            Self::HBondAcceptor => 1.15,
            Self::Aromatic => 1.05,
            Self::Hydrophobic => 0.95,
        }
    }
}

/// Single perceived 2D pharmacophore feature locus (atom or multi-atom ring/group centroid).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PharmacophorePoint {
    pub idx: usize,
    pub kind: PharmFeatureKind,
    /// Underlying heavy-atom indices in `Molecule::atoms` defining this pharmacophore locus.
    pub atoms: Vec<usize>,
    /// 2D skeletal world coordinates `[x, y]` (exact atom position or ring/cluster centroid).
    pub pos2d: [f32; 2],
    /// Concise chemical label (e.g. `"HBD (N8)"`, `"AROM-6"`, `"HYD (CF3)"`, `"POS (R3N)"`).
    pub label: String,
}

/// Canonicalized 3-point topological pharmacophore distance triangle (`Gobbi & Poppinger 1998`).
///
/// Vertex order `[0, 1, 2]` is canonicalized over all 6 vertex permutations so any two triangles
/// sharing the same `signature_hash` have vertex-by-vertex aligned `feat_indices`, `kinds`,
/// `bond_dists` (`[d01, d12, d20]`), and `dist_bins` (`[b01, b12, b20]`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PharmTriangle {
    /// Indices into `PharmacophoreMap::features` in canonicalized vertex order `[v0, v1, v2]`.
    pub feat_indices: [usize; 3],
    pub kinds: [PharmFeatureKind; 3],
    /// Exact topological bond distances `[d(v0,v1), d(v1,v2), d(v2,v0)]`.
    pub bond_dists: [u8; 3],
    /// Gobbi 8-level distance bins `[b(v0,v1), b(v1,v2), b(v2,v0)]` (`0..=7`).
    pub dist_bins: [u8; 3],
    /// Exact 64-bit canonical triangle signature hash.
    pub signature_hash: u64,
    /// Sum of topological bond edges `d01 + d12 + d20`.
    pub perimeter_bonds: u16,
    /// Human-readable canonical triangle code, e.g. `"HBD-AROM-HBA [2,4,5]"`.
    pub label: String,
}

/// 1024-bit folded Gobbi-Poppinger 2D pharmacophore triangle bitvector (`[u64; 16]`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pharm2DFingerprint {
    pub words: [u64; PHARM_FP_WORDS],
    pub on_bits: u32,
}

impl Default for Pharm2DFingerprint {
    fn default() -> Self {
        Self {
            words: [0u64; PHARM_FP_WORDS],
            on_bits: 0,
        }
    }
}

impl Pharm2DFingerprint {
    #[inline]
    pub fn set_bit(&mut self, bit: usize) {
        let idx = bit & 1023;
        let w = idx >> 6;
        let mask = 1u64 << (idx & 63);
        if (self.words[w] & mask) == 0 {
            self.words[w] |= mask;
            self.on_bits += 1;
        }
    }

    /// Hardware `POPCNT` Tanimoto (Jaccard) similarity in `[0.0, 1.0]`.
    #[inline]
    pub fn tanimoto(&self, other: &Self) -> f32 {
        if self.on_bits == 0 && other.on_bits == 0 {
            return 0.0;
        }
        let mut inter = 0u32;
        for i in 0..PHARM_FP_WORDS {
            inter += (self.words[i] & other.words[i]).count_ones();
        }
        let union = self.on_bits + other.on_bits - inter;
        if union == 0 {
            0.0
        } else {
            (inter as f32) / (union as f32)
        }
    }
}

/// Complete 2D topological pharmacophore map, triangle ensemble, `Pharm2DFingerprint`,
/// and `CATS2` 210-bin correlation vector for a single molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PharmacophoreMap {
    pub features: Vec<PharmacophorePoint>,
    pub counts_by_kind: [usize; 6],
    /// Pairwise topological bond distance matrix (`features.len() x features.len()`).
    pub feature_dists: Vec<Vec<u8>>,
    pub triangles: Vec<PharmTriangle>,
    pub fp: Pharm2DFingerprint,
    /// L2-normalized 210-element `CATS2` topological correlation vector (`21` pairs x `10` lags).
    pub cats2_vector: Vec<f32>,
    /// Canonical Bemis-Murcko framework key for scaffold-hopping detection.
    pub murcko_framework_key: String,
    pub murcko_label: String,
}

/// Matched 3-point consensus pharmacophore triangle linking a query triplet to a target triplet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchedPharmTriangle {
    pub signature_hash: u64,
    pub label: String,
    pub query_tri: PharmTriangle,
    pub target_tri: PharmTriangle,
    pub informativeness: f32,
}

/// Pairwise 2D pharmacophore alignment & scaffold-hopping comparison result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PharmacophoreAlignment {
    /// Composite 2D pharmacophore similarity (`0.65 * triangle_tanimoto + 0.35 * cats_cosine`).
    pub pharm_similarity: f32,
    /// Gobbi-Poppinger 1024-bit 3-point triangle Jaccard/Tanimoto similarity.
    pub triangle_tanimoto: f32,
    /// `CATS2` 210-dimensional cosine similarity.
    pub cats_cosine: f32,
    /// Circular radial `ECFP4` Tanimoto similarity (structural baseline).
    pub ecfp4_tanimoto: f32,
    pub same_murcko_scaffold: bool,
    /// `true` when Murcko frameworks differ, `pharm_similarity >= 0.40`, and `ecfp4_tanimoto <= 0.45`.
    pub is_scaffold_hop: bool,
    /// Positive enrichment `max(0, pharm_similarity - ecfp4_tanimoto)` across distinct scaffolds.
    pub scaffold_hop_enrichment: f32,
    /// Count of distinct canonical 3-point pharmacophore triangle signatures shared by both molecules.
    pub shared_triangle_count: usize,
    /// Top ranked consensus 3-point pharmacophore triangles for interactive 2D overlay.
    pub matched_triangles: Vec<MatchedPharmTriangle>,
}

/// Ranked library hit from screening a query molecule against a `DatasetAnalysis`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PharmacophoreHit {
    pub mol_idx: usize,
    pub alignment: PharmacophoreAlignment,
}

/// Gobbi & Poppinger (1998) 8 topological bond-distance bins:
/// `1`, `2`, `3`, `4`, `5`, `6`, `7-8`, and `9+` bonds (`0..=7`).
#[inline]
pub fn gobbi_distance_bin(bond_dist: u8) -> u8 {
    match bond_dist {
        0 | 1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        5 => 4,
        6 => 5,
        7 | 8 => 6,
        _ => 7,
    }
}

pub fn gobbi_bin_label(bin: u8) -> &'static str {
    match bin {
        0 => "1",
        1 => "2",
        2 => "3",
        3 => "4",
        4 => "5",
        5 => "6",
        6 => "7-8",
        _ => "9+",
    }
}

/// Computes the All-Pairs Shortest-Path (`APSP`) topological bond distance matrix across all
/// atoms in `mol` via unweighted Breadth-First Search (`u8::MAX` if disconnected).
pub fn compute_topological_distances(mol: &Molecule) -> Vec<Vec<u8>> {
    let n = mol.atoms.len();
    let mut dist = vec![vec![u8::MAX; n]; n];
    let mut q = VecDeque::with_capacity(n);

    for src in 0..n {
        if mol.atoms[src].elem == Element::H {
            continue;
        }
        dist[src][src] = 0;
        q.clear();
        q.push_back(src);
        while let Some(u) = q.pop_front() {
            let next_d = dist[src][u].saturating_add(1);
            if next_d == u8::MAX {
                continue;
            }
            for &(v, _) in &mol.adj[u] {
                if mol.atoms[v].elem == Element::H {
                    continue;
                }
                if dist[src][v] == u8::MAX {
                    dist[src][v] = next_d;
                    q.push_back(v);
                }
            }
        }
    }
    dist
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn unordered_pair_index(a: usize, b: usize) -> usize {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    // Map (lo, hi) with 0 <= lo <= hi < 6 into 0..21:
    // row_offset(lo) = lo * (13 - lo) / 2 + (hi - lo)
    (lo * (13 - lo)) / 2 + (hi - lo)
}

fn is_carbonyl_or_sulfonyl_like_c_or_s(mol: &Molecule, u: usize) -> bool {
    let el = mol.atoms[u].elem;
    if !matches!(el, Element::C | Element::S | Element::P) {
        return false;
    }
    mol.adj[u].iter().any(|&(v, bi)| {
        let b = &mol.bonds[bi];
        matches!(mol.atoms[v].elem, Element::O | Element::S | Element::N)
            && (b.order == BondOrder::Double || b.kekule_order == 2)
    })
}

fn is_nitro_nitrogen(mol: &Molecule, u: usize) -> bool {
    if mol.atoms[u].elem != Element::N {
        return false;
    }
    let ox_count = mol.adj[u]
        .iter()
        .filter(|&&(v, bi)| {
            mol.atoms[v].elem == Element::O
                && (mol.bonds[bi].order == BondOrder::Double
                    || mol.atoms[v].charge < 0
                    || mol.bonds[bi].kekule_order == 2)
        })
        .count();
    ox_count >= 2
}

fn centroid_of_atoms(mol: &Molecule, atoms: &[usize]) -> [f32; 2] {
    if atoms.is_empty() {
        return [0.0, 0.0];
    }
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for &u in atoms {
        sx += mol.atoms[u].pos2d[0];
        sy += mol.atoms[u].pos2d[1];
    }
    let inv = 1.0 / (atoms.len() as f32);
    [sx * inv, sy * inv]
}

/// Topological bond separation between two `PharmacophorePoint` loci:
/// - Identical atom anchor `[u]` vs `[u]` (e.g. `-OH` is both `HBD` and `HBA`) -> `0` bonds.
/// - Distinct multi-atom aromatic rings sharing a fused bridgehead bond (`[6,6]` quinazoline) -> `2` bonds.
/// - Ring-atom membership overlap (`[n]` pyridine `HBA` inside its `AROM-6` ring) -> `1` bond.
/// - Disjoint feature loci -> minimum heavy-atom shortest-path distance `min_{u in A, v in B} apsp[u][v]`,
///   plus `+1` per aromatic ring centroid so ring-center distances reflect true ring-to-substituent spacing.
fn feature_pair_bond_distance(
    apsp: &[Vec<u8>],
    fa: &PharmacophorePoint,
    fb: &PharmacophorePoint,
) -> u8 {
    if fa.atoms == fb.atoms {
        return 0;
    }
    let mut min_d = u8::MAX;
    let mut shared_atoms = 0usize;
    for &u in &fa.atoms {
        for &v in &fb.atoms {
            if u == v {
                shared_atoms += 1;
            }
            let d = apsp[u][v];
            if d < min_d {
                min_d = d;
            }
        }
    }
    if min_d == u8::MAX {
        return u8::MAX;
    }
    if fa.kind == PharmFeatureKind::Aromatic && fb.kind == PharmFeatureKind::Aromatic {
        if shared_atoms >= 2 {
            return 2; // Fused bicyclic aromatic rings (e.g. quinazoline, indole)
        }
        return min_d.saturating_add(2);
    }
    if fa.kind == PharmFeatureKind::Aromatic || fb.kind == PharmFeatureKind::Aromatic {
        if shared_atoms >= 1 {
            return 1; // Heteroatom feature embedded directly in the aromatic ring perimeter
        }
        return min_d.saturating_add(1);
    }
    if min_d == 0 {
        1
    } else {
        min_d
    }
}

/// Perceives all 6 canonical pharmacophore feature classes, constructs canonical 3-point
/// Gobbi-Poppinger distance triangles + 1024-bit fingerprint, and computes the 210-dim `CATS2`
/// topological correlation vector for `mol`.
pub fn extract_pharmacophore_map(mol: &Molecule) -> PharmacophoreMap {
    let n = mol.atoms.len();
    let apsp = compute_topological_distances(mol);
    let (murcko_framework_key, _, murcko_label, _) = murcko_signatures(mol);

    let mut features: Vec<PharmacophorePoint> = Vec::new();
    let mut counts_by_kind = [0usize; 6];
    let mut push_feat = |kind: PharmFeatureKind, atoms: Vec<usize>, label: String| {
        let pos2d = centroid_of_atoms(mol, &atoms);
        let idx = features.len();
        counts_by_kind[kind.type_index()] += 1;
        features.push(PharmacophorePoint {
            idx,
            kind,
            atoms,
            pos2d,
            label,
        });
    };

    // Track atoms already claimed by specialized acidic/CF3/aliphatic-ring motifs
    let mut neg_claimed = vec![false; n];
    let mut cf3_claimed = vec![false; n];

    // -------------------------------------------------------------------------
    // 1. NegIonizable (`NEG`): Carboxylic/sulfonic/phosphonic acids, tetrazoles, acylsulfonamides
    // -------------------------------------------------------------------------
    for r in &mol.rings {
        if r.len() == 5 && r.iter().all(|&u| mol.atoms[u].aromatic) {
            let n_cnt = r.iter().filter(|&&u| mol.atoms[u].elem == Element::N).count();
            let c_cnt = r.iter().filter(|&&u| mol.atoms[u].elem == Element::C).count();
            let has_nh_or_anion = r
                .iter()
                .any(|&u| mol.atoms[u].elem == Element::N && (mol.atoms[u].total_h() > 0 || mol.atoms[u].charge < 0));
            if n_cnt == 4 && c_cnt == 1 && has_nh_or_anion {
                for &u in r {
                    neg_claimed[u] = true;
                }
                push_feat(
                    PharmFeatureKind::NegIonizable,
                    r.clone(),
                    "NEG (Tetrazole)".to_string(),
                );
            }
        }
    }

    for u in 0..n {
        if neg_claimed[u] {
            continue;
        }
        let a = &mol.atoms[u];
        // Carboxylic / sulfonic / phosphonic acid center: C/S/P with (=O) and (-OH or -[O-])
        if matches!(a.elem, Element::C | Element::S | Element::P) {
            let mut oxo_nbrs = Vec::new();
            let mut acidic_oh_nbrs = Vec::new();
            for &(v, bi) in &mol.adj[u] {
                if mol.atoms[v].elem == Element::O {
                    let b = &mol.bonds[bi];
                    if b.order == BondOrder::Double || b.kekule_order == 2 {
                        oxo_nbrs.push(v);
                    } else if mol.atoms[v].total_h() > 0 || mol.atoms[v].charge < 0 {
                        acidic_oh_nbrs.push(v);
                    }
                }
            }
            if !oxo_nbrs.is_empty() && !acidic_oh_nbrs.is_empty() {
                let mut grp = vec![u];
                grp.extend(&oxo_nbrs);
                grp.extend(&acidic_oh_nbrs);
                for &g in &grp {
                    neg_claimed[g] = true;
                }
                let kind_lbl = match a.elem {
                    Element::C => "NEG (-COOH)",
                    Element::S => "NEG (-SO3H)",
                    _ => "NEG (-PO3H)",
                };
                push_feat(PharmFeatureKind::NegIonizable, grp, kind_lbl.to_string());
                continue;
            }
        }
        // Acylsulfonamide acidic NH: C(=O)-NH-S(=O)(=O)
        if a.elem == Element::N && (a.total_h() > 0 || a.charge < 0) {
            let mut has_acyl = false;
            let mut has_sulfonyl = false;
            for &(v, _) in &mol.adj[u] {
                if mol.atoms[v].elem == Element::C && is_carbonyl_or_sulfonyl_like_c_or_s(mol, v) {
                    has_acyl = true;
                } else if mol.atoms[v].elem == Element::S
                    && is_carbonyl_or_sulfonyl_like_c_or_s(mol, v)
                {
                    has_sulfonyl = true;
                }
            }
            if has_acyl && has_sulfonyl {
                neg_claimed[u] = true;
                push_feat(
                    PharmFeatureKind::NegIonizable,
                    vec![u],
                    format!("NEG (AcylSulf N{})", u + 1),
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. PosIonizable (`POS`): Physiological basic amines, amidines, guanidines, non-nitro cations
    // -------------------------------------------------------------------------
    let mut pos_claimed = vec![false; n];
    // First detect amidines / guanidines: sp2 C bonded to `=NH` and `-NH2`/`-NHR` (not carbonyl/amide)
    for u in 0..n {
        if mol.atoms[u].elem == Element::C && !mol.atoms[u].aromatic {
            let has_oxo_or_thio = mol.adj[u].iter().any(|&(v, bi)| {
                matches!(mol.atoms[v].elem, Element::O | Element::S)
                    && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
            });
            if has_oxo_or_thio {
                continue;
            }
            let mut imine_n = Vec::new();
            let mut amino_n = Vec::new();
            for &(v, bi) in &mol.adj[u] {
                if mol.atoms[v].elem == Element::N && !mol.atoms[v].aromatic {
                    if mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2 {
                        imine_n.push(v);
                    } else {
                        amino_n.push(v);
                    }
                }
            }
            if !imine_n.is_empty() && !amino_n.is_empty() {
                let mut grp = vec![u];
                grp.extend(&imine_n);
                grp.extend(&amino_n);
                for &g in &grp {
                    pos_claimed[g] = true;
                }
                let lbl = if amino_n.len() >= 2 {
                    "POS (Guanidine)"
                } else {
                    "POS (Amidine)"
                };
                push_feat(PharmFeatureKind::PosIonizable, grp, lbl.to_string());
            }
        }
    }

    for u in 0..n {
        if pos_claimed[u] || neg_claimed[u] {
            continue;
        }
        let a = &mol.atoms[u];
        if a.elem == Element::N {
            if is_nitro_nitrogen(mol, u) {
                continue;
            }
            if a.charge > 0 {
                pos_claimed[u] = true;
                push_feat(
                    PharmFeatureKind::PosIonizable,
                    vec![u],
                    format!("POS (N+{})", u + 1),
                );
                continue;
            }
            // Physiological basic aliphatic amine (pKa ~ 8-10.5): non-aromatic N with single bonds
            // whose heavy neighbors are exclusively sp3 carbons (no adjacent Ar, C=O, S=O, hetero)
            if !a.aromatic && a.charge == 0 && !mol.adj[u].is_empty() {
                let all_single_sp3_c = mol.adj[u].iter().all(|&(v, bi)| {
                    if mol.atoms[v].elem == Element::H {
                        return true;
                    }
                    mol.atoms[v].elem == Element::C
                        && !mol.atoms[v].aromatic
                        && mol.atoms[v].hyb == Hybridization::Sp3
                        && mol.bonds[bi].order == BondOrder::Single
                        && !is_carbonyl_or_sulfonyl_like_c_or_s(mol, v)
                });
                if all_single_sp3_c {
                    pos_claimed[u] = true;
                    push_feat(
                        PharmFeatureKind::PosIonizable,
                        vec![u],
                        format!("POS (Basic N{})", u + 1),
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. HBondDonor (`HBD`) & HBondAcceptor (`HBA`)
    // -------------------------------------------------------------------------
    for u in 0..n {
        let a = &mol.atoms[u];
        if !matches!(a.elem, Element::N | Element::O | Element::S) {
            continue;
        }
        // HBD: N, O, or S bearing >= 1 hydrogen and non-anionic
        if a.total_h() > 0 && a.charge >= 0 && !is_nitro_nitrogen(mol, u) {
            push_feat(
                PharmFeatureKind::HBondDonor,
                vec![u],
                format!("HBD ({}{})", a.elem.symbol(), u + 1),
            );
        }

        // HBA:
        if a.elem == Element::O && a.charge <= 0 {
            // Exclude aromatic furan oxygen unless isoxazole/oxazole (adjacent aromatic N makes ring polar,
            // though standard Gobbi treats non-aromatic O + carbonyl/ether/hydroxyl/sulfonyl O as HBA;
            // include non-aromatic O plus aromatic O adjacent to N)
            let is_arom_furan_only = a.aromatic
                && !mol.adj[u]
                    .iter()
                    .any(|&(v, _)| mol.atoms[v].elem == Element::N);
            if !is_arom_furan_only {
                push_feat(
                    PharmFeatureKind::HBondAcceptor,
                    vec![u],
                    format!("HBA (O{})", u + 1),
                );
            }
        } else if a.elem == Element::N && a.charge <= 0 && !is_nitro_nitrogen(mol, u) {
            if a.aromatic {
                // Pyridine / pyrimidine / quinazoline / pyrazole / isoxazole `sp2` lone-pair N:
                // aromatic N with 0 attached H and 2 heavy ring neighbors
                if a.total_h() == 0 && mol.heavy_degree(u) == 2 {
                    push_feat(
                        PharmFeatureKind::HBondAcceptor,
                        vec![u],
                        format!("HBA (n{})", u + 1),
                    );
                }
            } else {
                // Exclude amide / sulfonamide / aniline-amide delocalized N bonded to C=O, S=O, P=O
                let conjugated_ewg = mol
                    .adj[u]
                    .iter()
                    .any(|&(v, _)| is_carbonyl_or_sulfonyl_like_c_or_s(mol, v));
                // Also exclude diaryl / aryl secondary NH whose lone pair is delocalized into Ar
                let bonded_to_arom = mol.adj[u].iter().any(|&(v, _)| mol.atoms[v].aromatic);
                if !conjugated_ewg && (!bonded_to_arom || a.total_h() == 0) {
                    push_feat(
                        PharmFeatureKind::HBondAcceptor,
                        vec![u],
                        format!("HBA (N{})", u + 1),
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. Aromatic (`AROM`): Perceived 5- and 6-membered Hückel SSSR ring centroids
    // -------------------------------------------------------------------------
    for r in &mol.rings {
        if matches!(r.len(), 5 | 6) && r.iter().all(|&u| mol.atoms[u].aromatic) {
            push_feat(
                PharmFeatureKind::Aromatic,
                r.clone(),
                format!("AROM-{}", r.len()),
            );
        }
    }

    // -------------------------------------------------------------------------
    // 5. Hydrophobic (`HYD`): CF3 groups, aryl/alkyl Cl/Br/I, thioethers, aliphatic rings & lipophilic C islands
    // -------------------------------------------------------------------------
    // 5a. Trifluoromethyl / difluoromethyl (`-CF3` / `-CF2-`) lipophilic groups
    for u in 0..n {
        if mol.atoms[u].elem == Element::C && !mol.atoms[u].aromatic {
            let f_nbrs: Vec<usize> = mol.adj[u]
                .iter()
                .filter_map(|&(v, _)| (mol.atoms[v].elem == Element::F).then_some(v))
                .collect();
            if f_nbrs.len() >= 2 {
                cf3_claimed[u] = true;
                for &fv in &f_nbrs {
                    cf3_claimed[fv] = true;
                }
                let lbl = if f_nbrs.len() >= 3 {
                    "HYD (CF3)"
                } else {
                    "HYD (CF2)"
                };
                push_feat(PharmFeatureKind::Hydrophobic, vec![u], lbl.to_string());
            }
        }
    }

    // 5b. Lipophilic halogens (`Cl`, `Br`, `I`) and non-polar thioether `S`
    for u in 0..n {
        let a = &mol.atoms[u];
        if matches!(a.elem, Element::Cl | Element::Br | Element::I) && a.charge == 0 {
            push_feat(
                PharmFeatureKind::Hydrophobic,
                vec![u],
                format!("HYD ({}{})", a.elem.symbol(), u + 1),
            );
        } else if a.elem == Element::S && !a.aromatic && a.total_h() == 0 && a.charge == 0 {
            let polar_nbr = mol
                .adj[u]
                .iter()
                .any(|&(v, _)| matches!(mol.atoms[v].elem, Element::O | Element::N));
            if !polar_nbr {
                push_feat(
                    PharmFeatureKind::Hydrophobic,
                    vec![u],
                    format!("HYD (S{})", u + 1),
                );
            }
        }
    }

    // 5c. Non-polar aliphatic carbon islands (e.g. `-C#CH` ethynyl in Erlotinib, `p-CH3` tolyl in
    //     Celecoxib, `t-Bu`, `i-Pr`, or non-polar cycloalkyl rings)
    let mut is_lipophilic_c = vec![false; n];
    for u in 0..n {
        let a = &mol.atoms[u];
        if a.elem == Element::C && !a.aromatic && a.charge == 0 && !cf3_claimed[u] && !neg_claimed[u]
        {
            let bonded_to_hetero = mol.adj[u].iter().any(|&(v, _)| {
                !matches!(mol.atoms[v].elem, Element::C | Element::H)
            });
            if !bonded_to_hetero {
                // Exclude bare internal `-CH2-` linker carbons that sit strictly between two carbons
                // where one is polar-substituted unless terminal (`heavy_deg <= 1`), branched (`>= 3`),
                // unsaturated (`sp`/`sp2` alkynyl/alkenyl), or in an all-carbon ring
                let hdeg = mol.heavy_degree(u);
                let all_c_ring = a.in_ring
                    && mol.rings.iter().any(|r| {
                        r.contains(&u)
                            && r.iter()
                                .all(|&rv| mol.atoms[rv].elem == Element::C && !mol.atoms[rv].aromatic)
                    });
                if hdeg <= 1 || hdeg >= 3 || a.hyb != Hybridization::Sp3 || all_c_ring {
                    is_lipophilic_c[u] = true;
                }
            }
        }
    }

    let mut visited_c = vec![false; n];
    for u in 0..n {
        if is_lipophilic_c[u] && !visited_c[u] {
            let mut cluster = Vec::new();
            let mut dq = VecDeque::new();
            visited_c[u] = true;
            dq.push_back(u);
            while let Some(cur) = dq.pop_front() {
                cluster.push(cur);
                if cluster.len() >= 5 {
                    break;
                }
                for &(v, _) in &mol.adj[cur] {
                    if is_lipophilic_c[v] && !visited_c[v] {
                        visited_c[v] = true;
                        dq.push_back(v);
                    }
                }
            }
            let lbl = if cluster.len() == 1 {
                format!("HYD (C{})", cluster[0] + 1)
            } else {
                format!("HYD (C{}grp)", cluster.len())
            };
            push_feat(PharmFeatureKind::Hydrophobic, cluster, lbl);
        }
    }

    // -------------------------------------------------------------------------
    // 6. Pairwise Feature Distance Matrix + `CATS2` 210-Dim Correlation Vector
    // -------------------------------------------------------------------------
    let nf = features.len();
    let mut feature_dists = vec![vec![0u8; nf]; nf];
    for i in 0..nf {
        for j in (i + 1)..nf {
            let d = feature_pair_bond_distance(&apsp, &features[i], &features[j]);
            feature_dists[i][j] = d;
            feature_dists[j][i] = d;
        }
    }

    let mut raw_cats = vec![0.0f32; CATS2_DIM];
    for i in 0..nf {
        for j in (i + 1)..nf {
            let d = feature_dists[i][j];
            if d == u8::MAX {
                continue;
            }
            let lag = (d as usize).min(CATS2_MAX_LAG - 1);
            let pair_idx = unordered_pair_index(
                features[i].kind.type_index(),
                features[j].kind.type_index(),
            );
            let base = pair_idx * CATS2_MAX_LAG;
            raw_cats[base + lag] += 1.0;
            // Fuzzy ±1 topological bond-lag smoothing bridges 5-membered ring scaffold hops
            if lag > 0 {
                raw_cats[base + lag - 1] += 0.50;
            }
            if lag + 1 < CATS2_MAX_LAG {
                raw_cats[base + lag + 1] += 0.50;
            }
        }
    }
    // Type-frequency scaling per Schneider et al. (1999) / Reutlinger et al. (2013) `CATS2`
    for t1 in 0..6 {
        for t2 in t1..6 {
            let denom = (counts_by_kind[t1] + counts_by_kind[t2]) as f32;
            if denom > 0.0 {
                let p_idx = unordered_pair_index(t1, t2);
                for lag in 0..CATS2_MAX_LAG {
                    raw_cats[p_idx * CATS2_MAX_LAG + lag] /= denom;
                }
            }
        }
    }
    let norm_sq: f32 = raw_cats.iter().map(|v| v * v).sum();
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for v in &mut raw_cats {
            *v *= inv_norm;
        }
    }

    // -------------------------------------------------------------------------
    // 7. Gobbi-Poppinger Canonical 3-Point Distance Triangles & 1024-Bit Fingerprint
    // -------------------------------------------------------------------------
    let mut fp = Pharm2DFingerprint::default();

    // Encode 2-point pharmacophore pairs into bits `0..192` as base topological support
    for i in 0..nf {
        for j in (i + 1)..nf {
            let d = feature_dists[i][j];
            if d == 0 || d == u8::MAX {
                continue;
            }
            let (ka, kb) = if features[i].kind <= features[j].kind {
                (features[i].kind as u64, features[j].kind as u64)
            } else {
                (features[j].kind as u64, features[i].kind as u64)
            };
            let bin = gobbi_distance_bin(d) as u64;
            let h2 = splitmix64(0x2D02_0000 | (ka << 12) | (kb << 6) | bin);
            fp.set_bit((h2 as usize) % 192);
            // Soft +/-1 topological distance bin support to smooth 5- vs 6-membered bioisostere shifts
            if d > 1 {
                let b_lo = gobbi_distance_bin(d - 1) as u64;
                if b_lo != bin {
                    let h_lo = splitmix64(0x2D02_0000 | (ka << 12) | (kb << 6) | b_lo);
                    fp.set_bit((h_lo as usize) % 192);
                }
            }
            let b_hi = gobbi_distance_bin(d + 1) as u64;
            if b_hi != bin {
                let h_hi = splitmix64(0x2D02_0000 | (ka << 12) | (kb << 6) | b_hi);
                fp.set_bit((h_hi as usize) % 192);
            }
        }
    }

    let mut triangles: Vec<PharmTriangle> = Vec::new();
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    for i in 0..nf {
        for j in (i + 1)..nf {
            let d_ij = feature_dists[i][j];
            if d_ij == 0 || d_ij == u8::MAX {
                continue;
            }
            for k in (j + 1)..nf {
                let d_jk = feature_dists[j][k];
                let d_ki = feature_dists[k][i];
                if d_jk == 0 || d_jk == u8::MAX || d_ki == 0 || d_ki == u8::MAX {
                    continue;
                }

                let raw_idx = [i, j, k];
                let raw_k = [features[i].kind, features[j].kind, features[k].kind];
                let dist_mat = [[0u8, d_ij, d_ki], [d_ij, 0u8, d_jk], [d_ki, d_jk, 0u8]];

                let mut best_tuple: Option<([u8; 6], [usize; 3], [PharmFeatureKind; 3], [u8; 3])> =
                    None;
                for perm in &PERMS {
                    let p0 = perm[0];
                    let p1 = perm[1];
                    let p2 = perm[2];
                    let e01 = dist_mat[p0][p1];
                    let e12 = dist_mat[p1][p2];
                    let e20 = dist_mat[p2][p0];
                    let key = [
                        raw_k[p0] as u8,
                        raw_k[p1] as u8,
                        raw_k[p2] as u8,
                        gobbi_distance_bin(e01),
                        gobbi_distance_bin(e12),
                        gobbi_distance_bin(e20),
                    ];
                    let replace = match &best_tuple {
                        None => true,
                        Some((cur_key, _, _, _)) => key < *cur_key,
                    };
                    if replace {
                        best_tuple = Some((
                            key,
                            [raw_idx[p0], raw_idx[p1], raw_idx[p2]],
                            [raw_k[p0], raw_k[p1], raw_k[p2]],
                            [e01, e12, e20],
                        ));
                    }
                }

                if let Some((key, feat_indices, kinds, bond_dists)) = best_tuple {
                    let dist_bins = [key[3], key[4], key[5]];
                    let packed = ((key[0] as u64) << 40)
                        | ((key[1] as u64) << 32)
                        | ((key[2] as u64) << 24)
                        | ((key[3] as u64) << 16)
                        | ((key[4] as u64) << 8)
                        | (key[5] as u64);
                    let signature_hash = splitmix64(0x3D7A_C0DE_0000_0000 ^ packed);

                    // Fold into bits `192..1024` (832 triangle-dedicated bits, 2 bits per signature)
                    let bit_a = 192 + ((signature_hash as usize) % 832);
                    let bit_b = 192 + (((signature_hash >> 29) as usize) % 832);
                    fp.set_bit(bit_a);
                    fp.set_bit(bit_b);

                    let perimeter_bonds =
                        (bond_dists[0] as u16) + (bond_dists[1] as u16) + (bond_dists[2] as u16);
                    let label = format!(
                        "{}-{}-{} [{},{},{}]",
                        kinds[0].code(),
                        kinds[1].code(),
                        kinds[2].code(),
                        gobbi_bin_label(dist_bins[0]),
                        gobbi_bin_label(dist_bins[1]),
                        gobbi_bin_label(dist_bins[2])
                    );

                    triangles.push(PharmTriangle {
                        feat_indices,
                        kinds,
                        bond_dists,
                        dist_bins,
                        signature_hash,
                        perimeter_bonds,
                        label,
                    });
                }
            }
        }
    }

    PharmacophoreMap {
        features,
        counts_by_kind,
        feature_dists,
        triangles,
        fp,
        cats2_vector: raw_cats,
        murcko_framework_key,
        murcko_label,
    }
}

#[inline]
fn cats2_cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != CATS2_DIM || b.len() != CATS2_DIM {
        return 0.0;
    }
    let mut dot = 0.0f32;
    for i in 0..CATS2_DIM {
        dot += a[i] * b[i];
    }
    dot.clamp(0.0, 1.0)
}

fn triangle_informativeness(tri: &PharmTriangle) -> f32 {
    let type_score = tri.kinds[0].specificity_weight()
        + tri.kinds[1].specificity_weight()
        + tri.kinds[2].specificity_weight();
    let distinct_bonus = if tri.kinds[0] != tri.kinds[1]
        && tri.kinds[1] != tri.kinds[2]
        && tri.kinds[0] != tri.kinds[2]
    {
        0.55
    } else if tri.kinds[0] != tri.kinds[2] || tri.kinds[0] != tri.kinds[1] {
        0.25
    } else {
        0.0
    };
    let perim_bonus = ((tri.perimeter_bonds as f32).clamp(5.0, 22.0) - 5.0) * 0.04;
    type_score + distinct_bonus + perim_bonus
}

/// Aligns two `PharmacophoreMap` objects, computing Gobbi 3-point triangle Tanimoto,
/// `CATS2` cosine similarity, consensus 3-point distance triangles, and scaffold-hopping status.
pub fn align_pharmacophores(
    query: &PharmacophoreMap,
    target: &PharmacophoreMap,
    ecfp4_tanimoto: f32,
) -> PharmacophoreAlignment {
    let triangle_tanimoto = query.fp.tanimoto(&target.fp);
    let cats_cosine = cats2_cosine(&query.cats2_vector, &target.cats2_vector);
    let pharm_similarity = if query.triangles.is_empty() && target.triangles.is_empty() {
        cats_cosine
    } else {
        (0.65 * triangle_tanimoto + 0.35 * cats_cosine).clamp(0.0, 1.0)
    };

    let same_murcko_scaffold = !query.murcko_framework_key.is_empty()
        && query.murcko_framework_key != "ACYCLIC"
        && query.murcko_framework_key == target.murcko_framework_key;

    let is_scaffold_hop =
        !same_murcko_scaffold && pharm_similarity >= 0.40 && ecfp4_tanimoto <= 0.45;
    let scaffold_hop_enrichment = if !same_murcko_scaffold {
        (pharm_similarity - ecfp4_tanimoto).max(0.0)
    } else {
        0.0
    };

    // Index best representative triangle per canonical signature in Query & Target
    let mut q_by_sig: BTreeMap<u64, &PharmTriangle> = BTreeMap::new();
    for qt in &query.triangles {
        q_by_sig
            .entry(qt.signature_hash)
            .and_modify(|cur| {
                if qt.perimeter_bonds > cur.perimeter_bonds {
                    *cur = qt;
                }
            })
            .or_insert(qt);
    }

    let mut t_by_sig: BTreeMap<u64, &PharmTriangle> = BTreeMap::new();
    for tt in &target.triangles {
        t_by_sig
            .entry(tt.signature_hash)
            .and_modify(|cur| {
                if tt.perimeter_bonds > cur.perimeter_bonds {
                    *cur = tt;
                }
            })
            .or_insert(tt);
    }

    let mut matched_triangles = Vec::new();
    for (&sig, &qt) in &q_by_sig {
        if let Some(&tt) = t_by_sig.get(&sig) {
            if qt.kinds == tt.kinds && qt.dist_bins == tt.dist_bins {
                matched_triangles.push(MatchedPharmTriangle {
                    signature_hash: sig,
                    label: qt.label.clone(),
                    query_tri: qt.clone(),
                    target_tri: tt.clone(),
                    informativeness: triangle_informativeness(qt),
                });
            }
        }
    }

    let shared_triangle_count = matched_triangles.len();
    matched_triangles.sort_by(|a, b| {
        b.informativeness
            .partial_cmp(&a.informativeness)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.signature_hash.cmp(&b.signature_hash))
    });
    matched_triangles.truncate(12);

    PharmacophoreAlignment {
        pharm_similarity,
        triangle_tanimoto,
        cats_cosine,
        ecfp4_tanimoto,
        same_murcko_scaffold,
        is_scaffold_hop,
        scaffold_hop_enrichment,
        shared_triangle_count,
        matched_triangles,
    }
}

/// Screens an arbitrary `query_mol` (either a dataset compound or a live-sketched analog)
/// against all compounds in `ds`, returning ranked `PharmacophoreHit` alignments.
pub fn screen_pharmacophore_library(
    query_mol: &Molecule,
    ds: &DatasetAnalysis,
    scaffold_hops_only: bool,
    exclude_exact_smiles: bool,
) -> (PharmacophoreMap, Ecfp2048, Vec<PharmacophoreHit>) {
    let q_map = extract_pharmacophore_map(query_mol);
    let q_ecfp = compute_ecfp(query_mol, 2);

    let mut hits: Vec<PharmacophoreHit> = ds
        .mols
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            if exclude_exact_smiles && !query_mol.smiles.is_empty() && m.smiles == query_mol.smiles
            {
                return None;
            }
            let ecfp_sim = q_ecfp.tanimoto(&ds.fps[idx]);
            let alignment = align_pharmacophores(&q_map, &ds.pharmacophores[idx], ecfp_sim);
            if scaffold_hops_only && !alignment.is_scaffold_hop {
                return None;
            }
            Some(PharmacophoreHit { mol_idx: idx, alignment })
        })
        .collect();

    hits.sort_by(|a, b| {
        b.alignment
            .pharm_similarity
            .partial_cmp(&a.alignment.pharm_similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                ds.mols[b.mol_idx]
                    .pact
                    .partial_cmp(&ds.mols[a.mol_idx].pact)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    (q_map, q_ecfp, hits)
}
