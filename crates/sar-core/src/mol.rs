//! Core molecular graph representation, IUPAC standard atomic weights,
//! valence/hybridization perception, and Smallest Set of Smallest Rings (SSSR) cycle perception.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Supported chemical elements in medicinal chemistry & drug-discovery space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Element {
    H,
    B,
    C,
    N,
    O,
    F,
    Si,
    P,
    S,
    Cl,
    Se,
    Br,
    I,
    Other(u8),
}

impl Element {
    pub fn from_symbol(sym: &str) -> Option<Self> {
        match sym {
            "H" => Some(Self::H),
            "B" | "b" => Some(Self::B),
            "C" | "c" => Some(Self::C),
            "N" | "n" => Some(Self::N),
            "O" | "o" => Some(Self::O),
            "F" => Some(Self::F),
            "Si" => Some(Self::Si),
            "P" | "p" => Some(Self::P),
            "S" | "s" => Some(Self::S),
            "Cl" => Some(Self::Cl),
            "Se" | "se" => Some(Self::Se),
            "Br" => Some(Self::Br),
            "I" => Some(Self::I),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::B => "B",
            Self::C => "C",
            Self::N => "N",
            Self::O => "O",
            Self::F => "F",
            Self::Si => "Si",
            Self::P => "P",
            Self::S => "S",
            Self::Cl => "Cl",
            Self::Se => "Se",
            Self::Br => "Br",
            Self::I => "I",
            Self::Other(_) => "X",
        }
    }

    pub fn atomic_number(self) -> u8 {
        match self {
            Self::H => 1,
            Self::B => 5,
            Self::C => 6,
            Self::N => 7,
            Self::O => 8,
            Self::F => 9,
            Self::Si => 14,
            Self::P => 15,
            Self::S => 16,
            Self::Cl => 17,
            Self::Se => 34,
            Self::Br => 35,
            Self::I => 53,
            Self::Other(z) => z,
        }
    }

    /// Standard IUPAC atomic weights (Da).
    pub fn atomic_weight(self) -> f64 {
        match self {
            Self::H => 1.008,
            Self::B => 10.81,
            Self::C => 12.011,
            Self::N => 14.007,
            Self::O => 15.999,
            Self::F => 18.998,
            Self::Si => 28.085,
            Self::P => 30.974,
            Self::S => 32.06,
            Self::Cl => 35.45,
            Self::Se => 78.971,
            Self::Br => 79.904,
            Self::I => 126.904,
            Self::Other(z) => (z as f64) * 2.0,
        }
    }

    /// Allowed neutral valences per OpenSMILES v1.0 specification (`http://opensmiles.org/opensmiles.html`).
    pub fn default_valences(self) -> &'static [u8] {
        match self {
            Self::H => &[1],
            Self::B => &[3],
            Self::C => &[4],
            Self::N => &[3, 5],
            Self::O => &[2],
            Self::F => &[1],
            Self::Si => &[4],
            Self::P => &[3, 5],
            Self::S => &[2, 4, 6],
            Self::Cl => &[1, 3, 5, 7],
            Self::Se => &[2, 4, 6],
            Self::Br => &[1, 3, 5, 7],
            Self::I => &[1, 3, 5, 7],
            Self::Other(_) => &[0],
        }
    }

    /// Allowed valences adjusted for formal charge (e.g. `N+` -> 4, `O-` -> 1, `B-` -> 4)
    /// when assigning automatic implicit hydrogens in interactive 2D sketching.
    pub fn charged_valences(self, charge: i8) -> &'static [u8] {
        if charge == 0 {
            return self.default_valences();
        }
        match (self, charge) {
            (Self::B, -1) => &[4],
            (Self::C, 1 | -1) => &[3],
            (Self::N | Self::P, 1) => &[4],
            (Self::N | Self::P, -1) => &[2],
            (Self::O, 1) => &[3],
            (Self::O, -1) => &[1],
            (Self::S | Self::Se, 1) => &[3, 5],
            (Self::S | Self::Se, -1) => &[1],
            (Self::F | Self::Cl | Self::Br | Self::I, -1) => &[0],
            _ => self.default_valences(),
        }
    }

    /// CPK / skeletal depiction RGB color `[0..255]` for dark & light rendering.
    pub fn cpk_rgb(self, dark_bg: bool) -> [u8; 3] {
        match self {
            Self::H => if dark_bg { [203, 213, 225] } else { [71, 85, 105] },
            Self::B => [244, 114, 182],
            Self::C => if dark_bg { [226, 232, 240] } else { [30, 41, 59] },
            Self::N => if dark_bg { [96, 165, 250] } else { [37, 99, 235] },
            Self::O => if dark_bg { [248, 113, 113] } else { [220, 38, 38] },
            Self::F => if dark_bg { [52, 211, 153] } else { [5, 150, 105] },
            Self::Si => [251, 191, 36],
            Self::P => [251, 146, 60],
            Self::S => if dark_bg { [250, 204, 21] } else { [202, 138, 4] },
            Self::Cl => if dark_bg { [74, 222, 128] } else { [22, 163, 74] },
            Self::Se => [251, 146, 60],
            Self::Br => if dark_bg { [217, 119, 6] } else { [154, 52, 18] },
            Self::I => [168, 85, 247],
            Self::Other(_) => [148, 163, 184],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondOrder {
    pub fn numeric_order(self) -> f32 {
        match self {
            Self::Single => 1.0,
            Self::Aromatic => 1.5,
            Self::Double => 2.0,
            Self::Triple => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Hybridization {
    Sp,
    Sp2,
    Sp3,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Atom {
    pub idx: usize,
    pub elem: Element,
    pub charge: i8,
    pub aromatic: bool,
    pub bracketed: bool,
    pub explicit_h: u8,
    pub implicit_h: u8,
    pub chiral: i8,
    pub in_ring: bool,
    pub ring_sizes: Vec<usize>,
    pub hyb: Hybridization,
    pub pos2d: [f32; 2],
}

impl Atom {
    pub fn total_h(&self) -> u8 {
        self.explicit_h.saturating_add(self.implicit_h)
    }

    pub fn is_hetero(&self) -> bool {
        !matches!(self.elem, Element::C | Element::H)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bond {
    pub idx: usize,
    pub a: usize,
    pub b: usize,
    pub order: BondOrder,
    pub aromatic: bool,
    pub in_ring: bool,
    /// Kekulized integer bond order (`1`, `2`, or `3`) used for exact valence math and 2D depiction.
    pub kekule_order: u8,
}

impl Bond {
    pub fn other(&self, u: usize) -> usize {
        if self.a == u {
            self.b
        } else {
            self.a
        }
    }
}

/// Complete sanitized small-molecule graph with bioactivity metadata and 2D coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Molecule {
    pub id: String,
    pub name: String,
    pub target: String,
    pub metric: String,
    pub raw_nm: f64,
    pub pact: f64,
    pub source_url: String,
    pub smiles: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Adjacency list: `adj[u]` holds `(neighbor_atom_idx, bond_idx)`.
    pub adj: Vec<Vec<(usize, usize)>>,
    /// Perceived SSSR elementary rings (atom index cycles in perimeter order).
    pub rings: Vec<Vec<usize>>,
}

impl Molecule {
    pub fn new(smiles: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            target: "Unassigned".to_string(),
            metric: "IC50".to_string(),
            raw_nm: 100.0,
            pact: 7.0,
            source_url: String::new(),
            smiles: smiles.into(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            adj: Vec::new(),
            rings: Vec::new(),
        }
    }

    pub fn bond_between(&self, u: usize, v: usize) -> Option<usize> {
        self.adj
            .get(u)
            .and_then(|nbrs| nbrs.iter().find(|&&(w, _)| w == v).map(|&(_, b)| b))
    }

    pub fn heavy_degree(&self, u: usize) -> usize {
        self.adj[u]
            .iter()
            .filter(|&&(v, _)| self.atoms[v].elem != Element::H)
            .count()
    }

    /// Rebuilds adjacency lists, perceives SSSR rings, Kekulizes aromatic systems,
    /// assigns OpenSMILES implicit hydrogens, and perceives orbital hybridization.
    pub fn sanitize(&mut self) {
        let n = self.atoms.len();
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.idx = i;
        }
        for (bi, b) in self.bonds.iter_mut().enumerate() {
            b.idx = bi;
        }
        self.adj = vec![Vec::new(); n];
        for (bi, b) in self.bonds.iter().enumerate() {
            if b.a < n && b.b < n {
                self.adj[b.a].push((b.b, bi));
                self.adj[b.b].push((b.a, bi));
            }
        }

        self.perceive_sssr_rings();
        // Clear stale aromatic flags on any atoms or bonds that are no longer part of a ring
        for a in &mut self.atoms {
            if !a.in_ring && a.aromatic {
                a.aromatic = false;
            }
        }
        for b in &mut self.bonds {
            if !b.in_ring && (b.aromatic || b.order == BondOrder::Aromatic) {
                b.aromatic = false;
                b.order = if b.kekule_order == 2 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
            }
        }
        self.kekulize_and_assign_valences();
        self.perceive_hybridization();
    }

    /// Perceives the Smallest Set of Smallest Rings (SSSR) via shortest per-edge cycle search
    /// and GF(2) Gaussian elimination over bond-incidence bitvectors.
    fn perceive_sssr_rings(&mut self) {
        let n = self.atoms.len();
        let m = self.bonds.len();
        for a in &mut self.atoms {
            a.in_ring = false;
            a.ring_sizes.clear();
        }
        for b in &mut self.bonds {
            b.in_ring = false;
        }
        self.rings.clear();
        if n == 0 || m == 0 {
            return;
        }

        let mut visited = vec![false; n];
        let mut components = 0usize;
        for i in 0..n {
            if !visited[i] {
                components += 1;
                let mut q = VecDeque::new();
                q.push_back(i);
                visited[i] = true;
                while let Some(u) = q.pop_front() {
                    for &(v, _) in &self.adj[u] {
                        if !visited[v] {
                            visited[v] = true;
                            q.push_back(v);
                        }
                    }
                }
            }
        }
        let cyclomatic = (m + components).saturating_sub(n);
        if cyclomatic == 0 {
            return;
        }

        let mut candidate_cycles: Vec<(Vec<usize>, Vec<u64>)> = Vec::new();
        let words = m.div_ceil(64).max(1);

        for (skip_bi, bond) in self.bonds.iter().enumerate() {
            let start = bond.a;
            let goal = bond.b;
            let mut prev: Vec<Option<(usize, usize)>> = vec![None; n];
            let mut seen = vec![false; n];
            let mut q = VecDeque::new();
            seen[start] = true;
            q.push_back(start);
            let mut found = false;
            while let Some(u) = q.pop_front() {
                if u == goal {
                    found = true;
                    break;
                }
                for &(v, bi) in &self.adj[u] {
                    if bi == skip_bi || seen[v] {
                        continue;
                    }
                    seen[v] = true;
                    prev[v] = Some((u, bi));
                    q.push_back(v);
                }
            }
            if found {
                let mut atoms_cyc = vec![goal];
                let mut mask = vec![0u64; words];
                mask[skip_bi / 64] |= 1u64 << (skip_bi % 64);
                let mut cur = goal;
                while let Some((p, bi)) = prev[cur] {
                    mask[bi / 64] |= 1u64 << (bi % 64);
                    atoms_cyc.push(p);
                    if p == start {
                        break;
                    }
                    cur = p;
                }
                if (3..=28).contains(&atoms_cyc.len())
                    && !candidate_cycles.iter().any(|(_, m2)| m2 == &mask)
                {
                    candidate_cycles.push((atoms_cyc, mask));
                }
            }
        }

        candidate_cycles.sort_by_key(|(atoms_cyc, _)| atoms_cyc.len());

        let mut basis: Vec<Vec<u64>> = Vec::new();
        let mut selected: Vec<(Vec<usize>, Vec<u64>)> = Vec::new();

        for (atoms_cyc, mask) in candidate_cycles {
            let mut reduced = mask.clone();
            for bvec in &basis {
                if let Some(pivot_bit) = first_set_bit(bvec) {
                    if (reduced[pivot_bit / 64] >> (pivot_bit % 64)) & 1 == 1 {
                        for w in 0..words {
                            reduced[w] ^= bvec[w];
                        }
                    }
                }
            }
            if first_set_bit(&reduced).is_some() {
                basis.push(reduced);
                basis.sort_by_key(|v| first_set_bit(v).unwrap_or(usize::MAX));
                selected.push((atoms_cyc, mask));
                if selected.len() == cyclomatic {
                    break;
                }
            }
        }

        for (atoms_cyc, mask) in selected {
            let rlen = atoms_cyc.len();
            for &ai in &atoms_cyc {
                self.atoms[ai].in_ring = true;
                if !self.atoms[ai].ring_sizes.contains(&rlen) {
                    self.atoms[ai].ring_sizes.push(rlen);
                }
            }
            for bi in 0..m {
                if (mask[bi / 64] >> (bi % 64)) & 1 == 1 {
                    self.bonds[bi].in_ring = true;
                }
            }
            self.rings.push(atoms_cyc);
        }

        for ring in &self.rings {
            if ring.iter().all(|&ai| self.atoms[ai].aromatic) {
                for k in 0..ring.len() {
                    let u = ring[k];
                    let v = ring[(k + 1) % ring.len()];
                    if let Some(bi) = self.bond_between(u, v) {
                        if self.bonds[bi].order == BondOrder::Single || self.bonds[bi].aromatic {
                            self.bonds[bi].order = BondOrder::Aromatic;
                            self.bonds[bi].aromatic = true;
                        }
                    }
                }
            }
        }
    }

    /// Kekulizes aromatic bonds via exact perfect-matching backtracking first (falling back to
    /// greedy matching only for non-Kekule inputs), perceives Hückel aromatic rings on Kekulé inputs,
    /// and assigns OpenSMILES implicit hydrogens.
    fn kekulize_and_assign_valences(&mut self) {
        let n = self.atoms.len();
        let m = self.bonds.len();

        let mut arom_bonds = Vec::new();
        for (bi, b) in self.bonds.iter_mut().enumerate() {
            match b.order {
                BondOrder::Single => b.kekule_order = 1,
                BondOrder::Double => b.kekule_order = 2,
                BondOrder::Triple => b.kekule_order = 3,
                BondOrder::Aromatic => {
                    b.kekule_order = 1;
                    arom_bonds.push(bi);
                }
            }
        }

        if !arom_bonds.is_empty() {
            let mut can_have_double = vec![false; n];
            let mut needs_double = vec![false; n];

            for u in 0..n {
                if !self.atoms[u].aromatic {
                    continue;
                }
                let non_arom_bo: u8 = self.adj[u]
                    .iter()
                    .filter(|&&(_, bi)| self.bonds[bi].order != BondOrder::Aromatic)
                    .map(|&(_, bi)| self.bonds[bi].kekule_order)
                    .sum();
                let total_non_arom = non_arom_bo + self.atoms[u].explicit_h;
                let has_exo_double = self.adj[u]
                    .iter()
                    .any(|&(_, bi)| self.bonds[bi].order == BondOrder::Double);

                if has_exo_double {
                    can_have_double[u] = false;
                    needs_double[u] = false;
                } else {
                    match self.atoms[u].elem {
                        Element::C => {
                            let max_v = (4i8 + self.atoms[u].charge).max(0) as u8;
                            let arom_deg = self.adj[u]
                                .iter()
                                .filter(|&&(_, bi)| self.bonds[bi].order == BondOrder::Aromatic)
                                .count() as u8;
                            if total_non_arom + arom_deg < max_v {
                                can_have_double[u] = true;
                                needs_double[u] = true;
                            }
                        }
                        Element::N | Element::P => {
                            let arom_deg = self.adj[u]
                                .iter()
                                .filter(|&&(_, bi)| self.bonds[bi].order == BondOrder::Aromatic)
                                .count() as u8;
                            if self.atoms[u].explicit_h > 0
                                || arom_deg >= 3
                                || total_non_arom + arom_deg
                                    >= 3 + (self.atoms[u].charge.max(0) as u8)
                            {
                                can_have_double[u] =
                                    self.atoms[u].charge > 0 && total_non_arom + arom_deg < 4;
                                needs_double[u] = can_have_double[u];
                            } else {
                                can_have_double[u] = true;
                                needs_double[u] = true;
                            }
                        }
                        Element::O | Element::S | Element::Se => {
                            can_have_double[u] = self.atoms[u].charge > 0;
                            needs_double[u] = false;
                        }
                        _ => {}
                    }
                }
            }

            let cand_atoms: Vec<usize> = (0..n).filter(|&u| needs_double[u]).collect();
            let mut matched = vec![false; n];
            let adj_clone = self.adj.clone();

            fn strict_kekule(
                idx: usize,
                cand: &[usize],
                adj: &[Vec<(usize, usize)>],
                bonds: &mut [Bond],
                can_double: &[bool],
                matched: &mut [bool],
            ) -> bool {
                if idx >= cand.len() {
                    return true;
                }
                let u = cand[idx];
                if matched[u] {
                    return strict_kekule(idx + 1, cand, adj, bonds, can_double, matched);
                }
                for &(v, bi) in &adj[u] {
                    if bonds[bi].order == BondOrder::Aromatic && !matched[v] && can_double[v] {
                        matched[u] = true;
                        matched[v] = true;
                        bonds[bi].kekule_order = 2;
                        if strict_kekule(idx + 1, cand, adj, bonds, can_double, matched) {
                            return true;
                        }
                        bonds[bi].kekule_order = 1;
                        matched[u] = false;
                        matched[v] = false;
                    }
                }
                false
            }

            if !strict_kekule(
                0,
                &cand_atoms,
                &adj_clone,
                &mut self.bonds[..m],
                &can_have_double,
                &mut matched,
            ) {
                matched.fill(false);
                for &u in &cand_atoms {
                    if matched[u] {
                        continue;
                    }
                    for &(v, bi) in &adj_clone[u] {
                        if self.bonds[bi].order == BondOrder::Aromatic
                            && !matched[v]
                            && can_have_double[v]
                        {
                            matched[u] = true;
                            matched[v] = true;
                            self.bonds[bi].kekule_order = 2;
                            break;
                        }
                    }
                }
            }
        }

        // Perceive aromaticity on Kekulized 5- and 6-membered Hückel rings (including fused Kekulé sketches & V2000 SDFs)
        for ring in &self.rings {
            let rlen = ring.len();
            if rlen == 5 || rlen == 6 {
                let mut pi_electrons = 0usize;
                let mut all_sp2_ring = true;
                for &u in ring {
                    if self.atoms[u].aromatic {
                        pi_electrons += 1;
                        continue;
                    }
                    let ring_double = self.adj[u].iter().any(|&(v, bi)| {
                        self.bonds[bi].kekule_order == 2
                            && (ring.contains(&v) || self.bonds[bi].in_ring)
                    });
                    let exo_carbonyl = self.adj[u].iter().any(|&(v, bi)| {
                        self.bonds[bi].kekule_order == 2
                            && !ring.contains(&v)
                            && !self.bonds[bi].in_ring
                            && matches!(self.atoms[v].elem, Element::O | Element::S)
                    });
                    if ring_double {
                        pi_electrons += 1;
                    } else if exo_carbonyl {
                        // Ring carbonyl (e.g. pyridone / flavone): sp2 planar, contributes 0 pi e-
                    } else if matches!(self.atoms[u].elem, Element::N | Element::O | Element::S)
                        && self.adj[u].len() <= 3
                    {
                        pi_electrons += 2;
                    } else {
                        all_sp2_ring = false;
                        break;
                    }
                }
                if all_sp2_ring && (pi_electrons == 6 || pi_electrons == 10) {
                    for &u in ring {
                        self.atoms[u].aromatic = true;
                    }
                    for k in 0..rlen {
                        let u = ring[k];
                        let v = ring[(k + 1) % rlen];
                        if let Some(bi) = self.bond_between(u, v) {
                            self.bonds[bi].aromatic = true;
                        }
                    }
                }
            }
        }

        // Assign OpenSMILES implicit hydrogens using Kekulized integer bond orders
        for u in 0..n {
            if self.atoms[u].bracketed {
                self.atoms[u].implicit_h = 0;
                continue;
            }
            let bond_valence_sum: u8 = self.adj[u]
                .iter()
                .map(|&(_, bi)| self.bonds[bi].kekule_order)
                .sum();
            let used = bond_valence_sum.saturating_add(self.atoms[u].explicit_h);
            let valences = self.atoms[u].elem.charged_valences(self.atoms[u].charge);
            let target_v = valences
                .iter()
                .copied()
                .find(|&v| v >= used)
                .unwrap_or(used);
            self.atoms[u].implicit_h = target_v.saturating_sub(used);
        }
    }

    fn perceive_hybridization(&mut self) {
        for u in 0..self.atoms.len() {
            let has_triple = self.adj[u]
                .iter()
                .any(|&(_, bi)| self.bonds[bi].kekule_order == 3);
            let double_cnt = self.adj[u]
                .iter()
                .filter(|&&(_, bi)| self.bonds[bi].kekule_order == 2)
                .count();
            self.atoms[u].hyb = if has_triple || double_cnt >= 2 {
                Hybridization::Sp
            } else if self.atoms[u].aromatic || double_cnt == 1 {
                Hybridization::Sp2
            } else {
                Hybridization::Sp3
            };
        }
    }

    /// Computes exact Hill-system molecular formula (`C`, `H`, then alphabetical symbols).
    pub fn molecular_formula(&self) -> String {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut total_h = 0usize;
        for a in &self.atoms {
            if a.elem == Element::H {
                total_h += 1;
            } else {
                *counts.entry(a.elem.symbol()).or_insert(0) += 1;
                total_h += a.total_h() as usize;
            }
        }
        if total_h > 0 {
            *counts.entry("H").or_insert(0) += total_h;
        }

        let mut out = String::new();
        if let Some(&c) = counts.get("C") {
            out.push('C');
            if c > 1 {
                out.push_str(&c.to_string());
            }
            counts.remove("C");
            if let Some(&h) = counts.get("H") {
                out.push('H');
                if h > 1 {
                    out.push_str(&h.to_string());
                }
                counts.remove("H");
            }
        }
        for (sym, cnt) in counts {
            out.push_str(sym);
            if cnt > 1 {
                out.push_str(&cnt.to_string());
            }
        }
        out
    }

    /// Computes standard IUPAC molar mass (Da).
    pub fn molecular_weight(&self) -> f64 {
        let mut mw = 0.0;
        for a in &self.atoms {
            mw += a.elem.atomic_weight();
            if a.elem != Element::H {
                mw += (a.total_h() as f64) * Element::H.atomic_weight();
            }
        }
        mw
    }
}

fn first_set_bit(words: &[u64]) -> Option<usize> {
    for (wi, &w) in words.iter().enumerate() {
        if w != 0 {
            return Some(wi * 64 + (w.trailing_zeros() as usize));
        }
    }
    None
}
