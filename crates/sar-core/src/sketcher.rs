//! Interactive 2D bond/atom molecular graph sketcher, Weininger `CANGEN`-style canonical
//! OpenSMILES generator, Undo/Redo command history, ring-fusion & functional-group stamps,
//! and real-time multiparameter analog evaluator (`QED`, `CNS-MPO`, `PAINS`, `Free-Wilson pAct`,
//! and nearest `ECFP4` / `SALI` activity-cliff neighbors):
//! - Weininger, Weininger & Weininger (1989), *J. Chem. Inf. Comput. Sci.* 29, 97-101
//!   (`https://pubs.acs.org/jcisd8/article/29/2/97/1079225/SMILES-2-Algorithm-for-generation-of-unique-SMILES`)
//! - OpenSMILES v1.0 specification (`http://opensmiles.org/opensmiles.html`)
//! - Bickerton et al. (2012) `QED` (`https://pubmed.ncbi.nlm.nih.gov/22270643/`)
//! - Wager et al. (2010) `CNS-MPO` (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3368654/`)
//! - Baell & Holloway (2010) `PAINS` (`https://pubmed.ncbi.nlm.nih.gov/20131845/`)
//! - Guha & Van Drie (2008) `SALI` Activity Cliffs (`https://pubmed.ncbi.nlm.nih.gov/18686944/`)

use crate::alerts::{scan_structural_alerts, StructuralAlertMatch};
use crate::fingerprint::compute_ecfp;
use crate::layout2d::{compute_2d_coords, BOND_LEN_2D};
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::{parse_smiles, MolIoError};
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::rgroup_table::{predict_best_free_wilson, FreeWilsonMatrix, FreeWilsonPrediction};
use crate::DatasetAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::f32::consts::{PI, TAU};

const MAX_UNDO_HISTORY: usize = 64;

/// Ring templates available for one-click canvas placement, exocyclic sprouting, or edge fusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RingTemplate {
    Benzene,
    Pyridine,
    Pyrimidine,
    Cyclohexane,
    Piperazine,
    Cyclopentane,
    Furan,
    Thiophene,
}

impl RingTemplate {
    pub const ALL: [Self; 8] = [
        Self::Benzene,
        Self::Pyridine,
        Self::Pyrimidine,
        Self::Cyclohexane,
        Self::Piperazine,
        Self::Cyclopentane,
        Self::Furan,
        Self::Thiophene,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Benzene => "Ph (6-Arom)",
            Self::Pyridine => "Pyr (6-N)",
            Self::Pyrimidine => "Pym (6-N2)",
            Self::Cyclohexane => "CyHex (6)",
            Self::Piperazine => "Pipz (6-N2)",
            Self::Cyclopentane => "CyPent (5)",
            Self::Furan => "Furan (5-O)",
            Self::Thiophene => "Thio (5-S)",
        }
    }

    pub fn ring_size(self) -> usize {
        match self {
            Self::Cyclopentane | Self::Furan | Self::Thiophene => 5,
            _ => 6,
        }
    }

    /// Returns `(elements_around_perimeter, bond_orders_around_perimeter, is_aromatic_ring)`.
    /// Perimeter index `0` and `1` are the fusion edge when fusing onto an existing bond.
    fn spec(self) -> (Vec<Element>, Vec<BondOrder>, bool) {
        use BondOrder::{Aromatic, Double, Single};
        use Element::{C, N, O, S};
        match self {
            Self::Benzene => (
                vec![C, C, C, C, C, C],
                vec![Aromatic; 6],
                true,
            ),
            Self::Pyridine => (
                vec![C, C, C, N, C, C],
                vec![Aromatic; 6],
                true,
            ),
            Self::Pyrimidine => (
                vec![C, C, N, C, N, C],
                vec![Aromatic; 6],
                true,
            ),
            Self::Cyclohexane => (
                vec![C, C, C, C, C, C],
                vec![Single; 6],
                false,
            ),
            Self::Piperazine => (
                vec![N, C, C, N, C, C],
                vec![Single; 6],
                false,
            ),
            Self::Cyclopentane => (
                vec![C, C, C, C, C],
                vec![Single; 5],
                false,
            ),
            Self::Furan => (
                vec![C, C, C, O, C],
                vec![Single, Double, Single, Single, Double],
                true,
            ),
            Self::Thiophene => (
                vec![C, C, C, S, C],
                vec![Single, Double, Single, Single, Double],
                true,
            ),
        }
    }
}

/// Medicinal-chemistry R-group / warhead stamps for one-click sprouting on any atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunctionalGroup {
    Trifluoromethyl,
    Methoxy,
    Cyano,
    Ethynyl,
    Methylsulfonyl,
    Sulfonamide,
    Carboxamide,
    CarboxylicAcid,
    Dimethylamino,
    Acrylamide,
}

impl FunctionalGroup {
    pub const ALL: [Self; 10] = [
        Self::Trifluoromethyl,
        Self::Methoxy,
        Self::Cyano,
        Self::Ethynyl,
        Self::Methylsulfonyl,
        Self::Sulfonamide,
        Self::Carboxamide,
        Self::CarboxylicAcid,
        Self::Dimethylamino,
        Self::Acrylamide,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Trifluoromethyl => "-CF3",
            Self::Methoxy => "-OCH3",
            Self::Cyano => "-CN",
            Self::Ethynyl => "-C#CH",
            Self::Methylsulfonyl => "-SO2Me",
            Self::Sulfonamide => "-SO2NH2",
            Self::Carboxamide => "-CONH2",
            Self::CarboxylicAcid => "-COOH",
            Self::Dimethylamino => "-NMe2",
            Self::Acrylamide => "-NHCOCH=CH2",
        }
    }
}

/// Active interactive canvas tool in the 2D Molecular Sketcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SketchTool {
    DrawBond,
    MoveAtom,
    Erase,
    Ring(RingTemplate),
    Group(FunctionalGroup),
    Charge(i8),
}

/// Nearest dataset compound ranked by 2048-bit `ECFP4` Tanimoto similarity with prospective `SALI`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaliNeighborHit {
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    pub observed_pact: f64,
    pub tanimoto: f32,
    pub delta_pact: f64,
    pub prospective_sali: f32,
    pub is_cliff: bool,
}

/// Property & potency shift of the sketched analog relative to a pinned reference lead compound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParentLeadDelta {
    pub parent_mol_idx: Option<usize>,
    pub parent_name: String,
    pub parent_pact: f64,
    pub delta_mw: f64,
    pub delta_clogp: f64,
    pub delta_tpsa: f64,
    pub delta_qed: f64,
    pub delta_cns_mpo: f64,
    pub delta_pact_est: f64,
    pub tanimoto_to_parent: f32,
}

/// Real-time medicinal-chemistry scorecard recomputed automatically on every 2D sketcher edit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveDesignReport {
    pub canonical_smiles: String,
    pub profile: PhyschemProfile,
    pub alerts: Vec<StructuralAlertMatch>,
    pub alert_atoms: Vec<usize>,
    pub free_wilson: Option<FreeWilsonPrediction>,
    pub effective_pact: f64,
    pub effective_ic50_nm: f64,
    pub effective_lle: f64,
    pub pact_source: String,
    pub sali_neighbors: Vec<SaliNeighborHit>,
    pub max_sali_cliff: f32,
    pub cliff_count: usize,
    pub parent_delta: Option<ParentLeadDelta>,
}

/// Stateful 2D molecular editor with 64-step Undo/Redo history and pinned parent lead comparison.
#[derive(Clone, Debug)]
pub struct MolecularSketcher {
    pub mol: Molecule,
    pub tool: SketchTool,
    pub active_element: Element,
    pub active_bond_order: BondOrder,
    pub pinned_parent_idx: Option<usize>,
    pub pinned_parent_mol: Option<Molecule>,
    pub pinned_parent_profile: Option<PhyschemProfile>,
    undo_stack: Vec<Molecule>,
    redo_stack: Vec<Molecule>,
    pub revision: u64,
}

impl Default for MolecularSketcher {
    fn default() -> Self {
        // Default seed: Erlotinib or clean 4-anilinoquinazoline scaffold so user can immediately edit R-groups
        let mol = parse_smiles("COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC").unwrap_or_else(|_| {
            let mut m = Molecule::new("");
            m.sanitize();
            m
        });
        let mut s = Self {
            mol,
            tool: SketchTool::DrawBond,
            active_element: Element::C,
            active_bond_order: BondOrder::Single,
            pinned_parent_idx: None,
            pinned_parent_mol: None,
            pinned_parent_profile: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            revision: 1,
        };
        s.mol.smiles = molecule_to_smiles(&s.mol);
        s
    }
}

impl MolecularSketcher {
    pub fn from_dataset_compound(ds: &DatasetAnalysis, mol_idx: usize) -> Self {
        let mut s = Self::default();
        s.load_dataset_compound(ds, mol_idx);
        s
    }

    pub fn load_dataset_compound(&mut self, ds: &DatasetAnalysis, mol_idx: usize) {
        if mol_idx >= ds.mols.len() {
            return;
        }
        self.push_undo_snapshot();
        let src = ds.mols[mol_idx].clone();
        self.pinned_parent_idx = Some(mol_idx);
        self.pinned_parent_mol = Some(src.clone());
        self.pinned_parent_profile = Some(ds.profiles[mol_idx].clone());
        self.mol = src;
        self.mol.name = format!("{}-analog", self.mol.name);
        self.finish_mutation(false);
    }

    pub fn load_from_smiles(&mut self, smiles: &str) -> Result<(), MolIoError> {
        let parsed = parse_smiles(smiles)?;
        self.push_undo_snapshot();
        self.mol = parsed;
        if self.mol.name.is_empty() {
            self.mol.name = "Sketched-Analog".to_string();
        }
        self.finish_mutation(false);
        Ok(())
    }

    pub fn clear_canvas(&mut self) {
        if self.mol.atoms.is_empty() {
            return;
        }
        self.push_undo_snapshot();
        self.mol.atoms.clear();
        self.mol.bonds.clear();
        self.mol.adj.clear();
        self.mol.rings.clear();
        self.mol.smiles.clear();
        self.revision += 1;
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.undo_stack.pop() {
            self.redo_stack.push(self.mol.clone());
            self.mol = prev;
            self.revision += 1;
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.mol.clone());
            self.mol = next;
            self.revision += 1;
            true
        } else {
            false
        }
    }

    /// Saves an undo snapshot explicitly before beginning a continuous atom-drag gesture.
    pub fn begin_drag_snapshot(&mut self) {
        self.push_undo_snapshot();
    }

    /// Moves atom `u` to 2D Angstrom coordinates `pos` without creating an undo entry per frame.
    pub fn move_atom_to(&mut self, u: usize, pos: [f32; 2]) {
        if let Some(a) = self.mol.atoms.get_mut(u) {
            a.pos2d = pos;
            self.revision += 1;
        }
    }

    /// Re-runs deterministic 2D coordinate cleanup (`1.50 Å` bond lengths & fused-ring geometry).
    pub fn clean_2d_layout(&mut self) {
        if self.mol.atoms.is_empty() {
            return;
        }
        self.push_undo_snapshot();
        self.mol.sanitize();
        compute_2d_coords(&mut self.mol);
        self.mol.smiles = molecule_to_smiles(&self.mol);
        self.revision += 1;
    }

    /// Places a standalone atom at `pos` on empty canvas.
    pub fn add_floating_atom(&mut self, elem: Element, pos: [f32; 2]) -> usize {
        self.push_undo_snapshot();
        let idx = self.push_raw_atom(elem, false, pos);
        self.finish_mutation(false);
        idx
    }

    /// Mutates the element of atom `u` (clearing bracket overrides so implicit H updates cleanly).
    pub fn set_atom_element(&mut self, u: usize, elem: Element) {
        if u >= self.mol.atoms.len() {
            return;
        }
        self.push_undo_snapshot();
        self.mol.atoms[u].elem = elem;
        self.mol.atoms[u].bracketed = false;
        self.mol.atoms[u].explicit_h = 0;
        self.mol.atoms[u].charge = 0;
        if !matches!(elem, Element::C | Element::N | Element::O | Element::S | Element::P | Element::B)
        {
            self.mol.atoms[u].aromatic = false;
        }
        self.finish_mutation(false);
    }

    /// Adjusts formal charge on atom `u` by `delta` (clamped to `-2..=2`).
    pub fn adjust_atom_charge(&mut self, u: usize, delta: i8) {
        if u >= self.mol.atoms.len() {
            return;
        }
        self.push_undo_snapshot();
        let new_q = (self.mol.atoms[u].charge + delta).clamp(-2, 2);
        self.mol.atoms[u].charge = new_q;
        self.mol.atoms[u].bracketed = false;
        self.mol.atoms[u].explicit_h = 0;
        self.finish_mutation(false);
    }

    /// Sprouts a new bond of `order` and atom of `elem` from `anchor_u`, using `preferred_pos`
    /// if provided (e.g. mouse drag release) or automatic `120°`/`180°` steric-free angle otherwise.
    pub fn sprout_bond_and_atom(
        &mut self,
        anchor_u: usize,
        elem: Element,
        order: BondOrder,
        preferred_pos: Option<[f32; 2]>,
    ) -> Option<usize> {
        if anchor_u >= self.mol.atoms.len() {
            return None;
        }
        self.push_undo_snapshot();
        let pos = preferred_pos.unwrap_or_else(|| self.suggest_sprout_pos(anchor_u));
        let new_u = self.push_raw_atom(elem, false, pos);
        let eff_order = if order == BondOrder::Aromatic {
            BondOrder::Single
        } else {
            order
        };
        self.push_raw_bond(anchor_u, new_u, eff_order, false);
        self.finish_mutation(false);
        Some(new_u)
    }

    /// Connects two existing atoms `u` and `v`, or cycles the existing bond order between them.
    pub fn connect_or_cycle_bond(&mut self, u: usize, v: usize, default_order: BondOrder) {
        if u == v || u >= self.mol.atoms.len() || v >= self.mol.atoms.len() {
            return;
        }
        if let Some(bi) = self.mol.bond_between(u, v) {
            self.cycle_bond_order(bi);
            return;
        }
        self.push_undo_snapshot();
        let eff = if default_order == BondOrder::Aromatic {
            BondOrder::Single
        } else {
            default_order
        };
        self.push_raw_bond(u, v, eff, false);
        self.finish_mutation(false);
    }

    /// Cycles existing bond `bi` through `Single -> Double -> Triple -> Single`
    /// (converting aromatic ring bonds into explicit Kekulé orders first if clicked).
    pub fn cycle_bond_order(&mut self, bi: usize) {
        if bi >= self.mol.bonds.len() {
            return;
        }
        self.push_undo_snapshot();
        if self.mol.bonds[bi].aromatic || self.mol.bonds[bi].order == BondOrder::Aromatic {
            // Materialize current Kekulé orders on this ring so user can edit individual ring bonds
            for b in &mut self.mol.bonds {
                if b.aromatic || b.order == BondOrder::Aromatic {
                    b.aromatic = false;
                    b.order = if b.kekule_order == 2 {
                        BondOrder::Double
                    } else {
                        BondOrder::Single
                    };
                }
            }
            for a in &mut self.mol.atoms {
                a.aromatic = false;
            }
        }
        let next = match self.mol.bonds[bi].order {
            BondOrder::Single => BondOrder::Double,
            BondOrder::Double => BondOrder::Triple,
            BondOrder::Triple | BondOrder::Aromatic => BondOrder::Single,
        };
        self.mol.bonds[bi].order = next;
        self.mol.bonds[bi].aromatic = false;
        self.finish_mutation(false);
    }

    /// Sets explicit bond order `order` on existing bond `bi`.
    pub fn set_bond_order(&mut self, bi: usize, order: BondOrder) {
        if bi >= self.mol.bonds.len() {
            return;
        }
        self.push_undo_snapshot();
        self.mol.bonds[bi].order = order;
        self.mol.bonds[bi].aromatic = order == BondOrder::Aromatic;
        self.finish_mutation(false);
    }

    /// Deletes atom `target_u` and all incident bonds, re-indexing remaining atoms cleanly.
    pub fn delete_atom(&mut self, target_u: usize) {
        if target_u >= self.mol.atoms.len() {
            return;
        }
        self.push_undo_snapshot();
        self.mol.atoms.remove(target_u);
        self.mol
            .bonds
            .retain(|b| b.a != target_u && b.b != target_u);
        for b in &mut self.mol.bonds {
            if b.a > target_u {
                b.a -= 1;
            }
            if b.b > target_u {
                b.b -= 1;
            }
        }
        self.finish_mutation(false);
    }

    /// Deletes bond `target_bi` without removing its endpoint atoms.
    pub fn delete_bond(&mut self, target_bi: usize) {
        if target_bi >= self.mol.bonds.len() {
            return;
        }
        self.push_undo_snapshot();
        self.mol.bonds.remove(target_bi);
        self.finish_mutation(false);
    }

    /// Stamps `tpl` centered at `center` on empty canvas.
    pub fn stamp_ring_at(&mut self, center: [f32; 2], tpl: RingTemplate) {
        self.push_undo_snapshot();
        let (elems, orders, is_arom) = tpl.spec();
        let k = elems.len();
        let radius = BOND_LEN_2D / (2.0 * (PI / (k as f32)).sin());
        let mut ring_atoms = Vec::with_capacity(k);
        for (i, &el) in elems.iter().enumerate() {
            let theta = -PI * 0.5 + (i as f32) * TAU / (k as f32);
            let pos = [center[0] + radius * theta.cos(), center[1] + radius * theta.sin()];
            ring_atoms.push(self.push_raw_atom(el, is_arom, pos));
        }
        for i in 0..k {
            let bo = orders[i];
            self.push_raw_bond(
                ring_atoms[i],
                ring_atoms[(i + 1) % k],
                bo,
                bo == BondOrder::Aromatic,
            );
        }
        self.finish_mutation(false);
    }

    /// Sprouts an exocyclic ring `tpl` attached by a single bond from `anchor_u` (or turns an isolated
    /// degree-0 atom into vertex 0 of the ring).
    pub fn sprout_ring_from_atom(&mut self, anchor_u: usize, tpl: RingTemplate) {
        if anchor_u >= self.mol.atoms.len() {
            return;
        }
        if self.mol.adj[anchor_u].is_empty() {
            let p0 = self.mol.atoms[anchor_u].pos2d;
            self.push_undo_snapshot();
            let (elems, orders, is_arom) = tpl.spec();
            let k = elems.len();
            let radius = BOND_LEN_2D / (2.0 * (PI / (k as f32)).sin());
            let center = [p0[0] + radius, p0[1]];
            self.mol.atoms[anchor_u].aromatic |= is_arom;
            let mut ring_atoms = Vec::with_capacity(k);
            ring_atoms.push(anchor_u);
            for i in 1..k {
                let theta = PI + (i as f32) * TAU / (k as f32);
                let pos = [center[0] + radius * theta.cos(), center[1] + radius * theta.sin()];
                ring_atoms.push(self.push_raw_atom(elems[i], is_arom, pos));
            }
            for i in 0..k {
                let bo = orders[i];
                self.push_raw_bond(
                    ring_atoms[i],
                    ring_atoms[(i + 1) % k],
                    bo,
                    bo == BondOrder::Aromatic,
                );
            }
            self.finish_mutation(false);
            return;
        }

        self.push_undo_snapshot();
        let p_anchor = self.mol.atoms[anchor_u].pos2d;
        let attach_pos = self.suggest_sprout_pos(anchor_u);
        let dir = normalize2([attach_pos[0] - p_anchor[0], attach_pos[1] - p_anchor[1]]);

        let (elems, orders, is_arom) = tpl.spec();
        let k = elems.len();
        let radius = BOND_LEN_2D / (2.0 * (PI / (k as f32)).sin());
        let center = [
            attach_pos[0] + dir[0] * radius,
            attach_pos[1] + dir[1] * radius,
        ];
        let base_angle = (-dir[1]).atan2(-dir[0]);

        let mut ring_atoms = Vec::with_capacity(k);
        for (i, &el) in elems.iter().enumerate() {
            let theta = base_angle + (i as f32) * TAU / (k as f32);
            let pos = if i == 0 {
                attach_pos
            } else {
                [center[0] + radius * theta.cos(), center[1] + radius * theta.sin()]
            };
            ring_atoms.push(self.push_raw_atom(el, is_arom, pos));
        }
        // Exocyclic attachment bond from anchor_u to ring_atoms[0]
        self.push_raw_bond(anchor_u, ring_atoms[0], BondOrder::Single, false);
        for i in 0..k {
            let bo = orders[i];
            self.push_raw_bond(
                ring_atoms[i],
                ring_atoms[(i + 1) % k],
                bo,
                bo == BondOrder::Aromatic,
            );
        }
        self.finish_mutation(false);
    }

    /// Fuses ring `tpl` directly across existing bond `bond_idx` (`u -- v`), projecting the new
    /// polygon vertices outward away from existing neighbors to form clean bicyclic/tricyclic cores!
    pub fn fuse_ring_on_bond(&mut self, bond_idx: usize, tpl: RingTemplate) {
        if bond_idx >= self.mol.bonds.len() {
            return;
        }
        self.push_undo_snapshot();
        let u = self.mol.bonds[bond_idx].a;
        let v = self.mol.bonds[bond_idx].b;
        let pu = self.mol.atoms[u].pos2d;
        let pv = self.mol.atoms[v].pos2d;
        let mid = [(pu[0] + pv[0]) * 0.5, (pu[1] + pv[1]) * 0.5];
        let edge = [pv[0] - pu[0], pv[1] - pu[1]];
        let edge_len = (edge[0] * edge[0] + edge[1] * edge[1]).sqrt().max(0.25);
        let eu = [edge[0] / edge_len, edge[1] / edge_len];

        // Choose outward normal pointing away from existing neighbors of u and v
        let n1 = [-eu[1], eu[0]];
        let n2 = [eu[1], -eu[0]];
        let mut nbr_sum = [0.0f32, 0.0f32];
        let mut nbr_cnt = 0usize;
        for &w in &[u, v] {
            for &(nb, _) in &self.mol.adj[w] {
                if nb != u && nb != v {
                    nbr_sum[0] += self.mol.atoms[nb].pos2d[0] - mid[0];
                    nbr_sum[1] += self.mol.atoms[nb].pos2d[1] - mid[1];
                    nbr_cnt += 1;
                }
            }
        }
        let outward = if nbr_cnt > 0 {
            let dot1 = n1[0] * nbr_sum[0] + n1[1] * nbr_sum[1];
            if dot1 <= 0.0 {
                n1
            } else {
                n2
            }
        } else {
            n1
        };

        let (elems, orders, is_arom) = tpl.spec();
        let k = elems.len();
        let half_angle = PI / (k as f32);
        let apothem = edge_len / (2.0 * half_angle.tan());
        let radius = edge_len / (2.0 * half_angle.sin());
        let center = [mid[0] + outward[0] * apothem, mid[1] + outward[1] * apothem];

        let ang_u = (pu[1] - center[1]).atan2(pu[0] - center[0]);
        let ang_v = (pv[1] - center[1]).atan2(pv[0] - center[0]);
        let mut step = ang_v - ang_u;
        while step <= -PI {
            step += TAU;
        }
        while step > PI {
            step -= TAU;
        }
        let step_sign = if step >= 0.0 { 1.0 } else { -1.0 };
        let delta = step_sign * TAU / (k as f32);

        if is_arom {
            self.mol.atoms[u].aromatic = true;
            self.mol.atoms[v].aromatic = true;
            self.mol.bonds[bond_idx].order = BondOrder::Aromatic;
            self.mol.bonds[bond_idx].aromatic = true;
        }

        let mut ring_atoms = Vec::with_capacity(k);
        ring_atoms.push(u);
        ring_atoms.push(v);
        for i in 2..k {
            let theta = ang_u + (i as f32) * delta;
            let pos = [center[0] + radius * theta.cos(), center[1] + radius * theta.sin()];
            ring_atoms.push(self.push_raw_atom(elems[i], is_arom, pos));
        }

        for i in 1..k {
            let a = ring_atoms[i];
            let b = ring_atoms[(i + 1) % k];
            let bo = orders[i];
            self.push_raw_bond(a, b, bo, bo == BondOrder::Aromatic);
        }
        self.finish_mutation(false);
    }

    /// Sprouts a multi-atom medicinal-chemistry functional group (`-CF3`, `-SO2NH2`, `-CN`,
    /// covalent acrylamide, etc.) onto `anchor_u` with clean pre-oriented 2D coordinates.
    pub fn attach_functional_group(&mut self, anchor_u: usize, fg: FunctionalGroup) {
        if anchor_u >= self.mol.atoms.len() {
            return;
        }
        self.push_undo_snapshot();
        let p0 = self.mol.atoms[anchor_u].pos2d;
        let p1 = self.suggest_sprout_pos(anchor_u);
        let d = normalize2([p1[0] - p0[0], p1[1] - p0[1]]);
        let perp = [-d[1], d[0]];
        let b = BOND_LEN_2D;

        let step_from = |orig: [f32; 2], ang_offset: f32| -> [f32; 2] {
            let base = d[1].atan2(d[0]) + ang_offset;
            [orig[0] + b * base.cos(), orig[1] + b * base.sin()]
        };

        match fg {
            FunctionalGroup::Trifluoromethyl => {
                let c = self.push_raw_atom(Element::C, false, p1);
                self.push_raw_bond(anchor_u, c, BondOrder::Single, false);
                for &ang in &[0.0f32, PI * 0.5, -PI * 0.5] {
                    let f = self.push_raw_atom(Element::F, false, step_from(p1, ang));
                    self.push_raw_bond(c, f, BondOrder::Single, false);
                }
            }
            FunctionalGroup::Methoxy => {
                let o = self.push_raw_atom(Element::O, false, p1);
                let c = self.push_raw_atom(Element::C, false, step_from(p1, PI / 3.0));
                self.push_raw_bond(anchor_u, o, BondOrder::Single, false);
                self.push_raw_bond(o, c, BondOrder::Single, false);
            }
            FunctionalGroup::Cyano => {
                let c = self.push_raw_atom(Element::C, false, p1);
                let n = self.push_raw_atom(Element::N, false, step_from(p1, 0.0));
                self.push_raw_bond(anchor_u, c, BondOrder::Single, false);
                self.push_raw_bond(c, n, BondOrder::Triple, false);
            }
            FunctionalGroup::Ethynyl => {
                let c1 = self.push_raw_atom(Element::C, false, p1);
                let c2 = self.push_raw_atom(Element::C, false, step_from(p1, 0.0));
                self.push_raw_bond(anchor_u, c1, BondOrder::Single, false);
                self.push_raw_bond(c1, c2, BondOrder::Triple, false);
            }
            FunctionalGroup::Methylsulfonyl | FunctionalGroup::Sulfonamide => {
                let s = self.push_raw_atom(Element::S, false, p1);
                let o1 = self.push_raw_atom(
                    Element::O,
                    false,
                    [p1[0] + perp[0] * b, p1[1] + perp[1] * b],
                );
                let o2 = self.push_raw_atom(
                    Element::O,
                    false,
                    [p1[0] - perp[0] * b, p1[1] - perp[1] * b],
                );
                let term_el = if fg == FunctionalGroup::Sulfonamide {
                    Element::N
                } else {
                    Element::C
                };
                let tail = self.push_raw_atom(term_el, false, step_from(p1, 0.0));
                self.push_raw_bond(anchor_u, s, BondOrder::Single, false);
                self.push_raw_bond(s, o1, BondOrder::Double, false);
                self.push_raw_bond(s, o2, BondOrder::Double, false);
                self.push_raw_bond(s, tail, BondOrder::Single, false);
            }
            FunctionalGroup::Carboxamide | FunctionalGroup::CarboxylicAcid => {
                let c = self.push_raw_atom(Element::C, false, p1);
                let o_ket = self.push_raw_atom(Element::O, false, step_from(p1, PI / 3.0));
                let hetero = if fg == FunctionalGroup::Carboxamide {
                    Element::N
                } else {
                    Element::O
                };
                let x = self.push_raw_atom(hetero, false, step_from(p1, -PI / 3.0));
                self.push_raw_bond(anchor_u, c, BondOrder::Single, false);
                self.push_raw_bond(c, o_ket, BondOrder::Double, false);
                self.push_raw_bond(c, x, BondOrder::Single, false);
            }
            FunctionalGroup::Dimethylamino => {
                let n = self.push_raw_atom(Element::N, false, p1);
                let c1 = self.push_raw_atom(Element::C, false, step_from(p1, PI / 3.0));
                let c2 = self.push_raw_atom(Element::C, false, step_from(p1, -PI / 3.0));
                self.push_raw_bond(anchor_u, n, BondOrder::Single, false);
                self.push_raw_bond(n, c1, BondOrder::Single, false);
                self.push_raw_bond(n, c2, BondOrder::Single, false);
            }
            FunctionalGroup::Acrylamide => {
                let n = self.push_raw_atom(Element::N, false, p1);
                let p_c = step_from(p1, PI / 3.0);
                let c_ket = self.push_raw_atom(Element::C, false, p_c);
                let o_ket = self.push_raw_atom(Element::O, false, step_from(p_c, 2.0 * PI / 3.0));
                let p_ca = step_from(p_c, 0.0);
                let c_alpha = self.push_raw_atom(Element::C, false, p_ca);
                let c_beta = self.push_raw_atom(Element::C, false, step_from(p_ca, PI / 3.0));
                self.push_raw_bond(anchor_u, n, BondOrder::Single, false);
                self.push_raw_bond(n, c_ket, BondOrder::Single, false);
                self.push_raw_bond(c_ket, o_ket, BondOrder::Double, false);
                self.push_raw_bond(c_ket, c_alpha, BondOrder::Single, false);
                self.push_raw_bond(c_alpha, c_beta, BondOrder::Double, false);
            }
        }
        self.finish_mutation(false);
    }

    /// Computes a smart `120°` (`180°` for `sp` triple/allene atoms) outward sprouting coordinate
    /// around `u` maximizing angular clearance from existing bonds.
    pub fn suggest_sprout_pos(&self, u: usize) -> [f32; 2] {
        let p = self.mol.atoms[u].pos2d;
        let nbrs = &self.mol.adj[u];
        if nbrs.is_empty() {
            return [p[0] + BOND_LEN_2D * (PI / 6.0).cos(), p[1] - BOND_LEN_2D * (PI / 6.0).sin()];
        }
        if nbrs.len() == 1 {
            let v = nbrs[0].0;
            let bi = nbrs[0].1;
            let pv = self.mol.atoms[v].pos2d;
            let in_ang = (p[1] - pv[1]).atan2(p[0] - pv[0]);
            if self.mol.bonds[bi].kekule_order == 3 || self.mol.atoms[u].hyb == Hybridization::Sp {
                return [p[0] + BOND_LEN_2D * in_ang.cos(), p[1] + BOND_LEN_2D * in_ang.sin()];
            }
            // Alternate +/- 60 deg from incoming vector for natural zig-zag alkane chains
            let turn = if u % 2 == 0 { PI / 3.0 } else { -PI / 3.0 };
            let out_ang = in_ang + turn;
            return [p[0] + BOND_LEN_2D * out_ang.cos(), p[1] + BOND_LEN_2D * out_ang.sin()];
        }

        // >= 2 neighbors: find largest angular gap around `u` and bisect it!
        let mut angles: Vec<f32> = nbrs
            .iter()
            .map(|&(v, _)| {
                let pv = self.mol.atoms[v].pos2d;
                (pv[1] - p[1]).atan2(pv[0] - p[0])
            })
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_gap = -1.0f32;
        let mut best_mid = 0.0f32;
        for i in 0..angles.len() {
            let a1 = angles[i];
            let a2 = if i + 1 < angles.len() {
                angles[i + 1]
            } else {
                angles[0] + TAU
            };
            let gap = a2 - a1;
            if gap > best_gap {
                best_gap = gap;
                best_mid = a1 + gap * 0.5;
            }
        }
        [p[0] + BOND_LEN_2D * best_mid.cos(), p[1] + BOND_LEN_2D * best_mid.sin()]
    }

    /// Evaluates the current sketched molecule in real time (`< 1 ms`), computing canonical SMILES,
    /// `QED`, `CNS-MPO`, `PAINS` alerts, live `Free-Wilson` R-group potency, property deltas vs.
    /// the pinned parent lead, and top `ECFP4` Tanimoto + `SALI` activity-cliff neighbors!
    pub fn evaluate_live_design(
        &self,
        ds: &DatasetAnalysis,
        active_sar_matrix: Option<&FreeWilsonMatrix>,
    ) -> LiveDesignReport {
        let canonical_smiles = molecule_to_smiles(&self.mol);
        let alerts = scan_structural_alerts(&self.mol);
        let mut alert_atom_set = BTreeSet::new();
        for al in &alerts {
            for &ai in &al.atoms {
                alert_atom_set.insert(ai);
            }
        }
        let alert_atoms: Vec<usize> = alert_atom_set.into_iter().collect();

        let mut profile = evaluate_physchem(&self.mol, alerts.len());

        let free_wilson = predict_best_free_wilson(
            &self.mol,
            active_sar_matrix,
            ds.default_sar_matrix.as_ref(),
            &ds.mols,
            &ds.profiles,
        );

        let sketch_fp = compute_ecfp(&self.mol, 2);

        // Rank dataset molecules by ECFP4 Tanimoto similarity
        let mut sim_ranked: Vec<(usize, f32)> = ds
            .fps
            .iter()
            .enumerate()
            .map(|(i, fp)| (i, sketch_fp.tanimoto(fp)))
            .collect();
        sim_ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (effective_pact, pact_source) = if let Some(ref fw) = free_wilson {
            if let Some((_, ref name, obs_p)) = fw.exact_training_match {
                (
                    obs_p,
                    format!("Exact Series Match ({name}: obs pAct {obs_p:.2}, FW {:.2})", fw.predicted_pact),
                )
            } else {
                (
                    fw.predicted_pact,
                    format!("Free-Wilson R-Group Model ({})", fw.core_name),
                )
            }
        } else {
            // Similarity-weighted 3-NN ECFP4 analog potency fallback
            let top3: Vec<(usize, f32)> = sim_ranked
                .iter()
                .copied()
                .filter(|&(_, t)| t > 0.15)
                .take(3)
                .collect();
            if top3.is_empty() {
                (self.mol.pact, "Baseline Fallback pAct".to_string())
            } else {
                let w_sum: f64 = top3.iter().map(|&(_, t)| (t as f64).powi(2)).sum::<f64>().max(1e-6);
                let p_est: f64 = top3
                    .iter()
                    .map(|&(mi, t)| ((t as f64).powi(2)) * ds.mols[mi].pact)
                    .sum::<f64>()
                    / w_sum;
                (p_est, "ECFP4 3-NN Similarity Estimate".to_string())
            }
        };

        let effective_ic50_nm = 10f64.powf(9.0 - effective_pact.clamp(2.0, 14.0));
        let effective_lle = effective_pact - profile.clogp;
        profile.lle = effective_lle;

        // Build top 8 SALI neighbors using `effective_pact`
        let mut sali_neighbors = Vec::new();
        let mut max_sali_cliff = 0.0f32;
        let mut cliff_count = 0usize;

        for &(mi, tan) in sim_ranked.iter().take(8) {
            let obs = ds.mols[mi].pact;
            let dp = effective_pact - obs;
            // If exact self-match (`tan > 0.999` and `|dp| < 0.02`), SALI is 0
            let sali = if tan > 0.999 && dp.abs() < 0.02 {
                0.0
            } else {
                (dp.abs() as f32) / (1.0 - tan + 0.005)
            };
            let is_cliff = tan >= ds.min_tanimoto && sali >= ds.sali_cliff_threshold;
            if is_cliff {
                cliff_count += 1;
                if sali > max_sali_cliff {
                    max_sali_cliff = sali;
                }
            }
            sali_neighbors.push(SaliNeighborHit {
                mol_idx: mi,
                compound_id: ds.mols[mi].id.clone(),
                compound_name: ds.mols[mi].name.clone(),
                target: ds.mols[mi].target.clone(),
                observed_pact: obs,
                tanimoto: tan,
                delta_pact: dp,
                prospective_sali: sali,
                is_cliff,
            });
        }

        let parent_delta = if let (Some(ref pm), Some(ref pp)) =
            (&self.pinned_parent_mol, &self.pinned_parent_profile)
        {
            let parent_fp = compute_ecfp(pm, 2);
            Some(ParentLeadDelta {
                parent_mol_idx: self.pinned_parent_idx,
                parent_name: pm.name.clone(),
                parent_pact: pm.pact,
                delta_mw: profile.mw - pp.mw,
                delta_clogp: profile.clogp - pp.clogp,
                delta_tpsa: profile.tpsa - pp.tpsa,
                delta_qed: profile.qed_mean - pp.qed_mean,
                delta_cns_mpo: profile.cns_mpo - pp.cns_mpo,
                delta_pact_est: effective_pact - pm.pact,
                tanimoto_to_parent: sketch_fp.tanimoto(&parent_fp),
            })
        } else {
            None
        };

        LiveDesignReport {
            canonical_smiles,
            profile,
            alerts,
            alert_atoms,
            free_wilson,
            effective_pact,
            effective_ic50_nm,
            effective_lle,
            pact_source,
            sali_neighbors,
            max_sali_cliff,
            cliff_count,
            parent_delta,
        }
    }

    fn push_undo_snapshot(&mut self) {
        if self.undo_stack.len() >= MAX_UNDO_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.mol.clone());
        self.redo_stack.clear();
    }

    fn push_raw_atom(&mut self, elem: Element, aromatic: bool, pos2d: [f32; 2]) -> usize {
        let idx = self.mol.atoms.len();
        self.mol.atoms.push(Atom {
            idx,
            elem,
            charge: 0,
            aromatic,
            bracketed: false,
            explicit_h: 0,
            implicit_h: 0,
            chiral: 0,
            in_ring: false,
            ring_sizes: Vec::new(),
            hyb: Hybridization::Sp3,
            pos2d,
        });
        idx
    }

    fn push_raw_bond(&mut self, a: usize, b: usize, order: BondOrder, aromatic: bool) {
        if a == b
            || self
                .mol
                .bonds
                .iter()
                .any(|eb| (eb.a == a && eb.b == b) || (eb.a == b && eb.b == a))
        {
            return;
        }
        let idx = self.mol.bonds.len();
        let kekule_order = match order {
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            _ => 1,
        };
        self.mol.bonds.push(Bond {
            idx,
            a,
            b,
            order,
            aromatic,
            in_ring: false,
            kekule_order,
        });
    }

    fn finish_mutation(&mut self, relayout: bool) {
        self.mol.sanitize();
        if relayout {
            compute_2d_coords(&mut self.mol);
        }
        self.mol.smiles = molecule_to_smiles(&self.mol);
        self.revision += 1;
    }
}

fn normalize2(v: [f32; 2]) -> [f32; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if l > 1e-4 {
        [v[0] / l, v[1] / l]
    } else {
        [1.0, 0.0]
    }
}

/// Generates a deterministic, canonical Weininger `CANGEN`-style OpenSMILES v1.0 string
/// (`https://pubs.acs.org/jcisd8/article/29/2/97/1079225/SMILES-2-Algorithm-for-generation-of-unique-SMILES`)
/// that round-trips identically through `parse_smiles`.
pub fn molecule_to_smiles(mol: &Molecule) -> String {
    let n = mol.atoms.len();
    if n == 0 {
        return String::new();
    }

    // Ignore explicit H nodes attached to a heavy atom (fold them into parent H count during emission);
    // keep isolated H or H2 if no heavy neighbor exists.
    let mut active = vec![true; n];
    let mut extra_h_on_heavy = vec![0u8; n];
    for u in 0..n {
        if mol.atoms[u].elem == Element::H && mol.adj[u].len() == 1 {
            let nb = mol.adj[u][0].0;
            if mol.atoms[nb].elem != Element::H {
                active[u] = false;
                extra_h_on_heavy[nb] = extra_h_on_heavy[nb].saturating_add(1);
            }
        }
    }

    // Step 1: Morgan / Weininger extended-connectivity symmetry ranks (`CANON`)
    let mut ranks = compute_canonical_atom_ranks(mol, &active, &extra_h_on_heavy);

    // Step 2: Partition into connected components and emit canonical DFS trees (`GENES`)
    let mut visited = vec![false; n];
    let mut comp_smiles: Vec<(usize, String)> = Vec::new();

    for seed in 0..n {
        if !active[seed] || visited[seed] {
            continue;
        }
        // Gather component atoms and pick lowest canonical rank as root
        let mut comp = Vec::new();
        let mut q = std::collections::VecDeque::new();
        visited[seed] = true;
        q.push_back(seed);
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &(v, _) in &mol.adj[u] {
                if active[v] && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        let root = *comp
            .iter()
            .min_by_key(|&&u| (ranks[u], u))
            .unwrap_or(&seed);

        let frag = emit_component_canonical_smiles(mol, root, &active, &extra_h_on_heavy, &mut ranks);
        let min_r = comp.iter().map(|&u| ranks[u]).min().unwrap_or(0);
        comp_smiles.push((min_r, frag));
    }

    comp_smiles.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    comp_smiles
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>()
        .join(".")
}

fn compute_canonical_atom_ranks(
    mol: &Molecule,
    active: &[bool],
    extra_h: &[u8],
) -> Vec<usize> {
    let n = mol.atoms.len();
    let mut init_keys: Vec<(usize, (u8, u8, u8, i8, u8, u8, u8))> = Vec::new();

    for u in 0..n {
        if !active[u] {
            continue;
        }
        let a = &mol.atoms[u];
        let deg = mol.adj[u].iter().filter(|&&(v, _)| active[v]).count() as u8;
        let bo2: u8 = mol.adj[u]
            .iter()
            .filter(|&&(v, _)| active[v])
            .map(|&(_, bi)| {
                if mol.bonds[bi].aromatic {
                    3u8
                } else {
                    mol.bonds[bi].kekule_order.saturating_mul(2)
                }
            })
            .sum();
        let th = a.total_h().saturating_add(extra_h[u]);
        let key = (
            deg,
            bo2,
            a.elem.atomic_number(),
            a.charge,
            th,
            a.in_ring as u8,
            a.aromatic as u8,
        );
        init_keys.push((u, key));
    }

    init_keys.sort_by(|a, b| a.1.cmp(&b.1));
    let mut ranks = vec![0usize; n];
    let mut cur_r = 0usize;
    for i in 0..init_keys.len() {
        if i == 0 || init_keys[i].1 != init_keys[i - 1].1 {
            cur_r += 1;
        }
        ranks[init_keys[i].0] = cur_r;
    }

    for _round in 0..8 {
        let mut aug: Vec<(usize, usize, Vec<(u8, usize)>)> = Vec::with_capacity(init_keys.len());
        for &(u, _) in &init_keys {
            let mut nbrs: Vec<(u8, usize)> = mol.adj[u]
                .iter()
                .filter(|&&(v, _)| active[v])
                .map(|&(v, bi)| {
                    let bcode = if mol.bonds[bi].aromatic {
                        4u8
                    } else {
                        mol.bonds[bi].kekule_order
                    };
                    (bcode, ranks[v])
                })
                .collect();
            nbrs.sort();
            aug.push((u, ranks[u], nbrs));
        }
        aug.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
        let mut next_r = 0usize;
        let mut new_ranks = vec![0usize; n];
        for i in 0..aug.len() {
            if i == 0 || aug[i].1 != aug[i - 1].1 || aug[i].2 != aug[i - 1].2 {
                next_r += 1;
            }
            new_ranks[aug[i].0] = next_r;
        }
        let prev_max = ranks.iter().copied().max().unwrap_or(0);
        ranks = new_ranks;
        if next_r == prev_max {
            break;
        }
    }

    ranks
}

fn emit_component_canonical_smiles(
    mol: &Molecule,
    root: usize,
    active: &[bool],
    extra_h: &[u8],
    ranks: &mut [usize],
) -> String {
    let n = mol.atoms.len();
    let mut disc = vec![0usize; n];
    let mut parent = vec![usize::MAX; n];
    let mut children: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    // Ring closure back-edges `(ancestor_v, descendant_u, bond_idx)`
    let mut closures: Vec<(usize, usize, usize)> = Vec::new();
    let mut timer = 0usize;

    #[allow(clippy::too_many_arguments)]
    fn dfs_build(
        u: usize,
        mol: &Molecule,
        active: &[bool],
        ranks: &[usize],
        timer: &mut usize,
        disc: &mut [usize],
        parent: &mut [usize],
        children: &mut [Vec<(usize, usize)>],
        closures: &mut Vec<(usize, usize, usize)>,
    ) {
        *timer += 1;
        disc[u] = *timer;

        let mut nbrs: Vec<(usize, usize)> = mol.adj[u]
            .iter()
            .copied()
            .filter(|&(v, _)| active[v])
            .collect();
        // Canonical neighbor ordering: lower rank first; prefer higher bond order on ties
        nbrs.sort_by_key(|&(v, bi)| {
            let b_prio = if mol.bonds[bi].aromatic {
                2u8
            } else {
                4u8.saturating_sub(mol.bonds[bi].kekule_order)
            };
            (ranks[v], b_prio, v)
        });

        for (v, bi) in nbrs {
            if disc[v] == 0 {
                parent[v] = u;
                children[u].push((v, bi));
                dfs_build(v, mol, active, ranks, timer, disc, parent, children, closures);
            } else if v != parent[u] && disc[v] < disc[u] {
                closures.push((v, u, bi));
            }
        }
    }

    dfs_build(
        root,
        mol,
        active,
        ranks,
        &mut timer,
        &mut disc,
        &mut parent,
        &mut children,
        &mut closures,
    );

    // Sort ring closures deterministically by `(disc[ancestor], disc[descendant])`
    closures.sort_by_key(|&(anc, desc, _)| (disc[anc], disc[desc]));

    // Attach `(ring_num, optional_bond_str)` tags to ancestor (opening) and descendant (closing)
    let mut ring_tags: Vec<Vec<(u16, &'static str)>> = vec![Vec::new(); n];
    for (ri, &(anc, desc, bi)) in closures.iter().enumerate() {
        let rnum = ((ri % 99) + 1) as u16;
        let b_str = smiles_bond_token(mol, bi);
        // Emit explicit bond token on ring-opening tag only (OpenSMILES allows either or both matching)
        ring_tags[anc].push((rnum, b_str));
        ring_tags[desc].push((rnum, ""));
    }

    fn emit_node(
        u: usize,
        in_bi: Option<usize>,
        mol: &Molecule,
        extra_h: &[u8],
        children: &[Vec<(usize, usize)>],
        ring_tags: &[Vec<(u16, &'static str)>],
        out: &mut String,
    ) {
        if let Some(bi) = in_bi {
            out.push_str(smiles_bond_token(mol, bi));
        }
        out.push_str(&format_smiles_atom(mol, u, extra_h[u]));

        for &(rnum, b_tok) in &ring_tags[u] {
            out.push_str(b_tok);
            if rnum < 10 {
                out.push_str(&rnum.to_string());
            } else {
                out.push_str(&format!("%{rnum:02}"));
            }
        }

        let ch = &children[u];
        for (i, &(v, bi)) in ch.iter().enumerate() {
            let is_branch = i + 1 < ch.len();
            if is_branch {
                out.push('(');
            }
            emit_node(v, Some(bi), mol, extra_h, children, ring_tags, out);
            if is_branch {
                out.push(')');
            }
        }
    }

    let mut s = String::new();
    emit_node(root, None, mol, extra_h, &children, &ring_tags, &mut s);
    s
}

fn smiles_bond_token(mol: &Molecule, bi: usize) -> &'static str {
    let b = &mol.bonds[bi];
    let both_arom = mol.atoms[b.a].aromatic && mol.atoms[b.b].aromatic;
    if b.aromatic {
        ""
    } else {
        match b.kekule_order {
            3 => "#",
            2 => "=",
            _ => {
                if both_arom {
                    // Explicit `-` required between two aromatic atoms across a non-aromatic biaryl single bond!
                    "-"
                } else {
                    ""
                }
            }
        }
    }
}

fn format_smiles_atom(mol: &Molecule, u: usize, extra_h: u8) -> String {
    let a = &mol.atoms[u];
    let total_h = a.total_h().saturating_add(extra_h);

    let in_organic_subset = matches!(
        a.elem,
        Element::B
            | Element::C
            | Element::N
            | Element::O
            | Element::P
            | Element::S
            | Element::F
            | Element::Cl
            | Element::Br
            | Element::I
    );

    // Check what implicit H `mol.sanitize()` would assign to an unbracketed organic token
    let bond_kek_sum: u8 = mol.adj[u]
        .iter()
        .filter(|&&(v, _)| mol.atoms[v].elem != Element::H)
        .map(|&(_, bi)| mol.bonds[bi].kekule_order)
        .sum();
    let default_target = a
        .elem
        .default_valences()
        .iter()
        .copied()
        .find(|&v| v >= bond_kek_sum)
        .unwrap_or(bond_kek_sum);
    let expected_unbracketed_h = default_target.saturating_sub(bond_kek_sum);

    // Aromatic heteroatoms carrying a hydrogen (e.g. pyrrole/indole `[nH]`) MUST always be bracketed in OpenSMILES!
    let arom_hetero_with_h = a.aromatic && a.elem != Element::C && total_h > 0;
    let arom_unsupported = a.aromatic
        && !matches!(
            a.elem,
            Element::B | Element::C | Element::N | Element::O | Element::P | Element::S
        );

    let can_emit_bare = in_organic_subset
        && a.charge == 0
        && a.chiral == 0
        && !arom_hetero_with_h
        && !arom_unsupported
        && total_h == expected_unbracketed_h;

    if can_emit_bare {
        return if a.aromatic {
            match a.elem {
                Element::B => "b".to_string(),
                Element::C => "c".to_string(),
                Element::N => "n".to_string(),
                Element::O => "o".to_string(),
                Element::P => "p".to_string(),
                Element::S => "s".to_string(),
                _ => a.elem.symbol().to_string(),
            }
        } else {
            a.elem.symbol().to_string()
        };
    }

    // Emit explicit OpenSMILES bracket atom `[sym(@/@@)(H/Hk)(+/-)]`
    let mut out = String::from("[");
    let sym = if a.aromatic {
        match a.elem {
            Element::B => "b",
            Element::C => "c",
            Element::N => "n",
            Element::O => "o",
            Element::P => "p",
            Element::S => "s",
            Element::Se => "se",
            _ => a.elem.symbol(),
        }
    } else {
        a.elem.symbol()
    };
    out.push_str(sym);

    if a.chiral == 1 {
        out.push('@');
    } else if a.chiral == 2 {
        out.push_str("@@");
    }

    if total_h == 1 {
        out.push('H');
    } else if total_h > 1 {
        out.push_str(&format!("H{total_h}"));
    }

    if a.charge == 1 {
        out.push('+');
    } else if a.charge > 1 {
        out.push_str(&format!("+{}", a.charge));
    } else if a.charge == -1 {
        out.push('-');
    } else if a.charge < -1 {
        out.push_str(&format!("{}", a.charge));
    }

    out.push(']');
    out
}
