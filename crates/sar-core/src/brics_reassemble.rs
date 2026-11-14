//! Combinatorial Forward `BRICS` Fragment Recombiner & Multi-Objective Virtual Hybrid Scorer.
//!
//! Dissects active lead compounds (`pAct >= min_lead_pact`) across canonical `L1`–`L16` `BRICS`
//! retrosynthetic bonds, combinatorially recombines compatible scaffold and substituent synthons
//! across distinct parent leads (both 2-parent `Scaffold(A) + Donor(B)` and 3-parent
//! `Core(A) + Donor(B) + Donor(C)` chimeras), deduplicates against known dataset molecules via
//! Weininger `CANGEN` OpenSMILES, and ranks every virtual hybrid in parallel across:
//! 1. **Ridge `Free-Wilson` `pAct`** (with `ECFP4` + `2D Pharmacophore` 3-NN fallback for novel cores),
//! 2. **Bickerton `QED`** & **Wager `CNS-MPO`** drug-likeness,
//! 3. **Ertl & Schuffenhauer `SAScore`** synthetic accessibility (`1.0` easy to `10.0` difficult),
//! 4. **Quantitative `ADMET`** (`ESOL LogS`, `BOILED-Egg`, `Pfizer 3/75`, `PAINS`), and
//! 5. **3D Non-Dominated Pareto Optimality** (`Free-Wilson pAct` MAX, `QED` MAX, `SAScore` MIN).
//!
//! Primary Literature Grounding:
//! - Degen, Wegscheid-Gerlach, Zaliani & Rarey (2008), *ChemMedChem* 3(10), 1503–1507
//!   (`https://pubmed.ncbi.nlm.nih.gov/18792903/`): `BRICS` `L1`–`L16` fragment spaces & recombination.
//! - Free & Wilson (1964), *J. Med. Chem.* 7, 395–399 (`https://pubmed.ncbi.nlm.nih.gov/14221113/`).
//! - Bickerton et al. (2012), *Nat. Chem.* 4, 90–98 (`https://pubmed.ncbi.nlm.nih.gov/22270643/`): `QED`.
//! - Ertl & Schuffenhauer (2009), *J. Cheminform.* 1:8 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3225829/`): `SAScore`.

use crate::admet::{evaluate_admet, AdmetProfile};
use crate::alerts::scan_structural_alerts;
use crate::fingerprint::compute_ecfp;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Molecule};
use crate::pharmacophore::{align_pharmacophores, extract_pharmacophore_map};
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::rgroup_table::{
    build_free_wilson_for_smarts, predict_free_wilson_for_molecule, FreeWilsonMatrix,
    FreeWilsonPrediction, CURATED_CORE_TEMPLATES,
};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::{
    classify_brics_atom_env, evaluate_synthetic_accessibility, match_brics_reaction,
    SynthAccessProfile,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// User-configurable parameters controlling cross-lead `BRICS` recombination & multi-objective scoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BricsRecombineConfig {
    /// `"All"` (or empty) crosses leads within each target series; otherwise restricts to this target.
    pub target_filter: String,
    /// Minimum bioactivity (`pAct`) required for a dataset compound to contribute `BRICS` synthons.
    pub min_lead_pact: f64,
    /// Maximum allowed Ertl `SAScore` (`[1.0, 10.0]`) for retained virtual hybrids.
    pub max_sa_score: f64,
    /// Minimum required Bickerton `QED` (`[0.0, 1.0]`) for retained virtual hybrids.
    pub min_qed: f64,
    /// Suppress virtual hybrids triggering any Baell-Holloway `PAINS` or reactive warhead alert.
    pub hide_pains: bool,
    /// Perform a second-stage `Core(A) + Donor(B) + Donor(C)` 3-component recombination pass.
    pub include_three_component: bool,
    /// Allow cross-target polypharmacology / privileged-fragment chimeras when `target_filter == "All"`.
    pub allow_cross_target: bool,
    /// Maximum number of top-ranked virtual hybrids returned.
    pub max_hybrids: usize,
    /// Multi-objective utility weight on `predicted_pact`.
    pub weight_pact: f64,
    /// Multi-objective utility weight on `3.0 * QED`.
    pub weight_qed: f64,
    /// Multi-objective utility penalty weight on `0.45 * SAScore`.
    pub weight_sa: f64,
}

impl Default for BricsRecombineConfig {
    fn default() -> Self {
        Self {
            target_filter: "All".to_string(),
            min_lead_pact: 6.5,
            max_sa_score: 5.5,
            min_qed: 0.20,
            hide_pains: true,
            include_three_component: true,
            allow_cross_target: false,
            max_hybrids: 80,
            weight_pact: 1.00,
            weight_qed: 1.00,
            weight_sa: 1.00,
        }
    }
}

/// Newly formed forward `BRICS` coupling bond inside a `BricsVirtualHybrid`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridCouplingBond {
    pub atom_a: usize,
    pub atom_b: usize,
    pub pair_code: String,
    pub reaction_name: String,
}

/// Scored de novo virtual hybrid assembled by crossing `BRICS` synthons from `2` or `3` active leads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BricsVirtualHybrid {
    pub hybrid_id: String,
    pub hybrid_name: String,
    pub target: String,
    /// `2` for `Scaffold(A) + Donor(B)`, `3` for `Core(A) + Donor(B) + Donor(C)`.
    pub component_count: usize,
    pub scaffold_parent_idx: usize,
    pub scaffold_parent_name: String,
    pub donor_parent_indices: Vec<usize>,
    pub donor_parent_names: Vec<String>,
    /// Atom indices in `mol` inherited from Scaffold Parent A (colored sky-blue in 2D chimera view).
    pub scaffold_atoms: Vec<usize>,
    /// Atom indices in `mol` grafted from Donor Parent(s) B/C (colored emerald in 2D chimera view).
    pub donor_atoms: Vec<usize>,
    /// Newly formed forward `BRICS` coupling bond(s) joining scaffold and donor synthons.
    pub formed_bonds: Vec<HybridCouplingBond>,
    pub canonical_smiles: String,
    pub mol: Molecule,
    pub profile: PhyschemProfile,
    pub admet: AdmetProfile,
    pub sa_profile: SynthAccessProfile,
    pub free_wilson: Option<FreeWilsonPrediction>,
    pub predicted_pact: f64,
    pub predicted_ic50_nm: f64,
    pub predicted_lle: f64,
    pub pact_method: String,
    pub pains_count: usize,
    /// Signed potency gain relative to `scaffold_parent_idx`.
    pub delta_pact_vs_scaffold: f64,
    /// Signed `QED` gain relative to `scaffold_parent_idx`.
    pub delta_qed_vs_scaffold: f64,
    /// Signed `SAScore` change relative to `scaffold_parent_idx` (negative = easier to synthesize).
    pub delta_sa_vs_scaffold: f64,
    /// `true` if non-dominated across `(predicted_pact MAX, QED MAX, SAScore MIN)` with `0` PAINS alerts.
    pub pareto_front: bool,
    /// Weighted multi-objective score combining `Free-Wilson pAct`, `QED`, and `-SAScore`.
    pub utility_score: f64,
}

/// Summary report returned alongside a combinatorial `BRICS` recombination run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BricsRecombineSummary {
    pub active_leads_used: usize,
    pub unique_half_synthons: usize,
    pub raw_valid_chimeras: usize,
    pub novel_deduplicated: usize,
    pub retained_after_filters: usize,
    pub pareto_hybrid_count: usize,
}

/// Internal half-molecule synthon produced by cleaving a single non-ring `BRICS` bond.
#[derive(Clone)]
struct HalfSynthon {
    parent_mol_idx: usize,
    parent_name: String,
    parent_target: String,
    parent_pact: f64,
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
    anchor_atom: usize,
    brics_env: &'static str,
    bond_order: BondOrder,
    is_core_like: bool,
    dedup_key: String,
}

#[derive(Clone)]
struct RawHybridGraph {
    target: String,
    component_count: usize,
    scaffold_parent_idx: usize,
    scaffold_parent_name: String,
    donor_parent_indices: Vec<usize>,
    donor_parent_names: Vec<String>,
    scaffold_atoms: Vec<usize>,
    donor_atoms: Vec<usize>,
    formed_bonds: Vec<HybridCouplingBond>,
    canonical_smiles: String,
    mol: Molecule,
}

fn connected_component_excluding_bond(
    mol: &Molecule,
    start: usize,
    excluded_bond: usize,
) -> Vec<usize> {
    let mut visited = vec![false; mol.atoms.len()];
    let mut comp = Vec::new();
    let mut q = VecDeque::new();
    visited[start] = true;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        if mol.atoms[u].elem == Element::H {
            continue;
        }
        comp.push(u);
        for &(v, bi) in &mol.adj[u] {
            if bi == excluded_bond || mol.atoms[v].elem == Element::H {
                continue;
            }
            if !visited[v] {
                visited[v] = true;
                q.push_back(v);
            }
        }
    }
    comp
}

fn build_half_synthon(
    mol: &Molecule,
    mol_idx: usize,
    comp_atoms: &[usize],
    old_anchor: usize,
    brics_env: &'static str,
    bond_order: BondOrder,
) -> HalfSynthon {
    let mut old_to_new = vec![usize::MAX; mol.atoms.len()];
    let mut atoms = Vec::with_capacity(comp_atoms.len());
    let mut has_ring = false;

    for (ni, &ou) in comp_atoms.iter().enumerate() {
        old_to_new[ou] = ni;
        let sa = &mol.atoms[ou];
        if sa.in_ring {
            has_ring = true;
        }
        atoms.push(Atom {
            idx: ni,
            elem: sa.elem,
            charge: sa.charge,
            aromatic: sa.aromatic,
            bracketed: sa.charge != 0 || sa.chiral != 0,
            explicit_h: 0,
            implicit_h: 0,
            chiral: sa.chiral,
            in_ring: sa.in_ring,
            ring_sizes: Vec::new(),
            hyb: sa.hyb,
            pos2d: [0.0, 0.0],
        });
    }

    let mut bonds = Vec::new();
    for b in &mol.bonds {
        let na = old_to_new[b.a];
        let nb = old_to_new[b.b];
        if na != usize::MAX && nb != usize::MAX {
            let bi = bonds.len();
            bonds.push(Bond {
                idx: bi,
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: b.in_ring,
                kekule_order: b.kekule_order,
            });
        }
    }

    let anchor_atom = old_to_new[old_anchor];
    let is_core_like = has_ring && atoms.len() >= 7;

    let mut tmp = Molecule::new("");
    tmp.atoms = atoms.clone();
    tmp.bonds = bonds.clone();
    tmp.sanitize();
    let capped_smi = molecule_to_smiles(&tmp);
    let anchor_deg = tmp.adj.get(anchor_atom).map(|v| v.len()).unwrap_or(0);
    let dedup_key = format!(
        "{}|{}|{}{}|{}",
        capped_smi,
        brics_env,
        atoms[anchor_atom].elem.symbol(),
        anchor_deg,
        bond_order.numeric_order()
    );

    HalfSynthon {
        parent_mol_idx: mol_idx,
        parent_name: mol.name.clone(),
        parent_target: mol.target.clone(),
        parent_pact: mol.pact,
        atoms,
        bonds,
        anchor_atom,
        brics_env,
        bond_order,
        is_core_like,
        dedup_key,
    }
}

fn splice_two_half_synthons(
    core: &HalfSynthon,
    donor: &HalfSynthon,
    rxn_name: &str,
) -> Option<(Molecule, Vec<usize>, Vec<usize>, HybridCouplingBond)> {
    let total_heavy = core.atoms.len() + donor.atoms.len();
    if !(12..=48).contains(&total_heavy) {
        return None;
    }

    let mut mol = Molecule::new("");
    mol.target = core.parent_target.clone();
    mol.source_url = "https://pubmed.ncbi.nlm.nih.gov/18792903/".to_string();

    let mut scaffold_atoms = Vec::with_capacity(core.atoms.len());
    for (i, a) in core.atoms.iter().enumerate() {
        let mut ca = a.clone();
        ca.idx = i;
        mol.atoms.push(ca);
        scaffold_atoms.push(i);
    }

    let offset = core.atoms.len();
    let mut donor_atoms = Vec::with_capacity(donor.atoms.len());
    for (j, a) in donor.atoms.iter().enumerate() {
        let idx = offset + j;
        let mut da = a.clone();
        da.idx = idx;
        mol.atoms.push(da);
        donor_atoms.push(idx);
    }

    for b in &core.bonds {
        let bi = mol.bonds.len();
        mol.bonds.push(Bond {
            idx: bi,
            a: b.a,
            b: b.b,
            order: b.order,
            aromatic: b.aromatic,
            in_ring: false,
            kekule_order: b.kekule_order,
        });
    }
    for b in &donor.bonds {
        let bi = mol.bonds.len();
        mol.bonds.push(Bond {
            idx: bi,
            a: offset + b.a,
            b: offset + b.b,
            order: b.order,
            aromatic: b.aromatic,
            in_ring: false,
            kekule_order: b.kekule_order,
        });
    }

    let u = core.anchor_atom;
    let v = offset + donor.anchor_atom;
    let kekule_order = match core.bond_order {
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        _ => 1,
    };
    let join_bi = mol.bonds.len();
    mol.bonds.push(Bond {
        idx: join_bi,
        a: u,
        b: v,
        order: core.bond_order,
        aromatic: false,
        in_ring: false,
        kekule_order,
    });

    mol.sanitize();
    if mol.molecular_weight() > 680.0 {
        return None;
    }

    let coupling = HybridCouplingBond {
        atom_a: u,
        atom_b: v,
        pair_code: format!("{}+{}", core.brics_env, donor.brics_env),
        reaction_name: rxn_name.to_string(),
    };
    Some((mol, scaffold_atoms, donor_atoms, coupling))
}

/// Attempts a second-stage `BRICS` substitution on a 2-parent hybrid (`Pass 1`), replacing a second
/// untouched peripheral substituent on the Lead A scaffold with a compatible donor synthon from a
/// third distinct active lead `C` (`Core(A) + Donor(B) + Donor(C)`).
fn generate_three_component_chimeras(
    seed_hybrids: &[RawHybridGraph],
    donor_pool: &[HalfSynthon],
    existing_smiles: &BTreeSet<String>,
    seen_hybrids: &mut BTreeSet<String>,
    allow_cross_target: bool,
    max_extra: usize,
) -> Vec<RawHybridGraph> {
    let mut out = Vec::new();

    for seed in seed_hybrids.iter().take(24) {
        if out.len() >= max_extra {
            break;
        }
        let scaf_set: BTreeSet<usize> = seed.scaffold_atoms.iter().copied().collect();
        let first_join_pair = seed
            .formed_bonds
            .first()
            .map(|fb| (fb.atom_a.min(fb.atom_b), fb.atom_a.max(fb.atom_b)));

        for (bi, b) in seed.mol.bonds.iter().enumerate() {
            if out.len() >= max_extra {
                break;
            }
            if b.in_ring || b.aromatic {
                continue;
            }
            let pair_uv = (b.a.min(b.b), b.a.max(b.b));
            if Some(pair_uv) == first_join_pair {
                continue;
            }
            if !scaf_set.contains(&b.a) || !scaf_set.contains(&b.b) {
                continue;
            }
            let Some(ea) = classify_brics_atom_env(&seed.mol, b.a, b.b, b) else {
                continue;
            };
            let Some(eb) = classify_brics_atom_env(&seed.mol, b.b, b.a, b) else {
                continue;
            };
            if match_brics_reaction(ea, eb).is_none() {
                continue;
            }

            let comp_a = connected_component_excluding_bond(&seed.mol, b.a, bi);
            let comp_b = connected_component_excluding_bond(&seed.mol, b.b, bi);
            if comp_a.len() < 2 || comp_b.len() < 2 {
                continue;
            }

            let b0 = match seed.donor_atoms.first() {
                Some(&d0) => d0,
                None => continue,
            };
            let (keep_comp, keep_anchor, keep_env) = if comp_a.contains(&b0) {
                (&comp_a, b.a, ea)
            } else if comp_b.contains(&b0) {
                (&comp_b, b.b, eb)
            } else {
                continue;
            };

            if keep_comp.len() < 8 {
                continue;
            }

            for donor_c in donor_pool.iter().filter(|d| !d.is_core_like || d.atoms.len() <= 14) {
                if out.len() >= max_extra {
                    break;
                }
                if !allow_cross_target && donor_c.parent_target != seed.target {
                    continue;
                }
                if donor_c.parent_mol_idx == seed.scaffold_parent_idx
                    || seed.donor_parent_indices.contains(&donor_c.parent_mol_idx)
                {
                    continue;
                }
                if donor_c.bond_order != b.order {
                    continue;
                }
                let Some((rxn2, _)) = match_brics_reaction(keep_env, donor_c.brics_env) else {
                    continue;
                };
                if keep_comp.len() + donor_c.atoms.len() > 48 {
                    continue;
                }

                let mut old_to_new = vec![usize::MAX; seed.mol.atoms.len()];
                let mut mol3 = Molecule::new("");
                mol3.target = seed.target.clone();
                mol3.source_url = "https://pubmed.ncbi.nlm.nih.gov/18792903/".to_string();

                let mut new_scaf_atoms = Vec::new();
                let mut new_donor_atoms = Vec::new();
                for (ni, &ou) in keep_comp.iter().enumerate() {
                    old_to_new[ou] = ni;
                    let mut ca = seed.mol.atoms[ou].clone();
                    ca.idx = ni;
                    ca.explicit_h = 0;
                    ca.implicit_h = 0;
                    mol3.atoms.push(ca);
                    if scaf_set.contains(&ou) {
                        new_scaf_atoms.push(ni);
                    } else {
                        new_donor_atoms.push(ni);
                    }
                }

                let off = mol3.atoms.len();
                for (j, a) in donor_c.atoms.iter().enumerate() {
                    let idx = off + j;
                    let mut da = a.clone();
                    da.idx = idx;
                    mol3.atoms.push(da);
                    new_donor_atoms.push(idx);
                }

                for sb in &seed.mol.bonds {
                    let na = old_to_new[sb.a];
                    let nb = old_to_new[sb.b];
                    if na != usize::MAX && nb != usize::MAX {
                        let nbi = mol3.bonds.len();
                        mol3.bonds.push(Bond {
                            idx: nbi,
                            a: na,
                            b: nb,
                            order: sb.order,
                            aromatic: sb.aromatic,
                            in_ring: false,
                            kekule_order: sb.kekule_order,
                        });
                    }
                }
                for db in &donor_c.bonds {
                    let nbi = mol3.bonds.len();
                    mol3.bonds.push(Bond {
                        idx: nbi,
                        a: off + db.a,
                        b: off + db.b,
                        order: db.order,
                        aromatic: db.aromatic,
                        in_ring: false,
                        kekule_order: db.kekule_order,
                    });
                }

                let u_join = old_to_new[keep_anchor];
                let v_join = off + donor_c.anchor_atom;
                let k_ord = match b.order {
                    BondOrder::Double => 2,
                    BondOrder::Triple => 3,
                    _ => 1,
                };
                let nbi = mol3.bonds.len();
                mol3.bonds.push(Bond {
                    idx: nbi,
                    a: u_join,
                    b: v_join,
                    order: b.order,
                    aromatic: false,
                    in_ring: false,
                    kekule_order: k_ord,
                });

                mol3.sanitize();
                if mol3.molecular_weight() > 680.0 {
                    continue;
                }
                let can_smi = molecule_to_smiles(&mol3);
                if can_smi.is_empty()
                    || existing_smiles.contains(&can_smi)
                    || !seen_hybrids.insert(can_smi.clone())
                {
                    continue;
                }
                mol3.smiles = can_smi.clone();

                let mut formed = Vec::new();
                for fb in &seed.formed_bonds {
                    let na = old_to_new[fb.atom_a];
                    let nb = old_to_new[fb.atom_b];
                    if na != usize::MAX && nb != usize::MAX {
                        formed.push(HybridCouplingBond {
                            atom_a: na,
                            atom_b: nb,
                            pair_code: fb.pair_code.clone(),
                            reaction_name: fb.reaction_name.clone(),
                        });
                    }
                }
                formed.push(HybridCouplingBond {
                    atom_a: u_join,
                    atom_b: v_join,
                    pair_code: format!("{}+{}", keep_env, donor_c.brics_env),
                    reaction_name: rxn2.to_string(),
                });

                let mut d_indices = seed.donor_parent_indices.clone();
                d_indices.push(donor_c.parent_mol_idx);
                let mut d_names = seed.donor_parent_names.clone();
                d_names.push(donor_c.parent_name.clone());

                out.push(RawHybridGraph {
                    target: seed.target.clone(),
                    component_count: 3,
                    scaffold_parent_idx: seed.scaffold_parent_idx,
                    scaffold_parent_name: seed.scaffold_parent_name.clone(),
                    donor_parent_indices: d_indices,
                    donor_parent_names: d_names,
                    scaffold_atoms: new_scaf_atoms,
                    donor_atoms: new_donor_atoms,
                    formed_bonds: formed,
                    canonical_smiles: can_smi,
                    mol: mol3,
                });
            }
        }
    }
    out
}

#[inline]
fn compute_utility(pact: f64, qed: f64, sa: f64, pains: usize, cfg: &BricsRecombineConfig) -> f64 {
    cfg.weight_pact * pact + cfg.weight_qed * (3.0 * qed)
        - cfg.weight_sa * (0.45 * sa)
        - (pains as f64) * 1.50
}

/// Recomputes 3D `(Free-Wilson pAct MAX, QED MAX, SAScore MIN)` Pareto optimality and re-sorts
/// `hybrids` whenever the user adjusts objective weights (`weight_pact`, `weight_qed`, `weight_sa`).
pub fn rescore_and_sort_hybrids(
    hybrids: &mut [BricsVirtualHybrid],
    cfg: &BricsRecombineConfig,
) -> usize {
    for h in hybrids.iter_mut() {
        h.utility_score = compute_utility(
            h.predicted_pact,
            h.profile.qed_mean,
            h.sa_profile.sa_score,
            h.pains_count,
            cfg,
        );
        h.pareto_front = false;
    }

    let n = hybrids.len();
    let mut pareto_cnt = 0usize;
    for i in 0..n {
        if hybrids[i].pains_count > 0 {
            continue;
        }
        let pi = hybrids[i].predicted_pact;
        let qi = hybrids[i].profile.qed_mean;
        let si = hybrids[i].sa_profile.sa_score;
        let mut dominated = false;
        for j in 0..n {
            if i == j || hybrids[j].pains_count > 0 {
                continue;
            }
            let pj = hybrids[j].predicted_pact;
            let qj = hybrids[j].profile.qed_mean;
            let sj = hybrids[j].sa_profile.sa_score;
            let no_worse = pj >= pi - 1e-6 && qj >= qi - 1e-6 && sj <= si + 1e-6;
            let strictly_better = pj > pi + 1e-5 || qj > qi + 1e-5 || sj < si - 1e-5;
            if no_worse && strictly_better {
                dominated = true;
                break;
            }
        }
        if !dominated {
            hybrids[i].pareto_front = true;
            pareto_cnt += 1;
        }
    }

    hybrids.sort_by(|a, b| {
        b.utility_score
            .partial_cmp(&a.utility_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rk, h) in hybrids.iter_mut().enumerate() {
        h.hybrid_id = format!("BRICS-HYB-{:03}", rk + 1);
    }
    pareto_cnt
}

/// Dissects active leads in `ds` (`pAct >= cfg.min_lead_pact`) into `BRICS` half-synthons,
/// crosses compatible scaffold + substituent fragments across distinct parent leads, and scores
/// every virtual hybrid in parallel across `Free-Wilson pAct`, `QED`, `SAScore`, and `ADMET`.
pub fn recombine_brics_leads(
    ds: &DatasetAnalysis,
    active_sar_matrix: Option<&FreeWilsonMatrix>,
    cfg: &BricsRecombineConfig,
) -> (Vec<BricsVirtualHybrid>, BricsRecombineSummary) {
    let mut summary = BricsRecombineSummary::default();
    if ds.mols.len() < 2 {
        return (Vec::new(), summary);
    }

    let tf_trimmed = cfg.target_filter.trim();
    let filter_target = !tf_trimmed.is_empty() && !tf_trimmed.eq_ignore_ascii_case("All");

    // Collect top active leads (`<= 48` highest-potency leads to bound combinatorial explosion on 50k HTS)
    let mut lead_indices: Vec<usize> = (0..ds.mols.len())
        .filter(|&i| {
            let m = &ds.mols[i];
            if m.pact < cfg.min_lead_pact {
                return false;
            }
            if filter_target && m.target != tf_trimmed {
                return false;
            }
            true
        })
        .collect();
    lead_indices.sort_by(|&a, &b| {
        ds.mols[b]
            .pact
            .partial_cmp(&ds.mols[a].pact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    lead_indices.truncate(48);
    summary.active_leads_used = lead_indices.len();
    if lead_indices.len() < 2 {
        return (Vec::new(), summary);
    }

    // 1. Extract deduplicated BRICS HalfSynthons across all selected active leads
    let mut half_synthons: Vec<HalfSynthon> = Vec::new();
    let mut seen_synthon_keys: BTreeSet<String> = BTreeSet::new();

    for &mi in &lead_indices {
        let m = &ds.mols[mi];
        for (bi, b) in m.bonds.iter().enumerate() {
            if b.in_ring || b.aromatic {
                continue;
            }
            let Some(ea) = classify_brics_atom_env(m, b.a, b.b, b) else {
                continue;
            };
            let Some(eb) = classify_brics_atom_env(m, b.b, b.a, b) else {
                continue;
            };
            if match_brics_reaction(ea, eb).is_none() {
                continue;
            }

            let comp_a = connected_component_excluding_bond(m, b.a, bi);
            let comp_b = connected_component_excluding_bond(m, b.b, bi);
            if comp_a.len() < 2 || comp_b.len() < 2 {
                continue;
            }

            let hs_a = build_half_synthon(m, mi, &comp_a, b.a, ea, b.order);
            if seen_synthon_keys.insert(hs_a.dedup_key.clone()) {
                half_synthons.push(hs_a);
            }
            let hs_b = build_half_synthon(m, mi, &comp_b, b.b, eb, b.order);
            if seen_synthon_keys.insert(hs_b.dedup_key.clone()) {
                half_synthons.push(hs_b);
            }
        }
    }
    summary.unique_half_synthons = half_synthons.len();

    // Sort cores & donors by parent potency descending
    half_synthons.sort_by(|a, b| {
        b.parent_pact
            .partial_cmp(&a.parent_pact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let existing_smiles: BTreeSet<String> = ds.mols.iter().map(molecule_to_smiles).collect();
    let mut seen_hybrids: BTreeSet<String> = BTreeSet::new();
    let mut raw_candidates: Vec<RawHybridGraph> = Vec::new();
    let raw_cap = (cfg.max_hybrids * 4).clamp(120, 360);

    // 2. Pass 1: Cross Core-like HalfSynthons (Lead A) with Compatible Donor HalfSynthons (Lead B != Lead A)
    let core_pool: Vec<&HalfSynthon> = half_synthons.iter().filter(|h| h.is_core_like).collect();
    for &core in &core_pool {
        if raw_candidates.len() >= raw_cap {
            break;
        }
        for donor in &half_synthons {
            if raw_candidates.len() >= raw_cap {
                break;
            }
            if donor.parent_mol_idx == core.parent_mol_idx {
                continue;
            }
            if !cfg.allow_cross_target && donor.parent_target != core.parent_target {
                continue;
            }
            if donor.bond_order != core.bond_order {
                continue;
            }
            // Prefer core + substituent/ring-arm rather than fusing two giant 25-atom drugs
            if donor.atoms.len() > core.atoms.len() + 4 {
                continue;
            }
            let Some((rxn_name, _)) = match_brics_reaction(core.brics_env, donor.brics_env) else {
                continue;
            };
            let Some((mut hyb_mol, scaf_atoms, don_atoms, formed)) =
                splice_two_half_synthons(core, donor, rxn_name)
            else {
                continue;
            };
            summary.raw_valid_chimeras += 1;

            let can_smi = molecule_to_smiles(&hyb_mol);
            if can_smi.is_empty()
                || existing_smiles.contains(&can_smi)
                || !seen_hybrids.insert(can_smi.clone())
            {
                continue;
            }
            hyb_mol.smiles = can_smi.clone();

            raw_candidates.push(RawHybridGraph {
                target: core.parent_target.clone(),
                component_count: 2,
                scaffold_parent_idx: core.parent_mol_idx,
                scaffold_parent_name: core.parent_name.clone(),
                donor_parent_indices: vec![donor.parent_mol_idx],
                donor_parent_names: vec![donor.parent_name.clone()],
                scaffold_atoms: scaf_atoms,
                donor_atoms: don_atoms,
                formed_bonds: vec![formed],
                canonical_smiles: can_smi,
                mol: hyb_mol,
            });
        }
    }

    // 3. Pass 2: Convergent 3-Component (`Core(A) + Donor(B) + Donor(C)`) Double-Swap Chimeras
    if cfg.include_three_component && !raw_candidates.is_empty() {
        let extra = generate_three_component_chimeras(
            &raw_candidates,
            &half_synthons,
            &existing_smiles,
            &mut seen_hybrids,
            cfg.allow_cross_target,
            (cfg.max_hybrids * 3) / 2,
        );
        summary.raw_valid_chimeras += extra.len();
        raw_candidates.extend(extra);
    }

    summary.novel_deduplicated = raw_candidates.len();
    if raw_candidates.is_empty() {
        return (Vec::new(), summary);
    }

    // Pre-solve Free-Wilson matrices across all curated medicinal-chemistry templates once
    // so EGFR, Dopamine D2, and COX-2 hybrids all receive exact Ridge Free-Wilson QSAR scoring!
    let mut fw_models: Vec<FreeWilsonMatrix> = Vec::new();
    if let Some(am) = active_sar_matrix {
        fw_models.push(am.clone());
    }
    if let Some(dm) = &ds.default_sar_matrix {
        if fw_models.iter().all(|m| m.core_smarts != dm.core_smarts) {
            fw_models.push(dm.clone());
        }
    }
    for tpl in &CURATED_CORE_TEMPLATES {
        if fw_models.iter().any(|m| m.core_smarts == tpl.smarts) {
            continue;
        }
        let tf = (!tpl.target_hint.is_empty()).then_some(tpl.target_hint);
        if let Ok(mat) =
            build_free_wilson_for_smarts(&ds.mols, &ds.profiles, tpl.name, tpl.smarts, tf, 3, 0.15)
        {
            if mat.rows.len() >= 2 {
                fw_models.push(mat);
            }
        }
    }

    // 4. Parallel Rayon Multi-Objective Evaluation (`Free-Wilson pAct`, `QED`, `SAScore`, `ADMET`)
    let mut scored: Vec<BricsVirtualHybrid> = raw_candidates
        .into_par_iter()
        .filter_map(|mut raw| {
            compute_2d_coords(&mut raw.mol);
            let alerts = scan_structural_alerts(&raw.mol);
            let pains_count = alerts.len();
            if cfg.hide_pains && pains_count > 0 {
                return None;
            }

            let profile = evaluate_physchem(&raw.mol, pains_count);
            if profile.qed_mean < cfg.min_qed {
                return None;
            }

            let sa_profile = evaluate_synthetic_accessibility(&raw.mol);
            if sa_profile.sa_score > cfg.max_sa_score {
                return None;
            }

            let admet = evaluate_admet(&raw.mol, &profile);

            // 4a. Primary Potency Model: Ridge Free-Wilson QSAR across cached core matrices
            let mut best_fw: Option<FreeWilsonPrediction> = None;
            for mat in &fw_models {
                if let Some(pred) = predict_free_wilson_for_molecule(&raw.mol, mat) {
                    let better = match &best_fw {
                        None => true,
                        Some(cur) => pred.known_slot_count > cur.known_slot_count,
                    };
                    if better {
                        best_fw = Some(pred);
                    }
                }
            }

            // 4b. Blended ECFP4 + 2D Pharmacophore 3-NN Potency Model (for novel scaffold chimeras
            //     or when a Free-Wilson core has novel R-groups at some slots)
            let hyb_fp = compute_ecfp(&raw.mol, 2);
            let hyb_pm = extract_pharmacophore_map(&raw.mol);
            let mut nbr_sims: Vec<(f64, f64)> = ds
                .mols
                .iter()
                .enumerate()
                .filter(|(_, m)| cfg.allow_cross_target || m.target == raw.target)
                .map(|(idx, m)| {
                    let ecfp_s = hyb_fp.tanimoto(&ds.fps[idx]);
                    let pharm_al = align_pharmacophores(&hyb_pm, &ds.pharmacophores[idx], ecfp_s);
                    let blended = (0.60 * ecfp_s + 0.40 * pharm_al.pharm_similarity) as f64;
                    (blended, m.pact)
                })
                .collect();
            nbr_sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let top3 = &nbr_sims[..nbr_sims.len().min(3)];
            let w_sum: f64 = top3.iter().map(|(s, _)| s.max(0.05)).sum();
            let knn_pact = if w_sum > 0.0 {
                top3.iter().map(|(s, p)| s.max(0.05) * (*p)).sum::<f64>() / w_sum
            } else {
                ds.mols[raw.scaffold_parent_idx].pact
            };

            let (predicted_pact, pact_method) = if let Some(ref fw) = best_fw {
                if fw.novel_slot_count == 0 {
                    (fw.predicted_pact, format!("Free-Wilson ({})", fw.core_name))
                } else {
                    // Blend Free-Wilson known-slot projection with 3-NN similarity for novel R-group slots
                    let blend = 0.65 * fw.predicted_pact + 0.35 * knn_pact;
                    (
                        blend,
                        format!("FW+3NN ({}, {} novel R)", fw.core_name, fw.novel_slot_count),
                    )
                }
            } else {
                (knn_pact, "ECFP4+Pharm2D 3-NN".to_string())
            };

            let clamped_p = predicted_pact.clamp(3.0, 12.5);
            let predicted_ic50_nm = 10f64.powf(9.0 - clamped_p);
            let predicted_lle = clamped_p - profile.clogp;

            let short_donors = raw.donor_parent_names.join(" + ");
            let hybrid_name = format!("{} x {}", raw.scaffold_parent_name, short_donors);
            raw.mol.name = hybrid_name.clone();
            raw.mol.pact = clamped_p;
            raw.mol.raw_nm = predicted_ic50_nm;

            let scaf_p = ds.mols[raw.scaffold_parent_idx].pact;
            let scaf_q = ds.profiles[raw.scaffold_parent_idx].qed_mean;
            let scaf_sa = ds.synth_access[raw.scaffold_parent_idx].sa_score;
            let utility =
                compute_utility(clamped_p, profile.qed_mean, sa_profile.sa_score, pains_count, cfg);

            Some(BricsVirtualHybrid {
                hybrid_id: String::new(),
                hybrid_name,
                target: raw.target,
                component_count: raw.component_count,
                scaffold_parent_idx: raw.scaffold_parent_idx,
                scaffold_parent_name: raw.scaffold_parent_name,
                donor_parent_indices: raw.donor_parent_indices,
                donor_parent_names: raw.donor_parent_names,
                scaffold_atoms: raw.scaffold_atoms,
                donor_atoms: raw.donor_atoms,
                formed_bonds: raw.formed_bonds,
                canonical_smiles: raw.canonical_smiles,
                mol: raw.mol,
                profile,
                admet,
                sa_profile,
                free_wilson: best_fw,
                predicted_pact: clamped_p,
                predicted_ic50_nm,
                predicted_lle,
                pact_method,
                pains_count,
                delta_pact_vs_scaffold: clamped_p - scaf_p,
                delta_qed_vs_scaffold: 0.0, // set right below using profile.qed_mean
                delta_sa_vs_scaffold: 0.0,
                pareto_front: false,
                utility_score: utility,
            })
            .map(|mut h| {
                h.delta_qed_vs_scaffold = h.profile.qed_mean - scaf_q;
                h.delta_sa_vs_scaffold = h.sa_profile.sa_score - scaf_sa;
                h
            })
        })
        .collect();

    let pareto_cnt = rescore_and_sort_hybrids(&mut scored, cfg);
    scored.truncate(cfg.max_hybrids.max(1));
    // Recompute Pareto count within retained top list
    let final_pareto = rescore_and_sort_hybrids(&mut scored, cfg);
    let _ = pareto_cnt;
    summary.retained_after_filters = scored.len();
    summary.pareto_hybrid_count = final_pareto;

    (scored, summary)
}

/// Exports ranked `BricsVirtualHybrid` chimeras as a publication-ready `.csv` table.
pub fn export_brics_hybrids_csv(hybrids: &[BricsVirtualHybrid]) -> String {
    let mut out = String::from(
        "rank,hybrid_id,hybrid_name,target,components,scaffold_parent,donor_parents,couplings,pred_pact,pred_ic50_nm,pact_method,qed_mean,sa_score,tractability,cns_mpo,lle,esol_logs,boiled_egg,pains_count,pareto_3d,utility_score,delta_pact_vs_scaf,delta_qed_vs_scaf,delta_sa_vs_scaf,smiles,source_url\n",
    );
    for (rk, h) in hybrids.iter().enumerate() {
        let donors = h.donor_parent_names.join(" + ");
        let couplings = h
            .formed_bonds
            .iter()
            .map(|fb| format!("{} ({})", fb.pair_code, fb.reaction_name))
            .collect::<Vec<_>>()
            .join("; ");
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",{},\"{}\",\"{}\",\"{}\",{:.2},{:.3},\"{}\",{:.3},{:.2},\"{}\",{:.2},{:.2},{:.2},\"{}\",{},{},{:.3},{:+.2},{:+.3},{:+.2},{},\"{}\"\n",
            rk + 1,
            h.hybrid_id,
            h.hybrid_name,
            h.target,
            h.component_count,
            h.scaffold_parent_name,
            donors,
            couplings,
            h.predicted_pact,
            h.predicted_ic50_nm,
            h.pact_method,
            h.profile.qed_mean,
            h.sa_profile.sa_score,
            h.sa_profile.tractability.short_label(),
            h.profile.cns_mpo,
            h.predicted_lle,
            h.admet.esol_logs,
            h.admet.boiled_egg.label(),
            h.pains_count,
            h.pareto_front,
            h.utility_score,
            h.delta_pact_vs_scaffold,
            h.delta_qed_vs_scaffold,
            h.delta_sa_vs_scaffold,
            h.canonical_smiles,
            h.mol.source_url
        ));
    }
    out
}
