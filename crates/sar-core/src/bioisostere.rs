//! Tactical Medicinal-Chemistry Bioisostere & Late-Stage Analog Enumerator:
//! - Meanwell (2011), *J. Med. Chem.* 54(8), 2529-2591 (`https://pubmed.ncbi.nlm.nih.gov/21413808/`)
//! - Scans a lead or sketched `Molecule` for replaceable pharmacophoric motifs, performs exact
//!   molecular graph surgery, canonicalizes & deduplicates via Weininger `OpenSMILES`, computes
//!   2D skeletal coordinates, and scores every generated analog across `Free-Wilson pAct`, `QED`,
//!   `CNS-MPO`, `ESOL LogS`, `BOILED-Egg`, `Pfizer 3/75`, and `PAINS`.

use crate::admet::{evaluate_admet, AdmetProfile};
use crate::alerts::scan_structural_alerts;
use crate::fingerprint::compute_ecfp;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::rgroup_table::{predict_best_free_wilson, FreeWilsonMatrix, FreeWilsonPrediction};
use crate::sketcher::molecule_to_smiles;
use crate::DatasetAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A single de novo bioisosteric or late-stage analog generated from a parent lead molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BioisostereAnalog {
    pub rule_code: &'static str,
    pub rule_name: String,
    pub tactical_rationale: &'static str,
    pub mol: Molecule,
    pub canonical_smiles: String,
    pub profile: PhyschemProfile,
    pub admet: AdmetProfile,
    pub free_wilson: Option<FreeWilsonPrediction>,
    pub predicted_pact: f64,
    pub predicted_lle: f64,
    pub pains_count: usize,
    pub delta_pact: f64,
    pub delta_clogp: f64,
    pub delta_logs: f64,
    pub delta_qed: f64,
    pub delta_cns_mpo: f64,
    /// Multi-objective medicinal-chemistry utility score balancing potency, `QED`, `ESOL LogS`,
    /// `Pfizer 3/75` safety, and zero `PAINS` alerts.
    pub composite_mpo_score: f64,
}

/// Enumerates, 2D-lays-out, deduplicates, and multiparameter-scores tactical bioisosteric analogs
/// of `parent` against `ds` and optional `active_sar_matrix`.
pub fn generate_bioisosteres(
    parent: &Molecule,
    ds: &DatasetAnalysis,
    active_sar_matrix: Option<&FreeWilsonMatrix>,
) -> Vec<BioisostereAnalog> {
    if parent.atoms.is_empty() {
        return Vec::new();
    }

    let parent_alerts = scan_structural_alerts(parent);
    let parent_prof = evaluate_physchem(parent, parent_alerts.len());
    let parent_admet = evaluate_admet(parent, &parent_prof);
    let parent_fw = predict_best_free_wilson(
        parent,
        active_sar_matrix,
        ds.default_sar_matrix.as_ref(),
        &ds.mols,
        &ds.profiles,
    );
    let parent_pact = parent_fw
        .as_ref()
        .map(|f| f.predicted_pact)
        .unwrap_or(parent.pact);

    let mut raw_candidates: Vec<(&'static str, String, &'static str, Molecule)> = Vec::new();

    // 1. Aryl / Heteroaryl Halogens (-Cl, -Br, -F, -I) -> Halogen / Alkyne / Nitrile / CF3 Scan
    for u in 0..parent.atoms.len() {
        let el = parent.atoms[u].elem;
        if matches!(el, Element::Cl | Element::Br | Element::F | Element::I)
            && parent.adj[u].len() == 1
        {
            let anchor = parent.adj[u][0].0;
            if parent.atoms[anchor].aromatic {
                let src_sym = el.symbol();
                for (target_el, code, rationale) in [
                    (
                        Element::F,
                        "aryl_hal_to_F",
                        "Lowers MW & cLogP while blocking oxidative metabolism",
                    ),
                    (
                        Element::Cl,
                        "aryl_hal_to_Cl",
                        "Intermediate lipophilic sigma-withdrawing halogen",
                    ),
                    (
                        Element::Br,
                        "aryl_hal_to_Br",
                        "Deep hydrophobic kinase/GPCR pocket polarizability scan",
                    ),
                ] {
                    if target_el != el {
                        let mut m = parent.clone();
                        m.atoms[u].elem = target_el;
                        m.atoms[u].bracketed = false;
                        m.atoms[u].explicit_h = 0;
                        raw_candidates.push((
                            code,
                            format!("Aryl -{src_sym} -> -{}", target_el.symbol()),
                            rationale,
                            m,
                        ));
                    }
                }

                // Aryl -X -> -C#N (polar nitrile lowers cLogP by ~1.1 & improves ESOL LogS + LLE)
                {
                    let mut m = parent.clone();
                    m.atoms[u].elem = Element::C;
                    m.atoms[u].bracketed = false;
                    m.atoms[u].explicit_h = 0;
                    let n_idx = push_atom(&mut m, Element::N, false);
                    push_bond(&mut m, u, n_idx, BondOrder::Triple, false);
                    raw_candidates.push((
                        "aryl_hal_to_CN",
                        format!("Aryl -{src_sym} -> -C#N (Nitrile Bioisostere)"),
                        "Polar HBA vector lowering cLogP, boosting ESOL solubility & LLE",
                        m,
                    ));
                }

                // Aryl -X -> -C#CH (Erlotinib-class linear hydrophobic pocket vector)
                {
                    let mut m = parent.clone();
                    m.atoms[u].elem = Element::C;
                    m.atoms[u].bracketed = false;
                    m.atoms[u].explicit_h = 0;
                    let c2 = push_atom(&mut m, Element::C, false);
                    push_bond(&mut m, u, c2, BondOrder::Triple, false);
                    raw_candidates.push((
                        "aryl_hal_to_ethynyl",
                        format!("Aryl -{src_sym} -> -C#CH (Ethynyl Bioisostere)"),
                        "Linear sp-hybridized ATP-pocket vector (Erlotinib pharmacophore)",
                        m,
                    ));
                }

                // Aryl -X -> -CF3
                {
                    let mut m = parent.clone();
                    m.atoms[u].elem = Element::C;
                    m.atoms[u].bracketed = false;
                    m.atoms[u].explicit_h = 0;
                    for _ in 0..3 {
                        let f_idx = push_atom(&mut m, Element::F, false);
                        push_bond(&mut m, u, f_idx, BondOrder::Single, false);
                    }
                    raw_candidates.push((
                        "aryl_hal_to_CF3",
                        format!("Aryl -{src_sym} -> -CF3"),
                        "Strong inductive electron-withdrawing lipophilic bioisostere",
                        m,
                    ));
                }
            }
        }
    }

    // 2. Metabolic O-Alkyl Soft-Spot Block: Aryl -OCH3 -> -OCHF2 (Difluoromethoxy) & -OCD3/Ethoxy
    for u in 0..parent.atoms.len() {
        if parent.atoms[u].elem == Element::O && parent.adj[u].len() == 2 {
            let n0 = parent.adj[u][0].0;
            let n1 = parent.adj[u][1].0;
            let (ar_atom, me_atom) = if parent.atoms[n0].aromatic
                && parent.atoms[n1].elem == Element::C
                && parent.adj[n1].len() == 1
            {
                (Some(n0), Some(n1))
            } else if parent.atoms[n1].aromatic
                && parent.atoms[n0].elem == Element::C
                && parent.adj[n0].len() == 1
            {
                (Some(n1), Some(n0))
            } else {
                (None, None)
            };

            if let (Some(_), Some(c_me)) = (ar_atom, me_atom) {
                // -OCH3 -> -OCHF2
                let mut m_chf2 = parent.clone();
                m_chf2.atoms[c_me].bracketed = false;
                m_chf2.atoms[c_me].explicit_h = 0;
                for _ in 0..2 {
                    let f = push_atom(&mut m_chf2, Element::F, false);
                    push_bond(&mut m_chf2, c_me, f, BondOrder::Single, false);
                }
                raw_candidates.push((
                    "ome_to_ochf2",
                    "Aryl -OCH3 -> -OCHF2 (Difluoromethoxy)".to_string(),
                    "Blocks CYP450 O-demethylation soft spot while preserving conformational preference",
                    m_chf2,
                ));

                // -OCH3 -> -OCH2CH3
                let mut m_oet = parent.clone();
                m_oet.atoms[c_me].bracketed = false;
                m_oet.atoms[c_me].explicit_h = 0;
                let c_et = push_atom(&mut m_oet, Element::C, false);
                push_bond(&mut m_oet, c_me, c_et, BondOrder::Single, false);
                raw_candidates.push((
                    "ome_to_oet",
                    "Aryl -OCH3 -> -OCH2CH3 (Ethoxy Homolog)".to_string(),
                    "Solvent-front alkoxy homologation (PD-153035 -> Compound 56 motif)",
                    m_oet,
                ));
            }
        }
    }

    // 3. Carboxylic Acid (-C(=O)OH) -> 5-Tetrazolyl (-c1[nH]nnn1) & Acylsulfonamide (-C(=O)NHSO2Me)
    for u in 0..parent.atoms.len() {
        if parent.atoms[u].elem == Element::C && !parent.atoms[u].in_ring && parent.adj[u].len() == 3
        {
            let mut ket_o = None;
            let mut oh_o = None;
            let mut root_nb = None;
            for &(v, bi) in &parent.adj[u] {
                if parent.atoms[v].elem == Element::O
                    && parent.bonds[bi].kekule_order == 2
                    && parent.adj[v].len() == 1
                {
                    ket_o = Some(v);
                } else if parent.atoms[v].elem == Element::O
                    && parent.bonds[bi].kekule_order == 1
                    && parent.adj[v].len() == 1
                    && parent.atoms[v].charge <= 0
                {
                    oh_o = Some(v);
                } else {
                    root_nb = Some(v);
                }
            }
            if let (Some(ko), Some(oo), Some(_)) = (ket_o, oh_o, root_nb) {
                // 3a. Acylsulfonamide: replace -OH oxygen `oo` with -NH-S(=O)2-CH3
                let mut m_as = parent.clone();
                m_as.atoms[oo].elem = Element::N;
                m_as.atoms[oo].charge = 0;
                m_as.atoms[oo].bracketed = false;
                m_as.atoms[oo].explicit_h = 0;
                let s = push_atom(&mut m_as, Element::S, false);
                let o1 = push_atom(&mut m_as, Element::O, false);
                let o2 = push_atom(&mut m_as, Element::O, false);
                let c_me = push_atom(&mut m_as, Element::C, false);
                push_bond(&mut m_as, oo, s, BondOrder::Single, false);
                push_bond(&mut m_as, s, o1, BondOrder::Double, false);
                push_bond(&mut m_as, s, o2, BondOrder::Double, false);
                push_bond(&mut m_as, s, c_me, BondOrder::Single, false);
                raw_candidates.push((
                    "cooh_to_acylsulfonamide",
                    "-COOH -> -CONHSO2Me (Acylsulfonamide)".to_string(),
                    "Meanwell (2011) acidic pKa-matched carboxylic acid bioisostere",
                    m_as,
                ));

                // 3b. 5-Tetrazole: convert `u` into tetrazole C5, `ko` & `oo` into ring N1/N4, plus N2, N3!
                let mut m_tet = parent.clone();
                m_tet.atoms[u].aromatic = true;
                m_tet.atoms[ko].elem = Element::N;
                m_tet.atoms[ko].aromatic = true;
                m_tet.atoms[ko].bracketed = true;
                m_tet.atoms[ko].explicit_h = 1; // [nH] tautomer
                m_tet.atoms[oo].elem = Element::N;
                m_tet.atoms[oo].aromatic = true;
                m_tet.atoms[oo].charge = 0;
                m_tet.atoms[oo].bracketed = false;
                m_tet.atoms[oo].explicit_h = 0;
                if let Some(bi) = m_tet.bond_between(u, ko) {
                    m_tet.bonds[bi].order = BondOrder::Aromatic;
                    m_tet.bonds[bi].aromatic = true;
                }
                if let Some(bi) = m_tet.bond_between(u, oo) {
                    m_tet.bonds[bi].order = BondOrder::Aromatic;
                    m_tet.bonds[bi].aromatic = true;
                }
                let n2 = push_atom(&mut m_tet, Element::N, true);
                let n3 = push_atom(&mut m_tet, Element::N, true);
                push_bond(&mut m_tet, ko, n2, BondOrder::Aromatic, true);
                push_bond(&mut m_tet, n2, n3, BondOrder::Aromatic, true);
                push_bond(&mut m_tet, n3, oo, BondOrder::Aromatic, true);
                raw_candidates.push((
                    "cooh_to_tetrazole",
                    "-COOH -> 5-Tetrazolyl (-c1[nH]nnn1)".to_string(),
                    "Planar lipophilic carboxylic acid bioisostere improving GI permeability & metabolic stability",
                    m_tet,
                ));
            }
        }
    }

    // 4. Sulfonamide (-SO2NH2) <-> Methylsulfonyl (-SO2CH3) Switch
    for u in 0..parent.atoms.len() {
        if parent.atoms[u].elem == Element::S && parent.adj[u].len() == 4 {
            let mut oxo_cnt = 0usize;
            let mut nh2_atom = None;
            let mut me_atom = None;
            for &(v, bi) in &parent.adj[u] {
                if parent.atoms[v].elem == Element::O
                    && parent.bonds[bi].kekule_order == 2
                    && parent.adj[v].len() == 1
                {
                    oxo_cnt += 1;
                } else if parent.atoms[v].elem == Element::N
                    && parent.bonds[bi].kekule_order == 1
                    && parent.adj[v].len() == 1
                {
                    nh2_atom = Some(v);
                } else if parent.atoms[v].elem == Element::C
                    && parent.bonds[bi].kekule_order == 1
                    && parent.adj[v].len() == 1
                {
                    me_atom = Some(v);
                }
            }
            if oxo_cnt == 2 {
                if let Some(nv) = nh2_atom {
                    let mut m = parent.clone();
                    m.atoms[nv].elem = Element::C;
                    m.atoms[nv].bracketed = false;
                    m.atoms[nv].explicit_h = 0;
                    raw_candidates.push((
                        "so2nh2_to_so2me",
                        "-SO2NH2 -> -SO2CH3 (Methylsulfone Switch)".to_string(),
                        "Removes 2 HBDs, lowers TPSA by 26 A^2, and eliminates carbonic anhydrase off-target binding",
                        m,
                    ));
                } else if let Some(cv) = me_atom {
                    let mut m = parent.clone();
                    m.atoms[cv].elem = Element::N;
                    m.atoms[cv].bracketed = false;
                    m.atoms[cv].explicit_h = 0;
                    raw_candidates.push((
                        "so2me_to_so2nh2",
                        "-SO2CH3 -> -SO2NH2 (Primary Sulfonamide Switch)".to_string(),
                        "Adds polar HBD/HBA interactions and increases aqueous solubility",
                        m,
                    ));
                }
            }
        }
    }

    // 5. Aromatic Aza-Scan (Phenyl `cH` -> Pyridyl `n`) & Metabolic Para-Fluorination (`cH` -> `c-F`)
    let mut aza_count = 0usize;
    let mut fluor_count = 0usize;
    for ring in &parent.rings {
        if ring.len() == 6
            && ring
                .iter()
                .all(|&ai| parent.atoms[ai].aromatic && parent.atoms[ai].elem == Element::C)
        {
            for &ai in ring {
                // Unsubstituted aromatic C-H (`heavy_degree == 2`, `total_h() == 1`)
                if parent.heavy_degree(ai) == 2 && parent.atoms[ai].total_h() == 1 {
                    if aza_count < 3 {
                        let mut m_aza = parent.clone();
                        m_aza.atoms[ai].elem = Element::N;
                        m_aza.atoms[ai].aromatic = true;
                        m_aza.atoms[ai].bracketed = false;
                        m_aza.atoms[ai].explicit_h = 0;
                        raw_candidates.push((
                            "phenyl_aza_scan",
                            format!("Aza-Scan: Phenyl c{}H -> Pyridyl N", ai + 1),
                            "Drops cLogP by ~1.0 unit, raises ESOL LogS & LLE, and mitigates Pfizer 3/75 risk",
                            m_aza,
                        ));
                        aza_count += 1;
                    }
                    if fluor_count < 2 {
                        let mut m_f = parent.clone();
                        m_f.atoms[ai].bracketed = false;
                        m_f.atoms[ai].explicit_h = 0;
                        let f_idx = push_atom(&mut m_f, Element::F, false);
                        push_bond(&mut m_f, ai, f_idx, BondOrder::Single, false);
                        raw_candidates.push((
                            "aryl_fluorination",
                            format!("Fluorine Walk: Aryl c{}H -> c-F", ai + 1),
                            "Blocks hepatic CYP450 aromatic hydroxylation with minimal steric perturbation",
                            m_f,
                        ));
                        fluor_count += 1;
                    }
                }
            }
        }
    }

    // Deduplicate by canonical OpenSMILES, sanitize, compute 2D coordinates, and score!
    let parent_can = molecule_to_smiles(parent);
    let mut seen_smiles: BTreeSet<String> = BTreeSet::new();
    seen_smiles.insert(parent_can);

    let mut analogs = Vec::new();

    for (idx, (code, rule_name, rationale, mut cand)) in raw_candidates.into_iter().enumerate() {
        cand.sanitize();
        let can_smi = molecule_to_smiles(&cand);
        if can_smi.is_empty() || !seen_smiles.insert(can_smi.clone()) {
            continue;
        }
        compute_2d_coords(&mut cand);
        cand.smiles = can_smi.clone();
        cand.id = format!("BIO-{:02}", idx + 1);
        cand.name = format!("{} [{}]", parent.name, rule_name);
        cand.target = parent.target.clone();

        let alerts = scan_structural_alerts(&cand);
        let mut prof = evaluate_physchem(&cand, alerts.len());
        let admet = evaluate_admet(&cand, &prof);
        let fw = predict_best_free_wilson(
            &cand,
            active_sar_matrix,
            ds.default_sar_matrix.as_ref(),
            &ds.mols,
            &ds.profiles,
        );

        let pred_pact = if let Some(ref f) = fw {
            f.predicted_pact
        } else {
            // 3-NN ECFP4 similarity potency estimate
            let fp = compute_ecfp(&cand, 2);
            let mut sims: Vec<(usize, f32)> = ds
                .fps
                .iter()
                .enumerate()
                .map(|(i, dfp)| (i, fp.tanimoto(dfp)))
                .filter(|&(_, t)| t > 0.20)
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(3);
            if sims.is_empty() {
                parent_pact
            } else {
                let ws: f64 = sims.iter().map(|&(_, t)| (t as f64).powi(2)).sum::<f64>().max(1e-6);
                sims.iter()
                    .map(|&(mi, t)| ((t as f64).powi(2)) * ds.mols[mi].pact)
                    .sum::<f64>()
                    / ws
            }
        };

        let pred_lle = pred_pact - prof.clogp;
        prof.lle = pred_lle;
        cand.pact = pred_pact;
        cand.raw_nm = 10f64.powf(9.0 - pred_pact.clamp(2.0, 14.0));

        // Multi-objective MPO score: potency + 2.0*QED + 0.25*(CNS_MPO) + 0.35*ESOL_bonus + safety_bonus - PAINS_penalty
        let sol_term = ((admet.esol_logs + 6.0) / 4.0).clamp(-0.5, 1.0);
        let safety_bonus = (if admet.pfizer_3_75_pass { 0.35 } else { 0.0 })
            + (if admet.golden_triangle { 0.25 } else { 0.0 });
        let pains_pen = (alerts.len() as f64) * 1.25;
        let mpo_score = pred_pact + 2.0 * prof.qed_mean + 0.20 * prof.cns_mpo
            + 0.40 * sol_term
            + safety_bonus
            - pains_pen;

        analogs.push(BioisostereAnalog {
            rule_code: code,
            rule_name,
            tactical_rationale: rationale,
            mol: cand,
            canonical_smiles: can_smi,
            delta_pact: pred_pact - parent_pact,
            delta_clogp: prof.clogp - parent_prof.clogp,
            delta_logs: admet.esol_logs - parent_admet.esol_logs,
            delta_qed: prof.qed_mean - parent_prof.qed_mean,
            delta_cns_mpo: prof.cns_mpo - parent_prof.cns_mpo,
            profile: prof,
            admet,
            free_wilson: fw,
            predicted_pact: pred_pact,
            predicted_lle: pred_lle,
            pains_count: alerts.len(),
            composite_mpo_score: mpo_score,
        });
    }

    analogs.sort_by(|a, b| {
        b.composite_mpo_score
            .partial_cmp(&a.composite_mpo_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    analogs
}

fn push_atom(mol: &mut Molecule, elem: Element, aromatic: bool) -> usize {
    let idx = mol.atoms.len();
    mol.atoms.push(Atom {
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
        pos2d: [0.0, 0.0],
    });
    idx
}

fn push_bond(mol: &mut Molecule, a: usize, b: usize, order: BondOrder, aromatic: bool) {
    let idx = mol.bonds.len();
    let kekule_order = match order {
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        _ => 1,
    };
    mol.bonds.push(Bond {
        idx,
        a,
        b,
        order,
        aromatic,
        in_ring: false,
        kekule_order,
    });
}
