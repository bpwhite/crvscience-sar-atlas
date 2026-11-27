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

//! Core-aligned `R1/R2/R3/R4` R-group decomposition and Ridge-regularized Free-Wilson QSAR
//! matrix solver & live single-molecule analog potency predictor:
//! - Free & Wilson (1964), *J. Med. Chem.* 7, 395-399 (`https://pubmed.ncbi.nlm.nih.gov/14221113/`)
//! - Ridge-regularized one-hot R-group regression & prospective virtual analog enumeration
//!   (`http://practicalcheminformatics.blogspot.com/2018/05/free-wilson-analysis.html`,
//!   `https://www.mdpi.com/1420-3049/31/20/3775`)

use crate::mol::{Element, Molecule};
use crate::physchem::PhyschemProfile;
use crate::scaffold::{extract_murcko_scaffolds, wl_subgraph_hash};
use crate::smarts::{compile_smarts, match_smarts, SmartsQuery};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

pub struct CoreTemplate {
    pub name: &'static str,
    pub smarts: &'static str,
    pub target_hint: &'static str,
    pub description: &'static str,
}

/// Curated medicinal-chemistry core templates for one-click R1/R2/R3 SAR matrix alignment.
pub const CURATED_CORE_TEMPLATES: [CoreTemplate; 5] = [
    CoreTemplate {
        name: "4-Anilinoquinazoline (EGFR)",
        smarts: "c1ccc(Nc2ncnc3ccccc23)cc1",
        target_hint: "EGFR",
        description: "17-atom ATP-site pharmacophore aligning quinazoline C6/C7 tails and aniline 3'/4' halogens/alkynes",
    },
    CoreTemplate {
        name: "Quinazoline Core (EGFR)",
        smarts: "c1ncnc2ccccc12",
        target_hint: "EGFR",
        description: "10-atom bicyclic ring comparing whole C4-aniline headgroups against C6/C7 alkoxy chains",
    },
    CoreTemplate {
        name: "Phenothiazine (Dopamine D2)",
        smarts: "c1ccc2c(c1)Sc3ccccc3N2",
        target_hint: "Dopamine D2",
        description: "Tricyclic phenothiazine aligning C2 ring substituents (-CF3, -Cl) against N10 basic side chains",
    },
    CoreTemplate {
        name: "Tricyclic Piperazine (Dopamine D2)",
        smarts: "c1ccc2c(c1)[#8,#7,#16]c3ccccc3N=C2N4CCNCC4",
        target_hint: "Dopamine D2",
        description: "Dibenzoxazepine / dibenzodiazepine / dibenzothiazepine core aligning X-bridge, ring-Cl, and N-piperazine tails",
    },
    CoreTemplate {
        name: "Diaryl-5-Ring Coxib (COX-2)",
        smarts: "c1ccccc1-[r5]~[r5]-c2ccccc2",
        target_hint: "COX-2",
        description: "Vicinal diaryl 5-membered heterocycle aligning para-SO2Me/SO2NH2 and central furanone/pyrazole/isoxazole cores",
    },
];

/// A single aligned compound row inside the R1/R2/R3/R4 Free-Wilson table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RGroupRow {
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    /// Substituent label at each variable position `R1..Rk` (e.g. `["-[OCH2CH3]", "-[OCH2CH3]", "-[Br]"]`).
    pub r_groups: Vec<String>,
    /// Target atom indices matched to the core for 2D scaffold highlighting.
    pub core_atoms: Vec<usize>,
    pub observed_pact: f64,
    pub predicted_pact: f64,
    pub loo_predicted_pact: f64,
    /// Signed residual `observed_pact - predicted_pact` (`|res| >= 0.50` indicates non-additive epistatic cliff).
    pub residual: f64,
    pub non_additive_cliff: bool,
    pub clogp: f64,
    pub qed_mean: f64,
}

/// Fitted Free-Wilson additive contribution (`Delta pAct`) for a specific substituent at position `R_p`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubstituentEffect {
    pub slot_index: usize,
    pub slot_label: String,
    pub substituent: String,
    pub count: usize,
    /// Ridge Free-Wilson additive contribution (`sum_k coeff_pact = 0` per R-slot).
    pub coeff_pact: f64,
    pub mean_obs_pact: f64,
    pub mean_delta_clogp: f64,
}

/// Cell in the 2D `R1 x R2` medicinal-chemistry SAR matrix grid (either synthesized or predicted hole).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SarPivotCell {
    pub observed_mol: Option<usize>,
    pub observed_name: Option<String>,
    pub observed_pact: Option<f64>,
    pub predicted_pact: f64,
    pub residual: Option<f64>,
}

/// Prospective unsynthesized virtual analog enumerated from `R1 x R2 x R3` substituent combinations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VirtualAnalog {
    pub r_groups: Vec<String>,
    pub predicted_pact: f64,
    pub predicted_ic50_nm: f64,
    pub est_clogp: f64,
    pub est_lle: f64,
    pub shared_r_count: usize,
    pub nearest_observed_mol: usize,
    pub nearest_observed_name: String,
}

/// Per-slot contribution breakdown when scoring a live-sketched molecule against a `FreeWilsonMatrix`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FreeWilsonSlotContribution {
    pub slot_index: usize,
    pub slot_label: String,
    pub substituent: String,
    pub coeff_pact: f64,
    pub known_in_series: bool,
    pub train_count: usize,
}

/// Live Free-Wilson QSAR prediction for an arbitrary sketched or virtual `Molecule`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FreeWilsonPrediction {
    pub core_name: String,
    pub core_smarts: String,
    pub core_atoms: Vec<usize>,
    pub intercept_pact: f64,
    pub model_r2: f64,
    pub model_q2_loo: f64,
    pub slots: Vec<FreeWilsonSlotContribution>,
    pub predicted_pact: f64,
    pub predicted_ic50_nm: f64,
    pub known_slot_count: usize,
    pub novel_slot_count: usize,
    /// If the sketched R1..Rk vector matches an already-synthesized training row: `(mol_idx, name, obs_pact)`.
    pub exact_training_match: Option<(usize, String, f64)>,
}

/// Complete Core-Aligned R1/R2/R3/R4 Free-Wilson SAR Matrix model, 2D pivot grid, and virtual analog table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FreeWilsonMatrix {
    pub core_name: String,
    pub core_smarts: String,
    pub ridge_lambda: f64,
    pub slot_labels: Vec<String>,
    /// Core query-atom index corresponding to each `slot_labels[p]`.
    pub slot_query_atoms: Vec<usize>,
    pub rows: Vec<RGroupRow>,
    pub intercept_pact: f64,
    pub r2: f64,
    pub rmse: f64,
    pub q2_loo: f64,
    pub effects: Vec<SubstituentEffect>,
    pub pivot_r1_substituents: Vec<String>,
    pub pivot_r2_substituents: Vec<String>,
    /// `pivot_grid[r1_idx][r2_idx]` holding observed compound or predicted unsynthesized hole.
    pub pivot_grid: Vec<Vec<SarPivotCell>>,
    pub virtual_analogs: Vec<VirtualAnalog>,
}

impl FreeWilsonMatrix {
    /// Exports the R1..Rk decomposition table, observed/predicted potencies, and Free-Wilson substituent weights as CSV.
    pub fn export_csv(&self) -> String {
        let mut out = String::from("compound_id,compound_name,target,");
        for sl in &self.slot_labels {
            out.push_str(&format!("{sl},"));
        }
        out.push_str("observed_pact,predicted_pact,loo_pact,residual,non_additive_cliff,clogp,qed_mean\n");

        for r in &self.rows {
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",",
                r.compound_id, r.compound_name, r.target
            ));
            for g in &r.r_groups {
                out.push_str(&format!("\"{g}\","));
            }
            out.push_str(&format!(
                "{:.3},{:.3},{:.3},{:+.3},{},{:.2},{:.3}\n",
                r.observed_pact,
                r.predicted_pact,
                r.loo_predicted_pact,
                r.residual,
                r.non_additive_cliff,
                r.clogp,
                r.qed_mean
            ));
        }

        out.push_str("\n# Free-Wilson Substituent Contributions (Baseline Intercept = ");
        out.push_str(&format!(
            "{:.3}, R2 = {:.3}, RMSE = {:.3}, Q2_LOO = {:.3})\n",
            self.intercept_pact, self.r2, self.rmse, self.q2_loo
        ));
        out.push_str("slot,substituent,count,free_wilson_delta_pact,mean_observed_pact\n");
        for ef in &self.effects {
            out.push_str(&format!(
                "{},\"{}\",{},{:+.4},{:.3}\n",
                ef.slot_label, ef.substituent, ef.count, ef.coeff_pact, ef.mean_obs_pact
            ));
        }
        out
    }
}

/// Aligns a single sketched or virtual `mol` against an already-fitted `FreeWilsonMatrix`,
/// extracting its canonical `R1..Rk` substituent vector and computing its additive `predicted_pact`.
pub fn predict_free_wilson_for_molecule(
    mol: &Molecule,
    mat: &FreeWilsonMatrix,
) -> Option<FreeWilsonPrediction> {
    if mol.atoms.is_empty() {
        return None;
    }
    let query = compile_smarts(&mat.core_smarts).ok()?;
    let mres = match_smarts(mol, &query);
    if !mres.matched || mres.mappings.is_empty() {
        return None;
    }

    let (core_atoms, per_q_subst) = canonicalize_core_embedding(mol, &query, &mres.mappings);

    let mut pred_pact = mat.intercept_pact;
    let mut slots = Vec::with_capacity(mat.slot_query_atoms.len());
    let mut r_tuple = Vec::with_capacity(mat.slot_query_atoms.len());
    let mut known_cnt = 0usize;
    let mut novel_cnt = 0usize;

    for (si, &qp) in mat.slot_query_atoms.iter().enumerate() {
        let sub = per_q_subst
            .get(qp)
            .cloned()
            .unwrap_or_else(|| "-[H]".to_string());
        r_tuple.push(sub.clone());

        if let Some(ef) = mat
            .effects
            .iter()
            .find(|e| e.slot_index == si && e.substituent == sub)
        {
            pred_pact += ef.coeff_pact;
            known_cnt += 1;
            slots.push(FreeWilsonSlotContribution {
                slot_index: si,
                slot_label: mat.slot_labels[si].clone(),
                substituent: sub,
                coeff_pact: ef.coeff_pact,
                known_in_series: true,
                train_count: ef.count,
            });
        } else {
            novel_cnt += 1;
            slots.push(FreeWilsonSlotContribution {
                slot_index: si,
                slot_label: mat.slot_labels[si].clone(),
                substituent: sub,
                coeff_pact: 0.0,
                known_in_series: false,
                train_count: 0,
            });
        }
    }

    let exact_training_match = mat
        .rows
        .iter()
        .find(|r| r.r_groups == r_tuple)
        .map(|r| (r.mol_idx, r.compound_name.clone(), r.observed_pact));

    let clamped = pred_pact.clamp(2.0, 14.0);
    let predicted_ic50_nm = 10f64.powf(9.0 - clamped);

    Some(FreeWilsonPrediction {
        core_name: mat.core_name.clone(),
        core_smarts: mat.core_smarts.clone(),
        core_atoms,
        intercept_pact: mat.intercept_pact,
        model_r2: mat.r2,
        model_q2_loo: mat.q2_loo,
        slots,
        predicted_pact: pred_pact,
        predicted_ic50_nm,
        known_slot_count: known_cnt,
        novel_slot_count: novel_cnt,
        exact_training_match,
    })
}

/// Evaluates `mol` against `active_mat` first, falling back to `default_mat` and any matching
/// `CURATED_CORE_TEMPLATES` series in `ds_mols` so live 2D edits get instant Free-Wilson `pAct`
/// across EGFR quinazolines, D2 phenothiazines/dibenzoxazepines, or COX-2 coxibs automatically.
pub fn predict_best_free_wilson(
    mol: &Molecule,
    active_mat: Option<&FreeWilsonMatrix>,
    default_mat: Option<&FreeWilsonMatrix>,
    ds_mols: &[Molecule],
    ds_profiles: &[PhyschemProfile],
) -> Option<FreeWilsonPrediction> {
    if let Some(am) = active_mat {
        if let Some(pred) = predict_free_wilson_for_molecule(mol, am) {
            return Some(pred);
        }
    }
    if let Some(dm) = default_mat {
        let same_as_active = active_mat
            .map(|am| am.core_smarts == dm.core_smarts)
            .unwrap_or(false);
        if !same_as_active {
            if let Some(pred) = predict_free_wilson_for_molecule(mol, dm) {
                return Some(pred);
            }
        }
    }

    // Check curated templates Ordered by largest specific pharmacophore first
    for tpl in &CURATED_CORE_TEMPLATES {
        if let Ok(q) = compile_smarts(tpl.smarts) {
            if match_smarts(mol, &q).matched {
                if let Ok(mat) = build_free_wilson_for_smarts(
                    ds_mols,
                    ds_profiles,
                    tpl.name,
                    tpl.smarts,
                    Some(tpl.target_hint),
                    3,
                    0.15,
                ) {
                    if let Some(pred) = predict_free_wilson_for_molecule(mol, &mat) {
                        return Some(pred);
                    }
                }
            }
        }
    }

    None
}

/// Automatically selects the richest populated core (checking `CURATED_CORE_TEMPLATES` first,
/// then auto-converting the top Bemis-Murcko scaffold into SMARTS) and solves the Free-Wilson matrix.
pub fn build_auto_free_wilson_matrix(
    mols: &[Molecule],
    profiles: &[PhyschemProfile],
    max_slots: usize,
    ridge_lambda: f64,
) -> Option<FreeWilsonMatrix> {
    if mols.len() < 2 {
        return None;
    }

    let mut best_curated: Option<(usize, &CoreTemplate)> = None;
    let scan_limit = mols.len().min(600);
    for tpl in &CURATED_CORE_TEMPLATES {
        if let Ok(q) = compile_smarts(tpl.smarts) {
            let hits = mols[..scan_limit]
                .iter()
                .filter(|m| match_smarts(m, &q).matched)
                .count();
            if hits >= 3 && best_curated.map_or(true, |(bh, _)| hits > bh) {
                best_curated = Some((hits, tpl));
            }
        }
    }

    if let Some((_, tpl)) = best_curated {
        if let Ok(mat) = build_free_wilson_for_smarts(
            mols,
            profiles,
            tpl.name,
            tpl.smarts,
            Some(tpl.target_hint),
            max_slots,
            ridge_lambda,
        ) {
            if mat.rows.len() >= 3 {
                return Some(mat);
            }
        }
    }

    let scaffolds = extract_murcko_scaffolds(&mols[..scan_limit]);
    for sc in scaffolds.iter().take(5) {
        if sc.member_mols.len() < 2 {
            continue;
        }
        if let Some(auto_smarts) =
            scaffold_subgraph_to_smarts(&mols[sc.exemplar_mol], &sc.exemplar_atoms)
        {
            if let Ok(mat) = build_free_wilson_for_smarts(
                mols,
                profiles,
                &format!("Auto Core: {}", sc.framework_label),
                &auto_smarts,
                None,
                max_slots,
                ridge_lambda,
            ) {
                if mat.rows.len() >= 2 {
                    return Some(mat);
                }
            }
        }
    }

    None
}

/// Aligns dataset molecules matching `core_smarts` (optionally restricted to `target_filter`),
/// extracts canonical `R1..R_{max_slots}` substituent vectors, fits Ridge Free-Wilson QSAR,
/// builds the `R1 x R2` SAR matrix grid, and enumerates top unsynthesized virtual analogs.
pub fn build_free_wilson_for_smarts(
    mols: &[Molecule],
    profiles: &[PhyschemProfile],
    core_name: &str,
    core_smarts: &str,
    target_filter: Option<&str>,
    max_slots: usize,
    ridge_lambda: f64,
) -> Result<FreeWilsonMatrix, String> {
    let query = compile_smarts(core_smarts).map_err(|e| e.to_string())?;
    let qn = query.atoms.len();
    let max_slots = max_slots.clamp(1, 6);
    let lambda = ridge_lambda.clamp(1e-4, 25.0);

    let scan_n = mols.len().min(1200);

    struct RawAligned {
        mol_idx: usize,
        core_atoms: Vec<usize>,
        per_q_subst: Vec<String>,
    }

    let mut aligned: Vec<RawAligned> = Vec::new();
    for mi in 0..scan_n {
        if let Some(tf) = target_filter {
            if !tf.is_empty() && tf != "All" && mols[mi].target != tf {
                continue;
            }
        }
        let mres = match_smarts(&mols[mi], &query);
        if !mres.matched || mres.mappings.is_empty() {
            continue;
        }
        let (best_map, per_q_subst) =
            canonicalize_core_embedding(&mols[mi], &query, &mres.mappings);
        aligned.push(RawAligned {
            mol_idx: mi,
            core_atoms: best_map,
            per_q_subst,
        });
    }

    if aligned.len() < 2 {
        return Err(format!(
            "Core SMARTS '{}' matched only {} compound(s); at least 2 congeneric compounds are required.",
            core_smarts,
            aligned.len()
        ));
    }

    let mut varying_q: Vec<(usize, usize, usize)> = Vec::new();
    for q_pos in 0..qn {
        let mut distinct: BTreeSet<&str> = BTreeSet::new();
        let mut non_h = 0usize;
        for row in &aligned {
            let s = row.per_q_subst[q_pos].as_str();
            distinct.insert(s);
            if s != "-[H]" {
                non_h += 1;
            }
        }
        if distinct.len() >= 2 {
            varying_q.push((q_pos, non_h, distinct.len()));
        }
    }

    if varying_q.is_empty() {
        varying_q.push((0, aligned.len(), 1));
    }

    varying_q.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    varying_q.truncate(max_slots);

    let slot_query_atoms: Vec<usize> = varying_q.iter().map(|&(qp, _, _)| qp).collect();
    let p_slots = slot_query_atoms.len();
    let slot_labels: Vec<String> = (0..p_slots)
        .map(|k| format!("R{} (@c{})", k + 1, slot_query_atoms[k] + 1))
        .collect();

    let mut slot_vocabs: Vec<Vec<String>> = vec![Vec::new(); p_slots];
    let mut raw_r_rows: Vec<Vec<String>> = Vec::with_capacity(aligned.len());

    for al in &aligned {
        let mut r_vec = Vec::with_capacity(p_slots);
        for (si, &qp) in slot_query_atoms.iter().enumerate() {
            let sub = al.per_q_subst[qp].clone();
            if !slot_vocabs[si].contains(&sub) {
                slot_vocabs[si].push(sub.clone());
            }
            r_vec.push(sub);
        }
        raw_r_rows.push(r_vec);
    }
    for vocab in &mut slot_vocabs {
        vocab.sort();
    }

    let mut slot_col_offsets = Vec::with_capacity(p_slots);
    let mut total_feat_cols = 1usize;
    for vocab in &slot_vocabs {
        slot_col_offsets.push(total_feat_cols);
        total_feat_cols += vocab.len();
    }

    let m_rows = aligned.len();
    let d_cols = total_feat_cols;
    let mut a_mat = vec![0.0f64; m_rows * d_cols];
    let mut y_vec = Vec::with_capacity(m_rows);

    for (ri, al) in aligned.iter().enumerate() {
        a_mat[ri * d_cols] = 1.0;
        for si in 0..p_slots {
            let sub = &raw_r_rows[ri][si];
            let k_idx = slot_vocabs[si].iter().position(|v| v == sub).unwrap_or(0);
            let col = slot_col_offsets[si] + k_idx;
            a_mat[ri * d_cols + col] = 1.0;
        }
        y_vec.push(mols[al.mol_idx].pact);
    }

    let mut norm_mat = vec![0.0f64; d_cols * d_cols];
    let mut rhs = vec![0.0f64; d_cols];

    for r in 0..m_rows {
        let yr = y_vec[r];
        for c1 in 0..d_cols {
            let v1 = a_mat[r * d_cols + c1];
            if v1 == 0.0 {
                continue;
            }
            rhs[c1] += v1 * yr;
            for c2 in c1..d_cols {
                let v2 = a_mat[r * d_cols + c2];
                if v2 != 0.0 {
                    norm_mat[c1 * d_cols + c2] += v1 * v2;
                }
            }
        }
    }
    for c1 in 0..d_cols {
        for c2 in 0..c1 {
            norm_mat[c1 * d_cols + c2] = norm_mat[c2 * d_cols + c1];
        }
    }
    for c in 1..d_cols {
        norm_mat[c * d_cols + c] += lambda;
    }

    let inv_norm = invert_spd_matrix(&norm_mat, d_cols);
    let mut theta = vec![0.0f64; d_cols];
    for c1 in 0..d_cols {
        let mut acc = 0.0;
        for c2 in 0..d_cols {
            acc += inv_norm[c1 * d_cols + c2] * rhs[c2];
        }
        theta[c1] = acc;
    }

    let mut intercept_pact = theta[0];
    for si in 0..p_slots {
        let start = slot_col_offsets[si];
        let len = slot_vocabs[si].len();
        if len > 0 {
            let mean_slot: f64 = theta[start..start + len].iter().sum::<f64>() / (len as f64);
            intercept_pact += mean_slot;
            for c in start..start + len {
                theta[c] -= mean_slot;
            }
        }
    }
    theta[0] = intercept_pact;

    let y_mean: f64 = y_vec.iter().sum::<f64>() / (m_rows as f64);
    let mut ss_tot = 0.0f64;
    let mut ss_res = 0.0f64;
    let mut press_loo = 0.0f64;

    let mut rows: Vec<RGroupRow> = Vec::with_capacity(m_rows);
    for (ri, al) in aligned.iter().enumerate() {
        let mut y_hat = 0.0f64;
        for c in 0..d_cols {
            y_hat += a_mat[ri * d_cols + c] * theta[c];
        }

        let mut h_ii = 0.0f64;
        for c1 in 0..d_cols {
            let v1 = a_mat[ri * d_cols + c1];
            if v1 == 0.0 {
                continue;
            }
            for c2 in 0..d_cols {
                let v2 = a_mat[ri * d_cols + c2];
                if v2 != 0.0 {
                    h_ii += v1 * inv_norm[c1 * d_cols + c2] * v2;
                }
            }
        }

        let obs = y_vec[ri];
        let res = obs - y_hat;
        let denom = (1.0 - h_ii).max(0.08);
        let loo_err = if m_rows > 2 { res / denom } else { res };
        let loo_pred = obs - loo_err;

        ss_tot += (obs - y_mean).powi(2);
        ss_res += res * res;
        press_loo += loo_err * loo_err;

        let mol = &mols[al.mol_idx];
        let prof = &profiles[al.mol_idx];
        rows.push(RGroupRow {
            mol_idx: al.mol_idx,
            compound_id: mol.id.clone(),
            compound_name: mol.name.clone(),
            target: mol.target.clone(),
            r_groups: raw_r_rows[ri].clone(),
            core_atoms: al.core_atoms.clone(),
            observed_pact: obs,
            predicted_pact: y_hat,
            loo_predicted_pact: loo_pred,
            residual: res,
            non_additive_cliff: res.abs() >= 0.50,
            clogp: prof.clogp,
            qed_mean: prof.qed_mean,
        });
    }

    rows.sort_by(|a, b| {
        b.observed_pact
            .partial_cmp(&a.observed_pact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let rmse = (ss_res / (m_rows as f64)).sqrt();
    let r2 = if ss_tot > 1e-9 {
        (1.0 - ss_res / ss_tot).clamp(-1.0, 1.0)
    } else {
        1.0
    };
    let q2_loo = if ss_tot > 1e-9 {
        (1.0 - press_loo / ss_tot).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let mean_clogp: f64 =
        aligned.iter().map(|al| profiles[al.mol_idx].clogp).sum::<f64>() / (m_rows as f64);
    let mut effects: Vec<SubstituentEffect> = Vec::new();
    let mut coeff_lookup: Vec<HashMap<String, (f64, f64)>> = vec![HashMap::new(); p_slots];

    for si in 0..p_slots {
        let mut slot_effects = Vec::new();
        for (ki, sub) in slot_vocabs[si].iter().enumerate() {
            let col = slot_col_offsets[si] + ki;
            let beta = theta[col];
            let mut cnt = 0usize;
            let mut sum_p = 0.0f64;
            let mut sum_dlogp = 0.0f64;
            for (ri, al) in aligned.iter().enumerate() {
                if &raw_r_rows[ri][si] == sub {
                    cnt += 1;
                    sum_p += mols[al.mol_idx].pact;
                    sum_dlogp += profiles[al.mol_idx].clogp - mean_clogp;
                }
            }
            let mean_p = sum_p / (cnt.max(1) as f64);
            let mean_dl = sum_dlogp / (cnt.max(1) as f64);
            coeff_lookup[si].insert(sub.clone(), (beta, mean_dl));
            slot_effects.push(SubstituentEffect {
                slot_index: si,
                slot_label: slot_labels[si].clone(),
                substituent: sub.clone(),
                count: cnt,
                coeff_pact: beta,
                mean_obs_pact: mean_p,
                mean_delta_clogp: mean_dl,
            });
        }
        slot_effects.sort_by(|a, b| {
            b.coeff_pact
                .partial_cmp(&a.coeff_pact)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        effects.extend(slot_effects);
    }

    let pivot_r1_substituents: Vec<String> = effects
        .iter()
        .filter(|e| e.slot_index == 0)
        .take(12)
        .map(|e| e.substituent.clone())
        .collect();
    let pivot_r2_substituents: Vec<String> = if p_slots >= 2 {
        effects
            .iter()
            .filter(|e| e.slot_index == 1)
            .take(12)
            .map(|e| e.substituent.clone())
            .collect()
    } else {
        vec!["(Single-Slot)".to_string()]
    };

    let extra_slots_ref_beta: f64 = (2..p_slots)
        .map(|si| {
            effects
                .iter()
                .filter(|e| e.slot_index == si)
                .map(|e| e.coeff_pact)
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0)
        })
        .sum();

    let mut pivot_grid = Vec::with_capacity(pivot_r1_substituents.len());
    for r1_sub in &pivot_r1_substituents {
        let b1 = coeff_lookup[0].get(r1_sub).map(|x| x.0).unwrap_or(0.0);
        let mut grid_row = Vec::with_capacity(pivot_r2_substituents.len());
        for r2_sub in &pivot_r2_substituents {
            let b2 = if p_slots >= 2 {
                coeff_lookup[1].get(r2_sub).map(|x| x.0).unwrap_or(0.0)
            } else {
                0.0
            };

            let matching_row = rows.iter().find(|r| {
                &r.r_groups[0] == r1_sub && (p_slots < 2 || &r.r_groups[1] == r2_sub)
            });

            if let Some(obs_r) = matching_row {
                grid_row.push(SarPivotCell {
                    observed_mol: Some(obs_r.mol_idx),
                    observed_name: Some(obs_r.compound_name.clone()),
                    observed_pact: Some(obs_r.observed_pact),
                    predicted_pact: obs_r.predicted_pact,
                    residual: Some(obs_r.residual),
                });
            } else {
                grid_row.push(SarPivotCell {
                    observed_mol: None,
                    observed_name: None,
                    observed_pact: None,
                    predicted_pact: intercept_pact + b1 + b2 + extra_slots_ref_beta,
                    residual: None,
                });
            }
        }
        pivot_grid.push(grid_row);
    }

    let virtual_analogs = enumerate_virtual_analogs(
        &rows,
        &effects,
        &coeff_lookup,
        p_slots,
        intercept_pact,
        mean_clogp,
    );

    Ok(FreeWilsonMatrix {
        core_name: core_name.to_string(),
        core_smarts: core_smarts.to_string(),
        ridge_lambda: lambda,
        slot_labels,
        slot_query_atoms,
        rows,
        intercept_pact,
        r2,
        rmse,
        q2_loo,
        effects,
        pivot_r1_substituents,
        pivot_r2_substituents,
        pivot_grid,
        virtual_analogs,
    })
}

fn enumerate_virtual_analogs(
    observed_rows: &[RGroupRow],
    effects: &[SubstituentEffect],
    coeff_lookup: &[HashMap<String, (f64, f64)>],
    p_slots: usize,
    intercept: f64,
    mean_clogp: f64,
) -> Vec<VirtualAnalog> {
    if p_slots < 2 || observed_rows.is_empty() {
        return Vec::new();
    }

    let mut observed_tuples: BTreeSet<Vec<String>> = BTreeSet::new();
    for r in observed_rows {
        observed_tuples.insert(r.r_groups.clone());
    }

    let per_slot_cands: Vec<Vec<String>> = (0..p_slots)
        .map(|si| {
            effects
                .iter()
                .filter(|e| e.slot_index == si)
                .take(6)
                .map(|e| e.substituent.clone())
                .collect()
        })
        .collect();

    let mut analogs: Vec<VirtualAnalog> = Vec::new();
    let mut current: Vec<String> = Vec::with_capacity(p_slots);

    #[allow(clippy::too_many_arguments)]
    fn dfs_enum(
        si: usize,
        cands: &[Vec<String>],
        cur: &mut Vec<String>,
        observed_set: &BTreeSet<Vec<String>>,
        observed_rows: &[RGroupRow],
        lookup: &[HashMap<String, (f64, f64)>],
        intercept: f64,
        mean_clogp: f64,
        out: &mut Vec<VirtualAnalog>,
    ) {
        if si >= cands.len() {
            if observed_set.contains(cur) {
                return;
            }
            let mut pred_pact = intercept;
            let mut d_clogp = 0.0;
            for (slot_i, g) in cur.iter().enumerate() {
                if let Some(&(b, dl)) = lookup[slot_i].get(g) {
                    pred_pact += b;
                    d_clogp += dl;
                }
            }
            let est_clogp = mean_clogp + d_clogp;
            let est_lle = pred_pact - est_clogp;

            let mut best_shared = 0usize;
            let mut best_obs = &observed_rows[0];
            for r in observed_rows {
                let shared = r
                    .r_groups
                    .iter()
                    .zip(cur.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                if shared > best_shared
                    || (shared == best_shared && r.observed_pact > best_obs.observed_pact)
                {
                    best_shared = shared;
                    best_obs = r;
                }
            }

            out.push(VirtualAnalog {
                r_groups: cur.clone(),
                predicted_pact: pred_pact,
                predicted_ic50_nm: 10f64.powf(9.0 - pred_pact.clamp(2.0, 14.0)),
                est_clogp,
                est_lle,
                shared_r_count: best_shared,
                nearest_observed_mol: best_obs.mol_idx,
                nearest_observed_name: best_obs.compound_name.clone(),
            });
            return;
        }

        for g in &cands[si] {
            cur.push(g.clone());
            dfs_enum(
                si + 1,
                cands,
                cur,
                observed_set,
                observed_rows,
                lookup,
                intercept,
                mean_clogp,
                out,
            );
            cur.pop();
        }
    }

    dfs_enum(
        0,
        &per_slot_cands,
        &mut current,
        &observed_tuples,
        observed_rows,
        coeff_lookup,
        intercept,
        mean_clogp,
        &mut analogs,
    );

    analogs.sort_by(|a, b| {
        b.predicted_pact
            .partial_cmp(&a.predicted_pact)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.est_lle
                    .partial_cmp(&a.est_lle)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    analogs.truncate(25);
    analogs
}

fn canonicalize_core_embedding(
    mol: &Molecule,
    query: &SmartsQuery,
    mappings: &[Vec<usize>],
) -> (Vec<usize>, Vec<String>) {
    let mut best_map = mappings[0].clone();
    let mut best_subst = extract_substituents_for_embedding(mol, query, &mappings[0]);
    let mut best_rank = embedding_rank_key(&best_subst);

    for map in &mappings[1..] {
        let subst = extract_substituents_for_embedding(mol, query, map);
        let rank = embedding_rank_key(&subst);
        if rank > best_rank {
            best_rank = rank;
            best_map = map.clone();
            best_subst = subst;
        }
    }

    (best_map, best_subst)
}

fn embedding_rank_key(subst: &[String]) -> Vec<(u8, usize, String)> {
    subst
        .iter()
        .map(|s| {
            let non_h = (s != "-[H]") as u8;
            (non_h, s.len(), s.clone())
        })
        .collect()
}

fn extract_substituents_for_embedding(
    mol: &Molecule,
    query: &SmartsQuery,
    core_map: &[usize],
) -> Vec<String> {
    let qn = core_map.len();
    let tn = mol.atoms.len();
    let mut in_core = vec![None::<usize>; tn];
    for (qi, &ti) in core_map.iter().enumerate() {
        in_core[ti] = Some(qi);
    }

    let mut visited = vec![false; tn];
    let mut per_q_parts: Vec<Vec<String>> = vec![Vec::new(); qn];

    for (qi, &ti) in core_map.iter().enumerate() {
        if is_wildcard_or_multi_element(&query.atoms[qi]) {
            let sym = mol.atoms[ti].elem.symbol();
            let h = mol.atoms[ti].total_h();
            let endo = if h == 0 {
                format!("core:{sym}")
            } else {
                format!("core:{sym}H{}", if h > 1 { h.to_string() } else { String::new() })
            };
            per_q_parts[qi].push(endo);
        }
    }

    for start in 0..tn {
        if in_core[start].is_some() || visited[start] || mol.atoms[start].elem == Element::H {
            continue;
        }
        let mut comp_mask = vec![false; tn];
        let mut comp_atoms = Vec::new();
        let mut attached_q: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut q = VecDeque::new();
        visited[start] = true;
        comp_mask[start] = true;
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            comp_atoms.push(u);
            for &(v, _) in &mol.adj[u] {
                if let Some(qi) = in_core[v] {
                    attached_q.insert((qi, u));
                } else if !visited[v] && mol.atoms[v].elem != Element::H {
                    visited[v] = true;
                    comp_mask[v] = true;
                    q.push_back(v);
                }
            }
        }

        let distinct_q: BTreeSet<usize> = attached_q.iter().map(|&(qi, _)| qi).collect();
        if distinct_q.len() == 1 {
            if let Some(&(qi, r_root)) = attached_q.iter().next() {
                let bond_prefix = mol
                    .bond_between(core_map[qi], r_root)
                    .map(|bi| match mol.bonds[bi].kekule_order {
                        3 => "#",
                        2 => "=",
                        _ => "-",
                    })
                    .unwrap_or("-");
                let r_str = format_substituent_shorthand(mol, &comp_mask, r_root, bond_prefix);
                per_q_parts[qi].push(r_str);
            }
        } else if distinct_q.len() >= 2 {
            if let Some(&anchor_qi) = distinct_q.iter().next() {
                let bridge_lbl = format!(
                    "bridge:[{}#{:03x}]",
                    sub_formula(mol, &comp_mask),
                    wl_subgraph_hash(mol, &comp_mask, false) & 0xfff
                );
                per_q_parts[anchor_qi].push(bridge_lbl);
            }
        }
    }

    per_q_parts
        .into_iter()
        .map(|mut parts| {
            if parts.is_empty() {
                "-[H]".to_string()
            } else {
                parts.sort();
                parts.join(" + ")
            }
        })
        .collect()
}

fn is_wildcard_or_multi_element(expr: &crate::smarts::AtomExpr) -> bool {
    use crate::smarts::{AtomExpr, AtomPrim};
    match expr {
        AtomExpr::Or(_, _) => true,
        AtomExpr::Prim(
            AtomPrim::Wildcard
            | AtomPrim::AnyAromatic
            | AtomPrim::AnyAliphatic
            | AtomPrim::RingSize(_)
            | AtomPrim::InRingAny,
        ) => true,
        AtomExpr::And(a, b) => is_wildcard_or_multi_element(a) || is_wildcard_or_multi_element(b),
        _ => false,
    }
}

fn format_substituent_shorthand(
    mol: &Molecule,
    mask: &[bool],
    root: usize,
    bond_prefix: &str,
) -> String {
    let cnt = mask.iter().filter(|&&b| b).count();
    let el = mol.atoms[root].elem;
    let h = mol.atoms[root].total_h();

    if cnt == 1 {
        return match (el, h, bond_prefix) {
            (Element::C, 3, "-") => "-[CH3]".to_string(),
            (Element::N, 2, "-") => "-[NH2]".to_string(),
            (Element::O, 1, "-") => "-[OH]".to_string(),
            (Element::O, 0, "=") => "=[O]".to_string(),
            (Element::S, 0, "=") => "=[S]".to_string(),
            _ => format!("{}[{}]", bond_prefix, el.symbol()),
        };
    }

    if cnt == 2 {
        let other = (0..mol.atoms.len())
            .find(|&u| mask[u] && u != root)
            .unwrap_or(root);
        let inner_b = mol
            .bond_between(root, other)
            .map(|bi| match mol.bonds[bi].kekule_order {
                3 => "#",
                2 => "=",
                _ => "",
            })
            .unwrap_or("");
        return format!(
            "{}[{}{}{}]",
            bond_prefix,
            atom_h_str(mol, root),
            inner_b,
            atom_h_str(mol, other)
        );
    }

    if cnt == 3 {
        let nbrs: Vec<usize> = mol.adj[root]
            .iter()
            .map(|&(v, _)| v)
            .filter(|&v| mask[v])
            .collect();
        if nbrs.len() == 1 {
            let mid = nbrs[0];
            if let Some(leaf) = (0..mol.atoms.len()).find(|&u| mask[u] && u != root && u != mid) {
                return format!(
                    "{}[{}{}{}]",
                    bond_prefix,
                    atom_h_str(mol, root),
                    atom_h_str(mol, mid),
                    atom_h_str(mol, leaf)
                );
            }
        }
    }

    if cnt == 4 && el == Element::C {
        let f_cnt = mol.adj[root]
            .iter()
            .filter(|&&(v, _)| mask[v] && mol.atoms[v].elem == Element::F)
            .count();
        if f_cnt == 3 {
            return "-[CF3]".to_string();
        }
    }

    let sig = wl_subgraph_hash(mol, mask, false) & 0xfff;
    format!("{}[{}#{:03x}]", bond_prefix, sub_formula(mol, mask), sig)
}

fn atom_h_str(mol: &Molecule, u: usize) -> String {
    let s = mol.atoms[u].elem.symbol();
    let h = mol.atoms[u].total_h();
    match h {
        0 => s.to_string(),
        1 => format!("{s}H"),
        _ => format!("{s}H{h}"),
    }
}

fn sub_formula(mol: &Molecule, mask: &[bool]) -> String {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (u, &on) in mask.iter().enumerate() {
        if on && mol.atoms[u].elem != Element::H {
            *counts.entry(mol.atoms[u].elem.symbol()).or_insert(0) += 1;
        }
    }
    let mut out = String::new();
    if let Some(&c) = counts.get("C") {
        out.push('C');
        if c > 1 {
            out.push_str(&c.to_string());
        }
        counts.remove("C");
    }
    for (sym, cnt) in counts {
        out.push_str(sym);
        if cnt > 1 {
            out.push_str(&cnt.to_string());
        }
    }
    out
}

pub fn scaffold_subgraph_to_smarts(mol: &Molecule, keep_atoms: &[usize]) -> Option<String> {
    if keep_atoms.is_empty() {
        return None;
    }
    let n = mol.atoms.len();
    let mut keep = vec![false; n];
    for &u in keep_atoms {
        if u < n {
            keep[u] = true;
        }
    }
    let start = keep_atoms[0];

    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    let mut tree_children: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut back_edges: Vec<(usize, usize, usize)> = Vec::new();

    fn dfs_span(
        u: usize,
        mol: &Molecule,
        keep: &[bool],
        vis: &mut [bool],
        par: &mut [usize],
        children: &mut [Vec<(usize, usize)>],
        backs: &mut Vec<(usize, usize, usize)>,
    ) {
        vis[u] = true;
        for &(v, bi) in &mol.adj[u] {
            if !keep[v] {
                continue;
            }
            if !vis[v] {
                par[v] = u;
                children[u].push((v, bi));
                dfs_span(v, mol, keep, vis, par, children, backs);
            } else if v != par[u] && u < v {
                backs.push((u, v, bi));
            }
        }
    }

    dfs_span(
        start,
        mol,
        &keep,
        &mut visited,
        &mut parent,
        &mut tree_children,
        &mut back_edges,
    );

    let mut ring_tags: Vec<Vec<(u16, &'static str)>> = vec![Vec::new(); n];
    for (ri, &(u, v, bi)) in back_edges.iter().enumerate() {
        let rnum = (ri + 1) as u16;
        let b_str = smarts_bond_token(mol, bi);
        ring_tags[u].push((rnum, b_str));
        ring_tags[v].push((rnum, ""));
    }

    fn emit_dfs(
        u: usize,
        in_bi: Option<usize>,
        mol: &Molecule,
        children: &[Vec<(usize, usize)>],
        rtags: &[Vec<(u16, &'static str)>],
        out: &mut String,
    ) {
        if let Some(bi) = in_bi {
            out.push_str(smarts_bond_token(mol, bi));
        }
        let a = &mol.atoms[u];
        let atom_tok = match (a.elem, a.aromatic) {
            (Element::C, true) => "c".to_string(),
            (Element::C, false) => "C".to_string(),
            (Element::N, true) => "n".to_string(),
            (Element::N, false) => "N".to_string(),
            (Element::O, true) => "o".to_string(),
            (Element::O, false) => "O".to_string(),
            (Element::S, true) => "s".to_string(),
            (Element::S, false) => "S".to_string(),
            _ => format!("[#{}]", a.elem.atomic_number()),
        };
        out.push_str(&atom_tok);
        for &(rnum, b_tok) in &rtags[u] {
            out.push_str(b_tok);
            if rnum < 10 {
                out.push_str(&rnum.to_string());
            } else {
                out.push_str(&format!("%{rnum:02}"));
            }
        }
        let ch = &children[u];
        for (idx, &(v, bi)) in ch.iter().enumerate() {
            let branch = idx + 1 < ch.len();
            if branch {
                out.push('(');
            }
            emit_dfs(v, Some(bi), mol, children, rtags, out);
            if branch {
                out.push(')');
            }
        }
    }

    let mut sm = String::new();
    emit_dfs(start, None, mol, &tree_children, &ring_tags, &mut sm);
    Some(sm)
}

fn smarts_bond_token(mol: &Molecule, bi: usize) -> &'static str {
    let b = &mol.bonds[bi];
    if b.aromatic {
        ""
    } else {
        match b.kekule_order {
            3 => "#",
            2 => "=",
            _ => "-",
        }
    }
}

fn invert_spd_matrix(mat: &[f64], n: usize) -> Vec<f64> {
    let mut a = mat.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }

    for col in 0..n {
        let mut pivot = col;
        let mut max_abs = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = row;
            }
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }

        let diag = a[col * n + col];
        let inv_diag = if diag.abs() > 1e-12 { 1.0 / diag } else { 0.0 };
        for k in 0..n {
            a[col * n + k] *= inv_diag;
            inv[col * n + k] *= inv_diag;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor.abs() > 1e-15 {
                for k in 0..n {
                    a[row * n + k] -= factor * a[col * n + k];
                    inv[row * n + k] -= factor * inv[col * n + k];
                }
            }
        }
    }
    inv
}
