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

//! Multi-Objective Pareto Frontier Ranker (NSGA-II Non-Dominated Sorting, Crowding Distance,
//! 3D `(Potency, QED, ADMET)` Hypervolume, and `JSON`/`CSV` Frontier Export).

use crate::admet::{AdmetProfile, BoiledEggRegion, HergRisk};
use crate::DatasetAnalysis;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

/// Multi-objective frontier optimization preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParetoObjectiveProfile {
    /// Maximize `(pAct, QED, ADMET)` and minimize PAINS structural alerts.
    BalancedAdmet,
    /// Maximize `(pAct, QED, ADMET, CNS-MPO)` for blood-brain-barrier CNS optimization.
    CnsMpo,
    /// Maximize `(pAct, LLE, QED, ADMET)` to prioritize ligand lipophilic efficiency.
    PotencyAndLle,
}

impl ParetoObjectiveProfile {
    pub const ALL: [Self; 3] = [Self::BalancedAdmet, Self::CnsMpo, Self::PotencyAndLle];

    pub fn from_code(raw: &str) -> Self {
        let s = raw.trim().to_ascii_lowercase();
        if s.contains("cns") || s.contains("mpo") || s.contains("bbb") {
            Self::CnsMpo
        } else if s.contains("lle") || s.contains("eff") {
            Self::PotencyAndLle
        } else {
            Self::BalancedAdmet
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::BalancedAdmet => "balanced-admet",
            Self::CnsMpo => "cns-mpo",
            Self::PotencyAndLle => "potency-lle",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BalancedAdmet => "Balanced Potency / QED / ADMET",
            Self::CnsMpo => "CNS-MPO / BBB / QED / ADMET",
            Self::PotencyAndLle => "Potency / LLE / QED / ADMET",
        }
    }
}

/// Quantitative compound record ranked on a non-dominated Pareto shell (`pareto_rank >= 1`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoLeadEntry {
    pub mol_idx: usize,
    pub id: String,
    pub name: String,
    pub target: String,
    pub smiles: String,
    pub pact: f64,
    pub qed: f64,
    pub cns_mpo: f64,
    pub lle: f64,
    pub esol_logs: f64,
    pub sa_score: f64,
    pub pains_alerts: usize,
    pub admet_composite_score: f64,
    pub pareto_rank: usize,
    pub crowding_distance: f64,
    pub dominated_by_count: usize,
    pub dominates_count: usize,
    pub hypervolume_contrib_est: f64,
}

/// Complete NSGA-II non-dominated frontier report across a compound series.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoFrontierReport {
    pub profile: ParetoObjectiveProfile,
    pub total_compounds: usize,
    pub frontier_count: usize,
    pub front_layers: usize,
    pub hypervolume_3d: f64,
    pub entries: Vec<ParetoLeadEntry>,
}

/// Compute a normalized `[0.0, 1.0]` composite ADMET & developability score from quantitative ADMET + SA metrics.
pub fn compute_admet_composite_score(ad: &AdmetProfile, sa_score: f64, pains_alerts: usize) -> f64 {
    // ESOL solubility [-7..0] mapped to [0..1]
    let sol_s = ((ad.esol_logs + 7.0) / 5.5).clamp(0.0, 1.0);
    let egg_s = match ad.boiled_egg {
        BoiledEggRegion::YolkBbb => 1.0,
        BoiledEggRegion::WhiteHia => 0.85,
        BoiledEggRegion::OutsideLow => 0.25,
    };
    let pgp_s = if ad.pgp_substrate { 0.45 } else { 1.0 };
    let pfizer_s = if ad.pfizer_3_75_pass { 1.0 } else { 0.35 };
    let gsk_s = if ad.gsk_4_400_pass { 1.0 } else { 0.45 };
    let herg_s = match ad.herg_risk {
        HergRisk::Low => 1.0,
        HergRisk::Moderate => 0.55,
        HergRisk::High => 0.15,
    };
    let sa_s = (1.0 - (sa_score - 2.0) / 6.0).clamp(0.0, 1.0);
    let pains_penalty = if pains_alerts == 0 {
        1.0
    } else {
        (1.0 - 0.25 * (pains_alerts as f64)).max(0.1)
    };
    let raw = 0.20 * sol_s
        + 0.15 * egg_s
        + 0.10 * pgp_s
        + 0.15 * pfizer_s
        + 0.10 * gsk_s
        + 0.18 * herg_s
        + 0.12 * sa_s;
    (raw * pains_penalty).clamp(0.0, 1.0)
}

fn dominates(a: &[f64; 4], b: &[f64; 4]) -> bool {
    let ge_all = (0..4).all(|k| a[k] >= b[k] - 1e-9);
    let gt_any = (0..4).any(|k| a[k] > b[k] + 1e-7);
    ge_all && gt_any
}

/// Exact 3D dominated hypervolume indicator in normalized `[0, 1]^3` `(pAct, QED, ADMET)` space
/// relative to reference origin `(0.0, 0.0, 0.0)`.
fn compute_3d_hypervolume(front1_pts: &[[f64; 3]]) -> f64 {
    if front1_pts.is_empty() {
        return 0.0;
    }
    let mut pts: Vec<[f64; 3]> = front1_pts
        .iter()
        .map(|&[x, y, z]| [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), z.clamp(0.0, 1.0)])
        .collect();
    // Sort by z descending for exact 2D-slice sweep integration along z
    pts.sort_by(|a, b| b[2].partial_cmp(&a[2]).unwrap_or(std::cmp::Ordering::Equal));

    let mut hv = 0.0_f64;
    let mut active: Vec<[f64; 2]> = Vec::new();
    for i in 0..pts.len() {
        active.push([pts[i][0], pts[i][1]]);
        let z_hi = pts[i][2];
        let z_lo = if i + 1 < pts.len() { pts[i + 1][2] } else { 0.0 };
        let dz = (z_hi - z_lo).max(0.0);
        if dz <= 1e-12 {
            continue;
        }
        // Compute union area of rectangles [0, x] x [0, y] for active points
        let mut sorted = active.clone();
        sorted.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap_or(std::cmp::Ordering::Equal));
        let mut area_2d = 0.0_f64;
        let mut max_y = 0.0_f64;
        for j in 0..sorted.len() {
            let x_hi = sorted[j][0];
            let x_lo = if j + 1 < sorted.len() { sorted[j + 1][0] } else { 0.0 };
            if sorted[j][1] > max_y {
                max_y = sorted[j][1];
            }
            area_2d += (x_hi - x_lo).max(0.0) * max_y;
        }
        hv += area_2d * dz;
    }
    hv.clamp(0.0, 1.0)
}

/// Perform NSGA-II non-dominated sorting, crowding distance assignment, and 3D hypervolume
/// evaluation across all compounds in a `DatasetAnalysis`.
pub fn rank_pareto_frontier(
    ds: &DatasetAnalysis,
    profile: ParetoObjectiveProfile,
) -> ParetoFrontierReport {
    let n = ds.mols.len();
    if n == 0 {
        return ParetoFrontierReport {
            profile,
            total_compounds: 0,
            frontier_count: 0,
            front_layers: 0,
            hypervolume_3d: 0.0,
            entries: Vec::new(),
        };
    }

    let mut objs: Vec<[f64; 4]> = Vec::with_capacity(n);
    let mut admet_scores = Vec::with_capacity(n);
    let mut hv_pts = Vec::with_capacity(n);

    for i in 0..n {
        let m = &ds.mols[i];
        let p = &ds.profiles[i];
        let ad = &ds.admet[i];
        let sa = ds.synth_access[i].sa_score;
        let alerts = ds.alerts[i].len();
        let admet_score = compute_admet_composite_score(ad, sa, alerts);
        admet_scores.push(admet_score);

        let pact_safe = if m.pact.is_finite() { m.pact } else { 0.0 };
        let qed_safe = if p.qed_mean.is_finite() {
            p.qed_mean.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let obj4 = match profile {
            ParetoObjectiveProfile::BalancedAdmet => {
                [pact_safe, qed_safe, admet_score, -(alerts as f64)]
            }
            ParetoObjectiveProfile::CnsMpo => [
                pact_safe,
                qed_safe,
                admet_score,
                if p.cns_mpo.is_finite() { p.cns_mpo } else { 0.0 },
            ],
            ParetoObjectiveProfile::PotencyAndLle => [
                pact_safe,
                qed_safe,
                admet_score,
                if p.lle.is_finite() { p.lle } else { 0.0 },
            ],
        };
        objs.push(obj4);

        let norm_p = ((pact_safe - 4.0) / 6.0).clamp(0.05, 1.0);
        hv_pts.push([norm_p, qed_safe, admet_score]);
    }

    let mut dominates_list: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dom_count: Vec<usize> = vec![0; n];
    let mut init_dom_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut front1: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&objs[i], &objs[j]) {
                dominates_list[i].push(j);
            } else if dominates(&objs[j], &objs[i]) {
                dom_count[i] += 1;
            }
        }
        init_dom_count[i] = dom_count[i];
        if dom_count[i] == 0 {
            front1.push(i);
        }
    }

    let mut ranks = vec![0usize; n];
    let mut cur_front = front1;
    let mut rank_num = 1usize;
    while !cur_front.is_empty() {
        for &idx in &cur_front {
            ranks[idx] = rank_num;
        }
        let mut next_front = Vec::new();
        for &i in &cur_front {
            for &j in &dominates_list[i] {
                if dom_count[j] > 0 {
                    dom_count[j] -= 1;
                    if dom_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
        }
        fronts.push(cur_front);
        cur_front = next_front;
        rank_num += 1;
    }

    // NSGA-II crowding distance within each Pareto shell
    let mut crowding = vec![0.0_f64; n];
    for front in &fronts {
        if front.len() <= 2 {
            for &idx in front {
                crowding[idx] = f64::INFINITY;
            }
            continue;
        }
        for k in 0..4 {
            let mut order = front.clone();
            order.sort_by(|&a, &b| {
                objs[a][k]
                    .partial_cmp(&objs[b][k])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let first = order[0];
            let last = order[order.len() - 1];
            crowding[first] = f64::INFINITY;
            crowding[last] = f64::INFINITY;
            let span = objs[last][k] - objs[first][k];
            if span <= 1e-12 {
                continue;
            }
            for pos in 1..(order.len() - 1) {
                let idx = order[pos];
                if crowding[idx].is_finite() {
                    let prev_v = objs[order[pos - 1]][k];
                    let next_v = objs[order[pos + 1]][k];
                    crowding[idx] += (next_v - prev_v) / span;
                }
            }
        }
    }

    let front1_hv_pts: Vec<[f64; 3]> = fronts
        .first()
        .map(|f| f.iter().map(|&idx| hv_pts[idx]).collect())
        .unwrap_or_default();
    let hypervolume_3d = compute_3d_hypervolume(&front1_hv_pts);
    let frontier_count = fronts.first().map(|f| f.len()).unwrap_or(0);
    let front_layers = fronts.len();

    let mut entries: Vec<ParetoLeadEntry> = (0..n)
        .map(|i| {
            let m = &ds.mols[i];
            let p = &ds.profiles[i];
            let ad = &ds.admet[i];
            let sa = ds.synth_access[i].sa_score;
            let hv_proxy = hv_pts[i][0] * hv_pts[i][1] * hv_pts[i][2];
            ParetoLeadEntry {
                mol_idx: i,
                id: m.id.clone(),
                name: m.name.clone(),
                target: m.target.clone(),
                smiles: m.smiles.clone(),
                pact: m.pact,
                qed: p.qed_mean,
                cns_mpo: p.cns_mpo,
                lle: p.lle,
                esol_logs: ad.esol_logs,
                sa_score: sa,
                pains_alerts: ds.alerts[i].len(),
                admet_composite_score: admet_scores[i],
                pareto_rank: ranks[i],
                crowding_distance: crowding[i],
                dominated_by_count: init_dom_count[i],
                dominates_count: dominates_list[i].len(),
                hypervolume_contrib_est: hv_proxy,
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        a.pareto_rank
            .cmp(&b.pareto_rank)
            .then_with(|| {
                let ca = if a.crowding_distance.is_infinite() {
                    1.0e9
                } else {
                    a.crowding_distance
                };
                let cb = if b.crowding_distance.is_infinite() {
                    1.0e9
                } else {
                    b.crowding_distance
                };
                cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.hypervolume_contrib_est
                    .partial_cmp(&a.hypervolume_contrib_est)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    ParetoFrontierReport {
        profile,
        total_compounds: n,
        frontier_count,
        front_layers,
        hypervolume_3d,
        entries,
    }
}

fn json_escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn csv_escape(raw: &str) -> String {
    if raw.contains(',') || raw.contains('"') || raw.contains('\n') || raw.contains('\r') {
        format!("\"{}\"", raw.replace('"', "\"\""))
    } else {
        raw.to_string()
    }
}

/// Export the Pareto multi-objective frontier ranking to RFC 8259 JSON.
pub fn export_pareto_frontier_json(report: &ParetoFrontierReport) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("{\n");
    let _ = writeln!(out, "  \"profile\": \"{}\",", report.profile.code());
    let _ = writeln!(out, "  \"total_compounds\": {},", report.total_compounds);
    let _ = writeln!(out, "  \"frontier_count\": {},", report.frontier_count);
    let _ = writeln!(out, "  \"front_layers\": {},", report.front_layers);
    let _ = writeln!(out, "  \"hypervolume_3d\": {:.6},", report.hypervolume_3d);
    out.push_str("  \"entries\": [\n");
    for (idx, e) in report.entries.iter().enumerate() {
        let comma = if idx + 1 < report.entries.len() { "," } else { "" };
        let crowd = if e.crowding_distance.is_finite() {
            format!("{:.4}", e.crowding_distance)
        } else {
            "null".to_string()
        };
        out.push_str("    {\n");
        let _ = writeln!(out, "      \"pareto_rank\": {},", e.pareto_rank);
        let _ = writeln!(out, "      \"id\": \"{}\",", json_escape(&e.id));
        let _ = writeln!(out, "      \"name\": \"{}\",", json_escape(&e.name));
        let _ = writeln!(out, "      \"target\": \"{}\",", json_escape(&e.target));
        let _ = writeln!(out, "      \"pact\": {:.3},", e.pact);
        let _ = writeln!(out, "      \"qed\": {:.4},", e.qed);
        let _ = writeln!(out, "      \"cns_mpo\": {:.3},", e.cns_mpo);
        let _ = writeln!(out, "      \"lle\": {:.3},", e.lle);
        let _ = writeln!(out, "      \"esol_logs\": {:.3},", e.esol_logs);
        let _ = writeln!(out, "      \"sa_score\": {:.3},", e.sa_score);
        let _ = writeln!(out, "      \"pains_alerts\": {},", e.pains_alerts);
        let _ = writeln!(
            out,
            "      \"admet_composite_score\": {:.4},",
            e.admet_composite_score
        );
        let _ = writeln!(out, "      \"crowding_distance\": {crowd},");
        let _ = writeln!(out, "      \"crowding_boundary\": {},", !e.crowding_distance.is_finite());
        let _ = writeln!(out, "      \"dominated_by_count\": {},", e.dominated_by_count);
        let _ = writeln!(out, "      \"dominates_count\": {},", e.dominates_count);
        let _ = writeln!(
            out,
            "      \"hypervolume_contrib_est\": {:.5},",
            e.hypervolume_contrib_est
        );
        let _ = writeln!(out, "      \"smiles\": \"{}\"", json_escape(&e.smiles));
        let _ = writeln!(out, "    }}{comma}");
    }
    out.push_str("  ]\n}\n");
    out
}

/// Export the Pareto multi-objective frontier ranking to RFC 4180 CSV.
pub fn export_pareto_frontier_csv(report: &ParetoFrontierReport) -> String {
    let mut out = String::from(
        "pareto_rank,id,name,target,pact,qed,cns_mpo,lle,esol_logs,sa_score,pains_alerts,admet_composite_score,crowding_distance,dominated_by_count,dominates_count,hypervolume_contrib_est,smiles\n",
    );
    for e in &report.entries {
        let crowd = if e.crowding_distance.is_finite() {
            format!("{:.4}", e.crowding_distance)
        } else {
            "INF".to_string()
        };
        let _ = writeln!(
            out,
            "{},{},{},{},{:.3},{:.4},{:.3},{:.3},{:.3},{:.3},{},{:.4},{},{},{},{:.5},{}",
            e.pareto_rank,
            csv_escape(&e.id),
            csv_escape(&e.name),
            csv_escape(&e.target),
            e.pact,
            e.qed,
            e.cns_mpo,
            e.lle,
            e.esol_logs,
            e.sa_score,
            e.pains_alerts,
            e.admet_composite_score,
            crowd,
            e.dominated_by_count,
            e.dominates_count,
            e.hypervolume_contrib_est,
            csv_escape(&e.smiles),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load_benchmark_26, parse_smiles};

    #[test]
    fn test_pareto_frontier_non_dominated_sorting_and_exports() {
        let mols = load_benchmark_26();
        let ds = DatasetAnalysis::from_molecules(mols, 0.30, 6.0);
        let rep = rank_pareto_frontier(&ds, ParetoObjectiveProfile::BalancedAdmet);

        assert_eq!(rep.total_compounds, ds.mols.len());
        assert!(rep.frontier_count >= 1);
        assert!(rep.front_layers >= 2);
        assert!(rep.hypervolume_3d > 0.0 && rep.hypervolume_3d <= 1.0);

        for e in &rep.entries {
            if e.pareto_rank == 1 {
                assert_eq!(e.dominated_by_count, 0);
            } else {
                assert!(e.dominated_by_count >= 1);
            }
        }

        let json = export_pareto_frontier_json(&rep);
        assert!(json.contains("\"frontier_count\""));
        assert!(json.contains("\"hypervolume_3d\""));
        assert!(json.contains("CHEMBL553"));

        let csv = export_pareto_frontier_csv(&rep);
        assert_eq!(csv.lines().count(), rep.total_compounds + 1);
    }

    #[test]
    fn test_pareto_frontier_strict_dominance_and_empty_edge_case() {
        let empty = DatasetAnalysis::from_molecules(Vec::new(), 0.30, 6.0);
        let empty_rep = rank_pareto_frontier(&empty, ParetoObjectiveProfile::CnsMpo);
        assert_eq!(empty_rep.total_compounds, 0);
        assert_eq!(empty_rep.frontier_count, 0);
        assert_eq!(empty_rep.hypervolume_3d, 0.0);

        // Two molecules with identical scaffolds: one strictly higher potency must dominate the other
        let mut high = parse_smiles("c1ccccc1N").expect("valid");
        high.id = "HIGH".to_string();
        high.name = "HighLead".to_string();
        high.pact = 9.0;
        let mut low = parse_smiles("c1ccccc1N").expect("valid");
        low.id = "LOW".to_string();
        low.name = "LowLead".to_string();
        low.pact = 5.0;

        let ds = DatasetAnalysis::from_molecules(vec![high, low], 0.30, 6.0);
        let rep = rank_pareto_frontier(&ds, ParetoObjectiveProfile::PotencyAndLle);
        assert_eq!(rep.frontier_count, 1);
        assert_eq!(rep.entries[0].id, "HIGH");
        assert_eq!(rep.entries[0].pareto_rank, 1);
        assert_eq!(rep.entries[1].id, "LOW");
        assert_eq!(rep.entries[1].pareto_rank, 2);
    }
}
