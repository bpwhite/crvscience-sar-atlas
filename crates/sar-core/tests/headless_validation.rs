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

use sar_core::{
    export_pbpk_timecourse_csv, load_benchmark_26, run_active_learning_campaign,
    run_rbfe_network_campaign, simulate_oral_pbpk, ActiveLearningConfig, DatasetAnalysis,
    PbpkDosingConfig, RbfeNetworkConfig,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

struct PbpkSnap {
    compound_id: String,
    compound_name: String,
    fa_integrated: f64,
    fg_gut_wall: f64,
    fh_hepatic: f64,
    f_oral_pct: f64,
    cl_hepatic_blood_l_hr: f64,
    cl_renal_plasma_l_hr: f64,
    hepatic_extraction_eh: f64,
    vss_l_per_kg: f64,
    kp_uu_brain: f64,
    cmax_total_plasma_nm: f64,
    cmax_unbound_plasma_nm: f64,
    cmax_unbound_brain_nm: f64,
    tmax_plasma_hr: f64,
    auc_total_plasma_0_24_nm_hr: f64,
    auc_unbound_plasma_nm_hr: f64,
    auc_unbound_brain_nm_hr: f64,
    terminal_t_half_hr: f64,
    peak_ro_brain_pct: f64,
}

struct SarAtlasValidationData {
    // T1
    mol_count: usize,
    profile_count: usize,
    mean_cns_mpo: f64,
    mean_qed: f64,
    ro5_zero_violation_count: usize,
    veber_pass_count: usize,
    mmp_count: usize,
    max_mmp_abs_delta_pact: f64,
    cliff_edge_count: usize,
    max_sali: f64,
    pareto_lead_count: usize,
    // T2
    halo: PbpkSnap,
    lap: PbpkSnap,
    // T3
    rbfe_target: String,
    rbfe_n_nodes: usize,
    rbfe_n_edges: usize,
    rbfe_n_cycles: usize,
    rbfe_mean_lomap: f64,
    rbfe_mean_hysteresis_kcal: f64,
    rbfe_max_hysteresis_kcal: f64,
    rbfe_post_mle_max_cycle_error_kcal: f64,
    rbfe_dg_rmse_kcal: f64,
    rbfe_pact_r2: f64,
    rbfe_kendall_tau: f64,
    al_pool_size: usize,
    al_selected_batch_count: usize,
    al_batch_mean_pred_pact: f64,
    al_batch_mean_epistemic_std: f64,
    al_batch_mean_dissim: f64,
    al_top_name: String,
    al_top_ei: f64,
    al_top_selected: bool,
    // Files
    csv_path: PathBuf,
    csv_bytes: u64,
    json_path: PathBuf,
    json_bytes: u64,
}

static SAR_SUITE: OnceLock<SarAtlasValidationData> = OnceLock::new();

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn validation_dir() -> PathBuf {
    let dir = repo_root().join("validation");
    fs::create_dir_all(&dir).expect("create validation directory");
    dir
}

fn trapezoidal_auc(times: &[f32], conc: &[f32]) -> f64 {
    let mut auc = 0.0_f64;
    for i in 1..times.len().min(conc.len()) {
        let dt = (times[i] - times[i - 1]) as f64;
        if dt > 0.0 {
            auc += 0.5 * ((conc[i - 1] + conc[i]) as f64) * dt;
        }
    }
    auc
}

fn pbpk_snap(rep: &sar_core::OralPbpkReport) -> PbpkSnap {
    let auc_total_0_24 = trapezoidal_auc(&rep.time_hr, &rep.c_total_plasma_nm);
    PbpkSnap {
        compound_id: rep.compound_id.clone(),
        compound_name: rep.compound_name.clone(),
        fa_integrated: rep.fa_integrated,
        fg_gut_wall: rep.fg_gut_wall,
        fh_hepatic: rep.fh_hepatic,
        f_oral_pct: rep.f_oral_pct,
        cl_hepatic_blood_l_hr: rep.cl_hepatic_blood_l_hr,
        cl_renal_plasma_l_hr: rep.cl_renal_plasma_l_hr,
        hepatic_extraction_eh: rep.hepatic_extraction_eh,
        vss_l_per_kg: rep.vss_l_per_kg,
        kp_uu_brain: rep.kp_uu_brain,
        cmax_total_plasma_nm: rep.cmax_total_plasma_nm,
        cmax_unbound_plasma_nm: rep.cmax_unbound_plasma_nm,
        cmax_unbound_brain_nm: rep.cmax_unbound_brain_nm,
        tmax_plasma_hr: rep.tmax_plasma_hr,
        auc_total_plasma_0_24_nm_hr: auc_total_0_24,
        auc_unbound_plasma_nm_hr: rep.auc_unbound_plasma_nm_hr,
        auc_unbound_brain_nm_hr: rep.auc_unbound_brain_nm_hr,
        terminal_t_half_hr: rep.terminal_t_half_hr,
        peak_ro_brain_pct: rep.peak_ro_brain_pct,
    }
}

fn ensure_sar_validation() -> &'static SarAtlasValidationData {
    SAR_SUITE.get_or_init(|| {
        let val_dir = validation_dir();

        // ---------------------------------------------------------------------
        // T1: CNS MPO + QED + Lipinski Desirability & MMP Activity Cliffs
        // ---------------------------------------------------------------------
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let mol_count = ds.mols.len();
        let profile_count = ds.profiles.len();
        let mean_cns_mpo =
            ds.profiles.iter().map(|p| p.cns_mpo).sum::<f64>() / profile_count as f64;
        let mean_qed = ds.profiles.iter().map(|p| p.qed_mean).sum::<f64>() / profile_count as f64;
        let ro5_zero_violation_count = ds
            .profiles
            .iter()
            .filter(|p| p.ro5_violations == 0)
            .count();
        let veber_pass_count = ds.profiles.iter().filter(|p| p.veber_pass).count();
        let mmp_count = ds.mmps.len();
        let max_mmp_abs_delta_pact = ds
            .mmps
            .iter()
            .map(|m| m.delta_pact.abs())
            .fold(0.0_f64, f64::max);
        let cliff_edge_count = ds.edges.iter().filter(|e| e.is_cliff).count();
        let max_sali = ds
            .edges
            .iter()
            .map(|e| e.sali as f64)
            .fold(0.0_f64, f64::max);
        let pareto_lead_count = ds.pareto_leads.len();
        let pareto_names: Vec<String> = ds
            .pareto_leads
            .iter()
            .filter_map(|&idx| ds.mols.get(idx).map(|m| m.name.clone()))
            .collect();

        // ---------------------------------------------------------------------
        // T2: RK4 Compartmental Oral PBPK (CNS vs. Peripheral Separation)
        // ---------------------------------------------------------------------
        let halo_mol = ds
            .mols
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("Haloperidol"))
            .expect("Haloperidol in benchmark 26");
        let lap_mol = ds
            .mols
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("Lapatinib"))
            .expect("Lapatinib in benchmark 26");

        let cfg_pbpk = PbpkDosingConfig::default();
        let rep_halo = simulate_oral_pbpk(halo_mol, cfg_pbpk);
        let rep_lap = simulate_oral_pbpk(lap_mol, cfg_pbpk);

        let halo = pbpk_snap(&rep_halo);
        let lap = pbpk_snap(&rep_lap);

        let mut csv = String::from(
            "compound_id,compound_name,time_hr,gi_absorbed_pct,c_total_plasma_nm,c_unbound_plasma_nm,c_unbound_brain_nm,ro_plasma_pct,ro_brain_pct,target_ic50_nm\n",
        );
        for rep in [&rep_halo, &rep_lap] {
            let single_csv = export_pbpk_timecourse_csv(rep);
            for line in single_csv.lines().skip(1) {
                if !line.trim().is_empty() {
                    csv.push_str(&format!("{},{},{}\n", rep.compound_id, rep.compound_name, line));
                }
            }
        }
        let csv_path = val_dir.join("pbpk_plasma_profile.csv");
        fs::write(&csv_path, &csv).expect("write pbpk_plasma_profile.csv");
        let csv_bytes = fs::metadata(&csv_path).expect("stat pbpk csv").len();

        // ---------------------------------------------------------------------
        // T3: Lomap RBFE Closed-Cycle Network & MolPAL Active Learning
        // ---------------------------------------------------------------------
        let rbfe = run_rbfe_network_campaign(&ds, &RbfeNetworkConfig::default());
        let al = run_active_learning_campaign(&ds, &ActiveLearningConfig::default());

        let al_top = al.candidates.first().expect("ranked active learning candidates");
        let top_candidates_json: Vec<_> = al
            .candidates
            .iter()
            .take(5)
            .map(|c| {
                json!({
                    "rank": c.rank,
                    "candidate_id": c.candidate_id,
                    "candidate_name": c.candidate_name,
                    "predicted_pact": c.predicted_pact,
                    "epistemic_std": c.epistemic_std,
                    "expected_improvement": c.expected_improvement,
                    "ucb_score": c.ucb_score,
                    "selected_in_batch": c.selected_in_batch
                })
            })
            .collect();

        let summary = json!({
            "schema_version": "1.0",
            "repo": "crvscience-sar-atlas",
            "preprint_id": "APP-PREPRINT-09",
            "preprint_url": "https://crvscience.com/apps/preprints/sar-atlas-multiparameter-pbpk-rbfe",
            "concept_doi": "10.5281/zenodo.24222390",
            "version": "3.8.0",
            "testsPassed": 3,
            "testsTotal": 3,
            "tests": {
                "t1_cns_mpo_qed_lipinski_and_mmp_cliffs": {
                    "engine_calls": ["DatasetAnalysis::from_molecules", "load_benchmark_26"],
                    "dataset_molecules": mol_count,
                    "mean_cns_mpo": mean_cns_mpo,
                    "mean_qed": mean_qed,
                    "ro5_zero_violation_count": ro5_zero_violation_count,
                    "veber_pass_count": veber_pass_count,
                    "mmp_pair_count": mmp_count,
                    "max_mmp_abs_delta_pact": max_mmp_abs_delta_pact,
                    "sali_cliff_edge_count": cliff_edge_count,
                    "max_sali": max_sali,
                    "pareto_lead_count": pareto_lead_count,
                    "pareto_lead_names": pareto_names
                },
                "t2_rk4_oral_pbpk_and_brain_kpuu": {
                    "engine_calls": ["simulate_oral_pbpk", "export_pbpk_timecourse_csv"],
                    "dose_mg": cfg_pbpk.dose_mg,
                    "haloperidol": {
                        "compound_id": halo.compound_id,
                        "fa_integrated": halo.fa_integrated,
                        "fg_gut_wall": halo.fg_gut_wall,
                        "fh_hepatic": halo.fh_hepatic,
                        "f_oral_pct": halo.f_oral_pct,
                        "cl_hepatic_blood_l_hr": halo.cl_hepatic_blood_l_hr,
                        "cl_renal_plasma_l_hr": halo.cl_renal_plasma_l_hr,
                        "hepatic_extraction_eh": halo.hepatic_extraction_eh,
                        "vss_l_per_kg": halo.vss_l_per_kg,
                        "kp_uu_brain": halo.kp_uu_brain,
                        "cmax_total_plasma_nm": halo.cmax_total_plasma_nm,
                        "cmax_unbound_plasma_nm": halo.cmax_unbound_plasma_nm,
                        "cmax_unbound_brain_nm": halo.cmax_unbound_brain_nm,
                        "tmax_plasma_hr": halo.tmax_plasma_hr,
                        "auc_total_plasma_0_24_nm_hr": halo.auc_total_plasma_0_24_nm_hr,
                        "auc_unbound_plasma_nm_hr": halo.auc_unbound_plasma_nm_hr,
                        "auc_unbound_brain_nm_hr": halo.auc_unbound_brain_nm_hr,
                        "terminal_t_half_hr": halo.terminal_t_half_hr,
                        "peak_ro_brain_pct": halo.peak_ro_brain_pct
                    },
                    "lapatinib": {
                        "compound_id": lap.compound_id,
                        "fa_integrated": lap.fa_integrated,
                        "fg_gut_wall": lap.fg_gut_wall,
                        "fh_hepatic": lap.fh_hepatic,
                        "f_oral_pct": lap.f_oral_pct,
                        "cl_hepatic_blood_l_hr": lap.cl_hepatic_blood_l_hr,
                        "cl_renal_plasma_l_hr": lap.cl_renal_plasma_l_hr,
                        "hepatic_extraction_eh": lap.hepatic_extraction_eh,
                        "vss_l_per_kg": lap.vss_l_per_kg,
                        "kp_uu_brain": lap.kp_uu_brain,
                        "cmax_total_plasma_nm": lap.cmax_total_plasma_nm,
                        "cmax_unbound_plasma_nm": lap.cmax_unbound_plasma_nm,
                        "cmax_unbound_brain_nm": lap.cmax_unbound_brain_nm,
                        "tmax_plasma_hr": lap.tmax_plasma_hr,
                        "auc_total_plasma_0_24_nm_hr": lap.auc_total_plasma_0_24_nm_hr,
                        "auc_unbound_plasma_nm_hr": lap.auc_unbound_plasma_nm_hr,
                        "auc_unbound_brain_nm_hr": lap.auc_unbound_brain_nm_hr,
                        "terminal_t_half_hr": lap.terminal_t_half_hr,
                        "peak_ro_brain_pct": lap.peak_ro_brain_pct
                    },
                    "csv_artifact": "validation/pbpk_plasma_profile.csv",
                    "csv_bytes": csv_bytes
                },
                "t3_lomap_rbfe_closure_and_active_learning": {
                    "engine_calls": ["run_rbfe_network_campaign", "run_active_learning_campaign"],
                    "rbfe_resolved_target": rbfe.resolved_target,
                    "rbfe_n_nodes": rbfe.summary.n_nodes,
                    "rbfe_n_edges": rbfe.summary.n_edges,
                    "rbfe_n_cycles": rbfe.summary.n_cycles,
                    "rbfe_mean_lomap_score": rbfe.summary.mean_lomap_score,
                    "rbfe_mean_cycle_hysteresis_kcal": rbfe.summary.mean_cycle_hysteresis_kcal,
                    "rbfe_max_cycle_hysteresis_kcal": rbfe.summary.max_cycle_hysteresis_kcal,
                    "rbfe_post_mle_max_cycle_error_kcal": rbfe.summary.post_mle_max_cycle_error_kcal,
                    "rbfe_dg_rmse_kcal": rbfe.summary.dg_rmse_kcal,
                    "rbfe_pact_r2_pearson": rbfe.summary.pact_r2_pearson,
                    "rbfe_kendall_tau": rbfe.summary.kendall_tau,
                    "active_learning_pool_size": al.pool_size,
                    "active_learning_selected_batch_count": al.selected_batch_count,
                    "active_learning_batch_mean_predicted_pact": al.batch_mean_predicted_pact,
                    "active_learning_batch_mean_epistemic_std": al.batch_mean_epistemic_std,
                    "active_learning_batch_mean_pairwise_dissim": al.batch_mean_pairwise_dissim,
                    "active_learning_top5_candidates": top_candidates_json
                }
            }
        });

        let json_path = val_dir.join("sar_atlas_validation_summary.json");
        let pretty = serde_json::to_string_pretty(&summary).expect("serialize sar_atlas summary");
        fs::write(&json_path, format!("{pretty}\n")).expect("write sar_atlas_validation_summary.json");
        let json_bytes = fs::metadata(&json_path).expect("stat json").len();

        SarAtlasValidationData {
            mol_count,
            profile_count,
            mean_cns_mpo,
            mean_qed,
            ro5_zero_violation_count,
            veber_pass_count,
            mmp_count,
            max_mmp_abs_delta_pact,
            cliff_edge_count,
            max_sali,
            pareto_lead_count,
            halo,
            lap,
            rbfe_target: rbfe.resolved_target,
            rbfe_n_nodes: rbfe.summary.n_nodes,
            rbfe_n_edges: rbfe.summary.n_edges,
            rbfe_n_cycles: rbfe.summary.n_cycles,
            rbfe_mean_lomap: rbfe.summary.mean_lomap_score,
            rbfe_mean_hysteresis_kcal: rbfe.summary.mean_cycle_hysteresis_kcal,
            rbfe_max_hysteresis_kcal: rbfe.summary.max_cycle_hysteresis_kcal,
            rbfe_post_mle_max_cycle_error_kcal: rbfe.summary.post_mle_max_cycle_error_kcal,
            rbfe_dg_rmse_kcal: rbfe.summary.dg_rmse_kcal,
            rbfe_pact_r2: rbfe.summary.pact_r2_pearson,
            rbfe_kendall_tau: rbfe.summary.kendall_tau,
            al_pool_size: al.pool_size,
            al_selected_batch_count: al.selected_batch_count,
            al_batch_mean_pred_pact: al.batch_mean_predicted_pact,
            al_batch_mean_epistemic_std: al.batch_mean_epistemic_std,
            al_batch_mean_dissim: al.batch_mean_pairwise_dissim,
            al_top_name: al_top.candidate_name.clone(),
            al_top_ei: al_top.expected_improvement,
            al_top_selected: al_top.selected_in_batch,
            csv_path,
            csv_bytes,
            json_path,
            json_bytes,
        }
    })
}

#[test]
fn t1_cns_mpo_qed_lipinski_and_mmp_cliffs() {
    let d = ensure_sar_validation();
    assert_eq!(d.mol_count, 26);
    assert_eq!(d.profile_count, 26);
    assert!(d.mean_cns_mpo > 1.0 && d.mean_cns_mpo <= 6.0);
    assert!(d.mean_qed > 0.1 && d.mean_qed <= 1.0);
    assert!(d.ro5_zero_violation_count >= 10);
    assert!(d.veber_pass_count >= 10);
    assert!(d.mmp_count >= 1);
    assert!(d.max_mmp_abs_delta_pact.is_finite());
    assert!(d.cliff_edge_count >= 1 || d.max_sali >= 0.0);
    assert!(d.pareto_lead_count >= 1);
    assert!(d.json_path.exists() && d.json_bytes < 384 * 1024);
}

#[test]
fn t2_rk4_oral_pbpk_and_brain_kpuu() {
    let d = ensure_sar_validation();
    for snap in [&d.halo, &d.lap] {
        assert!(!snap.compound_name.is_empty());
        assert!(snap.fa_integrated > 0.0 && snap.fa_integrated <= 1.0);
        assert!(snap.fg_gut_wall > 0.0 && snap.fg_gut_wall <= 1.0);
        assert!(snap.fh_hepatic > 0.0 && snap.fh_hepatic <= 1.0);
        let expected_f = snap.fa_integrated * snap.fg_gut_wall * snap.fh_hepatic * 100.0;
        assert!(
            (snap.f_oral_pct - expected_f).abs() < 1e-2,
            "F_oral mass-balance identity must hold for {}: {} vs {}",
            snap.compound_name,
            snap.f_oral_pct,
            expected_f
        );
        assert!(snap.cl_hepatic_blood_l_hr >= 0.0 && snap.cl_renal_plasma_l_hr >= 0.0);
        assert!(snap.hepatic_extraction_eh >= 0.0 && snap.hepatic_extraction_eh <= 1.0);
        assert!(snap.vss_l_per_kg > 0.0 && snap.terminal_t_half_hr > 0.0);
        assert!(snap.cmax_total_plasma_nm > 0.0 && snap.cmax_unbound_plasma_nm > 0.0);
        assert!(snap.tmax_plasma_hr > 0.0 && snap.tmax_plasma_hr <= 24.0);
        assert!(snap.auc_total_plasma_0_24_nm_hr > 0.0 && snap.auc_unbound_plasma_nm_hr > 0.0);
    }
    assert!(
        d.halo.kp_uu_brain > d.lap.kp_uu_brain,
        "CNS-penetrant Haloperidol Kp,uu,brain ({}) must exceed peripherally restricted Lapatinib ({})",
        d.halo.kp_uu_brain,
        d.lap.kp_uu_brain
    );
    assert!(d.csv_path.exists() && d.csv_bytes < 384 * 1024);
}

#[test]
fn t3_lomap_rbfe_closure_and_active_learning() {
    let d = ensure_sar_validation();
    assert!(!d.rbfe_target.is_empty());
    assert!(d.rbfe_n_nodes >= 6 && d.rbfe_n_edges >= d.rbfe_n_nodes && d.rbfe_n_cycles >= 1);
    assert!(d.rbfe_mean_lomap > 0.3 && d.rbfe_mean_hysteresis_kcal.is_finite());
    assert!(d.rbfe_max_hysteresis_kcal.is_finite());
    assert!(
        d.rbfe_post_mle_max_cycle_error_kcal < 0.15,
        "MLE cycle closure residual must stay below 0.15 kcal/mol, got {}",
        d.rbfe_post_mle_max_cycle_error_kcal
    );
    assert!(d.rbfe_dg_rmse_kcal.is_finite() && d.rbfe_pact_r2.is_finite() && d.rbfe_kendall_tau.is_finite());
    assert!(d.al_pool_size >= 5 && d.al_selected_batch_count >= 1);
    assert!(d.al_batch_mean_pred_pact.is_finite() && d.al_batch_mean_epistemic_std >= 0.0);
    assert!(d.al_batch_mean_dissim.is_finite());
    assert!(!d.al_top_name.is_empty() && d.al_top_ei.is_finite() && d.al_top_selected);
    assert!(d.json_path.exists() && d.json_bytes < 384 * 1024);
}
