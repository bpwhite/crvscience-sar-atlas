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

//! Cross-Module Medicinal-Chemistry Lead Scorecard, Stage-Gate Candidate Target Profile (`TPP`)
//! Evaluator, and Standalone Zero-Dependency `.html` / `.svg` Dossier Exporter (`v2.1.0`).
//!
//! Synthesizes quantitative descriptors and mechanistic predictions across `SAR-Atlas` modules:
//! - Potency & Ligand Efficiency (`LE`, `LLE`, `LLE_7.4`, `SILE`, `Free-Wilson` QSAR, `SALI` cliffs)
//! - Physicochemical & `pH 7.4` Protomer/Tautomer Speciation (`QED`, `CNS-MPO`, `logD_7.4`, `3D NPR`)
//! - Quantitative `ADMET` & Safety (`ESOL`, `BOILED-Egg`, `Pfizer 3/75`, `GSK 4/400`, `hERG`, `PAINS`)
//! - `SMARTCyp` (`3A4`/`2D6`/`2C9`) Regioselective Clearance & Reactive Bioactivation Screening
//! - Mechanistic `RK4` Human Oral `PBPK` (`CAT`/`ACAT` `Fa * Fg * Fh`, `Rodgers-Rowland` `Vss`, `Kp,uu,brain`)
//! - Synthetic Tractability (`SAScore`, `BRICS`) & Tactical Backup/Rescue Paths (`Prodrug`, `Macrocycle`,
//!   `Covalent Warhead`, and `Meanwell Bioisostere` candidates)
//!
//! Primary References:
//! - Segall (2012), *Curr. Pharm. Des.* 18(9), 1292-1310 ("Multi-parameter optimization: identifying
//!   high quality compounds with a balance of properties": `https://pubmed.ncbi.nlm.nih.gov/22316157/`)
//!   & Segall (2014), *Expert Opin. Drug Discov.* (`https://pubmed.ncbi.nlm.nih.gov/24793080/`)
//! - Cook et al. (2014), *Nat. Rev. Drug Discov.* 13(6), 419-431 ("Lessons learned from the fate of
//!   AstraZeneca's drug pipeline: a five-dimensional framework": `https://pubmed.ncbi.nlm.nih.gov/24833294/`)
//!   & Morgan et al. (2018), *Nat. Rev. Drug Discov.* 17(3), 167-181 (`https://pubmed.ncbi.nlm.nih.gov/29348681/`)
//! - Derringer & Suich (1980) / Desirability Optimization Methodology (`https://pmc.ncbi.nlm.nih.gov/articles/PMC6217265/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC5650527/`)
//! - Bickerton et al. (2012) `QED` (`https://pubmed.ncbi.nlm.nih.gov/22270643/`), Wager et al. (2010)
//!   `CNS-MPO` (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3368654/`), Hopkins et al. (2004) `LE`
//!   (`https://pubmed.ncbi.nlm.nih.gov/15109945/`), and Leeson & Springthorpe (2007) `LLE`
//!   (`https://pubmed.ncbi.nlm.nih.gov/17971784/`)

use crate::admet::{evaluate_admet, BoiledEggRegion, HergRisk};
use crate::alerts::{scan_structural_alerts, AlertSeverity};
use crate::bioisostere::{generate_bioisosteres, BioisostereAnalog};
use crate::conformer3d::evaluate_3d_summary_profile;
use crate::covalent_warhead::{
    analyze_covalent_warheads_and_swaps, detect_covalent_warheads, DetectedWarhead,
    ReactivityWindow, WarheadSwapCandidate,
};
use crate::cyp_metabolism::{analyze_cyp_metabolism, CypMetabolismReport, MetabolicStabilityClass};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::layout2d::{compute_2d_coords, render_molecule_svg, render_molecule_svg_with_smarts};
use crate::macrocycle::{design_macrocycles, MacrocycleCandidate};
use crate::mol::Molecule;
use crate::pbpk_oral::{simulate_oral_pbpk, OralPbpkReport, PbpkDosingConfig, PBPK_TIME_POINTS};
use crate::physchem::evaluate_physchem;
use crate::pka_protomer::{evaluate_physiological_pka_74, PhysiologicalChargeSummary};
use crate::prodrug::{
    design_prodrugs, diagnose_parent_bcs_liability, BcsClass, ParentLiabilityDiagnosis,
    ProdrugCandidate,
};
use crate::rgroup_table::predict_best_free_wilson;
use crate::scaffold::murcko_signatures;
use crate::synth_access::{
    dissect_brics, evaluate_synthetic_accessibility, BricsRetrosynthesis, SynthAccessProfile,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::f64::consts::TAU;

/// Labels for the 6 normalized `[0..100]` multi-parameter desirability radar axes.
pub const DOSSIER_PILLAR_LABELS: [&str; 6] = [
    "Potency & LE/LLE",
    "Physchem & pKa",
    "ADMET & Safety",
    "CYP Stability",
    "Oral PBPK & RO",
    "Synthesis & BRICS",
];

/// Therapeutic Target Product Profile (`TPP`) preset tuning Stage-Gate thresholds and Derringer-Suich weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CampaignTargetProfile {
    /// Standard oral small molecule targeting systemic/peripheral tissue (`F_oral >= 30%`, `Cu,p >= IC50`).
    OralPeripheral,
    /// Oral CNS-penetrant small molecule (`CNS-MPO >= 4.0`, `Kp,uu,brain >= 0.30`, `Cu,brain >= IC50`).
    OralCnsPenetrant,
    /// Targeted Covalent Inhibitor (`TCI`) balancing selective `GSH t_1/2` (`4-80 h` or rev-cov) & `k_inact/K_I`.
    TargetedCovalent,
    /// Beyond-Rule-of-5 (`macrocycle`/`PROTAC`) or `BCS II-IV` Prodrug-Rescued oral candidate.
    BeyondRo5OrProdrug,
}

impl CampaignTargetProfile {
    pub const ALL: [Self; 4] = [
        Self::OralPeripheral,
        Self::OralCnsPenetrant,
        Self::TargetedCovalent,
        Self::BeyondRo5OrProdrug,
    ];

    pub fn from_code(raw: &str) -> Self {
        let s = raw.trim().to_ascii_lowercase();
        if s.contains("cns") || s.contains("brain") || s.contains("bbb") {
            Self::OralCnsPenetrant
        } else if s.contains("cov") || s.contains("tci") || s.contains("warhead") {
            Self::TargetedCovalent
        } else if s.contains("bro5")
            || s.contains("prodrug")
            || s.contains("macro")
            || s.contains("protac")
        {
            Self::BeyondRo5OrProdrug
        } else {
            Self::OralPeripheral
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::OralPeripheral => "oral-periph",
            Self::OralCnsPenetrant => "oral-cns",
            Self::TargetedCovalent => "covalent-tci",
            Self::BeyondRo5OrProdrug => "bro5-prodrug",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OralPeripheral => "Oral Peripheral Small Molecule",
            Self::OralCnsPenetrant => "Oral CNS-Penetrant Lead (BBB Kp,uu)",
            Self::TargetedCovalent => "Targeted Covalent Inhibitor (TCI)",
            Self::BeyondRo5OrProdrug => "bRo5 / Prodrug-Rescued Lead",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::OralPeripheral => "Oral Peripheral",
            Self::OralCnsPenetrant => "Oral CNS (BBB)",
            Self::TargetedCovalent => "Covalent TCI",
            Self::BeyondRo5OrProdrug => "bRo5 / Prodrug",
        }
    }

    /// Profile-specific weights across the 6 desirability pillars (summing to `1.0`).
    pub fn pillar_weights(self) -> [f64; 6] {
        match self {
            Self::OralPeripheral => [0.22, 0.16, 0.18, 0.14, 0.18, 0.12],
            Self::OralCnsPenetrant => [0.20, 0.20, 0.16, 0.14, 0.20, 0.10],
            Self::TargetedCovalent => [0.22, 0.14, 0.20, 0.14, 0.18, 0.12],
            Self::BeyondRo5OrProdrug => [0.24, 0.12, 0.16, 0.14, 0.20, 0.14],
        }
    }
}

/// Stage-Gate campaign nomination tier derived from `Derringer-Suich` desirability and gate failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressionTier {
    CandidateReady,
    LeadOptimization,
    ProdrugOrFormulationRescue,
    EarlyHitOrBackUp,
}

impl ProgressionTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::CandidateReady => "Stage-Gate: Candidate Nomination Ready",
            Self::LeadOptimization => "Lead Optimization (Targeted Fix)",
            Self::ProdrugOrFormulationRescue => "Prodrug / Formulation Rescue Candidate",
            Self::EarlyHitOrBackUp => "Early Hit / Back-Up Series Triage",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::CandidateReady => "Candidate Ready",
            Self::LeadOptimization => "Lead Opt",
            Self::ProdrugOrFormulationRescue => "Prodrug Rescue",
            Self::EarlyHitOrBackUp => "Hit / Triage",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::CandidateReady => [52, 211, 153],             // Emerald
            Self::LeadOptimization => [56, 189, 248],           // Sky Blue
            Self::ProdrugOrFormulationRescue => [251, 191, 36], // Amber
            Self::EarlyHitOrBackUp => [248, 113, 113],          // Coral
        }
    }

    pub fn badge_hex(self) -> &'static str {
        match self {
            Self::CandidateReady => "#059669",
            Self::LeadOptimization => "#0284c7",
            Self::ProdrugOrFormulationRescue => "#d97706",
            Self::EarlyHitOrBackUp => "#dc2626",
        }
    }
}

/// Traffic-light outcome for an individual Stage-Gate Target Product Profile criterion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriterionVerdict {
    Pass,
    Conditional,
    Fail,
}

impl CriterionVerdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Conditional => "WATCH",
            Self::Fail => "GAP",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Pass => [52, 211, 153],
            Self::Conditional => [251, 191, 36],
            Self::Fail => [248, 113, 113],
        }
    }

    pub fn hex(self) -> &'static str {
        match self {
            Self::Pass => "#059669",
            Self::Conditional => "#d97706",
            Self::Fail => "#dc2626",
        }
    }
}

/// Explicit Stage-Gate evaluation row comparing observed/predicted property against `TPP` threshold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageGateCriterion {
    #[serde(skip_deserializing)]
    pub code: &'static str,
    #[serde(skip_deserializing)]
    pub category: &'static str,
    #[serde(skip_deserializing)]
    pub metric_name: &'static str,
    #[serde(skip_deserializing)]
    pub target_window: &'static str,
    pub observed_value: String,
    pub desirability_0_1: f64,
    pub verdict: CriterionVerdict,
    #[serde(skip_deserializing)]
    pub source_module: &'static str,
}

/// Concrete next-DMTA-cycle structural design recommendation linked to a detected liability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MedChemOptimizationAction {
    pub priority: u8,
    #[serde(skip_deserializing)]
    pub domain: &'static str,
    pub issue_detected: String,
    pub structural_proposal: String,
    pub quantitative_impact: String,
    pub candidate_smiles: Option<String>,
}

/// Campaign context relationship (`SALI` activity cliff or `MMP` R-group transform) for a lead.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DossierNeighborNote {
    #[serde(skip_deserializing)]
    pub relation_kind: &'static str,
    pub compound_id: String,
    pub compound_name: String,
    pub summary: String,
}

/// Cross-module quantitative lead scorecard row for leaderboard ranking and CSV export.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeadScorecardRow {
    pub rank: usize,
    pub mol_idx: usize,
    pub id: String,
    pub name: String,
    pub target: String,
    pub smiles: String,
    pub scaffold_label: String,
    pub scaffold_key: String,
    pub pact: f64,
    pub ic50_nm: f64,
    pub le: f64,
    pub lle: f64,
    pub lle_74: f64,
    pub sile: f64,
    pub mw: f64,
    pub clogp: f64,
    pub logd_74: f64,
    pub tpsa: f64,
    pub fsp3: f64,
    pub qed: f64,
    pub cns_mpo: f64,
    pub esol_ug_ml: f64,
    pub bcs_class: BcsClass,
    pub boiled_egg: BoiledEggRegion,
    pub herg_risk: HergRisk,
    pub pfizer_3_75_pass: bool,
    pub pains_count: usize,
    pub cl_int_ml_min_kg: f64,
    pub microsomal_t_half_min: f64,
    pub reactive_metabolite_risk: bool,
    pub f_oral_pct: f64,
    pub kp_uu_brain: f64,
    pub target_compartment_t_above_ic50_hr: f64,
    pub peak_target_ro_pct: f64,
    pub sa_score: f64,
    pub brics_steps: usize,
    pub best_prodrug_f_oral_pct: f64,
    pub prodrug_fold_gain: f64,
    pub has_covalent_warhead: bool,
    pub gsh_half_life_h: Option<f64>,
    pub pareto_optimal: bool,
    pub pass_gates: usize,
    pub conditional_gates: usize,
    pub fail_gates: usize,
    pub pillar_scores: [f64; 6],
    pub weighted_score_100: f64,
    pub geometric_desirability: f64,
    pub tier: ProgressionTier,
}

/// Dataset-wide ranked campaign scorecard with Stage-Gate counts and scaffold-diversified backup selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignScorecardSummary {
    pub profile: CampaignTargetProfile,
    pub dose_mg: f64,
    pub total_compounds: usize,
    pub candidate_ready_count: usize,
    pub lead_opt_count: usize,
    pub prodrug_rescue_count: usize,
    pub early_hit_count: usize,
    pub pareto_count: usize,
    pub mean_geometric_desirability: f64,
    pub mean_f_oral_pct: f64,
    pub front_runner_idx: Option<usize>,
    pub backup_scaffold_idx: Option<usize>,
    pub rows: Vec<LeadScorecardRow>,
}

/// Complete cross-module medicinal-chemistry candidate dossier ready for GUI inspection or `.html`/`.svg` export.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeadCampaignDossier {
    pub profile: CampaignTargetProfile,
    pub pbpk_config: PbpkDosingConfig,
    pub scorecard: LeadScorecardRow,
    pub campaign_percentile: f64,
    pub gates: Vec<StageGateCriterion>,
    pub pbpk: OralPbpkReport,
    pub cyp: CypMetabolismReport,
    pub synth: SynthAccessProfile,
    pub brics: BricsRetrosynthesis,
    pub pka_74: PhysiologicalChargeSummary,
    #[serde(skip_deserializing)]
    pub shape3d_class: &'static str,
    pub npr1: f32,
    pub npr2: f32,
    pub dipole_debye: f32,
    pub bcs_diagnosis: ParentLiabilityDiagnosis,
    pub top_prodrug: Option<ProdrugCandidate>,
    pub top_macrocycle: Option<MacrocycleCandidate>,
    pub detected_warheads: Vec<DetectedWarhead>,
    pub top_warhead_swap: Option<WarheadSwapCandidate>,
    pub top_bioisosteres: Vec<BioisostereAnalog>,
    pub free_wilson_pred_pact: Option<f64>,
    pub campaign_neighbors: Vec<DossierNeighborNote>,
    pub optimization_actions: Vec<MedChemOptimizationAction>,
    pub structure_2d_svg_light: String,
    pub structure_2d_svg_dark: String,
    pub radar_svg_light: String,
    pub pk_curve_svg_light: String,
    pub cascade_svg_light: String,
}

#[inline]
fn ramp_maximize(val: f64, low: f64, high: f64) -> f64 {
    if val <= low {
        0.05
    } else if val >= high {
        1.0
    } else {
        0.05 + 0.95 * ((val - low) / (high - low).max(1e-9))
    }
}

#[inline]
fn ramp_minimize(val: f64, ideal_max: f64, unacceptable: f64) -> f64 {
    if val <= ideal_max {
        1.0
    } else if val >= unacceptable {
        0.05
    } else {
        1.0 - 0.95 * ((val - ideal_max) / (unacceptable - ideal_max).max(1e-9))
    }
}

#[inline]
fn verdict_from_d(d: f64) -> CriterionVerdict {
    if d >= 0.70 {
        CriterionVerdict::Pass
    } else if d >= 0.38 {
        CriterionVerdict::Conditional
    } else {
        CriterionVerdict::Fail
    }
}

fn escape_xml(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Evaluates the 6-pillar multi-parameter scorecard, 12 Stage-Gate `TPP` criteria, and `50 mg` (or custom)
/// oral `PBPK` profile for a single `Molecule`.
pub fn evaluate_molecule_scorecard(
    mol: &Molecule,
    mol_idx: usize,
    profile: CampaignTargetProfile,
    dose_mg: f64,
) -> (LeadScorecardRow, Vec<StageGateCriterion>, OralPbpkReport) {
    let alerts = scan_structural_alerts(mol);
    let pains_count = alerts
        .iter()
        .filter(|a| a.severity == AlertSeverity::PainsAssayInterference)
        .count();
    let prof = evaluate_physchem(mol, alerts.len());
    let admet = evaluate_admet(mol, &prof);
    let pka = evaluate_physiological_pka_74(mol);
    let synth = evaluate_synthetic_accessibility(mol);
    let cyp = analyze_cyp_metabolism(mol, false);
    let warheads = detect_covalent_warheads(mol);
    let prodrug_rep = design_prodrugs(mol, 0);

    let pbpk_cfg = PbpkDosingConfig {
        dose_mg: dose_mg.max(1.0),
        ..PbpkDosingConfig::default()
    };
    let pbpk = simulate_oral_pbpk(mol, pbpk_cfg);

    let hac = prof.heavy_atoms.max(1) as f64;
    let dg_bind = RT_LN10_KCAL * mol.pact;
    let le = dg_bind / hac;
    let sile = dg_bind / hac.powf(0.3);
    let lle_74 = mol.pact - pka.microstate_logd_74;

    let (scaf_key, _, scaf_lbl, _) = murcko_signatures(mol);
    let ic50_nm = if mol.raw_nm > 0.0 && mol.raw_nm.is_finite() {
        mol.raw_nm
    } else {
        10.0_f64.powf(9.0 - mol.pact)
    };

    let best_prodrug_f_oral = prodrug_rep
        .candidates
        .first()
        .map(|c| c.activation.predicted_active_f_oral_pct)
        .unwrap_or(pbpk.f_oral_pct);
    let prodrug_fold_gain = prodrug_rep
        .candidates
        .first()
        .map(|c| c.activation.f_oral_fold_gain)
        .unwrap_or(1.0);

    let reactive_met_risk = cyp.metabolites.iter().any(|m| m.is_reactive_bioactivation);
    let has_warhead = !warheads.is_empty();
    let gsh_t_half = warheads.first().map(|w| w.gsh_half_life_h);

    let mut gates = Vec::with_capacity(12);

    // Gate 1: Target Potency (pAct / IC50)
    let d_potency = ramp_maximize(mol.pact, 5.5, 7.8);
    gates.push(StageGateCriterion {
        code: "G01",
        category: "Potency & LE",
        metric_name: "Primary Target Potency (pAct / IC50)",
        target_window: "pAct >= 7.0 (IC50 <= 100 nM)",
        observed_value: format!("pAct {:.2} ({:.1} nM)", mol.pact, ic50_nm),
        desirability_0_1: d_potency,
        verdict: verdict_from_d(d_potency),
        source_module: "physchem.rs",
    });

    // Gate 2: Ligand & Lipophilic Efficiency (LE / LLE)
    let d_eff = match profile {
        CampaignTargetProfile::BeyondRo5OrProdrug => {
            0.45 * ramp_maximize(le, 0.16, 0.28) + 0.55 * ramp_maximize(lle_74, 2.0, 4.8)
        }
        _ => 0.50 * ramp_maximize(le, 0.22, 0.34) + 0.50 * ramp_maximize(prof.lle, 3.0, 5.5),
    };
    gates.push(StageGateCriterion {
        code: "G02",
        category: "Potency & LE",
        metric_name: "Ligand & Lipophilic Efficiency (LE / LLE)",
        target_window: if profile == CampaignTargetProfile::BeyondRo5OrProdrug {
            "LE >= 0.22, LLE_7.4 >= 3.5"
        } else {
            "LE >= 0.30 kcal/mol/HA, LLE >= 5.0"
        },
        observed_value: format!(
            "LE {:.2}, LLE {:.2}, LLE7.4 {:.2}",
            le, prof.lle, lle_74
        ),
        desirability_0_1: d_eff,
        verdict: verdict_from_d(d_eff),
        source_module: "fragment_grower.rs",
    });

    // Gate 3: Multiparameter Druglikeness (QED / CNS-MPO)
    let (d_mpo, mpo_window, mpo_obs) = match profile {
        CampaignTargetProfile::OralCnsPenetrant => (
            ramp_maximize(prof.cns_mpo, 3.0, 4.6),
            "Wager CNS-MPO >= 4.2 / 6.0",
            format!("CNS-MPO {:.2}, QED {:.2}", prof.cns_mpo, prof.qed_mean),
        ),
        CampaignTargetProfile::BeyondRo5OrProdrug => {
            let tpsa_d = if prof.tpsa <= 140.0 {
                1.0
            } else {
                ramp_minimize(prof.tpsa, 140.0, 220.0)
            };
            (
                0.60 * tpsa_d + 0.40 * ramp_maximize(prof.fsp3, 0.15, 0.35),
                "TPSA <= 140 A^2, Fsp3 >= 0.25",
                format!(
                    "MW {:.0} Da, TPSA {:.1} A^2, Fsp3 {:.2}",
                    prof.mw, prof.tpsa, prof.fsp3
                ),
            )
        }
        _ => (
            ramp_maximize(prof.qed_mean, 0.35, 0.68),
            "Bickerton QED >= 0.55, Ro5 <= 1",
            format!(
                "QED {:.2}, Ro5 viol {}, MW {:.0} Da",
                prof.qed_mean, prof.ro5_violations, prof.mw
            ),
        ),
    };
    gates.push(StageGateCriterion {
        code: "G03",
        category: "Physchem & pKa",
        metric_name: "Multiparameter Property Space",
        target_window: mpo_window,
        observed_value: mpo_obs,
        desirability_0_1: d_mpo,
        verdict: verdict_from_d(d_mpo),
        source_module: "physchem.rs",
    });

    // Gate 4: Physiological pH 7.4 logD & Ionization Speciation
    let logd = pka.microstate_logd_74;
    let d_logd = if (-0.5..=3.5).contains(&logd) {
        1.0
    } else if logd < -0.5 {
        ramp_maximize(logd, -2.5, -0.5)
    } else {
        ramp_minimize(logd, 3.5, 6.0)
    };
    gates.push(StageGateCriterion {
        code: "G04",
        category: "Physchem & pKa",
        metric_name: "pH 7.4 Microstate logD & Charge",
        target_window: "logD_7.4 in [-0.5, 3.5] (Golden Triangle)",
        observed_value: format!(
            "logD_7.4 {:.2} ({}, f_neut {:.0}%)",
            logd,
            pka.charge_class.label(),
            pka.neutral_fraction_74 * 100.0
        ),
        desirability_0_1: d_logd,
        verdict: verdict_from_d(d_logd),
        source_module: "pka_protomer.rs",
    });

    // Gate 5: Aqueous Solubility & BCS Dissolution
    let sol_ug = prodrug_rep.diagnosis.sol_ph68_ug_ml.max(admet.esol_ug_ml);
    let d_sol = if profile == CampaignTargetProfile::BeyondRo5OrProdrug && best_prodrug_f_oral >= 45.0
    {
        ramp_maximize(sol_ug, 2.0, 40.0).max(0.78)
    } else {
        ramp_maximize(sol_ug.log10(), 0.5, 2.0) // 3.16 ug/mL -> 100 ug/mL
    };
    gates.push(StageGateCriterion {
        code: "G05",
        category: "ADMET & Safety",
        metric_name: "Aqueous Solubility & BCS Class",
        target_window: "Sol_pH6.8 >= 50 ug/mL (BCS Class I/III)",
        observed_value: format!(
            "{:.1} ug/mL ({})",
            sol_ug,
            prodrug_rep.diagnosis.bcs_class.short_label()
        ),
        desirability_0_1: d_sol,
        verdict: verdict_from_d(d_sol),
        source_module: "admet.rs / prodrug.rs",
    });

    // Gate 6: Preclinical Toxicology (Pfizer 3/75, hERG & PAINS)
    let mut d_tox: f64 = 1.0;
    if pains_count > 0 {
        d_tox -= 0.55;
    }
    match admet.herg_risk {
        HergRisk::High => d_tox -= 0.35,
        HergRisk::Moderate => d_tox -= 0.14,
        HergRisk::Low => {}
    }
    if !admet.pfizer_3_75_pass {
        d_tox -= 0.16;
    }
    d_tox = d_tox.clamp(0.05, 1.0);
    gates.push(StageGateCriterion {
        code: "G06",
        category: "ADMET & Safety",
        metric_name: "PAINS, hERG & Pfizer 3/75 Safety",
        target_window: "0 PAINS, hERG Low/Mod, Pfizer 3/75 Pass",
        observed_value: format!(
            "PAINS {}, hERG {}, 3/75 {}",
            pains_count,
            admet.herg_risk.label(),
            if admet.pfizer_3_75_pass { "Pass" } else { "Risk" }
        ),
        desirability_0_1: d_tox,
        verdict: verdict_from_d(d_tox),
        source_module: "alerts.rs / admet.rs",
    });

    // Gate 7: SMARTCyp Hepatic Metabolic Stability (CL_int & t_1/2)
    let d_cl = ramp_minimize(cyp.cl_int_ml_min_kg, 18.0, 68.0);
    gates.push(StageGateCriterion {
        code: "G07",
        category: "CYP Stability",
        metric_name: "SMARTCyp Microsomal Clearance (CL_int)",
        target_window: "CL_int <= 20 mL/min/kg (t_1/2 >= 45 min)",
        observed_value: format!(
            "{:.1} mL/min/kg (t_1/2 {:.0} min, {})",
            cyp.cl_int_ml_min_kg,
            cyp.predicted_t_half_min,
            cyp.stability_class.short_label()
        ),
        desirability_0_1: d_cl,
        verdict: verdict_from_d(d_cl),
        source_module: "cyp_metabolism.rs",
    });

    // Gate 8: Reactive Metabolite Bioactivation Risk
    let d_bioact = if reactive_met_risk { 0.32 } else { 0.96 };
    gates.push(StageGateCriterion {
        code: "G08",
        category: "CYP Stability",
        metric_name: "Reactive Phase I/II Bioactivation",
        target_window: "Zero epoxide / quinone-imine / acyl-UGT alerts",
        observed_value: if reactive_met_risk {
            "Bioactivation Alert Present".to_string()
        } else {
            "Clean (Benign Phase I/II)".to_string()
        },
        desirability_0_1: d_bioact,
        verdict: verdict_from_d(d_bioact),
        source_module: "cyp_metabolism.rs",
    });

    // Gate 9: Human Oral Bioavailability Cascade (Fa * Fg * Fh)
    let effective_f_oral = if profile == CampaignTargetProfile::BeyondRo5OrProdrug {
        pbpk.f_oral_pct.max(best_prodrug_f_oral)
    } else {
        pbpk.f_oral_pct
    };
    let d_foral = ramp_maximize(effective_f_oral, 15.0, 55.0);
    gates.push(StageGateCriterion {
        code: "G09",
        category: "Oral PBPK & RO",
        metric_name: "Human Oral Bioavailability (F_oral = Fa*Fg*Fh)",
        target_window: "F_oral >= 30% at 50 mg fasted dose",
        observed_value: if profile == CampaignTargetProfile::BeyondRo5OrProdrug
            && best_prodrug_f_oral > pbpk.f_oral_pct + 5.0
        {
            format!(
                "Parent {:.1}% -> Prodrug {:.1}%",
                pbpk.f_oral_pct, best_prodrug_f_oral
            )
        } else {
            format!(
                "F_oral {:.1}% (Fa {:.0}%, Fg {:.0}%, Fh {:.0}%)",
                pbpk.f_oral_pct,
                pbpk.fa_integrated * 100.0,
                pbpk.fg_gut_wall * 100.0,
                pbpk.fh_hepatic * 100.0
            )
        },
        desirability_0_1: d_foral,
        verdict: verdict_from_d(d_foral),
        source_module: "pbpk_oral.rs",
    });

    // Gate 10: Compartment Selectivity / BBB Kp,uu,brain or Covalent Warhead Window
    let (d_comp, comp_metric, comp_win, comp_obs, comp_mod) = match profile {
        CampaignTargetProfile::OralCnsPenetrant => (
            ramp_maximize(pbpk.kp_uu_brain, 0.08, 0.35),
            "Unbound Brain Partitioning (Kp,uu,brain)",
            "Kp,uu,brain >= 0.30 (Low P-gp/BCRP efflux)",
            format!(
                "Kp,uu {:.2} (ER {:.1}x, {})",
                pbpk.kp_uu_brain,
                pbpk.bbb_efflux_ratio,
                pbpk.cns_class.short_label()
            ),
            "pbpk_oral.rs",
        ),
        CampaignTargetProfile::TargetedCovalent => {
            if let Some(w) = warheads.first() {
                let d_w = match w.window {
                    ReactivityWindow::TunedClinicalSweetSpot
                    | ReactivityWindow::ReversibleCovalent => 0.96,
                    ReactivityWindow::MildSelectiveIrreversible => 0.82,
                    ReactivityWindow::FastIrreversible => 0.52,
                    ReactivityWindow::HyperReactivePromiscuous => 0.15,
                    ReactivityWindow::NonCovalent => 0.25,
                };
                (
                    d_w,
                    "Electrophile GSH Half-Life & k_inact/K_I",
                    "GSH t_1/2 in 4-80 h or Reversible Covalent",
                    format!(
                        "{} (GSH t_1/2 {:.1} h, k_inact/K_I {:.0})",
                        w.warhead_name, w.gsh_half_life_h, w.kinact_over_ki_m_inv_s_inv
                    ),
                    "covalent_warhead.rs",
                )
            } else {
                (
                    0.25,
                    "Electrophile GSH Half-Life & k_inact/K_I",
                    "Tuned covalent warhead installed",
                    "Non-covalent scaffold (use Warhead Swapper)".to_string(),
                    "covalent_warhead.rs",
                )
            }
        }
        _ => {
            // Peripheral restriction or balanced Vss
            let vss_ok = if (0.3..=8.0).contains(&pbpk.vss_l_per_kg) {
                0.92
            } else {
                0.62
            };
            (
                vss_ok,
                "Rodgers-Rowland Volume & BBB Selectivity",
                "Vss 0.3-8.0 L/kg, controlled CNS exposure",
                format!(
                    "Vss {:.2} L/kg, Kp,uu,brain {:.2} ({})",
                    pbpk.vss_l_per_kg,
                    pbpk.kp_uu_brain,
                    admet.boiled_egg.label()
                ),
                "pbpk_oral.rs",
            )
        }
    };
    gates.push(StageGateCriterion {
        code: "G10",
        category: "Oral PBPK & RO",
        metric_name: comp_metric,
        target_window: comp_win,
        observed_value: comp_obs,
        desirability_0_1: d_comp,
        verdict: verdict_from_d(d_comp),
        source_module: comp_mod,
    });

    // Gate 11: Unbound Target Engagement Duration (Time Cu >= IC50 & Peak RO%)
    let (target_t_above, peak_ro, ro_label) = if profile == CampaignTargetProfile::OralCnsPenetrant {
        (
            pbpk.time_above_ic50_brain_hr,
            pbpk.peak_ro_brain_pct,
            "Unbound Brain Target Coverage (Cu,brain >= IC50)",
        )
    } else {
        (
            pbpk.time_above_ic50_plasma_hr,
            pbpk.peak_ro_plasma_pct,
            "Unbound Plasma Target Coverage (Cu,p >= IC50)",
        )
    };
    let d_ro = 0.55 * ramp_maximize(target_t_above, 2.0, 12.0)
        + 0.45 * ramp_maximize(peak_ro, 40.0, 85.0);
    gates.push(StageGateCriterion {
        code: "G11",
        category: "Oral PBPK & RO",
        metric_name: ro_label,
        target_window: "Time Cu >= IC50 >= 8 h, Peak RO >= 75%",
        observed_value: format!(
            "{:.1} h above IC50, Peak RO {:.1}%, t_1/2 {:.1} h",
            target_t_above, peak_ro, pbpk.terminal_t_half_hr
        ),
        desirability_0_1: d_ro,
        verdict: verdict_from_d(d_ro),
        source_module: "pbpk_oral.rs",
    });

    // Gate 12: Synthetic Accessibility (SAScore) & Convergent BRICS Steps
    let sa_max = if profile == CampaignTargetProfile::BeyondRo5OrProdrug {
        5.8
    } else {
        4.2
    };
    let d_synth = 0.70 * ramp_minimize(synth.sa_score, sa_max, sa_max + 2.8)
        + 0.30 * ramp_maximize(synth.brics_synthon_count as f64, 1.0, 3.0);
    gates.push(StageGateCriterion {
        code: "G12",
        category: "Synthesis & BRICS",
        metric_name: "Synthetic Tractability & BRICS Route",
        target_window: if profile == CampaignTargetProfile::BeyondRo5OrProdrug {
            "Ertl SAScore <= 5.8, convergent BRICS route"
        } else {
            "Ertl SAScore <= 4.2, <= 4 convergent steps"
        },
        observed_value: format!(
            "SAScore {:.2} ({}), {} BRICS cuts",
            synth.sa_score,
            synth.tractability.short_label(),
            synth.brics_cleavable_bonds
        ),
        desirability_0_1: d_synth,
        verdict: verdict_from_d(d_synth),
        source_module: "synth_access.rs",
    });

    // Aggregate 6 normalized [0..100] pillar scores
    let pillar_d = [
        (0.55 * d_potency + 0.45 * d_eff).clamp(0.05, 1.0),
        (0.55 * d_mpo + 0.45 * d_logd).clamp(0.05, 1.0),
        (0.42 * d_sol + 0.58 * d_tox).clamp(0.05, 1.0),
        (0.65 * d_cl + 0.35 * d_bioact).clamp(0.05, 1.0),
        (0.38 * d_foral + 0.27 * d_comp + 0.35 * d_ro).clamp(0.05, 1.0),
        d_synth.clamp(0.05, 1.0),
    ];
    let pillar_scores = [
        pillar_d[0] * 100.0,
        pillar_d[1] * 100.0,
        pillar_d[2] * 100.0,
        pillar_d[3] * 100.0,
        pillar_d[4] * 100.0,
        pillar_d[5] * 100.0,
    ];

    let weights = profile.pillar_weights();
    let mut weighted_score_100 = 0.0;
    let mut log_geom = 0.0;
    for i in 0..6 {
        weighted_score_100 += weights[i] * pillar_scores[i];
        log_geom += weights[i] * pillar_d[i].max(0.05).ln();
    }
    let geometric_desirability = (log_geom.exp() * 100.0).clamp(0.0, 100.0);

    let pass_gates = gates
        .iter()
        .filter(|g| g.verdict == CriterionVerdict::Pass)
        .count();
    let conditional_gates = gates
        .iter()
        .filter(|g| g.verdict == CriterionVerdict::Conditional)
        .count();
    let fail_gates = gates
        .iter()
        .filter(|g| g.verdict == CriterionVerdict::Fail)
        .count();

    let tier = if geometric_desirability >= 76.0 && fail_gates == 0 && pains_count == 0 {
        ProgressionTier::CandidateReady
    } else if pbpk.f_oral_pct < 32.0
        && prodrug_fold_gain >= 1.75
        && best_prodrug_f_oral >= 42.0
        && mol.pact >= 6.5
        && pains_count == 0
    {
        ProgressionTier::ProdrugOrFormulationRescue
    } else if geometric_desirability >= 58.0 && fail_gates <= 2 {
        ProgressionTier::LeadOptimization
    } else {
        ProgressionTier::EarlyHitOrBackUp
    };

    let row = LeadScorecardRow {
        rank: 1,
        mol_idx,
        id: if mol.id.is_empty() {
            format!("MOL-{}", mol_idx + 1)
        } else {
            mol.id.clone()
        },
        name: if mol.name.is_empty() {
            "Unnamed Compound".to_string()
        } else {
            mol.name.clone()
        },
        target: mol.target.clone(),
        smiles: mol.smiles.clone(),
        scaffold_label: scaf_lbl,
        scaffold_key: scaf_key,
        pact: mol.pact,
        ic50_nm,
        le,
        lle: prof.lle,
        lle_74,
        sile,
        mw: prof.mw,
        clogp: prof.clogp,
        logd_74: pka.microstate_logd_74,
        tpsa: prof.tpsa,
        fsp3: prof.fsp3,
        qed: prof.qed_mean,
        cns_mpo: prof.cns_mpo,
        esol_ug_ml: sol_ug,
        bcs_class: prodrug_rep.diagnosis.bcs_class,
        boiled_egg: admet.boiled_egg,
        herg_risk: admet.herg_risk,
        pfizer_3_75_pass: admet.pfizer_3_75_pass,
        pains_count,
        cl_int_ml_min_kg: cyp.cl_int_ml_min_kg,
        microsomal_t_half_min: cyp.predicted_t_half_min,
        reactive_metabolite_risk: reactive_met_risk,
        f_oral_pct: pbpk.f_oral_pct,
        kp_uu_brain: pbpk.kp_uu_brain,
        target_compartment_t_above_ic50_hr: target_t_above,
        peak_target_ro_pct: peak_ro,
        sa_score: synth.sa_score,
        brics_steps: synth.convergent_steps_est,
        best_prodrug_f_oral_pct: best_prodrug_f_oral,
        prodrug_fold_gain,
        has_covalent_warhead: has_warhead,
        gsh_half_life_h: gsh_t_half,
        pareto_optimal: prof.pareto_optimal,
        pass_gates,
        conditional_gates,
        fail_gates,
        pillar_scores,
        weighted_score_100,
        geometric_desirability,
        tier,
    };

    (row, gates, pbpk)
}

/// Evaluates and ranks an entire `DatasetAnalysis` campaign in parallel using `Rayon`.
pub fn evaluate_campaign_scorecard(
    ds: &DatasetAnalysis,
    profile: CampaignTargetProfile,
    dose_mg: f64,
) -> CampaignScorecardSummary {
    let mut rows: Vec<LeadScorecardRow> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(idx, mol)| {
            let (mut r, _, _) = evaluate_molecule_scorecard(mol, idx, profile, dose_mg);
            if let Some(p) = ds.profiles.get(idx) {
                r.pareto_optimal = p.pareto_optimal;
            }
            r
        })
        .collect();

    rows.sort_by(|a, b| {
        b.geometric_desirability
            .partial_cmp(&a.geometric_desirability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.weighted_score_100
                    .partial_cmp(&a.weighted_score_100)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    for (rk, row) in rows.iter_mut().enumerate() {
        row.rank = rk + 1;
    }

    let total_compounds = rows.len();
    let candidate_ready_count = rows
        .iter()
        .filter(|r| r.tier == ProgressionTier::CandidateReady)
        .count();
    let lead_opt_count = rows
        .iter()
        .filter(|r| r.tier == ProgressionTier::LeadOptimization)
        .count();
    let prodrug_rescue_count = rows
        .iter()
        .filter(|r| r.tier == ProgressionTier::ProdrugOrFormulationRescue)
        .count();
    let early_hit_count = rows
        .iter()
        .filter(|r| r.tier == ProgressionTier::EarlyHitOrBackUp)
        .count();
    let pareto_count = rows.iter().filter(|r| r.pareto_optimal).count();

    let mean_geometric_desirability = if total_compounds > 0 {
        rows.iter().map(|r| r.geometric_desirability).sum::<f64>() / (total_compounds as f64)
    } else {
        0.0
    };
    let mean_f_oral_pct = if total_compounds > 0 {
        rows.iter().map(|r| r.f_oral_pct).sum::<f64>() / (total_compounds as f64)
    } else {
        0.0
    };

    let front_runner_idx = rows.first().map(|r| r.mol_idx);
    let backup_scaffold_idx = if let Some(front) = rows.first() {
        rows.iter()
            .skip(1)
            .find(|r| {
                !r.scaffold_key.is_empty()
                    && r.scaffold_key != front.scaffold_key
                    && r.pains_count == 0
            })
            .or_else(|| rows.get(1))
            .map(|r| r.mol_idx)
    } else {
        None
    };

    CampaignScorecardSummary {
        profile,
        dose_mg,
        total_compounds,
        candidate_ready_count,
        lead_opt_count,
        prodrug_rescue_count,
        early_hit_count,
        pareto_count,
        mean_geometric_desirability,
        mean_f_oral_pct,
        front_runner_idx,
        backup_scaffold_idx,
        rows,
    }
}

/// Builds a full cross-module `LeadCampaignDossier` including 2D structure/alert/SoM SVGs,
/// 6-axis `Derringer-Suich` desirability radar SVG, `RK4` `PBPK` curve SVG, `Fa*Fg*Fh` cascade SVG,
/// `SMARTCyp` soft-spot table, `BRICS` retrosynthesis, and automated next-cycle structural proposals.
pub fn build_lead_campaign_dossier(
    mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    mol_idx: Option<usize>,
    profile: CampaignTargetProfile,
    pbpk_config: PbpkDosingConfig,
) -> LeadCampaignDossier {
    let mut working_mol = mol.clone();
    if working_mol.atoms.iter().all(|a| a.pos2d == [0.0, 0.0]) && working_mol.atoms.len() > 1 {
        compute_2d_coords(&mut working_mol);
    }

    let idx = mol_idx.unwrap_or(0);
    let (mut scorecard, mut gates, _) =
        evaluate_molecule_scorecard(&working_mol, idx, profile, pbpk_config.dose_mg);

    // Re-simulate oral PBPK with full user PbpkDosingConfig (includingQD/BID & Fed/Fasted & Efflux toggle)
    let pbpk = simulate_oral_pbpk(&working_mol, pbpk_config);
    scorecard.f_oral_pct = pbpk.f_oral_pct;
    scorecard.kp_uu_brain = pbpk.kp_uu_brain;
    scorecard.target_compartment_t_above_ic50_hr =
        if profile == CampaignTargetProfile::OralCnsPenetrant {
            pbpk.time_above_ic50_brain_hr
        } else {
            pbpk.time_above_ic50_plasma_hr
        };
    scorecard.peak_target_ro_pct = if profile == CampaignTargetProfile::OralCnsPenetrant {
        pbpk.peak_ro_brain_pct
    } else {
        pbpk.peak_ro_plasma_pct
    };
    if let Some(g9) = gates.iter_mut().find(|g| g.code == "G09") {
        g9.observed_value = format!(
            "F_oral {:.1}% (Fa {:.0}%, Fg {:.0}%, Fh {:.0}%, {})",
            pbpk.f_oral_pct,
            pbpk.fa_integrated * 100.0,
            pbpk.fg_gut_wall * 100.0,
            pbpk.fh_hepatic * 100.0,
            pbpk_config.prandial.short_label()
        );
    }
    if let Some(g11) = gates.iter_mut().find(|g| g.code == "G11") {
        g11.observed_value = format!(
            "{:.1} h above IC50 ({}), Peak RO {:.1}%, t_1/2 {:.1} h",
            scorecard.target_compartment_t_above_ic50_hr,
            pbpk_config.regimen.short_label(),
            scorecard.peak_target_ro_pct,
            pbpk.terminal_t_half_hr
        );
    }

    let cyp = analyze_cyp_metabolism(&working_mol, true);
    let synth = evaluate_synthetic_accessibility(&working_mol);
    let brics = dissect_brics(&working_mol);
    let pka_74 = evaluate_physiological_pka_74(&working_mol);
    let shape3d = evaluate_3d_summary_profile(&working_mol, true);
    let bcs_diagnosis = diagnose_parent_bcs_liability(&working_mol);
    let prodrug_rep = design_prodrugs(&working_mol, 0);
    let top_prodrug = prodrug_rep.candidates.into_iter().next();

    let macro_rep = design_macrocycles(&working_mol, 0);
    let top_macrocycle = macro_rep.candidates.into_iter().next();

    let detected_warheads = detect_covalent_warheads(&working_mol);

    let (
        top_warhead_swap,
        top_bioisosteres,
        free_wilson_pred_pact,
        campaign_neighbors,
        campaign_percentile,
    ) = if let Some(dataset) = ds {
        let cov_rep = analyze_covalent_warheads_and_swaps(
            &working_mol,
            dataset,
            dataset.default_sar_matrix.as_ref(),
            100.0,
        );
        let top_swap = cov_rep.swaps.into_iter().next();
        let mut bios = generate_bioisosteres(
            &working_mol,
            dataset,
            dataset.default_sar_matrix.as_ref(),
        );
        bios.truncate(3);

        let fw_pred = predict_best_free_wilson(
            &working_mol,
            dataset.default_sar_matrix.as_ref(),
            dataset.default_sar_matrix.as_ref(),
            &dataset.mols,
            &dataset.profiles,
        )
        .map(|p| p.predicted_pact);

        let mut neighbors = Vec::new();
        if let Some(m_idx) = mol_idx {
            if let Some(p) = dataset.profiles.get(m_idx) {
                scorecard.pareto_optimal = p.pareto_optimal;
            }
            // Top SALI cliff neighbor
            let mut my_edges: Vec<_> = dataset
                .edges
                .iter()
                .filter(|e| e.mol_a == m_idx || e.mol_b == m_idx)
                .collect();
            my_edges.sort_by(|a, b| {
                b.sali
                    .partial_cmp(&a.sali)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(edge) = my_edges.first() {
                let nb_idx = if edge.mol_a == m_idx {
                    edge.mol_b
                } else {
                    edge.mol_a
                };
                if let Some(nb_mol) = dataset.mols.get(nb_idx) {
                    neighbors.push(DossierNeighborNote {
                        relation_kind: if edge.is_cliff {
                            "SALI Activity Cliff"
                        } else {
                            "Nearest ECFP4 Neighbor"
                        },
                        compound_id: nb_mol.id.clone(),
                        compound_name: nb_mol.name.clone(),
                        summary: format!(
                            "Tanimoto {:.2}, |dpAct| {:.2}, SALI {:.1} (neighbor pAct {:.2})",
                            edge.tanimoto, edge.delta_pact, edge.sali, nb_mol.pact
                        ),
                    });
                }
            }
            // Top MMP transform
            if let Some(mmp) = dataset
                .mmps
                .iter()
                .find(|m| m.mol_from == m_idx || m.mol_to == m_idx)
            {
                let nb_idx = if mmp.mol_from == m_idx {
                    mmp.mol_to
                } else {
                    mmp.mol_from
                };
                if let Some(nb_mol) = dataset.mols.get(nb_idx) {
                    neighbors.push(DossierNeighborNote {
                        relation_kind: mmp.kind.label(),
                        compound_id: nb_mol.id.clone(),
                        compound_name: nb_mol.name.clone(),
                        summary: format!(
                            "{} -> dpAct {:+.2}, dcLogP {:+.2}, dLLE {:+.2}",
                            mmp.transform, mmp.delta_pact, mmp.delta_clogp, mmp.delta_lle
                        ),
                    });
                }
            }
        }

        // Compute rank & percentile against campaign
        let camp = evaluate_campaign_scorecard(dataset, profile, pbpk_config.dose_mg);
        let better_or_equal = camp
            .rows
            .iter()
            .filter(|r| r.geometric_desirability <= scorecard.geometric_desirability + 1e-6)
            .count();
        if let Some(m_idx) = mol_idx {
            if let Some(found) = camp.rows.iter().find(|r| r.mol_idx == m_idx) {
                scorecard.rank = found.rank;
            }
        }
        let pct = if camp.total_compounds > 0 {
            ((better_or_equal as f64) / (camp.total_compounds as f64) * 100.0).clamp(1.0, 100.0)
        } else {
            100.0
        };
        (top_swap, bios, fw_pred, neighbors, pct)
    } else {
        let single_ds = DatasetAnalysis::from_molecules(vec![working_mol.clone()], 0.30, 6.0);
        let cov_rep = analyze_covalent_warheads_and_swaps(&working_mol, &single_ds, None, 100.0);
        let top_swap = cov_rep.swaps.into_iter().next();
        let mut bios = generate_bioisosteres(&working_mol, &single_ds, None);
        bios.truncate(3);
        (top_swap, bios, None, Vec::new(), 100.0)
    };

    // Synthesize concrete next-DMTA-cycle structural design actions from cross-module engines
    let mut optimization_actions = Vec::new();
    let mut next_prio = 1u8;

    if scorecard.pains_count > 0 {
        let alerts = scan_structural_alerts(&working_mol);
        let desc = alerts
            .iter()
            .find(|a| a.severity == AlertSeverity::PainsAssayInterference)
            .map(|a| format!("{} ({})", a.code, a.description))
            .unwrap_or_else(|| "PAINS substructure".to_string());
        optimization_actions.push(MedChemOptimizationAction {
            priority: next_prio,
            domain: "PAINS / Assay Fidelity",
            issue_detected: format!("Detected promiscuous motif: {desc}"),
            structural_proposal:
                "Excise redox/reactive chromophore and replace with saturated or isosteric heterocycle."
                    .to_string(),
            quantitative_impact: "Eliminates non-specific assay interference and raises Safety Gate G06.".to_string(),
            candidate_smiles: top_bioisosteres.first().map(|b| b.canonical_smiles.clone()),
        });
        next_prio += 1;
    }

    if cyp.stability_class == MetabolicStabilityClass::HighClearanceLabile
        || cyp.cl_int_ml_min_kg > 26.0
        || scorecard.reactive_metabolite_risk
    {
        if let Some(site) = cyp.sites.first() {
            optimization_actions.push(MedChemOptimizationAction {
                priority: next_prio,
                domain: "SMARTCyp Soft-Spot Blocking",
                issue_detected: format!(
                    "Primary {} site at {} ({}) drives CL_int = {:.1} mL/min/kg{}",
                    site.dominant_isoform.label(),
                    site.atom_label,
                    site.reaction_kind.label(),
                    cyp.cl_int_ml_min_kg,
                    if scorecard.reactive_metabolite_risk {
                        " + reactive metabolite flag"
                    } else {
                        ""
                    }
                ),
                structural_proposal: site.blocking_strategy.to_string(),
                quantitative_impact: format!(
                    "Predicted to extend hepatic t_1/2 above 45 min and increase Fh from {:.0}%.",
                    pbpk.fh_hepatic * 100.0
                ),
                candidate_smiles: top_bioisosteres.first().map(|b| b.canonical_smiles.clone()),
            });
            next_prio += 1;
        }
    }

    if (pbpk.f_oral_pct < 38.0 || bcs_diagnosis.bcs_class != BcsClass::ClassI)
        && top_prodrug.is_some()
    {
        if let Some(pd) = &top_prodrug {
            if pd.activation.predicted_active_f_oral_pct > pbpk.f_oral_pct + 6.0 {
                optimization_actions.push(MedChemOptimizationAction {
                    priority: next_prio,
                    domain: "Bioreversible Prodrug Rescue",
                    issue_detected: format!(
                        "{}: {}",
                        bcs_diagnosis.bcs_class.short_label(),
                        bcs_diagnosis.primary_bottleneck
                    ),
                    structural_proposal: format!(
                        "Install {} ({}) on {} activated by {} (t_1/2 = {:.1} min).",
                        pd.promoiety_name,
                        pd.family.short_label(),
                        pd.handle_label,
                        pd.activation.primary_enzyme.short_label(),
                        pd.activation.activation_t_half_min
                    ),
                    quantitative_impact: format!(
                        "Lifts active oral bioavailability from {:.1}% to {:.1}% ({:.2}x gain).",
                        pbpk.f_oral_pct,
                        pd.activation.predicted_active_f_oral_pct,
                        pd.activation.f_oral_fold_gain
                    ),
                    candidate_smiles: Some(pd.canonical_smiles.clone()),
                });
                next_prio += 1;
            }
        }
    }

    if profile == CampaignTargetProfile::OralCnsPenetrant && pbpk.kp_uu_brain < 0.30 {
        optimization_actions.push(MedChemOptimizationAction {
            priority: next_prio,
            domain: "BBB Efflux / Kp,uu Tuning",
            issue_detected: format!(
                "Unbound brain Kp,uu,brain = {:.2} limited by apical efflux ratio {:.1}x (TPSA {:.1} A^2, HBD {}).",
                pbpk.kp_uu_brain, pbpk.bbb_efflux_ratio, scorecard.tpsa, working_mol.atoms.iter().filter(|a| a.total_h() > 0 && a.is_hetero()).count()
            ),
            structural_proposal: if let Some(mc) = &top_macrocycle {
                format!(
                    "Mask exposed HBD or cyclize via {} ({}-membered ring) to shield polarity and attenuate P-gp recognition.",
                    mc.tether_name, mc.preorg.macro_ring_size
                )
            } else {
                "N-methylate non-hinge amide/amine HBD or attenuate basic amine pKa below 8.0 to reduce P-gp/BCRP efflux.".to_string()
            },
            quantitative_impact: "Targets Kp,uu,brain >= 0.30 and CNS-MPO >= 4.2 for central target occupancy.".to_string(),
            candidate_smiles: top_macrocycle.as_ref().map(|m| m.canonical_smiles.clone()),
        });
        next_prio += 1;
    }

    if let Some(mc) = &top_macrocycle {
        if mc.preorg.potency_fold_boost >= 2.2 && optimization_actions.len() < 4 {
            optimization_actions.push(MedChemOptimizationAction {
                priority: next_prio,
                domain: "3D Macrocyclic Preorganization",
                issue_detected: format!(
                    "Acyclic conformation carries Kier Phi = {:.2} rotatable flexibility penalty.",
                    mc.preorg.parent_kier_phi
                ),
                structural_proposal: format!(
                    "Close {}-membered {} ({}) across {} (Core 3D RMSD {:.2} A).",
                    mc.preorg.macro_ring_size,
                    mc.tether_name,
                    mc.reaction.label(),
                    mc.pair_label,
                    mc.preorg.core_rmsd_3d_a
                ),
                quantitative_impact: format!(
                    "Entropy restriction (-T*dS = {:+.2} kcal/mol) predicts {:.1}x potency gain (pAct {:.2}).",
                    mc.preorg.entropy_ddg_kcal,
                    mc.preorg.potency_fold_boost,
                    mc.preorg.predicted_macro_pact
                ),
                candidate_smiles: Some(mc.canonical_smiles.clone()),
            });
            next_prio += 1;
        }
    }

    if profile == CampaignTargetProfile::TargetedCovalent {
        if let Some(sw) = &top_warhead_swap {
            optimization_actions.push(MedChemOptimizationAction {
                priority: next_prio,
                domain: "Covalent Electrophile Tuning",
                issue_detected: if let Some(w0) = detected_warheads.first() {
                    format!(
                        "Current warhead {} exhibits GSH t_1/2 = {:.1} h ({}).",
                        w0.warhead_name,
                        w0.gsh_half_life_h,
                        w0.window.short_label()
                    )
                } else {
                    "Parent lacks a tuned covalent electrophile.".to_string()
                },
                structural_proposal: format!(
                    "Swap/install {} ({}, clinical ref: {}).",
                    sw.template_name,
                    sw.reversibility.short_label(),
                    sw.clinical_precedent
                ),
                quantitative_impact: format!(
                    "Achieves TCI selectivity {:.1}/100 with k_inact/K_I = {:.0} M^-1 s^-1.",
                    sw.tci_selectivity_score, sw.kinact_over_ki
                ),
                candidate_smiles: Some(sw.canonical_smiles.clone()),
            });
            next_prio += 1;
        }
    }

    if let Some(bio) = top_bioisosteres.first() {
        if optimization_actions.len() < 4 {
            optimization_actions.push(MedChemOptimizationAction {
                priority: next_prio,
                domain: "Meanwell Tactical Bioisostere",
                issue_detected: format!(
                    "Lipophilicity/efficiency tuning opportunity (current LLE = {:.2}, logD_7.4 = {:.2}).",
                    scorecard.lle, scorecard.logd_74
                ),
                structural_proposal: format!("{} — {}", bio.rule_name, bio.tactical_rationale),
                quantitative_impact: format!(
                    "Predicted pAct {:.2} ({:+.2}), LLE {:.2}, dcLogP {:+.2}, dQED {:+.2}.",
                    bio.predicted_pact, bio.delta_pact, bio.predicted_lle, bio.delta_clogp, bio.delta_qed
                ),
                candidate_smiles: Some(bio.canonical_smiles.clone()),
            });
        }
    }

    // Build atom highlight sets for 2D SVG rendering:
    // Alert/PAINS atoms in red (#ef4444), SMARTCyp top SoMs + warhead electrophiles in cyan (#38bdf8)
    let alerts = scan_structural_alerts(&working_mol);
    let mut alert_atom_set = BTreeSet::new();
    for a in &alerts {
        if a.severity == AlertSeverity::PainsAssayInterference {
            for &ai in &a.atoms {
                alert_atom_set.insert(ai);
            }
        }
    }
    let alert_atoms: Vec<usize> = alert_atom_set.into_iter().collect();

    let mut focus_atom_set = BTreeSet::new();
    for s in cyp.sites.iter().take(2) {
        focus_atom_set.insert(s.atom_idx);
    }
    for w in &detected_warheads {
        focus_atom_set.insert(w.electrophile_atom);
    }
    let focus_atoms: Vec<usize> = focus_atom_set.into_iter().collect();

    let structure_2d_svg_light = render_molecule_svg_with_smarts(
        &working_mol,
        420,
        290,
        false,
        &alert_atoms,
        &focus_atoms,
        &[],
    );
    let structure_2d_svg_dark = render_molecule_svg_with_smarts(
        &working_mol,
        420,
        290,
        true,
        &alert_atoms,
        &focus_atoms,
        &[],
    );

    let radar_svg_light = render_dossier_radar_svg(
        &scorecard.pillar_scores,
        &format!(
            "Derringer-Suich D = {:.1} / 100",
            scorecard.geometric_desirability
        ),
        false,
    );
    let pk_curve_svg_light = render_pbpk_svg(&pbpk, scorecard.ic50_nm, false);
    let cascade_svg_light = render_oral_cascade_svg(&pbpk, top_prodrug.as_ref(), false);

    LeadCampaignDossier {
        profile,
        pbpk_config,
        scorecard,
        campaign_percentile,
        gates,
        pbpk,
        cyp,
        synth,
        brics,
        pka_74,
        shape3d_class: shape3d.shape_class,
        npr1: shape3d.npr1,
        npr2: shape3d.npr2,
        dipole_debye: shape3d.dipole_debye,
        bcs_diagnosis,
        top_prodrug,
        top_macrocycle,
        detected_warheads,
        top_warhead_swap,
        top_bioisosteres,
        free_wilson_pred_pact,
        campaign_neighbors,
        optimization_actions,
        structure_2d_svg_light,
        structure_2d_svg_dark,
        radar_svg_light,
        pk_curve_svg_light,
        cascade_svg_light,
    }
}

/// Renders a standalone 6-axis hexagonal multi-parameter desirability radar `.svg`.
pub fn render_dossier_radar_svg(
    pillar_scores: &[f64; 6],
    subtitle: &str,
    dark_theme: bool,
) -> String {
    let w = 420.0_f64;
    let h = 290.0_f64;
    let cx = 210.0_f64;
    let cy = 148.0_f64;
    let r_max = 92.0_f64;

    let bg = if dark_theme { "#0f172a" } else { "#ffffff" };
    let grid_stroke = if dark_theme { "#334155" } else { "#cbd5e1" };
    let text_main = if dark_theme { "#f1f5f9" } else { "#0f172a" };
    let text_sub = if dark_theme { "#94a3b8" } else { "#475569" };
    let poly_stroke = if dark_theme { "#38bdf8" } else { "#0284c7" };
    let poly_fill = if dark_theme { "#38bdf8" } else { "#0ea5e9" };

    let mut svg = String::with_capacity(4096);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" width=\"{w:.0}\" height=\"{h:.0}\">\n"
    ));
    svg.push_str(&format!(
        "  <rect width=\"100%\" height=\"100%\" rx=\"10\" fill=\"{bg}\"/>\n"
    ));
    svg.push_str(&format!(
        "  <text x=\"{cx:.0}\" y=\"22\" text-anchor=\"middle\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"12.5\" font-weight=\"700\" fill=\"{text_main}\">6-Pillar Multi-Parameter Desirability</text>\n"
    ));
    svg.push_str(&format!(
        "  <text x=\"{cx:.0}\" y=\"37\" text-anchor=\"middle\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"10.5\" fill=\"{text_sub}\">{}</text>\n",
        escape_xml(subtitle)
    ));

    // Concentric hexagonal rings (25, 50, 75, 100)
    for ring_pct in [0.25_f64, 0.50, 0.75, 1.00] {
        let mut pts = String::new();
        for k in 0..6 {
            let ang = -std::f64::consts::FRAC_PI_2 + (k as f64) * TAU / 6.0;
            let px = cx + r_max * ring_pct * ang.cos();
            let py = cy + r_max * ring_pct * ang.sin();
            pts.push_str(&format!("{px:.1},{py:.1} "));
        }
        let dash = if (ring_pct - 0.75).abs() < 1e-6 {
            " stroke-dasharray=\"3,3\""
        } else {
            ""
        };
        svg.push_str(&format!(
            "  <polygon points=\"{}\" fill=\"none\" stroke=\"{grid_stroke}\" stroke-width=\"1\"{dash}/>\n",
            pts.trim()
        ));
    }

    // Spokes & outer labels
    let mut data_pts = String::new();
    let mut vertex_coords = [[0.0_f64; 2]; 6];
    for k in 0..6 {
        let ang = -std::f64::consts::FRAC_PI_2 + (k as f64) * TAU / 6.0;
        let ox = cx + r_max * ang.cos();
        let oy = cy + r_max * ang.sin();
        svg.push_str(&format!(
            "  <line x1=\"{cx:.1}\" y1=\"{cy:.1}\" x2=\"{ox:.1}\" y2=\"{oy:.1}\" stroke=\"{grid_stroke}\" stroke-width=\"1\"/>\n"
        ));

        let val = (pillar_scores[k] / 100.0).clamp(0.05, 1.0);
        let vx = cx + r_max * val * ang.cos();
        let vy = cy + r_max * val * ang.sin();
        vertex_coords[k] = [vx, vy];
        data_pts.push_str(&format!("{vx:.1},{vy:.1} "));

        let lx = cx + (r_max + 22.0) * ang.cos();
        let ly = cy + (r_max + 17.0) * ang.sin();
        let anchor = if ang.cos() > 0.25 {
            "start"
        } else if ang.cos() < -0.25 {
            "end"
        } else {
            "middle"
        };
        svg.push_str(&format!(
            "  <text x=\"{lx:.1}\" y=\"{ly:.1}\" text-anchor=\"{anchor}\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"10\" font-weight=\"600\" fill=\"{text_main}\">{}</text>\n",
            escape_xml(DOSSIER_PILLAR_LABELS[k])
        ));
        svg.push_str(&format!(
            "  <text x=\"{lx:.1}\" y=\"{:.1}\" text-anchor=\"{anchor}\" font-family=\"monospace\" font-size=\"9.5\" fill=\"{poly_stroke}\">{:.0}/100</text>\n",
            ly + 11.5,
            pillar_scores[k]
        ));
    }

    svg.push_str(&format!(
        "  <polygon points=\"{}\" fill=\"{poly_fill}\" fill-opacity=\"0.24\" stroke=\"{poly_stroke}\" stroke-width=\"2.2\"/>\n",
        data_pts.trim()
    ));
    for [vx, vy] in vertex_coords {
        svg.push_str(&format!(
            "  <circle cx=\"{vx:.1}\" cy=\"{vy:.1}\" r=\"3.6\" fill=\"{poly_stroke}\" stroke=\"{bg}\" stroke-width=\"1.2\"/>\n"
        ));
    }
    svg.push_str("</svg>");
    svg
}

fn render_pbpk_svg(pbpk: &OralPbpkReport, ic50_nm: f64, dark_theme: bool) -> String {
    let w = 520.0_f64;
    let h = 250.0_f64;
    let pad_l = 54.0_f64;
    let pad_r = 18.0_f64;
    let pad_t = 38.0_f64;
    let pad_b = 36.0_f64;
    let pw = w - pad_l - pad_r;
    let ph = h - pad_t - pad_b;

    let bg = if dark_theme { "#0f172a" } else { "#ffffff" };
    let axis = if dark_theme { "#475569" } else { "#94a3b8" };
    let grid = if dark_theme { "#1e293b" } else { "#e2e8f0" };
    let txt = if dark_theme { "#f1f5f9" } else { "#0f172a" };
    let sub = if dark_theme { "#94a3b8" } else { "#475569" };

    let t_max = pbpk.config.regimen.horizon_hr().max(1.0);
    let mut c_peak = ic50_nm.max(1.0);
    for i in 0..PBPK_TIME_POINTS {
        c_peak = c_peak
            .max(pbpk.c_total_plasma_nm[i] as f64)
            .max(pbpk.c_unbound_plasma_nm[i] as f64)
            .max(pbpk.c_unbound_brain_nm[i] as f64);
    }
    let log_min = -1.5_f64; // ~0.03 nM
    let log_max = (c_peak.log10() + 0.45).max(1.5);

    let map_x = |t: f64| -> f64 { pad_l + (t / t_max).clamp(0.0, 1.0) * pw };
    let map_y = |c: f64| -> f64 {
        let lv = c.max(10.0_f64.powf(log_min)).log10();
        pad_t + (1.0 - ((lv - log_min) / (log_max - log_min)).clamp(0.0, 1.0)) * ph
    };

    let mut svg = String::with_capacity(5120);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" width=\"{w:.0}\" height=\"{h:.0}\">\n"
    ));
    svg.push_str(&format!(
        "  <rect width=\"100%\" height=\"100%\" rx=\"10\" fill=\"{bg}\"/>\n"
    ));
    svg.push_str(&format!(
        "  <text x=\"16\" y=\"21\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"12\" font-weight=\"700\" fill=\"{txt}\">RK4 Human Oral PBPK ({:.0} mg {}, {})</text>\n",
        pbpk.config.dose_mg,
        pbpk.config.regimen.short_label(),
        pbpk.config.prandial.short_label()
    ));

    // Legend
    svg.push_str(&format!(
        "  <text x=\"{:.0}\" y=\"21\" text-anchor=\"end\" font-family=\"system-ui, sans-serif\" font-size=\"9.5\" fill=\"{sub}\">Blue: Cu,plasma | Emerald: Cu,brain | Slate: Ctot | Coral: IC50</text>\n",
        w - 14.0
    ));

    // Log10 horizontal grid lines
    let mut exp_i = log_min.ceil() as i32;
    while (exp_i as f64) <= log_max {
        let gy = map_y(10.0_f64.powi(exp_i));
        svg.push_str(&format!(
            "  <line x1=\"{pad_l:.1}\" y1=\"{gy:.1}\" x2=\"{:.1}\" y2=\"{gy:.1}\" stroke=\"{grid}\" stroke-width=\"1\"/>\n",
            pad_l + pw
        ));
        let lbl = if exp_i >= 0 {
            format!("10^{exp_i}")
        } else {
            format!("10^{exp_i}")
        };
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"9\" fill=\"{sub}\">{lbl}</text>\n",
            pad_l - 6.0,
            gy + 3.0
        ));
        exp_i += 1;
    }

    // Time ticks
    for step in 0..=4 {
        let t_val = t_max * (step as f64) / 4.0;
        let gx = map_x(t_val);
        svg.push_str(&format!(
            "  <line x1=\"{gx:.1}\" y1=\"{pad_t:.1}\" x2=\"{gx:.1}\" y2=\"{:.1}\" stroke=\"{grid}\" stroke-width=\"1\"/>\n",
            pad_t + ph
        ));
        svg.push_str(&format!(
            "  <text x=\"{gx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"9.5\" fill=\"{sub}\">{t_val:.0}h</text>\n",
            pad_t + ph + 15.0
        ));
    }

    // Axes box
    svg.push_str(&format!(
        "  <rect x=\"{pad_l:.1}\" y=\"{pad_t:.1}\" width=\"{pw:.1}\" height=\"{ph:.1}\" fill=\"none\" stroke=\"{axis}\" stroke-width=\"1.1\"/>\n"
    ));

    // Target IC50 dashed line
    if ic50_nm > 0.0 {
        let y_ic50 = map_y(ic50_nm);
        svg.push_str(&format!(
            "  <line x1=\"{pad_l:.1}\" y1=\"{y_ic50:.1}\" x2=\"{:.1}\" y2=\"{y_ic50:.1}\" stroke=\"#ef4444\" stroke-width=\"1.4\" stroke-dasharray=\"5,4\"/>\n",
            pad_l + pw
        ));
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"9\" font-weight=\"700\" fill=\"#dc2626\">IC50 = {:.1} nM</text>\n",
            pad_l + pw - 4.0,
            (y_ic50 - 4.0).max(pad_t + 10.0),
            ic50_nm
        ));
    }

    let build_poly = |arr: &[f32; PBPK_TIME_POINTS]| -> String {
        let mut s = String::with_capacity(1024);
        for i in 0..PBPK_TIME_POINTS {
            let x = map_x(pbpk.time_hr[i] as f64);
            let y = map_y(arr[i] as f64);
            s.push_str(&format!("{x:.1},{y:.1} "));
        }
        s
    };

    svg.push_str(&format!(
        "  <polyline points=\"{}\" fill=\"none\" stroke=\"#94a3b8\" stroke-width=\"1.5\" stroke-dasharray=\"3,2\"/>\n",
        build_poly(&pbpk.c_total_plasma_nm).trim()
    ));
    svg.push_str(&format!(
        "  <polyline points=\"{}\" fill=\"none\" stroke=\"#0284c7\" stroke-width=\"2.3\"/>\n",
        build_poly(&pbpk.c_unbound_plasma_nm).trim()
    ));
    svg.push_str(&format!(
        "  <polyline points=\"{}\" fill=\"none\" stroke=\"#059669\" stroke-width=\"2.1\"/>\n",
        build_poly(&pbpk.c_unbound_brain_nm).trim()
    ));

    svg.push_str(&format!(
        "  <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"system-ui, sans-serif\" font-size=\"9.5\" fill=\"{sub}\">Time post-dose (hours, log10 nM concentration)</text>\n",
        pad_l + pw * 0.5,
        h - 5.0
    ));
    svg.push_str("</svg>");
    svg
}

fn render_oral_cascade_svg(
    pbpk: &OralPbpkReport,
    top_prodrug: Option<&ProdrugCandidate>,
    dark_theme: bool,
) -> String {
    let w = 520.0_f64;
    let h = 195.0_f64;
    let bg = if dark_theme { "#0f172a" } else { "#ffffff" };
    let txt = if dark_theme { "#f1f5f9" } else { "#0f172a" };
    let sub = if dark_theme { "#94a3b8" } else { "#475569" };
    let track = if dark_theme { "#1e293b" } else { "#f1f5f9" };

    let mut bars: Vec<(&str, f64, &str)> = vec![
        ("GI Absorbed (Fa)", pbpk.fa_integrated * 100.0, "#0284c7"),
        ("Gut-Wall Survival (Fg)", pbpk.fg_gut_wall * 100.0, "#0d9488"),
        ("Hepatic Escape (Fh)", pbpk.fh_hepatic * 100.0, "#7c3aed"),
        ("Systemic F_oral (Fa*Fg*Fh)", pbpk.f_oral_pct, "#2563eb"),
    ];
    if let Some(pd) = top_prodrug {
        if pd.activation.predicted_active_f_oral_pct > pbpk.f_oral_pct + 3.0 {
            bars.push((
                "Rescued Prodrug Active F_oral",
                pd.activation.predicted_active_f_oral_pct,
                "#059669",
            ));
        }
    }

    let mut svg = String::with_capacity(2560);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" width=\"{w:.0}\" height=\"{h:.0}\">\n"
    ));
    svg.push_str(&format!(
        "  <rect width=\"100%\" height=\"100%\" rx=\"10\" fill=\"{bg}\"/>\n"
    ));
    svg.push_str(&format!(
        "  <text x=\"16\" y=\"22\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"12\" font-weight=\"700\" fill=\"{txt}\">First-Pass Bioavailability Cascade (CAT/ACAT -&gt; Q_gut -&gt; Well-Stirred Liver)</text>\n"
    ));

    let bar_x = 196.0_f64;
    let bar_w = 256.0_f64;
    let row_gap = 29.0_f64;
    for (idx, (lbl, pct, col)) in bars.iter().enumerate() {
        let y = 38.0 + (idx as f64) * row_gap;
        let fill_w = (pct.clamp(0.0, 100.0) / 100.0) * bar_w;
        svg.push_str(&format!(
            "  <text x=\"{:.0}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"system-ui, sans-serif\" font-size=\"10.5\" font-weight=\"600\" fill=\"{sub}\">{}</text>\n",
            bar_x - 10.0,
            y + 12.5,
            escape_xml(lbl)
        ));
        svg.push_str(&format!(
            "  <rect x=\"{bar_x:.0}\" y=\"{y:.1}\" width=\"{bar_w:.0}\" height=\"16\" rx=\"4\" fill=\"{track}\"/>\n"
        ));
        svg.push_str(&format!(
            "  <rect x=\"{bar_x:.0}\" y=\"{y:.1}\" width=\"{fill_w:.1}\" height=\"16\" rx=\"4\" fill=\"{col}\"/>\n"
        ));
        svg.push_str(&format!(
            "  <text x=\"{:.0}\" y=\"{:.1}\" font-family=\"monospace\" font-size=\"10.5\" font-weight=\"700\" fill=\"{txt}\">{pct:.1}%</text>\n",
            bar_x + bar_w + 8.0,
            y + 12.5
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn strip_outer_svg_wrapper(svg_doc: &str) -> &str {
    let start = svg_doc
        .find('>')
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = svg_doc
        .rfind("</svg>")
        .unwrap_or(svg_doc.len());
    if start <= end {
        &svg_doc[start..end]
    } else {
        svg_doc
    }
}

/// Exports a single-file, publication-grade multi-panel `.svg` scorecard poster (`1240 x 840`)
/// embedding the 2D skeletal structure, 6-axis `Derringer-Suich` radar, `RK4` `PBPK` plot,
/// `Fa*Fg*Fh` cascade, and 12-criterion Stage-Gate checklist.
pub fn export_lead_dossier_svg(dossier: &LeadCampaignDossier) -> String {
    let sc = &dossier.scorecard;
    let tier_hex = sc.tier.badge_hex();
    let mut svg = String::with_capacity(24_576);
    svg.push_str(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1240 840\" width=\"1240\" height=\"840\">\n",
    );
    svg.push_str("  <rect width=\"1240\" height=\"840\" fill=\"#f8fafc\"/>\n");

    // Top Header Banner
    svg.push_str("  <rect x=\"20\" y=\"18\" width=\"1200\" height=\"76\" rx=\"12\" fill=\"#0f172a\"/>\n");
    svg.push_str(&format!(
        "  <text x=\"42\" y=\"50\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"20\" font-weight=\"800\" fill=\"#f8fafc\">{} ({}) — SAR-Atlas v2.1.0 Candidate Dossier</text>\n",
        escape_xml(&sc.name),
        escape_xml(&sc.id)
    ));
    svg.push_str(&format!(
        "  <text x=\"42\" y=\"74\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"12\" fill=\"#94a3b8\">Target: {} | Profile: {} | pAct {:.2} (IC50 {:.1} nM) | Scaffold: {}</text>\n",
        escape_xml(&sc.target),
        escape_xml(dossier.profile.label()),
        sc.pact,
        sc.ic50_nm,
        escape_xml(&sc.scaffold_label)
    ));
    svg.push_str(&format!(
        "  <rect x=\"930\" y=\"32\" width=\"270\" height=\"48\" rx=\"8\" fill=\"{tier_hex}\"/>\n"
    ));
    svg.push_str(&format!(
        "  <text x=\"1065\" y=\"53\" text-anchor=\"middle\" font-family=\"system-ui, sans-serif\" font-size=\"12.5\" font-weight=\"800\" fill=\"#ffffff\">{}</text>\n",
        escape_xml(sc.tier.short_label())
    ));
    svg.push_str(&format!(
        "  <text x=\"1065\" y=\"70\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"11.5\" font-weight=\"700\" fill=\"#f8fafc\">D = {:.1}/100 | Gates {}P/{}W/{}G</text>\n",
        sc.geometric_desirability, sc.pass_gates, sc.conditional_gates, sc.fail_gates
    ));

    // Panel 1: 2D Molecular Structure (top-left)
    svg.push_str("  <rect x=\"20\" y=\"110\" width=\"420\" height=\"290\" rx=\"10\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>\n");
    svg.push_str("  <svg x=\"20\" y=\"110\" width=\"420\" height=\"290\" viewBox=\"0 0 420 290\">\n");
    svg.push_str(strip_outer_svg_wrapper(&dossier.structure_2d_svg_light));
    svg.push_str("  </svg>\n");

    // Panel 2: 6-Axis Desirability Radar (top-middle)
    svg.push_str("  <rect x=\"456\" y=\"110\" width=\"420\" height=\"290\" rx=\"10\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>\n");
    svg.push_str("  <svg x=\"456\" y=\"110\" width=\"420\" height=\"290\" viewBox=\"0 0 420 290\">\n");
    svg.push_str(strip_outer_svg_wrapper(&dossier.radar_svg_light));
    svg.push_str("  </svg>\n");

    // Panel 3: Key Biophysical & Synthetic Summary Card (top-right)
    svg.push_str("  <rect x=\"892\" y=\"110\" width=\"328\" height=\"290\" rx=\"10\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>\n");
    svg.push_str("  <text x=\"910\" y=\"136\" font-family=\"system-ui, sans-serif\" font-size=\"13\" font-weight=\"700\" fill=\"#0f172a\">Biophysical &amp; Synthetic Snapshot</text>\n");
    let kv_rows = [
        ("MW / TPSA / Fsp3", format!("{:.1} Da / {:.1} A^2 / {:.2}", sc.mw, sc.tpsa, sc.fsp3)),
        ("cLogP / logD_7.4", format!("{:.2} / {:.2} ({})", sc.clogp, sc.logd_74, dossier.pka_74.charge_class.label())),
        ("LE / LLE / SILE", format!("{:.2} / {:.2} / {:.2}", sc.le, sc.lle, sc.sile)),
        ("QED / CNS-MPO", format!("{:.2} / {:.2}", sc.qed, sc.cns_mpo)),
        ("ESOL / BCS Class", format!("{:.1} ug/mL ({})", sc.esol_ug_ml, sc.bcs_class.short_label())),
        ("SMARTCyp CL_int", format!("{:.1} mL/min/kg (t1/2 {:.0}m)", sc.cl_int_ml_min_kg, sc.microsomal_t_half_min)),
        ("Oral F / Kp,uu,brain", format!("{:.1}% / {:.2}", sc.f_oral_pct, sc.kp_uu_brain)),
        ("SAScore / BRICS", format!("{:.2} / {} steps", sc.sa_score, sc.brics_steps)),
        ("3D Shape / Dipole", format!("{} / {:.2} D", dossier.shape3d_class, dossier.dipole_debye)),
    ];
    for (idx, (k, v)) in kv_rows.iter().enumerate() {
        let y = 162.0 + (idx as f64) * 25.0;
        svg.push_str(&format!(
            "  <text x=\"910\" y=\"{y:.0}\" font-family=\"system-ui, sans-serif\" font-size=\"10.5\" fill=\"#475569\">{}</text>\n",
            escape_xml(k)
        ));
        svg.push_str(&format!(
            "  <text x=\"1204\" y=\"{y:.0}\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"10.5\" font-weight=\"700\" fill=\"#0f172a\">{}</text>\n",
            escape_xml(v)
        ));
    }

    // Panel 4: RK4 PBPK Curve (bottom-left)
    svg.push_str("  <rect x=\"20\" y=\"416\" width=\"520\" height=\"250\" rx=\"10\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>\n");
    svg.push_str("  <svg x=\"20\" y=\"416\" width=\"520\" height=\"250\" viewBox=\"0 0 520 250\">\n");
    svg.push_str(strip_outer_svg_wrapper(&dossier.pk_curve_svg_light));
    svg.push_str("  </svg>\n");

    // Panel 5: Cascade Summary (bottom-left-footer)
    svg.push_str("  <rect x=\"20\" y=\"678\" width=\"520\" height=\"144\" rx=\"10\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>\n");
    svg.push_str("  <text x=\"36\" y=\"702\" font-family=\"system-ui, sans-serif\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">Next-DMTA-Cycle Structural Priorities</text>\n");
    for (idx, act) in dossier.optimization_actions.iter().take(3).enumerate() {
        let y = 724.0 + (idx as f64) * 32.0;
        svg.push_str(&format!(
            "  <text x=\"36\" y=\"{y:.0}\" font-family=\"system-ui, sans-serif\" font-size=\"10\" font-weight=\"700\" fill=\"#0284c7\">P{}: {}</text>\n",
            act.priority,
            escape_xml(act.domain)
        ));
        let line = truncate_str(&act.structural_proposal, 74);
        svg.push_str(&format!(
            "  <text x=\"36\" y=\"{:.0}\" font-family=\"system-ui, sans-serif\" font-size=\"9.5\" fill=\"#334155\">{}</text>\n",
            y + 13.0,
            escape_xml(&line)
        ));
    }
    if dossier.optimization_actions.is_empty() {
        svg.push_str("  <text x=\"36\" y=\"730\" font-family=\"system-ui, sans-serif\" font-size=\"10.5\" fill=\"#059669\">All Stage-Gate criteria within target windows; ready for preclinical nomination package.</text>\n");
    }

    // Panel 6: 12-Row Stage-Gate Checklist Table (bottom-right)
    svg.push_str("  <rect x=\"556\" y=\"416\" width=\"664\" height=\"406\" rx=\"10\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>\n");
    svg.push_str("  <text x=\"574\" y=\"441\" font-family=\"system-ui, sans-serif\" font-size=\"13\" font-weight=\"700\" fill=\"#0f172a\">12-Criterion Stage-Gate Target Product Profile Scorecard</text>\n");
    for (idx, g) in dossier.gates.iter().enumerate() {
        let y = 466.0 + (idx as f64) * 29.0;
        if idx % 2 == 1 {
            svg.push_str(&format!(
                "  <rect x=\"564\" y=\"{:.0}\" width=\"648\" height=\"26\" rx=\"4\" fill=\"#f8fafc\"/>\n",
                y - 16.0
            ));
        }
        let v_hex = g.verdict.hex();
        svg.push_str(&format!(
            "  <rect x=\"570\" y=\"{:.0}\" width=\"46\" height=\"18\" rx=\"4\" fill=\"{v_hex}\"/>\n",
            y - 12.0
        ));
        svg.push_str(&format!(
            "  <text x=\"593\" y=\"{y:.0}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"9.5\" font-weight=\"700\" fill=\"#ffffff\">{}</text>\n",
            g.verdict.label()
        ));
        svg.push_str(&format!(
            "  <text x=\"626\" y=\"{y:.0}\" font-family=\"system-ui, sans-serif\" font-size=\"10.2\" font-weight=\"600\" fill=\"#0f172a\">{} {}</text>\n",
            g.code,
            escape_xml(&truncate_str(g.metric_name, 34))
        ));
        svg.push_str(&format!(
            "  <text x=\"1204\" y=\"{y:.0}\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"9.8\" fill=\"#334155\">{}</text>\n",
            escape_xml(&truncate_str(&g.observed_value, 42))
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        out.push_str("...");
        out
    }
}

const DOSSIER_HTML_CSS: &str = r#"
:root {
  --ink: #0f172a;
  --muted: #475569;
  --border: #cbd5e1;
  --bg: #f1f5f9;
  --card: #ffffff;
  --accent: #0284c7;
  --pass: #059669;
  --watch: #d97706;
  --gap: #dc2626;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.45;
}
.page {
  max-width: 1180px;
  margin: 0 auto 28px auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 26px 30px;
  box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
}
.hero {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 12px;
  padding: 20px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  margin-bottom: 20px;
}
.hero h1 { margin: 0 0 6px 0; font-size: 22px; letter-spacing: -0.01em; }
.hero .meta { color: #cbd5e1; font-size: 13px; }
.hero .smiles {
  margin-top: 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11.5px;
  color: #7dd3fc;
  word-break: break-all;
}
.tier-box {
  min-width: 240px;
  text-align: center;
  padding: 12px 16px;
  border-radius: 10px;
  color: #ffffff;
  font-weight: 700;
}
.tier-box .big { font-size: 22px; font-family: ui-monospace, monospace; }
.tier-box .lbl { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.grid-2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-bottom: 18px;
}
.grid-3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 18px;
}
.panel {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  background: #ffffff;
}
.panel h2 {
  margin: 0 0 10px 0;
  font-size: 14.5px;
  color: #0f172a;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 6px;
}
.svg-wrap { text-align: center; overflow-x: auto; }
.svg-wrap svg { max-width: 100%; height: auto; }
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}
.kpi {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 9px 11px;
}
.kpi .k { font-size: 11px; color: var(--muted); font-weight: 600; }
.kpi .v { font-size: 15.5px; font-weight: 800; font-family: ui-monospace, monospace; margin-top: 2px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}
th {
  background: #f1f5f9;
  color: #0f172a;
  text-align: left;
  font-weight: 700;
  padding: 8px 10px;
  border-bottom: 2px solid var(--border);
}
td {
  padding: 7px 10px;
  border-bottom: 1px solid #e2e8f0;
  vertical-align: top;
}
td.num, th.num {
  text-align: right;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  color: #ffffff;
  font-size: 10.5px;
  font-weight: 700;
  font-family: ui-monospace, monospace;
}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
.refs { font-size: 11.5px; color: var(--muted); margin-top: 16px; }
.refs a { color: var(--accent); text-decoration: none; }
.refs a:hover { text-decoration: underline; }
@media print {
  body { background: #ffffff; padding: 0; }
  .page { box-shadow: none; border: none; margin: 0; max-width: 100%; page-break-after: always; }
}
"#;

fn render_single_dossier_section_html(dossier: &LeadCampaignDossier) -> String {
    let sc = &dossier.scorecard;
    let mut html = String::with_capacity(28_672);

    html.push_str("<section class=\"page\">\n");
    html.push_str("  <header class=\"hero\">\n");
    html.push_str("    <div>\n");
    html.push_str(&format!(
        "      <h1>{} <span style=\"color:#94a3b8;font-weight:500;\">({})</span></h1>\n",
        escape_xml(&sc.name),
        escape_xml(&sc.id)
    ));
    html.push_str(&format!(
        "      <div class=\"meta\">Target: <strong>{}</strong> &bull; TPP Profile: <strong>{}</strong> &bull; Campaign Rank: <strong>#{}</strong> ({:.0}th percentile) &bull; Murcko Scaffold: <strong>{}</strong></div>\n",
        escape_xml(&sc.target),
        escape_xml(dossier.profile.label()),
        sc.rank,
        dossier.campaign_percentile,
        escape_xml(&sc.scaffold_label)
    ));
    html.push_str(&format!(
        "      <div class=\"smiles\">SMILES: {}</div>\n",
        escape_xml(&sc.smiles)
    ));
    html.push_str("    </div>\n");
    html.push_str(&format!(
        "    <div class=\"tier-box\" style=\"background:{};\">\n",
        sc.tier.badge_hex()
    ));
    html.push_str(&format!(
        "      <div class=\"lbl\">{}</div>\n",
        escape_xml(sc.tier.short_label())
    ));
    html.push_str(&format!(
        "      <div class=\"big\">D = {:.1} / 100</div>\n",
        sc.geometric_desirability
    ));
    html.push_str(&format!(
        "      <div style=\"font-size:11.5px;opacity:0.92;\">Weighted {:.1} &bull; {} PASS / {} WATCH / {} GAP</div>\n",
        sc.weighted_score_100, sc.pass_gates, sc.conditional_gates, sc.fail_gates
    ));
    html.push_str("    </div>\n");
    html.push_str("  </header>\n");

    // 6-Card KPI Strip
    let kpis = [
        ("pAct / IC50", format!("{:.2} ({:.1} nM)", sc.pact, sc.ic50_nm)),
        ("LE / LLE / LLE7.4", format!("{:.2} / {:.2} / {:.2}", sc.le, sc.lle, sc.lle_74)),
        ("QED / CNS-MPO", format!("{:.2} / {:.2}", sc.qed, sc.cns_mpo)),
        ("pH 7.4 logD / Sol", format!("{:.2} / {:.0} ug/mL", sc.logd_74, sc.esol_ug_ml)),
        ("Oral F / Kp,uu,br", format!("{:.1}% / {:.2}", sc.f_oral_pct, sc.kp_uu_brain)),
        ("SAScore / BRICS", format!("{:.2} ({} steps)", sc.sa_score, sc.brics_steps)),
    ];
    html.push_str("  <div class=\"kpi-strip\">\n");
    for (k, v) in kpis {
        html.push_str(&format!(
            "    <div class=\"kpi\"><div class=\"k\">{}</div><div class=\"v\">{}</div></div>\n",
            escape_xml(k),
            escape_xml(&v)
        ));
    }
    html.push_str("  </div>\n");

    // Top 2-Column Visual Row: 2D Structure (with SoM + PAINS halos) + 6-Axis Desirability Radar
    html.push_str("  <div class=\"grid-2\">\n");
    html.push_str("    <div class=\"panel\">\n");
    html.push_str("      <h2>2D Skeletal Structure (Cyan: Top SMARTCyp SoM / Electrophile &bull; Coral: PAINS)</h2>\n");
    html.push_str("      <div class=\"svg-wrap\">");
    html.push_str(&dossier.structure_2d_svg_light);
    html.push_str("</div>\n");
    html.push_str(&format!(
        "      <div style=\"font-size:11.5px;color:#475569;margin-top:6px;\">3D Shape: <strong>{}</strong> (NPR1 {:.2}, NPR2 {:.2}) &bull; Dipole: <strong>{:.2} D</strong> &bull; Charge at pH 7.4: <strong>{}</strong>{}",
        escape_xml(dossier.shape3d_class),
        dossier.npr1,
        dossier.npr2,
        dossier.dipole_debye,
        escape_xml(dossier.pka_74.charge_class.label()),
        if let Some(fw) = dossier.free_wilson_pred_pact {
            format!(" &bull; Free-Wilson QSAR pAct: <strong>{fw:.2}</strong>")
        } else {
            String::new()
        }
    ));
    html.push_str("</div>\n    </div>\n");

    html.push_str("    <div class=\"panel\">\n");
    html.push_str("      <h2>Derringer-Suich 6-Axis Candidate Profile</h2>\n");
    html.push_str("      <div class=\"svg-wrap\">");
    html.push_str(&dossier.radar_svg_light);
    html.push_str("</div>\n    </div>\n");
    html.push_str("  </div>\n");

    // Second 2-Column Visual Row: RK4 Oral PBPK Curves + Fa*Fg*Fh Cascade
    html.push_str("  <div class=\"grid-2\">\n");
    html.push_str("    <div class=\"panel\">\n");
    html.push_str("      <h2>Physiologically Based Oral PK &amp; Unbound Brain Delivery</h2>\n");
    html.push_str("      <div class=\"svg-wrap\">");
    html.push_str(&dossier.pk_curve_svg_light);
    html.push_str("</div>\n    </div>\n");

    html.push_str("    <div class=\"panel\">\n");
    html.push_str("      <h2>GI Absorption, Gut-Wall &amp; Hepatic First-Pass Cascade</h2>\n");
    html.push_str("      <div class=\"svg-wrap\">");
    html.push_str(&dossier.cascade_svg_light);
    html.push_str("</div>\n");
    html.push_str(&format!(
        "      <div style=\"font-size:12px;color:#334155;margin-top:8px;\"><strong>Distribution &amp; Engagement:</strong> fu,plasma {:.3}, fu,brain {:.3}, Rodgers-Rowland Vss {:.2} L/kg ({:.0} L), terminal t<sub>1/2</sub> {:.1} h, Plasma Cu &ge; IC50 for <strong>{:.1} h</strong> (Peak RO {:.1}%), Brain Cu &ge; IC50 for <strong>{:.1} h</strong> (Peak RO {:.1}%).</div>\n",
        dossier.pbpk.fu_plasma,
        dossier.pbpk.fu_brain,
        dossier.pbpk.vss_l_per_kg,
        dossier.pbpk.vss_total_l,
        dossier.pbpk.terminal_t_half_hr,
        dossier.pbpk.time_above_ic50_plasma_hr,
        dossier.pbpk.peak_ro_plasma_pct,
        dossier.pbpk.time_above_ic50_brain_hr,
        dossier.pbpk.peak_ro_brain_pct
    ));
    html.push_str("    </div>\n");
    html.push_str("  </div>\n");

    // 12-Criterion Stage-Gate Table
    html.push_str("  <div class=\"panel\" style=\"margin-bottom:18px;\">\n");
    html.push_str("    <h2>12-Criterion Stage-Gate Target Product Profile Evaluation</h2>\n");
    html.push_str("    <table>\n      <thead><tr><th>Gate</th><th>Domain</th><th>Criterion</th><th>Target Window</th><th>Observed / Predicted</th><th class=\"num\">d<sub>i</sub></th><th>Status</th><th>Engine</th></tr></thead>\n      <tbody>\n");
    for g in &dossier.gates {
        html.push_str(&format!(
            "        <tr><td class=\"mono\">{}</td><td>{}</td><td><strong>{}</strong></td><td>{}</td><td class=\"mono\">{}</td><td class=\"num\">{:.2}</td><td><span class=\"badge\" style=\"background:{};\">{}</span></td><td class=\"mono\">{}</td></tr>\n",
            escape_xml(g.code),
            escape_xml(g.category),
            escape_xml(g.metric_name),
            escape_xml(g.target_window),
            escape_xml(&g.observed_value),
            g.desirability_0_1,
            g.verdict.hex(),
            g.verdict.label(),
            escape_xml(g.source_module)
        ));
    }
    html.push_str("      </tbody>\n    </table>\n  </div>\n");

    // Metabolism + Synthetic Route / Prodrug Rescue Row
    html.push_str("  <div class=\"grid-2\">\n");
    html.push_str("    <div class=\"panel\">\n");
    html.push_str("      <h2>SMARTCyp Regioselective Sites of Metabolism &amp; Phase I/II Metabolites</h2>\n");
    if dossier.cyp.sites.is_empty() {
        html.push_str("      <p style=\"font-size:12.5px;color:#475569;\">No labile CYP450 C-H / heteroatom sites detected.</p>\n");
    } else {
        html.push_str("      <table>\n        <thead><tr><th>Atom</th><th>Reaction</th><th>Isoform</th><th class=\"num\">3A4 Score</th><th>Structural Blocking Proposal</th></tr></thead>\n        <tbody>\n");
        for s in dossier.cyp.sites.iter().take(4) {
            html.push_str(&format!(
                "          <tr><td class=\"mono\"><strong>{}</strong></td><td>{}</td><td>{}</td><td class=\"num\">{:.1}</td><td>{}</td></tr>\n",
                escape_xml(&s.atom_label),
                escape_xml(s.reaction_kind.label()),
                escape_xml(s.dominant_isoform.label()),
                s.score_3a4,
                escape_xml(s.blocking_strategy)
            ));
        }
        html.push_str("        </tbody>\n      </table>\n");
    }
    if !dossier.cyp.metabolites.is_empty() {
        html.push_str("      <div style=\"margin-top:10px;font-size:12px;\"><strong>Major Predicted Metabolites:</strong> ");
        let items: Vec<String> = dossier
            .cyp
            .metabolites
            .iter()
            .take(3)
            .map(|m| {
                format!(
                    "{} ({}, {:+.1} Da, {:.0}%{})",
                    escape_xml(&m.reaction_name),
                    escape_xml(m.enzyme),
                    m.mass_shift_da,
                    m.predicted_abundance_pct,
                    if m.is_reactive_bioactivation {
                        ", REACTIVE ALERT"
                    } else {
                        ""
                    }
                )
            })
            .collect();
        html.push_str(&items.join(" &bull; "));
        html.push_str("</div>\n");
    }
    html.push_str("    </div>\n");

    html.push_str("    <div class=\"panel\">\n");
    html.push_str("      <h2>BRICS Retrosynthesis &amp; Prodrug / Backup Optionality</h2>\n");
    html.push_str(&format!(
        "      <div style=\"font-size:12.5px;margin-bottom:8px;\"><strong>Ertl SAScore:</strong> {:.2} ({}) &bull; <strong>BRICS Cuts:</strong> {} ({} convergent synthons) &bull; <strong>Stereocenters:</strong> {}</div>\n",
        dossier.synth.sa_score,
        escape_xml(dossier.synth.tractability.label()),
        dossier.synth.brics_cleavable_bonds,
        dossier.synth.brics_synthon_count,
        dossier.synth.chiral_centers
    ));
    if !dossier.brics.candidate_cuts.is_empty() {
        html.push_str("      <table>\n        <thead><tr><th>BRICS Pair</th><th>Forward Coupling Reaction</th><th class=\"num\">Priority</th></tr></thead>\n        <tbody>\n");
        for cut in dossier.brics.candidate_cuts.iter().take(4) {
            html.push_str(&format!(
                "          <tr><td class=\"mono\">{}</td><td>{}</td><td class=\"num\">{}</td></tr>\n",
                escape_xml(&cut.pair_code),
                escape_xml(cut.reaction_name),
                cut.priority
            ));
        }
        html.push_str("        </tbody>\n      </table>\n");
    }
    if let Some(pd) = &dossier.top_prodrug {
        let pd_svg = render_molecule_svg(&pd.mol, 300, 155, false, &pd.promoiety_atoms);
        html.push_str(&format!(
            "      <div style=\"margin-top:10px;padding-top:8px;border-top:1px solid #e2e8f0;font-size:12px;\"><strong>Top Bioreversible Prodrug:</strong> {} ({}, {}) &rarr; Active F<sub>oral</sub> <strong>{:.1}%</strong> ({:.2}x vs parent)</div>\n",
            escape_xml(pd.promoiety_name),
            escape_xml(pd.family.short_label()),
            escape_xml(pd.activation.primary_enzyme.short_label()),
            pd.activation.predicted_active_f_oral_pct,
            pd.activation.f_oral_fold_gain
        ));
        html.push_str("      <div class=\"svg-wrap\" style=\"margin-top:4px;\">");
        html.push_str(&pd_svg);
        html.push_str("</div>\n");
    }
    html.push_str("    </div>\n");
    html.push_str("  </div>\n");

    // Next-DMTA-Cycle Structural Design Actions
    html.push_str("  <div class=\"panel\">\n");
    html.push_str("    <h2>Prioritized Next-DMTA-Cycle Medicinal-Chemistry Recommendations</h2>\n");
    if dossier.optimization_actions.is_empty() {
        html.push_str("    <p style=\"font-size:12.5px;color:#059669;margin:0;\">All 12 Stage-Gate thresholds are met without structural alerts; compound qualifies for candidate nomination profiling.</p>\n");
    } else {
        html.push_str("    <table>\n      <thead><tr><th>Priority</th><th>Design Axis</th><th>Detected Liability</th><th>Concrete Structural Modification</th><th>Predicted Impact</th></tr></thead>\n      <tbody>\n");
        for act in &dossier.optimization_actions {
            let smi_note = act
                .candidate_smiles
                .as_ref()
                .map(|s| format!("<div class=\"mono\" style=\"color:#0284c7;margin-top:3px;\">{}</div>", escape_xml(s)))
                .unwrap_or_default();
            html.push_str(&format!(
                "        <tr><td class=\"num\"><strong>P{}</strong></td><td><strong>{}</strong></td><td>{}</td><td>{}{}</td><td>{}</td></tr>\n",
                act.priority,
                escape_xml(act.domain),
                escape_xml(&act.issue_detected),
                escape_xml(&act.structural_proposal),
                smi_note,
                escape_xml(&act.quantitative_impact)
            ));
        }
        html.push_str("      </tbody>\n    </table>\n");
    }

    if !dossier.campaign_neighbors.is_empty() {
        html.push_str("    <div style=\"margin-top:10px;font-size:12px;color:#334155;\"><strong>Campaign SAR Context:</strong> ");
        let notes: Vec<String> = dossier
            .campaign_neighbors
            .iter()
            .map(|n| {
                format!(
                    "<em>{}</em> vs {} ({}): {}",
                    escape_xml(n.relation_kind),
                    escape_xml(&n.compound_name),
                    escape_xml(&n.compound_id),
                    escape_xml(&n.summary)
                )
            })
            .collect();
        html.push_str(&notes.join(" &bull; "));
        html.push_str("</div>\n");
    }
    html.push_str("  </div>\n");

    html.push_str("  <footer class=\"refs\">\n");
    html.push_str("    <strong>Methodological References (SAR-Atlas v2.1.0):</strong> ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/22316157/\">Segall (2012) MPO</a> &bull; ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/24833294/\">Cook et al. (2014) 5R Framework</a> &bull; ");
    html.push_str("<a href=\"https://pmc.ncbi.nlm.nih.gov/articles/PMC6217265/\">Derringer-Suich Desirability</a> &bull; ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/22270643/\">Bickerton QED</a> &bull; ");
    html.push_str("<a href=\"https://pmc.ncbi.nlm.nih.gov/articles/PMC3368654/\">Wager CNS-MPO</a> &bull; ");
    html.push_str("<a href=\"https://pmc.ncbi.nlm.nih.gov/articles/PMC4055970/\">Rydberg SMARTCyp</a> &bull; ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/10486429/\">Yu-Amidon CAT</a> &bull; ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/16639716/\">Rodgers-Rowland Vss</a> &bull; ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/18058202/\">Hammarlund-Udenaes Kp,uu,brain</a> &bull; ");
    html.push_str("<a href=\"https://pmc.ncbi.nlm.nih.gov/articles/PMC3225829/\">Ertl SAScore</a> &bull; ");
    html.push_str("<a href=\"https://pubmed.ncbi.nlm.nih.gov/29700501/\">Rautio Prodrugs</a>.\n");
    html.push_str("  </footer>\n");
    html.push_str("</section>\n");

    html
}

/// Exports a 100% standalone, zero-dependency, print-ready `.html` medicinal-chemistry candidate dossier
/// with embedded vector `.svg` structure, 6-axis `Derringer-Suich` radar, `RK4` `PBPK` plot, and `BRICS`/`SMARTCyp` tables.
pub fn export_lead_dossier_html(dossier: &LeadCampaignDossier) -> String {
    let mut doc = String::with_capacity(32_768);
    doc.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n");
    doc.push_str("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    doc.push_str(&format!(
        "  <title>{} ({}) — SAR-Atlas v2.1.0 Medicinal-Chemistry Dossier</title>\n",
        escape_xml(&dossier.scorecard.name),
        escape_xml(&dossier.scorecard.id)
    ));
    doc.push_str("  <style>");
    doc.push_str(DOSSIER_HTML_CSS);
    doc.push_str("  </style>\n</head>\n<body>\n");
    doc.push_str(&render_single_dossier_section_html(dossier));
    doc.push_str("</body>\n</html>\n");
    doc
}

/// Exports a multi-compound Campaign Executive Leaderboard + Top-`N` Deep Candidate Dossiers as a single standalone `.html`.
pub fn export_campaign_dossier_html(
    ds: &DatasetAnalysis,
    summary: &CampaignScorecardSummary,
    top_n_dossiers: usize,
) -> String {
    let mut doc = String::with_capacity(65_536);
    doc.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n");
    doc.push_str("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    doc.push_str(&format!(
        "  <title>SAR-Atlas v2.1.0 Campaign Scorecard &amp; Dossier ({})</title>\n",
        escape_xml(summary.profile.label())
    ));
    doc.push_str("  <style>");
    doc.push_str(DOSSIER_HTML_CSS);
    doc.push_str("  </style>\n</head>\n<body>\n");

    // Executive Summary Cover Page
    doc.push_str("<section class=\"page\">\n");
    doc.push_str("  <header class=\"hero\">\n    <div>\n");
    doc.push_str("      <h1>SAR-Atlas v2.1.0 — Campaign Lead Scorecard &amp; Nomination Portfolio</h1>\n");
    doc.push_str(&format!(
        "      <div class=\"meta\">Target Product Profile: <strong>{}</strong> &bull; Compounds Evaluated: <strong>{}</strong> &bull; Oral PBPK Dose: <strong>{:.0} mg</strong></div>\n",
        escape_xml(summary.profile.label()),
        summary.total_compounds,
        summary.dose_mg
    ));
    doc.push_str("    </div>\n");
    doc.push_str("    <div class=\"tier-box\" style=\"background:#0284c7;\">\n");
    doc.push_str("      <div class=\"lbl\">Candidate Ready</div>\n");
    doc.push_str(&format!(
        "      <div class=\"big\">{} / {}</div>\n",
        summary.candidate_ready_count, summary.total_compounds
    ));
    doc.push_str(&format!(
        "      <div style=\"font-size:11.5px;opacity:0.92;\">Mean D = {:.1} &bull; Mean F<sub>oral</sub> = {:.1}%</div>\n",
        summary.mean_geometric_desirability, summary.mean_f_oral_pct
    ));
    doc.push_str("    </div>\n  </header>\n");

    let exec_kpis = [
        ("Candidate Ready", format!("{}", summary.candidate_ready_count)),
        ("Lead Optimization", format!("{}", summary.lead_opt_count)),
        ("Prodrug Rescue", format!("{}", summary.prodrug_rescue_count)),
        ("Hit / Back-Up Triage", format!("{}", summary.early_hit_count)),
        ("Pareto Front Leads", format!("{}", summary.pareto_count)),
        (
            "Scaffold-Distinct Backup",
            summary
                .backup_scaffold_idx
                .and_then(|i| ds.mols.get(i))
                .map(|m| m.name.clone())
                .unwrap_or_else(|| "None".to_string()),
        ),
    ];
    doc.push_str("  <div class=\"kpi-strip\">\n");
    for (k, v) in exec_kpis {
        doc.push_str(&format!(
            "    <div class=\"kpi\"><div class=\"k\">{}</div><div class=\"v\" style=\"font-size:14px;\">{}</div></div>\n",
            escape_xml(k),
            escape_xml(&v)
        ));
    }
    doc.push_str("  </div>\n");

    doc.push_str("  <div class=\"panel\">\n");
    doc.push_str("    <h2>Ranked Multi-Parameter Campaign Leaderboard</h2>\n");
    doc.push_str("    <table>\n      <thead><tr><th class=\"num\">Rank</th><th>Compound</th><th>Target</th><th class=\"num\">pAct</th><th class=\"num\">LE / LLE</th><th class=\"num\">QED / MPO</th><th class=\"num\">CL<sub>int</sub></th><th class=\"num\">F<sub>oral</sub>%</th><th class=\"num\">Kp,uu</th><th class=\"num\">SAScore</th><th class=\"num\">D<sub>geom</sub></th><th>Stage-Gate Tier</th></tr></thead>\n      <tbody>\n");
    for r in summary.rows.iter().take(30) {
        doc.push_str(&format!(
            "        <tr><td class=\"num\">{}</td><td><strong>{}</strong> <span class=\"mono\" style=\"color:#64748b;\">({})</span></td><td>{}</td><td class=\"num\">{:.2}</td><td class=\"num\">{:.2} / {:.2}</td><td class=\"num\">{:.2} / {:.1}</td><td class=\"num\">{:.1}</td><td class=\"num\">{:.1}%</td><td class=\"num\">{:.2}</td><td class=\"num\">{:.2}</td><td class=\"num\"><strong>{:.1}</strong></td><td><span class=\"badge\" style=\"background:{};\">{}</span></td></tr>\n",
            r.rank,
            escape_xml(&r.name),
            escape_xml(&r.id),
            escape_xml(&r.target),
            r.pact,
            r.le,
            r.lle,
            r.qed,
            r.cns_mpo,
            r.cl_int_ml_min_kg,
            r.f_oral_pct,
            r.kp_uu_brain,
            r.sa_score,
            r.geometric_desirability,
            r.tier.badge_hex(),
            escape_xml(r.tier.short_label())
        ));
    }
    doc.push_str("      </tbody>\n    </table>\n  </div>\n");
    doc.push_str("</section>\n");

    // Append full visual dossiers for the top N ranked compounds
    let pbpk_cfg = PbpkDosingConfig {
        dose_mg: summary.dose_mg,
        ..PbpkDosingConfig::default()
    };
    for row in summary.rows.iter().take(top_n_dossiers) {
        if let Some(mol) = ds.mols.get(row.mol_idx) {
            let dossier = build_lead_campaign_dossier(
                mol,
                Some(ds),
                Some(row.mol_idx),
                summary.profile,
                pbpk_cfg,
            );
            doc.push_str(&render_single_dossier_section_html(&dossier));
        }
    }

    doc.push_str("</body>\n</html>\n");
    doc
}

/// Exports the full campaign scorecard leaderboard as a flat `.csv` table.
pub fn export_campaign_scorecard_csv(summary: &CampaignScorecardSummary) -> String {
    let mut out = String::with_capacity(summary.rows.len() * 220 + 256);
    out.push_str("rank,id,name,target,profile,tier,d_geom_100,weighted_100,pass_gates,watch_gates,gap_gates,pact,ic50_nm,le,lle,lle_74,mw,clogp,logd_74,tpsa,qed,cns_mpo,sol_ug_ml,bcs_class,herg_risk,pains_count,cl_int_ml_min_kg,f_oral_pct,kp_uu_brain,t_above_ic50_hr,peak_ro_pct,sa_score,brics_steps,best_prodrug_f_oral_pct,pareto_optimal,smiles\n");
    for r in &summary.rows {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{:.2},{:.2},{},{},{},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{:.3},{:.2},{:.2},\"{}\",\"{}\",{},{:.2},{:.2},{:.3},{:.2},{:.1},{:.2},{},{:.2},{},\"{}\"\n",
            r.rank,
            r.id.replace('"', "'"),
            r.name.replace('"', "'"),
            r.target.replace('"', "'"),
            summary.profile.code(),
            r.tier.short_label(),
            r.geometric_desirability,
            r.weighted_score_100,
            r.pass_gates,
            r.conditional_gates,
            r.fail_gates,
            r.pact,
            r.ic50_nm,
            r.le,
            r.lle,
            r.lle_74,
            r.mw,
            r.clogp,
            r.logd_74,
            r.tpsa,
            r.qed,
            r.cns_mpo,
            r.esol_ug_ml,
            r.bcs_class.short_label(),
            r.herg_risk.label(),
            r.pains_count,
            r.cl_int_ml_min_kg,
            r.f_oral_pct,
            r.kp_uu_brain,
            r.target_compartment_t_above_ic50_hr,
            r.peak_target_ro_pct,
            r.sa_score,
            r.brics_steps,
            r.best_prodrug_f_oral_pct,
            r.pareto_optimal,
            r.smiles.replace('"', "'")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::load_benchmark_26;

    #[test]
    fn test_campaign_scorecard_and_standalone_html_svg_export() {
        let mols = load_benchmark_26();
        let ds = DatasetAnalysis::from_molecules(mols, 0.30, 6.0);
        let summary =
            evaluate_campaign_scorecard(&ds, CampaignTargetProfile::OralPeripheral, 50.0);
        assert_eq!(summary.total_compounds, 26);
        assert_eq!(summary.rows.len(), 26);
        assert!(summary.rows[0].geometric_desirability >= summary.rows[25].geometric_desirability);
        assert!(summary.front_runner_idx.is_some());
        assert!(summary.backup_scaffold_idx.is_some());

        let top_idx = summary.rows[0].mol_idx;
        let dossier = build_lead_campaign_dossier(
            &ds.mols[top_idx],
            Some(&ds),
            Some(top_idx),
            CampaignTargetProfile::OralPeripheral,
            PbpkDosingConfig::default(),
        );
        assert_eq!(dossier.gates.len(), 12);

        let html = export_lead_dossier_html(&dossier);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("Derringer-Suich"));

        let svg = export_lead_dossier_svg(&dossier);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>\n"));

        let camp_html = export_campaign_dossier_html(&ds, &summary, 2);
        assert!(camp_html.contains("Ranked Multi-Parameter Campaign Leaderboard"));

        let csv = export_campaign_scorecard_csv(&summary);
        assert_eq!(csv.lines().count(), 27);
    }
}
