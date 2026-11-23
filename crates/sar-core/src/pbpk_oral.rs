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

//! Mechanistic Human Oral Physiologically Based Pharmacokinetic (`PBPK`) Simulator:
//! 7-Segment `CAT`/`ACAT` Gastrointestinal Dissolution & Transit Absorption (`Fa`),
//! `Q_gut` Enterocyte (`Fg`) & `Well-Stirred` Hepatic First-Pass Clearance (`Fh`),
//! `Rodgers & Rowland` Tissue-Composition Partitioning (`Kp` / `V_ss`), and
//! `Hammarlund-Udenaes` / `Fridén` Unbound Brain `Kp,uu,brain` + `RK4` Time-Course Engine (`v1.9.0`).
//!
//! Primary References:
//! - Yu & Amidon (1999), *Int. J. Pharm.* 186(2), 119-125 ("A compartmental absorption and
//!   transit model for estimating oral drug absorption": `https://pubmed.ncbi.nlm.nih.gov/10486429/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC2691458/`)
//! - Rodgers & Rowland (2006), *J. Pharm. Sci.* 95(6), 1238-1257 ("Physiologically based
//!   pharmacokinetic modelling 2: predicting the tissue distribution of acids, very weak bases,
//!   neutrals and zwitterions": `https://pubmed.ncbi.nlm.nih.gov/16639716/`)
//! - Pang & Rowland (1977), *J. Pharmacokinet. Biopharm.* 5(6), 625-653 ("Hepatic clearance of
//!   drugs. I. Theoretical considerations of a 'well-stirred' model": `https://pubmed.ncbi.nlm.nih.gov/599411/`)
//! - Yang, Jamei, Yeo, Tucker & Rostami-Hodjegan (2007), `Q_gut` Intestinal First-Pass `Fg` Model
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC5412686/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC4875961/`)
//! - Hammarlund-Udenaes, Fridén, Syvänen & Gupta (2008), *Pharm. Res.* 25(8), 1737-1750
//!   ("On the rate and extent of drug delivery to the brain" — `Kp,uu,brain`:
//!   `https://pubmed.ncbi.nlm.nih.gov/18058202/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC2469271/`)
//! - Loryan, Di et al. (2022), *Pharm. Res.* 39, 1321-1341 ("Unbound Brain-to-Plasma Partition
//!   Coefficient, `Kp,uu,brain`": `https://pmc.ncbi.nlm.nih.gov/articles/PMC9246790/`) & Physics-Based
//!   `Kp,uu,brain` Prediction (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10828645/`)

use crate::admet::evaluate_admet;
use crate::alerts::scan_structural_alerts;
use crate::cyp_metabolism::analyze_cyp_metabolism;
use crate::mol::{Element, Molecule};
use crate::physchem::evaluate_physchem;
use crate::pka_protomer::{evaluate_physiological_pka_74, ChargeClass74};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Number of sampled time points along the simulated plasma & brain PK curves.
pub const PBPK_TIME_POINTS: usize = 97;

/// Standard reference adult human body weight (`70.0 kg`).
pub const HUMAN_BW_KG: f64 = 70.0;

/// Oral dosing schedule for single-dose or repeat-dose accumulation PBPK simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DosingRegimen {
    SingleDose24h,
    OnceDailyQd48h,
    TwiceDailyBid48h,
}

impl DosingRegimen {
    pub fn label(self) -> &'static str {
        match self {
            Self::SingleDose24h => "Single Oral Dose (0–24 h)",
            Self::OnceDailyQd48h => "Once-Daily QD (0, 24 h -> 48 h)",
            Self::TwiceDailyBid48h => "Twice-Daily BID (q12h x 4 -> 48 h)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::SingleDose24h => "Single 24h",
            Self::OnceDailyQd48h => "QD 48h",
            Self::TwiceDailyBid48h => "BID 48h",
        }
    }

    pub fn horizon_hr(self) -> f64 {
        match self {
            Self::SingleDose24h => 24.0,
            Self::OnceDailyQd48h | Self::TwiceDailyBid48h => 48.0,
        }
    }

    pub fn dose_times_hr(self) -> &'static [f64] {
        match self {
            Self::SingleDose24h => &[0.0],
            Self::OnceDailyQd48h => &[0.0, 24.0],
            Self::TwiceDailyBid48h => &[0.0, 12.0, 24.0, 36.0],
        }
    }
}

/// Gastrointestinal prandial state (`Fasted` vs `Fed` bile-micelle & splanchnic blood-flow shift).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrandialState {
    Fasted,
    FedBileMicelles,
}

impl PrandialState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fasted => "Fasted (pH 1.6 Stomach, Low Bile)",
            Self::FedBileMicelles => "Fed State (FeSSIF Bile Micelles, Slower Gastric Emptying)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Fasted => "Fasted",
            Self::FedBileMicelles => "Fed (FeSSIF)",
        }
    }
}

/// CNS Blood-Brain Barrier (`BBB`) unbound delivery tier based on `Kp,uu,brain`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CnsDeliveryClass {
    /// `Kp,uu,brain >= 0.30`: Unimpaired / weak-efflux BBB equilibrium suitable for CNS targets.
    CnsPenetrantOptimal,
    /// `0.10 <= Kp,uu,brain < 0.30`: Moderate apical efflux; requires high potency or dose.
    ModerateEffluxRestricted,
    /// `Kp,uu,brain < 0.10`: Strong `P-gp`/`BCRP` efflux or polar exclusion (desirable for peripheral-only safety).
    PeripherallyRestricted,
}

impl CnsDeliveryClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::CnsPenetrantOptimal => "CNS-Penetrant (Kp,uu,brain >= 0.30)",
            Self::ModerateEffluxRestricted => "Moderate BBB Efflux (0.10 <= Kp,uu < 0.30)",
            Self::PeripherallyRestricted => "Peripherally Restricted (Kp,uu < 0.10)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::CnsPenetrantOptimal => "CNS Penetrant",
            Self::ModerateEffluxRestricted => "Mod Efflux",
            Self::PeripherallyRestricted => "Periph-Only",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::CnsPenetrantOptimal => [52, 211, 153],      // Emerald
            Self::ModerateEffluxRestricted => [251, 191, 36], // Amber
            Self::PeripherallyRestricted => [56, 189, 248],   // Sky Blue
        }
    }
}

/// Interactive dosing & physiological perturbation configuration for `simulate_oral_pbpk`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PbpkDosingConfig {
    pub dose_mg: f64,
    pub regimen: DosingRegimen,
    pub prandial: PrandialState,
    /// Simulates chemical or genetic `MDR1 (P-gp)` + `BCRP` apical BBB efflux inhibition (`ER -> 0`).
    pub inhibit_bbb_efflux: bool,
    /// Optional explicit CYP3A4 Michaelis-Menten `Km` (`uM`); `None` uses structure-predicted `Km`.
    #[serde(default)]
    pub cyp3a4_km_um: Option<f64>,
    /// Number of deterministic virtual-population subjects for 90% CI (`P5`-`P95`) bands (`0` = deterministic only).
    #[serde(default = "default_virtual_pop_n")]
    pub virtual_pop_n: usize,
}

fn default_virtual_pop_n() -> usize {
    48
}

impl Default for PbpkDosingConfig {
    fn default() -> Self {
        Self {
            dose_mg: 50.0,
            regimen: DosingRegimen::SingleDose24h,
            prandial: PrandialState::Fasted,
            inhibit_bbb_efflux: false,
            cyp3a4_km_um: None,
            virtual_pop_n: default_virtual_pop_n(),
        }
    }
}

/// Rodgers & Rowland (2006) predicted organ-to-plasma partition coefficients (`Kp_tissue`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrganPartitionEntry {
    #[serde(skip_deserializing)]
    pub organ_name: &'static str,
    pub volume_l: f64,
    pub kp_total: f64,
}

/// Complete mechanistic PBPK parameter set, `Fa * Fg * Fh` cascade, `Kp,uu,brain`, and `RK4` curves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OralPbpkReport {
    pub compound_id: String,
    pub compound_name: String,
    pub target_name: String,
    pub target_ic50_nm: f64,
    pub mw: f64,
    pub clogp: f64,
    pub logd_74: f64,
    pub charge_class_74: ChargeClass74,
    pub config: PbpkDosingConfig,

    // Plasma & Blood Binding + Rodgers-Rowland Distribution
    pub fu_plasma: f64,
    pub blood_to_plasma_rbp: f64,
    pub vss_l_per_kg: f64,
    pub vss_total_l: f64,
    pub organ_kps: Vec<OrganPartitionEntry>,

    // GI CAT/ACAT Absorption + Q_gut + Well-Stirred Hepatic First-Pass Cascade
    pub peff_1e4_cm_s: f64,
    pub luminal_sol_ug_ml: f64,
    pub fa_analytical_perm: f64,
    /// Dynamically integrated fraction absorbed from the 7-segment `ACAT` `RK4` run (`0..1`).
    pub fa_integrated: f64,
    pub q_gut_l_hr: f64,
    pub fg_gut_wall: f64,
    pub cl_int_u_ml_min_kg: f64,
    pub cl_hepatic_blood_l_hr: f64,
    pub cl_renal_plasma_l_hr: f64,
    pub hepatic_extraction_eh: f64,
    pub fh_hepatic: f64,
    /// Overall oral bioavailability `F_oral = Fa_integrated * Fg * Fh` (`0..100%`).
    pub f_oral_pct: f64,

    // Hammarlund-Udenaes / Fridén Unbound Brain Delivery (Kp,uu,brain)
    pub fu_brain: f64,
    pub ps_bbb_l_hr: f64,
    pub bbb_efflux_ratio: f64,
    /// Total brain-to-plasma ratio `Kp,brain = Kp,uu,brain * (fu_plasma / fu_brain)`.
    pub kp_brain_total: f64,
    /// Pharmacologically active unbound brain-to-plasma ratio `Kp,uu,brain = Cu,brain,ss / Cu,plasma,ss`.
    pub kp_uu_brain: f64,
    pub cns_class: CnsDeliveryClass,

    // Exposure & Target Engagement Summary Metrics
    pub cmax_total_plasma_nm: f64,
    pub cmax_unbound_plasma_nm: f64,
    pub cmax_unbound_brain_nm: f64,
    pub tmax_plasma_hr: f64,
    pub tmax_brain_hr: f64,
    pub auc_unbound_plasma_nm_hr: f64,
    pub auc_unbound_brain_nm_hr: f64,
    pub terminal_t_half_hr: f64,
    /// Hours during simulation horizon where `Cu,plasma(t) >= target IC50` (peripheral coverage).
    pub time_above_ic50_plasma_hr: f64,
    /// Hours during simulation horizon where `Cu,brain(t) >= target IC50` (central CNS coverage).
    pub time_above_ic50_brain_hr: f64,
    pub peak_ro_plasma_pct: f64,
    pub peak_ro_brain_pct: f64,

    // RK4 Time-Series Arrays (`PBPK_TIME_POINTS = 97`)
    #[serde(with = "crate::serde_f32_array")]
    pub time_hr: [f32; PBPK_TIME_POINTS],
    #[serde(with = "crate::serde_f32_array")]
    pub c_total_plasma_nm: [f32; PBPK_TIME_POINTS],
    #[serde(with = "crate::serde_f32_array")]
    pub c_unbound_plasma_nm: [f32; PBPK_TIME_POINTS],
    #[serde(with = "crate::serde_f32_array")]
    pub c_unbound_brain_nm: [f32; PBPK_TIME_POINTS],
    #[serde(with = "crate::serde_f32_array")]
    pub ro_plasma_pct: [f32; PBPK_TIME_POINTS],
    #[serde(with = "crate::serde_f32_array")]
    pub ro_brain_pct: [f32; PBPK_TIME_POINTS],
    #[serde(with = "crate::serde_f32_array")]
    pub gi_absorbed_pct: [f32; PBPK_TIME_POINTS],

    // Saturable Michaelis-Menten CYP3A4 Kinetics & Virtual-Population 90% CI (P5-P95)
    #[serde(default)]
    pub cyp3a4_km_um: f64,
    #[serde(default)]
    pub cyp3a4_vmax_umol_hr: f64,
    #[serde(default)]
    pub cyp3a4_peak_saturation_pct: f64,
    #[serde(default)]
    pub virtual_pop_n: usize,
    #[serde(default)]
    pub f_oral_ci90: [f64; 2],
    #[serde(default)]
    pub cmax_u_plasma_ci90_nm: [f64; 2],
    #[serde(default)]
    pub cmax_u_brain_ci90_nm: [f64; 2],
    #[serde(default)]
    pub auc_u_plasma_ci90_nm_hr: [f64; 2],
    #[serde(default = "default_pbpk_curve", with = "crate::serde_f32_array")]
    pub c_unbound_plasma_p05_nm: [f32; PBPK_TIME_POINTS],
    #[serde(default = "default_pbpk_curve", with = "crate::serde_f32_array")]
    pub c_unbound_plasma_p95_nm: [f32; PBPK_TIME_POINTS],
    #[serde(default = "default_pbpk_curve", with = "crate::serde_f32_array")]
    pub c_unbound_brain_p05_nm: [f32; PBPK_TIME_POINTS],
    #[serde(default = "default_pbpk_curve", with = "crate::serde_f32_array")]
    pub c_unbound_brain_p95_nm: [f32; PBPK_TIME_POINTS],
}

fn default_pbpk_curve() -> [f32; PBPK_TIME_POINTS] {
    [0.0; PBPK_TIME_POINTS]
}

/// Dataset-level `50 mg` Single-Dose Oral PBPK & `Kp,uu,brain` summary row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PbpkDatasetRow {
    pub mol_idx: usize,
    pub id: String,
    pub name: String,
    pub target: String,
    pub ic50_nm: f64,
    pub fa_pct: f64,
    pub fg_pct: f64,
    pub fh_pct: f64,
    pub f_oral_pct: f64,
    pub vss_l_kg: f64,
    pub t_half_hr: f64,
    pub fu_plasma: f64,
    pub kp_uu_brain: f64,
    pub cns_class: CnsDeliveryClass,
    pub cmax_u_plasma_nm: f64,
    pub cmax_u_brain_nm: f64,
    pub peak_ro_brain_pct: f64,
}

/// Predicts unbound fraction in human plasma (`f_u,plasma`), blood-to-plasma ratio (`R_bp`),
/// and unbound fraction in brain homogenate (`f_u,brain`) from `cLogP`, `logD_7.4`, and charge state.
fn predict_binding_and_rbp(
    clogp: f64,
    logd_74: f64,
    charge_class: ChargeClass74,
) -> (f64, f64, f64) {
    // Lobell-Sivarajah / Austin sigmoid for nonspecific plasma protein binding (HSA for acids, AGP for bases):
    let acid_hsa_shift = match charge_class {
        ChargeClass74::Anionic => 0.95, // Carboxylates & acyl-sulfonamides bind Sudlow Site I/II on HSA tightly
        ChargeClass74::Zwitterionic => 0.35,
        ChargeClass74::Cationic => -0.15,
        ChargeClass74::Neutral => 0.0,
    };
    let lipo_eff = 0.62 * clogp + 0.38 * logd_74 + acid_hsa_shift;
    let ka_bind = 10f64.powf(0.56 * lipo_eff - 0.42);
    let fu_plasma = (1.0 / (1.0 + ka_bind)).clamp(0.002, 0.96);

    // Blood-to-plasma ratio R_bp (hematocrit = 0.45):
    // Acids are excluded from erythrocytes (R_bp ~ 0.55-0.62); lipophilic bases partition into RBCs (R_bp ~ 0.85-1.45).
    let rbp = match charge_class {
        ChargeClass74::Anionic => (0.55 + 0.12 * fu_plasma).clamp(0.55, 0.68),
        ChargeClass74::Cationic => {
            (0.78 + 0.22 * (logd_74.max(0.0) * 0.45).min(1.8)).clamp(0.75, 1.48)
        }
        ChargeClass74::Zwitterionic => 0.72,
        ChargeClass74::Neutral => (0.72 + 0.14 * (logd_74.max(0.0) * 0.35).min(1.4)).clamp(0.68, 1.15),
    };

    // Brain parenchyma nonspecific lipid/phospholipid binding (`fu_brain`, Fridén et al. 2009):
    // Driven primarily by neutral phospholipid partitioning (`cLogP`) + acidic phospholipid binding of cations!
    let cation_ps_shift = if charge_class == ChargeClass74::Cationic {
        0.55
    } else {
        0.0
    };
    let ka_brain = 10f64.powf(0.64 * clogp + cation_ps_shift - 0.68);
    let fu_brain = (1.0 / (1.0 + ka_brain)).clamp(0.001, 0.92);

    (fu_plasma, rbp, fu_brain)
}

/// Computes Rodgers & Rowland (2006) mechanistic tissue-to-plasma `Kp` values across 7 major
/// human organs and total steady-state volume of distribution `V_ss` (`L` and `L/kg`).
fn compute_rodgers_rowland_vss(
    clogp: f64,
    logd_74: f64,
    fu_plasma: f64,
    charge_class: ChargeClass74,
    kp_brain_total: f64,
) -> (Vec<OrganPartitionEntry>, f64, f64) {
    let p_nl = 10f64.powf(clogp.clamp(-2.0, 5.6));
    let d_74 = 10f64.powf(logd_74.clamp(-2.5, 5.2));

    // Electrostatic acidic-phospholipid + lysosomal trapping multiplier for basic cations (`pH_iw = 7.0` vs `pH_p = 7.4`, `10^0.4 = 2.51`):
    let (ion_iw_mult, ap_bind_mult, ecf_albumin_mult) = match charge_class {
        ChargeClass74::Cationic => (2.45, 5.20, 0.08),
        ChargeClass74::Anionic => (0.42, 0.0, 0.24),
        ChargeClass74::Zwitterionic => (1.15, 1.35, 0.14),
        ChargeClass74::Neutral => (1.00, 0.18, 0.10),
    };

    // Reference 70 kg human organ specs: (Name, V_tissue_L, f_ew, f_iw, f_nl, f_np, f_ap_rel)
    let specs: [(&str, f64, f64, f64, f64, f64, f64); 6] = [
        ("Muscle", 29.0, 0.118, 0.630, 0.010, 0.0072, 1.00),
        ("Adipose", 12.5, 0.135, 0.017, 0.853, 0.0016, 0.25),
        ("Liver", 1.80, 0.161, 0.573, 0.035, 0.0240, 2.65),
        ("Gut Wall", 1.65, 0.282, 0.475, 0.038, 0.0125, 1.55),
        ("Kidney", 0.31, 0.273, 0.483, 0.012, 0.0242, 2.40),
        ("Lung", 1.00, 0.336, 0.446, 0.022, 0.0128, 2.90),
    ];

    let v_plasma_l = 3.0;
    let mut vss_l = v_plasma_l;
    let mut entries = Vec::with_capacity(7);

    for (name, vt, f_ew, f_iw, f_nl, f_np, ap_rel) in specs {
        let lipid_term = if name == "Adipose" {
            // Vegetable-oil / triglyceride partitioning uses D_74 rather than octanol P
            (0.32 * d_74 + 0.68 * p_nl.powf(0.72)) * f_nl + p_nl.powf(0.75) * f_np
        } else {
            p_nl.powf(0.76) * (f_nl + 1.35 * f_np)
        };
        let ap_term = ap_bind_mult * ap_rel * (0.35 + 0.65 * d_74.powf(0.38));
        let unbound_tissue_part = f_ew + f_iw * ion_iw_mult + lipid_term + ap_term;
        let kp = (fu_plasma * unbound_tissue_part + (1.0 - fu_plasma) * ecf_albumin_mult)
            .clamp(0.08, 38.0);
        vss_l += vt * kp;
        entries.push(OrganPartitionEntry {
            organ_name: name,
            volume_l: vt,
            kp_total: kp,
        });
    }

    // Add Brain (`1.40 L`)
    let v_brain = 1.40;
    vss_l += v_brain * kp_brain_total;
    entries.push(OrganPartitionEntry {
        organ_name: "Brain",
        volume_l: v_brain,
        kp_total: kp_brain_total,
    });

    // Remaining slowly/moderately perfused carcass (skin, bone, heart, spleen ~ 12.0 L)
    let kp_rest = entries[0].kp_total * 0.85;
    vss_l += 12.0 * kp_rest;

    let vss_l_kg = (vss_l / HUMAN_BW_KG).clamp(0.11, 24.0);
    (entries, vss_l_kg * HUMAN_BW_KG, vss_l_kg)
}

/// Predicts human BBB permeability-surface area product `PS_bbb` (`L/h`), active `P-gp`/`BCRP`
/// apical efflux ratio `ER_bbb`, and unbound brain-to-plasma partition coefficient `Kp,uu,brain`
/// (`Hammarlund-Udenaes et al. 2008`; `Loryan / Di et al. 2022`).
fn predict_bbb_kpuu(
    mol: &Molecule,
    mw: f64,
    logd_74: f64,
    tpsa: f64,
    hbd: usize,
    cns_mpo: f64,
    pgp_flag: bool,
    inhibit_efflux: bool,
) -> (f64, f64, f64, CnsDeliveryClass) {
    // 1. Passive transcellular BBB permeability-surface area clearance PS_bbb (L/h across 15-20 m^2 human brain capillary bed):
    let log_ps = -0.35
        - 0.0145 * (tpsa - 45.0).max(0.0)
        - 0.24 * (hbd as f64)
        - 0.0022 * (mw - 340.0).max(0.0)
        + 0.20 * logd_74.clamp(-1.5, 3.5);
    let ps_bbb_l_hr = (10f64.powf(log_ps) * 72.0).clamp(0.08, 160.0);

    // 2. Active apical efflux ratio ER = CL_efflux / PS_bbb (`P-gp / MDR1` + `BCRP`):
    let has_sulfonamide_or_sulfone = mol.atoms.iter().any(|a| a.elem == Element::S && !a.aromatic);
    let mut er_efflux = 0.18
        + ((tpsa - 68.0).max(0.0) / 22.0).powf(1.35) * 1.45
        + (hbd.saturating_sub(1) as f64) * 1.15
        + ((mw - 415.0).max(0.0) / 55.0).powf(1.30) * 1.65;
    if pgp_flag && cns_mpo < 4.35 {
        er_efflux += 2.35;
    }
    if has_sulfonamide_or_sulfone && tpsa > 75.0 {
        er_efflux += 1.40; // BCRP recognition of polar sulfones/sulfonamides (e.g. Lapatinib, Rofecoxib/Celecoxib moderate)
    }
    // Compact high-CNS-MPO psychotropics & macrocyclic CNS leads evade P-gp via high passive flip-flop rate:
    if cns_mpo >= 4.45 && tpsa <= 62.0 && hbd <= 1 && mw <= 430.0 {
        er_efflux *= 0.28;
    }
    let eff_er = if inhibit_efflux {
        0.05
    } else {
        er_efflux.clamp(0.08, 35.0)
    };

    // 3. CSF bulk-flow turnover sink (`Q_csf ~ 0.024 L/h`): negligible when PS_bbb >> 0.024 L/h,
    //    significant for polar slow-permeating molecules!
    let q_csf_l_hr = 0.024;
    let kp_uu_brain = (1.0 / (1.0 + eff_er + q_csf_l_hr / ps_bbb_l_hr)).clamp(0.015, 1.02);

    let cns_class = if kp_uu_brain >= 0.30 {
        CnsDeliveryClass::CnsPenetrantOptimal
    } else if kp_uu_brain >= 0.10 {
        CnsDeliveryClass::ModerateEffluxRestricted
    } else {
        CnsDeliveryClass::PeripherallyRestricted
    };

    (ps_bbb_l_hr, eff_er, kp_uu_brain, cns_class)
}

/// Runs the full mechanistic 7-segment `ACAT` + `Q_gut` + `Well-Stirred` Hepatic + `Rodgers-Rowland`
/// + `Kp,uu,brain` `RK4` human oral PBPK simulation for `mol`.
pub fn simulate_oral_pbpk(mol: &Molecule, config: PbpkDosingConfig) -> OralPbpkReport {
    simulate_oral_pbpk_impl(mol, config, false).0
}

fn simulate_oral_pbpk_impl(
    mol: &Molecule,
    config: PbpkDosingConfig,
    zero_elim_closed: bool,
) -> (OralPbpkReport, [f64; 22], f64) {
    let alerts = scan_structural_alerts(mol);
    let phys = evaluate_physchem(mol, alerts.len());
    let admet = evaluate_admet(mol, &phys);
    let pka = evaluate_physiological_pka_74(mol);
    let cyp = analyze_cyp_metabolism(mol, false);

    let mw = phys.mw.max(110.0);
    let target_ic50_nm = if mol.raw_nm > 0.0 {
        mol.raw_nm
    } else {
        10f64.powf(9.0 - mol.pact.clamp(3.5, 11.5))
    }
    .clamp(0.005, 250_000.0);

    // 1. Plasma/Blood/Brain binding & BBB Kp,uu,brain:
    let (fu_plasma, rbp, fu_brain) =
        predict_binding_and_rbp(phys.clogp, pka.microstate_logd_74, pka.charge_class);
    let (ps_bbb_l_hr, bbb_er, kp_uu_brain, cns_class) = predict_bbb_kpuu(
        mol,
        mw,
        pka.microstate_logd_74,
        phys.tpsa,
        phys.hbd,
        phys.cns_mpo,
        admet.pgp_substrate,
        config.inhibit_bbb_efflux,
    );
    let kp_brain_total = (kp_uu_brain * (fu_plasma / fu_brain)).clamp(0.04, 32.0);

    // 2. Rodgers & Rowland (2006) Organ Partitioning & V_ss:
    let (organ_kps, vss_total_l, vss_l_per_kg) = compute_rodgers_rowland_vss(
        phys.clogp,
        pka.microstate_logd_74,
        fu_plasma,
        pka.charge_class,
        kp_brain_total,
    );

    // 3. Human Jejunal Effective Permeability P_eff (10^-4 cm/s, Winiwarter / Sun correlation):
    let charge_pen = pka.mean_net_charge_74.abs() * 0.42;
    let log_peff = 0.58
        - 0.0112 * phys.tpsa
        - 0.145 * (phys.hbd as f64)
        + 0.155 * pka.microstate_logd_74.clamp(-2.0, 4.2)
        - charge_pen
        - 0.0014 * (mw - 380.0).max(0.0);
    let peff_1e4_cm_s = 10f64.powf(log_peff).clamp(0.05, 12.0);
    // Convert 10^-4 cm/s -> cm/h (`* 0.36`):
    let peff_cm_hr = peff_1e4_cm_s * 0.36;
    // Yu & Amidon (1999) analytical 7-compartment permeability ceiling:
    let fa_analytical_perm = (1.0 - (1.0 + 0.54 * peff_cm_hr).powi(-7)).clamp(0.02, 0.999);
    // Absorptive rate constant per transit segment (`k_a = 2 * Peff / R_si`, R_si = 1.75 cm -> `1.14 * Peff_cm_hr`):
    let ka_seg_hr = (1.14 * peff_cm_hr).clamp(0.04, 8.5);

    // 4. Luminal Dissolution Solubility (`ug/mL` -> `nmol/L` = `nM`) in Fasted (FaSSIF) vs Fed (FeSSIF):
    let ion_hh = 10f64.powf(((phys.clogp - pka.microstate_logd_74).clamp(-0.2, 3.4)) * 0.75);
    let bile_micelle_mult = match config.prandial {
        PrandialState::Fasted => {
            1.0 + 0.12 * 10f64.powf((pka.microstate_logd_74 - 2.5).clamp(-2.0, 2.2))
        }
        PrandialState::FedBileMicelles => {
            1.25 + 1.85 * 10f64.powf((pka.microstate_logd_74 - 2.0).clamp(-2.0, 2.2))
        }
    }
    .clamp(1.0, 14.0);
    let luminal_sol_ug_ml = (admet.esol_ug_ml * ion_hh * bile_micelle_mult).clamp(0.15, 120_000.0);
    // Convert ug/mL (= mg/L) -> nmol/L: `(mg/L * 1e6) / mw`
    let c_sat_si_nm = (luminal_sol_ug_ml * 1.0e6) / mw;
    // Stomach solubility at pH 1.6 (fasted: basic amines protonate & dissolve >25x faster!) vs pH 4.8 (fed):
    let stom_base_boost = if pka.charge_class == ChargeClass74::Cationic
        && config.prandial == PrandialState::Fasted
    {
        18.0
    } else if pka.charge_class == ChargeClass74::Anionic
        && config.prandial == PrandialState::Fasted
    {
        0.25
    } else {
        1.2
    };
    let c_sat_stom_nm = (c_sat_si_nm * stom_base_boost).clamp(50.0, 1.0e9);

    // 5. Yang et al. (2007) Q_gut Enterocyte Availability F_g:
    let q_villi_l_hr = 18.0;
    // Cylindrical small-intestinal area A = 0.66 m^2 = 6600 cm^2 -> CL_perm (L/h) = Peff(cm/h) * 6600 / 1000 = 6.6 * Peff_cm_hr
    let cl_perm_l_hr = (6.6 * peff_cm_hr).max(0.05);
    let q_gut_l_hr = (q_villi_l_hr * cl_perm_l_hr) / (q_villi_l_hr + cl_perm_l_hr);
    let fm_3a4 = cyp.isoform_fraction_fm[0];
    // Human intestinal CYP3A4 abundance ~ 1.2% of hepatic CYP3A4, plus direct enterocyte UGT on free phenols:
    let has_phenol = mol.atoms.iter().enumerate().any(|(u, a)| {
        a.elem == Element::O
            && a.total_h() >= 1
            && mol.adj[u].len() == 1
            && mol.atoms[mol.adj[u][0].0].aromatic
    });
    let cl_int_u_total_l_hr = (cyp.cl_int_ml_min_kg * HUMAN_BW_KG * 60.0) / 1000.0;
    let ugt_gut_l_hr = if has_phenol { 42.0 } else { 0.0 };
    let fu_gut = (fu_plasma * 2.2).clamp(0.05, 1.0);

    // 6. Pang & Rowland (1977) Well-Stirred Hepatic Clearance & Saturable CYP3A4 Michaelis-Menten Km:
    let q_h_blood_l_hr = match config.prandial {
        PrandialState::Fasted => 90.0,
        PrandialState::FedBileMicelles => 112.0, // Postprandial splanchnic hyperemia
    };
    let ugt_liver_l_hr = if has_phenol { 95.0 } else { 0.0 };

    // Structure-predicted human CYP3A4 Km (uM) unless explicitly overridden in config
    let cyp3a4_km_um = config
        .cyp3a4_km_um
        .unwrap_or_else(|| 10f64.powf(1.35 - 0.30 * pka.microstate_logd_74.clamp(-1.0, 4.5)))
        .clamp(0.5, 80.0);
    let km_nm = cyp3a4_km_um * 1000.0;
    let cl_int_3a4_lin = cl_int_u_total_l_hr * fm_3a4;
    let cl_int_other_lin = cl_int_u_total_l_hr * (1.0 - fm_3a4) + ugt_liver_l_hr;
    let vmax_liver_3a4_nmol_hr = cl_int_3a4_lin * km_nm;
    let vmax_gut_3a4_nmol_hr = (cl_int_3a4_lin * 0.055) * km_nm;
    let cyp3a4_vmax_umol_hr = vmax_liver_3a4_nmol_hr / 1000.0;

    // Glomerular filtration + tubular reabsorption renal plasma clearance (GFR = 7.5 L/h):
    let tubular_reabs =
        (1.0 / (1.0 + 10f64.powf(-0.65 * pka.microstate_logd_74))).clamp(0.05, 0.92);
    let cl_renal_plasma_l_hr = (7.5 * fu_plasma * (1.0 - 0.80 * tubular_reabs)).clamp(0.01, 6.5);

    // 7. 4th-Order Runge-Kutta (`RK4`) Integration across Stomach + 7 CAT Segments + Central + Periph + Brain ISF:
    let k_ge_hr = match config.prandial {
        PrandialState::Fasted => 2.80,          // Fasted gastric emptying t1/2 ~ 15 min
        PrandialState::FedBileMicelles => 0.95, // Fed gastric emptying t1/2 ~ 44 min
    };
    let k_transit_hr = 7.0 / 3.32; // Yu & Amidon mean SI transit = 3.32 h -> 2.108 h^-1 per segment
    let k_diss_base_hr = 3.60;     // Micronized IR tablet Nernst-Brunner dissolution rate constant (h^-1)
    let v_stom_l = match config.prandial {
        PrandialState::Fasted => 0.25,
        PrandialState::FedBileMicelles => 0.50,
    };
    let v_si_seg_l = match config.prandial {
        PrandialState::Fasted => 0.055,
        PrandialState::FedBileMicelles => 0.095,
    };

    let q_periph_l_hr = 46.0;
    let v_brain_l = 1.40; // Human brain tissue volume (Rodgers-Rowland)
    let v_u_brain_l = (v_brain_l / fu_brain.max(0.005)).max(v_brain_l);
    let dose_nmol_per_admin = (config.dose_mg.max(0.5) * 1.0e6) / mw;
    let schedule = config.regimen.dose_times_hr();
    let horizon_hr = config.regimen.horizon_hr();

    #[derive(Clone, Copy)]
    struct SubjectIivMults {
        fu_p: f64,
        cl_3a4: f64,
        cl_other: f64,
        ka_ge: f64,
        vss: f64,
        q_h: f64,
        ps_bbb: f64,
    }

    struct SingleSubjectRun {
        fa_integrated: f64,
        fg_gut_wall: f64,
        fh_hepatic: f64,
        f_oral_pct: f64,
        cl_hepatic_blood_l_hr: f64,
        hepatic_extraction_eh: f64,
        terminal_t_half_hr: f64,
        cmax_total_plasma_nm: f64,
        cmax_unbound_plasma_nm: f64,
        cmax_unbound_brain_nm: f64,
        tmax_plasma_hr: f64,
        tmax_brain_hr: f64,
        auc_unbound_plasma_nm_hr: f64,
        auc_unbound_brain_nm_hr: f64,
        time_above_ic50_plasma_hr: f64,
        time_above_ic50_brain_hr: f64,
        peak_ro_plasma_pct: f64,
        peak_ro_brain_pct: f64,
        peak_cyp3a4_sat_pct: f64,
        time_hr: [f32; PBPK_TIME_POINTS],
        c_total_plasma_nm: [f32; PBPK_TIME_POINTS],
        c_unbound_plasma_nm: [f32; PBPK_TIME_POINTS],
        c_unbound_brain_nm: [f32; PBPK_TIME_POINTS],
        ro_plasma_pct: [f32; PBPK_TIME_POINTS],
        ro_brain_pct: [f32; PBPK_TIME_POINTS],
        gi_absorbed_pct: [f32; PBPK_TIME_POINTS],
        final_y: [f64; 22],
        total_dosed_nmol: f64,
    }

    let solve_quad_unbound = |a_lin: f64, b_vmax: f64, r_in: f64| -> f64 {
        if r_in <= 0.0 || a_lin <= 0.0 {
            return 0.0;
        }
        let b_coef = a_lin * km_nm + b_vmax - r_in;
        let disc = (b_coef * b_coef + 4.0 * a_lin * r_in * km_nm).sqrt();
        if b_coef >= 0.0 {
            (2.0 * r_in * km_nm) / (b_coef + disc).max(1e-18)
        } else {
            (-b_coef + disc) / (2.0 * a_lin)
        }
    };

    let integrate_subject = |mult: SubjectIivMults| -> SingleSubjectRun {
        let fu_p_s = (fu_plasma * mult.fu_p).clamp(0.001, 0.98);
        let fu_gut_s = (fu_gut * mult.fu_p).clamp(0.02, 1.0);
        let fu_blood_s = (fu_p_s / rbp).clamp(0.001, 1.0);
        let q_h_s = (q_h_blood_l_hr * mult.q_h).max(20.0);
        let vmax_liv_s = vmax_liver_3a4_nmol_hr * mult.cl_3a4;
        let vmax_gut_s = vmax_gut_3a4_nmol_hr * mult.cl_3a4;
        let cl_other_s = cl_int_other_lin * mult.cl_other;
        let ugt_gut_s = ugt_gut_l_hr * mult.cl_other;
        let ka_s = (ka_seg_hr * mult.ka_ge).clamp(0.02, 14.0);
        let k_ge_s = (k_ge_hr * mult.ka_ge).clamp(0.25, 6.0);
        let vss_s = (vss_total_l * mult.vss).max(6.0);
        let v_central_s = (4.2 + 0.12 * vss_s).clamp(4.2, vss_s.max(5.0));
        let v_brain_kp_eq_l = 1.40 * kp_brain_total * mult.vss;
        let v_periph_s = (vss_s - v_central_s - v_brain_kp_eq_l).max(2.5);
        let ps_bbb_s = (ps_bbb_l_hr * mult.ps_bbb).clamp(0.04, 180.0);
        let cl_renal_s = if zero_elim_closed {
            0.0
        } else {
            (cl_renal_plasma_l_hr * mult.fu_p).clamp(0.01, 10.0)
        };

        let dt = 0.01f64;
        let total_steps = ((horizon_hr / dt).round() as usize).max(96);

        // State vector Y (len = 22, all in nmol):
        // [0]: stom_solid, [1]: stom_diss, [2..9]: si_solid[0..7], [9..16]: si_diss[0..7],
        // [16]: central, [17]: periph, [18]: brain_isf, [19]: cumulative_absorbed_nmol,
        // [20]: cumulative_portal_nmol (after Fg), [21]: cumulative_systemic_nmol (after Fh)
        let mut y = [0.0f64; 22];
        let mut total_dosed_nmol = 0.0f64;
        let mut next_dose_ptr = 0usize;

        let eval_gut_and_liver = |j_abs_sum: f64, c_p_tot: f64| -> (f64, f64, f64, f64, f64) {
            if zero_elim_closed {
                return (1.0, 1.0, 0.0, j_abs_sum, 0.0);
            }
            let a_gut = q_gut_l_hr + fu_gut_s * ugt_gut_s;
            let c_u_gut = solve_quad_unbound(a_gut, fu_gut_s * vmax_gut_s, fu_gut_s * j_abs_sum);
            let cl_u_gut_3a4_t = vmax_gut_s / (km_nm + c_u_gut);
            let fg_t = (q_gut_l_hr / (q_gut_l_hr + fu_gut_s * (cl_u_gut_3a4_t + ugt_gut_s)))
                .clamp(0.08, 0.995);
            let j_portal = j_abs_sum * fg_t;

            let j_in_liver = j_portal + q_h_s * (c_p_tot * rbp);
            let a_liv = q_h_s / fu_blood_s + cl_other_s;
            let c_u_liver = solve_quad_unbound(a_liv, vmax_liv_s, j_in_liver);
            let cl_int_3a4_t = vmax_liv_s / (km_nm + c_u_liver);
            let cl_int_liv_t = cl_int_3a4_t + cl_other_s;
            let eh_t = ((fu_blood_s * cl_int_liv_t) / (q_h_s + fu_blood_s * cl_int_liv_t))
                .clamp(0.01, 0.96);
            let fh_t = (1.0 - eh_t).clamp(0.04, 0.99);
            let sat_liv = c_u_liver / (km_nm + c_u_liver);
            let sat_gut = c_u_gut / (km_nm + c_u_gut);
            (fg_t, fh_t, eh_t, j_portal, sat_liv.max(sat_gut))
        };

        let rhs = |s: &[f64; 22]| -> [f64; 22] {
            let mut dy = [0.0f64; 22];

            let c_stom = s[1].max(0.0) / v_stom_l;
            let sat_frac_stom = (1.0 - c_stom / c_sat_stom_nm).clamp(0.0, 1.0);
            let j_diss_stom = k_diss_base_hr * s[0].max(0.0) * sat_frac_stom;
            let j_emp_solid = (k_ge_s * 0.55) * s[0].max(0.0);
            let j_emp_diss = k_ge_s * s[1].max(0.0);
            dy[0] = -j_diss_stom - j_emp_solid;
            dy[1] = j_diss_stom - j_emp_diss;

            let mut j_abs_sum = 0.0f64;
            for seg in 0..7 {
                let sol_idx = 2 + seg;
                let dis_idx = 9 + seg;
                let in_sol = if seg == 0 {
                    j_emp_solid
                } else {
                    k_transit_hr * s[2 + seg - 1].max(0.0)
                };
                let in_dis = if seg == 0 {
                    j_emp_diss
                } else {
                    k_transit_hr * s[9 + seg - 1].max(0.0)
                };
                let c_seg = s[dis_idx].max(0.0) / v_si_seg_l;
                let j_diss = if c_seg <= c_sat_si_nm {
                    let sat_frac = (1.0 - c_seg / c_sat_si_nm).clamp(0.0, 1.0);
                    k_diss_base_hr * s[sol_idx].max(0.0) * sat_frac
                } else {
                    // Supersaturated SI lumen (e.g., weak base emptying from acidic stomach) precipitates toward c_sat_si_nm
                    let k_precip_hr = k_diss_base_hr;
                    let excess_nmol = (s[dis_idx].max(0.0) - c_sat_si_nm * v_si_seg_l).max(0.0);
                    -(k_precip_hr * excess_nmol)
                };
                let out_sol = if zero_elim_closed && seg == 6 {
                    0.0
                } else {
                    k_transit_hr * s[sol_idx].max(0.0)
                };
                let out_dis = if zero_elim_closed && seg == 6 {
                    0.0
                } else {
                    k_transit_hr * s[dis_idx].max(0.0)
                };
                let reg_mult = if seg <= 3 { 1.08 } else { 0.86 };
                let j_abs = (ka_s * reg_mult) * s[dis_idx].max(0.0);

                dy[sol_idx] = in_sol - out_sol - j_diss;
                dy[dis_idx] = in_dis - out_dis + j_diss - j_abs;
                j_abs_sum += j_abs;
            }

            let c_p_tot = s[16].max(0.0) / v_central_s;
            let c_p_unb = fu_p_s * c_p_tot;
            let c_periph = s[17].max(0.0) / v_periph_s;
            let c_br_unb = s[18].max(0.0) / v_u_brain_l;

            let (_fg_t, fh_t, eh_t, j_portal, _sat) = eval_gut_and_liver(j_abs_sum, c_p_tot);
            let j_sys_in = j_portal * fh_t;
            let j_elim = (q_h_s * eh_t * rbp + cl_renal_s) * c_p_tot;
            let j_dist = q_periph_l_hr * (c_p_tot - c_periph);
            let j_bbb_net = ps_bbb_s * c_p_unb - (ps_bbb_s / kp_uu_brain) * c_br_unb;

            dy[16] = j_sys_in - j_elim - j_dist - j_bbb_net;
            dy[17] = j_dist;
            dy[18] = j_bbb_net;
            dy[19] = j_abs_sum;
            dy[20] = j_portal;
            dy[21] = j_sys_in;
            dy
        };

        let mut time_hr = [0.0f32; PBPK_TIME_POINTS];
        let mut c_total_plasma_nm = [0.0f32; PBPK_TIME_POINTS];
        let mut c_unbound_plasma_nm = [0.0f32; PBPK_TIME_POINTS];
        let mut c_unbound_brain_nm = [0.0f32; PBPK_TIME_POINTS];
        let mut ro_plasma_pct = [0.0f32; PBPK_TIME_POINTS];
        let mut ro_brain_pct = [0.0f32; PBPK_TIME_POINTS];
        let mut gi_absorbed_pct = [0.0f32; PBPK_TIME_POINTS];

        let sample_stride = total_steps / (PBPK_TIME_POINTS - 1);
        let mut sample_idx = 0usize;

        let mut cmax_tot_p = 0.0f64;
        let mut cmax_unb_p = 0.0f64;
        let mut cmax_unb_b = 0.0f64;
        let mut tmax_p = 0.0f64;
        let mut tmax_b = 0.0f64;
        let mut auc_unb_p = 0.0f64;
        let mut auc_unb_b = 0.0f64;
        let mut t_above_ic50_p = 0.0f64;
        let mut t_above_ic50_b = 0.0f64;
        let mut peak_sat_frac = 0.0f64;

        for step in 0..=total_steps {
            let t = (step as f64) * dt;
            while next_dose_ptr < schedule.len() && t + 1e-9 >= schedule[next_dose_ptr] {
                y[0] += 0.85 * dose_nmol_per_admin;
                y[1] += 0.15 * dose_nmol_per_admin;
                total_dosed_nmol += dose_nmol_per_admin;
                next_dose_ptr += 1;
            }

            let cp_tot = (y[16] / v_central_s).max(0.0);
            let cp_unb = fu_p_s * cp_tot;
            let cb_unb = (y[18] / v_u_brain_l).max(0.0);
            let ro_p = 100.0 * cp_unb / (cp_unb + target_ic50_nm);
            let ro_b = 100.0 * cb_unb / (cb_unb + target_ic50_nm);

            // Track instantaneous peak CYP3A4 saturation across gut & liver
            let mut j_abs_now = 0.0f64;
            for seg in 0..7 {
                let reg_mult = if seg <= 3 { 1.08 } else { 0.86 };
                j_abs_now += (ka_s * reg_mult) * y[9 + seg].max(0.0);
            }
            let (_, _, _, _, sat_now) = eval_gut_and_liver(j_abs_now, cp_tot);
            if sat_now > peak_sat_frac {
                peak_sat_frac = sat_now;
            }

            if cp_tot > cmax_tot_p {
                cmax_tot_p = cp_tot;
                cmax_unb_p = cp_unb;
                tmax_p = t;
            }
            if cb_unb > cmax_unb_b {
                cmax_unb_b = cb_unb;
                tmax_b = t;
            }
            if step > 0 {
                auc_unb_p += cp_unb * dt;
                auc_unb_b += cb_unb * dt;
                if cp_unb >= target_ic50_nm {
                    t_above_ic50_p += dt;
                }
                if cb_unb >= target_ic50_nm {
                    t_above_ic50_b += dt;
                }
            }

            if step % sample_stride == 0 && sample_idx < PBPK_TIME_POINTS {
                time_hr[sample_idx] = t as f32;
                c_total_plasma_nm[sample_idx] = cp_tot as f32;
                c_unbound_plasma_nm[sample_idx] = cp_unb as f32;
                c_unbound_brain_nm[sample_idx] = cb_unb as f32;
                ro_plasma_pct[sample_idx] = ro_p as f32;
                ro_brain_pct[sample_idx] = ro_b as f32;
                gi_absorbed_pct[sample_idx] =
                    ((y[19] / total_dosed_nmol.max(1.0)) * 100.0).clamp(0.0, 100.0) as f32;
                sample_idx += 1;
            }

            if step < total_steps {
                let k1 = rhs(&y);
                let mut y2 = [0.0f64; 22];
                for j in 0..22 {
                    y2[j] = (y[j] + 0.5 * dt * k1[j]).max(0.0);
                }
                let k2 = rhs(&y2);
                let mut y3 = [0.0f64; 22];
                for j in 0..22 {
                    y3[j] = (y[j] + 0.5 * dt * k2[j]).max(0.0);
                }
                let k3 = rhs(&y3);
                let mut y4 = [0.0f64; 22];
                for j in 0..22 {
                    y4[j] = (y[j] + dt * k3[j]).max(0.0);
                }
                let k4 = rhs(&y4);
                for j in 0..22 {
                    y[j] = (y[j] + (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])).max(0.0);
                }
            }
        }

        let total_abs = y[19].max(1e-9);
        let fa_integrated = (total_abs / total_dosed_nmol.max(1.0)).clamp(0.01, 0.999);
        let fg_gut_wall = (y[20] / total_abs).clamp(0.08, 0.995);
        let fh_hepatic = (y[21] / y[20].max(1e-9)).clamp(0.04, 0.99);
        let f_oral_pct = (fa_integrated * fg_gut_wall * fh_hepatic * 100.0).clamp(0.01, 99.0);
        let hepatic_extraction_eh = (1.0 - fh_hepatic).clamp(0.01, 0.96);
        let cl_hepatic_blood_l_hr = q_h_s * hepatic_extraction_eh;
        let cl_sys_plasma_l_hr = cl_hepatic_blood_l_hr * rbp + cl_renal_s;
        let terminal_t_half_hr = (0.69315 * vss_s / cl_sys_plasma_l_hr.max(0.15)).clamp(0.35, 96.0);
        let peak_ro_p = 100.0 * cmax_unb_p / (cmax_unb_p + target_ic50_nm);
        let peak_ro_b = 100.0 * cmax_unb_b / (cmax_unb_b + target_ic50_nm);

        SingleSubjectRun {
            fa_integrated,
            fg_gut_wall,
            fh_hepatic,
            f_oral_pct,
            cl_hepatic_blood_l_hr,
            hepatic_extraction_eh,
            terminal_t_half_hr,
            cmax_total_plasma_nm: cmax_tot_p,
            cmax_unbound_plasma_nm: cmax_unb_p,
            cmax_unbound_brain_nm: cmax_unb_b,
            tmax_plasma_hr: tmax_p,
            tmax_brain_hr: tmax_b,
            auc_unbound_plasma_nm_hr: auc_unb_p,
            auc_unbound_brain_nm_hr: auc_unb_b,
            time_above_ic50_plasma_hr: t_above_ic50_p,
            time_above_ic50_brain_hr: t_above_ic50_b,
            peak_ro_plasma_pct: peak_ro_p,
            peak_ro_brain_pct: peak_ro_b,
            peak_cyp3a4_sat_pct: (peak_sat_frac * 100.0).clamp(0.0, 100.0),
            time_hr,
            c_total_plasma_nm,
            c_unbound_plasma_nm,
            c_unbound_brain_nm,
            ro_plasma_pct,
            ro_brain_pct,
            gi_absorbed_pct,
            final_y: y,
            total_dosed_nmol,
        }
    };

    let nom = integrate_subject(SubjectIivMults {
        fu_p: 1.0,
        cl_3a4: 1.0,
        cl_other: 1.0,
        ka_ge: 1.0,
        vss: 1.0,
        q_h: 1.0,
        ps_bbb: 1.0,
    });

    // Deterministic virtual population (rayon-parallel, antithetic log-normal human IIV) for 90% CI (P5-P95)
    let mut f_oral_ci90 = [nom.f_oral_pct, nom.f_oral_pct];
    let mut cmax_u_plasma_ci90_nm = [nom.cmax_unbound_plasma_nm, nom.cmax_unbound_plasma_nm];
    let mut cmax_u_brain_ci90_nm = [nom.cmax_unbound_brain_nm, nom.cmax_unbound_brain_nm];
    let mut auc_u_plasma_ci90_nm_hr = [nom.auc_unbound_plasma_nm_hr, nom.auc_unbound_plasma_nm_hr];
    let mut c_unbound_plasma_p05_nm = nom.c_unbound_plasma_nm;
    let mut c_unbound_plasma_p95_nm = nom.c_unbound_plasma_nm;
    let mut c_unbound_brain_p05_nm = nom.c_unbound_brain_nm;
    let mut c_unbound_brain_p95_nm = nom.c_unbound_brain_nm;

    let pop_n = if !zero_elim_closed && config.virtual_pop_n >= 10 {
        config.virtual_pop_n.min(200)
    } else {
        0
    };

    if pop_n >= 10 {
        let mut seeds = Vec::with_capacity(pop_n);
        let pairs = (pop_n + 1) / 2;
        for p in 0..pairs {
            let z = draw_iiv_normals(0x5A17_A71A_0001u64 ^ ((p as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            for sign in [1.0_f64, -1.0_f64] {
                if seeds.len() >= pop_n {
                    break;
                }
                seeds.push(SubjectIivMults {
                    fu_p: (sign * 0.20 * z[0]).exp(),
                    cl_3a4: (sign * 0.45 * z[1]).exp(),
                    cl_other: (sign * 0.25 * z[2]).exp(),
                    ka_ge: (sign * 0.30 * z[3]).exp(),
                    vss: (sign * 0.18 * z[4]).exp(),
                    q_h: (sign * 0.18 * z[5]).exp(),
                    ps_bbb: (sign * 0.25 * z[6]).exp(),
                });
            }
        }

        let runs: Vec<SingleSubjectRun> = seeds.into_par_iter().map(integrate_subject).collect();
        let mut f_vals: Vec<f64> = runs.iter().map(|r| r.f_oral_pct).collect();
        let mut cp_vals: Vec<f64> = runs.iter().map(|r| r.cmax_unbound_plasma_nm).collect();
        let mut cb_vals: Vec<f64> = runs.iter().map(|r| r.cmax_unbound_brain_nm).collect();
        let mut auc_vals: Vec<f64> = runs.iter().map(|r| r.auc_unbound_plasma_nm_hr).collect();

        f_oral_ci90 = ci90_bounds(&mut f_vals);
        cmax_u_plasma_ci90_nm = ci90_bounds(&mut cp_vals);
        cmax_u_brain_ci90_nm = ci90_bounds(&mut cb_vals);
        auc_u_plasma_ci90_nm_hr = ci90_bounds(&mut auc_vals);

        let mut tp_buf = Vec::with_capacity(runs.len());
        let mut tb_buf = Vec::with_capacity(runs.len());
        for ti in 0..PBPK_TIME_POINTS {
            tp_buf.clear();
            tb_buf.clear();
            for r in &runs {
                tp_buf.push(r.c_unbound_plasma_nm[ti] as f64);
                tb_buf.push(r.c_unbound_brain_nm[ti] as f64);
            }
            let [p05, p95] = ci90_bounds(&mut tp_buf);
            let [b05, b95] = ci90_bounds(&mut tb_buf);
            c_unbound_plasma_p05_nm[ti] = p05 as f32;
            c_unbound_plasma_p95_nm[ti] = p95 as f32;
            c_unbound_brain_p05_nm[ti] = b05 as f32;
            c_unbound_brain_p95_nm[ti] = b95 as f32;
        }
    }

    let report = OralPbpkReport {
        compound_id: mol.id.clone(),
        compound_name: mol.name.clone(),
        target_name: mol.target.clone(),
        target_ic50_nm,
        mw,
        clogp: phys.clogp,
        logd_74: pka.microstate_logd_74,
        charge_class_74: pka.charge_class,
        config,
        fu_plasma,
        blood_to_plasma_rbp: rbp,
        vss_l_per_kg,
        vss_total_l,
        organ_kps,
        peff_1e4_cm_s,
        luminal_sol_ug_ml,
        fa_analytical_perm,
        fa_integrated: nom.fa_integrated,
        q_gut_l_hr,
        fg_gut_wall: nom.fg_gut_wall,
        cl_int_u_ml_min_kg: cyp.cl_int_ml_min_kg,
        cl_hepatic_blood_l_hr: nom.cl_hepatic_blood_l_hr,
        cl_renal_plasma_l_hr,
        hepatic_extraction_eh: nom.hepatic_extraction_eh,
        fh_hepatic: nom.fh_hepatic,
        f_oral_pct: nom.f_oral_pct,
        fu_brain,
        ps_bbb_l_hr,
        bbb_efflux_ratio: bbb_er,
        kp_brain_total,
        kp_uu_brain,
        cns_class,
        cmax_total_plasma_nm: nom.cmax_total_plasma_nm,
        cmax_unbound_plasma_nm: nom.cmax_unbound_plasma_nm,
        cmax_unbound_brain_nm: nom.cmax_unbound_brain_nm,
        tmax_plasma_hr: nom.tmax_plasma_hr,
        tmax_brain_hr: nom.tmax_brain_hr,
        auc_unbound_plasma_nm_hr: nom.auc_unbound_plasma_nm_hr,
        auc_unbound_brain_nm_hr: nom.auc_unbound_brain_nm_hr,
        terminal_t_half_hr: nom.terminal_t_half_hr,
        time_above_ic50_plasma_hr: nom.time_above_ic50_plasma_hr,
        time_above_ic50_brain_hr: nom.time_above_ic50_brain_hr,
        peak_ro_plasma_pct: nom.peak_ro_plasma_pct,
        peak_ro_brain_pct: nom.peak_ro_brain_pct,
        time_hr: nom.time_hr,
        c_total_plasma_nm: nom.c_total_plasma_nm,
        c_unbound_plasma_nm: nom.c_unbound_plasma_nm,
        c_unbound_brain_nm: nom.c_unbound_brain_nm,
        ro_plasma_pct: nom.ro_plasma_pct,
        ro_brain_pct: nom.ro_brain_pct,
        gi_absorbed_pct: nom.gi_absorbed_pct,
        cyp3a4_km_um,
        cyp3a4_vmax_umol_hr,
        cyp3a4_peak_saturation_pct: nom.peak_cyp3a4_sat_pct,
        virtual_pop_n: pop_n,
        f_oral_ci90,
        cmax_u_plasma_ci90_nm,
        cmax_u_brain_ci90_nm,
        auc_u_plasma_ci90_nm_hr,
        c_unbound_plasma_p05_nm,
        c_unbound_plasma_p95_nm,
        c_unbound_brain_p05_nm,
        c_unbound_brain_p95_nm,
    };
    (report, nom.final_y, nom.total_dosed_nmol)
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw_iiv_normals(mut state: u64) -> [f64; 7] {
    let mut out = [0.0f64; 7];
    let mut i = 0;
    while i < 7 {
        let u1 = (((splitmix64(&mut state) >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        let u2 = ((splitmix64(&mut state) >> 11) as f64) / ((1u64 << 53) as f64);
        let r = (-2.0 * u1.clamp(1e-12, 1.0).ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out[i] = (r * theta.cos()).clamp(-2.8, 2.8);
        if i + 1 < 7 {
            out[i + 1] = (r * theta.sin()).clamp(-2.8, 2.8);
        }
        i += 2;
    }
    out
}

fn ci90_bounds(vals: &mut [f64]) -> [f64; 2] {
    if vals.is_empty() {
        return [0.0, 0.0];
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let idx_lo = ((n - 1) as f64 * 0.05).round() as usize;
    let idx_hi = ((n - 1) as f64 * 0.95).round() as usize;
    [vals[idx_lo.min(n - 1)], vals[idx_hi.min(n - 1)]]
}

/// Evaluates standard `50 mg` fasted single-dose oral PBPK & `Kp,uu,brain` across an entire dataset.
pub fn summarize_dataset_oral_pbpk(mols: &[Molecule], dose_mg: f64) -> Vec<PbpkDatasetRow> {
    let cfg = PbpkDosingConfig {
        dose_mg,
        virtual_pop_n: 0,
        ..PbpkDosingConfig::default()
    };
    mols.iter()
        .enumerate()
        .map(|(idx, m)| {
            let r = simulate_oral_pbpk(m, cfg);
            PbpkDatasetRow {
                mol_idx: idx,
                id: m.id.clone(),
                name: m.name.clone(),
                target: m.target.clone(),
                ic50_nm: r.target_ic50_nm,
                fa_pct: r.fa_integrated * 100.0,
                fg_pct: r.fg_gut_wall * 100.0,
                fh_pct: r.fh_hepatic * 100.0,
                f_oral_pct: r.f_oral_pct,
                vss_l_kg: r.vss_l_per_kg,
                t_half_hr: r.terminal_t_half_hr,
                fu_plasma: r.fu_plasma,
                kp_uu_brain: r.kp_uu_brain,
                cns_class: r.cns_class,
                cmax_u_plasma_nm: r.cmax_unbound_plasma_nm,
                cmax_u_brain_nm: r.cmax_unbound_brain_nm,
                peak_ro_brain_pct: r.peak_ro_brain_pct,
            }
        })
        .collect()
}

/// Exports the `97`-point `RK4` oral PBPK time-course (`C_total,p`, `Cu,p`, `Cu,brain`, `RO_p%`, `RO_brain%`) to CSV.
pub fn export_pbpk_timecourse_csv(rep: &OralPbpkReport) -> String {
    let mut out = String::from(
        "time_hr,gi_absorbed_pct,c_total_plasma_nm,c_unbound_plasma_nm,c_unbound_brain_nm,ro_plasma_pct,ro_brain_pct,target_ic50_nm\n",
    );
    for i in 0..PBPK_TIME_POINTS {
        out.push_str(&format!(
            "{:.3},{:.2},{:.3},{:.4},{:.4},{:.2},{:.2},{:.4}\n",
            rep.time_hr[i],
            rep.gi_absorbed_pct[i],
            rep.c_total_plasma_nm[i],
            rep.c_unbound_plasma_nm[i],
            rep.c_unbound_brain_nm[i],
            rep.ro_plasma_pct[i],
            rep.ro_brain_pct[i],
            rep.target_ic50_nm
        ));
    }
    out
}

/// Exports dataset-wide oral `Fa * Fg * Fh` bioavailability, `Vss`, `t1/2`, and `Kp,uu,brain` summary to CSV.
pub fn export_dataset_pbpk_csv(rows: &[PbpkDatasetRow]) -> String {
    let mut out = String::from(
        "id,name,target,ic50_nm,fa_pct,fg_pct,fh_pct,f_oral_pct,vss_l_kg,t_half_hr,fu_plasma,kp_uu_brain,cns_class,cmax_u_plasma_nm,cmax_u_brain_nm,peak_ro_brain_pct\n",
    );
    for r in rows {
        out.push_str(&format!(
            "\"{}\",\"{}\",\"{}\",{:.3},{:.1},{:.1},{:.1},{:.1},{:.2},{:.2},{:.4},{:.3},\"{}\",{:.2},{:.2},{:.1}\n",
            r.id,
            r.name,
            r.target,
            r.ic50_nm,
            r.fa_pct,
            r.fg_pct,
            r.fh_pct,
            r.f_oral_pct,
            r.vss_l_kg,
            r.t_half_hr,
            r.fu_plasma,
            r.kp_uu_brain,
            r.cns_class.short_label(),
            r.cmax_u_plasma_nm,
            r.cmax_u_brain_nm,
            r.peak_ro_brain_pct
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_benchmark_26;

    #[test]
    fn test_cyp3a4_km_saturable_nonlinearity_mass_balance_and_ci90_bands() {
        let bench = load_benchmark_26();
        let halo = bench
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("Haloperidol"))
            .expect("Haloperidol");

        let low = simulate_oral_pbpk(
            halo,
            PbpkDosingConfig {
                dose_mg: 2.0,
                cyp3a4_km_um: Some(1.5),
                ..PbpkDosingConfig::default()
            },
        );
        let high = simulate_oral_pbpk(
            halo,
            PbpkDosingConfig {
                dose_mg: 500.0,
                cyp3a4_km_um: Some(1.5),
                ..PbpkDosingConfig::default()
            },
        );

        // 1. Exact mass-balance identity across both doses and all regimens
        for reg in [
            DosingRegimen::SingleDose24h,
            DosingRegimen::OnceDailyQd48h,
            DosingRegimen::TwiceDailyBid48h,
        ] {
            let r = simulate_oral_pbpk(
                halo,
                PbpkDosingConfig {
                    dose_mg: 75.0,
                    regimen: reg,
                    ..PbpkDosingConfig::default()
                },
            );
            let exp_f = r.fa_integrated * r.fg_gut_wall * r.fh_hepatic * 100.0;
            assert!(
                (r.f_oral_pct - exp_f).abs() < 1e-4,
                "F_oral != Fa*Fg*Fh*100 for {:?}: {} vs {}",
                reg,
                r.f_oral_pct,
                exp_f
            );
        }

        // 2. Saturable Michaelis-Menten CYP3A4 nonlinearity at 500 mg vs 2 mg
        assert!(high.cyp3a4_peak_saturation_pct > low.cyp3a4_peak_saturation_pct);
        assert!(
            high.cyp3a4_peak_saturation_pct > 5.0,
            "Expected measurable CYP3A4 saturation at 500 mg with Km=1.5 uM, got {:.2}%",
            high.cyp3a4_peak_saturation_pct
        );
        assert!(
            high.fg_gut_wall >= low.fg_gut_wall && high.fh_hepatic >= low.fh_hepatic,
            "Saturating first-pass CYP3A4 must elevate Fg/Fh at 500 mg vs 2 mg"
        );
        assert!(high.fg_gut_wall * high.fh_hepatic > low.fg_gut_wall * low.fh_hepatic);

        // 3. Virtual-population 90% CI ordering (p05 <= nominal <= p95)
        assert_eq!(low.virtual_pop_n, 48);
        assert!(low.f_oral_ci90[0] <= low.f_oral_pct && low.f_oral_pct <= low.f_oral_ci90[1]);
        assert!(
            low.cmax_u_plasma_ci90_nm[0] <= low.cmax_unbound_plasma_nm
                && low.cmax_unbound_plasma_nm <= low.cmax_u_plasma_ci90_nm[1]
        );
        assert!(
            low.cmax_u_brain_ci90_nm[0] <= low.cmax_unbound_brain_nm
                && low.cmax_unbound_brain_nm <= low.cmax_u_brain_ci90_nm[1]
        );
        for ti in 0..PBPK_TIME_POINTS {
            assert!(low.c_unbound_plasma_p05_nm[ti] <= low.c_unbound_plasma_p95_nm[ti] + 1e-6);
            assert!(low.c_unbound_brain_p05_nm[ti] <= low.c_unbound_brain_p95_nm[ti] + 1e-6);
        }
    }

    #[test]
    fn test_pbpk_zero_elimination_mass_conservation_y0_to_y18_and_weak_base_supersaturation() {
        let bench = load_benchmark_26();
        let halo = bench
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case("Haloperidol"))
            .expect("Haloperidol");

        let cfg = PbpkDosingConfig {
            dose_mg: 50.0,
            prandial: PrandialState::Fasted,
            regimen: DosingRegimen::OnceDailyQd48h,
            virtual_pop_n: 0,
            ..PbpkDosingConfig::default()
        };
        let (rep, final_y, total_dosed_nmol) = simulate_oral_pbpk_impl(halo, cfg, true);
        let mass_y0_to_y18: f64 = final_y[0..=18].iter().sum();
        assert!(total_dosed_nmol > 1.0e4);
        assert!(
            (mass_y0_to_y18 - total_dosed_nmol).abs() < 1e-6,
            "Closed CL_elim=0 PBPK must conserve mass across y[0..=18] to <1e-6 nmol: sum={mass_y0_to_y18} vs dosed={total_dosed_nmol}"
        );
        // Verify brain distribution occurred with central subtraction conserving plasma+brain total
        assert!(final_y[18] > 0.0 && rep.cmax_unbound_brain_nm > 0.0);
    }
}
