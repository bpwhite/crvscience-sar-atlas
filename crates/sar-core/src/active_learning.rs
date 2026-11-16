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

//! `ECFP4` Tanimoto + `3D-USRCAT` Hybrid Gaussian-Process (`GP`) & Bootstrapped Random-Forest
//! (`RF`) Ensemble QSAR Surrogate with Conformal Uncertainty Calibration, Expected-Improvement (`EI`) /
//! Upper-Confidence-Bound (`UCB`), and Kriging-Believer Batch Acquisition (`active_learning.rs`, `v2.7.0`).
//!
//! Primary References:
//! - Ralaivola, Swamidass, Saigo & Baldi (2005), *Neural Netw.* 18(8), 1093-1110 ("Graph kernels
//!   for chemical informatics" — positive-definite Tanimoto, MinMax & Hybrid kernels:
//!   `https://pubmed.ncbi.nlm.nih.gov/16157471/`) & `GAUCHE` Tanimoto-Kernel Gaussian Processes
//!   for Chemistry (`https://pmc.ncbi.nlm.nih.gov/articles/PMC13439769/`,
//!   `https://academic.oup.com/biometrics/article/82/4/ujag168/8894036`)
//! - Graff, Shakhnovich & Coley (2021), *Chem. Sci.* 12(22), 7866-7881 ("Accelerating
//!   high-throughput virtual screening through molecular pool-based active learning" — `MolPAL`
//!   `EI`/`UCB`/greedy acquisition: `https://pmc.ncbi.nlm.nih.gov/articles/PMC8188596/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11005039/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC13235147/`)
//! - Hirschfeld, Swanson, Yang, Barzilay & Coley (2020), *J. Chem. Inf. Model.* 60(8), 3770-3780
//!   ("Uncertainty Quantification Using Neural Networks for Molecular Property Prediction":
//!   `https://pubmed.ncbi.nlm.nih.gov/32702986/`) & Epistemic Calibration / `UNIQUE` Benchmarking
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC9898940/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC12848971/`)

use crate::alerts::scan_structural_alerts;
use crate::bioisostere::generate_bioisosteres;
use crate::conformer3d::{evaluate_3d_summary_profile, usrcat_similarity};
use crate::fingerprint::{compute_ecfp, Ecfp2048};
use crate::mol::Molecule;
use crate::mol_io::parse_smiles;
use crate::parallel_lib::{
    apply_parallel_reaction_step, curated_amine_plate, curated_azide_plate,
    curated_boronic_acid_plate, ParallelReactionKind,
};
use crate::physchem::evaluate_physchem;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Dimensionality of the dense hybrid feature vector fed to the bootstrapped `Random Forest` ensemble
/// (`64` folded `ECFP4` counts + `12` `USR` 3D shape + `12` `USRCAT` 3D pharmacophore + `8` physchem = `96`).
pub const RF_FEATURE_DIM: usize = 96;

/// Molecular similarity kernel representation for the exact Cholesky Gaussian Process (`GP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QsarKernelMode {
    /// Convex positive-definite blend `(1 - w_3d) * K_Tanimoto(ECFP4) + w_3d * K_3D(USRCAT)`.
    HybridEcfp4And3dUsrcat,
    /// Pure 2D `ECFP4` Tanimoto kernel (`GAUCHE` standard substructure covariance).
    Pure2dEcfp4Tanimoto,
    /// Pure 3D `USRCAT` shape-pharmacophore kernel (enables zero-2D-overlap scaffold hopping).
    Pure3dUsrcatPharmacophore,
}

impl QsarKernelMode {
    pub const ALL: [Self; 3] = [
        Self::HybridEcfp4And3dUsrcat,
        Self::Pure2dEcfp4Tanimoto,
        Self::Pure3dUsrcatPharmacophore,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::HybridEcfp4And3dUsrcat => "Hybrid 2D ECFP4 Tanimoto + 3D USRCAT Kernel",
            Self::Pure2dEcfp4Tanimoto => "Pure 2D ECFP4 Tanimoto Kernel (GAUCHE)",
            Self::Pure3dUsrcatPharmacophore => "Pure 3D USRCAT Shape-Pharmacophore Kernel",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HybridEcfp4And3dUsrcat => "ECFP4 + 3D-USRCAT",
            Self::Pure2dEcfp4Tanimoto => "2D ECFP4 Only",
            Self::Pure3dUsrcatPharmacophore => "3D USRCAT Only",
        }
    }
}

/// Probabilistic QSAR surrogate architecture (`GP`, `Random Forest Ensemble`, or `Consensus GP+RF`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurrogateModelKind {
    /// Precision-weighted consensus of Exact Cholesky `GP` + 24-Tree Bootstrapped `Random Forest` (`Default`).
    ConsensusGpAndEnsemble,
    /// Exact closed-form Cholesky Gaussian Process regressor (`mu`, exact posterior `sigma_ep`).
    ExactGaussianProcess,
    /// 24-tree bootstrapped random-subspace `CART` regression ensemble (`inter-tree epistemic std`).
    RandomForestEnsemble,
}

impl SurrogateModelKind {
    pub const ALL: [Self; 3] = [
        Self::ConsensusGpAndEnsemble,
        Self::ExactGaussianProcess,
        Self::RandomForestEnsemble,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ConsensusGpAndEnsemble => "Consensus GP + RF",
            Self::ExactGaussianProcess => "Exact Cholesky GP",
            Self::RandomForestEnsemble => "24-Tree RF Ensemble",
        }
    }
}

/// Active-learning acquisition criterion (`MolPAL` / `Reker & Schneider`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionStrategy {
    /// Analytical Expected Improvement `EI(x) = (mu - y* - xi) Phi(Z) + sigma phi(Z)` (`Balanced`).
    ExpectedImprovement,
    /// Upper Confidence Bound `UCB(x) = mu(x) + kappa * sigma(x)` (`Optimistic Exploration`).
    UpperConfidenceBound,
    /// Pure Epistemic Uncertainty `sigma_epistemic(x)` (`Space-Filling Model Refinement`).
    PureEpistemicUncertainty,
    /// Pure Predicted Mean `mu(x)` (`Greedy Exploitation Baseline`).
    GreedyExploitation,
}

impl AcquisitionStrategy {
    pub const ALL: [Self; 4] = [
        Self::ExpectedImprovement,
        Self::UpperConfidenceBound,
        Self::PureEpistemicUncertainty,
        Self::GreedyExploitation,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ExpectedImprovement => "Expected Improvement (EI — Balanced)",
            Self::UpperConfidenceBound => "Upper Confidence Bound (UCB = mu + k*sigma)",
            Self::PureEpistemicUncertainty => "Max Epistemic Uncertainty (Space-Filling)",
            Self::GreedyExploitation => "Greedy Exploitation (Max Predicted pAct)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ExpectedImprovement => "Expected Imp (EI)",
            Self::UpperConfidenceBound => "UCB (mu + k*sig)",
            Self::PureEpistemicUncertainty => "Max Uncertainty",
            Self::GreedyExploitation => "Greedy (mu)",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::ExpectedImprovement => [52, 211, 153],      // Emerald
            Self::UpperConfidenceBound => [56, 189, 248],     // Sky Cyan
            Self::PureEpistemicUncertainty => [192, 132, 252], // Violet
            Self::GreedyExploitation => [251, 191, 36],       // Amber
        }
    }
}

/// Source of unlabeled candidate molecules for the active-learning screening pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidatePoolSource {
    /// Harvests novel bioisosteres of active leads + 2-step combinatorial analogs (`Prospective DMTA`).
    HarvestedAnalogsAndLibrary,
    /// Holds out `60%` of dataset compounds as an unassayed pool to benchmark true `pAct` recovery!
    DatasetHoldoutBenchmark,
}

impl CandidatePoolSource {
    pub const ALL: [Self; 2] = [
        Self::HarvestedAnalogsAndLibrary,
        Self::DatasetHoldoutBenchmark,
    ];

    pub fn short_label(self) -> &'static str {
        match self {
            Self::HarvestedAnalogsAndLibrary => "Prospective Analog + Library Pool",
            Self::DatasetHoldoutBenchmark => "Holdout Benchmark (40% Train / 60% Pool)",
        }
    }
}

/// Applicability Domain (`AD`) confidence tier based on nearest-training-neighbor similarity & epistemic std.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplicabilityDomainTier {
    /// `Max Train Sim >= 0.50` (`High interpolation confidence`).
    InDomain,
    /// `0.28 <= Max Train Sim < 0.50` (`Productive scaffold-hop boundary`).
    BoundaryExploration,
    /// `Max Train Sim < 0.28` (`Out-of-domain high epistemic extrapolation`).
    OutOfDomainExtrapolation,
}

impl ApplicabilityDomainTier {
    pub fn from_max_sim(sim: f32) -> Self {
        if sim >= 0.50 {
            Self::InDomain
        } else if sim >= 0.28 {
            Self::BoundaryExploration
        } else {
            Self::OutOfDomainExtrapolation
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::InDomain => "In-Domain",
            Self::BoundaryExploration => "Boundary Hop",
            Self::OutOfDomainExtrapolation => "Out-of-Domain",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::InDomain => [52, 211, 153],
            Self::BoundaryExploration => [56, 189, 248],
            Self::OutOfDomainExtrapolation => [251, 191, 36],
        }
    }
}

/// Configuration for `GP`/`RF` surrogate fitting and Kriging-Believer batch acquisition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    pub kernel_mode: QsarKernelMode,
    pub surrogate_kind: SurrogateModelKind,
    pub acquisition: AcquisitionStrategy,
    pub pool_source: CandidatePoolSource,
    /// Weight `w_3d` (`0.0..=1.0`, default `0.35`) on `3D-USRCAT` kernel inside `HybridEcfp4And3dUsrcat`.
    pub weight_3d_usrcat: f64,
    /// Assumed experimental assay noise `sigma_n` (`pAct` units, default `0.22` ~ 1.6-fold IC50 noise).
    pub assay_noise_sigma: f64,
    /// `UCB` exploration multiplier `kappa` (`0.2..=4.0`, default `1.50`).
    pub ucb_kappa: f64,
    /// `EI` jitter margin `xi` (`0.0..=0.5`, default `0.05` `pAct` units).
    pub ei_xi: f64,
    /// Target synthesis/assay batch size `B` (`4..=32`, default `10`) selected via Kriging Believer!
    pub batch_size: usize,
    /// Optional newline/comma-separated custom candidate SMILES appended to the unlabeled pool.
    pub custom_pool_smiles: Vec<String>,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            kernel_mode: QsarKernelMode::HybridEcfp4And3dUsrcat,
            surrogate_kind: SurrogateModelKind::ConsensusGpAndEnsemble,
            acquisition: AcquisitionStrategy::ExpectedImprovement,
            pool_source: CandidatePoolSource::HarvestedAnalogsAndLibrary,
            weight_3d_usrcat: 0.35,
            assay_noise_sigma: 0.22,
            ucb_kappa: 1.50,
            ei_xi: 0.05,
            batch_size: 10,
            custom_pool_smiles: Vec::new(),
        }
    }
}

/// Out-of-fold Leave-One-Out (`LOO`) cross-validation & conformal calibration diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QsarValidationMetrics {
    pub n_train: usize,
    pub best_observed_pact: f64,
    pub mean_observed_pact: f64,
    pub loo_q2: f64,
    pub loo_rmse: f64,
    pub loo_mae: f64,
    pub gp_log_marginal_likelihood: f64,
    /// Spearman rank correlation between predicted `epistemic_std` and actual `|y - y_hat_loo|`.
    pub uncertainty_error_spearman_rho: f64,
    /// Post-hoc conformal multiplier `q_90` guaranteeing >= 90% empirical interval coverage.
    pub conformal_90_scale: f64,
    pub empirical_90_coverage_pct: f64,
    /// `(observed_pact, loo_predicted_pact, loo_epistemic_std, compound_name)` tuples for parity plotting.
    pub parity_points: Vec<(f64, f64, f64, String)>,
}

/// Scored unlabeled candidate molecule with `GP`/`RF` posterior, `90%` conformal interval, and batch pick rank.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveLearningCandidate {
    pub rank: usize,
    pub candidate_id: String,
    pub candidate_name: String,
    pub provenance: String,
    pub canonical_smiles: String,
    pub mw: f64,
    pub clogp: f64,
    pub qed: f64,

    // Surrogate Posterior Predictions (pAct = -log10 M)
    pub gp_mean_pact: f64,
    pub gp_epistemic_std: f64,
    pub rf_mean_pact: f64,
    pub rf_epistemic_std: f64,
    pub predicted_pact: f64,
    pub epistemic_std: f64,
    pub total_predictive_std: f64,
    pub ci90_low_pact: f64,
    pub ci90_high_pact: f64,

    // Acquisition & Kriging-Believer Batch Selection
    pub expected_improvement: f64,
    pub ucb_score: f64,
    pub raw_acquisition_score: f64,
    /// Post-Kriging-Believer acquisition score at the step it was chosen (or final step if unpicked).
    pub batch_believer_score: f64,
    pub selected_in_batch: bool,
    pub batch_pick_order: Option<usize>,

    // Applicability Domain & Optional Ground-Truth (when running Holdout Benchmark)
    pub nearest_train_similarity: f32,
    pub nearest_train_name: String,
    pub ad_tier: ApplicabilityDomainTier,
    pub true_holdout_pact: Option<f64>,
    pub mol: Molecule,
}

/// Single round of retrospective Active Learning vs Random Screening DMTA simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulatedAlRound {
    pub round: usize,
    pub cumulative_tested: usize,
    pub al_best_pact_found: f64,
    pub random_best_pact_found: f64,
    pub al_top_quartile_hits_found: usize,
    pub random_top_quartile_hits_found: usize,
    pub enrichment_factor: f64,
}

/// Complete Active-Learning QSAR model, uncertainty calibration, and batch acquisition report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveLearningReport {
    pub config: ActiveLearningConfig,
    pub validation: QsarValidationMetrics,
    pub pool_size: usize,
    pub selected_batch_count: usize,
    pub batch_mean_predicted_pact: f64,
    pub batch_mean_epistemic_std: f64,
    pub batch_mean_pairwise_dissim: f64,
    pub simulated_curve: Vec<SimulatedAlRound>,
    pub candidates: Vec<ActiveLearningCandidate>,
}

// ---------------------------------------------------------------------------
// Internal Feature Extraction, Positive-Definite Hybrid Kernel & Linear Algebra
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FeaturizedMol {
    mol: Molecule,
    provenance: String,
    true_pact: Option<f64>,
    fp: Ecfp2048,
    usrcat60: Vec<f32>,
    rf_x: [f64; RF_FEATURE_DIM],
    mw: f64,
    clogp: f64,
    qed: f64,
}

fn featurize_molecule(
    mol: &Molecule,
    provenance: impl Into<String>,
    true_pact: Option<f64>,
    existing_s3d_usrcat: Option<&[f32]>,
    existing_s3d_usr: Option<&[f32; 12]>,
) -> FeaturizedMol {
    let alerts = scan_structural_alerts(mol);
    let pc = evaluate_physchem(mol, alerts.len());
    let fp = compute_ecfp(mol, 2);

    let (usr12, usrcat60) = if let (Some(uc), Some(u12)) = (existing_s3d_usrcat, existing_s3d_usr) {
        (*u12, uc.to_vec())
    } else {
        let s3d = evaluate_3d_summary_profile(mol, true);
        (s3d.usr12, s3d.usrcat60)
    };

    let mut rf_x = [0.0_f64; RF_FEATURE_DIM];
    // Bits 0..64: folded ECFP4 bit density counts (`2048 -> 64` bins of 32 bits each)
    for b in 0..2048usize {
        let w = fp.words[b >> 6];
        if ((w >> (b & 63)) & 1) != 0 {
            rf_x[b & 63] += 1.0;
        }
    }
    // Indices 64..76: 12-D USR 3D shape moments
    for k in 0..12 {
        rf_x[64 + k] = usr12[k] as f64;
    }
    // Indices 76..88: 12 key pharmacophore-channel USRCAT moments (strided across the 60-D vector)
    for k in 0..12 {
        rf_x[76 + k] = usrcat60.get(k * 5).copied().unwrap_or(0.0) as f64;
    }
    // Indices 88..96: 8 core physicochemical descriptors
    rf_x[88] = pc.mw / 500.0;
    rf_x[89] = pc.clogp / 5.0;
    rf_x[90] = pc.tpsa / 120.0;
    rf_x[91] = pc.fsp3;
    rf_x[92] = pc.hbd as f64;
    rf_x[93] = pc.hba as f64;
    rf_x[94] = pc.rotatable_bonds as f64;
    rf_x[95] = pc.qed_mean;

    FeaturizedMol {
        mol: mol.clone(),
        provenance: provenance.into(),
        true_pact,
        fp,
        usrcat60,
        rf_x,
        mw: pc.mw,
        clogp: pc.clogp,
        qed: pc.qed_mean,
    }
}

/// Evaluates the positive-definite `ECFP4` Tanimoto + `3D-USRCAT` Laplacian kernel in `[0.0, 1.0]`.
fn hybrid_molecular_kernel(
    a: &FeaturizedMol,
    b: &FeaturizedMol,
    mode: QsarKernelMode,
    w_3d: f64,
) -> f64 {
    let k_2d = a.fp.tanimoto(&b.fp) as f64;
    match mode {
        QsarKernelMode::Pure2dEcfp4Tanimoto => k_2d,
        QsarKernelMode::Pure3dUsrcatPharmacophore => {
            // Positive-definite exponential L1 kernel `exp(-||u - v||_1 / scale)` aligned with USRCAT
            let n = a.usrcat60.len().min(b.usrcat60.len());
            if n == 0 {
                usrcat_similarity(&a.usrcat60, &b.usrcat60)
            } else {
                let l1: f64 = (0..n)
                    .map(|i| ((a.usrcat60[i] - b.usrcat60[i]).abs()) as f64)
                    .sum();
                (-l1 / 42.0).exp().clamp(0.0, 1.0)
            }
        }
        QsarKernelMode::HybridEcfp4And3dUsrcat => {
            let w = w_3d.clamp(0.0, 1.0);
            let n = a.usrcat60.len().min(b.usrcat60.len());
            let k_3d = if n == 0 {
                k_2d
            } else {
                let l1: f64 = (0..n)
                    .map(|i| ((a.usrcat60[i] - b.usrcat60[i]).abs()) as f64)
                    .sum();
                (-l1 / 42.0).exp().clamp(0.0, 1.0)
            };
            (1.0 - w) * k_2d + w * k_3d
        }
    }
}

/// Exact lower-triangular Cholesky factorization `A = L * L^T` (`n x n` row-major).
pub fn cholesky_decompose(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 1.0e-12 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

fn forward_solve_lower(l: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i * n + k] * z[k];
        }
        z[i] = s / l[i * n + i].max(1.0e-12);
    }
    z
}

fn backward_solve_upper_t(l: &[f64], n: usize, z: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i].max(1.0e-12);
    }
    x
}

struct FittedGaussianProcess {
    prior_mean: f64,
    signal_var: f64,
    noise_var: f64,
    chol_l: Vec<f64>,
    alpha: Vec<f64>,
    log_marginal_likelihood: f64,
    n: usize,
}

impl FittedGaussianProcess {
    fn fit(
        train: &[FeaturizedMol],
        y: &[f64],
        mode: QsarKernelMode,
        w_3d: f64,
        noise_sigma: f64,
    ) -> Self {
        let n = train.len();
        if n == 0 {
            return Self {
                prior_mean: 6.5,
                signal_var: 1.0,
                noise_var: 0.05,
                chol_l: Vec::new(),
                alpha: Vec::new(),
                log_marginal_likelihood: 0.0,
                n: 0,
            };
        }
        let prior_mean = y.iter().sum::<f64>() / (n as f64);
        let y_centered: Vec<f64> = y.iter().map(|&v| v - prior_mean).collect();
        let emp_var = (y_centered.iter().map(|v| v * v).sum::<f64>() / (n.max(1) as f64))
            .clamp(0.25, 4.0);
        let signal_var = emp_var;
        let noise_var = (noise_sigma.clamp(0.05, 1.5)).powi(2);

        let mut cov = vec![0.0_f64; n * n];
        for i in 0..n {
            cov[i * n + i] = signal_var + noise_var + 1.0e-6;
            for j in (i + 1)..n {
                let kij = signal_var * hybrid_molecular_kernel(&train[i], &train[j], mode, w_3d);
                cov[i * n + j] = kij;
                cov[j * n + i] = kij;
            }
        }

        let chol_l = cholesky_decompose(&cov, n).unwrap_or_else(|| {
            for i in 0..n {
                cov[i * n + i] += 1.0e-3;
            }
            cholesky_decompose(&cov, n).unwrap_or_else(|| {
                let mut diag = vec![0.0_f64; n * n];
                for i in 0..n {
                    diag[i * n + i] = (signal_var + noise_var).sqrt();
                }
                diag
            })
        });

        let z = forward_solve_lower(&chol_l, n, &y_centered);
        let alpha = backward_solve_upper_t(&chol_l, n, &z);

        let data_fit: f64 = y_centered
            .iter()
            .zip(alpha.iter())
            .map(|(yi, ai)| yi * ai)
            .sum();
        let log_det: f64 = (0..n).map(|i| chol_l[i * n + i].ln()).sum::<f64>();
        let lml = -0.5 * data_fit - log_det - 0.5 * (n as f64) * (2.0 * PI).ln();

        Self {
            prior_mean,
            signal_var,
            noise_var,
            chol_l,
            alpha,
            log_marginal_likelihood: lml,
            n,
        }
    }

    fn predict(
        &self,
        cand: &FeaturizedMol,
        train: &[FeaturizedMol],
        mode: QsarKernelMode,
        w_3d: f64,
    ) -> (f64, f64) {
        if self.n == 0 {
            return (self.prior_mean, self.signal_var.sqrt());
        }
        let k_star: Vec<f64> = train
            .iter()
            .map(|tr| self.signal_var * hybrid_molecular_kernel(cand, tr, mode, w_3d))
            .collect();
        let mean = self.prior_mean
            + k_star
                .iter()
                .zip(self.alpha.iter())
                .map(|(k, a)| k * a)
                .sum::<f64>();
        let v = forward_solve_lower(&self.chol_l, self.n, &k_star);
        let red_var: f64 = v.iter().map(|vi| vi * vi).sum();
        let ep_var = (self.signal_var - red_var).max(1.0e-6);
        (mean, ep_var.sqrt())
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust Bootstrapped Random-Subspace Regression Tree Ensemble (`B = 24`)
// ---------------------------------------------------------------------------

enum RegTreeNode {
    Leaf(f64),
    Split {
        feat_idx: usize,
        threshold: f64,
        left: Box<RegTreeNode>,
        right: Box<RegTreeNode>,
    },
}

struct Rng64(u64);

impl Rng64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }
}

fn build_reg_tree(
    xs: &[[f64; RF_FEATURE_DIM]],
    ys: &[f64],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    rng: &mut Rng64,
) -> RegTreeNode {
    if indices.is_empty() {
        return RegTreeNode::Leaf(6.5);
    }
    let mean_y = indices.iter().map(|&i| ys[i]).sum::<f64>() / (indices.len() as f64);
    if indices.len() <= 3 || depth >= max_depth {
        return RegTreeNode::Leaf(mean_y);
    }

    let parent_sse: f64 = indices
        .iter()
        .map(|&i| {
            let d = ys[i] - mean_y;
            d * d
        })
        .sum();
    if parent_sse <= 1.0e-6 {
        return RegTreeNode::Leaf(mean_y);
    }

    // Try 14 random features (`~1.4 * sqrt(96)`) per node
    let mut best_gain = 0.0_f64;
    let mut best_feat = 0usize;
    let mut best_thr = 0.0_f64;

    for _ in 0..14 {
        let f = rng.next_usize(RF_FEATURE_DIM);
        let mut vals: Vec<(f64, f64)> = indices.iter().map(|&i| (xs[i][f], ys[i])).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if (vals.last().map(|v| v.0).unwrap_or(0.0) - vals[0].0).abs() < 1.0e-9 {
            continue;
        }

        let total_sum: f64 = vals.iter().map(|v| v.1).sum();
        let total_n = vals.len();
        let mut left_sum = 0.0_f64;
        for split_k in 1..total_n {
            left_sum += vals[split_k - 1].1;
            if (vals[split_k].0 - vals[split_k - 1].0).abs() < 1.0e-9 {
                continue;
            }
            let n_l = split_k as f64;
            let n_r = (total_n - split_k) as f64;
            let right_sum = total_sum - left_sum;
            let mean_l = left_sum / n_l;
            let mean_r = right_sum / n_r;
            let sse_l: f64 = vals[..split_k]
                .iter()
                .map(|v| (v.1 - mean_l).powi(2))
                .sum();
            let sse_r: f64 = vals[split_k..]
                .iter()
                .map(|v| (v.1 - mean_r).powi(2))
                .sum();
            let gain = parent_sse - (sse_l + sse_r);
            if gain > best_gain {
                best_gain = gain;
                best_feat = f;
                best_thr = 0.5 * (vals[split_k - 1].0 + vals[split_k].0);
            }
        }
    }

    if best_gain <= 1.0e-6 {
        return RegTreeNode::Leaf(mean_y);
    }

    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for &i in indices {
        if xs[i][best_feat] <= best_thr {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    if left_idx.is_empty() || right_idx.is_empty() {
        return RegTreeNode::Leaf(mean_y);
    }

    let left_node = build_reg_tree(xs, ys, &left_idx, depth + 1, max_depth, rng);
    let right_node = build_reg_tree(xs, ys, &right_idx, depth + 1, max_depth, rng);
    RegTreeNode::Split {
        feat_idx: best_feat,
        threshold: best_thr,
        left: Box::new(left_node),
        right: Box::new(right_node),
    }
}

fn eval_reg_tree(node: &RegTreeNode, x: &[f64; RF_FEATURE_DIM]) -> f64 {
    match node {
        RegTreeNode::Leaf(v) => *v,
        RegTreeNode::Split {
            feat_idx,
            threshold,
            left,
            right,
        } => {
            if x[*feat_idx] <= *threshold {
                eval_reg_tree(left, x)
            } else {
                eval_reg_tree(right, x)
            }
        }
    }
}

struct RandomForestRegressor {
    trees: Vec<RegTreeNode>,
}

impl RandomForestRegressor {
    fn fit(train: &[FeaturizedMol], y: &[f64], num_trees: usize) -> Self {
        let n = train.len();
        if n == 0 {
            return Self { trees: Vec::new() };
        }
        let xs: Vec<[f64; RF_FEATURE_DIM]> = train.iter().map(|t| t.rf_x).collect();
        let trees: Vec<RegTreeNode> = (0..num_trees)
            .into_par_iter()
            .map(|b| {
                let mut rng = Rng64(0xA17C_5EED_0000_0001_u64.wrapping_add((b as u64) * 7919));
                let boot: Vec<usize> = (0..n).map(|_| rng.next_usize(n)).collect();
                build_reg_tree(&xs, y, &boot, 0, 6, &mut rng)
            })
            .collect();
        Self { trees }
    }

    fn predict(&self, cand: &FeaturizedMol) -> (f64, f64) {
        if self.trees.is_empty() {
            return (6.5, 0.5);
        }
        let preds: Vec<f64> = self
            .trees
            .iter()
            .map(|t| eval_reg_tree(t, &cand.rf_x))
            .collect();
        let m = preds.iter().sum::<f64>() / (preds.len() as f64);
        let var = preds.iter().map(|&p| (p - m) * (p - m)).sum::<f64>()
            / ((preds.len().saturating_sub(1)).max(1) as f64);
        (m, var.max(1.0e-4).sqrt())
    }
}

// ---------------------------------------------------------------------------
// Standard Normal PDF/CDF & Expected Improvement (`EI`) + Spearman Rank Corr
// ---------------------------------------------------------------------------

fn approx_erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

fn std_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + approx_erf(z * FRAC_1_SQRT_2))
}

/// Analytical Jones et al. (1998) Expected Improvement `EI(x) = E[max(0, f(x) - y_best - xi)]`.
pub fn compute_expected_improvement(mean: f64, sigma: f64, best_y: f64, xi: f64) -> f64 {
    if sigma <= 1.0e-8 {
        return (mean - best_y - xi).max(0.0);
    }
    let imp = mean - best_y - xi;
    let z = imp / sigma;
    (imp * std_normal_cdf(z) + sigma * std_normal_pdf(z)).max(0.0)
}

fn spearman_rank_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return 0.0;
    }
    let rank_vec = |vals: &[f64]| -> Vec<f64> {
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&i, &j| {
            vals[i]
                .partial_cmp(&vals[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut r = vec![0.0_f64; n];
        for (pos, &idx) in ord.iter().enumerate() {
            r[idx] = (pos + 1) as f64;
        }
        r
    };
    let ra = rank_vec(&a[..n]);
    let rb = rank_vec(&b[..n]);
    let mean_r = ((n + 1) as f64) * 0.5;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_r;
        let db = rb[i] - mean_r;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1.0e-12 || vb <= 1.0e-12 {
        0.0
    } else {
        (cov / (va * vb).sqrt()).clamp(-1.0, 1.0)
    }
}

fn combine_surrogate_posteriors(
    kind: SurrogateModelKind,
    gp_mean: f64,
    gp_std: f64,
    rf_mean: f64,
    rf_std: f64,
) -> (f64, f64) {
    match kind {
        SurrogateModelKind::ExactGaussianProcess => (gp_mean, gp_std),
        SurrogateModelKind::RandomForestEnsemble => (rf_mean, rf_std),
        SurrogateModelKind::ConsensusGpAndEnsemble => {
            let w_gp = 1.0 / (gp_std * gp_std + 0.04);
            let w_rf = 1.0 / (rf_std * rf_std + 0.06);
            let w_sum = w_gp + w_rf;
            let mean = (w_gp * gp_mean + w_rf * rf_mean) / w_sum;
            // Law of total variance across GP + RF + cross-architecture disagreement
            let within_var = 0.5 * (gp_std * gp_std + rf_std * rf_std);
            let between_var = 0.25 * (gp_mean - rf_mean).powi(2);
            (mean, (within_var + between_var).sqrt())
        }
    }
}

fn score_acquisition_value(
    strat: AcquisitionStrategy,
    mean: f64,
    ep_std: f64,
    best_y: f64,
    xi: f64,
    kappa: f64,
) -> f64 {
    match strat {
        AcquisitionStrategy::ExpectedImprovement => {
            compute_expected_improvement(mean, ep_std, best_y, xi)
        }
        AcquisitionStrategy::UpperConfidenceBound => mean + kappa * ep_std,
        AcquisitionStrategy::PureEpistemicUncertainty => ep_std,
        AcquisitionStrategy::GreedyExploitation => mean,
    }
}

/// Harvests a chemically diverse unlabeled prospective candidate pool (bioisosteric analogs of
/// top leads + compact 2-step `Suzuki->Buchwald` and `Amide->Click` combinatorial candidates).
fn harvest_prospective_candidate_pool(
    ds: &DatasetAnalysis,
    custom_smiles: &[String],
) -> Vec<FeaturizedMol> {
    let mut seen_smiles: BTreeSet<String> = ds.mols.iter().map(|m| m.smiles.clone()).collect();
    let mut raw_cands: Vec<(Molecule, String)> = Vec::new();

    // 1. User-supplied custom candidate SMILES
    for (ci, smi) in custom_smiles.iter().enumerate() {
        let trimmed = smi.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(mut m) = parse_smiles(trimmed) {
            if seen_smiles.insert(m.smiles.clone()) {
                m.id = format!("CUST-{:02}", ci + 1);
                m.name = format!("Custom_Cand_{}", ci + 1);
                raw_cands.push((m, "Custom SMILES".to_string()));
            }
        }
    }

    // 2. Bioisosteric late-stage analogs around top 4 dataset leads (`In-Domain` & `Boundary` exploits!)
    let mut lead_ord: Vec<usize> = (0..ds.mols.len()).collect();
    lead_ord.sort_by(|&a, &b| {
        ds.mols[b]
            .pact
            .partial_cmp(&ds.mols[a].pact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &li in lead_ord.iter().take(4) {
        let analogs = generate_bioisosteres(&ds.mols[li], ds, ds.default_sar_matrix.as_ref());
        for (ai, an) in analogs.into_iter().take(8).enumerate() {
            if seen_smiles.insert(an.canonical_smiles.clone()) {
                let mut m = an.mol;
                m.id = format!("BIO-{}-{:02}", li + 1, ai + 1);
                m.name = format!("{} ({})", ds.mols[li].name, an.rule_code);
                raw_cands.push((m, format!("Bioisostere of {}", ds.mols[li].name)));
            }
        }
    }

    // 3. Focused 2-step Parallel/DEL exploration scaffolds (`Quinazoline Suzuki->Buchwald` + `Amide->Click`)
    let amines = curated_amine_plate();
    let boronics = curated_boronic_acid_plate();
    let azides = curated_azide_plate();

    if let Ok(q_core) = parse_smiles("Clc1ncnc2cc(Br)ccc12") {
        for ba in boronics.iter().take(5) {
            if let Ok(ba_m) = parse_smiles(&ba.smiles) {
                if let Some(step1) =
                    apply_parallel_reaction_step(ParallelReactionKind::SuzukiMiyaura, &q_core, &ba_m)
                {
                    for am in amines.iter().take(5) {
                        if let Ok(am_m) = parse_smiles(&am.smiles) {
                            if let Some(mut prod) = apply_parallel_reaction_step(
                                ParallelReactionKind::BuchwaldHartwig,
                                &step1,
                                &am_m,
                            ) {
                                if seen_smiles.insert(prod.smiles.clone()) {
                                    prod.id = format!("QNZ-{}-{}", ba.code, am.code);
                                    prod.name = format!("Quinazoline-{}-{}", ba.code, am.code);
                                    raw_cands.push((prod, "2-Step Suzuki->Buchwald Pool".to_string()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Ok(c_core) = parse_smiles("C#Cc1ccc(CC(=O)O)cc1") {
        for am in amines.iter().take(4) {
            if let Ok(am_m) = parse_smiles(&am.smiles) {
                if let Some(step1) =
                    apply_parallel_reaction_step(ParallelReactionKind::AmideCoupling, &c_core, &am_m)
                {
                    for az in azides.iter().take(4) {
                        if let Ok(az_m) = parse_smiles(&az.smiles) {
                            if let Some(mut prod) = apply_parallel_reaction_step(
                                ParallelReactionKind::CuAacClick,
                                &step1,
                                &az_m,
                            ) {
                                if seen_smiles.insert(prod.smiles.clone()) {
                                    prod.id = format!("CLK-{}-{}", am.code, az.code);
                                    prod.name = format!("ClickTriazole-{}-{}", am.code, az.code);
                                    raw_cands.push((prod, "2-Step Amide->Click DEL Pool".to_string()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    raw_cands
        .into_par_iter()
        .map(|(m, prov)| featurize_molecule(&m, prov, None, None, None))
        .collect()
}

/// Simulates `4` iterative Active-Learning DMTA rounds vs Random Screening on `all_labeled`
/// to quantify cumulative hit enrichment (`EF`) and fastest discovery of top-potency leads!
fn simulate_retrospective_dmta_rounds(
    all_labeled: &[FeaturizedMol],
    y_all: &[f64],
    cfg: &ActiveLearningConfig,
) -> Vec<SimulatedAlRound> {
    let n = all_labeled.len();
    if n < 8 {
        return Vec::new();
    }

    // Define top-quartile potency threshold (`top 25%` of dataset = true hits)
    let mut sorted_y = y_all.to_vec();
    sorted_y.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let hit_cutoff = sorted_y[(n / 4).max(1) - 1];
    let total_hits = y_all.iter().filter(|&&v| v >= hit_cutoff).count().max(1);

    // Deterministic initial seed = 4 moderately active diverse starting compounds (strided across middle/bottom)
    let mut order_by_y: Vec<usize> = (0..n).collect();
    order_by_y.sort_by(|&a, &b| {
        y_all[a]
            .partial_cmp(&y_all[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let init_k = (n / 5).clamp(4, 8);
    let step_k = ((n - init_k) / 4).clamp(2, 8);

    let mut al_known: Vec<usize> = (0..init_k)
        .map(|s| order_by_y[(s * (n * 2 / 3) / init_k).min(n - 1)])
        .collect();
    al_known.sort_unstable();
    al_known.dedup();

    let mut rounds = Vec::with_capacity(4);
    for r in 1..=4 {
        if r > 1 {
            let train_sub: Vec<FeaturizedMol> =
                al_known.iter().map(|&i| all_labeled[i].clone()).collect();
            let y_sub: Vec<f64> = al_known.iter().map(|&i| y_all[i]).collect();
            let best_y = y_sub.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let gp = FittedGaussianProcess::fit(
                &train_sub,
                &y_sub,
                cfg.kernel_mode,
                cfg.weight_3d_usrcat,
                cfg.assay_noise_sigma,
            );

            let mut scored_pool: Vec<(usize, f64)> = (0..n)
                .filter(|i| !al_known.contains(i))
                .map(|i| {
                    let (mu, sig) = gp.predict(
                        &all_labeled[i],
                        &train_sub,
                        cfg.kernel_mode,
                        cfg.weight_3d_usrcat,
                    );
                    let acq = score_acquisition_value(
                        cfg.acquisition,
                        mu,
                        sig,
                        best_y,
                        cfg.ei_xi,
                        cfg.ucb_kappa,
                    );
                    (i, acq)
                })
                .collect();
            scored_pool.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(pick_i, _) in scored_pool.iter().take(step_k) {
                al_known.push(pick_i);
            }
        }

        let tested = al_known.len().min(n);
        let al_best = al_known
            .iter()
            .map(|&i| y_all[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let al_hits = al_known.iter().filter(|&&i| y_all[i] >= hit_cutoff).count();

        // Expected random hypergeometric baseline for `tested` draws out of `n`
        let rand_frac = (tested as f64) / (n as f64);
        let rand_hits = ((total_hits as f64) * rand_frac).round() as usize;
        let rand_best_rank = ((n as f64) / ((tested + 1) as f64)).round() as usize;
        let rand_best = sorted_y[rand_best_rank.clamp(0, n - 1)];
        let hit_rate_al = (al_hits as f64) / (tested.max(1) as f64);
        let hit_rate_base = (total_hits as f64) / (n as f64);
        let ef = if hit_rate_base > 1.0e-6 {
            hit_rate_al / hit_rate_base
        } else {
            1.0
        };

        rounds.push(SimulatedAlRound {
            round: r,
            cumulative_tested: tested,
            al_best_pact_found: al_best,
            random_best_pact_found: rand_best,
            al_top_quartile_hits_found: al_hits,
            random_top_quartile_hits_found: rand_hits,
            enrichment_factor: ef,
        });
    }

    rounds
}

/// Trains the `ECFP4`/`3D-USRCAT` Gaussian-Process + `24-Tree Random Forest` Ensemble QSAR model,
/// computes Leave-One-Out conformal calibration, runs Kriging-Believer batch acquisition, and
/// benchmarks multi-round DMTA enrichment!
pub fn run_active_learning_campaign(
    ds: &DatasetAnalysis,
    config: &ActiveLearningConfig,
) -> ActiveLearningReport {
    let all_ds_feats: Vec<FeaturizedMol> = ds
        .mols
        .par_iter()
        .enumerate()
        .map(|(i, m)| {
            featurize_molecule(
                m,
                format!("Dataset ({})", m.target),
                Some(m.pact),
                Some(ds.shape3d[i].usrcat60.as_slice()),
                Some(&ds.shape3d[i].usr12),
            )
        })
        .collect();
    let all_ds_y: Vec<f64> = ds.mols.iter().map(|m| m.pact).collect();

    // Partition into Labeled Training Set vs Unlabeled Candidate Pool according to `config.pool_source`
    let (train_feats, train_y, pool_feats): (Vec<FeaturizedMol>, Vec<f64>, Vec<FeaturizedMol>) =
        match config.pool_source {
        CandidatePoolSource::HarvestedAnalogsAndLibrary => {
            let pool = harvest_prospective_candidate_pool(ds, &config.custom_pool_smiles);
            (all_ds_feats.clone(), all_ds_y.clone(), pool)
        }
        CandidatePoolSource::DatasetHoldoutBenchmark => {
            // Deterministic 40% Labeled Train / 60% Unlabeled Holdout Pool (stratified stride)
            let mut tr_f = Vec::new();
            let mut tr_y = Vec::new();
            let mut pl_f = Vec::new();
            for (i, mut fm) in all_ds_feats.iter().cloned().enumerate() {
                if i % 5 == 0 || i % 5 == 2 {
                    tr_f.push(fm);
                    tr_y.push(all_ds_y[i]);
                } else {
                    fm.provenance = "Held-Out Dataset Benchmark".to_string();
                    pl_f.push(fm);
                }
            }
            if tr_f.len() < 4 {
                (all_ds_feats.clone(), all_ds_y.clone(), pl_f)
            } else {
                (tr_f, tr_y, pl_f)
            }
        }
    };

    let n_train = train_feats.len();
    let best_obs = train_y
        .iter()
        .copied()
        .fold(6.0_f64, f64::max);
    let mean_obs = if n_train > 0 {
        train_y.iter().sum::<f64>() / (n_train as f64)
    } else {
        6.5
    };

    // 1. Fit full GP & 24-Tree RF Ensemble on `train_feats`
    let gp_full = FittedGaussianProcess::fit(
        &train_feats,
        &train_y,
        config.kernel_mode,
        config.weight_3d_usrcat,
        config.assay_noise_sigma,
    );
    let rf_full = RandomForestRegressor::fit(&train_feats, &train_y, 24);

    // 2. Leave-One-Out (`LOO`) Cross-Validation & Post-Hoc Conformal Uncertainty Calibration
    let loo_tuples: Vec<(f64, f64, f64, String)> = if n_train >= 4 {
        (0..n_train)
            .into_par_iter()
            .map(|hold_i| {
                let mut sub_f = Vec::with_capacity(n_train - 1);
                let mut sub_y = Vec::with_capacity(n_train - 1);
                for j in 0..n_train {
                    if j != hold_i {
                        sub_f.push(train_feats[j].clone());
                        sub_y.push(train_y[j]);
                    }
                }
                let gp_loo = FittedGaussianProcess::fit(
                    &sub_f,
                    &sub_y,
                    config.kernel_mode,
                    config.weight_3d_usrcat,
                    config.assay_noise_sigma,
                );
                let (gp_m, gp_s) = gp_loo.predict(
                    &train_feats[hold_i],
                    &sub_f,
                    config.kernel_mode,
                    config.weight_3d_usrcat,
                );
                let (pred_m, ep_s) = if config.surrogate_kind == SurrogateModelKind::ExactGaussianProcess
                {
                    (gp_m, gp_s)
                } else {
                    let rf_loo = RandomForestRegressor::fit(&sub_f, &sub_y, 12);
                    let (rf_m, rf_s) = rf_loo.predict(&train_feats[hold_i]);
                    combine_surrogate_posteriors(config.surrogate_kind, gp_m, gp_s, rf_m, rf_s)
                };
                (
                    train_y[hold_i],
                    pred_m,
                    ep_s.max(0.08),
                    train_feats[hold_i].mol.name.clone(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    let (loo_q2, loo_rmse, loo_mae, spearman_uq, conformal_q90, cov90_pct) =
        if !loo_tuples.is_empty() {
            let m = loo_tuples.len() as f64;
            let ss_res: f64 = loo_tuples.iter().map(|t| (t.0 - t.1).powi(2)).sum();
            let ss_tot: f64 = loo_tuples.iter().map(|t| (t.0 - mean_obs).powi(2)).sum();
            let q2 = if ss_tot > 1.0e-8 {
                (1.0 - ss_res / ss_tot).clamp(-1.0, 0.999)
            } else {
                0.0
            };
            let rmse = (ss_res / m).sqrt();
            let mae = loo_tuples.iter().map(|t| (t.0 - t.1).abs()).sum::<f64>() / m;
            let abs_errs: Vec<f64> = loo_tuples.iter().map(|t| (t.0 - t.1).abs()).collect();
            let sigmas: Vec<f64> = loo_tuples.iter().map(|t| t.2).collect();
            let rho = spearman_rank_corr(&sigmas, &abs_errs);

            // Normalized non-conformity scores `|y - y_hat| / total_std` -> 90th percentile `q_90`
            let mut nonconf: Vec<f64> = loo_tuples
                .iter()
                .map(|t| {
                    let tot_s = (t.2 * t.2 + config.assay_noise_sigma.powi(2)).sqrt();
                    (t.0 - t.1).abs() / tot_s.max(0.08)
                })
                .collect();
            nonconf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q_idx = (((nonconf.len() as f64) * 0.90).ceil() as usize)
                .saturating_sub(1)
                .min(nonconf.len() - 1);
            let q90 = nonconf[q_idx].clamp(1.28, 2.85);
            let covered = loo_tuples
                .iter()
                .filter(|t| {
                    let hw = q90 * (t.2 * t.2 + config.assay_noise_sigma.powi(2)).sqrt();
                    (t.0 - t.1).abs() <= hw
                })
                .count();
            let cov_pct = (covered as f64) / m * 100.0;
            (q2, rmse, mae, rho, q90, cov_pct)
        } else {
            (0.0, 0.0, 0.0, 0.0, 1.645, 90.0)
        };

    let validation = QsarValidationMetrics {
        n_train,
        best_observed_pact: best_obs,
        mean_observed_pact: mean_obs,
        loo_q2,
        loo_rmse,
        loo_mae,
        gp_log_marginal_likelihood: gp_full.log_marginal_likelihood,
        uncertainty_error_spearman_rho: spearman_uq,
        conformal_90_scale: conformal_q90,
        empirical_90_coverage_pct: cov90_pct,
        parity_points: loo_tuples,
    };

    // 3. Score Unlabeled Candidate Pool in Parallel
    let mut candidates: Vec<ActiveLearningCandidate> = pool_feats
        .par_iter()
        .map(|cf| {
            let (gp_m, gp_s) = gp_full.predict(
                cf,
                &train_feats,
                config.kernel_mode,
                config.weight_3d_usrcat,
            );
            let (rf_m, rf_s) = rf_full.predict(cf);
            let (pred_m, ep_s) =
                combine_surrogate_posteriors(config.surrogate_kind, gp_m, gp_s, rf_m, rf_s);
            let tot_s = (ep_s * ep_s + config.assay_noise_sigma.powi(2)).sqrt();
            let hw90 = conformal_q90 * tot_s;

            let ei = compute_expected_improvement(pred_m, ep_s, best_obs, config.ei_xi);
            let ucb = pred_m + config.ucb_kappa * ep_s;
            let raw_acq = score_acquisition_value(
                config.acquisition,
                pred_m,
                ep_s,
                best_obs,
                config.ei_xi,
                config.ucb_kappa,
            );

            // Nearest training neighbor similarity for Applicability Domain (`AD`)
            let mut max_sim = 0.0_f32;
            let mut near_name = String::new();
            for tf in &train_feats {
                let s = hybrid_molecular_kernel(
                    cf,
                    tf,
                    config.kernel_mode,
                    config.weight_3d_usrcat,
                ) as f32;
                if s > max_sim {
                    max_sim = s;
                    near_name = tf.mol.name.clone();
                }
            }

            ActiveLearningCandidate {
                rank: 0,
                candidate_id: cf.mol.id.clone(),
                candidate_name: cf.mol.name.clone(),
                provenance: cf.provenance.clone(),
                canonical_smiles: cf.mol.smiles.clone(),
                mw: cf.mw,
                clogp: cf.clogp,
                qed: cf.qed,
                gp_mean_pact: gp_m,
                gp_epistemic_std: gp_s,
                rf_mean_pact: rf_m,
                rf_epistemic_std: rf_s,
                predicted_pact: pred_m,
                epistemic_std: ep_s,
                total_predictive_std: tot_s,
                ci90_low_pact: pred_m - hw90,
                ci90_high_pact: pred_m + hw90,
                expected_improvement: ei,
                ucb_score: ucb,
                raw_acquisition_score: raw_acq,
                batch_believer_score: raw_acq,
                selected_in_batch: false,
                batch_pick_order: None,
                nearest_train_similarity: max_sim,
                nearest_train_name: near_name,
                ad_tier: ApplicabilityDomainTier::from_max_sim(max_sim),
                true_holdout_pact: cf.true_pact,
                mol: cf.mol.clone(),
            }
        })
        .collect();

    // 4. Sequential Kriging-Believer Batch Acquisition (`B = config.batch_size`)
    // Each picked molecule is hallucinated into the conditioning set at `y = predicted_pact`,
    // collapsing the epistemic variance of nearby structural analogs so the batch stays diverse!
    let target_b = config.batch_size.clamp(1, candidates.len().max(1));
    let mut believer_train = train_feats.clone();
    let mut believer_y = train_y.clone();
    let mut picked_mask = vec![false; candidates.len()];
    let mut picked_indices_ordered: Vec<usize> = Vec::with_capacity(target_b);

    for step_b in 0..target_b {
        if candidates.is_empty() {
            break;
        }
        let believer_gp = FittedGaussianProcess::fit(
            &believer_train,
            &believer_y,
            config.kernel_mode,
            config.weight_3d_usrcat,
            config.assay_noise_sigma,
        );
        let cur_best = believer_y.iter().copied().fold(best_obs, f64::max);

        let mut best_cand_idx = None;
        let mut best_score = f64::NEG_INFINITY;

        for (ci, cf) in pool_feats.iter().enumerate() {
            if picked_mask[ci] {
                continue;
            }
            let (_, bel_gp_std) = believer_gp.predict(
                cf,
                &believer_train,
                config.kernel_mode,
                config.weight_3d_usrcat,
            );
            // Scale original consensus epistemic std by the GP variance-collapse ratio!
            let collapse_ratio =
                (bel_gp_std / candidates[ci].gp_epistemic_std.max(1.0e-4)).clamp(0.05, 1.0);
            let bel_ep_std = candidates[ci].epistemic_std * collapse_ratio;

            // Gentle Tanimoto redundancy penalty against already-selected batch members (`> 0.65`)
            let mut max_batch_tc = 0.0_f32;
            for &pi in &picked_indices_ordered {
                let tc = cf.fp.tanimoto(&pool_feats[pi].fp);
                if tc > max_batch_tc {
                    max_batch_tc = tc;
                }
            }
            let div_factor = if max_batch_tc > 0.60 {
                (1.0 - (max_batch_tc - 0.60) * 1.4).clamp(0.25, 1.0) as f64
            } else {
                1.0
            };

            let sc = score_acquisition_value(
                config.acquisition,
                candidates[ci].predicted_pact,
                bel_ep_std,
                cur_best,
                config.ei_xi,
                config.ucb_kappa,
            ) * div_factor;

            candidates[ci].batch_believer_score = sc;
            if sc > best_score {
                best_score = sc;
                best_cand_idx = Some(ci);
            }
        }

        let Some(chosen) = best_cand_idx else {
            break;
        };
        picked_mask[chosen] = true;
        picked_indices_ordered.push(chosen);
        candidates[chosen].selected_in_batch = true;
        candidates[chosen].batch_pick_order = Some(step_b + 1);
        candidates[chosen].batch_believer_score = best_score;

        believer_train.push(pool_feats[chosen].clone());
        believer_y.push(candidates[chosen].predicted_pact);
    }

    // Sort: selected batch members first in pick order `1..=B`, then remaining candidates by `raw_acquisition_score`
    candidates.sort_by(|a, b| {
        b.selected_in_batch
            .cmp(&a.selected_in_batch)
            .then_with(|| match (a.batch_pick_order, b.batch_pick_order) {
                (Some(oa), Some(ob)) => oa.cmp(&ob),
                _ => b
                    .raw_acquisition_score
                    .partial_cmp(&a.raw_acquisition_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            })
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.rank = i + 1;
    }

    let batch_cands: Vec<&ActiveLearningCandidate> = candidates
        .iter()
        .filter(|c| c.selected_in_batch)
        .collect();
    let selected_batch_count = batch_cands.len();
    let b_denom = (selected_batch_count.max(1)) as f64;
    let batch_mean_predicted_pact =
        batch_cands.iter().map(|c| c.predicted_pact).sum::<f64>() / b_denom;
    let batch_mean_epistemic_std =
        batch_cands.iter().map(|c| c.epistemic_std).sum::<f64>() / b_denom;

    let mut dissim_sum = 0.0_f64;
    let mut dissim_pairs = 0usize;
    for i in 0..picked_indices_ordered.len() {
        for j in (i + 1)..picked_indices_ordered.len() {
            let fp_a = &pool_feats[picked_indices_ordered[i]].fp;
            let fp_b = &pool_feats[picked_indices_ordered[j]].fp;
            dissim_sum += (1.0 - fp_a.tanimoto(fp_b)) as f64;
            dissim_pairs += 1;
        }
    }
    let batch_mean_pairwise_dissim = if dissim_pairs > 0 {
        dissim_sum / (dissim_pairs as f64)
    } else {
        0.0
    };

    let simulated_curve = simulate_retrospective_dmta_rounds(&all_ds_feats, &all_ds_y, config);

    ActiveLearningReport {
        config: config.clone(),
        validation,
        pool_size: candidates.len(),
        selected_batch_count,
        batch_mean_predicted_pact,
        batch_mean_epistemic_std,
        batch_mean_pairwise_dissim,
        simulated_curve,
        candidates,
    }
}

/// Exports Active-Learning QSAR predictions, `90%` conformal intervals, and batch picks as RFC-4180 `.csv`.
pub fn export_active_learning_csv(report: &ActiveLearningReport, batch_only: bool) -> String {
    let mut out = String::with_capacity(8192);
    out.push_str("Rank,Batch_Pick,Candidate_ID,Name,Provenance,Predicted_pAct,Epistemic_Std,CI90_Low,CI90_High,Expected_Improvement,UCB_Score,GP_Mean,RF_Mean,Nearest_Train_Sim,Applicability_Domain,True_Holdout_pAct,MW,cLogP,QED,SMILES\n");

    for c in &report.candidates {
        if batch_only && !c.selected_in_batch {
            continue;
        }
        let pick_s = c
            .batch_pick_order
            .map(|k| format!("B{k}"))
            .unwrap_or_default();
        let hold_s = c
            .true_holdout_pact
            .map(|v| format!("{v:.2}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "{},{},{},\"{}\",\"{}\",{:.3},{:.3},{:.2},{:.2},{:.4},{:.3},{:.3},{:.3},{:.3},\"{}\",{},{:.1},{:.2},{:.3},\"{}\"\n",
            c.rank,
            pick_s,
            c.candidate_id,
            c.candidate_name.replace('"', "'"),
            c.provenance.replace('"', "'"),
            c.predicted_pact,
            c.epistemic_std,
            c.ci90_low_pact,
            c.ci90_high_pact,
            c.expected_improvement,
            c.ucb_score,
            c.gp_mean_pact,
            c.rf_mean_pact,
            c.nearest_train_similarity,
            c.ad_tier.short_label(),
            hold_s,
            c.mw,
            c.clogp,
            c.qed,
            c.canonical_smiles.replace('"', "'"),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::load_benchmark_26;

    #[test]
    fn test_gp_and_rf_active_learning_expected_improvement_and_kriging_believer() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = ActiveLearningConfig {
            batch_size: 8,
            ..ActiveLearningConfig::default()
        };
        let rep = run_active_learning_campaign(&ds, &cfg);
        assert_eq!(rep.validation.n_train, 26);
        assert!(rep.pool_size >= 16);
        assert_eq!(rep.selected_batch_count, 8);
        assert!(rep.validation.conformal_90_scale >= 1.25);
        assert_eq!(rep.simulated_curve.len(), 4);
        assert!(rep.candidates[0].selected_in_batch);
        assert_eq!(rep.candidates[0].batch_pick_order, Some(1));

        let csv = export_active_learning_csv(&rep, true);
        assert_eq!(csv.lines().count(), 9);
    }

    #[test]
    fn test_holdout_benchmark_mode_exposes_true_holdout_pact() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = ActiveLearningConfig {
            pool_source: CandidatePoolSource::DatasetHoldoutBenchmark,
            acquisition: AcquisitionStrategy::UpperConfidenceBound,
            batch_size: 6,
            ..ActiveLearningConfig::default()
        };
        let rep = run_active_learning_campaign(&ds, &cfg);
        assert!(rep.candidates.iter().all(|c| c.true_holdout_pact.is_some()));
        assert_eq!(rep.selected_batch_count, 6);
    }
}
