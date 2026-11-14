//! Pure-Rust 3D Distance-Geometry conformer generator (`ETKDG`-inspired torsional bounds +
//! triangle-inequality smoothing + Crippen-Havel metric-matrix embedding), analytical-gradient
//! `MMFF-lite` molecular mechanics minimizer with Gasteiger-Marsili `PEOE` partial charges,
//! Ballester-Richards Ultrafast Shape Recognition (`USR` 12-D & Schreyer-Blundell `USRCAT` 60-D),
//! Sauer-Schwarz Normalized Principal Moments (`NPR1`/`NPR2`) & Plane of Best Fit (`PBF`), and
//! Grant-Pickup / `ROCS` Gaussian 3D shape + chemical-color/electrostatic rigid-body superposition.
//!
//! Primary References:
//! - Riniker & Landrum (2015), *J. Chem. Inf. Model.* 55(12), 2562-2574 (`ETKDG` Distance Geometry:
//!   `https://pubmed.ncbi.nlm.nih.gov/26575315/`)
//! - Halgren (1996), *J. Comput. Chem.* 17(5-6), 490-519 (`MMFF94` Molecular Mechanics:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC4026342/`)
//! - Gasteiger & Marsili (1980), *Tetrahedron* 36, 3219-3228 (`PEOE` Partial Equalization of
//!   Orbital Electronegativity: `https://pubs.acs.org/jcisd8/article/43/6/1982/3805393/Fast-Assignment-of-Accurate-Partial-Atomic-Charges`)
//! - Ballester & Richards (2007), *J. Comput. Chem.* 28(10), 1711-1723 (`USR`:
//!   `https://pubmed.ncbi.nlm.nih.gov/17342716/`) & Schreyer & Blundell (2012), *J. Cheminform.* 4:27
//!   (`USRCAT`: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3505738/`)
//! - Grant, Gallardo & Pickup (1996) / Hawkins, Skillman & Nicholls (2007), *J. Med. Chem.* 50(1),
//!   74-82 (`ROCS` Gaussian Shape & Color Overlay: `https://pubmed.ncbi.nlm.nih.gov/17201411/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC6068280/`)
//! - Horn (1987) / Kabsch-Umeyama 3D Quaternion Rigid-Body Alignment (`https://pmc.ncbi.nlm.nih.gov/articles/PMC12951576/`)

use crate::fingerprint::compute_ecfp;
use crate::mol::{BondOrder, Element, Hybridization, Molecule};
use crate::scaffold::murcko_signatures;
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Dimensionality of the Ballester-Richards (2007) Ultrafast Shape Recognition (`USR`) descriptor.
pub const USR_DIM: usize = 12;

/// Dimensionality of the Schreyer-Blundell (2012) pharmacophore-extended `USRCAT` descriptor
/// (`All`, `Hydrophobic`, `Aromatic`, `H-Bond Acceptor`, `H-Bond Donor` x 12 moments = 60).
pub const USRCAT_DIM: usize = 60;

/// Grant-Pickup (1996) Gaussian amplitude `p` and exponent scale `kappa` (`alpha_i = kappa / R_i^2`).
const GAUSS_P: f64 = 2.70;
const GAUSS_KAPPA: f64 = 2.41798793;

/// Chemical feature ("color atom") types for `ROCS`-style 3D pharmacophoric & electrostatic overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RocsColorKind {
    HBondDonor,
    HBondAcceptor,
    Cation,
    Anion,
    AromaticRing,
    Hydrophobic,
}

impl RocsColorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::HBondDonor => "H-Bond Donor",
            Self::HBondAcceptor => "H-Bond Acceptor",
            Self::Cation => "Cation (+)",
            Self::Anion => "Anion (-)",
            Self::AromaticRing => "Aromatic Ring",
            Self::Hydrophobic => "Hydrophobic",
        }
    }

    pub fn short_code(self) -> &'static str {
        match self {
            Self::HBondDonor => "HBD",
            Self::HBondAcceptor => "HBA",
            Self::Cation => "POS",
            Self::Anion => "NEG",
            Self::AromaticRing => "AROM",
            Self::Hydrophobic => "HYD",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::HBondDonor => [56, 189, 248],   // Sky Blue
            Self::HBondAcceptor => [248, 113, 113], // Coral Red
            Self::Cation => [96, 165, 250],       // Royal Blue
            Self::Anion => [251, 113, 133],       // Rose
            Self::AromaticRing => [251, 191, 36], // Amber
            Self::Hydrophobic => [52, 211, 153],  // Emerald
        }
    }
}

/// 3D chemical feature center used in `ROCS` Color Tanimoto scoring and 3D viewport spheres.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RocsColorAtom {
    pub kind: RocsColorKind,
    pub pos3d: [f32; 3],
    #[serde(skip_deserializing)]
    pub label: &'static str,
}

/// Six-term `MMFF-lite` potential energy decomposition (in `kcal/mol`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MmffEnergyBreakdown {
    pub stretch_kcal: f64,
    pub bend_kcal: f64,
    pub oop_kcal: f64,
    pub torsion_kcal: f64,
    pub vdw_kcal: f64,
    pub electro_kcal: f64,
    pub total_kcal: f64,
    pub rms_grad: f64,
    pub iterations: usize,
}

/// Single minimized 3D conformer with geometric, inertial, electronic, and `USR`/`USRCAT` descriptors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conformer3d {
    pub rank: usize,
    pub seed_id: usize,
    pub coords3d: Vec<[f32; 3]>,
    pub energy: MmffEnergyBreakdown,
    pub rel_energy_kcal: f64,
    pub boltzmann_weight: f64,
    pub rmsd_to_lowest: f32,
    pub rg_angstrom: f32,
    pub pbf_angstrom: f32,
    pub npr1: f32,
    pub npr2: f32,
    pub gaussian_self_volume: f64,
    pub dipole_debye: f32,
    pub usr12: [f32; USR_DIM],
    pub usrcat60: Vec<f32>,
    pub color_atoms: Vec<RocsColorAtom>,
}

/// Configuration for `ETKDG`-inspired Distance-Geometry 3D conformer generation & `MMFF-lite` relaxation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ConformerGenConfig {
    pub num_trial_conformers: usize,
    pub max_kept_conformers: usize,
    pub mmff_max_iters: usize,
    pub rmsd_prune_cutoff: f32,
    pub energy_window_kcal: f64,
    pub random_seed: u64,
}

impl Default for ConformerGenConfig {
    fn default() -> Self {
        Self {
            num_trial_conformers: 8,
            max_kept_conformers: 5,
            mmff_max_iters: 110,
            rmsd_prune_cutoff: 0.45,
            energy_window_kcal: 12.0,
            random_seed: 0x5A17_3D12,
        }
    }
}

/// Deduplicated, Boltzmann-weighted 3D conformer ensemble for a small molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConformerEnsemble {
    pub gasteiger_charges: Vec<f32>,
    pub rotatable_dihedrals: usize,
    pub trial_count: usize,
    pub conformers: Vec<Conformer3d>,
}

/// Dataset-level 3D geometric, electrostatic, and `USR`/`USRCAT` summary stored on `DatasetAnalysis`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Summary3dProfile {
    pub coords3d: Vec<[f32; 3]>,
    pub gasteiger_charges: Vec<f32>,
    pub mmff_energy_kcal: f64,
    pub rg_angstrom: f32,
    pub pbf_angstrom: f32,
    pub npr1: f32,
    pub npr2: f32,
    #[serde(skip_deserializing)]
    pub shape_class: &'static str,
    pub dipole_debye: f32,
    pub gaussian_volume: f64,
    pub usr12: [f32; USR_DIM],
    pub usrcat60: Vec<f32>,
    pub color_atoms: Vec<RocsColorAtom>,
}

/// Optimal rigid-body 3D `ROCS` Gaussian shape + color/electrostatic superposition result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rocs3dAlignment {
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
    pub aligned_hit_coords: Vec<[f32; 3]>,
    pub aligned_hit_colors: Vec<RocsColorAtom>,
    pub best_hit_conformer_rank: usize,
    pub shape_overlap_vol: f64,
    pub shape_tanimoto: f64,
    pub shape_tversky_query: f64,
    pub color_feature_tanimoto: f64,
    pub esp_hodgkin_sim: f64,
    pub color_tanimoto: f64,
    pub combo_score: f64,
    pub combo_tanimoto: f64,
    pub usr_similarity: f64,
    pub usrcat_similarity: f64,
    pub centroid_rmsd: f32,
}

/// Ranked 3D virtual-screening / scaffold-hopping hit against a query 3D conformer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Overlay3dHit {
    pub mol_idx: usize,
    pub compound_id: String,
    pub compound_name: String,
    pub target: String,
    pub pact: f64,
    pub ecfp4_tanimoto_2d: f32,
    pub is_scaffold_hop: bool,
    pub npr1: f32,
    pub npr2: f32,
    pub pbf_angstrom: f32,
    pub dipole_debye: f32,
    pub alignment: Rocs3dAlignment,
}

/// Bondi (1964) van der Waals atomic radii (in Angstroms) used for steric bounds and Gaussian volumes.
pub fn bondi_vdw_radius(elem: Element) -> f32 {
    match elem {
        Element::H => 1.20,
        Element::B => 1.92,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::F => 1.47,
        Element::Si => 2.10,
        Element::P => 1.80,
        Element::S => 1.80,
        Element::Cl => 1.75,
        Element::Se => 1.90,
        Element::Br => 1.85,
        Element::I => 1.98,
        Element::Other(_) => 1.75,
    }
}

/// Single-atom covalent radii (in Angstroms) for equilibrium bond-length lookup.
fn covalent_radius(elem: Element, hyb: Hybridization, aromatic: bool) -> f64 {
    match elem {
        Element::H => 0.31,
        Element::B => 0.84,
        Element::C => {
            if aromatic {
                0.70
            } else {
                match hyb {
                    Hybridization::Sp => 0.69,
                    Hybridization::Sp2 => 0.73,
                    _ => 0.765,
                }
            }
        }
        Element::N => {
            if aromatic {
                0.68
            } else {
                match hyb {
                    Hybridization::Sp => 0.65,
                    Hybridization::Sp2 => 0.70,
                    _ => 0.71,
                }
            }
        }
        Element::O => match hyb {
            Hybridization::Sp2 => 0.63,
            _ => 0.66,
        },
        Element::F => 0.60,
        Element::Si => 1.11,
        Element::P => 1.07,
        Element::S => 1.04,
        Element::Cl => 1.00,
        Element::Se => 1.19,
        Element::Br => 1.15,
        Element::I => 1.34,
        Element::Other(_) => 0.85,
    }
}

/// Equilibrium bond length `r0` (Angstroms) and stretching force constant `kb` (`kcal/(mol*A^2)`).
fn ideal_bond_params(mol: &Molecule, b_idx: usize) -> (f64, f64) {
    let b = &mol.bonds[b_idx];
    let a1 = &mol.atoms[b.a];
    let a2 = &mol.atoms[b.b];
    let r_sum =
        covalent_radius(a1.elem, a1.hyb, a1.aromatic) + covalent_radius(a2.elem, a2.hyb, a2.aromatic);

    // Pauling-Schomaker-Stevenson bond-order contraction & electronegativity shortening
    let (bo_delta, kb) = if b.aromatic {
        (-0.015, 480.0)
    } else {
        match b.order {
            BondOrder::Single => {
                // Amide / aniline /aryl-aryl partial double-bond character
                let conj = a1.hyb == Hybridization::Sp2 && a2.hyb == Hybridization::Sp2;
                if conj {
                    (-0.045, 410.0)
                } else {
                    (0.0, 340.0)
                }
            }
            BondOrder::Double => (-0.135, 580.0),
            BondOrder::Triple => (-0.235, 760.0),
            BondOrder::Aromatic => (-0.015, 480.0),
        }
    };
    let r0 = (r_sum + bo_delta).clamp(1.05, 2.35);
    (r0, kb)
}

/// Ideal valence angle `theta0` (radians) at central atom `j` for triple `i - j - k`.
fn ideal_valence_angle_rad(mol: &Molecule, i: usize, j: usize, k: usize) -> (f64, f64) {
    let aj = &mol.atoms[j];
    // Check if i, j, k belong to a common small SSSR ring (3, 4, or 5-membered)
    for ring in &mol.rings {
        let len = ring.len();
        if len <= 5 && ring.contains(&i) && ring.contains(&j) && ring.contains(&k) {
            let deg: f64 = match len {
                3 => 60.0,
                4 => 89.5,
                5 => {
                    if aj.aromatic || aj.hyb == Hybridization::Sp2 {
                        108.0
                    } else {
                        104.5
                    }
                }
                _ => 109.47,
            };
            return (deg.to_radians(), 95.0);
        }
    }

    let deg: f64 = match aj.hyb {
        Hybridization::Sp => 180.0,
        Hybridization::Sp2 => 120.0,
        Hybridization::Sp3 => match aj.elem {
            Element::O | Element::S | Element::Se if aj.charge == 0 => 104.5,
            Element::N if aj.charge == 0 => 107.5,
            _ => 109.47,
        },
        Hybridization::Other => 109.47,
    };
    let k_ang = if aj.hyb == Hybridization::Sp { 55.0 } else { 80.0 };
    (deg.to_radians(), k_ang)
}

/// Iterative Gasteiger-Marsili (1980) Partial Equalization of Orbital Electronegativity (`PEOE`)
/// atomic charges, accounting for attached implicit hydrogens and symmetric resonance oxygens/nitrogens.
pub fn compute_gasteiger_charges(mol: &Molecule) -> Vec<f32> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }

    // Orbital electronegativity polynomial coefficients (a, b, c) in eV: chi(q) = a + b*q + c*q^2
    let params: Vec<(f64, f64, f64)> = mol
        .atoms
        .iter()
        .map(|a| match (a.elem, a.hyb, a.aromatic) {
            (Element::C, Hybridization::Sp, _) => (10.39, 9.45, 0.73),
            (Element::C, Hybridization::Sp2, _) | (Element::C, _, true) => (8.79, 9.32, 1.51),
            (Element::C, _, _) => (7.98, 9.18, 1.88),
            (Element::N, Hybridization::Sp, _) => (15.68, 11.70, -0.27),
            (Element::N, Hybridization::Sp2, _) | (Element::N, _, true) => (12.87, 11.15, 0.85),
            (Element::N, _, _) => (11.54, 10.82, 1.36),
            (Element::O, Hybridization::Sp2, _) | (Element::O, _, true) => (17.07, 13.79, 0.47),
            (Element::O, _, _) => (14.18, 12.92, 1.39),
            (Element::F, _, _) => (14.66, 13.85, 2.31),
            (Element::Cl, _, _) => (11.00, 9.69, 1.35),
            (Element::Br, _, _) => (10.08, 8.47, 1.16),
            (Element::I, _, _) => (9.90, 7.96, 0.96),
            (Element::S, _, _) => (10.14, 9.13, 1.38),
            (Element::P, _, _) => (8.90, 8.24, 0.96),
            (Element::B, _, _) => (5.98, 6.82, 1.60),
            (Element::Si, _, _) => (7.30, 6.56, 0.65),
            (Element::Se, _, _) => (9.80, 8.60, 1.18),
            (Element::H, _, _) => (7.17, 6.24, -0.56),
            (Element::Other(_), _, _) => (8.50, 8.50, 1.20),
        })
        .collect();

    // Seed initial charges from formal charges, splitting resonance-equivalent terminal oxygens
    let mut q_heavy: Vec<f64> = mol.atoms.iter().map(|a| a.charge as f64).collect();
    let mut q_hyd: Vec<Vec<f64>> = mol
        .atoms
        .iter()
        .map(|a| vec![0.0f64; a.total_h() as usize])
        .collect();

    let d_plus_h = 20.02f64;
    let mut damp = 0.5f64;

    for _iter in 0..8 {
        let chi_heavy: Vec<f64> = (0..n)
            .map(|i| {
                let (a, b, c) = params[i];
                let q = q_heavy[i].clamp(-1.5, 1.5);
                a + b * q + c * q * q
            })
            .collect();

        let mut dq_heavy = vec![0.0f64; n];

        // Heavy-heavy bond charge transfer
        for b in &mol.bonds {
            let u = b.a;
            let v = b.b;
            let cu = chi_heavy[u];
            let cv = chi_heavy[v];
            if (cu - cv).abs() < 1e-9 {
                continue;
            }
            // Normalize by cationic electronegativity of the less electronegative atom
            let donor = if cu < cv { u } else { v };
            let (da, db, dc) = params[donor];
            let denom = (da + db + dc).max(6.0);
            let delta = damp * (cv - cu) / denom;
            dq_heavy[u] += delta;
            dq_heavy[v] -= delta;
        }

        // Heavy-implicit-H bond charge transfer
        for u in 0..n {
            let cu = chi_heavy[u];
            let (da, db, dc) = params[u];
            let denom_u = (da + db + dc).max(6.0);
            for qh in &mut q_hyd[u] {
                let q_clamp = qh.clamp(-1.0, 1.0);
                let ch = 7.17 + 6.24 * q_clamp - 0.56 * q_clamp * q_clamp;
                let denom = if ch < cu { d_plus_h } else { denom_u };
                let delta = damp * (cu - ch) / denom;
                *qh += delta;
                dq_heavy[u] -= delta;
            }
        }

        for u in 0..n {
            q_heavy[u] += dq_heavy[u];
        }
        damp *= 0.5;
    }

    // Fold implicit-H partial charges onto heavy atoms for condensed-graph electrostatic overlap
    let mut total_condensed: Vec<f64> = (0..n)
        .map(|u| q_heavy[u] + q_hyd[u].iter().sum::<f64>())
        .collect();

    // Symmetrize resonance-equivalent terminal oxygens (carboxylates, sulfonamides, nitro)
    for u in 0..n {
        let term_oxy: Vec<usize> = mol.adj[u]
            .iter()
            .map(|&(nb, _)| nb)
            .filter(|&nb| {
                mol.atoms[nb].elem == Element::O
                    && mol.adj[nb].len() == 1
                    && mol.atoms[nb].total_h() == 0
            })
            .collect();
        if term_oxy.len() >= 2 {
            let avg = term_oxy.iter().map(|&o| total_condensed[o]).sum::<f64>()
                / (term_oxy.len() as f64);
            for o in term_oxy {
                total_condensed[o] = avg;
            }
        }
    }

    total_condensed.into_iter().map(|q| q as f32).collect()
}

/// Computes the 3D molecular dipole moment magnitude (in Debye; `1 e*A = 4.803205 D`)
/// relative to the center of geometry so charged ions also have a translation-invariant moment.
pub fn compute_dipole_debye(coords3d: &[[f32; 3]], charges: &[f32]) -> f32 {
    if coords3d.is_empty() || coords3d.len() != charges.len() {
        return 0.0;
    }
    let inv_n = 1.0 / (coords3d.len() as f64);
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut cz = 0.0f64;
    for p in coords3d {
        cx += p[0] as f64;
        cy += p[1] as f64;
        cz += p[2] as f64;
    }
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    let mut dx = 0.0f64;
    let mut dy = 0.0f64;
    let mut dz = 0.0f64;
    for (p, &q) in coords3d.iter().zip(charges.iter()) {
        let qf = q as f64;
        dx += qf * ((p[0] as f64) - cx);
        dy += qf * ((p[1] as f64) - cy);
        dz += qf * ((p[2] as f64) - cz);
    }
    ((dx * dx + dy * dy + dz * dz).sqrt() * 4.803205) as f32
}

/// Deterministic SplitMix64 PRNG for reproducible conformer distance-matrix sampling.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[derive(Clone)]
struct TorsionTerm {
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    v1: f64,
    v2: f64,
    v3: f64,
    is_rotatable: bool,
}

#[derive(Clone)]
struct ImproperTerm {
    center: usize,
    n1: usize,
    n2: usize,
    n3: usize,
    k_oop: f64,
    target_vol_sign: i8, // 0 = planar sp2; +1/-1 = tetrahedral chiral sign restraint
}

struct ForceFieldTopology {
    bonds: Vec<(usize, usize, f64, f64)>,
    angles: Vec<(usize, usize, usize, f64, f64)>,
    torsions: Vec<TorsionTerm>,
    impropers: Vec<ImproperTerm>,
    nonbonded: Vec<(usize, usize, f64, f64, f64, f64)>, // (u, v, r_star, eps, vdw_scale, elec_scale)
    rotatable_bonds: usize,
}

impl ForceFieldTopology {
    fn build(mol: &Molecule) -> Self {
        let n = mol.atoms.len();
        let mut bonds = Vec::with_capacity(mol.bonds.len());
        let mut bond_lookup = vec![vec![None; n]; n];

        for (bi, b) in mol.bonds.iter().enumerate() {
            let (r0, kb) = ideal_bond_params(mol, bi);
            bonds.push((b.a, b.b, r0, kb));
            bond_lookup[b.a][b.b] = Some((bi, r0));
            bond_lookup[b.b][b.a] = Some((bi, r0));
        }

        // Topological shortest path distances (in bond steps) to classify 1-2, 1-3, 1-4, >=1-5 pairs
        let mut topo_dist = vec![vec![usize::MAX; n]; n];
        for s in 0..n {
            topo_dist[s][s] = 0;
            let mut q = VecDeque::new();
            q.push_back(s);
            while let Some(u) = q.pop_front() {
                let du = topo_dist[s][u];
                for &(nb, _) in &mol.adj[u] {
                    if topo_dist[s][nb] == usize::MAX {
                        topo_dist[s][nb] = du + 1;
                        q.push_back(nb);
                    }
                }
            }
        }

        // Valence angles i - j - k
        let mut angles = Vec::new();
        for j in 0..n {
            let deg = mol.adj[j].len();
            if deg < 2 {
                continue;
            }
            for a_idx in 0..deg {
                for b_idx in (a_idx + 1)..deg {
                    let i = mol.adj[j][a_idx].0;
                    let k = mol.adj[j][b_idx].0;
                    let (theta0, k_ang) = ideal_valence_angle_rad(mol, i, j, k);
                    angles.push((i, j, k, theta0.cos(), k_ang));
                }
            }
        }

        // Proper torsions i - j - k - l
        let mut torsions = Vec::new();
        let mut rotatable_bonds = 0usize;
        for b in &mol.bonds {
            let j = b.a;
            let k = b.b;
            let aj = &mol.atoms[j];
            let ak = &mol.atoms[k];
            if aj.hyb == Hybridization::Sp || ak.hyb == Hybridization::Sp {
                continue;
            }
            let deg_j = mol.adj[j].len();
            let deg_k = mol.adj[k].len();
            if deg_j < 2 || deg_k < 2 {
                continue;
            }

            let is_amide = (aj.elem == Element::C
                && ak.elem == Element::N
                && mol.adj[j].iter().any(|&(nb, bi)| {
                    mol.atoms[nb].elem == Element::O && mol.bonds[bi].order == BondOrder::Double
                }))
                || (ak.elem == Element::C
                    && aj.elem == Element::N
                    && mol.adj[k].iter().any(|&(nb, bi)| {
                        mol.atoms[nb].elem == Element::O
                            && mol.bonds[bi].order == BondOrder::Double
                    }));

            let is_rot = !b.in_ring && b.order == BondOrder::Single && !b.aromatic && !is_amide;
            if is_rot {
                rotatable_bonds += 1;
            }

            let pair_scale = 1.0 / (((deg_j - 1) * (deg_k - 1)).max(1) as f64);
            let (v1, v2, v3) = if b.aromatic || b.order == BondOrder::Double {
                (0.0, 14.0 * pair_scale, 0.0)
            } else if is_amide {
                (0.0, 10.0 * pair_scale, 0.0)
            } else if aj.hyb == Hybridization::Sp2 && ak.hyb == Hybridization::Sp2 {
                // Biaryl / conjugated single bond
                (0.0, 3.2 * pair_scale, 0.0)
            } else if aj.hyb == Hybridization::Sp3 && ak.hyb == Hybridization::Sp3 {
                (0.35 * pair_scale, 0.0, 1.65 * pair_scale)
            } else {
                // sp2 - sp3
                (0.0, 0.65 * pair_scale, 0.85 * pair_scale)
            };

            for &(i, _) in &mol.adj[j] {
                if i == k {
                    continue;
                }
                for &(l, _) in &mol.adj[k] {
                    if l == j || l == i {
                        continue;
                    }
                    torsions.push(TorsionTerm {
                        i,
                        j,
                        k,
                        l,
                        v1,
                        v2,
                        v3,
                        is_rotatable: is_rot,
                    });
                }
            }
        }

        // Out-of-plane planarity & tetrahedral chirality restraints
        let mut impropers = Vec::new();
        for u in 0..n {
            let au = &mol.atoms[u];
            if (au.aromatic || au.hyb == Hybridization::Sp2) && mol.adj[u].len() == 3 {
                impropers.push(ImproperTerm {
                    center: u,
                    n1: mol.adj[u][0].0,
                    n2: mol.adj[u][1].0,
                    n3: mol.adj[u][2].0,
                    k_oop: 28.0,
                    target_vol_sign: 0,
                });
            } else if au.hyb == Hybridization::Sp3 && au.chiral != 0 && mol.adj[u].len() >= 3 {
                impropers.push(ImproperTerm {
                    center: u,
                    n1: mol.adj[u][0].0,
                    n2: mol.adj[u][1].0,
                    n3: mol.adj[u][2].0,
                    k_oop: 18.0,
                    target_vol_sign: au.chiral.signum(),
                });
            }
        }

        // Non-bonded pairs (1-4 scaled & >=1-5 unscaled)
        let mut nonbonded = Vec::new();
        for u in 0..n {
            let ru = bondi_vdw_radius(mol.atoms[u].elem) as f64;
            let eu = vdw_well_depth(mol.atoms[u].elem);
            for v in (u + 1)..n {
                let td = topo_dist[u][v];
                if td < 3 {
                    continue;
                }
                let rv = bondi_vdw_radius(mol.atoms[v].elem) as f64;
                let ev = vdw_well_depth(mol.atoms[v].elem);
                let r_star = ru + rv;
                let eps = (eu * ev).sqrt();
                let (vdw_sc, elec_sc) = if td == 3 { (0.50, 0.75) } else { (1.0, 1.0) };
                nonbonded.push((u, v, r_star, eps, vdw_sc, elec_sc));
            }
        }

        Self {
            bonds,
            angles,
            torsions,
            impropers,
            nonbonded,
            rotatable_bonds,
        }
    }
}

fn vdw_well_depth(elem: Element) -> f64 {
    match elem {
        Element::H => 0.020,
        Element::C => 0.068,
        Element::N => 0.065,
        Element::O => 0.075,
        Element::F => 0.055,
        Element::Cl => 0.210,
        Element::Br => 0.300,
        Element::I => 0.400,
        Element::S | Element::P | Element::Se => 0.200,
        _ => 0.080,
    }
}

/// Computes `1-4` cis (`phi = 0`) and trans (`phi = pi`) distances for four sequential atoms.
fn cis_trans_14_distances(
    r12: f64,
    r23: f64,
    r34: f64,
    theta123: f64,
    theta234: f64,
    phi_rad: f64,
) -> f64 {
    // Place j at (0,0,0), k at (r23, 0, 0), i in xy-plane
    let ix = r12 * theta123.cos();
    let iy = r12 * theta123.sin();
    let lx = r23 - r34 * theta234.cos();
    let ly = r34 * theta234.sin() * phi_rad.cos();
    let lz = r34 * theta234.sin() * phi_rad.sin();
    let dx = ix - lx;
    let dy = iy - ly;
    (dx * dx + dy * dy + lz * lz).sqrt()
}

/// Embeds initial 3D heavy-atom coordinates via `ETKDG`-inspired Distance Geometry:
/// 1-2/1-3/1-4 bounds + crystallographic torsion sampling + Floyd-Warshall triangle smoothing
/// + Crippen-Havel metric matrix top-3 eigenprojection.
fn embed_distance_geometry_3d(
    mol: &Molecule,
    topo: &ForceFieldTopology,
    rng: &mut SplitMix64,
) -> Vec<[f64; 3]> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![[0.0, 0.0, 0.0]];
    }
    if n == 2 {
        let r0 = topo.bonds.first().map(|b| b.2).unwrap_or(1.50);
        return vec![[-0.5 * r0, 0.0, 0.0], [0.5 * r0, 0.0, 0.0]];
    }

    let max_bound = (n as f64) * 1.65 + 4.0;
    let mut lb = vec![vec![0.0f64; n]; n];
    let mut ub = vec![vec![max_bound; n]; n];
    for u in 0..n {
        ub[u][u] = 0.0;
        let ru = bondi_vdw_radius(mol.atoms[u].elem) as f64;
        for v in (u + 1)..n {
            let rv = bondi_vdw_radius(mol.atoms[v].elem) as f64;
            let contact = 0.78 * (ru + rv);
            lb[u][v] = contact;
            lb[v][u] = contact;
        }
    }

    // 1-2 exact bond bounds
    let mut bond_r0 = vec![vec![0.0f64; n]; n];
    for &(u, v, r0, _) in &topo.bonds {
        bond_r0[u][v] = r0;
        bond_r0[v][u] = r0;
        lb[u][v] = (r0 - 0.015).max(0.9);
        lb[v][u] = lb[u][v];
        ub[u][v] = r0 + 0.015;
        ub[v][u] = ub[u][v];
    }

    // 1-3 geminal angle bounds via law of cosines
    for &(i, j, k, cos_th0, _) in &topo.angles {
        let r1 = bond_r0[i][j];
        let r2 = bond_r0[j][k];
        if r1 > 0.0 && r2 > 0.0 {
            let d13 = (r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * cos_th0).max(0.5).sqrt();
            lb[i][k] = lb[i][k].max(d13 - 0.045);
            lb[k][i] = lb[i][k];
            ub[i][k] = ub[i][k].min(d13 + 0.045).max(lb[i][k]);
            ub[k][i] = ub[i][k];
        }
    }

    // 1-4 torsional bounds (planar aromatic/double vs chair aliphatic ring vs ETKDG staggered rotamers)
    for tor in &topo.torsions {
        let (i, j, k, l) = (tor.i, tor.j, tor.k, tor.l);
        if bond_r0[i][l] > 0.0 {
            continue;
        }
        let r12 = bond_r0[i][j];
        let r23 = bond_r0[j][k];
        let r34 = bond_r0[k][l];
        if r12 <= 0.0 || r23 <= 0.0 || r34 <= 0.0 {
            continue;
        }
        let (th1, _) = ideal_valence_angle_rad(mol, i, j, k);
        let (th2, _) = ideal_valence_angle_rad(mol, j, k, l);
        let d_cis = cis_trans_14_distances(r12, r23, r34, th1, th2, 0.0);
        let d_trans = cis_trans_14_distances(r12, r23, r34, th1, th2, PI);

        let shared_arom_ring = mol
            .rings
            .iter()
            .find(|r| r.contains(&i) && r.contains(&j) && r.contains(&k) && r.contains(&l));

        if let Some(ring) = shared_arom_ring {
            let ring_planar = ring.iter().all(|&ra| {
                mol.atoms[ra].aromatic || mol.atoms[ra].hyb == Hybridization::Sp2
            });
            if ring_planar {
                // Planar cis inside aromatic / unsaturated ring
                lb[i][l] = lb[i][l].max(d_cis - 0.06);
                ub[i][l] = ub[i][l].min(d_cis + 0.06).max(lb[i][l]);
            } else {
                // Chair / envelope puckered 5- or 6-membered saturated ring (gauche ~ 56 deg)
                let d_chair = cis_trans_14_distances(r12, r23, r34, th1, th2, 56.0f64.to_radians());
                lb[i][l] = lb[i][l].max(d_chair - 0.14);
                ub[i][l] = ub[i][l].min(d_chair + 0.14).max(lb[i][l]);
            }
            lb[l][i] = lb[i][l];
            ub[l][i] = ub[i][l];
        } else if tor.is_rotatable {
            // ETKDG knowledge-informed preferred dihedral rotamer sampling
            let preferred_deg: &[f64] =
                if mol.atoms[j].hyb == Hybridization::Sp3 && mol.atoms[k].hyb == Hybridization::Sp3 {
                    &[60.0, 180.0, 180.0, 300.0]
                } else if mol.atoms[j].hyb == Hybridization::Sp2
                    && mol.atoms[k].hyb == Hybridization::Sp2
                {
                    &[35.0, 145.0, 215.0, 325.0]
                } else {
                    &[0.0, 90.0, 180.0, 270.0]
                };
            let pick = preferred_deg[(rng.next_u64() as usize) % preferred_deg.len()];
            let d_pref = cis_trans_14_distances(r12, r23, r34, th1, th2, pick.to_radians());
            let lo = d_cis.min(d_pref - 0.22).max(lb[i][l]);
            let hi = d_trans.max(d_pref + 0.22).min(ub[i][l]).max(lo + 0.02);
            lb[i][l] = lo;
            ub[i][l] = hi;
            lb[l][i] = lo;
            ub[l][i] = hi;
        } else {
            let lo = d_cis.max(lb[i][l]);
            let hi = d_trans.min(ub[i][l]).max(lo + 0.02);
            lb[i][l] = lo;
            ub[i][l] = hi;
            lb[l][i] = lo;
            ub[l][i] = hi;
        }
    }

    // Floyd-Warshall Triangle-Inequality Bounds Smoothing (Dress & Havel)
    for k in 0..n {
        for i in 0..n {
            if i == k {
                continue;
            }
            let u_ik = ub[i][k];
            let l_ik = lb[i][k];
            for j in (i + 1)..n {
                if j == k {
                    continue;
                }
                let cand_u = u_ik + ub[k][j];
                if cand_u < ub[i][j] {
                    ub[i][j] = cand_u;
                    ub[j][i] = cand_u;
                }
                let cand_l = (l_ik - ub[k][j]).max(lb[k][j] - u_ik);
                if cand_l > lb[i][j] {
                    let clamped = cand_l.min(ub[i][j]);
                    lb[i][j] = clamped;
                    lb[j][i] = clamped;
                }
            }
        }
    }

    // Sample symmetric trial distance matrix D_ij in [L_ij, U_ij]
    let mut dist_sq = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let lo = lb[i][j];
            let hi = ub[i][j].max(lo);
            let t = 0.20 + 0.60 * rng.next_f64();
            let d = lo + t * (hi - lo);
            let d2 = d * d;
            dist_sq[i][j] = d2;
            dist_sq[j][i] = d2;
        }
    }

    // Crippen-Havel Metric Matrix G_ij = 0.5 * (D0_i^2 + D0_j^2 - D_ij^2)
    let inv_n = 1.0 / (n as f64);
    let mut total_sum = 0.0f64;
    let mut row_sums = vec![0.0f64; n];
    for i in 0..n {
        let s: f64 = dist_sq[i].iter().sum();
        row_sums[i] = s;
        total_sum += s;
    }
    let mean_all = total_sum * inv_n * inv_n;
    let d0_sq: Vec<f64> = (0..n).map(|i| row_sums[i] * inv_n - 0.5 * mean_all).collect();

    let mut gram = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let g = 0.5 * (d0_sq[i] + d0_sq[j] - dist_sq[i][j]);
            gram[i][j] = g;
            gram[j][i] = g;
        }
    }

    // Top-3 deflated power-iteration eigenpairs of the symmetric Gram matrix
    let mut coords = vec![[0.0f64; 3]; n];
    for dim in 0..3 {
        let mut v: Vec<f64> = (0..n)
            .map(|i| (((i + 1) * (dim + 3)) as f64 * 0.6180339887).sin() + 0.1 * rng.next_f64())
            .collect();
        normalize_vec(&mut v);

        let mut eig = 0.0f64;
        for _ in 0..48 {
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                let mut acc = 0.0f64;
                for j in 0..n {
                    acc += gram[i][j] * v[j];
                }
                w[i] = acc;
            }
            eig = dot_vec(&v, &w);
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for i in 0..n {
                v[i] = w[i] / norm;
            }
        }

        let scale = eig.max(0.04).sqrt();
        for i in 0..n {
            coords[i][dim] = v[i] * scale;
        }

        // Hotelling deflation: G <- G - eig * v * v^T
        if eig > 0.0 {
            for i in 0..n {
                for j in i..n {
                    let sub = eig * v[i] * v[j];
                    gram[i][j] -= sub;
                    gram[j][i] = gram[i][j];
                }
            }
        }
    }

    // Ensure non-zero Z dispersion so planar seeds still allow out-of-plane ring puckering & sp3 tetrahedrality
    let z_var: f64 = coords.iter().map(|p| p[2] * p[2]).sum::<f64>() * inv_n;
    if z_var < 0.03 {
        for (i, a) in mol.atoms.iter().enumerate() {
            if !a.aromatic && a.hyb == Hybridization::Sp3 {
                coords[i][2] += (rng.next_f64() - 0.5) * 0.65;
            } else {
                coords[i][2] += (rng.next_f64() - 0.5) * 0.12;
            }
        }
    }

    coords
}

fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_vec(v: &mut [f64]) {
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nrm > 1e-12 {
        for x in v {
            *x /= nrm;
        }
    }
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Evaluates all 6 `MMFF-lite` potential energy terms and accumulates exact analytical Cartesian gradients.
fn eval_mmff_energy_and_grad(
    coords: &[[f64; 3]],
    topo: &ForceFieldTopology,
    charges: &[f32],
    grad: &mut [[f64; 3]],
) -> MmffEnergyBreakdown {
    for g in grad.iter_mut() {
        *g = [0.0, 0.0, 0.0];
    }
    let mut eb = MmffEnergyBreakdown::default();

    // 1. Bond Stretch: E = 0.5 * kb * (r - r0)^2
    for &(u, v, r0, kb) in &topo.bonds {
        let d = sub3(coords[u], coords[v]);
        let r = norm3(d).max(1e-6);
        let dr = r - r0;
        eb.stretch_kcal += 0.5 * kb * dr * dr;
        let f = kb * dr / r;
        for c in 0..3 {
            let gc = f * d[c];
            grad[u][c] += gc;
            grad[v][c] -= gc;
        }
    }

    // 2. Cosine-Harmonic Valence Angle Bend: E = 0.5 * k_ang * (cos_theta - cos_theta0)^2
    for &(i, j, k, cos0, k_ang) in &topo.angles {
        let u = sub3(coords[i], coords[j]);
        let v = sub3(coords[k], coords[j]);
        let ru = norm3(u).max(1e-6);
        let rv = norm3(v).max(1e-6);
        let cos_th = (dot3(u, v) / (ru * rv)).clamp(-0.999999, 0.999999);
        let dc = cos_th - cos0;
        eb.bend_kcal += 0.5 * k_ang * dc * dc;
        let d_e_dcos = k_ang * dc;

        let inv_ru = 1.0 / ru;
        let inv_rv = 1.0 / rv;
        let eu = [u[0] * inv_ru, u[1] * inv_ru, u[2] * inv_ru];
        let ev = [v[0] * inv_rv, v[1] * inv_rv, v[2] * inv_rv];

        for c in 0..3 {
            let gi = d_e_dcos * (ev[c] - cos_th * eu[c]) * inv_ru;
            let gk = d_e_dcos * (eu[c] - cos_th * ev[c]) * inv_rv;
            grad[i][c] += gi;
            grad[k][c] += gk;
            grad[j][c] -= gi + gk;
        }
    }

    // 3. Out-of-Plane Planarity & Chiral Volume Restraints
    // Signed volume V = (p1 - pc) . ((p2 - pc) x (p3 - pc))
    for imp in &topo.impropers {
        let pc = coords[imp.center];
        let a = sub3(coords[imp.n1], pc);
        let b = sub3(coords[imp.n2], pc);
        let d = sub3(coords[imp.n3], pc);
        let bxc = cross3(b, d);
        let vol = dot3(a, bxc);

        let target_v = if imp.target_vol_sign == 0 {
            0.0
        } else {
            (imp.target_vol_sign as f64) * 0.42
        };

        // For chiral centers, only penalize if volume approaches 0 or inverts sign
        let diff = if imp.target_vol_sign == 0 {
            vol
        } else if (imp.target_vol_sign > 0 && vol < target_v)
            || (imp.target_vol_sign < 0 && vol > target_v)
        {
            vol - target_v
        } else {
            0.0
        };

        if diff.abs() > 1e-9 {
            eb.oop_kcal += 0.5 * imp.k_oop * diff * diff;
            let pre = imp.k_oop * diff;
            let cxd = bxc;
            let dxa = cross3(d, a);
            let axb = cross3(a, b);
            for c in 0..3 {
                let g1 = pre * cxd[c];
                let g2 = pre * dxa[c];
                let g3 = pre * axb[c];
                grad[imp.n1][c] += g1;
                grad[imp.n2][c] += g2;
                grad[imp.n3][c] += g3;
                grad[imp.center][c] -= g1 + g2 + g3;
            }
        }
    }

    // 4. Fourier Torsional Potential: 0.5 * [V1*(1 + cos phi) + V2*(1 - cos 2phi) + V3*(1 + cos 3phi)]
    for tor in &topo.torsions {
        let b1 = sub3(coords[tor.j], coords[tor.i]);
        let b2 = sub3(coords[tor.k], coords[tor.j]);
        let b3 = sub3(coords[tor.l], coords[tor.k]);

        let m = cross3(b1, b2);
        let n = cross3(b2, b3);
        let m2 = dot3(m, m);
        let n2 = dot3(n, n);
        let b2_len = norm3(b2);
        if m2 < 1e-8 || n2 < 1e-8 || b2_len < 1e-6 {
            continue;
        }
        let mn = (m2 * n2).sqrt();
        let cos_phi = (dot3(m, n) / mn).clamp(-1.0, 1.0);
        let sin_phi = (dot3(cross3(m, n), b2) / (b2_len * mn)).clamp(-1.0, 1.0);

        let cos_2phi = 2.0 * cos_phi * cos_phi - 1.0;
        let cos_3phi = 4.0 * cos_phi * cos_phi * cos_phi - 3.0 * cos_phi;

        let e_tor = 0.5
            * (tor.v1 * (1.0 + cos_phi)
                + tor.v2 * (1.0 - cos_2phi)
                + tor.v3 * (1.0 + cos_3phi));
        eb.torsion_kcal += e_tor;

        // dE/dphi
        let sin_2phi = 2.0 * sin_phi * cos_phi;
        let sin_3phi = sin_phi * (4.0 * cos_phi * cos_phi - 1.0);
        let de_dphi = 0.5 * (-tor.v1 * sin_phi + 2.0 * tor.v2 * sin_2phi - 3.0 * tor.v3 * sin_3phi);

        // Blondel-Karplus singularity-free Cartesian dihedral derivatives
        let f_i_scale = -de_dphi * b2_len / m2;
        let f_l_scale = de_dphi * b2_len / n2;
        let f_i = [m[0] * f_i_scale, m[1] * f_i_scale, m[2] * f_i_scale];
        let f_l = [n[0] * f_l_scale, n[1] * f_l_scale, n[2] * f_l_scale];

        let b2_sq = b2_len * b2_len;
        let coef_j = dot3(b1, b2) / b2_sq;
        let coef_k = dot3(b3, b2) / b2_sq;

        for c in 0..3 {
            let gj = (coef_j - 1.0) * f_i[c] - coef_k * f_l[c];
            let gk = -coef_j * f_i[c] + (coef_k - 1.0) * f_l[c];
            grad[tor.i][c] += f_i[c];
            grad[tor.j][c] += gj;
            grad[tor.k][c] += gk;
            grad[tor.l][c] += f_l[c];
        }
    }

    // 5 & 6. Soft-Core Buffered Van der Waals + Distance-Dependent Dielectric Coulombic Electrostatics
    for &(u, v, r_star, eps, vdw_sc, elec_sc) in &topo.nonbonded {
        let d = sub3(coords[u], coords[v]);
        let r2 = dot3(d, d);
        let r = r2.sqrt().max(0.45);

        // Buffered soft-core 12-6 Lennard-Jones: u_eff = r_star / (r + 0.12 * r_star)
        let denom = r + 0.08 * r_star;
        let s = r_star / denom;
        let s2 = s * s;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;
        let e_v = vdw_sc * eps * (s12 - 2.0 * s6);
        eb.vdw_kcal += e_v;
        let de_vdw_dr = vdw_sc * eps * (-12.0 * s12 + 12.0 * s6) / denom;

        // Screened electrostatics: eps(r) = 4*r -> E_elec = 332.0637 * q_u * q_v / (4 * r^2)
        let qq = (charges[u] as f64) * (charges[v] as f64);
        let mut de_elec_dr = 0.0;
        if qq.abs() > 1e-8 {
            let r_eff2 = r2 + 0.35;
            let e_el = elec_sc * 83.0159 * qq / r_eff2;
            eb.electro_kcal += e_el;
            de_elec_dr = -2.0 * e_el * r / r_eff2;
        }

        let f_over_r = (de_vdw_dr + de_elec_dr) / r;
        for c in 0..3 {
            let gc = f_over_r * d[c];
            grad[u][c] += gc;
            grad[v][c] -= gc;
        }
    }

    eb.total_kcal = eb.stretch_kcal
        + eb.bend_kcal
        + eb.oop_kcal
        + eb.torsion_kcal
        + eb.vdw_kcal
        + eb.electro_kcal;
    let sum_g2: f64 = grad.iter().map(|g| dot3(*g, *g)).sum();
    eb.rms_grad = (sum_g2 / ((coords.len().max(1) * 3) as f64)).sqrt();
    eb
}

/// Minimizes a 3D coordinate vector under the `MMFF-lite` force field using adaptive
/// Polak-Ribiere Conjugate-Gradient with backtracking step-clamping, then centers at the origin.
fn minimize_mmff_lite(
    coords: &mut [[f64; 3]],
    topo: &ForceFieldTopology,
    charges: &[f32],
    max_iters: usize,
) -> MmffEnergyBreakdown {
    let n = coords.len();
    if n <= 1 {
        return MmffEnergyBreakdown::default();
    }

    let mut grad = vec![[0.0f64; 3]; n];
    let mut prev_grad = vec![[0.0f64; 3]; n];
    let mut dir = vec![[0.0f64; 3]; n];
    let mut trial = coords.to_vec();
    let mut trial_grad = vec![[0.0f64; 3]; n];

    let mut eb = eval_mmff_energy_and_grad(coords, topo, charges, &mut grad);
    for i in 0..n {
        dir[i] = [-grad[i][0], -grad[i][1], -grad[i][2]];
    }

    let mut step_scale = 0.012f64;
    let mut iters_done = 0usize;

    for it in 0..max_iters {
        iters_done = it + 1;
        if eb.rms_grad < 0.035 {
            break;
        }

        let max_dir = dir
            .iter()
            .map(|d| norm3(*d))
            .fold(1e-9f64, f64::max);
        let alpha_cap = (0.16 / max_dir).min(step_scale);

        let mut accepted = false;
        let mut alpha = alpha_cap;
        for _bt in 0..6 {
            for i in 0..n {
                for c in 0..3 {
                    trial[i][c] = coords[i][c] + alpha * dir[i][c];
                }
            }
            let cand_eb = eval_mmff_energy_and_grad(&trial, topo, charges, &mut trial_grad);
            if cand_eb.total_kcal <= eb.total_kcal + 1e-5 {
                prev_grad.copy_from_slice(&grad);
                coords.copy_from_slice(&trial);
                grad.copy_from_slice(&trial_grad);
                eb = cand_eb;
                step_scale = (step_scale * 1.18).min(0.085);
                accepted = true;
                break;
            }
            alpha *= 0.45;
        }

        if !accepted {
            // Reset to steepest descent
            for i in 0..n {
                dir[i] = [-grad[i][0], -grad[i][1], -grad[i][2]];
            }
            step_scale = (step_scale * 0.5).max(1e-4);
            continue;
        }

        // Polak-Ribiere beta with non-negative reset every 15 iterations
        if (it + 1) % 15 == 0 {
            for i in 0..n {
                dir[i] = [-grad[i][0], -grad[i][1], -grad[i][2]];
            }
        } else {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..n {
                den += dot3(prev_grad[i], prev_grad[i]);
                let dg = sub3(grad[i], prev_grad[i]);
                num += dot3(grad[i], dg);
            }
            let beta = if den > 1e-12 {
                (num / den).clamp(0.0, 2.5)
            } else {
                0.0
            };
            for i in 0..n {
                for c in 0..3 {
                    dir[i][c] = -grad[i][c] + beta * dir[i][c];
                }
            }
        }
    }

    // Center minimized conformer at heavy-atom centroid
    center_coords_f64(coords);
    eb.iterations = iters_done;
    eb
}

fn center_coords_f64(coords: &mut [[f64; 3]]) {
    if coords.is_empty() {
        return;
    }
    let inv = 1.0 / (coords.len() as f64);
    let mut c = [0.0f64; 3];
    for p in coords.iter() {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] *= inv;
    c[1] *= inv;
    c[2] *= inv;
    for p in coords.iter_mut() {
        p[0] -= c[0];
        p[1] -= c[1];
        p[2] -= c[2];
    }
}

/// Exact cyclic Jacobi eigendecomposition of a real symmetric `3x3` matrix `a`, returning
/// `(eigenvalues_ascending, orthonormal_eigenvectors_columns)`.
fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..24 {
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                let val = a[i][j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }
        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < 1e-15 {
            if a[p][q] >= 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            let phi = diff / (2.0 * a[p][q]);
            let sgn = if phi >= 0.0 { 1.0 } else { -1.0 };
            sgn / (phi.abs() + (1.0 + phi * phi).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = arp - s * (arq + tau * arp);
                a[p][r] = a[r][p];
                a[r][q] = arq + s * (arp - tau * arq);
                a[q][r] = a[r][q];
            }
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = vrp - s * (vrq + tau * vrp);
            v[r][q] = vrq + s * (vrp - tau * vrq);
        }
    }

    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[i][i].partial_cmp(&a[j][j]).unwrap_or(std::cmp::Ordering::Equal));
    let evals = [a[order[0]][order[0]], a[order[1]][order[1]], a[order[2]][order[2]]];
    let mut evecs = [[0.0f64; 3]; 3];
    for col in 0..3 {
        let src = order[col];
        for row in 0..3 {
            evecs[row][col] = v[row][src];
        }
    }
    // Enforce right-handed eigenvector basis (det == +1)
    let c0 = [evecs[0][0], evecs[1][0], evecs[2][0]];
    let c1 = [evecs[0][1], evecs[1][1], evecs[2][1]];
    let c2 = [evecs[0][2], evecs[1][2], evecs[2][2]];
    if dot3(cross3(c0, c1), c2) < 0.0 {
        for row in 0..3 {
            evecs[row][2] = -evecs[row][2];
        }
    }
    (evals, evecs)
}

/// Exact cyclic Jacobi eigendecomposition of a real symmetric `4x4` matrix (used for Horn's
/// quaternion optimal 3D Kabsch rotation), returning the unit eigenvector of the maximum eigenvalue.
fn jacobi_max_eigenvector_4x4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _sweep in 0..30 {
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let val = a[i][j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }
        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < 1e-15 {
            if a[p][q] >= 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            let phi = diff / (2.0 * a[p][q]);
            let sgn = if phi >= 0.0 { 1.0 } else { -1.0 };
            sgn / (phi.abs() + (1.0 + phi * phi).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..4 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = arp - s * (arq + tau * arp);
                a[p][r] = a[r][p];
                a[r][q] = arq + s * (arp - tau * arq);
                a[q][r] = a[r][q];
            }
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = vrp - s * (vrq + tau * vrp);
            v[r][q] = vrq + s * (vrp - tau * vrq);
        }
    }

    let mut best_col = 0usize;
    for col in 1..4 {
        if a[col][col] > a[best_col][best_col] {
            best_col = col;
        }
    }
    let mut q = [
        v[0][best_col],
        v[1][best_col],
        v[2][best_col],
        v[3][best_col],
    ];
    let nrm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if nrm > 1e-12 {
        for x in &mut q {
            *x /= nrm;
        }
    } else {
        q = [1.0, 0.0, 0.0, 0.0];
    }
    q
}

fn quat_to_rot_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Optimal proper right-handed 3D Kabsch-Horn quaternion superposition aligning `mobile` onto
/// `reference` (must have equal point count), returning `(rotation_3x3, translation_3, aligned_coords, rmsd)`.
pub fn kabsch_align_3d(
    mobile: &[[f32; 3]],
    reference: &[[f32; 3]],
) -> ([[f32; 3]; 3], [f32; 3], Vec<[f32; 3]>, f32) {
    let n = mobile.len().min(reference.len());
    if n == 0 {
        return (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
            Vec::new(),
            0.0,
        );
    }

    let inv_n = 1.0 / (n as f64);
    let mut cm = [0.0f64; 3];
    let mut cr = [0.0f64; 3];
    for i in 0..n {
        for c in 0..3 {
            cm[c] += mobile[i][c] as f64;
            cr[c] += reference[i][c] as f64;
        }
    }
    for c in 0..3 {
        cm[c] *= inv_n;
        cr[c] *= inv_n;
    }

    let mut s = [[0.0f64; 3]; 3];
    for i in 0..n {
        let pm = [
            (mobile[i][0] as f64) - cm[0],
            (mobile[i][1] as f64) - cm[1],
            (mobile[i][2] as f64) - cm[2],
        ];
        let pr = [
            (reference[i][0] as f64) - cr[0],
            (reference[i][1] as f64) - cr[1],
            (reference[i][2] as f64) - cr[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += pm[r] * pr[c];
            }
        }
    }

    // Horn (1987) 4x4 symmetric quaternion matrix N
    let n_mat = [
        [
            s[0][0] + s[1][1] + s[2][2],
            s[1][2] - s[2][1],
            s[2][0] - s[0][2],
            s[0][1] - s[1][0],
        ],
        [
            s[1][2] - s[2][1],
            s[0][0] - s[1][1] - s[2][2],
            s[0][1] + s[1][0],
            s[2][0] + s[0][2],
        ],
        [
            s[2][0] - s[0][2],
            s[0][1] + s[1][0],
            -s[0][0] + s[1][1] - s[2][2],
            s[1][2] + s[2][1],
        ],
        [
            s[0][1] - s[1][0],
            s[2][0] + s[0][2],
            s[1][2] + s[2][1],
            -s[0][0] - s[1][1] + s[2][2],
        ],
    ];

    let q = jacobi_max_eigenvector_4x4(n_mat);
    let rot = quat_to_rot_matrix(q);

    // t = cr - R * cm
    let r_cm = mat3_vec3(rot, cm);
    let trans = [cr[0] - r_cm[0], cr[1] - r_cm[1], cr[2] - r_cm[2]];

    let mut aligned = Vec::with_capacity(mobile.len());
    let mut sq_err = 0.0f64;
    for (i, p) in mobile.iter().enumerate() {
        let rp = mat3_vec3(rot, [p[0] as f64, p[1] as f64, p[2] as f64]);
        let ap = [rp[0] + trans[0], rp[1] + trans[1], rp[2] + trans[2]];
        if i < n {
            let dx = ap[0] - (reference[i][0] as f64);
            let dy = ap[1] - (reference[i][1] as f64);
            let dz = ap[2] - (reference[i][2] as f64);
            sq_err += dx * dx + dy * dy + dz * dz;
        }
        aligned.push([ap[0] as f32, ap[1] as f32, ap[2] as f32]);
    }
    let rmsd = (sq_err * inv_n).sqrt() as f32;

    let rot_f32 = [
        [rot[0][0] as f32, rot[0][1] as f32, rot[0][2] as f32],
        [rot[1][0] as f32, rot[1][1] as f32, rot[1][2] as f32],
        [rot[2][0] as f32, rot[2][1] as f32, rot[2][2] as f32],
    ];
    let trans_f32 = [trans[0] as f32, trans[1] as f32, trans[2] as f32];
    (rot_f32, trans_f32, aligned, rmsd)
}

#[inline]
fn mat3_vec3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
fn mat3_transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

/// Computes Radius of Gyration (`Rg`), Plane of Best Fit (`PBF`, Firth et al. 2012), and
/// Sauer-Schwarz (2003) Normalized Principal Moments of Inertia ratios (`NPR1 = I1/I3`, `NPR2 = I2/I3`).
fn compute_3d_shape_moments(mol: &Molecule, coords3d: &[[f32; 3]]) -> (f32, f32, f32, f32) {
    let n = mol.atoms.len().min(coords3d.len());
    if n <= 1 {
        return (0.0, 0.0, 0.5, 0.5);
    }

    let mut total_m = 0.0f64;
    let mut cm = [0.0f64; 3];
    for i in 0..n {
        let m = mol.atoms[i].elem.atomic_weight();
        total_m += m;
        cm[0] += m * (coords3d[i][0] as f64);
        cm[1] += m * (coords3d[i][1] as f64);
        cm[2] += m * (coords3d[i][2] as f64);
    }
    if total_m > 1e-9 {
        cm[0] /= total_m;
        cm[1] /= total_m;
        cm[2] /= total_m;
    }

    let mut inertia = [[0.0f64; 3]; 3];
    let mut cov = [[0.0f64; 3]; 3];
    let mut rg_sum = 0.0f64;

    for i in 0..n {
        let m = mol.atoms[i].elem.atomic_weight();
        let x = (coords3d[i][0] as f64) - cm[0];
        let y = (coords3d[i][1] as f64) - cm[1];
        let z = (coords3d[i][2] as f64) - cm[2];
        let r2 = x * x + y * y + z * z;
        rg_sum += m * r2;

        inertia[0][0] += m * (y * y + z * z);
        inertia[1][1] += m * (x * x + z * z);
        inertia[2][2] += m * (x * x + y * y);
        inertia[0][1] -= m * x * y;
        inertia[0][2] -= m * x * z;
        inertia[1][2] -= m * y * z;

        cov[0][0] += x * x;
        cov[1][1] += y * y;
        cov[2][2] += z * z;
        cov[0][1] += x * y;
        cov[0][2] += x * z;
        cov[1][2] += y * z;
    }
    inertia[1][0] = inertia[0][1];
    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    let rg = (rg_sum / total_m.max(1e-9)).sqrt() as f32;
    let (pmi, _) = jacobi_eigen_3x3(inertia);
    let i3 = pmi[2].max(1e-6);
    let npr1 = (pmi[0] / i3).clamp(0.0, 1.0) as f32;
    let npr2 = (pmi[1] / i3).clamp(1.0 - npr1 as f64, 1.0) as f32;

    // Plane of Best Fit (PBF): mean absolute distance along smallest-variance eigenvector of cov
    let (_, cov_vecs) = jacobi_eigen_3x3(cov);
    let normal = [cov_vecs[0][0], cov_vecs[1][0], cov_vecs[2][0]];
    let mut pbf_sum = 0.0f64;
    for i in 0..n {
        let p = [
            (coords3d[i][0] as f64) - cm[0],
            (coords3d[i][1] as f64) - cm[1],
            (coords3d[i][2] as f64) - cm[2],
        ];
        pbf_sum += dot3(p, normal).abs();
    }
    let pbf = (pbf_sum / (n as f64)) as f32;

    (rg, pbf, npr1, npr2)
}

fn classify_npr_shape(npr1: f32, npr2: f32) -> &'static str {
    // Vertices of Sauer-Schwarz triangle: Rod = (0, 1), Disc = (0.5, 0.5), Sphere = (1, 1)
    let d_rod = (npr1 - 0.0).hypot(npr2 - 1.0);
    let d_disc = (npr1 - 0.5).hypot(npr2 - 0.5);
    let d_sph = (npr1 - 1.0).hypot(npr2 - 1.0);
    if d_sph <= d_rod && d_sph <= d_disc {
        "Globular / Sphere"
    } else if d_disc <= d_rod {
        "Disc-Like"
    } else {
        "Rod-Like"
    }
}

/// Computes Ballester & Richards (2007) 12-D `USR` and Schreyer & Blundell (2012) 60-D `USRCAT`
/// ultrafast shape + pharmacophore moment descriptors.
pub fn compute_usr_descriptors(
    mol: &Molecule,
    coords3d: &[[f32; 3]],
) -> ([f32; USR_DIM], Vec<f32>) {
    let n = mol.atoms.len().min(coords3d.len());
    if n == 0 {
        return ([0.0; USR_DIM], vec![0.0; USRCAT_DIM]);
    }

    let pts: Vec<[f64; 3]> = coords3d
        .iter()
        .take(n)
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let inv_n = 1.0 / (n as f64);
    let mut ctd = [0.0f64; 3];
    for p in &pts {
        ctd[0] += p[0];
        ctd[1] += p[1];
        ctd[2] += p[2];
    }
    ctd[0] *= inv_n;
    ctd[1] *= inv_n;
    ctd[2] *= inv_n;

    let mut cst_idx = 0usize;
    let mut fct_idx = 0usize;
    let mut min_d = f64::INFINITY;
    let mut max_d = -1.0f64;
    for (i, p) in pts.iter().enumerate() {
        let d = norm3(sub3(*p, ctd));
        if d < min_d {
            min_d = d;
            cst_idx = i;
        }
        if d > max_d {
            max_d = d;
            fct_idx = i;
        }
    }
    let cst = pts[cst_idx];
    let fct = pts[fct_idx];

    let mut ftf_idx = 0usize;
    let mut max_f = -1.0f64;
    for (i, p) in pts.iter().enumerate() {
        let d = norm3(sub3(*p, fct));
        if d > max_f {
            max_f = d;
            ftf_idx = i;
        }
    }
    let ftf = pts[ftf_idx];
    let refs = [ctd, cst, fct, ftf];

    let moments_for_subset = |indices: &[usize]| -> [f32; USR_DIM] {
        if indices.is_empty() {
            return [0.0; USR_DIM];
        }
        let mut out = [0.0f32; USR_DIM];
        let inv_m = 1.0 / (indices.len() as f64);
        for (ri, rpt) in refs.iter().enumerate() {
            let dists: Vec<f64> = indices.iter().map(|&u| norm3(sub3(pts[u], *rpt))).collect();
            let mu1 = dists.iter().sum::<f64>() * inv_m;
            let mut m2 = 0.0f64;
            let mut m3 = 0.0f64;
            for &d in &dists {
                let diff = d - mu1;
                let diff2 = diff * diff;
                m2 += diff2;
                m3 += diff2 * diff;
            }
            let sigma = (m2 * inv_m).sqrt();
            let m3_avg = m3 * inv_m;
            let skew = m3_avg.signum() * m3_avg.abs().cbrt();
            out[ri * 3] = mu1 as f32;
            out[ri * 3 + 1] = sigma as f32;
            out[ri * 3 + 2] = skew as f32;
        }
        out
    };

    let all_idx: Vec<usize> = (0..n).collect();
    let usr12 = moments_for_subset(&all_idx);

    // USRCAT subsets: 1. All, 2. Hydrophobic, 3. Aromatic, 4. HBA, 5. HBD
    let mut hyd_idx = Vec::new();
    let mut arom_idx = Vec::new();
    let mut hba_idx = Vec::new();
    let mut hbd_idx = Vec::new();

    for (u, a) in mol.atoms.iter().take(n).enumerate() {
        if a.aromatic {
            arom_idx.push(u);
        }
        let polar_nbr = mol.adj[u]
            .iter()
            .any(|&(nb, _)| matches!(mol.atoms[nb].elem, Element::N | Element::O | Element::F));
        if matches!(
            a.elem,
            Element::C | Element::Cl | Element::Br | Element::I | Element::S
        ) && a.charge == 0
            && (!polar_nbr || a.elem != Element::C)
        {
            hyd_idx.push(u);
        }
        if matches!(a.elem, Element::N | Element::O | Element::F) && a.charge <= 0 {
            hba_idx.push(u);
        }
        if matches!(a.elem, Element::N | Element::O | Element::S) && a.total_h() > 0 {
            hbd_idx.push(u);
        }
    }

    let mut usrcat60 = Vec::with_capacity(USRCAT_DIM);
    usrcat60.extend_from_slice(&usr12);
    usrcat60.extend_from_slice(&moments_for_subset(&hyd_idx));
    usrcat60.extend_from_slice(&moments_for_subset(&arom_idx));
    usrcat60.extend_from_slice(&hba_idx_or_empty(&moments_for_subset, &hba_idx));
    usrcat60.extend_from_slice(&hba_idx_or_empty(&moments_for_subset, &hbd_idx));

    (usr12, usrcat60)
}

fn hba_idx_or_empty<F: Fn(&[usize]) -> [f32; USR_DIM]>(f: &F, idx: &[usize]) -> [f32; USR_DIM] {
    f(idx)
}

/// Ballester & Richards (2007) inverse-Manhattan `USR` similarity in `(0.0, 1.0]`.
pub fn usr_similarity(a: &[f32; USR_DIM], b: &[f32; USR_DIM]) -> f64 {
    let mut l1 = 0.0f64;
    for k in 0..USR_DIM {
        l1 += ((a[k] as f64) - (b[k] as f64)).abs();
    }
    1.0 / (1.0 + l1 / (USR_DIM as f64))
}

/// Schreyer & Blundell (2012) 60-D pharmacophore-aware `USRCAT` similarity in `(0.0, 1.0]`.
pub fn usrcat_similarity(a: &[f32], b: &[f32]) -> f64 {
    let m = a.len().min(b.len());
    if m == 0 {
        return 0.0;
    }
    let mut l1 = 0.0f64;
    for k in 0..m {
        l1 += ((a[k] as f64) - (b[k] as f64)).abs();
    }
    1.0 / (1.0 + l1 / (m as f64))
}

/// Extracts 3D `ROCS` chemical color atoms (`HBD`, `HBA`, `Cation`, `Anion`, `AromaticRing` centroids,
/// and `Hydrophobic` centers) on a 3D conformer.
fn perceive_3d_color_atoms(mol: &Molecule, coords3d: &[[f32; 3]]) -> Vec<RocsColorAtom> {
    let n = mol.atoms.len().min(coords3d.len());
    let mut colors = Vec::new();

    // 1. Aromatic ring centroids
    for ring in &mol.rings {
        if !ring.is_empty() && ring.iter().all(|&u| u < n && mol.atoms[u].aromatic) {
            let inv = 1.0 / (ring.len() as f32);
            let mut c = [0.0f32; 3];
            for &u in ring {
                c[0] += coords3d[u][0];
                c[1] += coords3d[u][1];
                c[2] += coords3d[u][2];
            }
            colors.push(RocsColorAtom {
                kind: RocsColorKind::AromaticRing,
                pos3d: [c[0] * inv, c[1] * inv, c[2] * inv],
                label: "Aromatic Ring Centroid",
            });
        }
    }

    // 2. Non-aromatic hydrophobic ring centroids & terminal lipophilic halogens/tert-butyls
    for ring in &mol.rings {
        if !ring.is_empty()
            && ring
                .iter()
                .all(|&u| u < n && !mol.atoms[u].aromatic && mol.atoms[u].elem == Element::C)
        {
            let inv = 1.0 / (ring.len() as f32);
            let mut c = [0.0f32; 3];
            for &u in ring {
                c[0] += coords3d[u][0];
                c[1] += coords3d[u][1];
                c[2] += coords3d[u][2];
            }
            colors.push(RocsColorAtom {
                kind: RocsColorKind::Hydrophobic,
                pos3d: [c[0] * inv, c[1] * inv, c[2] * inv],
                label: "Aliphatic Ring",
            });
        }
    }

    for u in 0..n {
        let a = &mol.atoms[u];
        let p = coords3d[u];

        if a.charge > 0
            || (a.elem == Element::N
                && !a.aromatic
                && a.hyb == Hybridization::Sp3
                && !mol.adj[u].iter().any(|&(nb, _)| {
                    mol.atoms[nb].hyb == Hybridization::Sp2 || mol.atoms[nb].elem == Element::S
                }))
        {
            colors.push(RocsColorAtom {
                kind: RocsColorKind::Cation,
                pos3d: p,
                label: "Basic / Cationic Center",
            });
        }

        if a.charge < 0 {
            colors.push(RocsColorAtom {
                kind: RocsColorKind::Anion,
                pos3d: p,
                label: "Acidic / Anionic Center",
            });
        }

        if matches!(a.elem, Element::N | Element::O | Element::S) && a.total_h() > 0 {
            colors.push(RocsColorAtom {
                kind: RocsColorKind::HBondDonor,
                pos3d: p,
                label: "H-Bond Donor",
            });
        }

        if matches!(a.elem, Element::O | Element::N) && a.charge <= 0 {
            let is_non_acceptor_n = a.elem == Element::N
                && ((a.aromatic && (a.total_h() > 0 || mol.heavy_degree(u) >= 3))
                    || (a.total_h() > 0
                        && mol.adj[u]
                            .iter()
                            .any(|&(nb, _)| mol.atoms[nb].hyb == Hybridization::Sp2 || mol.atoms[nb].elem == Element::S)));
            if !is_non_acceptor_n {
                colors.push(RocsColorAtom {
                    kind: RocsColorKind::HBondAcceptor,
                    pos3d: p,
                    label: "H-Bond Acceptor",
                });
            }
        }

        if matches!(a.elem, Element::Cl | Element::Br | Element::I)
            || (a.elem == Element::C
                && !a.in_ring
                && mol.adj[u].len() >= 3
                && mol.adj[u]
                    .iter()
                    .all(|&(nb, _)| matches!(mol.atoms[nb].elem, Element::C | Element::F)))
        {
            colors.push(RocsColorAtom {
                kind: RocsColorKind::Hydrophobic,
                pos3d: p,
                label: "Lipophilic Group",
            });
        }
    }

    colors
}

/// Pairwise second-order Grant-Pickup Gaussian molecular volume overlap `O_AB` (`Angstrom^3`).
fn gaussian_shape_overlap(
    mol_a: &Molecule,
    coords_a: &[[f64; 3]],
    mol_b: &Molecule,
    coords_b: &[[f64; 3]],
) -> f64 {
    let na = mol_a.atoms.len().min(coords_a.len());
    let nb = mol_b.atoms.len().min(coords_b.len());
    if na == 0 || nb == 0 {
        return 0.0;
    }

    let alphas_a: Vec<f64> = (0..na)
        .map(|i| {
            let r = bondi_vdw_radius(mol_a.atoms[i].elem) as f64;
            GAUSS_KAPPA / (r * r)
        })
        .collect();
    let alphas_b: Vec<f64> = (0..nb)
        .map(|j| {
            let r = bondi_vdw_radius(mol_b.atoms[j].elem) as f64;
            GAUSS_KAPPA / (r * r)
        })
        .collect();

    let p2 = GAUSS_P * GAUSS_P;
    let mut vol = 0.0f64;
    for i in 0..na {
        let ai = alphas_a[i];
        let pi = coords_a[i];
        for j in 0..nb {
            let aj = alphas_b[j];
            let sum_a = ai + aj;
            let k_exp = (ai * aj) / sum_a;
            let d = sub3(pi, coords_b[j]);
            let d2 = dot3(d, d);
            if k_exp * d2 > 14.0 {
                continue;
            }
            let ratio = PI / sum_a;
            let v_ij = p2 * ratio * ratio.sqrt() * (-k_exp * d2).exp();
            vol += v_ij;
        }
    }
    vol
}

/// Same-type `ROCS` color Gaussian overlap (`r_color = 1.05 A`, `alpha_c = 2.19`).
fn gaussian_color_overlap(colors_a: &[RocsColorAtom], colors_b: &[RocsColorAtom]) -> f64 {
    if colors_a.is_empty() || colors_b.is_empty() {
        return 0.0;
    }
    let alpha_c = 0.98f64;
    let k_exp = 0.5 * alpha_c;
    let pref = {
        let ratio = PI / (2.0 * alpha_c);
        GAUSS_P * GAUSS_P * ratio * ratio.sqrt()
    };
    let mut ov = 0.0f64;
    for ca in colors_a {
        for cb in colors_b {
            if ca.kind != cb.kind {
                continue;
            }
            let dx = (ca.pos3d[0] - cb.pos3d[0]) as f64;
            let dy = (ca.pos3d[1] - cb.pos3d[1]) as f64;
            let dz = (ca.pos3d[2] - cb.pos3d[2]) as f64;
            let d2 = dx * dx + dy * dy + dz * dz;
            if k_exp * d2 <= 12.0 {
                ov += pref * (-k_exp * d2).exp();
            }
        }
    }
    ov
}

/// Gaussian-smoothed 3D Electrostatic Potential Hodgkin similarity mapped to `[0.0, 1.0]`.
fn gaussian_esp_similarity(
    coords_a: &[[f64; 3]],
    charges_a: &[f32],
    coords_b: &[[f64; 3]],
    charges_b: &[f32],
) -> f64 {
    let beta = 0.32f64; // Smooth ~1.8 A electrostatic decay length
    let pair_int = |pa: &[[f64; 3]], qa: &[f32], pb: &[[f64; 3]], qb: &[f32]| -> f64 {
        let mut s = 0.0f64;
        for i in 0..pa.len().min(qa.len()) {
            let qi = qa[i] as f64;
            if qi.abs() < 1e-5 {
                continue;
            }
            for j in 0..pb.len().min(qb.len()) {
                let qj = qb[j] as f64;
                if qj.abs() < 1e-5 {
                    continue;
                }
                let d = sub3(pa[i], pb[j]);
                let d2 = dot3(d, d);
                if beta * d2 <= 14.0 {
                    s += qi * qj * (-beta * d2).exp();
                }
            }
        }
        s
    };

    let e_aa = pair_int(coords_a, charges_a, coords_a, charges_a);
    let e_bb = pair_int(coords_b, charges_b, coords_b, charges_b);
    if e_aa <= 1e-8 && e_bb <= 1e-8 {
        return 0.75; // Both essentially apolar
    }
    if e_aa <= 1e-8 || e_bb <= 1e-8 {
        return 0.25;
    }
    let e_ab = pair_int(coords_a, charges_a, coords_b, charges_b);
    let hodgkin = (2.0 * e_ab) / (e_aa + e_bb); // in [-1, 1]
    (0.5 * (hodgkin + 1.0)).clamp(0.0, 1.0)
}

/// Generates a Boltzmann-weighted, `MMFF-lite` minimized 3D conformer ensemble for `mol`.
pub fn generate_3d_conformer_ensemble(
    mol: &Molecule,
    cfg: &ConformerGenConfig,
) -> ConformerEnsemble {
    let charges = compute_gasteiger_charges(mol);
    let topo = ForceFieldTopology::build(mol);
    let trials = cfg.num_trial_conformers.max(1);
    let mut rng = SplitMix64::new(
        cfg.random_seed
            ^ ((mol.atoms.len() as u64) << 16)
            ^ ((mol.bonds.len() as u64) << 8),
    );

    let mut raw_confs: Vec<(usize, Vec<[f64; 3]>, MmffEnergyBreakdown)> = Vec::with_capacity(trials);
    for t_idx in 0..trials {
        let mut coords = embed_distance_geometry_3d(mol, &topo, &mut rng);
        let eb = minimize_mmff_lite(&mut coords, &topo, &charges, cfg.mmff_max_iters);
        raw_confs.push((t_idx + 1, coords, eb));
    }

    raw_confs.sort_by(|a, b| {
        a.2.total_kcal
            .partial_cmp(&b.2.total_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let e_min = raw_confs.first().map(|c| c.2.total_kcal).unwrap_or(0.0);
    let mut kept: Vec<Conformer3d> = Vec::new();

    for (seed_id, coords_f64, energy) in raw_confs {
        let rel_e = (energy.total_kcal - e_min).max(0.0);
        if !kept.is_empty() && rel_e > cfg.energy_window_kcal {
            continue;
        }
        let coords_f32: Vec<[f32; 3]> = coords_f64
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();

        // Deduplicate by 3D Kabsch heavy-atom RMSD against all lower-energy kept conformers
        let mut duplicate = false;
        let mut rmsd_to_best = 0.0f32;
        for (ki, existing) in kept.iter().enumerate() {
            let (_, _, _, rmsd) = kabsch_align_3d(&coords_f32, &existing.coords3d);
            if ki == 0 {
                rmsd_to_best = rmsd;
            }
            if rmsd < cfg.rmsd_prune_cutoff {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            continue;
        }

        let (rg, pbf, npr1, npr2) = compute_3d_shape_moments(mol, &coords_f32);
        let self_vol = gaussian_shape_overlap(mol, &coords_f64, mol, &coords_f64);
        let dipole = compute_dipole_debye(&coords_f32, &charges);
        let (usr12, usrcat60) = compute_usr_descriptors(mol, &coords_f32);
        let color_atoms = perceive_3d_color_atoms(mol, &coords_f32);

        kept.push(Conformer3d {
            rank: kept.len() + 1,
            seed_id,
            coords3d: coords_f32,
            energy,
            rel_energy_kcal: rel_e,
            boltzmann_weight: (-rel_e / 0.59248).exp(),
            rmsd_to_lowest: rmsd_to_best,
            rg_angstrom: rg,
            pbf_angstrom: pbf,
            npr1,
            npr2,
            gaussian_self_volume: self_vol,
            dipole_debye: dipole,
            usr12,
            usrcat60,
            color_atoms,
        });

        if kept.len() >= cfg.max_kept_conformers.max(1) {
            break;
        }
    }

    let z_sum: f64 = kept.iter().map(|c| c.boltzmann_weight).sum::<f64>().max(1e-12);
    for c in &mut kept {
        c.boltzmann_weight /= z_sum;
    }

    ConformerEnsemble {
        gasteiger_charges: charges,
        rotatable_dihedrals: topo.rotatable_bonds,
        trial_count: trials,
        conformers: kept,
    }
}

/// Evaluates a fast, deterministic 3D summary profile (`USR12`, `USRCAT60`, `NPR1/NPR2`, `PBF`,
/// `Rg`, `Dipole`, Gasteiger charges, and lowest-energy 3D conformer) for dataset-wide indexing.
pub fn evaluate_3d_summary_profile(mol: &Molecule, fast_hts_mode: bool) -> Summary3dProfile {
    let cfg = if fast_hts_mode {
        ConformerGenConfig {
            num_trial_conformers: 1,
            max_kept_conformers: 1,
            mmff_max_iters: 22,
            rmsd_prune_cutoff: 0.45,
            energy_window_kcal: 20.0,
            random_seed: 0x5A17_3D12,
        }
    } else {
        ConformerGenConfig {
            num_trial_conformers: 3,
            max_kept_conformers: 1,
            mmff_max_iters: 75,
            rmsd_prune_cutoff: 0.45,
            energy_window_kcal: 20.0,
            random_seed: 0x5A17_3D12,
        }
    };
    let ens = generate_3d_conformer_ensemble(mol, &cfg);
    if let Some(c0) = ens.conformers.into_iter().next() {
        let shape_class = classify_npr_shape(c0.npr1, c0.npr2);
        Summary3dProfile {
            coords3d: c0.coords3d,
            gasteiger_charges: ens.gasteiger_charges,
            mmff_energy_kcal: c0.energy.total_kcal,
            rg_angstrom: c0.rg_angstrom,
            pbf_angstrom: c0.pbf_angstrom,
            npr1: c0.npr1,
            npr2: c0.npr2,
            shape_class,
            dipole_debye: c0.dipole_debye,
            gaussian_volume: c0.gaussian_self_volume,
            usr12: c0.usr12,
            usrcat60: c0.usrcat60,
            color_atoms: c0.color_atoms,
        }
    } else {
        Summary3dProfile {
            coords3d: Vec::new(),
            gasteiger_charges: ens.gasteiger_charges,
            mmff_energy_kcal: 0.0,
            rg_angstrom: 0.0,
            pbf_angstrom: 0.0,
            npr1: 0.5,
            npr2: 0.5,
            shape_class: "Disc-Like",
            dipole_debye: 0.0,
            gaussian_volume: 0.0,
            usr12: [0.0; USR_DIM],
            usrcat60: vec![0.0; USRCAT_DIM],
            color_atoms: Vec::new(),
        }
    }
}

/// Computes the covariance principal frame `(centroid, 3x3_right_handed_axes)` of a 3D point cloud.
fn principal_axes_frame(pts: &[[f64; 3]]) -> ([f64; 3], [[f64; 3]; 3]) {
    if pts.is_empty() {
        return ([0.0; 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
    let inv = 1.0 / (pts.len() as f64);
    let mut c = [0.0f64; 3];
    for p in pts {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] *= inv;
    c[1] *= inv;
    c[2] *= inv;

    let mut cov = [[0.0f64; 3]; 3];
    for p in pts {
        let d = sub3(*p, c);
        for r in 0..3 {
            for col in 0..3 {
                cov[r][col] += d[r] * d[col];
            }
        }
    }
    let (_, evecs) = jacobi_eigen_3x3(cov);
    (c, evecs)
}

/// Rodrigues exponential-map rotation matrix `exp([omega]_x)` for small angular step `omega`.
fn rodrigues_rot(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = norm3(omega);
    if theta < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let k = [omega[0] / theta, omega[1] / theta, omega[2] / theta];
    let c = theta.cos();
    let s = theta.sin();
    let v = 1.0 - c;
    [
        [
            k[0] * k[0] * v + c,
            k[0] * k[1] * v - k[2] * s,
            k[0] * k[2] * v + k[1] * s,
        ],
        [
            k[1] * k[0] * v + k[2] * s,
            k[1] * k[1] * v + c,
            k[1] * k[2] * v - k[0] * s,
        ],
        [
            k[2] * k[0] * v - k[1] * s,
            k[2] * k[1] * v + k[0] * s,
            k[2] * k[2] * v + c,
        ],
    ]
}

/// Aligns `hit_mol` onto `query_mol` in 3D by multi-start principal-axes enumeration (`4` proper
/// `SO(3)` rotations) followed by 6-DOF analytical Gaussian shape + color overlap rigid-body ascent!
#[allow(clippy::too_many_arguments)]
pub fn align_rocs_3d(
    query_mol: &Molecule,
    query_coords: &[[f32; 3]],
    query_charges: &[f32],
    query_colors: &[RocsColorAtom],
    hit_mol: &Molecule,
    hit_coords: &[[f32; 3]],
    hit_charges: &[f32],
    hit_colors: &[RocsColorAtom],
) -> Rocs3dAlignment {
    let nq = query_mol.atoms.len().min(query_coords.len());
    let nh = hit_mol.atoms.len().min(hit_coords.len());
    if nq == 0 || nh == 0 {
        return Rocs3dAlignment {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            aligned_hit_coords: hit_coords.to_vec(),
            aligned_hit_colors: hit_colors.to_vec(),
            best_hit_conformer_rank: 1,
            shape_overlap_vol: 0.0,
            shape_tanimoto: 0.0,
            shape_tversky_query: 0.0,
            color_feature_tanimoto: 0.0,
            esp_hodgkin_sim: 0.0,
            color_tanimoto: 0.0,
            combo_score: 0.0,
            combo_tanimoto: 0.0,
            usr_similarity: 0.0,
            usrcat_similarity: 0.0,
            centroid_rmsd: 0.0,
        };
    }

    let q_pts: Vec<[f64; 3]> = query_coords[..nq]
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let h_pts: Vec<[f64; 3]> = hit_coords[..nh]
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let alphas_q: Vec<f64> = (0..nq)
        .map(|i| {
            let r = bondi_vdw_radius(query_mol.atoms[i].elem) as f64;
            GAUSS_KAPPA / (r * r)
        })
        .collect();
    let alphas_h: Vec<f64> = (0..nh)
        .map(|j| {
            let r = bondi_vdw_radius(hit_mol.atoms[j].elem) as f64;
            GAUSS_KAPPA / (r * r)
        })
        .collect();

    let (cq, axes_q) = principal_axes_frame(&q_pts);
    let (ch, axes_h) = principal_axes_frame(&h_pts);
    let axes_h_t = mat3_transpose(axes_h);

    // Full 24 proper chiral octahedral SO(3) axis-permutation+sign rotations of the principal frame,
    // augmented with tilts so non-diagonal moments of inertia cannot trap overlay orientation.
    let flips: [[f64; 3]; 4] = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let tilt = 0.61f64; // ~35 deg
    let tilt2 = 1.22f64; // ~70 deg
    let hpi = std::f64::consts::FRAC_PI_2;
    let tilt_mats: [[[f64; 3]; 3]; 15] = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        rodrigues_rot([0.0, 0.0, tilt]),
        rodrigues_rot([0.0, 0.0, -tilt]),
        rodrigues_rot([0.0, tilt, 0.0]),
        rodrigues_rot([0.0, -tilt, 0.0]),
        rodrigues_rot([0.0, 0.0, tilt2]),
        rodrigues_rot([0.0, 0.0, -tilt2]),
        rodrigues_rot([0.0, tilt2, 0.0]),
        rodrigues_rot([0.0, -tilt2, 0.0]),
        rodrigues_rot([hpi, 0.0, 0.0]),
        rodrigues_rot([-hpi, 0.0, 0.0]),
        rodrigues_rot([0.0, hpi, 0.0]),
        rodrigues_rot([0.0, -hpi, 0.0]),
        rodrigues_rot([0.0, 0.0, hpi]),
        rodrigues_rot([0.0, 0.0, -hpi]),
    ];

    let p2 = GAUSS_P * GAUSS_P;
    let alpha_c = 0.98f64;
    let kc_exp = 0.5 * alpha_c;
    let pref_c = {
        let ratio = PI / (2.0 * alpha_c);
        p2 * ratio * ratio.sqrt()
    };

    let mut best_obj = -1.0f64;
    let mut best_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut best_trans = [0.0f64; 3];

    for flip in flips {
        let mut flipped_q = axes_q;
        for row in 0..3 {
            for col in 0..3 {
                flipped_q[row][col] *= flip[col];
            }
        }
        let base_rot = mat3_mul(flipped_q, axes_h_t);
        for tmat in tilt_mats {
            let mut rot = mat3_mul(tmat, base_rot);
            let r_ch = mat3_vec3(rot, ch);
            let mut trans = [cq[0] - r_ch[0], cq[1] - r_ch[1], cq[2] - r_ch[2]];

            // Fast 34-step 6-DOF rigid-body gradient ascent on combined Gaussian Shape + Color objective
            let mut step_t = 0.055f64;
            let mut step_r = 0.040f64;

            for _it in 0..34 {
                let mut force_sum = [0.0f64; 3];
                let mut torque_sum = [0.0f64; 3];
                let mut cur_obj = 0.0f64;

                // Shape overlap forces & torques around cq
                for j in 0..nh {
                    let r_hj = mat3_vec3(rot, h_pts[j]);
                    let xj = [r_hj[0] + trans[0], r_hj[1] + trans[1], r_hj[2] + trans[2]];
                    let arm = sub3(xj, cq);
                    let aj = alphas_h[j];

                    for i in 0..nq {
                        let ai = alphas_q[i];
                        let sum_a = ai + aj;
                        let k_exp = (ai * aj) / sum_a;
                        let diff = sub3(q_pts[i], xj); // q_i - x_j
                        let d2 = dot3(diff, diff);
                        if k_exp * d2 > 12.0 {
                            continue;
                        }
                        let ratio = PI / sum_a;
                        let v_ij = p2 * ratio * ratio.sqrt() * (-k_exp * d2).exp();
                        cur_obj += v_ij;
                        // d(v_ij)/d(xj) = 2 * k_exp * v_ij * (q_i - x_j)
                        let s = 2.0 * k_exp * v_ij;
                        let fj = [s * diff[0], s * diff[1], s * diff[2]];
                        force_sum[0] += fj[0];
                        force_sum[1] += fj[1];
                        force_sum[2] += fj[2];
                        let tq = cross3(arm, fj);
                        torque_sum[0] += tq[0];
                        torque_sum[1] += tq[1];
                        torque_sum[2] += tq[2];
                    }
                }

                // Color feature overlap forces & torques (weight 0.65)
                for cb in hit_colors {
                    let p_cb = [cb.pos3d[0] as f64, cb.pos3d[1] as f64, cb.pos3d[2] as f64];
                    let r_cb = mat3_vec3(rot, p_cb);
                    let xj = [r_cb[0] + trans[0], r_cb[1] + trans[1], r_cb[2] + trans[2]];
                    let arm = sub3(xj, cq);

                    for ca in query_colors {
                        if ca.kind != cb.kind {
                            continue;
                        }
                        let qi = [ca.pos3d[0] as f64, ca.pos3d[1] as f64, ca.pos3d[2] as f64];
                        let diff = sub3(qi, xj);
                        let d2 = dot3(diff, diff);
                        if kc_exp * d2 > 10.0 {
                            continue;
                        }
                        let v_c = 2.40 * pref_c * (-kc_exp * d2).exp();
                        cur_obj += v_c;
                        let s = 2.0 * kc_exp * v_c;
                        let fj = [s * diff[0], s * diff[1], s * diff[2]];
                        force_sum[0] += fj[0];
                        force_sum[1] += fj[1];
                        force_sum[2] += fj[2];
                        let tq = cross3(arm, fj);
                        torque_sum[0] += tq[0];
                        torque_sum[1] += tq[1];
                        torque_sum[2] += tq[2];
                    }
                }

                if cur_obj > best_obj {
                    best_obj = cur_obj;
                    best_rot = rot;
                    best_trans = trans;
                }

                let fnrm = norm3(force_sum);
                let tnrm = norm3(torque_sum);
                if fnrm < 1e-4 && tnrm < 1e-4 {
                    break;
                }

                let dt_scale = (step_t / fnrm.max(1.0)).min(step_t);
                let dr_scale = (step_r / tnrm.max(1.0)).min(step_r);
                let d_omega = [
                    torque_sum[0] * dr_scale,
                    torque_sum[1] * dr_scale,
                    torque_sum[2] * dr_scale,
                ];
                let d_rot = rodrigues_rot(d_omega);

                // Rotate around cq + translate by force_sum * dt_scale
                // x' = cq + d_rot * (rot * h + trans - cq) + dt
                //    = (d_rot * rot) * h + (cq + d_rot * (trans - cq) + dt)
                let new_rot = mat3_mul(d_rot, rot);
                let tc = sub3(trans, cq);
                let r_tc = mat3_vec3(d_rot, tc);
                let new_trans = [
                    cq[0] + r_tc[0] + force_sum[0] * dt_scale,
                    cq[1] + r_tc[1] + force_sum[1] * dt_scale,
                    cq[2] + r_tc[2] + force_sum[2] * dt_scale,
                ];
                rot = new_rot;
                trans = new_trans;
                step_t *= 0.94;
                step_r *= 0.94;
            }
        }
    }

    // Transform hit coordinates & hit color atoms by best (best_rot, best_trans)
    let aligned_f64: Vec<[f64; 3]> = h_pts
        .iter()
        .map(|&p| {
            let rp = mat3_vec3(best_rot, p);
            [rp[0] + best_trans[0], rp[1] + best_trans[1], rp[2] + best_trans[2]]
        })
        .collect();
    let aligned_hit_coords: Vec<[f32; 3]> = aligned_f64
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();

    let aligned_hit_colors: Vec<RocsColorAtom> = hit_colors
        .iter()
        .map(|c| {
            let rp = mat3_vec3(
                best_rot,
                [c.pos3d[0] as f64, c.pos3d[1] as f64, c.pos3d[2] as f64],
            );
            RocsColorAtom {
                kind: c.kind,
                pos3d: [
                    (rp[0] + best_trans[0]) as f32,
                    (rp[1] + best_trans[1]) as f32,
                    (rp[2] + best_trans[2]) as f32,
                ],
                label: c.label,
            }
        })
        .collect();

    let o_qq = gaussian_shape_overlap(query_mol, &q_pts, query_mol, &q_pts);
    let o_hh = gaussian_shape_overlap(hit_mol, &aligned_f64, hit_mol, &aligned_f64);
    let o_qh = gaussian_shape_overlap(query_mol, &q_pts, hit_mol, &aligned_f64);

    let shape_denom = (o_qq + o_hh - o_qh).max(1e-9);
    let shape_tanimoto = (o_qh / shape_denom).clamp(0.0, 1.0);
    let shape_tversky_query =
        (o_qh / (0.95 * o_qq + 0.05 * o_hh).max(1e-9)).clamp(0.0, 1.0);

    let c_qq = gaussian_color_overlap(query_colors, query_colors);
    let c_hh = gaussian_color_overlap(&aligned_hit_colors, &aligned_hit_colors);
    let c_qh = gaussian_color_overlap(query_colors, &aligned_hit_colors);
    let color_feat_tan = if c_qq + c_hh - c_qh > 1e-9 {
        (c_qh / (c_qq + c_hh - c_qh)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let esp_sim = gaussian_esp_similarity(&q_pts, query_charges, &aligned_f64, hit_charges);
    let color_tanimoto = (0.65 * color_feat_tan + 0.35 * esp_sim).clamp(0.0, 1.0);
    let combo_score = shape_tanimoto + color_tanimoto; // OpenEye ROCS standard [0.0, 2.0]
    let combo_tanimoto = 0.5 * combo_score;

    let (usr_q, usrcat_q) = compute_usr_descriptors(query_mol, query_coords);
    let (usr_h, usrcat_h) = compute_usr_descriptors(hit_mol, &aligned_hit_coords);
    let usr_sim = usr_similarity(&usr_q, &usr_h);
    let usrcat_sim = usrcat_similarity(&usrcat_q, &usrcat_h);

    let mut ah_cent = [0.0f64; 3];
    for p in &aligned_f64 {
        ah_cent[0] += p[0];
        ah_cent[1] += p[1];
        ah_cent[2] += p[2];
    }
    let inv_nh = 1.0 / (nh as f64);
    ah_cent[0] *= inv_nh;
    ah_cent[1] *= inv_nh;
    ah_cent[2] *= inv_nh;
    let centroid_rmsd = norm3(sub3(cq, ah_cent)) as f32;

    Rocs3dAlignment {
        rotation: [
            [
                best_rot[0][0] as f32,
                best_rot[0][1] as f32,
                best_rot[0][2] as f32,
            ],
            [
                best_rot[1][0] as f32,
                best_rot[1][1] as f32,
                best_rot[1][2] as f32,
            ],
            [
                best_rot[2][0] as f32,
                best_rot[2][1] as f32,
                best_rot[2][2] as f32,
            ],
        ],
        translation: [
            best_trans[0] as f32,
            best_trans[1] as f32,
            best_trans[2] as f32,
        ],
        aligned_hit_coords,
        aligned_hit_colors,
        best_hit_conformer_rank: 1,
        shape_overlap_vol: o_qh,
        shape_tanimoto,
        shape_tversky_query,
        color_feature_tanimoto: color_feat_tan,
        esp_hodgkin_sim: esp_sim,
        color_tanimoto,
        combo_score,
        combo_tanimoto,
        usr_similarity: usr_sim,
        usrcat_similarity: usrcat_sim,
        centroid_rmsd,
    }
}

/// Evaluates `ROCS` 3D alignment across all conformers in `hit_ens` against the lowest-energy
/// conformer of `query_ens` and returns the highest-`ComboScore` pose.
pub fn align_rocs_ensemble_3d(
    query_mol: &Molecule,
    query_ens: &ConformerEnsemble,
    hit_mol: &Molecule,
    hit_ens: &ConformerEnsemble,
) -> Rocs3dAlignment {
    let Some(q0) = query_ens.conformers.first() else {
        return align_rocs_3d(query_mol, &[], &[], &[], hit_mol, &[], &[], &[]);
    };
    let mut best: Option<Rocs3dAlignment> = None;
    for hc in &hit_ens.conformers {
        let mut al = align_rocs_3d(
            query_mol,
            &q0.coords3d,
            &query_ens.gasteiger_charges,
            &q0.color_atoms,
            hit_mol,
            &hc.coords3d,
            &hit_ens.gasteiger_charges,
            &hc.color_atoms,
        );
        al.best_hit_conformer_rank = hc.rank;
        if best
            .as_ref()
            .map(|b| al.combo_score > b.combo_score)
            .unwrap_or(true)
        {
            best = Some(al);
        }
    }
    best.unwrap_or_else(|| align_rocs_3d(query_mol, &[], &[], &[], hit_mol, &[], &[], &[]))
}

/// Two-stage 3D virtual screening across `ds`:
/// 1. Ultrafast `USRCAT` 60-D pre-ranking across all visible dataset molecules (`O(N)` microseconds),
/// 2. Full parallel `Rayon` 6-DOF `ROCS` Gaussian shape + color/ESP superposition across the top
///    `top_rocs_rerank` candidates, sorted descending by `ComboScore (0.0..2.0)`.
pub fn screen_3d_overlay_library(
    query_mol: &Molecule,
    query_summary: &Summary3dProfile,
    ds: &DatasetAnalysis,
    visible_mask: Option<&[bool]>,
    scaffold_hops_only: bool,
    top_rocs_rerank: usize,
) -> Vec<Overlay3dHit> {
    if ds.mols.is_empty() || query_summary.coords3d.is_empty() {
        return Vec::new();
    }

    let q_fp = compute_ecfp(query_mol, 2);
    let (q_murcko, _, _, _) = murcko_signatures(query_mol);

    // Stage 1: USRCAT pre-filter across visible candidates
    let mut pre_candidates: Vec<(usize, f64, bool)> = (0..ds.mols.len())
        .filter_map(|i| {
            if let Some(mask) = visible_mask {
                if !mask.get(i).copied().unwrap_or(true) {
                    return None;
                }
            }
            let (ds_m_fw, _, _, _) = murcko_signatures(&ds.mols[i]);
            let is_hop = !q_murcko.is_empty() && ds_m_fw != q_murcko;
            if scaffold_hops_only && !is_hop {
                return None;
            }
            let usrcat = usrcat_similarity(&query_summary.usrcat60, &ds.shape3d[i].usrcat60);
            Some((i, usrcat, is_hop))
        })
        .collect();

    pre_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pre_candidates.truncate(top_rocs_rerank.max(20));

    // Stage 2: Parallel 6-DOF Gaussian ROCS superposition
    let mut hits: Vec<Overlay3dHit> = pre_candidates
        .par_iter()
        .map(|&(mi, _, is_hop)| {
            let hm = &ds.mols[mi];
            let hs = &ds.shape3d[mi];
            let al = align_rocs_3d(
                query_mol,
                &query_summary.coords3d,
                &query_summary.gasteiger_charges,
                &query_summary.color_atoms,
                hm,
                &hs.coords3d,
                &hs.gasteiger_charges,
                &hs.color_atoms,
            );
            let ecfp_sim = q_fp.tanimoto(&ds.fps[mi]);
            Overlay3dHit {
                mol_idx: mi,
                compound_id: hm.id.clone(),
                compound_name: hm.name.clone(),
                target: hm.target.clone(),
                pact: hm.pact,
                ecfp4_tanimoto_2d: ecfp_sim,
                is_scaffold_hop: is_hop,
                npr1: hs.npr1,
                npr2: hs.npr2,
                pbf_angstrom: hs.pbf_angstrom,
                dipole_debye: hs.dipole_debye,
                alignment: al,
            }
        })
        .collect();

    hits.sort_by(|a, b| {
        b.alignment
            .combo_score
            .partial_cmp(&a.alignment.combo_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

/// Formats a 3D conformer as a true 3D MDL V2000 `.sdf` record (`sar-atlas 1.2.0   3D`) with
/// `(x, y, z)` Angstrom coordinates, `M  CHG`, Gasteiger partial charges, and `MMFF-lite` energy tags.
pub fn format_3d_conformer_sdf(
    mol: &Molecule,
    coords3d: &[[f32; 3]],
    charges: &[f32],
    energy_kcal: f64,
    extra_tags: &[(&str, String)],
) -> String {
    let mut out = String::with_capacity(2560);
    let title = if mol.name.is_empty() { &mol.id } else { &mol.name };
    out.push_str(title);
    out.push('\n');
    out.push_str("  sar-atlas 1.2.0   3D\n");
    out.push_str("Generated by sar-core Distance-Geometry + MMFF-lite 3D engine\n");
    out.push_str(&format!(
        "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
        mol.atoms.len(),
        mol.bonds.len()
    ));

    let mut formal_chg: Vec<(usize, i8)> = Vec::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        if a.charge != 0 {
            formal_chg.push((i + 1, a.charge));
        }
        let p = coords3d.get(i).copied().unwrap_or([a.pos2d[0], a.pos2d[1], 0.0]);
        out.push_str(&format!(
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            p[0],
            p[1],
            p[2],
            a.elem.symbol()
        ));
    }

    for b in &mol.bonds {
        let btype: u8 = match b.order {
            BondOrder::Single => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => b.kekule_order.clamp(1, 2),
        };
        out.push_str(&format!(
            "{:>3}{:>3}{:>3}  0  0  0  0\n",
            b.a + 1,
            b.b + 1,
            btype
        ));
    }

    if !formal_chg.is_empty() {
        for chunk in formal_chg.chunks(8) {
            out.push_str(&format!("M  CHG{:>3}", chunk.len()));
            for &(idx1, chg) in chunk {
                out.push_str(&format!(" {:>3} {:>3}", idx1, chg));
            }
            out.push('\n');
        }
    }
    out.push_str("M  END\n");

    let q_csv = charges
        .iter()
        .map(|q| format!("{q:+.4}"))
        .collect::<Vec<_>>()
        .join(",");
    let base_tags: [(&str, String); 6] = [
        ("ID", mol.id.clone()),
        ("NAME", mol.name.clone()),
        ("TARGET", mol.target.clone()),
        ("PACT", format!("{:.2}", mol.pact)),
        ("MMFF_LITE_ENERGY_KCAL", format!("{energy_kcal:.3}")),
        ("GASTEIGER_PARTIAL_CHARGES", q_csv),
    ];
    for (k, v) in &base_tags {
        if !v.is_empty() {
            out.push_str(&format!("> <{k}>\n{v}\n\n"));
        }
    }
    for (k, v) in extra_tags {
        if !v.is_empty() {
            out.push_str(&format!("> <{k}>\n{v}\n\n"));
        }
    }
    out.push_str("$$$$\n");
    out
}

/// Formats a 2-record overlaid 3D MDL V2000 `.sdf` containing the reference `query_mol` pose and
/// the rigid-body `ROCS`-superimposed `hit_mol` pose ready for PyMOL / ChimeraX / Mol*.
pub fn format_3d_overlay_sdf(
    query_mol: &Molecule,
    query_summary: &Summary3dProfile,
    hit_mol: &Molecule,
    hit_charges: &[f32],
    hit_energy_kcal: f64,
    align: &Rocs3dAlignment,
) -> String {
    let mut out = format_3d_conformer_sdf(
        query_mol,
        &query_summary.coords3d,
        &query_summary.gasteiger_charges,
        query_summary.mmff_energy_kcal,
        &[("ROLE", "3D_QUERY_REFERENCE".to_string())],
    );
    let hit_tags = [
        ("ROLE", "3D_ROCS_ALIGNED_HIT".to_string()),
        ("ROCS_COMBO_SCORE", format!("{:.4}", align.combo_score)),
        ("ROCS_SHAPE_TANIMOTO", format!("{:.4}", align.shape_tanimoto)),
        ("ROCS_COLOR_TANIMOTO", format!("{:.4}", align.color_tanimoto)),
        ("ESP_HODGKIN_SIM", format!("{:.4}", align.esp_hodgkin_sim)),
        ("USR_SIMILARITY", format!("{:.4}", align.usr_similarity)),
        ("USRCAT_SIMILARITY", format!("{:.4}", align.usrcat_similarity)),
    ];
    out.push_str(&format_3d_conformer_sdf(
        hit_mol,
        &align.aligned_hit_coords,
        hit_charges,
        hit_energy_kcal,
        &hit_tags,
    ));
    out
}

/// Exports ranked 3D `ROCS` + `USR`/`USRCAT` virtual-screening hits to CSV.
pub fn export_3d_overlay_hits_csv(hits: &[Overlay3dHit]) -> String {
    let mut out = String::from(
        "rank,id,name,target,pact,rocs_combo_score,rocs_shape_tanimoto,rocs_color_tanimoto,esp_hodgkin_sim,usr_sim,usrcat_sim,ecfp4_2d_sim,scaffold_hop,npr1,npr2,pbf_angstrom,dipole_debye\n",
    );
    for (rk, h) in hits.iter().enumerate() {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.3},{:.3},{:.3},{:.2}\n",
            rk + 1,
            h.compound_id,
            h.compound_name,
            h.target,
            h.pact,
            h.alignment.combo_score,
            h.alignment.shape_tanimoto,
            h.alignment.color_tanimoto,
            h.alignment.esp_hodgkin_sim,
            h.alignment.usr_similarity,
            h.alignment.usrcat_similarity,
            h.ecfp4_tanimoto_2d,
            h.is_scaffold_hop,
            h.npr1,
            h.npr2,
            h.pbf_angstrom,
            h.dipole_debye
        ));
    }
    out
}
