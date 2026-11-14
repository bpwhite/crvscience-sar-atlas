//! Ertl & Schuffenhauer (2009) Synthetic Accessibility Score (`SAScore`) and
//! Degen et al. (2008) `BRICS` Retrosynthetic Bond Disconnector & Synthon Mapper.
//!
//! Primary Literature Grounding:
//! - Ertl & Schuffenhauer (2009), *J. Cheminform.* 1:8 (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3225829/`):
//!   Estimation of synthetic accessibility score (`SAScore`, scale `1.0` easy to `10.0` difficult)
//!   from extended-connectivity PubChem fragment log-odds (`fragmentScore`) minus topological/stereo
//!   complexity penalties (`sizePenalty`, `stereoPenalty`, `spiroPenalty`, `bridgePenalty`, `macroPenalty`).
//! - Ertl & Landrum Novartis / RDKit `sascorer.py` (`https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py`):
//!   Exact complexity penalty exponents (`n_atoms^1.005 - n_atoms`, `log10(n + 1)`), molecular symmetry
//!   correction (`0.5 * ln(n_atoms / n_unique_envs)`), and `[1, 10]` scaling + `> 8.0` log compression.
//! - Degen, Wegscheid-Gerlach, Zaliani & Rarey (2008), *ChemMedChem* 3(10), 1503–1507
//!   (`https://pubmed.ncbi.nlm.nih.gov/18792903/`): Breaking of Retrosynthetically Interesting
//!   Chemical Substructures (`BRICS`) across `L1`–`L16` medicinal-chemistry coupling environments.
//! - Lewell, Judd, Watson & Hann (1998), *J. Chem. Inf. Comput. Sci.* 38(3), 511–522
//!   (`https://pubmed.ncbi.nlm.nih.gov/9611787/`): `RECAP` retrosynthetic linkage archetypes.

use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::sketcher::molecule_to_smiles;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Medicinal-chemistry synthetic tractability tier per Ertl & Schuffenhauer (2009) (`SAScore = 6.0` cutoff).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthTractability {
    /// `SAScore <= 3.5`: 1–3 standard parallel-synthesis couplings from commercial building blocks.
    Easy,
    /// `3.5 < SAScore <= 6.0`: Multi-step lead-optimization synthesis.
    Moderate,
    /// `SAScore > 6.0`: Complex polycyclic, stereochemical, spiro/bridged, or macrocyclic synthesis.
    Challenging,
}

impl SynthTractability {
    pub fn from_sa_score(sa: f64) -> Self {
        if sa <= 3.5 {
            Self::Easy
        } else if sa <= 6.0 {
            Self::Moderate
        } else {
            Self::Challenging
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Easy => "Easy (Catalog / Parallel)",
            Self::Moderate => "Moderate (Multistep MedChem)",
            Self::Challenging => "Challenging (Complex / NP-like)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Easy => "Easy",
            Self::Moderate => "Moderate",
            Self::Challenging => "Hard",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Easy => [52, 211, 153],        // Emerald
            Self::Moderate => [251, 191, 36],    // Amber
            Self::Challenging => [248, 113, 113], // Coral Red
        }
    }
}

/// Complete Ertl & Schuffenhauer (2009) `SAScore` decomposition and `BRICS` summary metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthAccessProfile {
    /// Scaled Synthetic Accessibility Score in `[1.0, 10.0]` (`1.0` = easiest, `10.0` = hardest).
    pub sa_score: f64,
    pub tractability: SynthTractability,
    /// Mean `ECFC_4` / radius-2 fragment log-odds contribution (`[-2.5, +1.6]`).
    pub fragment_score: f64,
    /// `n_heavy^1.005 - n_heavy` (`sascorer.py`).
    pub size_penalty: f64,
    /// `log10(chiral_centers + 1)` (`sascorer.py`).
    pub stereo_penalty: f64,
    /// `log10(spiro_atoms + 1)` (`sascorer.py`).
    pub spiro_penalty: f64,
    /// `log10(bridgehead_atoms + 1)` (`sascorer.py`).
    pub bridgehead_penalty: f64,
    /// `log10(2)` (`0.30103`) if any ring has size `> 8`, else `0.0`.
    pub macrocycle_penalty: f64,
    /// Positive topological symmetry bonus `0.5 * ln(n_heavy / n_unique_envs)` (`sascorer.py` `score3`).
    pub symmetry_bonus: f64,
    pub heavy_atoms: usize,
    pub unique_env_count: usize,
    pub chiral_centers: usize,
    pub spiro_atoms: usize,
    pub bridgehead_atoms: usize,
    pub macrocycle_rings: usize,
    /// Atom indices contributing topological complexity (stereocenters, spiro, bridgehead, macrocycle/strained).
    pub complexity_atoms: Vec<usize>,
    /// Total candidate non-ring `BRICS` retrosynthetic bonds in `mol`.
    pub brics_cleavable_bonds: usize,
    /// Number of multi-atom building-block synthons obtained after greedy `BRICS` shredding.
    pub brics_synthon_count: usize,
    /// Estimated convergent coupling steps (`max(1, active_cuts)` adjusted for symmetry).
    pub convergent_steps_est: usize,
}

/// Single `BRICS` (`Degen et al. 2008`) retrosynthetically cleavable bond across `L1`–`L16`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BricsBondCut {
    pub bond_idx: usize,
    pub atom_a: usize,
    pub atom_b: usize,
    pub env_a: &'static str,
    pub env_b: &'static str,
    /// Canonical pair code, e.g. `"L14-L5"` or `"L16-L14"`.
    pub pair_code: String,
    /// Forward synthetic coupling reaction name (e.g. `"SNAr / Buchwald-Hartwig Amination"`).
    pub reaction_name: &'static str,
    /// Priority score (`1..=100`) ordering synthetic disconnections in multi-bond shredding.
    pub priority: u8,
    /// `true` if cleaved in the non-degenerate multi-bond `BRICS` synthon tree (`>= 2` heavy atoms per piece).
    pub selected_in_tree: bool,
}

/// Disconnected `BRICS` building-block synthon with 2D coordinates, reagent SMILES, and `SAScore`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BricsSynthon {
    pub idx: usize,
    /// `"Core Scaffold Brick"`, `"Ring Synthon"`, or `"Side-Chain / Linker Reagent"`.
    pub role: &'static str,
    /// Forward reagent archetype (e.g. `"Heteroaryl Electrophile ([14*])"`).
    pub reagent_archetype: String,
    /// Dummy attachment environment tags on this synthon (e.g. `["[14*]", "[3*]", "[3*]"]`).
    pub attachment_tags: Vec<String>,
    /// Hydrogen-capped canonical OpenSMILES of the building block.
    pub reagent_smiles: String,
    pub formula: String,
    pub mw: f64,
    /// Ertl `SAScore` of this isolated building block (`[1.0, 10.0]`).
    pub sa_score: f64,
    /// Atom indices in the parent `Molecule` belonging to this synthon.
    pub parent_atoms: Vec<usize>,
    /// Sanitized standalone `Molecule` with freshly computed 2D skeletal coordinates for GUI rendering.
    pub mol_2d: Molecule,
}

/// Complete `BRICS` retrosynthetic disconnection tree and building-block synthon inventory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BricsRetrosynthesis {
    pub parent_sa: SynthAccessProfile,
    pub candidate_cuts: Vec<BricsBondCut>,
    pub active_cuts: Vec<BricsBondCut>,
    pub synthons: Vec<BricsSynthon>,
    pub mean_synthon_sa: f64,
    pub max_synthon_sa: f64,
    /// Reduction in maximum molecular `SAScore` achieved by retrosynthetic disconnection (`parent_sa - max_synthon_sa`).
    pub convergence_sa_drop: f64,
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Computes radius-2 Morgan / Weisfeiler-Lehman topological environment hashes per atom for
/// `SAScore` symmetry (`score3`) and topological tetrahedral stereocenter perception.
fn compute_atom_morgan_hashes(mol: &Molecule, rounds: usize) -> Vec<u64> {
    let n = mol.atoms.len();
    let mut cur = vec![0u64; n];
    for (u, a) in mol.atoms.iter().enumerate() {
        let base = ((a.elem.atomic_number() as u64) << 24)
            | ((mol.heavy_degree(u) as u64) << 16)
            | ((a.total_h() as u64) << 10)
            | (((a.charge as i16 + 8) as u64) << 4)
            | ((a.aromatic as u64) << 2)
            | (a.in_ring as u64);
        cur[u] = splitmix64(base);
    }

    for r in 0..rounds {
        let mut next = cur.clone();
        for u in 0..n {
            if mol.atoms[u].elem == Element::H {
                continue;
            }
            let mut nbr_sigs: Vec<u64> = mol.adj[u]
                .iter()
                .filter(|&&(v, _)| mol.atoms[v].elem != Element::H)
                .map(|&(v, bi)| {
                    let b_code = match mol.bonds[bi].order {
                        BondOrder::Single => 1u64,
                        BondOrder::Double => 2u64,
                        BondOrder::Triple => 3u64,
                        BondOrder::Aromatic => 4u64,
                    };
                    splitmix64(cur[v] ^ (b_code.wrapping_mul(0x9E37_79B9)) ^ ((r as u64) << 40))
                })
                .collect();
            nbr_sigs.sort_unstable();
            let mut acc = splitmix64(cur[u] ^ ((r as u64 + 1) * 0x517C_C1B7_2722_0A95));
            for s in nbr_sigs {
                acc = splitmix64(acc.wrapping_add(s));
            }
            next[u] = acc;
        }
        cur = next;
    }
    cur
}

/// Evaluates the Ertl & Schuffenhauer (2009) PubChem fragment log-odds contribution for heavy atom `u`
/// based on its extended radius-2 (`ECFC_4` / HOSE-2) medicinal-chemistry environment.
fn atom_pubchem_fragment_log_odds(mol: &Molecule, u: usize) -> f64 {
    let a = &mol.atoms[u];
    let hdeg = mol.heavy_degree(u);
    let th = a.total_h();

    // Strained small rings (3- or 4-membered) or medium 7/8-membered rings reduce fragment frequency
    let min_ring = a.ring_sizes.iter().copied().min().unwrap_or(0);
    let ring_strain_adj = match min_ring {
        3 => -0.55,
        4 => -0.70,
        7 => -0.25,
        8 => -0.45,
        r if r >= 9 => -0.65,
        _ => 0.0,
    };

    // Count aromatic vs polar vs halogen neighbors
    let mut arom_nbrs = 0usize;
    let mut oxo_double = 0usize;
    let mut hetero_single = 0usize;
    let mut fluoro_nbrs = 0usize;
    for &(v, bi) in &mol.adj[u] {
        let nb = &mol.atoms[v];
        if nb.elem == Element::H {
            continue;
        }
        if nb.aromatic {
            arom_nbrs += 1;
        }
        if nb.elem == Element::F {
            fluoro_nbrs += 1;
        }
        let b = &mol.bonds[bi];
        if matches!(nb.elem, Element::O | Element::S)
            && (b.order == BondOrder::Double || b.kekule_order == 2)
        {
            oxo_double += 1;
        } else if nb.is_hetero() {
            hetero_single += 1;
        }
    }

    let base_score = match a.elem {
        Element::C => {
            if a.aromatic {
                // Ertl Fig. 2 Fragments 1, 3, 7, 9, 13, 26: aromatic CH & monosubstituted aryls are top 1% in PubChem
                if th >= 1 && hdeg == 2 {
                    1.22
                } else if hdeg == 3 && hetero_single <= 1 {
                    // Standard substituted aryl or benzo-ring fusion carbon
                    0.92
                } else {
                    // Polysubstituted / heteroatom-crowded heteroaromatic carbon
                    0.55
                }
            } else if oxo_double >= 1 {
                // Ertl Table 1: carbonyls in amides, esters, ureas, carboxylic acids (Fragments 2, 6, 8)
                1.08
            } else if fluoro_nbrs >= 2 {
                // Commercial -CF3 / -CF2- building block
                0.78
            } else if a.hyb == Hybridization::Sp3 {
                match hdeg {
                    1 => 1.15, // Methyl (Fragments 5, 18, 22)
                    2 => {
                        if a.in_ring {
                            0.82 // Piperazine / morpholine / piperidine / cyclohexyl CH2 (Fragment 17)
                        } else {
                            0.95 // Acyclic methylene linker (Fragment 4, 23)
                        }
                    }
                    3 => {
                        if hetero_single >= 2 {
                            -0.35 // Acetal / hemiaminal methine
                        } else {
                            0.32 // Branched methine CH
                        }
                    }
                    4 => -0.42, // Quaternary carbon center (neopentyl / gem-disubstituted / spiro)
                    _ => 0.40,
                }
            } else if a.hyb == Hybridization::Sp2 {
                0.52 // Acyclic alkene / imine carbon
            } else {
                0.48 // Terminal / internal alkyne or nitrile carbon
            }
        }
        Element::N => {
            if a.charge != 0 {
                -0.25
            } else if a.aromatic {
                // Pyridine / pyrimidine / quinazoline / pyrazole / indole aromatic N (Fragment 14)
                0.88
            } else if hdeg <= 2 && arom_nbrs >= 1 {
                // Aniline / diarylamine NH (Fragment 28)
                0.94
            } else if hdeg <= 3 {
                // Amide, sulfonamide, tertiary piperazine/piperidine amine (Fragments 12, 16, 17)
                0.86
            } else {
                -0.40
            }
        }
        Element::O => {
            if a.charge < 0 {
                0.55
            } else if a.aromatic {
                0.62 // Furan / oxazole / isoxazole O
            } else if hdeg == 1 {
                // Carbonyl =O, sulfonyl =O, or phenolic/aliphatic -OH (Fragments 8, 20)
                1.06
            } else {
                // Methoxy / ethoxy / morpholine ether -O- (Fragments 10, 19)
                0.96
            }
        }
        Element::F => 1.05,  // Aryl-F / CF3 (Fragment 21)
        Element::Cl => 1.02, // Aryl-Cl / Alkyl-Cl (Fragment 27)
        Element::Br => 0.85, // Common Suzuki/Buchwald aryl bromide reagent
        Element::I => 0.55,
        Element::S => {
            if oxo_double >= 2 {
                // Sulfonamide / sulfone -SO2- (Table 1 Sulfonamide formation)
                0.82
            } else if a.aromatic {
                0.78 // Thiophene / thiazole / phenothiazine ring S
            } else if oxo_double == 1 {
                0.15 // Sulfoxide (chiral sulfur)
            } else {
                0.52 // Thioether
            }
        }
        Element::P => -0.35,
        Element::B => 0.10, // Boronic acid / ester coupling handle
        Element::Si | Element::Se | Element::Other(_) => -0.85,
        Element::H => 0.0,
    };

    base_score + ring_strain_adj
}

/// Computes `SAScore` and topological complexity breakdown without invoking recursive synthon layout.
fn compute_sa_metrics_only(mol: &Molecule) -> (f64, f64, f64, f64, f64, f64, f64, f64, usize, usize, usize, usize, usize, usize, Vec<usize>) {
    let heavy_indices: Vec<usize> = (0..mol.atoms.len())
        .filter(|&u| mol.atoms[u].elem != Element::H)
        .collect();
    let n_heavy = heavy_indices.len();
    if n_heavy == 0 {
        return (
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, Vec::new(),
        );
    }

    // 1. PubChem ECFC_4 / HOSE-2 mean fragment score
    let frag_sum: f64 = heavy_indices
        .iter()
        .map(|&u| atom_pubchem_fragment_log_odds(mol, u))
        .sum();
    let fragment_score = frag_sum / (n_heavy as f64);

    // 2. Symmetry & topological stereocenter perception via 4-round Morgan hashes
    let env_hashes = compute_atom_morgan_hashes(mol, 4);
    let unique_envs: BTreeSet<u64> = heavy_indices.iter().map(|&u| env_hashes[u]).collect();
    let n_unique_envs = unique_envs.len().max(1);

    let mut complexity_set: BTreeSet<usize> = BTreeSet::new();

    // Detect explicit or topological tetrahedral stereocenters:
    // sp3 C (or quaternary N+) with 4 distinct substituents (`total_h <= 1` and all neighbor Morgan hashes distinct)
    let mut n_chiral = 0usize;
    for &u in &heavy_indices {
        let a = &mol.atoms[u];
        if a.chiral != 0 {
            n_chiral += 1;
            complexity_set.insert(u);
            continue;
        }
        if !a.aromatic
            && a.hyb == Hybridization::Sp3
            && matches!(a.elem, Element::C | Element::N)
            && a.total_h() <= 1
        {
            let heavy_nbrs: Vec<usize> = mol.adj[u]
                .iter()
                .filter_map(|&(v, bi)| {
                    (mol.atoms[v].elem != Element::H
                        && mol.bonds[bi].order == BondOrder::Single)
                        .then_some(v)
                })
                .collect();
            let total_substituents = heavy_nbrs.len() + (a.total_h() as usize);
            if total_substituents == 4 && heavy_nbrs.len() >= 3 {
                let mut branch_hashes: Vec<u64> =
                    heavy_nbrs.iter().map(|&v| env_hashes[v]).collect();
                branch_hashes.sort_unstable();
                branch_hashes.dedup();
                if branch_hashes.len() == heavy_nbrs.len() {
                    n_chiral += 1;
                    complexity_set.insert(u);
                }
            }
        }
    }

    // 3. Spiro & Bridgehead ring complexity perception across SSSR rings
    let nr = mol.rings.len();
    let mut ring_bond_sets: Vec<BTreeSet<usize>> = Vec::with_capacity(nr);
    for r in &mol.rings {
        let mut bset = BTreeSet::new();
        for idx in 0..r.len() {
            let u = r[idx];
            let v = r[(idx + 1) % r.len()];
            if let Some(bi) = mol.bond_between(u, v) {
                bset.insert(bi);
            }
        }
        ring_bond_sets.push(bset);
    }

    let mut spiro_atoms_set: BTreeSet<usize> = BTreeSet::new();
    let mut bridgehead_atoms_set: BTreeSet<usize> = BTreeSet::new();

    for r1 in 0..nr {
        for r2 in (r1 + 1)..nr {
            let shared_atoms: Vec<usize> = mol.rings[r1]
                .iter()
                .copied()
                .filter(|u| mol.rings[r2].contains(u))
                .collect();
            let shared_bonds = ring_bond_sets[r1].intersection(&ring_bond_sets[r2]).count();
            if shared_atoms.len() == 1 && shared_bonds == 0 {
                spiro_atoms_set.insert(shared_atoms[0]);
                complexity_set.insert(shared_atoms[0]);
            } else if shared_bonds >= 2 {
                // Bridged bicyclic system (e.g. tropane, norbornane, quinuclidine):
                // the two terminal atoms of the bridge path have >= 3 ring-bond incidences across r1 U r2
                for &ua in &shared_atoms {
                    let ring_deg_in_pair = mol.adj[ua]
                        .iter()
                        .filter(|&&(_, bi)| {
                            ring_bond_sets[r1].contains(&bi) || ring_bond_sets[r2].contains(&bi)
                        })
                        .count();
                    if ring_deg_in_pair >= 3 {
                        bridgehead_atoms_set.insert(ua);
                        complexity_set.insert(ua);
                    }
                }
            }
        }
    }
    // Also any atom belonging to >= 3 SSSR rings is a polycyclic bridgehead/pericondensed junction
    for &u in &heavy_indices {
        let r_membership = mol.rings.iter().filter(|r| r.contains(&u)).count();
        if r_membership >= 3 {
            bridgehead_atoms_set.insert(u);
            complexity_set.insert(u);
        }
    }

    let mut macrocycle_rings = 0usize;
    for r in &mol.rings {
        if r.len() > 8 {
            macrocycle_rings += 1;
            for &u in r {
                complexity_set.insert(u);
            }
        }
    }

    let n_spiro = spiro_atoms_set.len();
    let n_bridge = bridgehead_atoms_set.len();

    // Exact Ertl & Schuffenhauer (2009) / `sascorer.py` penalty formulas:
    let size_penalty = (n_heavy as f64).powf(1.005) - (n_heavy as f64);
    let stereo_penalty = ((n_chiral as f64) + 1.0).log10();
    let spiro_penalty = ((n_spiro as f64) + 1.0).log10();
    let bridgehead_penalty = ((n_bridge as f64) + 1.0).log10();
    let macrocycle_penalty = if macrocycle_rings > 0 {
        2.0_f64.log10()
    } else {
        0.0
    };

    let score2 = -size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridgehead_penalty
        - macrocycle_penalty;

    let symmetry_bonus = if n_heavy > n_unique_envs {
        ((n_heavy as f64) / (n_unique_envs as f64)).ln() * 0.5
    } else {
        0.0
    };

    let raw_sa = fragment_score + score2 + symmetry_bonus;
    // Scale `[-4.0, 2.5]` onto `[10.0, 1.0]` per `sascorer.py`:
    let mut sa_score = 11.0 - (raw_sa - (-4.0) + 1.0) / (2.5 - (-4.0)) * 9.0;
    if sa_score > 8.0 {
        sa_score = 8.0 + (sa_score + 1.0 - 9.0).max(1.0001).ln();
    }
    sa_score = sa_score.clamp(1.0, 10.0);

    (
        sa_score,
        fragment_score,
        size_penalty,
        stereo_penalty,
        spiro_penalty,
        bridgehead_penalty,
        macrocycle_penalty,
        symmetry_bonus,
        n_heavy,
        n_unique_envs,
        n_chiral,
        n_spiro,
        n_bridge,
        macrocycle_rings,
        complexity_set.into_iter().collect(),
    )
}

// =============================================================================
// Degen et al. (2008) `BRICS` L1..L16 Environment Classifier & Bond Disconnector
// =============================================================================

fn has_carbonyl_double_bond_o(mol: &Molecule, u: usize) -> bool {
    mol.atoms[u].elem == Element::C
        && !mol.atoms[u].aromatic
        && mol.adj[u].iter().any(|&(v, bi)| {
            mol.atoms[v].elem == Element::O
                && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
        })
}

fn has_sulfonyl_two_double_o(mol: &Molecule, u: usize) -> bool {
    if mol.atoms[u].elem != Element::S || mol.atoms[u].aromatic {
        return false;
    }
    let ox_cnt = mol.adj[u]
        .iter()
        .filter(|&&(v, bi)| {
            mol.atoms[v].elem == Element::O
                && (mol.bonds[bi].order == BondOrder::Double || mol.bonds[bi].kekule_order == 2)
        })
        .count();
    ox_cnt >= 2
}

fn ring_has_heteroatom(mol: &Molecule, u: usize) -> bool {
    mol.rings.iter().any(|r| {
        r.contains(&u)
            && r.iter()
                .any(|&rv| !matches!(mol.atoms[rv].elem, Element::C | Element::H))
    })
}

/// Assigns Degen et al. (2008) `BRICS` chemical environment tag (`L1`..`L16`) to atom `u`
/// across candidate non-ring bond `(u, other)`.
pub fn classify_brics_atom_env(
    mol: &Molecule,
    u: usize,
    other: usize,
    bond: &Bond,
) -> Option<&'static str> {
    let a = &mol.atoms[u];
    if bond.order == BondOrder::Double {
        // L7: Acyclic C=C olefinic carbon
        if a.elem == Element::C
            && !a.in_ring
            && !a.aromatic
            && mol.atoms[other].elem == Element::C
            && !mol.atoms[other].in_ring
        {
            return Some("L7");
        }
        return None;
    }
    if bond.order != BondOrder::Single {
        return None;
    }

    match a.elem {
        Element::C => {
            if a.aromatic {
                // L14: Heteroaromatic ring carbon; L16: All-carbon aromatic ring carbon (phenyl/naphthyl)
                if ring_has_heteroatom(mol, u) {
                    Some("L14")
                } else {
                    Some("L16")
                }
            } else if a.in_ring {
                // L13: Non-aromatic heterocycle ring carbon; L15: Non-aromatic carbocycle ring carbon
                if ring_has_heteroatom(mol, u) {
                    Some("L13")
                } else {
                    Some("L15")
                }
            } else if has_carbonyl_double_bond_o(mol, u) {
                // L6 if carbonyl is directly attached to a ring on its other substituent, else L1
                let attached_to_ring = mol
                    .adj[u]
                    .iter()
                    .any(|&(v, _)| v != other && mol.atoms[v].in_ring);
                if attached_to_ring {
                    Some("L6")
                } else {
                    Some("L1")
                }
            } else {
                // Acyclic non-carbonyl carbon:
                // Exclude -CF3 carbon from being cleaved off as a bare CF3 radical
                let f_cnt = mol.adj[u]
                    .iter()
                    .filter(|&&(v, _)| mol.atoms[v].elem == Element::F)
                    .count();
                if f_cnt >= 2 {
                    return None;
                }
                // L4 if across a heteroatom (O, N, S), else L8 across a ring/carbonyl C-C bond
                if matches!(mol.atoms[other].elem, Element::O | Element::N | Element::S) {
                    Some("L4")
                } else {
                    Some("L8")
                }
            }
        }
        Element::N => {
            // Exclude nitro / azide / quaternary N+
            if a.charge != 0 {
                return None;
            }
            if a.aromatic {
                Some("L9")
            } else if a.in_ring {
                // L10 if lactam/cyclic carbamate N, else L5 (e.g. piperazine / morpholine / piperidine N!)
                let adj_carbonyl_in_ring = mol
                    .adj[u]
                    .iter()
                    .any(|&(v, _)| v != other && mol.atoms[v].in_ring && has_carbonyl_double_bond_o(mol, v));
                if adj_carbonyl_in_ring {
                    Some("L10")
                } else {
                    Some("L5")
                }
            } else {
                Some("L5")
            }
        }
        Element::O => {
            if !a.in_ring && a.charge == 0 && mol.heavy_degree(u) == 2 {
                Some("L3")
            } else {
                None
            }
        }
        Element::S => {
            if has_sulfonyl_two_double_o(mol, u) {
                Some("L12")
            } else if !a.in_ring && !a.aromatic && a.charge == 0 && mol.heavy_degree(u) == 2 {
                Some("L11")
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Maps a `(env_a, env_b)` pair across a single/double non-ring bond to its canonical
/// Degen et al. (2008) `BRICS` reaction name and retrosynthetic priority, or `None` if uncleaved.
pub fn match_brics_reaction(env_a: &str, env_b: &str) -> Option<(&'static str, u8)> {
    let check = |x: &str, y: &str| -> Option<(&'static str, u8)> {
        match (x, y) {
            // L1 / L6 Carbonyl couplings
            ("L1" | "L6", "L5" | "L10") => Some(("Amide Coupling (HATU / EDC)", 96)),
            ("L1" | "L6", "L3") => Some(("Esterification / Acylation", 94)),
            // L12 Sulfonyl couplings
            ("L12", "L5" | "L10") => Some(("Sulfonamide Formation (SO2Cl + Amine)", 95)),
            // L14 Heteroaryl C-N & Biaryl couplings (e.g. 4-anilinoquinazoline SNAr!)
            ("L14", "L5") => Some(("SNAr / Buchwald-Hartwig Heteroaryl Amination", 92)),
            ("L14", "L9" | "L10") => Some(("Ullmann / Chan-Lam Heteroaryl C-N Coupling", 88)),
            ("L14", "L14" | "L16") => Some(("Suzuki-Miyaura Heterobiaryl Coupling", 90)),
            // L16 Phenyl/Aryl couplings
            ("L16", "L16") => Some(("Suzuki-Miyaura Biaryl Coupling", 89)),
            ("L16", "L5" | "L9" | "L10") => Some(("Buchwald-Hartwig / Ullmann C-N Coupling", 86)),
            // L3 Ether disconnections
            ("L3", "L14" | "L16") => Some(("Ullmann / SNAr Aryl Ether Coupling", 80)),
            ("L3", "L4" | "L13" | "L15") => Some(("Williamson Ether Synthesis", 84)),
            // L5 / L9 / L10 Amine alkylations & reductive aminations
            ("L5" | "L9" | "L10", "L4" | "L13" | "L15") => {
                Some(("Reductive Amination / N-Alkylation", 85))
            }
            // L7 Olefin metathesis / Wittig
            ("L7", "L7") => Some(("Olefin Metathesis / Wittig / Heck", 82)),
            // L11 Thioether couplings
            ("L11", "L4" | "L13" | "L14" | "L15" | "L16") => {
                Some(("Thioether S-Alkylation / Cross-Coupling", 78))
            }
            // L8 Ring-to-alkyl / alkynyl C-C cross-couplings (Sonogashira / Negishi / Sp2-Sp3)
            ("L8", "L13" | "L14" | "L15" | "L16") => {
                Some(("Sonogashira / Negishi C-C Cross-Coupling", 75))
            }
            _ => None,
        }
    };

    check(env_a, env_b).or_else(|| check(env_b, env_a))
}

fn heavy_component_sizes_with_cut_set(
    mol: &Molecule,
    cut_bonds: &BTreeSet<usize>,
) -> Vec<Vec<usize>> {
    let n = mol.atoms.len();
    let mut visited = vec![false; n];
    let mut comps = Vec::new();
    let mut q = VecDeque::new();

    for start in 0..n {
        if visited[start] || mol.atoms[start].elem == Element::H {
            continue;
        }
        let mut comp = Vec::new();
        visited[start] = true;
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &(v, bi) in &mol.adj[u] {
                if cut_bonds.contains(&bi) || mol.atoms[v].elem == Element::H {
                    continue;
                }
                if !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

/// Discovers all candidate `BRICS` bonds and selects the maximal priority-ordered multi-bond
/// retrosynthetic cut set where every resulting fragment contains `>= 2` heavy atoms.
fn compute_brics_cuts_and_components(mol: &Molecule) -> (Vec<BricsBondCut>, Vec<Vec<usize>>) {
    let mut candidates: Vec<BricsBondCut> = Vec::new();

    for (bi, b) in mol.bonds.iter().enumerate() {
        if b.in_ring || b.aromatic {
            continue;
        }
        if mol.atoms[b.a].elem == Element::H || mol.atoms[b.b].elem == Element::H {
            continue;
        }
        let Some(ea) = classify_brics_atom_env(mol, b.a, b.b, b) else {
            continue;
        };
        let Some(eb) = classify_brics_atom_env(mol, b.b, b.a, b) else {
            continue;
        };
        let Some((rxn_name, prio)) = match_brics_reaction(ea, eb) else {
            continue;
        };

        // Single-bond sanity filter: ensure cleaving `bi` alone leaves `>= 2` heavy atoms on both sides
        // (keeps terminal -OCH3, -NH2, -OH, -Cl, -C#CH? Wait: -C#CH has 2 heavy carbons! So C#CH is >= 2,
        // while bare -CH3 or -OH has 1 heavy atom and stays intact on its parent ring!)
        let mut single_set = BTreeSet::new();
        single_set.insert(bi);
        let comps = heavy_component_sizes_with_cut_set(mol, &single_set);
        if comps.iter().any(|c| c.len() < 2) {
            continue;
        }

        candidates.push(BricsBondCut {
            bond_idx: bi,
            atom_a: b.a,
            atom_b: b.b,
            env_a: ea,
            env_b: eb,
            pair_code: format!("{ea}-{eb}"),
            reaction_name: rxn_name,
            priority: prio,
            selected_in_tree: false,
        });
    }

    // Sort candidate disconnections by priority descending so core medicinal-chemistry couplings
    // (amides, sulfonamides, SNAr quinazoline C4-NH, Suzuki biaryls) cleave before peripheral linkers
    candidates.sort_by(|x, y| {
        y.priority
            .cmp(&x.priority)
            .then_with(|| x.bond_idx.cmp(&y.bond_idx))
    });

    let mut active_cut_bonds: BTreeSet<usize> = BTreeSet::new();
    for cand in &mut candidates {
        let mut trial = active_cut_bonds.clone();
        trial.insert(cand.bond_idx);
        let comps = heavy_component_sizes_with_cut_set(mol, &trial);
        if comps.iter().all(|c| c.len() >= 2) {
            active_cut_bonds.insert(cand.bond_idx);
            cand.selected_in_tree = true;
        }
    }

    let final_comps = heavy_component_sizes_with_cut_set(mol, &active_cut_bonds);
    (candidates, final_comps)
}

fn describe_reagent_archetype(tags: &[String], has_ring: bool, heavy_cnt: usize) -> String {
    if tags.is_empty() {
        return "Intact Drug / Lead Structure".to_string();
    }
    let first = tags[0].as_str();
    match first {
        "[1*]" | "[6*]" => "Carboxylic Acid / Acyl Chloride Donor".to_string(),
        "[3*]" => {
            if has_ring {
                "Phenolic / Heteroaryl-OH Building Block".to_string()
            } else {
                "Alkoxide / Alcohol Linker".to_string()
            }
        }
        "[4*]" => "Alkyl Halide / Sulfonate Electrophile".to_string(),
        "[5*]" | "[9*]" | "[10*]" => {
            if has_ring {
                "Aniline / Cyclic Amine Nucleophile".to_string()
            } else {
                "Primary / Secondary Amine Reagent".to_string()
            }
        }
        "[7*]" => "Alkene / Wittig / Heck Partner".to_string(),
        "[8*]" => {
            if heavy_cnt == 2 {
                "Terminal Alkyne / Vinyl Reagent".to_string()
            } else {
                "Organometallic / Alkyl Partner".to_string()
            }
        }
        "[11*]" => "Thiol / Thioether Nucleophile".to_string(),
        "[12*]" => "Sulfonyl Chloride Electrophile".to_string(),
        "[14*]" => "Heteroaryl Halide / Boronate Core".to_string(),
        "[16*]" => "Aryl Boronic Acid / Halide Partner".to_string(),
        "[13*]" | "[15*]" => "Saturated Ring Building Block".to_string(),
        _ => "Commercial Synthon".to_string(),
    }
}

/// Builds a standalone valence-sanitized `Molecule` with freshly computed 2D skeletal coordinates
/// for a disconnected `BRICS` synthon atom subset `comp_atoms`.
fn build_synthon_molecule(
    parent: &Molecule,
    comp_atoms: &[usize],
    syn_idx: usize,
) -> Molecule {
    let mut old_to_new = vec![usize::MAX; parent.atoms.len()];
    let mut sub = Molecule::new("");
    sub.id = format!("{}-SYN{}", parent.id, syn_idx + 1);
    sub.name = format!("Synthon #{}", syn_idx + 1);
    sub.target = parent.target.clone();
    sub.source_url = "https://pubmed.ncbi.nlm.nih.gov/18792903/".to_string();

    for (new_i, &old_u) in comp_atoms.iter().enumerate() {
        old_to_new[old_u] = new_i;
        let src_a = &parent.atoms[old_u];
        sub.atoms.push(Atom {
            idx: new_i,
            elem: src_a.elem,
            charge: src_a.charge,
            aromatic: src_a.aromatic,
            bracketed: src_a.charge != 0 || src_a.chiral != 0,
            explicit_h: 0,
            implicit_h: 0,
            chiral: src_a.chiral,
            in_ring: false,
            ring_sizes: Vec::new(),
            hyb: src_a.hyb,
            pos2d: src_a.pos2d,
        });
    }

    for b in &parent.bonds {
        let na = old_to_new[b.a];
        let nb = old_to_new[b.b];
        if na != usize::MAX && nb != usize::MAX {
            let bi = sub.bonds.len();
            sub.bonds.push(Bond {
                idx: bi,
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: false,
                kekule_order: b.kekule_order,
            });
        }
    }

    sub.sanitize();
    compute_2d_coords(&mut sub);
    sub.smiles = molecule_to_smiles(&sub);
    sub
}

/// Evaluates the fast Ertl & Schuffenhauer (2009) `SAScore` and `BRICS` summary counts for `mol`
/// (parallelized across entire HTS libraries via `Rayon` inside `DatasetAnalysis::from_molecules`).
pub fn evaluate_synthetic_accessibility(mol: &Molecule) -> SynthAccessProfile {
    let (
        sa_score,
        fragment_score,
        size_penalty,
        stereo_penalty,
        spiro_penalty,
        bridgehead_penalty,
        macrocycle_penalty,
        symmetry_bonus,
        heavy_atoms,
        unique_env_count,
        chiral_centers,
        spiro_atoms,
        bridgehead_atoms,
        macrocycle_rings,
        complexity_atoms,
    ) = compute_sa_metrics_only(mol);

    let (candidates, comps) = compute_brics_cuts_and_components(mol);
    let active_cnt = candidates.iter().filter(|c| c.selected_in_tree).count();
    let convergent_steps_est = if active_cnt == 0 {
        if heavy_atoms <= 12 { 1 } else { 2 }
    } else if symmetry_bonus >= 0.12 && active_cnt >= 2 {
        (active_cnt + 1) / 2
    } else {
        active_cnt
    };

    SynthAccessProfile {
        sa_score,
        tractability: SynthTractability::from_sa_score(sa_score),
        fragment_score,
        size_penalty,
        stereo_penalty,
        spiro_penalty,
        bridgehead_penalty,
        macrocycle_penalty,
        symmetry_bonus,
        heavy_atoms,
        unique_env_count,
        chiral_centers,
        spiro_atoms,
        bridgehead_atoms,
        macrocycle_rings,
        complexity_atoms,
        brics_cleavable_bonds: candidates.len(),
        brics_synthon_count: comps.len().max(1),
        convergent_steps_est,
    }
}

/// Performs full `BRICS` (`Degen et al. 2008`) retrosynthetic disconnection on `mol`, returning
/// all candidate & active cleavage bonds plus 2D-depicted `BricsSynthon` building blocks.
pub fn dissect_brics(mol: &Molecule) -> BricsRetrosynthesis {
    let parent_sa = evaluate_synthetic_accessibility(mol);
    let (candidate_cuts, mut comps) = compute_brics_cuts_and_components(mol);
    let active_cuts: Vec<BricsBondCut> = candidate_cuts
        .iter()
        .filter(|c| c.selected_in_tree)
        .cloned()
        .collect();

    // Sort components largest-first (Core Scaffold Bricks first, then Ring Synthons, then Side-Chain Linkers)
    comps.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));

    let mut synthons = Vec::with_capacity(comps.len());
    for (si, comp_atoms) in comps.into_iter().enumerate() {
        let atom_set: BTreeSet<usize> = comp_atoms.iter().copied().collect();
        let mut tags = Vec::new();
        for cut in &active_cuts {
            if atom_set.contains(&cut.atom_a) && !atom_set.contains(&cut.atom_b) {
                let num = cut.env_a.trim_start_matches('L');
                tags.push(format!("[{num}*]"));
            } else if atom_set.contains(&cut.atom_b) && !atom_set.contains(&cut.atom_a) {
                let num = cut.env_b.trim_start_matches('L');
                tags.push(format!("[{num}*]"));
            }
        }

        let mol_2d = build_synthon_molecule(mol, &comp_atoms, si);
        let (syn_sa, ..) = compute_sa_metrics_only(&mol_2d);
        let has_ring = !mol_2d.rings.is_empty();
        let role = if has_ring && comp_atoms.len() >= 8 && tags.len() >= 2 {
            "Core Scaffold Brick"
        } else if has_ring {
            "Ring Synthon"
        } else {
            "Side-Chain / Linker Reagent"
        };
        let reagent_archetype = describe_reagent_archetype(&tags, has_ring, comp_atoms.len());
        let formula = mol_2d.molecular_formula();
        let mw = mol_2d.molecular_weight();
        let reagent_smiles = mol_2d.smiles.clone();

        synthons.push(BricsSynthon {
            idx: si,
            role,
            reagent_archetype,
            attachment_tags: tags,
            reagent_smiles,
            formula,
            mw,
            sa_score: syn_sa,
            parent_atoms: comp_atoms,
            mol_2d,
        });
    }

    let (mean_synthon_sa, max_synthon_sa) = if synthons.is_empty() {
        (parent_sa.sa_score, parent_sa.sa_score)
    } else {
        let sum: f64 = synthons.iter().map(|s| s.sa_score).sum();
        let mx = synthons
            .iter()
            .map(|s| s.sa_score)
            .fold(1.0_f64, f64::max);
        (sum / (synthons.len() as f64), mx)
    };
    let convergence_sa_drop = (parent_sa.sa_score - max_synthon_sa).max(0.0);

    BricsRetrosynthesis {
        parent_sa,
        candidate_cuts,
        active_cuts,
        synthons,
        mean_synthon_sa,
        max_synthon_sa,
        convergence_sa_drop,
    }
}
