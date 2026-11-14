//! Evolutionary De Novo Molecule Generator (`de_novo_gen.rs`, `v2.2.0`):
//! Graph-Based Point/Group Mutations (`GB-GA` / `CReM`), Core & Peripheral Ring-Hop Bioisosteres
//! (including `sp3` `Escape-from-Flatland` bridged/spiro cages), and Synthesis-Aware `BRICS`
//! Synthon Crossovers (`AutoGrow4`), Guided by 3D `ROCS` / `USRCAT` Shape-Electrostatic Overlay,
//! 2D `Gobbi`/`CATS2` Pharmacophore Similarity, Ligand & Lipophilic Efficiency (`LE` / `LLE`),
//! and Ertl `SAScore` Synthetic Accessibility.
//!
//! Primary References:
//! - Jensen (2019), *Chem. Sci.* 10(12), 3567-3572 ("A graph-based genetic algorithm and generative
//!   model/Monte Carlo tree search for the exploration of chemical space":
//!   `https://pubmed.ncbi.nlm.nih.gov/30996948/`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC6438151/`)
//! - Brown, Fiscato, Segler & Vaucher (2019), *J. Chem. Inf. Model.* 59(3), 1096-1108 ("GuacaMol:
//!   Benchmarking Models for de Novo Molecular Design": `https://pubmed.ncbi.nlm.nih.gov/30887799/`)
//! - Spiegel & Durrant (2020), *J. Cheminform.* 12:25 ("AutoGrow4: an open-source genetic algorithm
//!   for de novo drug design and lead optimization": `https://pubmed.ncbi.nlm.nih.gov/33431021/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC7165399/`)
//! - Polishchuk (2020), *J. Cheminform.* 12:28 ("CReM: chemically reasonable mutations framework
//!   for structure generation": `https://pubmed.ncbi.nlm.nih.gov/33430959/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC7178718/`)
//! - Böhm, Flohr & Stahl (2004), *Drug Discov. Today Technol.* 1(3), 217-224 ("Scaffold hopping":
//!   `https://pubmed.ncbi.nlm.nih.gov/24981488/`) & Lovering, Bikker & Humblet (2009), *J. Med. Chem.*
//!   52(21), 6752-6756 ("Escape from flatland": `https://pubmed.ncbi.nlm.nih.gov/19827778/`)
//! - Hawkins, Skillman & Nicholls (2007) `ROCS` 3D Shape + Color (`https://pubmed.ncbi.nlm.nih.gov/17201411/`),
//!   Schreyer & Blundell (2012) `USRCAT` (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3505738/`),
//!   Hopkins et al. (2004) `LE` (`https://pubmed.ncbi.nlm.nih.gov/15109945/`), Leeson & Springthorpe
//!   (2007) `LLE` (`https://pubmed.ncbi.nlm.nih.gov/17971784/`), and Ertl & Schuffenhauer (2009)
//!   `SAScore` (`https://pmc.ncbi.nlm.nih.gov/articles/PMC3225829/`).

use crate::admet::{evaluate_admet, BoiledEggRegion};
use crate::alerts::{scan_structural_alerts, AlertSeverity};
use crate::conformer3d::{
    align_rocs_3d, evaluate_3d_summary_profile, usrcat_similarity, RocsColorAtom,
    Summary3dProfile,
};
use crate::fingerprint::{compute_ecfp, Ecfp2048};
use crate::fragment_grower::RT_LN10_KCAL;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::pharmacophore::{align_pharmacophores, extract_pharmacophore_map, PharmacophoreMap};
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::rgroup_table::predict_best_free_wilson;
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::{
    classify_brics_atom_env, evaluate_synthetic_accessibility, match_brics_reaction,
};
use crate::DatasetAnalysis;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Genetic operator class that produced an evolved candidate structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvolutionOperatorKind {
    SeedTemplate,
    GraphMutation,
    RingHop,
    BricsCrossover,
}

impl EvolutionOperatorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SeedTemplate => "Seed Parent",
            Self::GraphMutation => "Graph Mutation (GB-GA)",
            Self::RingHop => "Ring-Hop Bioisostere",
            Self::BricsCrossover => "BRICS Synthon Crossover",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::SeedTemplate => "Seed",
            Self::GraphMutation => "Graph-Mut",
            Self::RingHop => "Ring-Hop",
            Self::BricsCrossover => "BRICS-Xover",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::SeedTemplate => [148, 163, 184],   // Slate
            Self::GraphMutation => [56, 189, 248],   // Sky Blue
            Self::RingHop => [52, 211, 153],         // Emerald
            Self::BricsCrossover => [192, 132, 252], // Violet
        }
    }
}

/// Multi-objective fitness preset steering evolutionary exploration of chemical space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeNovoGoalPreset {
    /// Maximizes 3D `ROCS`/`USRCAT` + 2D `Gobbi`/`CATS2` pharmacophore alignment while rewarding novel Murcko cores (`ECFP4 <= 0.48`).
    ScaffoldHop3dRocs,
    /// Maximizes predicted `pAct`, Hopkins `LE`, and Leeson-Springthorpe `LLE` under `SAScore` & `MW` guardrails.
    PotencyAndEfficiency,
    /// Balanced Pareto discovery across 3D/2D target similarity, `LE`/`LLE`, `QED`, `Fsp3`, and synthetic accessibility.
    BalancedLeadDiscovery,
    /// Optimizes Wager `CNS-MPO`, `Fsp3` carbon saturation (`Escape from Flatland`), 3D shape match, and low `SAScore`.
    CnsEscapeFlatland,
}

impl DeNovoGoalPreset {
    pub const ALL: [Self; 4] = [
        Self::ScaffoldHop3dRocs,
        Self::PotencyAndEfficiency,
        Self::BalancedLeadDiscovery,
        Self::CnsEscapeFlatland,
    ];

    pub fn from_code(raw: &str) -> Self {
        let s = raw.trim().to_ascii_lowercase();
        if s.contains("pot") || s.contains("lle") || s.contains("eff") {
            Self::PotencyAndEfficiency
        } else if s.contains("cns") || s.contains("flat") || s.contains("sp3") {
            Self::CnsEscapeFlatland
        } else if s.contains("bal") {
            Self::BalancedLeadDiscovery
        } else {
            Self::ScaffoldHop3dRocs
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::ScaffoldHop3dRocs => "scaffold-hop",
            Self::PotencyAndEfficiency => "le-lle",
            Self::BalancedLeadDiscovery => "balanced",
            Self::CnsEscapeFlatland => "cns-sp3",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ScaffoldHop3dRocs => "3D ROCS & Pharmacophore Scaffold Hop",
            Self::PotencyAndEfficiency => "LE / LLE & Affinity Maturation",
            Self::BalancedLeadDiscovery => "Balanced Multi-Objective Discovery",
            Self::CnsEscapeFlatland => "CNS-MPO & Escape-from-Flatland (sp3)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ScaffoldHop3dRocs => "3D Scaffold Hop",
            Self::PotencyAndEfficiency => "LE / LLE Boost",
            Self::BalancedLeadDiscovery => "Balanced GA",
            Self::CnsEscapeFlatland => "CNS & sp3 Hop",
        }
    }

    /// Default `(w_3d_pharm, w_le_lle, w_sa, w_novelty)` weights for this discovery objective.
    pub fn default_weights(self) -> (f64, f64, f64, f64) {
        match self {
            Self::ScaffoldHop3dRocs => (0.38, 0.24, 0.16, 0.22),
            Self::PotencyAndEfficiency => (0.26, 0.44, 0.18, 0.12),
            Self::BalancedLeadDiscovery => (0.32, 0.30, 0.20, 0.18),
            Self::CnsEscapeFlatland => (0.28, 0.32, 0.18, 0.22),
        }
    }
}

/// Interactive evolutionary parameters for `evolve_de_novo_molecules`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeNovoConfig {
    pub goal: DeNovoGoalPreset,
    pub generations: usize,
    pub population_size: usize,
    pub max_returned: usize,
    pub max_mw: f64,
    pub max_sa_score: f64,
    pub enforce_pains_free: bool,
    pub enable_graph_mutations: bool,
    pub enable_ring_hops: bool,
    pub enable_brics_crossover: bool,
    pub random_seed: u64,
}

impl Default for DeNovoConfig {
    fn default() -> Self {
        Self {
            goal: DeNovoGoalPreset::ScaffoldHop3dRocs,
            generations: 6,
            population_size: 28,
            max_returned: 24,
            max_mw: 520.0,
            max_sa_score: 4.8,
            enforce_pains_free: true,
            enable_graph_mutations: true,
            enable_ring_hops: true,
            enable_brics_crossover: true,
            random_seed: 0xDE_40_70_22,
        }
    }
}

/// Telemetry point recorded at the close of each evolutionary generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationSnapshot {
    pub generation: usize,
    pub evaluated_count: usize,
    pub best_fitness: f64,
    pub mean_top10_fitness: f64,
    pub best_3d_pharm_sim: f64,
    pub best_lle: f64,
    pub mean_sa_score: f64,
    pub unique_scaffolds: usize,
    pub scaffold_hop_pct: f64,
}

/// Single evolved de novo candidate molecule with 3D `ROCS` superposition, 2D pharmacophore match,
/// `LE`/`LLE`/`SAScore` metrics, and full operator lineage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolvedCandidate {
    pub rank: usize,
    pub candidate_id: String,
    pub generation_born: usize,
    pub last_operator: EvolutionOperatorKind,
    pub operator_detail: String,
    pub lineage_trace: String,
    pub canonical_smiles: String,
    pub scaffold_key: String,
    pub scaffold_label: String,
    pub is_novel_scaffold: bool,
    pub is_true_scaffold_hop: bool,
    pub pareto_optimal: bool,

    // 3D ROCS / USRCAT & 2D Gobbi/CATS2 Similarity to Query Seed
    pub ecfp4_tanimoto: f64,
    pub pharm2d_similarity: f64,
    pub usrcat_3d_similarity: f64,
    pub rocs_shape_tanimoto: f64,
    pub rocs_color_tanimoto: f64,
    pub rocs_esp_similarity: f64,
    /// Hawkins `ROCS` normalized `ComboTanimoto = 0.5 * ShapeTanimoto + 0.5 * ColorTanimoto` (`[0..1]`).
    pub rocs_combo_tanimoto: f64,

    // Discovery Efficiency, Affinity & Synthesizability
    pub predicted_pact: f64,
    pub predicted_ic50_nm: f64,
    pub le: f64,
    pub lle: f64,
    pub lle_74: f64,
    pub sile: f64,
    pub sa_score: f64,
    pub physchem: PhyschemProfile,
    pub logd_74: f64,
    pub esol_ug_ml: f64,
    pub boiled_egg: BoiledEggRegion,
    pub shape3d_class: &'static str,

    /// Multi-objective evolutionary desirability fitness (`[0..100]`).
    pub fitness_score: f64,

    /// Rigid-body Kabsch/`ROCS`-aligned 3D coordinates in the reference query frame (`Angstrom`).
    pub aligned_coords3d: Vec<[f32; 3]>,
    pub gasteiger_charges: Vec<f32>,
    pub aligned_colors: Vec<RocsColorAtom>,
    pub mol: Molecule,
}

/// Complete evolutionary de novo design run report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeNovoEvolutionReport {
    pub query_id: String,
    pub query_name: String,
    pub query_smiles: String,
    pub query_pact: f64,
    pub query_scaffold_label: String,
    pub query_mol: Molecule,
    pub query_coords3d: Vec<[f32; 3]>,
    pub query_charges: Vec<f32>,
    pub query_colors: Vec<RocsColorAtom>,
    pub config: DeNovoConfig,
    pub total_unique_evaluated: usize,
    pub novel_scaffold_count: usize,
    pub true_scaffold_hop_count: usize,
    pub operator_counts: [usize; 3], // [GraphMutation, RingHop, BricsCrossover] among top returned
    pub trajectory: Vec<GenerationSnapshot>,
    pub candidates: Vec<EvolvedCandidate>,
}

struct RingHopTemplate {
    name: &'static str,
    smiles: &'static str,
    exits: usize,
    /// Atom indices inside `parse_smiles(smiles)` bearing the replaceable C-H / N-H attachment sites.
    attach_atoms: [usize; 2],
}

/// Curated 2-exit (core scaffold hop) & 1-exit (peripheral ring hop) medicinal-chemistry ring library,
/// spanning heteroaromatic azines/azoles AND `Escape-from-Flatland` (`sp3`) cages & saturated rings.
const RING_HOP_TEMPLATES: &[RingHopTemplate] = &[
    // 2-Exit Core Scaffold Hops (para / 1,4 / 1,3 vector-preserving replacements)
    RingHopTemplate {
        name: "Pyrimidine-2,5-diyl Core",
        smiles: "c1cncnc1",
        exits: 2,
        attach_atoms: [1, 4],
    },
    RingHopTemplate {
        name: "Pyrazine-2,5-diyl Core",
        smiles: "c1cnccn1",
        exits: 2,
        attach_atoms: [1, 4],
    },
    RingHopTemplate {
        name: "Pyridazine-3,6-diyl Core",
        smiles: "c1ccnnc1",
        exits: 2,
        attach_atoms: [0, 5],
    },
    RingHopTemplate {
        name: "Pyridine-2,5-diyl Core",
        smiles: "c1ccncc1",
        exits: 2,
        attach_atoms: [1, 4],
    },
    RingHopTemplate {
        name: "1,3-Thiazole-2,5-diyl Core",
        smiles: "c1cscn1",
        exits: 2,
        attach_atoms: [0, 3],
    },
    RingHopTemplate {
        name: "1,3,4-Oxadiazole-2,5-diyl Core",
        smiles: "c1nnco1",
        exits: 2,
        attach_atoms: [0, 3],
    },
    RingHopTemplate {
        name: "1H-Pyrazole-1,4-diyl Core",
        smiles: "c1cn[nH]c1",
        exits: 2,
        attach_atoms: [1, 3],
    },
    RingHopTemplate {
        name: "Bicyclo[1.1.1]pentane-1,3-diyl (BCP sp3)",
        smiles: "C12CC(C1)C2",
        exits: 2,
        attach_atoms: [0, 2],
    },
    RingHopTemplate {
        name: "Spiro[3.3]heptane-2,6-diyl (sp3)",
        smiles: "C1CC2(C1)CCC2",
        exits: 2,
        attach_atoms: [0, 5],
    },
    RingHopTemplate {
        name: "Piperidine-1,4-diyl (sp3)",
        smiles: "C1CCNCC1",
        exits: 2,
        attach_atoms: [0, 3],
    },
    RingHopTemplate {
        name: "Piperazine-1,4-diyl (sp3)",
        smiles: "C1CNCCN1",
        exits: 2,
        attach_atoms: [2, 5],
    },
    // 1-Exit Peripheral Ring Hops
    RingHopTemplate {
        name: "Pyridin-3-yl Ring",
        smiles: "c1ccncc1",
        exits: 1,
        attach_atoms: [1, 0],
    },
    RingHopTemplate {
        name: "Pyrimidin-5-yl Ring",
        smiles: "c1cncnc1",
        exits: 1,
        attach_atoms: [4, 0],
    },
    RingHopTemplate {
        name: "1-Methyl-1H-pyrazol-4-yl Ring",
        smiles: "Cn1ccnc1",
        exits: 1,
        attach_atoms: [3, 0],
    },
    RingHopTemplate {
        name: "Oxetan-3-yl (sp3 Polar Isostere)",
        smiles: "C1COC1",
        exits: 1,
        attach_atoms: [0, 0],
    },
    RingHopTemplate {
        name: "Morpholin-4-yl (sp3 Solubilizer)",
        smiles: "C1COCCN1",
        exits: 1,
        attach_atoms: [5, 0],
    },
    RingHopTemplate {
        name: "Tetrahydropyran-4-yl (sp3)",
        smiles: "C1CCOCC1",
        exits: 1,
        attach_atoms: [0, 0],
    },
    RingHopTemplate {
        name: "Cyclopropyl (sp3 Metabolic Shield)",
        smiles: "C1CC1",
        exits: 1,
        attach_atoms: [0, 0],
    },
];

/// Curated compact medicinal-chemistry donor synthons guaranteeing rich `BRICS` crossovers even in standalone 1-molecule runs.
const CURATED_CROSSOVER_DONORS: &[(&str, &str)] = &[
    ("4-Fluoroaniline", "Fc1ccc(N)cc1"),
    ("2-Methoxy-5-methylpyridine", "COc1ccc(C)cn1"),
    ("1-(Methylsulfonyl)piperazine", "CS(=O)(=O)N1CCNCC1"),
    ("4-(Morpholinomethyl)benzoic acid", "O=C(O)c1ccc(CN2CCOCC2)cc1"),
    ("5-Fluoro-1H-indazole", "Fc1ccc2[nH]ncc2c1"),
    ("3-(Trifluoromethyl)-1H-pyrazole", "FC(F)(F)c1cc[nH]n1"),
    ("Cyclopropanecarboxamide", "NC(=O)C1CC1"),
    ("6-Methoxyquinazoline-4-amine", "COc1ccc2ncnc(N)c2c1"),
];

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn rng_usize(state: &mut u64, upper: usize) -> usize {
    if upper <= 1 {
        0
    } else {
        (splitmix64(state) as usize) % upper
    }
}

/// Round-trips a raw `Molecule` graph through canonical `OpenSMILES` + `parse_smiles` to guarantee
/// 100% Kekulé/valence validity, single-component connectivity, and reasonable heavy-atom bounds.
fn sanitize_candidate_graph(mut raw: Molecule, max_mw: f64) -> Option<(String, Molecule)> {
    if raw.atoms.len() < 6 || raw.atoms.len() > 48 {
        return None;
    }
    raw.sanitize();
    // Ensure single connected component
    let mut visited = vec![false; raw.atoms.len()];
    let mut q = VecDeque::new();
    visited[0] = true;
    q.push_back(0usize);
    let mut seen = 0usize;
    while let Some(u) = q.pop_front() {
        seen += 1;
        for &(v, _) in &raw.adj[u] {
            if !visited[v] {
                visited[v] = true;
                q.push_back(v);
            }
        }
    }
    if seen != raw.atoms.len() {
        return None;
    }

    let can_smi = molecule_to_smiles(&raw);
    if can_smi.is_empty() || can_smi.contains('.') {
        return None;
    }
    let mut parsed = parse_smiles(&can_smi).ok()?;
    if parsed.atoms.len() < 6 || parsed.molecular_weight() > max_mw + 40.0 {
        return None;
    }
    // Check neutral/valid valence sanity (avoid hypervalent neutral carbons or weird radicals)
    for (u, a) in parsed.atoms.iter().enumerate() {
        let bo_sum: u8 = parsed.adj[u]
            .iter()
            .map(|&(_, bi)| parsed.bonds[bi].kekule_order)
            .sum();
        if a.elem == Element::C && a.charge == 0 && bo_sum > 4 {
            return None;
        }
        if matches!(a.elem, Element::N | Element::O) && a.charge == 0 && bo_sum > 3 {
            return None;
        }
    }
    parsed.smiles = molecule_to_smiles(&parsed);
    compute_2d_coords(&mut parsed);
    Some((parsed.smiles.clone(), parsed))
}

fn new_atom(idx: usize, elem: Element, aromatic: bool) -> Atom {
    Atom {
        idx,
        elem,
        charge: 0,
        aromatic,
        bracketed: false,
        explicit_h: 0,
        implicit_h: 0,
        chiral: 0,
        in_ring: aromatic,
        ring_sizes: Vec::new(),
        hyb: if aromatic {
            Hybridization::Sp2
        } else {
            Hybridization::Sp3
        },
        pos2d: [0.0, 0.0],
    }
}

/// Operator 1: Graph Point & Functional-Group Mutations (`GB-GA` / `CReM`).
fn generate_graph_mutations(
    parent: &Molecule,
    rng: &mut u64,
    max_mw: f64,
    limit: usize,
) -> Vec<(String, Molecule, String)> {
    let mut out = Vec::new();
    let n = parent.atoms.len();
    if n < 5 {
        return out;
    }

    // Collect candidate mutable sites
    let mut arom_ch = Vec::new();
    let mut arom_n = Vec::new();
    let mut open_ch = Vec::new();
    let mut terminal_halides_or_me = Vec::new();
    let mut chain_ch2 = Vec::new();

    for (u, a) in parent.atoms.iter().enumerate() {
        let deg = parent.adj[u].len();
        if a.aromatic && a.elem == Element::C && deg == 2 && a.total_h() >= 1 && a.charge == 0 {
            arom_ch.push(u);
        }
        if a.aromatic && a.elem == Element::N && deg == 2 && a.total_h() == 0 && a.charge == 0 {
            // Avoid mutating pyrimidine/quinazoline if adjacent to another ring N? Still safe to try; sanitize filters invalid Kekule
            arom_n.push(u);
        }
        if (a.elem == Element::C || a.elem == Element::N) && a.total_h() >= 1 && a.charge == 0 {
            open_ch.push(u);
        }
        if deg == 1
            && matches!(
                a.elem,
                Element::F | Element::Cl | Element::Br | Element::C | Element::O
            )
            && !a.in_ring
        {
            terminal_halides_or_me.push(u);
        }
        if !a.in_ring
            && a.elem == Element::C
            && deg == 2
            && a.hyb == Hybridization::Sp3
            && parent.adj[u]
                .iter()
                .all(|&(_, bi)| parent.bonds[bi].kekule_order == 1)
        {
            chain_ch2.push(u);
        }
    }

    let mut attempts = 0usize;
    while out.len() < limit && attempts < limit * 5 + 12 {
        attempts += 1;
        let op_pick = rng_usize(rng, 5);
        match op_pick {
            // 0: Aromatic C-H -> N aza-scan (LLE booster)
            0 if !arom_ch.is_empty() => {
                let u = arom_ch[rng_usize(rng, arom_ch.len())];
                let mut m = parent.clone();
                m.atoms[u].elem = Element::N;
                m.atoms[u].explicit_h = 0;
                m.atoms[u].implicit_h = 0;
                if let Some((smi, clean)) = sanitize_candidate_graph(m, max_mw) {
                    if smi != parent.smiles {
                        out.push((smi, clean, format!("Aza-scan [cH]->[n] @C{}", u + 1)));
                    }
                }
            }
            // 1: Aromatic N -> C-H de-aza scan
            1 if !arom_n.is_empty() => {
                let u = arom_n[rng_usize(rng, arom_n.len())];
                let mut m = parent.clone();
                m.atoms[u].elem = Element::C;
                if let Some((smi, clean)) = sanitize_candidate_graph(m, max_mw) {
                    if smi != parent.smiles {
                        out.push((smi, clean, format!("CH-restore [n]->[cH] @N{}", u + 1)));
                    }
                }
            }
            // 2: Sprout small polar/lipophilic/sp3 group at open C-H / N-H
            2 if !open_ch.is_empty() => {
                let u = open_ch[rng_usize(rng, open_ch.len())];
                let group_choice = rng_usize(rng, 8);
                let mut m = parent.clone();
                let base = m.atoms.len();
                let desc = match group_choice {
                    0 => {
                        if m.atoms[u].elem == Element::N {
                            continue;
                        }
                        m.atoms.push(new_atom(base, Element::F, false));
                        m.bonds.push(Bond {
                            idx: m.bonds.len(),
                            a: u,
                            b: base,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        format!("Fluoro sprout (-F) @atom {}", u + 1)
                    }
                    1 => {
                        m.atoms.push(new_atom(base, Element::C, false));
                        m.bonds.push(Bond {
                            idx: m.bonds.len(),
                            a: u,
                            b: base,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        format!("Magic-methyl sprout (-Me) @atom {}", u + 1)
                    }
                    2 => {
                        if m.atoms[u].elem == Element::N {
                            continue;
                        }
                        m.atoms.push(new_atom(base, Element::C, false));
                        m.atoms.push(new_atom(base + 1, Element::N, false));
                        let b0 = m.bonds.len();
                        m.bonds.push(Bond {
                            idx: b0,
                            a: u,
                            b: base,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        m.bonds.push(Bond {
                            idx: b0 + 1,
                            a: base,
                            b: base + 1,
                            order: BondOrder::Triple,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 3,
                        });
                        format!("Nitrile sprout (-CN) @atom {}", u + 1)
                    }
                    3 => {
                        if m.atoms[u].elem == Element::N {
                            continue;
                        }
                        m.atoms.push(new_atom(base, Element::O, false));
                        m.atoms.push(new_atom(base + 1, Element::C, false));
                        let b0 = m.bonds.len();
                        m.bonds.push(Bond {
                            idx: b0,
                            a: u,
                            b: base,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        m.bonds.push(Bond {
                            idx: b0 + 1,
                            a: base,
                            b: base + 1,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        format!("Methoxy sprout (-OMe) @atom {}", u + 1)
                    }
                    4 => {
                        if m.atoms[u].elem == Element::N {
                            continue;
                        }
                        m.atoms.push(new_atom(base, Element::O, false));
                        m.bonds.push(Bond {
                            idx: m.bonds.len(),
                            a: u,
                            b: base,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        format!("Hydroxyl HBD/HBA (-OH) @atom {}", u + 1)
                    }
                    5 => {
                        // Cyclopropyl sp3 ring sprout
                        m.atoms.push(new_atom(base, Element::C, false));
                        m.atoms.push(new_atom(base + 1, Element::C, false));
                        m.atoms.push(new_atom(base + 2, Element::C, false));
                        let b0 = m.bonds.len();
                        for (ia, ib) in [(u, base), (base, base + 1), (base + 1, base + 2), (base + 2, base)] {
                            let idx = m.bonds.len();
                            m.bonds.push(Bond {
                                idx,
                                a: ia,
                                b: ib,
                                order: BondOrder::Single,
                                aromatic: false,
                                in_ring: ia != u,
                                kekule_order: 1,
                            });
                        }
                        let _ = b0;
                        format!("Cyclopropyl sp3 sprout @atom {}", u + 1)
                    }
                    6 => {
                        if m.atoms[u].elem == Element::N {
                            continue;
                        }
                        m.atoms.push(new_atom(base, Element::N, false));
                        m.bonds.push(Bond {
                            idx: m.bonds.len(),
                            a: u,
                            b: base,
                            order: BondOrder::Single,
                            aromatic: false,
                            in_ring: false,
                            kekule_order: 1,
                        });
                        format!("Amino HBD sprout (-NH2) @atom {}", u + 1)
                    }
                    _ => {
                        if m.atoms[u].elem == Element::N {
                            continue;
                        }
                        // -CHF2 difluoromethyl lipophilic HBD bioisostere
                        m.atoms.push(new_atom(base, Element::C, false));
                        m.atoms.push(new_atom(base + 1, Element::F, false));
                        m.atoms.push(new_atom(base + 2, Element::F, false));
                        for (ia, ib) in [(u, base), (base, base + 1), (base, base + 2)] {
                            let idx = m.bonds.len();
                            m.bonds.push(Bond {
                                idx,
                                a: ia,
                                b: ib,
                                order: BondOrder::Single,
                                aromatic: false,
                                in_ring: false,
                                kekule_order: 1,
                            });
                        }
                        format!("Difluoromethyl (-CHF2) @atom {}", u + 1)
                    }
                };
                if let Some((smi, clean)) = sanitize_candidate_graph(m, max_mw) {
                    if smi != parent.smiles {
                        out.push((smi, clean, desc));
                    }
                }
            }
            // 3: Terminal substituent swap or prune (keeps MW compact and boosts LE!)
            3 if !terminal_halides_or_me.is_empty() => {
                let u = terminal_halides_or_me[rng_usize(rng, terminal_halides_or_me.len())];
                let sub_mode = rng_usize(rng, 4);
                let mut m = parent.clone();
                let desc = if sub_mode == 0 && m.atoms.len() > 10 {
                    // Prune terminal heavy atom to boost LE
                    m.atoms.remove(u);
                    m.bonds.retain(|b| b.a != u && b.b != u);
                    for b in &mut m.bonds {
                        if b.a > u {
                            b.a -= 1;
                        }
                        if b.b > u {
                            b.b -= 1;
                        }
                    }
                    format!("Prune terminal group @atom {} (+LE)", u + 1)
                } else {
                    let new_el = match sub_mode {
                        1 => Element::F,
                        2 => Element::Cl,
                        _ => Element::C,
                    };
                    m.atoms[u].elem = new_el;
                    m.atoms[u].charge = 0;
                    m.atoms[u].explicit_h = 0;
                    format!("Terminal swap -> -{} @atom {}", new_el.symbol(), u + 1)
                };
                if let Some((smi, clean)) = sanitize_candidate_graph(m, max_mw) {
                    if smi != parent.smiles {
                        out.push((smi, clean, desc));
                    }
                }
            }
            // 4: Acyclic CH2 -> O / NH heteroatom linker walk
            _ if !chain_ch2.is_empty() => {
                let u = chain_ch2[rng_usize(rng, chain_ch2.len())];
                // Avoid placing heteroatom adjacent to another heteroatom (hemiaminal/peroxide)
                let nbr_hetero = parent.adj[u]
                    .iter()
                    .any(|&(v, _)| parent.atoms[v].is_hetero());
                if nbr_hetero {
                    continue;
                }
                let mut m = parent.clone();
                let to_n = rng_usize(rng, 2) == 0;
                m.atoms[u].elem = if to_n { Element::N } else { Element::O };
                let lbl = if to_n {
                    format!("Linker walk -CH2- -> -NH- @C{}", u + 1)
                } else {
                    format!("Linker walk -CH2- -> -O- @C{}", u + 1)
                };
                if let Some((smi, clean)) = sanitize_candidate_graph(m, max_mw) {
                    if smi != parent.smiles {
                        out.push((smi, clean, lbl));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Operator 2: Core & Peripheral Ring-Hop Bioisosteres (`1-exit` and `2-exit` ring splicing,
/// including `BCP`, `Spiro[3.3]heptane`, `Piperidine`, `Oxadiazole`, `Thiazole`, and `Azines`).
fn generate_ring_hops(
    parent: &Molecule,
    rng: &mut u64,
    max_mw: f64,
    limit: usize,
) -> Vec<(String, Molecule, String)> {
    let mut out = Vec::new();
    if parent.rings.is_empty() {
        return out;
    }

    // Identify isolated (non-fused) rings with 1 or 2 single-bond exocyclic attachments
    let mut ring_candidates: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for (ri, ring) in parent.rings.iter().enumerate() {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        // Skip rings sharing atoms with another SSSR ring (preserve bicyclic hinge cores or hop non-fused rings cleanly)
        let is_fused = parent.rings.iter().enumerate().any(|(rj, other)| {
            ri != rj && ring.iter().any(|a| other.contains(a))
        });
        if is_fused {
            continue;
        }

        let ring_set: BTreeSet<usize> = ring.iter().copied().collect();
        let mut exo_anchors = Vec::new();
        let mut clean_exits = true;
        for &ru in ring {
            for &(nb, bi) in &parent.adj[ru] {
                if !ring_set.contains(&nb) {
                    if parent.bonds[bi].kekule_order != 1 {
                        clean_exits = false;
                        break;
                    }
                    exo_anchors.push(nb);
                }
            }
            if !clean_exits {
                break;
            }
        }
        if clean_exits && (exo_anchors.len() == 1 || exo_anchors.len() == 2) {
            ring_candidates.push((ring.clone(), exo_anchors));
        }
    }

    if ring_candidates.is_empty() {
        return out;
    }

    let mut attempts = 0usize;
    while out.len() < limit && attempts < limit * 6 + 10 {
        attempts += 1;
        let (ref old_ring, ref exo_nbrs) =
            ring_candidates[rng_usize(rng, ring_candidates.len())];
        let needed_exits = exo_nbrs.len();
        let matching_tmpls: Vec<&RingHopTemplate> = RING_HOP_TEMPLATES
            .iter()
            .filter(|t| t.exits == needed_exits)
            .collect();
        if matching_tmpls.is_empty() {
            continue;
        }
        let tmpl = matching_tmpls[rng_usize(rng, matching_tmpls.len())];
        let Ok(tmpl_mol) = parse_smiles(tmpl.smiles) else {
            continue;
        };

        let remove_set: BTreeSet<usize> = old_ring.iter().copied().collect();
        let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
        let mut combined = Molecule::new("");
        combined.target = parent.target.clone();
        combined.metric = parent.metric.clone();

        for (u, a) in parent.atoms.iter().enumerate() {
            if !remove_set.contains(&u) {
                let ni = combined.atoms.len();
                let mut ca = a.clone();
                ca.idx = ni;
                combined.atoms.push(ca);
                old_to_new.insert(u, ni);
            }
        }
        for b in &parent.bonds {
            if !remove_set.contains(&b.a) && !remove_set.contains(&b.b) {
                let idx = combined.bonds.len();
                combined.bonds.push(Bond {
                    idx,
                    a: old_to_new[&b.a],
                    b: old_to_new[&b.b],
                    order: b.order,
                    aromatic: b.aromatic,
                    in_ring: b.in_ring,
                    kekule_order: b.kekule_order,
                });
            }
        }

        let tmpl_offset = combined.atoms.len();
        for (tu, ta) in tmpl_mol.atoms.iter().enumerate() {
            let mut ca = ta.clone();
            ca.idx = tmpl_offset + tu;
            ca.explicit_h = 0;
            ca.implicit_h = 0;
            combined.atoms.push(ca);
        }
        for tb in &tmpl_mol.bonds {
            let idx = combined.bonds.len();
            combined.bonds.push(Bond {
                idx,
                a: tmpl_offset + tb.a,
                b: tmpl_offset + tb.b,
                order: tb.order,
                aromatic: tb.aromatic,
                in_ring: tb.in_ring,
                kekule_order: tb.kekule_order,
            });
        }

        // Stitch exocyclic anchor neighbors onto the template attachment sites
        let flip = needed_exits == 2 && rng_usize(rng, 2) == 1;
        let mut valid_stitch = true;
        for (k, &old_nb) in exo_nbrs.iter().enumerate() {
            let Some(&mapped_nb) = old_to_new.get(&old_nb) else {
                valid_stitch = false;
                break;
            };
            let t_slot = if flip { 1 - k } else { k };
            let t_atom = tmpl_offset + tmpl.attach_atoms[t_slot];
            // Avoid direct hetero-hetero single bond (N-N / N-O / O-O)
            if combined.atoms[mapped_nb].is_hetero() && combined.atoms[t_atom].is_hetero() {
                valid_stitch = false;
                break;
            }
            let idx = combined.bonds.len();
            combined.bonds.push(Bond {
                idx,
                a: mapped_nb,
                b: t_atom,
                order: BondOrder::Single,
                aromatic: false,
                in_ring: false,
                kekule_order: 1,
            });
        }
        if !valid_stitch {
            continue;
        }

        if let Some((smi, clean)) = sanitize_candidate_graph(combined, max_mw) {
            if smi != parent.smiles {
                out.push((smi, clean, format!("Ring-hop -> {}", tmpl.name)));
            }
        }
    }

    out
}

/// Extracts single-cut synthons `(kept_atoms_mol, anchor_atom_in_synthon, brics_env_opt)` from `mol`.
fn extract_crossover_synthons(mol: &Molecule) -> Vec<(Molecule, usize, Option<&'static str>)> {
    let mut synthons = Vec::new();
    for b in &mol.bonds {
        if b.in_ring || b.kekule_order != 1 {
            continue;
        }
        // BFS side A without bond `b.idx`
        let mut side_a = Vec::new();
        let mut vis = vec![false; mol.atoms.len()];
        let mut q = VecDeque::new();
        vis[b.a] = true;
        q.push_back(b.a);
        while let Some(u) = q.pop_front() {
            side_a.push(u);
            for &(v, bi) in &mol.adj[u] {
                if bi != b.idx && !vis[v] {
                    vis[v] = true;
                    q.push_back(v);
                }
            }
        }
        let side_b: Vec<usize> = (0..mol.atoms.len()).filter(|&u| !vis[u]).collect();
        if side_a.len() < 3 || side_b.len() < 3 {
            continue;
        }

        let env_a = classify_brics_atom_env(mol, b.a, b.b);
        let env_b = classify_brics_atom_env(mol, b.b, b.a);

        for (piece, anchor, env_tag) in [(&side_a, b.a, env_a), (&side_b, b.b, env_b)] {
            if piece.len() > 28 {
                continue;
            }
            let keep_set: BTreeSet<usize> = piece.iter().copied().collect();
            let mut map_idx = BTreeMap::new();
            let mut sub = Molecule::new("");
            for &u in piece {
                let ni = sub.atoms.len();
                let mut ca = mol.atoms[u].clone();
                ca.idx = ni;
                sub.atoms.push(ca);
                map_idx.insert(u, ni);
            }
            for mb in &mol.bonds {
                if keep_set.contains(&mb.a) && keep_set.contains(&mb.b) {
                    let idx = sub.bonds.len();
                    sub.bonds.push(Bond {
                        idx,
                        a: map_idx[&mb.a],
                        b: map_idx[&mb.b],
                        order: mb.order,
                        aromatic: mb.aromatic,
                        in_ring: mb.in_ring,
                        kekule_order: mb.kekule_order,
                    });
                }
            }
            synthons.push((sub, map_idx[&anchor], env_tag));
        }
    }
    synthons
}

/// Operator 3: Synthesis-Aware `BRICS` Synthon Crossover (`AutoGrow4` / `Degen L1–L16`).
fn generate_brics_crossovers(
    parent_a: &Molecule,
    donor_b: &Molecule,
    donor_label: &str,
    rng: &mut u64,
    max_mw: f64,
    limit: usize,
) -> Vec<(String, Molecule, String)> {
    let mut out = Vec::new();
    let syn_a = extract_crossover_synthons(parent_a);
    let syn_b = extract_crossover_synthons(donor_b);
    if syn_a.is_empty() || syn_b.is_empty() {
        return out;
    }

    let mut attempts = 0usize;
    while out.len() < limit && attempts < limit * 8 + 12 {
        attempts += 1;
        let (ref sa, anch_a, env_a) = syn_a[rng_usize(rng, syn_a.len())];
        let (ref sb, anch_b, env_b) = syn_b[rng_usize(rng, syn_b.len())];
        let total_atoms = sa.atoms.len() + sb.atoms.len();
        if !(11..=40).contains(&total_atoms) {
            continue;
        }

        // Prefer genuine BRICS L1-L16 reaction pairs when tags are present; otherwise enforce safe C-C / C-N / C-O coupling
        let rxn_desc = match (env_a, env_b) {
            (Some(ea), Some(eb)) => {
                if let Some((rxn_name, _)) = match_brics_reaction(ea, eb) {
                    format!("BRICS {ea}-{eb} ({rxn_name}) x {donor_label}")
                } else {
                    continue;
                }
            }
            _ => {
                if sa.atoms[anch_a].is_hetero() && sb.atoms[anch_b].is_hetero() {
                    continue;
                }
                format!("Synthon crossover x {donor_label}")
            }
        };

        let mut merged = sa.clone();
        let offset = merged.atoms.len();
        for (u, a) in sb.atoms.iter().enumerate() {
            let mut ca = a.clone();
            ca.idx = offset + u;
            merged.atoms.push(ca);
        }
        for b in &sb.bonds {
            let idx = merged.bonds.len();
            merged.bonds.push(Bond {
                idx,
                a: offset + b.a,
                b: offset + b.b,
                order: b.order,
                aromatic: b.aromatic,
                in_ring: b.in_ring,
                kekule_order: b.kekule_order,
            });
        }
        let idx = merged.bonds.len();
        merged.bonds.push(Bond {
            idx,
            a: anch_a,
            b: offset + anch_b,
            order: BondOrder::Single,
            aromatic: false,
            in_ring: false,
            kekule_order: 1,
        });

        if let Some((smi, clean)) = sanitize_candidate_graph(merged, max_mw) {
            if smi != parent_a.smiles && smi != donor_b.smiles {
                out.push((smi, clean, rxn_desc));
            }
        }
    }

    out
}

struct QueryAnchorContext<'a> {
    mol: &'a Molecule,
    fp: Ecfp2048,
    pharm: PharmacophoreMap,
    s3d: Summary3dProfile,
    ds: Option<&'a DatasetAnalysis>,
}

fn score_evolved_individual(
    mut cand_mol: Molecule,
    can_smi: String,
    gen_born: usize,
    op_kind: EvolutionOperatorKind,
    op_detail: String,
    lineage: String,
    qctx: &QueryAnchorContext<'_>,
    cfg: &DeNovoConfig,
) -> Option<EvolvedCandidate> {
    let alerts = scan_structural_alerts(&cand_mol);
    let pains_cnt = alerts
        .iter()
        .filter(|a| a.severity == AlertSeverity::PainsAssayInterference)
        .count();
    if cfg.enforce_pains_free && pains_cnt > 0 {
        return None;
    }

    let physchem = evaluate_physchem(&cand_mol, alerts.len());
    if physchem.mw > cfg.max_mw {
        return None;
    }
    let synth = evaluate_synthetic_accessibility(&cand_mol);
    if synth.sa_score > cfg.max_sa_score + 0.8 {
        return None;
    }

    let pka = evaluate_physiological_pka_74(&cand_mol);
    let admet = evaluate_admet(&cand_mol, &physchem);

    let fp = compute_ecfp(&cand_mol, 2);
    let ecfp_sim = qctx.fp.tanimoto(&fp) as f64;
    let pharm_map = extract_pharmacophore_map(&cand_mol);
    let pharm_align = align_pharmacophores(&qctx.pharm, &pharm_map, ecfp_sim as f32);
    let pharm2d_sim = pharm_align.pharm_similarity as f64;

    let c3d = evaluate_3d_summary_profile(&cand_mol, true);
    let usrcat_sim = usrcat_similarity(&qctx.s3d.usrcat60, &c3d.usrcat60);

    // Estimate binding affinity (pAct) combining Free-Wilson (when matching core) with
    // 3D USRCAT + 2D Pharmacophore Pocket-Recognition Transfer + Enthalpic Efficiency Bonus
    let fw_pact = qctx.ds.and_then(|ds| {
        predict_best_free_wilson(
            &cand_mol,
            ds.default_sar_matrix.as_ref(),
            ds.default_sar_matrix.as_ref(),
            &ds.mols,
            &ds.profiles,
        )
        .map(|p| p.predicted_pact)
    });

    let sim_anchor = 0.52 * usrcat_sim + 0.48 * pharm2d_sim;
    let size_penalty = if physchem.heavy_atoms > 34 {
        0.045 * ((physchem.heavy_atoms - 34) as f64)
    } else {
        0.0
    };
    let polar_bonus = if (35.0..=105.0).contains(&physchem.tpsa) && physchem.clogp <= 3.8 {
        0.28
    } else {
        -0.08
    };
    let fsp3_bonus = (physchem.fsp3 - 0.20).clamp(-0.10, 0.30) * 0.65;
    let pharm_transfer_pact =
        (qctx.mol.pact * (0.72 + 0.36 * sim_anchor) + polar_bonus + fsp3_bonus - size_penalty)
            .clamp(4.5, 9.8);
    let predicted_pact = match fw_pact {
        Some(fw) => (0.55 * fw + 0.45 * pharm_transfer_pact).clamp(4.5, 9.8),
        None => pharm_transfer_pact,
    };
    let predicted_ic50_nm = 10.0_f64.powf(9.0 - predicted_pact);

    cand_mol.pact = predicted_pact;
    cand_mol.raw_nm = predicted_ic50_nm;
    cand_mol.target = qctx.mol.target.clone();
    cand_mol.metric = "Pred_IC50".to_string();

    let hac = physchem.heavy_atoms.max(1) as f64;
    let dg = RT_LN10_KCAL * predicted_pact;
    let le = dg / hac;
    let sile = dg / hac.powf(0.30);
    let lle = predicted_pact - physchem.clogp;
    let lle_74 = predicted_pact - pka.microstate_logd_74;

    let is_novel_scaffold = !pharm_map.murcko_framework_key.is_empty()
        && pharm_map.murcko_framework_key != qctx.pharm.murcko_framework_key;
    let composite_3d_2d = 0.55 * usrcat_sim + 0.45 * pharm2d_sim;
    let is_true_scaffold_hop =
        is_novel_scaffold && composite_3d_2d >= 0.42 && ecfp_sim <= 0.52;

    // Multi-objective desirability components in [0..1]
    let d_sim = ((composite_3d_2d - 0.22) / 0.58).clamp(0.05, 1.0);
    let d_eff = (0.48 * ((le - 0.18) / 0.22).clamp(0.05, 1.0)
        + 0.52 * ((lle - 2.2) / 3.6).clamp(0.05, 1.0))
    .clamp(0.05, 1.0);
    let d_sa = ((cfg.max_sa_score + 0.8 - synth.sa_score) / 3.4).clamp(0.05, 1.0);
    let d_nov = match cfg.goal {
        DeNovoGoalPreset::ScaffoldHop3dRocs => {
            let hop_term = if is_novel_scaffold { 0.55 } else { 0.12 };
            let ecfp_sweet = if (0.15..=0.48).contains(&ecfp_sim) {
                0.45
            } else if ecfp_sim < 0.15 {
                0.25
            } else {
                0.45 * (1.0 - (ecfp_sim - 0.48) / 0.52).clamp(0.0, 1.0)
            };
            (hop_term + ecfp_sweet).clamp(0.05, 1.0)
        }
        DeNovoGoalPreset::CnsEscapeFlatland => {
            let d_mpo = (physchem.cns_mpo / 5.4).clamp(0.05, 1.0);
            let d_sp3 = (physchem.fsp3 / 0.45).clamp(0.05, 1.0);
            (0.55 * d_mpo + 0.45 * d_sp3).clamp(0.05, 1.0)
        }
        _ => {
            let hop_bonus = if is_novel_scaffold { 0.22 } else { 0.0 };
            (0.78 * physchem.qed_mean + hop_bonus).clamp(0.05, 1.0)
        }
    };

    let (w_sim, w_eff, w_sa, w_nov) = cfg.goal.default_weights();
    let pains_pen = if pains_cnt > 0 { 0.55 } else { 1.0 };
    let sa_soft_pen = if synth.sa_score > cfg.max_sa_score {
        0.78
    } else {
        1.0
    };
    let fitness_score = ((w_sim * d_sim + w_eff * d_eff + w_sa * d_sa + w_nov * d_nov)
        * 100.0
        * pains_pen
        * sa_soft_pen)
        .clamp(1.0, 99.9);

    let mut physchem_updated = physchem;
    physchem_updated.lle = lle;

    Some(EvolvedCandidate {
        rank: 0,
        candidate_id: String::new(),
        generation_born: gen_born,
        last_operator: op_kind,
        operator_detail: op_detail,
        lineage_trace: lineage,
        canonical_smiles: can_smi,
        scaffold_key: pharm_map.murcko_framework_key,
        scaffold_label: pharm_map.murcko_label,
        is_novel_scaffold,
        is_true_scaffold_hop,
        pareto_optimal: false,
        ecfp4_tanimoto: ecfp_sim,
        pharm2d_similarity: pharm2d_sim,
        usrcat_3d_similarity: usrcat_sim,
        rocs_shape_tanimoto: usrcat_sim,
        rocs_color_tanimoto: pharm2d_sim,
        rocs_esp_similarity: 0.50,
        rocs_combo_tanimoto: composite_3d_2d,
        predicted_pact,
        predicted_ic50_nm,
        le,
        lle,
        lle_74,
        sile,
        sa_score: synth.sa_score,
        physchem: physchem_updated,
        logd_74: pka.microstate_logd_74,
        esol_ug_ml: admet.esol_ug_ml,
        boiled_egg: admet.boiled_egg,
        shape3d_class: c3d.shape_class,
        fitness_score,
        aligned_coords3d: c3d.coords3d,
        gasteiger_charges: c3d.gasteiger_charges,
        aligned_colors: c3d.color_atoms,
        mol: cand_mol,
    })
}

/// Runs multi-generation evolutionary de novo molecule design seeded from `query_mol` (and optional
/// `DatasetAnalysis` crossover partners), returning ranked, diversity-pruned, 3D-`ROCS`-aligned candidates.
pub fn evolve_de_novo_molecules(
    query_mol: &Molecule,
    ds: Option<&DatasetAnalysis>,
    config: DeNovoConfig,
) -> DeNovoEvolutionReport {
    let mut seed_mol = query_mol.clone();
    if seed_mol.smiles.is_empty() {
        seed_mol.smiles = molecule_to_smiles(&seed_mol);
    }
    if seed_mol.atoms.iter().all(|a| a.pos2d == [0.0, 0.0]) && seed_mol.atoms.len() > 1 {
        compute_2d_coords(&mut seed_mol);
    }
    if seed_mol.pact <= 0.0 {
        seed_mol.pact = 7.20;
        seed_mol.raw_nm = 63.1;
    }

    let q_fp = compute_ecfp(&seed_mol, 2);
    let q_pharm = extract_pharmacophore_map(&seed_mol);
    let q_3d = evaluate_3d_summary_profile(&seed_mol, false);

    let qctx = QueryAnchorContext {
        mol: &seed_mol,
        fp: q_fp,
        pharm: q_pharm.clone(),
        s3d: q_3d.clone(),
        ds,
    };

    // Assemble crossover donor pool (active leads from DatasetAnalysis + curated medicinal-chemistry synthons)
    let mut donor_pool: Vec<(String, Molecule)> = Vec::new();
    if let Some(dataset) = ds {
        for m in dataset.mols.iter().filter(|m| m.pact >= 6.4).take(14) {
            if m.smiles != seed_mol.smiles {
                donor_pool.push((m.name.clone(), m.clone()));
            }
        }
    }
    for &(dname, dsmi) in CURATED_CROSSOVER_DONORS {
        if let Ok(dmol) = parse_smiles(dsmi) {
            donor_pool.push((dname.to_string(), dmol));
        }
    }

    let gens = config.generations.clamp(1, 20);
    let pop_cap = config.population_size.clamp(8, 80);
    let mut rng = config.random_seed ^ ((seed_mol.atoms.len() as u64).wrapping_mul(0x517C_C1B7));

    let mut seen_smiles: BTreeSet<String> = BTreeSet::new();
    seen_smiles.insert(seed_mol.smiles.clone());

    let mut archive: Vec<EvolvedCandidate> = Vec::new();
    let mut current_parents: Vec<(Molecule, String)> =
        vec![(seed_mol.clone(), "Gen0:Seed".to_string())];
    let mut trajectory: Vec<GenerationSnapshot> = Vec::with_capacity(gens);

    for gen in 1..=gens {
        let mut raw_proposals: Vec<(
            String,
            Molecule,
            EvolutionOperatorKind,
            String,
            String,
        )> = Vec::new();

        for (p_mol, p_lineage) in &current_parents {
            if config.enable_graph_mutations {
                for (smi, m, det) in generate_graph_mutations(p_mol, &mut rng, config.max_mw, 4) {
                    if seen_smiles.insert(smi.clone()) {
                        let trace = format!("{p_lineage} -> G{gen}:Mut({det})");
                        raw_proposals.push((
                            smi,
                            m,
                            EvolutionOperatorKind::GraphMutation,
                            det,
                            trace,
                        ));
                    }
                }
            }
            if config.enable_ring_hops {
                for (smi, m, det) in generate_ring_hops(p_mol, &mut rng, config.max_mw, 4) {
                    if seen_smiles.insert(smi.clone()) {
                        let trace = format!("{p_lineage} -> G{gen}:Hop({det})");
                        raw_proposals.push((smi, m, EvolutionOperatorKind::RingHop, det, trace));
                    }
                }
            }
            if config.enable_brics_crossover && !donor_pool.is_empty() {
                let (ref d_lbl, ref d_mol) = donor_pool[rng_usize(&mut rng, donor_pool.len())];
                for (smi, m, det) in
                    generate_brics_crossovers(p_mol, d_mol, d_lbl, &mut rng, config.max_mw, 3)
                {
                    if seen_smiles.insert(smi.clone()) {
                        let trace = format!("{p_lineage} -> G{gen}:Xover({d_lbl})");
                        raw_proposals.push((
                            smi,
                            m,
                            EvolutionOperatorKind::BricsCrossover,
                            det,
                            trace,
                        ));
                    }
                }
            }
        }

        // Score all novel offspring in parallel via Rayon
        let gen_scored: Vec<EvolvedCandidate> = raw_proposals
            .into_par_iter()
            .filter_map(|(smi, m, op_k, det, tr)| {
                score_evolved_individual(m, smi, gen, op_k, det, tr, &qctx, &config)
            })
            .collect();

        let eval_cnt = gen_scored.len();
        archive.extend(gen_scored);

        // Sort archive by fitness descending
        archive.sort_by(|a, b| {
            b.fitness_score
                .partial_cmp(&a.fitness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select diverse elitist mating pool for next generation (penalize >0.82 ECFP4 clones)
        let mut next_pool: Vec<(Molecule, String)> = Vec::with_capacity(pop_cap);
        let mut pool_fps: Vec<Ecfp2048> = Vec::with_capacity(pop_cap);
        for cand in &archive {
            if next_pool.len() >= pop_cap {
                break;
            }
            let c_fp = compute_ecfp(&cand.mol, 2);
            let too_close = pool_fps.iter().any(|pfp| pfp.tanimoto(&c_fp) > 0.85);
            if !too_close || next_pool.len() < 4 {
                pool_fps.push(c_fp);
                next_pool.push((cand.mol.clone(), cand.lineage_trace.clone()));
            }
        }
        if !next_pool.is_empty() {
            current_parents = next_pool;
        }

        // Record generation telemetry snapshot
        let top_slice = &archive[..archive.len().min(10)];
        let best_fit = top_slice.first().map(|c| c.fitness_score).unwrap_or(0.0);
        let mean_top10 = if !top_slice.is_empty() {
            top_slice.iter().map(|c| c.fitness_score).sum::<f64>() / (top_slice.len() as f64)
        } else {
            0.0
        };
        let best_sim = archive
            .iter()
            .take(25)
            .map(|c| c.rocs_combo_tanimoto)
            .fold(0.0_f64, f64::max);
        let best_lle = archive
            .iter()
            .take(25)
            .map(|c| c.lle)
            .fold(0.0_f64, f64::max);
        let mean_sa = if !top_slice.is_empty() {
            top_slice.iter().map(|c| c.sa_score).sum::<f64>() / (top_slice.len() as f64)
        } else {
            0.0
        };
        let scaf_set: BTreeSet<&str> = archive
            .iter()
            .take(40)
            .map(|c| c.scaffold_key.as_str())
            .collect();
        let hop_cnt = archive
            .iter()
            .take(40)
            .filter(|c| c.is_novel_scaffold)
            .count();
        let denom = archive.len().min(40).max(1) as f64;

        trajectory.push(GenerationSnapshot {
            generation: gen,
            evaluated_count: eval_cnt,
            best_fitness: best_fit,
            mean_top10_fitness: mean_top10,
            best_3d_pharm_sim: best_sim,
            best_lle,
            mean_sa_score: mean_sa,
            unique_scaffolds: scaf_set.len(),
            scaffold_hop_pct: (hop_cnt as f64) / denom * 100.0,
        });
    }

    let total_unique_evaluated = archive.len();

    // Diversity-filtered top candidate selection (max 3 per identical Murcko framework first, then backfill)
    let mut selected: Vec<EvolvedCandidate> = Vec::with_capacity(config.max_returned);
    let mut scaf_quota: BTreeMap<String, usize> = BTreeMap::new();
    let mut leftover: Vec<EvolvedCandidate> = Vec::new();
    for cand in archive {
        let cnt = scaf_quota.entry(cand.scaffold_key.clone()).or_insert(0);
        if *cnt < 3 && selected.len() < config.max_returned {
            *cnt += 1;
            selected.push(cand);
        } else if leftover.len() < config.max_returned {
            leftover.push(cand);
        }
    }
    for cand in leftover {
        if selected.len() >= config.max_returned {
            break;
        }
        selected.push(cand);
    }

    // Refine exact 3D Gaussian ROCS Shape + Color + ESP alignment in parallel on the final selected leads!
    selected.par_iter_mut().for_each(|cand| {
        let c3d = evaluate_3d_summary_profile(&cand.mol, false);
        let rocs = align_rocs_3d(
            &seed_mol,
            &q_3d.coords3d,
            &q_3d.gasteiger_charges,
            &q_3d.color_atoms,
            &cand.mol,
            &c3d.coords3d,
            &c3d.gasteiger_charges,
            &c3d.color_atoms,
        );
        cand.rocs_shape_tanimoto = rocs.shape_tanimoto;
        cand.rocs_color_tanimoto = rocs.color_tanimoto;
        cand.rocs_esp_similarity = rocs.esp_hodgkin_sim;
        cand.rocs_combo_tanimoto = rocs.combo_tanimoto;
        cand.aligned_coords3d = rocs.aligned_hit_coords;
        cand.gasteiger_charges = c3d.gasteiger_charges;
        cand.aligned_colors = rocs.aligned_hit_colors;
        cand.is_true_scaffold_hop = cand.is_novel_scaffold
            && (cand.rocs_combo_tanimoto >= 0.38 || cand.pharm2d_similarity >= 0.40)
            && cand.ecfp4_tanimoto <= 0.52;
    });

    // Re-sort final returned list incorporating exact 3D ROCS ComboTanimoto adjustment
    for cand in &mut selected {
        cand.fitness_score = (0.82 * cand.fitness_score + 18.0 * cand.rocs_combo_tanimoto)
            .clamp(1.0, 99.9);
    }
    selected.sort_by(|a, b| {
        b.fitness_score
            .partial_cmp(&a.fitness_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mark 3-objective non-dominated Pareto front (ROCS+Pharm2D MAX, LLE MAX, SAScore MIN)
    for i in 0..selected.len() {
        let si = 0.5 * selected[i].rocs_combo_tanimoto + 0.5 * selected[i].pharm2d_similarity;
        let li = selected[i].lle;
        let sai = selected[i].sa_score;
        let mut dominated = false;
        for j in 0..selected.len() {
            if i == j {
                continue;
            }
            let sj = 0.5 * selected[j].rocs_combo_tanimoto + 0.5 * selected[j].pharm2d_similarity;
            let lj = selected[j].lle;
            let saj = selected[j].sa_score;
            if sj >= si && lj >= li && saj <= sai && (sj > si || lj > li || saj < sai) {
                dominated = true;
                break;
            }
        }
        selected[i].pareto_optimal = !dominated;
    }

    let mut op_counts = [0usize; 3];
    for (idx, cand) in selected.iter_mut().enumerate() {
        cand.rank = idx + 1;
        cand.candidate_id = format!("EVO-{:03}", idx + 1);
        cand.mol.id = cand.candidate_id.clone();
        cand.mol.name = format!(
            "{}-G{}-{}",
            if seed_mol.name.is_empty() {
                "Lead"
            } else {
                &seed_mol.name
            },
            cand.generation_born,
            cand.rank
        );
        match cand.last_operator {
            EvolutionOperatorKind::GraphMutation => op_counts[0] += 1,
            EvolutionOperatorKind::RingHop => op_counts[1] += 1,
            EvolutionOperatorKind::BricsCrossover => op_counts[2] += 1,
            EvolutionOperatorKind::SeedTemplate => {}
        }
    }

    let novel_scaffold_count = selected.iter().filter(|c| c.is_novel_scaffold).count();
    let true_scaffold_hop_count = selected.iter().filter(|c| c.is_true_scaffold_hop).count();

    DeNovoEvolutionReport {
        query_id: if seed_mol.id.is_empty() {
            "QUERY".to_string()
        } else {
            seed_mol.id.clone()
        },
        query_name: if seed_mol.name.is_empty() {
            "Seed Template".to_string()
        } else {
            seed_mol.name.clone()
        },
        query_smiles: seed_mol.smiles.clone(),
        query_pact: seed_mol.pact,
        query_scaffold_label: q_pharm.murcko_label,
        query_mol: seed_mol,
        query_coords3d: q_3d.coords3d,
        query_charges: q_3d.gasteiger_charges,
        query_colors: q_3d.color_atoms,
        config,
        total_unique_evaluated,
        novel_scaffold_count,
        true_scaffold_hop_count,
        operator_counts: op_counts,
        trajectory,
        candidates: selected,
    }
}

/// Exports ranked evolved de novo candidates to `.csv`.
pub fn export_de_novo_candidates_csv(rep: &DeNovoEvolutionReport) -> String {
    let mut out = String::with_capacity(rep.candidates.len() * 240 + 256);
    out.push_str("rank,id,generation,operator,operator_detail,fitness_100,pareto_optimal,novel_scaffold,true_scaffold_hop,rocs_combo_tanimoto,rocs_shape_tanimoto,rocs_color_tanimoto,pharm2d_sim,ecfp4_tanimoto,pred_pact,pred_ic50_nm,le,lle,lle_74,sa_score,mw,clogp,logd_74,tpsa,fsp3,qed,cns_mpo,scaffold_label,smiles,lineage\n");
    for c in &rep.candidates {
        out.push_str(&format!(
            "{},\"{}\",{},\"{}\",\"{}\",{:.2},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{:.3},{:.3},{:.2},\"{}\",\"{}\",\"{}\"\n",
            c.rank,
            c.candidate_id,
            c.generation_born,
            c.last_operator.short_label(),
            c.operator_detail.replace('"', "'"),
            c.fitness_score,
            c.pareto_optimal,
            c.is_novel_scaffold,
            c.is_true_scaffold_hop,
            c.rocs_combo_tanimoto,
            c.rocs_shape_tanimoto,
            c.rocs_color_tanimoto,
            c.pharm2d_similarity,
            c.ecfp4_tanimoto,
            c.predicted_pact,
            c.predicted_ic50_nm,
            c.le,
            c.lle,
            c.lle_74,
            c.sa_score,
            c.physchem.mw,
            c.physchem.clogp,
            c.logd_74,
            c.physchem.tpsa,
            c.physchem.fsp3,
            c.physchem.qed_mean,
            c.physchem.cns_mpo,
            c.scaffold_label.replace('"', "'"),
            c.canonical_smiles.replace('"', "'"),
            c.lineage_trace.replace('"', "'")
        ));
    }
    out
}

/// Exports 3D `ROCS`-aligned evolved candidate structures as a multi-record MDL V2000 `.sdf` with discovery SD tags.
pub fn format_de_novo_3d_sdf(cands: &[EvolvedCandidate]) -> String {
    let mut sdf = String::with_capacity(cands.len() * 2048);
    for c in cands {
        let na = c.mol.atoms.len();
        let nb = c.mol.bonds.len();
        sdf.push_str(&format!("{}\n", c.candidate_id));
        sdf.push_str("  SAR-Atlas v2.2.0 DeNovo 3D ROCS-Aligned\n\n");
        sdf.push_str(&format!(
            "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
            na, nb
        ));
        for (i, a) in c.mol.atoms.iter().enumerate() {
            let p = c.aligned_coords3d.get(i).copied().unwrap_or([
                a.pos2d[0],
                a.pos2d[1],
                0.0,
            ]);
            sdf.push_str(&format!(
                "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
                p[0],
                p[1],
                p[2],
                a.elem.symbol()
            ));
        }
        for b in &c.mol.bonds {
            let bo = match b.order {
                BondOrder::Single => 1,
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
                BondOrder::Aromatic => 4,
            };
            sdf.push_str(&format!(
                "{:>3}{:>3}{:>3}  0  0  0  0\n",
                b.a + 1,
                b.b + 1,
                bo
            ));
        }
        sdf.push_str("M  END\n");
        for (tag, val) in [
            ("DeNovo_Rank", c.rank.to_string()),
            ("Operator", c.last_operator.label().to_string()),
            ("Operator_Detail", c.operator_detail.clone()),
            ("Fitness_100", format!("{:.2}", c.fitness_score)),
            ("ROCS_ComboTanimoto", format!("{:.3}", c.rocs_combo_tanimoto)),
            ("Pharm2D_Similarity", format!("{:.3}", c.pharm2d_similarity)),
            ("Predicted_pAct", format!("{:.2}", c.predicted_pact)),
            ("LE", format!("{:.3}", c.le)),
            ("LLE", format!("{:.2}", c.lle)),
            ("SAScore", format!("{:.2}", c.sa_score)),
            ("SMILES", c.canonical_smiles.clone()),
            ("Lineage", c.lineage_trace.clone()),
        ] {
            sdf.push_str(&format!(">  <{tag}>\n{val}\n\n"));
        }
        sdf.push_str("$$$$\n");
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::load_benchmark_26;

    #[test]
    fn test_de_novo_evolution_operators_and_3d_rocs_guidance() {
        let mols = load_benchmark_26();
        let ds = DatasetAnalysis::from_molecules(mols, 0.30, 6.0);
        // Seed with Erlotinib (CHEMBL553) or Celecoxib (CHEMBL118)
        let cel_idx = ds.mols.iter().position(|m| m.id == "CHEMBL118").unwrap_or(0);
        let cfg = DeNovoConfig {
            goal: DeNovoGoalPreset::ScaffoldHop3dRocs,
            generations: 4,
            population_size: 20,
            max_returned: 16,
            ..DeNovoConfig::default()
        };
        let rep = evolve_de_novo_molecules(&ds.mols[cel_idx], Some(&ds), cfg);
        assert!(rep.total_unique_evaluated >= 12);
        assert!(!rep.candidates.is_empty());
        assert_eq!(rep.trajectory.len(), 4);
        assert!(rep.novel_scaffold_count >= 1);
        for c in &rep.candidates {
            assert!(c.sa_score >= 1.0 && c.sa_score <= 10.0);
            assert_eq!(c.aligned_coords3d.len(), c.mol.atoms.len());
            assert!(parse_smiles(&c.canonical_smiles).is_ok());
        }
        let csv = export_de_novo_candidates_csv(&rep);
        assert_eq!(csv.lines().count(), rep.candidates.len() + 1);
        let sdf = format_de_novo_3d_sdf(&rep.candidates);
        assert!(sdf.contains("V2000") && sdf.contains("$$$$"));
    }
}
