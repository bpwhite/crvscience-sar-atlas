//! Targeted Protein Degradation (`TPD`) / `PROTAC` E3 Ligase Recruiter (`CRBN` & `VHL`),
//! Solvent-Exposed Exit-Vector Linkerology Enumerator, Beyond-Rule-of-5 (`bRo5`) Molecular
//! Chameleonicity (`dIMHB` / `ePSA` / AbbVie `AB-MPS`) Profiler, and Douglass et al. (2013)
//! Exact Three-Body Ternary-Complex & Hook-Effect Equilibrium Solver.
//!
//! Primary References:
//! - Troup, Fallan & Baud (2020), *Explor. Target Antitumor Ther.* 1(5), 273-312 ("Current
//!   strategies for the design of PROTAC linkers: a critical review":
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC9400730/`)
//! - Békés, Langley & Crews (2022), *Nat. Rev. Drug Discov.* 21(3), 181-200 ("PROTAC targeted
//!   protein degraders: the past is prologue": `https://pmc.ncbi.nlm.nih.gov/articles/PMC8765495/`)
//! - DeGoey, Chen, Cox & Wendt (2018), *J. Med. Chem.* 61(7), 2636-2651 ("Beyond the Rule of 5:
//!   Lessons Learned from AbbVie's Drugs and Compound Collection" — `AB-MPS = |cLogD - 3| + NAR + NRotB`:
//!   `https://pubmed.ncbi.nlm.nih.gov/28926247/`)
//! - Caron & Ermondi (2020-2024) ("Molecular properties, including chameleonicity, as essential
//!   tools for designing the next generation of oral bRo5 drugs" — dynamic intramolecular H-bonds
//!   `dIMHBs` & `ePSA` folding: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11542721/`,
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC9574858/`)
//! - Douglass, Miller, Sparer, Shapiro & Spiegel (2013), *J. Am. Chem. Soc.* 135(16), 6092-6099
//!   ("A comprehensive mathematical model for three-body binding equilibria" — `TF50`, `[ABC]_max`,
//!   and `TI50` Hook-Effect autoinhibition: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3717292/`)

use crate::alerts::scan_structural_alerts;
use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::{evaluate_physchem, PhyschemProfile};
use crate::pka_protomer::evaluate_physiological_pka_74;
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Number of logarithmic `[PROTAC]` dose points (`0.01 nM ..= 100,000 nM`) in the Hook-Effect curve.
pub const HOOK_CURVE_LEN: usize = 33;

/// E3 ubiquitin ligase family & clinical recruiter chemotype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum E3RecruiterKind {
    CrbnPomalidomide4NH,
    CrbnLenalidomide4C,
    CrbnThalidomide5Pip,
    CrbnPhenylGlutarimide,
    VhlVh032Amide,
    VhlVh032Phenolic,
}

impl E3RecruiterKind {
    pub const ALL: [Self; 6] = [
        Self::CrbnPomalidomide4NH,
        Self::CrbnLenalidomide4C,
        Self::CrbnThalidomide5Pip,
        Self::CrbnPhenylGlutarimide,
        Self::VhlVh032Amide,
        Self::VhlVh032Phenolic,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::CrbnPomalidomide4NH => "crbn_pom_4nh",
            Self::CrbnLenalidomide4C => "crbn_len_4c",
            Self::CrbnThalidomide5Pip => "crbn_5pip",
            Self::CrbnPhenylGlutarimide => "crbn_pg",
            Self::VhlVh032Amide => "vhl_vh032_lhs",
            Self::VhlVh032Phenolic => "vhl_vh032_rhs",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::CrbnPomalidomide4NH => "CRBN: Pomalidomide (4-NH)",
            Self::CrbnLenalidomide4C => "CRBN: Lenalidomide (Isoindolinone)",
            Self::CrbnThalidomide5Pip => "CRBN: 5-Piperazinyl-Thalidomide (ARV-110)",
            Self::CrbnPhenylGlutarimide => "CRBN: Phenyl-Glutarimide (Hydrolysis-Stable)",
            Self::VhlVh032Amide => "VHL: VH032 Left-Hand Amide (MZ1)",
            Self::VhlVh032Phenolic => "VHL: VH032 Right-Hand Phenolic (ACBI1)",
        }
    }

    pub fn ligase_family(self) -> &'static str {
        match self {
            Self::CrbnPomalidomide4NH
            | Self::CrbnLenalidomide4C
            | Self::CrbnThalidomide5Pip
            | Self::CrbnPhenylGlutarimide => "CRBN",
            Self::VhlVh032Amide | Self::VhlVh032Phenolic => "VHL",
        }
    }

    /// Binary dissociation constant `K_d,E3` (in `nM`) for the E3 substrate receptor.
    pub fn binary_kd_nm(self) -> f64 {
        match self {
            Self::CrbnPomalidomide4NH => 155.0,
            Self::CrbnLenalidomide4C => 235.0,
            Self::CrbnThalidomide5Pip => 145.0,
            Self::CrbnPhenylGlutarimide => 185.0,
            Self::VhlVh032Amide => 180.0,
            Self::VhlVh032Phenolic => 210.0,
        }
    }

    /// Intrinsic Protein-Protein Interaction (`PPI`) plastic-surface cooperativity multiplier.
    pub fn intrinsic_ppi_factor(self) -> f64 {
        match self {
            Self::CrbnPomalidomide4NH => 1.15,
            Self::CrbnLenalidomide4C => 1.20,
            Self::CrbnThalidomide5Pip => 1.45,
            Self::CrbnPhenylGlutarimide => 1.30,
            Self::VhlVh032Amide => 1.85,
            Self::VhlVh032Phenolic => 1.95,
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self.ligase_family() {
            "CRBN" => [52, 211, 153], // Emerald
            _ => [192, 132, 252],     // Violet
        }
    }

    /// Returns `(smiles_with_Br_dummy_anchor)` where the unique `Br` atom marks the linker exit vector!
    fn template_smiles_with_dummy(self) -> &'static str {
        match self {
            Self::CrbnPomalidomide4NH => "O=C1CCC(N2C(=O)c3cccc(NBr)c3C2=O)C(=O)N1",
            Self::CrbnLenalidomide4C => "O=C1CCC(N2Cc3c(Br)cccc3C2=O)C(=O)N1",
            Self::CrbnThalidomide5Pip => "O=C1CCC(N2C(=O)c3cc(N4CCN(Br)CC4)ccc3C2=O)C(=O)N1",
            Self::CrbnPhenylGlutarimide => "O=C1CCC(c2ccc(NBr)cc2)C(=O)N1",
            Self::VhlVh032Amide => "Cc1ncsc1c2ccc(CNC(=O)C3CC(O)CN3C(=O)C(NBr)C(C)(C)C)cc2",
            Self::VhlVh032Phenolic => "CC(=O)NC(C(C)(C)C)C(=O)N1CC(O)CC1C(=O)NCc2ccc(c3scnc3C)c(OBr)c2",
        }
    }
}

/// Linker architectural rigidity & polarity class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkerClass {
    FlexiblePeg,
    FlexibleAlkyl,
    SemiRigidHeterocycle,
    RigidSpiroOrAlkyne,
}

impl LinkerClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::FlexiblePeg => "Flexible PEG Ether",
            Self::FlexibleAlkyl => "Flexible Alkyl",
            Self::SemiRigidHeterocycle => "Semi-Rigid Piperazine/Click",
            Self::RigidSpiroOrAlkyne => "Rigid Oral Spiro/Bipiperidine",
        }
    }
}

/// Beyond-Rule-of-5 (`bRo5`) permeability & oral bioavailability regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bro5PermeabilityClass {
    OralBro5SweetSpot,
    ChameleonicPermeable,
    CellularToolDegrader,
    PermeabilityLimited,
}

impl Bro5PermeabilityClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::OralBro5SweetSpot => "Oral bRo5 Sweet Spot",
            Self::ChameleonicPermeable => "Chameleonic Permeable",
            Self::CellularToolDegrader => "Cellular Tool Degrader",
            Self::PermeabilityLimited => "Permeability-Limited",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::OralBro5SweetSpot => [52, 211, 153],   // Emerald
            Self::ChameleonicPermeable => [56, 189, 248], // Sky Blue
            Self::CellularToolDegrader => [251, 191, 36], // Amber
            Self::PermeabilityLimited => [248, 113, 113], // Coral Red
        }
    }
}

/// Solvent-exposed conjugation exit vector perceived on a Protein-of-Interest (`POI`) ligand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoiExitVector {
    pub vector_idx: usize,
    pub anchor_atom: usize,
    pub prune_atoms: Vec<usize>,
    pub needs_carbonyl_or_ch2_bridge: bool,
    pub label: String,
    #[serde(skip_deserializing)]
    pub rationale: &'static str,
    pub exposure_score: f64,
}

/// Quantitative Beyond-Rule-of-5 (`bRo5`) & Caron/Ermondi Chameleonicity profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChameleonicityProfile {
    /// DeGoey et al. (2018) AbbVie Multiparametric Score: `|logD_7.4 - 3.0| + NAR + NRotB` (`<= 14` oral sweet spot).
    pub abbv_mps: f64,
    /// Dynamic Intramolecular Hydrogen Bond (`dIMHB`) donor-acceptor pairs reachable in non-polar media.
    pub dimhb_count: usize,
    /// Static 2D Topological Polar Surface Area (`TPSA`, `A^2`, open aqueous cytosol state).
    pub tpsa_open_a2: f64,
    /// Predicted 3D Experimental/Exposed Polar Surface Area (`ePSA`, `A^2`, folded membrane state).
    pub epsa_folded_a2: f64,
    /// Polar surface masking upon aqueous -> lipid transition (`TPSA - ePSA`, `A^2`).
    pub delta_psa_masked_a2: f64,
    /// Caron/Ermondi Chameleonicity Index `[0.00 .. 1.00]` (`ChamelogD` / folding adaptability).
    pub chameleonicity_index: f64,
    /// Estimated `ChamelogD` lipophilicity shift (`logP_nonpolar_folded - logD_7.4_open`).
    pub chamelog_delta: f64,
    pub bro5_violations: usize,
    pub permeability_class: Bro5PermeabilityClass,
}

/// Douglass et al. (2013) exact three-body (`POI + PROTAC + E3`) ternary complex & Hook-Effect kinetics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TernaryHookProfile {
    /// Ternary cooperativity factor `alpha = K_d,binary / K_d,ternary` (`alpha > 1` positive cooperativity).
    pub cooperativity_alpha: f64,
    /// Peak fraction `[POI-PROTAC-E3]_max / min([POI]_tot, [E3]_tot)` in `[0.0 .. 1.0]`.
    pub max_ternary_fraction: f64,
    /// Concentration `[PROTAC]_t,max` (`nM`) yielding peak ternary complex formation.
    pub optimal_protac_nm: f64,
    /// `TF50` (`nM`, left-hand ascending half-maximal ternary formation concentration, `DC50` proxy).
    pub tf50_nm: f64,
    /// `TI50` (`nM`, right-hand descending Hook-Effect 50% autoinhibition concentration).
    pub ti50_nm: f64,
    /// Effective non-autoinhibited dosing window ratio `TI50 / TF50`.
    pub hook_window_ratio: f64,
    /// 33-point log10 `[PROTAC]` grid (`0.01 nM ..= 100,000 nM`).
    pub dose_grid_nm: Vec<f32>,
    /// Ternary complex `%` (`0..100%` of limiting protein) at each dose point.
    pub ternary_pct_curve: Vec<f32>,
    /// Unproductive binary `[POI-PROTAC] + [PROTAC-E3]` saturation `%` causing the Hook Effect.
    pub binary_competition_curve: Vec<f32>,
}

/// Single assembled heterobifunctional `PROTAC` degrader candidate (`POI + Linker + E3`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtacCandidate {
    pub rank: usize,
    pub candidate_id: String,
    pub e3_kind: E3RecruiterKind,
    #[serde(skip_deserializing)]
    pub linker_code: &'static str,
    #[serde(skip_deserializing)]
    pub linker_name: &'static str,
    pub linker_class: LinkerClass,
    pub linker_atom_span: usize,
    pub linker_length_angstrom: f64,
    pub exit_vector_label: String,
    pub physchem: PhyschemProfile,
    pub logd_74: f64,
    pub sa_score: f64,
    pub chameleon: ChameleonicityProfile,
    pub ternary: TernaryHookProfile,
    /// Composite Oral Degrader Desirability Score `[0..100]` uniting ternary `TF50`/cooperativity,
    /// `AB-MPS`, `ePSA` chameleonic permeability, and synthetic tractability.
    pub degrader_score: f64,
    pub canonical_smiles: String,
    pub poi_atoms: Vec<usize>,
    pub linker_atoms: Vec<usize>,
    pub e3_atoms: Vec<usize>,
    pub mol: Molecule,
}

/// Complete `PROTAC` design, exit-vector, chameleonicity, and Hook-Effect report for a `POI` lead.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtacDesignReport {
    pub poi_name: String,
    pub poi_kd_nm: f64,
    pub poi_conc_nm: f64,
    pub e3_conc_nm: f64,
    pub exit_vectors: Vec<PoiExitVector>,
    pub active_exit_vector_idx: usize,
    pub candidates: Vec<ProtacCandidate>,
}

struct LinkerTemplateSpec {
    code: &'static str,
    name: &'static str,
    class: LinkerClass,
    /// SMILES with `Cl` marking the `POI`-side attachment dummy and `I` marking the `E3`-side dummy!
    dummy_smiles: &'static str,
    span_bonds: usize,
    approx_span_angstrom: f64,
    hinge_flexibility: f64, // [0.0 rigid rod .. 1.0 flexible PEG/alkyl] enabling dIMHB folding
}

const PROTAC_LINKER_LIBRARY: [LinkerTemplateSpec; 10] = [
    LinkerTemplateSpec {
        code: "peg2",
        name: "PEG2 (-CH2CH2OCH2CH2OCH2-)",
        class: LinkerClass::FlexiblePeg,
        dummy_smiles: "ClCCOCCOCCI",
        span_bonds: 8,
        approx_span_angstrom: 9.4,
        hinge_flexibility: 0.95,
    },
    LinkerTemplateSpec {
        code: "peg3",
        name: "PEG3 (MZ1-Class Triethylene Glycol)",
        class: LinkerClass::FlexiblePeg,
        dummy_smiles: "ClCCOCCOCCOCCI",
        span_bonds: 11,
        approx_span_angstrom: 12.6,
        hinge_flexibility: 1.00,
    },
    LinkerTemplateSpec {
        code: "alkyl_c5",
        name: "Pentylene C5 (-(CH2)5-)",
        class: LinkerClass::FlexibleAlkyl,
        dummy_smiles: "ClCCCCCI",
        span_bonds: 6,
        approx_span_angstrom: 7.4,
        hinge_flexibility: 0.80,
    },
    LinkerTemplateSpec {
        code: "alkyl_c8",
        name: "Octylene C8 (-(CH2)8-)",
        class: LinkerClass::FlexibleAlkyl,
        dummy_smiles: "ClCCCCCCCCI",
        span_bonds: 9,
        approx_span_angstrom: 10.9,
        hinge_flexibility: 0.88,
    },
    LinkerTemplateSpec {
        code: "pip_c2",
        name: "Piperazine-Acetyl (-CH2C(=O)-Pip-CH2-)",
        class: LinkerClass::SemiRigidHeterocycle,
        dummy_smiles: "ClCC(=O)N1CCN(CCI)CC1",
        span_bonds: 8,
        approx_span_angstrom: 8.8,
        hinge_flexibility: 0.58,
    },
    LinkerTemplateSpec {
        code: "pip_pip_arv",
        name: "Piperidinyl-Piperazine (ARV-110 Oral Core)",
        class: LinkerClass::RigidSpiroOrAlkyne,
        dummy_smiles: "ClCC1CCN(C2CCN(CCI)CC2)CC1",
        span_bonds: 9,
        approx_span_angstrom: 9.8,
        hinge_flexibility: 0.38,
    },
    LinkerTemplateSpec {
        code: "spiro_3_3",
        name: "2,6-Diazaspiro[3.3]heptane (3D Oral Spiro)",
        class: LinkerClass::RigidSpiroOrAlkyne,
        dummy_smiles: "ClCC(=O)N1CC2(C1)CN(CC(=O)I)C2",
        span_bonds: 8,
        approx_span_angstrom: 8.4,
        hinge_flexibility: 0.42,
    },
    LinkerTemplateSpec {
        code: "alkyne_peg1",
        name: "Propargyl-PEG1 (-CH2-C#C-CH2OCH2CH2-)",
        class: LinkerClass::SemiRigidHeterocycle,
        dummy_smiles: "ClCC#CCOCCI",
        span_bonds: 7,
        approx_span_angstrom: 8.6,
        hinge_flexibility: 0.64,
    },
    LinkerTemplateSpec {
        code: "click_triazole",
        name: "1,4-Triazole-PEG1 (CuAAC Click Linker)",
        class: LinkerClass::SemiRigidHeterocycle,
        dummy_smiles: "ClCc1cn(CCOCCI)nn1",
        span_bonds: 9,
        approx_span_angstrom: 10.2,
        hinge_flexibility: 0.72,
    },
    LinkerTemplateSpec {
        code: "ethynyl_aryl",
        name: "4-Ethynylbenzyl Rigid Ruler (-C#C-Ph-CH2-)",
        class: LinkerClass::RigidSpiroOrAlkyne,
        dummy_smiles: "ClCC#Cc1ccc(CCI)cc1",
        span_bonds: 8,
        approx_span_angstrom: 9.6,
        hinge_flexibility: 0.22,
    },
];

/// Perceives and ranks solvent-exposed conjugation exit vectors on a `POI` small-molecule warhead.
pub fn detect_poi_exit_vectors(poi: &Molecule) -> Vec<PoiExitVector> {
    let n = poi.atoms.len();
    if n == 0 {
        return Vec::new();
    }

    // Distance from molecular 2D centroid as a fast topological periphery / rim proxy
    let inv_n = 1.0 / (n as f32);
    let cx: f32 = poi.atoms.iter().map(|a| a.pos2d[0]).sum::<f32>() * inv_n;
    let cy: f32 = poi.atoms.iter().map(|a| a.pos2d[1]).sum::<f32>() * inv_n;
    let radial = |u: usize| -> f64 {
        let dx = poi.atoms[u].pos2d[0] - cx;
        let dy = poi.atoms[u].pos2d[1] - cy;
        (dx.hypot(dy) as f64).clamp(0.5, 15.0)
    };

    let mut sites: Vec<PoiExitVector> = Vec::new();
    let mut seen_anchors: BTreeSet<usize> = BTreeSet::new();

    // 1. Saturated heterocyclic rim N/O or terminal aliphatic amine (`-NH-`, `-N(Me)-`, morpholine/piperazine rim)
    for (u, a) in poi.atoms.iter().enumerate() {
        if a.elem == Element::N && !a.aromatic {
            if a.total_h() >= 1 && poi.adj[u].len() <= 2 && seen_anchors.insert(u) {
                sites.push(PoiExitVector {
                    vector_idx: 0,
                    anchor_atom: u,
                    prune_atoms: Vec::new(),
                    needs_carbonyl_or_ch2_bridge: false,
                    label: format!("Solvent-Rim Amine N#{}", u + 1),
                    rationale: "Direct N-alkylation/acylation at solvent-exposed aliphatic NH handle",
                    exposure_score: 90.0 + radial(u),
                });
            } else if a.total_h() == 0 && poi.adj[u].len() == 3 {
                // Tertiary N-methyl (e.g. Afatinib / Chlorpromazine -N(CH3)2 -> demethylate one -CH3 for linker!)
                if let Some(&(me, _)) = poi.adj[u].iter().find(|&&(nb, _)| {
                    poi.atoms[nb].elem == Element::C && poi.adj[nb].len() == 1
                }) {
                    if seen_anchors.insert(u) {
                        sites.push(PoiExitVector {
                            vector_idx: 0,
                            anchor_atom: u,
                            prune_atoms: vec![me],
                            needs_carbonyl_or_ch2_bridge: false,
                            label: format!("Des-Methyl Tertiary Amine N#{}", u + 1),
                            rationale: "N-demethylative linker conjugation preserving basic tertiary amine",
                            exposure_score: 88.0 + radial(u),
                        });
                    }
                }
            }
        }
    }

    // 2. Terminal alkoxy `-O-CH3` or `-O-CH2CH3` or `-O-CH2CH2OCH3` (Gefitinib/Erlotinib C6/C7 quinazoline rim!)
    for (u, a) in poi.atoms.iter().enumerate() {
        if a.elem == Element::O && !a.in_ring && poi.adj[u].len() == 2 {
            let n0 = poi.adj[u][0].0;
            let n1 = poi.adj[u][1].0;
            for &(core_side, tail_side) in &[(n0, n1), (n1, n0)] {
                if poi.atoms[core_side].in_ring && !poi.atoms[tail_side].in_ring {
                    // Collect acyclic tail component beyond `u`
                    let mut tail = BTreeSet::new();
                    let mut q = VecDeque::new();
                    tail.insert(tail_side);
                    q.push_back(tail_side);
                    let mut hits_ring = false;
                    while let Some(cur) = q.pop_front() {
                        if poi.atoms[cur].in_ring {
                            hits_ring = true;
                            break;
                        }
                        for &(nb, _) in &poi.adj[cur] {
                            if nb != u && !tail.contains(&nb) {
                                tail.insert(nb);
                                q.push_back(nb);
                            }
                        }
                    }
                    if !hits_ring && tail.len() <= 6 && seen_anchors.insert(u) {
                        sites.push(PoiExitVector {
                            vector_idx: 0,
                            anchor_atom: u,
                            prune_atoms: tail.into_iter().collect(),
                            needs_carbonyl_or_ch2_bridge: false,
                            label: format!("Phenolic Ether Rim O#{}", u + 1),
                            rationale: "Replaces solvent-exposed alkoxy tail with phenolic PROTAC ether linker (Burslem EGFR strategy)",
                            exposure_score: 86.0 + radial(u),
                        });
                    }
                }
            }
        }
    }

    // 3. Primary sulfonamide `-SO2NH2` (Celecoxib / Valdecoxib coxib N-acylsulfonamide / N-alkyl exit vector!)
    for (u, a) in poi.atoms.iter().enumerate() {
        if a.elem == Element::N
            && a.total_h() >= 1
            && poi.adj[u].len() == 1
            && poi.atoms[poi.adj[u][0].0].elem == Element::S
            && seen_anchors.insert(u)
        {
            sites.push(PoiExitVector {
                vector_idx: 0,
                anchor_atom: u,
                prune_atoms: Vec::new(),
                needs_carbonyl_or_ch2_bridge: false,
                label: format!("Sulfonamide Rim N#{}", u + 1),
                rationale: "Conjugates linker at solvent-exposed sulfonamide NH2 handle",
                exposure_score: 84.0 + radial(u),
            });
        }
    }

    // 4. Terminal aryl halogen (`-Br`, `-Cl`, `-F`) or terminal alkyne (`-C#CH`)
    for (u, a) in poi.atoms.iter().enumerate() {
        if matches!(a.elem, Element::Br | Element::Cl | Element::I) && poi.adj[u].len() == 1 {
            let ar = poi.adj[u][0].0;
            if poi.atoms[ar].aromatic && seen_anchors.insert(ar) {
                sites.push(PoiExitVector {
                    vector_idx: 0,
                    anchor_atom: ar,
                    prune_atoms: vec![u],
                    needs_carbonyl_or_ch2_bridge: true,
                    label: format!("Aryl-{} Cross-Coupling C#{}", a.elem.symbol(), ar + 1),
                    rationale: "Sonogashira / Buchwald exit-vector conjugation at peripheral aryl halide",
                    exposure_score: 76.0 + radial(u),
                });
            }
        }
    }

    // Fallback: most peripheral C/N/O atom with available hydrogen or terminal methyl
    if sites.is_empty() {
        let best_u = (0..n)
            .filter(|&u| poi.atoms[u].total_h() > 0 || poi.adj[u].len() == 1)
            .max_by(|&a, &b| {
                radial(a)
                    .partial_cmp(&radial(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        sites.push(PoiExitVector {
            vector_idx: 0,
            anchor_atom: best_u,
            prune_atoms: Vec::new(),
            needs_carbonyl_or_ch2_bridge: poi.atoms[best_u].aromatic,
            label: format!(
                "Peripheral {}#{} Exit Vector",
                poi.atoms[best_u].elem.symbol(),
                best_u + 1
            ),
            rationale: "Outermost sterically accessible C-H / heteroatom functionalization vector",
            exposure_score: 70.0 + radial(best_u),
        });
    }

    sites.sort_by(|a, b| {
        b.exposure_score
            .partial_cmp(&a.exposure_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sites.truncate(4);
    for (i, s) in sites.iter_mut().enumerate() {
        s.vector_idx = i;
    }
    sites
}

/// Excises a single-bond dummy halogen (`target_dummy`) from `frag`, returning `(cleaned_mol, neighbor_anchor)`.
fn parse_and_excise_dummy(smiles: &str, dummy_elem: Element) -> Option<(Molecule, usize)> {
    let raw = parse_smiles(smiles).ok()?;
    let dummy_idx = raw.atoms.iter().position(|a| a.elem == dummy_elem)?;
    if raw.adj[dummy_idx].len() != 1 {
        return None;
    }
    let anchor_old = raw.adj[dummy_idx][0].0;
    let mut rem = BTreeSet::new();
    rem.insert(dummy_idx);
    prune_subgraph_keep_anchor(&raw, anchor_old, &rem)
}

/// Excises BOTH `Cl` (POI-end dummy) and `I` (E3-end dummy) from a linker template SMILES,
/// returning `(linker_mol, poi_end_atom, e3_end_atom)`.
fn parse_linker_two_dummies(smiles: &str) -> Option<(Molecule, usize, usize)> {
    let raw = parse_smiles(smiles).ok()?;
    let cl_idx = raw.atoms.iter().position(|a| a.elem == Element::Cl)?;
    let i_idx = raw.atoms.iter().position(|a| a.elem == Element::I)?;
    let left_old = raw.adj[cl_idx].first()?.0;
    let right_old = raw.adj[i_idx].first()?.0;

    let mut old_to_new = vec![None; raw.atoms.len()];
    let mut out = Molecule::new("");
    for (oi, a) in raw.atoms.iter().enumerate() {
        if oi == cl_idx || oi == i_idx {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == left_old || oi == right_old {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[oi] = Some(ni);
    }
    for b in &raw.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = out.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }
    Some((out, old_to_new[left_old]?, old_to_new[right_old]?))
}

fn prune_subgraph_keep_anchor(
    mol: &Molecule,
    anchor_old: usize,
    remove_set: &BTreeSet<usize>,
) -> Option<(Molecule, usize)> {
    if remove_set.contains(&anchor_old) {
        return None;
    }
    let mut old_to_new = vec![None; mol.atoms.len()];
    let mut out = Molecule::new("");
    for (oi, a) in mol.atoms.iter().enumerate() {
        if remove_set.contains(&oi) {
            continue;
        }
        let ni = out.atoms.len();
        let mut na = a.clone();
        na.idx = ni;
        if oi == anchor_old {
            na.explicit_h = 0;
            na.bracketed = na.charge != 0;
        }
        out.atoms.push(na);
        old_to_new[oi] = Some(ni);
    }
    for b in &mol.bonds {
        if let (Some(ua), Some(vb)) = (old_to_new[b.a], old_to_new[b.b]) {
            let bi = out.bonds.len();
            let mut nb = b.clone();
            nb.idx = bi;
            nb.a = ua;
            nb.b = vb;
            out.bonds.push(nb);
        }
    }
    Some((out, old_to_new[anchor_old]?))
}

fn append_fragment(dst: &mut Molecule, src: &Molecule) -> Vec<usize> {
    let offset = dst.atoms.len();
    let mut mapped = Vec::with_capacity(src.atoms.len());
    for (i, a) in src.atoms.iter().enumerate() {
        let mut na = a.clone();
        na.idx = offset + i;
        dst.atoms.push(na);
        mapped.push(offset + i);
    }
    for b in &src.bonds {
        let bi = dst.bonds.len();
        let mut nb = b.clone();
        nb.idx = bi;
        nb.a = offset + b.a;
        nb.b = offset + b.b;
        dst.bonds.push(nb);
    }
    mapped
}

fn add_single_bond(mol: &mut Molecule, a: usize, b: usize) {
    let bi = mol.bonds.len();
    mol.bonds.push(Bond {
        idx: bi,
        a,
        b,
        order: BondOrder::Single,
        aromatic: false,
        in_ring: false,
        kekule_order: 1,
    });
}

/// Assembles a full heterobifunctional `PROTAC` molecule (`POI + Linker + E3`) and returns
/// `(chimera_mol, poi_atom_indices, linker_atom_indices, e3_atom_indices)`.
fn assemble_protac_chimera(
    poi: &Molecule,
    ev: &PoiExitVector,
    linker: &LinkerTemplateSpec,
    e3: E3RecruiterKind,
) -> Option<(Molecule, Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rem_set: BTreeSet<usize> = ev.prune_atoms.iter().copied().collect();
    let (mut chimera, poi_anchor) = prune_subgraph_keep_anchor(poi, ev.anchor_atom, &rem_set)?;
    let poi_atoms: Vec<usize> = (0..chimera.atoms.len()).collect();

    // Optional -NH- bridge when grafting onto bare aromatic carbon so chemistry is an aniline/ether
    let mut linker_atoms = Vec::new();
    let eff_poi_anchor = if ev.needs_carbonyl_or_ch2_bridge {
        let nh_idx = chimera.atoms.len();
        chimera.atoms.push(Atom {
            idx: nh_idx,
            elem: Element::N,
            charge: 0,
            aromatic: false,
            bracketed: false,
            explicit_h: 0,
            implicit_h: 0,
            chiral: 0,
            in_ring: false,
            ring_sizes: Vec::new(),
            hyb: Hybridization::Sp3,
            pos2d: [0.0, 0.0],
        });
        add_single_bond(&mut chimera, poi_anchor, nh_idx);
        linker_atoms.push(nh_idx);
        nh_idx
    } else {
        poi_anchor
    };

    let (lnk_mol, lnk_left, lnk_right) = parse_linker_two_dummies(linker.dummy_smiles)?;
    let lnk_offset = chimera.atoms.len();
    let mapped_lnk = append_fragment(&mut chimera, &lnk_mol);
    linker_atoms.extend_from_slice(&mapped_lnk);
    add_single_bond(&mut chimera, eff_poi_anchor, lnk_offset + lnk_left);

    let (e3_mol, e3_anchor) = parse_and_excise_dummy(e3.template_smiles_with_dummy(), Element::Br)?;
    let e3_offset = chimera.atoms.len();
    let e3_atoms = append_fragment(&mut chimera, &e3_mol);
    add_single_bond(&mut chimera, lnk_offset + lnk_right, e3_offset + e3_anchor);

    chimera.sanitize();
    chimera.smiles = molecule_to_smiles(&chimera);
    compute_2d_coords(&mut chimera);

    Some((chimera, poi_atoms, linker_atoms, e3_atoms))
}

/// Evaluates DeGoey AbbVie `AB-MPS` and Caron/Ermondi dynamic `dIMHB` chameleonicity on an assembled `PROTAC`.
fn evaluate_bro5_chameleonicity(
    mol: &Molecule,
    prof: &PhyschemProfile,
    logd_74: f64,
    hinge_flex: f64,
) -> ChameleonicityProfile {
    let n = mol.atoms.len();
    // DeGoey et al. (2018) AbbVie Multiparametric Score: AB-MPS = |cLogD - 3| + NAR + NRotB
    let abbv_mps = (logd_74 - 3.0).abs()
        + (prof.aromatic_rings as f64)
        + (prof.rotatable_bonds as f64);

    // Topological all-pairs shortest path to count 5..=14 bond dynamic intramolecular H-bond pairs
    let mut donors = Vec::new();
    let mut acceptors = Vec::new();
    for (u, a) in mol.atoms.iter().enumerate() {
        if matches!(a.elem, Element::N | Element::O) && a.total_h() > 0 {
            donors.push(u);
        }
        if matches!(a.elem, Element::O | Element::N) && a.charge <= 0 {
            acceptors.push(u);
        }
    }

    let mut raw_dimhb_pairs = 0usize;
    let mut used_donors = BTreeSet::new();
    for &d in &donors {
        let mut dist = vec![usize::MAX; n];
        let mut q = VecDeque::new();
        dist[d] = 0;
        q.push_back(d);
        while let Some(u) = q.pop_front() {
            if dist[u] >= 14 {
                continue;
            }
            for &(nb, _) in &mol.adj[u] {
                if dist[nb] == usize::MAX {
                    dist[nb] = dist[u] + 1;
                    q.push_back(nb);
                }
            }
        }
        // Pseudo-ring size 6..=15 (topological bond separation 5..=14) forms dynamic folded IMHBs
        if acceptors
            .iter()
            .any(|&acc| acc != d && (5..=14).contains(&dist[acc]))
            && used_donors.insert(d)
        {
            raw_dimhb_pairs += 1;
        }
    }

    // Effective dynamic IMHBs scale with linker conformational folding capacity
    let eff_dimhb_f = (raw_dimhb_pairs as f64) * (0.45 + 0.55 * hinge_flex);
    let dimhb_count = eff_dimhb_f.round() as usize;

    // Masking of TPSA -> 3D membrane-folded ePSA (Ermondi & Caron 2020-2024)
    let steric_collapse_shield = prof.tpsa * (0.10 + 0.16 * hinge_flex);
    let imhb_shield = (dimhb_count as f64) * 16.5;
    let max_mask = prof.tpsa * 0.48;
    let delta_psa = (steric_collapse_shield + imhb_shield).min(max_mask);
    let epsa_folded = (prof.tpsa - delta_psa).max(35.0);

    // Chameleonicity index in [0..1] and ChamelogD lipophilicity gain upon folding
    let chameleonicity_index =
        ((delta_psa / 95.0) * 0.65 + (eff_dimhb_f / 5.0).min(1.0) * 0.35).clamp(0.05, 0.99);
    let chamelog_delta = (delta_psa * 0.024).clamp(0.15, 2.40);

    // Extended bRo5 limits (Doak / Pike / DeGoey oral degrader thresholds)
    let mut bro5_viol = 0usize;
    if prof.mw > 1000.0 {
        bro5_viol += 1;
    }
    if prof.hbd > 5 {
        bro5_viol += 1;
    }
    if epsa_folded > 165.0 {
        bro5_viol += 1;
    }
    if prof.rotatable_bonds > 20 {
        bro5_viol += 1;
    }
    if logd_74 < -1.0 || logd_74 > 7.5 {
        bro5_viol += 1;
    }

    let permeability_class =
        if prof.mw <= 960.0 && prof.hbd <= 4 && epsa_folded <= 142.0 && abbv_mps <= 16.5 {
            Bro5PermeabilityClass::OralBro5SweetSpot
        } else if epsa_folded <= 165.0 && chameleonicity_index >= 0.42 && prof.hbd <= 5 {
            Bro5PermeabilityClass::ChameleonicPermeable
        } else if bro5_viol <= 2 {
            Bro5PermeabilityClass::CellularToolDegrader
        } else {
            Bro5PermeabilityClass::PermeabilityLimited
        };

    ChameleonicityProfile {
        abbv_mps,
        dimhb_count,
        tpsa_open_a2: prof.tpsa,
        epsa_folded_a2: epsa_folded,
        delta_psa_masked_a2: delta_psa,
        chameleonicity_index,
        chamelog_delta,
        bro5_violations: bro5_viol,
        permeability_class,
    }
}

/// Solves Douglass et al. (2013) exact three-body equilibria (`POI + PROTAC + E3 <-> POI·PROTAC·E3`)
/// across 33 logarithmic `[PROTAC]` concentrations (`0.01 nM ..= 100,000 nM`).
pub fn solve_three_body_hook_curve(
    poi_kd_nm: f64,
    e3_kd_nm: f64,
    cooperativity_alpha: f64,
    poi_total_nm: f64,
    e3_total_nm: f64,
) -> TernaryHookProfile {
    let kp = poi_kd_nm.max(0.02);
    let ke = e3_kd_nm.max(1.0);
    let alpha = cooperativity_alpha.clamp(0.05, 50.0);
    let pt = poi_total_nm.max(1.0);
    let et = e3_total_nm.max(1.0);
    let limiting = pt.min(et);

    let mut dose_grid_nm = Vec::with_capacity(HOOK_CURVE_LEN);
    let mut ternary_pct_curve = Vec::with_capacity(HOOK_CURVE_LEN);
    let mut binary_competition_curve = Vec::with_capacity(HOOK_CURVE_LEN);

    let mut peak_frac = 0.0f64;
    let mut peak_idx = 0usize;

    // 33 log-spaced doses: 10^(-2.0 + k * 0.21875) nM -> [0.01 nM .. 100,000 nM]
    for step in 0..HOOK_CURVE_LEN {
        let log_c = -2.0 + (step as f64) * (7.0 / ((HOOK_CURVE_LEN - 1) as f64));
        let bt = 10.0f64.powf(log_c);
        dose_grid_nm.push(bt as f32);

        // Coupled mass-balance fixed-point / damped solver for free [P], [E], [B]:
        // [B] = bt / (1 + [P]/kp + [E]/ke + alpha*[P]*[E]/(kp*ke))
        // [P] = pt / (1 + [B]/kp * (1 + alpha*[E]/ke))
        // [E] = et / (1 + [B]/ke * (1 + alpha*[P]/kp))
        let mut p_free = pt * 0.5;
        let mut e_free = et * 0.5;
        let mut b_free = bt * 0.5;

        for _it in 0..45 {
            let b_next = bt
                / (1.0
                    + p_free / kp
                    + e_free / ke
                    + alpha * p_free * e_free / (kp * ke));
            let p_next = pt / (1.0 + (b_next / kp) * (1.0 + alpha * e_free / ke));
            let e_next = et / (1.0 + (b_next / ke) * (1.0 + alpha * p_next / kp));

            let err = (b_next - b_free).abs() + (p_next - p_free).abs() + (e_next - e_free).abs();
            b_free = 0.35 * b_free + 0.65 * b_next;
            p_free = 0.35 * p_free + 0.65 * p_next;
            e_free = 0.35 * e_free + 0.65 * e_next;
            if err < 1e-8 {
                break;
            }
        }

        let pbe = alpha * p_free * b_free * e_free / (kp * ke);
        let pb = p_free * b_free / kp;
        let be = b_free * e_free / ke;

        let frac = (pbe / limiting).clamp(0.0, 1.0);
        if frac > peak_frac {
            peak_frac = frac;
            peak_idx = step;
        }
        ternary_pct_curve.push((frac * 100.0) as f32);

        // Unproductive binary saturation of POI + E3
        let bin_sat = ((pb / pt + be / et) * 0.5 * 100.0).clamp(0.0, 100.0);
        binary_competition_curve.push(bin_sat as f32);
    }

    let optimal_protac_nm = dose_grid_nm[peak_idx] as f64;
    let half_peak_pct = (peak_frac * 50.0) as f32;

    // Interpolate TF50 (ascending limb) and TI50 (descending Hook-Effect limb)
    let mut tf50_nm = dose_grid_nm[0] as f64;
    for i in 1..=peak_idx {
        let y0 = ternary_pct_curve[i - 1];
        let y1 = ternary_pct_curve[i];
        if y0 <= half_peak_pct && y1 >= half_peak_pct {
            let t = ((half_peak_pct - y0) / (y1 - y0).max(1e-6)).clamp(0.0, 1.0) as f64;
            let l0 = (dose_grid_nm[i - 1] as f64).log10();
            let l1 = (dose_grid_nm[i] as f64).log10();
            tf50_nm = 10.0f64.powf(l0 + t * (l1 - l0));
            break;
        }
    }

    let mut ti50_nm = *dose_grid_nm.last().unwrap_or(&100_000.0) as f64;
    for i in (peak_idx + 1)..HOOK_CURVE_LEN {
        let y0 = ternary_pct_curve[i - 1];
        let y1 = ternary_pct_curve[i];
        if y0 >= half_peak_pct && y1 <= half_peak_pct {
            let t = ((y0 - half_peak_pct) / (y0 - y1).max(1e-6)).clamp(0.0, 1.0) as f64;
            let l0 = (dose_grid_nm[i - 1] as f64).log10();
            let l1 = (dose_grid_nm[i] as f64).log10();
            ti50_nm = 10.0f64.powf(l0 + t * (l1 - l0));
            break;
        }
    }

    let hook_window_ratio = (ti50_nm / tf50_nm.max(0.01)).max(1.0);

    TernaryHookProfile {
        cooperativity_alpha: alpha,
        max_ternary_fraction: peak_frac,
        optimal_protac_nm,
        tf50_nm,
        ti50_nm,
        hook_window_ratio,
        dose_grid_nm,
        ternary_pct_curve,
        binary_competition_curve,
    }
}

/// Enumerates and multiparameter-ranks `PROTAC` degraders across `selected_e3s` and all 10 linker
/// architectures at `exit_vector_choice` (`0..exit_vectors.len()`).
pub fn design_protac_series(
    poi: &Molecule,
    selected_e3s: &[E3RecruiterKind],
    exit_vector_choice: usize,
    poi_conc_nm: f64,
    e3_conc_nm: f64,
) -> ProtacDesignReport {
    let exit_vectors = detect_poi_exit_vectors(poi);
    let active_ev_idx = exit_vector_choice.min(exit_vectors.len().saturating_sub(1));
    let poi_kd_nm = (10.0f64.powf(9.0 - poi.pact.clamp(4.5, 10.5))).clamp(0.05, 25_000.0);

    let e3_list: &[E3RecruiterKind] = if selected_e3s.is_empty() {
        &E3RecruiterKind::ALL
    } else {
        selected_e3s
    };

    let mut candidates: Vec<ProtacCandidate> = Vec::new();
    let Some(ev) = exit_vectors.get(active_ev_idx) else {
        return ProtacDesignReport {
            poi_name: poi.name.clone(),
            poi_kd_nm,
            poi_conc_nm,
            e3_conc_nm,
            exit_vectors,
            active_exit_vector_idx: 0,
            candidates,
        };
    };

    for &e3 in e3_list {
        for lnk in &PROTAC_LINKER_LIBRARY {
            let Some((mut mol, poi_atoms, linker_atoms, e3_atoms)) =
                assemble_protac_chimera(poi, ev, lnk, e3)
            else {
                continue;
            };
            let cand_id = format!("{}-{}-{}", poi.id, lnk.code, e3.code());
            mol.id = cand_id.clone();
            mol.name = format!("{} + {} + {}", poi.name, lnk.code, e3.ligase_family());
            mol.target = format!("{} -> {}", poi.target, e3.ligase_family());

            let alerts = scan_structural_alerts(&mol);
            let prof = evaluate_physchem(&mol, alerts.len());
            let pka = evaluate_physiological_pka_74(&mol);
            let sa = evaluate_synthetic_accessibility(&mol);
            let cham = evaluate_bro5_chameleonicity(
                &mol,
                &prof,
                pka.microstate_logd_74,
                lnk.hinge_flexibility,
            );

            // Geometry-tuned ternary cooperativity alpha:
            // Optimal POI-E3 linker span ~ 8.4 - 10.5 A avoids steric POI/E3 clash (< 7.5 A)
            // while minimizing conformational entropic loss (> 11.5 A); rigid oral cores lock active pose!
            let span_opt_dist = (lnk.approx_span_angstrom - 9.3).abs();
            let geom_factor = (-0.24 * span_opt_dist * span_opt_dist).exp();
            let rigidity_entropy_bonus = 1.0 + 0.55 * (1.0 - lnk.hinge_flexibility);
            let alpha = (2.8 * e3.intrinsic_ppi_factor() * geom_factor * rigidity_entropy_bonus)
                .clamp(0.45, 18.0);

            let ternary = solve_three_body_hook_curve(
                poi_kd_nm,
                e3.binary_kd_nm(),
                alpha,
                poi_conc_nm,
                e3_conc_nm,
            );

            // Composite Oral Degrader Score [0..100]:
            // 1. Ternary peak efficacy & sub-10 nM TF50 potency (0..38)
            let tf50_pts = ((2.5 - ternary.tf50_nm.max(0.05).log10()) * 8.5).clamp(0.0, 24.0);
            let tern_pts = ternary.max_ternary_fraction * 14.0 + tf50_pts;
            // 2. Oral bRo5 AB-MPS & ePSA chameleonic permeability (0..42)
            let mps_pts = ((24.0 - cham.abbv_mps) * 1.65).clamp(0.0, 22.0);
            let epsa_pts = ((185.0 - cham.epsa_folded_a2) * 0.22).clamp(0.0, 14.0)
                + cham.chameleonicity_index * 6.0;
            // 3. Synthetic tractability & HBD restraint (0..20)
            let synth_pts = ((7.5 - sa.sa_score) * 2.5).clamp(0.0, 12.0)
                + if prof.hbd <= 4 { 8.0 } else { 2.0 };
            let degrader_score = (tern_pts + mps_pts + epsa_pts + synth_pts).clamp(2.0, 99.5);

            candidates.push(ProtacCandidate {
                rank: 0,
                candidate_id: cand_id,
                e3_kind: e3,
                linker_code: lnk.code,
                linker_name: lnk.name,
                linker_class: lnk.class,
                linker_atom_span: lnk.span_bonds,
                linker_length_angstrom: lnk.approx_span_angstrom,
                exit_vector_label: ev.label.clone(),
                physchem: prof,
                logd_74: pka.microstate_logd_74,
                sa_score: sa.sa_score,
                chameleon: cham,
                ternary,
                degrader_score,
                canonical_smiles: mol.smiles.clone(),
                poi_atoms,
                linker_atoms,
                e3_atoms,
                mol,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.degrader_score
            .partial_cmp(&a.degrader_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.rank = i + 1;
    }

    ProtacDesignReport {
        poi_name: poi.name.clone(),
        poi_kd_nm,
        poi_conc_nm,
        e3_conc_nm,
        exit_vectors,
        active_exit_vector_idx: active_ev_idx,
        candidates,
    }
}

/// Exports ranked `PROTAC` chimeras (`CRBN`/`VHL` + linkers + `bRo5` chameleonicity + Hook-Effect critical points) to CSV.
pub fn export_protac_candidates_csv(cands: &[ProtacCandidate]) -> String {
    let mut out = String::from(
        "rank,id,e3_recruiter,linker_code,linker_name,exit_vector,mw,logd_74,hbd,hba,nrotb,tpsa_open_a2,epsa_folded_a2,dimhb_count,chameleonicity_idx,abbv_mps,bro5_class,cooperativity_alpha,max_ternary_pct,tf50_nm,optimal_protac_nm,ti50_hook_nm,hook_window_ratio,sa_score,degrader_score,smiles\n",
    );
    for c in cands {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{:.1},{:.2},{},{},{},{:.1},{:.1},{},{:.3},{:.1},\"{}\",{:.2},{:.1},{:.2},{:.1},{:.0},{:.0},{:.2},{:.1},\"{}\"\n",
            c.rank,
            c.candidate_id,
            c.e3_kind.name(),
            c.linker_code,
            c.linker_name,
            c.exit_vector_label,
            c.physchem.mw,
            c.logd_74,
            c.physchem.hbd,
            c.physchem.hba,
            c.physchem.rotatable_bonds,
            c.chameleon.tpsa_open_a2,
            c.chameleon.epsa_folded_a2,
            c.chameleon.dimhb_count,
            c.chameleon.chameleonicity_index,
            c.chameleon.abbv_mps,
            c.chameleon.permeability_class.label(),
            c.ternary.cooperativity_alpha,
            c.ternary.max_ternary_fraction * 100.0,
            c.ternary.tf50_nm,
            c.ternary.optimal_protac_nm,
            c.ternary.ti50_nm,
            c.ternary.hook_window_ratio,
            c.sa_score,
            c.degrader_score,
            c.canonical_smiles
        ));
    }
    out
}
