//! Interactive 3D Tactile Ligand & Side-Chain Tug-and-Wiggle Force-Field Sculptor, Harmonic
//! Rubber-Band H-Bond Restraint Engine, `Blue-Fuse` Barrier-Crossing Optimizer, In-Pocket R-Group
//! Sprouter, and Timed Med-Chem Challenge Puzzle Suite (`foldit_sculpt.rs`, `v3.6.0`).
//!
//! Primary References:
//! - Cooper, Khatib, Treuille, Barbero, Lee, Beenen, Leaver-Fay, Baker & Popovic (2010), *Nature*
//!   466(7307), 756-760 ("Predicting protein structures with a multiplayer online game" — direct
//!   3D atom pulling, soft harmonic `"rubber bands"`, freezing degrees of freedom, adjustable
//!   steric repulsion, `"shake"` rotamer packing, and `"wiggle"` gradient minimization:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC2956414/`)
//! - Khatib, DiMaio, Cooper, Kazmierczyk, Gilski, Krzywda, Zabranska, Pichova, Thompson, Popovic,
//!   Jaskolski & Baker (2011), *Nat. Struct. Mol. Biol.* 18(10), 1175-1177 ("Crystal structure of a
//!   monomeric retroviral protease solved by protein folding game players" — human-guided barrier
//!   crossing & soften-wiggle-shake-harden macros:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC3705907/`)
//! - Kleffner, Flatten, Leaver-Fay, Baker, Siegel, Khatib & Cooper (2017), *Bioinformatics* 33(17),
//!   2765-2767 ("Foldit Standalone: a video game-derived protein structure manipulation interface
//!   using Rosetta": `https://pmc.ncbi.nlm.nih.gov/articles/PMC5860063/`)
//! - Drugit Small-Molecule Design Mode in Foldit (2025) ("Drugit: crowd-sourcing molecular design
//!   of non-peptidic VHL binders" — `PDB 5NVX` `VHL` E3 ligase puzzle rounds, `cart_bonded`
//!   internal ligand geometry strain, and multifactorial `TPSA`, `HBD`, and `cLogP` bonuses:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC11997059/`)
//! - Croll (2018), *Acta Crystallogr. D Struct. Biol.* 74(Pt 6), 519-530 ("ISOLDE: a physically
//!   realistic environment for model building into low-resolution electron-density maps" — live
//!   mouse-tugged harmonic position/distance restraints with explicit nonbonded vdW + electrostatics:
//!   `https://pmc.ncbi.nlm.nih.gov/articles/PMC6096486/`).

use crate::alerts::scan_structural_alerts;
use crate::allosteric_pocket::curated_allosteric_pdb_text;
use crate::mol::{Element, Hybridization, Molecule};
use crate::mol_io::parse_smiles;
use crate::physchem::evaluate_physchem;
use crate::pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, parse_pdb_structure, score_ligand_in_pocket,
    ParsedPdbStructure,
};
use crate::sketcher::molecule_to_smiles;
use crate::synth_access::evaluate_synthetic_accessibility;
use crate::watermap_solvation::curated_factor_xa_1ezq_pdb_text;
use serde::{Deserialize, Serialize};

/// Baseline Foldit/Drugit score offset (`8,000 pts` corresponds to a neutral `0.0 kcal/mol` pose; each `-1.0 kcal/mol` physical + med-chem gain adds `+260 pts`).
pub const FOLDIT_BASE_SCORE_PTS: f64 = 8_000.0;

/// Points awarded per `-1.0 kcal/mol` improvement in pure physical (unbanded) complex + med-chem objective energy.
pub const FOLDIT_PTS_PER_KCAL: f64 = 260.0;

/// Curated Foldit/Drugit medicinal chemistry challenge puzzles + freeform `.pdb` sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SculptPuzzlePreset {
    /// `Puzzle 1` (`PDB 5NVX` `VHL` E3 Ligase — Drugit 2025 benchmark): Satisfy `Tyr98` + `His115` H-bonds while keeping `TPSA <= 115 A^2`, `HBD <= 3`, and `cLogP 1.0..4.2`.
    Vhl5nvxDepeptidizationChallenge,
    /// `Puzzle 2` (`PDB 1M17` `EGFR` Gatekeeper Barrier): Soften repulsion, pull quinazoline past `Thr790`/`Met766`, `Shake` side chains, and lock the `Met793` hinge H-bond.
    Egfr1m17GatekeeperBarrierChallenge,
    /// `Puzzle 3` (`PDB 1EZQ` `Factor Xa` Unhappy Water Crush): Expel S1 `Tyr228` `HOH2012` hotspot (`dG = +2.6 kcal/mol`) via in-pocket `-Cl` / `-CN` sprouting + `Wiggle`.
    FactorXa1ezqVoidAndWaterCrusher,
    /// `Puzzle 4` (`PDB 3LN1` `COX-2` Selectivity Latch): Repack `Arg513`/`Val509` side chains (`Shake`) and sculpt ligand into the dewetted hydrophobic channel.
    Cox23ln1SelectivityLatchChallenge,
    /// `Puzzle 5` (`PDB 6OIM` `KRAS G12C` Switch-II Groove): Open `His95`/`Tyr96`/`Gln99` cryptic groove via rubber-band pull + `Blue-Fuse` cycle.
    Kras6oimSwitchIiCrypticPuzzle,
    /// Unlimited-time freeform 3D sculpting sandbox for any custom `.pdb` + dataset or 2D Sketcher molecule.
    CustomPdbFreeformSandbox,
}

impl SculptPuzzlePreset {
    pub const ALL: [Self; 6] = [
        Self::Vhl5nvxDepeptidizationChallenge,
        Self::Egfr1m17GatekeeperBarrierChallenge,
        Self::FactorXa1ezqVoidAndWaterCrusher,
        Self::Cox23ln1SelectivityLatchChallenge,
        Self::Kras6oimSwitchIiCrypticPuzzle,
        Self::CustomPdbFreeformSandbox,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => "PUZ-01-VHL-5NVX",
            Self::Egfr1m17GatekeeperBarrierChallenge => "PUZ-02-EGFR-1M17",
            Self::FactorXa1ezqVoidAndWaterCrusher => "PUZ-03-FXA-1EZQ",
            Self::Cox23ln1SelectivityLatchChallenge => "PUZ-04-COX2-3LN1",
            Self::Kras6oimSwitchIiCrypticPuzzle => "PUZ-05-KRAS-6OIM",
            Self::CustomPdbFreeformSandbox => "SANDBOX-CUSTOM-PDB",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => "P1: VHL 5NVX (Drugit)",
            Self::Egfr1m17GatekeeperBarrierChallenge => "P2: EGFR 1M17 (Gatekeeper)",
            Self::FactorXa1ezqVoidAndWaterCrusher => "P3: FXa 1EZQ (Water Crush)",
            Self::Cox23ln1SelectivityLatchChallenge => "P4: COX-2 3LN1 (Latch)",
            Self::Kras6oimSwitchIiCrypticPuzzle => "P5: KRAS 6OIM (Switch-II)",
            Self::CustomPdbFreeformSandbox => "Sandbox (.PDB)",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => {
                "Puzzle 1 — VHL E3 Ligase (5NVX): Non-Peptidic Tyr98/His115 Anchor & Oral TPSA Budget"
            }
            Self::Egfr1m17GatekeeperBarrierChallenge => {
                "Puzzle 2 — EGFR Kinase (1M17): Thr790 Gatekeeper Barrier Crossing & Met793 Hinge Lock"
            }
            Self::FactorXa1ezqVoidAndWaterCrusher => {
                "Puzzle 3 — Factor Xa (1EZQ): S1 Tyr228 Unhappy-Water Displacement & S4 Box Packing"
            }
            Self::Cox23ln1SelectivityLatchChallenge => {
                "Puzzle 4 — COX-2 (3LN1): Arg513/Val509 Side-Pocket Shake & Dewetted Channel Fit"
            }
            Self::Kras6oimSwitchIiCrypticPuzzle => {
                "Puzzle 5 — KRAS G12C (6OIM): Switch-II His95/Gln99 Cryptic Groove Remodeling"
            }
            Self::CustomPdbFreeformSandbox => {
                "Freeform 3D Force-Field Sculptor Sandbox (Custom .PDB + Dataset or Sketcher Lead)"
            }
        }
    }

    pub fn starter_smiles(self) -> &'static str {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => "CC1=C(C2=CC=C(CNC(=O)C3CC(O)CN3C(=O)C(C)(C)C)C=C2)SC=N1",
            Self::Egfr1m17GatekeeperBarrierChallenge => "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOC",
            Self::FactorXa1ezqVoidAndWaterCrusher => "COc1ccc(C(=O)Nc2ccc(N3CCCCC3=O)cc2)cc1",
            Self::Cox23ln1SelectivityLatchChallenge => "CS(=O)(=O)c1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(C)cc2)cc1",
            Self::Kras6oimSwitchIiCrypticPuzzle => "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C)c3cc(F)ccc23)CC1",
            Self::CustomPdbFreeformSandbox => "",
        }
    }

    pub fn time_budget_secs(self) -> f64 {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => 180.0,
            Self::Egfr1m17GatekeeperBarrierChallenge => 150.0,
            Self::FactorXa1ezqVoidAndWaterCrusher => 180.0,
            Self::Cox23ln1SelectivityLatchChallenge => 160.0,
            Self::Kras6oimSwitchIiCrypticPuzzle => 200.0,
            Self::CustomPdbFreeformSandbox => 0.0,
        }
    }

    /// `[Bronze, Silver, Gold, Platinum]` target Foldit point thresholds.
    pub fn medal_thresholds(self) -> [f64; 4] {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => [8_850.0, 9_350.0, 9_850.0, 10_350.0],
            Self::Egfr1m17GatekeeperBarrierChallenge => [8_900.0, 9_450.0, 9_950.0, 10_450.0],
            Self::FactorXa1ezqVoidAndWaterCrusher => [8_800.0, 9_300.0, 9_800.0, 10_300.0],
            Self::Cox23ln1SelectivityLatchChallenge => [8_950.0, 9_450.0, 9_950.0, 10_450.0],
            Self::Kras6oimSwitchIiCrypticPuzzle => [8_900.0, 9_400.0, 9_900.0, 10_400.0],
            Self::CustomPdbFreeformSandbox => [8_800.0, 9_300.0, 9_800.0, 10_300.0],
        }
    }

    pub fn max_tpsa_target(self) -> f64 {
        match self {
            Self::Vhl5nvxDepeptidizationChallenge => 118.0,
            _ => 132.0,
        }
    }

    pub fn max_clogp_target(self) -> f64 {
        match self {
            Self::Cox23ln1SelectivityLatchChallenge => 5.0,
            _ => 4.5,
        }
    }
}

/// Interactive 3D canvas mouse tool mode in `FolditSculptStudio`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SculptMouseTool {
    /// Drag to rotate 3D camera; click atom/residue to inspect.
    OrbitCamera,
    /// Click and drag any unfrozen ligand atom in 3D against the live force field (`ISOLDE` / `Foldit` Pull).
    TugAtom3d,
    /// Click a ligand atom then click a receptor polar anchor or water hotspot to attach a harmonic rubber band.
    DrawRubberBand,
    /// Click a ligand atom or flexible side chain to toggle its frozen pin lock.
    ToggleFreezePin,
    /// Click a flexible pocket side-chain tip to manually cycle its 4 `Dunbrack` rotamer states.
    CycleSideChainRotamer,
}

impl SculptMouseTool {
    pub const ALL: [Self; 5] = [
        Self::OrbitCamera,
        Self::TugAtom3d,
        Self::DrawRubberBand,
        Self::ToggleFreezePin,
        Self::CycleSideChainRotamer,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::OrbitCamera => "1: Orbit 3D",
            Self::TugAtom3d => "2: Tug Atom (Drag)",
            Self::DrawRubberBand => "3: Rubber Band",
            Self::ToggleFreezePin => "4: Freeze / Pin",
            Self::CycleSideChainRotamer => "5: Cycle Rotamer",
        }
    }
}

/// Achievement medal tier earned on a challenge puzzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SculptMedalTier {
    Unranked,
    Bronze,
    Silver,
    Gold,
    PlatinumVoidCrusher,
}

impl SculptMedalTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unranked => "Unranked (Keep Sculpting)",
            Self::Bronze => "BRONZE MEDAL",
            Self::Silver => "SILVER MEDAL",
            Self::Gold => "GOLD MEDAL",
            Self::PlatinumVoidCrusher => "PLATINUM VOID-CRUSHER",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Unranked => [148, 163, 184],
            Self::Bronze => [217, 119, 6],
            Self::Silver => [203, 213, 225],
            Self::Gold => [250, 204, 21],
            Self::PlatinumVoidCrusher => [52, 211, 153],
        }
    }
}

/// In-pocket `Drugit` medicinal chemistry R-group sprout warheads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SculptSproutWarhead {
    #[serde(skip_deserializing)]
    pub code: &'static str,
    #[serde(skip_deserializing)]
    pub label: &'static str,
    #[serde(skip_deserializing)]
    pub smiles_branch: &'static str,
    pub bond_len_a: f32,
}

pub const SCULPT_SPROUT_WARHEADS: [SculptSproutWarhead; 8] = [
    SculptSproutWarhead {
        code: "Cl",
        label: "+Chloro (-Cl)",
        smiles_branch: "Cl",
        bond_len_a: 1.74,
    },
    SculptSproutWarhead {
        code: "Me",
        label: "+Methyl (-CH3)",
        smiles_branch: "C",
        bond_len_a: 1.52,
    },
    SculptSproutWarhead {
        code: "F",
        label: "+Fluoro (-F)",
        smiles_branch: "F",
        bond_len_a: 1.36,
    },
    SculptSproutWarhead {
        code: "CN",
        label: "+Nitrile (-CN)",
        smiles_branch: "C#N",
        bond_len_a: 1.44,
    },
    SculptSproutWarhead {
        code: "OH",
        label: "+Hydroxyl (-OH)",
        smiles_branch: "O",
        bond_len_a: 1.41,
    },
    SculptSproutWarhead {
        code: "OMe",
        label: "+Methoxy (-OCH3)",
        smiles_branch: "OC",
        bond_len_a: 1.40,
    },
    SculptSproutWarhead {
        code: "CHF2",
        label: "+CHF2 (-CHF2)",
        smiles_branch: "C(F)F",
        bond_len_a: 1.51,
    },
    SculptSproutWarhead {
        code: "CF3",
        label: "+CF3 (-CF3)",
        smiles_branch: "C(F)(F)F",
        bond_len_a: 1.52,
    },
];

/// Flexible receptor side chain participating in `Shake` combinatorial rotamer repacking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptFlexibleSidechain {
    pub residue_id: String,
    pub res_name: String,
    pub anchor_ca_pos: [f32; 3],
    pub rotamer_tips: [[f32; 3]; 4],
    #[serde(skip_deserializing)]
    pub rotamer_labels: [&'static str; 4],
    pub rotamer_dunbrack_kcal: [f64; 4],
    pub active_rotamer: usize,
    pub tip_elem: Element,
    pub is_polar: bool,
    pub frozen: bool,
}

/// Target endpoint of a harmonic Foldit rubber-band restraint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BandAnchorTarget {
    ReceptorAnchor {
        anchor_idx: usize,
        label: String,
        pos: [f32; 3],
    },
    LigandAtom {
        atom_idx: usize,
    },
    SpacePin {
        label: String,
        pos: [f32; 3],
    },
}

/// Soft harmonic distance restraint (`E_band = 0.5 * k * (d - d_0)^2`) guiding `Wiggle` and 3D `Tug`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RubberBandRestraint {
    pub band_id: usize,
    pub lig_atom_idx: usize,
    pub target: BandAnchorTarget,
    pub target_dist_a: f32,
    pub spring_k_kcal_a2: f64,
    pub enabled: bool,
    pub current_dist_a: f32,
    pub strain_kcal: f64,
}

/// Fixed receptor atom + `WaterMap` hotspot summary cached inside `SculptSession` for sub-millisecond 60-FPS force evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptReceptorAtom {
    pub label: String,
    pub res_name: String,
    pub pos: [f32; 3],
    pub elem: Element,
    pub is_polar: bool,
    pub formal_charge: f64,
}

/// High-value active-site hydration hotspot (`WaterMap`) rewarded upon steric displacement or polar mimicry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptWaterTarget {
    pub site_id: String,
    pub pos: [f32; 3],
    /// Positive = unhappy hotspot (reward displacement); negative = conserved bridge (penalize nonpolar crush).
    pub dg_excess_kcal: f64,
    pub currently_displaced: bool,
}

/// Detailed energy & `Drugit` multi-parameter score decomposition for the current sculpted pose.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptScoreBreakdown {
    pub cart_bonded_strain_kcal: f64,
    pub vdw_dispersion_kcal: f64,
    pub steric_clash_kcal: f64,
    pub clash_count: usize,
    pub hbond_electrostatic_kcal: f64,
    pub formed_hbonds: usize,
    pub water_hotspot_kcal: f64,
    pub displaced_unhappy_waters: usize,
    pub sidechain_rotamer_kcal: f64,
    pub medchem_penalty_kcal: f64,
    pub rubber_band_spring_kcal: f64,
    /// True physical + med-chem energy (`kcal/mol`, unbanded, full `clashes_weight = 1.0`) used for leaderboard scoring!
    pub total_unbanded_dg_kcal: f64,
    /// Active steered potential (`kcal/mol`, including `clashes_weight` scaling + `rubber_band_spring_kcal`) minimized by `Wiggle`.
    pub steered_objective_kcal: f64,
    pub foldit_score_pts: f64,
    pub medal: SculptMedalTier,
    pub mw: f64,
    pub clogp: f64,
    pub tpsa: f64,
    pub hbd: usize,
    pub sa_score: f64,
    pub pains_count: usize,
    pub all_drugit_objectives_met: bool,
}

/// Reversible state snapshot for `Undo`, `Redo`, and `Restore Best`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptSnapshot {
    pub mol: Molecule,
    pub lig_coords: Vec<[f32; 3]>,
    pub frozen_lig_atoms: Vec<bool>,
    pub active_rotamers: Vec<usize>,
    pub bands: Vec<RubberBandRestraint>,
}

/// Recorded trajectory point along a player's sculpting session.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptTrajectoryStep {
    pub step_idx: usize,
    pub action_label: String,
    pub elapsed_secs: f64,
    pub foldit_score_pts: f64,
    pub unbanded_dg_kcal: f64,
    pub hbonds: usize,
    pub clashes: usize,
    pub smiles: String,
}

/// Complete interactive 3D Foldit/Drugit sculpting session state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptSession {
    pub preset: SculptPuzzlePreset,
    pub pdb_id: String,
    pub pocket_center: [f32; 3],
    pub mol: Molecule,
    pub lig_coords: Vec<[f32; 3]>,
    pub frozen_lig_atoms: Vec<bool>,
    pub receptor_atoms: Vec<SculptReceptorAtom>,
    pub sidechains: Vec<SculptFlexibleSidechain>,
    pub water_targets: Vec<SculptWaterTarget>,
    pub bands: Vec<RubberBandRestraint>,
    pub next_band_id: usize,
    /// Foldit steric repulsion dial (`0.10..=1.00`; lower temporarily to pull groups past tight gatekeepers).
    pub clashes_weight: f64,
    pub selected_lig_atom: Option<usize>,
    pub pending_band_lig_atom: Option<usize>,
    pub score: SculptScoreBreakdown,
    pub initial_score_pts: f64,
    pub best_score_pts: f64,
    pub best_snapshot: SculptSnapshot,
    pub undo_stack: Vec<SculptSnapshot>,
    pub redo_stack: Vec<SculptSnapshot>,
    pub trajectory: Vec<SculptTrajectoryStep>,
    pub moves_count: usize,
    pub wiggle_count: usize,
    pub shake_count: usize,
    pub tug_frames: usize,
    pub sprout_count: usize,
    pub elapsed_secs: f64,
    pub timer_running: bool,
    pub last_ui_time_s: Option<f64>,
}

/// Embedded `PDB 5NVX` Von Hippel-Lindau (`VHL`) E3 Ubiquitin Ligase active site (Drugit 2025 benchmark).
pub fn curated_vhl_5nvx_pdb_text() -> &'static str {
    r#"HEADER    LIGASE/INHIBITOR                        08-MAY-17   5NVX
TITLE     VON HIPPEL-LINDAU E3 LIGASE HYDROXYPROLINE POCKET (DRUGIT BENCHMARK)
ATOM      1  CA  TRP A  88       8.420   3.950   2.180  1.00 18.40           C
ATOM      2  CG  TRP A  88       9.180   2.820   1.640  1.00 18.80           C
ATOM      3  NE1 TRP A  88       9.840   1.960   2.450  1.00 19.20           N
ATOM      4  CZ2 TRP A  88       9.620   2.480   0.320  1.00 19.00           C
ATOM      5  CA  TYR A  98       4.850   2.640   3.920  1.00 16.80           C
ATOM      6  CG  TYR A  98       5.760   1.840   3.080  1.00 17.20           C
ATOM      7  OH  TYR A  98       6.680   1.020   2.380  1.00 17.60           O
ATOM      8  CB  PRO A  99       4.920   3.880   0.620  1.00 18.50           C
ATOM      9  CG  PRO A  99       5.820   4.420  -0.380  1.00 18.90           C
ATOM     10  CA  ILE A 109       6.250  -2.420   3.860  1.00 17.90           C
ATOM     11  CD1 ILE A 109       6.980  -1.480   2.950  1.00 18.30           C
ATOM     12  CA  HIS A 110       8.950  -2.180   2.920  1.00 17.40           C
ATOM     13  ND1 HIS A 110       8.320  -1.120   2.180  1.00 17.80           N
ATOM     14  NE2 HIS A 110       7.680  -0.280   2.960  1.00 18.10           N
ATOM     15  CA  SER A 111       9.820  -0.840   0.520  1.00 17.70           C
ATOM     16  OG  SER A 111       8.860  -0.180   1.280  1.00 18.20           O
ATOM     17  CA  TYR A 112      11.420   1.480   1.920  1.00 18.60           C
ATOM     18  OH  TYR A 112      10.480   0.920   2.780  1.00 19.10           O
ATOM     19  CA  HIS A 115       4.620  -0.920   0.480  1.00 16.90           C
ATOM     20  CG  HIS A 115       5.680  -0.380   1.320  1.00 17.30           C
ATOM     21  ND1 HIS A 115       6.480   0.580   0.880  1.00 17.50           N
ATOM     22  NE2 HIS A 115       6.280  -0.580   2.480  1.00 17.60           N
ATOM     23  CA  PHE A  91       7.420   4.120  -1.240  1.00 19.40           C
ATOM     24  CZ  PHE A  91       7.920   2.840  -0.680  1.00 19.80           C
ATOM     25  NH1 ARG A  69      11.860  -1.620   2.850  1.00 21.20           N
HETATM   26  C1  VHL A 501       7.450   0.620   1.820  1.00 18.00           C
HETATM   27  O2  VHL A 501       6.850   0.220   2.820  1.00 17.80           O
HETATM   28  N3  VHL A 501       8.480   1.420   1.520  1.00 18.20           N
HETATM   29  C4  VHL A 501       9.420   1.820   2.480  1.00 18.60           C
HETATM   30  C5  VHL A 501      10.480   2.580   1.920  1.00 19.00           C
HETATM   31  O   HOH A2101       8.920  -0.880   3.680  1.00 19.40           O
HETATM   32  O   HOH A2102       5.880   2.420   0.920  1.00 32.80           O
HETATM   33  O   HOH A2103      10.350   2.920   0.480  1.00 34.50           O
END
"#
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn covalent_radius_a(elem: Element) -> f32 {
    match elem {
        Element::H => 0.31,
        Element::B => 0.84,
        Element::C => 0.76,
        Element::N => 0.71,
        Element::O => 0.66,
        Element::F => 0.57,
        Element::P => 1.07,
        Element::S => 1.05,
        Element::Cl => 1.02,
        Element::Br => 1.20,
        Element::I => 1.39,
        _ => 0.80,
    }
}

#[inline]
fn sculpt_vdw_radius_a(elem: Element) -> f32 {
    match elem {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::F => 1.47,
        Element::P => 1.80,
        Element::S => 1.80,
        Element::Cl => 1.75,
        Element::Br => 1.85,
        Element::I => 1.98,
        _ => 1.72,
    }
}

#[inline]
fn ideal_bond_len_a(ea: Element, eb: Element, kekule_order: u8, aromatic: bool) -> f32 {
    let base = covalent_radius_a(ea) + covalent_radius_a(eb);
    if aromatic {
        (base - 0.11).max(1.28)
    } else {
        match kekule_order {
            3 => (base - 0.26).max(1.14),
            2 => (base - 0.16).max(1.20),
            _ => base.max(1.08),
        }
    }
}

fn load_sculpt_pdb(
    preset: SculptPuzzlePreset,
    custom_pdb_path: Option<&str>,
) -> (ParsedPdbStructure, Option<[f32; 3]>) {
    if preset == SculptPuzzlePreset::CustomPdbFreeformSandbox {
        if let Some(p) = custom_pdb_path {
            let t = p.trim();
            if !t.is_empty() {
                if let Ok(raw) = std::fs::read_to_string(t) {
                    if let Ok(parsed) = parse_pdb_structure(&raw) {
                        return (parsed, None);
                    }
                }
            }
        }
    }
    match preset {
        SculptPuzzlePreset::Vhl5nvxDepeptidizationChallenge
        | SculptPuzzlePreset::CustomPdbFreeformSandbox => (
            parse_pdb_structure(curated_vhl_5nvx_pdb_text()).expect("Valid 5NVX VHL PDB"),
            None,
        ),
        SculptPuzzlePreset::Egfr1m17GatekeeperBarrierChallenge => (
            parse_pdb_structure(curated_pocket_pdb_text(0)).expect("Valid 1M17 EGFR PDB"),
            None,
        ),
        SculptPuzzlePreset::FactorXa1ezqVoidAndWaterCrusher => (
            parse_pdb_structure(curated_factor_xa_1ezq_pdb_text()).expect("Valid 1EZQ FXa PDB"),
            None,
        ),
        SculptPuzzlePreset::Cox23ln1SelectivityLatchChallenge => (
            parse_pdb_structure(curated_pocket_pdb_text(2)).expect("Valid 3LN1 COX-2 PDB"),
            None,
        ),
        SculptPuzzlePreset::Kras6oimSwitchIiCrypticPuzzle => (
            parse_pdb_structure(curated_allosteric_pdb_text(1)).expect("Valid 6OIM KRAS PDB"),
            Some([12.90, 5.10, 8.20]),
        ),
    }
}

/// Initializes a new Foldit/Drugit 3D sculpting session for `preset`.
pub fn new_sculpt_session(
    preset: SculptPuzzlePreset,
    fallback_mol: Option<&Molecule>,
    custom_pdb_path: Option<&str>,
) -> SculptSession {
    let (pdb, center_override) = load_sculpt_pdb(preset, custom_pdb_path);
    let pocket = analyze_binding_pocket(&pdb, center_override, 9.5);

    let mol = if !preset.starter_smiles().is_empty() {
        let mut m = parse_smiles(preset.starter_smiles()).expect("Valid puzzle starter SMILES");
        m.id = preset.code().to_string();
        m.name = format!("{} Starter", preset.short_label());
        m
    } else if let Some(fb) = fallback_mol {
        fb.clone()
    } else {
        let mut m = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1").expect("Fallback SMILES");
        m.id = "SANDBOX-SEED".to_string();
        m.name = "Sandbox Lead".to_string();
        m
    };

    let placed = score_ligand_in_pocket(&mol, None, &pocket);
    let mut lig_coords = placed.aligned_ligand_coords3d;
    if lig_coords.len() < mol.atoms.len() {
        for a in mol.atoms.iter().skip(lig_coords.len()) {
            lig_coords.push([
                pocket.center[0] + a.pos2d[0] * 0.8,
                pocket.center[1] + a.pos2d[1] * 0.8,
                pocket.center[2],
            ]);
        }
    }

    // For challenge puzzles (1..5), apply a realistic starting perturbation (`~1.05 A` translation + slight twist)
    // so the player starts in an unoptimized pre-docked decoy with tangible barrier-crossing & H-bond headroom!
    if preset != SculptPuzzlePreset::CustomPdbFreeformSandbox {
        let shift = match preset {
            SculptPuzzlePreset::Vhl5nvxDepeptidizationChallenge => [0.92_f32, -0.68, 0.54],
            SculptPuzzlePreset::Egfr1m17GatekeeperBarrierChallenge => [1.08_f32, 0.62, -0.58],
            SculptPuzzlePreset::FactorXa1ezqVoidAndWaterCrusher => [-0.88_f32, 0.74, -0.45],
            SculptPuzzlePreset::Cox23ln1SelectivityLatchChallenge => [0.85_f32, -0.82, 0.52],
            SculptPuzzlePreset::Kras6oimSwitchIiCrypticPuzzle => [-0.95_f32, 0.68, 0.62],
            SculptPuzzlePreset::CustomPdbFreeformSandbox => [0.0, 0.0, 0.0],
        };
        for (idx, p) in lig_coords.iter_mut().enumerate() {
            let wob = ((idx as f32) * 1.37).sin() * 0.16;
            p[0] += shift[0] + wob;
            p[1] += shift[1] - wob * 0.7;
            p[2] += shift[2];
        }
    }

    // Build fixed receptor shell atoms + flexible pocket side chains (with 4 Dunbrack rotamer states each!).
    let mut receptor_atoms: Vec<SculptReceptorAtom> = Vec::new();
    let mut sidechains: Vec<SculptFlexibleSidechain> = Vec::new();

    for sa in pocket.shell_atoms.iter().take(96) {
        if matches!(sa.elem, Element::H) {
            continue;
        }
        let is_pol = matches!(sa.elem, Element::N | Element::O | Element::F)
            || sa.atom_name.starts_with('N')
            || sa.atom_name.starts_with('O');
        let q = match (sa.res_name.as_str(), is_pol) {
            ("ASP" | "GLU", true) => -0.55,
            ("ARG" | "LYS", true) => 0.55,
            ("HIS", true) => 0.22,
            (_, true) => -0.28,
            _ => 0.0,
        };
        let res_id = format!("{}{}", sa.res_name, sa.res_seq);
        let is_tip_atom = !matches!(sa.atom_name.as_str(), "N" | "CA" | "C" | "O" | "CB");
        if is_tip_atom
            && sidechains.len() < 10
            && !sidechains.iter().any(|s| s.residue_id == res_id)
        {
            // Find matching CA/CB anchor for this residue, or synthesize 2.4 A outward anchor.
            let ca_pos = pdb
                .protein_atoms
                .iter()
                .find(|a| a.res_seq == sa.res_seq && a.atom_name == "CA")
                .map(|a| a.pos)
                .unwrap_or([
                    sa.pos[0] + (sa.pos[0] - pocket.center[0]) * 0.35,
                    sa.pos[1] + (sa.pos[1] - pocket.center[1]) * 0.35,
                    sa.pos[2] + (sa.pos[2] - pocket.center[2]) * 0.35,
                ]);
            let vx = sa.pos[0] - ca_pos[0];
            let vy = sa.pos[1] - ca_pos[1];
            let vz = sa.pos[2] - ca_pos[2];
            // Build two orthogonal tangent vectors perpendicular to CA->tip for realistic chi1/chi2 swing (`~1.25 A`).
            let vn = (vx * vx + vy * vy + vz * vz).sqrt().max(0.6);
            let ux = vx / vn;
            let uy = vy / vn;
            let uz = vz / vn;
            let (tx1, ty1, tz1) = if ux.abs() < 0.8 {
                let n = (uy * uy + uz * uz).sqrt().max(1e-4);
                (0.0_f32, -uz / n, uy / n)
            } else {
                let n = (ux * ux + uz * uz).sqrt().max(1e-4);
                (uz / n, 0.0_f32, -ux / n)
            };
            let tx2 = uy * tz1 - uz * ty1;
            let ty2 = uz * tx1 - ux * tz1;
            let tz2 = ux * ty1 - uy * tx1;

            let r0 = sa.pos;
            let r1 = [
                sa.pos[0] + tx1 * 1.18,
                sa.pos[1] + ty1 * 1.18,
                sa.pos[2] + tz1 * 1.18,
            ];
            let r2 = [
                sa.pos[0] - tx1 * 0.95 + tx2 * 0.82,
                sa.pos[1] - ty1 * 0.95 + ty2 * 0.82,
                sa.pos[2] - tz1 * 0.95 + tz2 * 0.82,
            ];
            let r3 = [
                sa.pos[0] + ux * 0.55 - tx2 * 1.25,
                sa.pos[1] + uy * 0.55 - ty2 * 1.25,
                sa.pos[2] + uz * 0.55 - tz2 * 1.25,
            ];
            sidechains.push(SculptFlexibleSidechain {
                residue_id: res_id,
                res_name: sa.res_name.clone(),
                anchor_ca_pos: ca_pos,
                rotamer_tips: [r0, r1, r2, r3],
                rotamer_labels: ["g- (Crystal)", "t (Trans)", "g+ (Gauche+)", "Open-Out"],
                rotamer_dunbrack_kcal: [0.0, 0.28, 0.42, 0.78],
                active_rotamer: if preset == SculptPuzzlePreset::Egfr1m17GatekeeperBarrierChallenge
                    && sidechains.len() < 2
                {
                    2
                } else {
                    0
                },
                tip_elem: sa.elem,
                is_polar: is_pol,
                frozen: false,
            });
        } else {
            receptor_atoms.push(SculptReceptorAtom {
                label: format!("{}{}:{}", sa.res_name, sa.res_seq, sa.atom_name),
                res_name: sa.res_name.clone(),
                pos: sa.pos,
                elem: sa.elem,
                is_polar: is_pol,
                formal_charge: q,
            });
        }
    }

    let mut water_targets: Vec<SculptWaterTarget> = Vec::new();
    for (wi, w) in pdb.waters.iter().take(8).enumerate() {
        let dg_ex = if w.b_factor >= 26.0 {
            1.65 + ((w.b_factor as f64 - 26.0) * 0.11).clamp(0.0, 1.45)
        } else {
            -1.45
        };
        water_targets.push(SculptWaterTarget {
            site_id: format!("HOH{}", wi + 1),
            pos: w.pos,
            dg_excess_kcal: dg_ex,
            currently_displaced: false,
        });
    }

    let frozen_lig = vec![false; mol.atoms.len()];
    let dummy_snap = SculptSnapshot {
        mol: mol.clone(),
        lig_coords: lig_coords.clone(),
        frozen_lig_atoms: frozen_lig.clone(),
        active_rotamers: sidechains.iter().map(|s| s.active_rotamer).collect(),
        bands: Vec::new(),
    };
    let dummy_score = SculptScoreBreakdown {
        cart_bonded_strain_kcal: 0.0,
        vdw_dispersion_kcal: 0.0,
        steric_clash_kcal: 0.0,
        clash_count: 0,
        hbond_electrostatic_kcal: 0.0,
        formed_hbonds: 0,
        water_hotspot_kcal: 0.0,
        displaced_unhappy_waters: 0,
        sidechain_rotamer_kcal: 0.0,
        medchem_penalty_kcal: 0.0,
        rubber_band_spring_kcal: 0.0,
        total_unbanded_dg_kcal: 0.0,
        steered_objective_kcal: 0.0,
        foldit_score_pts: FOLDIT_BASE_SCORE_PTS,
        medal: SculptMedalTier::Unranked,
        mw: 0.0,
        clogp: 0.0,
        tpsa: 0.0,
        hbd: 0,
        sa_score: 0.0,
        pains_count: 0,
        all_drugit_objectives_met: false,
    };

    let mut sess = SculptSession {
        preset,
        pdb_id: pdb.pdb_id,
        pocket_center: pocket.center,
        mol,
        lig_coords,
        frozen_lig_atoms: frozen_lig,
        receptor_atoms,
        sidechains,
        water_targets,
        bands: Vec::new(),
        next_band_id: 1,
        clashes_weight: 1.0,
        selected_lig_atom: Some(0),
        pending_band_lig_atom: None,
        score: dummy_score,
        initial_score_pts: FOLDIT_BASE_SCORE_PTS,
        best_score_pts: 0.0,
        best_snapshot: dummy_snap,
        undo_stack: Vec::new(),
        redo_stack: Vec::new(),
        trajectory: Vec::new(),
        moves_count: 0,
        wiggle_count: 0,
        shake_count: 0,
        tug_frames: 0,
        sprout_count: 0,
        elapsed_secs: 0.0,
        timer_running: preset != SculptPuzzlePreset::CustomPdbFreeformSandbox,
        last_ui_time_s: None,
    };
    recompute_sculpt_score(&mut sess);
    sess.initial_score_pts = sess.score.foldit_score_pts;
    sess.best_score_pts = sess.score.foldit_score_pts;
    sess.best_snapshot = capture_snapshot(&sess);
    record_trajectory_step(&mut sess, "Start Puzzle Pose");
    sess
}

fn capture_snapshot(sess: &SculptSession) -> SculptSnapshot {
    SculptSnapshot {
        mol: sess.mol.clone(),
        lig_coords: sess.lig_coords.clone(),
        frozen_lig_atoms: sess.frozen_lig_atoms.clone(),
        active_rotamers: sess.sidechains.iter().map(|s| s.active_rotamer).collect(),
        bands: sess.bands.clone(),
    }
}

fn apply_snapshot(sess: &mut SculptSession, snap: SculptSnapshot) {
    sess.mol = snap.mol;
    sess.lig_coords = snap.lig_coords;
    sess.frozen_lig_atoms = snap.frozen_lig_atoms;
    for (sc, &rot) in sess.sidechains.iter_mut().zip(snap.active_rotamers.iter()) {
        sc.active_rotamer = rot.min(3);
    }
    sess.bands = snap.bands;
    if let Some(sel) = sess.selected_lig_atom {
        if sel >= sess.mol.atoms.len() {
            sess.selected_lig_atom = (!sess.mol.atoms.is_empty()).then_some(0);
        }
    }
    recompute_sculpt_score(sess);
}

pub fn push_undo_checkpoint(sess: &mut SculptSession) {
    if sess.undo_stack.len() >= 40 {
        sess.undo_stack.remove(0);
    }
    sess.undo_stack.push(capture_snapshot(sess));
    sess.redo_stack.clear();
}

pub fn undo_sculpt_move(sess: &mut SculptSession) -> bool {
    let Some(prev) = sess.undo_stack.pop() else {
        return false;
    };
    sess.redo_stack.push(capture_snapshot(sess));
    apply_snapshot(sess, prev);
    record_trajectory_step(sess, "Undo");
    true
}

pub fn redo_sculpt_move(sess: &mut SculptSession) -> bool {
    let Some(nxt) = sess.redo_stack.pop() else {
        return false;
    };
    sess.undo_stack.push(capture_snapshot(sess));
    apply_snapshot(sess, nxt);
    record_trajectory_step(sess, "Redo");
    true
}

pub fn restore_best_sculpt_pose(sess: &mut SculptSession) {
    push_undo_checkpoint(sess);
    let best = sess.best_snapshot.clone();
    apply_snapshot(sess, best);
    record_trajectory_step(sess, "Restore Personal Best");
}

fn record_trajectory_step(sess: &mut SculptSession, label: &str) {
    let idx = sess.trajectory.len();
    let smi = if sess.mol.smiles.is_empty() {
        molecule_to_smiles(&sess.mol)
    } else {
        sess.mol.smiles.clone()
    };
    sess.trajectory.push(SculptTrajectoryStep {
        step_idx: idx,
        action_label: label.to_string(),
        elapsed_secs: sess.elapsed_secs,
        foldit_score_pts: sess.score.foldit_score_pts,
        unbanded_dg_kcal: sess.score.total_unbanded_dg_kcal,
        hbonds: sess.score.formed_hbonds,
        clashes: sess.score.clash_count,
        smiles: smi,
    });
}

/// Evaluates both the true physical + med-chem Foldit score (unbanded, `clashes_weight = 1.0`) and
/// the steered objective (`clashes_weight`-scaled + active rubber bands) in sub-millisecond time.
pub fn recompute_sculpt_score(sess: &mut SculptSession) {
    let n_atoms = sess.mol.atoms.len().min(sess.lig_coords.len());
    let mut cart_bonded = 0.0_f64;

    // 1. `cart_bonded` 1-2 covalent bond stretch + 1-3 valence angle bend + intramolecular 1-4+ clash guard!
    for b in &sess.mol.bonds {
        if b.a < n_atoms && b.b < n_atoms {
            let r0 = ideal_bond_len_a(
                sess.mol.atoms[b.a].elem,
                sess.mol.atoms[b.b].elem,
                b.kekule_order,
                b.aromatic,
            );
            let d = dist3(sess.lig_coords[b.a], sess.lig_coords[b.b]);
            let dr = (d - r0) as f64;
            cart_bonded += 64.0 * dr * dr;
        }
    }
    for center_idx in 0..n_atoms {
        let Some(nbrs) = sess.mol.adj.get(center_idx) else {
            continue;
        };
        if nbrs.len() < 2 {
            continue;
        }
        let ideal_cos: f64 = match sess.mol.atoms[center_idx].hyb {
            Hybridization::Sp => -1.0,
            Hybridization::Sp2 => -0.50,
            _ => -0.3333,
        };
        let pc = sess.lig_coords[center_idx];
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                let u = nbrs[i].0;
                let v = nbrs[j].0;
                if u >= n_atoms || v >= n_atoms {
                    continue;
                }
                let pu = sess.lig_coords[u];
                let pv = sess.lig_coords[v];
                let d1 = dist3(pu, pc).max(1e-4);
                let d2 = dist3(pv, pc).max(1e-4);
                let cos_th = (((pu[0] - pc[0]) * (pv[0] - pc[0])
                    + (pu[1] - pc[1]) * (pv[1] - pc[1])
                    + (pu[2] - pc[2]) * (pv[2] - pc[2]))
                    / (d1 * d2))
                    .clamp(-1.0, 1.0) as f64;
                let dc = cos_th - ideal_cos;
                cart_bonded += 5.5 * dc * dc;
            }
        }
    }
    // Nonbonded 1-4+ internal ligand self-clash guard
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            let bonded_12 = sess
                .mol
                .adj
                .get(i)
                .map(|l| l.iter().any(|&(nb, _)| nb == j))
                .unwrap_or(false);
            if bonded_12 {
                continue;
            }
            let bonded_13 = sess
                .mol
                .adj
                .get(i)
                .map(|l| {
                    l.iter().any(|&(mid, _)| {
                        sess.mol
                            .adj
                            .get(mid)
                            .map(|l2| l2.iter().any(|&(nb2, _)| nb2 == j))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);
            if bonded_13 {
                continue;
            }
            let d = dist3(sess.lig_coords[i], sess.lig_coords[j]);
            if d < 2.35 {
                let ov = (2.35 - d) as f64;
                cart_bonded += 12.0 * ov * ov;
            }
        }
    }

    // 2. Nonbonded Protein-Ligand vdW + Steric Clash + H-Bond / Salt-Bridge across Fixed Shell + Active Side-Chain Rotamers!
    let mut vdw_disp = 0.0_f64;
    let mut clash_pen = 0.0_f64;
    let mut clash_count = 0_usize;
    let mut hbond_elec = 0.0_f64;
    let mut formed_hbonds = 0_usize;

    for i in 0..n_atoms {
        let lp = sess.lig_coords[i];
        let le = sess.mol.atoms[i].elem;
        let r_lig = sculpt_vdw_radius_a(le);
        let lig_polar = matches!(le, Element::N | Element::O | Element::F);

        for ra in &sess.receptor_atoms {
            let d = dist3(lp, ra.pos);
            if d > 5.80 {
                continue;
            }
            let r_sum = r_lig + sculpt_vdw_radius_a(ra.elem) - if lig_polar && ra.is_polar { 0.48 } else { 0.22 };
            if d < r_sum {
                let ov = (r_sum - d) as f64;
                clash_pen += 7.5 * ov * ov;
                if ov > 0.28 {
                    clash_count += 1;
                }
            } else {
                let x = (r_sum + 0.38) / d.max(2.0);
                let x6 = (x as f64).powi(6);
                vdw_disp += (0.145 * (x6 * x6 - 2.0 * x6)).clamp(-0.22, 0.30);
            }
            if lig_polar && ra.is_polar && (2.42..=3.52).contains(&d) {
                let well = (-(((d as f64) - 2.86) / 0.36).powi(2)).exp();
                let strength = if ra.formal_charge.abs() > 0.4 { 1.48 } else { 1.12 };
                hbond_elec -= strength * well;
                if well >= 0.45 {
                    formed_hbonds += 1;
                }
            }
        }

        for sc in &sess.sidechains {
            let tip = sc.rotamer_tips[sc.active_rotamer.min(3)];
            let d = dist3(lp, tip);
            if d > 5.80 {
                continue;
            }
            let r_sum = r_lig + sculpt_vdw_radius_a(sc.tip_elem)
                - if lig_polar && sc.is_polar { 0.50 } else { 0.24 };
            if d < r_sum {
                let ov = (r_sum - d) as f64;
                clash_pen += 8.2 * ov * ov;
                if ov > 0.26 {
                    clash_count += 1;
                }
            } else {
                let x = (r_sum + 0.36) / d.max(2.0);
                let x6 = (x as f64).powi(6);
                vdw_disp += (0.18 * (x6 * x6 - 2.0 * x6)).clamp(-0.28, 0.30);
            }
            if lig_polar && sc.is_polar && (2.40..=3.55).contains(&d) {
                let well = (-(((d as f64) - 2.84) / 0.38).powi(2)).exp();
                hbond_elec -= 1.35 * well;
                if well >= 0.45 {
                    formed_hbonds += 1;
                }
            }
        }
    }

    // 3. WaterMap Active-Site Hotspot Displacement vs Bridge Crushing
    let mut water_kcal = 0.0_f64;
    let mut disp_unhappy = 0_usize;
    for wt in &mut sess.water_targets {
        let mut f_ov = 0.0_f64;
        let mut polar_near = false;
        for i in 0..n_atoms {
            let d = dist3(sess.lig_coords[i], wt.pos);
            let pol = matches!(sess.mol.atoms[i].elem, Element::N | Element::O | Element::F);
            if d < 2.24 {
                f_ov += (1.0 - d / 2.24) as f64;
                if pol {
                    polar_near = true;
                }
            } else if pol && (2.40..=3.45).contains(&d) {
                polar_near = true;
            }
        }
        let f_cap = f_ov.clamp(0.0, 1.0);
        wt.currently_displaced = f_cap >= 0.35;
        if wt.dg_excess_kcal > 0.0 {
            if f_cap >= 0.35 {
                disp_unhappy += 1;
            }
            water_kcal -= wt.dg_excess_kcal * f_cap * 0.72;
        } else if f_cap >= 0.35 && !polar_near {
            water_kcal += (-wt.dg_excess_kcal) * f_cap * 0.75;
        } else if f_cap < 0.35 && polar_near {
            water_kcal -= 0.42;
        }
    }

    // 4. Side-chain Dunbrack rotamer internal energy
    let rotamer_kcal: f64 = sess
        .sidechains
        .iter()
        .map(|s| s.rotamer_dunbrack_kcal[s.active_rotamer.min(3)])
        .sum();

    // 5. Harmonic Rubber-Band Spring Energy (steers Wiggle/Tug, excluded from official Foldit Score!)
    let mut band_kcal = 0.0_f64;
    for b in &mut sess.bands {
        if b.lig_atom_idx >= n_atoms {
            b.current_dist_a = 0.0;
            b.strain_kcal = 0.0;
            continue;
        }
        let pa = sess.lig_coords[b.lig_atom_idx];
        let pb = match &b.target {
            BandAnchorTarget::ReceptorAnchor { pos, .. } => *pos,
            BandAnchorTarget::SpacePin { pos, .. } => *pos,
            BandAnchorTarget::LigandAtom { atom_idx } => {
                sess.lig_coords.get(*atom_idx).copied().unwrap_or(pa)
            }
        };
        let d = dist3(pa, pb);
        b.current_dist_a = d;
        let diff = (d - b.target_dist_a) as f64;
        let e_b = if b.enabled {
            0.5 * b.spring_k_kcal_a2 * diff * diff
        } else {
            0.0
        };
        b.strain_kcal = e_b;
        band_kcal += e_b;
    }

    // 6. Drugit Multifactorial Med-Chem Penalties / Bonuses (`TPSA`, `cLogP`, `HBD`, `SAScore`, `PAINS`)
    let alerts = scan_structural_alerts(&sess.mol);
    let phys = evaluate_physchem(&sess.mol, alerts.len());
    let sa = evaluate_synthetic_accessibility(&sess.mol).sa_score;
    let pains_cnt = alerts
        .iter()
        .filter(|a| a.family.eq_ignore_ascii_case("PAINS"))
        .count();
    let max_tpsa = sess.preset.max_tpsa_target();
    let max_clogp = sess.preset.max_clogp_target();

    let mut medchem_pen = 0.0_f64;
    if phys.tpsa > max_tpsa {
        medchem_pen += (phys.tpsa - max_tpsa) * 0.045;
    }
    if phys.clogp > max_clogp {
        medchem_pen += (phys.clogp - max_clogp) * 0.65;
    }
    if phys.hbd > 3 {
        medchem_pen += ((phys.hbd - 3) as f64) * 0.55;
    }
    if sa > 4.2 {
        medchem_pen += (sa - 4.2) * 0.50;
    }
    medchem_pen += (pains_cnt as f64) * 4.50;

    let unbanded_dg = vdw_disp
        + clash_pen
        + hbond_elec.clamp(-8.5, 0.0)
        + water_kcal
        + rotamer_kcal * 0.45
        + cart_bonded.min(45.0) * 0.55
        + medchem_pen;

    let cw = sess.clashes_weight.clamp(0.10, 1.00);
    let steered_obj = vdw_disp
        + clash_pen * cw
        + hbond_elec.clamp(-9.5, 0.0)
        + water_kcal
        + rotamer_kcal * 0.45
        + cart_bonded * 0.85
        + band_kcal;

    let pts = (FOLDIT_BASE_SCORE_PTS - unbanded_dg * FOLDIT_PTS_PER_KCAL).clamp(100.0, 16_000.0);
    let [t_br, t_si, t_go, t_pl] = sess.preset.medal_thresholds();
    let drugit_ok = pains_cnt == 0
        && clash_count == 0
        && cart_bonded <= 3.20
        && phys.tpsa <= max_tpsa
        && phys.clogp <= max_clogp
        && formed_hbonds >= 1;

    let medal = if pts >= t_pl && drugit_ok {
        SculptMedalTier::PlatinumVoidCrusher
    } else if pts >= t_go && pains_cnt == 0 && clash_count <= 1 {
        SculptMedalTier::Gold
    } else if pts >= t_si && pains_cnt == 0 {
        SculptMedalTier::Silver
    } else if pts >= t_br {
        SculptMedalTier::Bronze
    } else {
        SculptMedalTier::Unranked
    };

    sess.score = SculptScoreBreakdown {
        cart_bonded_strain_kcal: cart_bonded,
        vdw_dispersion_kcal: vdw_disp,
        steric_clash_kcal: clash_pen,
        clash_count,
        hbond_electrostatic_kcal: hbond_elec,
        formed_hbonds,
        water_hotspot_kcal: water_kcal,
        displaced_unhappy_waters: disp_unhappy,
        sidechain_rotamer_kcal: rotamer_kcal,
        medchem_penalty_kcal: medchem_pen,
        rubber_band_spring_kcal: band_kcal,
        total_unbanded_dg_kcal: unbanded_dg,
        steered_objective_kcal: steered_obj,
        foldit_score_pts: pts,
        medal,
        mw: phys.mw,
        clogp: phys.clogp,
        tpsa: phys.tpsa,
        hbd: phys.hbd,
        sa_score: sa,
        pains_count: pains_cnt,
        all_drugit_objectives_met: drugit_ok,
    };

    if pts > sess.best_score_pts + 0.5 {
        sess.best_score_pts = pts;
        sess.best_snapshot = capture_snapshot(sess);
    }
}

/// Evaluates the steered scalar potential for arbitrary candidate coordinates `coords` during line-search / finite-diff gradient steps.
fn eval_steered_coords_energy(
    sess: &SculptSession,
    coords: &[[f32; 3]],
    mouse_tug: Option<(usize, [f32; 3], f64)>,
) -> f64 {
    let n = sess.mol.atoms.len().min(coords.len());
    let cw = sess.clashes_weight.clamp(0.10, 1.00);
    let mut e = 0.0_f64;

    if let Some((t_idx, t_pos, k_tug)) = mouse_tug {
        if t_idx < n {
            let d = dist3(coords[t_idx], t_pos) as f64;
            e += 0.5 * k_tug * d * d;
        }
    }

    for b in &sess.mol.bonds {
        if b.a < n && b.b < n {
            let r0 = ideal_bond_len_a(
                sess.mol.atoms[b.a].elem,
                sess.mol.atoms[b.b].elem,
                b.kekule_order,
                b.aromatic,
            );
            let dr = (dist3(coords[b.a], coords[b.b]) - r0) as f64;
            e += 58.0 * dr * dr;
        }
    }

    for c_idx in 0..n {
        if let Some(nbrs) = sess.mol.adj.get(c_idx) {
            if nbrs.len() >= 2 {
                let ideal_cos: f64 = match sess.mol.atoms[c_idx].hyb {
                    Hybridization::Sp => -1.0,
                    Hybridization::Sp2 => -0.50,
                    _ => -0.3333,
                };
                let pc = coords[c_idx];
                for i in 0..nbrs.len() {
                    for j in (i + 1)..nbrs.len() {
                        let u = nbrs[i].0;
                        let v = nbrs[j].0;
                        if u < n && v < n {
                            let pu = coords[u];
                            let pv = coords[v];
                            let d1 = dist3(pu, pc).max(1e-4);
                            let d2 = dist3(pv, pc).max(1e-4);
                            let cos_th = (((pu[0] - pc[0]) * (pv[0] - pc[0])
                                + (pu[1] - pc[1]) * (pv[1] - pc[1])
                                + (pu[2] - pc[2]) * (pv[2] - pc[2]))
                                / (d1 * d2))
                                .clamp(-1.0, 1.0) as f64;
                            let dc = cos_th - ideal_cos;
                            e += 5.0 * dc * dc;
                        }
                    }
                }
            }
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let b12 = sess
                .mol
                .adj
                .get(i)
                .map(|l| l.iter().any(|&(nb, _)| nb == j))
                .unwrap_or(false);
            if b12 {
                continue;
            }
            let d = dist3(coords[i], coords[j]);
            if d < 2.28 {
                let ov = (2.28 - d) as f64;
                e += 10.0 * ov * ov;
            }
        }
    }

    for i in 0..n {
        let lp = coords[i];
        let le = sess.mol.atoms[i].elem;
        let r_lig = sculpt_vdw_radius_a(le);
        let lig_pol = matches!(le, Element::N | Element::O | Element::F);

        for ra in &sess.receptor_atoms {
            let d = dist3(lp, ra.pos);
            if d > 5.60 {
                continue;
            }
            let r_sum = r_lig + sculpt_vdw_radius_a(ra.elem) - if lig_pol && ra.is_polar { 0.48 } else { 0.22 };
            if d < r_sum {
                let ov = (r_sum - d) as f64;
                e += 7.5 * cw * ov * ov;
            } else {
                let x = (r_sum + 0.38) / d.max(2.0);
                let x6 = (x as f64).powi(6);
                e += (0.145 * (x6 * x6 - 2.0 * x6)).clamp(-0.22, 0.30);
            }
            if lig_pol && ra.is_polar && (2.30..=3.85).contains(&d) {
                let well = (-(((d as f64) - 2.86) / 0.38).powi(2)).exp();
                e -= 1.22 * well;
            }
        }

        for sc in &sess.sidechains {
            let tip = sc.rotamer_tips[sc.active_rotamer.min(3)];
            let d = dist3(lp, tip);
            if d > 5.60 {
                continue;
            }
            let r_sum = r_lig + sculpt_vdw_radius_a(sc.tip_elem)
                - if lig_pol && sc.is_polar { 0.50 } else { 0.24 };
            if d < r_sum {
                let ov = (r_sum - d) as f64;
                e += 8.2 * cw * ov * ov;
            } else {
                let x = (r_sum + 0.36) / d.max(2.0);
                let x6 = (x as f64).powi(6);
                e += (0.18 * (x6 * x6 - 2.0 * x6)).clamp(-0.28, 0.30);
            }
            if lig_pol && sc.is_polar && (2.30..=3.85).contains(&d) {
                let well = (-(((d as f64) - 2.84) / 0.40).powi(2)).exp();
                e -= 1.38 * well;
            }
        }
    }

    for wt in &sess.water_targets {
        let mut f_ov = 0.0_f64;
        for (i, &lp) in coords.iter().enumerate().take(n) {
            let d = dist3(lp, wt.pos);
            if d < 2.24 {
                f_ov += (1.0 - d / 2.24) as f64;
                let _ = i;
            }
        }
        let f_cap = f_ov.clamp(0.0, 1.0);
        if wt.dg_excess_kcal > 0.0 {
            e -= wt.dg_excess_kcal * f_cap * 0.72;
        }
    }

    for b in &sess.bands {
        if !b.enabled || b.lig_atom_idx >= n {
            continue;
        }
        let pa = coords[b.lig_atom_idx];
        let pb = match &b.target {
            BandAnchorTarget::ReceptorAnchor { pos, .. } => *pos,
            BandAnchorTarget::SpacePin { pos, .. } => *pos,
            BandAnchorTarget::LigandAtom { atom_idx } => coords.get(*atom_idx).copied().unwrap_or(pa),
        };
        let diff = (dist3(pa, pb) - b.target_dist_a) as f64;
        e += 0.5 * b.spring_k_kcal_a2 * diff * diff;
    }

    e
}

/// Core relaxation loop used by both `Wiggle` and real-time 60-FPS 3D `Tug`.
fn run_coordinate_relaxation(
    sess: &mut SculptSession,
    steps: usize,
    mouse_tug: Option<(usize, [f32; 3], f64)>,
) {
    let n = sess.mol.atoms.len().min(sess.lig_coords.len());
    if n == 0 {
        return;
    }
    let eps = 0.008_f32;
    let mut coords = sess.lig_coords.clone();
    let mut cur_e = eval_steered_coords_energy(sess, &coords, mouse_tug);

    for _ in 0..steps {
        let mut grad = vec![[0.0_f32; 3]; n];
        let mut max_g = 0.0_f32;
        for i in 0..n {
            if sess.frozen_lig_atoms.get(i).copied().unwrap_or(false) {
                continue;
            }
            for ax in 0..3 {
                let orig = coords[i][ax];
                coords[i][ax] = orig + eps;
                let ep = eval_steered_coords_energy(sess, &coords, mouse_tug);
                coords[i][ax] = orig - eps;
                let em = eval_steered_coords_energy(sess, &coords, mouse_tug);
                coords[i][ax] = orig;
                let g = ((ep - em) / (2.0 * (eps as f64))) as f32;
                let g_clamped = g.clamp(-40.0, 40.0);
                grad[i][ax] = g_clamped;
                if g_clamped.abs() > max_g {
                    max_g = g_clamped.abs();
                }
            }
        }
        if max_g < 0.02 {
            break;
        }
        // Backtracking Armijo step capped at `0.14 A` max per-atom displacement for silky stability!
        let mut step_scale = (0.12 / max_g).min(0.035);
        let mut accepted = false;
        for _ in 0..5 {
            let mut trial = coords.clone();
            for i in 0..n {
                if !sess.frozen_lig_atoms.get(i).copied().unwrap_or(false) {
                    trial[i][0] -= grad[i][0] * step_scale;
                    trial[i][1] -= grad[i][1] * step_scale;
                    trial[i][2] -= grad[i][2] * step_scale;
                }
            }
            let trial_e = eval_steered_coords_energy(sess, &trial, mouse_tug);
            if trial_e <= cur_e + 1e-5 {
                coords = trial;
                cur_e = trial_e;
                accepted = true;
                break;
            }
            step_scale *= 0.45;
        }
        if !accepted {
            break;
        }
    }

    sess.lig_coords = coords;
    recompute_sculpt_score(sess);
}

/// Real-time 60-FPS 3D mouse tug on `atom_idx` toward `target_pos3d` while bonded neighbors relax smoothly (`ISOLDE` / `Foldit` Pull).
pub fn drag_tug_atom_step(sess: &mut SculptSession, atom_idx: usize, target_pos3d: [f32; 3]) {
    if atom_idx >= sess.lig_coords.len() {
        return;
    }
    if sess.frozen_lig_atoms.get(atom_idx).copied().unwrap_or(false) {
        return;
    }
    sess.tug_frames += 1;
    run_coordinate_relaxation(sess, 4, Some((atom_idx, target_pos3d, 14.0)));
}

/// Foldit `Wiggle`: gradient-based minimization of all unfrozen ligand atoms under current `clashes_weight` and active rubber bands.
pub fn wiggle_minimize(sess: &mut SculptSession, iterations: usize) {
    push_undo_checkpoint(sess);
    sess.moves_count += 1;
    sess.wiggle_count += 1;
    // First apply a small rigid-body + local gradient settle so both macro-translation and internal torsions relax!
    let iters = iterations.clamp(4, 80);
    run_coordinate_relaxation(sess, iters, None);
    record_trajectory_step(sess, &format!("Wiggle ({iters} steps)"));
}

/// Foldit `Shake`: combinatorial `Dunbrack` rotamer repacking of all unfrozen pocket side chains against the current 3D ligand pose.
pub fn shake_pocket_rotamers(sess: &mut SculptSession) {
    push_undo_checkpoint(sess);
    sess.moves_count += 1;
    sess.shake_count += 1;

    for _pass in 0..2 {
        for sc_idx in 0..sess.sidechains.len() {
            if sess.sidechains[sc_idx].frozen {
                continue;
            }
            let mut best_r = sess.sidechains[sc_idx].active_rotamer;
            recompute_sculpt_score(sess);
            let mut best_e = sess.score.steered_objective_kcal;
            for cand_r in 0..4 {
                sess.sidechains[sc_idx].active_rotamer = cand_r;
                recompute_sculpt_score(sess);
                if sess.score.steered_objective_kcal < best_e - 1e-4 {
                    best_e = sess.score.steered_objective_kcal;
                    best_r = cand_r;
                }
            }
            sess.sidechains[sc_idx].active_rotamer = best_r;
        }
    }
    recompute_sculpt_score(sess);
    record_trajectory_step(sess, "Shake (Side-Chain Rotamers)");
}

/// Famous Foldit player barrier-crossing macro (`Blue Fuse` / `Quake`, Khatib et al. 2011):
/// Soften repulsion (`0.22`) -> `Wiggle` -> `Shake` -> ramp (`0.60`) -> `Wiggle` -> restore (`1.00`) -> `Shake` -> `Wiggle`.
pub fn run_blue_fuse_macro(sess: &mut SculptSession) {
    push_undo_checkpoint(sess);
    sess.moves_count += 1;
    let saved_cw = sess.clashes_weight;

    sess.clashes_weight = 0.22;
    run_coordinate_relaxation(sess, 16, None);
    for sc_idx in 0..sess.sidechains.len() {
        if !sess.sidechains[sc_idx].frozen {
            let mut best_r = sess.sidechains[sc_idx].active_rotamer;
            recompute_sculpt_score(sess);
            let mut best_e = sess.score.steered_objective_kcal;
            for r in 0..4 {
                sess.sidechains[sc_idx].active_rotamer = r;
                recompute_sculpt_score(sess);
                if sess.score.steered_objective_kcal < best_e {
                    best_e = sess.score.steered_objective_kcal;
                    best_r = r;
                }
            }
            sess.sidechains[sc_idx].active_rotamer = best_r;
        }
    }

    sess.clashes_weight = 0.60;
    run_coordinate_relaxation(sess, 18, None);

    sess.clashes_weight = 1.00;
    run_coordinate_relaxation(sess, 26, None);
    sess.wiggle_count += 3;
    sess.shake_count += 1;
    sess.clashes_weight = saved_cw;
    recompute_sculpt_score(sess);
    record_trajectory_step(sess, "Blue-Fuse Barrier Macro");
}

/// Attaches a harmonic rubber-band restraint between `lig_atom_idx` and a receptor anchor or water hotspot.
pub fn add_rubber_band_to_receptor(
    sess: &mut SculptSession,
    lig_atom_idx: usize,
    label: String,
    pos: [f32; 3],
    target_dist_a: f32,
) {
    if lig_atom_idx >= sess.mol.atoms.len() {
        return;
    }
    push_undo_checkpoint(sess);
    let bid = sess.next_band_id;
    sess.next_band_id += 1;
    sess.bands.push(RubberBandRestraint {
        band_id: bid,
        lig_atom_idx,
        target: BandAnchorTarget::ReceptorAnchor {
            anchor_idx: bid,
            label: label.clone(),
            pos,
        },
        target_dist_a,
        spring_k_kcal_a2: 3.6,
        enabled: true,
        current_dist_a: dist3(sess.lig_coords[lig_atom_idx], pos),
        strain_kcal: 0.0,
    });
    sess.moves_count += 1;
    recompute_sculpt_score(sess);
    record_trajectory_step(sess, &format!("Add Band -> {label}"));
}

/// Automatically perceives up to 3 high-gain unsatisfied polar H-bond or unhappy-water contacts and attaches smart rubber bands!
pub fn auto_band_polar_contacts(sess: &mut SculptSession) -> usize {
    push_undo_checkpoint(sess);
    let n = sess.mol.atoms.len().min(sess.lig_coords.len());
    let mut cands: Vec<(f32, usize, String, [f32; 3], f32)> = Vec::new();

    for i in 0..n {
        let pol = matches!(sess.mol.atoms[i].elem, Element::N | Element::O | Element::F);
        if pol {
            for ra in &sess.receptor_atoms {
                if ra.is_polar {
                    let d = dist3(sess.lig_coords[i], ra.pos);
                    if (2.95..=5.85).contains(&d) {
                        cands.push((d, i, ra.label.clone(), ra.pos, 2.85));
                    }
                }
            }
            for sc in &sess.sidechains {
                if sc.is_polar {
                    let tip = sc.rotamer_tips[sc.active_rotamer.min(3)];
                    let d = dist3(sess.lig_coords[i], tip);
                    if (2.95..=5.85).contains(&d) {
                        cands.push((d, i, format!("{}:Tip", sc.residue_id), tip, 2.84));
                    }
                }
            }
        } else {
            for wt in &sess.water_targets {
                if wt.dg_excess_kcal > 0.8 && !wt.currently_displaced {
                    let d = dist3(sess.lig_coords[i], wt.pos);
                    if (2.05..=4.85).contains(&d) {
                        cands.push((d - 0.5, i, wt.site_id.clone(), wt.pos, 0.85));
                    }
                }
            }
        }
    }

    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut added = 0_usize;
    let mut used_lig: Vec<usize> = Vec::new();
    for (_, li, lbl, pos, td) in cands {
        if used_lig.contains(&li) {
            continue;
        }
        used_lig.push(li);
        let bid = sess.next_band_id;
        sess.next_band_id += 1;
        sess.bands.push(RubberBandRestraint {
            band_id: bid,
            lig_atom_idx: li,
            target: BandAnchorTarget::ReceptorAnchor {
                anchor_idx: bid,
                label: lbl,
                pos,
            },
            target_dist_a: td,
            spring_k_kcal_a2: 3.5,
            enabled: true,
            current_dist_a: dist3(sess.lig_coords[li], pos),
            strain_kcal: 0.0,
        });
        added += 1;
        if added >= 3 {
            break;
        }
    }
    sess.moves_count += 1;
    recompute_sculpt_score(sess);
    record_trajectory_step(sess, &format!("Auto-Band ({added} restraints)"));
    added
}

/// Drugit Med-Chem In-Pocket Sprout: attaches `warhead` onto `anchor_atom_idx` in 3D, preserving existing coordinates and settling new atoms!
pub fn sprout_in_pocket_rgroup(
    sess: &mut SculptSession,
    anchor_atom_idx: usize,
    warhead_idx: usize,
) -> Result<String, String> {
    let wh = SCULPT_SPROUT_WARHEADS
        .get(warhead_idx)
        .copied()
        .ok_or_else(|| "Invalid warhead index".to_string())?;
    let n_old = sess.mol.atoms.len().min(sess.lig_coords.len());
    if anchor_atom_idx >= n_old {
        return Err("Selected anchor atom out of bounds".to_string());
    }

    let base_smi = if !sess.mol.smiles.trim().is_empty() {
        sess.mol.smiles.trim().to_string()
    } else {
        molecule_to_smiles(&sess.mol)
    };

    // Locate the `anchor_atom_idx`-th heavy atom token in `base_smi`
    let bytes = base_smi.as_bytes();
    let mut ord = 0_usize;
    let mut ins_pos: Option<usize> = None;
    let mut i = 0_usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'[' {
            if let Some(rel) = base_smi[i..].find(']') {
                let end_i = i + rel + 1;
                if ord == anchor_atom_idx {
                    ins_pos = Some(end_i);
                    break;
                }
                ord += 1;
                i = end_i;
                continue;
            }
        } else if (b == b'C' && i + 1 < bytes.len() && bytes[i + 1] == b'l')
            || (b == b'B' && i + 1 < bytes.len() && bytes[i + 1] == b'r')
        {
            let mut end_i = i + 2;
            while end_i < bytes.len() && (bytes[end_i].is_ascii_digit() || bytes[end_i] == b'%') {
                end_i += 1;
            }
            if ord == anchor_atom_idx {
                ins_pos = Some(end_i);
                break;
            }
            ord += 1;
            i = end_i;
            continue;
        } else if matches!(
            b,
            b'B' | b'C' | b'N' | b'O' | b'P' | b'S' | b'F' | b'I' | b'b' | b'c' | b'n' | b'o' | b'p' | b's'
        ) {
            let mut end_i = i + 1;
            while end_i < bytes.len() && (bytes[end_i].is_ascii_digit() || bytes[end_i] == b'%') {
                end_i += 1;
            }
            if ord == anchor_atom_idx {
                ins_pos = Some(end_i);
                break;
            }
            ord += 1;
            i = end_i;
            continue;
        }
        i += 1;
    }

    let cand_smi = if let Some(p) = ins_pos {
        format!("{}({}){}", &base_smi[..p], wh.smiles_branch, &base_smi[p..])
    } else {
        format!("{}({})", base_smi, wh.smiles_branch)
    };

    let mut new_mol = parse_smiles(&cand_smi)
        .or_else(|_| parse_smiles(&format!("{}({})", base_smi, wh.smiles_branch)))
        .map_err(|e| format!("Valence saturated at selected atom ({e})"))?;

    if new_mol.atoms.len() <= n_old {
        return Err("No new heavy atoms added".to_string());
    }
    let canon = molecule_to_smiles(&new_mol);
    new_mol.smiles = canon.clone();
    new_mol.id = sess.mol.id.clone();
    new_mol.name = format!("{}+{}", sess.mol.name, wh.code);

    push_undo_checkpoint(sess);

    // Compute 3D outward sprout vector from `anchor_atom_idx` away from its bonded neighbors!
    let anchor_p = sess.lig_coords[anchor_atom_idx];
    let mut nbr_c = [0.0_f32; 3];
    let mut cnt = 0_usize;
    if let Some(adj) = sess.mol.adj.get(anchor_atom_idx) {
        for &(nb, _) in adj {
            if nb < n_old {
                nbr_c[0] += sess.lig_coords[nb][0];
                nbr_c[1] += sess.lig_coords[nb][1];
                nbr_c[2] += sess.lig_coords[nb][2];
                cnt += 1;
            }
        }
    }
    let out_v = if cnt > 0 {
        let vx = anchor_p[0] - nbr_c[0] / (cnt as f32);
        let vy = anchor_p[1] - nbr_c[1] / (cnt as f32);
        let vz = anchor_p[2] - nbr_c[2] / (cnt as f32);
        let vn = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-4);
        [vx / vn, vy / vn, vz / vn]
    } else {
        [1.0, 0.0, 0.0]
    };

    // Map old coordinates before `anchor_atom_idx + 1`, insert new branch atoms along `out_v`, then append remaining tail atoms!
    let added = new_mol.atoms.len() - n_old;
    let insert_idx = (anchor_atom_idx + 1).min(n_old);
    let mut new_coords: Vec<[f32; 3]> = Vec::with_capacity(new_mol.atoms.len());
    let mut new_frozen: Vec<bool> = Vec::with_capacity(new_mol.atoms.len());

    for k in 0..insert_idx {
        new_coords.push(sess.lig_coords[k]);
        new_frozen.push(sess.frozen_lig_atoms.get(k).copied().unwrap_or(false));
    }
    for k in 0..added {
        let step_r = wh.bond_len_a + (k as f32) * 1.18;
        let lateral = ((k as f32) * 2.09).sin() * 0.48;
        new_coords.push([
            anchor_p[0] + out_v[0] * step_r + lateral * 0.4,
            anchor_p[1] + out_v[1] * step_r - lateral * 0.3,
            anchor_p[2] + out_v[2] * step_r + ((k as f32) * 1.5).cos() * 0.35,
        ]);
        new_frozen.push(false);
    }
    for k in insert_idx..n_old {
        new_coords.push(sess.lig_coords[k]);
        new_frozen.push(sess.frozen_lig_atoms.get(k).copied().unwrap_or(false));
    }

    sess.mol = new_mol;
    sess.lig_coords = new_coords;
    sess.frozen_lig_atoms = new_frozen;
    sess.bands.clear();
    sess.moves_count += 1;
    sess.sprout_count += 1;
    run_coordinate_relaxation(sess, 12, None);
    record_trajectory_step(sess, &format!("Sprout {}", wh.label));
    Ok(canon)
}

/// Exports the sculpted trajectory & score ledger to `.csv`.
pub fn export_sculpt_trajectory_csv(sess: &SculptSession) -> String {
    let mut out = String::from(
        "step,puzzle,pdb_id,action,elapsed_secs,foldit_score_pts,unbanded_dg_kcal,hbonds,clashes,smiles\n",
    );
    for st in &sess.trajectory {
        out.push_str(&format!(
            "{},{},{},\"{}\",{:.1},{:.1},{:+.3},{},{},\"{}\"\n",
            st.step_idx,
            sess.preset.code(),
            sess.pdb_id,
            st.action_label,
            st.elapsed_secs,
            st.foldit_score_pts,
            st.unbanded_dg_kcal,
            st.hbonds,
            st.clashes,
            st.smiles
        ));
    }
    out
}

/// Exports the sculpted 3D receptor + repacked side chains + ligand complex to `.pdb`.
pub fn format_sculpted_complex_pdb(sess: &SculptSession) -> String {
    let mut out = format!(
        "REMARK   1 SAR-ATLAS v3.6.0 FOLDIT / DRUGIT SCULPTED 3D COMPLEX\nREMARK   2 PUZZLE: {} (PDB {}) | SCORE: {:.1} PTS ({})\nREMARK   3 UNBANDED dG = {:+.3} KCAL/MOL | HBONDS = {} | CLASHES = {}\n",
        sess.preset.code(),
        sess.pdb_id,
        sess.score.foldit_score_pts,
        sess.score.medal.label(),
        sess.score.total_unbanded_dg_kcal,
        sess.score.formed_hbonds,
        sess.score.clash_count
    );
    let mut serial = 1_u32;
    for ra in &sess.receptor_atoms {
        let parts: Vec<&str> = ra.label.split(':').collect();
        let res_part = parts.first().copied().unwrap_or("UNK1");
        let atom_nm = parts.get(1).copied().unwrap_or("CA");
        let res_nm = &res_part[..res_part.len().min(3)];
        let res_no: i32 = res_part[res_nm.len()..].parse().unwrap_or(1);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 18.00          {:>2}\n",
            serial,
            atom_nm,
            res_nm,
            res_no,
            ra.pos[0],
            ra.pos[1],
            ra.pos[2],
            ra.elem.symbol()
        ));
        serial += 1;
    }
    for (si, sc) in sess.sidechains.iter().enumerate() {
        let tip = sc.rotamer_tips[sc.active_rotamer.min(3)];
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 14.00          {:>2}\n",
            serial,
            sc.tip_elem.symbol(),
            sc.res_name,
            900 + (si as i32),
            tip[0],
            tip[1],
            tip[2],
            sc.tip_elem.symbol()
        ));
        serial += 1;
    }
    for (idx, &lp) in sess.lig_coords.iter().enumerate() {
        let sym = sess
            .mol
            .atoms
            .get(idx)
            .map(|a| a.elem.symbol())
            .unwrap_or("C");
        let aname = format!("{}{}", sym, (idx + 1) % 100);
        out.push_str(&format!(
            "HETATM{:>5} {:<4} LIG L   1    {:>8.3}{:>8.3}{:>8.3}  1.00 12.00          {:>2}\n",
            serial, aname, lp[0], lp[1], lp[2], sym
        ));
        serial += 1;
    }
    out.push_str("END\n");
    out
}

/// Exports the sculpted 3D ligand pose to V2000 `.sdf` with Foldit score & Drugit property tags.
pub fn format_sculpted_ligand_sdf(sess: &SculptSession) -> String {
    let n = sess.mol.atoms.len().min(sess.lig_coords.len());
    let nb = sess
        .mol
        .bonds
        .iter()
        .filter(|b| b.a < n && b.b < n)
        .count();
    let mut sdf = format!(
        "{}\n  SAR-Atlas v3.6.0 Foldit/Drugit 3D Sculptor\n\n{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
        sess.mol.name, n, nb
    );
    for i in 0..n {
        let p = sess.lig_coords[i];
        let sym = sess.mol.atoms[i].elem.symbol();
        sdf.push_str(&format!(
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            p[0], p[1], p[2], sym
        ));
    }
    for b in &sess.mol.bonds {
        if b.a < n && b.b < n {
            let ord = b.kekule_order.clamp(1, 3);
            sdf.push_str(&format!("{:>3}{:>3}{:>3}  0  0  0  0\n", b.a + 1, b.b + 1, ord));
        }
    }
    sdf.push_str("M  END\n");
    sdf.push_str(&format!(
        "> <Foldit_Score_Pts>\n{:.1}\n\n> <Medal_Tier>\n{}\n\n> <Unbanded_dG_kcal>\n{:+.3}\n\n$$$$ \n",
        sess.score.foldit_score_pts,
        sess.score.medal.label(),
        sess.score.total_unbanded_dg_kcal
    ));
    sdf
}
