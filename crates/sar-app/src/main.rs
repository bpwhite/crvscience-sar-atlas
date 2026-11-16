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

//! `sar-atlas` (`v3.8.0`): Unified CLI, Native ANSI Terminal Console, and 5-Domain `eframe`/`wgpu` Desktop
//! Workbench for small-molecule cheminformatics, interactive 2D molecular sketching, quantitative
//! ADMET, mechanistic oral `PBPK` (`ACAT`, hepatic first-pass `Fg*Fh`, & unbound brain `Kp,uu,brain`),
//! bioreversible prodrug promoiety enumeration (`ester`/`phosphate`/`carbamate`/`oxazolidinone`/`PEPT1`),
//! `Rule-of-3` fragment exit-vector growing & `GE`/`LLE`/`LE` optimization,
//! terminal-pair `RCM`/lactam macrocyclization with 3D preorganization entropy scoring,
//! `SMARTCyp` (`3A4`/`2D6`/`2C9`) regioselective Phase I/II metabolism, physiological `pH 7.4`
//! micro-`pKa`/protomers, 3D `MMFF-lite` + `USR`/`ROCS` overlay, covalent warhead `GSH` tuning,
//! `PROTAC` (`CRBN`/`VHL`) `bRo5` chameleonicity, `ChEMBL`/`PubChem` REST, `SQLite 3`, and `MCS`.

mod gui;
mod tui;
mod wgpu_pass;

use clap::{Parser, Subcommand, ValueEnum};
use sar_core::{
    analyze_electrostatics_esp, export_dataset_esp_screen_csv, export_esp_contacts_csv, export_esp_designs_csv,
    format_esp_multipole_pdb, screen_dataset_electrostatics_esp, ElectrostaticsEspConfig, EspTargetPreset,
    export_public_structure_confidence_csv, fetch_and_assess_public_structure, PublicPdbFetchConfig,
    PublicStructureSource, PublicTargetPreset, auto_band_polar_contacts, export_sculpt_trajectory_csv,
    format_sculpted_complex_pdb, format_sculpted_ligand_sdf, new_sculpt_session, run_blue_fuse_macro,
    shake_pocket_rotamers, sprout_in_pocket_rgroup, wiggle_minimize, SculptPuzzlePreset, SCULPT_SPROUT_WARHEADS,
    analyze_watermap_solvation, export_dataset_watermap_screen_csv, export_watermap_rgroup_designs_csv,
    export_watermap_sites_csv, format_watermap_annotated_pdb, screen_dataset_watermap_solvation,
    WaterMapConfig, WaterMapPreset, export_dataset_hbond_screen_csv, export_hbond_network_edges_csv,
    export_hbond_network_nodes_csv, format_optimized_hbond_complex_pdb, optimize_pocket_hbond_network,
    screen_dataset_hbond_network, HBondNetworkConfig, HBondNetworkPreset, export_induced_fit_ensemble_csv,
    export_repacked_rotamers_csv, format_induced_fit_ensemble_sdf, score_ligand_induced_fit_ensemble,
    screen_dataset_induced_fit_ensemble, InducedFitConfig, InducedFitPreset, design_linked_fragments,
    export_linked_fragments_csv, format_linked_fragments_sdf, FragmentLinkerConfig, FragmentPairPreset,
    append_pipeline_run_sqlite, load_pipeline_ledger_sqlite, load_pipeline_recipe_json, run_discovery_pipeline_with_progress,
    save_pipeline_recipe_json, export_funnel_candidates_csv, export_funnel_stages_csv, format_promoted_survivors_sdf,
    DiscoveryFunnelPreset, DiscoveryPipelineConfig, analyze_torsion_strain_and_locks, export_locked_analogs_csv,
    export_torsion_profiles_csv, format_torsion_poses_sdf, screen_dataset_torsion_strain, BioactivePoseSource,
    export_rbfe_edges_csv, export_rbfe_nodes_csv, run_rbfe_network_campaign, RbfeNetworkConfig,
    RbfeSeriesPreset, RbfeTopologyMode, export_active_learning_csv, run_active_learning_campaign,
    AcquisitionStrategy, ActiveLearningConfig, CandidatePoolSource, QsarKernelMode, SurrogateModelKind,
    analyze_allosteric_network, curated_allosteric_pdb_text, export_allosteric_screen_csv,
    format_allosteric_poses_sdf, score_ligand_allosteric_coupling, screen_dataset_allosteric_pocket,
    CURATED_ALLOSTERIC_SYSTEMS, analyze_dual_pocket_selectivity, curated_offtarget_pdb_text,
    export_mutation_scan_csv, export_selectivity_screen_csv, format_dual_selectivity_poses_sdf,
    scan_pocket_residue_mutations, score_ligand_dual_selectivity, screen_dataset_selectivity,
    CURATED_SELECTIVITY_PAIRS, enumerate_parallel_library, export_parallel_library_csv, format_parallel_library_sdf,
    LibraryPreFilterMode, LibrarySynthesisMode, ParallelLibConfig, CURATED_LIBRARY_SCHEMES,
    analyze_binding_pocket, curated_pocket_pdb_text, export_pocket_complementarity_csv, fetch_rcsb_pdb_text,
    format_pocket_poses_sdf, parse_pdb_structure, score_ligand_in_pocket, screen_dataset_against_pocket,
    CURATED_PDB_POCKETS, evolve_de_novo_molecules, export_de_novo_candidates_csv, format_de_novo_3d_sdf,
    DeNovoConfig, DeNovoGoalPreset, build_lead_campaign_dossier, evaluate_campaign_scorecard,
    export_campaign_dossier_html, export_campaign_scorecard_csv, export_lead_dossier_html,
    export_lead_dossier_svg, CampaignTargetProfile, analyze_covalent_warheads_and_swaps, analyze_cyp_metabolism,
    analyze_pka_and_protomers, build_auto_free_wilson_matrix, build_curated_prodrug_parent,
    build_curated_ro3_seed_molecule, compute_kier_flexibility_phi, design_macrocycles, design_prodrugs,
    export_grown_fragments_csv, export_macrocycles_csv, export_metabolites_csv, export_dataset_pbpk_csv,
    export_pbpk_timecourse_csv, export_prodrugs_csv, format_grown_fragments_sdf, format_macrocycles_3d_sdf,
    format_prodrugs_sdf, simulate_oral_pbpk, summarize_dataset_oral_pbpk, DosingRegimen, PbpkDosingConfig,
    PrandialState, grow_fragment_along_vector, summarize_dataset_fragment_efficiency, summarize_dataset_prodrug_opportunities,
    CURATED_PRODRUG_PARENTS, CURATED_RO3_SEEDS, build_chem_heatmap_rgba, design_protac_series,
    export_protac_candidates_csv, export_warhead_swaps_csv, E3RecruiterKind, compile_smarts,
    compute_2d_coords, evaluate_3d_summary_profile, evaluate_physiological_pka_74, export_3d_overlay_hits_csv,
    format_3d_conformer_sdf, format_3d_overlay_sdf, generate_3d_conformer_ensemble, screen_3d_overlay_library,
    ConformerGenConfig, detect_and_align_series_mcs, dissect_brics, evaluate_admet, evaluate_synthetic_accessibility,
    export_brics_hybrids_csv, export_series_mcs_csv, fetch_chembl_target_series, fetch_pubchem_bioassay_aid,
    fetch_pubchem_compounds_by_identifiers, find_pairwise_mcs, format_sdf_dataset, generate_bioisosteres,
    load_benchmark_26, load_project_auto, match_smarts, molecule_to_smiles, pack_gpu_scene,
    parse_smiles, recombine_brics_leads, render_molecule_svg_with_smarts, save_project_auto,
    screen_pharmacophore_library, synthesize_hts_library, BricsRecombineConfig, ChemblQueryConfig,
    DatasetAnalysis, HeatmapLayer, McsAtomMatchMode, McsConfig, MolecularSketcher, NodeColorMode,
    PharmFeatureKind, SarProjectWorkspace,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "sar-atlas",
    version,
    about = "SAR-Atlas v3.8.0 — 5-Domain Small-Molecule, FBDD, Macrocycle, PROTAC, SMARTCyp, Prodrug & Oral PBPK Cheminformatics Workbench"
)]
struct Cli {
    /// Render an offscreen 1600x1000 GPU screenshot of the default SAR-Atlas workbench to PNG and exit.
    #[arg(long, value_name = "PNG_PATH")]
    screenshot: Option<PathBuf>,

    /// Select which workbench view to capture (1 = Chemical Space & SALI, 2 = RBFE Network, 3 = CYP & Metabolites).
    #[arg(long, value_name = "VIEW", default_value_t = 1)]
    screenshot_view: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Csv,
    Json,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the hardware-instanced `egui` + `wgpu` v3.8.0 5-domain (39-module + `Ctrl+K`) desktop workbench (default).
    Gui {
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long)]
        hts_synthetic: Option<usize>,
        #[arg(long, default_value_t = true)]
        vsync: bool,
        /// Render an offscreen 1600x1000 GPU screenshot to PNG and exit without opening a window.
        #[arg(long, value_name = "PNG_PATH")]
        screenshot: Option<PathBuf>,

        /// Select which workbench view to capture (1 = Chemical Space & SALI, 2 = RBFE Network, 3 = CYP & Metabolites).
        #[arg(long, value_name = "VIEW", default_value_t = 1)]
        screenshot_view: u8,
    },
    /// Launch the standalone zero-dependency ANSI terminal console.
    Tui {
        #[arg(short, long)]
        input: Option<PathBuf>,
    },
    /// Execute the configurable 6-stage molecule discovery cascade (`Pool Expansion -> PAINS/ADMET/SAScore -> 3D Pocket & Shoichet TEU -> Off-Target Selectivity -> Active-Learning QSAR -> Lomap BAR/MLE RBFE & Promotion`).
    #[command(alias = "funnel")]
    Pipeline {
        /// Campaign preset: `auto`, `egfr` (`1M17`), `cox2` (`3LN1`), or `drd2` (`6CM4`).
        #[arg(long, default_value = "egfr")]
        preset: String,
        /// Number of closed-loop `DMTA` rounds (`1..=5`, default `2`).
        #[arg(long, default_value_t = 2)]
        rounds: usize,
        /// Optional custom target `.pdb` file path overriding the preset receptor pocket.
        #[arg(long)]
        target_pdb: Option<PathBuf>,
        /// Optional custom off-target / mutant `.pdb` file path for Stage-4 dual-pocket selectivity.
        #[arg(long)]
        offtarget_pdb: Option<PathBuf>,
        /// Enable Stage-1 `Rule-of-3` `FBDD` fragment exit-vector growing stream.
        #[arg(long, default_value_t = true)]
        fbdd: bool,
        /// Enable Stage-1 `DeLinker` 3D sub-pocket fragment-linking stream.
        #[arg(long, default_value_t = true)]
        frag_link: bool,
        /// Enable Stage-3 `Sherman/Shoichet` flexible side-chain induced-fit Boltzmann ensemble scoring.
        #[arg(long, default_value_t = false)]
        induced_fit: bool,
        /// Enable Stage-3 `Protoss/Reduce` `His`/`Asp`/`Glu` tautomer & bridge-water H-bond network optimization.
        #[arg(long, default_value_t = true)]
        hbond_net: bool,
        /// Enable Stage-3 `WaterMap/IFST` inhomogeneous solvation hotspot displacement scoring.
        #[arg(long, default_value_t = true)]
        watermap: bool,
        /// Enable Stage-3 `Bauer-Mackey` 3D `ESP` & `XED` `sigma-hole` multipole complementarity scoring.
        #[arg(long, default_value_t = true)]
        esp_multipole: bool,
        /// Enable Stage-1 `RCM`/lactam macrocyclization preorganization stream.
        #[arg(long, default_value_t = false)]
        macrocycle: bool,
        /// Require tuned covalent warhead `GSH t_1/2` window in Stage 3.
        #[arg(long, default_value_t = false)]
        covalent_gsh: bool,
        /// Enable Stage-4 `GNM`/`ANM` allosteric cooperativity coupling branch.
        #[arg(long, default_value_t = false)]
        allosteric: bool,
        /// Minimum Bickerton `QED` desirability (`0.0..=0.9`, default `0.30`).
        #[arg(long, default_value_t = 0.30)]
        min_qed: f64,
        /// Maximum Ertl synthetic accessibility `SAScore` (`2.0..=7.5`, default `4.95`).
        #[arg(long, default_value_t = 4.95)]
        max_sascore: f64,
        /// Maximum 3D pocket empirical binding `dG` in `kcal/mol` (`default -5.80`).
        #[arg(long, default_value_t = -5.80)]
        max_pocket_dg: f64,
        /// Maximum `Shoichet` whole-molecule torsional strain in `TEU` (`default 6.50`).
        #[arg(long, default_value_t = 6.50)]
        max_teu: f64,
        /// Minimum dual-pocket selectivity `Delta pKd = pKd(Target) - pKd(OffTarget)` (`default -0.55`).
        #[arg(long, default_value_t = -0.55)]
        min_dpkd: f64,
        /// Maximum promoted survivors sharing any single `Bemis-Murcko` scaffold (`default 3`).
        #[arg(long, default_value_t = 3)]
        max_per_scaffold: usize,
        /// Minimum `ECFP4` dissimilarity (`1 - Tanimoto`) among promoted survivors (`default 0.14`).
        #[arg(long, default_value_t = 0.14)]
        min_dissim: f64,
        /// Number of top Stage-6 survivors to promote into the synthesis cohort (`default 8`).
        #[arg(long, default_value_t = 8)]
        promote_n: usize,
        /// Optional comma-separated custom candidate SMILES injected into Stage 1.
        #[arg(long, default_value = "")]
        smiles: String,
        /// Optional `.json` pipeline recipe file to load (CLI flags override loaded defaults when specified).
        #[arg(long)]
        load_recipe: Option<PathBuf>,
        /// Optional `.json` output path to save the active `DiscoveryPipelineConfig` recipe.
        #[arg(long)]
        save_recipe: Option<PathBuf>,
        /// Optional `SQLite 3` database path to append this run into `pipeline_runs` & `pipeline_promoted_leads`.
        #[arg(long)]
        ledger_sqlite: Option<PathBuf>,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Export only promoted synthesis survivors in `--export-candidates-csv`.
        #[arg(long, default_value_t = false)]
        promoted_only: bool,
        /// Optional `.csv` output path for the 6-stage attrition summary.
        #[arg(long)]
        export_stages_csv: Option<PathBuf>,
        /// Optional `.csv` output path for funnel candidate records.
        #[arg(long)]
        export_candidates_csv: Option<PathBuf>,
        /// Optional `.sdf` output path for 3D bioactive pocket poses of promoted survivors.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Scan `CSD`/`SMARTS` `360 deg` dihedral energy profiles, estimate `Shoichet` `TEU` & `Perola-Charifson` bioactive 3D conformer strain, and design rotatable-bond locking analogs (`ortho`-Me, `IMHB`, `alpha`-Me, small-ring cyclization).
    /// Align 3D sub-pocket fragment exit vectors (`DeLinker` 4-DOF), match across `18` rigid/`sp3`/heterocyclic linkers, and score `Jencks`/`Murray-Verdonk` super-additivity `Delta-G` & `Kabsch` `Core-RMSD`.
    /// Repack flexible binding-pocket `Dunbrack` side-chain rotamers (`Sherman IFD` / `AutoDockFR`) and score ligands across a 5-state `Fischer-Shoichet` Boltzmann receptor ensemble.
    /// Optimize active-site `His`/`Asp`/`Glu` tautomeric/protonation states (`PROPKA 3.1` + `Protoss`), `Asn`/`Gln`/`His` `180 deg` terminal flips (`Reduce`), and tetrahedral `H-O-H` bridge-water networks.
    /// Map active-site `IFST` / `WaterMap` / `GIST` `dH`/`-TdS`/`dG` hydration hotspots and design exit-vector matched displaceable-water R-group analogs.
    /// Run a `Foldit` / `Drugit` 3D ligand & side-chain sculpting puzzle or custom `.pdb` sandbox with rubber-band restraints, `Wiggle`, `Shake`, `Blue-Fuse`, and in-pocket R-group sprouting.
    /// Fetch open public `RCSB PDB` structures (plus `CCD` co-crystal ligand `SMILES`) or `EMBL-EBI AlphaFold DB` models (`UniProt` + `pLDDT` + `PAE`), audit `Karelina-Dror` pocket docking readiness, and export loop-trimmed `.pdb` files.
    /// Map 3D Connolly surface molecular electrostatic potential (`ESP`), `Bauer-Mackey` complementarity (`EC`, `EC_r`), `XED` `sigma-hole` halogen/chalcogen bonds, and `Wheeler-Houk` / `Gallivan-Dougherty` `pi` multipole interactions.
    #[command(aliases = ["sigma-hole", "xed"])]
    Esp {
        /// Target active-site `ESP` benchmark preset: `fxa` (`1EZQ`), `vhl` (`5NVX`), `egfr` (`1M17`), `drd2` (`6CM4`), `kras` (`6OIM`), or `custom`.
        #[arg(long, default_value = "fxa")]
        preset: String,
        /// Optional custom `.pdb` file path when `--preset custom` is selected.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// Interfacial interior dielectric constant (`2.0..=12.0`, default `4.0`).
        #[arg(long, default_value_t = 4.0)]
        dielectric: f64,
        /// Disable off-nucleus `XED` pseudoatoms to compare pure atom-centered Gasteiger charges.
        #[arg(long, default_value_t = false)]
        atom_centered_only: bool,
        /// Optional compound ID or name substring in the loaded dataset (defaults to lead #1).
        #[arg(long)]
        id: Option<String>,
        /// Optional custom query SMILES (overrides `--id` when non-empty).
        #[arg(long, default_value = "")]
        smiles: String,
        /// Screen all dataset compounds for 3D `ESP` & `sigma-hole` complementarity.
        #[arg(long, default_value_t = false)]
        screen_dataset: bool,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for detected `sigma-hole`, chalcogen, & `pi` contacts.
        #[arg(long)]
        export_contacts_csv: Option<PathBuf>,
        /// Optional `.csv` output path for bioisosteric `ESP`-tuning analog designs.
        #[arg(long)]
        export_designs_csv: Option<PathBuf>,
        /// Optional `.csv` output path for dataset-wide `ESP` complementarity screening.
        #[arg(long)]
        export_screen_csv: Option<PathBuf>,
        /// Optional `.pdb` output path containing explicit `XSH`/`XCH`/`XLP`/`XPI` multipole pseudoatoms.
        #[arg(long)]
        export_pdb: Option<PathBuf>,
    },
    /// Fetch open public `RCSB PDB` structures (plus `CCD` co-crystal ligand `SMILES`) or `EMBL-EBI AlphaFold DB` models (`UniProt` + `pLDDT` + `PAE`), audit `Karelina-Dror` pocket docking readiness, and export loop-trimmed `.pdb` files.
    #[command(aliases = ["rcsb", "afdb"])]
    FetchPdb {
        /// 4-character `RCSB PDB` ID (e.g. `1M17`, `5NVX`, `1EZQ`, `3LN1`).
        #[arg(long, default_value = "")]
        pdb_id: String,
        /// `UniProt` accession for public `AlphaFold DB` retrieval (e.g. `P01116`, `P00533`, `P40337`).
        #[arg(long, default_value = "")]
        uniprot: String,
        /// Optional built-in offline benchmark (`kras`, `egfr`, `vhl`, `fxa`, `cox2`) when neither `--pdb-id` nor `--uniprot` is given.
        #[arg(long, default_value = "kras")]
        preset: String,
        /// Optional local `.pdb` file path to audit offline.
        #[arg(long)]
        local_pdb: Option<PathBuf>,
        /// Optional local AlphaFold `PAE` `.json` file path.
        #[arg(long)]
        local_pae: Option<PathBuf>,
        /// Allow deterministic offline benchmark fallback if live HTTP endpoints are unreachable.
        #[arg(long, default_value_t = true)]
        offline_fallback: bool,
        /// Strip disordered residues with `pLDDT` below this threshold when writing `--export-clean-pdb`.
        #[arg(long, default_value_t = 55.0)]
        trim_low_plddt: f32,
        /// Optional `.csv` output path for per-residue `pLDDT` / `B-factor` & pocket readiness rows.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional `.pdb` output path for the sanitized / loop-trimmed receptor structure.
        #[arg(long)]
        export_clean_pdb: Option<PathBuf>,
    },
    /// Run a `Foldit` / `Drugit` 3D ligand & side-chain sculpting puzzle or custom `.pdb` sandbox with rubber-band restraints, `Wiggle`, `Shake`, `Blue-Fuse`, and in-pocket R-group sprouting.
    #[command(aliases = ["drugit", "foldit"])]
    Sculpt {
        /// Challenge puzzle preset: `vhl` (`5NVX`), `egfr` (`1M17`), `fxa` (`1EZQ`), `cox2` (`3LN1`), `kras` (`6OIM`), or `sandbox`.
        #[arg(long, default_value = "vhl")]
        puzzle: String,
        /// Optional custom `.pdb` file when `--puzzle sandbox` is selected.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// Automated optimization recipe to execute: `blue-fuse` (default), `wiggle`, `shake`, or `auto-band`.
        #[arg(long, default_value = "blue-fuse")]
        recipe: String,
        /// Optional in-pocket R-group warhead code (`Cl`, `Me`, `F`, `CN`, `OH`, `OMe`, `CHF2`, `CF3`) to sprout before relaxing.
        #[arg(long, default_value = "")]
        sprout: String,
        /// Anchor atom index (`0`-based) for `--sprout`.
        #[arg(long, default_value_t = 0)]
        anchor_atom: usize,
        /// Optional custom query SMILES in `--puzzle sandbox` mode.
        #[arg(long, default_value = "")]
        smiles: String,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for the sculpting trajectory ledger.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional `.pdb` output path for the sculpted 3D receptor + ligand complex.
        #[arg(long)]
        export_pdb: Option<PathBuf>,
        /// Optional `.sdf` output path for the sculpted 3D ligand conformer.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Map active-site `IFST` / `WaterMap` / `GIST` `dH`/`-TdS`/`dG` hydration hotspots and design exit-vector matched displaceable-water R-group analogs.
    #[command(alias = "ifst")]
    Watermap {
        /// Active-site hydration benchmark preset: `fxa` (`1EZQ`), `egfr` (`1M17`), `cox2` (`3LN1`), `kras` (`6OIM`), or `custom`.
        #[arg(long, default_value = "fxa")]
        preset: String,
        /// Optional custom `.pdb` receptor path when `--preset custom` is used.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// 3D `GIST` Cartesian voxel spacing in Angstrom (`0.50..=1.25`, default `0.75`).
        #[arg(long, default_value_t = 0.75)]
        voxel_spacing: f32,
        /// Minimum excess `dG_site` (`kcal/mol`) threshold for high-energy displaceable hotspots.
        #[arg(long, default_value_t = 1.20)]
        hotspot_dg: f64,
        /// Optional compound ID or name substring in the loaded dataset (defaults to lead #1).
        #[arg(long)]
        id: Option<String>,
        /// Optional custom query SMILES (overrides `--id` when non-empty).
        #[arg(long, default_value = "")]
        smiles: String,
        /// Screen all dataset compounds for `WaterMap` displacement and R-group headroom.
        #[arg(long, default_value_t = false)]
        screen_dataset: bool,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for `IFST` hydration-site `dH`/`-TdS`/`dG` thermodynamics.
        #[arg(long)]
        export_sites_csv: Option<PathBuf>,
        /// Optional `.csv` output path for designed displaceable-water R-group analogs.
        #[arg(long)]
        export_designs_csv: Option<PathBuf>,
        /// Optional `.csv` output path for dataset-wide `WaterMap` screening.
        #[arg(long)]
        export_screen_csv: Option<PathBuf>,
        /// Optional `.pdb` output path with `WME`/`WMS`/`WMD`/`WMB` hydration sites (`B-factor = dG_site`).
        #[arg(long)]
        export_pdb: Option<PathBuf>,
    },
    /// Optimize active-site `His`/`Asp`/`Glu` tautomeric/protonation states (`PROPKA 3.1` + `Protoss`), `Asn`/`Gln`/`His` `180 deg` terminal flips (`Reduce`), and tetrahedral `H-O-H` bridge-water networks.
    #[command(alias = "protoss")]
    HbondNet {
        /// Active-site H-bond benchmark preset: `egfr` (`1M17`), `cox2` (`3LN1`), `drd2` (`6CM4`), `kras` (`6OIM`), or `custom`.
        #[arg(long, default_value = "egfr")]
        preset: String,
        /// Optional custom `.pdb` file path when `--preset custom` is used.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// Target physiological or endosomal `pH` (`4.5..=9.5`, default `7.4`).
        #[arg(long, default_value_t = 7.4)]
        ph: f64,
        /// Optional compound ID or name substring in the loaded dataset (defaults to lead #1).
        #[arg(long)]
        id: Option<String>,
        /// Optional custom query SMILES (overrides `--id` when non-empty).
        #[arg(long, default_value = "")]
        smiles: String,
        /// Screen all compounds in the dataset for active-site H-bond & bridge-water networks.
        #[arg(long, default_value_t = false)]
        screen_dataset: bool,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for variable `His`/`Asp`/`Glu` & bridge-water node states.
        #[arg(long)]
        export_nodes_csv: Option<PathBuf>,
        /// Optional `.csv` output path for directional H-bond & water-relay edges.
        #[arg(long)]
        export_edges_csv: Option<PathBuf>,
        /// Optional `.csv` output path for dataset-wide H-bond network screening.
        #[arg(long)]
        export_screen_csv: Option<PathBuf>,
        /// Optional `.pdb` output path for the protonated/flipped receptor + explicit bridge-water + ligand complex.
        #[arg(long)]
        export_pdb: Option<PathBuf>,
    },
    /// Repack flexible binding-pocket `Dunbrack` side-chain rotamers (`Sherman IFD` / `AutoDockFR`) and score ligands across a 5-state `Fischer-Shoichet` Boltzmann receptor ensemble.
    #[command(alias = "ifd")]
    InducedFit {
        /// Receptor ensemble preset: `egfr` (`1M17`), `drd2` (`6CM4`), `cox2` (`3LN1`), `kras` (`6OIM`), or `custom`.
        #[arg(long, default_value = "egfr")]
        preset: String,
        /// Optional custom `.pdb` file path when `--preset custom` is used.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// Optional compound ID or name substring in the loaded dataset (defaults to lead #1).
        #[arg(long)]
        id: Option<String>,
        /// Optional custom query SMILES (overrides `--id` when non-empty).
        #[arg(long, default_value = "")]
        smiles: String,
        /// Maximum pocket-lining side chains allowed to repack (`2..=12`, default `6`).
        #[arg(long, default_value_t = 6)]
        max_flex: usize,
        /// Screen all compounds in the dataset across the 5-state Boltzmann ensemble.
        #[arg(long, default_value_t = false)]
        screen_dataset: bool,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for dataset ensemble-docking triage or single-ligand summary.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional `.csv` output path for per-state `Dunbrack` repacked side-chain rotamers.
        #[arg(long)]
        export_rotamers_csv: Option<PathBuf>,
        /// Optional `.sdf` output path for 5-state Boltzmann-weighted 3D ligand poses.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Align 3D sub-pocket fragment exit vectors (`DeLinker` 4-DOF), match across `18` rigid/`sp3`/heterocyclic linkers, and score `Jencks`/`Murray-Verdonk` super-additivity `Delta-G` & `Kabsch` `Core-RMSD`.
    #[command(alias = "delinker")]
    FragLink {
        /// Sub-pocket fragment pair preset: `egfr` (`1M17`), `cox2` (`3LN1`), `drd2` (`6CM4`), `bivalent`, `deconstruct`, or `custom`.
        #[arg(long, default_value = "egfr")]
        pair: String,
        /// Optional compound ID or name substring when `--pair deconstruct` is selected.
        #[arg(long)]
        id: Option<String>,
        /// Custom Fragment A SMILES when `--pair custom` is selected.
        #[arg(long, default_value = "COc1cc2ncnc(Cl)c2cc1OC")]
        frag_a: String,
        /// Custom Fragment B SMILES when `--pair custom` is selected.
        #[arg(long, default_value = "Brc1cccc(C#C)c1")]
        frag_b: String,
        /// PDB receptor pocket index (`0 = 1M17 EGFR`, `1 = 6CM4 DRD2`, `2 = 3LN1 COX-2`) for `--pair custom`.
        #[arg(long, default_value_t = 0)]
        pocket_idx: usize,
        /// Ranked 3D exit-vector pair index (`0 = best aligned`).
        #[arg(long, default_value_t = 0)]
        vector_pair: usize,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for the ranked 18-linker super-additivity leaderboard.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional `.sdf` output path for 3D poses of Fragment A, Fragment B, and linked chimeras.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Scan `CSD`/`SMARTS` `360 deg` dihedral energy profiles, estimate `Shoichet` `TEU` & `Perola-Charifson` bioactive 3D conformer strain, and design rotatable-bond locking analogs (`ortho`-Me, `IMHB`, `alpha`-Me, small-ring cyclization).
    #[command(alias = "torsion")]
    TorsionStrain {
        /// Optional compound ID or name substring in the loaded dataset (defaults to lead #1).
        #[arg(long)]
        id: Option<String>,
        /// Optional custom query SMILES (overrides `--id` when non-empty).
        #[arg(long, default_value = "")]
        smiles: String,
        /// Bioactive pose reference: `auto`, `egfr` (`1M17`), `drd2` (`6CM4`), `cox2` (`3LN1`), or `rocs`.
        #[arg(long, default_value = "auto")]
        pose: String,
        /// Screen and triage all compounds in the dataset for `Shoichet` `TEU` bioactive strain.
        #[arg(long, default_value_t = false)]
        screen_dataset: bool,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for per-bond `CSD`/`SMARTS` dihedral scan profiles.
        #[arg(long)]
        export_scan_csv: Option<PathBuf>,
        /// Optional `.csv` output path for conformationally locked rotatable-bond analogs.
        #[arg(long)]
        export_locks_csv: Option<PathBuf>,
        /// Optional `.sdf` output path for Free-Minimum vs Bioactive 3D conformer poses.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Build a `Lomap` `MCS` alchemical perturbation graph, run multi-`lambda` 2-leg (`Complex` vs `Solvent`) `Zwanzig`/`Bennett BAR` `RBFE`, audit `3`-/`4`-cycle closure hysteresis, and solve variance-weighted `MLE` node affinities.
    #[command(alias = "lomap")]
    Rbfe {
        /// Congeneric series preset: `egfr` (`1M17`), `cox2` (`3LN1`), `drd2` (`6CM4`), or `auto`.
        #[arg(long, default_value = "egfr")]
        series: String,
        /// Network topology: `lomap` (Cycle-Closed `MST` + Chords), `star` (Radial Hub), or `dense`.
        #[arg(long, default_value = "lomap")]
        topology: String,
        /// Stratified alchemical `lambda` windows (`5..=21`, default `11`).
        #[arg(long, default_value_t = 11)]
        lambda_windows: usize,
        /// `Lomap` `MCSR` exponential decay constant `beta` (`0.05..=0.25`, default `0.10`).
        #[arg(long, default_value_t = 0.10)]
        beta: f64,
        /// Minimum `Lomap` similarity score (`0.05..=0.80`, default `0.22`).
        #[arg(long, default_value_t = 0.22)]
        min_lomap: f64,
        /// Cycle-closure hysteresis warning threshold in `kcal/mol` (`default 0.80`).
        #[arg(long, default_value_t = 0.80)]
        hysteresis_warn: f64,
        /// Allow formal net-charge-changing edges (`Delta Q != 0`).
        #[arg(long, default_value_t = false)]
        allow_charge_change: bool,
        /// Optional comma-separated prospective analog SMILES to attach into the `RBFE` graph.
        #[arg(long, default_value = "")]
        smiles: String,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for `BAR`/`FEP`/`MLE` perturbation edges.
        #[arg(long)]
        export_edges_csv: Option<PathBuf>,
        /// Optional `.csv` output path for reconciled `MLE` ligand node free energies.
        #[arg(long)]
        export_nodes_csv: Option<PathBuf>,
    },
    /// Fit `ECFP4`/`3D-USRCAT` Gaussian-process & `24`-tree ensemble QSAR surrogates with conformal calibration, Expected Improvement (`EI`) / `UCB`, and Kriging-Believer batch acquisition.
    #[command(alias = "bo-qsar")]
    ActiveLearn {
        /// Similarity kernel: `hybrid` (`ECFP4 + 3D-USRCAT`), `2d` (`ECFP4` Tanimoto), or `3d` (`USRCAT`).
        #[arg(long, default_value = "hybrid")]
        kernel: String,
        /// Surrogate model: `consensus` (`GP + RF`), `gp` (Exact Cholesky `GP`), or `rf` (`24`-tree `RF`).
        #[arg(long, default_value = "consensus")]
        model: String,
        /// Acquisition strategy: `ei` (Expected Improvement), `ucb`, `uncertainty`, or `greedy`.
        #[arg(long, default_value = "ei")]
        acq: String,
        /// Pool source: `prospective` (bioisosteres + 2-step library) or `holdout` (40%/60% dataset split).
        #[arg(long, default_value = "prospective")]
        pool: String,
        /// `3D-USRCAT` kernel blend weight (`0.0..=1.0`, default `0.35`).
        #[arg(long, default_value_t = 0.35)]
        w3d: f64,
        /// `UCB` exploration weight `kappa` (`0.2..=4.0`, default `1.50`).
        #[arg(long, default_value_t = 1.50)]
        kappa: f64,
        /// Kriging-Believer synthesis/screening batch size (`2..=32`, default `10`).
        #[arg(long, default_value_t = 10)]
        batch_size: usize,
        /// Optional comma-separated custom candidate SMILES appended to the unlabeled pool.
        #[arg(long, default_value = "")]
        smiles: String,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Export only the Kriging-Believer selected batch in `--export-csv`.
        #[arg(long, default_value_t = false)]
        batch_only: bool,
        /// Optional `.csv` output path for active-learning predictions & batch picks.
        #[arg(long)]
        export_csv: Option<PathBuf>,
    },
    /// Detect normal-mode cryptic pockets via `GNM`/`ANM` elastic networks and score allosteric-to-orthosteric mechanical strain & cooperativity (`AlloSigMA` / `PRS` / `MWC`).
    #[command(alias = "cryptic")]
    Allosteric {
        /// Curated cryptic/allosteric benchmark (`5D41-EGFR-ALLO`, `6OIM-KRAS-SIIP`, `3K5V-ABL1-MYR`, or `0..=2`).
        #[arg(long, default_value = "5D41-EGFR-ALLO")]
        preset: String,
        /// Optional local `.pdb` structure file overriding `--preset`.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// `GNM` Kirchhoff harmonic contact cutoff in `Angstrom` (`7.5..=15.0`, default `10.0`).
        #[arg(long, default_value_t = 10.0)]
        cutoff: f32,
        /// Detected cryptic pocket index to dock/score (`0` = highest-scoring cryptic site).
        #[arg(long, default_value_t = 0)]
        site: usize,
        /// Optional single query ligand SMILES to score in the cryptic pocket.
        #[arg(long)]
        smiles: Option<String>,
        /// Optional ligand dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for ranked cryptic/allosteric modulators.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 3D MDL V2000 `.sdf` output path for cryptic-pocket-aligned poses.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Profile dual-pocket target vs off-target 3D cavity divergence, differential `WaterMap` hydration hotspots, and single-point residue-mutation `Delta-Delta G` (`ddG_bind`) drug-resistance / paralog shifts.
    Selectivity {
        /// Curated benchmark pair (`COX2-VS-COX1`, `EGFR-WT-VS-T790M`, `DRD2-VS-DRD3`, or `0..=2`).
        #[arg(long, default_value = "COX2-VS-COX1")]
        pair: String,
        /// Swap Target and Off-Target receptors (e.g. optimize mutant-selective `EGFR T790M` over `EGFR WT`!).
        #[arg(long, default_value_t = false)]
        swap: bool,
        /// Optional custom Target `.pdb` file path.
        #[arg(long)]
        target_pdb: Option<PathBuf>,
        /// Optional custom Off-Target `.pdb` file path.
        #[arg(long)]
        offtarget_pdb: Option<PathBuf>,
        /// Pocket extraction sphere radius in `Angstrom` (`7.0..=15.0`, default `10.5`).
        #[arg(long, default_value_t = 10.5)]
        radius: f32,
        /// Optional comma-separated binding-pocket missense mutations to include in `ddG_bind` scan (e.g. `T790M,C797S`).
        #[arg(long, default_value = "")]
        mutations: String,
        /// Optional single query ligand SMILES to inspect in detail.
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID from `--input` for residue-mutation `ddG_bind` scanning.
        #[arg(long)]
        id: Option<String>,
        /// Optional ligand dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for dual-pocket selectivity rankings.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional `.csv` output path for single-point residue-mutation `ddG_bind` table.
        #[arg(long)]
        export_mut_csv: Option<PathBuf>,
        /// Optional 3D MDL V2000 `.sdf` output path for selective poses.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Enumerate 2- and 3-step DNA-encoded (`DEL`) or medicinal-chemistry parallel (`Amide`, `Suzuki`, `Buchwald`, `CuAAC Click`) combinatorial libraries with `MaxMin` `ECFP4` diversity & 3D `PDB Pocket` / `ROCS` pre-filtering.
    #[command(alias = "parallel-lib")]
    DelLib {
        /// Built-in 2/3-step reaction scheme code or index (`0..=5`, e.g. `DEL-3C-AMIDE-SUZUKI-CLICK`, `MEDCHEM-2C-QUINAZOLINE`, `MEDCHEM-2C-AMIDE-CLICK`).
        #[arg(long, default_value = "DEL-3C-AMIDE-SUZUKI-CLICK")]
        scheme: String,
        /// Synthesis format: `del` (on-DNA 3-codon split-and-pool) or `plate` (96-well parallel synthesis).
        #[arg(long, default_value = "auto")]
        mode: String,
        /// Pre-filter & ranking objective: `hybrid` (`Pocket + ROCS`), `pocket`, `rocs`, or `maxmin`.
        #[arg(long, default_value = "hybrid")]
        prefilter: String,
        /// Number of building-block synthons per reaction cycle (`2..=10`, default `6`).
        #[arg(long, default_value_t = 6)]
        per_cycle: usize,
        /// Target `MaxMin` `ECFP4` diverse plate size (`4..=96`, default `24`).
        #[arg(long, default_value_t = 24)]
        plate_size: usize,
        /// Optional local `.pdb` receptor file for 3D pocket complementarity pre-filtering.
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// Built-in PDB pocket preset (`1M17-EGFR`, `6CM4-DRD2`, `3LN1-COX2`).
        #[arg(long, default_value = "1M17-EGFR")]
        preset: String,
        /// Optional custom central bifunctional/trifunctional scaffold SMILES.
        #[arg(long)]
        core_smiles: Option<String>,
        /// Optional reference query compound ID from `--input` for 3D `ROCS` alignment.
        #[arg(long)]
        id: Option<String>,
        /// Optional dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Export only the `MaxMin` diverse cherry-picked plate in `--export-csv`.
        #[arg(long, default_value_t = false)]
        plate_only: bool,
        /// Optional `.csv` output path for enumerated combinatorial library members.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 3D Pocket/ROCS-aligned MDL V2000 `.sdf` output path.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Parse `.pdb` binding cavities, profile buried `WaterMap` hydration sites & receptor H-bond/ionic pharmacophores, and score ligand complementarity (`Vina` + `WaterMap`).
    Pocket {
        /// Optional path to a local `.pdb` coordinate file (e.g. `data/egfr_1m17_pocket.pdb`).
        #[arg(long)]
        pdb: Option<PathBuf>,
        /// Optional 4-character RCSB PDB code to fetch live over HTTPS (e.g. `1M17`, `6CM4`, `3LN1`).
        #[arg(long)]
        rcsb: Option<String>,
        /// Built-in co-crystal PDB preset (`1M17-EGFR`, `6CM4-DRD2`, or `3LN1-COX2`).
        #[arg(long, default_value = "1M17-EGFR")]
        preset: String,
        /// Pocket extraction sphere radius in `Angstrom` (`6.5..=16.0`, default `10.5`).
        #[arg(long, default_value_t = 10.5)]
        radius: f32,
        /// Optional single query ligand SMILES to dock/score (omit to screen entire `--input` dataset).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional single compound ID from `--input` to inspect in detail.
        #[arg(long)]
        id: Option<String>,
        /// Optional ligand dataset `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json`.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Optional `.csv` output path for ranked pocket complementarity scores.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 3D pocket-aligned MDL V2000 `.sdf` output path.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Evolve novel molecules via graph mutations (`GB-GA`/`CReM`), core/peripheral ring hops (`BCP`/`sp3`/heterocycles), and `BRICS` crossovers guided by 3D `ROCS`, 2D pharmacophore, `LE`/`LLE`, and `SAScore`.
    DeNovo {
        /// Optional SMILES string of the seed template (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL553` Erlotinib, `CHEMBL118` Celecoxib). Defaults to `#1` lead.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Evolutionary objective (`scaffold-hop`, `le-lle`, `balanced`, or `cns-sp3`).
        #[arg(long, default_value = "scaffold-hop")]
        goal: String,
        /// Number of evolutionary generations (`1..=20`, default `6`).
        #[arg(long, default_value_t = 6)]
        generations: usize,
        /// Mating population size per generation (`8..=80`, default `28`).
        #[arg(long, default_value_t = 28)]
        pop_size: usize,
        /// Maximum allowed Ertl `SAScore` (`1.0..=8.0`, default `4.8`).
        #[arg(long, default_value_t = 4.8)]
        max_sa: f64,
        /// Deterministic RNG seed for reproducible evolution.
        #[arg(long, default_value_t = 0xDE40_7022)]
        seed: u64,
        /// Optional `.csv` output path for ranked evolved candidates.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 3D `ROCS`-aligned MDL V2000 `.sdf` output path.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Score leads across 6 multi-parameter pillars (`Derringer-Suich` desirability + 12 Stage-Gate `TPP` checks) and export standalone `.html` / `.svg` / `.csv` dossiers.
    Dossier {
        /// Optional SMILES string to score and export directly (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL553` Erlotinib, `CHEMBL54` Haloperidol). Omit for full campaign leaderboard.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Target Product Profile preset (`oral-periph`, `oral-cns`, `covalent-tci`, or `bro5-prodrug`).
        #[arg(long, default_value = "oral-periph")]
        profile: String,
        /// Oral PBPK simulation dose in `mg` (default `50.0`).
        #[arg(long, default_value_t = 50.0)]
        dose_mg: f64,
        /// Dosing regimen (`single`, `qd`, or `bid`).
        #[arg(long, default_value = "single")]
        regimen: String,
        /// Simulate fed (`FeSSIF` bile-micelle) gastrointestinal state.
        #[arg(long, default_value_t = false)]
        fed: bool,
        /// Simulate `BBB` `P-gp`/`BCRP` efflux inhibition.
        #[arg(long, default_value_t = false)]
        inhibit_efflux: bool,
        /// Number of top-ranked deep dossiers to embed when exporting a campaign `.html` (default `3`).
        #[arg(long, default_value_t = 3)]
        top_n: usize,
        /// Optional standalone `.html` dossier output path.
        #[arg(long)]
        export_html: Option<PathBuf>,
        /// Optional standalone multi-panel `.svg` scorecard poster output path (single-lead mode or #1 lead).
        #[arg(long)]
        export_svg: Option<PathBuf>,
        /// Optional campaign scorecard `.csv` output path.
        #[arg(long)]
        export_csv: Option<PathBuf>,
    },
    /// Simulate physiologically based oral absorption (`CAT`/`ACAT`), hepatic first-pass (`Fg*Fh`), `Rodgers-Rowland` `Vss`, and unbound brain `Kp,uu,brain` `RK4` PK curves.
    Pbpk {
        /// Optional SMILES string to simulate directly (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL54` Haloperidol, `CHEMBL554` Lapatinib). Omit for dataset PBPK & `Kp,uu,brain` summary.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Oral dose in `mg` (default `50.0`).
        #[arg(long, default_value_t = 50.0)]
        dose_mg: f64,
        /// Dosing regimen (`single`, `qd`, or `bid`).
        #[arg(long, default_value = "single")]
        regimen: String,
        /// Simulate fed high-fat / bile-micelle (`FeSSIF`) gastrointestinal state.
        #[arg(long, default_value_t = false)]
        fed: bool,
        /// Simulate chemical/genetic `BBB` `P-gp`/`BCRP` efflux inhibition.
        #[arg(long, default_value_t = false)]
        inhibit_efflux: bool,
        /// Optional `.csv` output path (writes `97`-point `RK4` PK time-course CSV or dataset PBPK summary CSV).
        #[arg(long)]
        export_csv: Option<PathBuf>,
    },
    /// Enumerate solubility- & permeability-masking prodrugs (`ester`, `phosphate`, `carbamate`, `oxazolidinone`, `PEPT1`) and `CES1`/`CES2`/`ALP` activation kinetics.
    Prodrug {
        /// Optional SMILES string of the active parent drug (overrides `--archetype` and `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional curated clinical FDA parent archetype (`ARC-OSLT`, `ARC-TEDZ`, `ARC-ACVR`, `ARC-MYCO`, `ARC-VALD`, `ARC-DABI`).
        #[arg(long)]
        archetype: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL118`, `CHEMBL726`, `CHEMBL54`). Omit all for dataset `BCS` & prodrug audit.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Maskable functional-group handle index (`0` = primary prioritized handle).
        #[arg(long, default_value_t = 0)]
        handle: usize,
        /// Optional `.csv` output path for ranked prodrugs (or dataset `BCS` rescue summary).
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 2D MDL V2000 `.sdf` output path for the enumerated prodrug structures.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Grow `Rule-of-3` (`Ro3`) fragment hits along synthetic exit vectors with `GE`, `LLE`, `LE`, and `SILE` optimization.
    FragGrow {
        /// Optional SMILES string of the fragment seed (overrides `--seed` and `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional curated clinical `Ro3` seed code or index (`RO3-7AZA`, `RO3-4AQZ`, `RO3-7DPP`, `RO3-INDZ`, `RO3-PHPY`, `RO3-BZIM`).
        #[arg(long)]
        seed: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL553`, `CHEMBL118`). Omit `--smiles`/`--seed`/`--id` for dataset `Ro3` & `LE`/`LLE` audit.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Exit-vector index (`0` = highest-sociability vector).
        #[arg(long, default_value_t = 0)]
        vector: usize,
        /// Baseline fragment `pAct` (`-log10 M`) when `--smiles` is supplied directly.
        #[arg(long, default_value_t = 4.10)]
        seed_pact: f64,
        /// Grow directly on full `--id` parent without deconstructing `MW > 300 Da` leads to their `Ro3` core.
        #[arg(long, default_value_t = false)]
        raw_parent: bool,
        /// Optional `.csv` output path for ranked grown candidates (or dataset `Ro3`/`LE`/`LLE` summary).
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 2D MDL V2000 `.sdf` output path for the grown fragment-to-lead structures.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Design terminal-pair `RCM` alkene/alkane, macrolactam, `N`-Me lactam, diether & `CuAAC` macrocycles with Kier `Phi` & 3D preorganization entropy (`-T*dS_conf`).
    Macrocycle {
        /// Optional SMILES string of the acyclic parent lead (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL553` Erlotinib, `CHEMBL939` Gefitinib). Omit for dataset macrocyclizability summary.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Terminal-pair rank index (`0` = highest-suitability 3D peripheral pair).
        #[arg(long, default_value_t = 0)]
        pair: usize,
        /// Optional `.csv` output path for the ranked 10-tether macrocycles (or dataset summary).
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 3D MDL V2000 `.sdf` output path for the `MMFF-lite` macrocyclic conformers.
        #[arg(long)]
        export_3d_sdf: Option<PathBuf>,
    },
    /// Predict `SMARTCyp` (`CYP3A4`/`2D6`/`2C9`) regioselective Sites of Metabolism (`SoM`) and enumerate Phase I / II metabolites.
    Metabolism {
        /// Optional SMILES string to analyze directly (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL118`, `DB00477`, `CHEMBL939`). Omit for dataset clearance summary.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Include 2-step sequential `Phase I -> Phase II` (`UGT`/`SULT`) conjugate metabolites.
        #[arg(long, default_value_t = true)]
        sequential_phase2: bool,
        /// Optional `.csv` output path (writes Phase I/II metabolites CSV or dataset hepatic clearance CSV).
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional MDL V2000 `.sdf` output path for the enumerated Phase I/II metabolite structures.
        #[arg(long)]
        export_metabolites_sdf: Option<PathBuf>,
    },
    /// Enumerate heterobifunctional `PROTAC` degraders (`CRBN`/`VHL` x 10 linkers), `bRo5` chameleonicity (`ePSA`/`AB-MPS`), and Hook-Effect equilibria.
    Protac {
        /// Optional SMILES string for the target `POI` ligand (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional `POI` compound ID or name from `--input` (defaults to `CHEMBL939` / Gefitinib).
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// E3 ubiquitin ligase filter (`"All"`, `"CRBN"`, or `"VHL"`).
        #[arg(long, default_value = "All")]
        e3: String,
        /// Perceived `POI` exit-vector index (`0` = highest solvent-exposure rim handle).
        #[arg(long, default_value_t = 0)]
        exit_vector: usize,
        /// Cellular target `[POI]_total` concentration in nM.
        #[arg(long, default_value_t = 100.0)]
        poi_nm: f64,
        /// Cellular `[E3]_total` concentration in nM.
        #[arg(long, default_value_t = 150.0)]
        e3_nm: f64,
        /// Number of top ranked `PROTAC` degraders to print.
        #[arg(long, default_value_t = 15)]
        top: usize,
        /// Optional `.csv` output path for ranked `PROTAC` chimeras (`bRo5`, chameleonicity & Hook-Effect metrics).
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional MDL V2000 `.sdf` output path for the assembled heterobifunctional `PROTAC` molecules.
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Profile covalent electrophile reactivity (`GSH t_1/2` & `k_inact/K_I`) and enumerate 12 reversible/irreversible warhead swaps.
    Warheads {
        /// Optional SMILES string to profile and swap directly (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (e.g. `CHEMBL1173655` / Afatinib). Omit for dataset-wide summary.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Free inhibitor concentration `[I]` in nM (`0.5..=10000.0`) for target-occupancy kinetics.
        #[arg(long, default_value_t = 50.0)]
        conc_nm: f64,
        /// Optional `.csv` output path (writes swapped warhead panel CSV or dataset covalent summary CSV).
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional MDL V2000 `.sdf` output path for the enumerated reversible/irreversible warhead swap series.
        #[arg(long)]
        export_swaps_sdf: Option<PathBuf>,
    },
    /// Generate Distance-Geometry + `MMFF-lite` 3D conformers and run `USR`/`USRCAT` + `ROCS` 3D shape-electrostatic overlay screening.
    #[command(name = "3d-overlay")]
    Overlay3d {
        /// Optional SMILES string for the 3D query molecule (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (defaults to `CHEMBL118` / Celecoxib).
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Number of Distance-Geometry trial conformers to sample and `MMFF-lite`-minimize (`1..=32`).
        #[arg(long, default_value_t = 10)]
        conformers: usize,
        /// Restrict 3D hits to true scaffold hops (distinct Bemis-Murcko framework from query).
        #[arg(long, default_value_t = false)]
        scaffold_hops_only: bool,
        /// Number of top `ROCS` 3D hits to print.
        #[arg(long, default_value_t = 15)]
        top: usize,
        /// Optional `.csv` output path for ranked 3D `ROCS` + `USR`/`USRCAT` virtual screening hits.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional 3D MDL V2000 `.sdf` path exporting the superimposed Query + #1 Hit 3D poses.
        #[arg(long)]
        export_overlay_sdf: Option<PathBuf>,
        /// Optional 3D MDL V2000 `.sdf` path exporting the full minimized conformer ensemble of the query.
        #[arg(long)]
        export_conformers_sdf: Option<PathBuf>,
    },
    /// Enumerate site-specific micro-`pKa` centers, coupled `pH 0-14` protomer charge states (`logD` & `pI`), and `MolVS` tautomers (default `pH 7.4`).
    Pka {
        /// Optional SMILES string to analyze directly (overrides `--id`).
        #[arg(long)]
        smiles: Option<String>,
        /// Optional compound ID or name from `--input` (or benchmark 26). Omit for dataset-wide `pH 7.4` summary.
        #[arg(long)]
        id: Option<String>,
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Target physiological or compartment pH (`0.0..=14.0`, default `7.4`).
        #[arg(long, default_value_t = 7.4)]
        ph: f64,
        /// Optional `.csv` output path for dataset-wide physiological `pH 7.4` micro-`pKa` & protomer metrics.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional MDL V2000 `.sdf` output path for the enumerated protomer microspecies of the target compound.
        #[arg(long)]
        export_protomers_sdf: Option<PathBuf>,
    },
    /// Detect the McGregor/`VF2` Maximum Common Substructure (`MCS`) core across a target series (or compound pair) and 2D-align R-groups.
    Mcs {
        /// Optional input `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Target series scope (`"EGFR"`, `"Dopamine D2"`, `"COX-2"`, or `"All"`).
        #[arg(short, long, default_value = "EGFR")]
        target: String,
        /// Optional pairwise mode: compare `--pair-a <ID>` against `--pair-b <ID>`.
        #[arg(long)]
        pair_a: Option<String>,
        #[arg(long)]
        pair_b: Option<String>,
        /// Enable bioisosteric ring-heteroatom matching (`C/N/O/S` in rings of matching size).
        #[arg(long, default_value_t = false)]
        any_ring_hetero: bool,
        /// Minimum series coverage fraction (`0.40..=1.00`) for multi-molecule MCS discovery.
        #[arg(long, default_value_t = 0.70)]
        threshold: f32,
        /// Optional `.csv` output path for the 2D-aligned R-group & core-superposition table.
        #[arg(long)]
        export_csv: Option<PathBuf>,
        /// Optional MDL V2000 `.sdf` output path with Procrustes-aligned 2D coordinates.
        #[arg(long)]
        export_aligned_sdf: Option<PathBuf>,
    },
    /// Cross dissected `BRICS` synthons across active leads and rank novel virtual hybrids by `Free-Wilson pAct`, `QED`, and `SAScore`.
    BricsRecombine {
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(short, long, default_value = "All")]
        target: String,
        #[arg(long, default_value_t = 6.5)]
        min_pact: f64,
        #[arg(long, default_value_t = 5.5)]
        max_sa: f64,
        #[arg(long, default_value_t = 0.20)]
        min_qed: f64,
        #[arg(long, default_value_t = true)]
        three_component: bool,
        #[arg(long, default_value_t = false)]
        cross_target: bool,
        #[arg(long, default_value_t = 20)]
        top: usize,
        #[arg(long)]
        export_csv: Option<PathBuf>,
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Compute Ertl & Schuffenhauer (2009) `SAScore` breakdown and Degen et al. (2008) `BRICS` retrosynthetic disconnection.
    Retrosynth {
        #[arg(long)]
        smiles: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long)]
        export_csv: Option<PathBuf>,
    },
    /// Map 2D topological pharmacophore features & 3-point distance triangles and rank alignment / scaffold-hop hits.
    Pharmacophore {
        #[arg(long)]
        smiles: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        scaffold_hops_only: bool,
        #[arg(long, default_value_t = 15)]
        top: usize,
        #[arg(long)]
        export_csv: Option<PathBuf>,
    },
    /// Fetch a live target series from EMBL-EBI `ChEMBL` REST API, curate/desalt, and optionally save to `.sqlite`, `.json`, or `.sdf`.
    FetchChembl {
        #[arg(long, default_value = "CHEMBL203")]
        target_id: String,
        #[arg(long, default_value = "EGFR")]
        label: String,
        #[arg(long, default_value_t = 6.5)]
        min_pchembl: f64,
        #[arg(long, default_value_t = 80)]
        limit: usize,
        #[arg(long, default_value = "B")]
        assay_type: String,
        #[arg(long)]
        save_project: Option<PathBuf>,
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Fetch active compounds from NCBI `PubChem` `PUG-REST` by BioAssay `--aid` or `--compounds` name/CID list.
    FetchPubchem {
        #[arg(long)]
        aid: Option<u64>,
        #[arg(long)]
        compounds: Option<String>,
        #[arg(long, default_value = "PubChem-Target")]
        label: String,
        #[arg(long, default_value_t = 60)]
        limit: usize,
        #[arg(long)]
        save_project: Option<PathBuf>,
    },
    /// Save an input `.smi`/`.csv`/`.sdf` dataset as a normalized `.sqlite` database or `.saratlas.json` workspace.
    SaveProject {
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(short, long, default_value = "sar_campaign.sqlite")]
        output: PathBuf,
        #[arg(long, default_value = "SAR-Campaign")]
        name: String,
    },
    /// Inspect a saved `.sqlite` or `.saratlas.json` workspace and print top Pareto/ADMET leads.
    InspectProject {
        #[arg(short, long)]
        project: PathBuf,
    },
    /// Evaluate any SMILES analog (`QED`, `CNS-MPO`, `ADMET`, `SAScore`, `PAINS`, `Free-Wilson pAct`, & `SALI` cliffs).
    SketchEval {
        #[arg(short, long)]
        smiles: String,
        #[arg(short, long)]
        input: Option<PathBuf>,
    },
    /// Print quantitative `ADMET` (`ESOL LogS`, `BOILED-Egg` GI/BBB, `P-gp`, `3/75`, `4/400`, `hERG`) table.
    Admet {
        #[arg(short, long)]
        input: Option<PathBuf>,
    },
    /// Enumerate & multiparameter-rank tactical Meanwell (2011) bioisosteric analogs of `--smiles` or `--id`.
    Bioisosteres {
        #[arg(long)]
        smiles: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[arg(short, long)]
        input: Option<PathBuf>,
    },
    /// Screen a dataset against a Daylight SMARTS substructure query (`--pattern`) in parallel.
    Smarts {
        #[arg(short, long)]
        pattern: String,
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long)]
        export_sdf: Option<PathBuf>,
    },
    /// Solve a core-aligned R1/R2/R3 Free-Wilson Ridge QSAR matrix & enumerate virtual analogs.
    Rgroup {
        #[arg(short, long)]
        core: Option<String>,
        #[arg(short, long)]
        target: Option<String>,
        #[arg(long, default_value_t = 3)]
        slots: usize,
        #[arg(long, default_value_t = 0.15)]
        lambda: f64,
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long)]
        export_csv: Option<PathBuf>,
    },
    /// Compute physicochemical descriptors, `ADMET`, `SAScore`, `QED`, `CNS-MPO`, `PAINS` alerts, and Pareto leads.
    Analyze {
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Csv)]
        format: OutputFormat,
        #[arg(long, default_value_t = false)]
        hide_pains: bool,
    },
    /// Rank top intra-target Guha-Van Drie `SALI` activity-cliff pairs.
    Cliffs {
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long, default_value_t = 0.30)]
        min_tanimoto: f32,
        #[arg(long, default_value_t = 6.0)]
        min_sali: f32,
        #[arg(long, default_value_t = 15)]
        top: usize,
    },
    /// Discover Hussain-Rea Matched Molecular Pairs (`MMP`) and bioisosteric core hops.
    Mmp {
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
    /// Render a 2D skeletal publication `.svg` vector diagram with optional SMARTS highlighting.
    ExportSvg {
        #[arg(long)]
        smiles: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[arg(long)]
        smarts: Option<String>,
        #[arg(short, long, default_value = "molecule_2d.svg")]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        light: bool,
    },
    /// Export sanitized 2D coordinates and SD bioactivity tags as a valid MDL V2000 `.sdf` file.
    ExportSdf {
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(short, long, default_value = "sar_export.sdf")]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        pareto_only: bool,
    },
    /// Run multi-core `Rayon` `ECFP4` + `Pharmacophore` + `SAScore/BRICS` + `ADMET` + `Pod` GPU benchmark.
    Bench {
        #[arg(short, long, default_value_t = 50_000)]
        compounds: usize,
    },
}

fn load_dataset_or_project(
    input: Option<&PathBuf>,
    min_tanimoto: f32,
    sali_cutoff: f32,
) -> DatasetAnalysis {
    if let Some(p) = input {
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "sqlite" | "sqlite3" | "db" | "json") {
            match load_project_auto(p) {
                Ok(sess) => return sess.ds,
                Err(err) => {
                    eprintln!("Error loading project {}: {err}", p.display());
                    std::process::exit(1);
                }
            }
        }
        match DatasetAnalysis::from_file(p, min_tanimoto, sali_cutoff) {
            Ok(ds) => return ds,
            Err(err) => {
                eprintln!("Error reading {}: {err}", p.display());
                std::process::exit(1);
            }
        }
    }
    DatasetAnalysis::from_molecules(load_benchmark_26(), min_tanimoto, sali_cutoff)
}

fn main() {
    let cli = Cli::parse();

    if let Some(out_path) = cli.screenshot {
        let ds = load_dataset_or_project(None, 0.30, 6.0);
        if let Err(err) = gui::run_screenshot(ds, &out_path, cli.screenshot_view) {
            eprintln!("Headless screenshot error: {err}");
            std::process::exit(1);
        }
        return;
    }

    match cli.command.unwrap_or(Commands::Gui {
        input: None,
        hts_synthetic: None,
        vsync: true,
        screenshot: None,
        screenshot_view: 1,
    }) {
        Commands::Gui {
            input,
            hts_synthetic,
            vsync,
            screenshot,
            screenshot_view,
        } => {
            let ds = if let Some(n) = hts_synthetic {
                DatasetAnalysis::from_molecules(synthesize_hts_library(n.max(10)), 0.45, 8.0)
            } else {
                load_dataset_or_project(input.as_ref(), 0.30, 6.0)
            };
            if let Some(out_path) = screenshot {
                if let Err(err) = gui::run_screenshot(ds, &out_path, screenshot_view) {
                    eprintln!("Headless screenshot error: {err}");
                    std::process::exit(1);
                }
            } else if let Err(err) = gui::run_gui(ds, vsync) {
                eprintln!("GUI error: {err}");
                std::process::exit(1);
            }
        }
        Commands::Tui { input } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if let Err(err) = tui::run_tui(&ds) {
                eprintln!("TUI error: {err}");
                std::process::exit(1);
            }
        }
        Commands::Pipeline {
            preset,
            rounds,
            target_pdb,
            offtarget_pdb,
            fbdd,
            frag_link,
            induced_fit,
            hbond_net,
            watermap,
            esp_multipole,
            macrocycle,
            covalent_gsh,
            allosteric,
            min_qed,
            max_sascore,
            max_pocket_dg,
            max_teu,
            min_dpkd,
            max_per_scaffold,
            min_dissim,
            promote_n,
            smiles,
            load_recipe,
            save_recipe,
            ledger_sqlite,
            input,
            promoted_only,
            export_stages_csv,
            export_candidates_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let funnel_preset = match preset.to_ascii_lowercase().as_str() {
                "auto" | "dataset" => DiscoveryFunnelPreset::AutoDatasetCampaign,
                "cox" | "cox2" | "cox-2" | "3ln1" => DiscoveryFunnelPreset::Cox2IsoformSelective,
                "drd2" | "d2" | "6cm4" => DiscoveryFunnelPreset::Drd2CnsBbbSelective,
                _ => DiscoveryFunnelPreset::EgfrKinasePanMutant,
            };
            let custom_input_smiles: Vec<String> = smiles
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();

            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let mut cfg = if let Some(ref rp) = load_recipe {
                load_pipeline_recipe_json(rp).expect("Failed to load --load-recipe JSON")
            } else {
                DiscoveryPipelineConfig {
                    preset: funnel_preset,
                    dmta_rounds: rounds,
                    custom_target_pdb_path: target_pdb.map(|p| p.to_string_lossy().to_string()),
                    custom_offtarget_pdb_path: offtarget_pdb.map(|p| p.to_string_lossy().to_string()),
                    include_fbdd_fragment_growing: fbdd,
                    include_fbdd_fragment_linking: frag_link,
                    include_macrocyclization: macrocycle,
                    require_covalent_warhead_window: covalent_gsh,
                    enable_allosteric_coupling_branch: allosteric,
                    min_qed,
                    max_sa_score: max_sascore,
                    max_pocket_dg_kcal: max_pocket_dg,
                    enable_induced_fit_ensemble: induced_fit,
                    optimize_pocket_hbond_network: hbond_net,
                    enable_watermap_solvation: watermap,
                    enable_esp_multipole_scoring: esp_multipole,
                    max_total_strain_teu: max_teu,
                    min_selectivity_delta_pkd: min_dpkd,
                    max_per_murcko_scaffold: max_per_scaffold,
                    min_promoted_ecfp4_dissim: min_dissim,
                    final_promote_count: promote_n,
                    custom_input_smiles: custom_input_smiles.clone(),
                    ..DiscoveryPipelineConfig::default()
                }
            };
            if !custom_input_smiles.is_empty() {
                cfg.custom_input_smiles = custom_input_smiles;
            }
            if let Some(ref sp_json) = save_recipe {
                save_pipeline_recipe_json(sp_json, &cfg)
                    .expect("Failed to write --save-recipe JSON");
                println!("Saved DiscoveryPipelineConfig recipe JSON -> {}", sp_json.display());
            }

            let rep = run_discovery_pipeline_with_progress(&ds, &cfg, |upd| {
                if upd.stage_num == 1 || upd.stage_num == 6 {
                    println!("  [{:>3.0}%] {}", upd.fraction_complete * 100.0, upd.stage_label);
                }
            });

            println!(
                "=== SAR-Atlas v3.8.0 Multi-Round Closed-Loop Discovery Funnel ==="
            );
            println!(
                "Recipe & PDBs     : {} | {} (PDB {} | {}) | {}",
                rep.config.recipe_name,
                rep.resolved_target,
                rep.target_pocket_pdb,
                rep.selectivity_pair_code,
                rep.pdb_source_note
            );
            println!(
                "Final Round Yield : Pool = {} -> Stage 4 = {} -> RBFE = {} -> Promoted = {} across {} Murcko Scaffolds ({:.1}% yield, {:.1} ms)",
                rep.initial_pool_size,
                rep.stage4_survivor_count,
                rep.rbfe_finalist_count,
                rep.promoted_survivor_count,
                rep.unique_promoted_scaffolds,
                rep.overall_yield_pct,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            for r in &rep.round_history {
                println!(
                    "  [DMTA Round {}/{}] Seeds: [{}] | Pool {} -> Promoted {} ({} Murcko) | Best RBFE dG {:+.2} kcal | Mean pAct {:.2} | Mean LLE {:.2} | Q^2 {:.2}",
                    r.round,
                    rep.round_history.len(),
                    r.seed_ids.join(", "),
                    r.pool_size,
                    r.promoted_count,
                    r.unique_promoted_scaffolds,
                    r.best_rbfe_dg_kcal,
                    r.mean_promoted_pact,
                    r.mean_promoted_lle,
                    r.surrogate_loo_q2
                );
            }
            println!(
                "
{:<6} {:<34} {:>6} {:>6} {:>6} {:>8} {:>9}  {}",
                "Stage", "Stage Gate", "Input", "Pass", "Drop", "Pass%", "CumYield%", "Criteria"
            );
            for s in &rep.stage_summaries {
                println!(
                    "S{:<5} {:<34} {:>6} {:>6} {:>6} {:>7.1}% {:>8.1}%  {}",
                    s.stage_num,
                    s.stage_title,
                    s.input_count,
                    s.passed_count,
                    s.dropped_count,
                    s.stage_pass_pct,
                    s.cumulative_yield_pct,
                    s.gate_criteria
                );
            }
            println!(
                "
{:<5} {:<10} {:<12} {:<11} {:>4} {:>5} {:>8} {:>5} {:>6} {:>7} {:>8} {:>5}  {}",
                "Rank", "Verdict", "ID", "Stream", "Rnd", "QED", "Pock_dG", "TEU", "dpKd", "AL_pAct", "RBFE_dG", "Score", "Murcko / Status"
            );
            for c in rep.candidates.iter().take(promote_n.max(14).min(28)) {
                let verd = if c.promoted_survivor { "PROMOTED" } else { "FILTERED" };
                println!(
                    "#{:<4} {:<10} {:<12} {:<11} R{:<3} {:>5.2} {:>+8.2} {:>5.2} {:>+6.2} {:>7.2} {:>+8.2} {:>5.1}  {} | {}",
                    c.rank,
                    verd,
                    c.candidate_id,
                    c.source_stream.short_label(),
                    c.dmta_round_born,
                    c.qed,
                    c.pocket_dg_kcal.unwrap_or(0.0),
                    c.total_strain_teu.unwrap_or(0.0),
                    c.selectivity_delta_pkd.unwrap_or(0.0),
                    c.al_predicted_pact.unwrap_or(0.0),
                    c.rbfe_mle_dg_kcal.unwrap_or(0.0),
                    c.composite_funnel_score,
                    c.murcko_scaffold,
                    c.status_summary
                );
            }
            if let Some(ref db_p) = ledger_sqlite {
                let run_id = append_pipeline_run_sqlite(db_p, &rep)
                    .expect("Failed to append SQLite pipeline ledger");
                let hist = load_pipeline_ledger_sqlite(db_p).unwrap_or_default();
                println!(
                    "Appended run #{} to SQLite pipeline ledger {} ({} total logged runs).",
                    run_id,
                    db_p.display(),
                    hist.len()
                );
            }
            if let Some(ref sp) = export_stages_csv {
                std::fs::write(sp, export_funnel_stages_csv(&rep))
                    .expect("Failed to write funnel stages CSV");
                println!("Wrote 6-stage attrition summary CSV -> {}", sp.display());
            }
            if let Some(ref cp) = export_candidates_csv {
                std::fs::write(cp, export_funnel_candidates_csv(&rep, promoted_only))
                    .expect("Failed to write funnel candidates CSV");
                println!("Wrote funnel candidates CSV -> {}", cp.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_promoted_survivors_sdf(&rep))
                    .expect("Failed to write promoted survivors 3D SDF");
                println!("Wrote promoted survivors 3D poses SDF -> {}", sdf_p.display());
            }
        }
        Commands::Esp {
            preset,
            pdb,
            dielectric,
            atom_centered_only,
            id,
            smiles,
            screen_dataset,
            input,
            export_contacts_csv,
            export_designs_csv,
            export_screen_csv,
            export_pdb,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let pr = if pdb.is_some() || preset.eq_ignore_ascii_case("custom") {
                EspTargetPreset::CustomPdbEspComplementarity
            } else {
                match preset.trim().to_ascii_lowercase().as_str() {
                    "vhl" | "5nvx" => EspTargetPreset::Vhl5nvxThiazoleChalcogenLock,
                    "egfr" | "1m17" => EspTargetPreset::Egfr1m17AnilineHalogenAndHingeEsp,
                    "drd2" | "6cm4" => EspTargetPreset::Drd26cm4CationPiAromaticCage,
                    "kras" | "6oim" => EspTargetPreset::Kras6oimFluorinatedPiGroove,
                    _ => EspTargetPreset::FactorXa1ezqHalogenAndS4PiBox,
                }
            };
            let cfg = ElectrostaticsEspConfig {
                preset: pr,
                custom_pdb_path: pdb.map(|p| p.to_string_lossy().to_string()),
                interface_dielectric: dielectric.clamp(1.5, 20.0),
                enable_xed_multipoles: !atom_centered_only,
                generate_esp_designs: true,
            };

            let query_mol = if !smiles.trim().is_empty() {
                let mut m = parse_smiles(smiles.trim()).expect("Invalid --smiles");
                m.id = "ESP-QUERY".to_string();
                m.name = "Custom ESP Query".to_string();
                m
            } else {
                id.as_deref()
                    .and_then(|q| {
                        ds.mols.iter().find(|m| {
                            m.id.eq_ignore_ascii_case(q)
                                || m.name.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
                        })
                    })
                    .or_else(|| ds.mols.first())
                    .cloned()
                    .expect("Dataset is empty")
            };

            let rep = analyze_electrostatics_esp(&query_mol, Some(&ds), &cfg);
            println!(
                "=== SAR-Atlas v3.8.0 3D ESP, Sigma-Hole & Pi-Multipole Complementarity Studio ==="
            );
            println!(
                "Target & Lead : {} (PDB {}) | {} ({})",
                rep.preset.label(),
                rep.pdb_id,
                rep.compound_name,
                rep.compound_id
            );
            println!(
                "Bauer-Mackey  : EC Integral = {:+.3} | EC_r (Pearson) = {:+.3} | EC_rho (Spearman) = {:+.3} | Fav = {:.1}% | Clash = {:.1}%",
                rep.bauer_mackey_ec_integral,
                rep.bauer_mackey_ec_pearson_r,
                rep.bauer_mackey_ec_spearman_rho,
                rep.esp_favorable_surface_pct,
                rep.esp_clash_surface_pct
            );
            println!(
                "Multipole dG  : Vs,max = {:+.1} | Vs,min = {:+.1} kcal/(mol*e) | XB = {} | ChB/Lock = {} | Pi/CatPi = {} | Total Mult dG = {:+.2} -> Refined dG = {:+.2} kcal/mol",
                rep.ligand_vs_max_kcal_e,
                rep.ligand_vs_min_kcal_e,
                rep.halogen_bond_count,
                rep.chalcogen_bond_and_lock_count,
                rep.pi_and_cation_pi_count,
                rep.total_multipole_dg_kcal,
                rep.esp_refined_pocket_dg_kcal
            );
            println!(
                "
{:<24} {:<10} {:<18} {:>7} {:>8} {:>9}  {}",
                "Interaction", "LigSite", "Partner", "Dist_A", "Angle", "Energy", "Rationale"
            );
            for c in &rep.contacts {
                println!(
                    "{:<24} {:<10} {:<18} {:>7.2} {:>8.1} {:>+9.2}  {}",
                    c.kind.short_label(),
                    c.ligand_site_label,
                    c.partner_label,
                    c.distance_a,
                    c.angle_deg,
                    c.energy_kcal,
                    c.rationale
                );
            }
            if !rep.esp_designs.is_empty() {
                println!(
                    "
{:<5} {:<28} {:>8} {:>8} {:>9} {:>7}  {}",
                    "Rank", "Modification", "BaseECr", "PredECr", "ddG_Mult", "Score", "SMILES"
                );
                for d in &rep.esp_designs {
                    println!(
                        "#{:<4} {:<28} {:>+8.3} {:>+8.3} {:>+9.2} {:>7.1}  {}",
                        d.rank,
                        d.modification_label,
                        d.baseline_ec_r,
                        d.predicted_ec_r,
                        d.ddg_multipole_kcal,
                        d.design_score,
                        d.designed_smiles
                    );
                }
            }

            if screen_dataset || export_screen_csv.is_some() {
                let rows = screen_dataset_electrostatics_esp(&ds, &cfg);
                if screen_dataset {
                    println!(
                        "
{:<5} {:<12} {:<22} {:>7} {:>7} {:>4} {:>4} {:>4} {:>9} {:>10}",
                        "Rank", "ID", "Name", "EC_Int", "EC_r", "XB", "ChB", "Pi", "Mult_dG", "Refined_dG"
                    );
                    for r in rows.iter().take(20) {
                        println!(
                            "#{:<4} {:<12} {:<22} {:>+7.3} {:>+7.3} {:>4} {:>4} {:>4} {:>+9.2} {:>+10.2}",
                            r.rank,
                            r.compound_id,
                            r.compound_name,
                            r.ec_integral,
                            r.ec_pearson_r,
                            r.halogen_bonds,
                            r.chalcogen_contacts,
                            r.pi_contacts,
                            r.total_multipole_dg_kcal,
                            r.esp_refined_pocket_dg_kcal
                        );
                    }
                }
                if let Some(ref sp) = export_screen_csv {
                    std::fs::write(sp, export_dataset_esp_screen_csv(&rows))
                        .expect("Failed to write dataset ESP screen CSV");
                    println!("Wrote dataset ESP complementarity CSV -> {}", sp.display());
                }
            }
            if let Some(ref cp) = export_contacts_csv {
                std::fs::write(cp, export_esp_contacts_csv(&rep))
                    .expect("Failed to write ESP contacts CSV");
                println!("Wrote multipole contacts CSV -> {}", cp.display());
            }
            if let Some(ref dp) = export_designs_csv {
                std::fs::write(dp, export_esp_designs_csv(&rep))
                    .expect("Failed to write ESP designs CSV");
                println!("Wrote ESP bioisostere designs CSV -> {}", dp.display());
            }
            if let Some(ref pp) = export_pdb {
                std::fs::write(pp, format_esp_multipole_pdb(&rep))
                    .expect("Failed to write ESP multipole PDB");
                println!("Wrote XED multipole-annotated PDB -> {}", pp.display());
            }
        }
        Commands::FetchPdb {
            pdb_id,
            uniprot,
            preset,
            local_pdb,
            local_pae,
            offline_fallback,
            trim_low_plddt,
            export_csv,
            export_clean_pdb,
        } => {
            let pr = match preset.trim().to_ascii_lowercase().as_str() {
                "egfr" | "1m17" => PublicTargetPreset::EgfrRcsb1m17ExperimentalHolo,
                "vhl" | "5nvx" | "p40337" => PublicTargetPreset::VhlAfP40337DisorderedTailTrim,
                "fxa" | "1ezq" => PublicTargetPreset::FactorXaRcsb1ezqExperimentalHolo,
                "cox2" | "3ln1" => PublicTargetPreset::Cox2Rcsb3ln1ExperimentalHolo,
                _ => PublicTargetPreset::KrasAfP01116SwitchIiOcclusion,
            };
            let src = if local_pdb.is_some() {
                PublicStructureSource::LocalPdbAndPaeFiles
            } else if !uniprot.trim().is_empty() {
                PublicStructureSource::AlphaFoldPublicDb
            } else if !pdb_id.trim().is_empty() {
                PublicStructureSource::RcsbExperimentalPdb
            } else {
                PublicStructureSource::OfflineBenchmarkPreset
            };
            let cfg = PublicPdbFetchConfig {
                source: src,
                preset: pr,
                rcsb_pdb_id: pdb_id.trim().to_string(),
                uniprot_accession: uniprot.trim().to_string(),
                local_pdb_path: local_pdb.map(|p| p.to_string_lossy().to_string()),
                local_pae_json_path: local_pae.map(|p| p.to_string_lossy().to_string()),
                allow_offline_fallback: offline_fallback,
                trim_low_plddt_cutoff: trim_low_plddt.clamp(30.0, 85.0),
            };

            let rep = fetch_and_assess_public_structure(&cfg)
                .expect("Public structure fetch & readiness audit failed");
            println!(
                "=== SAR-Atlas v3.8.0 Public RCSB PDB & AlphaFold DB Pocket Readiness Studio ==="
            );
            println!(
                "Structure     : {} | {} | Method: {}{}",
                rep.accession_code,
                rep.provenance_badge,
                rep.experimental_method,
                rep.resolution_a
                    .map(|r| format!(" ({r:.2} A)"))
                    .unwrap_or_default()
            );
            println!(
                "Confidence    : Global pLDDT = {:.1} (>=90: {:.1}% | 70-90: {:.1}% | 50-70: {:.1}% | <50: {:.1}%)",
                rep.global_mean_plddt,
                rep.pct_very_high_90,
                rep.pct_confident_70_90,
                rep.pct_low_50_70,
                rep.pct_disordered_lt50
            );
            println!(
                "Pocket Audit  : Volume = {:.0} A^3 | DScore = {:.2} | Pocket pLDDT = {:.1} (min {:.1}) | Pocket PAE = {:.2} A",
                rep.pocket_volume_a3,
                rep.pocket_druggability_dscore,
                rep.pocket_mean_plddt,
                rep.pocket_min_plddt,
                rep.pocket_mean_pae_a
            );
            println!(
                "Readiness     : {:.1}/100 -> {} | Recommended: {}",
                rep.readiness_score_0_100,
                rep.verdict.label(),
                rep.recommended_next_module
            );
            if !rep.apo_occluding_residues.is_empty() {
                println!(
                    "Apo Occluders : {}",
                    rep.apo_occluding_residues.join(", ")
                );
            }
            if !rep.disordered_intrusion_residues.is_empty() {
                println!(
                    "Loop Intruders: {} (Trimmed {} atoms at pLDDT < {:.0})",
                    rep.disordered_intrusion_residues.join(", "),
                    rep.trimmed_removed_atoms,
                    cfg.trim_low_plddt_cutoff
                );
            }
            if !rep.extracted_ligands.is_empty() {
                println!("
Co-Crystallized Ligands Extracted:");
                for lg in &rep.extracted_ligands {
                    println!(
                        "  [{}] {} ({} HA, MW {:.1}, cLogP {:.2}) | SMILES: {}",
                        lg.comp_id, lg.chemical_name, lg.heavy_atom_count, lg.mw, lg.clogp, lg.smiles
                    );
                }
            }
            if let Some(ref cp) = export_csv {
                std::fs::write(cp, export_public_structure_confidence_csv(&rep))
                    .expect("Failed to write pLDDT/readiness CSV");
                println!("Wrote per-residue confidence & readiness CSV -> {}", cp.display());
            }
            if let Some(ref pp) = export_clean_pdb {
                std::fs::write(pp, &rep.trimmed_clean_pdb_text)
                    .expect("Failed to write clean receptor PDB");
                println!(
                    "Wrote sanitized receptor PDB ({} disordered atoms trimmed) -> {}",
                    rep.trimmed_removed_atoms,
                    pp.display()
                );
            }
        }
        Commands::Sculpt {
            puzzle,
            pdb,
            recipe,
            sprout,
            anchor_atom,
            smiles,
            input,
            export_csv,
            export_pdb,
            export_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let pr = if pdb.is_some() || puzzle.eq_ignore_ascii_case("sandbox") || puzzle.eq_ignore_ascii_case("custom") {
                SculptPuzzlePreset::CustomPdbFreeformSandbox
            } else {
                match puzzle.trim().to_ascii_lowercase().as_str() {
                    "egfr" | "1m17" | "p2" => SculptPuzzlePreset::Egfr1m17GatekeeperBarrierChallenge,
                    "fxa" | "1ezq" | "p3" => SculptPuzzlePreset::FactorXa1ezqVoidAndWaterCrusher,
                    "cox2" | "3ln1" | "p4" => SculptPuzzlePreset::Cox23ln1SelectivityLatchChallenge,
                    "kras" | "6oim" | "p5" => SculptPuzzlePreset::Kras6oimSwitchIiCrypticPuzzle,
                    _ => SculptPuzzlePreset::Vhl5nvxDepeptidizationChallenge,
                }
            };
            let custom_m = (!smiles.trim().is_empty()).then(|| {
                let mut m = parse_smiles(smiles.trim()).expect("Invalid --smiles");
                m.id = "SCULPT-QUERY".to_string();
                m.name = "Custom Sculpt Lead".to_string();
                m
            });
            let fb_mol = custom_m.as_ref().or_else(|| ds.mols.first());
            let pdb_str = pdb.as_ref().map(|p| p.to_string_lossy().to_string());
            let mut sess = new_sculpt_session(pr, fb_mol, pdb_str.as_deref());

            if !sprout.trim().is_empty() {
                if let Some(wi) = SCULPT_SPROUT_WARHEADS
                    .iter()
                    .position(|w| w.code.eq_ignore_ascii_case(sprout.trim()))
                {
                    match sprout_in_pocket_rgroup(&mut sess, anchor_atom, wi) {
                        Ok(new_s) => println!("Sprouted {} -> {}", sprout.trim(), new_s),
                        Err(e) => println!("Sprout skipped: {e}"),
                    }
                }
            }

            match recipe.trim().to_ascii_lowercase().as_str() {
                "wiggle" => wiggle_minimize(&mut sess, 32),
                "shake" => shake_pocket_rotamers(&mut sess),
                "auto-band" | "band" => {
                    let _ = auto_band_polar_contacts(&mut sess);
                    wiggle_minimize(&mut sess, 28);
                }
                _ => {
                    let _ = auto_band_polar_contacts(&mut sess);
                    run_blue_fuse_macro(&mut sess);
                }
            }

            let [t_br, t_si, t_go, t_pl] = pr.medal_thresholds();
            println!("=== SAR-Atlas v3.8.0 Foldit / Drugit 3D Sculptor & Challenge Studio ===");
            println!("Puzzle        : {} (PDB {})", pr.title(), sess.pdb_id);
            println!(
                "Foldit Score  : {:.1} -> {:.1} pts [{}] (Par: Bronze {:.0} | Silver {:.0} | Gold {:.0} | Platinum {:.0})",
                sess.initial_score_pts,
                sess.score.foldit_score_pts,
                sess.score.medal.label(),
                t_br,
                t_si,
                t_go,
                t_pl
            );
            println!(
                "Physics Audit : Unbanded dG = {:+.2} kcal/mol | Strain = {:.2} | Clashes = {} | H-Bonds = {} | Displaced Waters = {}",
                sess.score.total_unbanded_dg_kcal,
                sess.score.cart_bonded_strain_kcal,
                sess.score.clash_count,
                sess.score.formed_hbonds,
                sess.score.displaced_unhappy_waters
            );
            println!(
                "Drugit Props  : MW = {:.1} | cLogP = {:.2} | TPSA = {:.1} A^2 | HBD = {} | SAScore = {:.2} | PAINS = {}",
                sess.score.mw,
                sess.score.clogp,
                sess.score.tpsa,
                sess.score.hbd,
                sess.score.sa_score,
                sess.score.pains_count
            );
            println!(
                "
{:<5} {:<30} {:>10} {:>12} {:>7} {:>7}",
                "Step", "Action", "Score_Pts", "Unbanded_dG", "HBonds", "Clashes"
            );
            for st in &sess.trajectory {
                println!(
                    "#{:<4} {:<30} {:>10.1} {:>+12.2} {:>7} {:>7}",
                    st.step_idx,
                    st.action_label,
                    st.foldit_score_pts,
                    st.unbanded_dg_kcal,
                    st.hbonds,
                    st.clashes
                );
            }

            if let Some(ref cp) = export_csv {
                std::fs::write(cp, export_sculpt_trajectory_csv(&sess))
                    .expect("Failed to write sculpt trajectory CSV");
                println!("Wrote Foldit sculpt trajectory CSV -> {}", cp.display());
            }
            if let Some(ref pp) = export_pdb {
                std::fs::write(pp, format_sculpted_complex_pdb(&sess))
                    .expect("Failed to write sculpted complex PDB");
                println!("Wrote sculpted 3D complex PDB -> {}", pp.display());
            }
            if let Some(ref sp) = export_sdf {
                std::fs::write(sp, format_sculpted_ligand_sdf(&sess))
                    .expect("Failed to write sculpted ligand SDF");
                println!("Wrote sculpted 3D ligand SDF -> {}", sp.display());
            }
        }
        Commands::Watermap {
            preset,
            pdb,
            voxel_spacing,
            hotspot_dg,
            id,
            smiles,
            screen_dataset,
            input,
            export_sites_csv,
            export_designs_csv,
            export_screen_csv,
            export_pdb,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let pr = if pdb.is_some() || preset.eq_ignore_ascii_case("custom") {
                WaterMapPreset::CustomPdbWaterMap
            } else {
                match preset.trim().to_ascii_lowercase().as_str() {
                    "egfr" | "1m17" => WaterMapPreset::Egfr1m17BackPocketAndHingeBridge,
                    "cox2" | "3ln1" => WaterMapPreset::Cox23ln1DewettedHydrophobicChannel,
                    "kras" | "6oim" => WaterMapPreset::Kras6oimSwitchIiCrypticSolvation,
                    _ => WaterMapPreset::FactorXa1ezqS1S4Hotspots,
                }
            };
            let cfg = WaterMapConfig {
                preset: pr,
                custom_pdb_path: pdb.map(|p| p.to_string_lossy().to_string()),
                gist_voxel_spacing_a: voxel_spacing.clamp(0.50, 1.25),
                hotspot_dg_threshold_kcal: hotspot_dg.clamp(0.50, 4.00),
                design_rgroup_vectors: true,
                ..WaterMapConfig::default()
            };

            let query_mol = if !smiles.trim().is_empty() {
                let mut m = parse_smiles(smiles.trim()).expect("Invalid --smiles");
                m.id = "WM-QUERY".to_string();
                m.name = "Custom WaterMap Query".to_string();
                m
            } else {
                id.as_deref()
                    .and_then(|q| {
                        ds.mols.iter().find(|m| {
                            m.id.eq_ignore_ascii_case(q)
                                || m.name.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
                        })
                    })
                    .or_else(|| ds.mols.first())
                    .cloned()
                    .expect("Dataset is empty")
            };

            let rep = analyze_watermap_solvation(&query_mol, Some(&ds), &cfg);
            println!(
                "=== SAR-Atlas v3.8.0 WaterMap / IFST Solvation Hotspot & R-Vector Studio ==="
            );
            println!(
                "Target & Lead : {} (PDB {}) | {} ({})",
                rep.preset.label(),
                rep.pdb_id,
                rep.compound_name,
                rep.compound_id
            );
            println!(
                "GIST & QT Map : {} voxels ({:.2} A) | {} principal sites (Unhappy={} | Dewetted={} | Conserved Bridge={})",
                rep.total_gist_voxels_evaluated,
                rep.gist_voxel_spacing_a,
                rep.total_hydration_sites,
                rep.unhappy_hotspot_sites,
                rep.dewetted_cavity_sites,
                rep.conserved_bridge_sites
            );
            println!(
                "Displacement  : Displaced Hotspots = {} | Retained Bridges = {} | dH_disp = {:+.2} | -TdS_disp = {:+.2} | Net dG_solv = {:+.2} | Refined dG = {:+.2} kcal/mol (Headroom = {:+.2})",
                rep.displaced_hotspot_count,
                rep.retained_bridge_count,
                rep.dh_displacement_gain_kcal,
                rep.tds_displacement_gain_kcal,
                rep.net_watermap_dg_kcal,
                rep.watermap_refined_dg_kcal,
                rep.untapped_proximal_hotspot_dg_kcal
            );
            println!(
                "
{:<7} {:<24} {:>5} {:>7} {:>7} {:>7} {:>6}  {:<22}  {}",
                "Site", "Thermodynamic Class", "Occ", "dH", "-TdS", "dG", "f_disp", "Ligand Status", "Lining"
            );
            for s in &rep.hydration_sites {
                println!(
                    "{:<7} {:<24} {:>5.2} {:>+7.2} {:>+7.2} {:>+7.2} {:>6.2}  {:<22}  {}",
                    s.site_id,
                    s.class.short_label(),
                    s.occupancy,
                    s.dh_excess_kcal,
                    s.neg_tds_excess_kcal,
                    s.dg_excess_kcal,
                    s.ligand_overlap_f_disp,
                    s.status.label(),
                    s.lining_residues.join(",")
                );
            }
            if !rep.rgroup_designs.is_empty() {
                println!(
                    "
{:<5} {:<14} {:<8} {:<18} {:>6} {:>8} {:>6} {:>7} {:>6}  {}",
                    "Rank", "DesignID", "Site", "R-Group", "DistA", "ddG_solv", "GE", "dLLE", "Score", "Designed SMILES"
                );
                for d in rep.rgroup_designs.iter().take(12) {
                    println!(
                        "#{:<4} {:<14} {:<8} {:<18} {:>6.2} {:>+8.2} {:>6.2} {:>+7.2} {:>6.1}  {}",
                        d.rank,
                        d.design_id,
                        d.target_site_id,
                        d.rgroup_name,
                        d.vector_distance_a,
                        d.ddg_solvation_gain_kcal,
                        d.group_efficiency_ge,
                        d.delta_lle,
                        d.design_score,
                        d.designed_smiles
                    );
                }
            }

            if screen_dataset || export_screen_csv.is_some() {
                let rows = screen_dataset_watermap_solvation(&ds, &cfg);
                if screen_dataset {
                    println!(
                        "
{:<5} {:<12} {:<22} {:>7} {:>7} {:>9} {:>10}  {}",
                        "Rank", "ID", "Name", "DispHot", "Bridges", "NetWM_dG", "Refined_dG", "Top R-Group Vector"
                    );
                    for r in rows.iter().take(20) {
                        println!(
                            "#{:<4} {:<12} {:<22} {:>7} {:>7} {:>+9.2} {:>+10.2}  {}",
                            r.rank,
                            r.compound_id,
                            r.compound_name,
                            r.displaced_hotspots,
                            r.retained_bridges,
                            r.net_watermap_dg_kcal,
                            r.watermap_refined_dg_kcal,
                            r.top_rgroup_suggestion
                        );
                    }
                }
                if let Some(ref sp) = export_screen_csv {
                    std::fs::write(sp, export_dataset_watermap_screen_csv(&rows))
                        .expect("Failed to write dataset WaterMap screen CSV");
                    println!("Wrote dataset WaterMap screen CSV -> {}", sp.display());
                }
            }
            if let Some(ref sp) = export_sites_csv {
                std::fs::write(sp, export_watermap_sites_csv(&rep))
                    .expect("Failed to write WaterMap sites CSV");
                println!("Wrote WaterMap hydration sites CSV -> {}", sp.display());
            }
            if let Some(ref dp) = export_designs_csv {
                std::fs::write(dp, export_watermap_rgroup_designs_csv(&rep))
                    .expect("Failed to write WaterMap R-group designs CSV");
                println!("Wrote WaterMap R-group vector designs CSV -> {}", dp.display());
            }
            if let Some(ref pp) = export_pdb {
                std::fs::write(pp, format_watermap_annotated_pdb(&rep))
                    .expect("Failed to write WaterMap annotated PDB");
                println!("Wrote WaterMap hotspot-annotated PDB -> {}", pp.display());
            }
        }
        Commands::HbondNet {
            preset,
            pdb,
            ph,
            id,
            smiles,
            screen_dataset,
            input,
            export_nodes_csv,
            export_edges_csv,
            export_screen_csv,
            export_pdb,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let pr = if pdb.is_some() || preset.eq_ignore_ascii_case("custom") {
                HBondNetworkPreset::CustomPdbHBondNetwork
            } else {
                match preset.trim().to_ascii_lowercase().as_str() {
                    "cox2" | "3ln1" => HBondNetworkPreset::Cox23ln1His90Gln178Sulfonamide,
                    "drd2" | "6cm4" => HBondNetworkPreset::Drd26cm4Asp114WaterSaltBridge,
                    "kras" | "6oim" => HBondNetworkPreset::Kras6oimHis95Gln99PolarGroove,
                    _ => HBondNetworkPreset::Egfr1m17Thr790WaterBridge,
                }
            };
            let cfg = HBondNetworkConfig {
                preset: pr,
                custom_pdb_path: pdb.map(|p| p.to_string_lossy().to_string()),
                target_ph: ph.clamp(4.0, 10.0),
                ..HBondNetworkConfig::default()
            };

            let query_mol = if !smiles.trim().is_empty() {
                let mut m = parse_smiles(smiles.trim()).expect("Invalid --smiles");
                m.id = "HBN-QUERY".to_string();
                m.name = "Custom HBN Query".to_string();
                m
            } else {
                id.as_deref()
                    .and_then(|q| {
                        ds.mols.iter().find(|m| {
                            m.id.eq_ignore_ascii_case(q)
                                || m.name.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
                        })
                    })
                    .or_else(|| ds.mols.first())
                    .cloned()
                    .expect("Dataset is empty")
            };

            let rep = optimize_pocket_hbond_network(&query_mol, Some(&ds), &cfg);
            println!(
                "=== SAR-Atlas v3.8.0 His/Asp/Glu Tautomer & Bridge-Water H-Bond Network Studio ==="
            );
            println!(
                "Target & Lead : {} (PDB {}, pH {:.1}) | {} ({}) -> {}",
                rep.preset.label(),
                rep.pdb_id,
                rep.target_ph,
                rep.compound_name,
                rep.compound_id,
                rep.selected_ligand_state_label
            );
            println!(
                "Global Solver : {} ({} microstate leaves) | Naive HBN dG = {:+.2} -> Optimized HBN dG = {:+.2} kcal/mol (Gain = {:+.2} kcal/mol)",
                rep.solver_method,
                rep.states_evaluated,
                rep.naive_network_dg_kcal,
                rep.optimized_network_dg_kcal,
                rep.ddg_hbn_optimization_gain_kcal
            );
            println!(
                "Network Audit : Direct H-Bonds = {} | Retained Bridge Waters = {} (Displaced = {}) | 180-deg Flips = {} | Clashes {} -> {} | Wyman dn_H+ = {:+.1}",
                rep.direct_protein_ligand_hbonds,
                rep.retained_bridge_waters,
                rep.displaced_waters,
                rep.flipped_180_residues,
                rep.naive_polar_clashes,
                rep.optimized_polar_clashes,
                rep.net_proton_uptake_at_ph
            );
            println!(
                "
{:<11} {:<22} {:>6} {:<16} {:<16} {:>9}  {}",
                "Node", "Category", "pKa", "NaiveState", "OptimalState", "Gain_kcal", "Mechanism"
            );
            for n in &rep.nodes {
                let nv = &n.modes[n.naive_mode_idx];
                let op = &n.modes[n.optimal_mode_idx];
                let pka_s = n
                    .propka_shifted_pka
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "{:<11} {:<22} {:>6} {:<16} {:<16} {:>+9.2}  {}",
                    n.node_id,
                    n.kind.short_label(),
                    pka_s,
                    nv.state_code,
                    op.state_code,
                    n.node_stabilization_gain_kcal,
                    op.description
                );
            }

            if screen_dataset || export_screen_csv.is_some() {
                let rows = screen_dataset_hbond_network(&ds, &cfg);
                if screen_dataset {
                    println!(
                        "
{:<5} {:<12} {:<22} {:>8} {:>8} {:>8} {:>9} {:>6} {:>5}  {}",
                        "Rank", "ID", "Name", "Naive_dG", "Opt_HBN", "Gain", "AdjPck_dG", "HBonds", "HOH", "LigState"
                    );
                    for r in rows.iter().take(20) {
                        println!(
                            "#{:<4} {:<12} {:<22} {:>+8.2} {:>+8.2} {:>+8.2} {:>+9.2} {:>6} {:>5}  {}",
                            r.rank,
                            r.compound_id,
                            r.compound_name,
                            r.naive_network_dg_kcal,
                            r.optimized_network_dg_kcal,
                            r.ddg_hbn_gain_kcal,
                            r.hbn_adjusted_pocket_dg_kcal,
                            r.direct_hbonds,
                            r.retained_bridge_waters,
                            r.selected_ligand_state
                        );
                    }
                }
                if let Some(ref sp) = export_screen_csv {
                    std::fs::write(sp, export_dataset_hbond_screen_csv(&rows))
                        .expect("Failed to write dataset HBN screen CSV");
                    println!("Wrote dataset H-bond network screen CSV -> {}", sp.display());
                }
            }
            if let Some(ref np) = export_nodes_csv {
                std::fs::write(np, export_hbond_network_nodes_csv(&rep))
                    .expect("Failed to write HBN nodes CSV");
                println!("Wrote variable His/Asp/Glu & bridge-water nodes CSV -> {}", np.display());
            }
            if let Some(ref ep) = export_edges_csv {
                std::fs::write(ep, export_hbond_network_edges_csv(&rep))
                    .expect("Failed to write HBN edges CSV");
                println!("Wrote directional H-bond network edges CSV -> {}", ep.display());
            }
            if let Some(ref pp) = export_pdb {
                std::fs::write(pp, format_optimized_hbond_complex_pdb(&rep))
                    .expect("Failed to write prepared complex PDB");
                println!("Wrote prepared protonated/bridge-water complex PDB -> {}", pp.display());
            }
        }
        Commands::InducedFit {
            preset,
            pdb,
            id,
            smiles,
            max_flex,
            screen_dataset,
            input,
            export_csv,
            export_rotamers_csv,
            export_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let pr = if pdb.is_some() || preset.eq_ignore_ascii_case("custom") {
                InducedFitPreset::CustomPdbFlexibleEnsemble
            } else {
                match preset.trim().to_ascii_lowercase().as_str() {
                    "drd2" | "6cm4" => InducedFitPreset::Drd26cm4ToggleSwitchAndCage,
                    "cox2" | "3ln1" => InducedFitPreset::Cox23ln1Arg513AndSidePocket,
                    "kras" | "6oim" => InducedFitPreset::Kras6oimHis95SwitchIIGroove,
                    _ => InducedFitPreset::Egfr1m17GatekeeperAndBackPocket,
                }
            };
            let cfg = InducedFitConfig {
                preset: pr,
                custom_pdb_path: pdb.map(|p| p.to_string_lossy().to_string()),
                max_flexible_residues: max_flex.clamp(2, 12),
                ..InducedFitConfig::default()
            };

            let query_mol = if !smiles.trim().is_empty() {
                let mut m = parse_smiles(smiles.trim()).expect("Invalid --smiles");
                m.id = "IFD-QUERY".to_string();
                m.name = "Custom IFD Query".to_string();
                m
            } else {
                id.as_deref()
                    .and_then(|q| {
                        ds.mols.iter().find(|m| {
                            m.id.eq_ignore_ascii_case(q)
                                || m.name.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
                        })
                    })
                    .or_else(|| ds.mols.first())
                    .cloned()
                    .expect("Dataset is empty")
            };

            let rep = score_ligand_induced_fit_ensemble(&query_mol, Some(&ds), &cfg);
            println!(
                "=== SAR-Atlas v3.8.0 Induced-Fit Side-Chain Repacking & Boltzmann Ensemble Studio ==="
            );
            println!(
                "Target & Lead : {} (PDB {}) | {} ({})",
                rep.preset.label(),
                rep.pdb_id,
                rep.compound_name,
                rep.compound_id
            );
            println!(
                "Thermodynamics: Rigid Holo dG = {:+.2} kcal/mol -> Boltzmann Ensemble dG = {:+.2} kcal/mol (ddG Rescue = {:+.2} kcal/mol | IFDScore = {:+.2})",
                rep.rigid_holo_dg_kcal,
                rep.boltzmann_ensemble_dg_kcal,
                rep.ddg_induced_fit_rescue_kcal,
                rep.best_sherman_ifd_score_kcal
            );
            println!(
                "State Shift   : Dominant {} ({:.1}% holo occupancy | KL shift = {:.3} nats | -TdS_rec = {:+.2} kcal/mol | {})",
                rep.dominant_state_code,
                rep.dominant_holo_occupancy_pct,
                rep.kl_population_shift_nats,
                rep.receptor_conf_entropy_cost_kcal,
                rep.mechanism.label()
            );
            println!(
                "
{:<16} {:>7} {:>7} {:>8} {:>9} {:>8} {:>9} {:>9} {:>8} {:>9}",
                "EnsembleState", "w_Apo%", "w_Holo%", "dG_Reorg", "PreRep_dG", "Post_dG", "RotStrn", "Eff_dG", "IFDScore", "Repacked"
            );
            for s in &rep.state_scores {
                println!(
                    "{:<16} {:>6.1}% {:>6.1}% {:>+8.2} {:>+9.2} {:>+8.2} {:>+9.2} {:>+9.2} {:>+8.2} {:>3} ({:.2}A)",
                    s.state_code,
                    s.apo_weight * 100.0,
                    s.holo_boltzmann_weight * 100.0,
                    s.apo_reorg_penalty_kcal,
                    s.pre_repack_dg_kcal,
                    s.post_repack_interaction_dg_kcal,
                    s.rotamer_strain_kcal,
                    s.effective_total_dg_kcal,
                    s.sherman_ifd_score_kcal,
                    s.repacked_residue_count,
                    s.repacked_sidechain_rmsd_a
                );
            }

            if screen_dataset || export_csv.is_some() {
                let rows = screen_dataset_induced_fit_ensemble(&ds, &cfg);
                if screen_dataset {
                    println!(
                        "
{:<5} {:<12} {:<22} {:>8} {:>8} {:>8} {:>8} {:<16} {:>6}  {}",
                        "Rank", "ID", "Name", "Rigid_dG", "Boltz_dG", "Rescue", "IFDScore", "DomState", "Occ%", "Mechanism"
                    );
                    for r in rows.iter().take(20) {
                        println!(
                            "#{:<4} {:<12} {:<22} {:>+8.2} {:>+8.2} {:>+8.2} {:>+8.2} {:<16} {:>5.1}%  {}",
                            r.rank,
                            r.compound_id,
                            r.compound_name,
                            r.rigid_holo_dg_kcal,
                            r.boltzmann_ensemble_dg_kcal,
                            r.ddg_induced_fit_rescue_kcal,
                            r.sherman_ifd_score_kcal,
                            r.dominant_state_code,
                            r.dominant_occupancy_pct,
                            r.mechanism.short_label()
                        );
                    }
                }
                if let Some(ref cp) = export_csv {
                    std::fs::write(cp, export_induced_fit_ensemble_csv(&rows))
                        .expect("Failed to write induced-fit ensemble CSV");
                    println!("Wrote dataset induced-fit Boltzmann ensemble CSV -> {}", cp.display());
                }
            }
            if let Some(ref rp) = export_rotamers_csv {
                std::fs::write(rp, export_repacked_rotamers_csv(&rep))
                    .expect("Failed to write repacked rotamers CSV");
                println!("Wrote per-state Dunbrack repacked rotamers CSV -> {}", rp.display());
            }
            if let Some(ref sp) = export_sdf {
                std::fs::write(sp, format_induced_fit_ensemble_sdf(&rep))
                    .expect("Failed to write induced-fit ensemble 3D SDF");
                println!("Wrote 5-state Boltzmann ensemble 3D poses SDF -> {}", sp.display());
            }
        }
        Commands::FragLink {
            pair,
            id,
            frag_a,
            frag_b,
            pocket_idx,
            vector_pair,
            input,
            export_csv,
            export_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let pr = match pair.trim().to_ascii_lowercase().as_str() {
                "cox2" | "3ln1" => FragmentPairPreset::Cox23ln1SulfonamideAndTolyl,
                "drd2" | "6cm4" => FragmentPairPreset::Drd26cm4PiperidineAndBenzisoxazole,
                "bivalent" | "allo" | "allosteric" => FragmentPairPreset::EgfrBivalentHingeToAllosteric,
                "deconstruct" | "split" | "lead" => FragmentPairPreset::DeconstructSelectedLead,
                "custom" => FragmentPairPreset::CustomSmilesSubpockets,
                _ => FragmentPairPreset::Egfr1m17HingeAndBackPocket,
            };
            let seed_mol = id
                .as_deref()
                .and_then(|q| {
                    ds.mols.iter().find(|m| {
                        m.id.eq_ignore_ascii_case(q)
                            || m.name.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
                    })
                })
                .or_else(|| ds.mols.first());

            let rep = design_linked_fragments(
                &FragmentLinkerConfig {
                    preset: pr,
                    vector_pair_rank: vector_pair,
                    custom_frag_a_smiles: frag_a,
                    custom_frag_b_smiles: frag_b,
                    custom_pocket_idx: pocket_idx,
                },
                seed_mol,
                Some(&ds),
            );
            println!(
                "=== SAR-Atlas v3.8.0 Sub-Pocket Fragment Linker & Super-Additivity Studio ==="
            );
            println!(
                "Pair Preset : {} (PDB {} — {})",
                rep.preset.label(),
                rep.target_pdb_id,
                rep.target_name
            );
            println!(
                "Fragment A  : {} ({}) | dG_A = {:+.2} kcal/mol (pKd = {:.2}, Ro3 LE = {:.2})",
                rep.frag_a_mol.name,
                rep.frag_a_mol.smiles,
                rep.dg_frag_a_kcal,
                rep.pkd_frag_a,
                rep.frag_a_ro3.seed_le
            );
            println!(
                "Fragment B  : {} ({}) | dG_B = {:+.2} kcal/mol (pKd = {:.2}, Ro3 LE = {:.2})",
                rep.frag_b_mol.name,
                rep.frag_b_mol.smiles,
                rep.dg_frag_b_kcal,
                rep.pkd_frag_b,
                rep.frag_b_ro3.seed_le
            );
            if let Some(vp) = rep.vector_pairs.get(rep.active_pair_idx) {
                println!(
                    "Exit Vector : {} <-> {} | d_anchor = {:.2} A (d_tip = {:.2} A) | alpha_A = {:.1} deg | alpha_B = {:.1} deg | phi = {:.1} deg | {}",
                    vp.vec_a.atom_label,
                    vp.vec_b.atom_label,
                    vp.anchor_distance_a,
                    vp.exit_tip_distance_a,
                    vp.alpha_a_deg,
                    vp.alpha_b_deg,
                    vp.dihedral_phi_deg,
                    vp.regime.label()
                );
            }
            println!(
                "
{:<5} {:<16} {:<13} {:>6} {:>6} {:>8} {:>8} {:>8} {:>9} {:>6} {:>6} {:>6}  {}",
                "Rank", "LinkerCode", "Class", "Geom%", "RMSD_A", "dG_rigid", "dG_strn", "dG_AB", "ddG_sup", "pKd", "LLE", "Score", "SMILES"
            );
            for c in &rep.candidates {
                println!(
                    "#{:<4} {:<16} {:<13} {:>6.1} {:>6.2} {:>+8.2} {:>+8.2} {:>+8.2} {:>+9.2} {:>6.2} {:>6.2} {:>6.1}  {}",
                    c.rank,
                    c.linker_code,
                    c.linker_class.short_label(),
                    c.geometry_match_score,
                    c.fragment_core_rmsd_a,
                    c.dg_rigid_entropy_saved_kcal,
                    c.dg_linker_strain_kcal,
                    c.dg_linked_ab_kcal,
                    c.ddg_superadditivity_kcal,
                    c.predicted_pkd,
                    c.lle,
                    c.composite_linker_score,
                    c.canonical_smiles
                );
            }
            if let Some(ref cp) = export_csv {
                std::fs::write(cp, export_linked_fragments_csv(&rep))
                    .expect("Failed to write linked fragments CSV");
                println!("Wrote linked fragments super-additivity CSV -> {}", cp.display());
            }
            if let Some(ref sp) = export_sdf {
                std::fs::write(sp, format_linked_fragments_sdf(&rep))
                    .expect("Failed to write linked fragments 3D SDF");
                println!("Wrote sub-pocket fragments + linked chimeras 3D SDF -> {}", sp.display());
            }
        }
        Commands::TorsionStrain {
            id,
            smiles,
            pose,
            screen_dataset,
            input,
            export_scan_csv,
            export_locks_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let pose_source = match pose.to_ascii_lowercase().as_str() {
                "egfr" | "1m17" => BioactivePoseSource::PocketEgfr1m17,
                "drd2" | "d2" | "6cm4" => BioactivePoseSource::PocketDrd26cm4,
                "cox" | "cox2" | "cox-2" | "3ln1" => BioactivePoseSource::PocketCox23ln1,
                "rocs" | "overlay" | "lead" => BioactivePoseSource::RocsReferenceOverlay,
                _ => BioactivePoseSource::AutoTargetPocket,
            };
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let query_mol = if !smiles.trim().is_empty() {
                let mut m = parse_smiles(smiles.trim()).expect("Valid query --smiles required");
                m.id = "QUERY".to_string();
                m.name = "Custom Query Ligand".to_string();
                m
            } else if let Some(ref qid) = id {
                let ql = qid.to_ascii_lowercase();
                ds.mols
                    .iter()
                    .find(|m| {
                        m.id.to_ascii_lowercase().contains(&ql)
                            || m.name.to_ascii_lowercase().contains(&ql)
                    })
                    .cloned()
                    .unwrap_or_else(|| ds.mols[0].clone())
            } else {
                ds.mols[0].clone()
            };

            let rep = analyze_torsion_strain_and_locks(&query_mol, Some(&ds), pose_source, true);
            println!(
                "=== SAR-Atlas v3.8.0 CSD/SMARTS Dihedral Scanner, Bioactive Strain & Rotor Locking ==="
            );
            println!(
                "Compound & Pose   : {} ({}) | Pose: {} | Free-vs-Bound Heavy RMSD = {:.2} A",
                rep.compound_name,
                rep.compound_id,
                rep.resolved_pose_label,
                rep.free_vs_bioactive_rmsd_a
            );
            println!(
                "Shoichet Strain   : Total = {:.2} TEU (<=6.0) | Max Single = {:.2} TEU (<=1.8) | Pass = {} | Tier: {}",
                rep.total_strain_teu,
                rep.max_single_bond_teu,
                rep.shoichet_filter_pass,
                rep.strain_tier.label()
            );
            println!(
                "Thermodynamics    : Strain dE = {:.2} kcal/mol | Rotor Entropy -TdS = {:.2} kcal/mol | Raw dG = {:+.2} -> Strain-Adj dG = {:+.2} kcal/mol ({:.1} ms)",
                rep.bioactive_strain_penalty_kcal,
                rep.total_conformational_entropy_kcal,
                rep.raw_pocket_dg_kcal,
                rep.strain_adjusted_dg_bind_kcal,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            println!(
                "
{:<4} {:<18} {:<20} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}  {}",
                "Rot", "Dihedral", "CSD_SMARTS_Rule", "FreeDeg", "BioDeg", "CSDPeak", "Barrier", "TEU", "dE_kcal", "Band"
            );
            for b in &rep.bond_profiles {
                println!(
                    "R{:<3} {:<18} {:<20} {:>8.1} {:>8.1} {:>8.1} {:>8.2} {:>7.2} {:>7.2}  {}",
                    b.rotor_idx + 1,
                    b.dihedral_label,
                    b.matched_rule_id,
                    b.free_pose_angle_deg,
                    b.bioactive_pose_angle_deg,
                    b.nearest_csd_peak_deg,
                    b.rotational_barrier_kcal,
                    b.bioactive_strain_teu,
                    b.bioactive_strain_kcal,
                    b.quality_band.short_label()
                );
            }
            if !rep.locked_analogs.is_empty() {
                println!(
                    "
{:<5} {:<10} {:<18} {:<24} {:>7} {:>9} {:>9}  {}",
                    "Rank", "Lock_ID", "Target_Rotor", "Strategy", "Barr", "dG_Preorg", "Pred_pAct", "Modification"
                );
                for c in rep.locked_analogs.iter().take(10) {
                    println!(
                        "#{:<4} {:<10} {:<18} {:<24} {:>7.2} {:>+9.2} {:>9.2}  {}",
                        c.rank,
                        c.candidate_id,
                        c.target_dihedral_label,
                        c.strategy.short_label(),
                        c.locked_barrier_kcal,
                        c.preorganization_gain_kcal,
                        c.predicted_locked_pact,
                        c.modification_summary
                    );
                }
            }
            if screen_dataset {
                let rows = screen_dataset_torsion_strain(&ds, pose_source);
                println!(
                    "
=== Dataset-Wide Bioactive Strain Triage ({} compounds) ===",
                    rows.len()
                );
                for r in rows.iter().take(12) {
                    println!(
                        "#{:<3} {:<12} {:<22} Nrot {:>2} | Total {:>5.2} TEU (Max {:>4.2}) | Strain {:>5.2} kcal | dG_adj {:>+6.2} | {}",
                        r.rank,
                        r.compound_id,
                        r.compound_name,
                        r.rotatable_bonds,
                        r.total_strain_teu,
                        r.max_single_teu,
                        r.bioactive_strain_kcal,
                        r.strain_adjusted_dg_kcal,
                        r.strain_tier.short_label()
                    );
                }
            }
            if let Some(ref sp) = export_scan_csv {
                std::fs::write(sp, export_torsion_profiles_csv(&rep))
                    .expect("Failed to write torsion scan CSV");
                println!("Wrote 1D dihedral scan CSV -> {}", sp.display());
            }
            if let Some(ref lp) = export_locks_csv {
                std::fs::write(lp, export_locked_analogs_csv(&rep))
                    .expect("Failed to write locked analogs CSV");
                println!("Wrote rotatable-bond locked analogs CSV -> {}", lp.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_torsion_poses_sdf(&rep))
                    .expect("Failed to write free/bioactive 3D SDF");
                println!("Wrote free & bioactive 3D poses SDF -> {}", sdf_p.display());
            }
        }
        Commands::Rbfe {
            series,
            topology,
            lambda_windows,
            beta,
            min_lomap,
            hysteresis_warn,
            allow_charge_change,
            smiles,
            input,
            export_edges_csv,
            export_nodes_csv,
        } => {
            let t0 = Instant::now();
            let series_preset = match series.to_ascii_lowercase().as_str() {
                "auto" | "dataset" => RbfeSeriesPreset::AutoDominantTarget,
                "cox" | "cox2" | "cox-2" | "3ln1" => RbfeSeriesPreset::Cox23ln1Diarylpyrazoles,
                "drd2" | "d2" | "6cm4" => RbfeSeriesPreset::Drd26cm4Arylpiperazines,
                _ => RbfeSeriesPreset::Egfr1m17Quinazolines,
            };
            let topology_mode = match topology.to_ascii_lowercase().as_str() {
                "star" | "radial" | "hub" => RbfeTopologyMode::StarRadialHub,
                "dense" | "redundant" => RbfeTopologyMode::DenseRedundantLomap,
                _ => RbfeTopologyMode::CycleClosedLomap,
            };
            let custom_analog_smiles: Vec<String> = smiles
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();

            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let cfg = RbfeNetworkConfig {
                series_preset,
                topology_mode,
                lomap_beta: beta,
                min_lomap_score: min_lomap,
                lambda_windows,
                allow_charge_changes: allow_charge_change,
                cycle_hysteresis_warn_kcal: hysteresis_warn,
                custom_analog_smiles,
                ..RbfeNetworkConfig::default()
            };
            let rep = run_rbfe_network_campaign(&ds, &cfg);

            println!(
                "=== SAR-Atlas v3.8.0 Lomap MCS Perturbation Network, Bennett BAR & Cycle-Closure MLE ==="
            );
            println!(
                "Receptor & Series : {} (PDB {}) | Topology: {} | Lambda Windows: {}",
                rep.resolved_target,
                rep.pocket_pdb_id,
                cfg.topology_mode.label(),
                cfg.lambda_windows
            );
            println!(
                "Graph & Overlap   : Nodes = {} | Edges = {} | Closed Cycles = {} ({:.0}% nodes in cycles) | Diameter = {} | Mean Lomap = {:.3} | Mean Overlap O_AB = {:.3}",
                rep.summary.n_nodes,
                rep.summary.n_edges,
                rep.summary.n_cycles,
                rep.summary.fraction_nodes_in_cycles * 100.0,
                rep.summary.graph_diameter,
                rep.summary.mean_lomap_score,
                rep.summary.mean_phase_space_overlap
            );
            println!(
                "Cycle Hysteresis  : Raw BAR Mean = {:.3} kcal/mol (Max {:.3}, Flagged {}) -> Post-MLE Max Residual = {:.6} kcal/mol ({:.1} ms)",
                rep.summary.mean_cycle_hysteresis_kcal,
                rep.summary.max_cycle_hysteresis_kcal,
                rep.summary.n_flagged_cycles,
                rep.summary.post_mle_max_cycle_error_kcal,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            println!(
                "Retrospective MLE : N_exp = {} | dG RMSE = {:.3} kcal/mol | MAE = {:.3} kcal/mol | R^2 = {:.3} | Kendall tau = {:+.3}",
                rep.summary.n_experimental_nodes,
                rep.summary.dg_rmse_kcal,
                rep.summary.dg_mae_kcal,
                rep.summary.pact_r2_pearson,
                rep.summary.kendall_tau
            );
            println!(
                "
{:<5} {:<26} {:<11} {:>6} {:>4} {:>15} {:>9} {:>9} {:>7}  {}",
                "Edge", "Perturbation (A -> B)", "Role", "Lomap", "MCS", "ddG_BAR+-Sig", "ddG_MLE", "FEP_Hyst", "Min_Ov", "Status"
            );
            for e in &rep.edges {
                let role = if e.is_spanning_tree_edge { "MST-Tree" } else { "Cyc-Chord" };
                println!(
                    "E{:<4} {:<26} {:<11} {:>6.3} {:>4} {:>+6.2} +- {:<5.2} {:>+9.2} {:>9.3} {:>7.3}  {}",
                    e.edge_idx + 1,
                    format!("{} -> {}", e.src_id, e.dst_id),
                    role,
                    e.lomap.total_lomap_score,
                    e.lomap.mcs_atoms,
                    e.raw_ddg_bar_kcal,
                    e.bar_std_err_kcal,
                    e.reconciled_ddg_mle_kcal,
                    e.fep_fwd_rev_hysteresis_kcal,
                    e.min_phase_space_overlap,
                    e.convergence_tier.short_label()
                );
            }
            if !rep.cycles.is_empty() {
                println!(
                    "
{:<13} {:<40} {:>12} {:>10} {:>8}  {}",
                    "Cycle", "Closed Loop Nodes", "Signed_BAR", "|Hyst|", "Z-Score", "Audit"
                );
                for cyc in &rep.cycles {
                    let loop_str = format!("{} -> {}", cyc.node_labels.join(" -> "), cyc.node_labels[0]);
                    println!(
                        "{:<13} {:<40} {:>+12.3} {:>10.3} {:>8.2}  {}",
                        cyc.cycle_id,
                        loop_str,
                        cyc.signed_closure_kcal,
                        cyc.abs_hysteresis_kcal,
                        cyc.closure_z_score,
                        if cyc.converged { "PASS" } else { "FLAGGED (>warn)" }
                    );
                }
            }
            println!(
                "
{:<5} {:<12} {:<28} {:>5} {:>14} {:>9} {:>9}",
                "Node", "ID", "Name", "Cyc", "dG_MLE+-Sig", "pAct_MLE", "pAct_Exp"
            );
            for nd in &rep.nodes {
                let exp_s = nd
                    .experimental_pact
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "N{:<4} {:<12} {:<28} {:>5} {:>+6.2} +- {:<4.2} {:>9.2} {:>9}",
                    nd.node_idx + 1,
                    nd.compound_id,
                    nd.compound_name,
                    nd.cycle_count,
                    nd.mle_dg_bind_kcal,
                    nd.mle_dg_std_kcal,
                    nd.mle_predicted_pact,
                    exp_s
                );
            }

            if let Some(ref ep) = export_edges_csv {
                std::fs::write(ep, export_rbfe_edges_csv(&rep))
                    .expect("Failed to write RBFE edges CSV");
                println!("Wrote RBFE perturbation edges CSV -> {}", ep.display());
            }
            if let Some(ref np) = export_nodes_csv {
                std::fs::write(np, export_rbfe_nodes_csv(&rep))
                    .expect("Failed to write RBFE nodes CSV");
                println!("Wrote reconciled MLE nodes CSV -> {}", np.display());
            }
        }
        Commands::ActiveLearn {
            kernel,
            model,
            acq,
            pool,
            w3d,
            kappa,
            batch_size,
            smiles,
            input,
            batch_only,
            export_csv,
        } => {
            let t0 = Instant::now();
            let kernel_mode = match kernel.to_ascii_lowercase().as_str() {
                "2d" | "ecfp" | "ecfp4" | "tanimoto" => QsarKernelMode::Pure2dEcfp4Tanimoto,
                "3d" | "usrcat" | "shape" => QsarKernelMode::Pure3dUsrcatPharmacophore,
                _ => QsarKernelMode::HybridEcfp4And3dUsrcat,
            };
            let surrogate_kind = match model.to_ascii_lowercase().as_str() {
                "gp" | "gaussian" => SurrogateModelKind::ExactGaussianProcess,
                "rf" | "forest" | "ensemble" => SurrogateModelKind::RandomForestEnsemble,
                _ => SurrogateModelKind::ConsensusGpAndEnsemble,
            };
            let acquisition = match acq.to_ascii_lowercase().as_str() {
                "ucb" => AcquisitionStrategy::UpperConfidenceBound,
                "uncertainty" | "var" | "explore" => AcquisitionStrategy::PureEpistemicUncertainty,
                "greedy" | "exploit" => AcquisitionStrategy::GreedyExploitation,
                _ => AcquisitionStrategy::ExpectedImprovement,
            };
            let pool_source = match pool.to_ascii_lowercase().as_str() {
                "holdout" | "benchmark" | "split" => CandidatePoolSource::DatasetHoldoutBenchmark,
                _ => CandidatePoolSource::HarvestedAnalogsAndLibrary,
            };
            let custom_pool_smiles: Vec<String> = smiles
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();

            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let cfg = ActiveLearningConfig {
                kernel_mode,
                surrogate_kind,
                acquisition,
                pool_source,
                weight_3d_usrcat: w3d,
                ucb_kappa: kappa,
                batch_size,
                custom_pool_smiles,
                ..ActiveLearningConfig::default()
            };
            let rep = run_active_learning_campaign(&ds, &cfg);

            println!(
                "=== SAR-Atlas v3.8.0 Active-Learning GP/RF QSAR & Batch Acquisition ==="
            );
            println!(
                "Surrogate & Kernel: {} | {} (w_3D = {:.2}) | Acquisition: {}",
                cfg.surrogate_kind.short_label(),
                cfg.kernel_mode.label(),
                cfg.weight_3d_usrcat,
                cfg.acquisition.label()
            );
            println!(
                "LOO Validation    : N_train = {} | Best pAct = {:.2} | LOO Q^2 = {:.3} | RMSE = {:.3} | UQ Spearman rho = {:+.2} | 90% CI q = {:.2} ({:.0}% cov)",
                rep.validation.n_train,
                rep.validation.best_observed_pact,
                rep.validation.loo_q2,
                rep.validation.loo_rmse,
                rep.validation.uncertainty_error_spearman_rho,
                rep.validation.conformal_90_scale,
                rep.validation.empirical_90_coverage_pct
            );
            println!(
                "Kriging Batch     : Pool {} -> Selected {} | Mean Pred pAct {:.2} +- {:.2} | Pairwise Dissimilarity {:.2} ({:.1} ms)",
                rep.pool_size,
                rep.selected_batch_count,
                rep.batch_mean_predicted_pact,
                rep.batch_mean_epistemic_std,
                rep.batch_mean_pairwise_dissim,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            for r in &rep.simulated_curve {
                println!(
                    "  [DMTA Round {}] Tested {:>2} | AL Best pAct {:.2} vs Rand {:.2} | Top-Quartile Hits {} vs {} (EF = {:.2}x)",
                    r.round,
                    r.cumulative_tested,
                    r.al_best_pact_found,
                    r.random_best_pact_found,
                    r.al_top_quartile_hits_found,
                    r.random_top_quartile_hits_found,
                    r.enrichment_factor
                );
            }
            println!(
                "{:<5} {:<5} {:<14} {:<22} {:>7} {:>6} {:>13} {:>7} {:>6} {:>6}  {}",
                "Rank", "Pick", "Cand_ID", "Name", "Pred", "+-Sig", "90%_CI", "EI", "UCB", "NearTc", "Provenance"
            );
            for c in rep.candidates.iter().take(batch_size.max(12).min(24)) {
                let pick = c
                    .batch_pick_order
                    .map(|k| format!("B{k}"))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "#{:<4} {:<5} {:<14} {:<22} {:>7.2} {:>6.2} [{:>4.2}, {:>4.2}] {:>7.4} {:>6.2} {:>6.2}  {}",
                    c.rank,
                    pick,
                    c.candidate_id,
                    c.candidate_name,
                    c.predicted_pact,
                    c.epistemic_std,
                    c.ci90_low_pact,
                    c.ci90_high_pact,
                    c.expected_improvement,
                    c.ucb_score,
                    c.nearest_train_similarity,
                    c.provenance
                );
            }

            if let Some(ref csv_p) = export_csv {
                std::fs::write(csv_p, export_active_learning_csv(&rep, batch_only))
                    .expect("Failed to write active-learning CSV");
                println!("Wrote active-learning QSAR CSV -> {}", csv_p.display());
            }
        }
        Commands::Allosteric {
            preset,
            pdb,
            cutoff,
            site,
            smiles,
            input,
            export_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let sys_idx = preset
                .parse::<usize>()
                .ok()
                .filter(|&k| k < CURATED_ALLOSTERIC_SYSTEMS.len())
                .or_else(|| {
                    CURATED_ALLOSTERIC_SYSTEMS.iter().position(|s| {
                        s.code.eq_ignore_ascii_case(&preset)
                            || s.pdb_id.eq_ignore_ascii_case(&preset)
                            || s.target_name.to_ascii_lowercase().contains(&preset.to_ascii_lowercase())
                    })
                })
                .unwrap_or(0);
            let sys = &CURATED_ALLOSTERIC_SYSTEMS[sys_idx];

            let (raw_pdb, ortho_hint, cryp_hint) = if let Some(ref p) = pdb {
                let txt = std::fs::read_to_string(p).unwrap_or_else(|e| {
                    eprintln!("Error reading PDB {}: {e}", p.display());
                    std::process::exit(1);
                });
                (txt, None, None)
            } else {
                (
                    curated_allosteric_pdb_text(sys_idx).to_string(),
                    Some(sys.orthosteric_center),
                    Some(sys.cryptic_seed_center),
                )
            };

            let parsed = parse_pdb_structure(&raw_pdb).unwrap_or_else(|e| {
                eprintln!("PDB parse error: {e}");
                std::process::exit(1);
            });
            let net = analyze_allosteric_network(&parsed, ortho_hint, cryp_hint, cutoff);
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let site_i = site.min(net.cryptic_pockets.len().saturating_sub(1));

            let rows = if let Some(ref smi) = smiles {
                let mut m = sar_core::parse_smiles(smi).unwrap_or_else(|e| {
                    eprintln!("SMILES parse error: {e}");
                    std::process::exit(1);
                });
                m.id = "QUERY".to_string();
                m.name = "Query_Modulator".to_string();
                score_ligand_allosteric_coupling(&m, None, &net, site_i)
                    .into_iter()
                    .collect()
            } else {
                screen_dataset_allosteric_pocket(&ds, &net, site_i)
            };

            println!(
                "=== SAR-Atlas v3.8.0 GNM/ANM Cryptic Pocket & Allosteric Strain Coupling ==="
            );
            println!(
                "PDB System        : {} — {} | {} Elastic Nodes | {} Springs | B-Factor r = {:+.2}",
                net.pdb_id,
                net.title,
                net.nodes.len(),
                net.spring_edges.len(),
                net.bfactor_pearson_r
            );
            println!(
                "Mechanical Hinges : {} | Shortest Allosteric Pathway: {} ({:.1} ms)",
                if net.mechanical_hinges.is_empty() {
                    "None".to_string()
                } else {
                    net.mechanical_hinges.join(", ")
                },
                net.primary_pathway_labels.join(" -> "),
                t0.elapsed().as_secs_f64() * 1000.0
            );
            for cp in &net.cryptic_pockets {
                println!(
                    "  [{}] Score {:.1}/100 | Dist {:.1} A | Vol {:.0} -> {:.0} A^3 (+{:.0}%) via ANM Mode #{} | OpenWork +{:.2} kcal | Coupling {:.0}%",
                    cp.site_id,
                    cp.cryptic_allosteric_score,
                    cp.dist_to_orthosteric_a,
                    cp.apo_volume_a3,
                    cp.open_holo_volume_a3,
                    cp.volume_expansion_pct,
                    cp.dominant_anm_mode,
                    cp.elastic_opening_work_kcal,
                    cp.orthosteric_coupling_efficiency * 100.0
                );
            }
            println!(
                "{:<5} {:<12} {:<20} {:>6} {:>8} {:>6} {:>8} {:>7} {:>8} {:>6}  {}",
                "Rank", "ID", "Name", "Score", "Open_dG", "Work", "App_dG", "Strain", "ddG_coop", "Alpha", "Modulation Mode"
            );
            for r in rows.iter().take(12) {
                println!(
                    "#{:<4} {:<12} {:<20} {:>6.1} {:>8.2} +{:>5.2} {:>8.2} {:>7.2} {:>+8.2} {:>6.2}  {}",
                    r.rank,
                    r.compound_id,
                    r.compound_name,
                    r.allosteric_discovery_score,
                    r.intrinsic_open_dg_kcal,
                    r.pocket_opening_work_kcal,
                    r.apparent_allosteric_dg_kcal,
                    r.orthosteric_strain_kcal,
                    r.cooperativity_ddg_kcal,
                    r.cooperativity_alpha,
                    r.modulation_mode.short_label()
                );
            }

            if let Some(ref csv_p) = export_csv {
                std::fs::write(csv_p, export_allosteric_screen_csv(&net, &rows))
                    .expect("Failed to write allosteric CSV");
                println!("Wrote allosteric screening CSV -> {}", csv_p.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_allosteric_poses_sdf(&net, &rows, 20))
                    .expect("Failed to write cryptic poses SDF");
                println!("Wrote 3D cryptic poses SDF -> {}", sdf_p.display());
            }
        }
        Commands::Selectivity {
            pair,
            swap,
            target_pdb,
            offtarget_pdb,
            radius,
            mutations,
            smiles,
            id,
            input,
            export_csv,
            export_mut_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let pair_idx = pair
                .parse::<usize>()
                .ok()
                .filter(|&k| k < CURATED_SELECTIVITY_PAIRS.len())
                .or_else(|| {
                    CURATED_SELECTIVITY_PAIRS.iter().position(|pr| {
                        pr.code.eq_ignore_ascii_case(&pair)
                            || pr.target_name.eq_ignore_ascii_case(&pair)
                            || pr.title.to_ascii_lowercase().contains(&pair.to_ascii_lowercase())
                    })
                })
                .unwrap_or(0);
            let pr = &CURATED_SELECTIVITY_PAIRS[pair_idx];

            let mut tgt_raw = if let Some(ref p) = target_pdb {
                std::fs::read_to_string(p).expect("Failed to read --target-pdb")
            } else {
                curated_pocket_pdb_text(pr.target_preset_idx).to_string()
            };
            let mut off_raw = if let Some(ref p) = offtarget_pdb {
                std::fs::read_to_string(p).expect("Failed to read --offtarget-pdb")
            } else {
                curated_offtarget_pdb_text(pair_idx).to_string()
            };
            if swap {
                std::mem::swap(&mut tgt_raw, &mut off_raw);
            }

            let tgt_parsed = parse_pdb_structure(&tgt_raw).unwrap_or_else(|e| {
                eprintln!("Target PDB parse error: {e}");
                std::process::exit(1);
            });
            let off_parsed = parse_pdb_structure(&off_raw).unwrap_or_else(|e| {
                eprintln!("Off-Target PDB parse error: {e}");
                std::process::exit(1);
            });
            let dual = analyze_dual_pocket_selectivity(&tgt_parsed, &off_parsed, radius);
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);

            let rows = if let Some(ref smi) = smiles {
                let mut m = sar_core::parse_smiles(smi).unwrap_or_else(|e| {
                    eprintln!("SMILES parse error: {e}");
                    std::process::exit(1);
                });
                m.id = "QUERY".to_string();
                m.name = "Query_Ligand".to_string();
                vec![score_ligand_dual_selectivity(&m, None, &dual)]
            } else {
                screen_dataset_selectivity(&ds, &dual)
            };

            let focus_mol = if let Some(ref smi) = smiles {
                let mut m = sar_core::parse_smiles(smi).unwrap();
                m.id = "QUERY".to_string();
                m.name = "Query_Ligand".to_string();
                m
            } else if let Some(ref qid) = id {
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(qid) || m.name.eq_ignore_ascii_case(qid))
                    .cloned()
                    .unwrap_or_else(|| rows[0].target_pose.mol.clone())
            } else {
                rows[0].target_pose.mol.clone()
            };

            let mut mut_tokens: Vec<String> = pr
                .signature_mutations
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            for tok in mutations.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                mut_tokens.push(tok.to_string());
            }
            let mut_scan = scan_pocket_residue_mutations(&focus_mol, &tgt_parsed, radius, &mut_tokens);

            println!(
                "=== SAR-Atlas v3.8.0 Dual-Pocket Selectivity, Differential WaterMap & Mutation ddG ==="
            );
            println!(
                "Target vs Decoy   : {} (Vol {:.0} A^3, DScore {:.2}) vs {} (Vol {:.0} A^3, DScore {:.2}) | dVol {:+.0} A^3",
                dual.target_pocket.pdb_id,
                dual.target_pocket.volume_angstrom3,
                dual.target_pocket.druggability_dscore,
                dual.offtarget_pocket.pdb_id,
                dual.offtarget_pocket.volume_angstrom3,
                dual.offtarget_pocket.druggability_dscore,
                dual.delta_volume_a3
            );
            println!(
                "3D Divergence     : {} Target-Exclusive Sub-Pocket Voxels | {} Differential Waters | {} Divergent Residues ({:.1} ms)",
                dual.target_exclusive_subpocket_pts.len(),
                dual.differential_waters.len(),
                dual.divergent_residues.len(),
                t0.elapsed().as_secs_f64() * 1000.0
            );
            for dw in dual.differential_waters.iter().take(4) {
                println!(
                    "  [Water {}] Tgt {:+.2} vs Off {:+.2} kcal -> Leverage {:+.2} kcal | {}",
                    dw.site_label,
                    dw.target_ddg_disp_kcal,
                    dw.offtarget_ddg_disp_kcal,
                    dw.selectivity_ddg_leverage_kcal,
                    dw.role.short_label()
                );
            }
            println!(
                "{:<5} {:<12} {:<20} {:>6} {:>8} {:>8} {:>8} {:>9}  {}",
                "Rank", "ID", "Name", "Score", "Tgt_dG", "Off_dG", "ddG_sel", "SI(Fold)", "Primary Driver"
            );
            for r in rows.iter().take(12) {
                println!(
                    "#{:<4} {:<12} {:<20} {:>6.1} {:>8.2} {:>8.2} {:>+8.2} {:>8.1}x  {}",
                    r.rank,
                    r.compound_id,
                    r.compound_name,
                    r.selectivity_discovery_score,
                    r.target_dg_kcal,
                    r.offtarget_dg_kcal,
                    r.ddg_selectivity_kcal,
                    r.selectivity_index_fold,
                    r.primary_selectivity_driver
                );
            }
            println!(
                "
Residue-Mutation ddG Scan on {} ({} in {} | Robustness {:.0}/100):",
                mut_scan.compound_name,
                mut_scan.compound_id,
                mut_scan.pdb_id,
                mut_scan.mutational_robustness_score
            );
            for e in mut_scan.entries.iter().take(8) {
                println!(
                    "  {:<8} [{:<24}] ddG={:+.2} kcal ({:>5.1}x) | {} | {}",
                    e.mutation_code,
                    e.category,
                    e.ddg_mutation_kcal,
                    e.fold_affinity_loss,
                    e.impact_class.short_label(),
                    e.structural_mechanism
                );
            }

            if let Some(ref csv_p) = export_csv {
                std::fs::write(csv_p, export_selectivity_screen_csv(&dual, &rows))
                    .expect("Failed to write selectivity CSV");
                println!("Wrote dual-pocket selectivity CSV -> {}", csv_p.display());
            }
            if let Some(ref mcsv_p) = export_mut_csv {
                std::fs::write(mcsv_p, export_mutation_scan_csv(&mut_scan))
                    .expect("Failed to write mutation ddG CSV");
                println!("Wrote residue-mutation ddG CSV -> {}", mcsv_p.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_dual_selectivity_poses_sdf(&dual, &rows, 20))
                    .expect("Failed to write selectivity poses SDF");
                println!("Wrote 3D selective poses SDF -> {}", sdf_p.display());
            }
        }
        Commands::DelLib {
            scheme,
            mode,
            prefilter,
            per_cycle,
            plate_size,
            pdb,
            preset,
            core_smiles,
            id,
            input,
            plate_only,
            export_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let scheme_idx = scheme
                .parse::<usize>()
                .ok()
                .filter(|&k| k < CURATED_LIBRARY_SCHEMES.len())
                .or_else(|| {
                    CURATED_LIBRARY_SCHEMES.iter().position(|sc| {
                        sc.code.eq_ignore_ascii_case(&scheme)
                            || sc.title.to_ascii_lowercase().contains(&scheme.to_ascii_lowercase())
                    })
                })
                .unwrap_or(0);
            let sc = &CURATED_LIBRARY_SCHEMES[scheme_idx];
            let synthesis_mode = match mode.to_ascii_lowercase().as_str() {
                "del" | "dna" | "on-dna" => LibrarySynthesisMode::DnaEncodedOnDna,
                "plate" | "medchem" | "parallel" => LibrarySynthesisMode::SolutionPhaseParallel,
                _ => sc.default_mode,
            };
            let prefilter_mode = match prefilter.to_ascii_lowercase().as_str() {
                "pocket" | "pdb" | "vina" => LibraryPreFilterMode::PdbPocketComplementarity,
                "rocs" | "3d" | "shape" => LibraryPreFilterMode::Rocs3dOverlay,
                "maxmin" | "diversity" => LibraryPreFilterMode::MaxMinDiversityFirst,
                _ => LibraryPreFilterMode::PocketAndRocsHybrid,
            };

            let pdb_raw = if let Some(ref p) = pdb {
                std::fs::read_to_string(p).unwrap_or_else(|e| {
                    eprintln!("Error reading PDB file {}: {e}", p.display());
                    std::process::exit(1);
                })
            } else {
                let pidx = CURATED_PDB_POCKETS
                    .iter()
                    .position(|pr| {
                        pr.code.eq_ignore_ascii_case(&preset)
                            || pr.pdb_id.eq_ignore_ascii_case(&preset)
                            || pr.target_name.eq_ignore_ascii_case(&preset)
                    })
                    .unwrap_or(0);
                curated_pocket_pdb_text(pidx).to_string()
            };
            let pocket_opt = parse_pdb_structure(&pdb_raw)
                .ok()
                .map(|parsed| analyze_binding_pocket(&parsed, None, 10.5));

            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let ref_mol = id
                .as_ref()
                .and_then(|q| {
                    ds.mols.iter().find(|m| {
                        m.id.eq_ignore_ascii_case(q) || m.name.eq_ignore_ascii_case(q)
                    })
                })
                .or_else(|| ds.mols.first());

            let cfg = ParallelLibConfig {
                scheme_idx,
                synthesis_mode,
                prefilter_mode,
                reagents_per_cycle: per_cycle,
                target_diverse_plate_size: plate_size,
                max_3d_scored_pool: (plate_size * 3).clamp(32, 120),
                custom_core_smiles: core_smiles,
                ..ParallelLibConfig::default()
            };
            let rep = enumerate_parallel_library(&cfg, ref_mol, pocket_opt.as_ref(), Some(&ds));

            println!(
                "=== SAR-Atlas v3.8.0 Combinatorial DEL & Parallel Library Enumerator ==="
            );
            println!(
                "Scheme & Format   : {} ({}) | {} ({}-step: {})",
                rep.scheme_code,
                rep.scheme_title,
                rep.synthesis_mode.label(),
                rep.step_count,
                rep.reaction_steps
                    .iter()
                    .map(|s| s.short_label())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            );
            println!(
                "Central Core      : {} ({}) | 3D Query: {} | PDB Pocket: {}",
                rep.core_name,
                rep.core_smiles,
                rep.reference_query_name,
                rep.reference_pocket_pdb.as_deref().unwrap_or("None")
            );
            println!(
                "Funnel & MaxMin   : Theoretical {} -> Unique {} -> Guardrail Pass {} -> MaxMin Plate {} ({:.1} ms)",
                rep.theoretical_combinatorial_size,
                rep.valid_unique_enumerated,
                rep.property_filter_passed,
                rep.selected_diverse_plate_count,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            println!(
                "Diversity & Hits  : ECFP4 Dissimilarity {:.3} -> {:.3} (Plate) | Scaffolds {} -> {} | Mean Fsp3 {:.2} | Fidelity {:.1}% | Pocket Hits {} | ROCS Hits {}",
                rep.mean_pairwise_dissimilarity_all,
                rep.mean_pairwise_dissimilarity_plate,
                rep.unique_murcko_scaffolds_all,
                rep.unique_murcko_scaffolds_plate,
                rep.mean_fsp3_plate,
                rep.mean_fidelity_pct_plate,
                rep.pocket_hit_count,
                rep.rocs_hit_count
            );
            println!(
                "{:<5} {:<5} {:<9} {:<16} {:>7} {:>8} {:>7} {:>7} {:>6} {:>6} {:>5}  {}",
                "Rank", "Pick", "Lib_ID", "Blocks", "Prio", "Pock_dG", "ROCS", "NearTc", "MW", "cLogP", "QED", "Barcode / Well"
            );
            for m in rep.members.iter().take(plate_size.min(25)) {
                let pick = m
                    .diversity_pick_order
                    .map(|k| format!("P{k}"))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "#{:<4} {:<5} {:<9} {:<16} {:>7.1} {:>8.2} {:>7.3} {:>7.2} {:>6.1} {:>6.2} {:>5.2}  {}",
                    m.rank,
                    pick,
                    m.library_id,
                    m.cycle_reagent_codes.join("+"),
                    m.composite_priority_score,
                    m.pocket_dg_kcal,
                    m.rocs_combo_score,
                    m.nearest_plate_tanimoto,
                    m.mw,
                    m.clogp,
                    m.qed,
                    m.barcode_or_well
                );
            }

            if let Some(ref csv_p) = export_csv {
                std::fs::write(csv_p, export_parallel_library_csv(&rep, plate_only))
                    .expect("Failed to write parallel library CSV");
                println!("Wrote combinatorial library CSV -> {}", csv_p.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_parallel_library_sdf(&rep, plate_size))
                    .expect("Failed to write parallel library 3D SDF");
                println!("Wrote 3D Pocket/ROCS-aligned plate SDF -> {}", sdf_p.display());
            }
        }
        Commands::Pocket {
            pdb,
            rcsb,
            preset,
            radius,
            smiles,
            id,
            input,
            export_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let pdb_raw = if let Some(ref p) = pdb {
                std::fs::read_to_string(p).unwrap_or_else(|e| {
                    eprintln!("Error reading PDB file {}: {e}", p.display());
                    std::process::exit(1);
                })
            } else if let Some(ref code) = rcsb {
                fetch_rcsb_pdb_text(code).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                })
            } else {
                let idx = CURATED_PDB_POCKETS
                    .iter()
                    .position(|pr| {
                        pr.code.eq_ignore_ascii_case(&preset)
                            || pr.pdb_id.eq_ignore_ascii_case(&preset)
                            || pr.target_name.eq_ignore_ascii_case(&preset)
                    })
                    .unwrap_or(0);
                curated_pocket_pdb_text(idx).to_string()
            };

            let parsed = parse_pdb_structure(&pdb_raw).unwrap_or_else(|e| {
                eprintln!("PDB parse error: {e}");
                std::process::exit(1);
            });
            let pocket = analyze_binding_pocket(&parsed, None, radius);
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);

            println!(
                "=== SAR-Atlas v3.8.0 PDB Binding-Site Cavity, WaterMap & Receptor Pharmacophore ==="
            );
            println!(
                "PDB Complex       : {} — {} | Center ({:.2}, {:.2}, {:.2}) A | Co-Crystal Ligand: {}",
                pocket.pdb_id,
                pocket.title,
                pocket.center[0],
                pocket.center[1],
                pocket.center[2],
                pocket.cocrystal_ligand_name.as_deref().unwrap_or("Apo / De Novo")
            );
            println!(
                "Cavity & DScore   : Vol {:.0} A^3 | Enclosure {:.0}% | Hydrophobic {:.0}% | DScore {:.2} ({})",
                pocket.volume_angstrom3,
                pocket.enclosure_fraction * 100.0,
                pocket.hydrophobic_fraction * 100.0,
                pocket.druggability_dscore,
                pocket.druggability_tier.label()
            );
            println!(
                "Hydration Sites   : {} total ({} Unhappy Displaceable Hotspots, {} Conserved Structural Bridges)",
                pocket.hydration_sites.len(),
                pocket.unhappy_water_count,
                pocket.structural_water_count
            );
            for w in &pocket.hydration_sites {
                println!(
                    "  [{}] ({:>6.2}, {:>6.2}, {:>6.2}) B={:>4.1} | HB={} | HydroEnc={} | dG_disp={:+.2} kcal/mol | {}",
                    w.site_id,
                    w.pos[0],
                    w.pos[1],
                    w.pos[2],
                    w.b_factor,
                    w.protein_hbond_count,
                    w.hydrophobic_enclosure_atoms,
                    w.displacement_ddg_bind_kcal,
                    w.role.short_label()
                );
            }
            println!(
                "Receptor Pharm3D  : {} projected cavity features",
                pocket.pharmacophore_points.len()
            );
            for fp in &pocket.pharmacophore_points {
                println!(
                    "  [{}] {:<16} <- {:<18} ({})",
                    fp.feature_id,
                    fp.kind.label(),
                    fp.receptor_origin,
                    fp.interaction_label
                );
            }

            let hits = if let Some(ref smi) = smiles {
                let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                    eprintln!("Invalid SMILES '{smi}': {e}");
                    std::process::exit(1);
                });
                m.id = "QUERY".to_string();
                m.name = "Query-Ligand".to_string();
                m.smiles = molecule_to_smiles(&m);
                compute_2d_coords(&mut m);
                vec![score_ligand_in_pocket(&m, None, &pocket)]
            } else if let Some(ref key) = id {
                let pos = ds
                    .mols
                    .iter()
                    .position(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .unwrap_or_else(|| {
                        eprintln!("Compound '{key}' not found.");
                        std::process::exit(1);
                    });
                vec![score_ligand_in_pocket(&ds.mols[pos], Some(pos), &pocket)]
            } else {
                screen_dataset_against_pocket(&ds, &pocket)
            };

            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "--- Pocket-Guided Complementarity Ranking ({} compounds, {ms:.1} ms) ---",
                hits.len()
            );
            for h in hits.iter().take(15) {
                println!(
                    "  #{:<2} {:<10} {:<20} | Score {:>5.1} | dG {:>6.2} kcal | pKd {:.2} | PockLE {:.2} | Pharm {:>3.0}% | UnhappyH2O {}",
                    h.rank,
                    h.compound_id,
                    h.compound_name,
                    h.complementarity_score,
                    h.total_dg_pocket_kcal,
                    h.predicted_pocket_pkd,
                    h.pocket_le,
                    h.pharmacophore_recall_pct,
                    h.displaced_unhappy_waters
                );
            }
            if let Some(ref csv_p) = export_csv {
                std::fs::write(csv_p, export_pocket_complementarity_csv(&pocket, &hits))
                    .expect("Failed to write pocket CSV");
                println!("Wrote pocket complementarity CSV -> {}", csv_p.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_pocket_poses_sdf(&pocket, &hits))
                    .expect("Failed to write pocket poses SDF");
                println!("Wrote 3D pocket-aligned V2000 SDF -> {}", sdf_p.display());
            }
        }
        Commands::DeNovo {
            smiles,
            id,
            input,
            goal,
            generations,
            pop_size,
            max_sa,
            seed,
            export_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let seed_mol = if let Some(ref smi) = smiles {
                let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                    eprintln!("Invalid SMILES '{smi}': {e}");
                    std::process::exit(1);
                });
                m.id = "SEED".to_string();
                m.name = "Query-Seed".to_string();
                m.target = "Target".to_string();
                m.metric = "IC50".to_string();
                m.raw_nm = 25.0;
                m.pact = 7.60;
                m.smiles = molecule_to_smiles(&m);
                compute_2d_coords(&mut m);
                m
            } else if let Some(ref key) = id {
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .cloned()
                    .unwrap_or_else(|| {
                        eprintln!("Compound '{key}' not found in dataset.");
                        std::process::exit(1);
                    })
            } else {
                ds.mols.first().cloned().unwrap_or_else(|| {
                    eprintln!("Dataset is empty.");
                    std::process::exit(1);
                })
            };

            let cfg = DeNovoConfig {
                goal: DeNovoGoalPreset::from_code(&goal),
                generations,
                population_size: pop_size,
                max_returned: 24,
                max_sa_score: max_sa,
                random_seed: seed,
                ..DeNovoConfig::default()
            };

            let rep = evolve_de_novo_molecules(&seed_mol, Some(&ds), cfg);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "=== SAR-Atlas v3.8.0 Evolutionary De Novo Molecule Generator ({ms:.1} ms) ==="
            );
            println!(
                "Seed Template     : {} ({}) | pAct {:.2} | Scaffold: {}",
                rep.query_name, rep.query_id, rep.query_pact, rep.query_scaffold_label
            );
            println!(
                "Goal & Exploration: {} | {} gens | {} unique sanitized graphs | {} novel scaffolds ({} true 3D/2D hops)",
                rep.config.goal.label(),
                rep.trajectory.len(),
                rep.total_unique_evaluated,
                rep.novel_scaffold_count,
                rep.true_scaffold_hop_count
            );
            for snap in &rep.trajectory {
                println!(
                    "  [Gen {:>2}] +{:>3} new | Best Fitness {:>5.1} | Top10 {:>5.1} | Best 3D/Pharm {:.3} | Best LLE {:.2} | Mean SA {:.2} | {} scaffolds ({:.0}% novel)",
                    snap.generation,
                    snap.evaluated_count,
                    snap.best_fitness,
                    snap.mean_top10_fitness,
                    snap.best_3d_pharm_sim,
                    snap.best_lle,
                    snap.mean_sa_score,
                    snap.unique_scaffolds,
                    snap.scaffold_hop_pct
                );
            }
            println!("--- Top Evolved De Novo Candidates ---");
            for c in rep.candidates.iter().take(15) {
                println!(
                    "  #{:<2} {} [G{:<2} {:<11}] Fit {:>5.1} | ROCS {:.3} | Pharm2D {:.3} | pAct {:.2} | LE {:.2}/LLE {:.2} | SA {:.2} | {} | {}",
                    c.rank,
                    c.candidate_id,
                    c.generation_born,
                    c.last_operator.short_label(),
                    c.fitness_score,
                    c.rocs_combo_tanimoto,
                    c.pharm2d_similarity,
                    c.predicted_pact,
                    c.le,
                    c.lle,
                    c.sa_score,
                    if c.is_true_scaffold_hop { "SCAFFOLD-HOP" } else if c.is_novel_scaffold { "Novel-Core" } else { "Analog" },
                    c.canonical_smiles
                );
            }
            if let Some(ref csv_p) = export_csv {
                std::fs::write(csv_p, export_de_novo_candidates_csv(&rep))
                    .expect("Failed to write evolved CSV");
                println!("Wrote {} evolved candidates -> {}", rep.candidates.len(), csv_p.display());
            }
            if let Some(ref sdf_p) = export_sdf {
                std::fs::write(sdf_p, format_de_novo_3d_sdf(&rep.candidates))
                    .expect("Failed to write 3D ROCS-aligned SDF");
                println!("Wrote 3D ROCS-aligned V2000 SDF -> {}", sdf_p.display());
            }
        }
        Commands::Dossier {
            smiles,
            id,
            input,
            profile,
            dose_mg,
            regimen,
            fed,
            inhibit_efflux,
            top_n,
            export_html,
            export_svg,
            export_csv,
        } => {
            let t0 = Instant::now();
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let prof_enum = CampaignTargetProfile::from_code(&profile);
            let reg_enum = match regimen.to_ascii_lowercase().as_str() {
                "qd" | "q24h" => DosingRegimen::OnceDailyQd48h,
                "bid" | "q12h" => DosingRegimen::TwiceDailyBid48h,
                _ => DosingRegimen::SingleDose24h,
            };
            let pbpk_cfg = PbpkDosingConfig {
                dose_mg,
                regimen: reg_enum,
                prandial: if fed {
                    PrandialState::FedBileMicelles
                } else {
                    PrandialState::Fasted
                },
                inhibit_bbb_efflux: inhibit_efflux,
            };

            if smiles.is_some() || id.is_some() {
                let (target_mol, ds_opt, idx_opt) = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-LEAD".to_string();
                    m.name = "Query-Lead".to_string();
                    m.target = "Target".to_string();
                    m.metric = "IC50".to_string();
                    m.raw_nm = 25.0;
                    m.pact = 7.60;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    (m, None, None)
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL553");
                    let pos = ds
                        .mols
                        .iter()
                        .position(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        });
                    (ds.mols[pos].clone(), Some(&ds), Some(pos))
                };

                let dossier = build_lead_campaign_dossier(
                    &target_mol,
                    ds_opt,
                    idx_opt,
                    prof_enum,
                    pbpk_cfg,
                );
                let sc = &dossier.scorecard;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "=== SAR-Atlas v3.8.0 Candidate Lead Dossier ({ms:.2} ms) ==="
                );
                println!(
                    "Compound          : {} ({}) | Target: {} | TPP: {}",
                    sc.name,
                    sc.id,
                    sc.target,
                    prof_enum.label()
                );
                println!(
                    "Desirability      : Derringer-Suich D = {:.1}/100 | Weighted = {:.1}/100 | Tier: {}",
                    sc.geometric_desirability,
                    sc.weighted_score_100,
                    sc.tier.label()
                );
                println!(
                    "Stage-Gate Tally  : {} PASS / {} WATCH / {} GAP (Rank #{}, {:.0}th percentile)",
                    sc.pass_gates,
                    sc.conditional_gates,
                    sc.fail_gates,
                    sc.rank,
                    dossier.campaign_percentile
                );
                for g in &dossier.gates {
                    println!(
                        "  [{}] {:<5} {:<40} | {:<34} (d={:.2})",
                        g.code,
                        g.verdict.label(),
                        g.metric_name,
                        g.observed_value,
                        g.desirability_0_1
                    );
                }
                if !dossier.optimization_actions.is_empty() {
                    println!("--- Prioritized Next-DMTA-Cycle Structural Proposals ---");
                    for act in &dossier.optimization_actions {
                        println!(
                            "  P{} [{}]: {} -> {}",
                            act.priority, act.domain, act.structural_proposal, act.quantitative_impact
                        );
                    }
                }
                if let Some(ref html_p) = export_html {
                    std::fs::write(html_p, export_lead_dossier_html(&dossier))
                        .expect("Failed to write HTML dossier");
                    println!("Wrote standalone HTML/SVG dossier -> {}", html_p.display());
                }
                if let Some(ref svg_p) = export_svg {
                    std::fs::write(svg_p, export_lead_dossier_svg(&dossier))
                        .expect("Failed to write SVG dossier poster");
                    println!("Wrote standalone SVG scorecard poster -> {}", svg_p.display());
                }
                if let Some(ref csv_p) = export_csv {
                    let camp = evaluate_campaign_scorecard(&ds, prof_enum, dose_mg);
                    std::fs::write(csv_p, export_campaign_scorecard_csv(&camp))
                        .expect("Failed to write scorecard CSV");
                    println!("Wrote campaign scorecard CSV -> {}", csv_p.display());
                }
            } else {
                let summary = evaluate_campaign_scorecard(&ds, prof_enum, dose_mg);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "=== SAR-Atlas v3.8.0 Campaign Lead Scorecard ({} compounds, {}, {:.2} ms) ===",
                    summary.total_compounds,
                    prof_enum.label(),
                    ms
                );
                println!(
                    "Candidate Ready: {} | Lead Opt: {} | Prodrug Rescue: {} | Hit/Triage: {} | Mean D_geom: {:.1} | Mean F_oral: {:.1}%",
                    summary.candidate_ready_count,
                    summary.lead_opt_count,
                    summary.prodrug_rescue_count,
                    summary.early_hit_count,
                    summary.mean_geometric_desirability,
                    summary.mean_f_oral_pct
                );
                for r in summary.rows.iter().take(15) {
                    println!(
                        "  #{:<2} {:<10} {:<20} | pAct {:.2} | LE {:.2}/LLE {:.1} | F_oral {:>5.1}% | Kp,uu {:.2} | D={:>5.1} | {}",
                        r.rank,
                        r.id,
                        r.name,
                        r.pact,
                        r.le,
                        r.lle,
                        r.f_oral_pct,
                        r.kp_uu_brain,
                        r.geometric_desirability,
                        r.tier.short_label()
                    );
                }
                if let Some(ref html_p) = export_html {
                    std::fs::write(
                        html_p,
                        export_campaign_dossier_html(&ds, &summary, top_n.max(1)),
                    )
                    .expect("Failed to write campaign HTML dossier");
                    println!(
                        "Wrote campaign leaderboard + top-{} HTML/SVG dossiers -> {}",
                        top_n.max(1),
                        html_p.display()
                    );
                }
                if let Some(ref svg_p) = export_svg {
                    if let Some(front_idx) = summary.front_runner_idx {
                        let doss = build_lead_campaign_dossier(
                            &ds.mols[front_idx],
                            Some(&ds),
                            Some(front_idx),
                            prof_enum,
                            pbpk_cfg,
                        );
                        std::fs::write(svg_p, export_lead_dossier_svg(&doss))
                            .expect("Failed to write front-runner SVG poster");
                        println!("Wrote front-runner SVG scorecard poster -> {}", svg_p.display());
                    }
                }
                if let Some(ref csv_p) = export_csv {
                    std::fs::write(csv_p, export_campaign_scorecard_csv(&summary))
                        .expect("Failed to write campaign scorecard CSV");
                    println!("Wrote campaign scorecard CSV -> {}", csv_p.display());
                }
            }
        }
        Commands::Pbpk {
            smiles,
            id,
            input,
            dose_mg,
            regimen,
            fed,
            inhibit_efflux,
            export_csv,
        } => {
            let t0 = Instant::now();
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let reg_enum = match regimen.to_ascii_lowercase().as_str() {
                "qd" | "q24h" => DosingRegimen::OnceDailyQd48h,
                "bid" | "q12h" => DosingRegimen::TwiceDailyBid48h,
                _ => DosingRegimen::SingleDose24h,
            };
            let cfg = PbpkDosingConfig {
                dose_mg,
                regimen: reg_enum,
                prandial: if fed {
                    PrandialState::FedBileMicelles
                } else {
                    PrandialState::Fasted
                },
                inhibit_bbb_efflux: inhibit_efflux,
            };

            if smiles.is_some() || id.is_some() {
                let target_mol = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-PBPK".to_string();
                    m.name = "Query-Compound".to_string();
                    m.target = "Target".to_string();
                    m.metric = "IC50".to_string();
                    m.raw_nm = 10.0;
                    m.pact = 8.0;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    m
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL54");
                    ds.mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        })
                };

                let rep = simulate_oral_pbpk(&target_mol, cfg);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "=== SAR-Atlas v3.8.0 Mechanistic Oral PBPK (ACAT + Q_gut + Hepatic First-Pass + Brain Kp,uu) ({ms:.2} ms) ==="
                );
                println!(
                    "Compound         : {} ({}) | {:.1} mg {} ({}) | Target {} IC50 {:.2} nM",
                    rep.compound_name,
                    rep.compound_id,
                    cfg.dose_mg,
                    cfg.regimen.short_label(),
                    cfg.prandial.short_label(),
                    rep.target_name,
                    rep.target_ic50_nm
                );
                println!(
                    "First-Pass F_oral: {:.1}%  [Fa(ACAT) = {:.1}% x Fg(Q_gut) = {:.1}% x Fh(Liver) = {:.1}%]",
                    rep.f_oral_pct,
                    rep.fa_integrated * 100.0,
                    rep.fg_gut_wall * 100.0,
                    rep.fh_hepatic * 100.0
                );
                println!(
                    "Distribution     : Vss = {:.2} L/kg ({:.1} L) | fu,p = {:.4} | R_bp = {:.2} | t1/2 = {:.1} h",
                    rep.vss_l_per_kg,
                    rep.vss_total_l,
                    rep.fu_plasma,
                    rep.blood_to_plasma_rbp,
                    rep.terminal_t_half_hr
                );
                println!(
                    "BBB & Brain ISF  : Kp,uu,brain = {:.3} ({}) | Total Kp,brain = {:.2} | fu,brain = {:.4} | Efflux ER = {:.2}x",
                    rep.kp_uu_brain,
                    rep.cns_class.short_label(),
                    rep.kp_brain_total,
                    rep.fu_brain,
                    rep.bbb_efflux_ratio
                );
                println!(
                    "Exposure & RO%   : Cu,p Cmax = {:.1} nM @ {:.2}h (T>IC50 {:.1}h, RO {:.0}%) | Cu,br Cmax = {:.1} nM @ {:.2}h (T>IC50 {:.1}h, Brain RO {:.0}%)",
                    rep.cmax_unbound_plasma_nm,
                    rep.tmax_plasma_hr,
                    rep.time_above_ic50_plasma_hr,
                    rep.peak_ro_plasma_pct,
                    rep.cmax_unbound_brain_nm,
                    rep.tmax_brain_hr,
                    rep.time_above_ic50_brain_hr,
                    rep.peak_ro_brain_pct
                );
                if let Some(ref out_csv) = export_csv {
                    let csv = export_pbpk_timecourse_csv(&rep);
                    std::fs::write(out_csv, csv).expect("Failed to write PBPK curve CSV");
                    println!("Wrote 97-point RK4 PBPK time-course -> {}", out_csv.display());
                }
            } else {
                let rows = summarize_dataset_oral_pbpk(&ds.mols, dose_mg);
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset Oral PBPK ({dose_mg:.0} mg Fasted) & Unbound Brain Kp,uu,brain Summary ({} compounds) ===",
                    rows.len()
                );
                for r in &rows {
                    println!(
                        "  {:<12} {:<20} | F_oral {:>4.1}% (Fa {:>3.0}/Fg {:>3.0}/Fh {:>3.0}) | Vss {:>5.2} L/kg | t1/2 {:>4.1}h | Kp,uu {:>5.3} ({:<13}) | Cu,br {:>6.1} nM (RO {:>3.0}%)",
                        r.id,
                        r.name,
                        r.f_oral_pct,
                        r.fa_pct,
                        r.fg_pct,
                        r.fh_pct,
                        r.vss_l_kg,
                        r.t_half_hr,
                        r.kp_uu_brain,
                        r.cns_class.short_label(),
                        r.cmax_u_brain_nm,
                        r.peak_ro_brain_pct
                    );
                }
                if let Some(ref out_csv) = export_csv {
                    let csv = export_dataset_pbpk_csv(&rows);
                    std::fs::write(out_csv, csv).expect("Failed to write dataset PBPK CSV");
                    println!("Wrote dataset PBPK & Kp,uu,brain summary -> {}", out_csv.display());
                }
            }
        }
        Commands::Prodrug {
            smiles,
            archetype,
            id,
            input,
            handle,
            export_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if smiles.is_some() || archetype.is_some() || id.is_some() {
                let parent_mol = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-PAR".to_string();
                    m.name = "Active-Parent".to_string();
                    m.target = "Query".to_string();
                    m.metric = "IC50".to_string();
                    m.pact = 7.80;
                    m.raw_nm = 15.8;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    m
                } else if let Some(ref ac) = archetype {
                    let aidx = ac
                        .parse::<usize>()
                        .ok()
                        .or_else(|| {
                            CURATED_PRODRUG_PARENTS.iter().position(|p| {
                                p.code.eq_ignore_ascii_case(ac) || p.name.eq_ignore_ascii_case(ac)
                            })
                        })
                        .unwrap_or(0);
                    build_curated_prodrug_parent(aidx)
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL118");
                    ds.mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        })
                };

                let rep = design_prodrugs(&parent_mol, handle);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let diag = &rep.diagnosis;
                println!(
                    "=== SAR-Atlas v3.8.0 Prodrug Promoiety Enumerator & Enzymatic Activation Profiler ({ms:.1} ms) ==="
                );
                println!(
                    "Active Parent    : {} ({}) | {} | Sol(pH6.8) {:.1} ug/mL (Do={:.1}x) | logD7.4 {:.2} | TPSA {:.0} A^2",
                    rep.parent_name,
                    rep.parent_id,
                    diag.bcs_class.short_label(),
                    diag.sol_ph68_ug_ml,
                    diag.dose_number_100mg,
                    diag.logd_74,
                    diag.tpsa
                );
                println!(
                    "Baseline Oral PK : Fa {:.0}% -> F_oral {:.0}% | Bottleneck: {}",
                    diag.parent_fa_pct, diag.parent_f_oral_pct, diag.primary_bottleneck
                );
                for h in &rep.handles {
                    let mark = if h.handle_idx == rep.active_handle_idx { "*" } else { " " };
                    println!(
                        "  [{mark}] H{:<2} {:<20} {:<36} | {}",
                        h.handle_idx + 1,
                        h.atom_label,
                        h.kind.label(),
                        h.rationale
                    );
                }
                println!(
                    "--- Ranked Bioreversible Prodrugs on H{} ({} structures) ---",
                    rep.active_handle_idx + 1,
                    rep.candidates.len()
                );
                for c in &rep.candidates {
                    let ac = &c.activation;
                    println!(
                        "  #{:<2} {:<12} {:<13} {:<12} | t1/2 {:>4.1}m (CES1/2 {:>4.1}x) | Sol {:>7.1} ({:>5.1}x) | Perm {:>4.2}x | F_oral {:>3.0}% ({:.2}x) | Score {:.1}",
                        c.rank,
                        c.promoiety_code,
                        c.family.short_label(),
                        ac.primary_enzyme.short_label(),
                        ac.activation_t_half_min,
                        ac.ces1_over_ces2_ratio,
                        ac.prodrug_sol_ph68_ug_ml,
                        ac.solubility_fold_change,
                        ac.permeability_fold_change,
                        ac.predicted_active_f_oral_pct,
                        ac.f_oral_fold_gain,
                        c.prodrug_score
                    );
                }
                if let Some(ref out_csv) = export_csv {
                    let csv = export_prodrugs_csv(&rep.candidates);
                    std::fs::write(out_csv, csv).expect("Failed to write prodrugs CSV");
                    println!("Wrote {} prodrug candidates -> {}", rep.candidates.len(), out_csv.display());
                }
                if let Some(ref out_sdf) = export_sdf {
                    let sdf = format_prodrugs_sdf(&rep.candidates);
                    std::fs::write(out_sdf, sdf).expect("Failed to write prodrugs SDF");
                    println!("Wrote {} 2D prodrug structures -> {}", rep.candidates.len(), out_sdf.display());
                }
            } else {
                let rows = summarize_dataset_prodrug_opportunities(&ds.mols);
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset BCS Liability & Best Prodrug Oral-Bioavailability Rescue ({} compounds) ===",
                    rows.len()
                );
                let mut csv_rows = String::from(
                    "id,name,bcs_class,parent_sol_ug_ml,parent_f_oral_pct,handles,best_promoiety,family,enzyme,rescued_f_oral_pct,f_oral_fold_gain
",
                );
                for r in &rows {
                    let fam = r.best_family.map(|f| f.short_label()).unwrap_or("-");
                    let enz = r.best_enzyme.map(|e| e.short_label()).unwrap_or("-");
                    println!(
                        "  {:<12} {:<20} | {:<17} | Sol {:>6.1} | F_oral {:>3.0}% -> {:>3.0}% ({:.2}x) | Hdl {} | {} ({}/{})",
                        r.id,
                        r.name,
                        r.bcs_class.short_label(),
                        r.parent_sol_ug_ml,
                        r.parent_f_oral_pct,
                        r.rescued_f_oral_pct,
                        r.f_oral_fold_gain,
                        r.handle_count,
                        r.best_promoiety,
                        fam,
                        enz
                    );
                    csv_rows.push_str(&format!(
                        "\"{}\",\"{}\",\"{}\",{:.1},{:.1},{},\"{}\",\"{}\",\"{}\",{:.1},{:.2}\n",
                        r.id,
                        r.name,
                        r.bcs_class.short_label(),
                        r.parent_sol_ug_ml,
                        r.parent_f_oral_pct,
                        r.handle_count,
                        r.best_promoiety,
                        fam,
                        enz,
                        r.rescued_f_oral_pct,
                        r.f_oral_fold_gain
                    ));
                }
                if let Some(ref out_csv) = export_csv {
                    std::fs::write(out_csv, csv_rows)
                        .expect("Failed to write dataset prodrug rescue CSV");
                    println!("Wrote dataset BCS & prodrug rescue summary -> {}", out_csv.display());
                }
            }
        }
        Commands::FragGrow {
            smiles,
            seed,
            id,
            input,
            vector,
            seed_pact,
            raw_parent,
            export_csv,
            export_sdf,
        } => {
            let t0 = Instant::now();
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if smiles.is_some() || seed.is_some() || id.is_some() {
                let (target_mol, deconstruct) = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-RO3".to_string();
                    m.name = "Query-Fragment".to_string();
                    m.target = "Query".to_string();
                    m.metric = "IC50".to_string();
                    m.pact = seed_pact;
                    m.raw_nm = 10f64.powf(9.0 - seed_pact);
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    (m, false)
                } else if let Some(ref sc) = seed {
                    let sidx = sc
                        .parse::<usize>()
                        .ok()
                        .or_else(|| {
                            CURATED_RO3_SEEDS.iter().position(|s| {
                                s.code.eq_ignore_ascii_case(sc) || s.name.eq_ignore_ascii_case(sc)
                            })
                        })
                        .unwrap_or(0);
                    (build_curated_ro3_seed_molecule(sidx), false)
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL553");
                    let found = ds
                        .mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        });
                    (found, !raw_parent)
                };

                let rep = grow_fragment_along_vector(
                    &target_mol,
                    vector,
                    deconstruct,
                    Some((&ds.mols, &ds.fps)),
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let aud = &rep.ro3_audit;
                println!(
                    "=== SAR-Atlas v3.8.0 Rule-of-3 Fragment Exit-Vector Grower & GE/LLE/LE Optimizer ({ms:.1} ms) ==="
                );
                println!(
                    "Seed Fragment    : {} ({}) | HAC {} | MW {:.1} Da | cLogP {:.2} (logD7.4 {:.2}) | Ro3 Viol: {}/6 (Sociability {:.0}/100)",
                    rep.seed_mol.name,
                    rep.seed_mol.id,
                    aud.heavy_atoms,
                    aud.mw,
                    aud.clogp,
                    aud.logd_74,
                    aud.ro3_violations,
                    aud.sociability_score
                );
                println!(
                    "Seed Efficiency  : pAct {:.2} | LE {:.3} kcal/mol/HA | LLE {:.2} | LLE_7.4 {:.2} | SILE {:.2}",
                    rep.seed_mol.pact, aud.seed_le, aud.seed_lle, aud.seed_lle_74, aud.seed_sile
                );
                for v in &rep.exit_vectors {
                    let mark = if v.vector_idx == rep.active_vector_idx { "*" } else { " " };
                    println!(
                        "  [{mark}] V{:<2} {:<14} {:<40} | Soc {:>4.1} | {}",
                        v.vector_idx + 1,
                        v.atom_label,
                        v.chemistry.label(),
                        v.sociability_weight,
                        v.rationale
                    );
                }
                println!(
                    "--- Ranked Grown Fragment-to-Lead Analogs on V{} ({} structures) ---",
                    rep.active_vector_idx + 1,
                    rep.candidates.len()
                );
                for c in &rep.candidates {
                    let e = &c.efficiency;
                    println!(
                        "  #{:<2} {:<12} {:<28} | HAC {:>2} (+{}) | pAct {:.2} ({:+.2}) | GE {:>+.3} | LE {:.3} | LLE {:>5.2} ({:+.2}) | QED {:.2} | SA {:.2} | {} ({:.1})",
                        c.rank,
                        c.synthon_code,
                        c.synthon_name,
                        e.grown_hac,
                        e.delta_hac,
                        e.predicted_pact,
                        e.delta_pact,
                        e.ge,
                        e.le,
                        e.lle,
                        e.delta_lle,
                        c.physchem.qed_mean,
                        c.sa_score,
                        c.regime.short_label(),
                        c.optimizer_score
                    );
                }
                if let Some(ref out_csv) = export_csv {
                    let csv = export_grown_fragments_csv(&rep.candidates);
                    std::fs::write(out_csv, csv).expect("Failed to write grown fragments CSV");
                    println!("Wrote {} grown analogs -> {}", rep.candidates.len(), out_csv.display());
                }
                if let Some(ref out_sdf) = export_sdf {
                    let sdf = format_grown_fragments_sdf(&rep.candidates);
                    std::fs::write(out_sdf, sdf).expect("Failed to write grown fragments SDF");
                    println!("Wrote {} grown 2D structures -> {}", rep.candidates.len(), out_sdf.display());
                }
            } else {
                let rows = summarize_dataset_fragment_efficiency(&ds.mols);
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset Rule-of-3, Ligand Efficiency (LE/LLE/SILE) & Sociability Audit ({} compounds) ===",
                    rows.len()
                );
                let mut csv_rows = String::from(
                    "id,name,hac,mw,pact,le_kcal_ha,lle,lle_74,sile,ro3_violations,strict_ro3_pass,exit_vectors,sociability_score
",
                );
                for r in &rows {
                    println!(
                        "  {:<12} {:<20} | HAC {:>2} | MW {:>5.1} | pAct {:.2} | LE {:.3} | LLE {:>5.2} | SILE {:.2} | Ro3 {}/6 | Vecs {:>2} (Soc {:>4.0})",
                        r.id,
                        r.name,
                        r.hac,
                        r.mw,
                        r.pact,
                        r.le,
                        r.lle,
                        r.sile,
                        r.ro3_violations,
                        r.exit_vector_count,
                        r.sociability_score
                    );
                    csv_rows.push_str(&format!(
                        "\"{}\",\"{}\",{},{:.1},{:.2},{:.3},{:.2},{:.2},{:.2},{},{},{},{:.1}\n",
                        r.id,
                        r.name,
                        r.hac,
                        r.mw,
                        r.pact,
                        r.le,
                        r.lle,
                        r.lle_74,
                        r.sile,
                        r.ro3_violations,
                        r.strict_ro3_pass,
                        r.exit_vector_count,
                        r.sociability_score
                    ));
                }
                if let Some(ref out_csv) = export_csv {
                    std::fs::write(out_csv, csv_rows)
                        .expect("Failed to write dataset fragment efficiency CSV");
                    println!("Wrote dataset Ro3 & LE/LLE audit -> {}", out_csv.display());
                }
            }
        }
        Commands::Macrocycle {
            smiles,
            id,
            input,
            pair,
            export_csv,
            export_3d_sdf,
        } => {
            let t0 = Instant::now();
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if smiles.is_some() || id.is_some() {
                let parent_mol = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-MAC".to_string();
                    m.name = "Acyclic-Parent".to_string();
                    m.target = "Query".to_string();
                    m.metric = "IC50".to_string();
                    m.raw_nm = 15.0;
                    m.pact = 7.82;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    m
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL553");
                    ds.mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        })
                };

                let rep = design_macrocycles(&parent_mol, pair);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "=== SAR-Atlas v3.8.0 Terminal-Pair RCM & Lactam Macrocyclizer + 3D Preorganization ({ms:.1} ms) ==="
                );
                println!(
                    "Acyclic Parent   : {} ({}) | pAct {:.2} | Parent Kier Phi {:.2} | Cyclizable Pairs: {}",
                    parent_mol.name,
                    parent_mol.id,
                    rep.parent_pact,
                    rep.parent_kier_phi,
                    rep.terminal_pairs.len()
                );
                for tp in &rep.terminal_pairs {
                    let mark = if tp.pair_idx == rep.active_pair_idx { "*" } else { " " };
                    println!(
                        "  [{mark}] Pair #{} {:<36} | Score {:>5.1} | {}",
                        tp.pair_idx + 1,
                        tp.label,
                        tp.suitability_score,
                        tp.rationale
                    );
                }
                println!(
                    "--- Ranked Macrocyclic Analogs ({} structures) ---",
                    rep.candidates.len()
                );
                for c in &rep.candidates {
                    println!(
                        "  #{:<2} {:<16} {:<24} | {:>2}-mer | Phi {:.2} ({:.1} rot) | -TdS {:>+5.2} + Strain {:>+5.2} = ddG {:>+5.2} kcal ({:>4.1}x -> pAct {:.2}) | CoreRMSD {:.3} A | {} | Score {:.1}",
                        c.rank,
                        c.tether_code,
                        c.reaction.label(),
                        c.preorg.macro_ring_size,
                        c.preorg.macro_kier_phi,
                        c.preorg.effective_rotors_frozen,
                        c.preorg.entropy_ddg_kcal,
                        c.preorg.core_strain_penalty_kcal,
                        c.preorg.net_ddg_bind_kcal,
                        c.preorg.potency_fold_boost,
                        c.preorg.predicted_macro_pact,
                        c.preorg.core_rmsd_3d_a,
                        c.druggability.short_label(),
                        c.macrocycle_score
                    );
                }
                if let Some(ref out_csv) = export_csv {
                    let csv = export_macrocycles_csv(&rep.candidates);
                    std::fs::write(out_csv, csv).expect("Failed to write macrocycles CSV");
                    println!("Wrote {} macrocycles -> {}", rep.candidates.len(), out_csv.display());
                }
                if let Some(ref out_sdf) = export_3d_sdf {
                    let sdf = format_macrocycles_3d_sdf(&rep.candidates);
                    std::fs::write(out_sdf, sdf).expect("Failed to write 3D macrocycles SDF");
                    println!(
                        "Wrote {} 3D MMFF-lite macrocycle conformers -> {}",
                        rep.candidates.len(),
                        out_sdf.display()
                    );
                }
            } else {
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset Macrocyclizability & 3D Preorganization Summary ({} compounds) ===",
                    ds.mols.len()
                );
                let mut csv_rows = String::from(
                    "id,name,parent_pact,kier_phi_parent,terminal_pairs,top_tether,reaction,ring_size,net_ddg_kcal,fold_boost,predicted_pact,core_rmsd_3d_a,druggability,macrocycle_score
",
                );
                for m in &ds.mols {
                    let phi = compute_kier_flexibility_phi(m);
                    let rep = design_macrocycles(m, 0);
                    if let Some(top) = rep.candidates.first() {
                        println!(
                            "  {:<12} {:<20} | Pairs {:>2} | Best: {:<15} ({:>2}-mer, {}) | ddG {:>+5.2} kcal ({:>4.1}x -> pAct {:.2}) | RMSD {:.3} A | {}",
                            m.id,
                            m.name,
                            rep.terminal_pairs.len(),
                            top.tether_code,
                            top.preorg.macro_ring_size,
                            top.reaction.label(),
                            top.preorg.net_ddg_bind_kcal,
                            top.preorg.potency_fold_boost,
                            top.preorg.predicted_macro_pact,
                            top.preorg.core_rmsd_3d_a,
                            top.druggability.short_label()
                        );
                        csv_rows.push_str(&format!(
                            "\"{}\",\"{}\",{:.2},{:.2},{},\"{}\",\"{}\",{},{:+.2},{:.1},{:.2},{:.3},\"{}\",{:.1}\n",
                            m.id,
                            m.name,
                            m.pact,
                            phi,
                            rep.terminal_pairs.len(),
                            top.tether_code,
                            top.reaction.label(),
                            top.preorg.macro_ring_size,
                            top.preorg.net_ddg_bind_kcal,
                            top.preorg.potency_fold_boost,
                            top.preorg.predicted_macro_pact,
                            top.preorg.core_rmsd_3d_a,
                            top.druggability.short_label(),
                            top.macrocycle_score
                        ));
                    } else {
                        println!(
                            "  {:<12} {:<20} | Pairs  0 | Compact/rigid scaffold (Phi {:.2})",
                            m.id, m.name, phi
                        );
                    }
                }
                if let Some(ref out_csv) = export_csv {
                    std::fs::write(out_csv, csv_rows)
                        .expect("Failed to write dataset macrocyclizability CSV");
                    println!("Wrote dataset macrocyclizability summary -> {}", out_csv.display());
                }
            }
        }
        Commands::Metabolism {
            smiles,
            id,
            input,
            sequential_phase2,
            export_csv,
            export_metabolites_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if smiles.is_some() || id.is_some() {
                let target_mol = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-MET".to_string();
                    m.name = "Query-Substrate".to_string();
                    m.target = "Query".to_string();
                    m.metric = "IC50".to_string();
                    m.raw_nm = 10.0;
                    m.pact = 8.0;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    m
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL118");
                    ds.mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        })
                };

                let rep = analyze_cyp_metabolism(&target_mol, sequential_phase2);
                println!(
                    "=== SAR-Atlas v3.8.0 SMARTCyp (3A4/2D6/2C9) Regioselectivity & Phase I/II Metabolite Report ==="
                );
                println!(
                    "Substrate        : {} ({}) | MW {:.2} Da | logD7.4 {:.2}",
                    target_mol.name, target_mol.id, rep.parent_mw, rep.parent_logd_74
                );
                println!(
                    "Hepatic Turnover : CL_int,u = {:.1} mL/min/kg (t1/2 ~ {:.0} min, {}) | fm: 3A4 {:.0}%, 2D6 {:.0}%, 2C9 {:.0}%",
                    rep.cl_int_ml_min_kg,
                    rep.predicted_t_half_min,
                    rep.stability_class.label(),
                    rep.isoform_fraction_fm[0] * 100.0,
                    rep.isoform_fraction_fm[1] * 100.0,
                    rep.isoform_fraction_fm[2] * 100.0
                );
                println!("--- Top Ranked SMARTCyp Sites of Metabolism ({}) ---", rep.sites.len());
                for (rk, s) in rep.sites.iter().take(6).enumerate() {
                    println!(
                        "  #{:<2} {:<6} {:<32} | Ea {:>5.1} kJ | Acc {:.2} | 3A4 {:>5.1} (#{}) | 2D6 {:>5.1} (#{}) | 2C9 {:>5.1} (#{}) | Dom: {}",
                        rk + 1,
                        s.atom_label,
                        s.reaction_kind.label(),
                        s.dft_ea_kj_mol,
                        s.accessibility_a,
                        s.score_3a4,
                        s.rank_3a4,
                        s.score_2d6,
                        s.rank_2d6,
                        s.score_2c9,
                        s.rank_2c9,
                        s.dominant_isoform.label()
                    );
                    println!("      Block Tip: {}", s.blocking_strategy);
                }
                println!(
                    "--- Enumerated Phase I & Phase II Biotransformation Metabolites ({}) ---",
                    rep.metabolites.len()
                );
                for m in &rep.metabolites {
                    let alert_tag = if m.is_reactive_bioactivation {
                        "[BIOACT]"
                    } else {
                        "        "
                    };
                    println!(
                        "  {:<4} {} [{:<10} {:<15}] {:<44} | dM {:>+7.2} Da | logD {:>5.2} ({:>+5.2}) | {:>4.1}% | {}",
                        m.metabolite_id,
                        alert_tag,
                        m.phase.short_label(),
                        m.enzyme,
                        m.reaction_name,
                        m.mass_shift_da,
                        m.logd_74,
                        m.delta_logd_74,
                        m.predicted_abundance_pct,
                        m.canonical_smiles
                    );
                }

                if let Some(csv_out) = export_csv {
                    std::fs::write(&csv_out, export_metabolites_csv(&rep))
                        .expect("Failed to write metabolites CSV");
                    println!("Exported Phase I/II metabolites CSV -> {}", csv_out.display());
                }
                if let Some(sdf_out) = export_metabolites_sdf {
                    let mols: Vec<_> = rep.metabolites.iter().map(|m| m.mol.clone()).collect();
                    std::fs::write(&sdf_out, format_sdf_dataset(&mols))
                        .expect("Failed to write metabolites SDF");
                    println!(
                        "Exported {} V2000 metabolites -> {}",
                        mols.len(),
                        sdf_out.display()
                    );
                }
            } else {
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset SMARTCyp Clearance & Bioactivation Summary ({} compounds) ===",
                    ds.mols.len()
                );
                for (i, m) in ds.mols.iter().enumerate() {
                    let mt = &ds.metabolism[i];
                    println!(
                        "  {:<22} ({:<11}) Top SoM {:<6} {:<30} | {:<6} {:>5.1} kJ | CLint {:>5.1} ({:<8}) | Bioact: {}",
                        m.name,
                        m.target,
                        mt.top_som_label,
                        mt.top_reaction,
                        mt.primary_isoform.label(),
                        mt.best_smartcyp_score,
                        mt.cl_int_ml_min_kg,
                        mt.stability_class.short_label(),
                        if mt.reactive_bioactivation_risk { "RISK" } else { "Clean" }
                    );
                }
                if let Some(csv_out) = export_csv {
                    std::fs::write(&csv_out, ds.export_metabolism_summary_csv())
                        .expect("Failed to write dataset metabolism CSV");
                    println!(
                        "Exported dataset SMARTCyp clearance CSV -> {}",
                        csv_out.display()
                    );
                }
            }
        }
        Commands::Protac {
            smiles,
            id,
            input,
            e3,
            exit_vector,
            poi_nm,
            e3_nm,
            top,
            export_csv,
            export_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let poi_mol = if let Some(ref smi) = smiles {
                let mut m = parse_smiles(smi).unwrap_or_else(|err| {
                    eprintln!("Invalid SMILES '{smi}': {err}");
                    std::process::exit(1);
                });
                m.id = "CLI-POI".to_string();
                m.name = "POI-Warhead".to_string();
                m.target = "Target-POI".to_string();
                m.metric = "IC50".to_string();
                m.raw_nm = 10.0;
                m.pact = 8.0;
                m.smiles = molecule_to_smiles(&m);
                compute_2d_coords(&mut m);
                m
            } else {
                let key = id.as_deref().unwrap_or("CHEMBL939");
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .cloned()
                    .unwrap_or_else(|| {
                        eprintln!("Compound '{key}' not found in dataset.");
                        std::process::exit(1);
                    })
            };

            let e3_filter: Vec<E3RecruiterKind> = E3RecruiterKind::ALL
                .iter()
                .copied()
                .filter(|k| {
                    e3.eq_ignore_ascii_case("All")
                        || k.ligase_family().eq_ignore_ascii_case(&e3)
                })
                .collect();

            let t0 = Instant::now();
            let rep = design_protac_series(&poi_mol, &e3_filter, exit_vector, poi_nm, e3_nm);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!(
                "=== SAR-Atlas v3.8.0 PROTAC (CRBN/VHL) Linkerology, bRo5 Chameleonicity & Hook-Effect Studio ({ms:.1} ms) ==="
            );
            println!(
                "POI Warhead      : {} ({}) | Binary Kd = {:.2} nM | [POI] = {:.0} nM, [E3] = {:.0} nM",
                rep.poi_name, poi_mol.id, rep.poi_kd_nm, rep.poi_conc_nm, rep.e3_conc_nm
            );
            println!("--- Perceived Solvent-Exposed POI Exit Vectors ({}) ---", rep.exit_vectors.len());
            for ev in &rep.exit_vectors {
                let mark = if ev.vector_idx == rep.active_exit_vector_idx { "*" } else { " " };
                println!(
                    "  [{}EV{}] {:<34} (Score {:.1}) — {}",
                    mark, ev.vector_idx, ev.label, ev.exposure_score, ev.rationale
                );
            }
            println!(
                "--- Top Ranked Heterobifunctional PROTAC Degraders ({} assembled) ---",
                rep.candidates.len()
            );
            for c in rep.candidates.iter().take(top) {
                println!(
                    "  #{:<2} {:<32} + {:<13} | MW {:>5.0} | TPSA {:>5.0}->ePSA {:>5.0} ({} dIMHB, Cham {:.2}) | AB-MPS {:>4.1} | alpha {:>4.2} | TF50 {:>5.2} nM .. TI50 {:>6.0} nM | Score {:>4.1}",
                    c.rank,
                    c.e3_kind.name(),
                    c.linker_code,
                    c.physchem.mw,
                    c.chameleon.tpsa_open_a2,
                    c.chameleon.epsa_folded_a2,
                    c.chameleon.dimhb_count,
                    c.chameleon.chameleonicity_index,
                    c.chameleon.abbv_mps,
                    c.ternary.cooperativity_alpha,
                    c.ternary.tf50_nm,
                    c.ternary.ti50_nm,
                    c.degrader_score
                );
            }

            if let Some(csv_out) = export_csv {
                std::fs::write(&csv_out, export_protac_candidates_csv(&rep.candidates))
                    .expect("Failed to write PROTAC CSV");
                println!("Exported PROTAC chimeras CSV -> {}", csv_out.display());
            }
            if let Some(sdf_out) = export_sdf {
                let mols: Vec<_> = rep.candidates.iter().map(|c| c.mol.clone()).collect();
                std::fs::write(&sdf_out, format_sdf_dataset(&mols))
                    .expect("Failed to write PROTAC SDF");
                println!(
                    "Exported {} V2000 PROTAC chimeras -> {}",
                    mols.len(),
                    sdf_out.display()
                );
            }
        }
        Commands::Warheads {
            smiles,
            id,
            input,
            conc_nm,
            export_csv,
            export_swaps_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if smiles.is_some() || id.is_some() {
                let target_mol = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-COV".to_string();
                    m.name = "Covalent-Query".to_string();
                    m.target = "EGFR".to_string();
                    m.metric = "IC50".to_string();
                    m.raw_nm = 5.0;
                    m.pact = 8.30;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    m
                } else {
                    let key = id.as_deref().unwrap_or("CHEMBL1173655");
                    ds.mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        })
                };

                let rep = analyze_covalent_warheads_and_swaps(
                    &target_mol,
                    &ds,
                    ds.default_sar_matrix.as_ref(),
                    conc_nm,
                );
                println!(
                    "=== SAR-Atlas v3.8.0 Covalent Electrophile Reactivity, GSH Half-Life & Warhead Swapper ==="
                );
                println!(
                    "Compound         : {} ({}) | Window: {} | [I] = {:.1} nM",
                    target_mol.name,
                    target_mol.id,
                    rep.primary_window.label(),
                    rep.inhibitor_conc_nm
                );
                if rep.detected.is_empty() {
                    println!("Detected Warhead : None (Non-covalent scaffold — running Ligand-First Covalentization)");
                } else {
                    for dw in &rep.detected {
                        println!(
                            "Detected Warhead : {} [{}] -> {} ({})",
                            dw.warhead_name,
                            dw.warhead_code,
                            dw.target_residue.label(),
                            dw.reversibility.label()
                        );
                        println!(
                            "GSH Reactivity   : base t1/2 {:.2} h | d_ind {:+.2} -> tuned GSH t1/2 = {:.2} h ({:.0} min) | k_GSH = {:.3} h^-1",
                            dw.base_gsh_half_life_h,
                            dw.inductive_shift_log10,
                            dw.gsh_half_life_h,
                            dw.gsh_half_life_h * 60.0,
                            dw.k_gsh_per_h
                        );
                        println!(
                            "Two-Step Kinetics: K_I = {:.2} nM | k_inact = {:.4} min^-1 | k_inact/K_I = {:.2e} M^-1 s^-1 | tau = {:.1} h",
                            dw.ki_nm,
                            dw.k_inact_per_min,
                            dw.kinact_over_ki_m_inv_s_inv,
                            dw.residence_time_h
                        );
                    }
                }

                println!(
                    "--- Enumerated Reversible & Irreversible Covalent Warhead Swap Panel ({}) ---",
                    rep.swaps.len()
                );
                for s in &rep.swaps {
                    let gsh_txt = s
                        .gsh_half_life_h
                        .map(|v| format!("{v:>6.2} h"))
                        .unwrap_or_else(|| "  NonCov".to_string());
                    println!(
                        "  #{:<2} {:<36} | {:<14} | GSH {} | k_inact/K_I {:>8.2e} | QED {:.2} | SA {:.2} | TCI {:>4.1}",
                        s.rank,
                        s.template_name,
                        s.reversibility.short_label(),
                        gsh_txt,
                        s.kinact_over_ki,
                        s.qed,
                        s.sa_score,
                        s.tci_selectivity_score
                    );
                }

                if let Some(csv_out) = export_csv {
                    std::fs::write(&csv_out, export_warhead_swaps_csv(&rep.swaps))
                        .expect("Failed to write warhead swaps CSV");
                    println!("Exported warhead swap panel CSV -> {}", csv_out.display());
                }
                if let Some(sdf_out) = export_swaps_sdf {
                    let mols: Vec<_> = rep.swaps.iter().map(|s| s.mol.clone()).collect();
                    std::fs::write(&sdf_out, format_sdf_dataset(&mols))
                        .expect("Failed to write warhead swaps SDF");
                    println!(
                        "Exported {} warhead-swapped V2000 analogs -> {}",
                        mols.len(),
                        sdf_out.display()
                    );
                }
            } else {
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset Covalent Warhead & GSH Reactivity Summary ({} compounds) ===",
                    ds.mols.len()
                );
                for (i, m) in ds.mols.iter().enumerate() {
                    let cv = &ds.covalent[i];
                    let gsh = cv
                        .gsh_half_life_h
                        .map(|v| format!("{v:>6.2} h"))
                        .unwrap_or_else(|| "       -".to_string());
                    println!(
                        "  {:<22} ({:<11}) {:<34} | {:<16} | GSH {} | SwapHandle: {}",
                        m.name,
                        m.target,
                        cv.primary_warhead_name,
                        cv.window.short_label(),
                        gsh,
                        if cv.swap_handle_available { "Yes" } else { "No" }
                    );
                }
                if let Some(csv_out) = export_csv {
                    std::fs::write(&csv_out, ds.export_covalent_summary_csv())
                        .expect("Failed to write dataset covalent CSV");
                    println!(
                        "Exported dataset covalent reactivity CSV -> {}",
                        csv_out.display()
                    );
                }
            }
        }
        Commands::Overlay3d {
            smiles,
            id,
            input,
            conformers,
            scaffold_hops_only,
            top,
            export_csv,
            export_overlay_sdf,
            export_conformers_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let query_mol = if let Some(ref smi) = smiles {
                let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                    eprintln!("Invalid SMILES '{smi}': {e}");
                    std::process::exit(1);
                });
                m.id = "CLI-3D".to_string();
                m.name = "3D-Query-Analog".to_string();
                m.target = "Query".to_string();
                m.metric = "IC50".to_string();
                m.raw_nm = 10.0;
                m.pact = 8.0;
                m.smiles = molecule_to_smiles(&m);
                m
            } else {
                let key = id.as_deref().unwrap_or("CHEMBL118");
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .cloned()
                    .unwrap_or_else(|| {
                        eprintln!("Compound '{key}' not found in dataset.");
                        std::process::exit(1);
                    })
            };

            let t0 = Instant::now();
            let cfg = ConformerGenConfig {
                num_trial_conformers: conformers.clamp(1, 32),
                max_kept_conformers: 6,
                mmff_max_iters: 120,
                rmsd_prune_cutoff: 0.45,
                energy_window_kcal: 12.0,
                random_seed: 0x5A17_3D12,
            };
            let ens = generate_3d_conformer_ensemble(&query_mol, &cfg);
            let q_sum = evaluate_3d_summary_profile(&query_mol, false);
            let hits = screen_3d_overlay_library(
                &query_mol,
                &q_sum,
                &ds,
                None,
                scaffold_hops_only,
                ds.mols.len().max(50),
            );
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!(
                "=== SAR-Atlas v3.8.0 Distance-Geometry 3D Conformer, MMFF-Lite & ROCS Shape-ESP Report ({ms:.1} ms) ==="
            );
            println!(
                "Query Molecule   : {} ({}) | {} rotatable bonds | {} kept conformers (from {} trials)",
                query_mol.name,
                query_mol.id,
                ens.rotatable_dihedrals,
                ens.conformers.len(),
                ens.trial_count
            );
            println!(
                "3D Descriptors   : {} (NPR1 {:.3}, NPR2 {:.3}) | PBF {:.3} A | Rg {:.2} A | Vol {:.1} A^3 | Dipole {:.2} D",
                q_sum.shape_class,
                q_sum.npr1,
                q_sum.npr2,
                q_sum.pbf_angstrom,
                q_sum.rg_angstrom,
                q_sum.gaussian_volume,
                q_sum.dipole_debye
            );
            println!("--- Minimized MMFF-Lite 3D Conformer Ensemble ---");
            for c in &ens.conformers {
                println!(
                    "  Conf #{:<2} | E_tot {:>7.2} kcal/mol (dE {:>+5.2}, {:>5.1}% Boltz) | RMSD {:>4.2} A | NPR ({:.2},{:.2}) | PBF {:.2} A | mu {:.2} D",
                    c.rank,
                    c.energy.total_kcal,
                    c.rel_energy_kcal,
                    c.boltzmann_weight * 100.0,
                    c.rmsd_to_lowest,
                    c.npr1,
                    c.npr2,
                    c.pbf_angstrom,
                    c.dipole_debye
                );
            }
            println!(
                "--- Top Ranked ROCS 3D Shape + Color/ESP Overlay Hits ({}) ---",
                if scaffold_hops_only { "Scaffold Hops Only" } else { "All Candidates" }
            );
            for (rk, h) in hits.iter().take(top).enumerate() {
                let hop_tag = if h.is_scaffold_hop { "[3D-HOP]" } else { "        " };
                println!(
                    "  #{:<2} {} {:<22} ({:<11} pAct {:.2}) | Combo {:.3}/2.0 | Shape {:.3} | Color {:.3} | USRCAT {:.3} | 2D {:.2}",
                    rk + 1,
                    hop_tag,
                    h.compound_name,
                    h.target,
                    h.pact,
                    h.alignment.combo_score,
                    h.alignment.shape_tanimoto,
                    h.alignment.color_tanimoto,
                    h.alignment.usrcat_similarity,
                    h.ecfp4_tanimoto_2d
                );
            }

            if let Some(csv_out) = export_csv {
                std::fs::write(&csv_out, export_3d_overlay_hits_csv(&hits))
                    .expect("Failed to write 3D overlay CSV");
                println!("Exported 3D ROCS/USR virtual screening CSV -> {}", csv_out.display());
            }
            if let Some(sdf_out) = export_overlay_sdf {
                if let Some(top_hit) = hits
                    .iter()
                    .find(|h| h.compound_id != query_mol.id)
                    .or_else(|| hits.first())
                {
                    let hm = &ds.mols[top_hit.mol_idx];
                    let hs = &ds.shape3d[top_hit.mol_idx];
                    let sdf_txt = format_3d_overlay_sdf(
                        &query_mol,
                        &q_sum,
                        hm,
                        &hs.gasteiger_charges,
                        hs.mmff_energy_kcal,
                        &top_hit.alignment,
                    );
                    std::fs::write(&sdf_out, sdf_txt).expect("Failed to write 3D overlay SDF");
                    println!(
                        "Exported superimposed 3D V2000 SDF ({} + {}) -> {}",
                        query_mol.name,
                        hm.name,
                        sdf_out.display()
                    );
                }
            }
            if let Some(conf_sdf_out) = export_conformers_sdf {
                let mut all_sdf = String::new();
                for c in &ens.conformers {
                    let tags = [
                        ("CONFORMER_RANK", format!("{}", c.rank)),
                        ("REL_ENERGY_KCAL", format!("{:.3}", c.rel_energy_kcal)),
                        ("BOLTZMANN_WEIGHT", format!("{:.4}", c.boltzmann_weight)),
                        ("NPR1_NPR2", format!("{:.3},{:.3}", c.npr1, c.npr2)),
                    ];
                    all_sdf.push_str(&format_3d_conformer_sdf(
                        &query_mol,
                        &c.coords3d,
                        &ens.gasteiger_charges,
                        c.energy.total_kcal,
                        &tags,
                    ));
                }
                std::fs::write(&conf_sdf_out, all_sdf)
                    .expect("Failed to write 3D conformers SDF");
                println!(
                    "Exported {} 3D V2000 conformers -> {}",
                    ens.conformers.len(),
                    conf_sdf_out.display()
                );
            }
        }
        Commands::Pka {
            smiles,
            id,
            input,
            ph,
            export_csv,
            export_protomers_sdf,
        } => {
            let target_ph = ph.clamp(0.0, 14.0);
            if smiles.is_some() || id.is_some() {
                let mol = if let Some(ref smi) = smiles {
                    let mut m = parse_smiles(smi).unwrap_or_else(|e| {
                        eprintln!("Invalid SMILES '{smi}': {e}");
                        std::process::exit(1);
                    });
                    m.id = "CLI-PKA".to_string();
                    m.name = "Query-Analog".to_string();
                    m.target = "Query".to_string();
                    m.metric = "IC50".to_string();
                    m.raw_nm = 10.0;
                    m.pact = 8.0;
                    m.smiles = molecule_to_smiles(&m);
                    compute_2d_coords(&mut m);
                    m
                } else {
                    let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
                    let key = id.as_deref().unwrap_or("");
                    ds.mols
                        .iter()
                        .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Compound '{key}' not found in dataset.");
                            std::process::exit(1);
                        })
                };
                let rep = analyze_pka_and_protomers(&mol, target_ph);
                let pi_txt = rep
                    .isoelectric_pi
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "None".to_string());
                println!(
                    "=== SAR-Atlas v3.8.0 Physiological Micro-pKa, Protomer & Tautomer Report ==="
                );
                println!(
                    "Compound         : {} ({}) | SMILES: {}",
                    mol.name, mol.id, mol.smiles
                );
                println!(
                    "State @ pH {:>4.1}  : {} | <Q> = {:+.2} | f_neutral = {:.1}% | pI = {}",
                    rep.target_ph,
                    rep.charge_class_at_ph.label(),
                    rep.mean_charge_at_ph,
                    rep.neutral_fraction_at_ph * 100.0,
                    pi_txt
                );
                println!(
                    "Compartment logD : Gastric (pH 2.0) = {:.2} | Jejunal (pH 6.5) = {:.2} | Plasma (pH 7.4) = {:.2} | @ pH {:.1} = {:.2}",
                    rep.gastric_logd_20,
                    rep.jejunal_logd_65,
                    rep.plasma_logd_74,
                    rep.target_ph,
                    rep.logd_at_ph
                );
                println!("--- Perceived Ionizable Micro-pKa Centers ({}) ---", rep.sites.len());
                for s in &rep.sites {
                    println!(
                        "  Atom #{:<3} ({:<6}) {:<34} | pKa0 {:>5.2} | d_ind {:>+5.2} -> micro-pKa {:>5.2} ({:>5.1}% ionized)",
                        s.atom_idx + 1,
                        s.kind.label(),
                        s.moiety_name,
                        s.base_pka,
                        s.inductive_shift,
                        s.micro_pka,
                        s.ionized_fraction_at_ph * 100.0
                    );
                }
                println!("--- Coupled Protomer Charge Microspecies @ pH {:.1} ---", rep.target_ph);
                for pr in rep.protomers.iter().take(8) {
                    println!(
                        "  #{:<2} Q={:+2}{} | {:>6.2}% | {:<32} | {}",
                        pr.rank,
                        pr.net_charge,
                        if pr.is_zwitterion { " (Zwit)" } else { "       " },
                        pr.fraction_at_target_ph * 100.0,
                        pr.state_label,
                        pr.canonical_smiles
                    );
                }
                println!("--- Prototropic Tautomers (MolVS 298 K Boltzmann Ranking) ---");
                for t in &rep.tautomers {
                    println!(
                        "  #{:<2} {:<32} | Score {:>5.1} | dG {:>+5.2} kcal/mol | {:>5.1}% | {}",
                        t.rank,
                        t.rule_name,
                        t.molvs_score,
                        t.rel_energy_kcal,
                        t.boltzmann_fraction * 100.0,
                        t.canonical_smiles
                    );
                }
                if let Some(sdf_out) = export_protomers_sdf {
                    let prot_mols: Vec<_> = rep.protomers.iter().map(|p| p.mol.clone()).collect();
                    std::fs::write(&sdf_out, format_sdf_dataset(&prot_mols))
                        .expect("Failed to write protomers SDF");
                    println!(
                        "Exported {} V2000 protomer microspecies -> {}",
                        prot_mols.len(),
                        sdf_out.display()
                    );
                }
            } else {
                let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
                println!(
                    "=== SAR-Atlas v3.8.0 Dataset Physiological pH 7.4 Micro-pKa & Protomer Summary ({} compounds) ===",
                    ds.mols.len()
                );
                for (i, m) in ds.mols.iter().enumerate() {
                    let pk = &ds.pka_74[i];
                    let ac = pk
                        .strongest_acid_pka
                        .map(|v| format!("{v:>5.2}"))
                        .unwrap_or_else(|| "    -".to_string());
                    let bs = pk
                        .strongest_base_pka
                        .map(|v| format!("{v:>5.2}"))
                        .unwrap_or_else(|| "    -".to_string());
                    println!(
                        "  {:<22} ({:<11}) {:<14} | <Q> {:>+5.2} | f0 {:>5.1}% | Acid {} | Base {} | logD7.4 {:>5.2} | {} prot / {} taut",
                        m.name,
                        m.target,
                        pk.charge_class.label(),
                        pk.mean_net_charge_74,
                        pk.neutral_fraction_74 * 100.0,
                        ac,
                        bs,
                        pk.microstate_logd_74,
                        pk.major_protomers_74,
                        pk.tautomer_count
                    );
                }
                if let Some(csv_out) = export_csv {
                    std::fs::write(&csv_out, ds.export_pka_protomers_csv())
                        .expect("Failed to write pH 7.4 CSV");
                    println!(
                        "Exported dataset pH 7.4 micro-pKa & protomer CSV -> {}",
                        csv_out.display()
                    );
                }
            }
        }
        Commands::Mcs {
            input,
            target,
            pair_a,
            pair_b,
            any_ring_hetero,
            threshold,
            export_csv,
            export_aligned_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let mut cfg = McsConfig::default();
            if any_ring_hetero {
                cfg.atom_mode = McsAtomMatchMode::AnyRingHetero;
            }
            cfg.threshold_fraction = threshold.clamp(0.35, 1.0);

            if let (Some(ka), Some(kb)) = (pair_a, pair_b) {
                let ma = ds
                    .mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(&ka) || m.name.eq_ignore_ascii_case(&ka))
                    .unwrap_or_else(|| {
                        eprintln!("Compound '{ka}' not found");
                        std::process::exit(1);
                    });
                let mb = ds
                    .mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(&kb) || m.name.eq_ignore_ascii_case(&kb))
                    .unwrap_or_else(|| {
                        eprintln!("Compound '{kb}' not found");
                        std::process::exit(1);
                    });
                match find_pairwise_mcs(ma, mb, &cfg) {
                    Some(res) => {
                        println!(
                            "=== SAR-Atlas v3.8.0 Pairwise McGregor/VF2 Maximum Common Substructure ==="
                        );
                        println!(
                            "Pair             : {} ({}) <-> {} ({})",
                            ma.name, ma.id, mb.name, mb.id
                        );
                        println!(
                            "MCS Size         : {} heavy atoms, {} bonds | MCS Tanimoto: {:.3} | 2D Core RMSD: {:.3} A ({} states)",
                            res.mcs_atom_count,
                            res.mcs_bond_count,
                            res.mcs_tanimoto,
                            res.core_rmsd_2d,
                            res.states_visited
                        );
                        println!("MCS SMARTS       : {}", res.mcs_smarts);
                    }
                    None => println!("No connected MCS >= {} atoms found.", cfg.min_core_atoms),
                }
                return;
            }

            let subset: Vec<usize> = (0..ds.mols.len())
                .filter(|&i| {
                    target.eq_ignore_ascii_case("All") || ds.mols[i].target.eq_ignore_ascii_case(&target)
                })
                .collect();

            let t0 = Instant::now();
            let Some(series) = detect_and_align_series_mcs(&ds, &subset, &target, &cfg) else {
                eprintln!(
                    "No multi-molecule MCS >= {} atoms found for scope '{target}'.",
                    cfg.min_core_atoms
                );
                std::process::exit(1);
            };
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!(
                "=== SAR-Atlas v3.8.0 Series McGregor/VF2 MCS & 2D Procrustes R-Group Alignment ({ms:.1} ms) ==="
            );
            println!(
                "Series Scope     : {} ({}/{} matched, {:.0}% coverage) | Mode: {}",
                series.target_scope,
                series.matched_member_count,
                series.total_input_molecules,
                series.coverage_fraction * 100.0,
                cfg.atom_mode.label()
            );
            println!(
                "Consensus Core   : {} atoms, {} bonds | Mean 2D Core RMSD: {:.4} A | Ref: {}",
                series.mcs_atom_count,
                series.mcs_bond_count,
                series.mean_core_rmsd_2d,
                series.reference_name
            );
            println!("MCS SMARTS       : {}", series.mcs_smarts);
            println!("Variable Ports   : {}", series.slot_labels.join(", "));
            for (rk, m) in series.members.iter().enumerate() {
                println!(
                    "  #{:<2} {:<22} (pAct {:.2}) | MCS Sim {:.3} | 2D RMSD {:.4} A | {}",
                    rk + 1,
                    m.compound_name,
                    m.pact,
                    m.mcs_tanimoto_to_ref,
                    m.core_rmsd_2d,
                    m.r_groups.join("  ")
                );
            }

            if let Some(csv_out) = export_csv {
                std::fs::write(&csv_out, export_series_mcs_csv(&series))
                    .expect("Failed to write MCS CSV");
                println!("Exported aligned MCS R-group CSV -> {}", csv_out.display());
            }
            if let Some(sdf_out) = export_aligned_sdf {
                let aligned_mols: Vec<_> =
                    series.members.iter().map(|m| m.aligned_mol.clone()).collect();
                std::fs::write(&sdf_out, format_sdf_dataset(&aligned_mols))
                    .expect("Failed to write aligned SDF");
                println!(
                    "Exported 2D Procrustes-aligned V2000 SDF -> {}",
                    sdf_out.display()
                );
            }
        }
        Commands::BricsRecombine {
            input,
            target,
            min_pact,
            max_sa,
            min_qed,
            three_component,
            cross_target,
            top,
            export_csv,
            export_sdf,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let cfg = BricsRecombineConfig {
                target_filter: target,
                min_lead_pact: min_pact,
                max_sa_score: max_sa,
                min_qed,
                hide_pains: true,
                include_three_component: three_component,
                allow_cross_target: cross_target,
                max_hybrids: 100,
                weight_pact: 1.00,
                weight_qed: 1.00,
                weight_sa: 1.00,
            };
            let t0 = Instant::now();
            let (hybrids, sum) =
                recombine_brics_leads(&ds, ds.default_sar_matrix.as_ref(), &cfg);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!(
                "=== SAR-Atlas v3.8.0 Combinatorial BRICS Fragment Recombiner ({ms:.1} ms) ==="
            );
            println!(
                "Pipeline Summary : {} active leads -> {} unique BRICS half-synthons -> {} valid splices -> {} novel chimeras -> {} retained ({} 3D Pareto)",
                sum.active_leads_used,
                sum.unique_half_synthons,
                sum.raw_valid_chimeras,
                sum.novel_deduplicated,
                sum.retained_after_filters,
                sum.pareto_hybrid_count
            );
            for (rk, h) in hybrids.iter().take(top).enumerate() {
                let p_tag = if h.pareto_front { "[PARETO]" } else { "        " };
                let rxns = h
                    .formed_bonds
                    .iter()
                    .map(|fb| fb.pair_code.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "  #{:<2} {} {:<34} ({:<11} {}c [{}]) | pAct {:.2} ({:+.2}) | QED {:.3} | SA {:.2} | Util {:.2} | {}",
                    rk + 1,
                    p_tag,
                    h.hybrid_name,
                    h.target,
                    h.component_count,
                    rxns,
                    h.predicted_pact,
                    h.delta_pact_vs_scaffold,
                    h.profile.qed_mean,
                    h.sa_profile.sa_score,
                    h.utility_score,
                    h.canonical_smiles
                );
            }
            if let Some(csv_out) = export_csv {
                std::fs::write(&csv_out, export_brics_hybrids_csv(&hybrids))
                    .expect("Failed to write BRICS hybrids CSV");
                println!("Exported BRICS hybrids CSV -> {}", csv_out.display());
            }
            if let Some(sdf_out) = export_sdf {
                let mols: Vec<_> = hybrids.iter().map(|h| h.mol.clone()).collect();
                std::fs::write(&sdf_out, format_sdf_dataset(&mols))
                    .expect("Failed to write BRICS hybrids SDF");
                println!("Exported BRICS hybrids V2000 SDF -> {}", sdf_out.display());
            }
        }
        Commands::Retrosynth {
            smiles,
            id,
            input,
            export_csv,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let target_mol = if let Some(smi) = smiles {
                let mut m = parse_smiles(&smi).unwrap_or_else(|e| {
                    eprintln!("Invalid SMILES '{smi}': {e}");
                    std::process::exit(1);
                });
                m.smiles = molecule_to_smiles(&m);
                m.id = "CLI-TARGET".to_string();
                m.name = "CLI Target Molecule".to_string();
                m
            } else {
                let key = id.as_deref().unwrap_or("CHEMBL553");
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .cloned()
                    .unwrap_or_else(|| ds.mols[0].clone())
            };

            let retro = dissect_brics(&target_mol);
            let sa = &retro.parent_sa;
            println!(
                "=== SAR-Atlas v3.8.0 Ertl SAScore & BRICS Retrosynthetic Disconnection ==="
            );
            println!(
                "Target           : {} ({}) | Formula: {} ({:.2} Da)",
                target_mol.name,
                target_mol.id,
                target_mol.molecular_formula(),
                target_mol.molecular_weight()
            );
            println!(
                "Ertl SAScore     : {:.2} / 10.0  [{}]  (Max Synthon SA: {:.2}, Convergence Drop: -{:.2})",
                sa.sa_score,
                sa.tractability.label(),
                retro.max_synthon_sa,
                retro.convergence_sa_drop
            );
            println!(
                "SA Decomposition : Frag {:.3} | Size -{:.3} | Stereo -{:.3} ({} chiral) | Spiro -{:.3} ({}) | Bridge -{:.3} ({}) | Macro -{:.3} | Sym +{:.3}",
                sa.fragment_score,
                sa.size_penalty,
                sa.stereo_penalty,
                sa.chiral_centers,
                sa.spiro_penalty,
                sa.spiro_atoms,
                sa.bridgehead_penalty,
                sa.bridgehead_atoms,
                sa.macrocycle_penalty,
                sa.symmetry_bonus
            );
            println!(
                "BRICS Cleavages  : {} active disconnections ({} candidate bonds) -> {} building-block synthons:",
                retro.active_cuts.len(),
                retro.candidate_cuts.len(),
                retro.synthons.len()
            );
            for (ci, cut) in retro.active_cuts.iter().enumerate() {
                println!(
                    "  Cut #{:<2} Bond {:>2} ({}{:<2} -- {}{:<2}) [{}] : {}",
                    ci + 1,
                    cut.bond_idx + 1,
                    target_mol.atoms[cut.atom_a].elem.symbol(),
                    cut.atom_a + 1,
                    target_mol.atoms[cut.atom_b].elem.symbol(),
                    cut.atom_b + 1,
                    cut.pair_code,
                    cut.reaction_name
                );
            }
            println!("Retrosynthetic BRICS Synthons:");
            for syn in &retro.synthons {
                println!(
                    "  Syn #{:<2} {:<26} | {:<10} {:>6.1} Da | SA {:.2} | Tags {:<14} | {} ({})",
                    syn.idx + 1,
                    syn.role,
                    syn.formula,
                    syn.mw,
                    syn.sa_score,
                    if syn.attachment_tags.is_empty() {
                        "-".to_string()
                    } else {
                        syn.attachment_tags.join(",")
                    },
                    syn.reagent_smiles,
                    syn.reagent_archetype
                );
            }
            if let Some(csv_out) = export_csv {
                std::fs::write(&csv_out, ds.export_synth_access_csv())
                    .expect("Failed to write SAScore/BRICS CSV");
                println!("Exported dataset SAScore & BRICS table -> {}", csv_out.display());
            }
        }
        Commands::Pharmacophore {
            smiles,
            id,
            input,
            scaffold_hops_only,
            top,
            export_csv,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let query_mol = if let Some(smi) = smiles {
                let mut m = parse_smiles(&smi).unwrap_or_else(|e| {
                    eprintln!("Invalid query SMILES '{smi}': {e}");
                    std::process::exit(1);
                });
                m.smiles = molecule_to_smiles(&m);
                m.id = "CLI-QUERY".to_string();
                m.name = "CLI Query Molecule".to_string();
                m
            } else {
                let key = id.as_deref().unwrap_or("CHEMBL118");
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .cloned()
                    .unwrap_or_else(|| ds.mols[0].clone())
            };

            let (q_map, _, hits) =
                screen_pharmacophore_library(&query_mol, &ds, scaffold_hops_only, true);

            println!(
                "=== SAR-Atlas v3.8.0 2D Topological Pharmacophore & Scaffold-Hopping Screen ==="
            );
            println!(
                "Query            : {} ({}) | Murcko: {}",
                query_mol.name, query_mol.id, q_map.murcko_label
            );
            let feat_summary: Vec<String> = PharmFeatureKind::ALL
                .iter()
                .map(|k| format!("{}={}", k.code(), q_map.counts_by_kind[k.type_index()]))
                .collect();
            println!(
                "Features ({:>2})    : {} | 3-Pt Triangles: {} | Pharm2D Bits: {}/1024",
                q_map.features.len(),
                feat_summary.join(", "),
                q_map.triangles.len(),
                q_map.fp.on_bits
            );
            println!(
                "Matched Hits     : {} {}",
                hits.len(),
                if scaffold_hops_only {
                    "(Scaffold Hops Only)"
                } else {
                    "(All Ranked Library Compounds)"
                }
            );
            for (rk, h) in hits.iter().take(top).enumerate() {
                let m = &ds.mols[h.mol_idx];
                let al = &h.alignment;
                let hop_tag = if al.is_scaffold_hop {
                    "SCAFFOLD HOP"
                } else if al.same_murcko_scaffold {
                    "Same Core"
                } else {
                    "Diff Core"
                };
                let top_tri = al
                    .matched_triangles
                    .first()
                    .map(|t| t.label.as_str())
                    .unwrap_or("-");
                println!(
                    "  #{:<2} {:<20} ({:<8} pAct {:.2}) | Pharm {:.3} (Tri {:.3}, CATS2 {:.3}) | ECFP4 {:.3} | SharedTri {:>3} | {:<12} | {}",
                    rk + 1,
                    m.name,
                    m.target,
                    m.pact,
                    al.pharm_similarity,
                    al.triangle_tanimoto,
                    al.cats_cosine,
                    al.ecfp4_tanimoto,
                    al.shared_triangle_count,
                    hop_tag,
                    top_tri
                );
            }
            if let Some(csv_out) = export_csv {
                std::fs::write(&csv_out, ds.export_pharmacophore_csv(&hits))
                    .expect("Failed to write pharmacophore CSV");
                println!("Exported pharmacophore alignment CSV -> {}", csv_out.display());
            }
        }
        Commands::FetchChembl {
            target_id,
            label,
            min_pchembl,
            limit,
            assay_type,
            save_project,
            export_sdf,
        } => {
            let cfg = ChemblQueryConfig {
                target_chembl_id: target_id,
                target_label: label.clone(),
                min_pchembl,
                max_compounds: limit,
                assay_type,
            };
            let (mols, rep) = match fetch_chembl_target_series(&cfg) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("ChEMBL fetch failed: {e}");
                    std::process::exit(1);
                }
            };
            let ds = DatasetAnalysis::from_molecules(mols, 0.30, 6.0);
            println!(
                "Fetched {} curated molecules from {} ({} raw scanned, {} desalted).",
                rep.deduplicated_molecules, rep.endpoint_url, rep.raw_records_scanned, rep.desalted_count
            );
            for (i, m) in ds.mols.iter().take(15).enumerate() {
                println!(
                    "  {:>2}. {:<14} {:<24} pAct {:.2} | QED {:.2} | SA {:.2} | LogS {:.2}",
                    i + 1,
                    m.id,
                    m.name,
                    m.pact,
                    ds.profiles[i].qed_mean,
                    ds.synth_access[i].sa_score,
                    ds.admet[i].esol_logs
                );
            }
            if let Some(proj_out) = save_project {
                let ws = SarProjectWorkspace::capture(
                    format!("ChEMBL-{label}"),
                    rep.endpoint_url,
                    &ds,
                    "",
                    "All",
                    "Auto Core",
                    "",
                    3,
                    0.15,
                    None,
                    None,
                );
                save_project_auto(&proj_out, &ws, &ds).expect("Failed to save project");
                println!("Saved workspace -> {}", proj_out.display());
            }
            if let Some(sdf_out) = export_sdf {
                std::fs::write(&sdf_out, format_sdf_dataset(&ds.mols)).expect("Failed to write SDF");
                println!("Exported V2000 SDF -> {}", sdf_out.display());
            }
        }
        Commands::FetchPubchem {
            aid,
            compounds,
            label,
            limit,
            save_project,
        } => {
            let (mols, rep) = if let Some(aid_num) = aid {
                fetch_pubchem_bioassay_aid(aid_num, &label, limit).unwrap_or_else(|e| {
                    eprintln!("PubChem AID fetch failed: {e}");
                    std::process::exit(1);
                })
            } else if let Some(cmp_list) = compounds {
                fetch_pubchem_compounds_by_identifiers(&cmp_list, &label, 7.5).unwrap_or_else(|e| {
                    eprintln!("PubChem compound lookup failed: {e}");
                    std::process::exit(1);
                })
            } else {
                eprintln!("Provide either --aid <AID> or --compounds \"name1,name2\"");
                std::process::exit(1);
            };
            let ds = DatasetAnalysis::from_molecules(mols, 0.30, 6.0);
            println!(
                "Imported {} molecules via {} ({})",
                ds.mols.len(),
                rep.source_name,
                rep.endpoint_url
            );
            if let Some(proj_out) = save_project {
                let ws = SarProjectWorkspace::capture(
                    format!("PubChem-{label}"),
                    rep.endpoint_url,
                    &ds,
                    "",
                    "All",
                    "Auto Core",
                    "",
                    3,
                    0.15,
                    None,
                    None,
                );
                save_project_auto(&proj_out, &ws, &ds).expect("Failed to save project");
                println!("Saved workspace -> {}", proj_out.display());
            }
        }
        Commands::SaveProject {
            input,
            output,
            name,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let fw_sm = ds
                .default_sar_matrix
                .as_ref()
                .map(|m| m.core_smarts.as_str())
                .unwrap_or("");
            let fw_nm = ds
                .default_sar_matrix
                .as_ref()
                .map(|m| m.core_name.as_str())
                .unwrap_or("Auto Core");
            let ws = SarProjectWorkspace::capture(
                &name,
                "Saved from SAR-Atlas CLI",
                &ds,
                "",
                "All",
                fw_nm,
                fw_sm,
                3,
                0.15,
                ds.mols.first(),
                None,
            );
            save_project_auto(&output, &ws, &ds).expect("Failed to save project workspace");
            println!(
                "Saved project '{}' ({} compounds, {} SALI edges) -> {}",
                name,
                ds.mols.len(),
                ds.edges.len(),
                output.display()
            );
        }
        Commands::InspectProject { project } => {
            let sess = load_project_auto(&project).unwrap_or_else(|e| {
                eprintln!("Failed to load project {}: {e}", project.display());
                std::process::exit(1);
            });
            println!(
                "=== SAR-Atlas v3.8.0 Project Workspace: {} ===",
                sess.project_name
            );
            println!("Notes          : {}", sess.notes);
            println!(
                "Compounds      : {} ({} Pareto leads, {} SALI cliff edges)",
                sess.ds.mols.len(),
                sess.ds.pareto_leads.len(),
                sess.ds.edges.iter().filter(|e| e.is_cliff).count()
            );
            println!(
                "Free-Wilson    : {} ({})",
                sess.fw_core_name, sess.fw_core_smarts
            );
            for &pi in sess.ds.pareto_leads.iter().take(10) {
                let m = &sess.ds.mols[pi];
                let p = &sess.ds.profiles[pi];
                let ad = &sess.ds.admet[pi];
                let sa = &sess.ds.synth_access[pi];
                println!(
                    "  [Pareto] {:<22} ({:<12}) pAct {:.2} | QED {:.3} | SA {:.2} | LogS {:.2} | {}",
                    m.name,
                    m.target,
                    m.pact,
                    p.qed_mean,
                    sa.sa_score,
                    ad.esol_logs,
                    ad.boiled_egg.label()
                );
            }
        }
        Commands::SketchEval { smiles, input } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let mut sketcher = MolecularSketcher::default();
            if let Err(e) = sketcher.load_from_smiles(&smiles) {
                eprintln!("Invalid SMILES '{smiles}': {e}");
                std::process::exit(1);
            }
            let rep = sketcher.evaluate_live_design(&ds, ds.default_sar_matrix.as_ref());
            let adm = evaluate_admet(&sketcher.mol, &rep.profile);
            let sa = evaluate_synthetic_accessibility(&sketcher.mol);
            let pk = evaluate_physiological_pka_74(&sketcher.mol);
            println!(
                "=== SAR-Atlas v3.8.0 Live Analog Design, ADMET, Micro-pKa & Synthetic Accessibility Report ==="
            );
            println!("Canonical SMILES : {}", rep.canonical_smiles);
            println!(
                "Potency Estimate : pAct {:.2} (IC50 {:.3} nM) via {}",
                rep.effective_pact, rep.effective_ic50_nm, rep.pact_source
            );
            println!(
                "Desirability     : QED {:.3} | CNS-MPO {:.2}/6.0 | LLE {:.2} | SAScore {:.2} ({}, {} BRICS cuts) | LogS {:.2} ({})",
                rep.profile.qed_mean,
                rep.profile.cns_mpo,
                rep.effective_lle,
                sa.sa_score,
                sa.tractability.short_label(),
                sa.brics_cleavable_bonds,
                adm.esol_logs,
                adm.boiled_egg.label()
            );
            println!(
                "pH 7.4 State     : {} (<Q> {:+.2}, f_neutral {:.1}%) | logD2.0 {:.2} -> logD7.4 {:.2} | {} protomers, {} tautomers",
                pk.charge_class.label(),
                pk.mean_net_charge_74,
                pk.neutral_fraction_74 * 100.0,
                pk.gastric_logd_20,
                pk.microstate_logd_74,
                pk.major_protomers_74,
                pk.tautomer_count
            );
        }
        Commands::Admet { input } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            for (i, m) in ds.mols.iter().enumerate() {
                let ad = &ds.admet[i];
                let sa = &ds.synth_access[i];
                println!(
                    "{:<22} {:<12} SA {:>4.2} | LogS {:>6.2} | {:<18} | 3/75 {} | hERG {}",
                    m.name,
                    m.target,
                    sa.sa_score,
                    ad.esol_logs,
                    ad.boiled_egg.label(),
                    if ad.pfizer_3_75_pass { "PASS" } else { "RISK" },
                    ad.herg_risk.label()
                );
            }
        }
        Commands::Bioisosteres { smiles, id, input } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let parent = if let Some(smi) = smiles {
                let mut m = parse_smiles(&smi).expect("Invalid SMILES");
                m.name = "CLI-Lead".to_string();
                m
            } else {
                let key = id.as_deref().unwrap_or("NSC693255");
                ds.mols
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(key) || m.name.eq_ignore_ascii_case(key))
                    .cloned()
                    .unwrap_or_else(|| ds.mols[0].clone())
            };
            for (rk, ba) in generate_bioisosteres(&parent, &ds, ds.default_sar_matrix.as_ref())
                .iter()
                .enumerate()
            {
                println!(
                    "  #{:<2} {:<38} | Pred pAct {:.2} ({:+.2}) | LogS {:.2} | {}",
                    rk + 1,
                    ba.rule_name,
                    ba.predicted_pact,
                    ba.delta_pact,
                    ba.admet.esol_logs,
                    ba.canonical_smiles
                );
            }
        }
        Commands::Smarts {
            pattern,
            input,
            export_sdf,
        } => {
            let query = compile_smarts(&pattern).expect("Invalid SMARTS");
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let results = ds.evaluate_smarts(&query);
            let hits: Vec<usize> = (0..ds.mols.len()).filter(|&i| results[i].matched).collect();
            println!(
                "SMARTS '{}' matched {}/{} compounds.",
                query.pattern,
                hits.len(),
                ds.mols.len()
            );
            if let Some(sdf_out) = export_sdf {
                let matched: Vec<_> = hits.iter().map(|&i| ds.mols[i].clone()).collect();
                std::fs::write(&sdf_out, format_sdf_dataset(&matched)).expect("Failed to write SDF");
            }
        }
        Commands::Rgroup {
            core,
            target,
            slots,
            lambda,
            input,
            export_csv,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let mat = if let Some(core_pat) = core {
                ds.solve_sar_matrix("CLI Core SMARTS", &core_pat, target.as_deref(), slots, lambda)
                    .expect("Free-Wilson error")
            } else {
                build_auto_free_wilson_matrix(&ds.mols, &ds.profiles, slots, lambda)
                    .expect("Could not auto-detect multi-compound core")
            };
            println!(
                "Free-Wilson {}: n={}, b0={:.3}, R^2={:.3}, Q^2={:.3}",
                mat.core_name,
                mat.rows.len(),
                mat.intercept_pact,
                mat.r2,
                mat.q2_loo
            );
            if let Some(csv_path) = export_csv {
                std::fs::write(&csv_path, mat.export_csv()).expect("Failed to write CSV");
            }
        }
        Commands::Analyze {
            input,
            output,
            format,
            hide_pains,
        } => {
            let mut ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            if hide_pains {
                let keep: Vec<_> = (0..ds.mols.len())
                    .filter(|&i| ds.alerts[i].is_empty())
                    .map(|i| ds.mols[i].clone())
                    .collect();
                ds = DatasetAnalysis::from_molecules(keep, 0.30, 6.0);
            }
            let payload = match format {
                OutputFormat::Csv => ds.export_summary_csv(),
                OutputFormat::Json => {
                    serde_json::to_string_pretty(&ds.profiles).unwrap_or_default()
                }
            };
            if let Some(out_path) = output {
                std::fs::write(&out_path, payload).expect("Failed to write analysis output");
            } else {
                print!("{payload}");
            }
        }
        Commands::Cliffs {
            input,
            min_tanimoto,
            min_sali,
            top,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), min_tanimoto, min_sali);
            for e in ds.edges.iter().filter(|e| e.same_target).take(top) {
                println!(
                    "{:<24} <-> {:<24} | Sim {:.3} | d_pAct {:.2} | SALI {:.2}",
                    ds.mols[e.mol_a].name,
                    ds.mols[e.mol_b].name,
                    e.tanimoto,
                    e.delta_pact,
                    e.sali
                );
            }
        }
        Commands::Mmp { input, top } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            for m in ds.mmps.iter().take(top) {
                println!(
                    "{} -> {} [{}]: d_pAct {:+.2}, d_LogP {:+.2}",
                    ds.mols[m.mol_from].name,
                    ds.mols[m.mol_to].name,
                    m.transform,
                    m.delta_pact,
                    m.delta_clogp
                );
            }
        }
        Commands::ExportSvg {
            smiles,
            id,
            smarts,
            output,
            light,
        } => {
            let mol = if let Some(smi) = smiles {
                parse_smiles(&smi).expect("Invalid SMILES")
            } else {
                let bench = load_benchmark_26();
                let target_id = id.as_deref().unwrap_or("BRIDGES-56");
                bench
                    .into_iter()
                    .find(|m| {
                        m.id.eq_ignore_ascii_case(target_id)
                            || m.name.eq_ignore_ascii_case(target_id)
                    })
                    .unwrap_or_else(|| {
                        parse_smiles("CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC").unwrap()
                    })
            };
            let alerts = sar_core::scan_structural_alerts(&mol);
            let alert_hl: Vec<usize> = alerts.iter().flat_map(|a| a.atoms.iter().copied()).collect();
            let (sm_atoms, sm_bonds) = if let Some(pat) = smarts {
                let q = compile_smarts(&pat).expect("Invalid SMARTS");
                let r = match_smarts(&mol, &q);
                (r.union_atoms, r.union_bonds)
            } else {
                (Vec::new(), Vec::new())
            };
            let svg = render_molecule_svg_with_smarts(
                &mol, 540, 380, !light, &alert_hl, &sm_atoms, &sm_bonds,
            );
            std::fs::write(&output, svg).expect("Failed to write SVG");
            println!(
                "Exported SVG ({}) -> {}",
                molecule_to_smiles(&mol),
                output.display()
            );
        }
        Commands::ExportSdf {
            input,
            output,
            pareto_only,
        } => {
            let ds = load_dataset_or_project(input.as_ref(), 0.30, 6.0);
            let selected: Vec<_> = (0..ds.mols.len())
                .filter(|&i| !pareto_only || ds.profiles[i].pareto_optimal)
                .map(|i| ds.mols[i].clone())
                .collect();
            std::fs::write(&output, format_sdf_dataset(&selected)).expect("Failed to write SDF");
        }
        Commands::Bench { compounds } => {
            let n = compounds.max(100);
            println!(
                "Running SAR-Atlas v3.8.0 multi-threaded Rayon benchmark ({n} compounds)..."
            );
            let t0 = Instant::now();
            let mols = synthesize_hts_library(n);
            let dt_gen = t0.elapsed();
            let t1 = Instant::now();
            let ds = DatasetAnalysis::from_molecules(mols, 0.55, 8.0);
            let dt_sar = t1.elapsed();
            let vis = vec![true; ds.mols.len()];
            let t2 = Instant::now();
            let (gpu_nodes, gpu_edges) = pack_gpu_scene(
                &ds.mols,
                &ds.chem_coords,
                &ds.profiles,
                &ds.alerts,
                &ds.edges,
                &vis,
                None,
                Some(0),
                NodeColorMode::PotencyViridis,
                false,
            );
            let _ = build_chem_heatmap_rgba(
                &ds.mols,
                &ds.chem_coords,
                &ds.profiles,
                &ds.edges,
                &vis,
                HeatmapLayer::PotencySurface,
            );
            let dt_gpu = t2.elapsed();
            println!(
                "  1. HTS synthesis: {:.1} ms | 2. ECFP4 + Pharm2D + pKa7.4 + 3D-ROCS + CovWarheads + SMARTCyp + BRICS + ADMET: {:.1} ms | 3. GPU pack ({}n/{}e): {:.1} ms",
                dt_gen.as_secs_f64() * 1000.0,
                dt_sar.as_secs_f64() * 1000.0,
                gpu_nodes.len(),
                gpu_edges.len(),
                dt_gpu.as_secs_f64() * 1000.0
            );
        }
    }
}
