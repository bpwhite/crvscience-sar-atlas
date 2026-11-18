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

//! `sar-core`: Pure-Rust small-molecule cheminformatics, Structure-Activity Relationship (`SAR`),
//! activity-cliff (`SALI`), Matched Molecular Pair (`MMP`), Core-Aligned `Free-Wilson` R-group
//! matrices, Daylight `SMARTS` subgraph search, interactive 2D `MolecularSketcher` with canonical
//! `OpenSMILES`, quantitative `ADMET` (`ESOL` / `BOILED-Egg`), `Meanwell` Bioisostere enumeration,
//! live `ChEMBL` / `PubChem` REST bioactivity ingestion, `.saratlas.json` / `SQLite` workspace
//! persistence, 2D topological `Pharmacophore` alignment, Ertl `SAScore` & `BRICS` retrosynthesis,
//! combinatorial `BRICS` virtual hybrids, McGregor/`VF2` `MCS` 2D series alignment, and
//! physiological `pH 7.4` micro-`pKa`/protomers, Distance-Geometry 3D conformers + `MMFF-lite` + `USR`/`ROCS` 3D overlay, covalent warhead `GSH` tuning, `PROTAC` `bRo5` chameleonicity, `SMARTCyp` Phase I/II metabolism, `RCM`/lactam macrocyclization with 3D preorganization entropy, `Rule-of-3` fragment exit-vector growing with `GE`/`LLE`/`LE` optimization, bioreversible prodrug promoiety enumeration (`ester`, `phosphate`, `carbamate`, `oxazolidinone`, `PEPT1`), and physiologically based oral absorption (`CAT`/`ACAT`), hepatic first-pass (`Fg * Fh`), `Rodgers-Rowland` `Vss`, and unbound brain `Kp,uu,brain` `RK4` pharmacokinetic simulation and cross-module `Derringer-Suich` lead scorecards + standalone `.html`/`.svg` dossiers and evolutionary graph-mutation / ring-hop / `BRICS` crossover de novo molecule generation and structure-based `.pdb` binding-site cavity, `WaterMap` hydration & receptor pharmacophore complementarity scoring and DNA-encoded (`DEL`) / parallel 2- and 3-step `Amide`, `Suzuki`, `Buchwald`, & `CuAAC Click` combinatorial library enumeration with `MaxMin` diversity and 3D `PDB Pocket` / `ROCS` pre-filtering and dual-pocket target vs off-target 3D cavity, differential `WaterMap`, and residue-mutation `Delta-Delta G` selectivity profiling and `GNM`/`ANM` normal-mode cryptic pocket detection & allosteric-to-orthosteric mechanical strain coupling and `ECFP4`/`3D-USRCAT` Gaussian-process + `24`-tree ensemble active-learning QSAR with Expected-Improvement & Kriging-Believer batch acquisition and `Lomap` `MCS` perturbation-graph planning, multi-`lambda` `Zwanzig`/`Bennett BAR` congeneric `RBFE` estimation, and `MLE` cycle-closure hysteresis reconciliation and hierarchical `CSD`/`SMARTS` dihedral energy-profile scanning, `Shoichet` `TEU` bioactive 3D conformer strain estimation, and rotatable-bond locking and configurable 6-stage molecule discovery funnel orchestration (`DeNovo`/`DEL`/`Rotor-Lock` -> `PAINS`/`ADMET`/`SAScore` -> `3D Pocket` & `Shoichet TEU` -> `Selectivity` -> `Active-Learning QSAR` -> `Lomap RBFE` & workspace promotion, and sub-pocket fragment-pair 3D exit-vector alignment, `18`-scaffold linker geometry matching, and `Jencks`/`Murray-Verdonk` super-additivity `Delta-G` scoring, and flexible binding-pocket `Dunbrack` side-chain rotamer repacking & `Fischer-Shoichet` Boltzmann receptor-ensemble docking, and active-site `His`/`Asp`/`Glu` coupled tautomer/protonation, `Asn`/`Gln`/`His` `180 deg` flip, and tetrahedral `H-O-H` bridge-water hydrogen-bond network optimization, and `IFST`/`WaterMap`/`GIST` `dH`/`-TdS`/`dG` active-site hydration hotspot mapping & displaceable-water R-group vector design, and `Foldit`/`Drugit`-style interactive 3D tactile ligand/side-chain force-field sculpting, rubber-band H-bond restraints, & timed med-chem challenge puzzles, and live public `RCSB PDB` & `AlphaFold DB` coordinate, ligand, per-residue `pLDDT`/`PAE` confidence & `Karelina-Dror` pocket readiness analysis, and 3D `Bauer-Mackey` molecular electrostatic potential (`ESP`) complementarity, `XED` off-center `sigma-hole` halogen/chalcogen bonds, & `Wheeler-Houk` `pi`-multipole recognition, `v3.8.0`).

pub mod active_learning;
pub mod admet;
pub mod alerts;
pub mod allosteric_pocket;
pub mod bioisostere;
pub mod brics_reassemble;
pub mod campaign_report;
pub mod chembl_ingest;
pub mod conformer3d;
pub mod covalent_warhead;
pub mod cyp_metabolism;
pub mod de_novo_gen;
pub mod discovery_pipeline;
pub mod electrostatics_esp;
pub mod fingerprint;
pub mod fixtures;
pub mod fragment_grower;
pub mod foldit_sculpt;
pub mod fragment_linker;
pub mod gpu_pack;
pub mod hbond_network;
pub mod induced_fit;
pub mod layout2d;
pub mod macrocycle;
pub mod mcs;
pub mod mmp;
pub mod mol;
pub mod mol_io;
pub mod parallel_lib;
pub mod pbpk_oral;
pub mod pharmacophore;
pub mod physchem;
pub mod pka_protomer;
pub mod pocket_cavity;
pub mod prodrug;
pub mod project_store;
pub mod protac_linker;
pub mod public_pdb_fetch;
pub mod rbfe_network;
pub mod rgroup_table;
pub mod scaffold;
pub mod selectivity_screen;
pub mod sketcher;
pub mod smarts;
pub mod synth_access;
pub mod torsion_strain;
pub mod watermap_solvation;

pub use active_learning::{
    cholesky_decompose, compute_expected_improvement, export_active_learning_csv,
    run_active_learning_campaign, AcquisitionStrategy, ActiveLearningCandidate,
    ActiveLearningConfig, ActiveLearningReport, ApplicabilityDomainTier, CandidatePoolSource,
    QsarKernelMode, QsarValidationMetrics, SimulatedAlRound, SurrogateModelKind, RF_FEATURE_DIM,
};
pub use admet::{
    evaluate_admet, in_boiled_egg_white, in_boiled_egg_yolk, AdmetProfile, BoiledEggRegion,
    HergRisk, SolubilityClass,
};
pub use alerts::{scan_structural_alerts, AlertSeverity, StructuralAlertMatch};
pub use allosteric_pocket::{
    analyze_allosteric_network, curated_allosteric_pdb_text, deform_pdb_along_anm_mode,
    export_allosteric_screen_csv, format_allosteric_poses_sdf, jacobi_eigen_symmetric,
    load_curated_allosteric_system, score_ligand_allosteric_coupling,
    screen_dataset_allosteric_pocket, AllostericLigandResult, AllostericModulationMode,
    AllostericNetworkReport, AnmNormalMode, CrypticPocketCandidate, CuratedAllostericPreset,
    ElasticResidueNode, CURATED_ALLOSTERIC_SYSTEMS,
};
pub use bioisostere::{generate_bioisosteres, BioisostereAnalog};
pub use brics_reassemble::{
    export_brics_hybrids_csv, recombine_brics_leads, rescore_and_sort_hybrids,
    BricsRecombineConfig, BricsRecombineSummary, BricsVirtualHybrid, HybridCouplingBond,
};
pub use campaign_report::{
    build_lead_campaign_dossier, evaluate_campaign_scorecard, evaluate_molecule_scorecard,
    export_campaign_dossier_html, export_campaign_scorecard_csv, export_lead_dossier_html,
    export_lead_dossier_svg, render_dossier_radar_svg, CampaignScorecardSummary,
    CampaignTargetProfile, CriterionVerdict, DossierNeighborNote, LeadCampaignDossier,
    LeadScorecardRow, MedChemOptimizationAction, ProgressionTier, StageGateCriterion,
    DOSSIER_PILLAR_LABELS,
};
pub use chembl_ingest::{
    desalt_largest_organic_smiles, extract_active_cids_from_pubchem_concise_csv,
    fetch_chembl_target_series, fetch_pubchem_bioassay_aid, fetch_pubchem_compounds_by_identifiers,
    parse_chembl_activities_json, parse_pubchem_property_table_into, ChemblQueryConfig,
    ChemblTargetPreset, IngestError, IngestReport, PubChemAssayPreset, CURATED_CHEMBL_TARGETS,
    CURATED_PUBCHEM_ASSAYS,
};
pub use conformer3d::{
    align_rocs_3d, align_rocs_ensemble_3d, bondi_vdw_radius, compute_dipole_debye,
    compute_gasteiger_charges, compute_usr_descriptors, evaluate_3d_summary_profile,
    export_3d_overlay_hits_csv, format_3d_conformer_sdf, format_3d_overlay_sdf,
    generate_3d_conformer_ensemble, kabsch_align_3d, screen_3d_overlay_library, usr_similarity,
    usrcat_similarity, Conformer3d, ConformerEnsemble, ConformerGenConfig, MmffEnergyBreakdown,
    Overlay3dHit, Rocs3dAlignment, RocsColorAtom, RocsColorKind, Summary3dProfile, USRCAT_DIM,
    USR_DIM,
};
pub use covalent_warhead::{
    analyze_covalent_warheads_and_swaps, detect_covalent_warheads, evaluate_covalent_summary,
    export_warhead_swaps_csv, CovalentDatasetSummary, CovalentWarheadReport, DetectedWarhead,
    ReactivityWindow, TargetResidue, WarheadReversibility, WarheadSwapCandidate, KINETIC_GRID_LEN,
};
pub use cyp_metabolism::{
    analyze_cyp_metabolism, evaluate_metabolic_summary, export_metabolites_csv,
    predict_smartcyp_sites, CypIsoform, CypMetabolismReport, MetabolicDatasetSummary,
    MetabolicStabilityClass, MetabolismPhase, MetaboliteSpecies, SmartCypSite, SomReactionKind,
};
pub use de_novo_gen::{
    evolve_de_novo_molecules, export_de_novo_candidates_csv, format_de_novo_3d_sdf,
    DeNovoConfig, DeNovoEvolutionReport, DeNovoGoalPreset, EvolutionOperatorKind,
    EvolvedCandidate, GenerationSnapshot,
};
pub use discovery_pipeline::{
    append_pipeline_run_sqlite, export_funnel_candidates_csv, export_funnel_stages_csv,
    format_promoted_survivors_sdf, load_pipeline_ledger_sqlite, load_pipeline_recipe_json,
    run_discovery_pipeline, run_discovery_pipeline_with_progress, save_pipeline_recipe_json,
    DiscoveryFunnelPreset, DiscoveryPipelineConfig, DiscoveryPipelineReport, DmtaRoundSummary,
    FunnelCandidateRecord, FunnelSourceStream, FunnelStageSummary, PipelineProgressUpdate,
    PipelineRunLedgerEntry,
};
pub use fingerprint::{
    compute_ecfp, compute_ecfp4_batch, compute_sali_landscape, embed_chemical_space_2d, Ecfp2048,
    SaliCliffEdge,
};
pub use fixtures::{load_benchmark_26, synthesize_hts_library, BENCHMARK_26_SPECS};
pub use fragment_grower::{
    build_curated_ro3_seed_molecule, evaluate_rule_of_three_audit, export_grown_fragments_csv,
    extract_ro3_seed_core, format_grown_fragments_sdf, grow_fragment_along_vector,
    perceive_fragment_exit_vectors, summarize_dataset_fragment_efficiency, CuratedRo3SeedSpec,
    EfficiencyMetrics, ExitVectorChemistry, FragmentDatasetRow, FragmentExitVector,
    FragmentGrowthReport, GrownFragmentCandidate, GrowthEfficiencyRegime, GrowthPocketRole,
    RuleOfThreeAudit, CURATED_RO3_SEEDS, RT_LN10_KCAL,
};
pub use fragment_linker::{
    design_linked_fragments, export_linked_fragments_csv, format_linked_fragments_sdf,
    perceive_fragment_exit_vectors_3d, rank_exit_vector_pairs_3d, ExitVector3d,
    ExitVectorPairGeometry, FragmentJoinRegime, FragmentLinkerConfig, FragmentLinkingReport,
    FragmentPairPreset, LinkedFragmentCandidate, LinkerScaffoldClass, LinkerTemplate,
    SuperAdditivityTier, DELINKER_POSE_PRESERVED_RMSD_A, LINKER_TEMPLATES,
    MURRAY_VERDONK_RIGID_ENTROPY_KCAL,
};
pub use gpu_pack::{
    build_chem_heatmap_rgba, inferno_cliff_rgb, pack_gpu_scene, viridis_rgb, GpuCameraUniform,
    GpuEdgeInstance, GpuNodeInstance, HeatmapLayer, NodeColorMode, HEATMAP_DIM,
};
pub use electrostatics_esp::{
    analyze_electrostatics_esp, analyze_esp_with_prebuilt, export_dataset_esp_screen_csv,
    export_esp_contacts_csv, export_esp_designs_csv, format_esp_multipole_pdb, load_esp_target_pdb,
    screen_dataset_electrostatics_esp, ConnollyEspVertex, DatasetEspScreenRow,
    ElectrostaticsEspConfig, ElectrostaticsEspReport, EspMultipoleDesign, EspTargetPreset,
    MultipoleContact, MultipoleContactKind, MultipolePseudoatom, MultipoleSiteKind,
    BAUER_MACKEY_ESP_CAP_KCAL_E, COULOMB_KCAL_A_PER_E2,
};
pub use public_pdb_fetch::{
    export_public_structure_confidence_csv, fetch_and_assess_public_structure,
    filter_pdb_removing_disordered_loops, ExtractedPdbLigand, PocketReadinessVerdict,
    PublicPdbFetchConfig, PublicStructureReadinessReport, PublicStructureSource,
    PublicTargetPreset, ResidueConfidenceBand, ResidueConfidenceRow,
    DEFAULT_DISORDERED_PLDDT_CUTOFF,
};
pub use foldit_sculpt::{
    add_rubber_band_to_receptor, auto_band_polar_contacts, curated_vhl_5nvx_pdb_text,
    drag_tug_atom_step, export_sculpt_trajectory_csv, format_sculpted_complex_pdb,
    format_sculpted_ligand_sdf, new_sculpt_session, push_undo_checkpoint,
    recompute_sculpt_score, redo_sculpt_move, restore_best_sculpt_pose, run_blue_fuse_macro,
    shake_pocket_rotamers, sprout_in_pocket_rgroup, undo_sculpt_move, wiggle_minimize,
    BandAnchorTarget, RubberBandRestraint, SculptFlexibleSidechain, SculptMedalTier,
    SculptMouseTool, SculptPuzzlePreset, SculptReceptorAtom, SculptScoreBreakdown,
    SculptSession, SculptSnapshot, SculptSproutWarhead, SculptTrajectoryStep, SculptWaterTarget,
    FOLDIT_BASE_SCORE_PTS, FOLDIT_PTS_PER_KCAL, SCULPT_SPROUT_WARHEADS,
};
pub use watermap_solvation::{
    analyze_watermap_solvation, analyze_watermap_with_prebuilt, curated_factor_xa_1ezq_pdb_text,
    export_dataset_watermap_screen_csv, export_watermap_rgroup_designs_csv,
    export_watermap_sites_csv, format_watermap_annotated_pdb, load_watermap_pdb,
    screen_dataset_watermap_solvation, DatasetWaterMapScreenRow, DisplaceableWaterRGroupDesign,
    GistHotspotVoxel, HydrationSiteClass, RGroupSolvationStrategy, SiteLigandStatus,
    WaterMapConfig, WaterMapHydrationSite, WaterMapPreset, WaterMapSolvationReport,
    WATERMAP_BULK_ENERGY_KCAL, WATERMAP_BULK_HBONDS, WATERMAP_BULK_NEIGHBORS,
    WATERMAP_DISPLACEMENT_CUTOFF_A,
};
pub use hbond_network::{
    export_dataset_hbond_screen_csv, export_hbond_network_edges_csv,
    export_hbond_network_nodes_csv, format_optimized_hbond_complex_pdb, load_hbond_network_pdb,
    optimize_hbond_network_with_prebuilt, optimize_pocket_hbond_network,
    screen_dataset_hbond_network, DatasetHBondScreenRow, HBondEdgeKind, HBondNetworkConfig,
    HBondNetworkEdge, HBondNetworkPreset, HBondNetworkReport, HbnNodeKind, HbnNodeMicrostate,
    HbnVariableNode, PolarVectorSite, ProtonLinkedPhPoint, HBN_PH_SCAN_POINTS,
    POLAR_H_BOND_LEN_A,
};
pub use induced_fit::{
    build_receptor_ensemble, export_induced_fit_ensemble_csv, export_repacked_rotamers_csv,
    format_induced_fit_ensemble_sdf, load_induced_fit_base_pdb, perceive_flexible_pocket_residues,
    rotate_point_around_axis_3d, score_ligand_induced_fit_ensemble,
    score_ligand_with_prebuilt_ensemble, screen_dataset_induced_fit_ensemble,
    BindingMechanismRegime, DatasetInducedFitRow, DunbrackRotamerId, EnsembleStateDockingScore,
    FlexiblePocketResidue, InducedFitConfig, InducedFitEnsembleReport, InducedFitPreset,
    ReceptorEnsembleMember, SideChainRepackMove, IFD_RT_298K_KCAL, SHERMAN_IFD_REORG_SCALE,
};
pub use layout2d::{
    compute_2d_coords, render_molecule_svg, render_molecule_svg_with_smarts, BOND_LEN_2D,
};
pub use macrocycle::{
    compute_kier_flexibility_phi, design_macrocycles, detect_cyclizable_terminal_pairs,
    export_macrocycles_csv, format_macrocycles_3d_sdf, CyclizableTerminalPair, MacroClosureReaction,
    MacroDruggabilityClass, MacrocycleCandidate, MacrocycleDesignReport, PreorganizationMetrics,
};
pub use mcs::{
    detect_and_align_series_mcs, export_series_mcs_csv, find_pairwise_mcs, solve_2d_procrustes,
    AlignedSeriesMember, McsAtomMatchMode, McsConfig, PairwiseMcsResult, SeriesMcsAlignment,
};
pub use mmp::{discover_mmps, MatchedMolecularPair, MmpKind};
pub use mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
pub use mol_io::{
    format_sdf_dataset, format_sdf_record, parse_delimited_dataset, parse_sdf_dataset,
    parse_sdf_record, parse_smiles, MolIoError,
};
pub use parallel_lib::{
    apply_parallel_reaction_step, curated_amine_plate, curated_azide_plate,
    curated_boronic_acid_plate, default_plate_for_reaction, enumerate_parallel_library,
    export_parallel_library_csv, format_parallel_library_sdf, select_maxmin_diverse_subset,
    BuildingBlockClass, BuildingBlockReagent, CuratedLibraryScheme, EnumeratedLibraryMember,
    LibraryPreFilterMode, LibrarySynthesisMode, ParallelLibConfig, ParallelLibraryReport,
    ParallelReactionKind, CURATED_LIBRARY_SCHEMES,
};
pub use pharmacophore::{
    align_pharmacophores, compute_topological_distances, extract_pharmacophore_map,
    gobbi_bin_label, gobbi_distance_bin, screen_pharmacophore_library, MatchedPharmTriangle,
    Pharm2DFingerprint, PharmFeatureKind, PharmTriangle, PharmacophoreAlignment, PharmacophoreHit,
    PharmacophoreMap, PharmacophorePoint, CATS2_DIM, CATS2_MAX_LAG, PHARM_FP_WORDS,
};
pub use physchem::{
    compute_bickerton_qed, compute_wager_cns_mpo, evaluate_physchem, mark_pareto_front,
    PhyschemProfile, QED_WEIGHT_MAX, QED_WEIGHT_MEAN,
};
pub use pka_protomer::{
    analyze_pka_and_protomers, enumerate_tautomers, evaluate_physiological_pka_74,
    perceive_ionizable_sites, standardize_neutral_parent, ChargeClass74, IonSiteKind,
    IonizableSite, PhysiologicalChargeSummary, PkaProtomerReport, ProtomerSpecies,
    TautomerSpecies, PH_GRID_LEN,
};
pub use pocket_cavity::{
    analyze_binding_pocket, curated_pocket_pdb_text, export_pocket_complementarity_csv,
    fetch_rcsb_pdb_text, format_pocket_poses_sdf, parse_pdb_structure, score_ligand_in_pocket,
    screen_dataset_against_pocket, CuratedPocketPreset, HydrationSite, ParsedPdbStructure,
    PdbAtom, PdbHetLigand, PdbWaterSite, PocketCavityAnalysis, PocketComplementarityResult,
    PocketDruggabilityTier, PocketLigandInteraction, PocketResidueSummary,
    ReceptorPharmacophorePoint, WaterThermodynamicRole, CURATED_PDB_POCKETS,
};
pub use pbpk_oral::{
    export_dataset_pbpk_csv, export_pbpk_timecourse_csv, simulate_oral_pbpk,
    summarize_dataset_oral_pbpk, CnsDeliveryClass, DosingRegimen, OralPbpkReport,
    OrganPartitionEntry, PbpkDatasetRow, PbpkDosingConfig, PrandialState, HUMAN_BW_KG,
    PBPK_TIME_POINTS,
};
pub use project_store::{
    load_project_auto, load_project_from_sqlite_conn, load_project_sqlite, save_project_auto,
    save_project_json, save_project_sqlite, save_project_to_sqlite_conn, LoadedProjectSession,
    ProjectStoreError, SarProjectWorkspace, StoredCompoundEntry, PROJECT_SCHEMA_VERSION,
};
pub use prodrug::{
    build_curated_prodrug_parent, design_prodrugs, diagnose_parent_bcs_liability,
    export_prodrugs_csv, format_prodrugs_sdf, perceive_maskable_handles,
    summarize_dataset_prodrug_opportunities, BcsClass, ByproductSafetyTier,
    CuratedProdrugParentSpec, EnzymaticActivationProfile, MaskableHandle, ParentLiabilityDiagnosis,
    ProdrugActivatingEnzyme, ProdrugCandidate, ProdrugDatasetRow, ProdrugDesignReport,
    ProdrugHandleKind, PromoietyFamily, CURATED_PRODRUG_PARENTS, PRODRUG_TIME_GRID_LEN,
};
pub use protac_linker::{
    design_protac_series, detect_poi_exit_vectors, export_protac_candidates_csv,
    solve_three_body_hook_curve, Bro5PermeabilityClass, ChameleonicityProfile, E3RecruiterKind,
    LinkerClass, PoiExitVector, ProtacCandidate, ProtacDesignReport, TernaryHookProfile,
    HOOK_CURVE_LEN,
};
pub use rbfe_network::{
    compute_lomap_pair_score, export_rbfe_edges_csv, export_rbfe_nodes_csv,
    molecule_formal_charge, run_rbfe_network_campaign, solve_bennett_bar, zwanzig_fep_kcal,
    LomapEdgeScore, RbfeConvergenceTier, RbfeLambdaWindow, RbfeLigandNode, RbfeNetworkConfig,
    RbfeNetworkReport, RbfePerturbationEdge, RbfeSeriesPreset, RbfeSummaryMetrics,
    RbfeThermodynamicCycle, RbfeTopologyMode, RBFE_RT_298K_KCAL,
};
pub use rgroup_table::{
    build_auto_free_wilson_matrix, build_free_wilson_for_smarts, predict_best_free_wilson,
    predict_free_wilson_for_molecule, scaffold_subgraph_to_smarts, CoreTemplate, FreeWilsonMatrix,
    FreeWilsonPrediction, FreeWilsonSlotContribution, RGroupRow, SarPivotCell, SubstituentEffect,
    VirtualAnalog, CURATED_CORE_TEMPLATES,
};
pub use scaffold::{
    extract_murcko_scaffolds, murcko_framework_mask, murcko_signatures, MurckoScaffold,
};
pub use selectivity_screen::{
    analyze_dual_pocket_selectivity, curated_offtarget_pdb_text, export_mutation_scan_csv,
    export_selectivity_screen_csv, format_dual_selectivity_poses_sdf,
    load_curated_selectivity_pair, mutate_pdb_residue, parse_mutation_token,
    scan_pocket_residue_mutations, score_ligand_dual_selectivity, screen_dataset_selectivity,
    CuratedSelectivityPair, DifferentialWaterRole, DifferentialWaterSite, DivergentResiduePair,
    DualPocketComparison, DualPocketLigandResult, MutationDdGEntry, MutationImpactClass,
    ResidueMutationScan, SelectivityTier, CURATED_SELECTIVITY_PAIRS,
};
pub use sketcher::{
    molecule_to_smiles, FunctionalGroup, LiveDesignReport, MolecularSketcher, ParentLeadDelta,
    RingTemplate, SaliNeighborHit, SketchTool,
};
pub use smarts::{
    compile_smarts, match_smarts, match_smarts_batch, AtomExpr, AtomPrim, BondExpr, BondPrim,
    SmartsError, SmartsMatchResult, SmartsPreset, SmartsQuery, CURATED_SMARTS_PRESETS,
};
pub use synth_access::{
    classify_brics_atom_env, dissect_brics, evaluate_synthetic_accessibility, match_brics_reaction,
    BricsBondCut, BricsRetrosynthesis, BricsSynthon, SynthAccessProfile, SynthTractability,
};
pub use torsion_strain::{
    analyze_torsion_strain_and_locks, compute_dihedral_deg_3d, export_locked_analogs_csv,
    export_torsion_profiles_csv, format_torsion_poses_sdf, match_csd_torsion_rule,
    perceive_scannable_dihedrals, screen_dataset_torsion_strain, BioactivePoseSource,
    BioactiveStrainTier, CsdTorsionRule, DatasetTorsionScreenRow, LockedAnalogCandidate,
    RotatableBondProfile, RotorLockStrategy, TorsionQualityBand, TorsionStrainReport,
    CSD_SMARTS_TORSION_LIBRARY, SHOICHET_SINGLE_TEU_CUTOFF, SHOICHET_TOTAL_TEU_CUTOFF,
    TORSION_RT_298K_KCAL, TORSION_SCAN_BINS,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

pub(crate) mod serde_f32_array {
    use serde::de::{Error, SeqAccess, Visitor};
    use serde::ser::SerializeTuple;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer, const N: usize>(
        arr: &[f32; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(N)?;
        for &v in arr {
            tup.serialize_element(&v)?;
        }
        tup.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<[f32; N], D::Error> {
        struct ArrayVisitor<const N: usize>;
        impl<'de, const N: usize> Visitor<'de> for ArrayVisitor<N> {
            type Value = [f32; N];
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "an array of {N} f32 values")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[f32; N], A::Error> {
                let mut out = [0.0f32; N];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| A::Error::invalid_length(i, &self))?;
                }
                Ok(out)
            }
        }
        deserializer.deserialize_tuple(N, ArrayVisitor::<N>)
    }
}

/// End-to-end analyzed SAR dataset holding sanitized molecules, descriptors, ADMET profiles,
/// physiological `pH 7.4` micro-`pKa`/protomer summaries, Ertl `SAScore` & `BRICS` counts,
/// fingerprints, 2D pharmacophore maps, 2D chemical-space coordinates, SALI activity cliffs,
/// Murcko scaffolds, MMPs, Free-Wilson SAR matrix, and PAINS alerts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub mols: Vec<Molecule>,
    pub profiles: Vec<PhyschemProfile>,
    pub admet: Vec<AdmetProfile>,
    pub pka_74: Vec<PhysiologicalChargeSummary>,
    pub synth_access: Vec<SynthAccessProfile>,
    pub alerts: Vec<Vec<StructuralAlertMatch>>,
    pub fps: Vec<Ecfp2048>,
    pub pharmacophores: Vec<PharmacophoreMap>,
    pub shape3d: Vec<Summary3dProfile>,
    pub covalent: Vec<CovalentDatasetSummary>,
    pub metabolism: Vec<MetabolicDatasetSummary>,
    pub chem_coords: Vec<[f32; 2]>,
    pub edges: Vec<SaliCliffEdge>,
    pub scaffolds: Vec<MurckoScaffold>,
    pub mmps: Vec<MatchedMolecularPair>,
    pub default_sar_matrix: Option<FreeWilsonMatrix>,
    pub pareto_leads: Vec<usize>,
    pub min_tanimoto: f32,
    pub sali_cliff_threshold: f32,
}

impl DatasetAnalysis {
    /// Runs the full parallel cheminformatics, ADMET, micro-`pKa`/protomer, 2D pharmacophore, and `SAScore`/`BRICS` pipeline.
    pub fn from_molecules(
        mols: Vec<Molecule>,
        min_tanimoto: f32,
        sali_cliff_threshold: f32,
    ) -> Self {
        let alerts: Vec<Vec<StructuralAlertMatch>> =
            mols.par_iter().map(scan_structural_alerts).collect();
        let alert_counts: Vec<usize> = alerts.iter().map(|v| v.len()).collect();
        let mut profiles: Vec<PhyschemProfile> = mols
            .par_iter()
            .zip(alert_counts.par_iter())
            .map(|(m, &ac)| evaluate_physchem(m, ac))
            .collect();
        let pareto_leads = mark_pareto_front(&mols, &mut profiles, &alert_counts);

        let admet: Vec<AdmetProfile> = mols
            .par_iter()
            .zip(profiles.par_iter())
            .map(|(m, p)| evaluate_admet(m, p))
            .collect();

        let pka_74: Vec<PhysiologicalChargeSummary> = mols
            .par_iter()
            .map(evaluate_physiological_pka_74)
            .collect();

        let synth_access: Vec<SynthAccessProfile> = mols
            .par_iter()
            .map(evaluate_synthetic_accessibility)
            .collect();

        let fps = compute_ecfp4_batch(&mols);
        let pharmacophores: Vec<PharmacophoreMap> =
            mols.par_iter().map(extract_pharmacophore_map).collect();
        let fast_3d = mols.len() > 500;
        let shape3d: Vec<Summary3dProfile> = mols
            .par_iter()
            .map(|m| evaluate_3d_summary_profile(m, fast_3d))
            .collect();
        let covalent: Vec<CovalentDatasetSummary> =
            mols.par_iter().map(evaluate_covalent_summary).collect();
        let metabolism: Vec<MetabolicDatasetSummary> =
            mols.par_iter().map(evaluate_metabolic_summary).collect();
        let chem_coords = embed_chemical_space_2d(&mols, &fps);
        let edges = compute_sali_landscape(&mols, &fps, min_tanimoto, sali_cliff_threshold, false);
        let scaffolds = extract_murcko_scaffolds(&mols);
        let mmps = discover_mmps(&mols, &profiles);
        let default_sar_matrix = build_auto_free_wilson_matrix(&mols, &profiles, 3, 0.15);

        Self {
            mols,
            profiles,
            admet,
            pka_74,
            synth_access,
            alerts,
            fps,
            pharmacophores,
            shape3d,
            covalent,
            metabolism,
            chem_coords,
            edges,
            scaffolds,
            mmps,
            default_sar_matrix,
            pareto_leads,
            min_tanimoto,
            sali_cliff_threshold,
        }
    }

    pub fn merge_imported_molecules(&mut self, incoming: Vec<Molecule>) -> usize {
        let mut seen: BTreeSet<String> = self.mols.iter().map(molecule_to_smiles).collect();
        let mut combined = self.mols.clone();
        let mut added = 0usize;
        for mut m in incoming {
            m.sanitize();
            let can = molecule_to_smiles(&m);
            if !can.is_empty() && seen.insert(can.clone()) {
                m.smiles = can;
                combined.push(m);
                added += 1;
            }
        }
        if added > 0 {
            let min_t = self.min_tanimoto;
            let sali_th = self.sali_cliff_threshold;
            *self = Self::from_molecules(combined, min_t, sali_th);
        }
        added
    }

    pub fn push_sketched_molecule(&mut self, mut mol: Molecule) -> usize {
        mol.sanitize();
        mol.smiles = molecule_to_smiles(&mol);
        let next_num = self.mols.len() + 1;
        if mol.id.is_empty() || self.mols.iter().any(|m| m.id == mol.id) {
            mol.id = format!("SKETCH-{next_num:03}");
        }
        if mol.name.is_empty() || self.mols.iter().any(|m| m.name == mol.name) {
            mol.name = format!("Analog-{next_num:03}");
        }
        if mol.source_url.is_empty() {
            mol.source_url = "http://opensmiles.org/opensmiles.html".to_string();
        }
        let mut all_mols = self.mols.clone();
        all_mols.push(mol);
        let min_t = self.min_tanimoto;
        let sali_th = self.sali_cliff_threshold;
        *self = Self::from_molecules(all_mols, min_t, sali_th);
        self.mols.len() - 1
    }

    pub fn evaluate_smarts(&self, query: &SmartsQuery) -> Vec<SmartsMatchResult> {
        match_smarts_batch(&self.mols, query)
    }

    pub fn solve_sar_matrix(
        &self,
        core_name: &str,
        core_smarts: &str,
        target_filter: Option<&str>,
        max_slots: usize,
        ridge_lambda: f64,
    ) -> Result<FreeWilsonMatrix, String> {
        build_free_wilson_for_smarts(
            &self.mols,
            &self.profiles,
            core_name,
            core_smarts,
            target_filter,
            max_slots,
            ridge_lambda,
        )
    }

    pub fn refresh_sali_edges(&mut self, min_tanimoto: f32, sali_cliff_threshold: f32) {
        self.min_tanimoto = min_tanimoto;
        self.sali_cliff_threshold = sali_cliff_threshold;
        self.edges = compute_sali_landscape(
            &self.mols,
            &self.fps,
            min_tanimoto,
            sali_cliff_threshold,
            false,
        );
    }

    pub fn from_file(
        path: &Path,
        min_tanimoto: f32,
        sali_cliff_threshold: f32,
    ) -> Result<Self, MolIoError> {
        let raw = std::fs::read_to_string(path)?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mols = if ext == "sdf"
            || ext == "mol"
            || raw.contains("V2000")
            || raw.contains("V3000")
            || raw.contains("M  V30")
        {
            parse_sdf_dataset(&raw)?
        } else {
            parse_delimited_dataset(&raw)?
        };
        Ok(Self::from_molecules(
            mols,
            min_tanimoto,
            sali_cliff_threshold,
        ))
    }

    pub fn export_summary_csv(&self) -> String {
        let mut out = String::from(
            "id,name,target,metric,raw_nm,pact,formula,mw,clogp,microstate_logd_74,charge_class_74,mean_charge_74,esol_logs,boiled_egg,pgp_sub,pfizer_3_75,gsk_4_400,herg_risk,sa_score,synth_tractability,brics_cuts,brics_synthons,hbd,hba,rotb,fsp3,qed_mean,cns_mpo,lle,ro5_violations,pains_alerts,pareto_lead,smiles,source_url\n",
        );
        for (i, m) in self.mols.iter().enumerate() {
            let p = &self.profiles[i];
            let ad = &self.admet[i];
            let pk = &self.pka_74[i];
            let sa = &self.synth_access[i];
            let alert_codes = if self.alerts[i].is_empty() {
                "Clean".to_string()
            } else {
                self.alerts[i]
                    .iter()
                    .map(|a| a.code)
                    .collect::<Vec<_>>()
                    .join(";")
            };
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",\"{}\",{:.4},{:.2},{},{:.2},{:.2},{:.2},\"{}\",{:.2},{:.2},\"{}\",{},{},{},\"{}\",{:.2},\"{}\",{},{},{},{},{},{:.2},{:.3},{:.2},{:.2},{},\"{}\",{},{},\"{}\"\n",
                m.id,
                m.name,
                m.target,
                m.metric,
                m.raw_nm,
                m.pact,
                p.formula,
                p.mw,
                p.clogp,
                pk.microstate_logd_74,
                pk.charge_class.label(),
                pk.mean_net_charge_74,
                ad.esol_logs,
                ad.boiled_egg.label(),
                ad.pgp_substrate,
                ad.pfizer_3_75_pass,
                ad.gsk_4_400_pass,
                ad.herg_risk.label(),
                sa.sa_score,
                sa.tractability.short_label(),
                sa.brics_cleavable_bonds,
                sa.brics_synthon_count,
                p.hbd,
                p.hba,
                p.rotatable_bonds,
                p.fsp3,
                p.qed_mean,
                p.cns_mpo,
                p.lle,
                p.ro5_violations,
                alert_codes,
                p.pareto_optimal,
                m.smiles,
                m.source_url
            ));
        }
        out
    }

    pub fn export_metabolism_summary_csv(&self) -> String {
        let mut out = String::from(
            "id,name,target,pact,top_som,top_reaction,primary_isoform,best_smartcyp_kj,cl_int_ml_min_kg,stability_class,phase2_handles,reactive_bioactivation,smiles,source_url\n",
        );
        for (i, m) in self.mols.iter().enumerate() {
            let mt = &self.metabolism[i];
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{:.2},\"{}\",\"{}\",\"{}\",{:.1},{:.1},\"{}\",{},{},\"{}\",\"{}\"\n",
                m.id,
                m.name,
                m.target,
                m.pact,
                mt.top_som_label,
                mt.top_reaction,
                mt.primary_isoform.label(),
                mt.best_smartcyp_score,
                mt.cl_int_ml_min_kg,
                mt.stability_class.short_label(),
                mt.phase2_direct_handles,
                mt.reactive_bioactivation_risk,
                m.smiles,
                m.source_url
            ));
        }
        out
    }

    pub fn export_covalent_summary_csv(&self) -> String {
        let mut out = String::from(
            "id,name,target,pact,has_warhead,primary_warhead,target_residue,reversibility,reactivity_window,gsh_half_life_h,kinact_over_ki_m_inv_s_inv,swap_handle_available,smiles,source_url\n",
        );
        for (i, m) in self.mols.iter().enumerate() {
            let cv = &self.covalent[i];
            let gsh = cv
                .gsh_half_life_h
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".to_string());
            let eff = cv
                .kinact_over_ki
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{:.2},{},\"{}\",\"{}\",\"{}\",\"{}\",{},{},{},\"{}\",\"{}\"\n",
                m.id,
                m.name,
                m.target,
                m.pact,
                cv.has_warhead,
                cv.primary_warhead_name,
                cv.target_residue.label(),
                cv.reversibility.short_label(),
                cv.window.short_label(),
                gsh,
                eff,
                cv.swap_handle_available,
                m.smiles,
                m.source_url
            ));
        }
        out
    }

    pub fn export_pka_protomers_csv(&self) -> String {
        let mut out = String::from(
            "id,name,target,pact,charge_class_74,dominant_charge,mean_charge_74,f_neutral_74,f_cation_74,f_anion_74,f_zwitterion_74,strongest_acid_pka,strongest_base_pka,isoelectric_pi,logd_20_gastric,logd_74_plasma,major_protomers_74,tautomer_count,dominant_protomer_smiles,source_url\n",
        );
        for (i, m) in self.mols.iter().enumerate() {
            let pk = &self.pka_74[i];
            let ac = pk
                .strongest_acid_pka
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".into());
            let bs = pk
                .strongest_base_pka
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".into());
            let pi = pk
                .isoelectric_pi
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".into());
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{:.2},\"{}\",{:+},{:+.2},{:.3},{:.3},{:.3},{:.3},{},{},{},{:.2},{:.2},{},{},{},\"{}\"\n",
                m.id,
                m.name,
                m.target,
                m.pact,
                pk.charge_class.label(),
                pk.dominant_net_charge,
                pk.mean_net_charge_74,
                pk.neutral_fraction_74,
                pk.cation_fraction_74,
                pk.anion_fraction_74,
                pk.zwitterion_fraction_74,
                ac,
                bs,
                pi,
                pk.gastric_logd_20,
                pk.microstate_logd_74,
                pk.major_protomers_74,
                pk.tautomer_count,
                pk.dominant_protomer_smiles,
                m.source_url
            ));
        }
        out
    }

    pub fn export_synth_access_csv(&self) -> String {
        let mut out = String::from(
            "id,name,target,pact,sa_score,tractability,fragment_score,size_penalty,stereo_penalty,spiro_penalty,bridge_penalty,macro_penalty,symmetry_bonus,chiral_centers,spiro_atoms,bridgehead_atoms,brics_cleavable_bonds,brics_synthons,est_convergent_steps,smiles,source_url\n",
        );
        for (i, m) in self.mols.iter().enumerate() {
            let sa = &self.synth_access[i];
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{:.2},{:.2},\"{}\",{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{},\"{}\"\n",
                m.id,
                m.name,
                m.target,
                m.pact,
                sa.sa_score,
                sa.tractability.short_label(),
                sa.fragment_score,
                sa.size_penalty,
                sa.stereo_penalty,
                sa.spiro_penalty,
                sa.bridgehead_penalty,
                sa.macrocycle_penalty,
                sa.symmetry_bonus,
                sa.chiral_centers,
                sa.spiro_atoms,
                sa.bridgehead_atoms,
                sa.brics_cleavable_bonds,
                sa.brics_synthon_count,
                sa.convergent_steps_est,
                m.smiles,
                m.source_url
            ));
        }
        out
    }

    pub fn export_cliffs_csv(&self) -> String {
        let mut out = String::from(
            "compound_a,compound_b,target,tanimoto_ecfp4,delta_pact,sali,is_activity_cliff\n",
        );
        for e in &self.edges {
            if !e.same_target {
                continue;
            }
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{:.3},{:.2},{:.2},{}\n",
                self.mols[e.mol_a].name,
                self.mols[e.mol_b].name,
                self.mols[e.mol_a].target,
                e.tanimoto,
                e.delta_pact,
                e.sali,
                e.is_cliff
            ));
        }
        out
    }

    pub fn export_mmp_csv(&self) -> String {
        let mut out = String::from(
            "from_compound,to_compound,target,mmp_type,transform,conserved_core,delta_pact,delta_clogp,delta_tpsa,delta_lle\n",
        );
        for mmp in &self.mmps {
            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",{:.2},{:.2},{:.2},{:.2}\n",
                self.mols[mmp.mol_from].name,
                self.mols[mmp.mol_to].name,
                mmp.target,
                mmp.kind.label(),
                mmp.transform,
                mmp.core_label,
                mmp.delta_pact,
                mmp.delta_clogp,
                mmp.delta_tpsa,
                mmp.delta_lle
            ));
        }
        out
    }

    pub fn export_pharmacophore_csv(&self, hits: &[PharmacophoreHit]) -> String {
        let mut out = String::from(
            "rank,id,name,target,pact,pharm_sim,triangle_tanimoto,cats2_cosine,ecfp4_tanimoto,shared_triangles,scaffold_hop,hop_enrichment,murcko_scaffold,top_consensus_triangle,smiles,source_url\n",
        );
        for (rank, h) in hits.iter().enumerate() {
            let m = &self.mols[h.mol_idx];
            let pm = &self.pharmacophores[h.mol_idx];
            let al = &h.alignment;
            let top_tri = al
                .matched_triangles
                .first()
                .map(|t| t.label.as_str())
                .unwrap_or("None");
            out.push_str(&format!(
                "{},\"{}\",\"{}\",\"{}\",{:.2},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},\"{}\",\"{}\",{},\"{}\"\n",
                rank + 1,
                m.id,
                m.name,
                m.target,
                m.pact,
                al.pharm_similarity,
                al.triangle_tanimoto,
                al.cats_cosine,
                al.ecfp4_tanimoto,
                al.shared_triangle_count,
                al.is_scaffold_hop,
                al.scaffold_hop_enrichment,
                pm.murcko_label,
                top_tri,
                m.smiles,
                m.source_url
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_26_benchmark_formulas_and_molecular_weights() {
        let expected: [(&str, &str, f64); 26] = [
            ("BRIDGES-56", "C18H18BrN3O2", 388.26),
            ("CHEMBL29197", "C16H14BrN3O2", 360.21),
            ("CHEMBL1173655", "C24H25ClFN5O3", 485.94),
            ("CHEMBL553", "C22H23N3O4", 393.44),
            ("NSC693255", "C16H14ClN3O2", 315.75),
            ("CHEMBL554", "C29H26ClFN4O4S", 581.06),
            ("CHEMBL939", "C22H24ClFN4O3", 446.91),
            ("CHEMBL24828", "C22H24BrFN4O2", 475.36),
            ("CHEMBL726", "C22H26F3N3OS", 437.53),
            ("CHEMBL567", "C21H26ClN3OS", 403.97),
            ("CHEMBL85", "C23H27FN4O2", 410.49),
            ("CHEMBL54", "C21H23ClFNO2", 375.87),
            ("CHEMBL708", "C21H21ClN4OS", 412.94),
            ("DB00477", "C17H19ClN2S", 318.86),
            ("CHEMBL1113", "C17H16ClN3O", 313.78),
            ("DB00334", "C17H20N4S", 312.43),
            ("CHEMBL831", "C18H18ClN3O", 327.81),
            ("CHEMBL42", "C18H19ClN4", 326.83),
            ("DB01224", "C21H25N3O2S", 383.51),
            ("CHEMBL122", "C17H14O4S", 314.36),
            ("CHEMBL118", "C17H14F3N3O2S", 381.37),
            ("CAS181695", "C16H14N2O3S", 314.36),
            ("DB00328", "C19H16ClNO4", 357.79),
            ("PAINS-CUR", "C21H20O6", 368.38),
            ("CHEMBL56337", "C15H13NO3S2", 319.40),
            ("PAINS-QUER", "C15H10O7", 302.24),
        ];

        let mols = load_benchmark_26();
        assert_eq!(mols.len(), 26);
        for (mol, (exp_id, exp_formula, exp_mw)) in mols.iter().zip(expected.iter()) {
            assert_eq!(mol.id, *exp_id);
            assert_eq!(mol.molecular_formula(), *exp_formula);
            assert!((mol.molecular_weight() - *exp_mw).abs() < 0.08);
        }
    }

    #[test]
    fn test_chembl_json_desalt_median_and_pubchem_parsers() {
        let sample_chembl = r#"{
            "activities": [
                {
                    "molecule_chembl_id": "CHEMBL553",
                    "molecule_pref_name": "ERLOTINIB HYDROCHLORIDE",
                    "canonical_smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC.Cl",
                    "standard_type": "IC50",
                    "pchembl_value": "8.50"
                },
                {
                    "molecule_chembl_id": "CHEMBL553",
                    "molecule_pref_name": "ERLOTINIB",
                    "canonical_smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
                    "standard_type": "IC50",
                    "pchembl_value": 8.90
                }
            ]
        }"#;
        let (mols, rep) = parse_chembl_activities_json(sample_chembl, "EGFR", 50).unwrap();
        assert_eq!(mols.len(), 1);
        assert_eq!(rep.desalted_count, 1);
        assert_eq!(mols[0].molecular_formula(), "C22H23N3O4");
        assert!((mols[0].pact - 8.70).abs() < 1e-4);

        let sample_pubchem_csv = "AID,SID,CID,Activity Outcome,Activity Name,Activity Value [uM]\n\
                                  1433,101,176870,Active,IC50,0.002\n\
                                  1433,102,999999,Inactive,IC50,50.0\n";
        let active = extract_active_cids_from_pubchem_concise_csv(sample_pubchem_csv, 20);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, 176870);
        assert!((active[0].2 - 8.70).abs() < 0.05);
    }

    #[test]
    fn test_sqlite_and_json_project_workspace_roundtrip() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let sketch = MolecularSketcher::from_dataset_compound(&ds, 4);

        let ws = SarProjectWorkspace::capture(
            "EGFR-Lead-Opt",
            "Quinazoline cliff campaign",
            &ds,
            "c1ncnc2ccccc12",
            "EGFR",
            "4-Anilinoquinazoline (EGFR)",
            "c1ccc(Nc2ncnc3ccccc23)cc1",
            3,
            0.15,
            Some(&sketch.mol),
            Some("NSC693255".to_string()),
        );

        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        save_project_to_sqlite_conn(&mut conn, &ws, &ds).unwrap();
        let loaded_sql = load_project_from_sqlite_conn(&conn).unwrap();
        assert_eq!(loaded_sql.project_name, "EGFR-Lead-Opt");
        assert_eq!(loaded_sql.ds.mols.len(), 26);
        assert_eq!(loaded_sql.active_smarts_query, "c1ncnc2ccccc12");
        assert!(loaded_sql.sketcher_mol.is_some());

        let json_str = serde_json::to_string(&ws).unwrap();
        let ws_back: SarProjectWorkspace = serde_json::from_str(&json_str).unwrap();
        let loaded_json = ws_back.into_loaded_session().unwrap();
        assert_eq!(loaded_json.ds.mols.len(), 26);
        assert_eq!(
            loaded_json.ds.mols[0].molecular_formula(),
            ds.mols[0].molecular_formula()
        );
    }

    #[test]
    fn test_2d_pharmacophore_features_triangles_and_cox2_scaffold_hop() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        assert_eq!(ds.pharmacophores.len(), 26);

        let indo_idx = ds.mols.iter().position(|m| m.id == "DB00328").unwrap();
        let indo_pm = &ds.pharmacophores[indo_idx];
        assert!(indo_pm.counts_by_kind[PharmFeatureKind::NegIonizable.type_index()] >= 1);
        assert!(indo_pm.counts_by_kind[PharmFeatureKind::Aromatic.type_index()] >= 2);
        assert!(indo_pm.counts_by_kind[PharmFeatureKind::Hydrophobic.type_index()] >= 1);

        let cloz_idx = ds.mols.iter().position(|m| m.id == "CHEMBL42").unwrap();
        let cloz_pm = &ds.pharmacophores[cloz_idx];
        assert!(cloz_pm.counts_by_kind[PharmFeatureKind::PosIonizable.type_index()] >= 1);

        let cel_idx = ds.mols.iter().position(|m| m.id == "CHEMBL118").unwrap();
        let val_idx = ds.mols.iter().position(|m| m.id == "CAS181695").unwrap();
        let ecfp_sim = ds.fps[cel_idx].tanimoto(&ds.fps[val_idx]);
        let align = align_pharmacophores(
            &ds.pharmacophores[cel_idx],
            &ds.pharmacophores[val_idx],
            ecfp_sim,
        );
        assert!(!align.same_murcko_scaffold);
        assert!(align.pharm_similarity > 0.55);
        assert!(align.shared_triangle_count >= 5);
        assert!(align.is_scaffold_hop);
    }

    #[test]
    fn test_ertl_sascore_complexity_penalties_and_brics_retrosynthesis() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        assert_eq!(ds.synth_access.len(), 26);

        let nsc_idx = ds.mols.iter().position(|m| m.id == "NSC693255").unwrap();
        let nsc_sa = &ds.synth_access[nsc_idx];
        assert!(nsc_sa.sa_score >= 1.5 && nsc_sa.sa_score <= 3.5);

        let complex_spiro_bridged =
            parse_smiles("CC1(C)CC2CCC1(C)C1(CCN(CC1)C(=O)C1CCCCCCCCCC1)O2").unwrap();
        let complex_sa = evaluate_synthetic_accessibility(&complex_spiro_bridged);
        assert!(complex_sa.spiro_atoms >= 1);
        assert!(complex_sa.bridgehead_atoms >= 2);
        assert!(complex_sa.macrocycle_rings >= 1);
        assert!(complex_sa.sa_score > nsc_sa.sa_score + 1.8);

        let cur_idx = ds.mols.iter().position(|m| m.id == "PAINS-CUR").unwrap();
        assert!(ds.synth_access[cur_idx].symmetry_bonus > 0.20);

        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        let retro = dissect_brics(&ds.mols[erl_idx]);
        assert!(retro.active_cuts.len() >= 3);
        assert!(retro.synthons.len() >= 4);
        assert!(retro.synthons.iter().all(|s| s.parent_atoms.len() >= 2));
        assert!(retro.max_synthon_sa < retro.parent_sa.sa_score);
    }

    #[test]
    fn test_brics_combinatorial_reassembler_and_pareto_scoring() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = BricsRecombineConfig::default();
        let (hybrids, summary) =
            recombine_brics_leads(&ds, ds.default_sar_matrix.as_ref(), &cfg);

        assert!(summary.active_leads_used >= 15);
        assert!(summary.unique_half_synthons >= 20);
        assert!(!hybrids.is_empty());
        assert!(summary.pareto_hybrid_count >= 1);

        let existing_smi: BTreeSet<String> = ds.mols.iter().map(molecule_to_smiles).collect();
        for h in &hybrids {
            assert!(!existing_smi.contains(&h.canonical_smiles));
            assert_eq!(h.pains_count, 0);
            assert!(h.sa_profile.sa_score <= cfg.max_sa_score);
            assert!(h.profile.qed_mean >= cfg.min_qed);
            assert!(h.free_wilson.is_some() || h.predicted_pact >= 5.0);
            assert!(!h.formed_bonds.is_empty());
        }
    }

    #[test]
    fn test_mcgregor_vf2_mcs_and_2d_procrustes_series_aligner() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);

        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        let gef_idx = ds.mols.iter().position(|m| m.id == "CHEMBL939").unwrap();
        let strict_cfg = McsConfig::default();
        let pair_mcs =
            find_pairwise_mcs(&ds.mols[erl_idx], &ds.mols[gef_idx], &strict_cfg).unwrap();
        assert!(pair_mcs.mcs_atom_count >= 19);
        assert!(pair_mcs.mcs_tanimoto > 0.45);
        assert!(pair_mcs.core_rmsd_2d < 0.25);

        let cel_idx = ds.mols.iter().position(|m| m.id == "CHEMBL118").unwrap();
        let val_idx = ds.mols.iter().position(|m| m.id == "CAS181695").unwrap();
        let mut bio_cfg = McsConfig::default();
        bio_cfg.atom_mode = McsAtomMatchMode::AnyRingHetero;
        let cox_mcs = find_pairwise_mcs(&ds.mols[cel_idx], &ds.mols[val_idx], &bio_cfg).unwrap();
        assert!(cox_mcs.mcs_atom_count >= 20);

        let egfr_indices: Vec<usize> = (0..ds.mols.len())
            .filter(|&i| ds.mols[i].target == "EGFR")
            .collect();
        let series_align =
            detect_and_align_series_mcs(&ds, &egfr_indices, "EGFR", &strict_cfg).unwrap();
        assert_eq!(series_align.matched_member_count, 8);
        assert!(series_align.mcs_atom_count >= 17);
        assert!(series_align.mean_core_rmsd_2d < 0.20);
        assert!(!series_align.slot_labels.is_empty());
    }

    #[test]
    fn test_physiological_micro_pka_protomers_and_tautomers() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        assert_eq!(ds.pka_74.len(), 26);

        // 1. Indomethacin (DB00328): carboxylic acid -> Anionic (-1) at pH 7.4 (> 99%)
        let indo_idx = ds.mols.iter().position(|m| m.id == "DB00328").unwrap();
        assert_eq!(ds.pka_74[indo_idx].charge_class, ChargeClass74::Anionic);
        assert!(ds.pka_74[indo_idx].anion_fraction_74 > 0.99);

        // 2. Chlorpromazine (DB00477): basic dimethylaminopropyl amine -> Cationic (+1) at pH 7.4
        let cpz_idx = ds.mols.iter().position(|m| m.id == "DB00477").unwrap();
        assert_eq!(ds.pka_74[cpz_idx].charge_class, ChargeClass74::Cationic);
        assert!(ds.pka_74[cpz_idx].cation_fraction_74 > 0.85);

        // 3. Gefitinib (CHEMBL939): diprotic base (morpholine pKa ~ 7.5 + quinazoline N1 pKa ~ 5.3)
        //    co-exists as both +1 and 0 protomers at pH 7.4!
        let gef_idx = ds.mols.iter().position(|m| m.id == "CHEMBL939").unwrap();
        let gef_rep = analyze_pka_and_protomers(&ds.mols[gef_idx], 7.4);
        assert!(gef_rep.sites.len() >= 2);
        assert!(ds.pka_74[gef_idx].major_protomers_74 >= 2);

        // 4. Ampholyte amino-acid test (e.g. Zwitterionic GABA `NCCCC(=O)O`) detects pI ~ 7.3 and Zwitterion!
        let gaba = parse_smiles("NCCCC(=O)O").unwrap();
        let gaba_rep = analyze_pka_and_protomers(&gaba, 7.4);
        assert_eq!(gaba_rep.charge_class_at_ph, ChargeClass74::Zwitterionic);
        assert!(gaba_rep.isoelectric_pi.is_some());

        // 5. Curcumin (PAINS-CUR) enumerates keto-enol tautomers!
        let cur_idx = ds.mols.iter().position(|m| m.id == "PAINS-CUR").unwrap();
        let cur_tauts = enumerate_tautomers(&ds.mols[cur_idx]);
        assert!(cur_tauts.len() >= 1);
    }


    #[test]
    fn test_conformer3d_dg_mmff_usr_and_rocs_overlay() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        assert_eq!(ds.shape3d.len(), 26);

        // 1. Benzene (`c1ccccc1`) is strictly planar (PBF ~ 0.00 A) vs Cyclohexane (`C1CCCCC1`) chair pucker (PBF > 0.15 A)
        let benz = parse_smiles("c1ccccc1").unwrap();
        let cyhex = parse_smiles("C1CCCCC1").unwrap();
        let b_3d = evaluate_3d_summary_profile(&benz, false);
        let c_3d = evaluate_3d_summary_profile(&cyhex, false);
        assert!(b_3d.pbf_angstrom < 0.03);
        assert!(c_3d.pbf_angstrom > 0.15);

        // 2. Gasteiger-Marsili PEOE partial charges conserve total formal charge and polarize C=O
        let indo_idx = ds.mols.iter().position(|m| m.id == "DB00328").unwrap();
        let q_sum: f32 = ds.shape3d[indo_idx].gasteiger_charges.iter().sum();
        assert!(q_sum.abs() < 0.02);
        assert!(ds.shape3d[indo_idx].dipole_debye > 0.5);

        // 3. Self-ROCS alignment yields Shape Tanimoto == 1.0 and ComboScore > 1.85
        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        let self_al = align_rocs_3d(
            &ds.mols[erl_idx],
            &ds.shape3d[erl_idx].coords3d,
            &ds.shape3d[erl_idx].gasteiger_charges,
            &ds.shape3d[erl_idx].color_atoms,
            &ds.mols[erl_idx],
            &ds.shape3d[erl_idx].coords3d,
            &ds.shape3d[erl_idx].gasteiger_charges,
            &ds.shape3d[erl_idx].color_atoms,
        );
        assert!(self_al.shape_tanimoto > 0.999);
        assert!(self_al.usr_similarity > 0.999);

        // 4. 3D Scaffold-hopping screen on Celecoxib (CHEMBL118) ranks Valdecoxib (CAS181695) near the top
        let cel_idx = ds.mols.iter().position(|m| m.id == "CHEMBL118").unwrap();
        let hits = screen_3d_overlay_library(
            &ds.mols[cel_idx],
            &ds.shape3d[cel_idx],
            &ds,
            None,
            true,
            26,
        );
        assert!(!hits.is_empty());
        assert!(hits[0].alignment.combo_score > 1.15);
    }


    #[test]
    fn test_covalent_warhead_afatinib_gsh_and_swapper() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        assert_eq!(ds.covalent.len(), 26);

        // 1. Afatinib (CHEMBL1173655) carries a tuned (E)-4-(dimethylamino)but-2-enamide Cys warhead
        let afa_idx = ds
            .mols
            .iter()
            .position(|m| m.id == "CHEMBL1173655")
            .unwrap();
        assert!(ds.covalent[afa_idx].has_warhead);
        assert_eq!(ds.covalent[afa_idx].target_residue, TargetResidue::Cysteine);
        assert_eq!(
            ds.covalent[afa_idx].window,
            ReactivityWindow::TunedClinicalSweetSpot
        );

        // 2. Full warhead swapper on Afatinib enumerates all 12 reversible/irreversible warheads + propionamide control
        let rep = analyze_covalent_warheads_and_swaps(
            &ds.mols[afa_idx],
            &ds,
            ds.default_sar_matrix.as_ref(),
            50.0,
        );
        assert_eq!(rep.swaps.len(), 12);
        assert!(rep
            .swaps
            .iter()
            .any(|s| s.reversibility == WarheadReversibility::ReversibleCovalent));
        assert!(rep
            .swaps
            .iter()
            .any(|s| s.reversibility == WarheadReversibility::NonCovalentControl));

        // 3. Non-covalent Erlotinib (CHEMBL553) supports ligand-first covalentization via its terminal C#C handle
        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        assert!(!ds.covalent[erl_idx].has_warhead);
        let erl_rep = analyze_covalent_warheads_and_swaps(
            &ds.mols[erl_idx],
            &ds,
            ds.default_sar_matrix.as_ref(),
            100.0,
        );
        assert_eq!(erl_rep.swaps.len(), 12);
    }


    #[test]
    fn test_protac_linker_crbn_vhl_chameleonicity_and_hook_effect() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        // Gefitinib (CHEMBL939) -> detects solvent-exposed morpholine/alkoxy exit vectors & builds 60 CRBN/VHL chimeras
        let gef_idx = ds.mols.iter().position(|m| m.id == "CHEMBL939").unwrap();
        let rep = design_protac_series(
            &ds.mols[gef_idx],
            &E3RecruiterKind::ALL,
            0,
            100.0,
            150.0,
        );
        assert!(!rep.exit_vectors.is_empty());
        assert_eq!(rep.candidates.len(), 60);

        let top = &rep.candidates[0];
        assert!(top.physchem.mw > 600.0);
        assert!(top.chameleon.epsa_folded_a2 < top.chameleon.tpsa_open_a2);
        assert!(top.ternary.ti50_nm > top.ternary.tf50_nm * 10.0);
        assert!(!top.poi_atoms.is_empty() && !top.linker_atoms.is_empty() && !top.e3_atoms.is_empty());
    }


    #[test]
    fn test_smartcyp_som_and_phase1_phase2_metabolite_enumeration() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        assert_eq!(ds.metabolism.len(), 26);

        // 1. Chlorpromazine (DB00477): phenothiazine thioether S-oxidation & N-demethylation / CYP2D6 basic amine!
        let cpz_idx = ds.mols.iter().position(|m| m.id == "DB00477").unwrap();
        let cpz_rep = analyze_cyp_metabolism(&ds.mols[cpz_idx], true);
        assert!(cpz_rep.sites.len() >= 5);
        assert_eq!(cpz_rep.sites[0].reaction_kind, SomReactionKind::SOxidation);
        assert!(!cpz_rep.metabolites.is_empty());

        // 2. Celecoxib (CHEMBL118): sulfonamide CYP2C9 anchor drives benzylic 4-methylphenyl hydroxylation!
        let cel_idx = ds.mols.iter().position(|m| m.id == "CHEMBL118").unwrap();
        let cel_rep = analyze_cyp_metabolism(&ds.mols[cel_idx], true);
        assert!(cel_rep
            .sites
            .iter()
            .take(2)
            .any(|s| s.reaction_kind == SomReactionKind::BenzylicHydroxylation));
        assert!(cel_rep
            .metabolites
            .iter()
            .any(|m| m.phase == MetabolismPhase::SequentialPhaseIToII));

        // 3. Indomethacin (DB00328): carboxylic acid forms direct Phase II acyl glucuronide (+176.03 Da)!
        let indo_idx = ds.mols.iter().position(|m| m.id == "DB00328").unwrap();
        let indo_rep = analyze_cyp_metabolism(&ds.mols[indo_idx], true);
        assert!(indo_rep
            .metabolites
            .iter()
            .any(|m| m.phase == MetabolismPhase::PhaseIIConjugate && (m.mass_shift_da - 176.03).abs() < 0.25));
    }


    #[test]
    fn test_macrocyclizer_rcm_lactam_kier_phi_and_3d_preorganization() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        // PD-153035 (CHEMBL29197) or Erlotinib (CHEMBL553): quinazoline C6/C7 alkoxy <-> aniline aryl wing!
        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        let rep = design_macrocycles(&ds.mols[erl_idx], 0);
        assert!(!rep.terminal_pairs.is_empty());
        assert_eq!(rep.candidates.len(), 10);

        let top = &rep.candidates[0];
        assert!(top.preorg.macro_ring_size >= 11);
        assert!(top.preorg.entropy_ddg_kcal < 0.0);
        assert!(top.mol.rings.iter().any(|r| r.len() >= 11));
    }

    #[test]
    fn test_fragment_grower_ro3_exit_vectors_and_ge_lle_le_optimization() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);

        // 1. Curated 7-Azaindole Vemurafenib seed (HAC=9, strict Rule-of-3 pass)
        let aza = build_curated_ro3_seed_molecule(0);
        let rep_aza = grow_fragment_along_vector(&aza, 0, false, Some((&ds.mols, &ds.fps)));
        assert!(rep_aza.ro3_audit.strict_ro3_pass);
        assert!(rep_aza.exit_vectors.len() >= 4);
        assert!(rep_aza.candidates.len() >= 14);
        assert!(rep_aza.candidates[0].efficiency.ge > 0.25);

        // 2. Deconstruct-to-Ro3 Core on Erlotinib (CHEMBL553, MW > 390 Da -> quinazoline Ro3 core <= 300 Da)
        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        let rep_erl = grow_fragment_along_vector(&ds.mols[erl_idx], 0, true, Some((&ds.mols, &ds.fps)));
        assert!(rep_erl.was_deconstructed_to_ro3);
        assert!(rep_erl.ro3_audit.mw <= 300.0);
        assert!(!rep_erl.candidates.is_empty());
    }

    #[test]
    fn test_prodrug_promoiety_enumerator_and_ces1_alp_activation() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);

        // 1. Oseltamivir Carboxylate (BCS III polar carboxylate) -> CES1 Ethyl Ester boosts oral F%!
        let oslt = build_curated_prodrug_parent(0);
        let rep_oslt = design_prodrugs(&oslt, 0);
        assert!(!rep_oslt.handles.is_empty());
        assert_eq!(rep_oslt.handles[0].kind, ProdrugHandleKind::CarboxylicAcid);
        assert!(rep_oslt.candidates.len() >= 7);
        assert!(rep_oslt.candidates[0].activation.f_oral_fold_gain > 1.5);

        // 2. Tedizolid Oxazolidinone Alcohol (BCS II brick-dust) -> ALP Phosphate boosts solubility >100x!
        let tedz = build_curated_prodrug_parent(1);
        let rep_tedz = design_prodrugs(&tedz, 0);
        assert!(rep_tedz
            .candidates
            .iter()
            .any(|c| c.family == PromoietyFamily::Phosphate && c.activation.solubility_fold_change > 100.0));

        // 3. Celecoxib (CHEMBL118, primary sulfonamide) -> Parecoxib-class & Oxazolidinone carbamate prodrugs!
        let cel_idx = ds.mols.iter().position(|m| m.id == "CHEMBL118").unwrap();
        let rep_cel = design_prodrugs(&ds.mols[cel_idx], 0);
        assert!(!rep_cel.candidates.is_empty());
        assert!(rep_cel
            .candidates
            .iter()
            .any(|c| c.family == PromoietyFamily::Oxazolidinone));
    }

    #[test]
    fn test_oral_pbpk_acat_first_pass_and_unbound_brain_kpuu() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);

        // 1. Haloperidol (CHEMBL54, CNS D2 antipsychotic, cationic base): high Kp,uu,brain >= 0.30 & large Vss > 2.5 L/kg!
        let hal_idx = ds.mols.iter().position(|m| m.id == "CHEMBL54").unwrap();
        let rep_hal = simulate_oral_pbpk(&ds.mols[hal_idx], PbpkDosingConfig::default());
        assert!(rep_hal.kp_uu_brain >= 0.30);
        assert!(rep_hal.vss_l_per_kg > 2.5);
        assert!(rep_hal.peak_ro_brain_pct > 50.0);

        // 2. Lapatinib (CHEMBL554, MW=581, TPSA=115, P-gp+ BCS II): peripherally restricted Kp,uu,brain < 0.10,
        //    and Fed FeSSIF state boosts oral bioavailability vs Fasted!
        let lap_idx = ds.mols.iter().position(|m| m.id == "CHEMBL554").unwrap();
        let rep_fast = simulate_oral_pbpk(&ds.mols[lap_idx], PbpkDosingConfig::default());
        let rep_fed = simulate_oral_pbpk(
            &ds.mols[lap_idx],
            PbpkDosingConfig {
                prandial: PrandialState::FedBileMicelles,
                ..PbpkDosingConfig::default()
            },
        );
        assert!(rep_fast.kp_uu_brain < 0.10);
        assert!(rep_fed.f_oral_pct > rep_fast.f_oral_pct);
    }

    #[test]
    fn test_v210_campaign_scorecard_and_html_svg_dossier() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let periph = evaluate_campaign_scorecard(&ds, CampaignTargetProfile::OralPeripheral, 50.0);
        assert_eq!(periph.total_compounds, 26);
        assert!(periph.front_runner_idx.is_some());
        assert!(periph.backup_scaffold_idx.is_some());

        // On Oral CNS profile, CNS-penetrant Haloperidol (CHEMBL54) outranks peripherally restricted Lapatinib (CHEMBL554)
        let hal_idx = ds.mols.iter().position(|m| m.id == "CHEMBL54").unwrap();
        let lap_idx = ds.mols.iter().position(|m| m.id == "CHEMBL554").unwrap();
        let (hal_sc, _, _) = evaluate_molecule_scorecard(
            &ds.mols[hal_idx],
            hal_idx,
            CampaignTargetProfile::OralCnsPenetrant,
            50.0,
        );
        let (lap_sc, _, _) = evaluate_molecule_scorecard(
            &ds.mols[lap_idx],
            lap_idx,
            CampaignTargetProfile::OralCnsPenetrant,
            50.0,
        );
        assert!(hal_sc.geometric_desirability > lap_sc.geometric_desirability);

        let dossier = build_lead_campaign_dossier(
            &ds.mols[hal_idx],
            Some(&ds),
            Some(hal_idx),
            CampaignTargetProfile::OralCnsPenetrant,
            PbpkDosingConfig::default(),
        );
        let html = export_lead_dossier_html(&dossier);
        let svg = export_lead_dossier_svg(&dossier);
        let csv = export_campaign_scorecard_csv(&periph);
        assert!(html.contains("<!DOCTYPE html>") && html.contains("<svg"));
        assert!(svg.starts_with("<svg") && svg.contains("Stage-Gate"));
        assert_eq!(csv.lines().count(), 27);
    }

    #[test]
    fn test_v220_evolutionary_de_novo_generator_and_scaffold_hopping() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let erl_idx = ds.mols.iter().position(|m| m.id == "CHEMBL553").unwrap();
        let cfg = DeNovoConfig {
            goal: DeNovoGoalPreset::ScaffoldHop3dRocs,
            generations: 5,
            population_size: 22,
            max_returned: 18,
            ..DeNovoConfig::default()
        };
        let rep = evolve_de_novo_molecules(&ds.mols[erl_idx], Some(&ds), cfg);
        assert!(rep.total_unique_evaluated >= 15);
        assert!(!rep.candidates.is_empty());
        assert!(rep.novel_scaffold_count >= 1);
        assert!(rep.candidates[0].rocs_combo_tanimoto > 0.25);
        assert!(rep.candidates.iter().any(|c| c.pareto_optimal));

        let csv = export_de_novo_candidates_csv(&rep);
        let sdf = format_de_novo_3d_sdf(&rep.candidates);
        assert_eq!(csv.lines().count(), rep.candidates.len() + 1);
        assert!(sdf.contains("V2000") && sdf.contains("ROCS_ComboTanimoto"));
    }

    #[test]
    fn test_v230_pdb_pocket_hydration_pharmacophore_and_screening() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for idx in 0..3 {
            let parsed = parse_pdb_structure(curated_pocket_pdb_text(idx)).unwrap();
            let pocket = analyze_binding_pocket(&parsed, None, 10.5);
            assert!(pocket.volume_angstrom3 >= 85.0);
            assert!(!pocket.hydration_sites.is_empty());
            assert!(!pocket.pharmacophore_points.is_empty());
            if idx == 0 {
                let hits = screen_dataset_against_pocket(&ds, &pocket);
                assert_eq!(hits.len(), 26);
                assert!(hits[0].total_dg_pocket_kcal < -3.0);
                let csv = export_pocket_complementarity_csv(&pocket, &hits);
                let sdf = format_pocket_poses_sdf(&pocket, &hits[..2]);
                assert_eq!(csv.lines().count(), 27);
                assert!(sdf.contains("V2000") && sdf.contains("1M17"));
            }
        }
    }

    #[test]
    fn test_v240_parallel_del_library_enumeration_diversity_and_3d_prefilter() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let parsed = parse_pdb_structure(curated_pocket_pdb_text(0)).unwrap();
        let pocket = analyze_binding_pocket(&parsed, None, 10.5);
        for scheme_idx in [0usize, 2usize, 3usize] {
            let cfg = ParallelLibConfig {
                scheme_idx,
                synthesis_mode: CURATED_LIBRARY_SCHEMES[scheme_idx].default_mode,
                prefilter_mode: LibraryPreFilterMode::PocketAndRocsHybrid,
                reagents_per_cycle: 4,
                target_diverse_plate_size: 10,
                max_3d_scored_pool: 20,
                ..ParallelLibConfig::default()
            };
            let rep = enumerate_parallel_library(&cfg, ds.mols.first(), Some(&pocket), Some(&ds));
            assert!(rep.valid_unique_enumerated >= 12);
            assert_eq!(rep.selected_diverse_plate_count, 10);
            assert!(rep.members[0].composite_priority_score > 20.0);
            let csv = export_parallel_library_csv(&rep, true);
            assert_eq!(csv.lines().count(), 11);
        }
    }

    #[test]
    fn test_v250_dual_pocket_selectivity_watermap_and_mutation_ddg() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for pair_idx in 0..3 {
            let dual = load_curated_selectivity_pair(pair_idx, false, 10.5).unwrap();
            assert!(!dual.differential_waters.is_empty());
            if pair_idx == 0 {
                let rows = screen_dataset_selectivity(&ds, &dual);
                assert_eq!(rows.len(), 26);
                let csv = export_selectivity_screen_csv(&dual, &rows);
                assert_eq!(csv.lines().count(), 27);
                let sdf = format_dual_selectivity_poses_sdf(&dual, &rows, 2);
                assert!(sdf.contains("DDG_SELECTIVITY_KCAL"));
            }
        }
        let egfr_pdb = parse_pdb_structure(curated_pocket_pdb_text(0)).unwrap();
        let mut_scan = scan_pocket_residue_mutations(
            &ds.mols[0],
            &egfr_pdb,
            10.5,
            &["T790M".to_string(), "C797S".to_string()],
        );
        assert!(!mut_scan.entries.is_empty());
        assert!(export_mutation_scan_csv(&mut_scan).lines().count() >= 4);
    }

    #[test]
    fn test_v260_gnm_anm_cryptic_pocket_and_allosteric_strain_coupling() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for sys_idx in 0..3 {
            let net = load_curated_allosteric_system(sys_idx, 10.0).unwrap();
            assert!(!net.cryptic_pockets.is_empty());
            assert!(!net.anm_modes.is_empty());
            if sys_idx == 0 {
                let rows = screen_dataset_allosteric_pocket(&ds, &net, 0);
                assert_eq!(rows.len(), 26);
                let csv = export_allosteric_screen_csv(&net, &rows);
                assert_eq!(csv.lines().count(), 27);
                let sdf = format_allosteric_poses_sdf(&net, &rows, 2);
                assert!(sdf.contains("ORTHOSTERIC_STRAIN_KCAL"));
            }
        }
    }

    #[test]
    fn test_v270_active_learning_gp_rf_expected_improvement_and_batch_acquisition() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for acq in AcquisitionStrategy::ALL {
            let cfg = ActiveLearningConfig {
                acquisition: acq,
                batch_size: 6,
                ..ActiveLearningConfig::default()
            };
            let rep = run_active_learning_campaign(&ds, &cfg);
            assert_eq!(rep.selected_batch_count, 6);
            assert!(rep.validation.conformal_90_scale >= 1.25);
            let csv = export_active_learning_csv(&rep, true);
            assert_eq!(csv.lines().count(), 7);
        }
    }

    #[test]
    fn test_v280_lomap_rbfe_network_bennett_bar_and_mle_cycle_closure() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for topo in RbfeTopologyMode::ALL {
            let cfg = RbfeNetworkConfig {
                series_preset: RbfeSeriesPreset::Egfr1m17Quinazolines,
                topology_mode: topo,
                lambda_windows: 11,
                ..RbfeNetworkConfig::default()
            };
            let rep = run_rbfe_network_campaign(&ds, &cfg);
            assert!(rep.summary.n_nodes >= 8);
            assert!(rep.summary.n_edges >= rep.summary.n_nodes - 1);
            assert!(rep.summary.post_mle_max_cycle_error_kcal < 1e-9);
            if topo == RbfeTopologyMode::CycleClosedLomap {
                assert!(rep.summary.n_cycles >= 2);
            }
            let e_csv = export_rbfe_edges_csv(&rep);
            let n_csv = export_rbfe_nodes_csv(&rep);
            assert_eq!(e_csv.lines().count(), rep.edges.len() + 1);
            assert_eq!(n_csv.lines().count(), rep.nodes.len() + 1);
        }
    }

    #[test]
    fn test_v290_csd_torsion_strain_and_rotatable_bond_locking() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        for ps in [BioactivePoseSource::AutoTargetPocket, BioactivePoseSource::RocsReferenceOverlay] {
            let rep = analyze_torsion_strain_and_locks(&ds.mols[0], Some(&ds), ps, true);
            assert!(rep.rotatable_bond_count >= 2);
            assert_eq!(rep.bond_profiles[0].scan_angles_deg.len(), TORSION_SCAN_BINS);
            assert!(!rep.locked_analogs.is_empty());
        }
        let rows = screen_dataset_torsion_strain(&ds, BioactivePoseSource::AutoTargetPocket);
        assert_eq!(rows.len(), ds.mols.len());
    }

    #[test]
    fn test_v310_multi_round_closed_loop_funnel_custom_pdb_and_sqlite_ledger() {
        let mut ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let n_before = ds.mols.len();
        let tmp = std::env::temp_dir();
        let custom_t_pdb = tmp.join("sar_v310_custom_egfr.pdb");
        let custom_o_pdb = tmp.join("sar_v310_custom_t790m.pdb");
        std::fs::write(&custom_t_pdb, curated_pocket_pdb_text(0)).expect("write target pdb");
        std::fs::write(&custom_o_pdb, curated_offtarget_pdb_text(1)).expect("write offtarget pdb");

        let cfg = DiscoveryPipelineConfig {
            recipe_name: "V310-CustomPDB-2Round".to_string(),
            preset: DiscoveryFunnelPreset::EgfrKinasePanMutant,
            dmta_rounds: 2,
            custom_target_pdb_path: Some(custom_t_pdb.to_string_lossy().to_string()),
            custom_offtarget_pdb_path: Some(custom_o_pdb.to_string_lossy().to_string()),
            include_fbdd_fragment_growing: true,
            include_macrocyclization: true,
            enable_allosteric_coupling_branch: true,
            max_per_murcko_scaffold: 2,
            final_promote_count: 6,
            ..DiscoveryPipelineConfig::default()
        };
        let rep = run_discovery_pipeline(&ds, &cfg);
        assert_eq!(rep.round_history.len(), 2);
        assert!(rep.promoted_survivor_count >= 1 && rep.promoted_survivor_count <= 6);
        assert!(rep.pdb_source_note.contains("Custom Target PDB"));

        let added = ds.merge_imported_molecules(rep.promoted_molecules());
        assert_eq!(ds.mols.len(), n_before + added);

        let db_path = tmp.join("sar_v310_co_persist.sqlite");
        let _ = std::fs::remove_file(&db_path);
        append_pipeline_run_sqlite(&db_path, &rep).expect("append ledger");
        let ws = SarProjectWorkspace::capture(
            "V310-Proj",
            "",
            &ds,
            "",
            "All",
            "Auto",
            "",
            3,
            0.15,
            None,
            None,
        )
        .with_pipeline_config(cfg.clone());
        save_project_sqlite(&db_path, &ws, &ds).expect("save sqlite without wiping ledger");
        let loaded = load_project_sqlite(&db_path).expect("load sqlite");
        assert_eq!(loaded.pipeline_config.unwrap().dmta_rounds, 2);
        let ledger = load_pipeline_ledger_sqlite(&db_path).expect("ledger intact");
        assert_eq!(ledger.len(), 1);

        let _ = std::fs::remove_file(&custom_t_pdb);
        let _ = std::fs::remove_file(&custom_o_pdb);
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_v320_subpocket_fragment_linker_and_superadditivity() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = FragmentLinkerConfig {
            preset: FragmentPairPreset::Egfr1m17HingeAndBackPocket,
            ..FragmentLinkerConfig::default()
        };
        let rep = design_linked_fragments(&cfg, ds.mols.first(), Some(&ds));
        assert_eq!(rep.target_pdb_id, "1M17");
        assert!(rep.candidates.len() >= 14);
        assert!(rep.candidates[0].ddg_superadditivity_kcal < 0.0);
        assert!(export_linked_fragments_csv(&rep).contains("ddg_superadditivity_kcal"));
    }

    #[test]
    fn test_v330_induced_fit_rotamer_repacking_and_boltzmann_ensemble() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = InducedFitConfig::default();
        let rep = score_ligand_induced_fit_ensemble(&ds.mols[0], Some(&ds), &cfg);
        assert_eq!(rep.state_scores.len(), 5);
        assert!(rep.boltzmann_ensemble_dg_kcal < -5.0);
        let rows = screen_dataset_induced_fit_ensemble(&ds, &cfg);
        assert_eq!(rows.len(), ds.mols.len());
        assert!(export_induced_fit_ensemble_csv(&rows).contains("boltzmann_ensemble_dg_kcal"));
    }

    #[test]
    fn test_v380_electrostatics_esp_sigma_hole_chalcogen_and_pi_complementarity() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let vhl_mol = parse_smiles("CC1=C(C2=CC=C(CNC(=O)C3CC(O)CN3C(=O)C(C)(C)C)C=C2)SC=N1")
            .expect("Valid VH032 thiazole SMILES");
        let cfg = ElectrostaticsEspConfig {
            preset: EspTargetPreset::Vhl5nvxThiazoleChalcogenLock,
            enable_xed_multipoles: true,
            generate_esp_designs: true,
            ..ElectrostaticsEspConfig::default()
        };
        let rep = analyze_electrostatics_esp(&vhl_mol, Some(&ds), &cfg);
        assert_eq!(rep.pdb_id, "5NVX");
        assert!(!rep.connolly_vertices.is_empty());
        assert!(!rep.multipole_pseudoatoms.is_empty());
        assert!((-1.0..=1.0).contains(&rep.bauer_mackey_ec_pearson_r));
        assert!(format_esp_multipole_pdb(&rep).contains("XCH"));
    }

    #[test]
    fn test_v370_public_pdb_and_alphafold_plddt_pae_pocket_readiness() {
        let cfg_af = PublicPdbFetchConfig {
            source: PublicStructureSource::OfflineBenchmarkPreset,
            preset: PublicTargetPreset::VhlAfP40337DisorderedTailTrim,
            trim_low_plddt_cutoff: 55.0,
            ..PublicPdbFetchConfig::default()
        };
        let rep = fetch_and_assess_public_structure(&cfg_af).expect("Offline AFDB VHL audit");
        assert!(rep.is_alphafold_plddt_scale);
        assert!(rep.trimmed_removed_atoms > 0);
        assert_eq!(rep.verdict, PocketReadinessVerdict::TrimDisorderedLoopsBeforeDocking);

        let cfg_rcsb = PublicPdbFetchConfig {
            source: PublicStructureSource::OfflineBenchmarkPreset,
            preset: PublicTargetPreset::EgfrRcsb1m17ExperimentalHolo,
            ..PublicPdbFetchConfig::default()
        };
        let rep_holo = fetch_and_assess_public_structure(&cfg_rcsb).expect("Offline RCSB 1M17 audit");
        assert!(!rep_holo.extracted_ligands.is_empty());
        assert!(export_public_structure_confidence_csv(&rep_holo).contains("effective_plddt"));
    }

    #[test]
    fn test_v360_foldit_sculpt_tug_wiggle_shake_bands_and_drugit_puzzles() {
        let mut sess = new_sculpt_session(
            SculptPuzzlePreset::Vhl5nvxDepeptidizationChallenge,
            None,
            None,
        );
        let start_pts = sess.score.foldit_score_pts;
        let _ = auto_band_polar_contacts(&mut sess);
        run_blue_fuse_macro(&mut sess);
        assert!(sess.score.foldit_score_pts >= start_pts - 1e-3);
        assert!(format_sculpted_complex_pdb(&sess).contains("5NVX"));
        assert!(format_sculpted_ligand_sdf(&sess).contains("Foldit_Score_Pts"));
    }

    #[test]
    fn test_v350_watermap_ifst_hydration_hotspots_and_rgroup_vector_designer() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = WaterMapConfig {
            preset: WaterMapPreset::FactorXa1ezqS1S4Hotspots,
            design_rgroup_vectors: true,
            ..WaterMapConfig::default()
        };
        let rep = analyze_watermap_solvation(&ds.mols[0], Some(&ds), &cfg);
        assert_eq!(rep.pdb_id, "1EZQ");
        assert!(!rep.hydration_sites.is_empty());
        assert!(export_watermap_sites_csv(&rep).contains("dh_excess_kcal"));
        assert!(format_watermap_annotated_pdb(&rep).contains("WATERMAP"));
    }

    #[test]
    fn test_v340_hbond_network_his_asp_tautomer_and_bridge_water_optimizer() {
        let ds = DatasetAnalysis::from_molecules(load_benchmark_26(), 0.30, 6.0);
        let cfg = HBondNetworkConfig {
            preset: HBondNetworkPreset::Egfr1m17Thr790WaterBridge,
            target_ph: 7.4,
            ..HBondNetworkConfig::default()
        };
        let rep = optimize_pocket_hbond_network(&ds.mols[0], Some(&ds), &cfg);
        assert_eq!(rep.pdb_id, "1M17");
        assert!(rep.optimized_network_dg_kcal <= rep.naive_network_dg_kcal + 1e-6);
        assert!(!rep.nodes.is_empty());
        assert!(export_hbond_network_nodes_csv(&rep).contains("optimal_state"));
    }

    #[test]
    fn test_gpu_pod_struct_wgsl_alignment() {
        assert_eq!(std::mem::size_of::<GpuCameraUniform>(), 48);
        assert_eq!(std::mem::size_of::<GpuNodeInstance>(), 32);
        assert_eq!(std::mem::size_of::<GpuEdgeInstance>(), 32);
    }
}
