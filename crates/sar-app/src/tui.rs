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

//! Standalone Zero-Dependency Pure-Std ANSI Terminal Console (`sar-atlas tui`, `v3.8.0`)
//!
//! Renders interactive colorized Unicode box-drawing tables across 23 cheminformatics views
//! (`Compounds & QED/MPO`, `SALI Cliffs`, `Matched Molecular Pairs`, `Scaffolds & PAINS`,
//! `Free-Wilson SAR Matrix`, and `Oral PBPK & Brain Kp,uu`) using only `std::io` (`stdin`/`stdout`)
//! and standard ANSI VT100 escape codes — zero third-party TUI crate dependencies and zero `unsafe`.

use sar_core::{
    analyze_electrostatics_esp, screen_dataset_electrostatics_esp, ElectrostaticsEspConfig,
    EspTargetPreset, fetch_and_assess_public_structure, PublicPdbFetchConfig, PublicStructureSource,
    PublicTargetPreset, auto_band_polar_contacts, new_sculpt_session, run_blue_fuse_macro,
    SculptPuzzlePreset, analyze_watermap_solvation, screen_dataset_watermap_solvation, WaterMapConfig,
    WaterMapPreset, optimize_pocket_hbond_network, screen_dataset_hbond_network, HBondNetworkConfig,
    HBondNetworkPreset, score_ligand_induced_fit_ensemble, screen_dataset_induced_fit_ensemble,
    InducedFitConfig, InducedFitPreset, design_linked_fragments, FragmentLinkerConfig, FragmentPairPreset,
    run_discovery_pipeline, DiscoveryFunnelPreset, DiscoveryPipelineConfig, analyze_torsion_strain_and_locks,
    BioactivePoseSource, TorsionQualityBand, run_rbfe_network_campaign, RbfeConvergenceTier,
    RbfeNetworkConfig, RbfeSeriesPreset, RbfeTopologyMode, run_active_learning_campaign, AcquisitionStrategy,
    ActiveLearningConfig, load_curated_allosteric_system, screen_dataset_allosteric_pocket,
    CURATED_ALLOSTERIC_SYSTEMS, load_curated_selectivity_pair, scan_pocket_residue_mutations,
    screen_dataset_selectivity, CURATED_SELECTIVITY_PAIRS, enumerate_parallel_library, LibraryPreFilterMode,
    ParallelLibConfig, CURATED_LIBRARY_SCHEMES, analyze_binding_pocket, curated_pocket_pdb_text,
    parse_pdb_structure, screen_dataset_against_pocket, CURATED_PDB_POCKETS, evolve_de_novo_molecules,
    DeNovoConfig, DeNovoGoalPreset, build_lead_campaign_dossier, compile_smarts, evaluate_campaign_scorecard,
    export_lead_dossier_html, simulate_oral_pbpk, CampaignTargetProfile, DatasetAnalysis, PbpkDosingConfig,
    ProgressionTier, SmartsMatchResult,
};
use std::io::{self, BufRead, Write};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[38;5;81m";
const EMERALD: &str = "\x1b[38;5;48m";
const AMBER: &str = "\x1b[38;5;221m";
const CORAL: &str = "\x1b[38;5;203m";
const VIOLET: &str = "\x1b[38;5;177m";
const SLATE: &str = "\x1b[38;5;246m";
const CLEAR_HOME: &str = "\x1b[2J\x1b[H";

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConsoleTab {
    Compounds,
    SaliCliffs,
    Mmps,
    ScaffoldsAndAlerts,
    FreeWilsonMatrix,
    OralPbpkAndKpuu,
    LeadScorecard,
    DeNovoEvolution,
    PocketCavity,
    ParallelLibrary,
    SelectivityProfiler,
    AllostericCryptic,
    ActiveLearningQsar,
    RbfeLomapBar,
    TorsionStrainLock,
    DiscoveryFunnel,
    FragmentLinker,
    InducedFitEnsemble,
    HBondNetwork,
    WaterMapSolvation,
    FolditSculpt,
    PublicPdbReadiness,
    ElectrostaticsEsp,
}

impl ConsoleTab {
    const ALL: [Self; 23] = [
        Self::Compounds,
        Self::SaliCliffs,
        Self::Mmps,
        Self::ScaffoldsAndAlerts,
        Self::FreeWilsonMatrix,
        Self::OralPbpkAndKpuu,
        Self::LeadScorecard,
        Self::DeNovoEvolution,
        Self::PocketCavity,
        Self::ParallelLibrary,
        Self::SelectivityProfiler,
        Self::AllostericCryptic,
        Self::ActiveLearningQsar,
        Self::RbfeLomapBar,
        Self::TorsionStrainLock,
        Self::DiscoveryFunnel,
        Self::FragmentLinker,
        Self::InducedFitEnsemble,
        Self::HBondNetwork,
        Self::WaterMapSolvation,
        Self::FolditSculpt,
        Self::PublicPdbReadiness,
        Self::ElectrostaticsEsp,
    ];

    fn title(self) -> &'static str {
        match self {
            Self::Compounds => "1: Compounds & QED/MPO",
            Self::SaliCliffs => "2: SALI Cliffs",
            Self::Mmps => "3: MMPs",
            Self::ScaffoldsAndAlerts => "4: Scaffolds & PAINS",
            Self::FreeWilsonMatrix => "5: Free-Wilson Matrix",
            Self::OralPbpkAndKpuu => "6: Oral PBPK & Kp,uu",
            Self::LeadScorecard => "7: Lead Scorecard",
            Self::DeNovoEvolution => "8: De Novo GA",
            Self::PocketCavity => "9: PDB Pocket",
            Self::ParallelLibrary => "10: DEL & Parallel Lib",
            Self::SelectivityProfiler => "11: Selectivity & ddG",
            Self::AllostericCryptic => "12: Cryptic & Allostery",
            Self::ActiveLearningQsar => "13: Active-Learn QSAR",
            Self::RbfeLomapBar => "14: Lomap RBFE & BAR",
            Self::TorsionStrainLock => "15: Torsion Strain & Lock",
            Self::DiscoveryFunnel => "16: Discovery Funnel",
            Self::FragmentLinker => "17: Fragment Linker",
            Self::InducedFitEnsemble => "18: Induced-Fit & Ensemble",
            Self::HBondNetwork => "19: H-Bond & Bridge Water",
            Self::WaterMapSolvation => "20: WaterMap & R-Vectors",
            Self::FolditSculpt => "21: Foldit Sculpt & Puzzles",
            Self::PublicPdbReadiness => "22: Public PDB & AFDB Readiness",
            Self::ElectrostaticsEsp => "23: 3D ESP & Sigma-Hole",
        }
    }
}

struct ConsoleState {
    tab_idx: usize,
    page: usize,
    page_size: usize,
    selected_mol: usize,
    pocket_preset_idx: usize,
    lib_scheme_idx: usize,
    sel_pair_idx: usize,
    allo_sys_idx: usize,
    al_acq_idx: usize,
    rbfe_series_idx: usize,
    torsion_pose_idx: usize,
    funnel_preset_idx: usize,
    frag_link_preset_idx: usize,
    ifd_preset_idx: usize,
    hbn_preset_idx: usize,
    wm_preset_idx: usize,
    sculpt_preset_idx: usize,
    pub_pdb_preset_idx: usize,
    esp_preset_idx: usize,
    hide_pains: bool,
    pareto_only: bool,
    smarts_query: String,
    smarts_error: Option<String>,
    smarts_results: Vec<SmartsMatchResult>,
    status_msg: String,
}

impl ConsoleState {
    fn recompute_smarts(&mut self, ds: &DatasetAnalysis) {
        let trimmed = self.smarts_query.trim();
        if trimmed.is_empty() {
            self.smarts_error = None;
            self.smarts_results.clear();
            return;
        }
        match compile_smarts(trimmed) {
            Ok(q) => {
                self.smarts_error = None;
                self.smarts_results = ds.evaluate_smarts(&q);
                let hits = self.smarts_results.iter().filter(|r| r.matched).count();
                self.status_msg = format!(
                    "SMARTS '{}' matched {}/{} compounds.",
                    trimmed,
                    hits,
                    ds.mols.len()
                );
            }
            Err(err) => {
                self.smarts_error = Some(err.to_string());
                self.smarts_results.clear();
            }
        }
    }

    fn visible_molecule_indices(&self, ds: &DatasetAnalysis) -> Vec<usize> {
        let smarts_active = !self.smarts_query.trim().is_empty() && self.smarts_error.is_none();
        (0..ds.mols.len())
            .filter(|&i| {
                if self.hide_pains && !ds.alerts[i].is_empty() {
                    return false;
                }
                if self.pareto_only && !ds.profiles[i].pareto_optimal {
                    return false;
                }
                if smarts_active {
                    let hit = self
                        .smarts_results
                        .get(i)
                        .map(|r| r.matched)
                        .unwrap_or(false);
                    if !hit {
                        return false;
                    }
                }
                true
            })
            .collect()
    }
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        format!("{:<width$}", s, width = max_chars)
    } else if max_chars > 2 {
        let mut cut: String = chars[..max_chars - 2].iter().collect();
        cut.push_str("..");
        cut
    } else {
        chars[..max_chars].iter().collect()
    }
}

/// Runs the standalone pure-std interactive ANSI terminal console.
pub fn run_tui(ds: &DatasetAnalysis) -> io::Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut out = io::stdout();

    let mut state = ConsoleState {
        tab_idx: 0,
        page: 0,
        page_size: 14,
        selected_mol: 0,
        pocket_preset_idx: 0,
        lib_scheme_idx: 0,
        sel_pair_idx: 0,
        allo_sys_idx: 0,
        al_acq_idx: 0,
        rbfe_series_idx: 1,
        torsion_pose_idx: 0,
        funnel_preset_idx: 1,
        frag_link_preset_idx: 0,
        ifd_preset_idx: 0,
        hbn_preset_idx: 0,
        wm_preset_idx: 0,
        sculpt_preset_idx: 0,
        pub_pdb_preset_idx: 0,
        esp_preset_idx: 0,
        hide_pains: false,
        pareto_only: false,
        smarts_query: String::new(),
        smarts_error: None,
        smarts_results: Vec::new(),
        status_msg: "Type 1..23 to switch view, e to cycle 3D ESP & Sigma-Hole target, u to cycle Public RCSB/AFDB target, z to cycle Foldit/Drugit 3D puzzle, m to cycle WaterMap solvation active site, t to cycle H-Bond & Bridge-Water active site, i to cycle Induced-Fit receptor ensemble, j to cycle Sub-Pocket Fragment Linker pair, f to cycle Discovery Funnel campaign, y to cycle Torsion pose reference, r to cycle Lomap RBFE series, b to cycle Active-Learning acquisition, k to cycle Cryptic/Allosteric system, x to cycle Selectivity pair, l to cycle DEL/Parallel scheme, w to cycle PDB pocket, g to evolve de novo analogs, h for HTML dossier, /<smarts> to filter, s <id> to inspect, n/p to page, or q to quit.".to_string(),
    };

    let mut line_buf = String::new();
    loop {
        render_ansi_frame(&mut out, ds, &state)?;
        write!(
            out,
            "\n{BOLD}{CYAN}sar-atlas>{RESET} "
        )?;
        out.flush()?;

        line_buf.clear();
        let n_read = reader.read_line(&mut line_buf)?;
        if n_read == 0 {
            // EOF on stdin (e.g. piped headless run)
            writeln!(out)?;
            break;
        }

        let cmd = line_buf.trim();
        if cmd.is_empty() || cmd.eq_ignore_ascii_case("n") || cmd.eq_ignore_ascii_case("next") {
            state.page += 1;
            continue;
        }
        if cmd.eq_ignore_ascii_case("q")
            || cmd.eq_ignore_ascii_case("quit")
            || cmd.eq_ignore_ascii_case("exit")
        {
            break;
        }
        if cmd.eq_ignore_ascii_case("p") || cmd.eq_ignore_ascii_case("prev") {
            state.page = state.page.saturating_sub(1);
            continue;
        }
        if let Ok(num) = cmd.parse::<usize>() {
            if (1..=ConsoleTab::ALL.len()).contains(&num) {
                state.tab_idx = num - 1;
                state.page = 0;
                state.status_msg = format!("Switched to view {}.", ConsoleTab::ALL[state.tab_idx].title());
                continue;
            }
        }
        if cmd.eq_ignore_ascii_case("a") || cmd.eq_ignore_ascii_case("pains") {
            state.hide_pains = !state.hide_pains;
            state.page = 0;
            state.status_msg = format!("Hide PAINS/Structural Alerts: {}", state.hide_pains);
            continue;
        }
        if cmd.eq_ignore_ascii_case("o") || cmd.eq_ignore_ascii_case("pareto") {
            state.pareto_only = !state.pareto_only;
            state.page = 0;
            state.status_msg = format!("Pareto-Optimal Leads Only: {}", state.pareto_only);
            continue;
        }
        if cmd.eq_ignore_ascii_case("c") || cmd.eq_ignore_ascii_case("clear") {
            state.smarts_query.clear();
            state.smarts_error = None;
            state.smarts_results.clear();
            state.hide_pains = false;
            state.pareto_only = false;
            state.page = 0;
            state.status_msg = "Cleared SMARTS and property filters.".to_string();
            continue;
        }
        if let Some(pat) = cmd.strip_prefix('/') {
            state.smarts_query = pat.trim().to_string();
            state.page = 0;
            state.recompute_smarts(ds);
            continue;
        }
        if let Some(rest) = cmd.strip_prefix("s ").or_else(|| cmd.strip_prefix("select ")) {
            let key = rest.trim();
            if let Ok(ridx) = key.parse::<usize>() {
                if ridx >= 1 && ridx <= ds.mols.len() {
                    state.selected_mol = ridx - 1;
                    state.status_msg = format!(
                        "Selected #{} {} ({}).",
                        ridx, ds.mols[state.selected_mol].name, ds.mols[state.selected_mol].id
                    );
                }
            } else if let Some(pos) = ds
                .mols
                .iter()
                .position(|m| m.id.eq_ignore_ascii_case(key) || m.name.to_ascii_lowercase().contains(&key.to_ascii_lowercase()))
            {
                state.selected_mol = pos;
                state.status_msg = format!(
                    "Selected #{} {} ({}).",
                    pos + 1,
                    ds.mols[pos].name,
                    ds.mols[pos].id
                );
            } else {
                state.status_msg = format!("No compound matching '{key}'.");
            }
            continue;
        }

        if cmd.eq_ignore_ascii_case("23") {
            state.tab_idx = 22;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("e")
            || cmd.eq_ignore_ascii_case("esp")
            || cmd.eq_ignore_ascii_case("xed")
            || cmd.eq_ignore_ascii_case("halogen")
        {
            state.tab_idx = 22;
            state.esp_preset_idx = (state.esp_preset_idx + 1) % 6;
            let pr = EspTargetPreset::ALL[state.esp_preset_idx];
            state.status_msg = format!(
                "Switched 3D ESP, Sigma-Hole & Pi-Multipole target to {} ({}).",
                pr.code(),
                pr.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("22") {
            state.tab_idx = 21;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("u")
            || cmd.eq_ignore_ascii_case("rcsb")
            || cmd.eq_ignore_ascii_case("afdb")
            || cmd.eq_ignore_ascii_case("uniprot")
        {
            state.tab_idx = 21;
            state.pub_pdb_preset_idx = (state.pub_pdb_preset_idx + 1) % 5;
            let pr = PublicTargetPreset::ALL[state.pub_pdb_preset_idx];
            state.status_msg = format!(
                "Switched Public RCSB PDB / AlphaFold DB target to {} ({}).",
                pr.code(),
                pr.short_label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("21") {
            state.tab_idx = 20;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("z")
            || cmd.eq_ignore_ascii_case("sculpt")
            || cmd.eq_ignore_ascii_case("foldit")
            || cmd.eq_ignore_ascii_case("drugit")
        {
            state.tab_idx = 20;
            state.sculpt_preset_idx = (state.sculpt_preset_idx + 1) % 5;
            let pr = SculptPuzzlePreset::ALL[state.sculpt_preset_idx];
            state.status_msg = format!(
                "Switched Foldit/Drugit 3D Med-Chem Challenge to {} ({}).",
                pr.code(),
                pr.title()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("20") {
            state.tab_idx = 19;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("m") || cmd.eq_ignore_ascii_case("watermap") || cmd.eq_ignore_ascii_case("ifst") {
            state.tab_idx = 19;
            state.wm_preset_idx = (state.wm_preset_idx + 1) % 5;
            let pr = WaterMapPreset::ALL[state.wm_preset_idx];
            state.status_msg = format!(
                "Switched WaterMap / IFST solvation active site to {} ({}).",
                pr.code(),
                pr.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("19") {
            state.tab_idx = 18;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("t") || cmd.eq_ignore_ascii_case("hbn") || cmd.eq_ignore_ascii_case("hbond") {
            state.tab_idx = 18;
            state.hbn_preset_idx = (state.hbn_preset_idx + 1) % 4;
            let pr = HBondNetworkPreset::ALL[state.hbn_preset_idx];
            state.status_msg = format!(
                "Switched His/Asp/Glu & Bridge-Water H-Bond Network target to {} ({}).",
                pr.code(),
                pr.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("18") {
            state.tab_idx = 17;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("i") || cmd.eq_ignore_ascii_case("ifd") || cmd.eq_ignore_ascii_case("ensemble") {
            state.tab_idx = 17;
            state.ifd_preset_idx = (state.ifd_preset_idx + 1) % 4;
            let pr = InducedFitPreset::ALL[state.ifd_preset_idx];
            state.status_msg = format!(
                "Switched Induced-Fit & Boltzmann Ensemble target to {} ({}).",
                pr.code(),
                pr.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("17") {
            state.tab_idx = 16;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("j") || cmd.eq_ignore_ascii_case("link") || cmd.eq_ignore_ascii_case("delinker") {
            state.tab_idx = 16;
            state.frag_link_preset_idx = (state.frag_link_preset_idx + 1) % FragmentPairPreset::ALL.len();
            let pr = FragmentPairPreset::ALL[state.frag_link_preset_idx];
            state.status_msg = format!(
                "Switched Sub-Pocket Fragment Linker pair to {} ({}).",
                pr.code(),
                pr.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("16") {
            state.tab_idx = 15;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("f") || cmd.eq_ignore_ascii_case("funnel") || cmd.eq_ignore_ascii_case("pipe") {
            state.tab_idx = 15;
            state.funnel_preset_idx = (state.funnel_preset_idx + 1) % DiscoveryFunnelPreset::ALL.len();
            let fp = DiscoveryFunnelPreset::ALL[state.funnel_preset_idx];
            state.status_msg = format!(
                "Switched 6-Stage Discovery Funnel campaign to {} ({}).",
                fp.code(),
                fp.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("15") {
            state.tab_idx = 14;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("y") || cmd.eq_ignore_ascii_case("tor") || cmd.eq_ignore_ascii_case("strain") {
            state.tab_idx = 14;
            state.torsion_pose_idx = (state.torsion_pose_idx + 1) % BioactivePoseSource::ALL.len();
            let ps = BioactivePoseSource::ALL[state.torsion_pose_idx];
            state.status_msg = format!(
                "Switched Torsion Strain bioactive pose reference to {}.",
                ps.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("14") {
            state.tab_idx = 13;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("r") || cmd.eq_ignore_ascii_case("rbfe") || cmd.eq_ignore_ascii_case("lomap") {
            state.tab_idx = 13;
            state.rbfe_series_idx = (state.rbfe_series_idx + 1) % RbfeSeriesPreset::ALL.len();
            let sp = RbfeSeriesPreset::ALL[state.rbfe_series_idx];
            state.status_msg = format!(
                "Switched Lomap RBFE congeneric series to {} ({}).",
                sp.code(),
                sp.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("13") {
            state.tab_idx = 12;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("b") || cmd.eq_ignore_ascii_case("al") || cmd.eq_ignore_ascii_case("bo") {
            state.tab_idx = 12;
            state.al_acq_idx = (state.al_acq_idx + 1) % AcquisitionStrategy::ALL.len();
            let acq = AcquisitionStrategy::ALL[state.al_acq_idx];
            state.status_msg = format!(
                "Switched Active-Learning acquisition strategy to {}.",
                acq.label()
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("12") {
            state.tab_idx = 11;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("k") || cmd.eq_ignore_ascii_case("allo") || cmd.eq_ignore_ascii_case("cryptic") {
            state.tab_idx = 11;
            state.allo_sys_idx = (state.allo_sys_idx + 1) % CURATED_ALLOSTERIC_SYSTEMS.len();
            let sys = &CURATED_ALLOSTERIC_SYSTEMS[state.allo_sys_idx];
            state.status_msg = format!(
                "Switched GNM/ANM cryptic-allosteric system to {} ({}).",
                sys.code, sys.title
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("11") {
            state.tab_idx = 10;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("x") || cmd.eq_ignore_ascii_case("sel") || cmd.eq_ignore_ascii_case("selectivity") {
            state.tab_idx = 10;
            state.sel_pair_idx = (state.sel_pair_idx + 1) % CURATED_SELECTIVITY_PAIRS.len();
            let pr = &CURATED_SELECTIVITY_PAIRS[state.sel_pair_idx];
            state.status_msg = format!(
                "Switched dual-pocket selectivity pair to {} ({}).",
                pr.code, pr.title
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("10") {
            state.tab_idx = 9;
            state.page = 0;
            continue;
        }
        if cmd.eq_ignore_ascii_case("l") || cmd.eq_ignore_ascii_case("del") || cmd.eq_ignore_ascii_case("lib") {
            state.tab_idx = 9;
            state.lib_scheme_idx = (state.lib_scheme_idx + 1) % CURATED_LIBRARY_SCHEMES.len();
            let sc = &CURATED_LIBRARY_SCHEMES[state.lib_scheme_idx];
            state.status_msg = format!(
                "Switched combinatorial scheme to {} ({}).",
                sc.code, sc.title
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("w") || cmd.eq_ignore_ascii_case("pocket") {
            state.tab_idx = 8;
            state.pocket_preset_idx = (state.pocket_preset_idx + 1) % CURATED_PDB_POCKETS.len();
            let pr = &CURATED_PDB_POCKETS[state.pocket_preset_idx];
            state.status_msg = format!(
                "Switched PDB receptor pocket to {} ({} — {}).",
                pr.code, pr.target_name, pr.ligand_name
            );
            continue;
        }
        if cmd.eq_ignore_ascii_case("g") || cmd.eq_ignore_ascii_case("evolve") || cmd.eq_ignore_ascii_case("denovo") {
            state.tab_idx = 7;
            state.page = 0;
            if !ds.mols.is_empty() {
                let idx = state.selected_mol.min(ds.mols.len() - 1);
                state.status_msg = format!(
                    "Evolved multi-generation de novo analogs from seed {} ({}).",
                    ds.mols[idx].name, ds.mols[idx].id
                );
            }
            continue;
        }
        if cmd.eq_ignore_ascii_case("h") || cmd.eq_ignore_ascii_case("html") || cmd.eq_ignore_ascii_case("dossier") {
            if !ds.mols.is_empty() {
                let idx = state.selected_mol.min(ds.mols.len() - 1);
                let m = &ds.mols[idx];
                let doss = build_lead_campaign_dossier(
                    m,
                    Some(ds),
                    Some(idx),
                    CampaignTargetProfile::OralPeripheral,
                    PbpkDosingConfig::default(),
                );
                let out_path = format!("dossier_{}.html", m.id.to_ascii_lowercase());
                match std::fs::write(&out_path, export_lead_dossier_html(&doss)) {
                    Ok(_) => {
                        state.status_msg = format!(
                            "Exported standalone HTML/SVG dossier for {} -> {}",
                            m.name, out_path
                        );
                    }
                    Err(e) => {
                        state.status_msg = format!("HTML export error: {e}");
                    }
                }
            }
            continue;
        }

        state.status_msg = format!(
            "Unrecognized command '{cmd}'. Commands: 1..23 (views), e (ESP/Sigma-Hole), u (RCSB/AFDB), z (Foldit Sculpt), m (WaterMap), t (H-Bond Net), i (Induced-Fit), j (Fragment Linker), f (Discovery Funnel), y (Torsion Strain), r (Lomap RBFE), b (Active-Learn), k (Allosteric), x (Selectivity), l (DEL/Lib), w (PDB Pocket), g (De Novo GA), h (HTML dossier), /<smarts>, c (clear), a (PAINS), o (Pareto), s <id>, n/p (page), q (quit)."
        );
    }

    Ok(())
}

fn render_ansi_frame(
    out: &mut impl Write,
    ds: &DatasetAnalysis,
    state: &ConsoleState,
) -> io::Result<()> {
    let vis = state.visible_molecule_indices(ds);
    write!(out, "{CLEAR_HOME}")?;

    // Top Banner
    writeln!(
        out,
        "{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗{RESET}"
    )?;
    writeln!(
        out,
        "{BOLD}{CYAN}║{RESET} {BOLD}SAR-Atlas v3.8.0 Native ANSI Terminal Console{RESET} {SLATE}(Zero External TUI Crates — Pure Rust std::io){RESET}               {BOLD}{CYAN}║{RESET}"
    )?;
    write!(out, "{BOLD}{CYAN}║{RESET} ")?;
    for (ti, tab) in ConsoleTab::ALL.iter().enumerate() {
        if ti == state.tab_idx {
            write!(out, "{BOLD}{EMERALD}[{}]{RESET}  ", tab.title())?;
        } else {
            write!(out, "{SLATE}{}{RESET}  ", tab.title())?;
        }
    }
    writeln!(out)?;
    writeln!(
        out,
        "{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{RESET}"
    )?;

    // Filter & Status Strip
    let sm_badge = if let Some(ref err) = state.smarts_error {
        format!("{CORAL}SMARTS Error: {err}{RESET}")
    } else if !state.smarts_query.is_empty() {
        let hits = state.smarts_results.iter().filter(|r| r.matched).count();
        format!(
            "{EMERALD}SMARTS: '{}' ({hits}/{} hits){RESET}",
            state.smarts_query,
            ds.mols.len()
        )
    } else {
        format!("{SLATE}SMARTS: <none> (type /pattern to filter){RESET}")
    };
    writeln!(
        out,
        "  Visible: {BOLD}{}/{}{RESET}  |  Hide PAINS [a]: {BOLD}{}{RESET}  |  Pareto [o]: {BOLD}{}{RESET}  |  {sm_badge}",
        vis.len(),
        ds.mols.len(),
        state.hide_pains,
        state.pareto_only
    )?;
    writeln!(
        out,
        "  {DIM}──────────────────────────────────────────────────────────────────────────────────────────────────────────────{RESET}"
    )?;

    match ConsoleTab::ALL[state.tab_idx] {
        ConsoleTab::Compounds => {
            let total_pages = vis.len().div_ceil(state.page_size).max(1);
            let pg = state.page.min(total_pages - 1);
            let start = pg * state.page_size;
            let end = (start + state.page_size).min(vis.len());

            writeln!(
                out,
                "  {BOLD}#    ID           Name                   Target         pAct    MW     logD7.4  TPSA   QED    MPO   SAScore  Alerts{RESET}  (Page {}/{})",
                pg + 1,
                total_pages
            )?;
            for &mi in &vis[start..end] {
                let m = &ds.mols[mi];
                let p = &ds.profiles[mi];
                let pk = &ds.pka_74[mi];
                let sa = &ds.synth_access[mi];
                let al = ds.alerts[mi].len();
                let mark = if mi == state.selected_mol {
                    format!("{EMERALD}>{RESET}")
                } else {
                    " ".to_string()
                };
                let al_str = if al == 0 {
                    format!("{EMERALD}Clean{RESET}")
                } else {
                    format!("{CORAL}{al} Alert{RESET}")
                };
                writeln!(
                    out,
                    " {mark}{:<4} {} {} {} {:>6.2} {:>6.1} {:>7.2} {:>6.1} {:>5.2} {:>5.2} {:>7.2}  {}",
                    mi + 1,
                    truncate_str(&m.id, 12),
                    truncate_str(&m.name, 22),
                    truncate_str(&m.target, 14),
                    m.pact,
                    p.mw,
                    pk.microstate_logd_74,
                    p.tpsa,
                    p.qed_mean,
                    p.cns_mpo,
                    sa.sa_score,
                    al_str
                )?;
            }
        }
        ConsoleTab::SaliCliffs => {
            let cliffs: Vec<_> = ds.edges.iter().filter(|e| e.same_target).collect();
            let total_pages = cliffs.len().div_ceil(state.page_size).max(1);
            let pg = state.page.min(total_pages - 1);
            let start = pg * state.page_size;
            let end = (start + state.page_size).min(cliffs.len());

            writeln!(
                out,
                "  {BOLD}Rank  Compound A             Compound B             Target          Tanimoto  dpAct    SALI Index{RESET}  (Page {}/{})",
                pg + 1,
                total_pages
            )?;
            for (rk, e) in cliffs[start..end].iter().enumerate() {
                let ma = &ds.mols[e.mol_a];
                let mb = &ds.mols[e.mol_b];
                let col = if e.is_cliff { CORAL } else { CYAN };
                writeln!(
                    out,
                    "  #{:<4} {} {} {}   {:.3}    {:>5.2}   {col}{:>8.1}{RESET}",
                    start + rk + 1,
                    truncate_str(&ma.name, 22),
                    truncate_str(&mb.name, 22),
                    truncate_str(&ma.target, 14),
                    e.tanimoto,
                    e.delta_pact,
                    e.sali
                )?;
            }
        }
        ConsoleTab::Mmps => {
            let total_pages = ds.mmps.len().div_ceil(state.page_size).max(1);
            let pg = state.page.min(total_pages - 1);
            let start = pg * state.page_size;
            let end = (start + state.page_size).min(ds.mmps.len());

            writeln!(
                out,
                "  {BOLD}Rank  From Lead              To Lead                R-Group Transform               dpAct    dLLE{RESET}  (Page {}/{})",
                pg + 1,
                total_pages
            )?;
            for (rk, mmp) in ds.mmps[start..end].iter().enumerate() {
                let ma = &ds.mols[mmp.mol_from];
                let mb = &ds.mols[mmp.mol_to];
                let xform = &mmp.transform;
                writeln!(
                    out,
                    "  #{:<4} {} {} {} {:>+6.2}  {:>+6.2}",
                    start + rk + 1,
                    truncate_str(&ma.name, 22),
                    truncate_str(&mb.name, 22),
                    truncate_str(&xform, 31),
                    mmp.delta_pact,
                    mmp.delta_lle
                )?;
            }
        }
        ConsoleTab::ScaffoldsAndAlerts => {
            let total_pages = ds.scaffolds.len().div_ceil(state.page_size).max(1);
            let pg = state.page.min(total_pages - 1);
            let start = pg * state.page_size;
            let end = (start + state.page_size).min(ds.scaffolds.len());

            writeln!(
                out,
                "  {BOLD}Scaffold Topology                               Members  Rings  Mean pAct  Max pAct  Spread{RESET}  (Page {}/{})",
                pg + 1,
                total_pages
            )?;
            for sc in &ds.scaffolds[start..end] {
                writeln!(
                    out,
                    "  {}  {:>5}    {:>3}     {:>6.2}    {:>6.2}   {:>6.2}",
                    truncate_str(&sc.framework_label, 46),
                    sc.member_mols.len(),
                    sc.ring_count,
                    sc.mean_pact,
                    sc.max_pact,
                    sc.pact_spread
                )?;
            }
        }
        ConsoleTab::FreeWilsonMatrix => {
            if let Some(ref fw) = ds.default_sar_matrix {
                writeln!(
                    out,
                    "  {BOLD}Core:{RESET} {} ({})  |  Matched Series: {BOLD}{}{RESET}  |  Intercept pAct0 = {BOLD}{:.2}{RESET}  |  R^2 = {BOLD}{:.3}{RESET} (RMSE {:.3})",
                    fw.core_name,
                    fw.core_smarts,
                    fw.rows.len(),
                    fw.intercept_pact,
                    fw.r2,
                    fw.rmse
                )?;
                writeln!(
                    out,
                    "  {BOLD}Slot   Substituent SMILES             Count   Ridge Weight (dpAct){RESET}"
                )?;
                for eff in fw.effects.iter().take(state.page_size) {
                    let col = if eff.coeff_pact >= 0.15 {
                        EMERALD
                    } else if eff.coeff_pact <= -0.15 {
                        CORAL
                    } else {
                        SLATE
                    };
                    writeln!(
                        out,
                        "  R{:<4}  {}  {:>4}    {col}{:>+8.3}{RESET}",
                        eff.slot_index + 1,
                        truncate_str(&eff.substituent, 28),
                        eff.count,
                        eff.coeff_pact
                    )?;
                }
            } else {
                writeln!(out, "  No multi-member congeneric scaffold series detected.")?;
            }
        }
        ConsoleTab::OralPbpkAndKpuu => {
            let total_pages = vis.len().div_ceil(state.page_size).max(1);
            let pg = state.page.min(total_pages - 1);
            let start = pg * state.page_size;
            let end = (start + state.page_size).min(vis.len());

            writeln!(
                out,
                "  {BOLD}ID           Name                   F_oral%  (Fa/Fg/Fh)     Vss(L/kg)  t1/2(h)  Kp,uu,br  CNS Class      Brain RO%{RESET}  (50 mg Fasted, Page {}/{})",
                pg + 1,
                total_pages
            )?;
            for &mi in &vis[start..end] {
                let m = &ds.mols[mi];
                let rep = simulate_oral_pbpk(m, PbpkDosingConfig::default());
                let cns_col = if rep.kp_uu_brain >= 0.30 {
                    EMERALD
                } else if rep.kp_uu_brain >= 0.10 {
                    AMBER
                } else {
                    CYAN
                };
                writeln!(
                    out,
                    "  {} {}  {:>5.1}%  ({:>2.0}/{:>2.0}/{:>2.0})     {:>6.2}    {:>5.1}   {cns_col}{:>6.3}    {}{RESET}   {:>5.0}%",
                    truncate_str(&m.id, 12),
                    truncate_str(&m.name, 20),
                    rep.f_oral_pct,
                    rep.fa_integrated * 100.0,
                    rep.fg_gut_wall * 100.0,
                    rep.fh_hepatic * 100.0,
                    rep.vss_l_per_kg,
                    rep.terminal_t_half_hr,
                    rep.kp_uu_brain,
                    truncate_str(rep.cns_class.short_label(), 13),
                    rep.peak_ro_brain_pct
                )?;
            }
        }
        ConsoleTab::LeadScorecard => {
            let summary = evaluate_campaign_scorecard(ds, CampaignTargetProfile::OralPeripheral, 50.0);
            let filtered_rows: Vec<_> = summary
                .rows
                .iter()
                .filter(|r| vis.contains(&r.mol_idx))
                .collect();
            let total_pages = filtered_rows.len().div_ceil(state.page_size).max(1);
            let pg = state.page.min(total_pages - 1);
            let start = pg * state.page_size;
            let end = (start + state.page_size).min(filtered_rows.len());

            writeln!(
                out,
                "  {BOLD}Rank  ID           Name                 pAct   LE/LLE     F_oral  Kp,uu   SAScore  Gates(P/W/G)  D_geom  Stage-Gate Tier{RESET}  (Page {}/{})",
                pg + 1,
                total_pages
            )?;
            for r in &filtered_rows[start..end] {
                let tier_col = match r.tier {
                    ProgressionTier::CandidateReady => EMERALD,
                    ProgressionTier::LeadOptimization => CYAN,
                    ProgressionTier::ProdrugOrFormulationRescue => AMBER,
                    ProgressionTier::EarlyHitOrBackUp => CORAL,
                };
                writeln!(
                    out,
                    "  #{:<4} {} {} {:>5.2}  {:.2}/{:<4.1}  {:>5.1}%  {:>5.2}   {:>6.2}   {:>2}/{:>2}/{:<2}      {BOLD}{:>5.1}{RESET}   {tier_col}{}{RESET}",
                    r.rank,
                    truncate_str(&r.id, 12),
                    truncate_str(&r.name, 20),
                    r.pact,
                    r.le,
                    r.lle,
                    r.f_oral_pct,
                    r.kp_uu_brain,
                    r.sa_score,
                    r.pass_gates,
                    r.conditional_gates,
                    r.fail_gates,
                    r.geometric_desirability,
                    truncate_str(r.tier.short_label(), 16)
                )?;
            }
        }
        ConsoleTab::DeNovoEvolution => {
            if ds.mols.is_empty() {
                writeln!(out, "  No seed molecule loaded.")?;
            } else {
                let idx = state.selected_mol.min(ds.mols.len() - 1);
                let cfg = DeNovoConfig {
                    goal: DeNovoGoalPreset::ScaffoldHop3dRocs,
                    generations: 4,
                    population_size: 20,
                    max_returned: 16,
                    ..DeNovoConfig::default()
                };
                let rep = evolve_de_novo_molecules(&ds.mols[idx], Some(ds), cfg);
                writeln!(
                    out,
                    "  {BOLD}Seed:{RESET} {} ({}) | Evaluated: {BOLD}{}{RESET} unique graphs | Novel Scaffolds: {EMERALD}{}{RESET} | True 3D/2D Hops: {CYAN}{}{RESET}",
                    rep.query_name,
                    rep.query_id,
                    rep.total_unique_evaluated,
                    rep.novel_scaffold_count,
                    rep.true_scaffold_hop_count
                )?;
                writeln!(
                    out,
                    "  {BOLD}Rank  ID       Gen  Operator     ROCS3D  Pharm2D  pAct   LE/LLE     SAScore  Hop?   Fitness  SMILES{RESET}"
                )?;
                for c in rep.candidates.iter().take(state.page_size) {
                    let hop_str = if c.is_true_scaffold_hop {
                        format!("{EMERALD}HOP {RESET}")
                    } else if c.is_novel_scaffold {
                        format!("{CYAN}New {RESET}")
                    } else {
                        format!("{SLATE}Same{RESET}")
                    };
                    writeln!(
                        out,
                        "  #{:<4} {:<8} G{:<2}  {}  {:>5.3}   {:>5.3}   {:>5.2}  {:.2}/{:<4.1}  {:>6.2}   {}   {BOLD}{:>5.1}{RESET}    {}",
                        c.rank,
                        c.candidate_id,
                        c.generation_born,
                        truncate_str(c.last_operator.short_label(), 11),
                        c.rocs_combo_tanimoto,
                        c.pharm2d_similarity,
                        c.predicted_pact,
                        c.le,
                        c.lle,
                        c.sa_score,
                        hop_str,
                        c.fitness_score,
                        truncate_str(&c.canonical_smiles, 28)
                    )?;
                }
            }
        }
        ConsoleTab::PocketCavity => {
            let pr = &CURATED_PDB_POCKETS[state.pocket_preset_idx % CURATED_PDB_POCKETS.len()];
            if let Ok(pdb) = parse_pdb_structure(curated_pocket_pdb_text(state.pocket_preset_idx)) {
                let pocket = analyze_binding_pocket(&pdb, None, 10.5);
                let hits = screen_dataset_against_pocket(ds, &pocket);
                writeln!(
                    out,
                    "  {BOLD}PDB {} ({}){RESET} | Vol {BOLD}{:.0} A^3{RESET} | DScore {EMERALD}{:.2}{RESET} | Unhappy H2O: {CORAL}{}{RESET} | Bridge H2O: {CYAN}{}{RESET} | Pharm3D: {BOLD}{}{RESET}  (press 'w' to cycle PDB)",
                    pocket.pdb_id,
                    pr.target_name,
                    pocket.volume_angstrom3,
                    pocket.druggability_dscore,
                    pocket.unhappy_water_count,
                    pocket.structural_water_count,
                    pocket.pharmacophore_points.len()
                )?;
                writeln!(
                    out,
                    "  {BOLD}Rank  ID           Name                 Target         Score   dG(kcal)  pKd   PockLE  Pharm%  UnhappyH2O  Top Contact{RESET}"
                )?;
                for h in hits.iter().take(state.page_size) {
                    let top_int = h
                        .key_interactions
                        .first()
                        .map(|it| format!("{} ({:.1}A)", it.receptor_partner, it.distance_a))
                        .unwrap_or_else(|| "vdW Enclosure".to_string());
                    writeln!(
                        out,
                        "  #{:<4} {} {} {} {BOLD}{:>5.1}{RESET}   {:>7.2}  {:>5.2}   {:>5.3}   {:>5.0}%      {:>2}      {}",
                        h.rank,
                        truncate_str(&h.compound_id, 12),
                        truncate_str(&h.compound_name, 20),
                        truncate_str(&h.target, 14),
                        h.complementarity_score,
                        h.total_dg_pocket_kcal,
                        h.predicted_pocket_pkd,
                        h.pocket_le,
                        h.pharmacophore_recall_pct,
                        h.displaced_unhappy_waters,
                        truncate_str(&top_int, 22)
                    )?;
                }
            }
        }
        ConsoleTab::ParallelLibrary => {
            let sc_idx = state.lib_scheme_idx % CURATED_LIBRARY_SCHEMES.len();
            let sc = &CURATED_LIBRARY_SCHEMES[sc_idx];
            let pocket_opt = parse_pdb_structure(curated_pocket_pdb_text(state.pocket_preset_idx))
                .ok()
                .map(|p| analyze_binding_pocket(&p, None, 10.5));
            let cfg = ParallelLibConfig {
                scheme_idx: sc_idx,
                synthesis_mode: sc.default_mode,
                prefilter_mode: LibraryPreFilterMode::PocketAndRocsHybrid,
                reagents_per_cycle: 5,
                target_diverse_plate_size: 16,
                max_3d_scored_pool: 32,
                ..ParallelLibConfig::default()
            };
            let q_mol = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1)));
            let rep = enumerate_parallel_library(&cfg, q_mol, pocket_opt.as_ref(), Some(ds));
            writeln!(
                out,
                "  {BOLD}{}{RESET} | Mode: {CYAN}{}{RESET} | Enum: {BOLD}{}{RESET} -> Diverse Plate: {EMERALD}{}{RESET} | Dissim: {:.2}->{:.2}  (press 'l' to cycle scheme)",
                rep.scheme_title,
                rep.synthesis_mode.short_label(),
                rep.valid_unique_enumerated,
                rep.selected_diverse_plate_count,
                rep.mean_pairwise_dissimilarity_all,
                rep.mean_pairwise_dissimilarity_plate
            )?;
            writeln!(
                out,
                "  {BOLD}Rank  Pick  LibID     Blocks         Priority  Pock_dG  ROCS   NearTc  MW     cLogP  Fsp3   QED   Barcode / Well{RESET}"
            )?;
            for m in rep.members.iter().take(state.page_size) {
                let pick_tag = m
                    .diversity_pick_order
                    .map(|k| format!("P{k:<2}"))
                    .unwrap_or_else(|| " - ".to_string());
                writeln!(
                    out,
                    "  #{:<4} {EMERALD}{}{RESET}   {}  {}  {BOLD}{:>6.1}{RESET}   {:>7.2}  {:>5.2}  {:>5.2}   {:>5.1}  {:>5.2}  {:>5.2}  {:>4.2}  {}",
                    m.rank,
                    pick_tag,
                    truncate_str(&m.library_id, 8),
                    truncate_str(&m.cycle_reagent_codes.join("+"), 13),
                    m.composite_priority_score,
                    m.pocket_dg_kcal,
                    m.rocs_combo_score,
                    m.nearest_plate_tanimoto,
                    m.mw,
                    m.clogp,
                    m.fsp3,
                    m.qed,
                    truncate_str(&m.barcode_or_well, 28)
                )?;
            }
        }
        ConsoleTab::SelectivityProfiler => {
            let pidx = state.sel_pair_idx % CURATED_SELECTIVITY_PAIRS.len();
            let pr = &CURATED_SELECTIVITY_PAIRS[pidx];
            if let Ok(dual) = load_curated_selectivity_pair(pidx, false, 10.5) {
                let rows = screen_dataset_selectivity(ds, &dual);
                let sel_m = &ds.mols[state.selected_mol.min(ds.mols.len().saturating_sub(1))];
                let tgt_pdb = parse_pdb_structure(curated_pocket_pdb_text(pr.target_preset_idx)).ok();
                let top_mut = tgt_pdb
                    .as_ref()
                    .and_then(|pdb| {
                        let extra: Vec<String> = pr
                            .signature_mutations
                            .iter()
                            .map(|s| (*s).to_string())
                            .collect();
                        let sc = scan_pocket_residue_mutations(sel_m, pdb, 10.5, &extra);
                        sc.entries.first().map(|e| {
                            format!("{} ({:+.2} kcal, {:.1}x)", e.mutation_code, e.ddg_mutation_kcal, e.fold_affinity_loss)
                        })
                    })
                    .unwrap_or_else(|| "None".to_string());

                writeln!(
                    out,
                    "  {BOLD}{}{RESET} | dVol {BOLD}{:+.0} A^3{RESET} | SubPocketPts: {EMERALD}{}{RESET} | DiffWaters: {CYAN}{}{RESET} | Top Mut ({}): {CORAL}{}{RESET}  (press 'x' to cycle)",
                    pr.title,
                    dual.delta_volume_a3,
                    dual.target_exclusive_subpocket_pts.len(),
                    dual.differential_waters.len(),
                    sel_m.name,
                    top_mut
                )?;
                writeln!(
                    out,
                    "  {BOLD}Rank  ID           Name                 Score  Tgt_dG   Off_dG   ddG_sel   SI(Fold)   Water_ddG  Primary Selectivity Driver{RESET}"
                )?;
                for r in rows.iter().take(state.page_size) {
                    writeln!(
                        out,
                        "  #{:<4} {} {} {BOLD}{:>5.1}{RESET}  {:>7.2}  {:>7.2}  {EMERALD}{:>+7.2}{RESET}   {:>7.1}x   {:>+7.2}    {}",
                        r.rank,
                        truncate_str(&r.compound_id, 12),
                        truncate_str(&r.compound_name, 20),
                        r.selectivity_discovery_score,
                        r.target_dg_kcal,
                        r.offtarget_dg_kcal,
                        r.ddg_selectivity_kcal,
                        r.selectivity_index_fold,
                        r.ddg_watermap_kcal,
                        truncate_str(&r.primary_selectivity_driver, 30)
                    )?;
                }
            }
        }
        ConsoleTab::AllostericCryptic => {
            let sidx = state.allo_sys_idx % CURATED_ALLOSTERIC_SYSTEMS.len();
            let sys = &CURATED_ALLOSTERIC_SYSTEMS[sidx];
            if let Ok(net) = load_curated_allosteric_system(sidx, 10.0) {
                let cp = &net.cryptic_pockets[0];
                let rows = screen_dataset_allosteric_pocket(ds, &net, 0);
                writeln!(
                    out,
                    "  {BOLD}{}{RESET} | Cryptic {}: {BOLD}{:.0}->{:.0} A^3 (+{:.0}%){RESET} via ANM M{} | Path: {CYAN}{}{RESET}  (press 'k' to cycle)",
                    sys.title,
                    cp.site_id,
                    cp.apo_volume_a3,
                    cp.open_holo_volume_a3,
                    cp.volume_expansion_pct,
                    cp.dominant_anm_mode,
                    net.primary_pathway_labels.join("->")
                )?;
                writeln!(
                    out,
                    "  {BOLD}Rank  ID           Name                 Score  Open_dG  Work   App_dG   Strain  ddG_coop  Alpha  Allosteric Mode{RESET}"
                )?;
                for r in rows.iter().take(state.page_size) {
                    writeln!(
                        out,
                        "  #{:<4} {} {} {BOLD}{:>5.1}{RESET}  {:>7.2}  +{:.2}  {EMERALD}{:>7.2}{RESET}  {:>5.2}   {:>+7.2}   {:>5.2}  {}",
                        r.rank,
                        truncate_str(&r.compound_id, 12),
                        truncate_str(&r.compound_name, 20),
                        r.allosteric_discovery_score,
                        r.intrinsic_open_dg_kcal,
                        r.pocket_opening_work_kcal,
                        r.apparent_allosteric_dg_kcal,
                        r.orthosteric_strain_kcal,
                        r.cooperativity_ddg_kcal,
                        r.cooperativity_alpha,
                        truncate_str(r.modulation_mode.short_label(), 26)
                    )?;
                }
            }
        }
        ConsoleTab::ActiveLearningQsar => {
            let acq = AcquisitionStrategy::ALL[state.al_acq_idx % AcquisitionStrategy::ALL.len()];
            let cfg = ActiveLearningConfig {
                acquisition: acq,
                batch_size: 10,
                ..ActiveLearningConfig::default()
            };
            let rep = run_active_learning_campaign(ds, &cfg);
            let ef4 = rep
                .simulated_curve
                .last()
                .map(|r| r.enrichment_factor)
                .unwrap_or(1.0);
            writeln!(
                out,
                "  {BOLD}GP + 24-Tree RF QSAR{RESET} | Acq: {CYAN}{}{RESET} | LOO Q^2={BOLD}{:.2}{RESET} (RMSE {:.2}) | CI90 Cov {:.0}% | Pool {} -> Batch {EMERALD}{}{RESET} | Sim EF {BOLD}{:.2}x{RESET}  (press 'b' to cycle)",
                acq.short_label(),
                rep.validation.loo_q2,
                rep.validation.loo_rmse,
                rep.validation.empirical_90_coverage_pct,
                rep.pool_size,
                rep.selected_batch_count,
                ef4
            )?;
            writeln!(
                out,
                "  {BOLD}Rank  Pick  Cand_ID        Name                   Pred_pAct  +-Sig   90%_CI         EI      UCB    Sim_AD  Provenance{RESET}"
            )?;
            for c in rep.candidates.iter().take(state.page_size) {
                let pick = c
                    .batch_pick_order
                    .map(|k| format!("B{k:<2}"))
                    .unwrap_or_else(|| " - ".to_string());
                writeln!(
                    out,
                    "  #{:<4} {EMERALD}{}{RESET}   {}  {}   {BOLD}{:>6.2}{RESET}    {:>5.2}   [{:.2},{:.2}]  {:>6.3}  {:>5.2}   {:>4.2}   {}",
                    c.rank,
                    pick,
                    truncate_str(&c.candidate_id, 13),
                    truncate_str(&c.candidate_name, 20),
                    c.predicted_pact,
                    c.epistemic_std,
                    c.ci90_low_pact,
                    c.ci90_high_pact,
                    c.expected_improvement,
                    c.ucb_score,
                    c.nearest_train_similarity,
                    truncate_str(&c.provenance, 22)
                )?;
            }
        }
        ConsoleTab::RbfeLomapBar => {
            let sp = RbfeSeriesPreset::ALL[state.rbfe_series_idx % RbfeSeriesPreset::ALL.len()];
            let cfg = RbfeNetworkConfig {
                series_preset: sp,
                topology_mode: RbfeTopologyMode::CycleClosedLomap,
                ..RbfeNetworkConfig::default()
            };
            let rep = run_rbfe_network_campaign(ds, &cfg);
            writeln!(
                out,
                "  {BOLD}Lomap RBFE & BAR{RESET} | Series: {CYAN}{}{RESET} ({}) | Nodes {} | Edges {} | Cycles {} (Max Hyst {AMBER}{:.2}{RESET} -> MLE {EMERALD}{:.3}{RESET} kcal/mol) | RMSE {BOLD}{:.2}{RESET} kcal/mol  (press 'r' to cycle)",
                sp.short_label(),
                rep.pocket_pdb_id,
                rep.summary.n_nodes,
                rep.summary.n_edges,
                rep.summary.n_cycles,
                rep.summary.max_cycle_hysteresis_kcal,
                rep.summary.post_mle_max_cycle_error_kcal,
                rep.summary.dg_rmse_kcal
            )?;
            writeln!(
                out,
                "  {BOLD}Edge  Perturbation (A -> B)            Role          Lomap  MCS  ddG_BAR +- Sig   ddG_MLE   FEP_Hyst  Min_Ov  Status{RESET}"
            )?;
            for e in rep.edges.iter().take(state.page_size) {
                let role = if e.is_spanning_tree_edge { "MST-Tree " } else { "Cyc-Chord" };
                let st_col = match e.convergence_tier {
                    RbfeConvergenceTier::ConvergedOptimal => EMERALD,
                    RbfeConvergenceTier::ModerateCaution => AMBER,
                    RbfeConvergenceTier::HighHysteresisFlagged => CORAL,
                };
                let pair_lbl = format!("{} -> {}", e.src_id, e.dst_id);
                writeln!(
                    out,
                    "  E{:<3}  {}   {}     {:>4.2}   {:>2}   {:>+6.2} +- {:.2}   {:>+6.2}     {:>5.2}    {:>5.3}  {st_col}{}{RESET}",
                    e.edge_idx + 1,
                    truncate_str(&pair_lbl, 28),
                    role,
                    e.lomap.total_lomap_score,
                    e.lomap.mcs_atoms,
                    e.raw_ddg_bar_kcal,
                    e.bar_std_err_kcal,
                    e.reconciled_ddg_mle_kcal,
                    e.fep_fwd_rev_hysteresis_kcal,
                    e.min_phase_space_overlap,
                    e.convergence_tier.short_label()
                )?;
            }
        }
        ConsoleTab::TorsionStrainLock => {
            let lead_idx = state.selected_mol.min(ds.mols.len().saturating_sub(1));
            let ps = BioactivePoseSource::ALL[state.torsion_pose_idx % BioactivePoseSource::ALL.len()];
            let rep = analyze_torsion_strain_and_locks(&ds.mols[lead_idx], Some(ds), ps, true);
            let pass_col = if rep.shoichet_filter_pass { EMERALD } else { CORAL };
            writeln!(
                out,
                "  {BOLD}CSD/SMARTS Torsion & Locking{RESET} | Lead: {CYAN}{} ({}){RESET} | Pose: {} | Total TEU {pass_col}{:.2}{RESET} (Max {:.2}) | Strain {:.2} kcal | N_rot {}  (press 'y' to cycle)",
                rep.compound_name,
                rep.compound_id,
                rep.resolved_pose_label,
                rep.total_strain_teu,
                rep.max_single_bond_teu,
                rep.bioactive_strain_penalty_kcal,
                rep.rotatable_bond_count
            )?;
            writeln!(
                out,
                "  {BOLD}Rot  Dihedral          CSD_Rule            Free_Deg  Bio_Deg  CSD_Peak  Barrier   TEU   Strain  Quality{RESET}"
            )?;
            for b in &rep.bond_profiles {
                let q_col = match b.quality_band {
                    TorsionQualityBand::RelaxedGreen => EMERALD,
                    TorsionQualityBand::ToleratedAmber => AMBER,
                    TorsionQualityBand::StrainedRed => CORAL,
                };
                writeln!(
                    out,
                    "  R{:<2}  {}  {}   {:>6.1}    {:>6.1}    {:>6.1}    {:>5.2}   {:>4.2}   {:>5.2}   {q_col}{}{RESET}",
                    b.rotor_idx + 1,
                    truncate_str(&b.dihedral_label, 16),
                    truncate_str(&b.matched_rule_id, 18),
                    b.free_pose_angle_deg,
                    b.bioactive_pose_angle_deg,
                    b.nearest_csd_peak_deg,
                    b.rotational_barrier_kcal,
                    b.bioactive_strain_teu,
                    b.bioactive_strain_kcal,
                    b.quality_band.short_label()
                )?;
            }
            if !rep.locked_analogs.is_empty() {
                writeln!(
                    out,
                    "  {BOLD}Top Rotatable-Bond Locks:{RESET}  Rank  Target_Dihedral   Strategy                 Barrier  dG_Preorg  Pred_pAct  SMILES"
                )?;
                for c in rep.locked_analogs.iter().take(5) {
                    writeln!(
                        out,
                        "                             #{:<3}  {}  {}  {:>5.2}    {EMERALD}{:>+6.2}{RESET}     {:>6.2}     {}",
                        c.rank,
                        truncate_str(&c.target_dihedral_label, 16),
                        truncate_str(c.strategy.short_label(), 23),
                        c.locked_barrier_kcal,
                        c.preorganization_gain_kcal,
                        c.predicted_locked_pact,
                        truncate_str(&c.canonical_smiles, 32)
                    )?;
                }
            }
        }
        ConsoleTab::DiscoveryFunnel => {
            let fp = DiscoveryFunnelPreset::ALL[state.funnel_preset_idx % DiscoveryFunnelPreset::ALL.len()];
            let cfg = DiscoveryPipelineConfig {
                preset: fp,
                final_promote_count: 8,
                ..DiscoveryPipelineConfig::default()
            };
            let rep = run_discovery_pipeline(ds, &cfg);
            writeln!(
                out,
                "  {BOLD}Closed-Loop DMTA Funnel ({}R){RESET} | {CYAN}{}{RESET} ({}/{}) | Pool {} -> Promoted {EMERALD}{}{RESET} ({} Murcko cores, {:.1}% yield)  (press 'f' to cycle)",
                rep.round_history.len(),
                fp.short_label(),
                rep.target_pocket_pdb,
                rep.selectivity_pair_code,
                rep.initial_pool_size,
                rep.promoted_survivor_count,
                rep.unique_promoted_scaffolds,
                rep.overall_yield_pct
            )?;
            for s in &rep.stage_summaries {
                writeln!(
                    out,
                    "    [S{}] {} : {:>3} -> {:>3} ({:>5.1}% stage | {:>5.1}% cum) | {}",
                    s.stage_num,
                    truncate_str(s.stage_title, 34),
                    s.input_count,
                    s.passed_count,
                    s.stage_pass_pct,
                    s.cumulative_yield_pct,
                    truncate_str(&s.gate_criteria, 44)
                )?;
            }
            writeln!(
                out,
                "  {BOLD}Rank  Status     ID         Source       QED   SASc   Pocket_dG  TEU   dpKd    AL_pAct  RBFE_dG  Score{RESET}"
            )?;
            for c in rep.candidates.iter().take(state.page_size.min(10)) {
                let st_badge = if c.promoted_survivor {
                    format!("{EMERALD}PROMOTED{RESET}")
                } else {
                    format!("{AMBER}Stage S{}{RESET}", c.furthest_stage_reached)
                };
                writeln!(
                    out,
                    "  #{:<3}  {}   {}  {}  {:.2}  {:.2}   {:>+6.2}     {:>4.2}  {:>+5.2}   {:>6.2}   {:>+6.2}   {BOLD}{:>5.1}{RESET}",
                    c.rank,
                    st_badge,
                    truncate_str(&c.candidate_id, 9),
                    truncate_str(c.source_stream.short_label(), 11),
                    c.qed,
                    c.sa_score,
                    c.pocket_dg_kcal.unwrap_or(0.0),
                    c.total_strain_teu.unwrap_or(0.0),
                    c.selectivity_delta_pkd.unwrap_or(0.0),
                    c.al_predicted_pact.unwrap_or(0.0),
                    c.rbfe_mle_dg_kcal.unwrap_or(0.0),
                    c.composite_funnel_score
                )?;
            }
        }
        ConsoleTab::FragmentLinker => {
            let pr = FragmentPairPreset::ALL[state.frag_link_preset_idx % FragmentPairPreset::ALL.len()];
            let sel_lead = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1)));
            let rep = design_linked_fragments(
                &FragmentLinkerConfig {
                    preset: pr,
                    ..FragmentLinkerConfig::default()
                },
                sel_lead,
                Some(ds),
            );
            writeln!(
                out,
                "  {BOLD}Sub-Pocket Fragment Linker{RESET} | Pair: {CYAN}{}{RESET} (PDB {}) | dG_A={:+.2} + dG_B={:+.2} = {:+.2} kcal/mol  (press 'j' to cycle)",
                pr.short_label(),
                rep.target_pdb_id,
                rep.dg_frag_a_kcal,
                rep.dg_frag_b_kcal,
                rep.dg_frag_a_kcal + rep.dg_frag_b_kcal
            )?;
            if let Some(vp) = rep.vector_pairs.get(rep.active_pair_idx) {
                writeln!(
                    out,
                    "    Exit Vectors: {} <-> {} | d_anchor={:.2} A | alpha_A={:.1} deg | alpha_B={:.1} deg | phi={:.1} deg | Regime: {}",
                    vp.vec_a.atom_label,
                    vp.vec_b.atom_label,
                    vp.anchor_distance_a,
                    vp.alpha_a_deg,
                    vp.alpha_b_deg,
                    vp.dihedral_phi_deg,
                    vp.regime.short_label()
                )?;
            }
            writeln!(
                out,
                "  {BOLD}Rank  Code             Class          Geom%  RMSD_A  dG_rigid  dG_strn  dG_AB    ddG_sup   pKd    LLE   Score{RESET}"
            )?;
            for c in rep.candidates.iter().take(state.page_size.min(12)) {
                let sup_col = if c.ddg_superadditivity_kcal <= -1.36 {
                    EMERALD
                } else if c.ddg_superadditivity_kcal <= -0.25 {
                    CYAN
                } else if c.ddg_superadditivity_kcal <= 0.85 {
                    AMBER
                } else {
                    CORAL
                };
                writeln!(
                    out,
                    "  #{:<3}  {}  {}  {:>5.1}  {:>5.2}   {:>+6.2}    {:>+6.2}   {:>+6.2}  {sup_col}{:>+7.2}{RESET}  {:>5.2}  {:>4.2}   {BOLD}{:>5.1}{RESET}",
                    c.rank,
                    truncate_str(c.linker_code, 15),
                    truncate_str(c.linker_class.short_label(), 13),
                    c.geometry_match_score,
                    c.fragment_core_rmsd_a,
                    c.dg_rigid_entropy_saved_kcal,
                    c.dg_linker_strain_kcal,
                    c.dg_linked_ab_kcal,
                    c.ddg_superadditivity_kcal,
                    c.predicted_pkd,
                    c.lle,
                    c.composite_linker_score
                )?;
            }
        }
        ConsoleTab::InducedFitEnsemble => {
            let pr = InducedFitPreset::ALL[state.ifd_preset_idx % 4];
            let cfg = InducedFitConfig {
                preset: pr,
                ..InducedFitConfig::default()
            };
            if let Some(sel_m) = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1))) {
                let rep = score_ligand_induced_fit_ensemble(sel_m, Some(ds), &cfg);
                writeln!(
                    out,
                    "  {BOLD}Induced-Fit & Boltzmann Ensemble{RESET} | {CYAN}{}{RESET} (PDB {}) | Lead: {} ({})  (press 'i' to cycle)",
                    pr.short_label(),
                    rep.pdb_id,
                    rep.compound_name,
                    rep.compound_id
                )?;
                writeln!(
                    out,
                    "    Rigid dG={:+.2} -> Boltzmann dG={EMERALD}{:+.2} kcal/mol{RESET} (Rescue {:+.2} | IFDScore {:+.2} | KL Shift {:.2} nats | {})",
                    rep.rigid_holo_dg_kcal,
                    rep.boltzmann_ensemble_dg_kcal,
                    rep.ddg_induced_fit_rescue_kcal,
                    rep.best_sherman_ifd_score_kcal,
                    rep.kl_population_shift_nats,
                    rep.mechanism.short_label()
                )?;
                for s in &rep.state_scores {
                    writeln!(
                        out,
                        "    [{}] w_apo={:>5.1}% -> w_holo={CYAN}{:>5.1}%{RESET} | dG_reorg={:+.2} | dG_rot={:+.2} | dG_eff={:+.2} | Repacked {} sc ({:.2} A)",
                        truncate_str(s.state_code, 15),
                        s.apo_weight * 100.0,
                        s.holo_boltzmann_weight * 100.0,
                        s.apo_reorg_penalty_kcal,
                        s.rotamer_strain_kcal,
                        s.effective_total_dg_kcal,
                        s.repacked_residue_count,
                        s.repacked_sidechain_rmsd_a
                    )?;
                }
            }
            let rows = screen_dataset_induced_fit_ensemble(ds, &cfg);
            writeln!(
                out,
                "  {BOLD}Rank  ID           Name                 Rigid_dG  Boltz_dG  Rescue   IFDScore  DomState         Occ%   Mechanism{RESET}"
            )?;
            for r in rows.iter().take(state.page_size.min(8)) {
                writeln!(
                    out,
                    "  #{:<3}  {}  {}  {:>+7.2}   {:>+7.2}   {:>+6.2}   {:>+7.2}   {}  {:>5.1}%  {}",
                    r.rank,
                    truncate_str(&r.compound_id, 11),
                    truncate_str(&r.compound_name, 19),
                    r.rigid_holo_dg_kcal,
                    r.boltzmann_ensemble_dg_kcal,
                    r.ddg_induced_fit_rescue_kcal,
                    r.sherman_ifd_score_kcal,
                    truncate_str(r.dominant_state_code, 15),
                    r.dominant_occupancy_pct,
                    r.mechanism.short_label()
                )?;
            }
        }
        ConsoleTab::HBondNetwork => {
            let pr = HBondNetworkPreset::ALL[state.hbn_preset_idx % 4];
            let cfg = HBondNetworkConfig {
                preset: pr,
                target_ph: 7.4,
                ..HBondNetworkConfig::default()
            };
            if let Some(sel_m) = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1))) {
                let rep = optimize_pocket_hbond_network(sel_m, Some(ds), &cfg);
                writeln!(
                    out,
                    "  {BOLD}His/Asp/Glu & Bridge-Water H-Bond Network{RESET} | {CYAN}{}{RESET} (PDB {}, pH {:.1}) | Lead: {}  (press 't' to cycle)",
                    pr.short_label(),
                    rep.pdb_id,
                    rep.target_ph,
                    rep.compound_name
                )?;
                writeln!(
                    out,
                    "    Naive dG={:+.2} -> Optimized dG={EMERALD}{:+.2} kcal/mol{RESET} (Gain {:+.2} | Clashes {}->{} | Bridge HOH: {} | 180-Flips: {} | dn_H+={:+.1})",
                    rep.naive_network_dg_kcal,
                    rep.optimized_network_dg_kcal,
                    rep.ddg_hbn_optimization_gain_kcal,
                    rep.naive_polar_clashes,
                    rep.optimized_polar_clashes,
                    rep.retained_bridge_waters,
                    rep.flipped_180_residues,
                    rep.net_proton_uptake_at_ph
                )?;
                for n in rep.nodes.iter().take(6) {
                    let op = &n.modes[n.optimal_mode_idx];
                    writeln!(
                        out,
                        "    [{}] {} -> {CYAN}{}{RESET} (dG_gain={:+.2} kcal) | {}",
                        truncate_str(&n.node_id, 9),
                        truncate_str(n.kind.short_label(), 21),
                        truncate_str(&op.state_code, 14),
                        n.node_stabilization_gain_kcal,
                        truncate_str(&op.description, 44)
                    )?;
                }
            }
            let rows = screen_dataset_hbond_network(ds, &cfg);
            writeln!(
                out,
                "  {BOLD}Rank  ID           Name                 Naive_dG  Opt_HBN   Gain     AdjPock_dG  HBonds  HOH  LigState{RESET}"
            )?;
            for r in rows.iter().take(state.page_size.min(7)) {
                writeln!(
                    out,
                    "  #{:<3}  {}  {}  {:>+7.2}   {:>+7.2}   {:>+6.2}   {:>+8.2}     {:>3}     {:>2}   {}",
                    r.rank,
                    truncate_str(&r.compound_id, 11),
                    truncate_str(&r.compound_name, 19),
                    r.naive_network_dg_kcal,
                    r.optimized_network_dg_kcal,
                    r.ddg_hbn_gain_kcal,
                    r.hbn_adjusted_pocket_dg_kcal,
                    r.direct_hbonds,
                    r.retained_bridge_waters,
                    truncate_str(&r.selected_ligand_state, 12)
                )?;
            }
        }
        ConsoleTab::WaterMapSolvation => {
            let pr = WaterMapPreset::ALL[state.wm_preset_idx % 5];
            let cfg = WaterMapConfig {
                preset: pr,
                ..WaterMapConfig::default()
            };
            if let Some(sel_m) = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1))) {
                let rep = analyze_watermap_solvation(sel_m, Some(ds), &cfg);
                writeln!(
                    out,
                    "  {BOLD}WaterMap / IFST Hydration-Site & R-Vector Studio{RESET} | {CYAN}{}{RESET} (PDB {}) | Lead: {}  (press 'm' to cycle)",
                    pr.short_label(),
                    rep.pdb_id,
                    rep.compound_name
                )?;
                writeln!(
                    out,
                    "    Sites: {} (Unhappy={} | Dewet={} | Bridge={}) | Displaced={} | Net dG_solv={EMERALD}{:+.2}{RESET} | Refined dG={:+.2} kcal/mol | Headroom={:+.2}",
                    rep.total_hydration_sites,
                    rep.unhappy_hotspot_sites,
                    rep.dewetted_cavity_sites,
                    rep.conserved_bridge_sites,
                    rep.displaced_hotspot_count,
                    rep.net_watermap_dg_kcal,
                    rep.watermap_refined_dg_kcal,
                    rep.untapped_proximal_hotspot_dg_kcal
                )?;
                for d in rep.rgroup_designs.iter().take(4) {
                    writeln!(
                        out,
                        "    R-Vec #{:<2} [{}] {} @ {} -> {} (dG_site={:+.2}) | ddG={EMERALD}{:+.2}{RESET} kcal | GE={:.2} | dLLE={:+.2}",
                        d.rank,
                        d.design_id,
                        truncate_str(d.rgroup_name, 18),
                        truncate_str(&d.anchor_atom_label, 8),
                        d.target_site_id,
                        d.target_site_dg_kcal,
                        d.ddg_solvation_gain_kcal,
                        d.group_efficiency_ge,
                        d.delta_lle
                    )?;
                }
            }
            let rows = screen_dataset_watermap_solvation(ds, &cfg);
            writeln!(
                out,
                "  {BOLD}Rank  ID           Name                 DispHot  Bridges  Net_WM_dG  Refined_dG  Top R-Group Vector{RESET}"
            )?;
            for r in rows.iter().take(state.page_size.min(6)) {
                writeln!(
                    out,
                    "  #{:<3}  {}  {}     {:>2}       {:>2}     {:>+8.2}    {:>+9.2}  {}",
                    r.rank,
                    truncate_str(&r.compound_id, 11),
                    truncate_str(&r.compound_name, 19),
                    r.displaced_hotspots,
                    r.retained_bridges,
                    r.net_watermap_dg_kcal,
                    r.watermap_refined_dg_kcal,
                    truncate_str(&r.top_rgroup_suggestion, 34)
                )?;
            }
        }
        ConsoleTab::FolditSculpt => {
            let pr = SculptPuzzlePreset::ALL[state.sculpt_preset_idx % 5];
            let sel_m = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1)));
            let mut sess = new_sculpt_session(pr, sel_m, None);
            let start_pts = sess.score.foldit_score_pts;
            let bands_n = auto_band_polar_contacts(&mut sess);
            run_blue_fuse_macro(&mut sess);
            let [t_br, t_si, t_go, t_pl] = pr.medal_thresholds();
            writeln!(
                out,
                "  {BOLD}Foldit / Drugit 3D Sculptor & Challenge Benchmark{RESET} | {CYAN}{}{RESET} (PDB {})  (press 'z' to cycle)",
                pr.title(),
                sess.pdb_id
            )?;
            writeln!(
                out,
                "    Start Score: {:.0} pts -> Auto-Band ({}) + Blue-Fuse Score: {EMERALD}{:.0} pts{RESET} [{}] | Par: B={:.0} S={:.0} G={:.0} P={:.0}",
                start_pts,
                bands_n,
                sess.score.foldit_score_pts,
                sess.score.medal.label(),
                t_br,
                t_si,
                t_go,
                t_pl
            )?;
            writeln!(
                out,
                "    Unbanded dG={:+.2} kcal/mol | Strain={:.2} | Clashes={} | H-Bonds={} | WaterHotspots={} | TPSA={:.1} | cLogP={:.2} | PAINS={}",
                sess.score.total_unbanded_dg_kcal,
                sess.score.cart_bonded_strain_kcal,
                sess.score.clash_count,
                sess.score.formed_hbonds,
                sess.score.displaced_unhappy_waters,
                sess.score.tpsa,
                sess.score.clogp,
                sess.score.pains_count
            )?;
            writeln!(
                out,
                "  {BOLD}Step  Action                        Score_Pts   Unbanded_dG   HBonds   Clashes{RESET}"
            )?;
            for st in &sess.trajectory {
                writeln!(
                    out,
                    "  #{:<3}  {}  {:>9.1}    {:>+9.2}      {:>3}       {:>3}",
                    st.step_idx,
                    truncate_str(&st.action_label, 28),
                    st.foldit_score_pts,
                    st.unbanded_dg_kcal,
                    st.hbonds,
                    st.clashes
                )?;
            }
        }
        ConsoleTab::PublicPdbReadiness => {
            let pr = PublicTargetPreset::ALL[state.pub_pdb_preset_idx % 5];
            let cfg = PublicPdbFetchConfig {
                source: PublicStructureSource::OfflineBenchmarkPreset,
                preset: pr,
                ..PublicPdbFetchConfig::default()
            };
            if let Ok(rep) = fetch_and_assess_public_structure(&cfg) {
                writeln!(
                    out,
                    "  {BOLD}Public RCSB PDB & AlphaFold DB Readiness{RESET} | {CYAN}{}{RESET} ({})  (press 'u' to cycle)",
                    rep.accession_code,
                    rep.provenance_badge
                )?;
                writeln!(
                    out,
                    "    Global pLDDT={:.1} (>=90: {:.0}% | <50: {:.0}%) | Pocket pLDDT={:.1} | Pocket PAE={:.2} A | Readiness={EMERALD}{:.1}/100{RESET}",
                    rep.global_mean_plddt,
                    rep.pct_very_high_90,
                    rep.pct_disordered_lt50,
                    rep.pocket_mean_plddt,
                    rep.pocket_mean_pae_a,
                    rep.readiness_score_0_100
                )?;
                writeln!(
                    out,
                    "    Verdict: {} | Next: {}",
                    rep.verdict.label(),
                    rep.recommended_next_module
                )?;
                writeln!(
                    out,
                    "  {BOLD}Residue   Chain  pLDDT   RMSF_A  Band                  Pocket  Occluding  Intrusion{RESET}"
                )?;
                for r in rep.residue_rows.iter().filter(|r| r.lines_binding_pocket || r.is_disordered_pocket_intrusion).take(8) {
                    writeln!(
                        out,
                        "  {:<8}    {}    {:>5.1}   {:>5.2}   {}   {:>5}     {:>5}      {:>5}",
                        format!("{}{}", r.res_name, r.res_seq),
                        r.chain_id,
                        r.effective_plddt,
                        r.est_rmsf_a,
                        truncate_str(r.band.short_label(), 20),
                        if r.lines_binding_pocket { "YES" } else { "-" },
                        if r.is_apo_occluding_sidechain { "APO!" } else { "-" },
                        if r.is_disordered_pocket_intrusion { "TRIM!" } else { "-" }
                    )?;
                }
            }
        }
        ConsoleTab::ElectrostaticsEsp => {
            let pr = EspTargetPreset::ALL[state.esp_preset_idx % 6];
            let cfg = ElectrostaticsEspConfig {
                preset: pr,
                ..ElectrostaticsEspConfig::default()
            };
            if let Some(sel_m) = ds.mols.get(state.selected_mol.min(ds.mols.len().saturating_sub(1))) {
                let rep = analyze_electrostatics_esp(sel_m, Some(ds), &cfg);
                writeln!(
                    out,
                    "  {BOLD}3D ESP, Sigma-Hole & Pi-Multipole Studio{RESET} | {CYAN}{}{RESET} (PDB {}) | Lead: {}  (press 'e' to cycle)",
                    pr.short_label(),
                    rep.pdb_id,
                    rep.compound_name
                )?;
                writeln!(
                    out,
                    "    Bauer-Mackey EC={:+.3} | EC_r={EMERALD}{:+.3}{RESET} | Fav={:.0}% Clash={:.0}% | XB={} | ChB={} | Pi/CatPi={} | Mult dG={:+.2} kcal/mol",
                    rep.bauer_mackey_ec_integral,
                    rep.bauer_mackey_ec_pearson_r,
                    rep.esp_favorable_surface_pct,
                    rep.esp_clash_surface_pct,
                    rep.halogen_bond_count,
                    rep.chalcogen_bond_and_lock_count,
                    rep.pi_and_cation_pi_count,
                    rep.total_multipole_dg_kcal
                )?;
                for c in rep.contacts.iter().take(4) {
                    writeln!(
                        out,
                        "    [{}] {} <-> {} (d={:.2} A, ang={:.0} deg) | {:+.2} kcal | {}",
                        truncate_str(c.kind.short_label(), 22),
                        truncate_str(&c.ligand_site_label, 8),
                        truncate_str(&c.partner_label, 14),
                        c.distance_a,
                        c.angle_deg,
                        c.energy_kcal,
                        truncate_str(&c.rationale, 38)
                    )?;
                }
            }
            let rows = screen_dataset_electrostatics_esp(ds, &cfg);
            writeln!(
                out,
                "  {BOLD}Rank  ID           Name                 EC_Int   EC_r    XB  ChB  Pi  Mult_dG   Refined_dG{RESET}"
            )?;
            for r in rows.iter().take(state.page_size.min(6)) {
                writeln!(
                    out,
                    "  #{:<3}  {}  {}  {:>+6.2}  {:>+6.2}   {:>2}   {:>2}  {:>2}  {:>+7.2}    {:>+8.2}",
                    r.rank,
                    truncate_str(&r.compound_id, 11),
                    truncate_str(&r.compound_name, 19),
                    r.ec_integral,
                    r.ec_pearson_r,
                    r.halogen_bonds,
                    r.chalcogen_contacts,
                    r.pi_contacts,
                    r.total_multipole_dg_kcal,
                    r.esp_refined_pocket_dg_kcal
                )?;
            }
        }
    }

    // Selected Compound Footer Summary
    if !ds.mols.is_empty() {
        let idx = state.selected_mol.min(ds.mols.len() - 1);
        let m = &ds.mols[idx];
        let ad = &ds.admet[idx];
        let mt = &ds.metabolism[idx];
        writeln!(
            out,
            "  {DIM}──────────────────────────────────────────────────────────────────────────────────────────────────────────────{RESET}"
        )?;
        writeln!(
            out,
            "  {BOLD}{VIOLET}Selected #{}:{RESET} {BOLD}{}{RESET} ({}) | {} = {:.2} nM (pAct {:.2}) | BOILED-Egg: {} | SMARTCyp SoM: {} ({}, CL_int {:.1})",
            idx + 1,
            m.name,
            m.id,
            m.metric,
            m.raw_nm,
            m.pact,
            ad.boiled_egg.label(),
            mt.top_som_label,
            mt.primary_isoform.label(),
            mt.cl_int_ml_min_kg
        )?;
    }
    writeln!(out, "  {AMBER}Status:{RESET} {}", state.status_msg)?;
    writeln!(
        out,
        "  {SLATE}Commands: [1..23] View | [e] ESP/Sigma-Hole | [u] RCSB/AFDB | [z] Foldit Sculpt | [m] WaterMap | [t] H-Bond Net | [i] IFD Ensemble | [j] Linker | [f] Funnel | [y] Torsion | [r] Lomap RBFE | [b] Active-Learn | [k] Allostery | [x] Selectivity | [l] DEL/Lib | [w] Cycle PDB | [g] De Novo GA | [h] HTML Dossier | [/<smarts>] Filter | [s <id/#>] Inspect | [a] PAINS | [o] Pareto | [n/p] Page | [q] Quit{RESET}"
    )?;

    Ok(())
}
