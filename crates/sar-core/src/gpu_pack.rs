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

//! Zero-allocation `bytemuck::{Pod, Zeroable}` GPU uniform and instance buffer packers
//! (`GpuCameraUniform` 48 B, `GpuNodeInstance` 32 B, `GpuEdgeInstance` 32 B) and `256x256` RGBA8
//! separable-Gaussian chemical-space heatmap rasterizer.

use crate::alerts::{AlertSeverity, StructuralAlertMatch};
use crate::fingerprint::SaliCliffEdge;
use crate::mol::Molecule;
use crate::physchem::PhyschemProfile;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const HEATMAP_DIM: usize = 256;

/// 48-byte camera and viewport uniform (`3 x vec4<f32>` in WGSL, zero implicit padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCameraUniform {
    pub pan: [f32; 2],
    pub zoom: f32,
    pub aspect: f32,
    pub viewport_px: [f32; 2],
    pub heatmap_alpha: f32,
    pub time_sec: f32,
    pub bounds_min: [f32; 2],
    pub bounds_span: [f32; 2],
}

/// 32-byte instanced compound glyph (`2 x vec4<f32>` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuNodeInstance {
    pub pos: [f32; 2],
    pub radius: f32,
    /// `0.0` = clean, `1.0` = warhead, `2.0` = PAINS, `3.0` = selected, `4.0` = Pareto, `5.0` = SMARTS hit spotlight.
    pub halo_kind: f32,
    pub color: [f32; 4],
}

/// 32-byte instanced SALI / Tanimoto similarity edge (`2 x vec4<f32>` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuEdgeInstance {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeatmapLayer {
    PotencySurface,
    SaliCliffIntensity,
    CnsMpoDesirability,
    CompoundDensity,
}

impl HeatmapLayer {
    pub fn label(self) -> &'static str {
        match self {
            Self::PotencySurface => "pAct Potency Landscape",
            Self::SaliCliffIntensity => "SALI Cliff Steepness",
            Self::CnsMpoDesirability => "CNS-MPO Desirability",
            Self::CompoundDensity => "Chemotype Density",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeColorMode {
    PotencyViridis,
    TargetSeries,
    QedScore,
    CnsMpoScore,
}

impl NodeColorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::PotencyViridis => "Potency (pAct)",
            Self::TargetSeries => "Assay Target",
            Self::QedScore => "Bickerton QED",
            Self::CnsMpoScore => "Wager CNS-MPO",
        }
    }
}

/// Packs visible compound nodes and SALI edges in parallel via `Rayon`, supporting optional
/// live SMARTS spotlighting (`smarts_hits` mask) across the GPU SALI landscape.
#[allow(clippy::too_many_arguments)]
pub fn pack_gpu_scene(
    mols: &[Molecule],
    coords: &[[f32; 2]],
    profiles: &[PhyschemProfile],
    alerts: &[Vec<StructuralAlertMatch>],
    edges: &[SaliCliffEdge],
    visible_mask: &[bool],
    smarts_hit_mask: Option<&[bool]>,
    selected_idx: Option<usize>,
    color_mode: NodeColorMode,
    cliffs_only_edges: bool,
) -> (Vec<GpuNodeInstance>, Vec<GpuEdgeInstance>) {
    let nodes: Vec<GpuNodeInstance> = (0..mols.len())
        .into_par_iter()
        .filter(|&i| visible_mask.get(i).copied().unwrap_or(true))
        .map(|i| {
            let pact_t = ((mols[i].pact - 4.5) / 6.5).clamp(0.0, 1.0) as f32;
            let base_radius = if mols.len() > 2000 {
                1.35 + pact_t * 1.15
            } else {
                3.4 + pact_t * 2.8
            };

            let is_smarts_hit = smarts_hit_mask
                .and_then(|m| m.get(i).copied())
                .unwrap_or(false);
            let smarts_active = smarts_hit_mask.is_some();

            let has_pains = alerts[i]
                .iter()
                .any(|a| a.severity == AlertSeverity::PainsAssayInterference);
            let has_warhead = !alerts[i].is_empty() && !has_pains;

            let halo_kind = if selected_idx == Some(i) {
                3.0
            } else if is_smarts_hit {
                5.0
            } else if has_pains {
                2.0
            } else if has_warhead {
                1.0
            } else if profiles[i].pareto_optimal {
                4.0
            } else {
                0.0
            };

            let rgb = match color_mode {
                NodeColorMode::PotencyViridis => viridis_rgb(pact_t),
                NodeColorMode::TargetSeries => target_palette_rgb(&mols[i].target),
                NodeColorMode::QedScore => viridis_rgb(profiles[i].qed_mean.clamp(0.0, 1.0) as f32),
                NodeColorMode::CnsMpoScore => {
                    viridis_rgb((profiles[i].cns_mpo / 6.0).clamp(0.0, 1.0) as f32)
                }
            };

            let alpha = if smarts_active && !is_smarts_hit && selected_idx != Some(i) {
                0.28
            } else {
                0.96
            };

            GpuNodeInstance {
                pos: coords[i],
                radius: if selected_idx == Some(i) {
                    base_radius * 1.38
                } else if is_smarts_hit {
                    base_radius * 1.18
                } else {
                    base_radius
                },
                halo_kind,
                color: [rgb[0], rgb[1], rgb[2], alpha],
            }
        })
        .collect();

    let max_edges = 24_000usize;
    let packed_edges: Vec<GpuEdgeInstance> = edges
        .iter()
        .filter(|e| {
            let va = visible_mask.get(e.mol_a).copied().unwrap_or(true);
            let vb = visible_mask.get(e.mol_b).copied().unwrap_or(true);
            va && vb && (!cliffs_only_edges || e.is_cliff)
        })
        .take(max_edges)
        .map(|e| {
            let col = if e.is_cliff {
                let alpha = ((e.sali / 35.0).clamp(0.45, 0.95)) as f32;
                [0.98, 0.28, 0.36, alpha]
            } else {
                let alpha = ((e.tanimoto - 0.30) * 0.65).clamp(0.14, 0.42);
                [0.38, 0.64, 0.96, alpha]
            };
            GpuEdgeInstance {
                p0: coords[e.mol_a],
                p1: coords[e.mol_b],
                color: col,
            }
        })
        .collect();

    (nodes, packed_edges)
}

/// Rasterizes a `256x256` RGBA8 chemical-space continuous field (`Potency`, `SALI`, `CNS-MPO`, or `Density`)
/// over world bounds `[-100, 100]^2` using a separable Gaussian kernel.
pub fn build_chem_heatmap_rgba(
    mols: &[Molecule],
    coords: &[[f32; 2]],
    profiles: &[PhyschemProfile],
    edges: &[SaliCliffEdge],
    visible_mask: &[bool],
    layer: HeatmapLayer,
) -> Vec<u8> {
    let dim = HEATMAP_DIM;
    let mut num = vec![0.0f32; dim * dim];
    let mut den = vec![0.0f32; dim * dim];

    let world_to_grid = |p: [f32; 2]| -> (usize, usize) {
        let gx = (((p[0] + 100.0) / 200.0) * ((dim - 1) as f32))
            .round()
            .clamp(0.0, (dim - 1) as f32) as usize;
        let gy = (((p[1] + 100.0) / 200.0) * ((dim - 1) as f32))
            .round()
            .clamp(0.0, (dim - 1) as f32) as usize;
        (gx, gy)
    };

    match layer {
        HeatmapLayer::SaliCliffIntensity => {
            for e in edges {
                if !visible_mask.get(e.mol_a).copied().unwrap_or(true)
                    || !visible_mask.get(e.mol_b).copied().unwrap_or(true)
                {
                    continue;
                }
                let mid = [
                    (coords[e.mol_a][0] + coords[e.mol_b][0]) * 0.5,
                    (coords[e.mol_a][1] + coords[e.mol_b][1]) * 0.5,
                ];
                let (gx, gy) = world_to_grid(mid);
                let v = (e.sali / 30.0).clamp(0.0, 1.0);
                num[gy * dim + gx] += v;
                den[gy * dim + gx] += 1.0;
            }
        }
        _ => {
            for i in 0..mols.len() {
                if !visible_mask.get(i).copied().unwrap_or(true) {
                    continue;
                }
                let (gx, gy) = world_to_grid(coords[i]);
                let val = match layer {
                    HeatmapLayer::PotencySurface => {
                        ((mols[i].pact - 4.5) / 6.5).clamp(0.0, 1.0) as f32
                    }
                    HeatmapLayer::CnsMpoDesirability => {
                        (profiles[i].cns_mpo / 6.0).clamp(0.0, 1.0) as f32
                    }
                    HeatmapLayer::CompoundDensity => 1.0,
                    HeatmapLayer::SaliCliffIntensity => 0.0,
                };
                num[gy * dim + gx] += val;
                den[gy * dim + gx] += 1.0;
            }
        }
    }

    let radius: isize = if mols.len() > 2000 { 6 } else { 14 };
    let sigma = (radius as f32) * 0.48;
    let weights: Vec<f32> = (-radius..=radius)
        .map(|k| (-((k as f32) * (k as f32)) / (2.0 * sigma * sigma)).exp())
        .collect();

    let blur_2d = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0f32; dim * dim];
        let mut dst = vec![0.0f32; dim * dim];
        for y in 0..dim {
            for x in 0..dim {
                let mut acc = 0.0f32;
                for (wi, k) in (-radius..=radius).enumerate() {
                    let sx = (x as isize + k).clamp(0, (dim - 1) as isize) as usize;
                    acc += src[y * dim + sx] * weights[wi];
                }
                tmp[y * dim + x] = acc;
            }
        }
        for y in 0..dim {
            for x in 0..dim {
                let mut acc = 0.0f32;
                for (wi, k) in (-radius..=radius).enumerate() {
                    let sy = (y as isize + k).clamp(0, (dim - 1) as isize) as usize;
                    acc += tmp[sy * dim + x] * weights[wi];
                }
                dst[y * dim + x] = acc;
            }
        }
        dst
    };

    let b_num = blur_2d(&num);
    let b_den = blur_2d(&den);
    let max_den = b_den.iter().copied().fold(1e-5f32, f32::max);

    let mut rgba = vec![0u8; dim * dim * 4];
    for idx in 0..(dim * dim) {
        let d = b_den[idx];
        if d < 1e-4 {
            continue;
        }
        let intensity = if layer == HeatmapLayer::CompoundDensity {
            (d / max_den).sqrt().clamp(0.0, 1.0)
        } else {
            (b_num[idx] / (d + 0.03)).clamp(0.0, 1.0)
        };
        let coverage = (d / (max_den * 0.28)).clamp(0.0, 1.0);
        let rgb = if layer == HeatmapLayer::SaliCliffIntensity {
            inferno_cliff_rgb(intensity)
        } else {
            viridis_rgb(intensity)
        };
        rgba[idx * 4] = (rgb[0] * 255.0) as u8;
        rgba[idx * 4 + 1] = (rgb[1] * 255.0) as u8;
        rgba[idx * 4 + 2] = (rgb[2] * 255.0) as u8;
        rgba[idx * 4 + 3] = (coverage * 215.0) as u8;
    }
    rgba
}

pub fn viridis_rgb(t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0);
    if x < 0.33 {
        let u = x / 0.33;
        lerp3([0.16, 0.19, 0.48], [0.12, 0.54, 0.58], u)
    } else if x < 0.66 {
        let u = (x - 0.33) / 0.33;
        lerp3([0.12, 0.54, 0.58], [0.28, 0.78, 0.43], u)
    } else {
        let u = (x - 0.66) / 0.34;
        lerp3([0.28, 0.78, 0.43], [0.98, 0.84, 0.18], u)
    }
}

fn inferno_cliff_rgb(t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0);
    if x < 0.5 {
        lerp3([0.14, 0.10, 0.32], [0.84, 0.22, 0.38], x * 2.0)
    } else {
        lerp3([0.84, 0.22, 0.38], [1.0, 0.78, 0.22], (x - 0.5) * 2.0)
    }
}

pub fn target_palette_rgb(target: &str) -> [f32; 3] {
    match target {
        "EGFR" => [0.22, 0.74, 0.97],
        "Dopamine D2" => [0.65, 0.54, 0.98],
        "COX-2" => [0.20, 0.83, 0.60],
        "Aldose Reductase" => [0.98, 0.57, 0.24],
        "CYP3A4" => [0.96, 0.44, 0.71],
        _ => {
            let mut h = 0u32;
            for b in target.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b as u32);
            }
            match h % 4 {
                0 => [0.29, 0.80, 0.85],
                1 => [0.95, 0.62, 0.26],
                2 => [0.52, 0.82, 0.38],
                _ => [0.88, 0.46, 0.68],
            }
        }
    }
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
