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

// SAR-Atlas 3-Pass Hardware-Instanced Chemical-Space & SALI Activity-Cliff Shader (v0.2.0)
// Pass 1: Continuous 256x256 bilinear potency / SALI / CNS-MPO / density field + topographic iso-contours (`vs_heatmap` / `fs_heatmap`)
// Pass 2: Anti-aliased instanced Tanimoto & SALI activity-cliff edges (`vs_edge` / `fs_edge`)
// Pass 3: Signed Distance Field (SDF) instanced compound nodes with SMARTS / PAINS / Warhead / Pareto / Selection halos (`vs_node` / `fs_node`)

struct CameraUniform {
    pan: vec2<f32>,
    zoom: f32,
    aspect: f32,
    viewport_px: vec2<f32>,
    heatmap_alpha: f32,
    time_sec: f32,
    bounds_min: vec2<f32>,
    bounds_span: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> cam: CameraUniform;

@group(0) @binding(1)
var t_heatmap: texture_2d<f32>;

@group(0) @binding(2)
var s_heatmap: sampler;

fn world_to_ndc(world_xy: vec2<f32>) -> vec2<f32> {
    let shifted = (world_xy - cam.pan) * cam.zoom;
    let sx = shifted.x / max(cam.aspect * 100.0, 1e-4);
    let sy = shifted.y / 100.0;
    return vec2<f32>(sx, sy);
}

fn ndc_to_world(ndc: vec2<f32>) -> vec2<f32> {
    let wx = (ndc.x * cam.aspect * 100.0) / max(cam.zoom, 1e-4) + cam.pan.x;
    let wy = (ndc.y * 100.0) / max(cam.zoom, 1e-4) + cam.pan.y;
    return vec2<f32>(wx, wy);
}

// -----------------------------------------------------------------------------
// PASS 1: Continuous Chemical-Space Heatmap + Subtle Coordinate Grid & Contours
// -----------------------------------------------------------------------------

struct HeatmapVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_heatmap(@builtin(vertex_index) vid: u32) -> HeatmapVarying {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let p = corners[vid];
    var out: HeatmapVarying;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_heatmap(in: HeatmapVarying) -> @location(0) vec4<f32> {
    let world = ndc_to_world(in.ndc);
    let uv = (world - cam.bounds_min) / max(cam.bounds_span, vec2<f32>(1.0, 1.0));

    let bg = vec3<f32>(0.045, 0.065, 0.105);
    let g_step = 25.0;
    let gx = abs(fract(world.x / g_step + 0.5) - 0.5) * g_step;
    let gy = abs(fract(world.y / g_step + 0.5) - 0.5) * g_step;
    let px_world = 200.0 / max(cam.viewport_px.y * cam.zoom, 1.0);
    let grid_mask = 1.0 - smoothstep(0.0, px_world * 1.35, min(gx, gy));
    var col = mix(bg, vec3<f32>(0.10, 0.145, 0.22), grid_mask * 0.55);

    if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && cam.heatmap_alpha > 0.01) {
        let tex_uv = vec2<f32>(uv.x, uv.y);
        let h = textureSample(t_heatmap, s_heatmap, tex_uv);
        let lum = dot(h.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let iso = abs(fract(lum * 7.0) - 0.5);
        let contour = (1.0 - smoothstep(0.0, 0.055, iso)) * h.a * 0.22;
        let field_alpha = clamp(h.a * cam.heatmap_alpha, 0.0, 0.85);
        col = mix(col, h.rgb + vec3<f32>(contour), field_alpha);
    }

    return vec4<f32>(col, 1.0);
}

// -----------------------------------------------------------------------------
// PASS 2: Instanced Anti-Aliased SALI Cliff & Similarity Edges
// -----------------------------------------------------------------------------

struct EdgeInput {
    @location(0) p0: vec2<f32>,
    @location(1) p1: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct EdgeVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) edge_dist: f32,
};

@vertex
fn vs_edge(@builtin(vertex_index) vid: u32, inst: EdgeInput) -> EdgeVarying {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(0.0,  1.0),
    );
    let q = quad[vid];

    let ndc0 = world_to_ndc(inst.p0);
    let ndc1 = world_to_ndc(inst.p1);
    let px0 = ndc0 * cam.viewport_px * 0.5;
    let px1 = ndc1 * cam.viewport_px * 0.5;
    let dir_px = px1 - px0;
    let seg_len = max(length(dir_px), 0.001);
    let n_px = vec2<f32>(-dir_px.y / seg_len, dir_px.x / seg_len);

    let is_cliff = select(0.0, 1.0, inst.color.r > 0.75);
    let half_w_px = mix(1.05, 2.35, is_cliff);
    let offset_ndc = (n_px * q.y * half_w_px) / max(cam.viewport_px * 0.5, vec2<f32>(1.0, 1.0));
    let pos_ndc = mix(ndc0, ndc1, q.x) + offset_ndc;

    var out: EdgeVarying;
    out.clip_pos = vec4<f32>(pos_ndc, 0.0, 1.0);
    out.color = inst.color;
    out.edge_dist = q.y;
    return out;
}

@fragment
fn fs_edge(in: EdgeVarying) -> @location(0) vec4<f32> {
    let aa = 1.0 - smoothstep(0.48, 1.0, abs(in.edge_dist));
    return vec4<f32>(in.color.rgb, in.color.a * aa);
}

// -----------------------------------------------------------------------------
// PASS 3: Instanced Signed Distance Field (SDF) Compound Glyphs & Alert/SMARTS Halos
// -----------------------------------------------------------------------------

struct NodeInput {
    @location(0) pos: vec2<f32>,
    @location(1) radius_halo: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct NodeVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) halo_kind: f32,
};

@vertex
fn vs_node(@builtin(vertex_index) vid: u32, inst: NodeInput) -> NodeVarying {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let uv = corners[vid];
    let halo_expand = select(1.35, 2.02, inst.radius_halo.y > 0.5);
    let r_world = inst.radius_halo.x * halo_expand;
    let world_pt = inst.pos + uv * r_world;

    var out: NodeVarying;
    out.clip_pos = vec4<f32>(world_to_ndc(world_pt), 0.0, 1.0);
    out.local_uv = uv * halo_expand;
    out.color = inst.color;
    out.halo_kind = inst.radius_halo.y;
    return out;
}

@fragment
fn fs_node(in: NodeVarying) -> @location(0) vec4<f32> {
    let r = length(in.local_uv);
    let fw = max(fwidth(r), 0.025);

    let core_alpha = 1.0 - smoothstep(1.0 - fw, 1.0 + fw, r);
    let rim_t = smoothstep(0.74 - fw, 0.84 + fw, r);
    let lit = in.color.rgb * (1.12 - 0.22 * r);
    var rgb = mix(lit, vec3<f32>(0.05, 0.08, 0.14), rim_t * 0.82);
    var alpha = core_alpha * in.color.a;

    // Halo rings: 1.0 = Warhead, 2.0 = PAINS, 3.0 = Selected, 4.0 = Pareto, 5.0 = SMARTS Substructure Hit
    if (in.halo_kind > 0.5) {
        let is_pulsing = (in.halo_kind > 2.5 && in.halo_kind < 3.5) || in.halo_kind > 4.5;
        let pulse = select(0.0, 0.08 * sin(cam.time_sec * 4.2), is_pulsing);
        let ring_center = 1.44 + pulse;
        let ring_band = 1.0 - smoothstep(0.09, 0.20 + fw, abs(r - ring_center));

        var halo_rgb = vec3<f32>(0.22, 0.88, 0.98); // 3.0 Selected cyan
        if (in.halo_kind > 0.5 && in.halo_kind < 1.5) {
            halo_rgb = vec3<f32>(0.98, 0.68, 0.16); // 1.0 Covalent/Metabolic amber
        } else if (in.halo_kind >= 1.5 && in.halo_kind < 2.5) {
            halo_rgb = vec3<f32>(0.96, 0.24, 0.32); // 2.0 PAINS crimson
        } else if (in.halo_kind >= 3.5 && in.halo_kind < 4.5) {
            halo_rgb = vec3<f32>(0.32, 0.96, 0.64); // 4.0 Pareto emerald
        } else if (in.halo_kind >= 4.5) {
            halo_rgb = vec3<f32>(0.22, 0.74, 0.98); // 5.0 SMARTS hit sky-cyan glow
        }

        rgb = mix(rgb, halo_rgb, ring_band * (1.0 - core_alpha * 0.62));
        alpha = max(alpha, ring_band * 0.94);
    }

    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(rgb, alpha);
}
