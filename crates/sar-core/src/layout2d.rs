//! Deterministic 2D skeletal coordinate layout (`1.50 Å` bond length, fused SSSR polygon bridging,
//! `120°`/`180°` chain placement, spring/steric relaxation, principal-axis alignment) and
//! publication-ready `.svg` vector export with optional PAINS and SMARTS substructure highlights.

use crate::mol::{Element, Hybridization, Molecule};
use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};

pub const BOND_LEN_2D: f32 = 1.50;

/// Computes deterministic 2D skeletal coordinates in Angstroms (`pos2d`) for `mol`.
pub fn compute_2d_coords(mol: &mut Molecule) {
    let n = mol.atoms.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        mol.atoms[0].pos2d = [0.0, 0.0];
        return;
    }

    let mut pos = vec![[0.0f32, 0.0f32]; n];
    let mut placed = vec![false; n];
    let mut placed_rings = vec![false; mol.rings.len()];

    let mut comp_offset_x = 0.0f32;

    for start_seed in 0..n {
        if placed[start_seed] {
            continue;
        }

        let mut comp_atoms = Vec::new();
        let mut seen = vec![false; n];
        let mut cq = VecDeque::new();
        seen[start_seed] = true;
        cq.push_back(start_seed);
        while let Some(u) = cq.pop_front() {
            comp_atoms.push(u);
            for &(v, _) in &mol.adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    cq.push_back(v);
                }
            }
        }

        let seed_ring = mol
            .rings
            .iter()
            .enumerate()
            .filter(|&(ri, r)| !placed_rings[ri] && r.iter().any(|&a| seen[a]))
            .max_by_key(|(_, r)| {
                let shared: usize = r
                    .iter()
                    .map(|&a| mol.rings.iter().filter(|other| other.contains(&a)).count())
                    .sum();
                if r.len() >= 10 { 500 + r.len() } else { (r.len() == 6) as usize * 100 + shared * 10 + r.len() }
            })
            .map(|(ri, _)| ri);

        if let Some(ri0) = seed_ring {
            place_regular_polygon(&mol.rings[ri0], &mut pos, &mut placed);
            placed_rings[ri0] = true;
        } else {
            let root = *comp_atoms
                .iter()
                .max_by_key(|&&u| mol.adj[u].len())
                .unwrap_or(&start_seed);
            pos[root] = [0.0, 0.0];
            placed[root] = true;
            if let Some(&(nbr, _)) = mol.adj[root].first() {
                pos[nbr] = [BOND_LEN_2D, 0.0];
                placed[nbr] = true;
            }
        }

        let mut progress = true;
        while progress {
            progress = false;

            for ri in 0..mol.rings.len() {
                if placed_rings[ri] {
                    continue;
                }
                let ring = &mol.rings[ri];
                if !ring.iter().any(|&a| seen[a]) {
                    continue;
                }
                let placed_in_ring: Vec<usize> =
                    (0..ring.len()).filter(|&k| placed[ring[k]]).collect();
                if placed_in_ring.len() >= 2 {
                    fuse_ring_on_shared_edge(ring, &mol.rings, &mut pos, &mut placed);
                    placed_rings[ri] = true;
                    progress = true;
                } else if placed_in_ring.len() == 1 {
                    let k0 = placed_in_ring[0];
                    let anchor = ring[k0];
                    let out_dir = outward_direction(anchor, &mol.adj[anchor], &pos, &placed);
                    place_spiro_ring(ring, k0, out_dir, &mut pos, &mut placed);
                    placed_rings[ri] = true;
                    progress = true;
                }
            }

            if progress {
                continue;
            }

            let placed_snapshot: Vec<usize> =
                comp_atoms.iter().copied().filter(|&u| placed[u]).collect();
            for u in placed_snapshot {
                let unplaced_nbrs: Vec<usize> = mol.adj[u]
                    .iter()
                    .map(|&(v, _)| v)
                    .filter(|&v| !placed[v])
                    .collect();
                if unplaced_nbrs.is_empty() {
                    continue;
                }

                let placed_nbrs: Vec<usize> = mol.adj[u]
                    .iter()
                    .map(|&(v, _)| v)
                    .filter(|&v| placed[v])
                    .collect();

                let base_angles = compute_substituent_angles(
                    u,
                    &placed_nbrs,
                    unplaced_nbrs.len(),
                    mol.atoms[u].hyb,
                    &pos,
                    mol,
                );

                for (slot, &v) in unplaced_nbrs.iter().enumerate() {
                    if placed[v] {
                        continue;
                    }
                    let theta = base_angles[slot];
                    pos[v] = [
                        pos[u][0] + BOND_LEN_2D * theta.cos(),
                        pos[u][1] + BOND_LEN_2D * theta.sin(),
                    ];
                    placed[v] = true;
                    progress = true;

                    for ri in 0..mol.rings.len() {
                        if !placed_rings[ri] {
                            if let Some(k0) = mol.rings[ri].iter().position(|&ra| ra == v) {
                                let dir = [theta.cos(), theta.sin()];
                                place_spiro_ring(&mol.rings[ri], k0, dir, &mut pos, &mut placed);
                                placed_rings[ri] = true;
                            }
                        }
                    }
                }
            }
        }

        relax_component(mol, &comp_atoms, &mut pos);
        align_principal_axis(&comp_atoms, &mut pos);

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut mean_y = 0.0f32;
        for &u in &comp_atoms {
            min_x = min_x.min(pos[u][0]);
            max_x = max_x.max(pos[u][0]);
            mean_y += pos[u][1];
        }
        mean_y /= comp_atoms.len() as f32;
        for &u in &comp_atoms {
            pos[u][0] = pos[u][0] - min_x + comp_offset_x;
            pos[u][1] -= mean_y;
        }
        comp_offset_x += (max_x - min_x) + 3.0;
    }

    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for p in &pos {
        cx += p[0];
        cy += p[1];
    }
    cx /= n as f32;
    cy /= n as f32;
    for (i, a) in mol.atoms.iter_mut().enumerate() {
        a.pos2d = [pos[i][0] - cx, pos[i][1] - cy];
    }
}

fn place_regular_polygon(ring: &[usize], pos: &mut [[f32; 2]], placed: &mut [bool]) {
    let k = ring.len();
    let radius = BOND_LEN_2D / (2.0 * (PI / (k as f32)).sin());
    let phase = PI / 2.0;
    for (i, &ai) in ring.iter().enumerate() {
        let ang = phase + TAU * (i as f32) / (k as f32);
        pos[ai] = [radius * ang.cos(), radius * ang.sin()];
        placed[ai] = true;
    }
}

fn fuse_ring_on_shared_edge(
    ring: &[usize],
    all_rings: &[Vec<usize>],
    pos: &mut [[f32; 2]],
    placed: &mut [bool],
) {
    let k = ring.len();
    let mut shared_edge = None;
    for idx in 0..k {
        let a = ring[idx];
        let b = ring[(idx + 1) % k];
        if placed[a] && placed[b] {
            shared_edge = Some(idx);
            break;
        }
    }
    let Some(idx0) = shared_edge else {
        if let Some(k0) = (0..k).find(|&i| placed[ring[i]]) {
            place_spiro_ring(ring, k0, [1.0, 0.0], pos, placed);
        }
        return;
    };

    let u = ring[idx0];
    let v = ring[(idx0 + 1) % k];
    let dx = pos[v][0] - pos[u][0];
    let dy = pos[v][1] - pos[u][1];
    let edge_len = (dx * dx + dy * dy).sqrt().max(0.5);
    let mid = [(pos[u][0] + pos[v][0]) * 0.5, (pos[u][1] + pos[v][1]) * 0.5];

    let mut pcx = 0.0f32;
    let mut pcy = 0.0f32;
    let mut pcnt = 0usize;
    for other in all_rings {
        if other.contains(&u)
            && other.contains(&v)
            && other.iter().any(|&ai| ai != u && ai != v && !ring.contains(&ai) && placed[ai])
        {
            for &ai in other {
                if ai != u && ai != v && placed[ai] {
                    pcx += pos[ai][0];
                    pcy += pos[ai][1];
                    pcnt += 1;
                }
            }
        }
    }
    if pcnt == 0 {
        for (ai, &is_p) in placed.iter().enumerate() {
            if is_p && ai != u && ai != v {
                pcx += pos[ai][0];
                pcy += pos[ai][1];
                pcnt += 1;
            }
        }
    }

    let n1 = [-dy / edge_len, dx / edge_len];
    let n2 = [dy / edge_len, -dx / edge_len];
    let normal = if pcnt > 0 {
        let to_old = [pcx / (pcnt as f32) - mid[0], pcy / (pcnt as f32) - mid[1]];
        if n1[0] * to_old[0] + n1[1] * to_old[1] < n2[0] * to_old[0] + n2[1] * to_old[1] {
            n1
        } else {
            n2
        }
    } else {
        n1
    };

    let half_angle = PI / (k as f32);
    let apothem = (edge_len * 0.5) / half_angle.tan();
    let radius = (edge_len * 0.5) / half_angle.sin();
    let center = [mid[0] + normal[0] * apothem, mid[1] + normal[1] * apothem];

    let ang_u = (pos[u][1] - center[1]).atan2(pos[u][0] - center[0]);
    let ang_v = (pos[v][1] - center[1]).atan2(pos[v][0] - center[0]);
    let mut delta = ang_v - ang_u;
    while delta <= -PI {
        delta += TAU;
    }
    while delta > PI {
        delta -= TAU;
    }
    let step = if delta >= 0.0 {
        TAU / (k as f32)
    } else {
        -TAU / (k as f32)
    };

    for offset in 0..k {
        let r_idx = (idx0 + offset) % k;
        let ai = ring[r_idx];
        if !placed[ai] {
            let ang = ang_u + step * (offset as f32);
            pos[ai] = [center[0] + radius * ang.cos(), center[1] + radius * ang.sin()];
            placed[ai] = true;
        }
    }
}

fn place_spiro_ring(
    ring: &[usize],
    anchor_k: usize,
    dir: [f32; 2],
    pos: &mut [[f32; 2]],
    placed: &mut [bool],
) {
    let k = ring.len();
    let anchor = ring[anchor_k];
    let radius = BOND_LEN_2D / (2.0 * (PI / (k as f32)).sin());
    let dnorm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-4);
    let ux = dir[0] / dnorm;
    let uy = dir[1] / dnorm;
    let center = [pos[anchor][0] + ux * radius, pos[anchor][1] + uy * radius];
    let ang0 = (pos[anchor][1] - center[1]).atan2(pos[anchor][0] - center[0]);
    let step = TAU / (k as f32);
    for offset in 1..k {
        let ai = ring[(anchor_k + offset) % k];
        if !placed[ai] {
            let ang = ang0 + step * (offset as f32);
            pos[ai] = [center[0] + radius * ang.cos(), center[1] + radius * ang.sin()];
            placed[ai] = true;
        }
    }
}

fn outward_direction(
    u: usize,
    nbrs: &[(usize, usize)],
    pos: &[[f32; 2]],
    placed: &[bool],
) -> [f32; 2] {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut cnt = 0usize;
    for &(v, _) in nbrs {
        if placed[v] {
            sx += pos[u][0] - pos[v][0];
            sy += pos[u][1] - pos[v][1];
            cnt += 1;
        }
    }
    if cnt == 0 || (sx.abs() < 1e-4 && sy.abs() < 1e-4) {
        [1.0, 0.0]
    } else {
        let len = (sx * sx + sy * sy).sqrt();
        [sx / len, sy / len]
    }
}

fn compute_substituent_angles(
    u: usize,
    placed_nbrs: &[usize],
    count: usize,
    hyb: Hybridization,
    pos: &[[f32; 2]],
    mol: &Molecule,
) -> Vec<f32> {
    if placed_nbrs.is_empty() {
        return (0..count)
            .map(|i| TAU * (i as f32) / (count.max(1) as f32))
            .collect();
    }

    if placed_nbrs.len() == 1 {
        let p = placed_nbrs[0];
        let incoming = (pos[u][1] - pos[p][1]).atan2(pos[u][0] - pos[p][0]);
        if hyb == Hybridization::Sp && count == 1 {
            return vec![incoming];
        }
        if count == 1 {
            let gp_opt = mol.adj[p]
                .iter()
                .map(|&(w, _)| w)
                .find(|&w| w != u && (pos[w][0] != 0.0 || pos[w][1] != 0.0));
            let bend = PI / 3.0;
            if let Some(gp) = gp_opt {
                let prev_dir = (pos[p][1] - pos[gp][1]).atan2(pos[p][0] - pos[gp][0]);
                let mut turn = incoming - prev_dir;
                while turn <= -PI {
                    turn += TAU;
                }
                while turn > PI {
                    turn -= TAU;
                }
                return vec![if turn >= 0.0 {
                    incoming - bend
                } else {
                    incoming + bend
                }];
            }
            return vec![incoming + bend];
        } else if count == 2 {
            return vec![incoming + PI / 3.0, incoming - PI / 3.0];
        } else {
            return vec![incoming, incoming + PI * 0.52, incoming - PI * 0.52];
        }
    }

    let out = outward_direction(u, &mol.adj[u], pos, &vec![true; mol.atoms.len()]);
    let bisector = out[1].atan2(out[0]);
    if count == 1 {
        vec![bisector]
    } else if count == 2 {
        vec![bisector - PI / 4.2, bisector + PI / 4.2]
    } else {
        (0..count)
            .map(|k| bisector + ((k as f32) - 1.0) * (PI / 3.5))
            .collect()
    }
}

fn relax_component(mol: &Molecule, comp_atoms: &[usize], pos: &mut [[f32; 2]]) {
    if comp_atoms.len() <= 3 {
        return;
    }
    for _iter in 0..28 {
        let mut disp = vec![[0.0f32, 0.0f32]; mol.atoms.len()];

        for b in &mol.bonds {
            let dx = pos[b.b][0] - pos[b.a][0];
            let dy = pos[b.b][1] - pos[b.a][1];
            let d = (dx * dx + dy * dy).sqrt().max(0.05);
            let err = (d - BOND_LEN_2D) * 0.25;
            let fx = (dx / d) * err;
            let fy = (dy / d) * err;
            let wa = if mol.atoms[b.a].in_ring { 0.15 } else { 0.5 };
            let wb = if mol.atoms[b.b].in_ring { 0.15 } else { 0.5 };
            disp[b.a][0] += fx * wa;
            disp[b.a][1] += fy * wa;
            disp[b.b][0] -= fx * wb;
            disp[b.b][1] -= fy * wb;
        }

        for (ii, &u) in comp_atoms.iter().enumerate() {
            for &v in &comp_atoms[ii + 1..] {
                if mol.bond_between(u, v).is_some() {
                    continue;
                }
                let dx = pos[v][0] - pos[u][0];
                let dy = pos[v][1] - pos[u][1];
                let d2 = dx * dx + dy * dy;
                if d2 < 1.32 {
                    let d = d2.sqrt().max(0.08);
                    let push = (1.15 - d).max(0.0) * 0.22;
                    let ux = if d > 0.09 { dx / d } else { 0.707 };
                    let uy = if d > 0.09 { dy / d } else { 0.707 };
                    let wu = if mol.atoms[u].in_ring { 0.08 } else { 0.65 };
                    let wv = if mol.atoms[v].in_ring { 0.08 } else { 0.65 };
                    disp[u][0] -= ux * push * wu;
                    disp[u][1] -= uy * push * wu;
                    disp[v][0] += ux * push * wv;
                    disp[v][1] += uy * push * wv;
                }
            }
        }

        for &u in comp_atoms {
            pos[u][0] += disp[u][0].clamp(-0.22, 0.22);
            pos[u][1] += disp[u][1].clamp(-0.22, 0.22);
        }
    }
}

fn align_principal_axis(comp_atoms: &[usize], pos: &mut [[f32; 2]]) {
    let n = comp_atoms.len() as f32;
    if n < 2.0 {
        return;
    }
    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    for &u in comp_atoms {
        mx += pos[u][0];
        my += pos[u][1];
    }
    mx /= n;
    my /= n;

    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;
    for &u in comp_atoms {
        let x = pos[u][0] - mx;
        let y = pos[u][1] - my;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let c = (-theta).cos();
    let s = (-theta).sin();
    for &u in comp_atoms {
        let x = pos[u][0] - mx;
        let y = pos[u][1] - my;
        pos[u][0] = x * c - y * s;
        pos[u][1] = x * s + y * c;
    }
}

/// Renders a standalone publication-quality 2D skeletal `.svg` diagram of `mol`,
/// optionally highlighting `highlight_atoms` (e.g., PAINS or covalent warhead substructure).
pub fn render_molecule_svg(
    mol: &Molecule,
    width: u32,
    height: u32,
    dark_theme: bool,
    highlight_atoms: &[usize],
) -> String {
    render_molecule_svg_with_smarts(mol, width, height, dark_theme, highlight_atoms, &[], &[])
}

/// Renders a standalone 2D skeletal `.svg` diagram overlaying both PAINS/alert atoms (`#ef4444`)
/// and Daylight SMARTS substructure atom & bond matches (`#38bdf8`).
pub fn render_molecule_svg_with_smarts(
    mol: &Molecule,
    width: u32,
    height: u32,
    dark_theme: bool,
    alert_atoms: &[usize],
    smarts_atoms: &[usize],
    smarts_bonds: &[usize],
) -> String {
    let w = width.max(200) as f32;
    let h = height.max(160) as f32;
    let bg = if dark_theme { "#0f172a" } else { "#ffffff" };
    let bond_col = if dark_theme { "#cbd5e1" } else { "#1e293b" };
    let sub_col = if dark_theme { "#94a3b8" } else { "#475569" };

    if mol.atoms.is_empty() {
        return format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\"><rect width=\"100%\" height=\"100%\" fill=\"{bg}\"/></svg>"
        );
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for a in &mol.atoms {
        min_x = min_x.min(a.pos2d[0]);
        max_x = max_x.max(a.pos2d[0]);
        min_y = min_y.min(a.pos2d[1]);
        max_y = max_y.max(a.pos2d[1]);
    }
    let pad = 34.0f32;
    let footer_h = 30.0f32;
    let span_x = (max_x - min_x).max(2.0);
    let span_y = (max_y - min_y).max(2.0);
    let scale = ((w - 2.0 * pad) / span_x)
        .min((h - 2.0 * pad - footer_h) / span_y)
        .clamp(12.0, 42.0);

    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    let map_pt = |p: [f32; 2]| -> [f32; 2] {
        [
            w * 0.5 + (p[0] - cx) * scale,
            (h - footer_h) * 0.5 - (p[1] - cy) * scale,
        ]
    };

    let mut svg = String::with_capacity(4608);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.0} {:.0}\" width=\"{:.0}\" height=\"{:.0}\">\n",
        w, h, w, h
    ));
    svg.push_str(&format!(
        "  <rect width=\"100%\" height=\"100%\" rx=\"10\" fill=\"{bg}\"/>\n"
    ));

    // SMARTS bond glow underlays
    for &bi in smarts_bonds {
        if let Some(b) = mol.bonds.get(bi) {
            let p1 = map_pt(mol.atoms[b.a].pos2d);
            let p2 = map_pt(mol.atoms[b.b].pos2d);
            svg.push_str(&format!(
                "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#38bdf8\" stroke-opacity=\"0.38\" stroke-width=\"{:.1}\" stroke-linecap=\"round\"/>\n",
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                (scale * 0.42).clamp(6.0, 12.0)
            ));
        }
    }

    // SMARTS atom halos
    for &ai in smarts_atoms {
        if let Some(a) = mol.atoms.get(ai) {
            let p = map_pt(a.pos2d);
            svg.push_str(&format!(
                "  <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.1}\" fill=\"#38bdf8\" fill-opacity=\"0.26\" stroke=\"#38bdf8\" stroke-width=\"1.3\"/>\n",
                p[0],
                p[1],
                scale * 0.38
            ));
        }
    }

    // PAINS / warhead alert halos
    for &ai in alert_atoms {
        if let Some(a) = mol.atoms.get(ai) {
            let p = map_pt(a.pos2d);
            svg.push_str(&format!(
                "  <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.1}\" fill=\"#ef4444\" fill-opacity=\"0.22\" stroke=\"#ef4444\" stroke-width=\"1.2\"/>\n",
                p[0],
                p[1],
                scale * 0.42
            ));
        }
    }

    // Skeletal bonds
    for b in &mol.bonds {
        let p1 = map_pt(mol.atoms[b.a].pos2d);
        let p2 = map_pt(mol.atoms[b.b].pos2d);
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let blen = (dx * dx + dy * dy).sqrt().max(1.0);
        let nx = -dy / blen;
        let ny = dx / blen;

        if b.kekule_order == 1 {
            svg.push_str(&format!(
                "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{bond_col}\" stroke-width=\"1.9\" stroke-linecap=\"round\"/>\n",
                p1[0], p1[1], p2[0], p2[1]
            ));
        } else if b.kekule_order == 2 {
            let shared_ring = mol
                .rings
                .iter()
                .find(|r| r.contains(&b.a) && r.contains(&b.b));
            if let Some(ring) = shared_ring {
                let mut rcx = 0.0f32;
                let mut rcy = 0.0f32;
                for &ra in ring {
                    let rp = map_pt(mol.atoms[ra].pos2d);
                    rcx += rp[0];
                    rcy += rp[1];
                }
                rcx /= ring.len() as f32;
                rcy /= ring.len() as f32;
                let mx = (p1[0] + p2[0]) * 0.5;
                let my = (p1[1] + p2[1]) * 0.5;
                let sign = if (rcx - mx) * nx + (rcy - my) * ny >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let off = (scale * 0.14).clamp(3.0, 5.2) * sign;
                let trim = 0.15f32;
                let q1 = [p1[0] + dx * trim + nx * off, p1[1] + dy * trim + ny * off];
                let q2 = [p2[0] - dx * trim + nx * off, p2[1] - dy * trim + ny * off];
                svg.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{bond_col}\" stroke-width=\"1.9\" stroke-linecap=\"round\"/>\n",
                    p1[0], p1[1], p2[0], p2[1]
                ));
                svg.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{bond_col}\" stroke-width=\"1.7\" stroke-linecap=\"round\"/>\n",
                    q1[0], q1[1], q2[0], q2[1]
                ));
            } else {
                let off = (scale * 0.085).clamp(2.0, 3.4);
                svg.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{bond_col}\" stroke-width=\"1.8\" stroke-linecap=\"round\"/>\n",
                    p1[0] + nx * off, p1[1] + ny * off, p2[0] + nx * off, p2[1] + ny * off
                ));
                svg.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{bond_col}\" stroke-width=\"1.8\" stroke-linecap=\"round\"/>\n",
                    p1[0] - nx * off, p1[1] - ny * off, p2[0] - nx * off, p2[1] - ny * off
                ));
            }
        } else {
            let off = (scale * 0.12).clamp(2.6, 4.2);
            for mult in [-1.0f32, 0.0, 1.0] {
                svg.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{bond_col}\" stroke-width=\"1.6\" stroke-linecap=\"round\"/>\n",
                    p1[0] + nx * off * mult,
                    p1[1] + ny * off * mult,
                    p2[0] + nx * off * mult,
                    p2[1] + ny * off * mult
                ));
            }
        }
    }

    for a in &mol.atoms {
        let show_label = a.elem != Element::C || a.charge != 0;
        if !show_label {
            continue;
        }
        let p = map_pt(a.pos2d);
        let [r, g, b] = a.elem.cpk_rgb(dark_theme);
        let mut label = a.elem.symbol().to_string();
        let th = a.total_h();
        if th == 1 && matches!(a.elem, Element::N | Element::O | Element::S) {
            label.push('H');
        } else if th > 1 && matches!(a.elem, Element::N | Element::O | Element::S) {
            label.push_str(&format!("H{}", th));
        }
        if a.charge > 0 {
            label.push('+');
        } else if a.charge < 0 {
            label.push('-');
        }

        let mask_r = (scale * 0.28).clamp(6.5, 11.5);
        svg.push_str(&format!(
            "  <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.1}\" fill=\"{bg}\"/>\n",
            p[0], p[1], mask_r
        ));
        svg.push_str(&format!(
            "  <text x=\"{:.2}\" y=\"{:.2}\" fill=\"rgb({r},{g},{b})\" font-family=\"ui-sans-serif, system-ui, sans-serif\" font-size=\"{:.1}\" font-weight=\"700\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
            p[0],
            p[1],
            (scale * 0.42).clamp(10.0, 14.5),
            label
        ));
    }

    let caption = format!(
        "{} ({})  |  {}  |  {:.2} Da  |  pAct {:.2}",
        if mol.name.is_empty() { &mol.id } else { &mol.name },
        mol.target,
        mol.molecular_formula(),
        mol.molecular_weight(),
        mol.pact
    );
    svg.push_str(&format!(
        "  <text x=\"{:.1}\" y=\"{:.1}\" fill=\"{sub_col}\" font-family=\"ui-monospace, monospace\" font-size=\"11\" text-anchor=\"middle\">{}</text>\n",
        w * 0.5,
        h - 10.0,
        xml_escape(&caption)
    ));
    svg.push_str("</svg>\n");
    svg
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
