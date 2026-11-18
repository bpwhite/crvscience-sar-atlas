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

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by Cargo"));

    // Always generate the 64x64 RGBA icon for egui ViewportBuilder::with_icon on every OS.
    let icon64 = render_sar_atlas_icon(64);
    fs::write(out_dir.join("icon_64.rgba"), &icon64).expect("write icon_64.rgba");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let sizes = [16u32, 32, 48, 64, 256];
    let ico_path = out_dir.join("sar_atlas.ico");
    write_multi_size_ico(&ico_path, &sizes).expect("write sar_atlas.ico");

    let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap_or_else(|_| "3".to_string());
    let minor = env::var("CARGO_PKG_VERSION_MINOR").unwrap_or_else(|_| "8".to_string());
    let patch = env::var("CARGO_PKG_VERSION_PATCH").unwrap_or_else(|_| "0".to_string());
    let ver_commas = format!("{major},{minor},{patch},0");
    let ver_dots = format!("{major}.{minor}.{patch}.0");
    let ico_escaped = ico_path.to_string_lossy().replace('\\', "/");

    let rc_text = format!(
        r#"1 ICON "{ico_escaped}"

1 VERSIONINFO
FILEVERSION {ver_commas}
PRODUCTVERSION {ver_commas}
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
            VALUE "CompanyName", "CRV Science Network\0"
            VALUE "FileDescription", "SAR-Atlas: Small-Molecule Cheminformatics, RBFE, CYP & Oral PBPK Workbench\0"
            VALUE "FileVersion", "{ver_dots}\0"
            VALUE "InternalName", "sar-atlas\0"
            VALUE "LegalCopyright", "Copyright CRV Science Network (GPL-3.0-or-later)\0"
            VALUE "OriginalFilename", "11_SAR-Atlas.exe\0"
            VALUE "ProductName", "SAR-Atlas\0"
            VALUE "ProductVersion", "{ver_dots}\0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0409, 1200
    END
END
"#
    );

    let rc_path = out_dir.join("sar_atlas.rc");
    fs::write(&rc_path, rc_text).expect("write sar_atlas.rc");

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "msvc" {
        if let Some(rc_exe) = find_windows_rc_exe() {
            let res_path = out_dir.join("sar_atlas.res");
            let status = Command::new(&rc_exe)
                .args(["/nologo", "/fo"])
                .arg(&res_path)
                .arg(&rc_path)
                .status();
            if matches!(status, Ok(s) if s.success()) {
                println!("cargo:rustc-link-arg-bin=sar-atlas={}", res_path.display());
            } else {
                println!("cargo:warning=rc.exe failed compiling sar_atlas.rc; skipping PE resources");
            }
        } else {
            println!("cargo:warning=rc.exe not found; skipping PE resources");
        }
    } else {
        let obj_path = out_dir.join("sar_atlas_res.o");
        let compiled = ["windres", "x86_64-w64-mingw32-windres"]
            .into_iter()
            .any(|tool| {
                let status = Command::new(tool)
                    .arg("-i")
                    .arg(&rc_path)
                    .args(["-O", "coff", "-o"])
                    .arg(&obj_path)
                    .status();
                matches!(status, Ok(s) if s.success())
            });
        if compiled {
            println!("cargo:rustc-link-arg-bin=sar-atlas={}", obj_path.display());
        } else {
            println!("cargo:warning=windres not found/failed; skipping PE resources");
        }
    }
}

fn find_windows_rc_exe() -> Option<PathBuf> {
    if let Ok(path_env) = env::var("PATH") {
        for dir in env::split_paths(&path_env) {
            let cand = dir.join("rc.exe");
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    let kits_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if let Ok(entries) = fs::read_dir(kits_root) {
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|d| d.path().join("x64").join("rc.exe")))
            .filter(|p| p.is_file())
            .collect();
        versions.sort();
        if let Some(latest) = versions.pop() {
            return Some(latest);
        }
    }
    None
}

fn write_multi_size_ico(path: &Path, sizes: &[u32]) -> std::io::Result<()> {
    let count = sizes.len() as u16;
    let mut frames: Vec<(u32, Vec<u8>)> = Vec::with_capacity(sizes.len());
    for &sz in sizes {
        let rgba = render_sar_atlas_icon(sz);
        frames.push((sz, encode_dib_frame(sz, &rgba)));
    }

    let mut ico = Vec::new();
    // ICONDIR
    ico.extend_from_slice(&0u16.to_le_bytes());
    ico.extend_from_slice(&1u16.to_le_bytes());
    ico.extend_from_slice(&count.to_le_bytes());

    let mut offset = 6u32 + 16u32 * (sizes.len() as u32);
    for (sz, dib) in &frames {
        let wh = if *sz >= 256 { 0u8 } else { *sz as u8 };
        ico.push(wh);
        ico.push(wh);
        ico.push(0);
        ico.push(0);
        ico.extend_from_slice(&1u16.to_le_bytes());
        ico.extend_from_slice(&32u16.to_le_bytes());
        ico.extend_from_slice(&(dib.len() as u32).to_le_bytes());
        ico.extend_from_slice(&offset.to_le_bytes());
        offset += dib.len() as u32;
    }
    for (_, dib) in frames {
        ico.extend_from_slice(&dib);
    }
    fs::write(path, ico)
}

fn encode_dib_frame(sz: u32, rgba: &[u8]) -> Vec<u8> {
    let w = sz as usize;
    let h = sz as usize;
    let and_stride = ((w + 31) / 32) * 4;
    let mut out = Vec::with_capacity(40 + w * h * 4 + and_stride * h);

    // BITMAPINFOHEADER (40 bytes)
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(sz as i32).to_le_bytes());
    out.extend_from_slice(&((sz * 2) as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&((w * h * 4) as u32).to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    // Bottom-up BGRA XOR mask
    for y_inv in 0..h {
        let y = h - 1 - y_inv;
        for x in 0..w {
            let idx = (y * w + x) * 4;
            out.push(rgba[idx + 2]);
            out.push(rgba[idx + 1]);
            out.push(rgba[idx]);
            out.push(rgba[idx + 3]);
        }
    }

    // Bottom-up 1-bpp AND mask
    for y_inv in 0..h {
        let y = h - 1 - y_inv;
        let mut row = vec![0u8; and_stride];
        for x in 0..w {
            let a = rgba[(y * w + x) * 4 + 3];
            if a < 16 {
                row[x / 8] |= 1 << (7 - (x % 8));
            }
        }
        out.extend_from_slice(&row);
    }
    out
}

fn render_sar_atlas_icon(size: u32) -> Vec<u8> {
    let s = size as usize;
    let mut rgba = vec![0u8; s * s * 4];
    let aa = 1.6_f32 / (size as f32).max(16.0);

    // Aromatic benzene hexagon vertices in normalized [-1, 1] space
    let r_hex = 0.43_f32;
    let mut hex = [[0.0_f32; 2]; 6];
    for (k, pt) in hex.iter_mut().enumerate() {
        let ang = (k as f32) * std::f32::consts::FRAC_PI_3 - std::f32::consts::FRAC_PI_2;
        *pt = [r_hex * ang.cos(), r_hex * ang.sin()];
    }
    // Substituent vectors: upper-right emerald pharmacophore, lower-left coral/amber SALI cliff
    let sub_em_base = hex[1];
    let sub_em_tip = [sub_em_base[0] + 0.25, sub_em_base[1] - 0.14];
    let sub_cliff_base = hex[4];
    let sub_cliff_tip = [sub_cliff_base[0] - 0.25, sub_cliff_base[1] + 0.14];

    for py in 0..s {
        for px in 0..s {
            let nx = ((px as f32) + 0.5) / (size as f32) * 2.0 - 1.0;
            let ny = ((py as f32) + 0.5) / (size as f32) * 2.0 - 1.0;

            let d_box = sd_rounded_box(nx, ny, 0.86, 0.86, 0.22);
            let alpha_box = smooth_cov(d_box, aa);
            if alpha_box <= 0.001 {
                continue;
            }

            // Deep navy gradient backdrop (#0D1526 -> #132342)
            let g = ((ny + 1.0) * 0.5).clamp(0.0, 1.0);
            let mut col = [
                13.0 + 6.0 * g,
                21.0 + 14.0 * g,
                38.0 + 28.0 * g,
            ];

            // Cyan border (#22D3EE)
            let border_cov = smooth_cov((d_box + 0.035).abs() - 0.025, aa);
            col = blend_rgb(col, [34.0, 211.0, 238.0], border_cov * 0.90);

            // Subtle inner aromatic ring glow
            let d_glow = (nx * nx + ny * ny).sqrt() - 0.28;
            let glow_cov = (-d_glow * d_glow * 14.0).exp() * 0.22;
            col = blend_rgb(col, [56.0, 189.0, 248.0], glow_cov);

            // Substituent bonds
            let d_sb1 = sd_segment([nx, ny], sub_em_base, sub_em_tip) - 0.045;
            let d_sb2 = sd_segment([nx, ny], sub_cliff_base, sub_cliff_tip) - 0.045;
            col = blend_rgb(col, [16.0, 185.0, 129.0], smooth_cov(d_sb1, aa));
            col = blend_rgb(col, [244.0, 63.0, 94.0], smooth_cov(d_sb2, aa));

            // Hexagon outer bonds (#E2E8F0)
            let mut d_hex = f32::INFINITY;
            for k in 0..6 {
                let d = sd_segment([nx, ny], hex[k], hex[(k + 1) % 6]) - 0.052;
                d_hex = d_hex.min(d);
            }
            col = blend_rgb(col, [226.0, 232.0, 240.0], smooth_cov(d_hex, aa));

            // Alternating inner double-bond chords (#38BDF8)
            let mut d_dbl = f32::INFINITY;
            for k in [0usize, 2, 4] {
                let a = [hex[k][0] * 0.72, hex[k][1] * 0.72];
                let b = [hex[(k + 1) % 6][0] * 0.72, hex[(k + 1) % 6][1] * 0.72];
                d_dbl = d_dbl.min(sd_segment([nx, ny], a, b) - 0.032);
            }
            col = blend_rgb(col, [56.0, 189.0, 248.0], smooth_cov(d_dbl, aa));

            // Pharmacophore emerald node (#10B981) & SALI cliff coral/amber node (#F43F5E / #F59E0B)
            let d_em = ((nx - sub_em_tip[0]).powi(2) + (ny - sub_em_tip[1]).powi(2)).sqrt() - 0.13;
            let d_cliff =
                ((nx - sub_cliff_tip[0]).powi(2) + (ny - sub_cliff_tip[1]).powi(2)).sqrt() - 0.13;
            col = blend_rgb(col, [16.0, 185.0, 129.0], smooth_cov(d_em, aa));
            col = blend_rgb(col, [244.0, 63.0, 94.0], smooth_cov(d_cliff, aa));
            let d_cliff_core =
                ((nx - sub_cliff_tip[0]).powi(2) + (ny - sub_cliff_tip[1]).powi(2)).sqrt() - 0.055;
            col = blend_rgb(col, [245.0, 158.0, 11.0], smooth_cov(d_cliff_core, aa));

            let idx = (py * s + px) * 4;
            rgba[idx] = col[0].clamp(0.0, 255.0) as u8;
            rgba[idx + 1] = col[1].clamp(0.0, 255.0) as u8;
            rgba[idx + 2] = col[2].clamp(0.0, 255.0) as u8;
            rgba[idx + 3] = (alpha_box * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgba
}

fn sd_rounded_box(px: f32, py: f32, bx: f32, by: f32, r: f32) -> f32 {
    let qx = px.abs() - (bx - r);
    let qy = py.abs() - (by - r);
    let ox = qx.max(0.0);
    let oy = qy.max(0.0);
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0) - r
}

fn sd_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let pax = p[0] - a[0];
    let pay = p[1] - a[1];
    let bax = b[0] - a[0];
    let bay = b[1] - a[1];
    let denom = (bax * bax + bay * bay).max(1e-6);
    let h = ((pax * bax + pay * bay) / denom).clamp(0.0, 1.0);
    let dx = pax - bax * h;
    let dy = pay - bay * h;
    (dx * dx + dy * dy).sqrt()
}

fn smooth_cov(d: f32, aa: f32) -> f32 {
    ((-d / aa + 0.5).clamp(0.0, 1.0)).powf(1.0)
}

fn blend_rgb(dst: [f32; 3], src: [f32; 3], alpha: f32) -> [f32; 3] {
    let a = alpha.clamp(0.0, 1.0);
    [
        dst[0] * (1.0 - a) + src[0] * a,
        dst[1] * (1.0 - a) + src[1] * a,
        dst[2] * (1.0 - a) + src[2] * a,
    ]
}
