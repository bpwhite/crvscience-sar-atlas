//! OpenSMILES v1.0 (`http://opensmiles.org/opensmiles.html`) and MDL V2000 `.sdf` / `.mol`
//! parser and writer, plus `.smi` and `.csv` multi-compound batch ingestion.

use crate::layout2d::compute_2d_coords;
use crate::mol::{Atom, Bond, BondOrder, Element, Hybridization, Molecule};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MolIoError {
    #[error("Empty SMILES string")]
    EmptySmiles,
    #[error("Unexpected SMILES token '{0}' at byte {1}")]
    UnexpectedToken(char, usize),
    #[error("Unclosed bracket atom at byte {0}")]
    UnclosedBracket(usize),
    #[error("Unbalanced branch parenthesis in SMILES")]
    UnbalancedBranch,
    #[error("Unclosed ring closure {0} in SMILES")]
    UnclosedRing(u16),
    #[error("Malformed MDL V2000 record: {0}")]
    InvalidSdf(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Parses an OpenSMILES v1.0 string, perceives SSSR rings & implicit valences, and computes 2D coordinates.
pub fn parse_smiles(smiles: &str) -> Result<Molecule, MolIoError> {
    let trimmed = smiles.trim();
    if trimmed.is_empty() {
        return Err(MolIoError::EmptySmiles);
    }
    let core_smi = trimmed.split_whitespace().next().unwrap_or(trimmed);
    let mut mol = Molecule::new(core_smi);

    let bytes = core_smi.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    let mut prev_atom: Option<usize> = None;
    let mut branch_stack: Vec<Option<usize>> = Vec::new();
    let mut pending_bond: Option<BondOrder> = None;
    let mut open_rings: HashMap<u16, (usize, Option<BondOrder>)> = HashMap::new();

    while i < len {
        let ch = bytes[i] as char;
        match ch {
            '(' => {
                branch_stack.push(prev_atom);
                i += 1;
            }
            ')' => {
                prev_atom = branch_stack.pop().ok_or(MolIoError::UnbalancedBranch)?;
                pending_bond = None;
                i += 1;
            }
            '.' => {
                prev_atom = None;
                pending_bond = None;
                i += 1;
            }
            '-' => {
                pending_bond = Some(BondOrder::Single);
                i += 1;
            }
            '=' => {
                pending_bond = Some(BondOrder::Double);
                i += 1;
            }
            '#' => {
                pending_bond = Some(BondOrder::Triple);
                i += 1;
            }
            ':' => {
                pending_bond = Some(BondOrder::Aromatic);
                i += 1;
            }
            '/' | '\\' => {
                // Cis/trans directional single bond in OpenSMILES
                if pending_bond.is_none() {
                    pending_bond = Some(BondOrder::Single);
                }
                i += 1;
            }
            '%' => {
                if i + 2 >= len
                    || !(bytes[i + 1] as char).is_ascii_digit()
                    || !(bytes[i + 2] as char).is_ascii_digit()
                {
                    return Err(MolIoError::UnexpectedToken('%', i));
                }
                let rnum = ((bytes[i + 1] - b'0') as u16) * 10 + ((bytes[i + 2] - b'0') as u16);
                let cur = prev_atom.ok_or(MolIoError::UnexpectedToken('%', i))?;
                handle_ring_closure(&mut mol, cur, rnum, pending_bond.take(), &mut open_rings);
                i += 3;
            }
            '0'..='9' => {
                let rnum = (bytes[i] - b'0') as u16;
                let cur = prev_atom.ok_or(MolIoError::UnexpectedToken(ch, i))?;
                handle_ring_closure(&mut mol, cur, rnum, pending_bond.take(), &mut open_rings);
                i += 1;
            }
            '[' => {
                let start = i + 1;
                let mut end = start;
                while end < len && bytes[end] != b']' {
                    end += 1;
                }
                if end >= len {
                    return Err(MolIoError::UnclosedBracket(i));
                }
                let inner = &core_smi[start..end];
                let atom = parse_bracket_atom(inner, mol.atoms.len())?;
                let new_idx = mol.atoms.len();
                mol.atoms.push(atom);
                if let Some(pa) = prev_atom {
                    add_parsed_bond(&mut mol, pa, new_idx, pending_bond.take());
                } else {
                    pending_bond = None;
                }
                prev_atom = Some(new_idx);
                i = end + 1;
            }
            _ => {
                // Organic subset two-letter (`Cl`, `Br`) or one-letter (`B,C,N,O,P,S,F,I` / `b,c,n,o,p,s`)
                let (elem, aromatic, step) = if i + 1 < len {
                    let two = &core_smi[i..i + 2];
                    match two {
                        "Cl" => (Element::Cl, false, 2),
                        "Br" => (Element::Br, false, 2),
                        _ => parse_single_organic(ch, i)?,
                    }
                } else {
                    parse_single_organic(ch, i)?
                };

                // Collapse explicit `[H]` or unbracketed `H` onto parent heavy atom if attached
                let new_idx = mol.atoms.len();
                mol.atoms.push(Atom {
                    idx: new_idx,
                    elem,
                    charge: 0,
                    aromatic,
                    bracketed: false,
                    explicit_h: 0,
                    implicit_h: 0,
                    chiral: 0,
                    in_ring: false,
                    ring_sizes: Vec::new(),
                    hyb: Hybridization::Sp3,
                    pos2d: [0.0, 0.0],
                });
                if let Some(pa) = prev_atom {
                    add_parsed_bond(&mut mol, pa, new_idx, pending_bond.take());
                } else {
                    pending_bond = None;
                }
                prev_atom = Some(new_idx);
                i += step;
            }
        }
    }

    if !branch_stack.is_empty() {
        return Err(MolIoError::UnbalancedBranch);
    }
    if let Some(&rnum) = open_rings.keys().next() {
        return Err(MolIoError::UnclosedRing(rnum));
    }

    mol.sanitize();
    compute_2d_coords(&mut mol);
    Ok(mol)
}

fn parse_single_organic(ch: char, pos: usize) -> Result<(Element, bool, usize), MolIoError> {
    match ch {
        'B' => Ok((Element::B, false, 1)),
        'C' => Ok((Element::C, false, 1)),
        'N' => Ok((Element::N, false, 1)),
        'O' => Ok((Element::O, false, 1)),
        'P' => Ok((Element::P, false, 1)),
        'S' => Ok((Element::S, false, 1)),
        'F' => Ok((Element::F, false, 1)),
        'I' => Ok((Element::I, false, 1)),
        'H' => Ok((Element::H, false, 1)),
        'b' => Ok((Element::B, true, 1)),
        'c' => Ok((Element::C, true, 1)),
        'n' => Ok((Element::N, true, 1)),
        'o' => Ok((Element::O, true, 1)),
        'p' => Ok((Element::P, true, 1)),
        's' => Ok((Element::S, true, 1)),
        '*' => Ok((Element::Other(0), false, 1)),
        _ => Err(MolIoError::UnexpectedToken(ch, pos)),
    }
}

fn parse_bracket_atom(inner: &str, idx: usize) -> Result<Atom, MolIoError> {
    let bytes = inner.as_bytes();
    let n = bytes.len();
    let mut p = 0usize;

    // Optional isotope digits
    while p < n && (bytes[p] as char).is_ascii_digit() {
        p += 1;
    }
    if p >= n {
        return Err(MolIoError::UnclosedBracket(0));
    }

    // Element symbol (1 or 2 letters, e.g., `n`, `C`, `Cl`, `Br`, `Si`, `Se`, `se`, `H`, `*`)
    let first = bytes[p] as char;
    let aromatic = first.is_ascii_lowercase();
    let (elem, step) = if first == '*' {
        (Element::Other(0), 1)
    } else if p + 1 < n && (bytes[p + 1] as char).is_ascii_lowercase() && (bytes[p + 1] as char) != 'h' {
        let two = &inner[p..p + 2];
        if let Some(el) = Element::from_symbol(two) {
            (el, 2)
        } else if let Some(el) = Element::from_symbol(&inner[p..p + 1]) {
            (el, 1)
        } else {
            (Element::Other(6), 1)
        }
    } else if let Some(el) = Element::from_symbol(&inner[p..p + 1]) {
        (el, 1)
    } else {
        (Element::Other(6), 1)
    };
    p += step;

    let mut chiral = 0i8;
    let mut explicit_h = 0u8;
    let mut charge = 0i8;

    while p < n {
        match bytes[p] as char {
            '@' => {
                p += 1;
                if p < n && bytes[p] == b'@' {
                    chiral = 2;
                    p += 1;
                } else {
                    chiral = 1;
                }
            }
            'H' => {
                p += 1;
                if p < n && (bytes[p] as char).is_ascii_digit() {
                    let mut cnt = 0u8;
                    while p < n && (bytes[p] as char).is_ascii_digit() {
                        cnt = cnt.saturating_mul(10).saturating_add(bytes[p] - b'0');
                        p += 1;
                    }
                    explicit_h = cnt;
                } else {
                    explicit_h = 1;
                }
            }
            '+' => {
                p += 1;
                if p < n && bytes[p] == b'+' {
                    while p < n && bytes[p] == b'+' {
                        charge += 1;
                        p += 1;
                    }
                    charge += 1;
                } else if p < n && (bytes[p] as char).is_ascii_digit() {
                    let mut mag = 0i8;
                    while p < n && (bytes[p] as char).is_ascii_digit() {
                        mag = mag.saturating_mul(10).saturating_add((bytes[p] - b'0') as i8);
                        p += 1;
                    }
                    charge = mag;
                } else {
                    charge += 1;
                }
            }
            '-' => {
                p += 1;
                if p < n && bytes[p] == b'-' {
                    while p < n && bytes[p] == b'-' {
                        charge -= 1;
                        p += 1;
                    }
                    charge -= 1;
                } else if p < n && (bytes[p] as char).is_ascii_digit() {
                    let mut mag = 0i8;
                    while p < n && (bytes[p] as char).is_ascii_digit() {
                        mag = mag.saturating_mul(10).saturating_add((bytes[p] - b'0') as i8);
                        p += 1;
                    }
                    charge = -mag;
                } else {
                    charge -= 1;
                }
            }
            ':' => {
                // Atom-map class `:1` -> skip to end
                break;
            }
            _ => {
                p += 1;
            }
        }
    }

    Ok(Atom {
        idx,
        elem,
        charge,
        aromatic,
        bracketed: true,
        explicit_h,
        implicit_h: 0,
        chiral,
        in_ring: false,
        ring_sizes: Vec::new(),
        hyb: Hybridization::Sp3,
        pos2d: [0.0, 0.0],
    })
}

fn add_parsed_bond(mol: &mut Molecule, a: usize, b: usize, explicit_order: Option<BondOrder>) {
    if a == b || mol.bonds.iter().any(|eb| (eb.a == a && eb.b == b) || (eb.a == b && eb.b == a)) {
        return;
    }
    let both_arom = mol.atoms[a].aromatic && mol.atoms[b].aromatic;
    let order = explicit_order.unwrap_or(if both_arom {
        BondOrder::Aromatic
    } else {
        BondOrder::Single
    });
    let idx = mol.bonds.len();
    mol.bonds.push(Bond {
        idx,
        a,
        b,
        order,
        aromatic: order == BondOrder::Aromatic,
        in_ring: false,
        kekule_order: 1,
    });
}

fn handle_ring_closure(
    mol: &mut Molecule,
    cur: usize,
    rnum: u16,
    explicit_order: Option<BondOrder>,
    open_rings: &mut HashMap<u16, (usize, Option<BondOrder>)>,
) {
    if let Some((other_atom, open_order)) = open_rings.remove(&rnum) {
        let resolved = explicit_order.or(open_order);
        add_parsed_bond(mol, other_atom, cur, resolved);
    } else {
        open_rings.insert(rnum, (cur, explicit_order));
    }
}

/// Parses `.smi` or `.csv` / `.tsv` small-molecule text tables.
/// Supports headered CSVs (`id,name,target,metric,raw_nm,pact,smiles,ref`) or standard `SMILES Name pAct` `.smi` files.
pub fn parse_delimited_dataset(text: &str) -> Result<Vec<Molecule>, MolIoError> {
    let mut out = Vec::new();
    let mut lines = text.lines().filter(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with('#')
    });

    let Some(first_line) = lines.next() else {
        return Ok(out);
    };

    let is_csv = first_line.contains(',');
    let first_lower = first_line.to_ascii_lowercase();
    let has_header = first_lower.contains("smiles") || first_lower.starts_with("id,");

    let mut header_cols: Vec<String> = Vec::new();
    let mut pending_first = None;
    if has_header {
        header_cols = split_csv_row(first_line)
            .into_iter()
            .map(|s| s.trim().to_ascii_lowercase())
            .collect();
    } else {
        pending_first = Some(first_line);
    }

    for (row_idx, line) in pending_first.into_iter().chain(lines).enumerate() {
        if is_csv {
            let cols = split_csv_row(line);
            if cols.is_empty() {
                continue;
            }
            let find_col = |names: &[&str], fallback: usize| -> Option<&str> {
                if !header_cols.is_empty() {
                    for &nm in names {
                        if let Some(pos) = header_cols.iter().position(|h| h == nm) {
                            return cols.get(pos).map(|s| s.as_str());
                        }
                    }
                    None
                } else {
                    cols.get(fallback).map(|s| s.as_str())
                }
            };

            let smi = find_col(&["smiles", "canonical_smiles", "smi"], 0)
                .unwrap_or("")
                .trim();
            if smi.is_empty() {
                continue;
            }
            let mut mol = parse_smiles(smi)?;
            mol.id = find_col(&["id", "chembl_id", "compound_id"], 1)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("CMPD-{:04}", row_idx + 1));
            mol.name = find_col(&["name", "compound", "title"], 2)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| mol.id.clone());
            mol.target = find_col(&["target", "assay_target"], 3)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "Target-A".to_string());
            mol.metric = find_col(&["metric", "standard_type"], 4)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "IC50".to_string());
            if let Some(raw_s) = find_col(&["raw_nm", "ic50_nm", "ki_nm"], 5) {
                if let Ok(v) = raw_s.trim().parse::<f64>() {
                    mol.raw_nm = v.max(1e-6);
                    mol.pact = 9.0 - mol.raw_nm.log10();
                }
            }
            if let Some(pact_s) = find_col(&["pact", "pic50", "pki", "pchembl_value"], 6) {
                if let Ok(v) = pact_s.trim().parse::<f64>() {
                    mol.pact = v;
                    if mol.raw_nm == 100.0 {
                        mol.raw_nm = 10f64.powf(9.0 - v);
                    }
                }
            }
            if let Some(u) = find_col(&["ref", "source_url", "url"], 7) {
                mol.source_url = u.trim().to_string();
            }
            out.push(mol);
        } else {
            // Standard whitespace/tab `.smi`: `SMILES [ID] [NAME] [PACT] [TARGET]`
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let mut mol = parse_smiles(parts[0])?;
                mol.id = parts[1].trim().to_string();
                mol.name = parts[2].trim().to_string();
                if let Some(t) = parts.get(3) {
                    mol.target = t.trim().to_string();
                }
                if let Some(p) = parts.get(4).and_then(|s| s.trim().parse::<f64>().ok()) {
                    mol.pact = p;
                    mol.raw_nm = 10f64.powf(9.0 - p);
                }
                if let Some(u) = parts.get(5) {
                    mol.source_url = u.trim().to_string();
                }
                out.push(mol);
            } else {
                let ws: Vec<&str> = line.split_whitespace().collect();
                if ws.is_empty() {
                    continue;
                }
                let mut mol = parse_smiles(ws[0])?;
                mol.id = ws
                    .get(1)
                    .map(|s| (*s).to_string())
                    .unwrap_or_else(|| format!("CMPD-{:04}", row_idx + 1));
                mol.name = ws.get(2).map(|s| (*s).to_string()).unwrap_or_else(|| mol.id.clone());
                if let Some(p) = ws.get(3).and_then(|s| s.parse::<f64>().ok()) {
                    mol.pact = p;
                    mol.raw_nm = 10f64.powf(9.0 - p);
                }
                out.push(mol);
            }
        }
    }
    Ok(out)
}

fn split_csv_row(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

/// Parses a multi-record MDL V2000 `.sdf` or single `.mol` file.
pub fn parse_sdf_dataset(sdf_text: &str) -> Result<Vec<Molecule>, MolIoError> {
    let mut out = Vec::new();
    for raw_rec in sdf_text.split("$$$$") {
        let rec = raw_rec.trim_matches(|c| c == '\r' || c == '\n');
        if rec.trim().is_empty() {
            continue;
        }
        let mol = parse_sdf_record(rec)?;
        out.push(mol);
    }
    Ok(out)
}

/// Parses a single MDL V2000 `.mol` / `.sdf` block including 2D coordinates and SD property tags.
pub fn parse_sdf_record(record: &str) -> Result<Molecule, MolIoError> {
    let lines: Vec<&str> = record.lines().collect();
    if lines.len() < 4 {
        return Err(MolIoError::InvalidSdf("Fewer than 4 header lines".to_string()));
    }

    let title = lines[0].trim().to_string();
    let counts_line = lines[3];
    let (n_atoms, n_bonds) = if counts_line.len() >= 6 {
        let na = counts_line[0..3]
            .trim()
            .parse::<usize>()
            .ok();
        let nb = counts_line[3..6]
            .trim()
            .parse::<usize>()
            .ok();
        if let (Some(a), Some(b)) = (na, nb) {
            (a, b)
        } else {
            let ws: Vec<&str> = counts_line.split_whitespace().collect();
            if ws.len() < 2 {
                return Err(MolIoError::InvalidSdf("Cannot parse V2000 counts line".to_string()));
            }
            (
                ws[0].parse().map_err(|_| MolIoError::InvalidSdf("Bad atom count".into()))?,
                ws[1].parse().map_err(|_| MolIoError::InvalidSdf("Bad bond count".into()))?,
            )
        }
    } else {
        return Err(MolIoError::InvalidSdf("Counts line too short".to_string()));
    };

    if lines.len() < 4 + n_atoms + n_bonds {
        return Err(MolIoError::InvalidSdf("Truncated atom or bond block".to_string()));
    }

    let mut mol = Molecule::new("");
    mol.name = if title.is_empty() { "SDF-Compound".to_string() } else { title.clone() };
    mol.id = mol.name.clone();

    let mut has_nonzero_coords = false;

    for ai in 0..n_atoms {
        let aline = lines[4 + ai];
        let ws: Vec<&str> = aline.split_whitespace().collect();
        if ws.len() < 4 {
            return Err(MolIoError::InvalidSdf(format!("Malformed atom line {}", ai + 1)));
        }
        let x: f32 = ws[0].parse().unwrap_or(0.0);
        let y: f32 = ws[1].parse().unwrap_or(0.0);
        if x.abs() > 1e-4 || y.abs() > 1e-4 {
            has_nonzero_coords = true;
        }
        let sym = ws[3];
        let elem = Element::from_symbol(sym).unwrap_or(Element::C);
        let chg_code: i8 = ws.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
        let charge = match chg_code {
            1 => 3,
            2 => 2,
            3 => 1,
            5 => -1,
            6 => -2,
            7 => -3,
            _ => 0,
        };
        mol.atoms.push(Atom {
            idx: ai,
            elem,
            charge,
            aromatic: false,
            bracketed: charge != 0,
            explicit_h: 0,
            implicit_h: 0,
            chiral: 0,
            in_ring: false,
            ring_sizes: Vec::new(),
            hyb: Hybridization::Sp3,
            pos2d: [x, y],
        });
    }

    // Collapse explicit H atoms in V2000 SDF into parent heavy atom `explicit_h` so skeletal/ECFP match SMILES!
    let mut raw_bonds: Vec<(usize, usize, u8)> = Vec::with_capacity(n_bonds);
    for bi in 0..n_bonds {
        let bline = lines[4 + n_atoms + bi];
        let (a1, a2, btype) = if bline.len() >= 9 {
            let p1 = bline[0..3].trim().parse::<usize>().ok();
            let p2 = bline[3..6].trim().parse::<usize>().ok();
            let pt = bline[6..9].trim().parse::<u8>().ok();
            if let (Some(u), Some(v), Some(t)) = (p1, p2, pt) {
                (u, v, t)
            } else {
                let ws: Vec<&str> = bline.split_whitespace().collect();
                (
                    ws[0].parse().unwrap_or(1),
                    ws[1].parse().unwrap_or(1),
                    ws[2].parse().unwrap_or(1),
                )
            }
        } else {
            let ws: Vec<&str> = bline.split_whitespace().collect();
            (
                ws.first().and_then(|s| s.parse().ok()).unwrap_or(1),
                ws.get(1).and_then(|s| s.parse().ok()).unwrap_or(1),
                ws.get(2).and_then(|s| s.parse().ok()).unwrap_or(1),
            )
        };
        if a1 >= 1 && a2 >= 1 && a1 <= n_atoms && a2 <= n_atoms {
            raw_bonds.push((a1 - 1, a2 - 1, btype));
        }
    }

    // Parse M  CHG lines and SD data tags (`> <TAG>`)
    let mut line_idx = 4 + n_atoms + n_bonds;
    while line_idx < lines.len() {
        let l = lines[line_idx];
        if l.starts_with("M  CHG") {
            let ws: Vec<&str> = l[6..].split_whitespace().collect();
            if let Some(cnt) = ws.first().and_then(|s| s.parse::<usize>().ok()) {
                for k in 0..cnt {
                    if let (Some(a_s), Some(c_s)) = (ws.get(1 + 2 * k), ws.get(2 + 2 * k)) {
                        if let (Ok(ai1), Ok(chg)) = (a_s.parse::<usize>(), c_s.parse::<i8>()) {
                            if ai1 >= 1 && ai1 <= mol.atoms.len() {
                                mol.atoms[ai1 - 1].charge = chg;
                                mol.atoms[ai1 - 1].bracketed = chg != 0;
                            }
                        }
                    }
                }
            }
        } else if l.trim_start().starts_with('>') {
            if let (Some(lt), Some(gt)) = (l.find('<'), l.rfind('>')) {
                if gt > lt {
                    let tag = l[lt + 1..gt].trim().to_ascii_uppercase();
                    let val = lines.get(line_idx + 1).map(|s| s.trim()).unwrap_or("");
                    match tag.as_str() {
                        "ID" | "CHEMBL_ID" | "COMPOUND_ID" => {
                            if !val.is_empty() {
                                mol.id = val.to_string();
                            }
                        }
                        "NAME" | "GENERIC_NAME" | "TITLE" => {
                            if !val.is_empty() {
                                mol.name = val.to_string();
                            }
                        }
                        "TARGET" | "ASSAY_TARGET" => {
                            if !val.is_empty() {
                                mol.target = val.to_string();
                            }
                        }
                        "METRIC" | "STANDARD_TYPE" => {
                            if !val.is_empty() {
                                mol.metric = val.to_string();
                            }
                        }
                        "RAW_NM" | "IC50_NM" | "KI_NM" => {
                            if let Ok(v) = val.parse::<f64>() {
                                mol.raw_nm = v.max(1e-6);
                                mol.pact = 9.0 - mol.raw_nm.log10();
                            }
                        }
                        "PACT" | "PIC50" | "PKI" => {
                            if let Ok(v) = val.parse::<f64>() {
                                mol.pact = v;
                            }
                        }
                        "SMILES" | "CANONICAL_SMILES" => {
                            if !val.is_empty() {
                                mol.smiles = val.to_string();
                            }
                        }
                        "REF" | "SOURCE_URL" | "REFERENCE" => {
                            if !val.is_empty() {
                                mol.source_url = val.to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        line_idx += 1;
    }

    // Map heavy atoms while folding explicit H neighbors into `explicit_h`
    let mut old_to_new: Vec<Option<usize>> = vec![None; n_atoms];
    let mut heavy_atoms: Vec<Atom> = Vec::new();
    for (old_i, a) in mol.atoms.iter().enumerate() {
        if a.elem != Element::H {
            let ni = heavy_atoms.len();
            let mut cloned = a.clone();
            cloned.idx = ni;
            old_to_new[old_i] = Some(ni);
            heavy_atoms.push(cloned);
        }
    }

    let mut final_bonds: Vec<Bond> = Vec::new();
    for (u_old, v_old, btype) in raw_bonds {
        match (old_to_new[u_old], old_to_new[v_old]) {
            (Some(u), Some(v)) => {
                let order = match btype {
                    2 => BondOrder::Double,
                    3 => BondOrder::Triple,
                    4 => {
                        heavy_atoms[u].aromatic = true;
                        heavy_atoms[v].aromatic = true;
                        BondOrder::Aromatic
                    }
                    _ => BondOrder::Single,
                };
                let idx = final_bonds.len();
                final_bonds.push(Bond {
                    idx,
                    a: u,
                    b: v,
                    order,
                    aromatic: order == BondOrder::Aromatic,
                    in_ring: false,
                    kekule_order: 1,
                });
            }
            (Some(u), None) => {
                heavy_atoms[u].explicit_h = heavy_atoms[u].explicit_h.saturating_add(1);
            }
            (None, Some(v)) => {
                heavy_atoms[v].explicit_h = heavy_atoms[v].explicit_h.saturating_add(1);
            }
            (None, None) => {}
        }
    }

    mol.atoms = heavy_atoms;
    mol.bonds = final_bonds;
    mol.sanitize();
    // Note: in V2000 SDFs without explicit H records, standard uncharged heavy atoms still get implicit Hs
    // via `bracketed = charge != 0`, while any attached explicit H increments `explicit_h` cleanly.
    if !has_nonzero_coords {
        compute_2d_coords(&mut mol);
    }
    Ok(mol)
}

/// Formats a single `Molecule` as a valid MDL V2000 `.sdf` record (ending with `$$$$`).
pub fn format_sdf_record(mol: &Molecule) -> String {
    let mut out = String::with_capacity(2048);
    let title = if mol.name.is_empty() { &mol.id } else { &mol.name };
    out.push_str(title);
    out.push('\n');
    out.push_str("  sar-atlas 0.1.0   2D\n");
    out.push_str("Generated by sar-core V2000 writer\n");
    out.push_str(&format!(
        "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
        mol.atoms.len(),
        mol.bonds.len()
    ));

    let mut charged: Vec<(usize, i8)> = Vec::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        if a.charge != 0 {
            charged.push((i + 1, a.charge));
        }
        out.push_str(&format!(
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            a.pos2d[0],
            a.pos2d[1],
            0.0,
            a.elem.symbol()
        ));
    }

    for b in &mol.bonds {
        let btype: u8 = match b.order {
            BondOrder::Single => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => b.kekule_order.clamp(1, 2),
        };
        out.push_str(&format!(
            "{:>3}{:>3}{:>3}  0  0  0  0\n",
            b.a + 1,
            b.b + 1,
            btype
        ));
    }

    if !charged.is_empty() {
        for chunk in charged.chunks(8) {
            out.push_str(&format!("M  CHG{:>3}", chunk.len()));
            for &(idx1, chg) in chunk {
                out.push_str(&format!(" {:>3} {:>3}", idx1, chg));
            }
            out.push('\n');
        }
    }
    out.push_str("M  END\n");

    let tags = [
        ("ID", mol.id.as_str()),
        ("NAME", mol.name.as_str()),
        ("TARGET", mol.target.as_str()),
        ("METRIC", mol.metric.as_str()),
        ("RAW_NM", &format!("{:.4}", mol.raw_nm)),
        ("PACT", &format!("{:.2}", mol.pact)),
        ("SMILES", mol.smiles.as_str()),
        ("REF", mol.source_url.as_str()),
    ];
    for (k, v) in tags {
        if !v.is_empty() {
            out.push_str(&format!("> <{}>\n{}\n\n", k, v));
        }
    }
    out.push_str("$$$$\n");
    out
}

/// Formats an entire compound slice as a multi-record MDL V2000 `.sdf` string.
pub fn format_sdf_dataset(mols: &[Molecule]) -> String {
    let mut out = String::with_capacity(mols.len() * 1536);
    for m in mols {
        out.push_str(&format_sdf_record(m));
    }
    out
}
