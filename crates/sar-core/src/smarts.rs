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

//! Daylight SMARTS (`https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html`)
//! substructure query compiler and `GreatestConstraintFirst` + `VF2` (`Cordella et al. 2004`,
//! `Bonnici & Giugno 2013`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC3633016/`) subgraph
//! monomorphism matcher with recursive `$(...)` environment support and `Rayon` batch screening.

use crate::mol::{Atom, Bond, Element, Hybridization, Molecule};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SmartsError {
    #[error("Empty SMARTS pattern")]
    EmptyPattern,
    #[error("Unexpected token '{0}' at byte {1}")]
    UnexpectedToken(char, usize),
    #[error("Unclosed bracket '[' starting at byte {0}")]
    UnclosedBracket(usize),
    #[error("Unclosed recursive '$(' starting at byte {0}")]
    UnclosedRecursive(usize),
    #[error("Unbalanced branch parenthesis in SMARTS")]
    UnbalancedBranch,
    #[error("Unclosed ring closure {0} in SMARTS")]
    UnclosedRing(u16),
    #[error("Empty atom or bond expression at byte {0}")]
    EmptyExpression(usize),
}

/// Logical AST for Daylight SMARTS atom expressions (`!` > `&` > `,` > `;`).
#[derive(Clone, Debug)]
pub enum AtomExpr {
    Prim(AtomPrim),
    Not(Box<AtomExpr>),
    And(Box<AtomExpr>, Box<AtomExpr>),
    Or(Box<AtomExpr>, Box<AtomExpr>),
}

#[derive(Clone, Debug)]
pub enum AtomPrim {
    Wildcard,
    AnyAromatic,
    AnyAliphatic,
    AtomicNum(u8),
    Element { elem: Element, aromatic: Option<bool> },
    HeavyDegree(usize),
    TotalConnectivity(usize),
    TotalHydrogens(u8),
    ImplicitHydrogens(Option<u8>),
    Valence(u8),
    InRingAny,
    NotInRing,
    RingMembershipCount(usize),
    RingSize(usize),
    Charge(i8),
    Hybridization(Hybridization),
    Recursive(SmartsQuery),
}

/// Logical AST for Daylight SMARTS bond expressions (`!` > `&` > `,` > `;`).
#[derive(Clone, Debug)]
pub enum BondExpr {
    Prim(BondPrim),
    Not(Box<BondExpr>),
    And(Box<BondExpr>, Box<BondExpr>),
    Or(Box<BondExpr>, Box<BondExpr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondPrim {
    SingleOrAromatic,
    SingleAliphatic,
    Double,
    Triple,
    Aromatic,
    Any,
    Ring,
}

#[derive(Clone, Debug)]
pub struct QueryBond {
    pub a: usize,
    pub b: usize,
    pub expr: BondExpr,
}

/// Compiled Daylight SMARTS query graph ready for thread-safe parallel matching.
#[derive(Clone, Debug)]
pub struct SmartsQuery {
    pub pattern: String,
    pub atoms: Vec<AtomExpr>,
    pub bonds: Vec<QueryBond>,
    pub adj: Vec<Vec<(usize, usize)>>,
    /// Connected components, each ordered via `GreatestConstraintFirst` (`(query_u, parent_edges)`).
    pub search_plans: Vec<Vec<QueryStep>>,
}

#[derive(Clone, Debug)]
pub struct QueryStep {
    pub q_atom: usize,
    /// Already-matched query neighbors `(prev_q_atom, query_bond_idx)`.
    pub constraints: Vec<(usize, usize)>,
}

/// Result of matching a `SmartsQuery` against a target `Molecule`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SmartsMatchResult {
    pub matched: bool,
    /// Count of unique atom-set embeddings in the target molecule.
    pub match_count: usize,
    /// Union of all matched target atom indices (sorted) for 2D skeletal highlighting.
    pub union_atoms: Vec<usize>,
    /// Union of all matched target bond indices (sorted) for 2D bond highlighting.
    pub union_bonds: Vec<usize>,
    /// Up to 32 distinct query->target atom index embeddings (including symmetric core flips for R-group alignment).
    pub mappings: Vec<Vec<usize>>,
}

pub struct SmartsPreset {
    pub name: &'static str,
    pub smarts: &'static str,
    pub description: &'static str,
}

/// Curated medicinal-chemistry SMARTS presets covering core scaffolds, bioisosteres, and warheads.
pub const CURATED_SMARTS_PRESETS: [SmartsPreset; 10] = [
    SmartsPreset {
        name: "Quinazoline",
        smarts: "c1ncnc2ccccc12",
        description: "4-Anilinoquinazoline ATP-competitive kinase core (EGFR series)",
    },
    SmartsPreset {
        name: "Phenothiazine",
        smarts: "c1ccc2c(c1)Sc3ccccc3N2",
        description: "Tricyclic phenothiazine neuroleptic scaffold (D2 antagonists)",
    },
    SmartsPreset {
        name: "7-Ring Tricyclic",
        smarts: "[r7]",
        description: "Any atom in a 7-membered ring (dibenzoxazepine / clozapine / olanzapine)",
    },
    SmartsPreset {
        name: "Piperazine",
        smarts: "N1CCNCC1",
        description: "Aliphatic 1,4-piperazine basic amine motif",
    },
    SmartsPreset {
        name: "Sulfonyl",
        smarts: "[#16](=O)(=O)[#7,#6]",
        description: "Sulfonamide or methylsulfone COX-2 / kinase selectivity pocket group",
    },
    SmartsPreset {
        name: "CF3 Group",
        smarts: "C(F)(F)F",
        description: "Lipophilic metabolically blocked trifluoromethyl substituent",
    },
    SmartsPreset {
        name: "Enone / Acrylamide",
        smarts: "[#6]=[#6]-[#6]=O",
        description: "Conjugated Michael acceptor (covalent acrylamides & curcuminoids)",
    },
    SmartsPreset {
        name: "Terminal Alkyne",
        smarts: "[#6]#[CH1]",
        description: "Ethynyl substituent (e.g. Erlotinib meta-ethynylaniline)",
    },
    SmartsPreset {
        name: "Ortho-Catechol",
        smarts: "[OH]c1ccccc1[OH]",
        description: "1,2-Dihydroxyaryl redox-active catechol motif",
    },
    SmartsPreset {
        name: "Aryl Halide",
        smarts: "c-[F,Cl,Br,I]",
        description: "Halogenated aromatic ring (F, Cl, Br, or I)",
    },
];

/// Compiles a Daylight SMARTS pattern string into a `SmartsQuery`.
pub fn compile_smarts(pattern: &str) -> Result<SmartsQuery, SmartsError> {
    let trimmed = pattern.trim();
    if trimmed.is_empty() {
        return Err(SmartsError::EmptyPattern);
    }

    let bytes = trimmed.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    let mut atoms: Vec<AtomExpr> = Vec::new();
    let mut bonds: Vec<QueryBond> = Vec::new();
    let mut prev_atom: Option<usize> = None;
    let mut branch_stack: Vec<Option<usize>> = Vec::new();
    let mut pending_bond: Option<BondExpr> = None;
    let mut open_rings: HashMap<u16, (usize, Option<BondExpr>)> = HashMap::new();

    while i < len {
        let ch = bytes[i] as char;
        if ch.is_ascii_whitespace() {
            break;
        }
        match ch {
            '(' => {
                branch_stack.push(prev_atom);
                i += 1;
            }
            ')' => {
                prev_atom = branch_stack.pop().ok_or(SmartsError::UnbalancedBranch)?;
                pending_bond = None;
                i += 1;
            }
            '.' => {
                prev_atom = None;
                pending_bond = None;
                i += 1;
            }
            '[' => {
                let end = find_matching_bracket(trimmed, i)?;
                let inner = &trimmed[i + 1..end];
                let aexpr = parse_bracket_atom_expr(inner, i + 1)?;
                let cur = atoms.len();
                atoms.push(aexpr);
                if let Some(pa) = prev_atom {
                    let bexpr = pending_bond
                        .take()
                        .unwrap_or(BondExpr::Prim(BondPrim::SingleOrAromatic));
                    bonds.push(QueryBond { a: pa, b: cur, expr: bexpr });
                } else {
                    pending_bond = None;
                }
                prev_atom = Some(cur);
                i = end + 1;
            }
            '%' => {
                if i + 2 >= len
                    || !(bytes[i + 1] as char).is_ascii_digit()
                    || !(bytes[i + 2] as char).is_ascii_digit()
                {
                    return Err(SmartsError::UnexpectedToken('%', i));
                }
                let rnum = ((bytes[i + 1] - b'0') as u16) * 10 + ((bytes[i + 2] - b'0') as u16);
                let cur = prev_atom.ok_or(SmartsError::UnexpectedToken('%', i))?;
                close_or_open_smarts_ring(
                    &mut bonds,
                    cur,
                    rnum,
                    pending_bond.take(),
                    &mut open_rings,
                );
                i += 3;
            }
            '0'..='9' => {
                let rnum = (bytes[i] - b'0') as u16;
                let cur = prev_atom.ok_or(SmartsError::UnexpectedToken(ch, i))?;
                close_or_open_smarts_ring(
                    &mut bonds,
                    cur,
                    rnum,
                    pending_bond.take(),
                    &mut open_rings,
                );
                i += 1;
            }
            '-' | '=' | '#' | ':' | '~' | '@' | '!' | '/' | '\\' => {
                if prev_atom.is_none() {
                    return Err(SmartsError::UnexpectedToken(ch, i));
                }
                let start = i;
                while i < len && is_bond_expr_char(bytes[i] as char) {
                    i += 1;
                }
                let b_slice = &trimmed[start..i];
                pending_bond = Some(parse_bond_expr(b_slice, start)?);
            }
            _ => {
                let (aexpr, step) = parse_unbracketed_smarts_atom(&trimmed[i..], i)?;
                let cur = atoms.len();
                atoms.push(aexpr);
                if let Some(pa) = prev_atom {
                    let bexpr = pending_bond
                        .take()
                        .unwrap_or(BondExpr::Prim(BondPrim::SingleOrAromatic));
                    bonds.push(QueryBond { a: pa, b: cur, expr: bexpr });
                } else {
                    pending_bond = None;
                }
                prev_atom = Some(cur);
                i += step;
            }
        }
    }

    if !branch_stack.is_empty() {
        return Err(SmartsError::UnbalancedBranch);
    }
    if let Some(&rnum) = open_rings.keys().next() {
        return Err(SmartsError::UnclosedRing(rnum));
    }
    if atoms.is_empty() {
        return Err(SmartsError::EmptyPattern);
    }

    let n = atoms.len();
    let mut adj = vec![Vec::new(); n];
    for (bi, b) in bonds.iter().enumerate() {
        adj[b.a].push((b.b, bi));
        adj[b.b].push((b.a, bi));
    }

    let search_plans = build_search_plans(&atoms, &adj);

    Ok(SmartsQuery {
        pattern: trimmed.to_string(),
        atoms,
        bonds,
        adj,
        search_plans,
    })
}

fn is_bond_expr_char(ch: char) -> bool {
    matches!(
        ch,
        '-' | '=' | '#' | ':' | '~' | '@' | '!' | '&' | ',' | ';' | '/' | '\\'
    )
}

fn find_matching_bracket(s: &str, open_pos: usize) -> Result<usize, SmartsError> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut p = open_pos;
    while p < bytes.len() {
        match bytes[p] {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(p);
                }
            }
            _ => {}
        }
        p += 1;
    }
    Err(SmartsError::UnclosedBracket(open_pos))
}

fn find_matching_paren(s: &str, open_pos: usize) -> Result<usize, SmartsError> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut p = open_pos;
    while p < bytes.len() {
        match bytes[p] {
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = bracket_depth.saturating_sub(1),
            b'(' if bracket_depth == 0 => depth += 1,
            b')' if bracket_depth == 0 => {
                depth -= 1;
                if depth == 0 {
                    return Ok(p);
                }
            }
            _ => {}
        }
        p += 1;
    }
    Err(SmartsError::UnclosedRecursive(open_pos))
}

fn close_or_open_smarts_ring(
    bonds: &mut Vec<QueryBond>,
    cur: usize,
    rnum: u16,
    pending_bond: Option<BondExpr>,
    open_rings: &mut HashMap<u16, (usize, Option<BondExpr>)>,
) {
    if let Some((other, open_b)) = open_rings.remove(&rnum) {
        let expr = pending_bond
            .or(open_b)
            .unwrap_or(BondExpr::Prim(BondPrim::SingleOrAromatic));
        if other != cur {
            bonds.push(QueryBond { a: other, b: cur, expr });
        }
    } else {
        open_rings.insert(rnum, (cur, pending_bond));
    }
}

fn parse_unbracketed_smarts_atom(rem: &str, byte_pos: usize) -> Result<(AtomExpr, usize), SmartsError> {
    if rem.starts_with("Cl") {
        return Ok((
            AtomExpr::Prim(AtomPrim::Element {
                elem: Element::Cl,
                aromatic: Some(false),
            }),
            2,
        ));
    }
    if rem.starts_with("Br") {
        return Ok((
            AtomExpr::Prim(AtomPrim::Element {
                elem: Element::Br,
                aromatic: Some(false),
            }),
            2,
        ));
    }
    let ch = rem
        .chars()
        .next()
        .ok_or(SmartsError::EmptyExpression(byte_pos))?;
    let prim = match ch {
        '*' => AtomPrim::Wildcard,
        'a' => AtomPrim::AnyAromatic,
        'A' => AtomPrim::AnyAliphatic,
        'B' => AtomPrim::Element { elem: Element::B, aromatic: Some(false) },
        'C' => AtomPrim::Element { elem: Element::C, aromatic: Some(false) },
        'N' => AtomPrim::Element { elem: Element::N, aromatic: Some(false) },
        'O' => AtomPrim::Element { elem: Element::O, aromatic: Some(false) },
        'P' => AtomPrim::Element { elem: Element::P, aromatic: Some(false) },
        'S' => AtomPrim::Element { elem: Element::S, aromatic: Some(false) },
        'F' => AtomPrim::Element { elem: Element::F, aromatic: Some(false) },
        'I' => AtomPrim::Element { elem: Element::I, aromatic: Some(false) },
        'b' => AtomPrim::Element { elem: Element::B, aromatic: Some(true) },
        'c' => AtomPrim::Element { elem: Element::C, aromatic: Some(true) },
        'n' => AtomPrim::Element { elem: Element::N, aromatic: Some(true) },
        'o' => AtomPrim::Element { elem: Element::O, aromatic: Some(true) },
        'p' => AtomPrim::Element { elem: Element::P, aromatic: Some(true) },
        's' => AtomPrim::Element { elem: Element::S, aromatic: Some(true) },
        _ => return Err(SmartsError::UnexpectedToken(ch, byte_pos)),
    };
    Ok((AtomExpr::Prim(prim), 1))
}

fn parse_bracket_atom_expr(inner: &str, base_pos: usize) -> Result<AtomExpr, SmartsError> {
    if inner.is_empty() {
        return Err(SmartsError::EmptyExpression(base_pos));
    }
    let clean = strip_top_level_atom_map(inner);
    if clean == "H" {
        return Ok(AtomExpr::Prim(AtomPrim::AtomicNum(1)));
    }
    let parts = split_top_level(clean, ';');
    let mut acc: Option<AtomExpr> = None;
    let mut offset = 0usize;
    for part in parts {
        let sub = parse_atom_or(part, base_pos + offset)?;
        acc = Some(match acc {
            None => sub,
            Some(prev) => AtomExpr::And(Box::new(prev), Box::new(sub)),
        });
        offset += part.len() + 1;
    }
    acc.ok_or(SmartsError::EmptyExpression(base_pos))
}

fn parse_atom_or(s: &str, base_pos: usize) -> Result<AtomExpr, SmartsError> {
    let parts = split_top_level(s, ',');
    let mut acc: Option<AtomExpr> = None;
    let mut offset = 0usize;
    for part in parts {
        let sub = parse_atom_high_and(part, base_pos + offset)?;
        acc = Some(match acc {
            None => sub,
            Some(prev) => AtomExpr::Or(Box::new(prev), Box::new(sub)),
        });
        offset += part.len() + 1;
    }
    acc.ok_or(SmartsError::EmptyExpression(base_pos))
}

fn parse_atom_high_and(s: &str, base_pos: usize) -> Result<AtomExpr, SmartsError> {
    if s.is_empty() {
        return Err(SmartsError::EmptyExpression(base_pos));
    }
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut p = 0usize;
    let mut acc: Option<AtomExpr> = None;

    while p < n {
        if bytes[p] == b'&' {
            p += 1;
            if p >= n {
                return Err(SmartsError::EmptyExpression(base_pos + p));
            }
        }
        let mut negated = false;
        while p < n && bytes[p] == b'!' {
            negated = !negated;
            p += 1;
        }
        if p >= n {
            return Err(SmartsError::EmptyExpression(base_pos + p));
        }

        let (prim, step) = parse_single_atom_prim(&s[p..], base_pos + p, acc.is_some() || negated)?;
        p += step;

        let mut node = AtomExpr::Prim(prim);
        if negated {
            node = AtomExpr::Not(Box::new(node));
        }
        acc = Some(match acc {
            None => node,
            Some(prev) => AtomExpr::And(Box::new(prev), Box::new(node)),
        });
    }

    acc.ok_or(SmartsError::EmptyExpression(base_pos))
}

fn parse_single_atom_prim(
    rem: &str,
    pos: usize,
    _has_preceding: bool,
) -> Result<(AtomPrim, usize), SmartsError> {
    let bytes = rem.as_bytes();
    let ch = bytes[0] as char;

    if rem.starts_with("$(") {
        let close = find_matching_paren(rem, 1)?;
        let sub_pat = &rem[2..close];
        let sub_q = compile_smarts(sub_pat)?;
        return Ok((AtomPrim::Recursive(sub_q), close + 1));
    }

    if ch.is_ascii_digit() {
        let (_, d_len) = read_usize(rem);
        if d_len < rem.len() {
            let (inner_p, inner_step) =
                parse_single_atom_prim(&rem[d_len..], pos + d_len, false)?;
            return Ok((inner_p, d_len + inner_step));
        }
        return Err(SmartsError::UnexpectedToken(ch, pos));
    }

    match ch {
        '*' => Ok((AtomPrim::Wildcard, 1)),
        'a' => Ok((AtomPrim::AnyAromatic, 1)),
        'A' => Ok((AtomPrim::AnyAliphatic, 1)),
        '#' => {
            let (num, d_len) = read_usize(&rem[1..]);
            if d_len == 0 {
                return Err(SmartsError::UnexpectedToken('#', pos));
            }
            Ok((AtomPrim::AtomicNum(num as u8), 1 + d_len))
        }
        'D' => {
            let (deg, d_len) = read_usize(&rem[1..]);
            Ok((
                AtomPrim::HeavyDegree(if d_len > 0 { deg } else { 1 }),
                1 + d_len,
            ))
        }
        'X' => {
            let (conn, d_len) = read_usize(&rem[1..]);
            Ok((
                AtomPrim::TotalConnectivity(if d_len > 0 { conn } else { 1 }),
                1 + d_len,
            ))
        }
        'H' => {
            let (hc, d_len) = read_usize(&rem[1..]);
            Ok((
                AtomPrim::TotalHydrogens(if d_len > 0 { hc as u8 } else { 1 }),
                1 + d_len,
            ))
        }
        'h' => {
            let (hc, d_len) = read_usize(&rem[1..]);
            Ok((
                AtomPrim::ImplicitHydrogens(if d_len > 0 { Some(hc as u8) } else { None }),
                1 + d_len,
            ))
        }
        'v' => {
            let (val, d_len) = read_usize(&rem[1..]);
            Ok((
                AtomPrim::Valence(if d_len > 0 { val as u8 } else { 1 }),
                1 + d_len,
            ))
        }
        'R' => {
            let (rc, d_len) = read_usize(&rem[1..]);
            if d_len == 0 {
                Ok((AtomPrim::InRingAny, 1))
            } else if rc == 0 {
                Ok((AtomPrim::NotInRing, 1 + d_len))
            } else {
                Ok((AtomPrim::RingMembershipCount(rc), 1 + d_len))
            }
        }
        'r' => {
            let (rsz, d_len) = read_usize(&rem[1..]);
            if d_len == 0 {
                Ok((AtomPrim::InRingAny, 1))
            } else {
                Ok((AtomPrim::RingSize(rsz), 1 + d_len))
            }
        }
        '+' => {
            let mut p = 1usize;
            if p < bytes.len() && bytes[p] == b'+' {
                let mut q = 1i8;
                while p < bytes.len() && bytes[p] == b'+' {
                    q += 1;
                    p += 1;
                }
                Ok((AtomPrim::Charge(q), p))
            } else {
                let (mag, d_len) = read_usize(&rem[1..]);
                Ok((
                    AtomPrim::Charge(if d_len > 0 { mag as i8 } else { 1 }),
                    1 + d_len,
                ))
            }
        }
        '-' => {
            let mut p = 1usize;
            if p < bytes.len() && bytes[p] == b'-' {
                let mut q = -1i8;
                while p < bytes.len() && bytes[p] == b'-' {
                    q -= 1;
                    p += 1;
                }
                Ok((AtomPrim::Charge(q), p))
            } else {
                let (mag, d_len) = read_usize(&rem[1..]);
                Ok((
                    AtomPrim::Charge(if d_len > 0 { -(mag as i8) } else { -1 }),
                    1 + d_len,
                ))
            }
        }
        '^' => {
            let (hyb_code, d_len) = read_usize(&rem[1..]);
            let hyb = match hyb_code {
                1 => Hybridization::Sp,
                2 => Hybridization::Sp2,
                3 => Hybridization::Sp3,
                _ => return Err(SmartsError::UnexpectedToken('^', pos)),
            };
            Ok((AtomPrim::Hybridization(hyb), 1 + d_len))
        }
        '@' => {
            if rem.starts_with("@@") {
                Ok((AtomPrim::Wildcard, 2))
            } else {
                Ok((AtomPrim::Wildcard, 1))
            }
        }
        _ => {
            if bytes.len() >= 2 {
                let c2 = bytes[1] as char;
                if (c2.is_ascii_lowercase() && c2 != 'h' && c2 != 'r')
                    || matches!(&rem[..2], "Cl" | "Br" | "Si" | "Se" | "se")
                {
                    if let Some(el) = Element::from_symbol(&rem[..2]) {
                        let arom = if ch.is_ascii_lowercase() {
                            Some(true)
                        } else {
                            Some(false)
                        };
                        return Ok((AtomPrim::Element { elem: el, aromatic: arom }, 2));
                    }
                }
            }
            if let Some(el) = Element::from_symbol(&rem[..1]) {
                let arom = if ch.is_ascii_lowercase() {
                    Some(true)
                } else {
                    Some(false)
                };
                return Ok((AtomPrim::Element { elem: el, aromatic: arom }, 1));
            }
            Err(SmartsError::UnexpectedToken(ch, pos))
        }
    }
}

fn parse_bond_expr(s: &str, base_pos: usize) -> Result<BondExpr, SmartsError> {
    let parts = split_top_level(s, ';');
    let mut acc: Option<BondExpr> = None;
    for part in parts {
        let or_parts = split_top_level(part, ',');
        let mut or_acc: Option<BondExpr> = None;
        for op in or_parts {
            let and_sub = parse_bond_high_and(op, base_pos)?;
            or_acc = Some(match or_acc {
                None => and_sub,
                Some(prev) => BondExpr::Or(Box::new(prev), Box::new(and_sub)),
            });
        }
        let sub = or_acc.ok_or(SmartsError::EmptyExpression(base_pos))?;
        acc = Some(match acc {
            None => sub,
            Some(prev) => BondExpr::And(Box::new(prev), Box::new(sub)),
        });
    }
    acc.ok_or(SmartsError::EmptyExpression(base_pos))
}

fn parse_bond_high_and(s: &str, base_pos: usize) -> Result<BondExpr, SmartsError> {
    if s.is_empty() {
        return Err(SmartsError::EmptyExpression(base_pos));
    }
    let bytes = s.as_bytes();
    let mut p = 0usize;
    let mut acc: Option<BondExpr> = None;
    while p < bytes.len() {
        if bytes[p] == b'&' {
            p += 1;
            if p >= bytes.len() {
                return Err(SmartsError::EmptyExpression(base_pos + p));
            }
        }
        let mut neg = false;
        while p < bytes.len() && bytes[p] == b'!' {
            neg = !neg;
            p += 1;
        }
        if p >= bytes.len() {
            return Err(SmartsError::EmptyExpression(base_pos + p));
        }
        let bprim = match bytes[p] as char {
            '-' | '/' | '\\' => BondPrim::SingleAliphatic,
            '=' => BondPrim::Double,
            '#' => BondPrim::Triple,
            ':' => BondPrim::Aromatic,
            '~' => BondPrim::Any,
            '@' => BondPrim::Ring,
            c => return Err(SmartsError::UnexpectedToken(c, base_pos + p)),
        };
        p += 1;
        let mut node = BondExpr::Prim(bprim);
        if neg {
            node = BondExpr::Not(Box::new(node));
        }
        acc = Some(match acc {
            None => node,
            Some(prev) => BondExpr::And(Box::new(prev), Box::new(node)),
        });
    }
    acc.ok_or(SmartsError::EmptyExpression(base_pos))
}

fn strip_top_level_atom_map(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b':' if depth == 0 => return &s[..i],
            _ => {}
        }
    }
    s
}

fn split_top_level(s: &str, delim: char) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut last = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.saturating_sub(1),
            _ if b == (delim as u8) && depth == 0 => {
                out.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[last..]);
    out
}

fn read_usize(s: &str) -> (usize, usize) {
    let mut val = 0usize;
    let mut cnt = 0usize;
    for b in s.bytes() {
        if b.is_ascii_digit() {
            val = val.saturating_mul(10).saturating_add((b - b'0') as usize);
            cnt += 1;
        } else {
            break;
        }
    }
    (val, cnt)
}

fn build_search_plans(atoms: &[AtomExpr], adj: &[Vec<(usize, usize)>]) -> Vec<Vec<QueryStep>> {
    let n = atoms.len();
    let mut in_plan = vec![false; n];
    let mut plans = Vec::new();

    for start_comp in 0..n {
        if in_plan[start_comp] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start_comp];
        let mut comp_seen = vec![false; n];
        comp_seen[start_comp] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &(v, _) in &adj[u] {
                if !comp_seen[v] {
                    comp_seen[v] = true;
                    stack.push(v);
                }
            }
        }

        let mut steps = Vec::with_capacity(comp.len());
        let seed = *comp
            .iter()
            .max_by_key(|&&u| specificity_score(&atoms[u]) * 16 + adj[u].len() * 10)
            .unwrap_or(&start_comp);

        in_plan[seed] = true;
        steps.push(QueryStep {
            q_atom: seed,
            constraints: Vec::new(),
        });

        while steps.len() < comp.len() {
            let next_q = *comp
                .iter()
                .filter(|&&u| !in_plan[u])
                .max_by_key(|&&u| {
                    let already_nbrs = adj[u].iter().filter(|&&(v, _)| in_plan[v]).count();
                    already_nbrs * 1000 + specificity_score(&atoms[u]) * 16 + adj[u].len() * 8
                })
                .unwrap();

            let constraints: Vec<(usize, usize)> = adj[next_q]
                .iter()
                .filter(|&&(v, _)| in_plan[v])
                .copied()
                .collect();
            in_plan[next_q] = true;
            steps.push(QueryStep {
                q_atom: next_q,
                constraints,
            });
        }

        plans.push(steps);
    }

    plans
}

fn specificity_score(expr: &AtomExpr) -> usize {
    match expr {
        AtomExpr::Prim(p) => match p {
            AtomPrim::Wildcard => 1,
            AtomPrim::AnyAliphatic | AtomPrim::AnyAromatic => 3,
            AtomPrim::Element { elem: Element::C, .. } => 4,
            AtomPrim::Element { .. } | AtomPrim::AtomicNum(_) => 8,
            AtomPrim::Recursive(_) => 12,
            AtomPrim::RingSize(_) | AtomPrim::RingMembershipCount(_) => 9,
            _ => 5,
        },
        AtomExpr::Not(inner) => specificity_score(inner) + 2,
        AtomExpr::And(a, b) => specificity_score(a) + specificity_score(b),
        AtomExpr::Or(a, b) => specificity_score(a).min(specificity_score(b)),
    }
}

impl AtomExpr {
    pub fn matches(&self, mol: &Molecule, u: usize) -> bool {
        match self {
            Self::Prim(p) => p.matches(mol, u),
            Self::Not(inner) => !inner.matches(mol, u),
            Self::And(a, b) => a.matches(mol, u) && b.matches(mol, u),
            Self::Or(a, b) => a.matches(mol, u) || b.matches(mol, u),
        }
    }
}

impl AtomPrim {
    pub fn matches(&self, mol: &Molecule, u: usize) -> bool {
        let a: &Atom = &mol.atoms[u];
        match self {
            Self::Wildcard => a.elem != Element::H,
            Self::AnyAromatic => a.aromatic,
            Self::AnyAliphatic => !a.aromatic && a.elem != Element::H,
            Self::AtomicNum(z) => a.elem.atomic_number() == *z,
            Self::Element { elem, aromatic } => {
                if a.elem != *elem {
                    return false;
                }
                match aromatic {
                    Some(req_arom) => a.aromatic == *req_arom,
                    None => true,
                }
            }
            Self::HeavyDegree(d) => mol.heavy_degree(u) == *d,
            Self::TotalConnectivity(x) => mol.heavy_degree(u) + (a.total_h() as usize) == *x,
            Self::TotalHydrogens(h) => a.total_h() == *h,
            Self::ImplicitHydrogens(opt_h) => match opt_h {
                Some(h) => a.implicit_h == *h,
                None => a.implicit_h >= 1,
            },
            Self::Valence(v) => {
                let bo_sum: u8 = mol.adj[u]
                    .iter()
                    .map(|&(_, bi)| mol.bonds[bi].kekule_order)
                    .sum();
                bo_sum.saturating_add(a.total_h()) == *v
            }
            Self::InRingAny => a.in_ring,
            Self::NotInRing => !a.in_ring,
            Self::RingMembershipCount(cnt) => {
                let r_cnt = mol.rings.iter().filter(|r| r.contains(&u)).count();
                r_cnt == *cnt
            }
            Self::RingSize(sz) => a.ring_sizes.contains(sz),
            Self::Charge(q) => a.charge == *q,
            Self::Hybridization(hyb) => a.hyb == *hyb,
            Self::Recursive(sub_q) => match_anchored_atom0(mol, sub_q, u),
        }
    }
}

impl BondExpr {
    pub fn matches(&self, bond: &Bond) -> bool {
        match self {
            Self::Prim(p) => p.matches(bond),
            Self::Not(inner) => !inner.matches(bond),
            Self::And(a, b) => a.matches(bond) && b.matches(bond),
            Self::Or(a, b) => a.matches(bond) || b.matches(bond),
        }
    }
}

impl BondPrim {
    pub fn matches(self, bond: &Bond) -> bool {
        match self {
            Self::SingleOrAromatic => bond.aromatic || bond.kekule_order == 1,
            Self::SingleAliphatic => !bond.aromatic && bond.kekule_order == 1,
            Self::Double => !bond.aromatic && bond.kekule_order == 2,
            Self::Triple => bond.kekule_order == 3,
            Self::Aromatic => bond.aromatic,
            Self::Any => true,
            Self::Ring => bond.in_ring,
        }
    }
}

fn match_anchored_atom0(mol: &Molecule, sub_q: &SmartsQuery, target_root: usize) -> bool {
    let qn = sub_q.atoms.len();
    if qn == 0 || qn > mol.atoms.len() {
        return false;
    }
    if !sub_q.atoms[0].matches(mol, target_root) {
        return false;
    }
    if qn == 1 {
        return true;
    }

    let mut visited_q = vec![false; qn];
    visited_q[0] = true;
    let mut steps = Vec::with_capacity(qn - 1);
    while steps.len() + 1 < qn {
        let Some(next_q) = (0..qn).filter(|&u| !visited_q[u]).max_by_key(|&u| {
            sub_q.adj[u].iter().filter(|&&(v, _)| visited_q[v]).count() * 100 + sub_q.adj[u].len()
        }) else {
            break;
        };
        let constraints: Vec<(usize, usize)> = sub_q.adj[next_q]
            .iter()
            .filter(|&&(v, _)| visited_q[v])
            .copied()
            .collect();
        visited_q[next_q] = true;
        steps.push(QueryStep {
            q_atom: next_q,
            constraints,
        });
    }

    let mut q2t = vec![usize::MAX; qn];
    let mut used_t = vec![false; mol.atoms.len()];
    q2t[0] = target_root;
    used_t[target_root] = true;

    anchored_dfs(0, &steps, sub_q, mol, &mut q2t, &mut used_t)
}

fn anchored_dfs(
    step_idx: usize,
    steps: &[QueryStep],
    query: &SmartsQuery,
    mol: &Molecule,
    q2t: &mut [usize],
    used_t: &mut [bool],
) -> bool {
    if step_idx >= steps.len() {
        return true;
    }
    let step = &steps[step_idx];
    let q = step.q_atom;
    let q_deg = query.adj[q].len();

    if let Some(&(first_prev_q, first_q_bi)) = step.constraints.first() {
        let prev_t = q2t[first_prev_q];
        for &(cand_t, t_bi) in &mol.adj[prev_t] {
            if used_t[cand_t] || mol.adj[cand_t].len() < q_deg {
                continue;
            }
            if !query.bonds[first_q_bi].expr.matches(&mol.bonds[t_bi]) {
                continue;
            }
            if !query.atoms[q].matches(mol, cand_t) {
                continue;
            }
            let rest_ok = step.constraints[1..].iter().all(|&(pq, qbi)| {
                mol.bond_between(q2t[pq], cand_t)
                    .is_some_and(|tbi| query.bonds[qbi].expr.matches(&mol.bonds[tbi]))
            });
            if !rest_ok {
                continue;
            }
            q2t[q] = cand_t;
            used_t[cand_t] = true;
            if anchored_dfs(step_idx + 1, steps, query, mol, q2t, &mut *used_t) {
                return true;
            }
            used_t[cand_t] = false;
            q2t[q] = usize::MAX;
        }
        false
    } else {
        for cand_t in 0..mol.atoms.len() {
            if used_t[cand_t]
                || mol.adj[cand_t].len() < q_deg
                || !query.atoms[q].matches(mol, cand_t)
            {
                continue;
            }
            q2t[q] = cand_t;
            used_t[cand_t] = true;
            if anchored_dfs(step_idx + 1, steps, query, mol, q2t, &mut *used_t) {
                return true;
            }
            used_t[cand_t] = false;
            q2t[q] = usize::MAX;
        }
        false
    }
}

/// Matches a compiled `SmartsQuery` against `mol`, returning unique atom-set count (`match_count`)
/// plus up to 32 distinct `q2t` automorphic embeddings for symmetry-canonicalized R-group alignment.
pub fn match_smarts(mol: &Molecule, query: &SmartsQuery) -> SmartsMatchResult {
    let qn = query.atoms.len();
    let tn = mol.atoms.len();
    if qn == 0 || qn > tn {
        return SmartsMatchResult::default();
    }

    let mut compat = vec![vec![false; tn]; qn];
    for q in 0..qn {
        let q_deg = query.adj[q].len();
        let mut any = false;
        for t in 0..tn {
            if mol.adj[t].len() >= q_deg && query.atoms[q].matches(mol, t) {
                compat[q][t] = true;
                any = true;
            }
        }
        if !any {
            return SmartsMatchResult::default();
        }
    }

    let all_steps: Vec<&QueryStep> = query.search_plans.iter().flat_map(|p| p.iter()).collect();
    let mut q2t = vec![usize::MAX; qn];
    let mut used_t = vec![false; tn];
    let mut seen_atom_sets: Vec<Vec<usize>> = Vec::new();
    let mut mappings: Vec<Vec<usize>> = Vec::new();
    let mut raw_visits = 0usize;
    let mut union_atoms_set = BTreeSet::new();
    let mut union_bonds_set = BTreeSet::new();

    #[allow(clippy::too_many_arguments)]
    fn search_all(
        idx: usize,
        steps: &[&QueryStep],
        query: &SmartsQuery,
        mol: &Molecule,
        compat: &[Vec<bool>],
        q2t: &mut [usize],
        used_t: &mut [bool],
        raw_visits: &mut usize,
        seen_sets: &mut Vec<Vec<usize>>,
        mappings: &mut Vec<Vec<usize>>,
        u_atoms: &mut BTreeSet<usize>,
        u_bonds: &mut BTreeSet<usize>,
    ) {
        if *raw_visits >= 128 || seen_sets.len() >= 64 {
            return;
        }
        if idx >= steps.len() {
            *raw_visits += 1;
            if mappings.len() < 32 {
                mappings.push(q2t.to_vec());
            }
            let mut sorted_atoms = q2t.to_vec();
            sorted_atoms.sort_unstable();
            if !seen_sets.contains(&sorted_atoms) {
                seen_sets.push(sorted_atoms);
                for &ta in q2t.iter() {
                    u_atoms.insert(ta);
                }
                for qb in &query.bonds {
                    if let Some(tbi) = mol.bond_between(q2t[qb.a], q2t[qb.b]) {
                        u_bonds.insert(tbi);
                    }
                }
            }
            return;
        }

        let step = steps[idx];
        let q = step.q_atom;

        if let Some(&(first_pq, first_qbi)) = step.constraints.first() {
            let prev_t = q2t[first_pq];
            for &(cand_t, t_bi) in &mol.adj[prev_t] {
                if used_t[cand_t] || !compat[q][cand_t] {
                    continue;
                }
                if !query.bonds[first_qbi].expr.matches(&mol.bonds[t_bi]) {
                    continue;
                }
                let rest_ok = step.constraints[1..].iter().all(|&(pq, qbi)| {
                    mol.bond_between(q2t[pq], cand_t)
                        .is_some_and(|tbi| query.bonds[qbi].expr.matches(&mol.bonds[tbi]))
                });
                if !rest_ok {
                    continue;
                }
                q2t[q] = cand_t;
                used_t[cand_t] = true;
                search_all(
                    idx + 1,
                    steps,
                    query,
                    mol,
                    compat,
                    q2t,
                    used_t,
                    raw_visits,
                    seen_sets,
                    mappings,
                    u_atoms,
                    u_bonds,
                );
                used_t[cand_t] = false;
                q2t[q] = usize::MAX;
            }
        } else {
            for cand_t in 0..mol.atoms.len() {
                if used_t[cand_t] || !compat[q][cand_t] {
                    continue;
                }
                q2t[q] = cand_t;
                used_t[cand_t] = true;
                search_all(
                    idx + 1,
                    steps,
                    query,
                    mol,
                    compat,
                    q2t,
                    used_t,
                    raw_visits,
                    seen_sets,
                    mappings,
                    u_atoms,
                    u_bonds,
                );
                used_t[cand_t] = false;
                q2t[q] = usize::MAX;
            }
        }
    }

    search_all(
        0,
        &all_steps,
        query,
        mol,
        &compat,
        &mut q2t,
        &mut used_t,
        &mut raw_visits,
        &mut seen_atom_sets,
        &mut mappings,
        &mut union_atoms_set,
        &mut union_bonds_set,
    );

    SmartsMatchResult {
        matched: !seen_atom_sets.is_empty(),
        match_count: seen_atom_sets.len(),
        union_atoms: union_atoms_set.into_iter().collect(),
        union_bonds: union_bonds_set.into_iter().collect(),
        mappings,
    }
}

/// Evaluates a compiled `SmartsQuery` across an entire dataset in parallel via `Rayon`.
pub fn match_smarts_batch(mols: &[Molecule], query: &SmartsQuery) -> Vec<SmartsMatchResult> {
    mols.par_iter().map(|m| match_smarts(m, query)).collect()
}
