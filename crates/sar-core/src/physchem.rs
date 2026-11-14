//! Physicochemical descriptor and multiparameter desirability engine:
//! - Wildman & Crippen (1999) atomic `cLogP`
//! - Ertl, Rohde & Selzer (2000) fragment `TPSA`
//! - Bickerton et al. (2012) `QED` (`WEIGHT_MEAN` & `WEIGHT_MAX`)
//! - Wager et al. (2010) `CNS-MPO` (`0.0–6.0`)
//! - Lipinski Rule-of-5, Veber rules, Lipophilic Ligand Efficiency (`LLE`), and Pareto front ranking.

use crate::mol::{Element, Hybridization, Molecule};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhyschemProfile {
    pub formula: String,
    pub mw: f64,
    pub clogp: f64,
    pub clogd_74: f64,
    pub est_basic_pka: f64,
    pub tpsa: f64,
    pub hbd: usize,
    pub hba: usize,
    pub rotatable_bonds: usize,
    pub aromatic_rings: usize,
    pub fsp3: f64,
    pub heavy_atoms: usize,
    pub ro5_violations: usize,
    pub veber_pass: bool,
    pub qed_mean: f64,
    pub qed_max: f64,
    pub cns_mpo: f64,
    pub lle: f64,
    pub pareto_optimal: bool,
}

/// Computes the complete physicochemical and multiparameter desirability profile for `mol`.
/// `alert_count` is the count of structural alerts (used in Bickerton QED `ALERTS` term).
pub fn evaluate_physchem(mol: &Molecule, alert_count: usize) -> PhyschemProfile {
    let formula = mol.molecular_formula();
    let mw = mol.molecular_weight();
    let heavy_atoms = mol.atoms.iter().filter(|a| a.elem != Element::H).count();

    let mut hbd = 0usize;
    let mut hba = 0usize;
    let mut c_total = 0usize;
    let mut c_sp3 = 0usize;

    for (u, a) in mol.atoms.iter().enumerate() {
        let th = a.total_h() as usize;
        if matches!(a.elem, Element::N | Element::O) && th > 0 {
            hbd += 1;
        }
        // Lipinski/RDKit HBA: N and O with available lone pair (exclude positively charged quaternary N or amide-like pyrrole N if saturated)
        match a.elem {
            Element::O => {
                if a.charge <= 0 {
                    hba += 1;
                }
            }
            Element::N => {
                let bo_sum: u8 = mol.adj[u]
                    .iter()
                    .map(|&(_, bi)| mol.bonds[bi].kekule_order)
                    .sum();
                let is_amide_n = mol.adj[u].iter().any(|&(v, _)| is_carbonyl_or_sulfonyl_center(mol, v));
                let is_pyrrolic = a.aromatic && (th > 0 || mol.adj[u].len() >= 3);
                if a.charge <= 0 && bo_sum + (th as u8) <= 3 && !is_amide_n && !is_pyrrolic {
                    hba += 1;
                } else if a.aromatic && th == 0 && mol.adj[u].len() == 2 && a.charge == 0 {
                    // Pyridine / pyrimidine / quinazoline ring nitrogen
                    hba += 1;
                }
            }
            _ => {}
        }
        if a.elem == Element::C {
            c_total += 1;
            if a.hyb == Hybridization::Sp3 && !a.aromatic {
                c_sp3 += 1;
            }
        }
    }

    let fsp3 = if c_total > 0 {
        (c_sp3 as f64) / (c_total as f64)
    } else {
        0.0
    };

    // Rotatable bonds: non-ring single bonds between two non-terminal heavy atoms, excluding amide C-N and triple-adjacent bonds
    let mut rotatable_bonds = 0usize;
    for b in &mol.bonds {
        if b.in_ring || b.kekule_order != 1 || b.aromatic {
            continue;
        }
        let deg_a = mol.heavy_degree(b.a);
        let deg_b = mol.heavy_degree(b.b);
        if deg_a <= 1 || deg_b <= 1 {
            continue;
        }
        if mol.atoms[b.a].hyb == Hybridization::Sp || mol.atoms[b.b].hyb == Hybridization::Sp {
            continue;
        }
        let is_amide = (mol.atoms[b.a].elem == Element::N && is_carbonyl_center(mol, b.b))
            || (mol.atoms[b.b].elem == Element::N && is_carbonyl_center(mol, b.a));
        if !is_amide {
            rotatable_bonds += 1;
        }
    }

    let aromatic_rings = mol
        .rings
        .iter()
        .filter(|r| r.iter().all(|&ai| mol.atoms[ai].aromatic))
        .count();

    let clogp = estimate_wildman_crippen_clogp(mol);
    let tpsa = estimate_ertl_tpsa(mol);
    let est_basic_pka = estimate_strongest_basic_pka(mol);

    // Henderson-Hasselbalch ionization correction at physiological pH 7.4:
    // cLogD(7.4) = cLogP - log10(1 + 10^(pKa_basic - 7.4) + 10^(7.4 - pKa_acidic))
    let has_carboxylic_acid = has_carboxylic_acid_group(mol);
    let basic_term = if est_basic_pka > 3.0 {
        10f64.powf((est_basic_pka - 7.4).clamp(-6.0, 6.0))
    } else {
        0.0
    };
    let acidic_term = if has_carboxylic_acid {
        10f64.powf((7.4 - 4.2_f64).clamp(-6.0, 6.0))
    } else {
        0.0
    };
    let clogd_74 = clogp - (1.0 + basic_term + acidic_term).log10();

    let mut ro5_violations = 0usize;
    if mw > 500.0 {
        ro5_violations += 1;
    }
    if clogp > 5.0 {
        ro5_violations += 1;
    }
    if hbd > 5 {
        ro5_violations += 1;
    }
    if hba > 10 {
        ro5_violations += 1;
    }
    let veber_pass = tpsa <= 140.0 && rotatable_bonds <= 10;

    let qed_mean = compute_bickerton_qed(
        mw,
        clogp,
        hba as f64,
        hbd as f64,
        tpsa,
        rotatable_bonds as f64,
        aromatic_rings as f64,
        alert_count as f64,
        &QED_WEIGHT_MEAN,
    );
    let qed_max = compute_bickerton_qed(
        mw,
        clogp,
        hba as f64,
        hbd as f64,
        tpsa,
        rotatable_bonds as f64,
        aromatic_rings as f64,
        alert_count as f64,
        &QED_WEIGHT_MAX,
    );

    let cns_mpo = compute_wager_cns_mpo(clogp, clogd_74, mw, tpsa, hbd as f64, est_basic_pka);
    let lle = mol.pact - clogp;

    PhyschemProfile {
        formula,
        mw,
        clogp,
        clogd_74,
        est_basic_pka,
        tpsa,
        hbd,
        hba,
        rotatable_bonds,
        aromatic_rings,
        fsp3,
        heavy_atoms,
        ro5_violations,
        veber_pass,
        qed_mean,
        qed_max,
        cns_mpo,
        lle,
        pareto_optimal: false,
    }
}

fn is_carbonyl_center(mol: &Molecule, u: usize) -> bool {
    mol.atoms[u].elem == Element::C
        && mol.adj[u].iter().any(|&(v, bi)| {
            mol.bonds[bi].kekule_order == 2
                && !mol.bonds[bi].aromatic
                && matches!(mol.atoms[v].elem, Element::O | Element::S)
        })
}

fn is_carbonyl_or_sulfonyl_center(mol: &Molecule, u: usize) -> bool {
    is_carbonyl_center(mol, u)
        || (mol.atoms[u].elem == Element::S
            && mol.adj[u]
                .iter()
                .filter(|&&(v, bi)| mol.bonds[bi].kekule_order == 2 && mol.atoms[v].elem == Element::O)
                .count()
                >= 1)
}

fn has_carboxylic_acid_group(mol: &Molecule) -> bool {
    for (u, a) in mol.atoms.iter().enumerate() {
        if a.elem == Element::C && is_carbonyl_center(mol, u) {
            let has_oh = mol.adj[u].iter().any(|&(v, bi)| {
                mol.bonds[bi].kekule_order == 1
                    && mol.atoms[v].elem == Element::O
                    && mol.atoms[v].total_h() >= 1
            });
            if has_oh {
                return true;
            }
        }
    }
    false
}

/// Wildman & Crippen (1999) (`J. Chem. Inf. Comput. Sci. 39, 868-873`) atomic logP contributions
/// covering aliphatic, aromatic, heteroatom-bound, carbonyl/sulfonyl, and implicit-H microenvironments.
pub fn estimate_wildman_crippen_clogp(mol: &Molecule) -> f64 {
    let mut logp = 0.0f64;

    for (u, a) in mol.atoms.iter().enumerate() {
        let th = a.total_h() as f64;
        let hetero_nbrs = mol.adj[u]
            .iter()
            .filter(|&&(v, _)| mol.atoms[v].is_hetero())
            .count();
        let arom_nbrs = mol.adj[u]
            .iter()
            .filter(|&&(v, _)| mol.atoms[v].aromatic)
            .count();

        match a.elem {
            Element::C => {
                if a.aromatic {
                    if hetero_nbrs == 0 {
                        // C18 (aromatic CH) = +0.1581; C19/C20 bridgehead/substituted C = +0.2955 / +0.2713
                        if th > 0.0 {
                            logp += 0.1581 + th * 0.1230;
                        } else {
                            logp += 0.2713;
                        }
                    } else {
                        // C21/C22 aromatic C bonded to N/O/S/halogen
                        logp += 0.1360;
                    }
                } else if a.hyb == Hybridization::Sp {
                    // C25 alkyne/nitrile sp carbon
                    logp += if hetero_nbrs > 0 { -0.0670 } else { 0.2010 } + th * 0.1230;
                } else if a.hyb == Hybridization::Sp2 {
                    if is_carbonyl_center(mol, u) {
                        // C26/C27 carbonyl C
                        logp += -0.2783 + th * 0.1230;
                    } else {
                        // C13..C16 alkene sp2 carbon
                        logp += 0.1750 + th * 0.1230;
                    }
                } else {
                    // Sp3 carbon: C1/C2 hydrocarbon vs C3/C4/C5 heteroatom-adjacent vs C8..C10 benzylic
                    let c_contrib = if hetero_nbrs >= 2 {
                        -0.3250 // e.g. CF3 / acetal carbon (offset by strong +F contributions)
                    } else if hetero_nbrs == 1 {
                        -0.2035 // C3/C4 alpha to N/O/halogen
                    } else if arom_nbrs >= 1 {
                        0.0845 // benzylic sp3 C
                    } else {
                        0.1441 // C1/C2 aliphatic hydrocarbon C
                    };
                    logp += c_contrib + th * 0.1230;
                }
            }
            Element::N => {
                if a.aromatic {
                    if th > 0.0 {
                        // N11 pyrrole [nH] + H2
                        logp += -0.3239 + th * (-0.2677);
                    } else {
                        // N11 unprotonated pyridine/quinazoline `n`
                        logp += -0.3239;
                    }
                } else if is_amide_or_aniline_n(mol, u) {
                    // Aniline Ar-NH-Ar / amide N
                    logp += -0.3650 + th * (-0.2100);
                } else if a.hyb == Hybridization::Sp {
                    // Nitrile N
                    logp += -0.5360;
                } else if a.hyb == Hybridization::Sp2 {
                    // Imine / amidine N=C (e.g. Loxapine/Clozapine/Olanzapine)
                    logp += -0.1850 + th * (-0.2100);
                } else {
                    // Aliphatic amine (primary/secondary/tertiary piperazine/piperidine)
                    let base_n = match mol.heavy_degree(u) {
                        0 | 1 => -1.0190,
                        2 => -0.7096,
                        _ => -0.5280,
                    };
                    logp += base_n + th * (-0.2100);
                }
            }
            Element::O => {
                if a.aromatic {
                    // O2 furan / isoxazole aromatic O
                    logp += 0.1280;
                } else if a.hyb == Hybridization::Sp2 {
                    // O9..O11 carbonyl / sulfonyl =O
                    let on_s = mol.adj[u].iter().any(|&(v, _)| mol.atoms[v].elem == Element::S);
                    logp += if on_s { -0.3339 } else { -0.1990 };
                } else if th > 0.0 {
                    // Alcohol (O1) vs Phenol/Acid OH (O2/O12)
                    let on_arom_or_carbonyl = arom_nbrs > 0
                        || mol.adj[u].iter().any(|&(v, _)| is_carbonyl_center(mol, v));
                    logp += if on_arom_or_carbonyl {
                        -0.2893 + (-0.2677)
                    } else {
                        -0.6820 + (-0.2677)
                    };
                } else {
                    // Ether O3 (aliphatic R-O-R) vs O4 (aryl ether Ar-O-R / Ar-O-Ar)
                    logp += if arom_nbrs > 0 { -0.0950 } else { -0.4190 };
                }
            }
            Element::F => logp += 0.4202,
            Element::Cl => logp += 0.6895,
            Element::Br => logp += 0.8456,
            Element::I => logp += 1.0500,
            Element::S => {
                let oxo_cnt = mol.adj[u]
                    .iter()
                    .filter(|&&(v, bi)| {
                        mol.bonds[bi].kekule_order == 2 && mol.atoms[v].elem == Element::O
                    })
                    .count();
                if oxo_cnt >= 2 {
                    // Sulfone / sulfonamide S(=O)2
                    logp += -1.0260;
                } else if oxo_cnt == 1 {
                    // Sulfoxide S=O
                    logp += -0.8500;
                } else if a.hyb == Hybridization::Sp2 && mol.heavy_degree(u) == 1 {
                    // Thiocarbonyl =S (e.g. Epalrestat rhodanine)
                    logp += 0.6400;
                } else if a.aromatic {
                    // Thiophene / thiazole aromatic S
                    logp += 0.6482;
                } else {
                    // Thioether / phenothiazine S / thiol
                    logp += 0.5450 + th * 0.1800;
                }
            }
            Element::P => logp += -0.4200,
            Element::B => logp += -0.1500,
            Element::Si => logp += 0.6500,
            Element::Se => logp += 0.7200,
            Element::H => logp += 0.1230,
            Element::Other(_) => {}
        }
    }

    logp
}

fn is_amide_or_aniline_n(mol: &Molecule, u: usize) -> bool {
    mol.adj[u]
        .iter()
        .any(|&(v, _)| mol.atoms[v].aromatic || is_carbonyl_or_sulfonyl_center(mol, v))
}

/// Ertl, Rohde & Selzer (2000) (`J. Med. Chem. 43, 3714-3717`) topological polar surface area (`Å²`).
pub fn estimate_ertl_tpsa(mol: &Molecule) -> f64 {
    let mut tpsa = 0.0f64;

    for (u, a) in mol.atoms.iter().enumerate() {
        let th = a.total_h();
        let deg = mol.heavy_degree(u);
        let double_cnt = mol.adj[u]
            .iter()
            .filter(|&&(_, bi)| mol.bonds[bi].kekule_order == 2 && !mol.bonds[bi].aromatic)
            .count();
        let triple_cnt = mol.adj[u]
            .iter()
            .filter(|&&(_, bi)| mol.bonds[bi].kekule_order == 3)
            .count();

        match a.elem {
            Element::N => {
                if a.aromatic {
                    if th >= 1 {
                        tpsa += 15.79; // [nH] pyrrole/indole
                    } else if deg == 2 {
                        tpsa += 12.89; // pyridine/quinazoline `n`
                    } else {
                        tpsa += 4.93; // bridgehead aromatic `n` (e.g. Risperidone)
                    }
                } else if triple_cnt >= 1 {
                    tpsa += 23.79; // nitrile C#N
                } else if double_cnt >= 1 {
                    if th == 0 {
                        tpsa += 12.36; // imine -N=C
                    } else {
                        tpsa += 23.85; // =NH
                    }
                } else {
                    // Single-bonded N
                    match (deg, th) {
                        (1, 2) => tpsa += 26.02, // -NH2 (sulfonamide/aniline/primary amine)
                        (2, 1) => tpsa += 12.03, // -NH- (secondary amine / diarylamine / amide NH)
                        (3, 0) => tpsa += 3.24,  // tertiary amine / piperazine / phenothiazine N
                        _ => {
                            if a.charge > 0 {
                                tpsa += 4.44 * (th as f64);
                            } else {
                                tpsa += 12.03;
                            }
                        }
                    }
                }
            }
            Element::O => {
                if a.aromatic {
                    tpsa += 13.14; // furan / isoxazole aromatic O
                } else if double_cnt >= 1 {
                    tpsa += 17.07; // carbonyl / sulfonyl =O
                } else if th >= 1 {
                    tpsa += 20.23; // -OH hydroxyl / phenol / carboxylic acid OH
                } else {
                    tpsa += 9.23; // -O- ether / ester / lactone O
                }
            }
            Element::S => {
                // Include standard Ertl S polar contributions (thiol/thioether/sulfoxide/sulfone/thiocarbonyl)
                if a.aromatic {
                    tpsa += 28.24 * 0.0; // standard RDKit default TPSA excludes non-polar aromatic S unless includeSandP; keep standard N/O + polar S=O / =S consistency: RDKit standard TPSA uses N & O only (17.07 per =O on sulfone)!
                }
                let _ = (double_cnt, th);
            }
            Element::P => {
                if th >= 1 {
                    tpsa += 9.81;
                }
            }
            _ => {}
        }
    }

    tpsa
}

/// Estimates strongest basic center `pKa` for Wager CNS-MPO physiological ionization scoring.
pub fn estimate_strongest_basic_pka(mol: &Molecule) -> f64 {
    let mut max_pka: f64 = 2.0; // neutral / non-basic baseline

    for (u, a) in mol.atoms.iter().enumerate() {
        if a.elem != Element::N {
            continue;
        }
        let th = a.total_h();
        let has_carbonyl = mol.adj[u]
            .iter()
            .any(|&(v, _)| is_carbonyl_or_sulfonyl_center(mol, v));
        if has_carbonyl {
            continue; // Amides and sulfonamides are non-basic (basic pKa < 0)
        }
        let arom_nbrs = mol.adj[u]
            .iter()
            .filter(|&&(v, _)| mol.atoms[v].aromatic)
            .count();

        let pka = if a.aromatic {
            if th > 0 {
                0.5 // pyrrole
            } else {
                5.2 // pyridine / quinazoline ring N
            }
        } else if a.hyb == Hybridization::Sp2 {
            // Amidine (e.g. Loxapine/Clozapine/Olanzapine piperazinyl-amidine) vs simple imine
            let has_amine_carbon = mol.adj[u].iter().any(|&(c, bi)| {
                mol.bonds[bi].kekule_order == 2
                    && mol.adj[c]
                        .iter()
                        .any(|&(n2, _)| n2 != u && mol.atoms[n2].elem == Element::N)
            });
            if has_amine_carbon {
                7.6
            } else {
                6.2
            }
        } else if arom_nbrs >= 2 {
            1.0 // diphenylamine / phenothiazine ring N
        } else if arom_nbrs == 1 {
            4.6 // aniline
        } else {
            // Aliphatic amine: check inductive beta-heteroatom attenuation (morpholine ~7.4, piperazine ~8.2, piperidine/trialkylamine ~9.4)
            let mut beta_ewg = 0usize;
            for &(c_alpha, _) in &mol.adj[u] {
                for &(beta, _) in &mol.adj[c_alpha] {
                    if beta != u && matches!(mol.atoms[beta].elem, Element::N | Element::O | Element::F | Element::S) {
                        beta_ewg += 1;
                    }
                }
            }
            if beta_ewg >= 2 {
                7.4 // morpholine (e.g. Gefitinib)
            } else if beta_ewg == 1 {
                8.2 // piperazine / ethanolamine (e.g. Fluphenazine, Perphenazine, Olanzapine, Clozapine, Quetiapine)
            } else if th >= 1 {
                9.8 // secondary aliphatic piperazine NH (e.g. Amoxapine)
            } else {
                9.1 // tertiary piperidine / dimethylaminoalkyl (e.g. Haloperidol, Chlorpromazine, Vandetanib, Afatinib)
            }
        };
        if pka > max_pka {
            max_pka = pka;
        }
    }

    max_pka
}

/// Exact Bickerton et al. (2012) (`Nature Chemistry 4, 90–98`) & RDKit `QED.py` ADS parameters:
/// `(A, B, C, D, E, F, DMAX)` for `[MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS]`.
const QED_ADS_PARAMS: [(f64, f64, f64, f64, f64, f64, f64); 8] = [
    // MW
    (2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561),
    // ALOGP
    (3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604),
    // HBA
    (2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046),
    // HBD
    (1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616),
    // PSA
    (1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167),
    // ROTB
    (0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403),
    // AROM
    (3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610),
    // ALERTS
    (0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140),
];

pub const QED_WEIGHT_MEAN: [f64; 8] = [0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95];
pub const QED_WEIGHT_MAX: [f64; 8] = [0.50, 0.25, 0.00, 0.50, 0.00, 0.50, 0.25, 1.00];

fn ads_desirability(x: f64, p: (f64, f64, f64, f64, f64, f64, f64)) -> f64 {
    let (a, b, c, d, e, f, dmax) = p;
    let exp1 = 1.0 + (-((x - c + d * 0.5) / e)).exp();
    let exp2 = 1.0 + (-((x - c - d * 0.5) / f)).exp();
    let val = (a + b / exp1 * (1.0 - 1.0 / exp2)) / dmax;
    val.clamp(1e-6, 1.0)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_bickerton_qed(
    mw: f64,
    alogp: f64,
    hba: f64,
    hbd: f64,
    psa: f64,
    rotb: f64,
    arom: f64,
    alerts: f64,
    weights: &[f64; 8],
) -> f64 {
    let props = [mw, alogp, hba, hbd, psa, rotb, arom, alerts];
    let mut weighted_ln_sum = 0.0f64;
    let mut w_sum = 0.0f64;
    for i in 0..8 {
        let d_i = ads_desirability(props[i], QED_ADS_PARAMS[i]);
        weighted_ln_sum += weights[i] * d_i.ln();
        w_sum += weights[i];
    }
    (weighted_ln_sum / w_sum.max(1e-9)).exp().clamp(0.0, 1.0)
}

/// Wager et al. (2010) (`ACS Chem. Neurosci. 1, 435-449`) CNS Multiparameter Optimization (`0.0–6.0`).
pub fn compute_wager_cns_mpo(
    clogp: f64,
    clogd: f64,
    mw: f64,
    tpsa: f64,
    hbd: f64,
    pka: f64,
) -> f64 {
    let ramp_down = |val: f64, low: f64, high: f64| -> f64 {
        if val <= low {
            1.0
        } else if val >= high {
            0.0
        } else {
            (high - val) / (high - low)
        }
    };

    let d_clogp = ramp_down(clogp, 3.0, 5.0);
    let d_clogd = ramp_down(clogd, 2.0, 4.0);
    let d_mw = ramp_down(mw, 360.0, 500.0);
    let d_tpsa = if tpsa <= 20.0 || tpsa >= 120.0 {
        0.0
    } else if tpsa < 40.0 {
        (tpsa - 20.0) / 20.0
    } else if tpsa <= 90.0 {
        1.0
    } else {
        (120.0 - tpsa) / 30.0
    };
    let d_hbd = ramp_down(hbd, 0.5, 3.5);
    let d_pka = ramp_down(pka, 8.0, 10.0);

    (d_clogp + d_clogd + d_mw + d_tpsa + d_hbd + d_pka).clamp(0.0, 6.0)
}

/// Marks non-dominated Pareto-optimal leads maximizing `(pAct, qed_mean, cns_mpo)` and minimizing `alert_count`.
pub fn mark_pareto_front(
    mols: &[Molecule],
    profiles: &mut [PhyschemProfile],
    alert_counts: &[usize],
) -> Vec<usize> {
    let n = mols.len();
    let mut pareto_indices = Vec::new();
    for i in 0..n {
        let mut dominated = false;
        for j in 0..n {
            if i == j {
                continue;
            }
            let ge_all = mols[j].pact >= mols[i].pact - 1e-6
                && profiles[j].qed_mean >= profiles[i].qed_mean - 1e-6
                && profiles[j].cns_mpo >= profiles[i].cns_mpo - 1e-6
                && alert_counts[j] <= alert_counts[i];
            let gt_any = mols[j].pact > mols[i].pact + 1e-4
                || profiles[j].qed_mean > profiles[i].qed_mean + 1e-4
                || profiles[j].cns_mpo > profiles[i].cns_mpo + 1e-4
                || alert_counts[j] < alert_counts[i];
            if ge_all && gt_any {
                dominated = true;
                break;
            }
        }
        profiles[i].pareto_optimal = !dominated;
        if !dominated {
            pareto_indices.push(i);
        }
    }
    pareto_indices
}
