# SAR-Atlas (`v3.8.0`) Scientific Methods & Verification

## 1. Oral Compartmental PBPK (`pbpk_oral.rs`)

### Model Architecture (`simulate_oral_pbpk`)
- **GI Dissolution & 7-Segment ACAT Model**: Models solid (`stom_solid`, `si_solid[0..7]`) and dissolved (`stom_diss`, `si_diss[0..7]`) drug species with pH-dependent stomach vs. small-intestine solubility (`c_sat_stom_nm`, `c_sat_si_nm`, Prandial FaSSIF/FeSSIF bile micelle enhancement) and Yu-Amidon segmental transit (`k_transit_hr = 7 / 3.32 h^-1`).
- **Supersaturated Weak-Base Precipitation**: When luminal dissolved concentration `c_seg` exceeds `c_sat_si_nm` (e.g., cationic bases emptying from low-pH stomach into neutral SI segments), dissolved excess `(c_seg - c_sat_si_nm) * V_si_seg` precipitates cleanly back to `si_solid` at first-order rate `k_precip_hr`, preserving exact solid + dissolved mass balance.
- **First-Pass Gut & Liver (`Q_gut`, `Well-Stirred`, Saturable CYP3A4)**: Evaluates nonlinear Michaelis-Menten CYP3A4 saturation (`Km`, `Vmax`) + UGT clearance via `F_oral = Fa * Fg * Fh`.
- **Plasma-Brain Mass Conservation & Nonspecific Brain Binding (`V_u,brain`)**:
  - Central plasma (`y[16]`), peripheral tissues (`y[17]`), and brain parenchyma (`y[18]`) form a strictly mass-conserving closed ODE network across `y[0..=18]`.
  - Unbound brain concentration reflects nonspecific lipid/phospholipid tissue partitioning via `V_u,brain = V_brain / fu_brain` (`V_brain = 1.40 L`), so `C_u,brain = y[18] / V_u,brain`.
  - Net BBB flux `j_bbb_net = PS_bbb * C_u,plasma - (PS_bbb / Kp,uu,brain) * C_u,brain` enters `dy[18]` and is symmetrically subtracted from central plasma `dy[16]` (`dy[16] = j_sys_in - j_elim - j_dist - j_bbb_net`).
  - Integrated via 4th-order Runge-Kutta (`RK4`, `dt = 0.01 h`) plus antithetic log-normal 48-subject virtual population (`P5–P95` CI90 envelopes).

## 2. Alchemical RBFE, Bennett BAR & Solvation (`rbfe_network.rs`)

### Self-Consistent Bennett Acceptance Ratio (`solve_bennett_bar`)
- Given forward and reverse perturbation work arrays `du_fwd_kcal`, `du_rev_kcal`, `solve_bennett_bar` locates `C` via safeguarded Newton-Raphson + bisection on the Fermi-Dirac logistic moment equation:
  $$\langle f(\beta \Delta U_{\text{fwd}} - C) \rangle_0 = \langle f(\beta \Delta U_{\text{rev}} + C) \rangle_1$$
- Computes asymptotic standard error from the true Bennett (1976) / Shirts & Chodera (2008) reduced variance without artificial `0.002 / (n0 + n1)` floors (retaining only a `1e-12` numerical underflow guard):
  $$\mathrm{Var}(\beta \Delta G) = \max\!\left(\frac{1}{n_0}\left(\frac{\langle f_0^2 \rangle}{\langle f_0 \rangle^2} - 1\right) + \frac{1}{n_1}\left(\frac{\langle f_1^2 \rangle}{\langle f_1 \rangle^2} - 1\right), 10^{-12}\right)$$

### Born Continuum Ionic Solvation (`estimate_aqueous_solvation_dg_kcal`)
- Combines polar (`TPSA`, `HBD`, `HBA`), hydrophobic cavitation (`cLogP`, heavy atoms), and size-dependent Born electrostatics:
  $$r_{\text{eff}} = \left(\sum_i R_{\text{vdW},i}^3\right)^{1/3} + 0.65\text{ \AA}, \qquad \Delta G_{\text{Born}} = -163.88\,\frac{q^2}{r_{\text{eff}}}\text{ kcal/mol}$$
- Correctly reproduces stronger aqueous hydration favorability for compact monovalent/divalent ions relative to bulky molecular ions of identical formal charge.

## 3. GPU Chemical-Space Scalar Contours & 3D Geometry (`chem_space.wgsl`, `gui.rs`)

- **Scalar Iso-Contours (`t_scalar` `R8Unorm`)**: Separable Gaussian fields write quantized `intensity` `[0, 255]` directly into `scalar_texture`, so `fs_heatmap` extracts topographic iso-contours directly from true normalized scalar levels rather than non-monotonic perceptual colormap RGB luminance.
- **Screen-Space Node Clamping (`vs_node`)**: Clamps rendered compound core pixel radii to `[3.0, 18.0]` px using `(viewport_px.y * 0.5) * zoom / 100.0`, preserving visibility at overview zoom and preventing over-expansion at deep zoom.
- **Dynamic Bounding-Sphere 3D Projection**: `paint_3d_molecular_viewport` computes the ligand/overlay bounding radius `r_bound` dynamically (`cam_dist = max(3.2 * r_bound, 26.0)`), preventing perspective clipping on large macrocycles and PROTACs alongside multi-layer Lambert/specular atom shading, depth fog, 3D `X/Y/Z` gizmo, Ångström scale bar, and Gasteiger charge atom picking.

## 4. Offline Benchmark Telemetry (`validation/sar_atlas_validation_summary.json`)

- **T1 (`load_benchmark_26`)**: `26` curated molecules; mean `CNS-MPO = 4.87`, mean `QED = 0.715`, `11` Pareto leads, `12` MMP pairs (`5` SALI cliffs, max `SALI = 8.63`).
- **T2 (`simulate_oral_pbpk`, 50 mg fasted)**: CNS-penetrant Haloperidol (`CHEMBL54`, `Kp,uu,brain = 0.923`) cleanly separates from peripherally restricted Lapatinib (`CHEMBL554`, `Kp,uu,brain = 0.062`) while conserving mass across closed GI/systemic/brain compartments (`y[0..=18]` error `< 1e-6` nmol).
- **T3 (`run_rbfe_network_campaign` & `run_active_learning_campaign`)**: Cycle-closed Lomap Dopamine D2 graph (`11` nodes, `15` edges, `6` cycles, pre-MLE max hysteresis `2.03 kcal/mol` -> `0.000 kcal/mol` post-MLE, `RMSE = 0.99 kcal/mol`, Pearson `R^2 = 0.582`, Kendall `tau = 0.455`) paired with hybrid-kernel GP+RF active learning (`70` candidates, `10` promoted).
