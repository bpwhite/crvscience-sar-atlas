# Headless Validation Artifacts (`crvscience-sar-atlas`)

Generated directly by running the workspace Rust integration test suite:

```bash
cargo test -p sar-core --test headless_validation
```

All outputs are kept strictly under `384 KB` (`393,216` bytes) per file and verify real execution of `sar_core` (`v3.8.0`):

1. **`pbpk_plasma_profile.csv`** (`T2`): 97-point (`0.000 -> 24.000 h`) RK4 compartmental oral PBPK time-course curves (`export_pbpk_timecourse_csv`) comparing CNS-penetrant **Haloperidol** (`CHEMBL54`) against peripherally restricted **Lapatinib** (`CHEMBL554`).
2. **`sar_atlas_validation_summary.json`** (`T1 + T2 + T3`): Machine-readable telemetry record capturing `T1` (`DatasetAnalysis` CNS MPO, QED, Lipinski Ro5, Veber, MMP activity cliffs, Pareto leads), `T2` (`simulate_oral_pbpk` `Fa * Fg * Fh == F_oral` balance, `CL`, `Vss`, `Kp,uu,brain`, `Cmax`, `Tmax`, `AUC0-24`), and `T3` (`run_rbfe_network_campaign` MLE cycle closure `< 0.15 kcal/mol` + `run_active_learning_campaign` MolPAL Kriging-Believer batch selection).
