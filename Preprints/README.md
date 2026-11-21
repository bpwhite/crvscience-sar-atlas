# Preprints for SAR-Atlas (`crvscience-sar-atlas`)

**Author:** Bryan P. White (ORCID: [0000-0003-4191-3511](https://orcid.org/0000-0003-4191-3511))  
**Hub:** https://crvscience.com/apps

This directory contains the self-contained LaTeX source (`.tex`) and single-column PDF (`.pdf`) editions for both:

1. **Companion Application Preprint (`APP-PREPRINT-09`)**
   - **Title:** SAR-Atlas: Drug Design, Body Distribution Modeling, and Candidate Screening in Rust
   - **Files:** `sar-atlas-multiparameter-pbpk-rbfe.tex`, `sar-atlas-multiparameter-pbpk-rbfe.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222390
   - **Figshare Mirror:** https://figshare.com/articles/preprint/SAR-Atlas_Drug_Design_Body_Distribution_Modeling_and_Candidate_Screening_in_Rust/35013766
   - **Web Preprint:** https://crvscience.com/apps/preprints/sar-atlas-multiparameter-pbpk-rbfe

2. **CRV Science Public Apps Network Overview (`APP-PREPRINT-01`)**
   - **Title:** The CRV Science App Network: Twenty Open-Source Rust Programs for Science, Data, and Simulation
   - **Files:** `crv-science-public-apps-network.tex`, `crv-science-public-apps-network.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222398
   - **Figshare Mirror:** https://figshare.com/articles/preprint/The_CRV_Science_App_Network_Seventeen_Open-Source_Rust_Programs_for_Science_Data_and_Simulation/35013736
   - **Web Preprint:** https://crvscience.com/apps/preprints/crv-science-public-apps-network

## Compiling Standalone LaTeX

Both `.tex` files inline their bibliography so each compiles standalone with standard `pdflatex`:

```bash
pdflatex sar-atlas-multiparameter-pbpk-rbfe.tex
pdflatex crv-science-public-apps-network.tex
```
