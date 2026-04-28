<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of QuantLet: SFM_ch4_full_pipeline

Published in: Statistics of Financial Markets (SFM)

Description: End-to-end EMH weak-form testing pipeline on S&P 500 (^GSPC, 2015-2024) using real data via yfinance. Computes ACF with confidence band, Ljung-Box Q(10) and Q(20), Wald-Wolfowitz runs test on signs, variance ratio profile VR(q) for q in {2,5,10,20,60,120} with Lo-MacKinlay heteroskedasticity-robust 95% CI, Chow-Denning multiple variance ratio MV1 vs SMM critical value, stationary block bootstrap 95% CI for VR(q), and 1-year rolling rho_1 and Ljung-Box p-value (Adaptive Markets Hypothesis check). All numerical results printed to stdout for inclusion in seminar slides; produces 4-panel summary figure.

Keywords: EMH, weak form, ACF, Ljung-Box, runs test, variance ratio, Lo-MacKinlay, Chow-Denning, block bootstrap, AMH, rolling, S&P 500, yfinance

Author: Daniel Traian Pele

Submitted: Monday, 27 April 2026

Datafile: yfinance ^GSPC (2015-01-01 to 2024-12-31)

Output: sfm_ch4_full_pipeline.pdf

```
