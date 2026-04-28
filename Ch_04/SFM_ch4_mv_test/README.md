<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of QuantLet: SFM_ch4_mv_test

Published in: Statistics of Financial Markets (SFM)

Description: Chow & Denning (1993) Multiple Variance Ratio test on five markets (S&P 500, EUR/USD, Gold, EEM, Bitcoin) using horizons q in {2, 5, 10, 20}. Computes individual heteroscedasticity-robust Z*(q) statistics (Lo--MacKinlay 1988) and the MV1 = max_q |Z*(q)| statistic, comparing to the Studentized Maximum Modulus (SMM) critical value at the 5% level. Avoids size distortion of running m separate VR tests with Bonferroni correction.

Keywords: multiple variance ratio, Chow Denning, joint test, SMM, Studentized Maximum Modulus, EMH, weak form, Lo MacKinlay, yfinance

Author: Daniel Traian Pele

Submitted: Saturday, 26 April 2026

Datafile: yfinance ^GSPC, EURUSD=X, GC=F, EEM, BTC-USD (2018-2025)

Output: sfm_ch4_mv_test.pdf

```
