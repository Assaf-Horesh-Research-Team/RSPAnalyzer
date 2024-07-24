# RSPAnalyzer

This library is designed for analyzing the physical parameters of 
light curves and spectra from various radio supernovae. 
It uses Chevaliers 1998 Equation 1 model and the MCMC algorithm.

It requires as an input a csv file which contains the measured values, 
and it must be on the following format:

Its columns should be:
delta_t | flux | flux_err | freq 

where:
delta_t is the days since explosion in days units
flux is the flux measured in mJY units
flux_err is the err in the flux (as stated on the article/already proccessed as you want) also in mJY
freq is the telescope's frequency in Ghz



## Installation

```bash
pip install RSPanalyzer
