## AdaFDR

Git repo to reproduce the results of the paper "AdaFDR: a Fast, Powerful and Covariate-Adaptive Approach to Multiple Hypothesis Testing", 2018.

The python package can be found at the git repo [adafdr](https://github.com/martinjzhang/adafdr)

## RNA-seq experiments
### Python methods 
The three RNA-seq datasets are incorporated as part of the *adafdr* package and hence can be loaded directly. 
Jupyter notebook files are provided to reproduce the results of *adafdr*, along with baseline methods *BH* and *SBH*.
- airway: `./vignettes/airway.ipynb`
- bottomly: `./vignettes/bottomly.ipynb`
- passila: `./vignettes/passila.ipynb`

### R methods

## Simulations
### Data
The simulation data can be found in the data folder [AdaFDRpaper_data]()
The simulation data (Fig. 3, Supp. Fig. 2) can be found in the data folder, each with ten repetitions
generated using different random seeds. 
- Simulated data with one covariate: `./simu_1d_bump_slope`
- Simulated data with weakly-dependent p-values: `./simu_data_wd`
- Simulated data with strongly-dependent p-values: `./simu_data_sd`
- Simulated data used in AdaPT: `./data_adapt`
- Simulated data used in IHW: `./simu_data_ihw`
- Simulated data with two covariates: `./simu_2d_bump_slope`
- Simulated data with ten covariates: `./simu_10d_bump_slope`

The simulation from the SummarizedBenchmark paper:
- Yeast RNA-Seq *in silico* experiment: `./data_yeast` 
- Polyester RNA-Seq *in silico* experiment: `./data_polyester` 
- Polyester RNA-Seq *in silico* experiment with uninformative covariate: `./data_polyester_ui` 
- Simulation varying the number of tests: `./data_ntest` 
- Simulation varying the non-null proportion: `./data_prop_alt` 

### Python methods 
All simulation data can be downloaded from `...`
### R methods 

## GTEx data
### Python methods 
Only the GTEx data for the two adipose tissues are provided, which can be downloaded from `...`
### R methods

## Comparison with MuTHER data
