# 1 GTEx experiments 
## Data
- For those whose are interested in the main GTEx experiments in Fig. 2, a smaller data for the tissue *Adipose_Subcutaneous* is available at [GTEx_Adipose_Subcutaneous](https://osf.io/c5yk6/)

- The data for all GTEx experiments is available at [GTEx_full](https://osf.io/9vrax/) 

## 1.1 Multiple testing for 17 tissues with four covariates (Fig. 2a)
- Code:
  - Main script: <br/>
  `AdaFDRpaper/experiments/analysis_gtex.py`
  - Testing for tissue `TISSUE_NAME` <br/>
  `python ./experiments/analysis_gtex.py -d 'load_GTEx' -o 'GTEx_TISSUE_NAME' -n 'TISSUE_NAME' `
  - Script for bach processing <br/>
  `AdaFDRpaper/experiments/call_analysis_gtex_0.sh` <br/>
  `AdaFDRpaper/experiments/call_analysis_gtex_1.sh` <br/>
  `AdaFDRpaper/experiments/call_analysis_gtex_2.sh` <br/>
  `AdaFDRpaper/experiments/call_analysis_gtex_3.sh` <br/>
  
- Data: <br/>
(`TISSUE_NAME` can be any of the 17 tissues) <br/>
`GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered` <br/>
The same data for *Adipose_Subcutaneous* is also available at <br/>
`GTEx_Adipose_Subcutaneous/Adipose_Subcutaneous.allpairs.txt.processed.filtered`

- Data format: <br/>
  Only data with p-values `P<0.01` or `P>0.99` are kept <br/>
  col1: gene-SNP name; col2: gene expression; col3: AAF; col4: distance from TSS; 
  col5: chromatin states (15 states model, used by the paper); col6: chromatin states 
  (15 states model, not used by the paper); col7: p-value;

## 1.2 Multiple testing with augmented p-values (Fig. 2b, Supp. Fig. 1a)
This experiment is done for four tissues `{Adipose_Subcutaneous, Adipose_Visceral_Omentum, Colon_Sigmoid, Colon_Transverse}`
- Code:
  - Main script: <br/>
  `AdaFDRpaper/experiments/analysis_gtex.py`
  - Testing with augmented p-value from a related tissue for the tissue `TISSUE_NAME` being 
  one of `{Adipose_Subcutaneous, Adipose_Visceral_Omentum, Colon_Sigmoid, Colon_Transverse}` <br/>
  `python ./experiments/analysis_gtex.py -d 'load_GTEx' -o 'GTEx_TISSUE_NAME_aug' -n 'TISSUE_NAME-aug'`
  - Testing with augmented p-value from an unrelated tissue for the tissue `TISSUE_NAME` being 
  one of `{Adipose_Subcutaneous, Adipose_Visceral_Omentum, Colon_Sigmoid, Colon_Transverse}` <br/>
  `python ./experiments/analysis_gtex.py -d 'load_GTEx' -o 'GTEx_TISSUE_NAME_a_ur' -n 'TISSUE_NAME-a_ur'`
  - Script for bach processing <br/>
  `AdaFDRpaper/experiments/call_analysis_gtex_0.sh` <br/>
  `AdaFDRpaper/experiments/call_analysis_gtex_1.sh` <br/>
  
- Data: <br/>
(`TISSUE_NAME` can be one of `{Adipose_Subcutaneous, Adipose_Visceral_Omentum, Colon_Sigmoid, Colon_Transverse}`)
  - Augmented p-value from a related tissue
 `GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered.augmented.txt`
  - Augmented p-value from an unrelated tissue
 `GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered.augmented_not_related.txt`
 
- Data format: <br/>
  Only data with p-values `P<0.01` or `P>0.99` are kept <br/>
  col1: gene-SNP name; col2: gene expression; col3: AAF; col4: distance from TSS; 
  col5: chromatin states (15 states model, used by the paper); col6: chromatin states 
  (15 states model, not used by the paper); col7: p-value; col8: augmented p-value

## 1.3 Validation by the MuTHER data (Fig. 2d, Supp. Fig. 1c)
This experiment is done for three tissues: 
  1. Use the MuTHER adipose eQTL data to validate GTEx Adipose_Subcutaneous discoveries
  2. Use the MuTHER adipose eQTL data to validate GTEx Adipose_Visceral_Omentum discoveries
  3. Use the MuTHER lymphocytes (LCL) eQTL data to validate GTEx Cells_EBV-transformed_lymphocytes discoveries

The details are as follows:
- Code:
  1. Match the GTEx discoveries with the MuTHER p-values: <br/>
      - The two adipose tissues:
      `AdaFDRpaper/experiments_v1/generate_GTEx_MuTHER_comparison_data_adipose.py`
      - The Cells_EBV-transformed_lymphocytes tissue: <br/>
      `AdaFDRpaper/experiments_v1/generate_GTEx_MuTHER_comparison_data_LCL.py`
  2. Generate the figures: <br/>
  `AdaFDRpaper/experiments_v1/validate_GTEx_by_MuTHER.ipynb`
  
- Data: <br/>
  - The GTEx discoveries from experiment 1.1: <br/>
  Here we save the testing results. The result for `TISSUE_NAME` is available at <br/>
  `GTEx_full/test_results/result_GTEx_TISSUE_NAME.pickle`
  - The MuTHER data: <br/>
  `GTEx_full/MuTHER_validation/MuTHER_cis_results_chrall.txt`<br/>
  Data format: col1: chromesome number; col2: gene name; col3: SNP ID; col 4: p-value for fat (adipose); col 5: p-value for LCL; p-value for skin (not used)
  - Other data: <br/>
    - SNP information:<br/>
    `GTEx_full/MuTHER_validation/snp_feat.txt`<br/>
    Data format: col1: SNP symbol; col2: SNP ID
    - Gene information:<br/>
    `GTEx_full/MuTHER_validation/gencode.v19.genes.patched_contigs.gtf`<br/>
    Gene symbol and gene ID
    
## 1.4 Validation by the MuTHER data (Supp. Fig. 2)
- Code: <br/>
`AdaFDRpaper/experiments_v1/GTEx_AdaFDR_only_result.ipynb`

- Data: The result for `TISSUE_NAME` is available at <br/>
  `GTEx_full/test_results/result_GTEx_TISSUE_NAME.pickle`
  
## 1.5 Multiple testing for 17 tissues with each covariate separately (Supp. Fig. 3)
- Code:
  - Main script: <br/>
  `AdaFDRpaper/experiments_v1/analysis_gtex_uni_covariate.py.py`
  - Testing for tissue `TISSUE_NAME` <br/>
  `python ./experiments_v1/analysis_gtex_uni_covariate.py -d 'load_GTEx' -o 'GTEx_TISSUE_NAME' -n 'TISSUE_NAME' `
  - Script for bach processing <br/>
  `AdaFDRpaper/experiments_v1/call_analysis_gtex_uni_covariate_0.sh` <br/>
  `AdaFDRpaper/experiments_v1/call_analysis_gtex_uni_covariate_1.sh` <br/>
  
- Data: <br/>
(`TISSUE_NAME` can be any of the 17 tissues) <br/>
`GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered` <br/>
The same data for *Adipose_Subcutaneous* is also available at <br/>
`GTEx_Adipose_Subcutaneous/Adipose_Subcutaneous.allpairs.txt.processed.filtered`

- Data format: <br/>
  Only data with p-values `P<0.01` or `P>0.99` are kept <br/>
  col1: gene-SNP name; col2: gene expression; col3: AAF; col4: distance from TSS; 
  col5: chromatin states (15 states model, used by the paper); col6: chromatin states 
  (15 states model, not used by the paper); col7: p-value;
  
## 1.6 Assumption check (Supp. Figs. 4-5)
The experiment requires to use the full data without p-value filtering, which can be very large. The data is hence omitted. The corresponding code is <br/>
`GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered`

# 2. Other experiments
## 2.1 Schematic (Fig. 1. Fig. 5)
- Code: <br/>
`AdaFDRpaper/experiments/generate_figure_schematic.ipynb`

## 2.2 Other applications (Fig. 3) and simulations (Fig. 4, Supp. Figs. 8-10)
See `README.md`

## 2.3 Algorithm stability (Supp. Fig. 7)
- Code: <br/>
`AdaFDRpaper/experiments_v1/algorithm_stability.ipynb`





