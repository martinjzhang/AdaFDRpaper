# GTEx experiments 
## Data
- For those whose are interested in the main GTEx experiments in Fig. 2, a smaller data for the tissue *Adipose_Subcutaneous* is available at [GTEx_Adipose_Subcutaneous](https://osf.io/c5yk6/)

- The data for all GTEx experiments is available at [GTEx_full](aaa) 

## Multiple testing for 17 tissues with four covariates (Fig. 2a)
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
  col1: gene-SNP name; col2: gene expression; col3: AAF; col4: distance from TSS; 
  col5: chromatin states (15 states model, used by the paper); col6: chromatin states 
  (15 states model, not used by the paper); col7: p-value;

## Multiple testing with augmented p-values (Fig. 2b, Supp. Fig. 1a)
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
(`TISSUE_NAME` can be one of {Adipose_Subcutaneous, Adipose_Visceral_Omentum, Colon_Sigmoid, Colon_Transverse})
  - Augmented p-value from a related tissue
 `GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered.augmented.txt`
  - Augmented p-value from an unrelated tissue
 `GTEx_full/GTEx_data_17_tissue/TISSUE_NAME.allpairs.txt.processed.filtered.augmented_not_related.txt`
 
- Data format: <br/>
  col1: gene-SNP name; col2: gene expression; col3: AAF; col4: distance from TSS; 
  col5: chromatin states (15 states model, used by the paper); col6: chromatin states 
  (15 states model, not used by the paper); col7: p-value; col8: augmented p-value

## Validation by the MuTHER data





# Other experiments
