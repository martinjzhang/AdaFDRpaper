import os

data_path = '/data3/martin/gtex_data'
metadata_path = os.path.join(data_path, 'gtex_metadata.csv')
utils_path = os.path.join(data_path, 'gtex_utils')
association_path = os.path.join(data_path, 'GTEx_Analysis_v7_eQTL_all_associations')

files = [
"Adipose_Subcutaneous.allpairs.txt",
"Adipose_Visceral_Omentum.allpairs.txt",
"Artery_Aorta.allpairs.txt",
"Breast_Mammary_Tissue.allpairs.txt",
"Cells_EBV-transformed_lymphocytes.allpairs.txt",
"Colon_Sigmoid.allpairs.txt",
"Colon_Transverse.allpairs.txt",
"Esophagus_Gastroesophageal_Junction.allpairs.txt",
"Esophagus_Mucosa.allpairs.txt",
"Esophagus_Muscularis.allpairs.txt",
"Heart_Atrial_Appendage.allpairs.txt",
"Heart_Left_Ventricle.allpairs.txt",
"Lung.allpairs.txt",
"Muscle_Skeletal.allpairs.txt",
"Pancreas.allpairs.txt",
"Stomach.allpairs.txt",
"Whole_Blood.allpairs.txt"
]

roadmap_dict = {
'Adipose_Subcutaneous.allpairs.txt': 'E063',
'Adipose_Visceral_Omentum.allpairs.txt': 'E063',
'Artery_Aorta.allpairs.txt': 'E065',
'Breast_Mammary_Tissue.allpairs.txt': 'E027',
'Cells_EBV-transformed_lymphocytes.allpairs.txt': 'E116',
'Colon_Sigmoid.allpairs.txt': 'E106',
'Colon_Transverse.allpairs.txt': 'E075',# 'E076',
'Esophagus_Gastroesophageal_Junction.allpairs.txt': 'E079',
'Esophagus_Mucosa.allpairs.txt': 'E079',
'Esophagus_Muscularis.allpairs.txt': 'E079',
'Heart_Atrial_Appendage.allpairs.txt': 'E104',
'Heart_Left_Ventricle.allpairs.txt': 'E095',
'Lung.allpairs.txt': 'E096',
'Muscle_Skeletal.allpairs.txt': 'E107', #'E108',
'Pancreas.allpairs.txt': 'E098',
'Stomach.allpairs.txt': 'E110', #'E111',
'Whole_Blood.allpairs.txt': 'E062'}

tissue_dict = {
"Adipose_Subcutaneous.allpairs.txt": 'Adipose - Subcutaneous',
"Adipose_Visceral_Omentum.allpairs.txt": 'Adipose - Visceral (Omentum)',
"Artery_Aorta.allpairs.txt": 'Artery - Aorta' ,
"Breast_Mammary_Tissue.allpairs.txt":  'Breast - Mammary Tissue',
"Cells_EBV-transformed_lymphocytes.allpairs.txt":  'Cells - EBV-transformed lymphocytes',
"Colon_Sigmoid.allpairs.txt": 'Colon - Sigmoid',
"Colon_Transverse.allpairs.txt": 'Colon - Transverse',
"Esophagus_Gastroesophageal_Junction.allpairs.txt": 'Esophagus - Gastroesophageal Junction',
"Esophagus_Mucosa.allpairs.txt": 'Esophagus - Mucosa',
"Esophagus_Muscularis.allpairs.txt": 'Esophagus - Muscularis',
"Heart_Atrial_Appendage.allpairs.txt": 'Heart - Atrial Appendage',
"Heart_Left_Ventricle.allpairs.txt": 'Heart - Left Ventricle',
"Lung.allpairs.txt": 'Lung',
"Muscle_Skeletal.allpairs.txt":  'Muscle - Skeletal',
"Pancreas.allpairs.txt":  'Pancreas',
"Stomach.allpairs.txt": 'Stomach',
"Whole_Blood.allpairs.txt": 'Whole Blood'
}
