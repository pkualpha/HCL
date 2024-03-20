1. requirements:
tensorflow 1.15 for data preprocessing

2. modify storage path in global_settings.py

3. download dataset and organize as following:
${raw_data}/mimic/PROCEDURES_ICD.csv
${raw_data}/mimic/LABEVENTS.csv
${raw_data}/mimic/DIAGNOSES_ICD.csv
${raw_data}/mimic/ADMISSIONS.csv
${raw_data}/eicu/patient.csv
${raw_data}/eicu/diagnosis.csv
${raw_data}/eicu/admissionDx.csv
${raw_data}/eicu/treatment.csv

4. data preprocess
train/val/test split for 10 times
python -m data_preprocessing.preprocess_eicu
python -m data_preprocessing.preprocess_mimic


python -m data_preprocessing.graph_construction_eicu
python -m data_preprocessing.graph_construction_mimic


5. train
python sweep_hcl.py --task {ihm/readmission} --fold {i} --gpu {i} --hgnn_num_layer 1 --dataset {eicu/mimic} --bs 512 --gnn_name sage
