This is the replication package of the paper "Concept-drift-adaptive anomaly detector for marine sensor data streams". In this package, you will find in the following folders:
01_data: data used in the study
	|- 01_label: data with anomalies that we identified together with domain experts. The data format follows [1]. AdapAD assumes new data subjects follows this data format. 
	|- 02_labelling_process: explanations of anomaly labelling process for the three datasets used in the study.
02_AdapAD_code: source code of AdapAD.
03_benchmark: results of the benchmark stage discussed in Section 3. See README.txt to replicate our experiments in the benchmark stage.
04_validation: results that we showed in Section 5. Validation.
05_misc: 
	|- 01_development_progress: experimental results of AdapAD v0.2 on the benchmark data.
	|- 02_compare_training_size: experimental results when we attempted to increase training size of semi-supervised algorithms during the benchmark study stage (see Section 2.1 in the paper).
	
Reference:
[1] Sebastian Schmidl, Phillip Wenig, and Thorsten Papenbrock. Anomaly detection in time series: a comprehensive evaluation. Proceedings of the VLDB Endowment, 15(9):1779–1797, 2022.

All the exeperiments of AdapAD and state-of-the-art algorithms were executed on:
- Ubuntu LTS 20.04 OS
- Python 3.8
- Docker
