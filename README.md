<p align="center">
<img src="./assets/headline.png" style="width: 70%;"/>
</p>

<h1 align="center">
<h2 align="center">Concept-drift-adaptive anomaly detector for marine sensor data streams</h2>

AdapAD is a semi-supervised anomaly detector designed to detect anomalies in univariate marine data streams. The operation flow of AdapAD is illustrated below.

<div align="center">
  <img src="./assets/operation_flow.png" alt="operation_flow" style="width: 70%;">
</div>

If you use AdapAD in your project or research, please cite the following paper:

- [Internet of Things, Elsevier, 2024](https://www.sciencedirect.com/science/article/pii/S254266052400355X)

### Reference

> Nguyen, N.T., Heldal, R. and Pelliccione, P., 2024. Concept-drift-adaptive anomaly detector for marine sensor data streams. Internet of Things, p.101414.

```bibtex
@article{nguyen2024concept,
  title={Concept-drift-adaptive anomaly detector for marine sensor data streams},
  author={Nguyen, Ngoc-Thanh and Heldal, Rogardt and Pelliccione, Patrizio},
  journal={Internet of Things},
  pages={101414},
  year={2024},
  publisher={Elsevier}
}
```

All the exeperiments of AdapAD and state-of-the-art algorithms were executed on:
- Ubuntu LTS 20.04 OS
- Python 3.8
- Docker

The framework can be used in a wide range of applications, especially in root cause analysis. In our work, we have demonstrated the usefulness and practicability of MuMSAD in two real-world applications, requested by many of industrial and research collaborators.
- Automatic multi-parameter marine data quality control. The figure below demonstrates the overview idea of the application.
- Automatic identification of malfunctioning sensors in Remotely Operated Vehicles, which is requested by one of our industrial collaborator.

<div align="center">
  <img src="./assets/motivation.PNG" alt="Motivation" style="width: 50%;">
</div>

Our work is under review at The IEEE International Conference on Data Engineering (ICDE) 2025 Industry and Applications Track.
MuMSAD is an extension of [MSAD](https://github.com/boniolp/MSAD), which is orignally designed for automatic selection of univariate anomaly detectors. MuMSAD is fully compatible to features supported in the original framework.

## Installation

To install and use MuMSAD from source, you will need the following tools:

- `git`
- `conda` (anaconda or miniconda)

#### Steps for installation

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/ntnguyen-so/AdapAD_alg
cd AdapAD_alg/02_AdapAD_code
```

**Step 2:** Create and activate a `conda` environment named `MSAD`.

```bash
conda env create --file environment.yml
conda activate AdapAD_alg
pip install -r requirements.txt
```

**Step 3:** Installation complete!

## Structure

Below you can find the structure of the repository.

- 01_data: data used in the study
  - 01_label: data with anomalies that we identified together with domain experts. The data format follows [1]. AdapAD assumes new data subjects follows this data format. 
  - 02_labelling_process: explanations of anomaly labelling process for the three datasets used in the study.
- 02_AdapAD_code: source code of AdapAD.
- 03_benchmark: results of the benchmark stage discussed in Section 3. See README.txt to replicate our experiments in the benchmark stage.
- 04_validation: results that we showed in Section 5. Validation.
- 05_misc: 
  - 01_development_progress: experimental results of AdapAD v0.2 on the benchmark data.
  - 02_compare_training_size: experimental results when we attempted to increase training size of semi-supervised algorithms during the benchmark study stage (see Section 2.1 in the paper).
	
