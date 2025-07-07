# GDM Diagnosis Using Machine Learning

This repository implements and benchmarks multiple machine‑learning pipelines for early detection of Gestational Diabetes Mellitus (GDM). The work extends the empirical study published in *Journal of Ayurveda and Integrative Medicine* in 2024, where we demonstrated the usefulness of ML models alongside Ayurvedic management strategies for GDM.([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/39662422/?utm_source=chatgpt.com))

## Repository structure

| File                                 | Purpose                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------- |
| `pima-indians-diabetes.csv`          | Source dataset with eight clinical attributes and binary diabetes outcome |
| `Imputations_with_RF_and_LR.ipynb`   | Baseline and model‑based imputation experiments                           |
| `RandomForestIterativeImputer.ipynb` | Iterative imputation with tree estimators                                 |
| `FeatureImportances.ipynb`           | Model‑driven feature importance analysis                                  |
| `Classifiers_with_wrapper.ipynb`     | Training and evaluation of individual classifiers using wrapper selection |
| `Ensemble_Learning.ipynb`            | Bagging, boosting and stacking ensembles                                  |

## Data

We use the Pima Indians Diabetes dataset (768 records, 8 predictors). Zeros in clinical columns except **Pregnancies** are treated as missing and replaced with median values before modelling.

| Column                   | Description                           |
| ------------------------ | ------------------------------------- |
| Pregnancies              | Number of pregnancies                 |
| Glucose                  | Plasma glucose concentration (2‑hour) |
| BloodPressure            | Diastolic blood pressure (mm Hg)      |
| SkinThickness            | Triceps skin fold thickness (mm)      |
| Insulin                  | 2‑hour serum insulin (mu U/ml)        |
| BMI                      | Body mass index (kg/m²)               |
| DiabetesPedigreeFunction | Family history score                  |
| Age                      | Age in years                          |
| Outcome                  | Diabetes diagnosis label              |

## Methods

### Imputation

* Simple: mean, median and mode
* Model‑based: Random Forest and Logistic Regression regressors
* Iterative: chained Random Forest regressors

### Models

* Logistic Regression
* Support Vector Machine
* K‑Nearest Neighbours
* Decision Tree and Random Forest
* Gradient Boosting (XGBoost or AdaBoost depending on environment)
* Voting and stacking ensembles

### Metrics

* Accuracy
* Precision, Recall, F1
* ROC‑AUC
* Confusion matrix with five‑fold stratified cross‑validation

## Quick results

| Model                         | Accuracy | ROC‑AUC |
| ----------------------------- | -------- | ------- |
| Logistic Regression (scaled)  | 0.708    | 0.813   |
| Random Forest (median impute) | 0.740    | 0.816   |
| Random Forest (raw zeros)     | 0.747    | 0.813   |

Top five features by Random Forest importance:

1. Glucose (0.27)
2. BMI (0.16)
3. Age (0.13)
4. DiabetesPedigreeFunction (0.13)
5. BloodPressure (0.09)

## How to reproduce

```bash
git clone https://github.com/SnehaDharne/gdm‑diagnosis.git
cd gdm‑diagnosis

# create and activate environment
conda create -n gdmml python=3.9
conda activate gdmml
pip install -r requirements.txt

# launch notebooks
jupyter lab
```

Run notebooks in the sequence listed in the Repository structure section.

## Citation

If you use this code or findings, please cite:

```
@article{shetty2024cdss,
  title={A machine learning‑based clinical decision support system for effective stratification of gestational diabetes mellitus and management through Ayurveda},
  author={Shetty, Nisha P and Shetty, Jayashree and Hegde, Veeraj and Dharne, Sneha D and Kv, Mamtha},
  journal={Journal of Ayurveda and Integrative Medicine},
  year={2024},
  volume={15},
  number={6},
  pages={101051},
  doi={10.1016/j.jaim.2024.101051}
}
```

## Author

Sneha Dharne
[snehadattadharne@gmail.com](mailto:snehadattadharne@gmail.com) • [LinkedIn](https://www.linkedin.com/in/snehadharne) • [GitHub](https://github.com/SnehaDharne)

Feel free to open issues or pull requests to improve the project.
