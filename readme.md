# README

This repository contains a survival analysis pipeline for machine learning model development and validation using Monte Carlo Cross-Validation (MCCV) with 100 simulations. External validation of the model is also included.
The goal is to develop a pre-operative (T0) prognostic model for Renal Cell Carcinoma (RCC), with cancer-specific mortality (CSM) as the endpoint.

## Project Structure

The notebooks and scripts are designed to be **run in sequence** based on their numerical prefix (00, 01, 02, etc.). Each step builds upon the previous ones.

### Execution Order

#### 00 - Data Preprocessing
- **Goal**: Clean and preprocess raw data for modeling
- `00_preprocessing_larcher.ipynb` - Process data starting from a dataset curated by a Clinician
- `00_preprocessing_raw.ipynb` - Process data starting from the Raw dataset

#### 01 - Exploratory Data Analysis
- **Goal**: Understand data distribution, patterns, and relationships
- `01_eda_raw.ipynb` - EDA and Kaplan-Meier curves on the pre-processed dataset

#### 02 - Survival Model Fine-tuning (GRANT Features)
- **Goal**: Fine-tune Cox models using GRANT features as baseline with comprehensive validation
- Uses MCCV with 100 simulations via papermill orchestration
- `02_survival_grant_finetune_*.ipynb` - Core fine-tuning notebooks
- `02_monte_carlo_orchestrator.ipynb` - Orchestrator for MCCV simulations using papermill
- `02_monte_carlo_collector.ipynb` - Collects metrics from all MCCV simulations

#### 03 - Feature Selection
- **Goal**: Perform feature selection using Random Survival Forest and analyze feature importance (permutation importance)
- Uses MCCV with 100 simulations via papermill orchestration
- `03_survival_feature_selection_*.ipynb` - Feature selection notebooks
- `03_monte_carlo_orchestrator.ipynb` - Orchestrator for MCCV simulations using papermill
- `03_monte_carlo_collector.ipynb` - Collects importance metrics

#### 04 - Survival Models Training
- **Goal**: Train and validate various survival models with selected features
- Uses cluster-based parallel execution for computationally intensive models
- `04_survival_models_raw_csm.py` - Core training script
- `04_monte_carlo_orchestrator_cluster.ipynb` - Orchestrator for MCCV simulations using Azure cluster
- `04_monte_carlo_collector.ipynb` - Collects model performance metrics

#### 05 - Model Comparison
- **Goal**: Compare performance of all trained models (internal validation)
- `05_compare_models.ipynb` - Comprehensive model comparison over the MCCV simulations, and selection of the best model

#### 06 - Final Model Training
- **Goal**: Train best performing model on full (internal) dataset
- `06_train_full_dataset.ipynb` - Train final model on 100% of internal data

#### 07 - Explainable AI (XAI)
- **Goal**: Generate model interpretability analysis
- `07_xai.ipynb` - Feature importance, SHAP values, and model explanation

#### 08 - External Validation
- **Goal**: Validate final model on external dataset with risk stratification
- `08_external_validation_risk_stratification_bootstrap.ipynb` - Bootstrap validation and Kaplan-Meier curves for the risk groups

#### 09 - Review - General comments
- **Goal**: Address review comments
- `09_review_comments.ipynb` - Handle general Reviewer feedback
- `09_review_comments_eda.ipynb` - Generate additional plots

#### 10 - Review - SSIGN model 
- **Goal**: Create an univariate Cox model using the SSIGN score as feature (now abbreviated as SSIGN model), same methodology as in paper https://pmc.ncbi.nlm.nih.gov/articles/PMC5536178/
- Uses MCCV with 100 simulations via papermill orchestration
- `10_review_survival_ssign_finetune*.ipynb` - Core notebook
- `10_monte_carlo_orchestrator.ipynb` - Orchestrator for MCCV simulations using papermill
- `10_monte_carlo_collector.ipynb` - Collects metrics from all MCCV simulations

#### 11 - Review - Model Comparison
- **Goal**: Compare performance of all trained models (internal validation), incliding the SSIGN model
- `11_review_compare_models.ipynb` - Comprehensive model comparison over the MCCV simulations

#### 12 - Review - Refit
- **Goal**: Train SSIGN model on full (internal) dataset
- `12_review_train_full_dataset.ipynb` - Train SSIGN model on 100% of internal data

#### 13 - Review - External Validation
- **Goal**: Validate SSIGN model on external dataset, compare performance with the models from the main manuscript
- `13_review_external_validation_bootstrap.ipynb` - Bootstrap validation on external dataset

#### 14 - Review - Byun Model Recreation
- **Goal**: Re-create Cox model from Byun et al. paper and validate on DBURI dataset
- `14_byun_from_hr.ipynb` - Recreate Byun model using hazard ratios from published paper and validate on external data

#### 15 - Review - Inclusion Criteria Validation
- **Goal**: Verify patient counts and data integrity after applying inclusion/exclusion criteria
- `15_check_inclusion_drop.ipynb` - Check numerical consistency of patient filtering process

#### 16 - Review - Preprocessing Feature Tracking
- **Goal**: Document feature retention through each preprocessing step
- `16_review_preprocessing_larcher.ipynb` - Track features through Clinician's preprocessing pipeline
- `16_review_preprocessing_raw.ipynb` - Track features through Raw preprocessing pipeline
- CSV files document retained features after each preprocessing step (step1-9) for both pipelines

### Validation Methodology

The project employs **Monte Carlo Cross-Validation (MCCV) with 100 simulations** for robust internal validation:

- **Low-training models**: Use `papermill` to execute notebooks with different random seeds
- **Computationally intensive models**: Use cluster-based parallel execution  
- **Architecture**: 
  - **Orchestrator**: Injects random seeds into notebooks/scripts and manages parallel execution
  - **Collector**: Aggregates metrics from all experiments for analysis and comparison

Ref. Chapter 8.5.2 from https://www.ncbi.nlm.nih.gov/books/NBK543527/pdf/Bookshelf_NBK543527.pdf 
### Key Directories

- `artifacts/` - Stored model metrics and results
- `papermill/` - Papermill-generated notebook executions  
- `figures/` - Generated plots and visualizations
- `src/` - Source code and utility functions

### Environment

- Environment configuration: `env_yaml/uc2_pat.yml`
- Utility functions: `04_survival_models/src/uc2_functions.py`
