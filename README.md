# Wine Quality Prediction: SVM & Logistic Regression Implementation

## Project Overview

This project implements **Support Vector Machine (SVM)** and **Logistic Regression (LR)** algorithms from scratch to predict wine quality. The task is binary classification: predicting whether wines are "good" (quality ≥ 6) or "bad" (quality < 6) using both red and white wine datasets from the UCI Machine Learning Repository.

## Dataset

- **Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Files**: `winequality-red.csv` and `winequality-white.csv`
- **Features**: 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, etc.)
- **Total samples**: ~6,500 wines (red + white combined)

## Requirements & Usage

### Dependencies
- Python 3.8+
- NumPy
- Pandas  
- Matplotlib
- Seaborn
- Jupyter Notebook

### Installation & Execution
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in sequential order:
   - `notebooks/data_exploration.ipynb` - Dataset analysis and visualization
   - `notebooks/data_preprocessing.ipynb` - Feature engineering and data preparation
   - `notebooks/models_evaluation.ipynb` - Model training and performance evaluation

### Implementation Structure
All core implementations are located in the `src/` directory:
- `models.py` - SVM and Logistic Regression algorithm implementations
- `model_selection.py` - Grid search and cross-validation utilities  
- `util.py` - Performance metrics and helper functions

## Key Results

### Performance Summary
- **Logistic Regression** consistently outperforms SVM across all configurations
- **Polynomial kernels** improve both models, with LR showing more controlled gains
- **Best accuracy**: 77.5% (Polynomial LR) vs 74.5% (Polynomial SVM)

## Academic Integrity

This project was completed in accordance with academic integrity policies. All implementations are original work, with appropriate citations for theoretical foundations and mathematical formulations.

## License

This project is for educational purposes. The UCI Wine Quality dataset is publicly available under its original terms.

## Contact

- **Author**: Alessandro Sarchi
- **Email**: alessandro.sarchi02@gmail.com
- **Course**: Statistical methods for Machine Learning
- **Institution**: University of Milan

---

## Academic Declaration

*I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work. I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.*