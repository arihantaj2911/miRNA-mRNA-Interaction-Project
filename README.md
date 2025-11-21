# Predicting miRNA–mRNA Interactions Using Random Forest

## Project Description
This project predicts miRNA–mRNA interactions using sequence-derived features and a Random Forest model. The goal is to identify key features influencing interactions and achieve high predictive performance.

## Repository Structure
```
data/          # Sample datasets
src/           # Python scripts for feature extraction and model training
notebooks/     # Jupyter notebooks for analysis and visualization
report/        # Project report and figures
README.md      # Project overview and instructions
ai_usuage.md
```

## Installation
To install required packages, run:
```
pip install -r requirements.txt
```

## Usage
- Train and evaluate the Random Forest model:
```
python src/model.py
```
- Explore data and visualizations:
```
jupyter notebook notebooks/eda.ipynb
```

## Key Features & Insights
- Sequence-derived features can predict miRNA–mRNA interactions.
- Random Forest model achieves ROC-AUC > 0.75.
- Seed region matching is the most influential feature.

## Next Steps / Future Work
- Improve negative sampling to reduce noise.
- Test additional features (e.g., minimum free energy, binding-site conservation).
- Fine-tune Random Forest hyperparameters and optimize performance.
- Enhance visualization and interpretability.
- Prepare final polished report with complete figures.


