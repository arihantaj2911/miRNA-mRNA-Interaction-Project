miRNA–mRNA Interaction Prediction

A machine learning project to predict potential interactions between miRNA and mRNA sequences using Logistic Regression and Random Forest classifiers. This project supports bioinformatics research by applying supervised learning to biological sequence-derived features.

Repository Structure
miRNA-mRNA-Interaction-Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── src/
│   ├── feature_extraction.py
│   ├── model_training.py
│   
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│
├── results/
│   ├── metrics.txt
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│
├── README.md
├── requirements.txt
└── ai_usage.md

🧬 Project Overview

MicroRNAs regulate gene expression by binding to complementary mRNA sequences. Since experimental validation is expensive, computational models help predict potential miRNA–mRNA interactions efficiently.
This project builds two machine learning models:

Logistic Regression

Random Forest Classifier

Both models are evaluated using standard classification metrics.

⚙️ Features Used

Seed region complementarity

GC content

Minimum free energy (MFE)

Sequence alignment scores

Target accessibility

Interaction probability features

🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Preprocessing
python src/feature_extraction.py

3️⃣ Train Models
python src/model_training.py

4️⃣ View Results

All output files (metrics, ROC curves, confusion matrices) appear in the results/ folder.

📊 Model Performance Metrics
🔹 Logistic Regression Metrics
accuracy:   0.6562
precision:  0.6582
recall:     0.6500
f1_score:   0.6541
roc_auc:    0.7212

🔹 Random Forest Metrics
accuracy:   0.7406
precision:  1.0000
recall:     0.4813
f1_score:   0.6498
roc_auc:    0.7816

📈 Visualizations

Add images to the results/ folder and they will render here:

![Confusion Matrix]
<img width="1920" height="1440" alt="confusion_matrix_rf" src="https://github.com/user-attachments/assets/b40e5156-a301-4df3-8b33-0d880972c389" />

![ROC Curves](results/roc_curves.png)
<img width="1920" height="1440" alt="roc_curve_rf" src="https://github.com/user-attachments/assets/6da3b926-f826-4c31-92f5-26b9530f1deb" />


🧪 Tools & Technologies
Category	Tools
Programming	Python 3.9
ML Libraries	Scikit-learn, Pandas, NumPy
Visualization	Matplotlib, Seaborn
Bioinformatics	ViennaRNA (optional), BLAST (optional)
Notebook	Jupyter / Google Colab
📚 Dataset & References (APA Style)

Chou, C. H., Shrestha, S., Yang, C. D., et al. (2018). miRTarBase update 2018: a resource for experimentally validated microRNA–target interactions. Nucleic Acids Research, 46(D1), D296–D302.

Altschul, S. F., Gish, W., Miller, W., Myers, E. W., & Lipman, D. J. (1990). Basic local alignment search tool. Journal of Molecular Biology, 215(3), 403–410.

QIIME2 Development Team. (2024). QIIME 2 User Documentation. https://qiime2.org/
