import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ========== Report Function ==========
def generate_report(model, X_train, y_train, X_test, y_test, filename="BreastCancer_Report.pdf"):
    """
    Generates a one-page PDF report with evaluation metrics and training-validation plots.
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Learning Curve (Train vs Validation Accuracy)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="accuracy",
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
    plt.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()

    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp_img.name)
    plt.close()

    # PDF Report
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Breast Cancer Classification Report</b>", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Accuracy: {acc:.4f}", styles['Normal']))
    story.append(Paragraph(f"Precision: {prec:.4f}", styles['Normal']))
    story.append(Paragraph(f"Recall: {rec:.4f}", styles['Normal']))
    story.append(Paragraph(f"F1-Score: {f1:.4f}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Training-Validation Accuracy Plot:", styles['Heading2']))
    story.append(Image(tmp_img.name, width=400, height=250))

    doc.build(story)
    print(f"✅ Report saved as {filename}")


# ========== Main Function ==========
def main():
    # Load Dataset
    df = pd.read_csv("breast-cancer.csv")
    print(df.head())

    # Encode Diagnosis
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Drop less useful columns
    drop_cols = [
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
        'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst',
        'perimeter_mean','perimeter_se','area_mean','area_se',
        'concavity_mean','concavity_se','concave points_mean','concave points_se'
    ]
    df = df.drop(drop_cols, axis=1)

    # Features & Target
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    # Standardize
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # Models
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("Naive Bayes", GaussianNB()),
        ("SVM", SVC())
    ]

    print("\n--- Model Evaluation ---")
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

    # Choose Final Model (Random Forest)
    final_model = RandomForestClassifier()
    final_model.fit(X_train, y_train)

    # Generate Report
    generate_report(final_model, X_train, y_train, X_test, y_test, "BreastCancer_Report.pdf")


# ========== Run ==========
if __name__ == "__main__":
    main()
