import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
from sklearn.model_selection import train_test_split                         
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ---------------- Generate Report ----------------
def generate_report(model, X_test, y_test, filename="Model_Report.pdf"):
    """
    Generates a one-page PDF report with evaluation metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp_img.name)
    plt.close()

    # PDF generation
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Iris Classification Report</b>", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Accuracy: {acc:.4f}", styles['Normal']))
    story.append(Paragraph(f"Precision: {prec:.4f}", styles['Normal']))
    story.append(Paragraph(f"Recall: {rec:.4f}", styles['Normal']))
    story.append(Paragraph(f"F1-Score: {f1:.4f}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Confusion Matrix:", styles['Heading2']))
    story.append(Image(tmp_img.name, width=300, height=250))

    doc.build(story)
    print(f"✅ Report saved as {filename}")

# ---------------- Main ----------------
def main():
    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Split dataset
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = [
        ('LR', LogisticRegression(max_iter=200)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC())
    ]

    # Evaluate models
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results, names = [], []

    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

    # Compare algorithms
    plt.figure()
    plt.boxplot(results)
    plt.xticks(range(1, len(names)+1), names)
    plt.title("Algorithm Comparison")
    plt.show()

    # Choose final model (SVM for example)
    final_model = SVC()
    final_model.fit(X_train, y_train)

    predictions = final_model.predict(X_test)
    print("\nFinal Model Evaluation (SVM):")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    # Generate PDF report
    generate_report(final_model, X_test, y_test, "Iris_Report.pdf")

# Run
if __name__ == "__main__":
    main()
