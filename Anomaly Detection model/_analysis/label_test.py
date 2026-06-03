import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

df = pd.read_csv("labels_GWIMC.csv")

# Convert floats to ints
df["Predicted labels"] = df["Predicted labels"].astype(int)
df["Ground truth labels"] = df["Ground truth labels"].astype(int)

y_pred = df["Predicted labels"]
y_true = df["Ground truth labels"]

cm = confusion_matrix(y_true, y_pred)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Confusion Matrix:\n", cm)


class_names = ["anomalous", "non-anomalous"]
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
    yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
