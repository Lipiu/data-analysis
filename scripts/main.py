import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('dataset/Heart_Disease_Prediction.csv')
df.columns = [
    "Age", "Sex", "ChestPain", "BP", "Cholesterol", "FBS",
    "EKG", "MaxHR", "ExerciseAngina", "STDepression",
    "Slope", "NumVessels", "Thallium", "HeartDisease"
]

df["HeartDisease"] = df["HeartDisease"].map({
    "Presence": 1,
    "Absence": 0
})

counts = df["HeartDisease"].value_counts().sort_index()

plt.bar(["Absence (0)", "Presence (1)"], counts.values)
plt.title("Heart Disease Class Distribution")
plt.ylabel("Count")
plt.show()


X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

y_pred = lda.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.colorbar()

plt.xticks([0, 1], ["Absence", "Presence"])
plt.yticks([0, 1], ["Absence", "Presence"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred, target_names=["Absence", "Presence"]))

X_lda = lda.transform(X_train_scaled)

plt.hist(X_lda[y_train == 0], bins=30, alpha=0.6, label="Absence")
plt.hist(X_lda[y_train == 1], bins=30, alpha=0.6, label="Presence")

plt.title("LDA Projection")
plt.xlabel("LDA Component 1")
plt.ylabel("Frequency")
plt.legend()
plt.show()

coefficients = pd.Series(
    lda.coef_[0],
    index = X.columns
).sort_values(key=abs)

plt.figure(figsize=(8,6))
plt.barh(coefficients.index, coefficients.values)
plt.title("LDA Feature Importance")
plt.xlabel("Coefficient Value")
plt.show()

print(coefficients)

age_absence_disease = df[df["HeartDisease"] == 0]["Age"]
age_presence_disease = df[df["HeartDisease"] == 1]["Age"]
plt.hist(age_absence_disease, bins=20, alpha=0.6, label="Absence")
plt.hist(age_presence_disease, bins=30, alpha=0.6, label="Presence")
plt.title("Age Distribution by Heart Disease Status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

bp_absence = df[df["HeartDisease"] == 0]["BP"]
bp_presence = df[df["HeartDisease"] == 1]["BP"]
plt.hist(bp_absence, bins=20, alpha=0.6, label="Absence")
plt.hist(bp_presence, bins=20, alpha=0.6, label="Presence")
plt.title("Reseting Blood Pressure by Heart Disease Status")
plt.xlabel("Blood Pressure (mm Hg)")
plt.ylabel("Count")
plt.legend()
plt.show()

chol_absence = df[df["HeartDisease"] == 0]["Cholesterol"]
chol_presence = df[df["HeartDisease"] == 1]["Cholesterol"]
plt.hist(chol_absence, bins=25, alpha=0.6, label="Absence")
plt.hist(chol_presence, bins=25, alpha=0.6, label="Presence")
plt.title("Cholesterol Levels by Heart Disease Status")
plt.xlabel("Cholesterol (mg/dl)")
plt.ylabel("Count")
plt.legend()
plt.show()

fbs_counts = pd.crosstab(df["FBS"], df["HeartDisease"])
fbs_counts.plot(kind="bar")
plt.title("Fasting Blood Sugar vs Heart Disease")
plt.xlabel("FBS (0 = ≤120 mg/dl, 1 = >120 mg/dl)")
plt.ylabel("Count")
plt.legend(["Absence", "Presence"])
plt.show()