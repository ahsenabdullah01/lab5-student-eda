import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],
    "Gender": [
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Female"
    ],
    "StudyHours": [
        2, 3, 4, 5, 6, 2.5,
        7, 4.5, 3.5, 6.5, 5.5, 5.5
    ],
    "Attendance(%)": [
        80, 85, 90, 75, 92, 78,
        95, 88, 82, 91, 89, 89
    ],
    "Marks": [
        65, 70, np.nan, 72, 85, 68,
        90, np.nan, 74, 82, 78, 78
    ]
}

pd.DataFrame(data).to_csv("student_performance.csv", index=False)

df = pd.read_csv("student_performance.csv")

print("Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df_clean = df.drop_duplicates().copy()

df_clean["Marks"] = df_clean["Marks"].fillna(
    df_clean["Marks"].median()
)

print("\nAfter cleaning:", df_clean.shape)

print("\nCleaned Dataset:")
print(df_clean)

plt.figure(figsize=(8, 5))
plt.hist(df_clean["Marks"], bins=6, edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Marks Distribution")
plt.savefig("marks_histogram.png")
plt.show()

plt.figure(figsize=(6, 5))
plt.boxplot(df_clean["Marks"])
plt.ylabel("Marks")
plt.title("Marks Boxplot")
plt.savefig("marks_boxplot.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(
    df_clean["StudyHours"],
    df_clean["Marks"]
)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.savefig("studyhours_vs_marks.png")
plt.show()

print("\nCorrelation Matrix:")
correlation_matrix = df_clean[
    ["StudyHours", "Attendance(%)", "Marks"]
].corr()

print(correlation_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("\nAverage Marks and Attendance by Gender:")

gender_summary = df_clean.groupby("Gender")[
    ["Marks", "Attendance(%)"]
].mean()

print(gender_summary)

print("\nSaved:")
print("student_performance.csv")
print("marks_histogram.png")
print("marks_boxplot.png")
print("studyhours_vs_marks.png")
print("correlation_heatmap.png")