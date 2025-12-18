import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="HR Attrition Analytics & Prediction",
    layout="wide"
)

# -----------------------------
# Load Dataset (SAFE)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("hr_attrition_final_dataset.csv")

df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔎 Filters")

dept = st.sidebar.multiselect(
    "Department", df["Department"].unique(), df["Department"].unique()
)
gender = st.sidebar.multiselect(
    "Gender", df["Gender"].unique(), df["Gender"].unique()
)
job = st.sidebar.multiselect(
    "Job Role", df["JobRole"].unique(), df["JobRole"].unique()
)

filtered_df = df[
    (df["Department"].isin(dept)) &
    (df["Gender"].isin(gender)) &
    (df["JobRole"].isin(job))
]

# -----------------------------
# KPI Section
# -----------------------------
st.title("📊 HR Attrition Analytics Dashboard")

total = len(filtered_df)
attrition_count = filtered_df[filtered_df["Attrition"] == "Yes"].shape[0]
attrition_rate = (attrition_count / total) * 100 if total else 0

c1, c2, c3 = st.columns(3)
c1.metric("👥 Total Employees", total)
c2.metric("🚪 Attrition Count", attrition_count)
c3.metric("📉 Attrition Rate (%)", f"{attrition_rate:.2f}")

st.divider()

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("📌 Attrition Distribution")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="Attrition", ax=ax)
st.pyplot(fig)

st.subheader("🏢 Attrition by Department")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="Department", hue="Attrition", ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

st.subheader("💰 Income vs Attrition")
fig, ax = plt.subplots()
sns.boxplot(data=filtered_df, x="Attrition", y="MonthlyIncome", ax=ax)
st.pyplot(fig)

st.subheader("😊 Satisfaction Correlation")
satisfaction_cols = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]
fig, ax = plt.subplots()
sns.heatmap(filtered_df[satisfaction_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.divider()

# -----------------------------
# MACHINE LEARNING
# -----------------------------
st.header("🤖 Attrition Prediction")

ml_df = df.copy()
le = LabelEncoder()

for col in ml_df.select_dtypes(include="object"):
    ml_df[col] = le.fit_transform(ml_df[col])

X = ml_df.drop("Attrition", axis=1)
y = ml_df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model_type = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

if model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.success(f"🎯 Model Accuracy: {acc*100:.2f}%")

st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

if model_type == "Random Forest":
    st.subheader("📌 Feature Importance")
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    imp.head(10).plot(kind="barh", ax=ax)
    st.pyplot(fig)

st.divider()

# -----------------------------
# Individual Prediction
# -----------------------------
st.header("🧍 Individual Employee Risk Prediction")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, float(X[col].mean()))

input_df = pd.DataFrame([input_data])
pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

if pred == 1:
    st.error(f"⚠️ High Attrition Risk ({prob*100:.2f}%)")
else:
    st.success(f"✅ Low Attrition Risk ({prob*100:.2f}%)")

st.divider()

st.subheader("📄 Filtered Data Preview")
st.dataframe(filtered_df, use_container_width=True)
