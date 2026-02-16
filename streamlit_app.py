import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
# -----------------------------
# Step 1: Load Data
# -----------------------------
st.header("Step 1: Load Data")

df = pd.read_csv("chrun.csv")

st.write("Raw Data Preview")
st.dataframe(df.head())

# -----------------------------
# Step 2: Data Cleaning
# -----------------------------
st.header("Step 2: Data Cleaning")

# Drop customerID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

st.write("Shape after cleaning:", df.shape)

# -----------------------------
# Step 3: Encode Target
# -----------------------------
st.header("Step 3: Encode Target")

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

st.write("Encoded Churn Column:")
st.dataframe(df[['Churn']].head())


# -----------------------------
# Step 4: Feature / Target Split
# -----------------------------
st.header("Step 4: Feature Engineering")

X = df.drop('Churn', axis=1)
y = df['Churn']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

st.write("Features Shape:", X.shape)

# -----------------------------
# Step 5: Train/Test Split
# -----------------------------
st.header("Step 5: Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Training size:", X_train.shape)
st.write("Testing size:", X_test.shape)

# -----------------------------
# Step 6: Train Model
# -----------------------------
st.header("Step 6: Train Random Forest Model")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

# -----------------------------
# Step 7: Model Evaluation
# -----------------------------
st.header("Step 7: Model Evaluation")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"Accuracy: {accuracy:.2f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# -----------------------------
# Step 8: Churn Probability
# -----------------------------
st.header("Step 8: Churn Probability")

y_prob = model.predict_proba(X_test)[:, 1]

df_test = X_test.copy()
df_test['Churn_Prob'] = y_prob

st.dataframe(df_test.head())

# -----------------------------
# Step 9: Risk Level Classification
# -----------------------------
st.header("Step 9: Risk Level Classification")

def risk_level(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df_test['Risk_Level'] = df_test['Churn_Prob'].apply(risk_level)

st.dataframe(df_test[['Churn_Prob','Risk_Level']].head())

# Risk Distribution
st.subheader("Risk Level Distribution")
st.bar_chart(df_test['Risk_Level'].value_counts())

# -----------------------------
# Step 10: Probability Distribution
# -----------------------------
st.header("Step 10: Churn Probability Distribution")

fig2, ax2 = plt.subplots(figsize=(8,4))
sns.histplot(df_test['Churn_Prob'], bins=20, kde=True, ax=ax2)
ax2.set_title("Churn Probability Distribution")
st.pyplot(fig2)

# -----------------------------
# Step 11: Feature Importance
# -----------------------------
st.header("Step 11: Feature Importance")

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

fig3, ax3 = plt.subplots()
top_features.plot(kind='barh', ax=ax3)
ax3.set_title("Top 10 Important Features")
ax3.invert_yaxis()
st.pyplot(fig3)

st.success("🎯 Dashboard Completed Successfully!")
