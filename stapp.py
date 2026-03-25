import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------
# Load your trained XGBoost model
# ---------------------------
model = joblib.load('xgb_model.pkl')  # Save your trained model earlier with joblib
feature_names = joblib.load('features.pkl')  # List of feature names used for training

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

# Title
st.title("📊 Student Dropout Prediction Dashboard")
st.markdown("""
This tool predicts **student dropout risk levels** using academic, demographic, and socio-economic data.  
Upload your dataset or manually enter student details to get actionable insights.
""")

# Sidebar input method
st.sidebar.header("Data Input Options")
data_option = st.sidebar.radio("Choose input method:", ("Upload CSV", "Manual Entry"))

# Risk level function
def risk_level(prob):
    if prob >= 0.70:
        return "🔴 High Risk"
    elif 0.40 <= prob < 0.70:
        return "🟨 Medium Risk"
    else:
        return "🟩 Low Risk"

# ---------------------------
# Load data
# ---------------------------
if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Student Data")
        st.dataframe(data.head())
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    st.sidebar.markdown("Enter student details:")
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0)
    data = pd.DataFrame([inputs])

# ---------------------------
# Prediction
# ---------------------------
probs = model.predict_proba(data)[:, 1]  # Probability of dropout
labels = [risk_level(p) for p in probs]
data['Dropout Probability'] = probs
data['Risk Level'] = labels

# ---------------------------
# Show results
# ---------------------------
st.write("### 🧑‍🎓 Student Risk Assessment")
st.dataframe(data[['Dropout Probability', 'Risk Level'] + feature_names])

# Summary chart
st.write("### 📈 Class Summary")
summary = data['Risk Level'].value_counts().reindex(
    ["🔴 High Risk", "🟨 Medium Risk", "🟩 Low Risk"]
).fillna(0)
st.bar_chart(summary)

# Suggested actions
st.write("### 📝 Suggested Actions")
for i, row in data.iterrows():
    if row['Risk Level'] == "🔴 High Risk":
        action = "📞 Call guardian immediately"
    elif row['Risk Level'] == "🟨 Medium Risk":
        action = "👩‍🏫 Assign mentor and monitor progress"
    else:
        action = "✅ Continue regular follow-up"
    st.markdown(f"**Student {i+1}**: {action}")

# ---------------------------
# Optional: Feature importance with SHAP
# ---------------------------
if st.checkbox("Show Feature Importance"):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    st.write("### 🔍 Feature Importance")
    shap.summary_plot(shap_values, data, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
# =======================
# Graphs Section
# =======================

st.header("📊 Visual Insights")


# 2. Feature Importance
st.subheader("Feature Importance (Why model predicts dropouts)")
importances = model.feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(10)  # Top 10 features
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax, palette="viridis")
st.pyplot(fig)

# 3. Absences vs Dropout Probability
if "absences" in data.columns:
    st.subheader("Absences vs Dropout Probability")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=data["absences"], y=data["Dropout Probability"], hue=data["Risk Level"], ax=ax2, palette={"🔴 High Risk":"red","🟨 Medium Risk":"yellow","🟩 Low Risk":"green"})
    ax2.set_xlabel("Number of Absences")
    ax2.set_ylabel("Predicted Dropout Probability")
    st.pyplot(fig2)

