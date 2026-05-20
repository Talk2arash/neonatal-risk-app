from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# APP INIT
# =========================
app = Flask(__name__)

os.makedirs("static", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

education_enc = joblib.load("encoders/education.pkl")
epilepsy_enc = joblib.load("encoders/epilepsy.pkl")
medication_enc = joblib.load("encoders/medication.pkl")
therapy_enc = joblib.load("encoders/therapy.pkl")

# SHAP EXPLAINER
explainer = shap.Explainer(model)

# =========================
# FEATURE NAMES
# =========================
feature_names = [
    "Age", "BMI", "Education", "Epilepsy Type",
    "Seizure Frequency", "Duration", "Medication",
    "Dosage", "Therapy Type", "Gestational Age",
    "Hypertension", "Diabetes", "Smoking", "Alcohol"
]

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template("dashboard.html")

# =========================
# PREDICTION ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    # =========================
    # INPUT COLLECTION
    # =========================
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    seizure = float(request.form['seizure'])
    duration = float(request.form['duration'])
    dosage = float(request.form['dosage'])
    gest_age = float(request.form['gest_age'])

    hypertension = int(request.form['hypertension'])
    diabetes = int(request.form['diabetes'])
    smoking = int(request.form['smoking'])
    alcohol = int(request.form['alcohol'])

    education = education_enc.transform([request.form['education']])[0]
    epilepsy = epilepsy_enc.transform([request.form['epilepsy']])[0]
    medication = medication_enc.transform([request.form['medication']])[0]
    therapy = therapy_enc.transform([request.form['therapy']])[0]

    # =========================
    # FEATURE VECTOR
    # =========================
    X = np.array([[age, bmi, education, epilepsy,
                   seizure, duration, medication,
                   dosage, therapy, gest_age,
                   hypertension, diabetes,
                   smoking, alcohol]])

    # =========================
    # PREPROCESSING
    # =========================
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    selected_features = np.array(feature_names)[selector.get_support()]

    # =========================
    # MODEL PREDICTION
    # =========================
    prob = model.predict_proba(X_selected)[0][1]
    pred = model.predict(X_selected)[0]

    # =========================
    # RISK CLASSIFICATION
    # =========================
    if prob < 0.3:
        risk_level = "LOW RISK"
        color = "green"
    elif prob < 0.7:
        risk_level = "MODERATE RISK"
        color = "orange"
    else:
        risk_level = "HIGH RISK"
        color = "red"

    # =========================
    # SHAP EXPLANATION
    # =========================
    shap_values = explainer(X_selected)
    sv = shap_values[0]

    if len(sv.values.shape) > 1:
        sv = sv[:, 1]

    sv.feature_names = selected_features

    # =========================
    # WATERFALL PLOT
    # =========================
    plt.figure()
    shap.plots.waterfall(sv, show=False)
    waterfall_path = "static/waterfall.png"
    plt.savefig(waterfall_path, bbox_inches='tight')
    plt.close()

    # =========================
    # BAR PLOT
    # =========================
    plt.figure()
    shap.plots.bar(sv, show=False)
    bar_path = "static/bar.png"
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()

    # =========================
    # SHAP TEXT EXPLANATION
    # =========================
    shap_vals = sv.values
    idx = np.argsort(np.abs(shap_vals))[::-1]

    explanation = []
    for i in idx[:5]:
        direction = "increased" if shap_vals[i] > 0 else "decreased"
        explanation.append(
            f"{selected_features[i]} {direction} risk ({shap_vals[i]:.3f})"
        )

    shap_text = " | ".join(explanation)

    # =========================
    # PDF REPORT GENERATION
    # =========================
    report_filename = f"Neonatal_Report_{np.random.randint(1000,9999)}.pdf"
    report_path = os.path.join("reports", report_filename)

    doc = SimpleDocTemplate(report_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("NEONATAL RISK PREDICTION REPORT", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles['Normal']))
    content.append(Paragraph(f"<b>Probability:</b> {round(prob*100,2)}%", styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("TOP SHAP EXPLANATION:", styles['Heading2']))
    content.append(Paragraph(shap_text, styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            "DISCLAIMER: This system is a clinical decision support tool and should not replace medical judgment.",
            styles['Italic']
        )
    )

    doc.build(content)

    # =========================
    # RETURN RESULT PAGE DATA
    # =========================
    return render_template(
        "result.html",
        prediction=int(pred),
        probability=round(prob * 100, 2),
        risk_level=risk_level,
        color=color,
        shap_text=shap_text,
        waterfall=waterfall_path,
        bar=bar_path,
        report_file=report_path
    )

# =========================
# DOWNLOAD REPORT ROUTE
# =========================
@app.route('/download_report/<path:filename>')
def download_report(filename):
    return send_file(filename, as_attachment=True)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)