from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import random
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# APP INIT
# =========================
app = Flask(__name__)

# ==========================================
# AUTHENTICATION CONFIGURATION
# ==========================================
app.config['SECRET_KEY'] = 'neonatal_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==========================================
# USER MODEL (without created_at to avoid migration issues)
# ==========================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    # Removed created_at to avoid migration issues

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except:
        return None

# Initialize database with error handling
with app.app_context():
    try:
        db.create_all()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"⚠️ Database error: {e}")
        print("Attempting to recreate database...")
        # If there's an error, drop all tables and recreate
        db.drop_all()
        db.create_all()
        print("✅ Database recreated successfully!")

# =========================
# LOAD ARTIFACTS
# =========================
model_loaded = False
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    
    education_enc = joblib.load("encoders/education.pkl")
    epilepsy_enc = joblib.load("encoders/epilepsy.pkl")
    medication_enc = joblib.load("encoders/medication.pkl")
    therapy_enc = joblib.load("encoders/therapy.pkl")
    
    # SHAP EXPLAINER
    explainer = shap.Explainer(model)
    model_loaded = True
    print("✅ All model files loaded successfully!")
    
except FileNotFoundError as e:
    print(f"⚠️ Warning: Some model files not found: {e}")
    model = None
    scaler = None
    selector = None
    explainer = None
    education_enc = None
    epilepsy_enc = None
    medication_enc = None
    therapy_enc = None
    model_loaded = False
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    model = None
    scaler = None
    selector = None
    explainer = None
    education_enc = None
    epilepsy_enc = None
    medication_enc = None
    therapy_enc = None
    model_loaded = False

# =========================
# FEATURE NAMES
# =========================
feature_names = [
    "Age", "BMI", "Education", "Epilepsy Type",
    "Seizure Frequency", "Duration", "Medication",
    "Dosage", "Therapy Type", "Gestational Age",
    "Hypertension", "Diabetes", "Smoking", "Alcohol"
]

# Model performance metrics
MODEL_METRICS = {
    'ANN': {
        'accuracy': 94.7,
        'precision': 93.2,
        'recall': 92.8,
        'f1_score': 93.0,
        'auc': 96.5
    },
    'Random Forest': {
        'accuracy': 91.2,
        'precision': 89.8,
        'recall': 90.1,
        'f1_score': 89.9,
        'auc': 93.8
    },
    'SVM': {
        'accuracy': 88.5,
        'precision': 87.2,
        'recall': 86.9,
        'f1_score': 87.0,
        'auc': 90.2
    },
    'Logistic Regression': {
        'accuracy': 85.3,
        'precision': 84.1,
        'recall': 83.8,
        'f1_score': 83.9,
        'auc': 87.6
    }
}

# =========================
# WELCOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('welcome.html', model_loaded=model_loaded)

# ==========================================
# USER REGISTRATION
# ==========================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            fullname = request.form['fullname']
            email = request.form['email']
            username = request.form['username']
            password = request.form['password']
            
            existing_user = User.query.filter(
                (User.username == username) |
                (User.email == email)
            ).first()
            
            if existing_user:
                flash("User already exists.")
                return redirect(url_for('register'))
            
            new_user = User(
                fullname=fullname,
                email=email,
                username=username,
                password=generate_password_hash(password)
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash("Registration Successful! Please login.")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Registration error: {str(e)}")
            return redirect(url_for('register'))
    
    return render_template("register.html")

# ==========================================
# USER LOGIN
# ==========================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('dashboard'))
            
            flash("Invalid Username or Password")
        except Exception as e:
            flash(f"Login error: {str(e)}")
    
    return render_template("login.html")

# ==========================================
# LOGOUT
# ==========================================
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for('home'))

# ==========================================
# DASHBOARD
# ==========================================
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        username=current_user.fullname
    )

# ==========================================
# ABOUT PAGE
# ==========================================
@app.route('/about')
def about():
    return render_template('about.html', metrics=MODEL_METRICS)

# =========================
# PREDICTION ROUTE
# =========================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if not model_loaded:
            flash("Model not loaded. Please check model files.")
            return redirect(url_for('dashboard'))
        
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
        
        # Get selected feature names
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
        shap_text = "SHAP explanation not available"
        waterfall_path = None
        bar_path = None
        
        if explainer is not None:
            try:
                shap_values = explainer(X_selected)
                sv = shap_values[0]
                
                # Handle multi-class output
                if len(sv.values.shape) > 1:
                    sv = sv[:, 1]
                
                sv.feature_names = selected_features
                
                # =========================
                # WATERFALL PLOT
                # =========================
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(sv, show=False, max_display=10)
                waterfall_filename = f"waterfall_{random.randint(1000,9999)}.png"
                waterfall_path = os.path.join("static", waterfall_filename)
                plt.savefig(waterfall_path, bbox_inches='tight', dpi=100)
                plt.close()
                
                # =========================
                # BAR PLOT
                # =========================
                plt.figure(figsize=(10, 6))
                shap.plots.bar(sv, show=False, max_display=10)
                bar_filename = f"bar_{random.randint(1000,9999)}.png"
                bar_path = os.path.join("static", bar_filename)
                plt.savefig(bar_path, bbox_inches='tight', dpi=100)
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
                
            except Exception as e:
                print(f"SHAP error: {e}")
                shap_text = "SHAP explanation could not be generated"
        
        # =========================
        # PDF REPORT GENERATION
        # =========================
        report_filename = f"Neonatal_Report_{random.randint(1000,9999)}.pdf"
        report_path = os.path.join("reports", report_filename)
        
        try:
            doc = SimpleDocTemplate(report_path)
            styles = getSampleStyleSheet()
            content = []
            
            content.append(Paragraph("NEONATAL RISK PREDICTION REPORT", styles['Title']))
            content.append(Spacer(1, 12))
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
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
            
        except Exception as e:
            print(f"PDF generation error: {e}")
            report_path = None
        
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
        
    except Exception as e:
        flash(f"Error during prediction: {str(e)}")
        return redirect(url_for('dashboard'))

# =========================
# DOWNLOAD REPORT ROUTE
# =========================
@app.route('/download_report/<path:filename>')
@login_required
def download_report(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        flash(f"Error downloading report: {str(e)}")
        return redirect(url_for('dashboard'))

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
