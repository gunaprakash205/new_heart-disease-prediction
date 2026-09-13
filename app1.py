
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="❤️",
    layout="wide"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f7f8fa;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    .risk-low {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #e9f7ef;
        border: 1px solid #b7e4c7;
    }

    .risk-high {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #fdecec;
        border: 1px solid #f5b5b5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Risk Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning prediction with model-based explainability'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "⚠️ This application is an educational machine-learning "
    "project. It is not a medical diagnosis or a substitute "
    "for professional medical advice."
)


# ==========================================================
# PATIENT INPUT
# ==========================================================

st.header("🧑‍⚕️ Patient Information")

st.write(
    "Enter the patient's health information below to generate "
    "a model prediction."
)


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50.0,
        max_value=250.0,
        value=120.0
    )


with col2:

    chol = st.number_input(
        "Cholesterol",
        min_value=50.0,
        max_value=600.0,
        value=200.0
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [False, True]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "normal",
            "lv hypertrophy",
            "st-t abnormality"
        ]
    )

    thalch = st.number_input(
        "Maximum Heart Rate",
        min_value=50.0,
        max_value=250.0,
        value=150.0
    )


with col3:

    exang = st.selectbox(
        "Exercise Induced Angina",
        [False, True]
    )

    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [
            "flat",
            "upsloping",
            "downsloping"
        ]
    )

    ca = st.number_input(
        "Number of Major Vessels",
        min_value=0,
        max_value=4,
        value=0
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "normal",
            "fixed defect",
            "reversable defect"
        ]
    )


st.divider()


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

predict = st.button(
    "🚀 Analyze Heart Disease Risk",
    use_container_width=True
)


if predict:

    # ======================================================
    # CREATE INPUT DATAFRAME
    # ======================================================

    input_data = pd.DataFrame({

        "age": [age],
        "trestbps": [trestbps],
        "chol": [chol],
        "thalch": [thalch],
        "oldpeak": [oldpeak],
        "ca": [ca],

        "sex_Male": [
            1 if sex == "Male" else 0
        ],

        "cp_atypical angina": [
            1 if cp == "atypical angina" else 0
        ],

        "cp_non-anginal": [
            1 if cp == "non-anginal" else 0
        ],

        "cp_typical angina": [
            1 if cp == "typical angina" else 0
        ],

        "fbs_True": [
            1 if fbs else 0
        ],

        "restecg_normal": [
            1 if restecg == "normal" else 0
        ],

        "restecg_st-t abnormality": [
            1 if restecg == "st-t abnormality" else 0
        ],

        "exang_True": [
            1 if exang else 0
        ],

        "slope_flat": [
            1 if slope == "flat" else 0
        ],

        "slope_upsloping": [
            1 if slope == "upsloping" else 0
        ],

        "thal_normal": [
            1 if thal == "normal" else 0
        ],

        "thal_reversable defect": [
            1 if thal == "reversable defect" else 0
        ]
    })


    # ======================================================
    # FEATURE ALIGNMENT
    # ======================================================

    if hasattr(scaler, "feature_names_in_"):

        input_data = input_data.reindex(
            columns=scaler.feature_names_in_,
            fill_value=0
        )


    # ======================================================
    # SCALE DATA
    # ======================================================

    scaled_input = scaler.transform(input_data)


    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    prediction = model.predict(scaled_input)

    probability = model.predict_proba(scaled_input)

    disease_probability = probability[0][1] * 100

    no_disease_probability = probability[0][0] * 100

    confidence = np.max(probability) * 100


    # ======================================================
    # RESULT
    # ======================================================

    st.header("📌 Prediction")

    if prediction[0] == 1:

        st.markdown(
            """
            <div class="risk-high">
                <h2>⚠️ Model Prediction: Higher Risk</h2>
                <p>The model classified this input as Class 1.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="risk-low">
                <h2>✅ Model Prediction: Lower Risk</h2>
                <p>The model classified this input as Class 0.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # ======================================================
    # PROBABILITY METRICS
    # ======================================================

    st.subheader("📈 Model Probability")

    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "Heart Disease Probability",
            f"{disease_probability:.2f}%"
        )


    with m2:

        st.metric(
            "No Disease Probability",
            f"{no_disease_probability:.2f}%"
        )


    with m3:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


    st.progress(
        int(disease_probability)
    )


    st.caption(
        "These probabilities are model outputs and are not "
        "clinically validated risk scores."
    )


    # ======================================================
    # PATIENT SUMMARY
    # ======================================================

    st.subheader("🧾 Patient Input Summary")

    summary_data = pd.DataFrame({

        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain",
            "Resting BP",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Angina",
            "Oldpeak",
            "Slope",
            "Major Vessels",
            "Thalassemia"
        ],

        "Value": [
            age,
            sex,
            cp,
            trestbps,
            chol,
            str(fbs),
            restecg,
            thalch,
            str(exang),
            oldpeak,
            slope,
            ca,
            thal
        ]
    })


    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # SHAP EXPLAINABILITY
    # ======================================================

    st.divider()

    st.header("🧠 Explainable AI")

    st.write(
        "SHAP explains how individual features influenced "
        "the Random Forest prediction."
    )


    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        scaled_input
    )


    # Handle different SHAP versions

    if isinstance(shap_values, list):

        shap_value = shap_values[1][0]

    else:

        if len(shap_values.shape) == 3:

            shap_value = shap_values[0, :, 1]

        else:

            shap_value = shap_values[0]


    # ======================================================
    # SHAP DATAFRAME
    # ======================================================

    shap_df = pd.DataFrame({

        "Feature": input_data.columns,

        "SHAP Value": shap_value
    })


    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )


    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # ======================================================
    # TOP SHAP FEATURES
    # ======================================================

    st.subheader("🔎 Top Features Influencing This Prediction")


    top_features = shap_df.head(8).copy()

    top_features = top_features.sort_values(
        "SHAP Value"
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    ax.barh(
        top_features["Feature"],
        top_features["SHAP Value"]
    )


    ax.axvline(
        0,
        linewidth=1
    )


    ax.set_xlabel(
        "SHAP Value"
    )


    ax.set_ylabel(
        "Feature"
    )


    ax.set_title(
        "Feature Contribution to This Prediction"
    )


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )


    # ======================================================
    # SHAP INTERPRETATION
    # ======================================================

    st.info(
        """
        How to read this chart:

        • Positive SHAP values push the prediction toward
          Class 1 (heart disease).

        • Negative SHAP values push the prediction toward
          Class 0 (no heart disease).

        • Larger absolute SHAP values indicate stronger
          influence on this individual prediction.

        SHAP explains the model's behavior; it does not
        establish medical causation.
        """
    )


    # ======================================================
    # TOP CONTRIBUTIONS
    # ======================================================

    st.subheader("📌 Key Model Contributions")


    for _, row in shap_df.head(6).iterrows():

        feature = row["Feature"]

        value = row["SHAP Value"]


        if value > 0:

            st.write(
                f"🔴 **{feature}** → pushed the prediction "
                f"toward heart disease "
                f"({value:+.4f})"
            )

        else:

            st.write(
                f"🔵 **{feature}** → pushed the prediction "
                f"toward no heart disease "
                f"({value:+.4f})"
            )


    # ======================================================
    # MODEL PERFORMANCE
    # ======================================================

    st.divider()

    st.header("🏆 Model Performance")


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "Accuracy",
            "86.96%"
        )


    with p2:

        st.metric(
            "Precision",
            "84.82%"
        )


    with p3:

        st.metric(
            "Recall",
            "93.14%"
        )


    with p4:

        st.metric(
            "F1 Score",
            "88.79%"
        )


    st.caption(
        "Performance measured on the held-out test set."
    )


    # ======================================================
    # MODEL COMPARISON
    # ======================================================

    st.subheader("📊 Algorithm Comparison")


    model_results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "SVM",
            "KNN"
        ],

        "Accuracy": [
            0.842391,
            0.793478,
            0.869565,
            0.826087,
            0.853261,
            0.853261
        ],

        "Precision": [
            0.841121,
            0.801887,
            0.848214,
            0.830189,
            0.831858,
            0.850467
        ],

        "Recall": [
            0.882353,
            0.833333,
            0.931373,
            0.862745,
            0.921569,
            0.892157
        ],

        "F1 Score": [
            0.861244,
            0.817308,
            0.887850,
            0.846154,
            0.874419,
            0.870813
        ]
    })


    display_results = model_results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2).astype(str) + "%"


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "🏆 Random Forest achieved the strongest overall "
        "held-out test performance among the evaluated models."
    )


    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    st.subheader("🔄 Cross-Validation")


    st.write(
        "The original Random Forest was evaluated using "
        "5-fold cross-validation."
    )


    cv_scores = [
        0.81756757,
        0.83673469,
        0.80272109,
        0.78231293,
        0.82993197
    ]


    cv_df = pd.DataFrame({

        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],

        "Accuracy": [
            f"{x * 100:.2f}%"
            for x in cv_scores
        ]
    })


    c1, c2 = st.columns([2, 1])


    with c1:

        st.dataframe(
            cv_df,
            use_container_width=True,
            hide_index=True
        )


    with c2:

        st.metric(
            "Average CV Accuracy",
            "81.39%"
        )


    st.caption(
        "Cross-validation provides a more robust estimate "
        "of model performance across different data splits."
    )


    # ======================================================
    # FINAL NOTE
    # ======================================================

    st.divider()

    st.subheader("ℹ️ About This Project")

    st.write(
        """
        This project demonstrates an end-to-end machine-learning
        workflow for heart disease classification:

        • Data preprocessing and missing-value handling
        • Categorical feature encoding
        • Multiple machine-learning algorithms
        • Model evaluation
        • Cross-validation
        • Hyperparameter tuning
        • Random Forest model selection
        • SHAP-based explainability
        • Streamlit deployment
        """
    )

    st.caption(
        "Educational machine-learning application. "
        "Not intended for clinical diagnosis."
    )