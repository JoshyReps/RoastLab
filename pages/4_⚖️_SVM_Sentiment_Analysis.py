import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import apply_custom_theme, render_sidebar_info, load_model, load_all_metrics

st.set_page_config(
    page_title="4. SVM Sentiment | RoastLab",
    page_icon="⚖️",
    layout="wide"
)

apply_custom_theme()
render_sidebar_info()

st.markdown("<h2 class='brand-title'>4. Support Vector Machine: Review Sentiment Classifier</h2>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Linear Kernel Boundary with TF-IDF Feature Extraction</p>", unsafe_allow_html=True)

model = load_model("svm_sentiment")
metrics = load_all_metrics().get("svm", {})

col_s1, col_s2 = st.columns([1, 1])
with col_s1:
    st.markdown("### ☕ Business Context & Relevance")
    st.write("""
    Customer retention in specialty cafes depends heavily on immediate response to negative experiences. 
    RoastLab parses customer reviews from QR-code feedback forms and Google Maps in real time.
    
    **Why Support Vector Machines (SVM)?**
    - Text classification with TF-IDF yields very high-dimensional sparse feature spaces.
    - Linear SVM finds the **maximum margin hyperplane** $\\mathbf{w}^T \\mathbf{x} + b = 0$ that cleanly separates positive from negative reviews:
      $$\\min_{\\mathbf{w}, b} \\frac{1}{2} \\|\\mathbf{w}\\|^2 \\quad \\text{subject to} \\quad y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1$$
    - Negative reviews are flagged immediately and sent to store managers for immediate service recovery.
    """)
    st.markdown(f"""
    - **Kernel Type:** `Linear Kernel (Maximum Margin Hyperplane)`
    - **Feature Extraction:** `TF-IDF (1-2 N-grams, English stop words removed)`
    - **Support Vectors Count:** `{metrics.get('support_vectors', 24)}`
    - **Model Accuracy:** `{metrics.get('accuracy', 1.0):.1%}`
    """)
    
with col_s2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Live Review Sentiment Tester")
    sample_reviews = [
        "The cold brew was incredibly smooth, floral, and gave me the cleanest energy boost!",
        "Waited over 25 minutes for a simple black americano, and the counter was completely dirty.",
        "Best artisanal latte in town! Barista made stunning latte art and the froth was silky.",
        "The iced latte tasted watered down like ice cubes with milk. Extremely disappointed."
    ]
    sample_placeholder = "-- Pick a sample (optional) --"

    def load_sample():
        """Copies a chosen sample into the feedback box, then resets the dropdown so it can be re-picked."""
        choice = st.session_state["svm_sample"]
        if choice != sample_placeholder:
            st.session_state["svm_review_text"] = choice
        st.session_state["svm_sample"] = sample_placeholder

    # Primary input: type your own feedback
    with st.form("svm_feedback_form"):
        review_text = st.text_area(
            "✍️ Type your own customer feedback",
            key="svm_review_text",
            height=130,
            placeholder="e.g. Delicious pour-over and fast barista service!"
        )
        st.form_submit_button("🔍 Analyze Feedback")

    # Secondary: quick-fill with a curated sample
    st.selectbox(
        "...or try a sample review:",
        [sample_placeholder] + sample_reviews,
        key="svm_sample",
        on_change=load_sample
    )

    if not model:
        st.warning("Model file not found. Please train models first.")
    elif review_text.strip():
        if model.named_steps["tfidf"].transform([review_text]).nnz == 0:
            st.info(
                "🤔 The model doesn't recognize any of those words, so it can't judge this feedback. "
                "Try mentioning the drink, service, or atmosphere (e.g., *\"slow service\"*, *\"smooth latte\"*)."
            )
        else:
            pred_sent = model.predict([review_text])[0]
            decision = model.decision_function([review_text])[0]

            if pred_sent == "Positive":
                st.success(f"✅ **Sentiment:** POSITIVE (Margin Distance: `+{abs(decision):.2f}`)")
                st.write("🟢 **Store Action:** Logged to Barista Recognition Board & Google Review Highlights.")
            else:
                st.error(f"⚠️ **Sentiment:** NEGATIVE (Margin Distance: `-{abs(decision):.2f}`)")
                st.write("🔴 **Store Action:** High-priority ticket opened. Branch Manager alerted for customer outreach.")
    else:
        st.caption("Type feedback above and press **Analyze Feedback**, or pick a sample.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 💡 Why Support Vector Machines over Simple Keyword Matching?")
st.write("""
Keyword counting fails when customers use nuanced sentence structures (e.g., *"not bad at all"* vs *"not good"*). 
SVM maps combinations of uni-grams and bi-grams into a continuous feature space and evaluates the distance from the decision boundary, 
allowing RoastLab to gauge not just positive/negative sentiment, but **severity and confidence**.
""")