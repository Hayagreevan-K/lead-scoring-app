# app/streamlit_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Lead Scoring (Pipeline)", layout="wide")
st.title("Sales Lead Scoring — Robust Pipeline Demo")

# Paths
PIPELINE_PATH = "models/lead_pipeline_compressed.joblib"
os.makedirs("models", exist_ok=True)

# ---------------------
# Helpers
# ---------------------
@st.cache_data(show_spinner=False)
def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        try:
            pipe = joblib.load(PIPELINE_PATH)
            return pipe
        except Exception as e:
            st.warning(f"Failed to load saved pipeline: {e}")
            return None
    return None

def build_pipeline(numeric_cols, categorical_cols, classifier=None):
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        # sklearn >=1.6 uses sparse_output instead of sparse
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(("cat", ohe, categorical_cols))
    preproc = ColumnTransformer(transformers, remainder="drop")
    pipe = Pipeline([("preproc", preproc), ("clf", classifier)])
    return pipe

def detect_target_candidates(df):
    candidates = [c for c in df.columns if 'convert' in c.lower() or 'converted' in c.lower() or 'target' in c.lower()]
    if not candidates:
        for c in df.columns:
            u = df[c].dropna().unique()
            try:
                if set(u).issubset({0,1}) or len(u) == 2:
                    candidates.append(c)
                    break
            except Exception:
                continue
    return candidates

def extract_feature_names(pipeline, input_df):
    try:
        preproc = pipeline.named_steps.get('preproc', pipeline.named_steps.get('preprocessor', None))
        if preproc is None:
            return None
        # get_feature_names_out expects string column names
        names = preproc.get_feature_names_out(input_df.columns)
        return list(names)
    except Exception:
        return None

def get_pipeline_classifier_name(pipeline):
    """
    Return (estimator_object, estimator_class_name) robustly from a sklearn Pipeline.
    """
    try:
        if hasattr(pipeline, "steps") and len(pipeline.steps) > 0:
            clf_obj = pipeline.steps[-1][1]
            return clf_obj, type(clf_obj).__name__
        for key in ("clf", "model", "estimator"):
            if key in getattr(pipeline, "named_steps", {}):
                clf_obj = pipeline.named_steps[key]
                return clf_obj, type(clf_obj).__name__
    except Exception:
        return None, None
    return None, None

# ---------------------
# UI: upload / sample
# ---------------------
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload leads CSV", type=["csv"])
    use_sample = st.button("Use small synthetic demo dataset")
with col2:
    st.write("Pipeline artifact:")
    pipeline = load_pipeline()
    if pipeline is not None:
        clf_obj, clf_name = get_pipeline_classifier_name(pipeline)
        if clf_name:
            st.success(f"Saved pipeline found — Model type: {clf_name}")
        else:
            st.success("Saved pipeline found.")
    else:
        st.info("No saved pipeline found. You can upload labeled data and train a pipeline in-app.")

# Prepare uploaded_df
if use_sample:
    st.info("Creating small synthetic demo dataset.")
    np.random.seed(42)
    n = 800
    demo = pd.DataFrame({
        'lead_source': np.random.choice(['ad', 'organic', 'referral', 'partner'], size=n),
        'visits': np.random.poisson(3, size=n),
        'time_on_site': np.abs(np.random.normal(120, 60, size=n)),
        'country': np.random.choice(['IN','US','UK','AU'], size=n),
        'previous_purchases': np.random.randint(0,5,size=n),
    })
    demo['Converted'] = ((demo['visits']>3) & (demo['previous_purchases']>0)).astype(int)
    uploaded_df = demo.copy()
elif uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV or use the sample dataset to proceed.")
    st.stop()

st.subheader("Preview uploaded data")
st.dataframe(uploaded_df.head(10))

# Detect target candidates and let user choose
candidates = detect_target_candidates(uploaded_df)
target_col = None
if candidates:
    target_col = st.selectbox("Detected target candidates (choose if you want to train)", options=candidates, index=0)
else:
    target_col = st.text_input("Enter target column name (if you want to train). Leave blank to just score with saved pipeline:", value="")
if target_col == "":
    target_col = None

# ---------------------
# If training: build pipeline and train
# ---------------------
if target_col and target_col in uploaded_df.columns:
    st.subheader("Train pipeline from uploaded data (optional)")
    if st.button("Train pipeline now"):
        raw_X = uploaded_df.drop(columns=[target_col]).copy()
        y = uploaded_df[target_col].astype(int).copy()

        numeric_cols = raw_X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = raw_X.select_dtypes(include=['object','category','bool']).columns.tolist()

        st.write(f"Detected numeric cols: {numeric_cols}")
        st.write(f"Detected categorical cols: {categorical_cols}")

        pipe = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols,
                              classifier=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))

        X_train, X_test, y_train, y_test = train_test_split(raw_X, y, stratify=y, test_size=0.2, random_state=42)
        with st.spinner("Training pipeline..."):
            pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        st.success(f"Trained pipeline. ROC-AUC on holdout: {auc:.3f}")

        try:
            joblib.dump(pipe, PIPELINE_PATH, compress=3)
            st.write(f"Saved compressed pipeline to {PIPELINE_PATH}")
            pipeline = pipe
        except Exception as e:
            st.warning(f"Failed to save pipeline: {e}")
            pipeline = pipe
else:
    st.info("No target selected for training. If a saved pipeline exists it will be used for scoring.")

# ---------------------
# Ensure pipeline available
# ---------------------
pipeline = pipeline if 'pipeline' in locals() and pipeline is not None else load_pipeline()
if pipeline is None:
    st.warning("No pipeline available. Provide labeled data and train, or upload a pipeline at models/lead_pipeline_compressed.joblib.")
    st.stop()

# ---------------------
# Scoring: use raw uploaded data (predict_proba handles preprocessing)
# ---------------------
st.subheader("Score leads using pipeline")
raw_for_scoring = uploaded_df.copy()
if target_col and target_col in raw_for_scoring.columns:
    raw_for_scoring = raw_for_scoring.drop(columns=[target_col])

# ---------- Robust column-name normalization & alignment ----------
# 1) ensure DataFrame column names are strings
raw_for_scoring.columns = raw_for_scoring.columns.map(lambda c: str(c))

# 2) try to extract expected columns from pipeline preprocessor and normalize them too
expected_cols = None
try:
    preproc = pipeline.named_steps.get('preproc', pipeline.named_steps.get('preprocessor', None))
    if preproc is not None and hasattr(preproc, 'transformers_'):
        expected_cols = []
        for name, transformer, cols in preproc.transformers_:
            if cols in ('drop', 'passthrough') or cols is None:
                continue
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected_cols.extend([str(x) for x in cols])
            else:
                # cols might be a single column name or slice, try to stringify
                try:
                    expected_cols.append(str(cols))
                except Exception:
                    pass
        # dedupe while preserving order
        seen = set()
        expected_cols = [x for x in expected_cols if not (x in seen or seen.add(x))]
except Exception:
    expected_cols = None

# 3) align input to expected columns if available
if expected_cols:
    for c in expected_cols:
        if c not in raw_for_scoring.columns:
            raw_for_scoring[c] = 0
    other_cols = [c for c in raw_for_scoring.columns if c not in expected_cols]
    reorder = expected_cols + other_cols
    seen = set()
    reorder = [c for c in reorder if not (c in seen or seen.add(c))]
    raw_for_scoring = raw_for_scoring.reindex(columns=reorder, fill_value=0)
else:
    # fallback: ensure unique string column names
    raw_for_scoring.columns = raw_for_scoring.columns.map(lambda c: str(c))
    if raw_for_scoring.columns.duplicated().any():
        cols = []
        counts = {}
        for c in raw_for_scoring.columns:
            counts[c] = counts.get(c, 0) + 1
            cols.append(c if counts[c] == 1 else f"{c}_{counts[c]}")
        raw_for_scoring.columns = cols

# Try prediction
try:
    probs_all = pipeline.predict_proba(raw_for_scoring)[:,1]
except Exception as e:
    st.error(
        "Model prediction failed after column normalization. "
        "This usually means the pipeline expects different columns than supplied."
    )
    # helpful debug info (safe)
    st.write("Input columns (sample):", list(raw_for_scoring.columns)[:100])
    if expected_cols:
        st.write("Expected columns (sample):", expected_cols[:100])
    st.stop()

out = uploaded_df.copy()
out['lead_score'] = probs_all
out['lead_rank'] = out['lead_score'].rank(ascending=False, method='first')
out_sorted = out.sort_values('lead_score', ascending=False)

# Display top N
st.subheader("Top scored leads")
top_n = st.number_input("Show top N leads", min_value=5, max_value=1000, value=50, step=5)
st.dataframe(out_sorted.head(top_n))

# Score distribution
st.subheader("Score distribution")
fig, ax = plt.subplots(figsize=(6,3))
sns.histplot(out_sorted['lead_score'], bins=40, kde=True, ax=ax)
ax.set_xlabel("Lead score")
st.pyplot(fig)

# Thresholding
st.subheader("Threshold & flags")
threshold = st.slider("Select threshold for 'High Priority' leads", 0.0, 1.0, 0.6, 0.01)
out_sorted['high_priority'] = (out_sorted['lead_score'] >= threshold).astype(int)
st.write(f"High priority leads (score >= {threshold}): {int(out_sorted['high_priority'].sum())}")

# Feature importance (if tree model)
clf_obj, clf_name = get_pipeline_classifier_name(pipeline)
if clf_obj is not None and hasattr(clf_obj, 'feature_importances_'):
    st.subheader("Top feature importances")
    feat_names = extract_feature_names(pipeline, raw_for_scoring)
    try:
        importances = clf_obj.feature_importances_
        if feat_names is not None and len(feat_names) == len(importances):
            feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
            fig2, ax2 = plt.subplots(figsize=(6,6))
            sns.barplot(x=feat_imp.values[::-1], y=feat_imp.index[::-1], ax=ax2)
            st.pyplot(fig2)
        else:
            feat_imp = pd.Series(importances).sort_values(ascending=False).head(30)
            st.write("Top importances (index-based):")
            st.table(feat_imp)
    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")
elif clf_obj is not None and hasattr(clf_obj, 'coef_'):
    st.subheader("Top coefficients (linear model)")
    try:
        feat_names = extract_feature_names(pipeline, raw_for_scoring)
        coefs = clf_obj.coef_[0]
        if feat_names is not None and len(feat_names) == len(coefs):
            coefs_s = pd.Series(coefs, index=feat_names).sort_values(key=abs, ascending=False).head(30)
            st.table(coefs_s)
        else:
            st.table(pd.Series(coefs).head(30))
    except Exception as e:
        st.warning(f"Could not show coefficients: {e}")

# Download options
st.subheader("Download scored leads")
csv = out_sorted.to_csv(index=False)
st.download_button("Download all scored leads (CSV)", csv, "scored_leads_full.csv", "text/csv")
st.download_button("Download top leads (CSV)", out_sorted.head(top_n).to_csv(index=False), f"scored_leads_top{top_n}.csv", "text/csv")

st.markdown("---")
st.caption("Note: For production, ensure the same schema is used for training & scoring and implement monitoring for data drift.")
