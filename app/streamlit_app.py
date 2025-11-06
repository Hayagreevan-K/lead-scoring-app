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
        # sklearn >=1.6 uses sparse_output
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(("cat", ohe, categorical_cols))
    preproc = ColumnTransformer(transformers, remainder="drop")
    pipe = Pipeline([("preproc", preproc), ("clf", classifier)])
    return pipe

def detect_target_candidates(df):
    candidates = [c for c in df.columns if 'convert' in c.lower() or 'converted' in c.lower() or 'target' in c.lower()]
    if not candidates:
        for c in df.columns:
            try:
                u = df[c].dropna().unique()
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
        names = preproc.get_feature_names_out(input_df.columns)
        return list(names)
    except Exception:
        return None

def get_pipeline_classifier_name(pipeline):
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
        probs_test = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs_test)
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
# Scoring prep
# ---------------------
st.subheader("Score leads using pipeline")
raw_for_scoring = uploaded_df.copy()
if target_col and target_col in raw_for_scoring.columns:
    raw_for_scoring = raw_for_scoring.drop(columns=[target_col])

# ---------- Robust column-name normalization & alignment ----------
# ensure DataFrame column names are strings
raw_for_scoring.columns = raw_for_scoring.columns.map(lambda c: str(c))

# try to extract expected columns from pipeline preprocessor and normalize them too
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
                try:
                    expected_cols.append(str(cols))
                except Exception:
                    pass
        # dedupe while preserving order
        seen = set()
        expected_cols = [x for x in expected_cols if not (x in seen or seen.add(x))]
except Exception:
    expected_cols = None

# add missing expected columns filled with zeros, reorder
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

# ---------------- Diagnostic block: run BEFORE predict_proba ----------------
st.subheader("🔎 Deep diagnostics (developer)")

# 1) Classifier sanity
clf_obj, clf_name = get_pipeline_classifier_name(pipeline)
if clf_obj is None:
    st.error("Could not detect estimator inside pipeline.")
else:
    st.write("Estimator type:", clf_name)
    try:
        classes = getattr(clf_obj, "classes_", None)
        if classes is not None:
            st.write("Classifier classes_ (sample):", classes[:20], "  total classes:", len(classes))
        else:
            st.write("Classifier has no classes_ attribute (maybe not fitted).")
    except Exception as e:
        st.write("Error reading classes_:", e)

# 2) Preprocessor transform shape & variance
try:
    preproc = pipeline.named_steps.get('preproc', pipeline.named_steps.get('preprocessor', None))
    if preproc is None:
        st.info("Pipeline has no preprocessor step exposed as 'preproc'/'preprocessor'. Skipping preproc checks.")
    else:
        sample_df = raw_for_scoring.head(min(200, len(raw_for_scoring))).copy()
        with st.spinner("Running preprocessor.transform for diagnostics..."):
            X_trans = preproc.transform(sample_df)
        Xt = np.asarray(X_trans)
        st.write("Transformed feature matrix shape (sample):", getattr(X_trans, "shape", "unknown"))
        col_var = np.var(Xt, axis=0)
        uniq_counts = [np.unique(Xt[:,i]).size for i in range(Xt.shape[1])]
        n_const = int((col_var <= 1e-8).sum())
        st.write(f"Transformed features: {Xt.shape[1]} cols, {n_const} constant (near-zero var) columns.")
        low_var_idx = np.argsort(col_var)[:20]
        low_var_sample = [{"col_index": int(i), "variance": float(col_var[i]), "unique_vals": int(uniq_counts[i])} for i in low_var_idx[:20]]
        st.table(low_var_sample)
        st.write("First 10 transformed feature values (sample rows):")
        display_df = pd.DataFrame(Xt[:, :10])
        st.dataframe(display_df.head(8))
except Exception as e:
    st.warning(f"Preprocessor diagnostic failed: {e}")

# 3) Count expected cols missing originally
try:
    if expected_cols:
        missing_original = [c for c in expected_cols if c not in uploaded_df.columns.map(str)]
        st.write("Number of expected cols missing from uploaded file (filled with zeros):", len(missing_original))
        if len(missing_original) > 0:
            st.write("Sample missing expected columns:", missing_original[:200])
except Exception:
    pass

# 4) Try prediction (safe) and show stats
try:
    probs_all = pipeline.predict_proba(raw_for_scoring)[:,1]
    st.success("predict_proba ran successfully.")
    st.write({
        "min": float(probs_all.min()),
        "max": float(probs_all.max()),
        "mean": float(probs_all.mean()),
        "unique_count": int(len(np.unique(probs_all))),
        "n_rows": int(len(probs_all))
    })
except Exception as e:
    st.error(f"predict_proba failed: {e}")
    st.stop()

# 5) If classifier appears unfitted or single-class, offer immediate retrain (diagnostic)
try:
    classes = getattr(clf_obj, "classes_", None)
    if classes is None or len(classes) < 2:
        st.warning("Classifier does not appear to be fitted or has fewer than 2 classes. You should retrain.")
        if target_col and target_col in uploaded_df.columns:
            if st.button("Retrain pipeline now (diagnostic)"):
                st.info("Retraining pipeline on uploaded labeled data (diagnostic only).")
                raw_X = uploaded_df.drop(columns=[target_col]).copy()
                y = uploaded_df[target_col].astype(int).copy()
                num_cols = raw_X.select_dtypes(include=['number']).columns.tolist()
                cat_cols = raw_X.select_dtypes(include=['object','category','bool']).columns.tolist()
                test_pipe = build_pipeline(numeric_cols=num_cols, categorical_cols=cat_cols,
                                           classifier=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
                X_train, X_test, y_train, y_test = train_test_split(raw_X, y, stratify=y, test_size=0.2, random_state=42)
                with st.spinner("Training..."):
                    test_pipe.fit(X_train, y_train)
                test_probs = test_pipe.predict_proba(raw_for_scoring)[:,1]
                st.write("Retrain test predict stats:", {
                    "min": float(test_probs.min()),
                    "max": float(test_probs.max()),
                    "mean": float(test_probs.mean()),
                    "unique_count": int(len(np.unique(test_probs)))
                })
                st.success("Retraining diagnostic complete.")
except Exception:
    pass

# -------------------------
# DIAGNOSTIC + PLOTTING + DOWNLOAD
# -------------------------
out = uploaded_df.copy()
out['lead_score'] = probs_all
out['lead_rank'] = out['lead_score'].rank(ascending=False, method='first')
out_sorted = out.sort_values('lead_score', ascending=False)

# diagnostics summary
st.subheader("Prediction diagnostics summary")
min_s, max_s = float(out_sorted['lead_score'].min()), float(out_sorted['lead_score'].max())
mean_s = float(out_sorted['lead_score'].mean())
unique_count = int(out_sorted['lead_score'].nunique())
st.write({
    "min_score": min_s,
    "max_score": max_s,
    "mean_score": mean_s,
    "unique_scores": unique_count,
    "total_leads": len(out_sorted)
})

st.markdown("**Top 10 scored leads (sanity check)**")
st.dataframe(out_sorted.head(10).reset_index(drop=True))

st.markdown("**Columns with zero or near-zero variance (in scoring input)**")
var_report = raw_for_scoring.var(numeric_only=True).sort_values()
zero_var = var_report[var_report <= 1e-8]
if len(zero_var) > 0:
    st.write(f"Found {len(zero_var)} numeric columns with near-zero variance:")
    st.table(zero_var)
else:
    st.write("No numeric columns with near-zero variance detected.")

st.markdown("**Columns added during alignment (missing in upload → filled with zeros)**")
if expected_cols:
    missing_original = [c for c in expected_cols if c not in uploaded_df.columns.map(str)]
    if missing_original:
        st.write(f"{len(missing_original)} expected columns were missing and filled with zeros.")
        st.write(missing_original[:200])
    else:
        st.write("No expected columns were missing from the upload.")
else:
    st.write("No pipeline expected-columns metadata available to compare.")

if unique_count == 1:
    st.warning("All predicted scores are identical — debugging actions above will help identify why.")
    st.info("If you uploaded labeled data, use the 'Retrain pipeline now (diagnostic)' button to check variability.")

# --- Correct plotting: histogram with KDE, mean & threshold lines
st.subheader("Score distribution (corrected plot)")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(out_sorted['lead_score'], bins=30, kde=True, stat="count", ax=ax, edgecolor='black')
ax.axvline(mean_s, color='red', linestyle='--', linewidth=1, label=f"Mean: {mean_s:.3f}")
try:
    threshold
except NameError:
    threshold = None
if threshold is not None:
    ax.axvline(threshold, color='green', linestyle='--', linewidth=1, label=f"Threshold: {threshold:.2f}")
ax.set_xlabel("Predicted probability (lead score)")
ax.set_ylabel("Count")
ax.set_title("Lead Score Probability Distribution")
ax.legend()
st.pyplot(fig)

# Top-N and downloads
st.subheader("Top scored leads")
top_n = st.number_input("Show top N leads", min_value=5, max_value=1000, value=50, step=5)
st.dataframe(out_sorted.head(top_n))

st.subheader("Download scored leads")
csv = out_sorted.to_csv(index=False)
st.download_button("Download all scored leads (CSV)", csv, "scored_leads_full.csv", "text/csv")
st.download_button("Download top leads (CSV)", out_sorted.head(top_n).to_csv(index=False), f"scored_leads_top{top_n}.csv", "text/csv")

st.markdown("---")
st.caption("Note: For production, ensure the same schema is used for training & scoring and implement monitoring for data drift.")
