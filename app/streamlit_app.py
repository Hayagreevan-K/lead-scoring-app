# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Lead Scoring Demo", layout="wide")

# -----------------------
# Config: compressed model filenames (from Kaggle training with joblib.compress)
# -----------------------
MODEL_PATH = "models/lead_model_compressed.joblib"
SCALER_PATH = "models/scaler_compressed.joblib"
os.makedirs("models", exist_ok=True)

# -----------------------
# Utility / Preprocess functions
# -----------------------
def basic_clean(df):
    df = df.copy()
    # drop obvious id columns
    drop_candidates = [c for c in df.columns if c.lower().strip() in ('lead_id','id','leadid','slno','sr_no','serial')]
    drop_candidates += [c for c in df.columns if 'url' in c.lower() or 'link' in c.lower()]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors='ignore')
    return df

def simple_impute(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('unknown')
    return df

def map_bool_like(df):
    df = df.copy()
    bool_map = {'yes':1,'no':0,'y':1,'n':0,'true':1,'false':0,'1':1,'0':0}
    for c in df.select_dtypes(include='object').columns:
        s = df[c].dropna().astype(str).str.lower().unique()[:20]
        if set(s).issubset(set(bool_map.keys())):
            df[c] = df[c].map(lambda x: bool_map.get(str(x).lower(), np.nan)).fillna(0).astype(int)
    return df

def encode_features(df, drop_first=True, low_card_thresh=15):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    low_card = [c for c in cat_cols if df[c].nunique() <= low_card_thresh]
    high_card = [c for c in cat_cols if c not in low_card]
    # one-hot low-card
    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=drop_first)
    # frequency encode high-card
    for c in high_card:
        freq = df[c].value_counts(normalize=True)
        df[c + '_freq'] = df[c].map(freq).fillna(0)
        df.drop(columns=[c], inplace=True)
    return df

def prepare_features(raw_df, target_col=None):
    df = basic_clean(raw_df)
    df = simple_impute(df)
    df = map_bool_like(df)
    df_feat = encode_features(df)
    # build feature list
    if target_col and target_col in df_feat.columns:
        feature_cols = [c for c in df_feat.columns if c != target_col]
    else:
        feature_cols = df_feat.columns.tolist()
    return df_feat, feature_cols

# -----------------------
# Model helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    model, scaler = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            st.warning("Could not load saved model/scaler: " + str(e))
    return model, scaler

def train_quick_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    return model, scaler, auc

# -----------------------
# Streamlit UI
# -----------------------
st.title("Sales Lead Scoring — Demo")
st.markdown("Upload a leads CSV (with a target column for training) or use the sample dataset. The app will try to load compressed model artifacts from `models/`.")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    sample_btn = st.button("Use sample demo dataset")
with col2:
    st.write("Model artifacts:")
    model, scaler = load_artifacts()
    if model is not None:
        st.success(f"Pretrained compressed model found: {MODEL_PATH}")
        st.write("Model type:", type(model).__name__)
    else:
        st.info("No pretrained compressed model found. You can upload labeled data to train a quick model.")

# Create demo dataset if requested
if sample_btn:
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
else:
    uploaded_df = None
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Error reading CSV: " + str(e))

if uploaded_df is None:
    st.info("Upload a CSV or click 'Use sample demo dataset' to proceed.")
    st.stop()

st.subheader("Preview data")
st.dataframe(uploaded_df.head(10))

# Detect possible target columns
possible_targets = [c for c in uploaded_df.columns if 'convert' in c.lower() or 'converted' in c.lower() or 'target' in c.lower()]
target_col = None
if possible_targets:
    target_col = st.selectbox("Detected target candidates (choose if you want to train)", options=possible_targets, index=0)
else:
    target_col = st.text_input("Enter target column name (if training). Leave empty to just score with pretrained model:", value="")

if target_col == "":
    target_col = None

# Prepare features
df_feat, feature_cols = prepare_features(uploaded_df, target_col=target_col)
st.write(f"Prepared features count: {len(feature_cols)}")

# Option to train quick model if labeled data present
if target_col and target_col in df_feat.columns:
    st.subheader("Train model from uploaded data (optional)")
    if st.button("Train model now"):
        X = df_feat[[c for c in feature_cols if c != target_col]].copy() if target_col in feature_cols else df_feat[feature_cols].copy()
        y = df_feat[target_col].astype(int)
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        model_tr, scaler_tr, auc = train_quick_model(X, y)
        st.success(f"Trained RandomForest. ROC-AUC on holdout: {auc:.3f}")
        model = model_tr
        scaler = scaler_tr
        # Save compressed artifacts (small)
        try:
            joblib.dump(model, MODEL_PATH, compress=3)
            joblib.dump(scaler, SCALER_PATH, compress=3)
            st.write(f"Saved compressed model/scaler to {MODEL_PATH} and {SCALER_PATH}")
        except Exception as e:
            st.warning("Could not save artifacts: " + str(e))
else:
    st.info("No labeled target provided for training. Will use pretrained compressed model if available to score uploaded leads.")

# Ensure model exists
if model is None:
    st.warning("No model available. Provide labeled data and click 'Train model now' OR place compressed model files in models/ and reload.")
    st.stop()

# Build X_all for scoring
X_all = df_feat[feature_cols].copy()
# If target present in features, drop it before scoring
if target_col and target_col in X_all.columns:
    X_all = X_all.drop(columns=[target_col])

numeric_cols = X_all.select_dtypes(include='number').columns.tolist()
if scaler is not None and numeric_cols:
    try:
        X_all[numeric_cols] = scaler.transform(X_all[numeric_cols])
    except Exception as e:
        st.warning("Scaler transform failed; attempting fit_transform.")
        scaler = StandardScaler()
        X_all[numeric_cols] = scaler.fit_transform(X_all[numeric_cols])

# Predict probabilities
try:
    probs = model.predict_proba(X_all)[:,1]
except Exception as e:
    st.error("Model prediction failed: " + str(e))
    st.stop()

out = uploaded_df.copy()
out['lead_score'] = probs
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

# Thresholding & flags
st.subheader("Threshold & Flags")
threshold = st.slider("Select lead score threshold to mark 'High Priority' leads", 0.0, 1.0, 0.6, 0.01)
out_sorted['high_priority'] = (out_sorted['lead_score'] >= threshold).astype(int)
hp_count = int(out_sorted['high_priority'].sum())
st.write(f"High priority leads (score >= {threshold}): {hp_count}")

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    st.subheader("Top feature importances")
    try:
        feat_imp = pd.Series(model.feature_importances_, index=X_all.columns).sort_values(ascending=False).head(20)
        fig2, ax2 = plt.subplots(figsize=(6,6))
        sns.barplot(x=feat_imp.values[::-1], y=feat_imp.index[::-1], ax=ax2)
        st.pyplot(fig2)
    except Exception as e:
        st.warning("Could not plot feature importances: " + str(e))
elif hasattr(model, 'coef_'):
    st.subheader("Top coefficients (linear model)")
    coefs = pd.Series(model.coef_[0], index=X_all.columns).sort_values(key=abs, ascending=False).head(20)
    st.table(coefs)

# Download options
st.subheader("Download scored leads")
csv = out_sorted.to_csv(index=False)
st.download_button("Download all scored leads (CSV)", csv, "scored_leads_full.csv", "text/csv")
st.download_button("Download top leads (CSV)", out_sorted.head(top_n).to_csv(index=False), f"scored_leads_top{top_n}.csv", "text/csv")

st.markdown("---")
st.caption("Note: For production, ensure identical feature pipeline used for training & scoring and add model monitoring.")
