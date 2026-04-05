import os, logging, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from feature_engineering import build_training_features, FEATURE_COLS
from config import DATA_DIR, MODEL_DIR

logger = logging.getLogger(__name__)
os.makedirs(MODEL_DIR, exist_ok=True)

def _result_model():
    if HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=42, verbosity=0)
    return GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)

def _binary_model():
    if HAS_XGB:
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, verbosity=0)
    return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

def train(hist_df: pd.DataFrame = None):
    if hist_df is None or hist_df.empty:
        path = f"{DATA_DIR}/all_results.csv"
        if os.path.exists(path):
            hist_df = pd.read_csv(path)
        else:
            logger.error("No data. Run python main.py train first.")
            return None

    logger.info("Building training features …")
    train_df = build_training_features(hist_df)

    if train_df is None or len(train_df) < 50:
        logger.error(f"Too few samples ({len(train_df) if train_df is not None else 0}). Check API key and season in config.py")
        return None

    X        = train_df[FEATURE_COLS].fillna(0)
    y_result = train_df["result"]
    y_over   = train_df["over25"]
    y_btts   = train_df["btts"]

    logger.info("Training 1X2 model …")
    Xtr, Xte, ytr, yte = train_test_split(X, y_result, test_size=0.2, random_state=42)
    rm = _result_model(); rm.fit(Xtr, ytr)
    acc = accuracy_score(yte, rm.predict(Xte))
    logger.info(f"1X2 accuracy: {acc:.3f}")
    print(classification_report(yte, rm.predict(Xte), target_names=["Home","Draw","Away"]))

    logger.info("Training Over/Under model …")
    om = _binary_model(); om.fit(X, y_over)
    cvo = cross_val_score(om, X, y_over, cv=5, scoring="accuracy")
    logger.info(f"O/U accuracy: {cvo.mean():.3f}")

    logger.info("Training BTTS model …")
    bm = _binary_model(); bm.fit(X, y_btts)
    cvb = cross_val_score(bm, X, y_btts, cv=5, scoring="accuracy")
    logger.info(f"BTTS accuracy: {cvb.mean():.3f}")

    models = {
        "result_model": rm, "over25_model": om, "btts_model": bm,
        "feature_cols": FEATURE_COLS,
        "trained_at":   pd.Timestamp.now().isoformat(),
        "n_samples":    len(train_df),
        "result_accuracy": float(acc),
        "over25_cv_acc":   float(cvo.mean()),
        "btts_cv_acc":     float(cvb.mean()),
    }
    path = f"{MODEL_DIR}/football_models.pkl"
    with open(path, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"Models saved → {path}")
    return models

def load_models() -> dict:
    path = f"{MODEL_DIR}/football_models.pkl"
    if not os.path.exists(path):
        logger.warning("No saved models. Run: python main.py train")
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def model_summary() -> str:
    m = load_models()
    if not m:
        return "No models trained yet."
    return (
        f"Trained:  {m.get('trained_at','?')}\n"
        f"Samples:  {m.get('n_samples',0)}\n"
        f"1X2 acc:  {m.get('result_accuracy',0):.1%}\n"
        f"O/U acc:  {m.get('over25_cv_acc',0):.1%}\n"
        f"BTTS acc: {m.get('btts_cv_acc',0):.1%}"
    )