import time
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Генерация тестового датасета
X, y = make_classification(
    n_samples=50000, n_features=20, n_informative=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# 2. Оценка LightGBM
lgb_model = LGBMClassifier(random_state=42, verbose=-1)

start = time.time()
lgb_model.fit(X_train, y_train)
lgb_train_time = time.time() - start

start = time.time()
lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
lgb_infer_time = time.time() - start

lgb_auc = roc_auc_score(y_test, lgb_preds)

# 3. Оценка CatBoost
cb_model = CatBoostClassifier(random_state=42, verbose=0)

start = time.time()
cb_model.fit(X_train, y_train)
cb_train_time = time.time() - start

start = time.time()
cb_preds = cb_model.predict_proba(X_test)[:, 1]
cb_infer_time = time.time() - start

cb_auc = roc_auc_score(y_test, cb_preds)

# 4. XGB оценка
xgb_model = XGBClassifier(
    random_state=42,
    verbosity=0,
    eval_metric='logloss',
    n_jobs=-1
)

start = time.time()
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - start

start = time.time()
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
xgb_infer_time = time.time() - start

xgb_auc = roc_auc_score(y_test, xgb_preds)

# 5. Вывод результатов
results = pd.DataFrame(
    {
        "Model": ["LightGBM", "CatBoost", "XGBoost"],
        "ROC-AUC": [lgb_auc, cb_auc, xgb_auc],
        "Train Time (s)": [lgb_train_time, cb_train_time, xgb_train_time],
        "Inference Time (s)": [lgb_infer_time, cb_infer_time, xgb_infer_time],
    }
)

print(results)