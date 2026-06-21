#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report , confusion_matrix , f1_score,recall_score,precision_score,accuracy_score,roc_curve,auc)

#import dataset

df=pd.read_csv("data/processed_credit_data.csv")
print(df.head())

#feature selection and train test split

X=df.drop("Risk",axis=1)
y=df["Risk"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "saved_models/scaler.pkl")

#create model
xgb=XGBClassifier(n_estimators=100,max_depth=8,random_state=42,learning_rate=0.1,eval_metric="logloss")

#train model
xgb.fit(X_train,y_train)

print("Classes:", xgb.classes_)
print("model training completed")

#model prediction
xgb_pred=xgb.predict(X_test)
y_prob=xgb.predict_proba(X_test)[:,1]

#model evaluation

print("XGBoost results")
print("accuracy score",accuracy_score(y_test,xgb_pred))
print("precision score",precision_score(y_test,xgb_pred))
print("recall score",recall_score(y_test,xgb_pred))
print("f1 score",f1_score(y_test,xgb_pred))

print("classification report:\n",classification_report(y_test,xgb_pred))

plt.figure(figsize=(8,5))
cm=confusion_matrix(y_test,xgb_pred)
print(cm)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("confusion matrix xgboost")
plt.xlabel("predicted")
plt.ylabel("actual")

plt.tight_layout()

plt.savefig(
    "images/xgboost_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#roc curve
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(8,5))

plt.plot(fpr,tpr,color="blue",label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost")

plt.legend()
plt.tight_layout()

plt.savefig(
    "images/xgboost_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#feature importance

importance=pd.DataFrame(
    {"feature":X.columns,
     "importance":xgb.feature_importances_}
     )

importance=importance.sort_values(
    by="importance",
    ascending=False
)

print("feature importance")
print(importance)

sns.barplot(
    data=importance,
    x="importance",
    y="feature",
    hue="feature",
    palette="viridis",
    legend=False
)

plt.title("XGBoost Feature Importance")
plt.tight_layout()

plt.savefig(
    "images/xgboost_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#save model

joblib.dump(xgb,"saved_models/xgboost.pkl")

print("XGBoost model saved successfully!")





