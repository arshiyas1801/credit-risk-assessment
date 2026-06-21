#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_curve,auc)

#load dataset
df=pd.read_csv("data/processed_credit_data.csv")
print(df.head())

#feature selection and train test split
X=df.drop('Risk',axis=1)
y=df['Risk']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#create model
rf=RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42)

#train model
rf.fit(X_train,y_train)
print("model training completed")

#model prediction
rf_pred=rf.predict(X_test)
y_prob=rf.predict_proba(X_test)[:,1]

#model evaluation
print("Random forest results")
print("accuracy score:",accuracy_score(y_test,rf_pred))
print("precision score:",precision_score(y_test,rf_pred))
print("recall score:",recall_score(y_test,rf_pred))
print("f1 score:",f1_score(y_test,rf_pred))

print("classification report:\n",classification_report(y_test,rf_pred))

plt.figure(figsize=(10,6))
cm=confusion_matrix(y_test,rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig(
    "images/random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#roc curve
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(8,5))
plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Postive Rate")
plt.title("Random Forest ROC curve")

plt.legend()


plt.savefig(
    "images/random_forest_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#feature importance

importance=pd.DataFrame({"feature":X.columns,"importance":rf.feature_importances_})
importance=importance.sort_values(by="importance",ascending=False)

print("Feature Importance:-")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="importance",
    y="feature",
    hue="feature",
    palette="viridis",
    legend=False
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    "images/random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#save model

joblib.dump(rf,"saved_models/random_forest.pkl")
print("\nRandom Forest model saved successfully!")



