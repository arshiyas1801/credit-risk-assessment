#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

#import dataset
df=pd.read_csv("data/processed_credit_data.csv")
print(df.head())

#feature selection and train test split

X=df.drop("Risk",axis=1)
y=df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#feature scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#save scaler
joblib.dump(scaler,"saved_models/scaler.pkl")

#create model
model = Sequential()

model.add(Dense(
    16,
    activation='relu',
    input_shape=(X_train.shape[1],)
))
model.add(Dropout(0.2))

model.add(Dense(
    8,
    activation='relu'
))
model.add(Dropout(0.2))

model.add(Dense(
    1,
    activation='sigmoid'
))

#compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#train model
history=model.fit(
    X_train,y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("model training completed")

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Neural Network Training History")
plt.legend()

plt.savefig(
    "images/neural_network_training_history.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#predictions
y_prob=model.predict(X_test).ravel()
nn_pred=(y_prob > 0.5).astype(int)

#evaluation
print("neural network results")
print("Accuracy:", accuracy_score(y_test, nn_pred))
print("Precision:", precision_score(y_test, nn_pred))
print("Recall:", recall_score(y_test, nn_pred))
print("F1 Score:", f1_score(y_test, nn_pred))

print("classification report",classification_report(y_test,nn_pred))

#confusion matrix

plt.figure(figsize=(8,5))
cm=confusion_matrix(y_test,nn_pred)

sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix neural network")

plt.tight_layout()
plt.savefig(
    "images/neural_network_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
    )
plt.show()

#roc curve
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("neural network roc curve")
plt.legend()

plt.tight_layout()
plt.savefig("images/neural_network_roc_curve.png",dpi=300,bbox_inches="tight")
plt.show()

#save model
model.save("saved_models/neural_network.keras")
print("neural network model saved successfully")
