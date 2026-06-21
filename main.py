#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#import data

df=pd.read_csv('data/german_credit_data.csv')

#basic information about the dataset

print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print("null values",df.isnull().sum())
print("duplicate rows",df.duplicated().sum())
print(df.columns)


#data cleaning

df['Saving accounts']=df['Saving accounts'].fillna('unknown')
df['Checking account']=df['Checking account'].fillna('unknown')

print(df.isnull().sum())

#data visualization

#plot 1 
plt.figure(num="Age vs Risk", figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Risk', bins=20, kde=True,multiple='stack', palette='Set1')
plt.title("Age Distribution Histogram by Credit Risk Status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/age_vs_risk_histogram.png", dpi=300, bbox_inches="tight")

#plot 2
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Saving accounts', hue='Risk', palette='Set1')
plt.title('Saving Account Status vs Credit Risk')
plt.xlabel('Saving Account Type')  
plt.ylabel('Count') 
plt.tight_layout() 
plt.savefig("images/saving_accounts_vs_risk.png", dpi=300, bbox_inches="tight")

#plot 3
plt.figure(num="Credit Amount & Housing vs Risk", figsize=(10, 6))
sns.boxplot(data=df, x='Housing', y='Credit amount',hue='Risk', palette='Set1')          # hue kept — compares good/bad
plt.title('Credit Amount Distribution by Housing and Risk Status')
plt.xlabel('Housing Type')
plt.ylabel('Credit Amount')
plt.ylim(0, 12000)                              
plt.tight_layout() 
plt.savefig("images/credit_amount_housing_boxplot.png", dpi=300, bbox_inches="tight")

#plot 4
plt.figure(num="Correlation Heatmap", figsize=(10, 6))
numerical_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numerical_df.corr(numeric_only=True) , annot=True, cmap='coolwarm',fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")

#plot 5
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Duration', y='Credit amount',hue='Risk', palette='Set1', alpha=0.6)
plt.title('Loan Duration vs Credit Amount by Risk')
plt.xlabel('Loan Duration (months)')  
plt.ylabel('Credit Amount') 
plt.tight_layout()
plt.savefig("images/duration_vs_credit_amount.png", dpi=300, bbox_inches="tight")

#plot 6
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Purpose', hue='Risk', palette='Set1')
plt.title('Loan Purpose vs Credit Risk')
plt.xlabel('Purpose')  
plt.ylabel('Count') 
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/purpose_vs_risk.png", dpi=300, bbox_inches="tight")

plt.show()

#data preprocessing

#label encoding
from sklearn.preprocessing import LabelEncoder

categorical_cols = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
    "Risk"
]
for columns in categorical_cols:
    print({columns})
    print(df[columns].value_counts())


categorical_cols = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
    "Risk"
]

encoders = {}

for column in categorical_cols:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    encoders[column] = le

print(df.head())

print(df["Risk"].unique())
print(df["Risk"].value_counts())

#feature selection and train test split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Risk', axis=1)
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("training features",X_train.shape)
print("testing features",X_test.shape)

print("training labels",y_train.shape)
print("testing labels",y_test.shape)


#save preprocessed data
processed_df = pd.DataFrame(X, columns=X.columns)
processed_df['Risk']=y.values

processed_df.to_csv("data/processed_credit_data.csv",index=False)



# save scaler
joblib.dump(scaler, "saved_models/scaler.pkl")

# save label encoders
joblib.dump(encoders, "saved_models/label_encoders.pkl")

