#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report


# In[3]:


class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load(self, path='historic.csv'):
        self.df = pd.read_csv(path)

    def preprocess(self):
        df = self.df.copy()
        df.drop('item_no', axis=1, inplace=True)

        # Encode target label
        df['success_indicator'] = df['success_indicator'].apply(lambda x: 1 if x == 'top' else 0)

        # Label encode categorical features
        for col in ['category', 'main_promotion', 'color']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop('success_indicator', axis=1)
        y = df['success_indicator']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        y_pred = self.model.predict(self.X_test)
        print("Evaluation Report:\n")
        print(classification_report(self.y_test, y_pred))

    def predict(self, input_file='prediction_input.csv'):
        df_pred = pd.read_csv(input_file)
        df = df_pred.copy()

        # Apply same label encoding
        for col in ['category', 'main_promotion', 'color']:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])

        X_pred = df.drop('item_no', axis=1)
        X_pred_scaled = self.scaler.transform(X_pred)

        predictions = self.model.predict(X_pred_scaled)

        # Return item_no with prediction
        result = pd.DataFrame({
            'item_no': df_pred['item_no'],
            'prediction': ['top' if p == 1 else 'flop' for p in predictions]
        })

        print(result.head())
        return result


