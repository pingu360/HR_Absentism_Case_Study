#!/usr/bin/env python
# coding: utf-8

# In[144]:


# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# create the special class that we are going to use from here on to predict new data
class absenteeism_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.logreg_simplified = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # Group the reasons
            df = pd.read_csv(data_file,delimiter=',')
            bins = [1, 14, 17, 21, 28]
            labels = ['1', '2', '3', '4']
            df['Reason'] = pd.cut(df['Reason for Absence'], bins=bins, labels=labels, include_lowest=True)
            df = df.drop(labels = ['Reason for Absence', 'ID'], axis = 1)
            
            # Convert the Date
            df['Date'] = pd.to_datetime(df['Date'] , format='%d/%m/%Y')
            df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month Value'] = df['Date'].dt.month
            df['Day of the Week'] = df['Date'].dt.dayofweek
            df = df.drop(labels = 'Date', axis = 1)
            
            # Define the mapping dictionary
            mapping = {1: 0, 2: 1, 3: 1, 4: 1}
            # Method 1: Using map() function
            df['Education'] = df['Education'].map(mapping)
            
            #Reorder columns
            columns = [
            'Reason',
            'Month Value',
            'Day of the Week',
            'Transportation Expense',
            'Distance to Work',
            'Age',
            'Daily Work Load Average',
            'Body Mass Index',
            'Education',
            'Children',
            'Pets',
            'Absenteeism Time in Hours',
            ]
            
            # Create dummy variables for the "Category" column
            dummy_variables = pd.get_dummies(df['Reason'], prefix='Reason')
            # Concatenate the dummy variables with the original DataFrame
            df = pd.concat([dummy_variables ,df], axis=1)
            # Drop the Reason column
            df = df.drop('Reason',axis = 1)
            self.preprocessed_data = df.copy()
            
            
            columns_to_scale = ['Transportation Expense','Age','Children','Pets']
            df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
            
            columns_to_drop = ['Body Mass Index', 'Month Value',
                   'Daily Work Load Average', 'Distance to Work', 'Education',
                   'Day of the Week']
            
            df = df.drop(labels = columns_to_drop ,axis = 1)
            self.data = df
            

            
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.logreg_simplified.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.logreg_simplified.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.logreg_simplified.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.logreg_simplified.predict(self.data)
                return self.preprocessed_data

