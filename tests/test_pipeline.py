import pytest
import pandas as pd
from train_model import model, X_test, y_test, X_train, y_train
from sklearn.metrics import r2_score

# Test accuracy (R² Score)
def test_accuracy():
    preds = model.predict(X_test)
    accuracy = r2_score(y_test, preds)
    assert accuracy > 0.8, "Accuracy (R² Score) is below the acceptable threshold!"

# Additional test for data quality
#def test_missing_values():
#    data = pd.read_csv('credit_data.csv')  # Change filename if necessary
#    assert data.isnull().sum().sum() == 0, "Dataset contains missing values!"

# Integration test for pipeline
def test_pipeline_execution():
    assert len(X_train) > 0, "Training data is empty!"
    assert len(y_train) > 0, "Labels are empty!"

