# Experimental Results

This document records all controlled experiments conducted in this project.
Each experiment varies **one primary factor** (prediction horizon or training duration) while keeping the dataset, preprocessing steps, and evaluation metrics consistent.

All models are evaluated using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

---

## Test 1 — Short-Term Forecasting (Baseline)

**Epochs:** 20  
**Prediction Horizon:** 1 hour  

### Results
- **Naive:** MSE = 0.008779 | MAE = 0.061699  
- **MLP:** MSE = 0.006956 | MAE = 0.058883  
- **LSTM:** MSE = 0.008009 | MAE = 0.064839  

### Best Model
**Multilayer Perceptron (MLP)**

### Insight
Simple feedforward models outperform both naive and sequence-based models for short-term energy forecasting, indicating that immediate feature relationships dominate one-hour-ahead predictions.

---

## Test 2 — Effect of Increased Training Duration

**Epochs:** 40  
**Prediction Horizon:** 1 hour  

### Results
- **Naive:** MSE = 0.008779 | MAE = 0.061699  
- **MLP:** MSE = 0.007348 | MAE = 0.063676  
- **LSTM:** MSE = 0.008757 | MAE = 0.066729  

### Best Model
**Multilayer Perceptron (MLP)**

### Insight
Increasing the number of training epochs did not improve performance and slightly degraded generalization, suggesting mild overfitting. Increased training duration does not necessarily result in improved predictive accuracy.

---

## Test 3 — Longer-Horizon Forecasting

**Epochs:** 40  
**Prediction Horizon:** 6 hours  

### Results
- **Naive:** MSE = 0.027286 | MAE = 0.126172  
- **MLP:** MSE = 0.012992 | MAE = 0.094923  
- **LSTM:** MSE = 0.011007 | MAE = 0.077650  

### Best Model
**Long Short-Term Memory (LSTM)**

### Insight
As the prediction horizon increases, temporal dependencies become more important. The LSTM outperforms simpler models, demonstrating the value of temporal memory for longer-term energy consumption forecasting.

---

## Test 4 — Regularization via Reduced Training

**Epochs:** 20  
**Prediction Horizon:** 6 hours  

### Results
- **Naive:** MSE = 0.027286 | MAE = 0.126172  
- **MLP:** MSE = 0.012768 | MAE = 0.092890  
- **LSTM:** MSE = 0.009570 | MAE = 0.074245  

### Best Model
**Long Short-Term Memory (LSTM)**

### Insight
Reduccing the number of training epochs improved generalization across all models. The LSTM performs best for longer forecasting horizons when training duration is carefully controlled, highlighting the importance of regularization for sequence-based architectures.

---

## Summary of Findings

Across all experiments, the following conclusions were observed:

- Model performance strongly depends on prediction horizon.
- MLP models perform best for short-term (1-hour-ahead) forecasting.
- LSTM models become advantageous for longer horizons due to their ability to model temporal dependencies.
- Increasing training duration does not guarantee better generalization and may lead to overfitting.
- Controlled training duration improves performance, particularly for LSTM models.

These findings emphasize the importance of aligning model architecture and training strategy with the temporal characteristics of the forecasting task.
