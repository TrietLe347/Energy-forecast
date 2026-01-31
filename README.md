# Energy Consumption Forecasting with Deep Learning

This is my first personal deep learning project, completed alongside my *Intro to Deep Learning* course at California State University of San Marcos.  
The project explores short-term and medium-term household energy consumption forecasting using multiple neural network architectures, with a focus on how prediction horizon and training strategy affect model performance.

## Models Evaluated
- Naive baseline (persistence model)
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)

## Dataset
The project uses the *Individual Household Electric Power Consumption* dataset from the UCI Machine Learning Repository.  
It contains over four years of minute-level household energy measurements, which are resampled to hourly resolution for modeling.

Dataset link:  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Experiments
Four controlled experiments were conducted to evaluate the impact of prediction horizon and training duration:

| Test   | Epochs | Horizon | Best Model | Key Insight                                                             |
|--------|--------|---------|------------|-------------------------------------------------------------------------|
| Test 1 | 20     | 1 hour  | MLP        | Simple models excel at short horizons                                   |
| Test 2 | 40     | 1 hour  | MLP        | More training does not improve generalization                           |
| Test 3 | 40     | 6 hours | LSTM       | Temporal memory matters for longer horizons                             |
| Test 4 | 20     | 6 hours | LSTM       | Reduced training improves generalization for longer-horizon forecasting |

Detailed experimental results are documented in experiments.md

## Key Findings
- Model performance strongly depends on the prediction horizon.
- MLPs outperform sequence-based models for short-term forecasting tasks.
- LSTMs become advantageous as the forecast horizon increases.
- Increased model complexity or training duration does not guarantee better generalization.

## Technologies
- Python
- NumPy, Pandas, Scikit-learn
- TensorFlow / Keras
- Matplotlib

## Status
This is an independent personal project created to deepen my understanding of deep learning for time-series forecasting and experimental model evaluation.


## Follow-Up Work: Regularization and Generalization

During this project, I observed that increasing the number of training epochs beyond an optimal point led to overfitting, resulting in degraded performance on unseen data. This behavior was especially noticeable for longer prediction horizons, where model complexity and training duration had a stronger impact on generalization.
Through this process, I learned about regularization techniques such as early stopping and dropout, which are commonly used to mitigate overfitting in deep learning models. As a follow-up, I plan to apply these techniques in a separate project to evaluate how they improve generalization performance for medium-term energy forecasting while keeping the dataset and evaluation metrics consistent.
