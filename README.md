# EMNIST Character Classification with TensorFlow

This project demonstrates the implementation, training, and fine-tuning of two state-of-the-art neural network architectures—Xception and DenseNet—on the EMNIST dataset. The results of the two models are compared to evaluate their performance. Additionally, a user-friendly GUI has been created using Streamlit for visualizing and interacting with the models.

## Features

- **Model Training and Fine-Tuning:**
  - Xception: Implemented and trained from scratch.
  - DenseNet: Implemented and trained from scratch.
  - Fine-tuning both models for better performance.
- **Dataset:**
  - EMNIST dataset (available on Kaggle): Extended MNIST for handwritten character recognition.
- **Comparison:**
  - Analyzing and comparing the performance metrics of Xception and DenseNet.
- **GUI:**
  - A Streamlit-based graphical user interface (`app.py`) for interacting with the models and visualizing predictions.

## Dependencies

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## Run the GUI

To run the Streamlit app, run the Deployment notebook `deployment.ipynb`.

## Results

The results of the comparison is in terms of:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **Training Time**

These metrics are shown in both, the notebook and the documentation `report.pdf`.

## Dataset

The EMNIST dataset is available on Kaggle. You can download it from [here](https://www.kaggle.com/crawford/emnist).

## Acknowledgements

- [EMNIST Dataset](https://www.kaggle.com/crawford/emnist)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/abs/1608.06993)
