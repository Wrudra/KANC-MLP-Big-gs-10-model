# KANC MLP (Big) (gs = 10) Model

This repository contains an implementation of the **KANC MLP (Big) (gs = 10)** model from the paper *"Convolutional Kolmogorov-Arnold Networks"* [Link to the Paper](https://arxiv.org/abs/2406.13155). The **KANC MLP (Big) (gs = 10)** model was trained for 10 epochs on Kaggle using a P100 GPU, with an 80-20 train-validation split on the Fashion MNIST dataset.

### Model Architecture

The model architecture is as follows:

- **Input**: 28x28x1 grayscale Fashion MNIST images.
- **Layers**:
  - 3 KAN Convolutional layers (1→16, 16→16, 16→16), each with 3x3 kernels, `grid_size=10`.
  - 2 MaxPooling layers (2x2) after the first two conv layers.
  - Flatten to 144 features (3x3x16).
  - Fully connected layer (144→10).
  - Log softmax for 10-class classification.
- **Parameters**: ~33.54K (per the paper).
- **Output**: 10 classes (Fashion MNIST labels).
  
### Test Set Metrics

After training, the model was evaluated on the 10,000-image Fashion MNIST test set, yielding the following results:

- **Accuracy**: 87.66%
- **Precision**: 0.88
- **Recall**: 0.88
- **F1 Score**: 0.88

These metrics indicate balanced performance across the 10 classes, with 87.66% of test images correctly classified and consistent precision, recall, and F1 scores.

### Training Dynamics

The following plots illustrate the model’s training progress over 10 epochs:

![image](https://github.com/user-attachments/assets/f61cc618-26a3-42d9-be1a-44391798e744)

- **Training and Validation Loss**:
  - Training loss decreases steadily from ~0.65 to ~0.35.
  - Validation loss decreases from ~0.55 to ~0.40, closely tracking training loss, indicating good generalization with no significant overfitting.
- **Validation Accuracy**:
  - Increases from ~81% to ~87% on the validation set (12,000 images), showing consistent improvement.
  - The upward trend suggests potential for higher accuracy with more epochs.
- **Learning Rate Schedule**:
  - Remains constant at 0.001, as the `ReduceLROnPlateau` scheduler (with `patience=2`) did not trigger a reduction.

### Comparison to Paper and Potential

The paper *"Convolutional Kolmogorov-Arnold Networks"* reports that the **KANC MLP (Big) (gs = 10)** model achieves an accuracy of 89.15% on Fashion MNIST, with precision, recall, and F1 scores of 89.22%, 89.15%, and 89.14%, respectively. In comparison:

- **This Run**: Achieves 87.66% accuracy, which is 1.49% lower than the paper’s result. Precision, recall, and F1 scores (0.88 each) are also ~1.3-1.4% lower.
- **Potential for Improvement**:
  - The steady decrease in loss and the upward trend in validation accuracy (reaching ~87% by epoch 10) suggest that training for more epochs (e.g., 15-20) could close the gap to the paper’s 89.15%.
  - The constant learning rate (0.001) may have contributed to slight fluctuations in validation accuracy. Reducing the learning rate (e.g., to 0.0005 after epoch 5) or adjusting the scheduler (`patience=1`) could stabilize training and help achieve the paper’s performance.

### Improving Performance

To enhance the model’s accuracy, consider the following adjustments:

- **More Epochs**: Extend training to 15-20 epochs to potentially reach ~89% accuracy, as the validation accuracy is still increasing.
- **Learning Rate**: Adjust the scheduler (`patience=1`) or manually reduce the learning rate to 0.0005 after epoch 5 to stabilize training and improve convergence.
- **Grid Size**: Experiment with `grid_size=15` to increase the spline’s flexibility, which may boost performance at the cost of more parameters.
