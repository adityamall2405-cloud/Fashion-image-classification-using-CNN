# Fashion-MNIST Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) for classifying Fashion-MNIST images into 10 different clothing categories.

## Dataset

The dataset consists of:
- **Training data**: 60,000 images (6,000 per class) in the `train/` folder
- **Test data**: 10,000 images (1,000 per class) in the `test/` folder

### Classes (0-9):
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## Project Structure

```
.
├── train/              # Training images organized by class (0-9)
├── test/               # Test images organized by class (0-9)
├── fashion_mnist_cnn.ipynb  # Main Jupyter notebook for training and testing
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook:**
   - Open `fashion_mnist_cnn.ipynb` in Jupyter

## Usage

1. Run all cells in the notebook sequentially
2. The notebook will:
   - Load and preprocess the data
   - Build a CNN model
   - Train the model with validation
   - Evaluate on test data
   - Generate visualizations (confusion matrix, sample predictions, etc.)
   - Save the trained model

## Model Architecture

The CNN model consists of:
- **3 Convolutional Blocks** with increasing filters (32, 64, 128)
- **Batch Normalization** layers for stable training
- **Max Pooling** layers for dimensionality reduction
- **Dropout** layers to prevent overfitting
- **Dense layers** (512, 256, 10) for classification

## Training

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20% of training data

## Results

After training, the model will display:
- Training and validation accuracy/loss curves
- Test accuracy and loss
- Classification report with precision, recall, and F1-score
- Confusion matrix
- Per-class accuracy analysis
- Sample predictions with confidence scores

## Saved Models

The trained model will be saved as:
- `fashion_mnist_cnn_model.h5` - Keras H5 format
- `fashion_mnist_cnn_model_savedmodel/` - TensorFlow SavedModel format

## Requirements

- Python 3.8+
- TensorFlow 2.13.0+
- Keras 2.13.0+
- NumPy, Matplotlib, Seaborn, Pandas
- Pillow, scikit-learn
- Jupyter Notebook

## Notes

- The model uses data augmentation techniques (normalization, reshaping)
- Early stopping and learning rate reduction callbacks are used to prevent overfitting
- All images are 28x28 grayscale images

