# MNIST-CNN-Streamlit-Digit-Classifier
CNN-based handwritten digit classification using the MNIST dataset with an interactive Streamlit application.
# MNIST CNN Streamlit Digit Classifier

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset** and provides an interactive **Streamlit web application** for digit prediction.

## 📌 Project Overview

This project demonstrates the complete workflow of an image classification system:

**MNIST Dataset → Data Preprocessing → CNN Training → Model Evaluation → Prediction → Streamlit Application**

The CNN learns patterns from handwritten digit images and predicts one of the ten classes from **0 to 9**.

The Streamlit application allows a user to upload a digit image and receive:

- Predicted digit
- Prediction confidence
- Prediction probabilities
- Test accuracy
- Test loss
- Actual vs predicted sample results
- Training and validation accuracy curves
- Training and validation loss curves

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand the MNIST image classification dataset.
2. Preprocess grayscale images for CNN training.
3. Build and train a CNN using TensorFlow/Keras.
4. Evaluate the model using unseen test data.
5. Generate actual vs predicted results.
6. Visualize training and validation performance.
7. Develop an interactive Streamlit application for digit prediction.

---

## 🗂️ Dataset

The project uses the **MNIST handwritten digit dataset**.

- Image type: Grayscale
- Image size: **28 × 28 pixels**
- Number of classes: **10**
- Classes: **0, 1, 2, 3, 4, 5, 6, 7, 8, 9**

The pixel values are normalized from the original range of 0–255 to a range of 0–1 before training.

---

## 🧠 CNN Architecture

The CNN used in this project contains the following layers:

| Layer | Configuration |
|---|---|
| Input | 28 × 28 × 1 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Feature vector |
| Dense | 128 neurons, ReLU |
| Dropout | 0.2 |
| Output | 10 neurons, Softmax |

The **softmax** output provides probabilities for the ten digit classes.

---

## ⚙️ Training Configuration

The model was trained using:

- Optimizer: **Adam**
- Loss function: **Sparse Categorical Cross-Entropy**
- Evaluation metric: **Accuracy**
- Epochs: **5**
- Batch size: **128**
- Validation split: **10%**

---

## 📊 Results

The trained CNN achieved the following results:

| Metric | Result |
|---|---:|
| Training Accuracy | **99.03%** |
| Validation Accuracy | **98.98%** |
| Test Accuracy | **98.98%** |
| Training Loss | **0.0311** |
| Validation Loss | **0.0363** |
| Test Loss | **0.0330** |

### Final Performance

The model achieved approximately **98.98% accuracy on the MNIST test dataset**, showing strong performance in handwritten digit classification.

---

## 🌐 Streamlit Application

The project includes a Streamlit application that provides an interactive interface for the trained CNN.

### Main features

**1. Image Upload**

Users can upload a PNG, JPG, or JPEG image containing a handwritten digit.

**2. Digit Prediction**

The CNN predicts the digit from **0 to 9**.

**3. Confidence Score**

The application displays the confidence associated with the prediction.

**4. Prediction Probability Chart**

The probability distribution for all ten digit classes is displayed.

**5. Model Performance**

The application displays the test accuracy and test loss.

**6. CNN Results**

Sample actual and predicted labels are displayed.

**7. Training Curves**

Training/validation accuracy and loss graphs are available inside the application.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MNIST-CNN-Streamlit-Digit-Classifier.git
cd MNIST-CNN-Streamlit-Digit-Classifier
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install the required libraries

```bash
pip install tensorflow streamlit numpy matplotlib pillow
```

### 3. Run the Streamlit application

```bash
streamlit run B_Abhishek_CNN_Streamlit_Project.py
```

The application will open in your browser.

> **Note:** The final Python file is self-contained and trains the CNN when the application starts. Therefore, the first launch may take some time.

---

## 📁 Project Files

```text
MNIST-CNN-Streamlit-Digit-Classifier/
│
├── B_Abhishek_CNN_Streamlit_Project.py
├── B_Abhishek_CNN_Streamlit_Project.pdf
└── README.md
```

### `B_Abhishek_CNN_Streamlit_Project.py`

Contains both:

- CNN image classification
- Streamlit application

### `B_Abhishek_CNN_Streamlit_Project.pdf`

Contains the project report, methodology, CNN architecture, results, evaluation, application details, challenges, and conclusion.

### `README.md`

Provides an overview of the project and instructions for running it.

---

## 🔄 Project Workflow

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
Image Normalization
      ↓
CNN Model Creation
      ↓
CNN Training
      ↓
Model Evaluation
      ↓
Prediction
      ↓
Streamlit Application
      ↓
User Uploads Digit Image
      ↓
Predicted Digit + Confidence
```

---

## 🧪 Testing

The Streamlit application was tested using MNIST handwritten digit images.

The application successfully:

- Loaded the CNN workflow
- Displayed model performance
- Accepted image uploads
- Predicted handwritten digits
- Displayed confidence values
- Displayed prediction probabilities
- Displayed CNN results
- Displayed training and validation curves

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Pillow**
- **Streamlit**
- **Google Colab**
- **GitHub**

---

## 💡 Challenges

Some challenges encountered during development included:

- Setting up Streamlit in Google Colab
- Connecting the Streamlit application for testing
- Processing uploaded digit images
- Ensuring the final Python file was self-contained
- Testing the application with different MNIST images

---

## 🔮 Future Improvements

Possible improvements include:

- Saving and loading the trained model instead of retraining during application startup.
- Improving preprocessing for real-world handwritten images.
- Adding image drawing directly inside the Streamlit application.
- Adding more detailed analysis of incorrect predictions.
- Improving the user interface.
- Deploying the application permanently using a cloud hosting platform.

---

## 👨‍💻 Author

**B. Abhishek**

B.Tech Computer Science and Engineering  
Central University of Andhra Pradesh

---

## 📜 Project Status

**Completed and Tested ✅**

This project was developed as an academic CNN image classification and Streamlit application project.

---

⭐ If you found this project useful, consider giving the repository a star!
