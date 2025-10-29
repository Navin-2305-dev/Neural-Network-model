# 🧠 Building a Neural Network from Scratch using NumPy and Pandas

This project demonstrates how to **build, train, and evaluate a simple neural network from scratch** using only **NumPy** and **Pandas**, without relying on any high-level deep learning frameworks such as TensorFlow or PyTorch.  
It provides a clear, educational implementation of **forward propagation, backward propagation, and gradient descent** for image classification tasks.

---

## 🚀 Project Overview

The main goal of this project is to help understand **how a neural network learns** through core mathematical operations.  
The model is trained on a labeled dataset (like MNIST) and can classify images (digits 0–9) using a **2-layer feedforward neural network**.

### 🔍 Key Concepts Covered
- Data preprocessing with **Pandas** and **NumPy**
- Neural network initialization
- **Forward Propagation** with ReLU and Softmax activations
- **Loss computation** using cross-entropy
- **Backward Propagation** (manual gradient computation)
- **Parameter updates** using Gradient Descent
- Model **accuracy evaluation**

---

## 🧩 Technologies Used
- **Python 3.x**
- **NumPy** — for numerical computations  
- **Pandas** — for data loading and preprocessing  
- **Matplotlib** — for visualization (optional)

---

## 🧠 Neural Network Architecture

| Layer | Type | Activation | Description |
|-------|------|-------------|--------------|
| Input | 784 neurons | — | Flattened 28×28 grayscale image |
| Hidden | 10 neurons | ReLU | Learns intermediate features |
| Output | 10 neurons | Softmax | Predicts class probabilities |

---

## 🧮 Implementation Steps

1. **Data Loading**
   ```python
   data = pd.read_csv('/content/sample_data/train.csv')
   ```
   The dataset is loaded and converted into NumPy arrays for faster computation.

2. **Data Preprocessing**
   - Normalization of input features (`X / 255.0`)
   - Splitting dataset into training and development sets

3. **Model Initialization**
   ```python
   def init_params():
       W1 = np.random.randn(10, 784) * 0.01
       b1 = np.zeros((10, 1))
       W2 = np.random.randn(10, 10) * 0.01
       b2 = np.zeros((10, 1))
       return W1, b1, W2, b2
   ```

4. **Forward Propagation**
   - Linear transformation → ReLU activation → Softmax output

5. **Backward Propagation**
   - Gradient computation for each parameter  
   - Parameter update using learning rate `α`

6. **Model Training**
   ```python
   W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=500)
   ```

7. **Evaluation**
   - Predictions using `np.argmax()`
   - Accuracy calculation on the dev set

---

## 📊 Example Output

```
Iteration: 100  Accuracy: 85.6%
Iteration: 200  Accuracy: 89.3%
Iteration: 500  Accuracy: 92.1%
```

---

## 🧪 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Neural-Network-Numpy.git
   cd Neural-Network-Numpy
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. Open the Jupyter Notebook or run as a Python script:
   ```bash
   jupyter notebook Copy_of_Welcome_To_Colab.ipynb
   ```

4. (Optional) Modify dataset path in the code:
   ```python
   data = pd.read_csv('/content/sample_data/train.csv')
   ```

---

## 📚 Learning Outcome

By going through this notebook, you’ll:
- Understand the **mathematical foundation** of neural networks  
- Learn how to implement **gradient descent** manually  
- Gain confidence in debugging and improving neural models  
- Build a foundation before transitioning to advanced frameworks like TensorFlow or PyTorch  

---

## 🏆 Future Improvements
- Add more hidden layers and non-linearities  
- Implement mini-batch gradient descent  
- Include dropout regularization  
- Visualize training loss and accuracy graphs  
- Extend to **Convolutional Neural Networks (CNNs)**  

---

## 👨‍💻 Author
**Navin B**  
🎓 Department of Artificial Intelligence and Data Science  
Sri Sairam Engineering College  
💡 Passionate about AI, ML, and deep learning  
