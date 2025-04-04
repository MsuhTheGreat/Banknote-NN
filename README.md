#  **🌟 My Implementation of Neural Networks Using Hidden Layers 🌟**
This project implements a fully connected neural network from scratch using Python and NumPy. It includes forward propagation, backpropagation, and gradient descent for training a model to classify banknotes as genuine or counterfeit. 💰🔍

## **✨ Features**
✅ Implements a **customizable** multi-layer neural network.  
✅ Uses **tanh** activation for hidden layers & **sigmoid** for output.  
✅ Trains on the **Banknote Authentication Dataset**. 🏦  
✅ Saves trained weights & biases for later testing. 🧠💾  
✅ **Custom accuracy formula** based on loss function. 📊  

## **📂 Dataset**
The project uses the **Banknote Authentication Dataset** from the UCI Machine Learning Repository. It contains **four extracted features** from banknote images.

📌 **Source:**  [UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt)  

📌 **Features:**

- 🖼 Variance of wavelet-transformed image

- 📈 Skewness of wavelet-transformed image

- 📊 Kurtosis of wavelet-transformed image

- 🔢 Entropy of the image

📌 **Labels:**

- ❌ 0 → Fake banknote

- ✅ 1 → Genuine banknote

## **🛠 Installation & Setup**
### **1️⃣ Clone the Repository**
```
git clone https://github.com/MsuhTheGreat/Banknote-NN.git
cd Banknote-NN
```
### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```
python -m venv venv  

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
### **3️⃣ Install Dependencies**
```
pip install -r requirements.txt
```  
## **🚀 Usage**
### **🎯 Training the Model**
Run `train.py` to train the neural network:
```
python source/train.py
```
You'll be prompted to enter:  
📌 Number of **epochs** ⏳  
📌 **Learning rate** (Alpha) 🎚  
📌 Number of **layers** 🏗  
📌 Number of **neurons** in each layer 🔢  

🔹 **Weight Initialization**: Weights are initialized using **Xavier** method.  
🔹 **Training Progress**: Cost updates displayed every **1000** epochs.  
🔹 **Model Saved To:** `models/` directory. 💾  

### **🧪 Testing the Model**
Run `test.py` to evaluate the trained model:
```
python source/test.py
```  
You'll be asked to enter the number of layers used during training. The script will then:  
✅ **Perform Forward Propagation** on test data.  
✅ **Calculate Accuracy** using both traditional formula & **my own custom formula.** 📊

#### **Example Output**  
##### **When I used Tanh**  
Accuracy According To My Custom Formula: 99.7289911031016  
Accuracy According To Traditional Formula: 100.0  
##### **When I used ReLU**  
Accuracy According To My Custom Formula: 99.78671729471988  
Accuracy According To Traditional Formula: 100.0  

## **📁 Project Structure**
```
Banknote-NN/
│── data/                      # Stores test data 📂
│── models/                    # Stores trained weights & biases 🧠
│── source/
│   ├── train.py               # Training script 🏋️‍♂️
│   ├── test.py                # Testing script 🧪
│── .gitignore                 # Ignore unnecessary files 🚫
│── LICENSE                    # MIT License 📜
│── README.md                  # Project documentation 📖
│── Requirements.txt           # Required dependencies 📦
```
## **🎯 Accuracy Calculation**
### **✅ Traditional Accuracy**
$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100
$$  

### **🔥 Custom Accuracy Formula**
Uses the sigmoid of the cost function:

$$
J_{\text{sigmoid}} = \frac{1}{1 + e^{-J}}
$$

$$
\text{Accuracy} = (1 - J_{\text{sigmoid}}) \times 200
$$

### **🔮 Intuition for Custom Accuracy Formula**
In Deep Learning, the cost function measures the inaccuracy of a model. I wanted to transform it into a percentage-based accuracy measure. Since the cost function ranges from [0, ∞), I applied the sigmoid function to map it into a bounded range. Through experimentation, I found that multiplying by 200 effectively scales the accuracy, making it more interpretable.

## **🤝 Contributing**
If you’d like to contribute:

1. Fork the repo
2. Create a new branch
3. Commit your changes
4. Open a pull request
## **📜 License**
This project is open-source and available under the MIT License.
## **💡 Author**
Created by MsuhTheGreat. Feel free to connect! 😊