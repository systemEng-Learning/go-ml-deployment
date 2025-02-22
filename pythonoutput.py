import numpy as np
import onnxruntime as ort

# Define the classifier weights
coefficients = np.array([
    -0.50130147, 0.8419699, -2.2627807, -0.96755445, 0.5673314,
    -0.37099984, -0.17644618, -0.8879377, -0.06602986, -0.47097006,
    2.4392269, 1.8554921
]).reshape(3, 4)  # Reshaped to (num_classes, input_dim)

intercepts = np.array([9.628701, 1.8089765, -11.437677])
class_labels = np.array([0, 1, 2])  # Integer class labels

# Define Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Prevent overflow
    return e_x / e_x.sum(axis=1, keepdims=True)

# **Step 1: Linear Classification**
def linear_classifier(input_tensor):
    logits = np.dot(input_tensor, coefficients.T) + intercepts
    probabilities = softmax(logits)  # Softmax activation
    predicted_label = np.argmax(probabilities, axis=1)
    return predicted_label, probabilities

# **Step 2: Cast label to int64**
def cast_to_int64(label):
    return label.astype(np.int64)

# **Step 3: Normalize Probabilities**
def l1_normalize(probabilities):
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)

# **Step 4: ZipMap (Mapping probabilities to class labels)**
def zipmap(probabilities):
    return [{class_labels[i]: prob[i] for i in range(len(class_labels))} for prob in probabilities]

# **Test the model with a sample input**
input_tensor = np.array([[1.0, 2.0, 3.0, 4.0]])  # Example input

# Execute the pipeline
label, probability_tensor = linear_classifier(input_tensor)
output_label = cast_to_int64(label)
probabilities = l1_normalize(probability_tensor)
output_probability = zipmap(probabilities)

# **Print the results**
print("Predicted Label:", output_label)
print("Class Probabilities:", output_probability)
