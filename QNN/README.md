# QNN (v3.1.0) Implementation Documentation

## 1. Introduction
This documentation provides a comprehensive guide to the QNN (v3.1.0) implementation. Our hybrid quantum-classical neural network framework combines traditional convolutional neural networks (CNNs) with quantum circuits to create advanced hybrid models that can leverage quantum computational advantages while maintaining classical deep learning strengths.

The implementation is specifically designed for data privacy and security applications, featuring quantum entanglement strategies, noise modeling, adversarial defense mechanisms, and feature encoding techniques.

## 2. Hybrid Quantum-Classical Approach
Our implementation uses a hybrid approach where:

- A classical CNN extracts features from input data
- These features are encoded into quantum states
- Quantum operations are applied to process information
- Quantum measurements generate outputs that are combined with classical outputs

## 3. Installation and Requirements
TBC

## 4. Implementation Architecture
Our QNN implementation follows a modular architecture with these major components:

### 4.1 High-Level Architecture
```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Classical    │    │   Feature     │    │    Quantum    │    │    Dynamic    │
│  CNN Model    │───>│  Extraction   │───>│    Circuit    │───>│   Weighting   │───> Output
└───────────────┘    └───────────────┘    └───────────────┘    └───────────────┘
```

### 4.2 Data Flow
1. **Input Processing**: Raw data is processed by a classical CNN
2. **Feature Extraction**: Intermediate features are extracted from CNN layers
3. **Feature Dimensionality Reduction**: FDR methods reduce feature dimensions
4. **Quantum Encoding**: Classical features are encoded into quantum states
5. **Quantum Processing**: Quantum circuit applies rotations and entanglement operations
6. **Quantum Measurement**: Quantum states are measured to produce probabilities
7. **Hybrid Combination**: Classical and quantum outputs are combined adaptively
8. **Output Generation**: Final classification output is produced

## 5. Key Components

### 5.1 FeatureExtractor Class
Purpose: Extracts intermediate features from the provided CNN model

```python
# Example usage
feature_extractor = FeatureExtractor(resnet_model, layer_name="layer3")
output = feature_extractor(input_data)
features = feature_extractor.get_features()
```

Key methods:
- `__init__(model, layer_name)`: Initializes extractor with specified layer
- `_register_hooks()`: Sets up hooks to capture intermediate features
- `get_features()`: Returns extracted features from the last forward pass

### 5.2 Entanglement Layer
Purpose: Applies different entanglement patterns to quantum qubits

```python
# Available entanglement types
entanglement_types = [
    'no_entanglement_ansatz',
    'linear_entanglement_ansatz',
    'full_entanglement_ansatz',
    'star_entanglement_ansatz'
]
```

Implemented patterns:
- **No entanglement**: Qubits operate independently
- **Linear entanglement**: Adjacent qubits are entangled (nearest-neighbor)
- **Full entanglement**: All-to-all qubit entanglement
- **Star entanglement**: Central qubit connected to all others

### 5.3 DynamicWeightingModule
Purpose: Adaptively balances classical and quantum components

```python
# Example initialization
weighting_module = DynamicWeightingModule(input_dim=7, hidden_dim=128, num_layers=3)
alpha = weighting_module(classical_output, quantum_output)
```

Key features:
- Extracts statistical properties from both outputs
- Calculates dynamic weighting based on features like mean, variance, skewness, correlation
- Uses attention-like mechanism to focus on the most reliable component

### 5.4 Hybrid Forward Function
Purpose: Core function implementing hybrid quantum-classical processing

Key steps:
1. Processes input through classical CNN
2. Applies feature extraction and encoding
3. Executes quantum circuit with appropriate parameters
4. Applies adaptive weighting of outputs
5. Returns final hybrid result

## 6. Step-by-Step Usage Guide
### 6.1 Basic Implementation
```python
import torch
import torch.nn as nn
from torchvision.models import resnet18
from QNN import get_qnn_model

# Step 1: Prepare your classical CNN model
cnn_model = resnet18(pretrained=True)
cnn_model.fc = nn.Linear(512, 10)  # For 10-class classification

# Step 2: Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 3: Create QNN model with defaults
qnn_model = get_qnn_model(cnn_model, device)

# Step 4: Use the model for inference
input_data = torch.randn(1, 3, 224, 224)  # Example input
output = qnn_model(input_data)
predicted_class = torch.argmax(output, dim=1)
```

### 6.2 Training Loop Example
```python
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set up data loaders
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
model = get_qnn_model(cnn_model, device)
optimizer = optim.Adam([
    {'params': model.cnn_model.parameters(), 'lr': 1e-4},
    {'params': [model.theta, model.phi, model.raw_noise], 'lr': 1e-2}
])

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = F.nll_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
```

### 6.3 Customizing the QNN Model
```python
# Create custom QNN model
custom_qnn = get_qnn_model(
    cnn_model=cnn_model,
    device=device,
    output_dim=10,                        # Number of classes
    circuit_depth=3,                      # Deeper quantum circuit
    batch_size_limit=32,                  # Memory-efficient batching
    param_mode='vector',                  # Use vector parameters
    num_params=8,                         # More trainable parameters
    noise_strength=0.1,                   # Increased noise for robustness
    use_shots=True,                       # Use shot-based sampling
    adversarial_defense=True,             # Enable defense mechanisms
    feature_dim_reduction=64,             # Reduce feature dimensions
    entanglement_type='full_entanglement_ansatz',  # Full entanglement
    use_dynamic_weights=True,             # Enable adaptive weighting
    use_backprop=False,                   # Use parameter-shift differentiation
    encoding_method='enhanced_angle',     # Enhanced encoding method
    noise_model='mixed'                   # Mixed noise model
)
```

## 7. Parameters and Configuration
### 7.1 Comprehensive Parameter List

| Parameter | Description | Type | Default | Range/Options |
|-----------|-------------|------|---------|---------------|
| cnn_model | Classical CNN model | nn.Module | Required | Any PyTorch model |
| device | Computation device | torch.device | Required | 'cuda' or 'cpu' |
| output_dim | Output dimension | int | 10 | ≥ 2 |
| circuit_depth | Quantum circuit depth | int | 2 | ≥ 1 |
| batch_size_limit | Max quantum batch size | int | 64 | ≥ 1 |
| param_mode | Parameter mode | str | 'vector' | 'scalar' or 'vector' |
| num_params | Number of parameters | int | 4 | ≥ 1 |
| noise_strength | Quantum noise strength | float | 0.05 | 0.0 to 1.0 |
| use_shots | Use shot-based sampling | bool | True | True/False |
| adversarial_defense | Enable adversarial defense | bool | True | True/False |
| feature_dim_reduction | Feature dimension after reduction | int | None | None or ≥ 1 |
| entanglement_type | Entanglement pattern | str | 'linear_entanglement_ansatz' | See section 8 |
| use_dynamic_weights | Enable dynamic weighting | bool | True | True/False |
| use_backprop | Use backpropagation | bool | True | True/False |
| encoding_method | Feature encoding method | str | 'enhanced_angle' | 'angle' or 'enhanced_angle' |
| noise_model | Quantum noise model | str | 'depolarizing' | See section 10 |
| feature_layer | CNN layer for feature extraction | str | None | Layer name or None |

### 7.2 QNN Model Configuration Example
```python
# Configure QNN with different settings for different tasks
# Image classification configuration
image_classification_qnn = get_qnn_model(
    cnn_model=cnn_model,
    device=device,
    output_dim=1000,                      # ImageNet classes
    circuit_depth=2,                      # Balance performance
    batch_size_limit=16,                  # For high-res images
    encoding_method='enhanced_angle',     # Better preservation of spatial info
    entanglement_type='linear_entanglement_ansatz'  # Efficient entanglement
)

# Adversarial robustness configuration
robust_qnn = get_qnn_model(
    cnn_model=cnn_model,
    device=device,
    noise_strength=0.15,                  # Higher noise
    adversarial_defense=True,             # Enable defense
    use_shots=True,                       # Adds natural noise
    noise_model='mixed',                  # Multiple noise types
    entanglement_type='full_entanglement_ansatz'  # More complex entanglement
)
```

## 8. Entanglement Approaches
### 8.1 No Entanglement Ansatz
Description: Qubits remain independent with no entanglement operations
Implementation:
```python
# No operations needed
pass
```
Use case: Baseline quantum processing, quantum feature embedding

### 8.2 Linear Entanglement Ansatz
Description: Each qubit is entangled with its adjacent qubits
Implementation:
```python
for i in range(num_qubits - 1):
    qml.CNOT(wires=[i, i+1])
```
Use case: Efficiently captures nearest-neighbor correlations with low circuit depth

### 8.3 Full Entanglement Ansatz
Description: All-to-all connections between qubits
Implementation:
```python
for i in range(num_qubits):
    for j in range(i + 1, num_qubits):
        qml.CNOT(wires=[i, j])
```
Use case: Maximum entanglement for capturing complex patterns, at cost of circuit depth

### 8.4 Star Entanglement Ansatz
Description: One central qubit entangled with all others
Implementation:
```python
central_qubit = 0
for i in range(1, num_qubits):
    qml.CNOT(wires=[central_qubit, i])
```
Use case: Efficient compromise between connectivity and circuit depth

### 8.5 Changing Entanglement Type at Runtime
```python
# Create model with default entanglement
model = get_qnn_model(cnn_model, device)

# Change entanglement type
model.set_entanglement_type('full_entanglement_ansatz')
```

## 9. Encoding Methods

Encoding classical data into quantum states is a crucial aspect of quantum machine learning. The QNN implementation offers two primary encoding methods:

### 9.1 Standard Angle Encoding

**Method name: `angle`**

This method uses RY rotations to encode classical features into quantum states:

1. **Normalization**: Features are first normalized by subtracting mean and dividing by standard deviation
2. **Angle Mapping**: The normalized features are transformed to rotation angles within [-π, π] using the arctangent function
3. **Qubit Assignment**:
   - If features > qubits: Features are combined using weighted averages
   - If features < qubits: Features are duplicated across multiple qubits

**Code Example**:
```python
# Calculate rotation angles for standard encoding
feature_mean = cnn_features.mean(dim=1, keepdim=True)
feature_std = cnn_features.std(dim=1, keepdim=True) + 1e-6
normalized_features = (cnn_features - feature_mean) / feature_std
angles = torch.pi + torch.atan(normalized_features) * (torch.pi / 2)
```

**Circuit Implementation**:
```
for i in range(num_qubits):
    qml.RY(angles[i], wires=i)
```

### 9.2 Enhanced Angle Encoding

**Method name: `enhanced_angle`**

This method preserves more of the original data structure by using all three rotation gates (RX, RY, RZ) per qubit:

1. **Feature Preprocessing**:
   - Normalize features
   - Calculate features per qubit (3 features per qubit: one for each rotation)
   
2. **Dimension Handling**:
   - When features < needed: Create mapping matrix to expand dimensions
   - When features > needed:
     - Method 1: Select top features
     - Method 2: Use SVD for dimensionality reduction
     - Method 3: Fallback to strided selection if SVD fails
   
3. **Rotation Angle Generation**:
   - RX angles scaled to [-π, π] range
   - RY angles scaled to [-π, π] range
   - RZ angles scaled to [-2π, 2π] range

**Code Example**:
```python
# Call the enhanced encoding function
rx_angles, ry_angles, rz_angles = angle_encoding_enhanced(features, num_qubits)
```

**Circuit Implementation**:
```
for i in range(num_qubits):
    qml.RX(rx_angles[i], wires=i)
    qml.RY(ry_angles[i], wires=i)
    qml.RZ(rz_angles[i], wires=i)
```

### 9.3 Encoding Comparison and Selection

| Aspect | Standard Encoding | Enhanced Encoding |
|--------|-------------------|-------------------|
| Data Preservation | Moderate | High |
| Circuit Complexity | Low | Medium |
| Parameter Count | num_qubits | 3 × num_qubits |
| Backprop Compatibility | Full | Full |
| Use Case | Simple problems, limited qubits | Complex data, pattern recognition |

To select an encoding method, set the `encoding_method` parameter during model initialization or use `set_encoding_method()` to change it dynamically.

## 10. Training and Optimization

The QNN model is designed to be trained using standard PyTorch optimization techniques:

### 10.1 Loss Functions

The hybrid model outputs log probabilities, making it compatible with standard classification loss functions:

```python
criterion = nn.CrossEntropyLoss()
```

### 10.2 Optimizer Configuration

Due to the hybrid nature, consider using different learning rates for classical and quantum parameters:

```python
# Separate parameter groups
classical_params = []
quantum_params = []

for name, param in qnn_model.named_parameters():
    if name in ['theta', 'phi', 'raw_noise']:
        quantum_params.append(param)
    else:
        classical_params.append(param)

optimizer = torch.optim.Adam([
    {'params': classical_params, 'lr': 1e-3},
    {'params': quantum_params, 'lr': 1e-2}
])
```

### 10.3 Training Loop

```python
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
```

### 10.4 Differentiation Methods

The QNN supports two differentiation methods:

1. **Backpropagation** (`use_backprop=True`):
   - Faster training
   - Incompatible with shots and noise models
   - Uses Pennylane's quantum gradient computation

2. **Parameter-shift** (`use_backprop=False`):
   - Compatible with all quantum features
   - Slower but more physically realistic
   - Required for shot-based or noisy simulations

## 11. Quantum Circuit Design

### 11.1 Circuit Structure

The quantum circuit follows this general structure:

1. **Initial Encoding Layer**: Encode classical data into quantum states
2. **Repeated Variational Blocks** (controlled by `circuit_depth`):
   - Optional noise rotations for robustness
   - Entanglement layer (configurable pattern)
   - Parameterized rotation layer with trainable parameters
   - Re-application of data encoding with decay
   - Optional noise operations (physically realistic quantum noise)
3. **Measurement**: Calculate probabilities for all computational basis states

### 11.2 Parameter Design

Quantum parameters can be configured in two modes:

1. **Scalar Mode** (`param_mode='scalar'`):
   - Single trainable parameter for all qubits
   - More constrained model, fewer parameters

2. **Vector Mode** (`param_mode='vector'`):
   - Individual parameter for each qubit 
   - More expressive model, more parameters to train
   - `num_params` controls the vector size

### 11.3 Entanglement Architecture

The choice of entanglement pattern significantly impacts model expressivity and training:

1. **No Entanglement**: Simplest model, no quantum advantage
2. **Linear Entanglement**: Good balance of expressivity and circuit depth
3. **Full Entanglement**: Most expressive, but can be harder to train
4. **Star Entanglement**: Efficient for certain problem structures

## 12. Adversarial Defense Mechanism

The QNN implements adversarial defense capabilities:

### 12.1 Quantum Noise as Defense

Quantum noise naturally creates uncertainty in predictions, making the model more robust against adversarial attacks:

```python
# Noise strength is trainable
noise_strength = constrained_noise_activation(self.raw_noise).item()
```

### 12.2 Input Randomization

When `adversarial_defense` is enabled and `defense_mode=True`:

```python
if defense_mode and self.adversarial_defense:
    noise_level = constrained_noise_activation(self.raw_noise).item() * 2
    noise_level = min(0.1, noise_level)  # Cap at 0.1
    input_noise = torch.randn_like(input_data) * noise_level
    input_data = input_data + input_noise
```

### 12.3 Output Smoothing

```python
# Apply smoothing to quantum outputs
kernel_size = 3
if quantum_output_probabilities.shape[1] >= kernel_size:
    padding = (kernel_size - 1) // 2
    smoothed_quantum = F.avg_pool1d(
        quantum_output_probabilities.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    ).squeeze(1)
```

## 13. Examples and Use Cases

### 13.1 Image Classification

```python
# Create a QNN model for MNIST classification
qnn_model = get_qnn_model(
    cnn_model=cnn_model,
    device=device,
    output_dim=10,
    circuit_depth=2,
    encoding_method='enhanced_angle',
    entanglement_type='linear_entanglement_ansatz'
)

# Train and evaluate
train_model(qnn_model, train_loader, optimizer, criterion, epochs=5)
accuracy = evaluate_model(qnn_model, test_loader)
```

### 13.2 Adversarial Testing

```python
# Create adversarial examples
adversarial_examples = generate_fgsm_attack(
    model=qnn_model,
    images=test_images,
    labels=test_labels,
    epsilon=0.1
)

# Test standard accuracy
standard_accuracy = evaluate_model(qnn_model, test_loader)

# Test adversarial accuracy with defense mode enabled
adversarial_accuracy = evaluate_adversarial_examples(
    qnn_model, 
    adversarial_examples,
    test_labels,
    defense_mode=True
)

print(f"Standard accuracy: {standard_accuracy:.2f}%")
print(f"Adversarial accuracy: {adversarial_accuracy:.2f}%")
```

## 14. Future Directions

### 14.1 Hardware Compatibility

The current QNN implementation is designed for simulators but can be extended to real quantum hardware:

```python
# Example of adapting to real quantum hardware
dev_hardware = qml.device('qiskit.ibmq', wires=num_qubits, backend='ibmq_manila')
```

### 14.2 Advanced Quantum Features

Potential extensions include:

1. **Quantum Kernels**: Using quantum circuits as kernel functions
2. **Quantum Embedding Layers**: Learning optimal encodings 
3. **Quantum Error Mitigation**: Adding error correction techniques

## 15. Conclusion

The QNN implementation provides a sophisticated integration of classical and quantum computing. By combining CNNs with quantum circuits, it offers potential advantages in certain machine learning tasks while providing flexibility in model design and configuration.

The modular architecture allows for experimentation with different quantum circuit designs, encoding methods, and entanglement patterns. The dynamic weighting system intelligently combines classical and quantum outputs to maximize model performance.

As quantum hardware continues to improve, this hybrid approach provides a pathway to leverage quantum advantages while maintaining the proven capabilities of classical deep learning.
