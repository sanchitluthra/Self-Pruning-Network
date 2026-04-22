
# Self-Pruning Neural Network
**Sanchit Luthra | 102303486 | Tredence AI Intern Case Study**

## 🛠️ My Solution: How the Pruning Works
Every weight in the network is paired with a learnable **gate score**. Instead of pruning after training, the network learns which connections to cut *during* the training process:

* **Sigmoid Activation:** We apply `sigmoid(gate_score)` to scale the value between 0 and 1.
  * Gates pushed near **0.0** → The weight is pruned.
  * Gates pushed near **1.0** → The weight survives.
* **Custom Sparsity Loss:** `Loss = CrossEntropy + λ ∑(gates)`. The λ (L1 penalty) acts as a constant downward pressure, relentlessly forcing non-essential gates to zero.
* **Warm-up Phase (Epochs 1-3):** The network trains normally with zero pruning pressure (`λ = 0`). It  learn basic image features before it starts pruning weight.
* **Dynamic Annealing (Epoch 4+):** The pruning  (`λ`) slowly ramps up to the target value. This  approach allows the network to gracefully adapt as its weights are deleted.

## 📊 Results & Trade-off Analysis
The model was trained for 10 epochs. To ensure a thorough evaluation, I conducted an ablation study comparing a positive gate initialization against a neutral gate initialization.

### Baseline: Gate Scores Initialized at 1.0 ($\sigma \approx 0.73$)
Starting with highly active gates provides a strong initial signal but requires more L1 pressure to push weak connections down to zero.

| Pruning Strictness | Target λ | Final Test Accuracy | Sparsity (Weights Pruned) |
| :--- | :--- | :--- | :--- |
| **Low** | 2e-06 | 56.18% | 49.48% |
| **Medium** | 8e-06 | 57.05% | 76.93% |
| **High** | 2e-05 | 56.94% | 88.15% |

### Optimized: Gate Scores Initialized at 0.0 ($\sigma = 0.50$)
Starting the gates at a perfectly neutral `0.0` gives the network an unbiased starting state. The gradients can naturally push critical connections up and weak connections down, accelerating the pruning process.

| Pruning Strictness | Target λ | Final Test Accuracy | Sparsity (Weights Pruned) |
| :--- | :--- | :--- | :--- |
| **Low** | 2e-06 | 57.00% | 49.77% |
| **Medium** | 8e-06 | 57.42% | 77.03% |
| **High** | 2e-05 | **57.45%** | **88.21%** |

> **Note:** The **High λ** configuration with the **0.0 initialization** proved to be the best overall model. It successfully severed over 88% of the network's parameters while actually achieving the highest classification accuracy of the entire experiment.

## 📈 Visual Proof
### Baseline: Gates Initialized at 1.0
<img width="722" height="468" alt="Screenshot 2026-04-22 at 10 09 52 PM" src="https://github.com/user-attachments/assets/e7303393-ae56-4f18-9543-73dc58af50f2" />


### Optimized: Gates Initialized at 0.0
<img width="720" height="470" alt="Screenshot 2026-04-22 at 10 10 16 PM" src="https://github.com/user-attachments/assets/97770ab4-895b-466b-85ed-851c90a85778" />
