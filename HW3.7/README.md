# homework 3.7

Artificial neural networks: mathematical formulation, activation functions, cost functions, backpropagation, and the training algorithm.

[view the notebook](HW3_7_neural_networks.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aaxhirrr/mat-422-fall-2026/blob/AASHIR/hw3-7/HW3.7/HW3_7_neural_networks.ipynb)

Open in Colab and choose **Runtime > Run all**. The notebook uses only NumPy and Matplotlib, with four explicitly listed XOR inputs and fixed starting weights. There are no outside data files or package installs.

The notebook has 8 code cells, 3 figures, and saved outputs. All 14 checks at the end should print `True`.

| assigned topic | example |
|---|---|
| 3.7.1 mathematical formulation | a 2-input, 4-hidden-unit, 1-output network, array shapes, and a hand-checked forward calculation |
| 3.7.2 activation functions | sigmoid, tanh, and ReLU, their derivatives, and the collapse of two affine layers |
| 3.7.3 cost function | squared error and binary cross-entropy, with a stable logit-based calculation |
| 3.7.4 backpropagation | chain-rule equations, explicit gradients, and finite-difference checks of all 17 parameters |
| 3.7.5 backpropagation algorithm | a full-batch gradient-descent loop, loss curve, XOR predictions, and a decision-boundary plot |

The four XOR points are the complete toy training example. The reported accuracy is on those four points, not an estimate of performance on unseen data.
