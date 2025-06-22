
# Neural-Network-Approximation

This project explores the theoretical and practical limits of approximating **1D and 2D convex and non-convex functions** using **ReLU-based neural networks**. Inspired by recent theoretical work, we aim to evaluate and compare different architectures—including traditional dense models and Axon-style networks—for function approximation tasks.

---

## 🧠 Project Goals

- Approximate **1D and 2D functions** \( f : [0,1] \rightarrow \mathbb{R} \) and \( f : [0,1]^2 \rightarrow \mathbb{R} \) using ReLU networks.
- Focus on both **convex** and **non-convex** functions.
- Compare traditional feedforward architectures with **Axon-style networks** (greedy layer growing).
- Evaluate performance and convergence relative to **theoretical approximation bounds** (Liu & Liang 2021, Fokina & Oseledets 2023).
- Understand the practical limitations of classical neural regressors when faced with complex geometries or higher dimensions.

---

## 📁 Project Structure

```
project/
├── data/           # Data generators for 1D and 2D test functions
├── model/          # Network definitions (ReLU, piecewise-linear, Axon-style, etc.)
├── utils/          # Training loop, evaluation, and visualization tools
├── tests/          # Unit tests for core components
├── experiments/    # Jupyter notebooks with results and analysis
├── .gitlab-ci.yml  # GitLab CI/CD configuration
├── setup.py        # Project environment setup
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation & Setup

Create and activate the Python virtual environment:

```bash
python setup.py
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.bat     # Windows
```

Then launch Jupyter notebooks for experiments:

```bash
jupyter notebook
```

---

## ✅ Running Unit Tests

Ensure everything works correctly with:

```bash
pytest tests/
```

---

## 📌 CI/CD

GitLab CI/CD automatically runs all tests on push and merge requests via `.gitlab-ci.yml`.

---

## 📚 References

1. **Liu, B., Liang, Y.** (2021). *Optimal Function Approximation with ReLU Neural Networks*, Neurocomputing, Vol. 435.  
2. **Fokina, D., Oseledets, I.** (2023). *Growing Axons: Greedy Learning of Neural Networks with Application to Function Approximation*, Russian Journal of Numerical Analysis and Mathematical Modelling.  
3. GitHub (Axon algorithm implementation): [https://github.com/dashafok/axon-approximation](https://github.com/dashafok/axon-approximation)
