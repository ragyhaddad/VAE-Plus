# CyclicalVAE

A state-of-the-art implementation of a Variational Autoencoder (VAE) leveraging **cyclical annealing schedules** to mitigate posterior collapse, a common issue in VAEs. This repository provides a modular, easy-to-use, and high-performance implementation designed to outperform publicly available VAE models.

---

## 🚀 Features

- **Cyclical Annealing**: Mitigates posterior collapse by dynamically adjusting the KL divergence weight (β) during training.
- **Customizable Architecture**: Easily modify encoder/decoder layers and latent space dimensions.
- **Scalable**: Supports both CPU and GPU training for datasets of varying sizes.
- **Visualization Tools**: Built-in utilities for visualizing latent space and reconstruction quality.
- **Benchmarking**: Compare with standard VAE implementations using pre-configured benchmarks.

---

## 📖 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [How It Works](#how-it-works)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/CyclicalVAE.git
cd vae-cyc
pip install -r requirements.txt
