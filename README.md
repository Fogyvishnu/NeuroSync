# NeuroSync - EEG Brain-Computer Interface System

<div align="center">

![NeuroSync Logo](docs/images/logo.png)
*Syncing Minds with Machines*

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/NeuroSync.svg?style=social)](https://github.com/yourusername/NeuroSync)

**Advanced MATLAB-based EEG signal processing and Brain-Computer Interface implementation**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Examples](#-examples) • [Documentation](#-documentation)

</div>

## Overview

NeuroSync is a comprehensive EEG Brain-Computer Interface (BCI) system implemented in MATLAB. It provides a complete pipeline for EEG signal processing, feature extraction, machine learning classification, and real-time BCI applications.

### Key Capabilities
- **Complete EEG Processing Pipeline** from raw signals to classification
- **Real-time BCI Interface** with visual feedback
- **Multiple ML Models** (SVM, Random Forest, LDA, Neural Networks)
- **Comprehensive Visualization Dashboard**
- **Simulated & Real EEG Data Support**

## Features

### Signal Processing
- ✅ Bandpass & notch filtering
- ✅ Artifact detection & removal (eye blinks, muscle artifacts)
- ✅ Common Average Reference (CAR)
- ✅ ICA for component analysis
- ✅ Time-frequency analysis

### Feature Extraction
- ✅ Time-domain features (mean, variance, skewness, kurtosis)
- ✅ Frequency-domain features (band powers, spectral edge)
- ✅ Hjorth parameters
- ✅ Connectivity features
- ✅ Custom feature sets

### Machine Learning
- ✅ Support Vector Machines (SVM)
- ✅ Random Forest
- ✅ Linear Discriminant Analysis (LDA)
- ✅ Neural Networks
- ✅ Ensemble methods
- ✅ Cross-validation & hyperparameter tuning

### Visualization
- ✅ Real-time EEG display
- ✅ Topographical brain maps
- ✅ Time-frequency representations
- ✅ Classification results dashboard
- ✅ Interactive BCI interface

## Installation

### Prerequisites
- MATLAB R2020b or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- (Optional) Deep Learning Toolbox
- (Optional) EEGLAB (for advanced ICA)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Fogyvishnu/NeuroSync.git
cd NeuroSync
