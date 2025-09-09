# CrackingCandyCrush

**AI system to detect game state given a screenshot and select the optimal move in Candy Crush Saga.**

---

## Overview
This project implements a complete pipeline for **automated gameplay in Candy Crush Saga**.  
It combines:
- **YOLOv8** for real-time object detection of candies and board elements,  
- **ResNet (18/34)** for candy classification and objective detection,  
- **OCR (Tesseract + EasyOCR)** for reading score and level objectives,  
- **Monte Carlo Tree Search (MCTS)** for optimal move selection and strategy planning.  

The system takes a raw screenshot of the game board, extracts the full game state, and outputs the next best move to maximize performance.  

This repository was developed as part of an academic research project on applying **computer vision and search-based decision-making** to real-world match-3 puzzle games.  

---

## Features
- Real-time **object detection** of candies and blockers using YOLOv8.  
- **ResNet-based classification** for detailed candy types and objectives.  
- **OCR pipeline** for reading in-game text (e.g., moves left, goals).  
- **Move simulation engine** with stochastic behavior modeling.  
- **MCTS-based solver** for optimal move selection under uncertainty.  
- Modular design for extending to future Candy Crush Saga mechanics.  

---

## Installation
### Prerequisites
- Python 3.9+  
- CUDA-capable GPU (recommended)  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)  

### Install dependencies
```bash
pip install -r requirements.txt
```
---
## Usage
To Train computer vision models
```bash
python candy_vision_train.py
```
Test train models
```bash
python candy_testing.py
```
To run on Candy Crush Game
```bash
python main.py
```
---
## Dataset
- Candy Dataset: For classification (individual candy images) in "candy_dataset" folder.
- YOLO Dataset: For object detection (candies, blockers, objectives) in "data/images/train", "data/images/label" contains label of grids, labelling done using labelImg
- Testing Images: For benchmarking (data/test/).
---
## Results
- Object Detection Module(YOLOv8) succeeded on all 30 tests.
- Object Classification Module(ResNet) succeeded on all 30 tests and training dataset.
- MCTS and other method solvers achieving 97.5% completion rate.
## Acknowledgments
- Ultralytics YOLO
- PyTorch
- EasyOCR
- TesseractOCR
- ResNet
- EfficientNet
