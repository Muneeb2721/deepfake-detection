🧠 DeepFake Detection Using Swin Transformer

This project implements a DeepFake vs Real image classification pipeline using the Swin Transformer model and Hugging Face's `Trainer` API. The dataset is processed, balanced using oversampling, and augmented for training. Evaluation metrics and visualizations are also included.

📁 Table of Contents

- [Cell1: Install Required Packages](#cell1-install-required-packages)
- [Cell2: Load Dataset & Apply Oversampling](#cell2-load-dataset--apply-oversampling)
- [Cell3: Label Mapping + Split Dataset](#cell3-label-mapping--split-dataset)
- [Cell4: Load Swin Transformer + Transforms](#cell4-load-swin-transformer--transforms)
- [Cell5: Collator + Metrics + CSV/Plot Export](#cell5-collator--metrics--csvplot-export)
- [Cell6: Training Arguments + Trainer Setup](#cell6-training-arguments--trainer-setup)
- [Cell7: Train and Evaluate](#cell7-train-and-evaluate)
- [📊 Visual Outputs](#-visual-outputs)
- [📁 File Structure](#-file-structure)
- [🚀 Deployment (optional)](#-deployment-optional)
- [📌 References](#-references)

Cell1: Install Required Packages

Installs the necessary Python packages for dataset handling, model loading, training, evaluation, and image transformations.

Cell2: Load Dataset & Apply Oversampling

- Loads DeepFake vs Real images dataset from Hugging Face Datasets.
- Applies RandomOverSampler from imbalanced-learn to balance class distribution.

Cell3: Label Mapping + Split Dataset

- Maps string labels to numeric values.
- Casts the label column for model compatibility.
- Splits the dataset into 60% training and 40% testing sets.

Cell4: Load Swin Transformer + Transforms

- Loads the `microsoft/swin-base-patch4-window7-224` pre-trained model.
- Applies augmentations like random rotation and sharpness adjustment to training images.

Cell5: Collator + Metrics + CSV/Plot Export

- Defines a custom collator for the Trainer.
- Calculates accuracy, precision, recall, and F1-score.
- Generates:
  - `classification_report.csv`
  - `confusion_matrix.csv`
  - `confusion_matrix_plot.png`

Cell6: Training Arguments + Trainer Setup

- Configures training parameters:
  - Batch size
  - Epochs
  - Evaluation and saving strategies
- Loads the best model based on validation F1-score.

Cell7: Train and Evaluate

- Trains the model for 2 epochs.
- Automatically resumes if interrupted.
- Evaluates and saves metrics post-training.

📊 Visual Outputs

After training, the following plots are generated:

- Confusion Matrix Plot
- Loss vs Epochs
- Accuracy vs Epochs
- Loss vs Steps
- Accuracy vs Steps
- Loss vs Wall-Time
- Accuracy vs Wall-Time

📁 File Structure
🔹 classification_report.csv
🔹 confusion_matrix.csv
🔹 confusion_matrix_plot.png
🔹 training_args.json
🔹 output_dir/
🔹 ├── checkpoint-*/  
🔹 └── config.json
🔹 README.md
🔹 *.ipynb
```

📌 References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [imbalanced-learn](https://imbalanced-learn.org/)

