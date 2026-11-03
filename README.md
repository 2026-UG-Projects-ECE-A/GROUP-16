README
KiMORE Post-Stroke Rehabilitation 3D Skeleton Motion Forecasting & Assessment
A Deep Learning framework built in PyTorch to model, simulate, and assess post-stroke rehabilitation exercises using 3D skeleton animation data from the KiMORE (Kinect Motion Recovery) dataset.

📌 Project Overview
This project processes spatial-temporal 3D body skeleton joint data to perform two primary tasks:

Seq2Seq 3D Motion Trajectory Forecasting (RehabSeq2SeqLSTM): Predicts future 3D joint movements given historical motion frames.
Bi-LSTM Motor Quality Assessment (RehabBiLSTMEvaluator): Evaluates movement execution quality and classifies subjects into Healthy Controls (G001) vs. Motor-Impaired / Post-Stroke Patients (G002–G005), achieving 94% classification accuracy.
📊 Dataset Structure
The raw dataset in kimore/skeleton contains 2,806 .skeleton files representing 3D skeleton sequences recorded via Kinect v2 (25 body joints per frame with 
x
,
y
,
z
 coordinates).

Subjects: 27 subjects (Healthy Group G001, Impaired Groups G002–G005)
Exercises: 8 rehabilitation exercises (E002 through E009)
Pre-processing: Coordinates are normalized relative to Joint 0 (pelvis/spine base) for spatial origin invariance.
📂 Project Architecture
Stroke_DL/
├── dataset.py                # Raw .skeleton parser & sliding window data loader
├── models.py                 # PyTorch Seq2Seq LSTM & Bi-LSTM architectures
├── train.py                  # Training pipeline for trajectory forecasting model
├── simulate.py               # Trajectory simulation evaluation & joint inspection script
├── eval_confusion_matrix.py  # Bi-LSTM training, evaluation, and heatmap confusion matrix generator
├── requirements.txt          # Python dependencies
├── LSTM_REHABILITATION_GUIDE.md # Technical feasibility & reference design guide
├── confusion_matrix.png      # Generated confusion matrix plot
├── lstm_rehab_simulator.pth  # Trained Seq2Seq LSTM model weights
└── bilstm_motor_evaluator.pth# Trained Bi-LSTM classification model weights
🚀 Getting Started & Execution Guide
1. Environment Setup
Create and activate a Python virtual environment, then install dependencies:

# Create virtual environment
python -m venv venv

# Install dependencies (CPU or GPU supported)
.\venv\Scripts\pip.exe install -r requirements.txt
GPU Acceleration: If an NVIDIA GPU is present, PyTorch will automatically utilize CUDA acceleration.

2. Verify Dataset Parsing
Run dataset.py to inspect the KiMORE skeleton files and verify sliding-window feature extraction:

.\venv\Scripts\python.exe dataset.py
Output sample:

Dataset loaded successfully. Total sliding-window samples: 7111
Sample Input Shape: torch.Size([59, 75]), Target Shape: torch.Size([59, 75])
3. Train Seq2Seq Motion Simulator
Train the RehabSeq2SeqLSTM network to forecast 3D joint trajectories:

# Default: 15 epochs
.\venv\Scripts\python.exe train.py kimore\skeleton 15
4. Run Trajectory Forecasting Simulation
Evaluate the trained trajectory forecasting model on test sequences:

.\venv\Scripts\python.exe simulate.py
Results:

Trajectory Prediction MSE: 0.000147
Compares predicted vs. ground truth 
(
x
,
y
,
z
)
 coordinates for pelvis and head joints across sequence frames.
5. Evaluate Motor Function Assessor & Generate Confusion Matrix
Train the RehabBiLSTMEvaluator and generate the classification confusion matrix (confusion_matrix.png):

.\venv\Scripts\python.exe eval_confusion_matrix.py 10
Results Summary:

Overall Accuracy: 94%
Impaired (G002–G005) Recall: 97%
Healthy (G001) Precision: 90%
📈 Model Performance & Metrics
Model	Task	Key Metric	Result
RehabSeq2SeqLSTM	3D Trajectory Forecasting	Mean Squared Error (MSE)	0.000147
RehabBiLSTMEvaluator	Healthy vs. Impaired Assessment	Test Accuracy / F1-Score	94% / 0.94
📜 License & Acknowledgments
Dataset: KiMORE (Kinect Motion Recovery) Dataset.
Framework: PyTorch, Scikit-Learn, Seaborn, Matplotlib.
