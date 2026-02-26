🧬 LactoSuite AI: Professional Lactic Acid Yield Predictor

LactoSuite AI is an advanced expert system leveraging high-performance Machine Learning (Stacking Ensemble) to predict Lactic Acid production yields during fermentation. The system features a robust synchronization between a Python Flask backend and a modern, interactive React dashboard.

🚀 Key Features

AI-Powered Prediction: Utilizing a sophisticated Stacking Ensemble model with high precision ($R^2 = 84.66\%$).

16-Feature Intelligence: Analyzes comprehensive fermentation parameters across three dimensions: Medium Composition, Fermentation Mode, and Environmental Kinetics.

Interactive Dashboard: A real-time UI built with React and Tailwind CSS for instant yield estimation and parameter adjustment.

Session Intelligence: Built-in history logging system with the ability to export simulation data as professional CSV reports.

🛠️ Project Structure

lactic_acid_api_server.py: Flask-based API for model serving and real-time feature engineering.

lactic_acid_local_sync.html: The primary user interface and simulation dashboard.

lactic_acid_best_model.pkl: The "AI Brain" - a serialized trained model pipeline (Polynomial + PowerTransform).

lactic_acid_analysis.py: Master script for exploratory data analysis (EDA) and model training.

💻 Installation & Usage

Install Dependencies:

pip install -r requirements.txt


Launch the API Server:

python lactic_acid_api_server.py


Access the Dashboard:
Simply open lactic_acid_local_sync.html in any modern web browser.

📊 Model Performance Metrics (Stacking Ensemble)

The current model has been verified against experimental datasets with the following results:

R-squared ($R^2$): 0.8466

MAE (Mean Absolute Error): 12.51 g/L

RMSE (Root Mean Square Error): 18.06 g/L

MAPE (Mean Absolute Percentage Error): 49.74 %

System Architect: Krittima Anekthanakul,Ph.D.

Optimizing Biotechnology through Artificial Intelligence