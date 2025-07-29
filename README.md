# Energy Demand Forecasting

This project implements time series forecasting for energy demand using machine learning techniques.

## Project Structure

```
energy_demand_forecasting/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Preprocessed data
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for EDA
├── outputs/           # Model outputs, plots, etc.
├── src/               # Source code
│   ├── load_and_preprocess.py
│   ├── train_model.py
│   └── predict.py
├── main.py            # Main execution script
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd energy_demand_forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place your time series data in `data/raw/`
   - The expected format is CSV with a column named `DE_load_actual_entsoe_transparency`

## Usage

### Training the Model

Run the main script to train the model:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train an XGBoost model with Optuna hyperparameter optimization
- Save the trained model and optimization study

### Making Predictions

Use the prediction script:

```bash
python src/predict.py
```

## Data

The project expects energy demand time series data with the following characteristics:
- 15-minute intervals
- Target variable: `DE_load_actual_entsoe_transparency`
- Additional features for forecasting

## Model

- **Algorithm:** XGBoost
- **Hyperparameter Optimization:** Optuna
- **Evaluation:** Time series cross-validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here] 