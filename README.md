# Sales & Demand Forecasting
### Future Interns – Machine Learning Task 1

A beginner-level ML project that forecasts monthly sales using the Superstore dataset.

---

## Project Structure

```
sales_forecasting_project/
│── app.py                   # Streamlit web app
│── model.py                 # Model training and forecasting logic
│── data_preprocessing.py    # Data loading and cleaning
│── Sales_Forecasting.ipynb  # Jupyter notebook (step-by-step walkthrough)
│── requirements.txt
│── README.md
```

---

## What This Project Does

- Loads and cleans the Superstore Sales CSV
- Aggregates daily sales into monthly totals
- Engineers time-based features (trend, seasonality)
- Trains a Random Forest model to predict sales
- Forecasts sales for the next 12 months
- Shows results in an interactive Streamlit app

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`  
Upload the `Sample_-_Superstore.csv` file from the sidebar.

### 3. Or open the Jupyter Notebook
```bash
jupyter notebook Sales_Forecasting.ipynb
```

Place `Sample_-_Superstore.csv` in the same folder before running.

---

## Dataset

**Superstore Sales Dataset**  
Source: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

---

## Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Features | Month trend, Year, Month, Sine/Cosine seasonality |
| Metrics | MAE, RMSE |
| Forecast horizon | Configurable (3–24 months) |

---

## Business Interpretation

The forecast helps businesses:
- **Plan inventory** before high-demand months
- **Manage cash flow** by knowing expected revenue
- **Schedule staffing** based on busy/slow seasons
- **Run promotions** during predicted low-sales periods

---

*Built for Future Interns ML Task 1*
