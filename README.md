# ✈️ Flight Price Prediction — Feature Engineering & ML

An end-to-end machine learning project on Indian domestic flight prices, covering datetime feature extraction, string parsing, ordinal encoding, and a tuned Random Forest Regressor.

---

## 📂 Dataset

**Source:** `flight_price.xlsx` (local)

| Property | Value |
|---|---|
| Rows | 10,683 |
| Columns (raw) | 11 |
| Columns (final model input) | 15 |
| Target Variable | `Price` (INR) |

**Raw Features:** Airline, Date_of_Journey, Source, Destination, Route, Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info, Price

---

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — encoding, model training, evaluation

---

## 🔧 Feature Engineering & Data Cleaning

This project is heavily feature-engineering focused. Almost every raw column required transformation before it could be used.

### Step-by-Step Transformations

| Column | Raw Format | Transformation |
|---|---|---|
| `Date_of_Journey` | `"24/03/2019"` | Split into `Date`, `Month`, `Year` as integers; original column dropped |
| `Arrival_Time` | `"01:10 22 Mar"` (some with date suffix) | Stripped trailing date info using `.split(' ')[0]`; extracted `Arrival_Hour`, `Arrival_Minutes` |
| `Dep_Time` | `"22:20"` | Split on `:` to extract `Departure_Hour`, `Departure_Minutes` |
| `Duration` | `"2h 50m"`, `"19h"` (no minutes) | Parsed hours and minutes separately into `Duration_Hour`, `Duration_Minutes`; missing minutes defaulted to `0`; original column dropped |
| `Total_Stops` | `"non-stop"`, `"1 stop"`, `"2 stops"`, `NaN` | Mapped to integers `{non-stop:0, 1 stop:1, 2 stops:2, 3 stops:3, 4 stops:4}`; NaN filled with mode (`1 stop` → `1`) |
| `Route` | `"BLR → DEL"` | Identified 1 null row (index 9039, Air India Delhi→Cochin) — dropped along with column |
| `Airline`, `Source`, `Destination`, `Additional_Info` | String categories | Encoded using `OrdinalEncoder` from scikit-learn; results concatenated back as float columns |

### Null Summary (raw data)

| Column | Nulls |
|---|---|
| `Route` | 1 |
| `Total_Stops` | 1 |
| All others | 0 |

Both nulls belonged to the same corrupt row (index 9039), which was dropped.

---

## 🧠 Model

### Feature Set

```python
X = ['Airline', 'Total_Stops', 'Arrival_Hour', 'Arrival_Minutes',
     'Departure_Hour', 'Departure_Minutes', 'Duration_Hour', 'Duration_Minutes']
y = 'Price'
```

> Note: `Source`, `Destination`, `Additional_Info`, `Date`, `Month`, `Year` are in `final_df` but were excluded from the training features in this iteration.

### Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

### Algorithm — Random Forest Regressor

Two configurations were tried; the final tuned model:

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

### Results

| Metric | Value |
|---|---|
| MAE | ₹1,713.72 |
| RMSE | ₹2,993.67 |
| R² Score | 0.577 |

The model explains ~58% of price variance using only 8 features. Adding `Source`, `Destination`, and date features is a clear next step to improve R².

---

## 📁 Project Structure

```
📦 flight-price-prediction/
├── FlightPrice.ipynb       # Main notebook
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/flight-price-prediction.git
cd flight-price-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

# Place the dataset
# Add flight_price.xlsx to the project root (or update the path in cell 2)

# Launch notebook
jupyter notebook FlightPrice.ipynb
```

> `openpyxl` is required for `pd.read_excel()` to work.

---

## 📌 Key Takeaways

- Datetime and time fields in raw string format need careful parsing — especially when arrival times carry next-day date suffixes.
- `Duration` without a minutes component (e.g., `"19h"`) must be handled explicitly or it throws errors on split.
- `OrdinalEncoder` was chosen over one-hot encoding to keep dimensionality low; a more principled approach for non-ordinal categories like `Airline` would be target encoding.
- R² of 0.577 with only 8 features leaves room to improve — including `Source`, `Destination`, and date features in the model is the most direct path forward.
