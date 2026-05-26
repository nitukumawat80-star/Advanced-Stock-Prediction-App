# Advanced Stock Prediction App

An upgraded Streamlit app with:

- User login and registration
- Personalized recent search history
- Watchlist management
- ML-based stock prediction (Random Forest)
- Top Indian company shares dashboard
- Multi-share performance and connection graphs
- Share dependency analysis (beta, correlation, R2)
- Built-in stock chatbot assistant

## Quick Start

1. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

3. Run the app:

```powershell
py -m streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Main Pages

- `Prediction`: run price prediction, view OHLC/volume/technical graphs, save results.
- `Top Indian Shares`: ranked Indian large-cap snapshot with comparison graphs.
- `Connections and Dependency`: correlation matrix and one-share-to-another dependency analysis.
- `Recent Searches`: per-user prediction history and watchlist snapshots.
- `Assistant`: chat interface for ticker and app-help questions.

## Notes

- Local app data (users, history, watchlist) is stored in the `data/` folder.
- This project is for learning and experimentation only.
- Predictions are model estimates, not financial advice.
