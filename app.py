import datetime as dt
import hashlib
import json
import math
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATA_DIR = Path("data")
USERS_FILE = DATA_DIR / "users.json"
HISTORY_FILE = DATA_DIR / "search_history.json"
WATCHLIST_FILE = DATA_DIR / "watchlists.json"
MAX_RECENT_SEARCHES = 15

TOP_INDIAN_COMPANIES = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank": "AXISBANK.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Sun Pharma": "SUNPHARMA.NS",
}

INDIAN_SECTOR_HINTS = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT Services",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "INFY.NS": "IT Services",
    "BHARTIARTL.NS": "Telecom",
    "SBIN.NS": "Banking",
    "LT.NS": "Infrastructure",
    "ITC.NS": "Consumer",
    "HINDUNILVR.NS": "Consumer",
    "KOTAKBANK.NS": "Banking",
    "BAJFINANCE.NS": "Finance",
    "AXISBANK.NS": "Banking",
    "M&M.NS": "Automobile",
    "SUNPHARMA.NS": "Pharma",
}

FEATURE_COLUMNS = [
    "close",
    "return_1d",
    "return_5d",
    "sma_5",
    "sma_20",
    "ema_10",
    "rsi_14",
    "volatility_20",
    "volume_change",
]


@dataclass
class PredictionResult:
    latest_close: float
    predicted_close: float
    lower_bound: float
    upper_bound: float
    expected_move_pct: float
    mae: float
    r2: float
    chart_df: pd.DataFrame
    test_df: pd.DataFrame


def ensure_storage_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (USERS_FILE, HISTORY_FILE, WATCHLIST_FILE):
        if not path.exists():
            path.write_text("{}", encoding="utf-8")


def read_json_file(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_json_file(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def password_digest(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> Tuple[bool, str]:
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return False, "Username must be 3-30 chars using letters, numbers, or underscore."
    if len(password) < 6:
        return False, "Password must have at least 6 characters."

    users = read_json_file(USERS_FILE)
    if username in users:
        return False, "Username already exists."

    salt = secrets.token_hex(16)
    hashed = password_digest(password, salt)
    users[username] = {
        "password_hash": f"{salt}${hashed}",
        "created_at": dt.datetime.utcnow().isoformat(),
    }
    write_json_file(USERS_FILE, users)
    return True, "Registration completed. You can login now."


def authenticate_user(username: str, password: str) -> bool:
    users = read_json_file(USERS_FILE)
    user = users.get(username)
    if not user:
        return False

    password_hash = user.get("password_hash", "")
    if "$" not in password_hash:
        return False
    salt, stored_hash = password_hash.split("$", 1)
    return password_digest(password, salt) == stored_hash


def get_search_history(username: str) -> List[Dict]:
    history_data = read_json_file(HISTORY_FILE)
    items = history_data.get(username, [])
    return items if isinstance(items, list) else []


def add_search_history(username: str, item: Dict) -> None:
    history_data = read_json_file(HISTORY_FILE)
    user_items = history_data.get(username, [])
    if not isinstance(user_items, list):
        user_items = []
    user_items.insert(0, item)
    history_data[username] = user_items[:MAX_RECENT_SEARCHES]
    write_json_file(HISTORY_FILE, history_data)


def clear_search_history(username: str) -> None:
    history_data = read_json_file(HISTORY_FILE)
    history_data[username] = []
    write_json_file(HISTORY_FILE, history_data)


def get_watchlist(username: str) -> List[str]:
    watchlists = read_json_file(WATCHLIST_FILE)
    user_watchlist = watchlists.get(username, [])
    if not isinstance(user_watchlist, list):
        return []
    return sorted(set(user_watchlist))


def add_to_watchlist(username: str, ticker: str) -> None:
    watchlists = read_json_file(WATCHLIST_FILE)
    user_watchlist = watchlists.get(username, [])
    if not isinstance(user_watchlist, list):
        user_watchlist = []
    if ticker not in user_watchlist:
        user_watchlist.append(ticker)
    watchlists[username] = sorted(set(user_watchlist))
    write_json_file(WATCHLIST_FILE, watchlists)


def remove_from_watchlist(username: str, ticker: str) -> None:
    watchlists = read_json_file(WATCHLIST_FILE)
    user_watchlist = watchlists.get(username, [])
    if not isinstance(user_watchlist, list):
        user_watchlist = []
    user_watchlist = [item for item in user_watchlist if item != ticker]
    watchlists[username] = user_watchlist
    write_json_file(WATCHLIST_FILE, watchlists)


def normalize_ticker(raw_ticker: str, market: str) -> str:
    ticker = raw_ticker.strip().upper().replace(" ", "")
    if market == "India (NSE)" and "." not in ticker and not ticker.startswith("^"):
        ticker = f"{ticker}.NS"
    return ticker


@st.cache_data(show_spinner=False)
def download_data(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError("No data found for this ticker/date range.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    renamed = data[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return renamed


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["volume_change"] = df["volume"].pct_change()
    return df.dropna().copy()


def train_and_predict(features_df: pd.DataFrame, horizon_days: int) -> PredictionResult:
    dataset = features_df.copy()
    dataset["target"] = dataset["close"].shift(-horizon_days)
    dataset = dataset.dropna().copy()

    if len(dataset) < 120:
        raise ValueError("Not enough rows for training. Increase history period.")

    X = dataset[FEATURE_COLUMNS]
    y = dataset["target"]

    split_index = int(len(dataset) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)

    latest_row = X.tail(1)
    predicted_close = float(model.predict(latest_row)[0])

    tree_preds = np.array([tree.predict(latest_row)[0] for tree in model.estimators_])
    lower_bound = float(np.percentile(tree_preds, 10))
    upper_bound = float(np.percentile(tree_preds, 90))

    latest_close = float(features_df["close"].iloc[-1])
    expected_move_pct = ((predicted_close - latest_close) / latest_close) * 100

    test_df = pd.DataFrame({"Actual": y_test, "Predicted": test_pred}, index=y_test.index)
    chart_df = features_df[["close"]].copy().rename(columns={"close": "Close"})

    return PredictionResult(
        latest_close=latest_close,
        predicted_close=predicted_close,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        expected_move_pct=expected_move_pct,
        mae=float(mae),
        r2=float(r2),
        chart_df=chart_df,
        test_df=test_df,
    )


@st.cache_data(show_spinner=False, ttl=900)
def get_top_indian_snapshot() -> pd.DataFrame:
    end_date = dt.date.today() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=180)
    rows: List[Dict] = []

    for company_name, ticker in TOP_INDIAN_COMPANIES.items():
        try:
            data = download_data(ticker, start_date, end_date)
            closes = data["close"].dropna()
            if len(closes) < 25:
                continue

            latest = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) > 1 else latest
            month_anchor = float(closes.iloc[-22]) if len(closes) > 22 else float(closes.iloc[0])
            day_change = ((latest - prev) / prev) * 100 if prev else 0.0
            month_change = ((latest - month_anchor) / month_anchor) * 100 if month_anchor else 0.0

            vol = closes.pct_change().dropna().tail(22).std() * math.sqrt(252) * 100
            rows.append(
                {
                    "Company": company_name,
                    "Ticker": ticker,
                    "Sector": INDIAN_SECTOR_HINTS.get(ticker, "Unknown"),
                    "Last Close": round(latest, 2),
                    "1D Change %": round(day_change, 2),
                    "1M Return %": round(month_change, 2),
                    "Volatility %": round(float(vol) if not np.isnan(vol) else 0.0, 2),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("1M Return %", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


@st.cache_data(show_spinner=False, ttl=900)
def get_returns_for_tickers(
    tickers: Tuple[str, ...], start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:
    frames: Dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            data = download_data(ticker, start_date, end_date)
            frames[ticker] = data["close"]
        except Exception:
            continue

    if len(frames) < 2:
        raise ValueError("Need at least two tickers with valid data.")

    combined = pd.DataFrame(frames).dropna(how="all").ffill().dropna(how="any")
    returns = combined.pct_change().dropna()
    if returns.empty:
        raise ValueError("Could not compute returns for selected tickers.")
    return returns


@st.cache_data(show_spinner=False, ttl=300)
def get_quick_ticker_snapshot(ticker: str) -> Dict[str, float]:
    data = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("Ticker data unavailable.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    closes = data["Close"].dropna()
    if len(closes) < 2:
        raise ValueError("Not enough data for snapshot.")

    latest = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    month_anchor = float(closes.iloc[-22]) if len(closes) > 22 else float(closes.iloc[0])
    return {
        "latest": latest,
        "day_change_pct": ((latest - prev) / prev) * 100 if prev else 0.0,
        "month_change_pct": ((latest - month_anchor) / month_anchor) * 100 if month_anchor else 0.0,
    }


def dependency_metrics(source_returns: pd.Series, target_returns: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([source_returns, target_returns], axis=1).dropna()
    if len(aligned) < 20:
        raise ValueError("Not enough overlapping data for dependency analysis.")

    x = aligned.iloc[:, 0].to_numpy()
    y = aligned.iloc[:, 1].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "beta": float(slope),
        "intercept": float(intercept),
        "correlation": correlation,
        "r2": float(r2),
    }


def setup_session_state() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("chat_history", [])


def render_auth_screen() -> None:
    if st.session_state["logged_in"]:
        return

    st.title("Stock Prediction and Analysis Platform")
    st.caption("Login to access personalized predictions, recent searches, and assistant tools.")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", use_container_width=True)
        if submit_login:
            if authenticate_user(username.strip(), password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username.strip()
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            username = st.text_input("New username")
            password = st.text_input("New password", type="password")
            submit_register = st.form_submit_button("Create account", use_container_width=True)
        if submit_register:
            ok, message = register_user(username.strip(), password)
            if ok:
                st.success(message)
            else:
                st.error(message)

    st.stop()


def render_sidebar(username: str) -> str:
    with st.sidebar:
        st.header(f"Welcome, {username}")
        st.caption("Personalized stock workspace")
        page = st.radio(
            "Navigate",
            [
                "Prediction",
                "Top Indian Shares",
                "Connections and Dependency",
                "Recent Searches",
                "Assistant",
            ],
        )
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["chat_history"] = []
            st.rerun()
    return page


def render_prediction_page(username: str) -> None:
    st.header("Prediction Engine")
    st.caption("Train model on historical data, estimate future close price, and track your searches.")

    user_watchlist = get_watchlist(username)
    default_ticker = user_watchlist[0] if user_watchlist else "AAPL"

    col1, col2, col3 = st.columns(3)
    with col1:
        market = st.selectbox("Market", ["US / Global", "India (NSE)"])
    with col2:
        raw_ticker = st.text_input("Ticker symbol", value=default_ticker)
    with col3:
        horizon_days = st.slider("Prediction horizon (days)", min_value=1, max_value=30, value=7)

    history_years = st.slider("Years of history", min_value=1, max_value=10, value=5)
    run_prediction = st.button("Run prediction", type="primary")

    with st.expander("Watchlist", expanded=False):
        add_watch_col, remove_watch_col = st.columns(2)
        with add_watch_col:
            new_watch = st.text_input("Add ticker to watchlist", value="")
            if st.button("Add to watchlist"):
                ticker_to_add = normalize_ticker(new_watch, market)
                if ticker_to_add:
                    add_to_watchlist(username, ticker_to_add)
                    st.success(f"Added {ticker_to_add}")
                    st.rerun()
        with remove_watch_col:
            current_watchlist = get_watchlist(username)
            if current_watchlist:
                remove_choice = st.selectbox("Remove ticker", current_watchlist)
                if st.button("Remove from watchlist"):
                    remove_from_watchlist(username, remove_choice)
                    st.success(f"Removed {remove_choice}")
                    st.rerun()
            else:
                st.info("Your watchlist is empty.")

    if not run_prediction:
        st.info("Choose your settings and click Run prediction.")
        return

    ticker = normalize_ticker(raw_ticker, market)
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
        return

    end_date = dt.date.today() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=history_years * 365)

    try:
        with st.spinner("Downloading data and training model..."):
            raw_df = download_data(ticker, start_date, end_date)
            features_df = build_features(raw_df)
            result = train_and_predict(features_df, horizon_days)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    add_search_history(
        username,
        {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "horizon_days": horizon_days,
            "predicted_close": round(result.predicted_close, 2),
            "expected_move_pct": round(result.expected_move_pct, 2),
        },
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest close", f"${result.latest_close:,.2f}")
    m2.metric(f"Predicted close (+{horizon_days}d)", f"${result.predicted_close:,.2f}")
    m3.metric("Expected move", f"{result.expected_move_pct:+.2f}%")

    m4, m5, m6 = st.columns(3)
    m4.metric("Prediction interval (10-90%)", f"${result.lower_bound:,.2f} - ${result.upper_bound:,.2f}")
    m5.metric("Model MAE", f"${result.mae:,.2f}")
    m6.metric("Model R2", f"{result.r2:.3f}")

    st.subheader("Price History Graph")
    st.line_chart(result.chart_df)

    st.subheader("OHLC Graph")
    ohlc_df = raw_df[["open", "high", "low", "close"]].copy().rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    st.line_chart(ohlc_df)

    st.subheader("Volume Graph")
    st.bar_chart(raw_df[["volume"]].rename(columns={"volume": "Volume"}))

    st.subheader("Backtest: Actual vs Predicted")
    st.line_chart(result.test_df)

    indicator_df = pd.DataFrame(
        {
            "Close": raw_df["close"],
            "SMA20": raw_df["close"].rolling(20).mean(),
            "EMA10": raw_df["close"].ewm(span=10, adjust=False).mean(),
        }
    ).dropna()
    if not indicator_df.empty:
        st.subheader("Technical Trend Graph")
        st.line_chart(indicator_df.tail(250))

        latest_signal = "Bullish" if indicator_df["Close"].iloc[-1] > indicator_df["SMA20"].iloc[-1] else "Bearish"
        st.caption(f"Trend signal based on Close vs SMA20: {latest_signal}")


def render_recent_searches_page(username: str) -> None:
    st.header("Recently Searched Data")
    recent = get_search_history(username)

    if not recent:
        st.info("No search history yet. Run a prediction first.")
    else:
        history_df = pd.DataFrame(recent)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("Clear search history"):
            clear_search_history(username)
            st.success("Search history cleared.")
            st.rerun()

    st.subheader("Watchlist Snapshot")
    watchlist = get_watchlist(username)
    if not watchlist:
        st.info("Add tickers to watchlist from the Prediction page.")
        return

    snapshot_rows: List[Dict] = []
    for ticker in watchlist:
        try:
            snap = get_quick_ticker_snapshot(ticker)
            snapshot_rows.append(
                {
                    "Ticker": ticker,
                    "Latest": round(snap["latest"], 2),
                    "1D Change %": round(snap["day_change_pct"], 2),
                    "1M Change %": round(snap["month_change_pct"], 2),
                }
            )
        except Exception:
            snapshot_rows.append({"Ticker": ticker, "Latest": "N/A", "1D Change %": "N/A", "1M Change %": "N/A"})

    snapshot_df = pd.DataFrame(snapshot_rows)
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)


def render_top_indian_page() -> None:
    st.header("Top Indian Company Shares")
    st.caption("Live snapshot of leading Indian companies with return and volatility ranking.")

    top_df = get_top_indian_snapshot()
    if top_df.empty:
        st.error("Could not load Indian market snapshot right now.")
        return

    gainer = top_df.iloc[0]
    loser = top_df.sort_values("1M Return %", ascending=True).iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Top 1M gainer", f"{gainer['Ticker']} ({gainer['1M Return %']:+.2f}%)")
    c2.metric("Lowest 1M return", f"{loser['Ticker']} ({loser['1M Return %']:+.2f}%)")

    st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.subheader("Top Shares Performance Graph")
    compare_choices = st.multiselect(
        "Compare selected Indian shares",
        options=top_df["Ticker"].tolist(),
        default=top_df["Ticker"].tolist()[:4],
    )

    if len(compare_choices) >= 2:
        end_date = dt.date.today() + dt.timedelta(days=1)
        start_date = end_date - dt.timedelta(days=180)
        try:
            returns = get_returns_for_tickers(tuple(compare_choices), start_date, end_date)
            normalized = (1 + returns).cumprod()
            normalized = normalized / normalized.iloc[0]
            st.line_chart(normalized)
        except Exception as exc:
            st.warning(f"Comparison chart unavailable: {exc}")
    else:
        st.info("Select at least 2 shares to compare performance.")

    st.subheader("Sector Distribution")
    sector_counts = top_df["Sector"].value_counts()
    st.bar_chart(sector_counts)


def render_connections_page() -> None:
    st.header("Company Connection Graph and Dependency")
    st.caption("Analyze how shares move together and how one share statistically depends on another.")

    default_tickers = list(TOP_INDIAN_COMPANIES.values())[:6]
    selected_tickers = st.multiselect(
        "Select shares for connection graph",
        options=list(TOP_INDIAN_COMPANIES.values()),
        default=default_tickers,
    )
    if len(selected_tickers) < 2:
        st.info("Select at least 2 shares.")
        return

    period = st.select_slider("Lookback period", options=["3M", "6M", "1Y", "2Y"], value="6M")
    days_map = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    end_date = dt.date.today() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=days_map[period])

    try:
        returns = get_returns_for_tickers(tuple(selected_tickers), start_date, end_date)
    except Exception as exc:
        st.error(f"Connection analysis failed: {exc}")
        return

    st.subheader("Normalized Performance Graph")
    normalized = (1 + returns).cumprod()
    normalized = normalized / normalized.iloc[0]
    st.line_chart(normalized)

    st.subheader("Company Connection Matrix (Correlation)")
    corr = returns.corr().round(3)
    st.dataframe(corr, use_container_width=True)

    threshold = st.slider("Connection strength threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    edges: List[Dict] = []
    tickers = corr.columns.tolist()
    for i, source in enumerate(tickers):
        for target in tickers[i + 1 :]:
            val = float(corr.loc[source, target])
            if abs(val) >= threshold:
                edges.append(
                    {
                        "Source": source,
                        "Target": target,
                        "Correlation": round(val, 3),
                        "Relation": "Positive" if val >= 0 else "Negative",
                        "Strength": round(abs(val), 3),
                    }
                )

    st.subheader("Strong Company Connections")
    if edges:
        edges_df = pd.DataFrame(edges).sort_values("Strength", ascending=False)
        st.dataframe(edges_df, use_container_width=True, hide_index=True)
    else:
        st.info("No pair passed the selected threshold.")

    st.subheader("One Share Dependent on Another")
    base_ticker = st.selectbox("Driver share", selected_tickers, index=0)
    target_options = [ticker for ticker in selected_tickers if ticker != base_ticker]
    dependent_ticker = st.selectbox("Dependent share", target_options)

    try:
        metrics = dependency_metrics(returns[base_ticker], returns[dependent_ticker])
    except Exception as exc:
        st.warning(f"Dependency analysis unavailable: {exc}")
        return

    d1, d2, d3 = st.columns(3)
    d1.metric("Dependency beta", f"{metrics['beta']:.3f}")
    d2.metric("Correlation", f"{metrics['correlation']:.3f}")
    d3.metric("R2", f"{metrics['r2']:.3f}")

    implied_target_move = metrics["intercept"] + metrics["beta"] * 0.01
    st.caption(
        f"Interpretation: if {base_ticker} moves +1.0% in a day, "
        f"{dependent_ticker} is expected to move about {implied_target_move * 100:+.2f}%."
    )

    scatter_df = returns[[base_ticker, dependent_ticker]].dropna().rename(
        columns={base_ticker: "Driver Return", dependent_ticker: "Dependent Return"}
    )
    st.scatter_chart(scatter_df, x="Driver Return", y="Dependent Return")


def chatbot_reply(prompt: str, username: str) -> str:
    text = prompt.strip()
    lowered = text.lower()

    if not text:
        return "Please type a question."
    if any(word in lowered for word in ("hi", "hello", "hey")):
        return "Hello. Ask me about top Indian shares, recent searches, or a ticker like RELIANCE.NS."
    if "help" in lowered:
        return (
            "Try asking: 'top indian shares', 'recent searches', "
            "'price of TCS.NS', or 'how to analyze dependency'."
        )
    if "top" in lowered and "indian" in lowered:
        symbols = ", ".join(list(TOP_INDIAN_COMPANIES.values())[:8])
        return f"Popular Indian tickers: {symbols}."
    if "recent" in lowered:
        recent = get_search_history(username)
        if not recent:
            return "No recent searches found yet."
        previews = [f"{item['ticker']} ({item['expected_move_pct']}%)" for item in recent[:5]]
        return "Recent predictions: " + ", ".join(previews)
    if "dependency" in lowered or "connection" in lowered:
        return (
            "Open the 'Connections and Dependency' page, choose at least two shares, "
            "then use the dependency section for beta, correlation, and R2."
        )

    ticker_candidates = re.findall(r"\b[A-Z]{2,12}(?:\.NS)?\b", text.upper())
    if ticker_candidates:
        ticker = ticker_candidates[0]
        if "." not in ticker and ticker in [item.split(".")[0] for item in TOP_INDIAN_COMPANIES.values()]:
            ticker = f"{ticker}.NS"
        try:
            snap = get_quick_ticker_snapshot(ticker)
            return (
                f"{ticker} latest close: {snap['latest']:.2f}. "
                f"1D: {snap['day_change_pct']:+.2f}%, 1M: {snap['month_change_pct']:+.2f}%."
            )
        except Exception:
            return f"I could not fetch data for {ticker}. Check the ticker symbol."

    return "I can help with ticker snapshots, top Indian shares, recent searches, and dependency guidance."


def render_assistant_page(username: str) -> None:
    st.header("Stock Chatbot Assistant")
    st.caption("Ask quick questions about tickers, top Indian shares, and your recent app activity.")

    if not st.session_state["chat_history"]:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": (
                    "Hello. I am your stock assistant. Ask me about a ticker or type "
                    "'top indian shares' or 'recent searches'."
                ),
            }
        )

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input("Type your message")
    if not user_prompt:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    answer = chatbot_reply(user_prompt, username)
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)


def main() -> None:
    st.set_page_config(page_title="Advanced Stock Prediction App", layout="wide")
    ensure_storage_files()
    setup_session_state()
    render_auth_screen()

    st.title("Advanced Stock Prediction App")
    st.info("This app is for educational use only and not investment advice.")

    username = st.session_state["username"]
    page = render_sidebar(username)

    if page == "Prediction":
        render_prediction_page(username)
    elif page == "Top Indian Shares":
        render_top_indian_page()
    elif page == "Connections and Dependency":
        render_connections_page()
    elif page == "Recent Searches":
        render_recent_searches_page(username)
    else:
        render_assistant_page(username)


if __name__ == "__main__":
    main()
