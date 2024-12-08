# Pionex Grid Trading

This project implements a grid trading strategy using the Pionex API to fetch cryptocurrency price data. It analyzes the data to determine optimal entry points and sets up a grid trading strategy with defined limits, take-profit, and stop-loss levels.

## Project Structure

```
pionex-grid-trading
├── src
│   ├── fetch_data.py       # Fetches the last 14 days of 4-hour price data from Pionex API
│   ├── analysis.py         # Analyzes price data to determine optimal entry points and predictions
│   ├── strategy.py         # Defines the grid trading strategy and calculates grids and leverage
│   └── main.py             # Entry point for the application
├── requirements.txt        # Lists project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pionex-grid-trading
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will fetch the latest price data, analyze it, and execute the grid trading strategy based on the defined parameters.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.