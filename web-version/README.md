# Web Version - Grid Trading Dashboard

This is the web-based version that deploys to GitHub Pages and provides a real-time dashboard for monitoring Bitcoin grid trading strategies.

## Features

- 📊 Real-time grid trading strategy analysis
- 📉 Drawdown analysis and visualization
- 🔄 Automatic data updates every 4 hours via GitHub Actions
- 💰 Free API usage (Pionex public API)
- 🌐 Accessible anywhere via GitHub Pages

## How It Works

1. **GitHub Actions** runs automatically every 4 hours
2. Fetches latest Bitcoin price data from Pionex API (free)
3. Calculates optimal grid trading parameters
4. Analyzes drawdown and performance metrics
5. Generates static JSON files with the analysis
6. Deploys to GitHub Pages for public access

## Setup

### Enable GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" section
3. Set source to "GitHub Actions"
4. The workflow will automatically deploy on the next push

### Manual Deployment

To manually trigger a deployment:

1. Go to "Actions" tab in your repository
2. Select "Deploy to GitHub Pages" workflow
3. Click "Run workflow"

## File Structure

```
web-version/
├── index.html              # Main dashboard page
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   ├── js/
│   │   └── app.js         # Dashboard logic
│   └── data/              # Generated data (auto-updated)
│       ├── analysis_7d.json
│       ├── analysis_14d.json
│       └── analysis_30d.json
└── api/
    ├── trading_api.py      # API functions
    ├── generate_data.py    # Data generation script
    └── generate_mock_data.py  # Mock data for testing
```

## Local Testing

To test locally:

```bash
cd web-version
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## API Keys (Optional)

If you want to use premium features or higher rate limits:

1. Go to repository Settings → Secrets → Actions
2. Add secret: `PIONEX_API_KEY`

The application works with free APIs by default.

## Customization

### Change Update Frequency

Edit `.github/workflows/deploy.yml`:

```yaml
schedule:
  - cron: '0 */4 * * *'  # Every 4 hours
```

### Modify Trading Parameters

Edit `web-version/api/trading_api.py` to adjust:
- Grid count calculation
- Leverage calculation
- Risk factors
- Price predictions

## Troubleshooting

**Data not updating:**
- Check GitHub Actions logs
- Verify API endpoint is accessible
- Check rate limits

**Page not loading:**
- Verify GitHub Pages is enabled
- Check workflow completed successfully
- Clear browser cache

## Technologies

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js
- **Backend**: Python (data generation)
- **APIs**: Pionex Public API (free)
- **Deployment**: GitHub Actions + GitHub Pages
