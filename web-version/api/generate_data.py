"""
Script to generate static JSON data for the web interface.
This runs in GitHub Actions and outputs data that can be served via GitHub Pages.
"""
import json
import os
from trading_api import get_full_analysis

def generate_static_data():
    """Generate static JSON data files for the web interface"""
    
    # Create output directory
    output_dir = "../static/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate analysis for different time periods
    analyses = {
        "7d": get_full_analysis(days=7),
        "14d": get_full_analysis(days=14),
        "30d": get_full_analysis(days=30)
    }
    
    # Save each analysis
    for period, data in analyses.items():
        output_file = os.path.join(output_dir, f"analysis_{period}.json")
        with open(output_file, 'w') as f:
            json.dump(data, indent=2, fp=f, default=str)
        print(f"Generated {output_file}")
    
    # Check if all analyses were successful
    all_success = all(data.get("success", False) for data in analyses.values())
    
    if all_success:
        # Generate a combined summary
        summary = {
            "last_updated": analyses["14d"]["timestamp"],
            "symbol": "BTC_USDT_PERP",
            "periods": {
                "7d": {
                    "current_price": analyses["7d"]["strategy"]["current_price"],
                    "max_drawdown": analyses["7d"]["drawdown"]["max_drawdown"]
                },
                "14d": {
                    "current_price": analyses["14d"]["strategy"]["current_price"],
                    "max_drawdown": analyses["14d"]["drawdown"]["max_drawdown"]
                },
                "30d": {
                    "current_price": analyses["30d"]["strategy"]["current_price"],
                    "max_drawdown": analyses["30d"]["drawdown"]["max_drawdown"]
                }
            }
        }
        
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, indent=2, fp=f, default=str)
        print(f"Generated {summary_file}")
    else:
        print("Warning: Some analyses failed. Summary not generated.")
    
    print("\nData generation complete!")

if __name__ == "__main__":
    generate_static_data()
