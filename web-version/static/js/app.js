// Global variables
let chart = null;
let currentPeriod = '14d';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadData();
    
    // Set up period selector
    document.getElementById('periodSelect').addEventListener('change', function(e) {
        currentPeriod = e.target.value;
        loadData();
    });
});

// Load data from JSON files
async function loadData() {
    try {
        showLoading();
        
        const response = await fetch(`static/data/analysis_${currentPeriod}.json`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }
        
        const data = await response.json();
        
        if (data.success) {
            updateDashboard(data);
        } else {
            showError(data.error || 'Unknown error occurred');
        }
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load data. Please try again later.');
    }
}

// Update all dashboard elements
function updateDashboard(data) {
    // Update stats cards
    document.getElementById('currentPrice').textContent = `$${formatNumber(data.strategy.current_price)}`;
    document.getElementById('maxDrawdown').textContent = `${data.drawdown.max_drawdown}%`;
    document.getElementById('gridCount').textContent = data.strategy.grid_count;
    document.getElementById('leverage').textContent = `${data.strategy.leverage}x`;
    
    # Apply color based on drawdown
    const DRAWDOWN_THRESHOLD = -10;  // Threshold for significant drawdown
    const drawdownElement = document.getElementById('maxDrawdown');
    if (data.drawdown.max_drawdown < DRAWDOWN_THRESHOLD) {
        drawdownElement.classList.add('negative');
    } else {
        drawdownElement.classList.remove('negative');
    }
    
    // Update strategy details
    document.getElementById('entryPrice').textContent = `$${formatNumber(data.strategy.entry_price)}`;
    document.getElementById('upperLimit').textContent = `$${formatNumber(data.strategy.upper_limit)}`;
    document.getElementById('lowerLimit').textContent = `$${formatNumber(data.strategy.lower_limit)}`;
    document.getElementById('priceRange').textContent = `$${formatNumber(data.strategy.price_range)}`;
    document.getElementById('predictedHigh').textContent = `$${formatNumber(data.strategy.predicted_high)}`;
    document.getElementById('predictedLow').textContent = `$${formatNumber(data.strategy.predicted_low)}`;
    
    // Update drawdown analysis
    document.getElementById('maxDrawdownDetail').textContent = `${data.drawdown.max_drawdown}%`;
    document.getElementById('maxDrawdownDate').textContent = formatDate(data.drawdown.max_drawdown_date);
    document.getElementById('currentDrawdown').textContent = `${data.drawdown.current_drawdown}%`;
    document.getElementById('peakPrice').textContent = `$${formatNumber(data.drawdown.peak_price)}`;
    
    // Update last updated time
    document.getElementById('lastUpdated').textContent = formatDate(data.timestamp);
    
    // Update chart
    updateChart(data.price_data);
    
    hideLoading();
}

// Update the price chart
function updateChart(priceData) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Prepare chart data
    const labels = priceData.map(item => formatDate(item.time));
    const prices = priceData.map(item => item.close);
    const highs = priceData.map(item => item.high);
    const lows = priceData.map(item => item.low);
    
    // Destroy existing chart if it exists
    if (chart) {
        chart.destroy();
    }
    
    // Create new chart
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Close Price',
                    data: prices,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'High',
                    data: highs,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.05)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Low',
                    data: lows,
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.05)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#ffffff',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#4CAF50',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#b0b7c3',
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#b0b7c3',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

// Utility functions
function formatNumber(num) {
    return parseFloat(num).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function showLoading() {
    const statValues = document.querySelectorAll('.stat-value');
    statValues.forEach(el => el.classList.add('loading'));
}

function hideLoading() {
    const statValues = document.querySelectorAll('.stat-value');
    statValues.forEach(el => el.classList.remove('loading'));
}

function showError(message) {
    // Create error notification element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.textContent = `Error: ${message}`;
    errorDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #f44336; color: white; padding: 15px 25px; border-radius: 8px; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.3);';
    
    document.body.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.style.opacity = '0';
        errorDiv.style.transition = 'opacity 0.5s';
        setTimeout(() => errorDiv.remove(), 500);
    }, 5000);
    
    hideLoading();
}
