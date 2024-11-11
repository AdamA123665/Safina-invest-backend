import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Define the Asset data class
@dataclass
class Asset:
    name: str
    returns: pd.Series
    confidence_score: float = 5.0
    asset_class: str = 'equity'
    ticker: str = None
    info: str = None
    color: str = None
    expected_return: float = None
    expected_volatility: float = None
    sharpe_ratio: float = None
    sortino_ratio: float = None
    max_drawdown: float = None
    var_95: float = None
    cvar_95: float = None
    beta: float = None
    tracking_error: float = None
    rolling_metrics: Dict = None

class PortfolioRequest(BaseModel):
    initial_investment: float
    risk_tolerance: float

    @validator('initial_investment')
    def validate_investment(cls, v):
        if v < 1000 or v > 10000000:
            raise ValueError('Initial investment must be between 1,000 and 10,000,000')
        return v

    @validator('risk_tolerance')
    def validate_risk(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Risk tolerance must be between 1 and 10')
        return v

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the AdvancedIslamicPortfolioOptimizer class
class AdvancedIslamicPortfolioOptimizer:

    def __init__(self,
                 risk_tolerance: float,
                 initial_investment: float,
                 risk_free_rate: float = 0.04):
        self.assets: Dict[str, Asset] = {}
        self.risk_tolerance = risk_tolerance
        self.initial_investment = initial_investment
        self.trading_days_per_year = 252
        self.risk_free_rate = risk_free_rate
        self.max_allocation = self._calculate_max_allocation()
        self.defensive_ratio = self._calculate_defensive_ratio()
        self.portfolio_metrics = {}

    def _calculate_max_allocation(self) -> float:
        return max(-0.05 * (self.risk_tolerance - 7)**2 + 0.8, 1.0)

    def _calculate_defensive_ratio(self) -> float:
        return max(0.6 - (self.risk_tolerance / 10)**1.2 * 0.6, 0.0)

    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series for given returns."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown

    def add_asset(self, name: str, returns: pd.Series, confidence_score: float,
                  asset_class: str, ticker: str, info: str, color: str):
        asset = Asset(
            name=name,
            returns=returns,
            confidence_score=confidence_score,
            asset_class=asset_class,
            ticker=ticker,
            info=info,
            color=color
        )
        metrics = self._calculate_asset_metrics(returns)
        (asset.expected_return, asset.expected_volatility, asset.sharpe_ratio,
         asset.sortino_ratio, asset.max_drawdown, asset.var_95, asset.cvar_95,
         asset.beta, asset.tracking_error) = metrics
        asset.rolling_metrics = self._calculate_rolling_metrics(returns)
        self.assets[name] = asset

    def _calculate_asset_metrics(self, returns: pd.Series) -> Tuple:
        ann_return = returns.mean() * self.trading_days_per_year
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        sharpe_ratio = (ann_return - self.risk_free_rate
                        ) / volatility if volatility != 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(
            self.trading_days_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = (ann_return - self.risk_free_rate
                         ) / downside_std if downside_std != 0 else 0
        max_drawdown = self._calculate_drawdown(returns).min()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(
            returns[returns <= var_95]) > 0 else 0
        beta, tracking_error = None, None
        return (ann_return, volatility, sharpe_ratio, sortino_ratio,
                max_drawdown, var_95, cvar_95, beta, tracking_error)

    def _calculate_rolling_metrics(self,
                                   returns: pd.Series,
                                   window: int = 252) -> Dict:
        rolling_returns = returns.rolling(window=window).mean() * 252
        rolling_volatility = returns.rolling(
            window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns -
                          self.risk_free_rate) / rolling_volatility
        rolling_drawdown = self._calculate_drawdown(returns)
        return {
            'returns': rolling_returns,
            'volatility': rolling_volatility,
            'sharpe': rolling_sharpe,
            'drawdown': rolling_drawdown
        }

    def optimize_portfolio(self) -> dict:
        """Portfolio optimization using all calculated metrics and ensuring diversification."""
        returns_df = pd.DataFrame({asset.name: asset.returns 
                                   for asset in self.assets.values()})
        
        # Prepare optimization inputs
        ann_returns = np.array([asset.expected_return for asset in self.assets.values()])
        volatilities = np.array([asset.expected_volatility for asset in self.assets.values()])
        cvars = np.array([asset.cvar_95 for asset in self.assets.values()])
        sortino_ratios = np.array([asset.sortino_ratio for asset in self.assets.values()])
        covariance = returns_df.cov() * self.trading_days_per_year
        
        # Adjust lambda values based on risk tolerance
        # Risk tolerance scale is from 1 (low risk) to 10 (high risk)
        risk_level = (self.risk_tolerance - 1) / 9  # Normalized risk level from 0 to 1
        
        # Further amplify return weight for higher risk, maintaining balance at lower risk
        lambda_return = (risk_level ** 2 + risk_level * 2)  # Stronger emphasis on return at high risk
        
        # Decay for volatility and CVaR to maintain stability at low risk levels
        lambda_volatility = (1 - risk_level) ** 3  # Slightly softened decay for more flexibility at low risk
        lambda_cvar = (1 - risk_level) ** 3
        
        # Parabolic shape for diversification, slightly boosted at low risk, sharply reduced at high risk
        lambda_diversification = 3.5 * risk_level * (1 - risk_level) + 0.1 * (1 - risk_level)  # Higher initial boost, steep drop-off at high risk
        
        # Sortino ratio increases linearly with risk tolerance
        lambda_sortino = 0.5 * risk_level
        
        # Normalize lambdas so they sum to 1
        total_lambda = lambda_return + lambda_volatility + lambda_cvar + lambda_sortino + lambda_diversification
        lambda_return /= total_lambda
        lambda_volatility /= total_lambda
        lambda_cvar /= total_lambda
        lambda_sortino /= total_lambda
        lambda_diversification /= total_lambda

        def objective(weights):
            # Portfolio metrics
            portfolio_return = np.dot(weights, ann_returns)
            portfolio_vol = np.sqrt(weights.T @ covariance @ weights)
            portfolio_cvar = np.dot(weights, cvars)
            portfolio_sortino = (portfolio_return - self.risk_free_rate) / (np.sqrt(np.dot(weights ** 2, volatilities ** 2)))
            # Diversification penalty (sum of weighted correlations)
            correlation_matrix = returns_df.corr().values
            diversification_penalty = np.sum(np.outer(weights, weights) * correlation_matrix)
            
            # Objective function
            score = (lambda_return * portfolio_return) - \
                    (lambda_volatility * portfolio_vol) - \
                    (lambda_cvar * portfolio_cvar) + \
                    (lambda_sortino * portfolio_sortino) - \
                    (lambda_diversification * diversification_penalty)
            
            return -score  # Negative because we minimize in optimization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            # Defensive assets minimum allocation
            {'type': 'ineq', 'fun': lambda x: np.sum(x[self._get_defensive_indices()]) - self.defensive_ratio}
        ]
        
        # Bounds
        bounds = [(0, self.max_allocation) for _ in range(len(self.assets))]
        
        # Optimize
        result = minimize(objective, 
                          x0=np.array([1.0/len(self.assets)] * len(self.assets)),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        
        # Check if optimization was successful
        if not result.success:
            raise ValueError("Optimization failed: " + result.message)
        
        # Calculate portfolio metrics
        self.portfolio_metrics = self._calculate_portfolio_metrics(result.x, returns_df)
        
        # Store optimal weights
        self.optimal_weights = result.x
        
        return self.portfolio_metrics

    def _get_defensive_indices(self) -> List[int]:
        """Get indices of defensive assets."""
        return [i for i, asset in enumerate(self.assets.values())
                if asset.asset_class == 'defensive']

    def _calculate_portfolio_metrics(self, weights, returns_df):
        """Updated portfolio metrics calculation to handle NaN values"""
        ann_returns = returns_df.mean() * self.trading_days_per_year
        portfolio_return = np.dot(weights, ann_returns)
        covariance_matrix = returns_df.cov() * self.trading_days_per_year
        portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)

        # Handle potential NaN in Sharpe ratio
        sharpe_ratio = 0.0
        if portfolio_volatility != 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Handle potential NaN in Sortino ratio
        downside_returns = returns_df[returns_df < 0]
        sortino_ratio = 0.0
        if not downside_returns.empty:
            downside_covariance = downside_returns.cov() * self.trading_days_per_year
            downside_volatility = np.sqrt(weights.T @ downside_covariance @ weights)
            if downside_volatility != 0:
                sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_volatility

        # Calculate max drawdown with NaN handling
        portfolio_returns = returns_df @ weights
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Ensure no NaN values in weights
        weights_dict = {
            asset.name: float(weight) if not np.isnan(weight) else 0.0
            for asset, weight in zip(self.assets.values(), weights)
        }

        portfolio_metrics = {
            'Expected Return': float(portfolio_return) if not np.isnan(portfolio_return) else 0.0,
            'Volatility': float(portfolio_volatility) if not np.isnan(portfolio_volatility) else 0.0,
            'Sharpe Ratio': float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
            'Sortino Ratio': float(sortino_ratio) if not np.isnan(sortino_ratio) else 0.0,
            'Max Drawdown': float(max_drawdown) if not np.isnan(max_drawdown) else 0.0,
            'Weights': weights_dict
        }

        return portfolio_metrics

    def simulate_portfolio_returns(self, num_simulations=1000, num_years=10):
        """Perform Monte Carlo simulations to project future returns over multiple years."""
        weights = self.optimal_weights
        returns_df = pd.DataFrame(
            {asset.name: asset.returns
             for asset in self.assets.values()})
        # Use historical mean returns and covariance matrix
        mean_returns = returns_df.mean()
        covariance_matrix = returns_df.cov()
        
        num_days_per_year = 252
        total_days = num_years * num_days_per_year
        
        # Simulate returns
        simulated_portfolio_values = []
        for i in range(num_simulations):
            simulated_daily_returns = np.random.multivariate_normal(
                mean_returns, covariance_matrix, total_days)
            simulated_portfolio_daily_returns = simulated_daily_returns @ weights
            cumulative_returns = np.cumprod(1 + simulated_portfolio_daily_returns)
            # Get the portfolio value at the end of each year
            yearly_values = cumulative_returns[(np.arange(num_years) + 1) * num_days_per_year - 1]
            simulated_portfolio_values.append(yearly_values)
        
        # Convert to a NumPy array for easier manipulation
        simulated_portfolio_values = np.array(simulated_portfolio_values)
        
        # Calculate statistics
        projected_return_mean = np.mean(simulated_portfolio_values, axis=0)
        projected_return_5th_percentile = np.percentile(simulated_portfolio_values, 5, axis=0)
        projected_return_95th_percentile = np.percentile(simulated_portfolio_values, 95, axis=0)
        
        self.projected_returns = {
            'years': list(range(1, num_years + 1)),
            'mean': projected_return_mean.tolist(),
            '5th_percentile': projected_return_5th_percentile.tolist(),
            '95th_percentile': projected_return_95th_percentile.tolist()
        }

    # Methods to generate dashboard data
    def create_dashboard(self) -> dict:
        """Modified to return structured data instead of Plotly figure"""
        performance_data = self._get_performance_data()
        volatility_data = self._get_volatility_data()
        risk_metrics = self._get_risk_metrics()
        weights_data = self._get_weights_data()
        projected_returns_data = self._get_projected_returns_data()
        asset_info = self._get_asset_info()
        research_articles = self._get_research_articles()
        
        return {
            'performance': performance_data,
            'volatility': volatility_data,
            'risk_metrics': risk_metrics,
            'weights': weights_data,
            'projected_returns': projected_returns_data,
            'asset_info': asset_info,
            'research_articles': research_articles
        }

    def _get_performance_data(self):
        returns_df = pd.DataFrame({asset.name: asset.returns for asset in self.assets.values()})
        cumulative_returns = (1 + returns_df).cumprod()
        
        return {
            'dates': cumulative_returns.index.strftime('%Y-%m-%d').tolist(),
            'series': [
                {
                    'name': col,
                    'values': cumulative_returns[col].fillna(0).tolist()
                }
                for col in cumulative_returns.columns
            ]
        }

    def _get_volatility_data(self):
        data = {
            'dates': [],
            'series': []
        }
        for asset in self.assets.values():
            volatility = asset.rolling_metrics['volatility']
            if not data['dates']:
                data['dates'] = volatility.index.strftime('%Y-%m-%d').tolist()
            data['series'].append({
                'name': asset.name,
                'values': [
                    float(val) if not np.isnan(val) else 0.0 
                    for val in volatility.tolist()
                ]
            })
        return data

    def _get_risk_metrics(self):
        metrics = ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
        values = [
            float(self.portfolio_metrics.get(metric, 0)) 
            if not np.isnan(self.portfolio_metrics.get(metric, 0)) 
            else 0.0 
            for metric in metrics
        ]
        return {
            'labels': metrics,
            'values': values
        }

    def _get_weights_data(self):
        weights = self.portfolio_metrics.get('Weights', {})
        return {
            'labels': list(weights.keys()),
            'values': [
                float(val) if not np.isnan(val) else 0.0 
                for val in weights.values()
            ]
        }
    
    def _get_projected_returns_data(self):
        if not hasattr(self, 'projected_returns'):
            self.simulate_portfolio_returns()
        return self.projected_returns

    def _get_asset_info(self):
        return [
            {
                'name': asset.name,
                'ticker': asset.ticker,
                'info': asset.info,
                'color': asset.color
            }
            for asset in self.assets.values()
        ]

    def _get_research_articles(self):
        # Research articles with additional details
        return [
            {
                'title': 'The US Election',
                'content': """Since Harris's crushing defeat, Trump trades have surged. Financial stocks have received a significant boost with the prospect of deregulation on the horizon. Tech stocks, particularly Tesla, have also rallied sharply, while fossil fuel-based energy companies are enjoying a renewed resurgence.

                With inflation showing no signs of cooling, the 10-year Treasury yield has spiked, leading to a bear steepening. A "higher for longer" approach is now a reality, with the US dollar making essential gains against the GBP, EUR, and JPY. Meanwhile, Asian markets have stumbled, as tariffs may push large conglomerates to reduce their offshore sales.

                Overall, this is a strong uplift for the US market, but it remains to be seen if these higher rates will start to slow the US economy.""",
                'date': '2024-11-05',
                'image_url': 'https://www.ft.com/__origami/service/image/v2/images/raw/ftcms%3A6f22b49f-c9e1-4ddf-9cc6-eead253330d0?source=next-article&fit=scale-down&quality=highest&width=1440&dpr=1',
                'link': 'https://www.example.com/us-election-article'
            },
            {
                'title': 'Election day thoughts',
                'content': 'An overview of how recent global events are affecting investments, with a focus on emerging markets...',
                'date': '2024-11-04',
                'image_url': 'https://www.ealingtimes.co.uk/resources/images/18730697/?type=responsive-gallery-fullscreen',
                'link': 'https://www.example.com/global-events-article'
            },
            {
                'title': 'Iran vs the world',
                'content': 'Understanding the principles of Sharia-compliant investing, including benefits and popular strategies...',
                'date': '2024-11-03',
                'image_url': 'https://t3.ftcdn.net/jpg/00/12/34/84/360_F_12348489_HuKrpd65r1iDhzIn0k6oGzBGXkq9Z00h.jpg',
                'link': 'https://www.example.com/sharia-investing-article'
            }
        ]


# Define the function to process the Excel data
def process_excel_data(excel_file: str):
    data = pd.read_excel(excel_file)
    assets_prices = {}
    for i in range(0, len(data.columns) - 1, 2):
        date_col, price_col = data.columns[i], data.columns[i + 1]
        if 'Unnamed' in price_col:
            continue
        asset_name, dates, prices = price_col.strip(), data[date_col], data[price_col]
        asset_df = pd.DataFrame({'Date': dates, asset_name: prices}).dropna()
        asset_df.set_index('Date', inplace=True)
        assets_prices[asset_name] = asset_df[asset_name]
    
    df = pd.DataFrame(assets_prices)
    df.index = pd.to_datetime(df.index)
    return df.pct_change().dropna()

# Define the function to create the portfolio
def create_islamic_portfolio(
        returns_data: pd.DataFrame, risk_tolerance: float,
        initial_investment: float) -> Tuple[dict, dict]:
    optimizer = AdvancedIslamicPortfolioOptimizer(
        risk_tolerance=risk_tolerance, initial_investment=initial_investment)
    
    # Define asset classes
    asset_classes = {
        'Titans 100': 'equity',
        'World ESG': 'equity',
        'USA ESG': 'equity',
        'EM ESG': 'equity',
        'EUROPE ESG': 'equity',
        'Sukuk': 'defensive',
        'REIT': 'defensive',
        'Gold': 'defensive'
    }
    
    # Define asset tickers
    asset_tickers = {
        'Titans 100': 'IGDA or HIPS',
        'World ESG': 'HIWS or ISWD',
        'USA ESG': 'HLAL, HIUS, ISUS',
        'EM ESG': 'ISDE, HIES',
        'EUROPE ESG': 'HIPS',
        'Sukuk': 'HBKS, SKUK',
        'REIT': 'HIND LN',
        'Gold': 'SGLD, SGLN'
    }
    
    # Define asset information
    asset_infos = {
        'Titans 100': 'The 100 largest Sharia-compliant companies globally. These are quite volatile but provide the largest return over 10 years. Invesco and HSBC provide ETFs which track these funds',
        'World ESG': 'Global ESG-focused Sharia-compliant equities. This equity fund is much more stable and allows you to get exposure to a basket of stocks around the world. It is market cap weighted and has lower returns.',
        'USA ESG': 'US-based ESG-focused Sharia-compliant equities. This provides strong returns but is more volitile. The US has a track record of doing well the past 40 years :)',
        'EM ESG': 'Emerging markets ESG-focused Sharia-compliant equities. This gives you specific exposure to mainly chinese and indian equities. Historically they have done well but with higher rates, it hasnt performed the best ',
        'EUROPE ESG': 'European ESG-focused Sharia-compliant equities. This goves you exposure to eurpoean stocks which do well but not as good as America',
        'Sukuk': 'Islamic bonds compliant with Sharia law. This is the most stable and you proabably wont lose your money but the return % historically is the lowest',
        'REIT': 'Real Estate Investment Trusts that are Sharia-compliant. This gives you exposure to real estate, finally something i know... Yes its better than Sukuk but will give you less than some equity funds and doenst perform best in high interest rate environmets',
        'Gold': 'Investment in physical gold assets. We all know what this is. Since the dawn of time gold has been a go to investment class. And recently its even outperfomed the S&P'
    }
    
    # Define asset colors
    asset_colors = {
        'Titans 100': '#0088FE',
        'World ESG': '#00C49F',
        'USA ESG': '#FFBB28',
        'EM ESG': '#FF8042',
        'EUROPE ESG': '#8A2BE2',
        'Sukuk': '#FF69B4',
        'REIT': '#A52A2A',
        'Gold': '#FFD700'  # Gold color
    }
    
    # Add assets to the optimizer
    for asset_name in returns_data.columns:
        optimizer.add_asset(
            name=asset_name,
            returns=returns_data[asset_name],
            confidence_score=5.0,
            asset_class=asset_classes.get(asset_name, 'equity'),
            ticker=asset_tickers.get(asset_name, ''),
            info=asset_infos.get(asset_name, ''),
            color=asset_colors.get(asset_name, '#FFFFFF')
        )
    portfolio_metrics = optimizer.optimize_portfolio()
    dashboard_data = optimizer.create_dashboard()
    return portfolio_metrics, dashboard_data

# Define the API endpoint
@app.post("/api/portfolio/optimize")
async def create_portfolio(request: PortfolioRequest):
    try:
        excel_file = 'ETFreturns.equites.xlsx'
        returns_df = process_excel_data(excel_file)

        portfolio_metrics, dashboard_data = create_islamic_portfolio(
            returns_data=returns_df,
            risk_tolerance=request.risk_tolerance,
            initial_investment=request.initial_investment
        )

        # Clean any NaN values from the metrics
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(x) for x in obj]
            elif isinstance(obj, (float, np.float32, np.float64)):
                return float(obj) if not np.isnan(obj) else 0.0
            return obj

        # Clean both portfolio metrics and dashboard data
        cleaned_metrics = clean_nan(portfolio_metrics)
        cleaned_dashboard = clean_nan(dashboard_data)

        response_data = {
            'portfolio_metrics': cleaned_metrics,
            'dashboard_data': cleaned_dashboard
        }

        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
