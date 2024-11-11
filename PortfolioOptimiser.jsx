import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Input,
  Label,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/';
import { Info, PieChart, DollarSign, TrendingUp, BarChart2 } from 'lucide-react';

const PortfolioOptimizer = () => {
  const [initialInvestment, setInitialInvestment] = useState(10000);
  const [riskTolerance, setRiskTolerance] = useState(5);
  const [portfolioData, setPortfolioData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleOptimize = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/create_portfolio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initial_investment: initialInvestment,
          risk_tolerance: riskTolerance
        })
      });

      if (!response.ok) {
        throw new Error('Optimization failed');
      }

      const data = await response.json();

      // Prepare the chart data
      const chartData = prepareChartData(data.graph.data);
      data.graph.chartData = chartData;

      setPortfolioData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const prepareChartData = (graphData) => {
    // Combine all assets into a single time series for the portfolio performance
    const dateSet = new Set();
    graphData.forEach(asset => {
      asset.x.forEach(date => dateSet.add(date));
    });
    const dates = Array.from(dateSet).sort();

    const combinedData = dates.map(date => {
      const point = { date };
      graphData.forEach(asset => {
        const index = asset.x.indexOf(date);
        point[asset.name] = index !== -1 ? asset.y[index] : null;
      });
      return point;
    });

    return combinedData;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold text-center text-blue-800">Islamic Sharia Portfolio Optimizer</h1>

      <Card>
        <CardHeader>
          <CardTitle>Portfolio Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Initial Investment ($)</Label>
              <Input
                type="number"
                value={initialInvestment}
                onChange={(e) => setInitialInvestment(Number(e.target.value))}
                min={1000}
                max={1000000}
                step={1000}
              />
            </div>
            <div>
              <Label>Risk Tolerance</Label>
              <Slider
                value={[riskTolerance]}
                onValueChange={(val) => setRiskTolerance(val[0])}
                min={1}
                max={10}
                step={1}
              />
              <span className="text-sm text-gray-500">Current: {riskTolerance}/10</span>
            </div>
          </div>
          <Button
            onClick={handleOptimize}
            disabled={isLoading}
            className="w-full"
          >
            {isLoading ? 'Optimizing...' : 'Optimize Portfolio'}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
          {error}
        </div>
      )}

      {portfolioData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Portfolio Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {Object.entries(portfolioData.portfolio_metrics).map(([key, value]) =>
                    key !== 'Weights' && (
                      <TableRow key={key}>
                        <TableCell>{key}</TableCell>
                        <TableCell className="text-right">
                          {typeof value === 'number'
                            ? (value * 100).toFixed(2) + '%'
                            : value}
                        </TableCell>
                      </TableRow>
                    )
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Asset Allocation */}
          <Card>
            <CardHeader>
              <CardTitle>Asset Allocation</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell className="text-right">Allocation</TableCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(portfolioData.portfolio_metrics.Weights).map(([asset, weight]) => (
                    <TableRow key={asset}>
                      <TableCell>{asset}</TableCell>
                      <TableCell className="text-right">
                        {(weight * 100).toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Performance Chart */}
          <Card className="col-span-full">
            <CardHeader>
              <CardTitle>Portfolio Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={portfolioData.graph.chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(tick) => new Date(tick).toLocaleDateString()} />
                  <YAxis />
                  <Tooltip labelFormatter={(label) => new Date(label).toLocaleDateString()} />
                  <Legend />
                  {Object.keys(portfolioData.portfolio_metrics.Weights).map((assetName, index) => (
                    <Line
                      key={assetName}
                      type="monotone"
                      dataKey={assetName}
                      name={assetName}
                      stroke={`hsl(${index * 60}, 70%, 50%)`}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default PortfolioOptimizer;
