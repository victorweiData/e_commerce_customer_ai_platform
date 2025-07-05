import React, { useState, useEffect } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, PieChart, Pie,
    Cell, Scatter, RadarChart, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, Radar
  } from 'recharts'

  import { Users, DollarSign, Star, TrendingUp, AlertTriangle, Crown } from 'lucide-react';

const CustomerSegmentDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [clusterData, setClusterData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Define cluster names and types mapping
  const clusterConfig = {
    0: { name: "Budget Conscious", type: "standard" },
    1: { name: "High-Value Premium", type: "premium" },
    2: { name: "Payment Diversified", type: "standard" },
    3: { name: "Dissatisfied Regular", type: "at-risk" },
    4: { name: "Low Engagement", type: "low-value" }
  };

  // Load data from JSON file
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Try to read the JSON file
        // try to fetch your JSON from public/segment_summary.json
        const resp = await fetch('/segment_summary.json');
        if (!resp.ok) throw new Error(resp.statusText);
        const parsedData = (await resp.json()) as RawCluster[];
        
        // Enhance data with calculated fields and metadata
        const enhancedData = parsedData.map(cluster => ({
          ...cluster,
          name: clusterConfig[cluster.cluster]?.name || `Cluster ${cluster.cluster}`,
          type: clusterConfig[cluster.cluster]?.type || "standard",
          annualValue: Math.round(cluster.avgOrderValue),
          totalValue: Math.round(cluster.customers * cluster.avgOrderValue)
        }));
        
        setClusterData(enhancedData);
        setError(null);
      } catch (err) {
        console.error('Error loading data:', err);
        setError('Failed to load segment data. Please ensure the file exists at dashboard/public/segment_summary.json');
        
        // Fallback to sample data for demonstration
        const fallbackData = [
          {
            cluster: 0,
            name: "Budget Conscious",
            customers: 61555,
            percentage: 61.9,
            avgOrderValue: 115.23,
            avgInstallments: 1.94,
            avgReviewScore: 4.75,
            reviewRate: 1.01,
            categoryDiversity: 1.00,
            annualValue: 115,
            totalValue: 7093184,
            type: "standard"
          },
          {
            cluster: 1,
            name: "High-Value Premium",
            customers: 14539,
            percentage: 14.6,
            avgOrderValue: 388.95,
            avgInstallments: 8.07,
            avgReviewScore: 4.23,
            reviewRate: 1.01,
            categoryDiversity: 1.02,
            annualValue: 389,
            totalValue: 5654934,
            type: "premium"
          }
        ];
        setClusterData(fallbackData);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Calculate derived metrics
  const totalCustomers = clusterData.reduce((sum, cluster) => sum + cluster.customers, 0);
  const totalRevenue = clusterData.reduce((sum, cluster) => sum + cluster.totalValue, 0);
  const premiumPercentage = clusterData.find(c => c.type === 'premium')?.percentage || 0;
  const atRiskPercentage = clusterData.filter(c => c.type === 'at-risk' || c.type === 'low-value')
    .reduce((sum, cluster) => sum + cluster.percentage, 0);

  const getClusterColor = (type) => {
    switch(type) {
      case 'premium': return '#10B981';
      case 'standard': return '#3B82F6';
      case 'at-risk': return '#F59E0B';
      case 'low-value': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getClusterIcon = (type) => {
    switch(type) {
      case 'premium': return <Crown className="w-5 h-5" />;
      case 'standard': return <Users className="w-5 h-5" />;
      case 'at-risk': return <AlertTriangle className="w-5 h-5" />;
      case 'low-value': return <TrendingUp className="w-5 h-5 rotate-180" />;
      default: return <Users className="w-5 h-5" />;
    }
  };

  const radarData = clusterData.map(cluster => ({
    cluster: `Cluster ${cluster.cluster}`,
    'Order Value': Math.round((cluster.avgOrderValue / 400) * 100),
    'Review Score': Math.round((cluster.avgReviewScore / 5) * 100),
    'Installments': Math.round((cluster.avgInstallments / 10) * 100),
    'Category Diversity': Math.round(cluster.categoryDiversity * 100),
    'Review Rate': Math.round(cluster.reviewRate * 100)
  }));

  const getRecommendations = (cluster) => {
    const recommendations = {
      'premium': {
        title: 'High-Value Premium',
        description: 'VIP programs, exclusive early access, premium support, flexible financing options. These customers drive profit.',
        priority: 'Short-term',
        color: 'green'
      },
      'standard': {
        title: cluster.name,
        description: cluster.avgReviewScore > 4.5 
          ? 'Volume discounts, loyalty rewards, bundle deals. Happy customers who value good prices - your foundation.'
          : 'Payment innovation testing, cross-sell opportunities, financial service partnerships. Tech-savvy segment.',
        priority: 'Long-term',
        color: 'blue'
      },
      'at-risk': {
        title: 'Dissatisfied Regular',
        description: 'URGENT: Root cause analysis, quality improvements, proactive outreach, service recovery programs.',
        priority: 'Immediate',
        color: 'red'
      },
      'low-value': {
        title: 'Low Engagement',
        description: 'Onboarding improvements, product discovery features, personalized recommendations, win-back campaigns.',
        priority: 'Short-term',
        color: 'orange'
      }
    };
    
    return recommendations[cluster.type] || recommendations['standard'];
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading customer segment data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Customer Segment Analysis</h1>
          <p className="text-lg text-gray-600">
            K-means clustering analysis of {totalCustomers.toLocaleString()} customers across {clusterData.length} distinct segments
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
              <p className="text-yellow-800">{error}</p>
            </div>
            <p className="text-sm text-yellow-700 mt-1">Showing sample data for demonstration purposes.</p>
          </div>
        )}

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Customers</p>
                <p className="text-2xl font-bold text-gray-900">{totalCustomers.toLocaleString()}</p>
              </div>
              <Users className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Revenue Potential</p>
                <p className="text-2xl font-bold text-gray-900">${(totalRevenue / 1000000).toFixed(1)}M</p>
              </div>
              <DollarSign className="w-8 h-8 text-green-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Premium Customers</p>
                <p className="text-2xl font-bold text-gray-900">{premiumPercentage.toFixed(1)}%</p>
              </div>
              <Crown className="w-8 h-8 text-yellow-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">At-Risk Customers</p>
                <p className="text-2xl font-bold text-gray-900">{atRiskPercentage.toFixed(1)}%</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg w-fit">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'segments', label: 'Segment Details' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content based on active tab */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Customer Distribution */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Customer Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={clusterData}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="customers"
                    label={({ name, percentage }) => `${name}: ${percentage}%`}
                  >
                    {clusterData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getClusterColor(entry.type)} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value.toLocaleString(), 'Customers']} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Revenue by Segment */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Revenue Potential by Segment</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={clusterData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis tickFormatter={(value) => `${(value/1000000).toFixed(1)}M`} />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Total Value']} />
                  <Bar dataKey="totalValue" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'segments' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {clusterData.map((cluster) => (
              <div key={cluster.cluster} className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
                <div className="flex items-center gap-3 mb-4">
                  <div className={`p-2 rounded-lg`} style={{backgroundColor: getClusterColor(cluster.type) + '20'}}>
                    {getClusterIcon(cluster.type)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{cluster.name}</h3>
                    <p className="text-sm text-gray-600">Cluster {cluster.cluster}</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Customers:</span>
                    <span className="font-medium">{cluster.customers.toLocaleString()} ({cluster.percentage}%)</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Order Value:</span>
                    <span className="font-medium">${cluster.avgOrderValue.toFixed(0)}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Review Score:</span>
                    <span className="font-medium flex items-center gap-1">
                      {cluster.avgReviewScore.toFixed(2)}
                      <Star className="w-4 h-4 text-yellow-500 fill-current" />
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Installments:</span>
                    <span className="font-medium">{cluster.avgInstallments.toFixed(1)}</span>
                  </div>
                  
                  <div className="pt-3 border-t border-gray-100">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium text-gray-900">Annual Value:</span>
                      <span className="font-bold text-lg">${cluster.annualValue}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Strategic Recommendations */}
        <div className="mt-8 bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategic Recommendations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {clusterData.map((cluster) => {
              const rec = getRecommendations(cluster);
              const colorClasses = {
                'green': 'bg-green-50 border-green-200',
                'blue': 'bg-blue-50 border-blue-200',
                'red': 'bg-red-50 border-red-200',
                'orange': 'bg-orange-50 border-orange-200',
                'purple': 'bg-purple-50 border-purple-200'
              };
              
              return (
                <div key={cluster.cluster} className={`p-4 rounded-lg border ${colorClasses[rec.color]}`}>
                  <div className="flex items-center gap-2 mb-2">
                    {getClusterIcon(cluster.type)}
                    <h4 className={`font-semibold text-${rec.color}-900`}>{cluster.name}</h4>
                  </div>
                  <p className={`text-sm text-${rec.color}-800 mb-2`}>
                    <strong>{cluster.percentage}% • ${cluster.avgOrderValue.toFixed(0)} AOV • {cluster.avgReviewScore.toFixed(1)}★ Rating</strong>
                  </p>
                  <p className={`text-sm text-${rec.color}-800`}>{rec.description}</p>
                </div>
              );
            })}
          </div>
          
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-2">Data Source Information:</h4>
            <div className="text-sm text-gray-600">
              <p>• Dashboard automatically loads data from: <code className="bg-gray-200 px-1 rounded">dashboard/public/segment_summary.json</code></p>
              <p>• Updates in real-time when the JSON file is modified</p>
              <p>• Fallback to sample data if file is not accessible</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CustomerSegmentDashboard;