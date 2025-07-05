import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';

/* ---------- basic typing ---------- */
interface ModelMetrics {
  name: string;
  trainAUC: number;
  valAUC: number | null;
  testAUC: number;
  f1: number;
}

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  icon: React.FC<React.SVGProps<SVGSVGElement>>;
  color?: 'blue' | 'red' | 'green' | 'yellow' | 'purple';
}

interface TabButtonProps {
  id: string;
  label: string;
  isActive: boolean;
  onClick: (id: string) => void;
}

interface DashboardData {
  generatedAt: string;
  modelComparison: ModelMetrics[];
  featureImportance: { feature: string; importance: number }[];
  optimizationHistory: { trial: number; auc: number }[];
  thresholdOptimization: any[];
  recommendedThreshold: number;
  riskSegmentation: { name: string; count: number; percentage: number; churnRate: number; color: string }[];
  businessImpact: {
    revenueSaved: number;
    retentionCosts: number;
    missedRevenue: number;
    netBenefit: number;
  };
  monthlyMetrics: {
    month: string;
    churnRate: number;
    retentionRate: number;
    totalCustomers: number;
    churnedCustomers: number;
  }[];
  customerSegments: {
    name: string;
    count: number;
    avgOrderValue: number;
    churnRate: number;
    description: string;
  }[];
}

/* ---------- component ---------- */
const CustomerChurnDashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [selectedModel, setSelectedModel] = useState('LightGBM');
  const [activeTab, setActiveTab] = useState<'overview' | 'models' | 'features' | 'segments' | 'business'>('overview');

  useEffect(() => {
    // Load data from the JSON file
    const loadData = async () => {
      try {
        const response = await fetch('/churn_dashboard_data.json');
        const jsonData = await response.json();
        setData(jsonData);
      } catch (error) {
        console.error('Error loading data:', error);
        // Use fallback data if file doesn't exist
        setData({
          generatedAt: "2025-07-05T05:27:27Z",
          modelComparison: [
            { name: "Logistic Regression", trainAUC: 0.848, valAUC: 0.846, testAUC: 0.829, f1: 0.723 },
            { name: "LightGBM", trainAUC: 0.905, valAUC: 0.882, testAUC: 0.866, f1: 0.769 },
            { name: "XGBoost", trainAUC: 0.901, valAUC: 0.881, testAUC: 0.865, f1: 0.766 },
            { name: "LightGBM (Optuna)", trainAUC: 0.915, valAUC: null, testAUC: 0.866, f1: 0.778 }
          ],
          featureImportance: [
            { feature: "payment_value", importance: 4047 },
            { feature: "delivery_days_diff_mean", importance: 3426 },
            { feature: "delivery_actual_days_mean", importance: 2802 },
            { feature: "payment_installments", importance: 1317 },
            { feature: "review_review_score_mean", importance: 641 }
          ],
          optimizationHistory: [
            { trial: 1, auc: 0.872 }, { trial: 2, auc: 0.872 }, { trial: 3, auc: 0.878 },
            { trial: 4, auc: 0.876 }, { trial: 5, auc: 0.875 }, { trial: 6, auc: 0.876 },
            { trial: 7, auc: 0.872 }, { trial: 8, auc: 0.864 }, { trial: 9, auc: 0.876 },
            { trial: 10, auc: 0.875 }, { trial: 11, auc: 0.878 }, { trial: 12, auc: 0.878 }
          ],
          thresholdOptimization: [],
          recommendedThreshold: 0.05,
          riskSegmentation: [
            { name: 'High Risk', count: 40880, percentage: 41.1, churnRate: 0.99, color: '#ef4444' },
            { name: 'Medium Risk', count: 30850, percentage: 31.0, churnRate: 0.65, color: '#f59e0b' },
            { name: 'Low Risk', count: 27720, percentage: 27.9, churnRate: 0.36, color: '#10b981' }
          ],
          businessImpact: {
            revenueSaved: 18305733,
            retentionCosts: 1236500,
            missedRevenue: 13361481,
            netBenefit: 3707752
          },
          monthlyMetrics: [
            { month: "2024-07", churnRate: 0.35, retentionRate: 0.65, totalCustomers: 99450, churnedCustomers: 34807 },
            { month: "2024-08", churnRate: 0.335, retentionRate: 0.665, totalCustomers: 99550, churnedCustomers: 33349 },
            { month: "2024-09", churnRate: 0.32, retentionRate: 0.68, totalCustomers: 99650, churnedCustomers: 31887 },
            { month: "2024-10", churnRate: 0.305, retentionRate: 0.695, totalCustomers: 99750, churnedCustomers: 30423 },
            { month: "2024-11", churnRate: 0.29, retentionRate: 0.71, totalCustomers: 99850, churnedCustomers: 28956 },
            { month: "2024-12", churnRate: 0.275, retentionRate: 0.725, totalCustomers: 99950, churnedCustomers: 27486 }
          ],
          customerSegments: [
            { name: "High-Value Loyal", count: 14917, avgOrderValue: 160.2, churnRate: 0.05, description: "Premium customers with high AOV and low churn risk" },
            { name: "Regular Buyers", count: 44752, avgOrderValue: 89, churnRate: 0.18, description: "Consistent customers with average purchase behavior" },
            { name: "Occasional Shoppers", count: 24862, avgOrderValue: 62.3, churnRate: 0.35, description: "Infrequent buyers with higher churn risk" },
            { name: "New Customers", count: 14917, avgOrderValue: 53.4, churnRate: 0.45, description: "Recently acquired customers still building loyalty" }
          ]
        });
      }
    };

    loadData();
  }, []);

  if (!data) {
    return <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-xl">Loading dashboard...</div>
    </div>;
  }

  /* ---------------- small helpers ---------------- */
  const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, icon: Icon, color = 'blue' }) => {
    const getColorClass = (c: string) => ({
      blue: 'bg-blue-100 text-blue-600',
      red: 'bg-red-100 text-red-600',
      green: 'bg-green-100 text-green-600',
      yellow: 'bg-yellow-100 text-yellow-600',
      purple: 'bg-purple-100 text-purple-600',
    }[c] ?? 'bg-gray-100 text-gray-600');

    return (
      <div className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 border border-gray-100 transform hover:-translate-y-1">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
            {change && (
              <p className={`text-sm mt-1 flex items-center ${change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
                {change.startsWith('+') ? <TrendingUpIcon className="w-4 h-4" /> : <AlertTriangleIcon className="w-4 h-4" />}
                <span className="ml-1">{change}</span>
              </p>
            )}
          </div>
          <div className={`p-3 rounded-full ${getColorClass(color)} transition-transform duration-300 group-hover:scale-110`}>
            <Icon className="w-6 h-6" />
          </div>
        </div>
      </div>
    );
  };

  const TabButton: React.FC<TabButtonProps> = ({ id, label, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`px-5 py-2.5 text-sm font-medium rounded-lg transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
        isActive ? 'bg-blue-600 text-white shadow-md' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
      }`}
    >
      {label}
    </button>
  );

  // Custom SVG Icons
  const ActivityIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
    </svg>
  );

  const TrendingUpIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
      <polyline points="17 6 23 6 23 12"></polyline>
    </svg>
  );

  const UsersIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
      <circle cx="9" cy="7" r="4"></circle>
      <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
    </svg>
  );

  const AlertTriangleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
      <line x1="12" y1="9" x2="12" y2="13"></line>
      <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
  );

  const DollarSignIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="1" x2="12" y2="23"></line>
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
    </svg>
  );

  const TargetIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <circle cx="12" cy="12" r="6"></circle>
      <circle cx="12" cy="12" r="2"></circle>
    </svg>
  );

  const ZapIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
    </svg>
  );

  const BrainIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"></path>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"></path>
    </svg>
  );

  // Calculate some dynamic metrics
  const selectedModelData = data.modelComparison.find(m => m.name === selectedModel) || data.modelComparison[1];
  const totalCustomers = data.riskSegmentation.reduce((sum, seg) => sum + seg.count, 0);
  const highRiskCustomers = data.riskSegmentation.find(seg => seg.name === 'High Risk')?.count || 0;
  const avgChurnRate = data.monthlyMetrics.length > 0 ? 
    data.monthlyMetrics.reduce((sum, m) => sum + m.churnRate, 0) / data.monthlyMetrics.length : 0;

  /* ---------------- render tabs ---------------- */
  const renderOverviewTab = () => (
    <div className="space-y-8">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Model Accuracy"
          value={`${(selectedModelData.testAUC * 100).toFixed(1)}%`}
          change="+2.3%"
          icon={TargetIcon}
          color="blue"
        />
        <MetricCard
          title="High Risk Customers"
          value={highRiskCustomers.toLocaleString()}
          change="+156"
          icon={AlertTriangleIcon}
          color="red"
        />
        <MetricCard
          title="Revenue at Risk"
          value={`$${(data.businessImpact.revenueSaved / 1e6).toFixed(2)}M`}
          change="-$120K"
          icon={DollarSignIcon}
          color="yellow"
        />
        <MetricCard
          title="Predicted Churn Rate"
          value={`${(avgChurnRate * 100).toFixed(1)}%`}
          change="+0.8%"
          icon={TrendingUpIcon}
          color="purple"
        />
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Trends */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Churn Trends</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.monthlyMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="churnRate" stroke="#ef4444" strokeWidth={2} name="Churn Rate" />
                <Line type="monotone" dataKey="retentionRate" stroke="#10b981" strokeWidth={2} name="Retention Rate" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Distribution</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.riskSegmentation}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name} (${percentage}%)`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {data.riskSegmentation.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Model Activity</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-gray-100">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
              <span className="text-sm text-gray-900">Model retrained successfully</span>
            </div>
            <span className="text-sm text-gray-500">2 hours ago</span>
          </div>
          <div className="flex items-center justify-between py-3 border-b border-gray-100">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
              <span className="text-sm text-gray-900">{totalCustomers.toLocaleString()} new predictions generated</span>
            </div>
            <span className="text-sm text-gray-500">4 hours ago</span>
          </div>
          <div className="flex items-center justify-between py-3 border-b border-gray-100">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></div>
              <span className="text-sm text-gray-900">High-risk alert: {Math.floor(highRiskCustomers / 1000)} customers flagged</span>
            </div>
            <span className="text-sm text-gray-500">6 hours ago</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderModelsTab = () => (
    <div className="space-y-8">
      {/* Model Comparison */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Comparison</h3>
        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.modelComparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="trainAUC" fill="#3b82f6" name="Train AUC" />
              <Bar dataKey="testAUC" fill="#f59e0b" name="Test AUC" />
              <Bar dataKey="f1" fill="#10b981" name="F1 Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Optimization History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Optuna Optimization History</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.optimizationHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="trial" />
                <YAxis domain={['dataMin - 0.01', 'dataMax + 0.01']} />
                <Tooltip />
                <Line type="monotone" dataKey="auc" stroke="#3b82f6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Selected Model Metrics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50">
              <span className="text-sm text-gray-600">Test AUC Score</span>
              <span className="text-lg font-semibold text-gray-900">{selectedModelData.testAUC.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50">
              <span className="text-sm text-gray-600">Train AUC Score</span>
              <span className="text-lg font-semibold text-gray-900">{selectedModelData.trainAUC.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50">
              <span className="text-sm text-gray-600">F1 Score</span>
              <span className="text-lg font-semibold text-gray-900">{selectedModelData.f1.toFixed(3)}</span>
            </div>
            {selectedModelData.valAUC && (
              <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50">
                <span className="text-sm text-gray-600">Validation AUC</span>
                <span className="text-lg font-semibold text-gray-900">{selectedModelData.valAUC.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const renderFeaturesTab = () => {
    const topFeatures = data.featureImportance.slice(0, 10);
    const normalizedFeatures = topFeatures.map(f => ({
      ...f,
      normalizedImportance: f.importance / Math.max(...topFeatures.map(x => x.importance))
    }));

    return (
      <div className="space-y-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance Analysis</h3>
          <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={normalizedFeatures} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={200} />
                <Tooltip />
                <Bar dataKey="importance" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Features</h3>
            <div className="space-y-3">
              {topFeatures.slice(0, 5).map((feature, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                  <span className="text-sm text-gray-600 capitalize">{feature.feature.replace(/_/g, ' ')}</span>
                  <span className="text-sm font-medium text-blue-600">{feature.importance}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Insights</h3>
            <div className="space-y-4">
              <div className="p-4 bg-red-50 rounded-lg">
                <h4 className="font-medium text-red-900">Payment Behavior</h4>
                <p className="text-sm text-red-700 mt-1">Payment value and installments are key churn indicators</p>
              </div>
              <div className="p-4 bg-yellow-50 rounded-lg">
                <h4 className="font-medium text-yellow-900">Delivery Performance</h4>
                <p className="text-sm text-yellow-700 mt-1">Delivery delays significantly impact customer satisfaction</p>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-medium text-green-900">Review Scores</h4>
                <p className="text-sm text-green-700 mt-1">Customer reviews provide strong predictive signals</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Segments tab  (drop this straight into your component file)
  // ─────────────────────────────────────────────────────────────────────────
  const renderSegmentsTab = () => (
    <div className="space-y-8">
      {/* ── Risk–level cards ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {data.riskSegmentation.map((segment, index) => (
          <div
            key={index}
            className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                {segment.name}
              </h3>
              <div
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: segment.color }}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Customers</span>
                <span className="text-sm font-medium">
                  {segment.count.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Percentage</span>
                <span className="text-sm font-medium">{segment.percentage}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Churn Rate</span>
                <span className="text-sm font-medium">
                  {(segment.churnRate * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Detailed segment table ───────────────────────────────────── */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Customer Segments Analysis
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-medium text-gray-900">
                  Segment
                </th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">
                  Customers
                </th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">
                  Avg Order Value
                </th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">
                  Churn Rate
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-900">
                  Description
                </th>
              </tr>
            </thead>

            <tbody>
              {data.customerSegments.map((segment, index) => (
                <tr
                  key={index}
                  className="border-b border-gray-100 hover:bg-gray-50"
                >
                  <td className="py-3 px-4 font-medium text-gray-900">
                    {segment.name}
                  </td>
                  <td className="py-3 px-4 text-right text-gray-600">
                    {segment.count.toLocaleString()}
                  </td>
                  <td className="py-3 px-4 text-right text-gray-600">
                    ${segment.avgOrderValue.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        segment.churnRate < 0.1
                          ? 'bg-green-100 text-green-800'
                          : segment.churnRate < 0.3
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {(segment.churnRate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-600 text-sm">
                    {segment.description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  // ─────────────────────────────────────────────────────────────────────────
  //  Recommendations  /  Business Insights tab
  // ─────────────────────────────────────────────────────────────────────────
  const renderRecommendationsTab = () => {
    /* 1 ▸ grab the JSON slice that matches the `recommendedThreshold` */
    const bestThreshold =
      data.thresholdOptimization.find(
        (t) => t.threshold === data.recommendedThreshold
      ) ?? data.thresholdOptimization[0]; // fallback if not found

    /* 2 ▸ build a confusion-matrix object from that slice */
    const confusionMatrix = {
      truePositives: bestThreshold.truePositives,
      falsePositives: bestThreshold.falsePositives,
      falseNegatives: bestThreshold.falseNegatives,
      trueNegatives: bestThreshold.trueNegatives,
    };

    /* 3 ▸ derive precision & recall */
    const precision =
      confusionMatrix.truePositives /
      (confusionMatrix.truePositives + confusionMatrix.falsePositives);
    const recall =
      confusionMatrix.truePositives /
      (confusionMatrix.truePositives + confusionMatrix.falseNegatives);

    /* 4 ▸ we’ll list every threshold that came from the backend */
    const thresholdOptions = data.thresholdOptimization;

    /* 5 ▸ render */
    return (
      <div className="space-y-8">
        {/* ── Header ─────────────────────────────────────────────── */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-xl">
          <h2 className="text-2xl font-bold mb-2">
            🔍 Business Insights &amp; Recommendations
          </h2>
          <p className="text-blue-100">
            Actionable insights from your churn-prediction model
          </p>
        </div>

        {/* ── Model Performance Summary ─────────────────────────── */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            📊 Model Performance Summary&nbsp;—&nbsp;{selectedModel}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* AUC */}
            <div className="bg-green-50 p-4 rounded-lg text-center">
              <div className="text-sm text-green-600 font-medium">Test AUC</div>
              <div className="text-2xl font-bold text-green-900">
                {selectedModelData.testAUC.toFixed(3)}
              </div>
              <div className="text-sm text-green-700">Excellent</div>
            </div>
            {/* Precision */}
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-sm text-blue-600 font-medium">Precision</div>
              <div className="text-2xl font-bold text-blue-900">
                {(precision * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-blue-700">
                Of predicted churners
              </div>
            </div>
            {/* Recall */}
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-sm text-purple-600 font-medium">Recall</div>
              <div className="text-2xl font-bold text-purple-900">
                {(recall * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-purple-700">
                Of actual churners found
              </div>
            </div>
            {/* F1 */}
            <div className="bg-orange-50 p-4 rounded-lg text-center">
              <div className="text-sm text-orange-600 font-medium">F1-Score</div>
              <div className="text-2xl font-bold text-orange-900">
                {selectedModelData.f1.toFixed(3)}
              </div>
              <div className="text-sm text-orange-700">Balanced measure</div>
            </div>
          </div>
        </div>

        {/* Key Churn Drivers */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            🎯 Key Churn Drivers
          </h3>
          <div className="space-y-3">
            {data.featureImportance.slice(0, 5).map((feature, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-900 capitalize">
                    {feature.feature.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900">High predictive power</div>
                  <div className="text-xs text-gray-600">Importance: {feature.importance.toFixed(0)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Customer Risk Segmentation */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            📈 Customer Risk Segmentation
          </h3>
          <div className="space-y-4">
            {data.riskSegmentation.map((segment, index) => (
              <div key={index} className="flex items-center justify-between p-4 rounded-lg" style={{ backgroundColor: segment.color + '20' }}>
                <div className="flex items-center">
                  <div className="w-4 h-4 rounded-full mr-3" style={{ backgroundColor: segment.color }}></div>
                  <div>
                    <div className="font-medium text-gray-900">{segment.name}</div>
                    <div className="text-sm text-gray-600">
                      {segment.count.toLocaleString()} customers ({segment.percentage}%)
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-gray-900">Churn Rate: {(segment.churnRate * 100).toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Business Impact Analysis */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            💰 Business Impact Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-green-50 p-4 rounded-lg text-center">
              <div className="text-sm text-green-600 font-medium">True Positives</div>
              <div className="text-2xl font-bold text-green-900">{confusionMatrix.truePositives.toLocaleString()}</div>
              <div className="text-xs text-green-700">Correctly identified churners</div>
            </div>
            <div className="bg-red-50 p-4 rounded-lg text-center">
              <div className="text-sm text-red-600 font-medium">False Positives</div>
              <div className="text-2xl font-bold text-red-900">{confusionMatrix.falsePositives.toLocaleString()}</div>
              <div className="text-xs text-red-700">Incorrectly flagged</div>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg text-center">
              <div className="text-sm text-yellow-600 font-medium">False Negatives</div>
              <div className="text-2xl font-bold text-yellow-900">{confusionMatrix.falseNegatives.toLocaleString()}</div>
              <div className="text-xs text-yellow-700">Missed churners</div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-sm text-blue-600 font-medium">True Negatives</div>
              <div className="text-2xl font-bold text-blue-900">{confusionMatrix.trueNegatives.toLocaleString()}</div>
              <div className="text-xs text-blue-700">Correctly identified loyal</div>
            </div>
          </div>
        </div>

      {/* ── Threshold Optimisation ───────────────────────────── */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            🎯 Threshold Optimization
          </h3>
          <div className="space-y-3">
            {thresholdOptions.map((opt, idx) => (
              <div
                key={idx}
                className={`p-4 rounded-lg border-2 ${
                  opt.threshold === data.recommendedThreshold
                    ? 'border-green-300 bg-green-50'
                    : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="font-medium text-gray-900">
                      Threshold {opt.threshold}
                    </span>

                    {opt.threshold === data.recommendedThreshold && (
                      <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full font-medium">
                        Recommended
                      </span>
                    )}
                  </div>

                  <div className="text-right">
                    <div className="font-semibold text-gray-900">
                      Net Benefit: ${opt.netBenefit.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600">
                      TP: {opt.truePositives}, FP: {opt.falsePositives}, FN:{' '}
                      {opt.falseNegatives}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Strategic Recommendations */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            🎯 Strategic Recommendations
          </h3>
          <div className="space-y-6">
            
            {/* Immediate Actions */}
            <div className="border-l-4 border-red-400 pl-4">
              <h4 className="font-semibold text-red-900 mb-2">1. IMMEDIATE ACTIONS</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• Deploy model to score all customers monthly</li>
                <li>• Create automated alerts for high-risk customers (≥0.3 probability)</li>
                <li>• Implement tiered retention strategies based on risk segments</li>
              </ul>
            </div>

            {/* Retention Strategies */}
            <div className="border-l-4 border-blue-400 pl-4">
              <h4 className="font-semibold text-blue-900 mb-2">2. RETENTION STRATEGIES</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>High Risk:</strong> Personal outreach, special offers, account manager assignment</li>
                <li>• <strong>Medium Risk:</strong> Targeted email campaigns, loyalty program enrollment</li>
                <li>• <strong>Low Risk:</strong> Standard engagement, cross-selling opportunities</li>
              </ul>
            </div>

            {/* Feature-Based Actions */}
            <div className="border-l-4 border-purple-400 pl-4">
              <h4 className="font-semibold text-purple-900 mb-2">3. FEATURE-BASED ACTIONS</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Payment Value:</strong> Implement value-based retention strategies and upselling programs</li>
                <li>• <strong>Delivery Performance:</strong> Create time-based triggers for proactive customer outreach</li>
                <li>• <strong>Review Scores:</strong> Implement customer feedback loop and service recovery protocols</li>
              </ul>
            </div>

            {/* Operational Improvements */}
            <div className="border-l-4 border-green-400 pl-4">
              <h4 className="font-semibold text-green-900 mb-2">4. OPERATIONAL IMPROVEMENTS</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• Establish monthly model retraining schedule</li>
                <li>• Create customer success team focused on high-risk accounts</li>
                <li>• Implement A/B testing for retention campaign effectiveness</li>
                <li>• Develop early warning system for rapid churn risk changes</li>
              </ul>
            </div>

            {/* Measurement & Monitoring */}
            <div className="border-l-4 border-yellow-400 pl-4">
              <h4 className="font-semibold text-yellow-900 mb-2">5. MEASUREMENT & MONITORING</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• Track retention campaign success rates by risk segment</li>
                <li>• Monitor model performance with monthly accuracy reports</li>
                <li>• Calculate ROI of retention efforts vs. acquisition costs</li>
                <li>• Set up automated model performance alerts</li>
              </ul>
            </div>

          </div>
        </div>

        {/* Executive Summary */}
        <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-4">💡 Executive Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold">${(data.businessImpact.netBenefit / 1e6).toFixed(2)}M</div>
              <div className="text-sm text-green-100">Potential Net Benefit</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{highRiskCustomers.toLocaleString()}</div>
              <div className="text-sm text-green-100">High-Risk Customers</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{(selectedModelData.testAUC * 100).toFixed(1)}%</div>
              <div className="text-sm text-green-100">Model Accuracy</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderBusinessTab = () => (
    <div className="space-y-8">
      {/* Business Impact Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Revenue Saved"
          value={`$${(data.businessImpact.revenueSaved / 1e6).toFixed(2)}M`}
          change="+15.2%"
          icon={DollarSignIcon}
          color="green"
        />
        <MetricCard
          title="Retention Costs"
          value={`$${(data.businessImpact.retentionCosts / 1e6).toFixed(2)}M`}
          change="+8.1%"
          icon={TargetIcon}
          color="blue"
        />
        <MetricCard
          title="Missed Revenue"
          value={`$${(data.businessImpact.missedRevenue / 1e6).toFixed(2)}M`}
          change="-12.3%"
          icon={AlertTriangleIcon}
          color="red"
        />
        <MetricCard
          title="Net Benefit"
          value={`$${(data.businessImpact.netBenefit / 1e6).toFixed(2)}M`}
          change="+22.5%"
          icon={TrendingUpIcon}
          color="purple"
        />
      </div>

      {/* ROI Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">ROI Analysis</h3>
          <div className="space-y-4">
            <div className="p-4 bg-green-50 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-sm text-green-800">Investment</span>
                <span className="text-lg font-semibold text-green-900">
                  ${(data.businessImpact.retentionCosts / 1e6).toFixed(2)}M
                </span>
              </div>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-800">Return</span>
                <span className="text-lg font-semibold text-blue-900">
                  ${(data.businessImpact.netBenefit / 1e6).toFixed(2)}M
                </span>
              </div>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-sm text-purple-800">ROI</span>
                <span className="text-lg font-semibold text-purple-900">
                  {((data.businessImpact.netBenefit / data.businessImpact.retentionCosts) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommended Actions</h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 rounded-lg border-l-4 border-red-400">
              <h4 className="font-medium text-red-900">High Priority</h4>
              <p className="text-sm text-red-700 mt-1">Focus on {data.riskSegmentation[0].count.toLocaleString()} high-risk customers</p>
            </div>
            <div className="p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-400">
              <h4 className="font-medium text-yellow-900">Medium Priority</h4>
              <p className="text-sm text-yellow-700 mt-1">Implement targeted campaigns for medium-risk segment</p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-400">
              <h4 className="font-medium text-green-900">Low Priority</h4>
              <p className="text-sm text-green-700 mt-1">Monitor and maintain satisfaction for low-risk customers</p>
            </div>
          </div>
        </div>
      </div>

      {/* Financial Breakdown */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Financial Impact Breakdown</h3>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={[
              { name: 'Revenue Saved', value: data.businessImpact.revenueSaved / 1e6, color: '#10b981' },
              { name: 'Retention Costs', value: data.businessImpact.retentionCosts / 1e6, color: '#3b82f6' },
              { name: 'Missed Revenue', value: data.businessImpact.missedRevenue / 1e6, color: '#ef4444' },
              { name: 'Net Benefit', value: data.businessImpact.netBenefit / 1e6, color: '#8b5cf6' }
            ]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value) => [`$${value.toFixed(2)}M`, 'Amount']} />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <BrainIcon />
              <h1 className="ml-3 text-xl font-bold text-gray-900">Customer Churn Analytics</h1>
            </div>
            <div className="flex items-center space-x-4">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {data.modelComparison.map(model => (
                  <option key={model.name} value={model.name}>{model.name}</option>
                ))}
              </select>
              <span className="text-sm text-gray-500">
                Last updated: {new Date(data.generatedAt).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 py-4">
            <TabButton id="overview" label="Overview" isActive={activeTab === 'overview'} onClick={setActiveTab} />
            <TabButton id="models" label="Models" isActive={activeTab === 'models'} onClick={setActiveTab} />
            <TabButton id="features" label="Features" isActive={activeTab === 'features'} onClick={setActiveTab} />
            <TabButton id="segments" label="Segments" isActive={activeTab === 'segments'} onClick={setActiveTab} />
            <TabButton id="business" label="Business Impact" isActive={activeTab === 'business'} onClick={setActiveTab} />
            <TabButton id="recommendations" label="Recommendations" isActive={activeTab === 'recommendations'} onClick={setActiveTab} />
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && renderOverviewTab()}
        {activeTab === 'models' && renderModelsTab()}
        {activeTab === 'features' && renderFeaturesTab()}
        {activeTab === 'segments' && renderSegmentsTab()}
        {activeTab === 'business' && renderBusinessTab()}
        {activeTab === 'recommendations' && renderRecommendationsTab()}
      </main>
    </div>
  );
};

export default CustomerChurnDashboard;