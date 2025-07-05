import { useState } from 'react';
import CustomerChurnDashboard from './components/CustomerChurnDashboard';
import CustomerSegmentDashboard from './components/CustomerSegmentDashboard';

const App: React.FC = () => {
  const [view, setView] = useState<'churn' | 'segment'>('churn');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* very bare-bones nav */}
      <nav className="bg-white shadow-sm border-b px-6 py-3 flex space-x-4 sticky top-0 z-20">
        <button
          onClick={() => setView('churn')}
          className={`px-4 py-2 rounded-lg text-sm font-medium
            ${view === 'churn'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
        >
          Churn dashboard
        </button>
        <button
          onClick={() => setView('segment')}
          className={`px-4 py-2 rounded-lg text-sm font-medium
            ${view === 'segment'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
        >
          Segment dashboard
        </button>
      </nav>

      {/* render whichever dashboard is active */}
      {view === 'churn' ? <CustomerChurnDashboard /> : <CustomerSegmentDashboard />}
    </div>
  );
};

export default App;