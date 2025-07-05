import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

// ← add this line:
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)