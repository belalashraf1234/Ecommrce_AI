import React from 'react';
import ReactDOM from 'react-dom/client';

import './index.css';
import App from './App';
import "./Main.css";

import { Provider } from 'react-redux';
import { persistor, store } from './Redux/store';
import { PersistGate } from 'redux-persist/lib/integration/react';
ReactDOM.createRoot(document.getElementById('root')).render(
  
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
   <React.StrictMode>
            <App />
  </React.StrictMode>,
    </PersistGate>
      
  </Provider> 
 
)
