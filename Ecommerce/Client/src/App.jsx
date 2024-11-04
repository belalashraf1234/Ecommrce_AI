

import { useSelector } from 'react-redux'
import Cart from './Pages/Cart'
import Home from './Pages/Home'
import Login from './Pages/Login'
import Product from './Pages/Product'
import ProductList from './Pages/ProductList'
import Register from './Pages/Register'
import { BrowserRouter as Router, Routes, Route,Navigate} from 'react-router-dom';
const App = () => {
  const user=useSelector(state=>state.user.currentUser)||null;
  
  return (

    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/cart" element={<Cart />} />
        <Route path="/login" element={user? <Navigate to="/" /> :<Login />} />
        <Route path="/product/:id" element={<Product />} />
        <Route path="/products/:category" element={<ProductList />} />
        <Route path="/register" element={user? <Navigate to="/" /> :<Register />} />
      </Routes>
    </Router>
  );
};

export default App
