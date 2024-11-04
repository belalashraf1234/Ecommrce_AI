import { ShoppingCartOutlined } from '@mui/icons-material';
import SearchIcon from '@mui/icons-material/Search';
import { Badge } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { Link } from 'react-router-dom';
import { logout } from '../Redux/userRedux';
const NavBar = () => {
  const dispatch=useDispatch();
  const user=useSelector(state=>state.user.currentUser);
  const quantity=useSelector(state=>state.cart.quantity);
  return (
    <div className="h-14 flex justify-between items-center overflow-hidden">
      <div className="flex-1  flex items-center">
      <span className="text-xl cursor-pointer ">EN</span>
      <div className='border-solid border border-gray-500 flex items-center ml-6 p-1 '>
        <input type="text" placeholder="Search" className="outline-none bg-transparent border-none" />
        <SearchIcon style={{color:"gray",fontSize:"16px"}} />
      </div>
      </div>
      <div className="flex-1 text-center">
        <Link to={'/'}>
          <h1 className='font-bold '>
          Cliq
        </h1>
        </Link>
      </div>
      <div className="flex-1 flex items-center justify-end">
       { (user&&<>
        <Link>
        <div onClick={()=>dispatch(logout())} className='text-sm cursor-pointer ml-6'>logout</div>
       </Link><Link>
       <div className='text-sm cursor-pointer ml-6'>{user.username}</div>
       </Link>
       </>)||<>
       <Link to={'/register'}>
        <div className='text-sm cursor-pointer ml-6'>Register</div>
        </Link>
        
        <Link to={'/login'}>
        <div className='text-sm cursor-pointer ml-6'>Login</div>
        </Link>
       </>}
        
        <div className='text-sm cursor-pointer ml-6'>
        <Link to={'/cart'}>
        <Badge badgeContent={quantity} color='primary' >
              <ShoppingCartOutlined />
          </Badge>
        </Link>
          
        </div>
      </div>
    </div>
  )
}

export default NavBar
