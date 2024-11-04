import { FavoriteBorderOutlined, SearchOutlined, ShoppingCartOutlined } from "@mui/icons-material"
import { Link } from "react-router-dom"


const Product = ({item}) => {
  
  return (
    <div className="group flex-1 m-1 min-w-[400px] h-[350px] flex items-center justify-center bg-[#f5fbfd] relative">
        <div className="w-60 h-60 rounded-full z-0  bg-white absolute">
          
        </div>
      <img src={item.image} className="h-3/4 z-40 " />
      <div className="w-full h-full absolute top-0 left-0 cursor-pointer  group-hover:opacity-100 opacity-0 z-50 flex items-center justify-center  bg-white bg-opacity-20   transition duration-500 ease hover:scale-100 ">
            <div className=" w-12 h-12 rounded-full bg-white items-center flex m-[10px] transition-all duration-500 ease justify-center ">
                <ShoppingCartOutlined/>
            </div>
            <div className=" w-12 h-12 rounded-full bg-white items-center flex m-[10px] transition-all duration-500 ease justify-center">
                
                  <Link to={`/product/${item.productId}`}>
                  <SearchOutlined/>
                  </Link>
                
            </div>
            <div className=" w-12 h-12 rounded-full bg-white items-center flex m-[10px] transition-all duration-500 ease justify-center">
                <FavoriteBorderOutlined/>
            </div>
      </div>
    </div>
  )
}

export default Product
