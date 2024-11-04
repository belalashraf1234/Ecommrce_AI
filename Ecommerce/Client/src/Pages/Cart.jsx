import { Add, Remove } from "@mui/icons-material";
import Announcment from "../Component/Announcment";
import Footer from "../Component/Footer"; 
import NavBar from "../Component/NavBar"; 
import { useSelector } from "react-redux";

const Cart = () => {
const cart = useSelector((state) => state.cart);
console.log(cart);
  return (
    <div>
      <NavBar />
      <Announcment />
      <div className="p-5 md:p-10">
        <h1 className="text-center font-light text-2xl">YOUR BAG</h1>
        <div className="flex items-center justify-between py-5">
          <button className="py-2 px-5 font-semibold cursor-pointer border border-black">
            CONTINUE SHOPPING
          </button>
          <div className="hidden md:flex">
            <span className="underline cursor-pointer mx-3">Shopping Bag(2)</span>
            <span className="underline cursor-pointer mx-3">Your Wishlist (0)</span>
          </div>
          <button className="py-2 px-5 font-semibold cursor-pointer bg-black text-white">
            CHECKOUT NOW
          </button>
        </div>
        <div className="flex flex-col md:flex-row justify-between">
          <div className="flex-[2]">
            {
              cart.products.map((product) => {
                return (
                  <div key={product.productId}>
                  <div key={product.productId}>
                 <div className="flex flex-col md:flex-row justify-between mb-5">
              <div className="flex-2 flex">
                <img
                  className="w-48"
                  src={product.image}
                />
                <div className="p-5 flex flex-col justify-around">
                  <span>
                    <b>Product:</b> {product.title}
                  </span>
                  <span>
                    <b>ID:</b> {product.productId}
                  </span>
                  <div  style={{backgroundColor:product.color}} className="w-5 h-5 rounded-full "></div>
                  <span>
                    <b>Size:</b> {product.size}
                  </span>
                </div>
              </div>
              <div className="flex-1 flex flex-col items-center justify-center">
                <div className="flex items-center mb-5">
                  <Add className="cursor-pointer" />
                  <div className="text-xl mx-3">2</div>
                  <Remove className="cursor-pointer" />
                </div>
                <div className="text-3xl font-light">{product.price}</div>
              </div>
            </div>
            <hr className="bg-gray-200 border-none h-px" />

                 </div>
                  </div>
                
                )
              })
            }
            
          </div>
          <div className="flex-1 border border-gray-300 rounded-lg p-5 h-80">
            <h1 className="text-2xl font-light">ORDER SUMMARY</h1>
            <div className="my-7 flex justify-between">
              <span>Subtotal</span>
              <span>{cart.total}</span>
            </div>
            <div className="my-7 flex justify-between">
              <span>Estimated Shipping</span>
              <span>$ 5.90</span>
            </div>
            <div className="my-7 flex justify-between">
              <span>Shipping Discount</span>
              <span>$ -5.90</span>
            </div>
            <div className="my-7 flex justify-between font-medium text-xl">
              <span>Total</span>
              <span>{cart.total}</span>
            </div>
            <button className="w-full py-2 bg-black text-white font-semibold">
              CHECKOUT NOW
            </button>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Cart;