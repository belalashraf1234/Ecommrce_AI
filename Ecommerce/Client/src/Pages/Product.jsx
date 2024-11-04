import { Remove,Add } from "@mui/icons-material"; 
import Announcment from "../Component/Announcment";
import Footer from "../Component/Footer";
import NavBar from "../Component/NavBar";
import NewsLetter from "../Component/NewsLetter";
import { useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { publicRequest } from "../requestedMethods";
import { addProduct } from "../Redux/cartRedux";
import { useDispatch } from "react-redux";
import CommentSection from "../Component/CommentSection";

const Product = () => {
  const location=useLocation();
  const id=location.pathname.split('/')[2]
  const [product,setProduct]=useState({});
  const [quantity,setQuantity]=useState(1);
  const [color,setColor]=useState(null);
  const [size,setSize]=useState(null);
const Dispatch=useDispatch();
  useEffect(()=>{
    const getProduct=async()=>{
      try{
          const res=await publicRequest.get(`/products/find/${id}`);
          setProduct(res.data);
          console.log(res.data);
      }
      catch(err){
        console.log(err);
    }};
    getProduct();
  },[id]);

const handleClick1=(type)=>{
  if(type==="dec"){
    quantity>1 && setQuantity(quantity-1);
  }
  else{
    setQuantity(quantity+1);
  }
}

const handleClick=()=>{
  console.log("aas");
  Dispatch(addProduct({...product,quantity,color,size}));
 
}

  return (
    <div>
      <NavBar />
      <Announcment />
      <div className="p-12 flex flex-col md:flex-row">
        <div className="flex-1">
          <img
            className="w-full h-[90vh] object-cover md:h-[40vh]"
            src={product.image}
            alt="Product"
          />
        </div>
        <div className="flex-1 p-0 md:p-12">
          <h1 className="font-light text-2xl">{product.title}</h1>
          <p className="my-5">
           {product.description}
          </p>
          <span className="font-thin text-4xl">{product.price}</span>
          <div className="w-full md:w-1/2 my-7 flex justify-between">
            <div className="flex items-center">
              <span className="text-xl font-light">Color</span>
              {product.colors?.map((c)=>{
                return <div key={c} onClick={()=>setColor(c)} style={{backgroundColor:c.color}} className="w-6 h-6 rounded-full  ml-2 cursor-pointer"></div>
              })}
            </div>
            <div className="flex items-center">
              <span className="text-xl font-light">Size</span>
              <select className="ml-2 p-1 border border-gray-300">
              {product.sizes?.map((s)=>{
                return <option onClick={(e)=>setSize(e.target.value)} key={s.sizeId}>{s.size}</option>
              })}
              </select>
            </div>
          </div>
          <div className="w-full md:w-1/2 flex items-center justify-between">
            <div  className="flex items-center font-semibold">
              <Remove onClick={()=>handleClick1("dec")} className="cursor-pointer" />
              <div className="w-8 h-8 border border-teal-500 rounded-lg flex items-center justify-center mx-1">{quantity}</div>
              <Add onClick={()=>handleClick1("inc")} on className="cursor-pointer" />
            </div>
          
            <button onClick={handleClick} className="py-3 px-6 border-2 border-teal-500 bg-white cursor-pointer font-semibold hover:bg-gray-100">
              ADD TO CART
            </button>
          </div>
        </div>
      </div>
      <CommentSection id={id}/>
      <NewsLetter />
      <Footer />
    </div>
  );
};

export default Product;