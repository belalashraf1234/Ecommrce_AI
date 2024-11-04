
import { useEffect, useState } from "react";

import Product from "./Product"
import axios from "axios";
const Products = ({cat,filters,sort}) => {
  const [products,setProducts]=useState([]);
  const [filteredProducts,setFilteredProducts]=useState([]);

  useEffect(()=>{
    const getProucts=async()=>{
      try{
        const res=await axios.get(cat?`http://localhost:5115/products?category=${cat}`
      :`http://localhost:5115/products`)
        setProducts(res.data);

      }
      catch(err){
        console.log(err);
    }
    
    }
    getProucts();
  },[cat])


useEffect(()=>{
  cat&&setFilteredProducts(
    products.filter(product => {
      // Convert the filters object into an array of entries
      return Object.entries(filters).every(([key, value]) => {
          if (key === 'color') {
              // Check if the color exists in the product's colors array
              return product.colors.some(color => color.color === value);
          } else if (key === 'size') {
              // Check if the size exists in the product's sizes array
              return product.sizes.some(size => size.size === value);
          } else {
              // For other keys (not arrays), check directly
              return product[key] === value;
          }
      });
  }

  ))

},[products,cat,filters])
useEffect(()=>{
if(sort==="newest")
{
  setFilteredProducts(prev=>
    [...prev].sort((a,b)=>a.createdAt-b.createdAt))
}
if(sort==="asc")
{
  setFilteredProducts(prev=>
    [...prev].sort((a,b)=>a.price - b.price))
}
if(sort==="desc")
{
  setFilteredProducts(prev=>
    [...prev].sort((a,b)=> b.price - a.price))
}
},[sort])
  
  return (
    <div className="p-5 flex flex-wrap justify-between   ">
      {
        cat?filteredProducts.map((item)=>(
          <Product item={item} key={item.productId}/>
        )):
        products.slice(0,8).map((item)=>(
          <Product item={item} key={item.productId}/>
        ))
      }
    </div>
  )
}

export default Products
