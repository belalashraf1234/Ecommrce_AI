import { useState } from "react";
import { useLocation } from "react-router-dom";
import NavBar from "../Component/NavBar";
import Announcment from "../Component/Announcment";
import Products from "../Component/Products";
import NewsLetter from "../Component/NewsLetter";
import Footer from "../Component/Footer";


const ProductList = () => {
  const location=useLocation();
  const cat=location.pathname.split("/")[2];
  const [filter,setFilter]=useState({});
  const [sort,setSort]=useState("newest");

const handleChange = (e) => {
    const value=e.target.value;
    setFilter({...filter,
      [e.target.name]:value
    })
  }
  return (
    <div>
      <NavBar />
      <Announcment />
      <h1 className="m-5 text-2xl">Dresses</h1>
      <div className="flex justify-between m-5 flex-col md:flex-row">
        <div className="flex flex-col md:flex-row m-5">
          <span className="text-xl font-semibold mb-2 md:mb-0 md:mr-5">Filter Products:</span>
          <select name="color" className="p-2 mb-2 md:mb-0 md:mr-5" onChange={handleChange}>
            <option disabled selected>
              Color
            </option>
            <option >white</option>
            <option>black</option>
            <option>red</option>
            <option>blue</option>
            <option>yellow</option>
            <option>green</option>
          </select>
          <select name="size" className="p-2" onChange={handleChange}>
            <option disabled selected>
              Size
            </option>
            <option>XS</option>
            <option>S</option>
            <option>M</option>
            <option>L</option>
            <option>XL</option>
          </select>
        </div>
        <div className="flex flex-col md:flex-row m-5">
          <span className="text-xl font-semibold mb-2 md:mb-0 md:mr-5">Sort Products:</span>
          <select onChange={(e)=>setSort(e.target.value)} className="p-2">
            <option  value="newest" selected>Newest</option>
            <option value='asc'>Price (asc)</option>
            <option value='desc'>Price (desc)</option>
          </select>
        </div>
      </div>
      <Products cat={cat} filters={filter} sort={sort} />
      <NewsLetter />
      <Footer />
    </div>
  );
};

export default ProductList;