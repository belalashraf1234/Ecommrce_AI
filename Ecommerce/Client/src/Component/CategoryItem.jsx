import { Link } from "react-router-dom";

const CategoryItem = ({ item }) => {
  return (
    <div className="flex-1 m-1 h-70vh relative flex bg-slate-600">
      <Link to={`/products/${item.cat}`}>
        <img src={item.img} className="w-full h-full object-cover" />
        <div className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center ">
          <h1 className="text-white mb-5 ">{item.title}</h1>
          <button className="bg-white text-gray-400 border-none p-2 cursor-pointer font-medium rounded-lg">
            SHOP NOW
          </button>
        </div>
      </Link>
    </div>
  );
};

export default CategoryItem;
