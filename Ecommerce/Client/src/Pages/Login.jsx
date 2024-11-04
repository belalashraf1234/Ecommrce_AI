import { useState } from "react";
import { login } from "../Redux/apiCalls";
import { useDispatch, useSelector } from "react-redux";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
const dispatch=useDispatch();
const {error}=useSelector(state=>state.user);

const handleLoggin=(e)=>{
  e.preventDefault();
  login(dispatch,{username,password})
}

    return (
      <div className="w-screen h-screen bg-gradient-to-r from-white/50 via-white/50 to-white/50 bg-cover bg-center flex items-center justify-center" style={{ backgroundImage: 'url("https://images.pexels.com/photos/6984650/pexels-photo-6984650.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940")' }}>
        <div className="w-4/5 md:w-1/4 p-5 bg-white">
          <h1 className="text-2xl font-light">SIGN IN</h1>
          <form className="flex flex-col">
            <input onChange={(e)=>setUsername(e.target.value)} className="flex-1 min-w-40 my-2 p-2 border border-gray-300" placeholder="username" />
            <input onChange={(e)=>setPassword(e.target.value)} className="flex-1 min-w-40 my-2 p-2 border border-gray-300" placeholder="password" type="password" />
            <button onClick={handleLoggin} className="w-2/5 py-3 my-2 bg-teal-500 text-white cursor-pointer">LOGIN</button>
            {error&&<span>something went Error</span>}
            <a className="my-1 text-xs underline cursor-pointer">DO NOT YOU REMEMBER THE PASSWORD?</a>
            <a className="my-1 text-xs underline cursor-pointer">CREATE A NEW ACCOUNT</a>
          </form>
        </div>
      </div>
    );
  };
  
  export default Login;