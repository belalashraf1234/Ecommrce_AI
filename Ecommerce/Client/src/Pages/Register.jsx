import { useDispatch, useSelector } from "react-redux";
import { register } from "../Redux/apiCalls";
import { useState } from "react";

const Register = () => {
  const [name, setName] = useState('');
  const [lastName, setLastName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordError, setPasswordError] = useState(false);
  const {error}=useSelector(state=>state.user);
  const dispatch=useDispatch();
  const handleRegister = (e) => {
    e.preventDefault();
     if(password!==confirmPassword){
      setPasswordError(true);
      return;
     }
     register(dispatch,{name,lastName,username,email,password});

  }

    return (
    <div>
       <div className="w-screen h-screen bg-gradient-to-r from-white/50 to-white/50 bg-cover bg-center flex items-center justify-center" style={{ backgroundImage: 'url("https://images.pexels.com/photos/6984661/pexels-photo-6984661.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940")' }}>
       <div className="w-11/12 md:w-2/5 p-5 bg-white">
          <h1 className="text-2xl font-light">CREATE AN ACCOUNT</h1>
          <form className="flex flex-wrap">
            <input onChange={(e)=>setName(e.target.value)} className="flex-1 min-w-2/5 m-2 p-2" placeholder="name" />
            <input onChange={(e)=>setLastName(e.target.value)} className="flex-1 min-w-2/5 m-2 p-2" placeholder="last name" />
            <input onChange={(e)=>setUsername(e.target.value)} className="flex-1 min-w-2/5 m-2 p-2" placeholder="username" />
            <input onChange={(e)=>setEmail(e.target.value)} className="flex-1 min-w-2/5 m-2 p-2" placeholder="email" />
            <input type="password" onChange={(e)=>setPassword(e.target.value)} className="flex-1 min-w-2/5 m-2 p-2" placeholder="password" / >
            <input type="password" onChange={(e)=>setConfirmPassword(e.target.value)} className="flex-1 min-w-2/5 m-2 p-2" placeholder="confirm password" />
            <span className="text-xs m-2">
              By creating an account, I consent to the processing of my personal data in accordance with the <b>PRIVACY POLICY</b>
            </span>
            <button onClick={handleRegister} className="w-2/5 border-none p-3 bg-teal-500 text-white cursor-pointer m-2">CREATE</button>
            {error&&<span>something went Error</span>}
            {passwordError&&<span>Check Passowrd again </span>}
          </form>
        </div>
      </div>
    </div>
    );
  };
  
  export default Register;