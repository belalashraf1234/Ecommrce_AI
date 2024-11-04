import { loginFailure, loginStart, loginSuccess } from "./userRedux";
import { publicRequest } from "../requestedMethods";
export const register=async (dispatch,user)=>{
  dispatch(loginStart());
  try {
    const res = await publicRequest.post("/register", user);
    dispatch(loginSuccess(res.data));
    console.log(res)
  } catch (err) {
    dispatch(loginFailure());
  }
}
export const login = async (dispatch, user) => {
  dispatch(loginStart());
  try {
    const res = await publicRequest.post("/login", user);
    dispatch(loginSuccess(res.data));
    console.log(res)
  } catch (err) {
    dispatch(loginFailure());
  }
};