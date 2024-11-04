import axios from "axios";
const BaseUrl = "http://localhost:5115/";

export const publicRequest=axios.create({
    baseURL:BaseUrl,
});

export const userRequest=axios.create(
    {
        baseURL:BaseUrl,
       
    }
);