import { ArrowLeftOutlined, ArrowRightOutlined } from "@mui/icons-material"
import { sliderItems } from "../Pages/data"
import { useState } from "react"
const Slider = () => {
    const [slideIndex,setSlideIndex] = useState(0);
    const handelClick=(direction)=>{
        if(direction==="left"){
            setSlideIndex(slideIndex>0?slideIndex-1:2);
        }
        else{
            setSlideIndex(slideIndex<2?slideIndex+1:0);
        }
    }
    const transform = {
        transform:`translateX(${slideIndex * -100}vw)`
    }
  return (

    <div className="w-full h-screen flex relative   bg-orange-500 overflow-hidden">
      
        {sliderItems.map((item)=>(
            <div style={transform} key={item.id} className="h-full  flex transition duration-[1500ms] ease">
            <div className="h-screen w-screen flex items-center ">
                    
                    <div className="w-full h-full bg-red-400">
                    <img className=" h-screen w-screen"  src={item.img}/>
                    </div>
                   
            </div>
      </div>
        ))}
      <div onClick={()=>handelClick("right")} className="w-12 h-12 bg-slate-100 cursor-pointer opacity-50 flex items-center justify-center border rounded-full absolute  top-0 bottom-0 right-2 m-auto  ">
        <ArrowRightOutlined/>
      </div>
      <div onClick={()=>handelClick("left")} className="w-12 h-12 bg-slate-100 cursor-pointer opacity-50 flex items-center justify-center border rounded-full absolute  top-0 bottom-0 left-2 m-auto  ">
        <ArrowLeftOutlined/>
      </div>
    </div>
  )
}

export default Slider
