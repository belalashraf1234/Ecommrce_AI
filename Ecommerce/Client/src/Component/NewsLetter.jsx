import { Send } from "@mui/icons-material"


const NewsLetter = () => {
  return (
    <div className="h-[60vh] bg-[#fcf5f5] flex items-center justify-center flex-col">
      <h1 className="font text-[70px] mb-[20px]  ">
      Newsletter
      </h1>
      <div className="text-[24px] font-[300] mb-[20px] ">
        Get timely updates from your favorite products.
      </div>
      <div className="w-1/2 h-[50px]  bg-white flex justify-between  border border-solid border-gray-400">
            <input type="text"  className="flex-[8] p-2 w-3/4 border-none  pl-5"/>
            <button className="flex-1 border-none bg-teal-700">
                <Send/>
            </button>
      </div>
    </div>
  )
}

export default NewsLetter
