import { Facebook, Instagram, MailOutline, Phone, Room, Twitter } from "@mui/icons-material";

const Footer = () => {
  return (
    <div className="flex ">
      <div className="flex-1 flex flex-col p-5 ">
        <h1>CiC</h1>
        <p className="my-5">
          this is Gruad Project for cic Bis department 2024
        </p>
        <div className="flex ">
          <div className="w-10 h-10 rounded-full text-white bg-slate-900 flex items-center justify-center mr-5">
            <Facebook />
          </div>
          <div className="w-10 h-10 rounded-full text-white bg-slate-900 flex items-center justify-center mr-5">
            <Instagram />
          </div>
          <div className="w-10 h-10 rounded-full text-white bg-slate-900 flex items-center justify-center mr-5">
            <Twitter />
          </div>
        </div>
      </div>
      <div className="flex-1 p-5">
        <h1 className="mb-[30px]">Usfel Links</h1>
        <ul className="m-0 p-0 list-none flex flex-wrap">
          <li className="w-1/2 mb-3  ">Home</li>
          <li className="w-1/2 mb-3 ">Cart</li>
          <li className="w-1/2 mb-3 ">Mans</li>
          <li className="w-1/2 mb-3 ">Womans</li>
          <li className="w-1/2 mb-3 ">Accessories</li>
          <li className="w-1/2 mb-3 ">MyAccount</li>
          <li className="w-1/2 mb-3 ">OrderTracking </li>
          <li className="w-1/2 mb-3 ">whishlist</li>
          <li className="w-1/2 mb-3 ">Terms</li>
        </ul>
      </div>
      <div className="flex-1 p-5">
      <h1 className="mb-[30px]">Contact</h1>
      <div className="mb-5 flex items-center">  <Room style={{marginRight:"10px"}}/> Cic College</div>
      <div className="mb-5 flex items-center"> <Phone style={{marginRight:"10px"}}/> </div>
      <div className="mb-5 flex items-center"> <MailOutline style={{marginRight:"10px"}} /></div>
      </div>
      
      
    </div>
  );
};

export default Footer;
