import Announcment from "../Component/Announcment"
import Categories from "../Component/Categories"
import Footer from "../Component/Footer"
import NavBar from "../Component/NavBar"
import NewsLetter from "../Component/NewsLetter"
import Products from "../Component/Products"
import Slider from "../Component/Slider"


const Home = () => {
  return (
    <div>
    <Announcment />
      <NavBar />
      <Slider/>
      <Categories/>
      <Products/>
      <NewsLetter/>
      <Footer/>
    </div>
  )
}

export default Home
