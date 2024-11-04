
import { useState } from 'react';
import  './Comment.css'
import axios from 'axios';
import { useSelector } from 'react-redux';


const CommentForm = ({ id,modal }) => {
  const user=useSelector(state=>state.user.currentUser.userId);
  const [text, setText] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res=await axios.post("http://localhost:5115/predict/",{ProductId:id,commentText:text,userId:user});
    if(res.data.probability<0.0010){
      modal(true);
    }
    setText('');
  };


  return (
    <form className="comment-form" onSubmit={handleSubmit}>

      <textarea
        placeholder="Your Comment"
        value={text}
        onChange={(e) => setText(e.target.value)}
      ></textarea>
      <button type="submit">Add Comment</button>
    </form>
  );
};
  export default CommentForm;