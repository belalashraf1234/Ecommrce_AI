import axios from 'axios';
import  { useEffect, useState } from 'react'
import Comment from './Comment';

const CommentList = ({id}) => {
  const [comments, setComments] = useState([]);
  useEffect(  ()=>{
   const retriveComments = async () => {
      const response = await axios.get(`http://localhost:5115/comments/${id}`);
      setComments(response.data);
      console.log(response.data);
   };
    retriveComments();
  },[comments]);
const filerdComments=comments.filter(comment=>comment.probability>0.0010);

  return (
    <>
      {filerdComments.map((comment) => (
        <Comment key={comment.id} comment={comment} />
      ))}
    </>
  )
}

export default CommentList