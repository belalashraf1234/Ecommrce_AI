import React from 'react'
const Comment = ({ comment }) => {
    
  return (
    <div>
      <div className="comment">
        <div className="comment-header">
          <strong>{comment.username}</strong>
        </div>
        <div className='flex justify-between'>
        <p>{comment.commentText}</p>
        <p>{comment.prediction}</p>
        </div>
      </div>
    </div>
  )
}


export default Comment;