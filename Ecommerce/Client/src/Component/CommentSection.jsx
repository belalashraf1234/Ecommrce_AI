
import CommentList from "./CommentList";
import CommentForm from "./CommentForm";
import  './Comment.css'
import { useState } from "react";
import Modal from "./Modal";
const CommentSection = ({id}) => {
  const [showModal, setShowModal] = useState(false);
  const handleOpenModal = () => {
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
  };
  
    return (
      <div className="comment-section">
        <h2 >Comments</h2>
        <Modal show={showModal} onClose={handleCloseModal}>
        <p>Sorry but your review is suspect to be fake review</p>
      </Modal>
        <CommentList id={id}  />
        <CommentForm modal={handleOpenModal} id={id}  />
      </div>
    );
  };
  
  export default CommentSection;