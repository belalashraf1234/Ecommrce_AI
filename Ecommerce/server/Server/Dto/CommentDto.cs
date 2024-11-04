namespace Server;
using System.ComponentModel.DataAnnotations;

public record class CommentDto
{
        [Required]
        public int ProductId { get; set; }

        [Required]
        public int UserId { get; set; }

        [Required]
        [StringLength(1000, ErrorMessage = "Comment cannot be longer than 1000 characters.")]
        public string? CommentText { get; set; }
}
