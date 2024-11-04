namespace BackEndServer.Entities;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;


public class Cart
{
    [Key]
    public int CartId { get; set; }
    
    [Required]
    public string? UserId { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    
    public virtual ICollection<CartProduct>? Products { get; set; }
}
