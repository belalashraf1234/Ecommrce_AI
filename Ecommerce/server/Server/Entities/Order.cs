namespace BackEndServer.Entities;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

public class Order
{
    [Key]
    public int OrderId { get; set; }
    
    [Required]
    public string? UserId { get; set; }
    
    [Required]
    public decimal Amount { get; set; }
    
    [Required]
    public string? Address { get; set; }
    
    public string Status { get; set; } = "pending";
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    
    public virtual ICollection<OrderProduct>? Products { get; set; }
}