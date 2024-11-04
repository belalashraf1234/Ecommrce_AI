using System.ComponentModel.DataAnnotations;

namespace BackEndServer.Entities;

public class Product
{
    [Key]
    public int ProductId { get; set; }
    
    [Required]
    [StringLength(100)]
    public string? Title { get; set; }
    
    [Required]
    public string? Description { get; set; }
    
    [Required]
    public string? Image { get; set; }
    
    public decimal Price { get; set; }
    
    public bool InStock { get; set; } = true;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    
    public virtual ICollection<ProductCategory>? Categories { get; set; }
    
    public virtual ICollection<ProductSize>? Sizes { get; set; }
    
    public virtual ICollection<ProductColor>? Colors { get; set; }
}