using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BackEndServer.Entities;

public class ProductCategory
{
    [Key]
    public int CategoryId { get; set; }
    
    public int ProductId { get; set; }
    
    [StringLength(100)]
    public string? Category { get; set; }
    
    [ForeignKey("ProductId")]
    public virtual Product? Product { get; set; }
}
