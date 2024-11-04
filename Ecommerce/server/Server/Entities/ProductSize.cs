namespace BackEndServer.Entities;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

public class ProductSize
{
 [Key]
    public int SizeId { get; set; }
    
    public int ProductId { get; set; }
    
    [StringLength(50)]
    public string? Size { get; set; }
    
    [ForeignKey("ProductId")]
    public virtual Product? Product { get; set; }
}
