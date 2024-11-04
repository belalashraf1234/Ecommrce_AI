namespace BackEndServer.Entities;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
public class ProductColor
{
    [Key]
    public int ColorId { get; set; }
    
    public int ProductId { get; set; }
    
    [StringLength(50)]
    public string? Color { get; set; }
    
    [ForeignKey("ProductId")]
    public virtual Product? Product { get; set; }
}
