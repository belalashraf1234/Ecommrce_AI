namespace BackEndServer.Entities;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

public class CartProduct
{
     [Key]
    public int CartProductId { get; set; }
    
    public int CartId { get; set; }
    
    public string? ProductId { get; set; }
    
    public int Quantity { get; set; } = 1;
    
    [ForeignKey("CartId")]
    public virtual Cart? Cart { get; set; }

}
