using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BackEndServer.Entities;

public class OrderProduct
{
    [Key]
    public int OrderProductId { get; set; }
    
    public int OrderId { get; set; }
    
    public string? ProductId { get; set; }
    
    public int Quantity { get; set; } = 1;
    
    [ForeignKey("OrderId")]
    public virtual Order? Order { get; set; }
}