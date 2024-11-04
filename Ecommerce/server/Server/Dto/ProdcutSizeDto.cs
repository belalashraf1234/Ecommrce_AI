namespace Server;

public record class ProdcutSizeDto
{
 public int SizeId { get; set; }
    public int ProductId { get; set; }
    public string ?Size { get; set; }
}
