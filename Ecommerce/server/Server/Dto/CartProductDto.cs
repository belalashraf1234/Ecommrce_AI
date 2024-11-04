namespace Server;

public record class CartProductDto
{
public int CartProductId { get; set; }
    public int CartId { get; set; }
    public string? ProductId { get; set; }
    public int Quantity { get; set; }
}
