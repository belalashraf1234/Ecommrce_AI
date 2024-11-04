namespace Server;

public record class OrderProductDto
{
public int OrderProductId { get; set; }
    public string? ProductId { get; set; }
    public int Quantity { get; set; }
}
