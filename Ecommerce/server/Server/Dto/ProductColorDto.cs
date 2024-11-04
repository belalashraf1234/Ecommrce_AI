namespace Server;

public record class ProductColorDto
{
 public int? ColorId { get; set; }
    public int? ProductId { get; set; }
    public string? Color { get; set; }
}
