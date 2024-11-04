namespace Server;

public record class ProductUpdateDto
{
public string? Title { get; set; }
    public string? Description { get; set; }
    public string? Image { get; set; }
    public decimal Price { get; set; }
    public bool InStock { get; set; }
    public ICollection<string> ?Categories { get; set; }
    public ICollection<string>? Sizes { get; set; }
    public ICollection<string> ?Colors { get; set; }

}
