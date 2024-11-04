namespace Server;

public record class ProductDto
{
      public int ProductId { get; set; }
    public string? Title { get; set; }
    public string? Description { get; set; }
    public string? Image { get; set; }
    public decimal Price { get; set; }
    public bool InStock { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public ICollection<string>? Categories { get; set; }
    public ICollection<string> ?Sizes { get; set; }
    public ICollection<string>? Colors { get; set; }

}
