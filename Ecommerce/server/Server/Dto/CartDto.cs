namespace Server;

public record class CartDto
{    public int CartId { get; set; }
    public string? UserId { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public ICollection<CartProductDto>? Products { get; set; }

}
