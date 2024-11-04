namespace Server;

public record class OrderDto
{
public int OrderId { get; set; }
    public string? UserId { get; set; }
    public decimal Amount { get; set; }
    public string? Address { get; set; }
    public string ?Status { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public ICollection<OrderProductDto>? Products { get; set; }
}
