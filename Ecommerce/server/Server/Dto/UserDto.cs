namespace Server;

public record class UserDto
{
 public int? UserId { get; set; }
    public string? Username { get; set; }
    public string? Password { get; set; }
    public string ?Email { get; set; }
    public bool ?IsAdmin { get; set; }
    public string? Img { get; set; }
    public DateTime? CreatedAt { get; set; }
    public DateTime? UpdatedAt { get; set; }
}
