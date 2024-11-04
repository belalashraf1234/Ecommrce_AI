using System.ComponentModel.DataAnnotations;

namespace Server;

public record class LoginDto
{
    [Required]
    public string? Username { get; set; }

    [Required]
    public string? Password { get; set; }
}
