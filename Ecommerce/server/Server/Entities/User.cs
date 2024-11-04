namespace BackEndServer.Entities;
using System.ComponentModel.DataAnnotations;

public class User
{
    [Key]
    public int UserId { get; set; }
    
    [Required]
    [StringLength(100)]
    public string? Username { get; set; }
    
    [Required]
    [EmailAddress]
    [StringLength(100)]
    public string? Email { get; set; }
    
    [Required]
    [StringLength(255)]
    public string? Password { get; set; }
    
    public bool IsAdmin { get; set; } = false;
    
    [StringLength(255)]
    public string? Img { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
