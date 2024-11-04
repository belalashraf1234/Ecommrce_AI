using BackEndServer.Entities;
using Microsoft.EntityFrameworkCore;

namespace BackEndServer.Data;

public class ServerContext(DbContextOptions<ServerContext> options):DbContext(options)
{
    public DbSet<User> Users { get; set; }
    public DbSet<Product> Products { get; set; }
    public DbSet<Order> Orders { get; set; }
    public DbSet<OrderProduct> OrderProducts { get; set; }
    public DbSet<Cart> Carts { get; set; }
    public DbSet<CartProduct> CartProducts { get; set; }
     public DbSet<ProductCategory> ProductCategories { get; set; }
      public DbSet<ProductColor> ProductColors { get; set; }
       public DbSet<ProductSize> ProductSizes { get; set; }
        public DbSet<Comment> Comments { get; set; }
        
    protected override  void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);
      modelBuilder.Entity<User>()
        .Property(u => u.Img)
        .HasDefaultValue(null);
        modelBuilder.Entity<User>()
            .Property(u => u.CreatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");

        modelBuilder.Entity<User>()
            .Property(u => u.UpdatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");

        modelBuilder.Entity<Product>()
            .Property(p => p.CreatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");
            
        modelBuilder.Entity<Product>()
            .Property(p => p.UpdatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");
        modelBuilder.Entity<Order>()
            .Property(o => o.CreatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");
            
        modelBuilder.Entity<Order>()
            .Property(o => o.UpdatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");
         modelBuilder.Entity<Cart>()
            .Property(c => c.CreatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");
            
        modelBuilder.Entity<Cart>()
            .Property(c => c.UpdatedAt)
            .HasDefaultValueSql("CURRENT_TIMESTAMP");
           
    }   

    
    
}
