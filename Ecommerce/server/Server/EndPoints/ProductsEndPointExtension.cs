using BackEndServer.Data;
using BackEndServer.Entities;
using Microsoft.EntityFrameworkCore;

namespace Server;

public static class ProductEndPointExtension
{
public static WebApplication MapProductEndPoints(this WebApplication app){
       app.MapPost("/products", async (ProductDto productDto, ServerContext db) =>
{
    var product = new Product
    {
        Title = productDto.Title,
        Description = productDto.Description,
        Image = productDto.Image,
        Price = productDto.Price,
        InStock = productDto.InStock,
        CreatedAt = DateTime.UtcNow,
        UpdatedAt = DateTime.UtcNow,
        Categories = productDto.Categories?.Select(c => new ProductCategory { Category = c }).ToList(),
        Sizes = productDto.Sizes?.Select(s => new ProductSize { Size = s }).ToList(),
        Colors = productDto.Colors?.Select(col => new ProductColor { Color = col }).ToList(),
    };

    db.Products.Add(product);
    await db.SaveChangesAsync();

    return Results.Created($"/products/{product.ProductId}", product);
});

app.MapPut("/products/{id}", async (int id, ProductDto productDto, ServerContext db) =>
{
    var product = await db.Products.Include(p => p.Categories).Include(p => p.Sizes).Include(p => p.Colors).FirstOrDefaultAsync(p => p.ProductId == id);

    if (product == null)
        return Results.NotFound();

    product.Title = productDto.Title;
    product.Description = productDto.Description;
    product.Image = productDto.Image;
    product.Price = productDto.Price;
    product.InStock = productDto.InStock;
    product.UpdatedAt = DateTime.UtcNow;

    // Update related collections
    product.Categories = productDto.Categories?.Select(c => new ProductCategory { Category = c }).ToList();
    product.Sizes = productDto.Sizes?.Select(s => new ProductSize { Size = s }).ToList();
    product.Colors = productDto.Colors?.Select(col => new ProductColor { Color = col }).ToList();

    await db.SaveChangesAsync();

    return Results.Ok(product);
});

app.MapDelete("/products/{id}", async (int id, ServerContext db) =>
{
    var product = await db.Products.FindAsync(id);

    if (product == null)
        return Results.NotFound();

    db.Products.Remove(product);
    await db.SaveChangesAsync();

    return Results.Ok("Product has been deleted...");
});

app.MapGet("/products/find/{id}", async (int id, ServerContext db) =>
{
    var product = await db.Products.Include(p => p.Categories).Include(p => p.Sizes).Include(p => p.Colors).FirstOrDefaultAsync(p => p.ProductId == id);

    if (product == null)
        return Results.NotFound();

    return Results.Ok(product);
});

app.MapGet("/products", async (string? qNew, string? qCategory, ServerContext db) =>
{
    IQueryable<Product> query = db.Products.Include(p => p.Categories).Include(p => p.Sizes).Include(p => p.Colors);

    if (!string.IsNullOrEmpty(qNew))
    {
        query = query.OrderByDescending(p => p.CreatedAt).Take(1);
    }
    else if (!string.IsNullOrEmpty(qCategory))
    {
        query = query.Where(p => p.Categories.Any(c => c.Category == qCategory));
    }

    var products = await query.ToListAsync();
    return Results.Ok(products);
});


        return app;
    }
}
