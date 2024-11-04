using BackEndServer.Data;
using BackEndServer.Entities;
using Microsoft.EntityFrameworkCore;

namespace Server;

public static class CArtEndPointsExtension
{
public static WebApplication MapCartEndPoints(this WebApplication app){
     app.MapPost("/carts", async (ServerContext db, CartDto cartDto) =>
{
    var cart = new Cart
    {
        UserId = cartDto.UserId,
        CreatedAt = DateTime.UtcNow,
        UpdatedAt = DateTime.UtcNow,
        Products = cartDto.Products?.Select(p => new CartProduct
        {
            ProductId = p.ProductId,
            Quantity = p.Quantity
        }).ToList()
    };

    db.Carts.Add(cart);
    await db.SaveChangesAsync();

    return Results.Created($"/api/carts/{cart.CartId}", cart);
});
app.MapPut("/carts/{id:int}", async (int id, CartDto cartDto, ServerContext db) =>
{
    var cart = await db.Carts.Include(c => c.Products).FirstOrDefaultAsync(c => c.CartId == id);

    if (cart == null)
    {
        return Results.NotFound();
    }

    cart.UserId = cartDto.UserId;
    cart.UpdatedAt = DateTime.UtcNow;
    cart.Products = cartDto.Products?.Select(p => new CartProduct
    {
        ProductId = p.ProductId,
        Quantity = p.Quantity
    }).ToList();

    await db.SaveChangesAsync();

    return Results.Ok(cart);
});
app.MapDelete("/carts/{id:int}", async (int id, ServerContext db) =>
{
    var cart = await db.Carts.FindAsync(id);

    if (cart == null)
    {
        return Results.NotFound();
    }

    db.Carts.Remove(cart);
    await db.SaveChangesAsync();

    return Results.Ok("Cart has been deleted...");
});
app.MapGet("/carts/find/{userId}", async (string userId, ServerContext db) =>
{
    var cart = await db.Carts.Include(c => c.Products)
                              .FirstOrDefaultAsync(c => c.UserId == userId);

    if (cart == null)
    {
        return Results.NotFound();
    }

    return Results.Ok(cart);
});
app.MapGet("/carts", async (ServerContext db) =>
{
    var carts = await db.Carts.Include(c => c.Products).ToListAsync();
    return Results.Ok(carts);
});
        return app;
    }
}
