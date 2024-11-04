using BackEndServer.Data;
using BackEndServer.Entities;
using Microsoft.EntityFrameworkCore;

namespace Server;

public static class OrderEndPointExtension
{
public static WebApplication MapOrderEndPoints(this WebApplication app){
      app.MapPost("/orders", async (ServerContext db, OrderDto orderDto) =>
{
    var order = new Order
    {
        UserId = orderDto.UserId,
        Amount = orderDto.Amount,
        Address = orderDto.Address,
        Status = orderDto.Status ?? "pending",
        CreatedAt = DateTime.UtcNow,
        UpdatedAt = DateTime.UtcNow,
        Products = orderDto.Products?.Select(p => new OrderProduct
        {
            ProductId = p.ProductId,
            Quantity = p.Quantity
        }).ToList()
    };

    db.Orders.Add(order);
    await db.SaveChangesAsync();

    return Results.Created($"/orders/{order.OrderId}", order);
});
app.MapPut("/orders/{id:int}", async (int id, OrderDto orderDto, ServerContext db) =>
{
    var order = await db.Orders.Include(o => o.Products).FirstOrDefaultAsync(o => o.OrderId == id);

    if (order == null)
    {
        return Results.NotFound();
    }

    order.UserId = orderDto.UserId;
    order.Amount = orderDto.Amount;
    order.Address = orderDto.Address;
    order.Status = orderDto.Status ?? "pending";
    order.UpdatedAt = DateTime.UtcNow;
    order.Products = orderDto.Products?.Select(p => new OrderProduct
    {
        ProductId = p.ProductId,
        Quantity = p.Quantity
    }).ToList();

    await db.SaveChangesAsync();

    return Results.Ok(order);
});
app.MapDelete("/orders/{id:int}", async (int id, ServerContext db) =>
{
    var order = await db.Orders.FindAsync(id);

    if (order == null)
    {
        return Results.NotFound();
    }

    db.Orders.Remove(order);
    await db.SaveChangesAsync();

    return Results.Ok("Order has been deleted...");
});
app.MapGet("/orders/find/{userId}", async (string userId, ServerContext db) =>
{
    var orders = await db.Orders.Include(o => o.Products).Where(o => o.UserId == userId).ToListAsync();

    if (!orders.Any())
    {
        return Results.NotFound();
    }

    return Results.Ok(orders);
});
app.MapGet("/orders", async (ServerContext db) =>
{
    var orders = await db.Orders.Include(o => o.Products).ToListAsync();
    return Results.Ok(orders);
});

// GET /api/orders/income
app.MapGet("orders/income", async (ServerContext db, int? productId) =>
{
    var date = DateTime.UtcNow;
    var lastMonth = new DateTime(date.Year, date.Month, 1).AddMonths(-1);
    var previousMonth = lastMonth.AddMonths(-1);

    var income = await db.Orders
                         .Where(o => o.CreatedAt >= previousMonth)
                         .Where(o => !productId.HasValue || o.Products!.Any(p => p.ProductId == productId.ToString()))
                         .GroupBy(o => new { o.CreatedAt.Month })
                         .Select(g => new
                         {
                             Month = g.Key.Month,
                             Total = g.Sum(o => o.Amount)
                         })
                         .ToListAsync();

    return Results.Ok(income);
});

    return app;
    }
}
