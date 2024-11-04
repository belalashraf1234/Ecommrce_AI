using BackEndServer.Data;
using Microsoft.EntityFrameworkCore;

namespace Server;

public static  class UserEndPointExtension
{
public static WebApplication MapUserEndPoints(this WebApplication app){
    app.MapPut("/users/{id:int}", async (int id, UserDto userDto, ServerContext db) =>
{
    if (!string.IsNullOrEmpty(userDto.Password))
    {
        // Encrypt password if it's provided
        // You need to implement the encryption logic
    }

    var user = await db.Users.FindAsync(id);

    if (user == null)
    {
        return Results.NotFound();
    }

    user.Username = userDto.Username;
    user.Email = userDto.Email;
    user.IsAdmin = userDto.IsAdmin ?? false;
    user.Img = userDto.Img;
    user.UpdatedAt = DateTime.UtcNow;

    await db.SaveChangesAsync();

    return Results.Ok(user);
});

// DELETE /api/users/{id}
app.MapDelete("/users/{id:int}", async (int id, ServerContext db) =>
{
    var user = await db.Users.FindAsync(id);

    if (user == null)
    {
        return Results.NotFound();
    }

    db.Users.Remove(user);
    await db.SaveChangesAsync();

    return Results.Ok("User has been deleted...");
});

// GET /api/users/find/{id}
app.MapGet("/users/find/{id:int}", async (int id, ServerContext db) =>
{
    var user = await db.Users.FindAsync(id);

    if (user == null)
    {
        return Results.NotFound();
    }

    // Exclude password from the response
    var userDto = new UserDto
    {
        UserId = user.UserId,
        Username = user.Username,
        Email = user.Email,
        IsAdmin = user.IsAdmin,
        Img = user.Img,
        CreatedAt = user.CreatedAt,
        UpdatedAt = user.UpdatedAt
    };

    return Results.Ok(userDto);
});

// GET /api/users
app.MapGet("/users", async (ServerContext db) =>
{
    var users = await db.Users.ToListAsync();
    return Results.Ok(users);
});

// GET /api/users/stats
app.MapGet("/users/stats", async (ServerContext db) =>
{
    var date = DateTime.UtcNow;
    var lastYear = new DateTime(date.Year - 1, date.Month, date.Day);

    var data = await db.Users
                        .Where(u => u.CreatedAt >= lastYear)
                        .GroupBy(u => u.CreatedAt.Month)
                        .Select(g => new { Month = g.Key, Total = g.Count() })
                        .ToListAsync();

    return Results.Ok(data);
});

    return app;
    }
}
