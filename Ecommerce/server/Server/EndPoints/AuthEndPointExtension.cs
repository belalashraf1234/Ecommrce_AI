using BackEndServer.Data;
using BackEndServer.Entities;
using Microsoft.EntityFrameworkCore;

namespace Server;

public static class AuthEndPointExtension
{
public static WebApplication MapAuthEndPoints(this WebApplication app){
       app.MapPost("/register", async (UserDto userDto, ServerContext db) =>
{
    var newUser = new User
    {
        Username = userDto.Username,
        Email = userDto.Email,
        Password = userDto.Password
    };

    db.Users.Add(newUser);
    await db.SaveChangesAsync();

    return Results.Created($"/users/{newUser.UserId}", newUser);
});    
    app.MapPost("/login", async (LoginDto loginDto, ServerContext db) =>
{
    var user = await db.Users.SingleOrDefaultAsync(u => u.Username == loginDto.Username);
    if (user == null || user.Password != loginDto.Password)
    {
        return Results.Unauthorized();
    }

    return Results.Ok(new { user.UserId, user.Username, user.Email, user.IsAdmin, user.Img });
});


    return app;
    }
}
