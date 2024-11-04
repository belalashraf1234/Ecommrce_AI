using Microsoft.EntityFrameworkCore;

namespace BackEndServer.Data;

public static class DataExtension
{
public static void MigrateDb(this WebApplication app)
{
   using var scope = app.Services.CreateScope();
   var dbContext=scope.ServiceProvider.GetRequiredService<ServerContext>();
      dbContext.Database.Migrate();

}};
