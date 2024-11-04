using System.Text;
using System.Text.Json.Serialization;
using BackEndServer.Data;
using BackEndServer.Entities;
using Microsoft.ML;
using Microsoft.ML.Data;
using Server;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowSpecificOrigins",
        policy =>
        {
            policy.AllowAnyOrigin() // Specify allowed origins
                  .AllowAnyHeader()
                  .AllowAnyMethod();
        });
});

builder.Services.AddSqlite<ServerContext>("Data Source=Ecommerce.db");
builder.Services.Configure<Microsoft.AspNetCore.Http.Json.JsonOptions>(options => options.SerializerOptions.ReferenceHandler = ReferenceHandler.IgnoreCycles);
builder.Services.AddSingleton(LoadModel());
// Method to load the model
PredictionEngine<ProductReview, ProductReviewPrediction> LoadModel()
{
    // Load the model
    string modelPath = Path.Combine(@"E:\Ecommerce\server\Server", "model.zip");
    var mlContext = new MLContext();
    var loadedModel = mlContext.Model.Load(modelPath, out var modelInsputSchema);
    
    // Create a prediction engine
    return mlContext.Model.CreatePredictionEngine<ProductReview, ProductReviewPrediction>(loadedModel);
}
var app = builder.Build();





app.UseCors("AllowSpecificOrigins");
app.MapGet("/", () => "Welcome to the Product Review Sentiment Analysis API!");
app.MapAuthEndPoints();
app.MapOrderEndPoints();
app.MapProductEndPoints();
app.MapUserEndPoints();
app.MapCartEndPoints();
app.MapCommentEndPoints();

app.Run();




// Define data classes



public class ProductReview
{
    [LoadColumn(0)]
    public string ReviewText { get; set; }
}

public class ProductReviewPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}








