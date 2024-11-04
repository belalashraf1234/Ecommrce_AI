using BackEndServer.Data;
using BackEndServer.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using Microsoft.ML.Data;
using Server;


public static class AiModelEndPoint
{
public static WebApplication MapCommentEndPoints(this WebApplication app){
  
  
app.MapPost("/predict",async (CommentDto request, PredictionEngine<ProductReview, ProductReviewPrediction> predEngine,ServerContext db) =>
{   var username = db.Users.FirstOrDefault(u => u.UserId == request.UserId);
    var review = new Comment { CommentText = request.CommentText,UserId = request.UserId,ProductId = request.ProductId};
    var predict=new ProductReview(){ReviewText=review.CommentText!};
    var prediction = predEngine.Predict(predict);
    db.Comments.Add(review);
    await db.SaveChangesAsync();
    
    return Results.Ok(new
    {   Username = username!.Username,
        Text = review.CommentText,
        Prediction = prediction.Prediction ? "Positive" : "Negative",
        Probability = prediction.Probability,
        Score = prediction.Score
    });

});

app.MapGet("/comments/{productId:int}", async (
    int productId,
    PredictionEngine<ProductReview, ProductReviewPrediction> predEnginePool,
    ServerContext db) =>
{
    // Fetch comments for the specified product ID
    var comments = await db.Comments
        .Where(comment => comment.ProductId == productId)
        .ToListAsync();

    // Prepare a list to hold comments with their predictions
    var commentsWithPredictions = new List<CommentWithPredictionDto>();

    // Predict reviews for each comment
    foreach (var comment in comments)
    {
        var commentReview = new ProductReview() { ReviewText = comment.CommentText! };
        var prediction = predEnginePool.Predict(commentReview);

        var username = await db.Users
            .Where(u => u.UserId == comment.UserId)
            .Select(u => u.Username)
            .FirstOrDefaultAsync();

        commentsWithPredictions.Add(new CommentWithPredictionDto
        {
            CommentText = comment.CommentText!,
            CommentId = comment.CommentId,
            UserId = comment.UserId,
            Username = username,
            ProductId = comment.ProductId,
            Prediction = prediction.Prediction ? "Positive" : "Negative",
            Probability = prediction.Probability,
            Score = prediction.Score
        });
    }

    // Return the list of comments with predictions


    // Return the comments with their predictions in JSON format
    return Results.Ok(commentsWithPredictions);
});


    return app;
}







}
