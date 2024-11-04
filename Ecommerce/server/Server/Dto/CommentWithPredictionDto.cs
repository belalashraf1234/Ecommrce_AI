namespace Server;

public class CommentWithPredictionDto
{
    public string CommentText { get; set; }
    public string Username { get; set; }
    public int CommentId { get; set; }
    public int UserId { get; set; }
    public int ProductId { get; set; }
    public string Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}