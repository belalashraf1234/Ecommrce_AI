using Microsoft.ML;
using Microsoft.ML.Data;

string dataPath = @"E:\Ecommerce\server\AiModel\\Products_Review.txt";
var mlContext = new MLContext();

var data = mlContext.Data.LoadFromTextFile<ProductReview>(dataPath, hasHeader: false, separatorChar: '\t');
var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
var trainData = trainTestData.TrainSet;
var testData = trainTestData.TestSet;

var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ProductReview.ReviewText))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

var model = pipeline.Fit(trainData);

mlContext.Model.Save(model, trainData.Schema, "model.zip");

public class ProductReview
{
    [LoadColumn(0)]
    public string ReviewText { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }
}