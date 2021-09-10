using Microsoft.ML;
using System;

namespace MulticlassCategorize
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var trainingData = mlContext.Data.LoadFromTextFile<Model.NewsInput>("dataset/training_dataset.csv", hasHeader: true, separatorChar: ',');
            var testData = mlContext.Data.LoadFromTextFile<Model.NewsInput>("dataset/training_dataset.csv", hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "KeyLabel")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "FeaturizedTitle"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Body", outputColumnName: "FeaturizedBody"))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", new string[] { "FeaturizedTitle", "FeaturizedBody" }))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(labelColumnName: "KeyLabel", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: "PredictedLabel", outputColumnName: "PredictLabel"));

            var model = pipeline.Fit(trainingData);

            var predict = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predict, labelColumnName: "KeyLabel", scoreColumnName: "Score");

            mlContext.Model.Save(model, trainingData.Schema, "SavedModel.zip");

            PredictionEngine<Model.NewsInput, Model.NewsOutput> predictionEngine =
                mlContext.Model.CreatePredictionEngine<Model.NewsInput, Model.NewsOutput>(model, trainingData.Schema);

            var result = predictionEngine.Predict(new Model.NewsInput()
            {
                Title = "The dark man killed 3 women too",
                Body = "A man called the dark man killed three women because they were speaking very loudly in the metro station. A lot of mistress saw the event"
            });
            Console.WriteLine(result.PredictLabel);
        }
        public static void LoadModal()
        {
            var mlContext = new MLContext();
            DataViewSchema schema = null;
            var model = mlContext.Model.Load("", out schema);
            PredictionEngine<Model.NewsInput, Model.NewsOutput> predictionEngine =
                mlContext.Model.CreatePredictionEngine<Model.NewsInput, Model.NewsOutput>(model, schema);

            var result = predictionEngine.Predict(new Model.NewsInput()
            {
                Title = "The dark man killed 3 women too",
                Body = "A man called the dark man killed three women because they were speaking very loudly in the metro station. A lot of mistress saw the event"
            });
            Console.WriteLine(result.PredictLabel);
        }
    }
}
