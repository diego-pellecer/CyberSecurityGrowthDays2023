using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

 // Simple image classification problem to classify grayscale images of handwritten digits (0-9).
 // NOTE: Note that this example assumes that you have already trained a TensorFlow model and exported it as a .pb file, 
 // which can be loaded using the LoadTensorFlowModel() method.
 
namespace CNNExample
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load data
            var data = context.Data.LoadFromEnumerable(GetData());

            // Split data into training and testing sets
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Define pipeline
            var pipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.LoadRawImageBytes("Image", "ImagePath"))
                .Append(context.Transforms.ResizeImages("Image", 28, 28, "Image"))
                .Append(context.Transforms.ExtractPixels("Image", interleavePixelColors: true))
                .Append(context.Model.LoadTensorFlowModel("model.pb")
                .ScoreTensorFlowModel(new[] { "dense/BiasAdd" }, new[] { "Input" }, true)
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

            // Train model
            var model = pipeline.Fit(trainTestData.TrainSet);

            // Evaluate model
            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");

            // Make prediction on new data
            var predictionEngine = context.Model.CreatePredictionEngine<HandwrittenDigit, HandwrittenDigitPrediction>(model);
            var prediction = predictionEngine.Predict(new HandwrittenDigit { ImagePath = "test.png" });
            Console.WriteLine($"Predicted digit: {prediction.PredictedLabel}");
        }

        static HandwrittenDigit[] GetData()
        {
            var dataDirectory = @"path/to/data/folder";
            var digits = Enumerable.Range(0, 10);

            return digits
                .SelectMany(digit => Directory.GetFiles(Path.Combine(dataDirectory, digit.ToString()))
                    .Select(file => new HandwrittenDigit
                    {
                        ImagePath = file,
                        Label = digit
                    }))
                .ToArray();
        }
    }

    class HandwrittenDigit
    {
        public string ImagePath { get; set; }
        public int Label { get; set; }
    }

    class HandwrittenDigitPrediction
    {
        [ColumnName("PredictedLabel")]
        public int PredictedLabel { get; set; }
    }
}
