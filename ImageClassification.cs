using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

/* This code assumes that you have already trained a TensorFlow model and exported it as a .pb file. 
You can load the model using the LoadTensorFlowModel() method and use the ScoreTensorFlowModel() method to score the images. 
The ExtractPixels() method is used to extract the pixel values from the images and convert them to a format that can be fed into the neural network. 
Finally, the MapKeyToValue() method is used to convert the predicted class labels back to their original string values.
*/

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
                .Append(context.Transforms.ResizeImages("Image", 28, 28))
                .Append(context.Transforms.ExtractPixels("Image", interleavePixelColors: true))
                .Append(context.Transforms.Conversion.MapKeyToValue("Label"))
                .Append(context.Model.LoadTensorFlowModel("model.pb")
                    .ScoreTensorFlowModel(new[] { "dense_1/BiasAdd" }, new[] { "conv2d_input" }, true)
                    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

            // Train model
            var model = pipeline.Fit(trainTestData.TrainSet);

            // Evaluate model
            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");

            // Make prediction on new data
            var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictionEngine.Predict(new ImageData { ImagePath = "test.png" });
            Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
        }

        static IEnumerable<ImageData> GetData()
        {
            var dataDirectory = @"path/to/data/folder";
            var classes = new List<string> { "class1", "class2", "class3" };

            foreach (var className in classes)
            {
                var images = System.IO.Directory.GetFiles(System.IO.Path.Combine(dataDirectory, className))
                    .Select(imagePath => new ImageData { ImagePath = imagePath, Label = className });

                foreach (var image in images)
                {
                    yield return image;
                }
            }
        }
    }

    class ImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class ImagePrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
