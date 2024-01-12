// See https://aka.ms/new-console-template for more information

using AnomalyDetection_MeterReading;
using Microsoft.ML;

var mlContext = new MLContext();

var _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
//assign the Number of records in dataset file to constant variable
var _docsize = 36;

var dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

// var serialzedData = System.Text.Json.JsonSerializer.Serialize(dataView);
Console.WriteLine("Completed loading the scheme...");

DetectSpike(mlContext, _docsize, dataView);

// Use IID spike estimator

IDataView CreateEmptyDataView(MLContext context)
{
    // Create empty DataView. We just need the schema to call Fit() for the time series transforms
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    return context.Data.LoadFromEnumerable(enumerableData);
}

void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(
        outputColumnName: nameof(ProductSalesPrediction.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales), confidence: 95d, pvalueHistoryLength: docSize / 4);
    
    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
    
    IDataView transformedData = iidSpikeTransform.Transform(productSales);
    
    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);
    
    Console.WriteLine("Alert\tScore\tP-Value");
    
    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- Spike detected";
            }

            Console.WriteLine(results);
        }
    }
    Console.WriteLine("");
}
