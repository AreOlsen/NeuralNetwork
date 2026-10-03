using System.Globalization;
using CsvHelper;
using NeuralNetwork.Activations;
using NeuralNetwork.Layers.Tensor.Convolutional;
using NeuralNetwork.Loss;

namespace NeuralNetwork;

public class Program
{
    private const int ImageSize = 28;
    private const int Channels = 1;

    public static void Main(string[] args)
    {
        Console.WriteLine("Reading CSV...");
        string csvPath = "/Users/areolsen/Desktop/NeuralNetCSharp/NeuralNetwork/NeuralNetwork/mnist_train.csv";
        var (x, y) = ReadCSV(csvPath, "label");

        Console.WriteLine("Creating network...");
        ConvolutionNetwork net = new(
            inputHeight: ImageSize,
            inputWidth: ImageSize,
            inputDepth: Channels,
            kernelCount: 3,
            kernelWidth: 3,
            kernelHeight: 3,
            stride: 1,
            padding: 1,
            convolutionLayers: 2,
            convulationActivationType: ActivationType.ReLU,
            hiddenLayers: 2,
            hiddenNodes: 32,
            hiddenActivationType: ActivationType.LeakyReLU,
            outputNodes: 10,
            outputActivationType: ActivationType.Softmax);

        SoftmaxCrossEntropy<double[,,]> cross = new(net);
        Console.WriteLine("Training...");
        List<double> loss = net.Train(10, 0.01d, x, y, cross);
    }

    private static (List<double[,,]> x, List<List<double>> y) ReadCSV(string filepath, string yColumn)
    {
        var xData = new List<double[,,]>();
        var yData = new List<List<double>>();

        using (var reader = new StreamReader(filepath))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            csv.Read();
            csv.ReadHeader();
            
            var xColumns = csv.HeaderRecord!
                .Where(col => col != yColumn)
                .ToArray();

            while (csv.Read())
            {
                var flatPixels = new double[xColumns.Length];
                for (int i = 0; i < xColumns.Length; i++)
                {
                    flatPixels[i] = csv.GetField<double>(xColumns[i]);
                }

                var image = new double[Channels, ImageSize, ImageSize];
                for (int r = 0; r < ImageSize; r++) 
                {
                    for (int c = 0; c < ImageSize; c++) 
                    {
                        int flatIndex = r * ImageSize + c;
                        image[0, c, r] = flatPixels[flatIndex] / 255.0;
                    }
                }
                xData.Add(image);

                // Extract target label
                int label = csv.GetField<int>(yColumn);
                var yRow = new List<double>(new double[10]);
                yRow[label] = 1.0;
                yData.Add(yRow);
            }
        }

        return (xData, yData);
    }
}