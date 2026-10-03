namespace NeuralNetwork;

using NeuralNetwork.Activations;
using NeuralNetwork.Layers.Node;
using NeuralNetwork.Layers.Node.Dense;
using NeuralNetwork.Loss;

public class DenseNetwork : INetwork<List<double>> {
    private readonly List<DenseLayer> _dense = new();
    private readonly NodeInputLayer _input;
    private readonly NodeOutputLayer _output;
    public int OutputSize { get;  }
    public int InputSize { get;  }

    public DenseNetwork(int inputNodes, int hiddenLayers, int hiddenNodes, ActivationType hiddenActivationType,  int outputNodes, ActivationType outputActivationType)
    {
        InputSize = inputNodes;
        OutputSize = outputNodes;

        NodeInputLayer input = new(inputNodes);
        _input = input;
        _dense.Add(new DenseLayer(input,hiddenNodes,hiddenActivationType));
        for (int i = 1; i < hiddenLayers; i++){
            DenseLayer dense = new(_dense[i-1], hiddenNodes, hiddenActivationType);
            _dense.Add(dense);
        }
        _output = new NodeOutputLayer(_dense[^1], outputNodes, outputActivationType);
    }

	public List<double> Predict(List<double> x)
	{
	    _input.SetInputs(x);
		for(int i = 0; i<_dense.Count; i++){
		    _dense[i].Forward();
		}
        _output.Forward();
        return _output.GetValues();
	}

	public List<double> Train(int epochs, double learningRate, List<List<double>> x, List<List<double>> y, ILoss<List<double>> loss){
        List<double> lossHistory = new(epochs);
        for (int epoch = 0; epoch < epochs; epoch++){
            double lossValue = loss.Loss(x, y);
            lossHistory.Add(lossValue);
            if(double.IsNaN(lossValue)) break;
            for (int iteration = 0; iteration< x.Count; iteration++ ){
                List<double> gradient = loss.LossGradient(x[iteration],y[iteration]);
                _output.SetErrors(gradient);
                _output.Backward(learningRate);
                for (int layer = _dense.Count-1; layer >= 0; layer--){
                    _dense[layer].Backward(learningRate);
                }
            }
        }
        return lossHistory;
    }
}
