

using NeuralNetwork.Activations;
using NeuralNetwork.Layers.Node;
using NeuralNetwork.Layers.Node.Dense;
using NeuralNetwork.Loss;
using NeuralNetwork.Layers.Tensor;
using NeuralNetwork.Layers.Tensor.Convolutional;

namespace NeuralNetwork;

public class ConvolutionNetwork : INetwork<double[,,]> {
    private readonly List<ConvolutionLayer> _convs = new();
    private readonly List<Pool> _pools = new();
    private readonly List<DenseLayer> _dense = new();
    private readonly  TensorInputLayer _input;
    public readonly Flatten _flatten;
    private readonly NodeOutputLayer _output;
    public ConvolutionNetwork(int inputHeight, int inputWidth, int inputDepth, int kernelCount, int kernelWidth, int kernelHeight, int stride, int padding, int convolutionLayers, ActivationType convulationActivationType, int hiddenLayers, int hiddenNodes, ActivationType hiddenActivationType,  int outputNodes, ActivationType outputActivationType)
    {
        TensorInputLayer input = new(inputWidth,inputHeight,inputDepth);
        _input = input;

        _convs.Add(new ConvolutionLayer(input,kernelCount,kernelWidth,kernelHeight,stride,padding,convulationActivationType));
        _pools.Add(new Pool(_convs[0],2,2));
        for (int i = 1; i < convolutionLayers; i++){
            ConvolutionLayer conv = new(_pools[^1],kernelCount,kernelWidth,kernelHeight,stride,padding,convulationActivationType);
            _convs.Add(conv);
            Pool pool = new Pool(_convs[^1],2,2);
            _pools.Add(pool);
        }

        _flatten = new Flatten(_pools[^1]);
        _dense.Add(new DenseLayer(_flatten,hiddenNodes,hiddenActivationType));
        for (int i = 1; i < hiddenLayers; i++){
            DenseLayer dense = new(_dense[i-1], hiddenNodes, hiddenActivationType);
            _dense.Add(dense);
        }
        _output = new NodeOutputLayer(_dense[^1], outputNodes, outputActivationType);
    }


	public List<double> Predict(double[,,] x)
	{
	    _input.SetInputs(x);
        
        for(int  i = 0; i < _convs.Count; i++)
        {
            _convs[i].Forward();
            _pools[i].Forward();
        }
        _flatten.Forward();
		for(int i = 0; i<_dense.Count; i++){
		    _dense[i].Forward();
		}


        _output.Forward();
        return _output.GetValues();
	}

	public List<double> Train(int epochs, double learningRate, List<double[,,]> x, List<List<double>> y, ILoss<double[,,]> loss){
        List<double> lossHistory = new(epochs);
        for (int epoch = 0; epoch < epochs; epoch++){
            double lossValue = loss.Loss(x, y);
            lossHistory.Add(lossValue);
            Console.WriteLine($"Epoch number: {epoch+1}, loss:{lossValue}.");

            if(double.IsNaN(lossValue)) break;

            for (int iteration = 0; iteration< x.Count; iteration++ ){
                List<double> gradient = loss.LossGradient(x[iteration],y[iteration]);

                _output.SetErrors(gradient);
                _output.Backward(learningRate);
                for (int layer = _dense.Count-1; layer >= 0; layer--){
                    _dense[layer].Backward(learningRate);
                }
                _flatten.Backward(learningRate);
                for(int layer = _convs.Count-1; layer>=0; layer--)
                {
                    _pools[layer].Backward(learningRate);
                    _convs[layer].Backward(learningRate);
                }
                _input.Backward(learningRate);
            }

        }
        return lossHistory;
    }
}
