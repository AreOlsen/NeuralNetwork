using NeuralNetwork.Activations;

namespace NeuralNetwork.Layers.Tensor.Convolutional;

public class ConvolutionLayer : TensorLayer
{
    private readonly TensorLayer _input;
    private readonly IActivation _activation;
    public int Padding {get;}
    public int Stride {get;}
    private List<Kernel> _kernels = new();

    public ConvolutionLayer(TensorLayer input, int kernelCount, int kernelWidth, int kernelHeight, int stride, int padding, ActivationType activationType)
    {
        _input = input;
        Stride=stride;
        Padding=padding;
        _activation=ActivationFactory.Create(activationType);

        for(int i = 0; i < kernelCount; i++)
        {
            _kernels.Add(new Kernel(this,input.Value.Depth,kernelWidth,kernelHeight));
        }
        int outputHeight = (input.Value.Height + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth  = (input.Value.Width  + 2 * padding - kernelWidth ) / stride + 1;
        Value = new Tensor(kernelCount,outputWidth,outputHeight);
    }

    override
    public void Forward()
    {
        foreach(Channel channel in Value.Channels)
        {
            channel.Clear(); // Clears NetInputs and Values back to 0
        }
        
        for(int i = 0;  i < _kernels.Count; i++)
        {
            Kernel kernel = _kernels[i];    
            Channel channel = Value.Channels[i];
            kernel.Forward(_input.Value, channel);
            channel.ApplyActivation(_activation);
        }
    }

    override
    public void Backward(double learningRate)
    {
        Tensor input = _input.Value;

        foreach(Channel channel in input.Channels)
        {
            channel.ClearErrors();
        }

        int valueDepth = Value.Depth;
        for(int outputDepth = 0; outputDepth<valueDepth; outputDepth++)
        {
            Channel outputChannel = Value.Channels[outputDepth];
            Kernel kernel = _kernels[outputDepth];
            int outputWidth = outputChannel.Width;
            int outputHeight = outputChannel.Height;

            for(int column = 0; column < outputWidth; column++)
            {
                for(int row = 0; row < outputHeight; row++)
                {
                    double error = outputChannel.Errors[column,row]*_activation.Derivative(outputChannel.NetInputs[column,row]);
                    kernel.AccumulateInputError(input,column,row,error);
                    kernel.AccumulateBiasDerivative(error);
                    kernel.AccumulateWeightGradient(input, column, row, error);
                }
            }
        }

        foreach(Kernel kernel in _kernels)
        {
                kernel.Backward(learningRate);
        }
    }
}
