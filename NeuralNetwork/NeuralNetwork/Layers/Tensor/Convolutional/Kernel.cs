namespace NeuralNetwork.Layers.Tensor.Convolutional;


public class Kernel  {
    public double[,,] Weights {get;}
    public double[,,] WeightGradient {get;}
    public int Width {get;}
    public int Height {get;}
    public double Bias {get; private set;}
    public double BiasDerivative {get; private set;}
    private readonly ConvolutionLayer _layer;

    public Kernel(ConvolutionLayer layer, int depth, int width, int height)
    {   
        _layer = layer;
        Weights = new double[depth,width,height];
        double scale = Math.Sqrt(2.0 / (depth * width * height));
        for(int i = 0; i < depth; i++)
            for(int ii = 0; ii < width; ii++)
                for(int iii=0; iii < height; iii++)
                    Weights[i,ii,iii]=(Random.Shared.NextDouble()*2-1)*scale;
        WeightGradient = new double[depth,width,height];
        Width=width;
        Height=height;
        Bias=0.01d;
        BiasDerivative=0d;
    }

    public void Forward(Tensor input, Channel output)
    {
        int stride  = _layer.Stride;
        int padding = _layer.Padding;
        int layerHeight = _layer.Value.Height;
        int layerWidth = _layer.Value.Width;

        //Go through all value coords in output channel produced.
        for(int outCol = 0; outCol < layerWidth; outCol++){
            for(int outRow = 0; outRow < layerHeight; outRow++)
            {
                //Go through all input channels and apply convolution.
                double sum = Bias;
                for(int depth = 0;  depth < input.Channels.Count; depth++)
                {
                    for(int kernelX = 0; kernelX < Width; kernelX++)
                    {
                        for(int kernelY = 0; kernelY < Height; kernelY++)
                        {
                            //Get original input position.
                            int inCol = outCol*stride-padding + kernelX;
                            int inRow = outRow*stride-padding+kernelY;
                            if(inCol< 0 || inRow < 0 || inCol>=input.Width || inRow>=input.Height) continue;

                            //Get weighted input for channel.
                            sum+=input.Channels[depth].Values[inCol,inRow]*Weights[depth,kernelX,kernelY];
                        }
                    }
                }
                
                //Apply convolution.
                output.NetInputs[outCol,outRow]=sum;
            }
        }
    }

    public void Backward(double learningRate)
    {
        Bias -= BiasDerivative*learningRate;
        BiasDerivative=0;

        for(int depth = 0;  depth < Weights.GetLength(0); depth++)
        {
            for(int x = 0; x < Weights.GetLength(1); x++)
            {
                for(int y = 0;  y < Weights.GetLength(2); y++)
                {
                    Weights[depth, x, y] -= WeightGradient[depth, x, y]*learningRate;
                    WeightGradient[depth, x, y]=0;
                }
            }   
        }
    }

    public void AccumulateBiasDerivative(double outChannelError)
    {
        BiasDerivative+=outChannelError;
    }


    public void AccumulateWeightGradient(Tensor input, int column, int row, double outputError)
    {
            int stride = _layer.Stride;
        int padding = _layer.Padding;
        for(int inputDepth = 0; inputDepth < input.Depth; inputDepth++)
        {
            Channel inputChannel = input.Channels[inputDepth];

            for(int kernelX = 0; kernelX < Width; kernelX++)
            {
                for(int kernelY = 0; kernelY < Height; kernelY++)
                {
                    int inputRow = row*stride-padding+kernelY;
                    int inputColumn = column*stride-padding + kernelX;

                    if (inputRow < 0 || inputColumn < 0 ||
                        inputRow >= input.Height || inputColumn >= input.Width) continue;
                        
                    double inputValue = inputChannel.Values[inputColumn,inputRow];
                    WeightGradient[inputDepth,kernelX,kernelY]+=inputValue*outputError;
                }
            }
        }
    }
    
    public void AccumulateInputError(Tensor input, int column, int row, double outputError)
    {
        int stride = _layer.Stride;
        int padding = _layer.Padding;
        for(int inputDepth = 0; inputDepth < input.Depth; inputDepth++)
        {
            Channel inputChannel = input.Channels[inputDepth];

            for(int kernelX = 0; kernelX < Width; kernelX++)
            {
                for(int kernelY = 0; kernelY < Height; kernelY++)
                {
                    int inputRow = row*stride-padding+kernelY;
                    int inputColumn = column*stride-padding + kernelX;

                    if (inputRow < 0 || inputColumn < 0 ||
                        inputRow >= input.Height || inputColumn >= input.Width) continue;


                    double kernelWeight = Weights[inputDepth, kernelX, kernelY];
                    inputChannel.Errors[inputColumn,inputRow]+=outputError*kernelWeight;
                }
            }
        }
    }
}