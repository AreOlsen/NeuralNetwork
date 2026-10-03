namespace NeuralNetwork.Layers.Tensor;

public class Pool : TensorLayer
{
    private readonly TensorLayer _input;
    private readonly int _stride;
    private readonly int _poolSize;
    private (int col, int row)[,,] _maxSource;

    public Pool(TensorLayer inputLayer, int poolSize, int stride)
    {
        _input = inputLayer;
        _stride=stride;
        _poolSize=poolSize;
        int outputHeight = (inputLayer.Value.Height - poolSize) / stride + 1;
        int outputWidth  = (inputLayer.Value.Width  - poolSize) / stride + 1;
        Value = new Tensor(inputLayer.Value.Depth,outputWidth,outputHeight);
        _maxSource=new (int,int)[inputLayer.Value.Depth,outputWidth,outputHeight];
    }

    override
    public void Forward(){
        Tensor input = _input.Value;

        for(int channel = 0; channel < input.Depth; channel++)
        {
            Channel inChannel = input.Channels[channel];
            Channel outChannel = Value.Channels[channel];

            //Go through all values to write to in output channel.
            for(int outRow = 0; outRow < outChannel.Height; outRow++)
            {
                for(int outCol = 0; outCol < outChannel.Width; outCol++)
                {
                    //Get over window, get max value.
                    int windowRowStart = outRow*_stride;
                    int windowColStart= outCol*_stride;
                    double maxValue = double.MinValue;
                    for(int windowRow = 0; windowRow<_poolSize; windowRow++)
                    {
                        for(int windowCol = 0; windowCol <_poolSize; windowCol++)
                        {
                            double value = inChannel.Values[windowColStart+windowCol,windowRowStart+windowRow];
                            if(value>maxValue){
                                maxValue=value;
                                _maxSource[channel, outCol,outRow]=(windowColStart+windowCol,windowRowStart+windowRow);
                            }
                        }
                    }
                    //Set cell to max value found in window.
                    outChannel.Values[outCol,outRow]=maxValue;
                }
            }
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

        for(int depth = 0; depth < input.Depth; depth++)
        {
            Channel inChannel = input.Channels[depth];
            Channel outChannel = Value.Channels[depth];
            for(int row = 0; row < outChannel.Height; row++)
            {
                for(int col = 0; col < outChannel.Width; col++)
                {
                    var (maxColumn, maxRow) = _maxSource[depth,col,row];
                    inChannel.Errors[maxColumn,maxRow]+=outChannel.Errors[col,row];
                    outChannel.Errors[col,row]=0;
                }
            }
        }
    }
}