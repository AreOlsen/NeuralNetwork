    namespace NeuralNetwork.Layers.Node.Dense;

    using NeuralNetwork.Activations;

    public class DenseNode : Node {
        private readonly List<Connection> _inputs = new();
        private readonly IActivation _activation;
        public double NetInput { get; private set; }
        private double _bias=0.01d;

        public DenseNode(List<Node> inputNodes, IActivation activation){
            _activation=activation;
            foreach(Node input in inputNodes){
                _inputs.Add(new Connection(input,this, inputNodes.Count));
            }
        }

        public void Backward(double learningRate){
            AccumulateError();
            UpdateBias(learningRate);
            UpdateIncomingConnections(learningRate);
            Error = 0;
    }

        private void AccumulateError(){
            foreach(Connection con in _inputs){
                Node input = con.inputNode;
                input.Error += input.ValueDerivative() * Error * con.Weight;
            }
        }

        private void UpdateBias(double learningRate){
            _bias -= learningRate * Error;
        }

        private void UpdateIncomingConnections(double learningRate){
            foreach(Connection con in _inputs){
                con.UpdateWeight(learningRate);
            }
        }

        public void ForwardInputs()
        {
            NetInput = _bias + _inputs.Sum(node => node.GetValue());
        }

        public void ForwardValues()
        {
            Value = _activation.Calculate(NetInput);
            Error = 0;
        }

        override
        public double ValueDerivative(){
            return _activation.Derivative(NetInput);
        }
    }
