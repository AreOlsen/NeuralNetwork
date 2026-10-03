namespace NeuralNetwork.Layers.Node.Dense;

public record Connection(Node inputNode, DenseNode outputNode, int fanIn) {

    public double Weight { get; private set; } = (Random.Shared.NextDouble()*2-1)*Math.Sqrt(2.0/fanIn);

    public void UpdateWeight(double learningRate){
        Weight -= learningRate * outputNode.Error * inputNode.Value;
    }

	public double GetValue(){
	    return Weight*inputNode.Value;
	}
}
