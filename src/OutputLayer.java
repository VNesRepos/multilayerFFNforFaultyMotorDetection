/**
 * This class applies the sigmoid activation function to the output layer’s weighted input vector
 * to produce the network’s final activated outputs.
 * the output has two probabilities for each class, 1 per neuron
 */
public class OutputLayer {

    double[] nonActivatedOutput;

    public OutputLayer(double[] outputInputVector){
      nonActivatedOutput = outputInputVector;
    }
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    public double[] finalOutput(){
        double[] finalOutput = new double[nonActivatedOutput.length];
        for (int i = 0; i < 2; i++){
            finalOutput[i] = sigmoid(nonActivatedOutput[i]);
        }
        return finalOutput;
    }
}
