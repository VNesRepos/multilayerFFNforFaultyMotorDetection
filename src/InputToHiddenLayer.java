import java.util.Random;

/**
 * This class takes an input vector, weight matrix, and bias array
 * to compute the weighted input values for the hidden layer.
 */
public class InputToHiddenLayer {


    double[]v;
    int n;
    double[] b1;
    double[][] wMatrix;

    public InputToHiddenLayer(double[] vector, int numHiddenNeurons, double[][] weightMatrix, double[] biases) {
    v  = vector;
    n = numHiddenNeurons;
    wMatrix = weightMatrix;
    b1 = biases;

    }
    // This method calculates the weighted input (sum of (inputs × weights + bias) for each hidden neuron
    public double[] calcHiddenInput (){
        double[] weightedInput = new double[n];
        for (int i = 0; i < wMatrix[0].length; i++) {
            double tempSum = b1[i];
            for (int j = 0; j < v.length; j++) {
                tempSum += v[j] * wMatrix[j][i];
            }
            weightedInput[i] = tempSum;
        }
        return weightedInput;
    }

}
