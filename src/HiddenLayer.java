/**
 * This class represents the hidden layer of the neural network.
 * Its only purpose is to apply the sigmoid activation function to the input vector
 * and stores the resulting activated outputs to use as non-weighted inputs to the output layer
 */

public class HiddenLayer {
    double[] h;
    double[] HidLayOut;
    public HiddenLayer(double[] hiddenInputVector) {
        h = hiddenInputVector;
    }

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    public double[] activatedHiddenLayer() {
        double[] hiddenLayerOutput = new double[h.length];
        for (int i = 0; i < h.length; i++) {
            hiddenLayerOutput[i] = sigmoid(h[i]);
        }
        HidLayOut = hiddenLayerOutput;
        return hiddenLayerOutput;
    }
}
