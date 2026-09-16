import java.util.Random;
import java.util.Scanner;
import java.io.*;
public class Main {

    Scanner scanner;
    Random rand = new Random(2025);


    public Main() {
        //data file names
        /**	1.	L30fft_32.out
         2.	L30fft_64.out
         3.	L30fft16.out
         4.	L30fft25.out
         5.	L30fft150.out
         6.	L30fft1000.out
         **/

        {

            try {
                scanner = new Scanner(new File("L30fft_64.out"));//change the data file here
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }
        //makes a matrix out of the file, each row of the matrix has the motors verdict 1 or 0 followed by the data vector of that motor
        int motors = scanner.nextInt();// save first int in the file to use as number of motors(though it's always 53)
        int dataPoints = scanner.nextInt() + 1;// save then number of bins of the data file chosen for training and testing and add an extra +1 space for the first column that has the verdicts
        double[][] dataMatrix = new double[motors][dataPoints];//matrix to store the data from the file, both x vectors and y results
        for (int i = 0; i < motors; i++) {// go through row by row
            double[]row = new double[dataPoints]; //an array to hold the verdict and its vector
            for (int j = 0; j < dataPoints; j++) {//fill the temporary array with the x and y data for a single sample
                row[j] = scanner.nextDouble();
            }
            dataMatrix[i] = row;// add the array x and y(single complete feed forward sample)  to the matrix(2D array) that holds all the samples both x's and y data
        }


        //makes a matrix of just the vectors, each of the length that corresponds to the number of bins in the file we are using
        //just the x label data
        double[][] inputMatrix = new double[motors][dataPoints-1];// size of the matrix of the x label data(input vector) = number of motors(rows) * granularity of the data vector - 1 column(y label data)
        double maxForNorm = 0;
        double minForNorm = 0;
        for (int i = 0; i < motors; i++) {
            for (int j = 1; j < dataPoints; j++) {

                inputMatrix[i][j-1] = dataMatrix[i][j];
                    //keep track of the smallest and largest bin in the data matrix for normalization
                if(inputMatrix[i][j-1] > maxForNorm) {
                    maxForNorm = inputMatrix[i][j-1];//new Max
                }
                if(inputMatrix[i][j-1] < minForNorm) {
                    minForNorm = inputMatrix[i][j-1];//new Min
                }
            }
        }


        int hiddenNeurons = inputMatrix[0].length/5; //number of hidden neurons the network will use is set with this variable that is then passed in the parameters

        //normalizing the input matrix based on the smallest and largest bin
        for (int i = 0; i < motors; i++) {
            for (int j = 0; j < dataPoints-1; j++) {
                inputMatrix[i][j] = (inputMatrix[i][j] - minForNorm) / (maxForNorm - minForNorm);//makes the smallest bin the new 0
                                                                                                //  / (maxForNorm - minForNorm) largest value is now 1
            }
        }

        // an array of the verdicts labeled data made from separating the y label data from x labeled in the dataMatrix
        // always ends up being an array of 34 0's followed by 19 1's as our 53 motors are always sorted by the good motors followed by bad
        double[] motorVerdicts = new double[motors];
        for (int i=0; i<motors; i++) {
            motorVerdicts[i] = dataMatrix[i][0];
        }
        double[][] motorVerdictVectors = verdictsforOutputNeurons(motorVerdicts,motors);//
        // verdictsforOutputNeurons make the list of y results into a 2D vector with 2 columns ready for the output layer to use for evaluation
        //of the output neurons



//////////////////////////////////////////////////////Part B/////////////////////////////////////////////////////////
        SeperateIn3EvenKFolds even = new SeperateIn3EvenKFolds(inputMatrix[0].length,inputMatrix,motorVerdicts);
        even.In3EvenKFolds();// make 3 Evenly sized* (18,18,17) inputs each with same ratio of good to bad motors
        even.shuffledKFold(even.k1, even.kv1, rand );// shuffle the first k-fold
        even.shuffledKFold(even.k2, even.kv2, rand );// shuffle the second k-fold
        even.shuffledKFold(even.k3, even.kv3, rand );// shuffles the third k-fold
        //storing the k - Folds that the SeperateIn3EvenKFolds class made and we then shuffled (for simplicity)
        double[][] kFold1x = even.k1;
        double[][] kFold1y = verdictsforOutputNeurons(even.kv1,18);
        double[][] kFold2x = even.k2;
        double[][] kFold2y = verdictsforOutputNeurons(even.kv2,18);
        double[][] kFold3x = even.k3;
        double[][] kFold3y = verdictsforOutputNeurons(even.kv3,17);

        //first combined Folds for training, their vectors and their verdicts
        double[][] combinedXFolds = add2KFolds(kFold1x,kFold2x);
        double[][] combinedYFolds = add2KFolds(kFold1y,kFold2y);

        //data per 100 epoch is recorded for Graph 1 to map 1+2 VS 3
        BackPropogation KfBpTrain1 = new BackPropogation(combinedXFolds ,0.1,combinedYFolds,hiddenNeurons);
        //KfBpTrain1.BackPropogate(2000);
        BackPropogation KfBpTest1 = new BackPropogation(kFold3x,0.1,kFold3y,hiddenNeurons);
        //KfBpTest1.BackPropogate(2000);

        //second combined Folds for training, their vectors and their verdicts
        double[][] combinedXFolds2 = add2KFolds(kFold1x,kFold3x);
        double[][] combinedYFolds2 = add2KFolds(kFold1y,kFold3y);

        //data per 100 epoch is recorded for Graph 1 to map 1+3 VS 2
        BackPropogation KfBpTrain2 = new BackPropogation(combinedXFolds2 ,0.1,combinedYFolds2,hiddenNeurons);
        //KfBpTrain2.BackPropogate(2000);
        BackPropogation KfBpTest2 = new BackPropogation(kFold2x,0.1,kFold2y,hiddenNeurons);
        //KfBpTest2.BackPropogate(2000);

        // Third combined Folds for training, their vectors and their verdicts
        double[][] combinedXFolds3 = add2KFolds(kFold2x,kFold3x);
        double[][] combinedYFolds3 = add2KFolds(kFold2y,kFold3y);

        //data per 100 epoch is recorded for Graph 1 to map 2+3 VS 1
        BackPropogation KfBpTrain3 = new BackPropogation(combinedXFolds3 ,0.1,combinedYFolds3,hiddenNeurons);
        //KfBpTrain3.BackPropogate(2000);
        BackPropogation KfBpTest3 = new BackPropogation(kFold1x,0.1,kFold1y,hiddenNeurons);
        //KfBpTest3.BackPropogate(2000);

        ///////////////////////////////////////////////Part C/////////////////////////////////////////////////////////

        ///graphs for part C
        //KfBpTrain1.BackPropogateWithMomentum(20000,combinedXFolds,combinedYFolds);
        //KfBpTest1.BackPropogateWithMomentum(20000,kFold3x,kFold3y);

        //KfBpTrain2.BackPropogateWithMomentum(20000,combinedXFolds2,combinedYFolds2);
        //KfBpTest2.BackPropogateWithMomentum(20000,kFold2x,kFold2y);

        //KfBpTrain3.BackPropogateWithMomentum(20000,combinedXFolds3,combinedYFolds3);
        //KfBpTest3.BackPropogateWithMomentum(20000,kFold1x,kFold1y);
        ///////////////////////////////////////////////Part D//////////////////////////////////////////////////////////
        //results for 32 BINS, 150 BINS, 1000 BINS

        //For each of the datasets 20000 epoch of training are done 5 times, the final global loss is recorded
        //Following the training the BackPropogateWithMomentum method gets the test set to run 1 epoch and record
        //the results of the trained network on unseen data
        // Note: both 20000 training epochs and the 1 test epoch need to run consecutively in one run for test to happen on a trained network

        // RESULTS for 32 Bins DATA-------------------------------------------------------------------------------


        KfBpTrain1.BackPropogateWithMomentum(20000,combinedXFolds,combinedYFolds);
        //final loss value #1 (0.0010199149786071866)
        //final loss value #2 (0.0012786089148295711)
        //final loss value #3 (0.001100707413535555)
        //final loss value #4 (0.0010772497455831518)
        //final loss value #5 (0.001090208235506141)
        //double averageL1 = (0.0012786089148295711 + 0.001090208235506141 + 0.001100707413535555 + 0.0010772497455831518 + 0.0010199149786071866)/5;
        //System.out.println(averageL1); //0.001113337857612321 average loss for 5 A + B runs


        KfBpTrain1.BackPropogateWithMomentum(1,kFold3x,kFold3y);
        // 0.13427032826860968 average loss for the test set C after training on AB
        // 0.13205788005036814 average loss for the test set C after training on AB
        // 0.13040837538334502  average loss for the test set C after training on AB
        // 0.13209934576638577 average loss for the test set C after training on AB
        // 0.13402225916460594 average loss for the test set for C after training on AB
        //double averageTestL1 = (0.13402225916460594 + 0.13209934576638577 + 0.13040837538334502 + 0.13205788005036814  + 0.13427032826860968)/5;
        //System.out.println(averageTestL1); //0.1325716377266629  average loss during testing for 5 runs on C TEST SET after trained on A + B

        //KfBpTrain2.BackPropogateWithMomentum(20000,combinedXFolds2,combinedYFolds);
        //final loss value #1 (0.02968717997782947)
        //final loss value #2 (0.029817660817661646)
        //final loss value #3 (0.029653777246616032)
        //final loss value #4 (0.0298558443462488)
        //final loss value #5 (0.02962703479556911)
        //double averageL2 = (0.02968717997782947 + 0.029817660817661646 + 0.029653777246616032 + 0.0298558443462488 + 0.02962703479556911)/5;
        //System.out.println(averageL2); //0.02972829943678501 average loss for 5 A + C training runs


        //KfBpTrain2.BackPropogateWithMomentum(1,kFold2x,kFold2y);
        // 0.14873003443800817 average loss for the test set B after training on AC
        // 0.15400912683962076 average loss for the test set B after training on AC
        // 0.1783127624949284 average loss for the test set B after training on AC
        // 0.15216178115847487 average loss for the test set B after training on AC
        // 0.14876275431581534 average loss for the test set B after training on AC
        //double averageTestL2 = (0.14873003443800817 + 0.15400912683962076 + 0.1783127624949284 + 0.15216178115847487 + 0.14876275431581534)/5;
        //System.out.println(averageTestL2);//0.1230381767806565 average loss for test set B trained on A + C

        // RESULTS for 150 Bins DATA-----------------------------------------------------------------------------

        //KfBpTrain1.BackPropogateWithMomentum(20000,combinedXFolds,combinedYFolds);
        //          final loss value #1 (3.1780312225462983E-4)
        //        //final loss value #2 (3.1677125334368027E-4)
        //        //final loss value #3 (3.1512665474338445E-4)
        //        //final loss value #4 (3.1772133802566196E-4)
        //        //final loss value #5 (3.165162277021562E-4)
        //double averageL1 = (3.1780312225462983E-4 + 3.1677125334368027E-4 +3.1512665474338445E-4 +3.1772133802566196E-4 +3.165162277021562E-4)/5;
        //System.out.println(averageL1); //3.1678771921390253E-4 average loss for 5 A + B runs

        //KfBpTrain1.BackPropogateWithMomentum(1,kFold3x,kFold3y);
        // 0.07945677214085571 average loss for the test set C after training on AB
        // 0.07918908552702991 average loss for the test set C after training on AB
        // 0.0785186327884611 average loss for the test set C after training on AB
        // 0.07941152071781822 average loss for the test set C after training on AB
        // 0.07971412008383386 average loss for the test set C after training on AB
        //double averageTestL1 = (0.07945677214085571 + 0.07918908552702991 + 0.0785186327884611 + 0.07941152071781822 + 0.07971412008383386)/5;
        //System.out.println(averageTestL1);//0.07925802625159976 average loss for test set C trained on A + B

        //KfBpTrain2.BackPropogateWithMomentum(20000,combinedXFolds2,combinedYFolds2);
        //          final loss value #1 (1.6873213701555927E-4)
        //        //final loss value #2 (1.7037265311163938E-4)
        //        //final loss value #3 (1.6874974302409545E-4)
        //        //final loss value #4 (1.6807773214572502E-4)
        //        //final loss value #5 (1.6848933924275294E-4)
        //double averageL2 = (1.6873213701555927E-4 + 1.7037265311163938E-4 + 1.6874974302409545E-4 + 1.6848933924275294E-4 +1.6848933924275294E-4)/5;
        //System.out.println(averageL2); //1.6896664232736E-4 average loss for 5 A + C runs


        //KfBpTrain2.BackPropogateWithMomentum(1,kFold2x,kFold2x);
        // 0.1312960794537338 average loss for the test set C after training on AC
        // 0.13140560510240323 average loss for the test set B after training on AC
        // 0.13148661042559281 average loss for the test set B after training on AC
        // 0.13073167518654882 average loss for the test set B after training on AC
        // 0.1308178175972955 average loss for the test set B after training on AC
        //double averageTestL1 = (0.1312960794537338 + 0.13140560510240323 + 0.13148661042559281 + 0.13073167518654882 + 0.1308178175972955)/5;
        //System.out.println(averageTestL1);//0.13114755755311486 average loss for test set B trained on A + C

        // RESULTS for 1000 Bins DATA-----------------------------------------------------------------------------
        KfBpTrain1.BackPropogateWithMomentum(20000,combinedXFolds,combinedYFolds);
        //          final loss value #1 (9.11353661672318E-5)
        //        //final loss value #2 (9.168176081796602E-5)
        //        //final loss value #3 (9.134202156391506E-5)
        //        //final loss value #4 (9.12265507711206E-5)
        //        //final loss value #5 (9.141279602722494E-5)
        //double averageL1 = (9.141279602722494E-5 + 9.12265507711206E-5 + 9.134202156391506E-5 + 09.168176081796602E-5 + 9.11353661672318E-5)/5;
        //System.out.println(averageL1); //9.135969906949167E-5 average loss for 5 A + B runs


        KfBpTrain1.BackPropogateWithMomentum(1,kFold3x,kFold3y);
        // 0.0032594811026664294 average loss for the test set C after training on AB
        // 0.0017769969441065107 average loss for the test set C after training on AB
        // 0.001872739738956315 average loss for the test set C after training on AB
        // 0.0027645653307478892 average loss for the test set C after training on AB
        // 0.0032594811026664294 average loss for the test set C after training on AB
        //double averageTestL1 = (0.0032594811026664294 + 0.0017769969441065107 + 0.001872739738956315 + 0.0027645653307478892 + 0.0032594811026664294)/5;
        //System.out.println(averageTestL1);//0.0025866528438287144 average loss for test set C trained on A + B

        //KfBpTrain2.BackPropogateWithMomentum(20000,combinedXFolds2,combinedYFolds2);
        //          final loss value #1 (6.728043790750363E-5)
        //        //final loss value #2 (6.763232678386706E-5)
        //        //final loss value #3 (6.731463698356701E-5)
        //        //final loss value #4 (6.718789387938938E-5)
        //        //final loss value #5 (6.782823743982344E-5)
        //double averageL2 = (6.728043790750363E-5 + 6.763232678386706E-5 + 6.731463698356701E-5 + 6.718789387938938E-5  + 6.782823743982344E-5)/5;
        //System.out.println(averageL2); //6.74487065988301E-5 average loss for 5 A + C runs

        //KfBpTrain2.BackPropogateWithMomentum(1,kFold2x,kFold2x);
        // 0.12828885495075323 average loss for the test set C after training on AC
        // 0.14623964350830085 average loss for the test set B after training on AC
        // 0.1217601556485612 average loss for the test set B after training on AC
        // 0.14510886034859014 average loss for the test set B after training on AC
        // 0.13837617806559996 average loss for the test set B after training on AC
        //double averageTestL1 = (0.12828885495075323 + 0.14623964350830085 + 0.1217601556485612 + 0.14510886034859014 + 0.13837617806559996)/5;
        //System.out.println(averageTestL1);//0.13595473850436107 average loss for test set B trained on A + C

        //////////////////////////////////////////Part E/////////////////////////////////////////////////////////////
        //KfBpTrain2.BackPropogationWith5Layers(2000,combinedXFolds2,combinedYFolds2);
        //KfBpTrain2.BackPropogationWith5Layers(1,kFold2x,kFold2y);//0.13350779888650763 for 1000 bins test
                                                                    //0.17556423035781674 for 16 bins test

        //KfBpTrain2.BackPropogateWithMomentum(200000,combinedXFolds2,combinedYFolds2);
        //KfBpTrain2.BackPropogateWithMomentum(1,kFold2x,kFold2y); //0.045857645525385675 for 1000 bins test
                                                                                // 0.16227985191276464 for 16 bins




    }
    // method that helps format the target data to a format that has results per output neuron
    private double[][] verdictsforOutputNeurons(double[] motorVerdicts, int motors) {
        double[][] motorVerdictVectors = new double[motors][2];
        for (int i=0; i<motors; i++) {
            if(motorVerdicts[i]==0) {
                motorVerdictVectors[i][0] = 1;
                motorVerdictVectors[i][1] = 0;
            }else{
                motorVerdictVectors[i][0] = 0;
                motorVerdictVectors[i][1] = 1;
            }
        }
        return motorVerdictVectors;

    }
    //a method for combining two K-folds to make a single fold ready to be used for trainig the network
        private double[][] add2KFolds(double[][] foldA, double[][] foldB) {
            double[][] kFoldAB = new double[foldA.length + foldB.length][foldA[0].length];
            int index = 0;

            for (int i = 0; i < foldA.length; i++) {
                kFoldAB[index++] = foldA[i];
            }
            for (int i = 0; i < foldB.length; i++) {
                kFoldAB[index++] = foldB[i];
            }
            return kFoldAB;
        }




    public static void main(String[] args) {
       new Main();
        }
    }