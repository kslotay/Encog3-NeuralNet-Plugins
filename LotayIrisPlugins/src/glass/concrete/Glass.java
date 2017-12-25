/*
 Kulvinder Lotay
 
 glass.concrete.Glass plugin for Iris project, for creating and testing
 of a neural network for glass data

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package glass.concrete;

import java.text.SimpleDateFormat;
import java.util.Date;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import process.plugin.AbstractIris;
import process.util.Constant;
import process.util.Helper;

/**
 *
 * @author Kulvinder
 * This class overrides the createNetwork and testNetwork
 * methods of superclass AbstractIris, implementing
 * the required features as per the required specs
 * 
 */
public class Glass extends AbstractIris {    
    
    // Modify Cosntant class in process.util to change static field
    // CLASSIFYING = "glass" instead of "iris"
    
    // Modifications to input data:
    // Added header row to glass.csv file as is expected in Helper function
    // Removed ID row
    // Replaced classifier numbers with categorical names
    
    // Defines the column data types in glass-data.csv
    public final static char[] GLASS_DATA_TYPES = {
        Constant.TYPE_DECIMAL,  // refractive index
        Constant.TYPE_DECIMAL,  // Na Sodium
        Constant.TYPE_DECIMAL,  // Mg Magnesium
        Constant.TYPE_DECIMAL,  // Al Aluminium
        Constant.TYPE_DECIMAL,  // Si Silicon
        Constant.TYPE_DECIMAL,  // K Potassium
        Constant.TYPE_DECIMAL,  // Ca Calcium
        Constant.TYPE_DECIMAL,  // Ba Barium
        Constant.TYPE_DECIMAL,  // Fe Iron
        Constant.TYPE_NOMINAL,  // Glass classification
    };
    
    /*
    * Constructor
    */
    public Glass() {
        super("glass", "glass.csv", GLASS_DATA_TYPES);
    }
    
    @Override
    public void createNetwork() {
        // Create Neural Network using BasicNetwork from Encog
        // Assign to protected BasicNetwork member variable network
        network = new BasicNetwork();
        
        // Add input layer with bias, and 4 neurons
        network.addLayer(new BasicLayer(null, true, 9));
        
        // Add hidden layer with tanh activation function, bias, 36 neurons
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 36));       
        
        // Add output layer with tanh activation function, no bias, 2 neurons
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 5));    
        
        // Finalize network structure
        network.getStructure().finalizeStructure();
        
        // Reset network
        network.reset();
    }

    @Override
    public void testNetwork() {
        // Check data for the provided conditions, lengths must match up
        assert(allInputs != null && allInputs.length != 0);
        assert(allIdeals != null && allIdeals.length != 0);
        assert(allInputs.length == allIdeals.length);
        
        // Get number of input columns, -1 as last column is the glass type
        int numCols = Helper.headers.size() - 1;
        
        // Create double precision 2D arrays for testInputs
        double[][] testInputs = new double[numTestRows][numCols];      
        
        // Populate testInputs from allInputs
        for(int row=testStart; row <= testEnd; row++){            
            for(int col=0; col < numCols; col++){
                testInputs[row-testStart][col] = allInputs[row][col];               
            }
        }
        
        // Reassign numCols to value for testIdeals
        numCols = equilateral.encode(0).length;
        // Test for matching length after reassignment
        assert(equilateral.encode(0).length == numCols);
        
        // Create double precision 2D arrays for testIdeals
        double[][] testIdeals = new double[numTestRows][numCols];
        
         // Populate testIdeals from allIdeals
        for(int row=testStart; row <= testEnd; row++){
            for(int col=0; col < numCols; col++){
                testIdeals[row-testStart][col] = allIdeals[row][col];
            }
        }
        
        // Create BasicMLDataSet using testInputs and testIdeals
        BasicMLDataSet testingSet = new BasicMLDataSet(testInputs, testIdeals);
        
        // Trackers for total pass throughs, and passcount
        int totalCount = 0;
        int passCount = 0;
        
        // String holding PASSED or FAILED depending on outcome
        String passFail = "";
        
        // Formatted date for output report
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        String timeStamp = dateFormat.format(date);
        
        System.out.println("Glass ANN Report " + timeStamp + ":-");
        // Formatted string for report header row
        System.out.println(String.format("%3s %-36s %-36s %-6s", "Row", "Actual", "Predicted", "Test"));
        
        // Iterate through input data, interacting with network as needed
        for(MLDataPair dataPair: testingSet) {
            // Invoke compute on network using pair input data from BasicMLDataSet testingSet
            final MLData outputData = network.compute(dataPair.getInput());
            
            // Decode the outputData index for subtype returned from compute
            int predictedIndex = equilateral.decode(outputData.getData());
            // Decode ideal index for subtype
            int actualIndex = equilateral.decode(dataPair.getIdeal().getData());
            
            // Find the corresponding subtype for ideal and predicted
            String actualData = subtypes.get(actualIndex);
            String predictedData = subtypes.get(predictedIndex);
            
            // Increment total pass through count
            totalCount++;
            
            // Increment pass count if the data matches
            if(predictedData.equals(actualData)){
                passCount++;
                // Set passFail to PASSED
                passFail = "PASSED";
            }
            else {
                // Set passFail to FAILED on failure
                passFail = "FAILED";
            }
            
            // Print out results in correct format
            System.out.println(String.format("%3d %-36s %-36s %6s", (testStart+totalCount), actualData, predictedData, passFail));
        }
        
        // Print out statistics
        System.out.println("Passed: " + passCount + "/" + totalCount);
        System.out.println("Failed: " + (totalCount - passCount) + "/" + totalCount);
        
        // Calculate and print out the success rate
        double successRate = ((double)passCount/(double)totalCount);
        System.out.println("Success Rate: " + (int)(successRate * 100) + "%");
    }
    
}
