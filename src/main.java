import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.NotSerializableException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.*;



public class main {
	int evonum=50;
	int popsize=30;
	static int outputLayersNode=10;
	static int inputLayersNode=784;
	static int numHiddenLayers=2;
	static int HiddenLayersNode=200;
	static int trainingSize=10000;
	static double learningRate=0.00001;
	static double right=0;
	static File storage=new File("brain.txt");
	//input layer nodes ,number of hidden layers, nodes in hidden layers,output layer nodes
	
	
	 public static void main(String[] args) throws IOException{
		 NeuralNetwork n  = new NeuralNetwork(inputLayersNode,numHiddenLayers,HiddenLayersNode,outputLayersNode);
		 System.out.println("success");
		
		 //Training data
//		 MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
		 
		 //Test data
		 MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
		 
		 
		 //simply comment this out to start a new NN!
		 n=readFromFile(storage);
		 
		 
		 
		 for (int x=0;x<trainingSize;x++) {
			 System.out.println(x);
			 //Gets training data ThreadLocalRandom.current().nextInt(0, 59999 + 1)
			 MnistMatrix matrix=mnistMatrix[x];
			  int label=matrix.getLabel();
			  //gets the answer Array
			  double[] answer=labelToArray(label);
			  
			  double[] input = new double[784];
			  int counter=0;
		        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
		            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
		                input[counter]=matrix.getValue(r, c);
		                counter++;
		            }
		            
		        }
		        
		        //
			 
			 double[] output=n.Evaluate(input);
			 ArrayList<Layer> networkLayers= n.getLayer();
			 
			 
			 
			 
			 //finding cost variable
			 double tempcost;
			 double cost[]= new double[output.length];
			 for (int i=0;i<output.length;i++) {
				 tempcost=answer[i]-output[i];
				 cost[i]=tempcost;
			 }
	
//			 System.out.print(Arrays.toString(output));
//			 double avgCost=0;
//			 for (int i=0;i<cost.length;i++) {
//				 avgCost+=cost[i];
//			 }
//			 avgCost=avgCost/cost.length;
			 //System.out.println(avgCost);
			
			 if (label==getAnswer(output)) {
				 System.out.println(true);
				 right++;
			 }
			 else {
				 System.out.println(getAnswer(output));
				 System.out.println(label);
			 }
			 
			 
			 
			 RealMatrix Cost = new Array2DRowRealMatrix(cost);
			 
			 
			//derivative sigmoid output 
			 double dOutput[][]= dSigmoid(output);
//			 RealMatrix dOutputM = new Array2DRowRealMatrix(dOutput);
			 
			 //getting gradient
			 
			 
			 
//			 System.out.println(Arrays.toString(output));
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 double[][] gradient=Multiply(Cost.getData(),dOutput);
			 gradient=Multiply(gradient,learningRate);
			
			 
			
			 RealMatrix gradientM = new Array2DRowRealMatrix(gradient);
			 
			 
			 
			 ArrayList<Node> prevNodes =networkLayers.get(networkLayers.size()-2).getNodes();
			
			 
			 double[] values= new double[prevNodes.size()];
			 for (int j=0;j<prevNodes.size();j++) {
				 values[j]=prevNodes.get(j).getValue();
			 }
			 RealMatrix prevLayerOut = new Array2DRowRealMatrix(values);
			 prevLayerOut=prevLayerOut.transpose();
			 
			 
			 
			 
			 
			 
			 
			 
			 //Obtaining delta
			 gradientM=gradientM.multiply(prevLayerOut);
			 
			 

			 
			 
			 
			 
			 ArrayList<Node> currentNodes =networkLayers.get(networkLayers.size()-1).getNodes();
			 
			 //Changes the values
			 for (int j=0;j<prevNodes.size();j++) {
				 double[] deltas=gradientM.getColumn(j);
				 prevNodes.get(j).changeOutWeights(deltas);
				 

			 }
			
			 for (int j=0;j<currentNodes.size();j++) {
				 
				currentNodes.get(j).changeBias(gradient[j][0]);
				 
				
			 }
			 
			 
			
			 RealMatrix costPreviousLayer=Cost;
			 
			 
			 
			 
			 
			 
			 
			 
			 //Loops for each layer of the neural network except the last
			 for (int i=networkLayers.size()-2;i>0;i-- ) {
				 
				 //Obtains the nodes in that layer and previous
				 ArrayList<Node> nodes =networkLayers.get(i).getNodes();
				 ArrayList<Node> PrevNodes =networkLayers.get(i-1).getNodes();
				 int forwardNodeNum =networkLayers.get(i+1).getNodes().size();
				
				 RealMatrix weights=new Array2DRowRealMatrix(forwardNodeNum,nodes.size());
				 double[] currentHiddenOutputs=new double[nodes.size()];
				 //Loops through each node
				 for (int t=0;t<nodes.size();t++ ) {
					 Node currentNode=nodes.get(t);
					 
					 //puts weights and values of the node into array column
					 double tempWeight[]= currentNode.getOutWeights();
					 weights.setColumn(t,tempWeight);
					 double tempValue= currentNode.getValue();
					 currentHiddenOutputs[t]=tempValue;
				 }
				 //gets outputs of prevNodes
				 double prevHiddenOutputs[]=new double[PrevNodes.size()];
				 for (int t=0;t<PrevNodes.size();t++ ) {
					
					 Node currentNode=PrevNodes.get(t);
					 
					 //puts Value of the node into array column
					 double tempValue= currentNode.getValue();
					 prevHiddenOutputs[t]=tempValue;
				 }
				 
				 RealMatrix weights_t=weights.transpose();
				 
				 //hidden layer error
				 RealMatrix hiddenErrors = weights_t.multiply(costPreviousLayer);
				
				 
				 //gradient
				 
				 double[][] hiddenGradient=dSigmoid(currentHiddenOutputs);
				 hiddenGradient=Multiply(hiddenGradient,hiddenErrors.getData());
				 RealMatrix hiddenGradientM=new Array2DRowRealMatrix(Multiply(hiddenGradient, learningRate));
				 
				 RealMatrix PrevLayerOut = new Array2DRowRealMatrix(prevHiddenOutputs);
				 PrevLayerOut=PrevLayerOut.transpose();
				 
				 //Obtaining delta
				 hiddenGradientM=hiddenGradientM.multiply(PrevLayerOut);
				
				 
				currentNodes =networkLayers.get(networkLayers.size()-1).getNodes();
				 //changeing values
				 for (int j=0;j<PrevNodes.size();j++) {
					 double[] deltas=hiddenGradientM.getColumn(j);
					 PrevNodes.get(j).changeOutWeights(deltas);
					
				 }
				 
				 for (int j=0;j<currentNodes.size();j++) {
					 
						currentNodes.get(j).changeBias(gradient[j][0]);
						 
						
					 }
				 
				 
				 
				 
				 
				 
				 costPreviousLayer=hiddenErrors;
				 
				 
			 }
//			 System.out.println(Arrays.toString(cost));
			 
		 }
		 
		 //just comment this line out when using the test set!
//		 writeToFile(storage, n);
		 
		 System.out.println("accuracy of "+right/trainingSize*100+"% ("+right+"out of "+trainingSize+")");
		 System.out.println("done");
	 }
	 static double[][] Multiply(double[][] a,double[][] b) {
		  double[][] data=new double[a.length][a[0].length];
		      // hadamard product
		      for ( int i = 0; i < a.length; i++) {
		        for ( int j = 0; j < a[0].length; j++) {
		        
		          data[i][j] = a[i][j]*b[i][j];
		          
		        }
		      }
		      return data;
		    
	 
	 }
	 static double[][] Multiply(double[][] a,double b) {
		  double[][] data=new double[a.length][a[0].length];
		  
		      for ( int i = 0; i < a[0].length; i++) {
		        for ( int j = 0; j < a.length; j++) {
		        
		          data[j][i] = a[j][i]*b;
		          
		        }
		      }
		      return data;
		    
	 
	 }
	 static double[][] dSigmoid(double[] input) {
		//derivative sigmoid output 
		 double dOutput[][]= new double[input.length][1];
		 for (int j=0;j<input.length;j++) {
			 dOutput[j][0]=input[j]*(1-input[j]);
		 }
		 return dOutput;
	 }
	 static void print2D(double[][] matrix) {
		 for (int i = 0; i < matrix.length; i++) {
			    for (int j = 0; j < matrix[i].length; j++) {
			        System.out.print(matrix[i][j] + " ");
			    }
			    System.out.println();
			}
	 }
	 static double[] labelToArray(int l) {
		 double[] array= {0,0,0,0,0,0,0,0,0,0};
		 array[l]=1; 
		 return array; 
	 }
	 public static void writeToFile(File path, NeuralNetwork data)
	 {
	     try(ObjectOutputStream write= new ObjectOutputStream (new FileOutputStream(path)))
	     {
	         write.writeObject(data);
	     }
	     catch(NotSerializableException nse)
	     {
	         //do something
	    	 System.out.print("help");
	     }
	     catch(IOException eio)
	     {
	         //do something
	    	 
	     }
	 }


	 public static NeuralNetwork readFromFile(File path)
	 {
	     NeuralNetwork data = null;

	     try(ObjectInputStream inFile = new ObjectInputStream(new FileInputStream(path)))
	     {
	         data = (NeuralNetwork) inFile.readObject();
	         return data;
	     }
	     catch(ClassNotFoundException cnfe)
	     {
	         //do something
	    	 System.out.print("help");
	     }
	     catch(FileNotFoundException fnfe)
	     {
	         //do something
	    	 System.out.print("help");
	     }
	     catch(IOException e)
	     {
	         //do something
	    	 System.out.print("help");
	     }
	     return data;
	 }
	 static int getAnswer(double[] output) {
		 int answer=-1;
		 double highest=0;
		 for (int i=0;i<output.length;i++) {
			 if (output[i]>highest) {
				 
				 answer=i;
				 highest=output[i];
			 }
		 }
		 return answer;
	 }
	 private static void printMnistMatrix(final MnistMatrix matrix) {
	        System.out.println("label: " + matrix.getLabel());
	        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
	            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
	                System.out.print(matrix.getValue(r, c) + " ");
	            }
	            System.out.println();
	        }
	    }
}
