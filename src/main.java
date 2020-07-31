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







//NN Template
//Created by Chunfeng Li
//Released 07/30/2020
//This is simply a template for a Feed Forward network, NO TRAINING is happening





public class main {
	
	static int outputLayersNode=10;
	static int inputLayersNode=784;
	static int numHiddenLayers=2;
	static int HiddenLayersNode=200;
	
	static File storage=new File("brain.txt");
	//input layer nodes ,number of hidden layers, nodes in hidden layers,output layer nodes
	
	
	 public static void main(String[] args) throws IOException{
		 NeuralNetwork n  = new NeuralNetwork(inputLayersNode,numHiddenLayers,HiddenLayersNode,outputLayersNode);
		 System.out.println("success");
		
		
		 
		 
		 //simply comment this out to start a new NN!
		 n=readFromFile(storage);
		 
		 //INPUT for network as an ARRAY
		double[] input = new double[784];
		
		//Output of network as an ARRAY
		double[] output=n.Evaluate(input);
			 

//		 just comment this line out when using the test set!
		 writeToFile(storage, n);
		 
		 
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
	
}
