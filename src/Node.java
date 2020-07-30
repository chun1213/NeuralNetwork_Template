import java.util.ArrayList;
import java.util.Random;
public class Node implements  java.io.Serializable{
	int type=0;
	ArrayList <Connection> outConnections;
	ArrayList <Connection> inConnections;
	double value=0;
	double input=0;
	double bias;
	
	public Node(int t) {
		outConnections=new ArrayList<Connection>();
		inConnections=new ArrayList<Connection>();
		type=t;
		bias=Math.random()*1 -0.5*1;
	}
	//adds outbound connection to next layer
	public void addOutputConnection(Connection c) {
		outConnections.add(c);
	}
	//adds inbound connection from previous layer
	public void addInputConnection(Connection c) {
		inConnections.add(c);
	}
	//for hidden and output nodes
	public void SetInput() {
		for (int i=0;i<inConnections.size();i++) {
		input+=inConnections.get(i).getValue();
		
		}
		input+=bias;
		evaluate();
	}
	public double getInput() {
		return input;
	}
	
	//for input layer nodes
	public void SetInput(double in) {
		input=in;
		if (type!=1) {
			evaluate();
		}
	}
	
	
	public void evaluate() {
		//for hidden nodes
		if (type==0) {
			value=(1/( 1 + Math.pow(Math.E,(-1*input))));
			//evaluates connection value
			for (int i=0;i<outConnections.size();i++) {
				outConnections.get(i).setValue(value);
			}
		}
		
		//for output nodes 
		else if (type==1) {
			value=(1/( 1 + Math.pow(Math.E,(-1*input))));
//			if (value<0.5) {
//            	value =0;
//            }
//            else if (value>0.5) {
//            	value =1;
//            }
			
		}
		
		//for bias node
		else if (type==2) {
			value=1;
		}
		input=0;
		
	}
	public double getValue() {
		return value;
	}
	public double[] getOutWeights() {
		double weights[]= new double[outConnections.size()]; 
		for(int i=0;i<outConnections.size();i++) {
			weights[i]=outConnections.get(i).getWeight();
		}
		return weights;
	}
	public void changeOutWeights(double[] w) {
		 
		for(int i=0;i<outConnections.size();i++) {
			outConnections.get(i).changeWeight(w[i]);
		}
		
	}
	public void changeBias(double b) {
		bias=bias+b;
	}
}
