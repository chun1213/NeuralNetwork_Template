import java.util.ArrayList;
public class NeuralNetwork implements  java.io.Serializable{
	ArrayList<Layer> networkLayers;
	ArrayList<Connection> Connections;
	public NeuralNetwork(int nOfInputs, int nOfHiddenLayers,int nOfHiddenNodes, int nOfOutputs){
		//inputlayer
		 networkLayers = new ArrayList<Layer>();
		Layer inputLayer= new Layer();
		 for (int i=0;i<nOfInputs;i++) {
			 inputLayer.addNode(0);
		 }
		
		 networkLayers.add(inputLayer);
		 
		 //hiddenlayers
		 for (int i=0;i<nOfHiddenLayers;i++) {
			 Layer hiddenLayer = new Layer();
			 for (int x=0;x<nOfHiddenNodes;x++) {
				 hiddenLayer.addNode(0);
				 
			 }
			 
			 networkLayers.add(hiddenLayer);
		 }
		 
		 //outputlayer
		 Layer outputLayer= new Layer();
		 for (int i=0;i<nOfOutputs;i++) {
			 outputLayer.addNode(1);	 
		 }
		 networkLayers.add(outputLayer);
		 
		 addConnections();
		 
	}
	
	public void addConnections() {
		Connections = new ArrayList<Connection>();
		ArrayList<Node> nextLayer;
        ArrayList<Node> currentLayer;
		for (int i=0;i<networkLayers.size()-1;i++) {
			currentLayer=networkLayers.get(i).getNodes();
			nextLayer=networkLayers.get(i+1).getNodes();
			for (int x=0;x<currentLayer.size();x++){
				for (int n=0;n<nextLayer.size();n++){
					Connection tempConnection=new Connection();
					nextLayer.get(n).addInputConnection(tempConnection);
					currentLayer.get(x).addOutputConnection(tempConnection);
				}
			}
		}
	}
	public double[] Evaluate(double[] in) {
		double[] output = new double[networkLayers.get(networkLayers.size()-1).getNodes().size()];
		if (in.length==networkLayers.get(0).getNodes().size()) {
			
			//adds input into inputlayer and evaluates
			ArrayList<Node> inputNodes=networkLayers.get(0).getNodes();
			for (int i=0;i<inputNodes.size();i++) {
				
				inputNodes.get(i).SetInput(in[i]);
			}
			
			//evaluates hidden layers and output layer
			for (int i=1;i<networkLayers.size();i++) {
				inputNodes=networkLayers.get(i).getNodes();
				for (int x=0;x<inputNodes.size();x++) {
				
					inputNodes.get(x).SetInput();
				}
			}
			
			//returns output
			ArrayList<Node> outputNodes=networkLayers.get(networkLayers.size()-1).getNodes();
			for (int i=0;i<outputNodes.size();i++) {
				output[i]=outputNodes.get(i).getValue();
			}
			
			
			
			
			return output;
		}
		else {
			System.out.println("input does not match specified inpuut layer");
			return output;
		}
		
	}
	public ArrayList<Layer> getLayer() {
		return networkLayers;
	}
}
