import java.util.ArrayList;

public class Layer implements  java.io.Serializable{
	ArrayList<Node> Nodes;
	
	public Layer(){
	    	
	        Nodes = new ArrayList<Node>();
	        
	}
	
	public void addNode(int t) {
		Nodes.add(new Node( t));
		
	}
	
	
	public ArrayList<Node> getNodes(){
		
		return Nodes;
		}
	
	
	
	
}

