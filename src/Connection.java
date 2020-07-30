import java.util.Random;

public class Connection implements  java.io.Serializable{
	double weight;
	
	double value;
	int initialWeightInterval=1;
	public Connection() {
		
		weight=Math.random()*initialWeightInterval -0.5*initialWeightInterval;
		
	}
	public void setValue(double nodeValue) {
		value=weight*nodeValue;
	}
	
	public double getValue() {
		return value;
	}
	public double getWeight() {
		return weight;
	}
	
	public void changeWeight(double w) {
	
		weight=weight+w;
	
	}
}
