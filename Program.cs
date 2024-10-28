using neural;
class Program{
    static int Main(String[] args){
        Network model = new KsomNetwork(9,5,"D:/Учебники/НМ/Kohonen neural network/Dataset.txt","D:/Учебники/НМ/Kohonen neural network/Testset.csv");
        model.PrintWeights();
        model.train(400);
        model.PrintWeights();
        model.test();
        return 0;
    }
}