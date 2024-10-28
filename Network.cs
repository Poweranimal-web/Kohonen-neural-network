using System.IO;
using System.Text;
using System.Linq;
namespace neural{
    interface Network{
        public void train(int epoch);
        public void test();
        public void PrintWeights();
        public void ReadTrainCSVFile(string path);
        public void ReadTestCSVFile(string path);
    }
    class KsomNetwork : Network{
        static int amount_inputs = 0;
        static int amount_outputs = 0;
        List<float[,]> TrainDataSets = new List<float[,]>();
        List<float[,]> TestDataSets = new List<float[,]>();
        float[,] Weights;
        float[,] Outputs;
        int? numberWinNeuron;
        float alpha = 0.01f;
        public KsomNetwork(int input, int output, string trainPath,string testPath){
            amount_inputs = input;
            amount_outputs = output;
            Weights = new float[amount_outputs, amount_inputs];
            Outputs = new float[amount_outputs,1];
            ReadTrainCSVFile(trainPath);
            ReadTestCSVFile(testPath);
            GenerateWeights();
        }
        float[,] ConvertTo2DMatrix(float[] array){
            Console.WriteLine(array.Length);
            float[,] matrix = new float[array.Length,1];
            for (int i = 0; i < array.Length; i++)
            {
                matrix[i,0] = array[i];
            }
            return matrix;


        }
        public void ReadTrainCSVFile(string path){
            StreamReader? csvFile = null;
            try{
                Console.WriteLine("Reading train file ...");
                csvFile = new StreamReader(path);
                while (csvFile.Peek() >= 0)
                {
                    float[] arrayInstance = Array.ConvertAll(csvFile.ReadLine().Split(";"), float.Parse); 
                    float[,] matrix = ConvertTo2DMatrix(arrayInstance);
                    TrainDataSets.Add(matrix);
                }
            }
            catch(Exception e){
                Console.WriteLine($"{e.Message}");
            }
            finally{
                Console.WriteLine("Read train file!!");
                csvFile.Close();
            }

        }
        public void ReadTestCSVFile(string path){
            StreamReader? csvFile = null;
            try{
                Console.WriteLine("Reading test file...");
                csvFile = new StreamReader(path);
                while (csvFile.Peek() >= 0)
                {
                    float[] arrayInstance = Array.ConvertAll(csvFile.ReadLine().Split(";"), float.Parse);  
                    float[,] matrix = ConvertTo2DMatrix(arrayInstance);
                    TestDataSets.Add(matrix);
                }
            }
            // catch(Exception e){
            //     Console.WriteLine($"{e.Message}");
            // }
            finally{
                Console.WriteLine("Read test file!!");
                csvFile.Close();
            }

        }

        void GenerateWeights(){
            Random rnd = new Random(10);
            double sumIntOut = amount_inputs + amount_outputs;
            float min_value_weights = -(float)((Math.Sqrt(6))/(Math.Sqrt(sumIntOut)));
            float max_value_weights = (float)((Math.Sqrt(6))/(Math.Sqrt(sumIntOut)));
            Console.WriteLine($"Max value:{max_value_weights}, Min value: {min_value_weights}");
            for (int row = 0; row < Weights.GetLength(0); row++)
            {
                for (int col = 0; col < Weights.GetLength(1); col++)
                {
                    Weights[row, col] = min_value_weights + (float)rnd.NextDouble() * (max_value_weights-min_value_weights);
                }
            }
        }
        void GenerateWeights(int inputs){
            for (int row = 0; row < Weights.GetLength(0); row++)
            {
                for (int col = 0; col < Weights.GetLength(1); col++)
                {
                    Weights[row, col] = (float)1/inputs;
                }
            }
        }
        public void PrintWeights(){
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int f = 0; f < Weights.GetLength(1); f++)
                {
                    Console.Write($"{Weights[i,f]} ");  
                }
                Console.WriteLine("");
            }
        }
        public void PrintOutputs(){
            for (int i = 0; i < Outputs.GetLength(0); i++)
            {
                for (int f = 0; f < Outputs.GetLength(1); f++)
                {
                    Console.Write($"{Outputs[i,f]} ");  
                }
                Console.WriteLine("");
            }
        }
        public void PrintInputVector(float[,] vector, ref int counts){
            if(counts > 0){
                for (int i = 0; i < vector.GetLength(0); i++)
                {
                    Console.Write($"{vector[i,0]};");
                }
                Console.WriteLine("");
            }
            counts--;
            
        }
        public void PrintInputVector(float[,] vector){
                for (int i = 0; i < vector.GetLength(0); i++)
                {
                    Console.Write($"{vector[i,0]};");
                }
        }
        public void Print(List<float[,]> vectors){
            for (int i = 0; i < vectors.Count; i++)
            {
                for (int row = 0; row < vectors[i].GetLength(0); row++)
                {
                    
                    Console.Write($"{vectors[i][row,0]};"); 
                    

                }
                Console.WriteLine("");
            }
        }
        object[] GetMax(){
            object[] MaxElementData = new object[2];
            float maxValue = 0;
            int maxIndexValue = 0;
            for (int f = 0; f < Outputs.GetLength(0); f++)
            {
                if (maxValue < Outputs[f,0]){
                        maxValue = Outputs[f,0];
                        maxIndexValue = f;
                }
            }
            MaxElementData[0] = maxValue;
            MaxElementData[1] = maxIndexValue;
            return MaxElementData;

            
        }
        object[] GetMax(int epoch){
            object[] MaxElementData = new object[2];
            float maxValue = 0;
            int maxIndexValue = 0;
            if (epoch>=0 && epoch<=2){
                Console.WriteLine("");
                for (int f = 0; f < Outputs.GetLength(0); f++)
                {
                    Console.WriteLine($"Output{f}: {Outputs[f,0]}");
                    if (maxValue < Outputs[f,0]){
                        maxValue = Outputs[f,0];
                        maxIndexValue = f;
                    }
                }
                Console.WriteLine("");
            }
            else {
                for (int f = 0; f < Outputs.GetLength(0); f++)
                {
                    Console.WriteLine($"Output{f}: {Outputs[f,0]}");
                    if (maxValue < Outputs[f,0]){
                        maxValue = Outputs[f,0];
                        maxIndexValue = f;
                    }
                }
                Console.WriteLine("");
            }
            MaxElementData[0] = maxValue;
            MaxElementData[1] = maxIndexValue;
            return MaxElementData;

            
        }
        void forward_pass(float[,] input, int[] neurons){
            Random rnd = new Random();
            int[] dropout = {rnd.Next(0,neurons.Length),rnd.Next(0,neurons.Length),rnd.Next(0,neurons.Length)};
            for (int row = 0; row < Weights.GetLength(0); row++)
            {
                
                float stream_result = 0;
                if (!dropout.Contains(row)){
                    for (int col = 0; col < Weights.GetLength(1); col++)
                    {
                            
                            stream_result += Weights[row,col] * input[col,0]; 
                            
                    }
                }
                Outputs[row,0] = stream_result;
                stream_result = 0;
            }
        }
        void forward_pass(float[,] input){
            for (int row = 0; row < Weights.GetLength(0); row++)
            {
                float stream_result = 0;
                for (int col = 0; col < Weights.GetLength(1); col++)
                {
                    
                    stream_result += Weights[row,col] * input[col,0]; 
                    
                }
                Outputs[row,0] = stream_result;
                stream_result = 0;
            }
        }
        void backward_pass(float[,] input,int epoch,int indexMaxValueNeuron){
            float B_t = CalculateRadius(epoch);
            for (int row=0; row < Weights.GetLength(0); row++){
                if (IsInNeighborhood(row, indexMaxValueNeuron, B_t)){
                    for (int i = 0; i < amount_inputs; i++)
                    {
                        float x = input[i,0];
                        Weights[row, i] = (float)(Weights[row, i] + (alpha*(x-Weights[row, i])));
                    }
                }
            }

        } 
        void transformVector(ref float[,] input, float B){
            for (int i = 0; i < input.GetLength(0); i++)
            {
                input[i,0] = (float)(input[i,0]*B) +(float)((1-B)/Math.Sqrt(amount_inputs));
            }

        }
        float CalculateRadius(int epoch) {
            float initialRadius = 4.0f;
            float timeConstant = 1.0f;
            return initialRadius * (float)Math.Exp(-epoch / timeConstant);
        }
        bool IsInNeighborhood(int neuronIndex, int winnerIndex, float radius) {
            return Math.Sqrt(Math.Pow(neuronIndex - winnerIndex,2)) <= radius;
        }
        public void train(int epoch){
            float B = 0;
            Console.WriteLine("Training....");
            Console.WriteLine("Goes through layers...");
            object[] MaxValue = null;
            int[] neurons = new int[amount_outputs];
            int count = 20;
            for (int i = 0; i < epoch; i++)
            {
                B = (float)(i/epoch);
                for (int p = 0; p < TrainDataSets.Count; p++)
                {
                    float[,] vector = TrainDataSets[p]; 
                    transformVector(ref vector, (float)B);
                    PrintInputVector(vector, ref count);
                    forward_pass(vector,neurons);
                    MaxValue = GetMax();
                    neurons[(int)MaxValue[1]] = neurons[(int)MaxValue[1]] + 1; 
                    backward_pass(TrainDataSets[p], i,(int)MaxValue[1]);
                }
            }
        }
        public void test(){
            Console.WriteLine("Testing....");
            Console.WriteLine("Goes through layers...");
            for (int p = 0; p < TestDataSets.Count; p++)
            {
                    
                float[,] vector = TestDataSets[p]; 
                forward_pass(vector);
                object[] MaxValue = GetMax();
                PrintInputVector(vector);
                Console.Write($" class: {(int)MaxValue[1]}");
                Console.WriteLine("");

            }
        }
    }
}