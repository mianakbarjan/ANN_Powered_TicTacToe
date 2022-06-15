#include <iostream>
#include <string>
#include <fstream>
#include<sstream>
#include <vector>
#include<cmath>
#include <cassert>
using namespace std;
char square[10] = { 'o','1','2','3','4','5','6','7','8','9' };
void ConvertData()
{
	ifstream file("tic-tac-toe.txt");
	ofstream file1("temp.txt", ios::app);
	for (unsigned i = 0; i < 958; i++) {
		unsigned j = 0;
		if (i < 626) {
			string array[627];
			file >> array[i];
			string a = array[i];
			file1 << "input: ";
			for (unsigned i = 0; i < 18; i += 2) {
				if (a[i] == 'x') {
					file1 << 1.0 << " ";
				}
				else if (a[i] == 'o') {
					file1 << 2.0 << " ";
				}
				else if (a[i] == 'b') {
					file1 << 0.0 << " ";
				}
			}
			file1 << endl << "output: " << 1.0 << endl;
		}
		else if (i >= 626) {
			string array[333];
			file >> array[j];
			string a = array[j];
			file1 << "input: ";
			for (unsigned i = 0; i < 18; i += 2) {
				if (a[i] == 'x') {
					file1 << 1.0 << " ";
				}
				else if (a[i] == 'o') {
					file1 << 2.0 << " ";
				}
				else if (a[i] == 'b') {
					file1 << 0.0 << " ";
				}
			}
			file1 << endl << "output: " << -1.0 << endl;
			j++;
		}
	}
	file.close();
	file1.close();
	remove("tic-tac-toe.txt");
	rename("temp.txt", "tic-tac-toe.txt");

}


int checkwin();
void board();
struct Link //Contains weight for each link between neuron from one layer and each neuron of the next layer
{
	double Weight;
	double ChangeInWeight;
};
/******************NEURON CLASS***********************/
class Neuron
{
private:
	double OutputValue; //Output Value that the Neuron shoots
	vector<Link> OutputWeight;
	unsigned Index;
	static double ActivationFunction(double x) //Activation hyperbolic function
	{
		return tanh(x);
	}
	static double DerivativeActivationFunction(double x) //Derivative of hyperbolic function to calculate change
	{
		return 1 - x * x;
	}
	double Gradient;
	static double TrainingRate;
	static double Alpha; //Multiplier of Last Weight Change
public:
	Neuron(unsigned NumberOfOutputs, unsigned Ind) //Total Number of Connections with Next layer
	{
		for (unsigned Outputs = 0; Outputs <= NumberOfOutputs; Outputs++)
		{
			OutputWeight.push_back(Link()); //Creates a New Link between Neuron from one layer and each Neuron of the Next
			OutputWeight.back().Weight = rand() / double(RAND_MAX); //Assigns Random Weights to each Link of each Neuron
		}
		Index = Ind;
	}
	void SetOutputValue(double Value)
	{
		OutputValue = Value;
	}
	double GetOutputValue()
	{
		return OutputValue;
	}
	void FeedForward(vector<Neuron>& PreviousLayer)
	{
		double sum = 0.0;

		for (unsigned Neurone = 0; Neurone < PreviousLayer.size(); Neurone++)
		{
			sum += PreviousLayer[Neurone].GetOutputValue() * PreviousLayer[Neurone].OutputWeight[Index].Weight;
			OutputValue = ActivationFunction(sum);
		}
	}
	void CalculateOutputGradients(double TargetValue)
	{
		double Change = TargetValue - OutputValue;
		Gradient = Change * Neuron::DerivativeActivationFunction(OutputValue);
	}
	double SumDOW(vector<Neuron>& NextLayer)
	{
		double sum = 0.0;
		for (unsigned Neurone = 0; Neurone < NextLayer.size() - 1; Neurone++)
		{
			sum += OutputWeight[Neurone].Weight * NextLayer[Neurone].Gradient;
		}
		return sum;
	}
	void CalculateHiddenGradients(vector<Neuron>& NextLayer)
	{
		double DOW = SumDOW(NextLayer);
		Gradient = DOW * Neuron::DerivativeActivationFunction(OutputValue);
	}
	void UpdateInputWeights(vector<Neuron>& PreviousLayer)
	{
		for (unsigned Neurone = 0; Neurone < PreviousLayer.size(); Neurone++)
		{
			Neuron& neurone = PreviousLayer[Neurone];
			double OldChangeWeight = neurone.OutputWeight[Index].ChangeInWeight;
			double NewChangeWeight = TrainingRate * neurone.GetOutputValue() * Gradient + Alpha * OldChangeWeight;
			neurone.OutputWeight[Index].ChangeInWeight = NewChangeWeight;
			neurone.OutputWeight[Index].Weight += NewChangeWeight;

		}
	}
};
double Neuron::TrainingRate = 0.4;
double Neuron::Alpha = 0.8;
/****************NEURAL NETWORK CLASS *******************/
class NeuralNetwork
{
private:
	vector<vector<Neuron>>Layers;
	double Error;
	double RecentAverageError;
	static double RecentAverageSmoothingFactor;
public:
	NeuralNetwork(vector<unsigned>& neuronsinlayers) //constructor
	{
		unsigned NumberofLayers = neuronsinlayers.size();
		for (unsigned LayerNumber = 0; LayerNumber < NumberofLayers; LayerNumber++)
		{
			unsigned NumberOfOutputs;
			if (LayerNumber == NumberofLayers - 1)
			{
				NumberOfOutputs = 0;
			}
			else
			{
				NumberOfOutputs = neuronsinlayers[LayerNumber + 1];
			}
			Layers.push_back(vector<Neuron>()); //new layer

			for (unsigned NeuronNum = 0; NeuronNum <= neuronsinlayers[LayerNumber]; ++NeuronNum) //to add specified number of neurons to each layer + a bias neuron
			{
				Layers.back().push_back(Neuron(NumberOfOutputs, NeuronNum));
				//adding neurons to current layer

			}

		}

	}
	double getRecentAverageError(void) const { return RecentAverageError; }
	void FeedForward(vector<double>& InputValues)
	{
		for (unsigned input = 0; input < InputValues.size(); input++)
		{
			Layers[0][input].SetOutputValue(InputValues[input]); //Assigns the input values to the Input Neurones

		}
		//Forward Propagation
		for (unsigned LayerNumber = 1; LayerNumber < Layers.size(); LayerNumber++)
		{
			vector<Neuron>& PreviousLayer = Layers[LayerNumber - 1];
			for (unsigned Neurone = 0; Neurone < Layers[LayerNumber].size(); Neurone++)
			{
				Layers[LayerNumber][Neurone].FeedForward(PreviousLayer);
			}
		}
	}
	void BackPropagation(vector<double>& TargetValues)
	{
		// Calculating the overall Net Error
		vector<Neuron>& OutputLayer = Layers.back();
		Error = 0.0;
		for (unsigned Neurone = 0; Neurone < OutputLayer.size() - 1; Neurone++)
		{
			double Change = TargetValues[Neurone] - OutputLayer[Neurone].GetOutputValue();
			Error += Change * Change;
		}
		Error /= OutputLayer.size() - 1; //Average Error Squared
		Error = sqrt(Error); //RootMeanSquare;
		RecentAverageError = (RecentAverageError * RecentAverageSmoothingFactor + Error) / (RecentAverageSmoothingFactor + 1.0);
		//Gradient Calculation for All Layers
		for (unsigned Neurone = 0; Neurone < OutputLayer.size() - 1; Neurone++)
		{
			OutputLayer[Neurone].CalculateOutputGradients(TargetValues[Neurone]);
		}
		for (unsigned LayerNum = Layers.size() - 2; LayerNum > 0; LayerNum--)
		{
			vector<Neuron>& HiddenLayer = Layers[LayerNum];
			vector<Neuron>& NextLayer = Layers[LayerNum + 1];
			for (unsigned Neurone = 0; Neurone < HiddenLayer.size(); Neurone++)
			{
				HiddenLayer[Neurone].CalculateHiddenGradients(NextLayer);
			}
		}
		for (unsigned LayerNum = Layers.size() - 1; LayerNum > 0; LayerNum--)
		{
			vector<Neuron>& layer = Layers[LayerNum];
			vector<Neuron>& PreviousLayer = Layers[LayerNum - 1];
			for (unsigned Neurone = 0; Neurone < layer.size() - 1; Neurone++)
			{
				layer[Neurone].UpdateInputWeights(PreviousLayer);
			}
		}
	}
	void GetResults(vector<double>& ResultValues)
	{
		ResultValues.clear();
		for (unsigned Neurone = 0; Neurone < Layers.back().size() - 1; ++Neurone)
		{
			ResultValues.push_back(Layers.back()[Neurone].GetOutputValue());
		}
	}
};
double NeuralNetwork::RecentAverageSmoothingFactor = 250.0;
/****************TRAINING DATA + TRAINING****************/
class TrainingData
{
private:
	ifstream m_trainingDataFile;
public:
	TrainingData(const string filename) {
		m_trainingDataFile.open(filename.c_str());
	}


	bool isEof(void) { return m_trainingDataFile.eof(); }

	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double>& inputVals) {
		inputVals.clear();
		string line;
		getline(m_trainingDataFile, line);
		stringstream ss(line);
		string label;
		ss >> label;
		if (label.compare("input:") == 0) {
			double oneValue;
			while (ss >> oneValue) {
				inputVals.push_back(oneValue);
			}
		}
		return inputVals.size();
	}
	unsigned getTargetOutputs(vector<double>& targetOutputVals) {
		targetOutputVals.clear();
		string line;
		getline(m_trainingDataFile, line);
		stringstream ss(line);

		string label;
		ss >> label;
		if (label.compare("output:") == 0) {
			double oneValue;
			while (ss >> oneValue) {
				targetOutputVals.push_back(oneValue);
			}
		}
		return targetOutputVals.size();
	}
};

void showVectorVals(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}







void board()
{
	system("cls");
	cout << "\n\n\tTic Tac Toe\n\n";
	cout << "Computer (X)  -  Player 2 (O)" << endl << endl;
	cout << endl;
	cout << "     |     |     " << endl;
	cout << "  " << square[1] << "  |  " << square[2] << "  |  " << square[3] << endl;
	cout << "_____|_____|_____" << endl;
	cout << "     |     |     " << endl;
	cout << "  " << square[4] << "  |  " << square[5] << "  |  " << square[6] << endl;
	cout << "_____|_____|_____" << endl;
	cout << "     |     |     " << endl;
	cout << "  " << square[7] << "  |  " << square[8] << "  |  " << square[9] << endl;
	cout << "     |     |     " << endl << endl;
}
int checkwin()
{
	if (square[1] == square[2] && square[2] == square[3])
		return 1;
	else if (square[4] == square[5] && square[5] == square[6])
		return 1;
	else if (square[7] == square[8] && square[8] == square[9])
		return 1;
	else if (square[1] == square[4] && square[4] == square[7])
		return 1;
	else if (square[2] == square[5] && square[5] == square[8])
		return 1;
	else if (square[3] == square[6] && square[6] == square[9])
		return 1;
	else if (square[1] == square[5] && square[5] == square[9])
		return 1;
	else if (square[3] == square[5] && square[5] == square[7])
		return 1;
	else if (square[1] != '1' && square[2] != '2' && square[3] != '3'
		&& square[4] != '4' && square[5] != '5' && square[6] != '6'
		&& square[7] != '7' && square[8] != '8' && square[9] != '9')
		return 0;
	else
		return -1;
}



int main()
{
	ConvertData();
	TrainingData TrainData("tic-tac-toe.txt");
	vector<unsigned>NeuronsInLayers;
	NeuronsInLayers.push_back(9); //Input Layer, 9 Neurons for 9 Inputs --> 1,2,0
	NeuronsInLayers.push_back(3); //Hidden Layer, 3 Neurons
	NeuronsInLayers.push_back(1); //Output Layer, 1 Neuron for 1 Output --> +1, -1
	NeuralNetwork Net(NeuronsInLayers);
	vector<double> InputValues, TargetValues, ResultValues;
	unsigned TrainingPass = 0;
	while (!TrainData.isEof()) {
		TrainingPass++;
		cout << endl << "Pass " << TrainingPass;

		// Get new input data and feed it forward:
		if (TrainData.getNextInputs(InputValues) != NeuronsInLayers[0]) {
			break;
		}
		showVectorVals(": Inputs:", InputValues);
		Net.FeedForward(InputValues);
		// Collect the net's actual output results:
		Net.GetResults(ResultValues);
		showVectorVals("Outputs:", ResultValues);
		// Train the net what the outputs should have been:
		TrainData.getTargetOutputs(TargetValues);
		showVectorVals("Targets:", TargetValues);
		assert(TargetValues.size() == NeuronsInLayers.back());
		Net.BackPropagation(TargetValues);
		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "
			<< Net.getRecentAverageError() << endl;
	}

	InputValues.clear();
	for (int i = 0; i < 9; i++)
	{
		InputValues.push_back(0);
	}
	int player = 1, i, choice;
	char mark;
	do
	{
		board();
		if (player % 2 == 1)
		{
			player = 1;
		}
		else
		{
			player = 2;
		}

		if (player == 1)
		{

			cout << "Player " << player << endl;
			mark = 'x';
			int MoveDone = 0;
			while (MoveDone == 0)
			{ //closing bracket add
				double BestOption = -100; //
				int BestPosition = 0;

				for (int i = 0; i < 9; i++)
				{
					if (InputValues[i] == 0)
					{
						InputValues[i] = 1;
						Net.FeedForward(InputValues);
						Net.GetResults(ResultValues);

						if (ResultValues[0] > BestOption)
						{
							BestOption = ResultValues[0];
							BestPosition = i;
						}
						InputValues[i] = 0;
					}
				}
				choice = BestPosition + 1; //check
				if (choice == 1 && square[1] == '1')
				{
					square[1] = mark;
					InputValues[0] = 1;
					MoveDone = 1;
				}
				else if (choice == 2 && square[2] == '2')
				{
					square[2] = mark;
					InputValues[1] = 1;
					MoveDone = 1;
				}
				else if (choice == 3 && square[3] == '3')
				{
					square[3] = mark;
					InputValues[2] = 1;
					MoveDone = 1;
				}
				else if (choice == 4 && square[4] == '4')
				{
					square[4] = mark;
					InputValues[3] = 1;
					MoveDone = 1;
				}
				else if (choice == 5 && square[5] == '5')
				{
					square[5] = mark;
					InputValues[4] = 1;
					MoveDone = 1;
				}
				else if (choice == 6 && square[6] == '6')
				{
					square[6] = mark;
					InputValues[5] = 1;
					MoveDone = 1;
				}
				else if (choice == 7 && square[7] == '7')
				{
					square[7] = mark;
					InputValues[6] = 1;
					MoveDone = 1;
				}
				else if (choice == 8 && square[8] == '8')
				{
					square[8] = mark;
					InputValues[7] = 1;
					MoveDone = 1;
				}
				else if (choice == 9 && square[9] == '9')
				{
					square[9] = mark;
					InputValues[8] = 1;
					MoveDone = 1;
				}
			}
			i = checkwin();
			player++;
			board();
		}
		else if (player == 2)
		{
			cout << "Player, enter a number: ";
			cin >> choice;
			mark = 'o';
			if (choice == 1 && square[1] == '1')
			{
				square[1] = mark;
				InputValues[0] = 2;
			}
			else if (choice == 2 && square[2] == '2')
			{
				square[2] = mark;
				InputValues[1] = 2;
			}
			else if (choice == 3 && square[3] == '3')
			{
				square[3] = mark;
				InputValues[2] = 2;
			}
			else if (choice == 4 && square[4] == '4')
			{
				square[4] = mark;
				InputValues[3] = 2;
			}
			else if (choice == 5 && square[5] == '5')
			{
				square[5] = mark;
				InputValues[4] = 2;
			}
			else if (choice == 6 && square[6] == '6')
			{
				square[6] = mark;
				InputValues[5] = 2;
			}
			else if (choice == 7 && square[7] == '7')
			{
				square[7] = mark;
				InputValues[6] = 2;
			}
			else if (choice == 8 && square[8] == '8')
			{
				square[8] = mark;
				InputValues[7] = 2;
			}
			else if (choice == 9 && square[9] == '9')
			{
				square[9] = mark;
				InputValues[8] = 2;
			}
			else
			{
				cout << "Invalid move ";
				player--;
				cin.ignore();
				cin.get();
			}
			i = checkwin();
			player++;
		}
	} while (i == -1);
	board();
	if (i == 1)
	{
		if (--player == 1)
		{
			cout << "==>\aComputer won";
		}
		else
		{
			cout << "==>\aPlayer won";
		}
	}
	else
		cout << "==>\aGame draw";
	cout << endl << "Before running this code again, please replace the newly created" << endl << "tic-tac-toe.txt file with the original tic-tac-toe file or an error" << " will generate" << endl << endl;
	return 0;
}