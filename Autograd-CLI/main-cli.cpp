#include <stdio.h>
#include <iostream>
#include <random>
#include <format>
#include "../src/vector.h"
#include "../src/node.h"
#include "../src/sgd.h"
#include "../src/tensor4d.h"
#include "../src/conv2d.h"
#include "../mnist.h"
#include "../cifar.h"
#include "../progressmanager.h"
#include "../trainer.h"



template<typename T>
void ShuffleDataset(std::vector<NodePtr<T>>& xs, std::vector<NodePtr<T>>& labels)
{
	static std::random_device rd;
	static std::mt19937 rng(rd());

	for (size_t i = xs.size() - 1; i > 0; i--)
	{
		std::uniform_int_distribution<size_t> dist(0, i);
		size_t j = dist(rng);

		std::swap(xs[i], xs[j]);
		std::swap(labels[i], labels[j]);
	}
}



void ShuffleDataset(std::vector<std::vector<float>>& xs, std::vector<float>& labels)
{
	static std::random_device rd;
	static std::mt19937 rng(rd());

	for (size_t i = xs.size() - 1; i > 0; i--)
	{
		std::uniform_int_distribution<size_t> dist(0, i);
		size_t j = dist(rng);

		std::swap(xs[i], xs[j]);
		std::swap(labels[i], labels[j]);
	}
}



void ShuffleDataset(std::vector<Tensor4D>& xs, std::vector<float>& labels)
{
	static std::random_device rd;
	static std::mt19937 rng(rd());

	for (size_t i = xs.size() - 1; i > 0; i--)
	{
		std::uniform_int_distribution<size_t> dist(0, i);
		size_t j = dist(rng);

		std::swap(xs[i], xs[j]);
		std::swap(labels[i], labels[j]);
	}
}



struct Config
{
	std::string mode;//train, validate
	std::string dataset;
	std::string model;
	int epochs = 1;

	float lr = 0.0005f;
	float momentum = 0.5f;
	float weight_decay = 0.001f;
};



Config ParseArguments(int argc, char** argv)
{
	Config cfg;

	if (argc < 1)
		throw std::runtime_error("No command");

	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];

		if (arg == "train")
			cfg.mode = arg;
		else if (arg == "validate")
			cfg.mode = arg;
		else if (arg == "calibrate")
			cfg.mode = arg;
		else if (arg == "stats")
			cfg.mode = arg;

		else if (arg == "--dataset")
			cfg.dataset = argv[++i];
		else if (arg == "--model")
			cfg.model = argv[++i];
		else if (arg == "--epochs")
			cfg.epochs = std::stoi(argv[++i]);
		else if (arg == "--lr")
			cfg.lr = std::stof(argv[++i]);
		else if (arg == "--momentum")
			cfg.momentum = std::stof(argv[++i]);
		else if (arg == "--weight_decay")
			cfg.weight_decay = std::stof(argv[++i]);
	}

	//if (cfg.mode != "train" && cfg.mode != "validate" && cfg.mode != "stats")
		//throw std::runtime_error("Unknown mode");
	//if (cfg.dataset.empty())
		//throw std::runtime_error("Unknown dataset");
	//if (cfg.model.empty())
		//throw std::runtime_error("Unknown model");

	return cfg;
}



int main(int argc, char** argv)
{
	auto config = ParseArguments(argc, argv);

	//config.mode = "train";
	//config.model = "simple-res.tape";
	//config.dataset = "cifar-10";

	
	Trainer::DataSet data;


	if (config.dataset == "mnist")
	{
		MNist mnist = MNist();

		for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
		{
			data.train_data.push_back(mnist.GetTrainImageTensor(i));
			Tensor4D label({ 1,1,1,1 });
			label.At(0, 0, 0, 0) = (float)mnist.GetTrainingLabelData(i);
			data.train_labels.push_back(label);
		}

		for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
		{
			data.valid_data.push_back(mnist.GetValidationImageTensor(i));
			Tensor4D label({ 1,1,1,1 });
			label.At(0, 0, 0, 0) = (float)mnist.GetValidationLabelData(i);
			data.valid_labels.push_back(label);
		}
	}

	else if (config.dataset == "fashion-mnist")
	{
		MNist mnist = MNist("fashion-mnist");

		for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
		{
			data.train_data.push_back(mnist.GetTrainImageTensor(i));
			Tensor4D label({ 1,1,1,1 });
			label.At(0, 0, 0, 0) = (float)mnist.GetTrainingLabelData(i);
			data.train_labels.push_back(label);
		}

		for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
		{
			data.valid_data.push_back(mnist.GetValidationImageTensor(i));
			Tensor4D label({ 1,1,1,1 });
			label.At(0, 0, 0, 0) = (float)mnist.GetValidationLabelData(i);
			data.valid_labels.push_back(label);
		}
	}

	else if (config.dataset == "cifar-10")
	{
		Cifar mnist = Cifar();

		for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
		{
			data.train_data.push_back(mnist.GetTrainImageTensor(i));
			Tensor4D label({ 1,1,1,1 });
			label.At(0, 0, 0, 0) = (float)mnist.GetTrainingLabelData(i);
			data.train_labels.push_back(label);
		}

		for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
		{
			data.valid_data.push_back(mnist.GetValidationImageTensor(i));
			Tensor4D label({ 1,1,1,1 });
			label.At(0, 0, 0, 0) = (float)mnist.GetValidationLabelData(i);
			data.valid_labels.push_back(label);
		}
	}
	else if (config.mode != "stats")
	{
		std::cout << "Unknown dataset!" << std::endl;
		return -1;
	}

	std::cout << "Loading model ..." << std::endl;


	auto tape = TapeRecorder<Tensor4D>();
	SGD<Tensor4D> sgd(config.lr, config.momentum, config.weight_decay);
	const int batch_size = 64;

	if (!tape.LoadFromFile(config.model))
	{
		std::cout << "Error loading model!" << std::endl;
		return -1;
	}

	std::cout << "Model loaded" << std::endl;

	sgd.SetTrainableParams(tape);
	std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;


	if (config.mode == "train")
	{
		Trainer trainer(data, tape, sgd);

		for (int e = 0; e < config.epochs; e++)
		{
			trainer.TrainingLoop();

			tape.SaveToFile(config.model);

			std::cout << "\n\nEpoch " << (e+1) << " complete. Model saved!\n"  << std::endl;
		}
	}
	else if (config.mode == "validate")
	{
		Trainer trainer(data, tape, sgd);

		trainer.Validate();
	}
	else if (config.mode == "calibrate")
	{
		Trainer trainer(data, tape, sgd);

		trainer.Calibrate();
		trainer.Validate();
	}
	else if (config.mode == "stats")
	{
		tape.PrintTape();
	}
	else
		std::cout << "Unknown command!" << std::endl;

	return 0;
}