#include <stdio.h>
#include <iostream>
#include <random>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#include "src/scalar.h"
#include "src/vector.h"
#include "src/node.h"
#include "src/neuron.h"
#include "src/layer.h"
#include "src/sgd.h"
#include "src/tensor4d.h"
#include "src/conv2d.h"
#include "mnist.h"
#include "cifar.h"



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



void MNist_Tensor4D()
{
	MNist mnist = MNist();
	//MNist mnist = MNist("mnist-fashion");

	mnist.PrintTrainImage(0);
	mnist.PrintTrainImage(1);
	mnist.PrintTrainImage(2);

	std::vector<Tensor4D> xs_train;
	std::vector<float> labels_train;

	std::vector<Tensor4D> xs_val;
	std::vector<float> labels_val;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		//xs_train_old.push_back(mnist.GetTrainingImageData(i));
		xs_train.push_back(mnist.GetTrainImageTensor(i));
		labels_train.push_back((float)mnist.GetTrainingLabelData(i));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		xs_val.push_back(Tensor4D({ 1, 1, 28*28, 1 }, mnist.GetValidationImageData(i)));
		//xs_val.push_back(mnist.GetValidationImageTensor(i));
		labels_val.push_back((float)mnist.GetValidationLabelData(i));
	}


	auto tape = TapeRecorder<Tensor4D>();
	SGD<Tensor4D> sgd(0.001f, 0.3f);
	const int batch_size = 1;

	{
		//auto x = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28, 28 }), "input");
		auto x = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28, 28 }), "input");
		auto label = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 1, 1 }), "label");

		Conv2D conv1(1,  8, 3, 1, 1, Activation::ReLU);
		Conv2D conv2(8, 16, 3, 1, 1, Activation::ReLU);
		//Layer4D L1(900, 128, Activation::ReLU,     InitType::Xavier);
		//Layer4D L1(28*28, 128, Activation::ReLU,     InitType::Xavier);
		Layer4D L1(3136, 128, Activation::ReLU, InitType::Xavier);
		//Layer4D L1(1568, 128, Activation::ReLU, InitType::Xavier);
		Layer4D L2(128,  10, Activation::Identity, InitType::Xavier);

		//auto out_conv1 = conv1.Conv(x)->MaxPool2D(2, 2);
		//auto out_conv2 = conv2.Conv(out_conv1)->MaxPool2D(2, 2);

		//auto out_conv1 = conv1.Conv(x)->MaxPool2D(2, 2);

		auto out_conv1 = conv1.Conv(x);
		auto out_conv2 = conv2.Conv(out_conv1)->MaxPool2D(2, 2);
		auto flattened = out_conv2->Flatten();

		//auto flattened = x->Flatten();
		//auto flattened = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28*28, 1 }), "flattened");
		flattened->SetLabel("flattened");

		auto h1 = L1.Forward(flattened);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto conv_weights = tape.SetValue("conv_weights");
	auto dconv_weights = tape.GetGradient("conv_weights");
	auto input  = tape.SetValue("input");
	auto dinput = tape.GetGradient("input");
	auto flattened = tape.SetValue("flattened");
	auto dflattened = tape.GetGradient("flattened");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	tape.PrintTape();

	size_t train_size = xs_train.size() / 1;


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < train_size;)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_train.size())
			{
				input->ViewBatch(actual_batch_size).CopyFrom(xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_train[i + j];

				if (pred_class == target_class) correct++;
			}

			i += actual_batch_size;
		}


		for (size_t i = 0; i < xs_val.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_val.size())
			{
				input->ViewBatch(actual_batch_size).CopyFrom(xs_val[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0,{ labels_val[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_val[i + j];

				if (pred_class == target_class) val_correct++;
			}

			i += actual_batch_size;
		}

		return std::tuple{ (float)correct * 100.0f / train_size, (float)val_correct * 100.0f / xs_val.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 5; epoch++)
	{
		ShuffleDataset(xs_train, labels_train);


		epoch_loss = 0.0f;

		DWORD start_time = timeGetTime();

		for (size_t i = 0; i < train_size; i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs_train.size());

			float batch_loss = 0.0f;

			size_t actual_batch_size = 0;

			for (size_t j = i; j < end; j++)
			{
				input->ViewBatch(actual_batch_size).CopyFrom(xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			if (actual_batch_size != batch_size)
				continue;

			tape.Forward();
			tape.ZeroGradients();
			tape.Backward();

			epoch_loss += loss->At(0, 0, 0, 0) * actual_batch_size;
			batch_loss += loss->At(0, 0, 0, 0) * actual_batch_size;

			if (i / batch_size % (train_size / 50) == 0)
				std::cout << "Batch " << i / batch_size + 1 << " of " << train_size / batch_size << " batch_loss " << batch_loss / batch_size << std::endl;

			sgd.Step();
		}

		epoch_loss /= train_size;

		auto epoch_duration = timeGetTime() - start_time;
		auto samples_per_second = (float)train_size / epoch_duration * 1000.0f;

		std::cout << "batch_size: " << batch_size << std::endl;
		std::cout << "samples_per_second: " << samples_per_second << std::endl;
		std::cout << "Epoch " << epoch + 1 << " Loss: " << epoch_loss << std::endl;
		auto [train_acc, val_acc] = calculate_acc();
		std::cout << "Train Accuracy: " << train_acc << "%" << std::endl;
		std::cout << "Valid Accuracy: " << val_acc << "%" << std::endl;

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	/*auto [train_acc, val_acc] = calculate_acc();
	std::cout << "Final Train Accuracy: " << train_acc << "%" << std::endl;
	std::cout << "Final Valid Accuracy: " << val_acc   << "%" << std::endl;*/
}



void MNist_Tensor()
{
	MNist mnist = MNist();
	//MNist mnist = MNist("mnist-fashion");

	mnist.PrintTrainImage(0);
	mnist.PrintTrainImage(1);
	mnist.PrintTrainImage(2);

	std::vector<std::vector<float>> xs_train;
	std::vector<float> labels_train;

	std::vector<std::vector<float>> xs_val;
	std::vector<float> labels_val;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		//xs.push_back(Node<Tensor>::Create(Tensor({ 28*28, 1 }, mnist.GetTrainingImageData(i))));
		xs_train.push_back(mnist.GetTrainingImageData(i));
		labels_train.push_back((float)mnist.GetTrainingLabelData(i));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		xs_val.push_back(mnist.GetValidationImageData(i));
		labels_val.push_back((float)mnist.GetValidationLabelData(i));
	}


	auto tape = TapeRecorder<Tensor>();
	SGD<Tensor> sgd(0.03f, 0.7f);
	const int batch_size = 2;

	{
		auto x = Node<Tensor>::Create(Tensor({ 28 * 28, batch_size }), "input");
		auto label = Node<Tensor>::Create(Tensor({ batch_size }), "label");

		LayerTensor L1(784, 128, Activation::ReLU, InitType::Xavier);
		LayerTensor L2(128,  10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		//auto loss = (  (pred - label)->ElementwiseMul(pred - label)  )->Sum();
		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto input  = tape.SetValue("input");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs_train.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_train.size())
			{
				input->SetColumn(actual_batch_size, xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				input->Print();
				output->Print();
				auto output_row = output->ViewColumn(j);
				
				//Convert one hot to class index
				int pred_class = (int)output_row.ArgMax();
				int target_class = (int)labels_train[i + j];

				if (pred_class == target_class) correct++;
			}

			i += actual_batch_size;
		}


		for (size_t i = 0; i < xs_val.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_val.size())
			{
				input->SetColumn(actual_batch_size, xs_val[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, { labels_val[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				auto output_row = output->ViewColumn(j);

				//Convert one hot to class index
				int pred_class = (int)output_row.ArgMax();
				int target_class = (int)labels_val[i + j];

				if (pred_class == target_class) val_correct++;
			}

			i += actual_batch_size;
		}

		return std::tuple{ (float)correct * 100.0f / xs_train.size(), (float)val_correct * 100.0f / xs_val.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 3; epoch++)
	{
		//ShuffleDataset(xs_train, labels_train);


		epoch_loss = 0.0f;

		DWORD start_time = timeGetTime();

		for (size_t i = 0; i < xs_train.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs_train.size());

			float batch_loss = 0.0f;

			size_t actual_batch_size = 0;

			for (size_t j = i; j < end; j++)
			{
				input->SetColumn(actual_batch_size, xs_train[j]);
				label->SetColumn(actual_batch_size, { labels_train[j] });
				actual_batch_size++;
			}

			tape.Forward();
			tape.ZeroGradients();
			tape.Backward();

			epoch_loss += loss->At({ 0 });
			batch_loss += loss->At({ 0 });

			if (i / batch_size % 1000 == 0)
				std::cout << "Batch " << i / batch_size + 1 << " of " << xs_train.size() / batch_size << " batch_loss " << batch_loss / batch_size << std::endl;

			sgd.Step();
		}

		epoch_loss /= xs_train.size();

		auto epoch_duration = timeGetTime() - start_time;
		auto samples_per_second = (float)xs_train.size() / epoch_duration * 1000.0f;

		std::cout << "batch_size: " << batch_size << std::endl;
		std::cout << "samples_per_second: " << samples_per_second << std::endl;
		std::cout << "Epoch " << epoch + 1 << " Loss: " << epoch_loss << std::endl;
		auto [train_acc, val_acc] = calculate_acc();
		std::cout << "Train Accuracy: " << train_acc << "%" << std::endl;
		std::cout << "Valid Accuracy: " << val_acc << "%" << std::endl;

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	/*auto [train_acc, val_acc] = calculate_acc();
	std::cout << "Final Train Accuracy: " << train_acc << "%" << std::endl;
	std::cout << "Final Valid Accuracy: " << val_acc   << "%" << std::endl;*/
}



void MNist_Test()
{
	//MNist mnist = MNist();
	MNist mnist = MNist("mnist-fashion");

	mnist.PrintTrainImage(0);
	mnist.PrintTrainImage(1);
	mnist.PrintTrainImage(2);

	std::vector<NodePtr<Matrix>> xs;
	std::vector<NodePtr<Matrix>> labels;

	std::vector<NodePtr<Matrix>> val_xs;
	std::vector<NodePtr<Matrix>> val_labels;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs.push_back(Node<Matrix>::Create(mnist.GetTrainingImageData(i)));

		int label = mnist.GetTrainingLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		//labels.push_back(Node<Matrix>::Create(one_hot));
		labels.push_back(Node<Matrix>::Create({ (float)label }));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		val_xs.push_back(Node<Matrix>::Create(mnist.GetValidationImageData(i)));

		int label = mnist.GetValidationLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		//val_labels.push_back(Node<Matrix>::Create(one_hot));
		val_labels.push_back(Node<Matrix>::Create({ (float)label }));
	}


	auto tape = TapeRecorder<Matrix>();
	SGD<Matrix> sgd(0.03f, 0.7f);
	const int batch_size = 8;

	{
		auto x = Node<Matrix>::CreateWithSize(784, "input");
		auto label = Node<Matrix>::CreateWithSize(1, "label");

		Layer3D L1(784, 128, Activation::ReLU, InitType::Xavier);
		Layer3D L2(128, 10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		//auto loss = (  (pred - label)->ElementwiseMul(pred - label)  )->Sum();
		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = (int)output->ArgMax();
			int target_class = (int)labels[i]->GetValue()[0];

			if (pred_class == target_class) correct++;
		}

		for (size_t i = 0; i < val_xs.size(); i++)
		{
			*input = val_xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = (int)output->ArgMax();
			int target_class = (int)val_labels[i]->GetValue()[0];

			if (pred_class == target_class) val_correct++;
		}

		return std::tuple{ (float)correct * 100.0f / xs.size(), (float)val_correct * 100.0f / val_xs.size() };
		};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 10; epoch++)
	{
		ShuffleDataset(xs, labels);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			//if (i / batch_size % 500 == 0)
				//std::cout << "Batch " << i / batch_size << " of " << xs.size() / batch_size << std::endl;

			size_t end = std::min(i + batch_size, xs.size());

			size_t actual_batch_size = end - i;

			float batch_loss = 0.0f;

			tape.ZeroGradients();

			for (size_t j = i; j < end; j++)
			{
				*input = xs[j]->GetValue();
				*label = labels[j]->GetValue();

				tape.Forward();
				tape.Backward();

				epoch_loss += loss->GetValue()[0];
				batch_loss += loss->GetValue()[0];
			}

			sgd.Step(1.0f / actual_batch_size);
		}

		epoch_loss /= xs.size();

		//if (epoch % 2 == 0)
		std::cout << "Epoch " << epoch + 1 << " Loss: " << epoch_loss << std::endl;
		auto [train_acc, val_acc] = calculate_acc();
		std::cout << "Train Accuracy: " << train_acc << "%" << std::endl;
		std::cout << "Valid Accuracy: " << val_acc << "%" << std::endl;

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	/*auto [train_acc, val_acc] = calculate_acc();
	std::cout << "Final Train Accuracy: " << train_acc << "%" << std::endl;
	std::cout << "Final Valid Accuracy: " << val_acc   << "%" << std::endl;*/
}



void Cifar_Test()
{
	Cifar mnist = Cifar();

	mnist.PrintTrainImage(0);
	mnist.PrintTrainImage(1);
	mnist.PrintTrainImage(2);

	std::vector<NodePtr<Matrix>> xs;
	std::vector<NodePtr<Matrix>> labels;

	std::vector<NodePtr<Matrix>> val_xs;
	std::vector<NodePtr<Matrix>> val_labels;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs.push_back(Node<Matrix>::Create(mnist.GetTrainingImageData(i)));

		int label = mnist.GetTrainingLabelData(i);

		labels.push_back(Node<Matrix>::Create({ (float)label }));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		val_xs.push_back(Node<Matrix>::Create(mnist.GetValidationImageData(i)));

		int label = mnist.GetValidationLabelData(i);

		val_labels.push_back(Node<Matrix>::Create({ (float)label }));
	}


	auto tape = TapeRecorder<Matrix>();
	SGD<Matrix> sgd(0.03f, 0.7f);
	const int batch_size = 8;

	{
		auto x = Node<Matrix>::CreateWithSize(32*32 * 3, "input");
		auto label = Node<Matrix>::CreateWithSize(1, "label");

		Layer3D L1(32*32*3, 128, Activation::ReLU, InitType::Xavier);
		Layer3D L2(128,  10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto input  = tape.SetValue("input");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = (int)output->ArgMax();
			int target_class = (int)labels[i]->GetValue()[0];

			if (pred_class == target_class) correct++;
		}

		for (size_t i = 0; i < val_xs.size(); i++)
		{
			*input = val_xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = (int)output->ArgMax();
			int target_class = (int)val_labels[i]->GetValue()[0];

			if (pred_class == target_class) val_correct++;
		}

		return std::tuple{ (float)correct * 100.0f / xs.size(), (float)val_correct * 100.0f / val_xs.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 20; epoch++)
	{
		ShuffleDataset(xs, labels);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			//if (i / batch_size % 500 == 0)
				//std::cout << "Batch " << i / batch_size << " of " << xs.size() / batch_size << std::endl;

			size_t end = std::min(i + batch_size, xs.size());

			size_t actual_batch_size = end - i;

			float batch_loss = 0.0f;

			tape.ZeroGradients();

			for (size_t j = i; j < end; j++)
			{
				*input = xs[j]->GetValue();
				*label = labels[j]->GetValue();

				tape.Forward();
				tape.Backward();

				epoch_loss += loss->GetValue()[0];
				batch_loss += loss->GetValue()[0];
			}

			sgd.Step(1.0f / actual_batch_size);
		}

		epoch_loss /= xs.size();

		//if (epoch % 2 == 0)
		std::cout << "Epoch " << epoch+1 << " Loss: " << epoch_loss << std::endl;
		auto [train_acc, val_acc] = calculate_acc();
		std::cout << "Train Accuracy: " << train_acc << "%" << std::endl;
		std::cout << "Valid Accuracy: " << val_acc << "%" << std::endl;

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	/*auto [train_acc, val_acc] = calculate_acc();
	std::cout << "Final Train Accuracy: " << train_acc << "%" << std::endl;
	std::cout << "Final Valid Accuracy: " << val_acc   << "%" << std::endl;*/
}



void SpiralClassification_MSE()
{
	std::vector<NodePtr<Vector>> xs;
	std::vector<NodePtr<Vector>> labels;

	const int points_per_class = 50;
	const float pi = 3.14159265f;

	for (int class_id = 0; class_id < 2; class_id++)
	{
		for (int i = 0; i < points_per_class; i++)
		{
			float r = (float)i / points_per_class;
			float t = 4.0f * pi * r + class_id * pi;

			float x = r * std::sin(t);
			float y = r * std::cos(t);

			xs.push_back(Node<Vector>::Create({ x, y }));

			labels.push_back(Node<Vector>::Create(
				class_id == 0 ? std::initializer_list<float>{ -1.0f }
			: std::initializer_list<float>{ 1.0f }
				));
		}
	}

	Layer2D<Vector> L1(2, 32, Activation::Tanh);
	Layer2D<Vector> L2(32, 32, Activation::Tanh);
	Layer2D<Vector> L3(32, 1, Activation::Tanh);

	SGD<Vector> sgd(0.01f);
	const int batch_size = 8;

	// IMPORTANT: collect params ONCE
	{
		auto x = xs[0];
		auto h1 = L1.Forward(x);
		auto h2 = L2.Forward(h1);
		auto out = L3.Forward(h2);

		auto loss = (out - labels[0]) * (out - labels[0]);
		sgd.SetTrainableParams(loss->CollectParams());
	}

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 2000; epoch++)
	{
		ShuffleDataset(xs, labels);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs.size());

			NodePtr<Vector> batch_loss = nullptr;

			for (size_t j = i; j < end; j++)
			{
				auto out = L3.Forward(L2.Forward(L1.Forward(xs[j])));
				auto loss = (out - labels[j]) * (out - labels[j]);

				epoch_loss += loss->GetValue()[0];

				batch_loss = batch_loss ? (batch_loss + loss) : loss;
			}

			batch_loss = batch_loss->Sum();

			batch_loss->Backwards();
			sgd.Step();
		}

		epoch_loss /= xs.size();

		if (epoch % 20 == 0)
			std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
	}

	int correct = 0;

	for (size_t i = 0; i < xs.size(); i++)
	{
		auto out = L3.Forward(L2.Forward(L1.Forward(xs[i])));

		float pred = out->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		int pc = pred > 0.0f ? 1 : 0;
		int tc = target > 0.0f ? 1 : 0;

		if (pc == tc) correct++;
	}

	std::cout << "Final Accuracy: " << (float)correct / xs.size() << std::endl;
}



void XOR()
{
		std::vector<NodePtr<Vector>> xs = {
			Node<Vector>::Create({ 0.0f, 0.0f }),
			Node<Vector>::Create({ 0.0f, 1.0f }),
			Node<Vector>::Create({ 1.0f, 0.0f }),
			Node<Vector>::Create({ 1.0f, 1.0f })
		};

		std::vector<NodePtr<Vector>> label = {
			Node<Vector>::Create({ 0.0f }),
			Node<Vector>::Create({ 1.0f }),
			Node<Vector>::Create({ 1.0f }),
			Node<Vector>::Create({ 0.0f })
		};

		Layer2D<Vector> L1(2, 3, Activation::Tanh);
		Layer2D<Vector> L2(3, 1, Activation::Tanh);

		float lr = 0.1f;

		for (size_t epoch = 0; epoch < 200; epoch++)
		{
			float epoch_loss = 0.0f;

			for (size_t i = 0; i < xs.size(); i++)
			{
				auto x = xs[i];

				auto l1_out = L1.Forward(x);
				auto out = L2.Forward(l1_out);

				auto diff = out - label[i];
				auto loss = diff * diff;

				auto params = loss->CollectParams();

				loss->Backwards();

				//printf("L1 W grad: ");
				//L1.GetWeight(0)->Print();

				//printf("L2 W grad: ");
				//L2.GetWeight(0)->Print();

				loss->SetLabel("loss");
				//loss->Print();

				//printf("W0 before: %f\n", L1.GetWeight(0)->GetValue()[0]);

				for (size_t i = 0; i < L1.GetOutputLength(); i++)
				{
					L1.GetWeight(i)->GetValue() -= lr * L1.GetWeight(i)->GetGradient();
					L1.GetBias(i)->GetValue() -= lr * L1.GetBias(i)->GetGradient();
				}
				for (size_t i = 0; i < L2.GetOutputLength(); i++)
				{
					L2.GetWeight(i)->GetValue() -= lr * L2.GetWeight(i)->GetGradient();
					L2.GetBias(i)->GetValue() -= lr * L2.GetBias(i)->GetGradient();
				}

				//printf("W0 after: %f\n", L1.GetWeight(0)->GetValue()[0]);

				// Loss sammeln (nur Value!)
				epoch_loss += loss->GetValue()[0];
			}

			printf("Epoch %zu - Loss: %f\n", epoch, epoch_loss / xs.size());
		}

		for (size_t i = 0; i < xs.size(); i++)
		{
			auto x = xs[i];

			auto l1_out = L1.Forward(x);
			auto out = L2.Forward(l1_out);

			x->SetLabel("x");
			out->SetLabel("out");

			x->Print();
			out->Print();
		}
}



void Sine()
{
	{
		// training data: y = sin(x)
		std::vector<NodePtr<Vector>> xs;
		std::vector<NodePtr<Vector>> labels;

		const int N = 20;

		for (int i = 0; i < N; i++)
		{
			float x = -3.1415f + i * (6.2830f / (N - 1)); // [-pi, pi]
			float y = std::sin(x);

			xs.push_back(Node<Vector>::Create({ x }));
			labels.push_back(Node<Vector>::Create({ y }));
		}

		// simple MLP: 1 → 8 → 8 → 1
		Layer2D<Vector> L1(1, 8, Activation::Tanh);
		Layer2D<Vector> L2(8, 8, Activation::Tanh);
		Layer2D<Vector> L3(8, 1, Activation::Tanh);

		SGD<Vector> sgd(0.02f);

		float epoch_loss = 0.0f;

		for (size_t epoch = 0; epoch < 800; epoch++)
		{
			epoch_loss = 0.0f;

			for (size_t i = 0; i < xs.size(); i++)
			{
				auto x = xs[i];

				auto h1 = L1.Forward(x);
				auto h2 = L2.Forward(h1);
				auto pred = L3.Forward(h2);

				auto diff = pred - labels[i];
				auto loss = diff * diff;

				sgd.SetTrainableParams(loss->CollectParams());

				loss->Backwards();
				sgd.Step();

				epoch_loss += loss->GetValue()[0];
			}

			epoch_loss /= xs.size();

			// optional debug
			if (epoch % 20 == 0)
				std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
		}

		// final sanity check: model learned function shape
		for (size_t i = 0; i < xs.size(); i++)
		{
			auto x = xs[i];

			auto h1 = L1.Forward(x);
			auto h2 = L2.Forward(h1);
			auto out = L3.Forward(h2);

			float pred = out->GetValue()[0];
			float target = labels[i]->GetValue()[0];
		}
	}
}



int main()
{
	auto a = Node<Scalar>::Create(2.0f);
	auto b = Node<Scalar>::Create(-3.0f);
	auto c = Node<Scalar>::Create(10.0f);

	auto d = a * b + c;

	d->Backwards();

	// d = a * b + c
	// dd/da = b = -3
	// dd/db = a = 2
	// dd/dc = 1

	printf("d = %.4f\n", d->GetValue().GetValue()[0]);
	printf("dd/dd = %.4f\n", d->GetGradient().GetValue()[0]);
	printf("dd/da = %.4f\n", a->GetGradient().GetValue()[0]);
	printf("dd/db = %.4f\n", b->GetGradient().GetValue()[0]);
	printf("dd/dc = %.4f\n", c->GetGradient().GetValue()[0]);


	
	{
		auto x = Node<Vector>::Create({ 0.5f, 0.0f });
		auto w = Node<Vector>::Create({ 1.0f, 0.5f });
		auto b2 = Node<Vector>::Create({ 1.0f });

		auto out = (w * x)->ElementwiseAdd(b2);


		printf("\n\nout = %.4f %.4f\n", out->GetValue().m_pValues[0], out->GetValue().m_pValues[1]);


		out->Backwards();

		printf("dout/dout = %.4f, %.4f\n", out->GetGradient().m_pValues[0], out->GetGradient().m_pValues[1]);

		printf("dout/dx0 = %.4f\n", x->GetGradient().m_pValues[0]);
		printf("dout/dx1 = %.4f\n", x->GetGradient().m_pValues[1]);
	}


	//Neuron
	{
		auto x = Node<Vector>::Create({ 0.0f, 0.0f });

		Neuron2D n(2, Activation::Tanh);
		auto out = n.Forward(x);

		out->Backwards();

		printf("\n\nout = %.4f %.4f\n", out->GetValue().m_pValues[0], out->GetValue().m_pValues[1]);
		printf("dout/dout = %.4f, %.4f\n", out->GetGradient().m_pValues[0], out->GetGradient().m_pValues[1]);
	}

	//Kapathy
	{
		auto x1 = Node<Scalar>::Create(2.0f, "x1");
		auto x2 = Node<Scalar>::Create(0.0f, "x2");

		auto w1 = Node<Scalar>::Create(-3.0f, "w1");
		auto w2 = Node<Scalar>::Create(1.0f, "w2");

		auto b = Node<Scalar>::Create(6.881375f, "b");

		auto x1w1 = x1 * w1;
		auto x2w2 = x2 * w2;
		x1w1->SetLabel("x1w1");
		x2w2->SetLabel("x2w2");

		auto x1w1x2w2 = x1w1 + x2w2;
		x1w1x2w2->SetLabel("x1w1x2w2");

		auto n = x1w1x2w2 + b;
		auto o = n->Tanh();

		n->SetLabel("n");
		o->SetLabel("o");

		o->Backwards();

		printf("\n\n");

		x1->Print();
		x2->Print();
		w1->Print();
		w2->Print();

		x1w1->Print();
		x2w2->Print();

		x1w1x2w2->Print();

		b->Print();
		n->Print();
		o->Print();
	}

	//XOR
	//XOR();

	//Sine();

	//SpiralClassification_MSE();

	//MNist_Test();

	//Cifar_Test();


	//MNist_Tensor();
	MNist_Tensor4D();


	return 0;
}