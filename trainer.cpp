#pragma once
#include <format>
#include <random>
#include <chrono>
#include "trainer.h"
#include "progressmanager.h"



void ShuffleDataset(std::vector<Tensor4D>& xs, std::vector<Tensor4D>& labels)
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



void Trainer::TrainingLoop()
{
	ProgressManager mgr;

	ShuffleDataset(m_Data.train_data, m_Data.train_labels);

	auto input  = m_Tape.SetValue("input");
	auto label  = m_Tape.SetValue("label");
	auto output = m_Tape.SetValue("output");
	auto loss   = m_Tape.SetValue("loss");

	float epoch_loss = 0.0f;
	size_t batch_size = input->GetBatches();
	size_t train_size = m_Data.train_data.size();

	auto task = mgr.createBar("Epoch " + std::to_string(epochs_done + 1), train_size);
	float old_loss = 0.0f;
	float old_acc  = 0.0f;

	for (size_t i = 0; i < train_size; i += batch_size)
	{
		size_t end = std::min(i + batch_size, train_size);

		float batch_loss = 0.0f;

		size_t actual_batch_size = 0;

		for (size_t j = i; j < end; j++)
		{
			input->ViewBatch(actual_batch_size).CopyFrom(m_Data.train_data[i + actual_batch_size]);
			//label->SetColumn(actual_batch_size, 0, 0, m_Data.train_labels[i + actual_batch_size].Data()[0]);
			label->At(actual_batch_size, 0, 0, 0) = m_Data.train_labels[i + actual_batch_size].Data()[0];
			actual_batch_size++;
		}

		if (actual_batch_size != batch_size)
			continue;

		auto start_time = std::chrono::high_resolution_clock::now();
		m_Tape.Forward();

		m_Tape.ZeroGradients();
		m_Tape.Backward();

		auto end_time = std::chrono::high_resolution_clock::now();

		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		size_t correct = 0;
		for (size_t j = 0; j < actual_batch_size; j++)
		{
			//Convert one hot to class index
			int pred_class = (int)output->ArgMax(j, 0);
			int target_class = (int)m_Data.train_labels[i + j].At(0, 0, 0, 0);

			if (pred_class == target_class) correct++;
		}

		epoch_loss += loss->At(0, 0, 0, 0) * actual_batch_size;
		batch_loss += loss->At(0, 0, 0, 0);

		float passes_per_second = (float)batch_size / batch_duration * 1000.0f;
		float eta = ((float)(train_size - i) / passes_per_second) / 60.0f;
		float acc = (float)correct * 100.0f / actual_batch_size;

		if (i > 0)
		{
			acc  = 0.9f * old_acc  + 0.1f * acc;
			batch_loss = 0.9f * old_loss + 0.1f * batch_loss;
		}

		old_acc  = acc;
		old_loss = batch_loss;

		std::string desc = std::format("passes/s: {:.2f} Loss: {:.4f} ETA: {:.1f}m Acc: {:.2f}%", passes_per_second, batch_loss, eta, acc);
		task.update(i);
		task.description(desc);
		//std::cout << desc << std::endl;

		m_sgd.Step();
	}

	epochs_done++;
}