#pragma once
#include <string>
#include <vector>
#include <stdexcept>



class Cifar
{
public:
	Cifar(const std::string& FolderName = "cifar-10")
	{
		//m_TrainNumImages = ReadImageData("datasets/" + FolderName + "/data_batch_%d.bin");
		m_TrainNumImages = ReadImageData("datasets/" + FolderName + "/train.bin");
		//m_ValidNumImages = ReadImageData("datasets/" + FolderName + "/test_batch.bin");
	}
	~Cifar()
	{
		delete[] m_TrainImages;
		delete[] m_TrainLabels;
		delete[] m_ValidationImages;
		delete[] m_ValidationLabels;
	}

	int GetTrainingNumberOfImages() const { return m_TrainNumImages; }
	int GetValidationNumberOfImages() const { return m_ValidNumImages; }

	std::vector<float> GetTrainingImageData(size_t Index) const;
	std::vector<float> GetValidationImageData(size_t Index) const;
	int GetTrainingLabelData(size_t Index) const;
	int GetValidationLabelData(size_t Index) const;

	void SaveDataSplit(const std::string& BaseFilename, int NumFiles);

	void PrintTrainImage(size_t Index);


private:
	int ReadImageData(const std::string& Filename);

	unsigned char* m_TrainImages;
	unsigned char* m_TrainLabels;
	unsigned char* m_ValidationImages;
	unsigned char* m_ValidationLabels;

	uint32_t m_TrainNumImages = 0;
	uint32_t m_ValidNumImages = 0;
};