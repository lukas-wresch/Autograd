#pragma once
#include <string>
#include <vector>
#include <stdexcept>



class MNist
{
public:
	MNist(const std::string& FolderName = "mnist")
	{
		m_TrainNumImages   = ReadImageData("datasets/" + FolderName + "/train-images.idx3-ubyte", &m_TrainImages);
		int TrainNumLabels = ReadLabelData("datasets/" + FolderName + "/train-labels.idx1-ubyte", &m_TrainLabels);

		if (m_TrainNumImages < 0)
			throw std::runtime_error("MNist: Failed to read training images");
		if (TrainNumLabels < 0)
			throw std::runtime_error("MNist: Failed to read training labels");
		if (m_TrainNumImages != TrainNumLabels)
			throw std::runtime_error("MNist: Number of images and labels does not match");

		m_ValidNumImages   = ReadImageData("datasets/" + FolderName + "/t10k-images.idx3-ubyte", &m_ValidationImages);
		int ValidNumLabels = ReadLabelData("datasets/" + FolderName + "/t10k-labels.idx1-ubyte", &m_ValidationLabels);

		if (m_ValidNumImages < 0)
			throw std::runtime_error("MNist: Failed to read training images");
		if (ValidNumLabels < 0)
			throw std::runtime_error("MNist: Failed to read training labels");
		if (m_ValidNumImages != ValidNumLabels)
			throw std::runtime_error("MNist: Number of images and labels does not match");
	}
	~MNist()
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

	void PrintTrainImage(size_t Index);


private:
	int ReadImageData(const std::string& Filename, unsigned char** Data);
	int ReadLabelData(const std::string& Filename, unsigned char** Data);

	unsigned char* m_TrainImages;
	unsigned char* m_TrainLabels;
	unsigned char* m_ValidationImages;
	unsigned char* m_ValidationLabels;

	uint32_t m_Height = 0;
	uint32_t m_Width  = 0;
	uint32_t m_TrainNumImages = 0;
	uint32_t m_ValidNumImages = 0;
};