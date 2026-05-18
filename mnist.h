#pragma once
#include <string>
#include <vector>
#include <stdexcept>



class MNist
{
public:
	MNist()
	{
		m_NumImages = ReadImageData("datasets/mnist/train-images.idx3-ubyte",   &m_TrainImages);
		int NumLabels = ReadLabelData("datasets/mnist/train-labels.idx1-ubyte", &m_TrainLabels);

		if (m_NumImages < 0)
			throw std::runtime_error("MNist: Failed to read training images");
		if (NumLabels < 0)
			throw std::runtime_error("MNist: Failed to read training labels");
		if (m_NumImages != NumLabels)
			throw std::runtime_error("MNist: Number of images and labels does not match");
	}
	~MNist()
	{
		delete[] m_TrainImages;
		delete[] m_TrainLabels;
	}

	int GetTrainingNumberOfImages() const { return m_NumImages; }

	std::vector<float> GetTrainingImageData(size_t Index) const;
	int GetTrainingLabelData(size_t Index) const;

	void PrintTrainImage(size_t Index);


private:
	int ReadImageData(const std::string& Filename, unsigned char** Data);
	int ReadLabelData(const std::string& Filename, unsigned char** Data);

	unsigned char* m_TrainImages;
	unsigned char* m_TrainLabels;

	uint32_t m_Height = 0;
	uint32_t m_Width  = 0;
	uint32_t m_NumImages = 0;
};