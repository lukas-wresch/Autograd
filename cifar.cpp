#include "cifar.h"



int Cifar::ReadImageData(const std::string& Filename)
{
    if (Filename.find("%d") != std::string::npos)
    {
		m_TrainImages = new unsigned char[50000 * 32 * 32 * 3];
        m_TrainLabels = new unsigned char[50000];
        m_TrainNumImages = 0;

        //%d vorhanden
        for (int i = 1; i < 100 && m_TrainNumImages < 50000; i++)
        {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), Filename.c_str(), i);

            FILE* fp = nullptr;
            errno_t err = fopen_s(&fp, buffer, "rb");
            if (err != 0 || !fp)
                break;

            while (true)
            {
                if (fread(&m_TrainLabels[m_TrainNumImages], 1, 1, fp) != 1)
                    break;

                if (fread(&m_TrainImages[m_TrainNumImages * 32 * 32 * 3], 32 * 32 * 3, 1, fp) != 1)
                    break;

                m_TrainNumImages++;
            }

			fclose(fp);
        }

        return m_TrainNumImages;
    }

    else
    {
        m_ValidationImages = new unsigned char[10000 * 32 * 32 * 3];
        m_ValidationLabels = new unsigned char[10000];
        m_ValidNumImages = 0;

        FILE* fp = nullptr;
        errno_t err = fopen_s(&fp, Filename.c_str(), "rb");
        if (err != 0 || !fp)
            return 0;

        while (m_ValidNumImages < 10000)
        {
            if (fread(&m_ValidationLabels[m_ValidNumImages], 1, 1, fp) != 1)
                break;

            if (fread(&m_ValidationImages[m_ValidNumImages * 32 * 32 * 3], 32 * 32 * 3, 1, fp) != 1)
                break;

            m_ValidNumImages++;
        }

        fclose(fp);
        return m_ValidNumImages;
    }
}



std::vector<float> Cifar::GetTrainingImageData(size_t Index) const
{
    if (Index >= m_TrainNumImages)
        throw std::runtime_error("Cifar: Index out of range");

    std::vector<float> image_data(32 * 32 * 3);
    for (uint32_t i = 0; i < 32 * 32 * 3; i++)
        image_data[i] = (float)m_TrainImages[Index * 32 * 32 * 3 + i] / 255.0f;

    return image_data;
}



std::vector<float> Cifar::GetValidationImageData(size_t Index) const
{
    if (Index >= m_ValidNumImages)
        throw std::runtime_error("Cifar: Index out of range");

    std::vector<float> image_data(32 * 32 * 3);
    for (uint32_t i = 0; i < 32 * 32 * 3; i++)
        image_data[i] = (float)m_ValidationImages[Index * 32 * 32 * 3 + i] / 255.0f;

    return image_data;
}



int Cifar::GetTrainingLabelData(size_t Index) const
{
    if (Index >= m_TrainNumImages)
        throw std::runtime_error("MNist: Index out of range");
    return m_TrainLabels[Index];
}



int Cifar::GetValidationLabelData(size_t Index) const
{
    if (Index >= m_ValidNumImages)
        throw std::runtime_error("MNist: Index out of range");
    return m_ValidationLabels[Index];
}



void Cifar::PrintTrainImage(size_t Index)
{
	if (Index >= m_TrainNumImages)
        throw std::runtime_error("Cifar: Index out of range");

    for (uint32_t y = 0; y < 32; y++)
    {
        for (uint32_t x = 0; x < 32; x++)
        {
            unsigned char p = m_TrainImages[y * 32 + x + Index * 32 * 32 * 3];
            
            if (p > 128)
                printf("#");
            else
                printf(" ");
        }

        printf("\n");
    }

	printf("Label: %u\n", m_TrainLabels[Index]);
}