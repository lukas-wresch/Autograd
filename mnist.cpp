#include "mnist.h"



uint32_t read_be_uint32(FILE* fp)
{
    unsigned char bytes[4];

    if (fread(bytes, 1, 4, fp) != 4)
    {
        fprintf(stderr, "Fehler beim Lesen\n");
        exit(1);
    }

    return ((uint32_t)bytes[0] << 24) |
        ((uint32_t)bytes[1] << 16) |
        ((uint32_t)bytes[2] << 8) |
        ((uint32_t)bytes[3]);
}



int MNist::ReadImageData(const std::string& Filename, unsigned char** Data)
{
    FILE* fp = nullptr;
    errno_t err = fopen_s(&fp, Filename.c_str(), "rb");
    if (err != 0 || !fp)
        return -1;

    // Header lesen
    uint32_t magic = read_be_uint32(fp);
    uint32_t num_images = read_be_uint32(fp);
    m_Height = read_be_uint32(fp);
    m_Width  = read_be_uint32(fp);

    printf("Magic Number : %u\n", magic);
    printf("Bilder       : %u\n", num_images);
    printf("Hoehe        : %u\n", m_Height);
    printf("Breite       : %u\n", m_Width);

    // Speicher für alle Bilder
	size_t total_size = (size_t)m_Height * m_Width * num_images;
    *Data = new unsigned char[total_size];

    /* Bilddaten lesen */
    size_t read = fread(*Data, 1, total_size, fp);

    fclose(fp);

    if (read != total_size)
    {
        delete[] *Data;
        *Data = nullptr;
        return -1;
    }

    return num_images;
}



int MNist::ReadLabelData(const std::string& Filename, unsigned char** Data)
{
    FILE* fp = nullptr;
    errno_t err = fopen_s(&fp, Filename.c_str(), "rb");
    if (err != 0 || !fp)
        return -1;

    // Header lesen
    uint32_t magic = read_be_uint32(fp);
    uint32_t num_labels = read_be_uint32(fp);

    printf("Magic Number : %u\n", magic);

    // Speicher für alle Bilder
    *Data = new unsigned char[num_labels];

    /* Bilddaten lesen */
    size_t read = fread(*Data, 1, num_labels, fp);

    fclose(fp);

    if (read != num_labels)
    {
        delete[] * Data;
        *Data = nullptr;
        return -1;
    }

    return num_labels;
}



std::vector<float> MNist::GetTrainingImageData(size_t Index) const
{
    if (Index >= m_TrainNumImages)
        throw std::runtime_error("MNist: Index out of range");

    std::vector<float> image_data(m_Width * m_Height);
    for (uint32_t i = 0; i < m_Width * m_Height; i++)
        image_data[i] = (float)m_TrainImages[Index * m_Height * m_Width + i] / 255.0f;

    return image_data;
}



std::vector<float> MNist::GetValidationImageData(size_t Index) const
{
    if (Index >= m_ValidNumImages)
        throw std::runtime_error("MNist: Index out of range");

    std::vector<float> image_data(m_Width * m_Height);
    for (uint32_t i = 0; i < m_Width * m_Height; i++)
        image_data[i] = (float)m_ValidationImages[Index * m_Height * m_Width + i] / 255.0f;

    return image_data;
}



int MNist::GetTrainingLabelData(size_t Index) const
{
    if (Index >= m_TrainNumImages)
        throw std::runtime_error("MNist: Index out of range");
    return m_TrainLabels[Index];
}



int MNist::GetValidationLabelData(size_t Index) const
{
    if (Index >= m_ValidNumImages)
        throw std::runtime_error("MNist: Index out of range");
    return m_ValidationLabels[Index];
}



void MNist::PrintTrainImage(size_t Index)
{
	if (Index >= m_TrainNumImages)
        throw std::runtime_error("MNist: Index out of range");

    for (uint32_t y = 0; y < m_Height; y++)
    {
        for (uint32_t x = 0; x < m_Width; x++)
        {
            unsigned char p = m_TrainImages[y * m_Width + x + Index * m_Height * m_Width];
            
            if (p > 128)
                printf("#");
            else
                printf(" ");
        }

        printf("\n");
    }

	printf("Label: %u\n", m_TrainLabels[Index]);
}