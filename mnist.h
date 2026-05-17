#pragma once



class MNist
{
public:
	MNist();
	~MNist();


private:
	unsigned char* m_TrainImages;
	unsigned char* m_TrainLabels;
};