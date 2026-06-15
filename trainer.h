#pragma once
#include <vector>
#include "src/tensor4d.h"
#include "src/tape.h"
#include "src/sgd.h"



class Trainer
{
public:
    struct DataSet
    {
        std::vector<Tensor4D> train_data;
        std::vector<Tensor4D> train_labels;

        std::vector<Tensor4D> valid_data;
        std::vector<Tensor4D> valid_labels;
    };


    Trainer(const DataSet& Data, TapeRecorder<Tensor4D>& Tape, SGD<Tensor4D>& Sgd) : m_Data(Data), m_Tape(Tape), m_sgd(Sgd)
    {}

    void TrainingLoop();

    void Validate();

private:
    DataSet m_Data;
    TapeRecorder<Tensor4D>& m_Tape;
    SGD<Tensor4D>& m_sgd;

    size_t epochs_done = 0;
};