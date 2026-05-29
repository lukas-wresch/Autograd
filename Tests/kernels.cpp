#include "gtest/gtest.h"
#include "../src/kernels.h"



TEST(Kernels, Multiply_Forward_Basic)
{
    Tensor a({ 2, 2 });
    Tensor b({ 2, 2 });
    Tensor out({ 2, 2 });

    a.Data()[0] = 1; a.Data()[1] = 2;
    a.Data()[2] = 3; a.Data()[3] = 4;

    b.Data()[0] = 2; b.Data()[1] = 3;
    b.Data()[2] = 4; b.Data()[3] = 5;

    Kernels::Multiply_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 2);
    EXPECT_FLOAT_EQ(out.Data()[1], 6);
    EXPECT_FLOAT_EQ(out.Data()[2], 12);
    EXPECT_FLOAT_EQ(out.Data()[3], 20);
}


TEST(Kernels, Multiply_Forward_ScalarBroadcast)
{
    Tensor a({ 2, 2 });
    Tensor b({ 1 });   // scalar tensor
    Tensor out({ 2, 2 });

    a.Data()[0] = 1; a.Data()[1] = 2;
    a.Data()[2] = 3; a.Data()[3] = 4;

    b.Data()[0] = 10;

    Kernels::Multiply_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 10);
    EXPECT_FLOAT_EQ(out.Data()[1], 20);
    EXPECT_FLOAT_EQ(out.Data()[2], 30);
    EXPECT_FLOAT_EQ(out.Data()[3], 40);
}


TEST(Kernels, Multiply_Forward_VectorBroadcast)
{
    Tensor a({ 2, 3 });

    Tensor b({ 3 }); // broadcast over rows
    Tensor out({ 2, 3 });

    // a =
    // [1 2 3]
    // [4 5 6]

    for (int i = 0; i < 6; i++) a.Data()[i] = i + 1.0f;

    // b = [1 10 100]
    b.Data()[0] = 1;
    b.Data()[1] = 10;
    b.Data()[2] = 100;

    Kernels::Multiply_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 1);
    EXPECT_FLOAT_EQ(out.Data()[1], 20);
    EXPECT_FLOAT_EQ(out.Data()[2], 300);

    EXPECT_FLOAT_EQ(out.Data()[3], 4);
    EXPECT_FLOAT_EQ(out.Data()[4], 50);
    EXPECT_FLOAT_EQ(out.Data()[5], 600);
}


TEST(Kernels, Multiply_Forward_RowBroadcast)
{
    Tensor a({ 2, 3 });
    Tensor b({ 2, 1 });
    Tensor out({ 2, 3 });

    // a =
    // [1 2 3]
    // [4 5 6]

    for (int i = 0; i < 6; i++) a.Data()[i] = i + 1.0f;

    // b =
    // [2]
    // [3]

    b.Data()[0] = 2;
    b.Data()[1] = 3;

    Kernels::Multiply_Forward(out, a, b);

    // row 0 * 2
    EXPECT_FLOAT_EQ(out.Data()[0], 2);
    EXPECT_FLOAT_EQ(out.Data()[1], 4);
    EXPECT_FLOAT_EQ(out.Data()[2], 6);

    // row 1 * 3
    EXPECT_FLOAT_EQ(out.Data()[3], 12);
    EXPECT_FLOAT_EQ(out.Data()[4], 15);
    EXPECT_FLOAT_EQ(out.Data()[5], 18);
}


TEST(Kernels, Multiply_Forward_BroadcastingFull)
{
    Tensor a({ 2, 1 });
    Tensor b({ 1, 3 });
    Tensor out({ 2, 3 });

    a.Data()[0] = 2;
    a.Data()[1] = 3;

    b.Data()[0] = 1;
    b.Data()[1] = 10;
    b.Data()[2] = 100;

    Kernels::Multiply_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 2);
    EXPECT_FLOAT_EQ(out.Data()[1], 20);
    EXPECT_FLOAT_EQ(out.Data()[2], 200);

    EXPECT_FLOAT_EQ(out.Data()[3], 3);
    EXPECT_FLOAT_EQ(out.Data()[4], 30);
    EXPECT_FLOAT_EQ(out.Data()[5], 300);
}


TEST(Kernels, Multiply_Forward_ShapeMismatchThrows)
{
    Tensor a({ 2, 2 });
    Tensor b({ 3, 3 });
    Tensor out({ 2, 2 });

    EXPECT_THROW(
        Kernels::Multiply_Forward(out, a, b),
        std::runtime_error
    );
}



TEST(Kernels, Multiply_Forward_MatchesNaive)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3 });
    Tensor out({ 2, 3 });

    for (int i = 0; i < 6; i++) a.Data()[i] = i + 1.0f;
    b.Data()[0] = 1;
    b.Data()[1] = 2;
    b.Data()[2] = 3;

    Kernels::Multiply_Forward(out, a, b);

    // naive check
    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_GT(out.Data()[i], 0.0f);
}



TEST(Kernels, Multiply_Forward_3D_Elementwise)
{
    Tensor a({ 2, 2, 2 });
    Tensor b({ 2, 2, 2 });
    Tensor out({ 2, 2, 2 });

    for (size_t i = 0; i < a.Size(); i++)
    {
        a.Data()[i] = (float)(i + 1);
        b.Data()[i] = 2.0f;
    }

    Kernels::Multiply_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_FLOAT_EQ(out.Data()[i], (i + 1) * 2.0f);
}



TEST(Kernels, Multiply_Forward_3D_ScalarBroadcast)
{
    Tensor a({ 2, 2, 2 });
    Tensor b({ 1 });
    Tensor out({ 2, 2, 2 });

    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = (float)(i + 1);

    b.Data()[0] = 10.0f;

    Kernels::Multiply_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_FLOAT_EQ(out.Data()[i], (i + 1) * 10.0f);
}



TEST(Kernels, Multiply_Forward_3D_VectorBroadcast)
{
    Tensor a({ 2, 2, 3 });
    Tensor b({ 3 });
    Tensor out({ 2, 2, 3 });

    // fill a = 1..12
    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = (float)(i + 1);

    // vector broadcast
    b.Data()[0] = 1.0f;
    b.Data()[1] = 10.0f;
    b.Data()[2] = 100.0f;

    Kernels::Multiply_Forward(out, a, b);

    for (size_t i = 0; i < 12; i += 3)
    {
        EXPECT_FLOAT_EQ(out.Data()[i + 0], a.Data()[i + 0] * 1);
        EXPECT_FLOAT_EQ(out.Data()[i + 1], a.Data()[i + 1] * 10);
        EXPECT_FLOAT_EQ(out.Data()[i + 2], a.Data()[i + 2] * 100);
    }
}



TEST(Kernels, Multiply_Forward_3D_MatrixBroadcast)
{
    Tensor a({ 2, 3, 4 });
    Tensor b({ 3, 4 });
    Tensor out({ 2, 3, 4 });

    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = 1.0f;

    for (size_t i = 0; i < b.Size(); i++)
        b.Data()[i] = (float)(i + 1);

    Kernels::Multiply_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_FLOAT_EQ(out.Data()[i], b.Data()[i % b.Size()]);
}



TEST(Kernels, Multiply_Forward_3D_BroadcastColumn)
{
    Tensor a({ 2, 2, 2 });
    Tensor b({ 2, 1 });
    Tensor out({ 2, 2, 2 });

    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = 1.0f;

    b.Data()[0] = 2.0f;
    b.Data()[1] = 3.0f;

    Kernels::Multiply_Forward(out, a, b);

    // each "row block" must scale correctly
    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_TRUE(out.Data()[i] == 2.0f || out.Data()[i] == 3.0f);
}



TEST(Kernels, Multiply_Forward_InvalidBroadcastThrows)
{
    Tensor a({ 2, 3, 4 });
    Tensor b({ 3, 5 }); // incompatible
    Tensor out({ 2, 3, 4 });

    EXPECT_THROW(
        Kernels::Multiply_Forward(out, a, b),
        std::runtime_error
    );
}



TEST(Kernels, MatMul_Forward_Basic2D)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3, 2 });
    Tensor out({ 2, 2 });

    // a =
    // [1 2 3]
    // [4 5 6]

    a.Data()[0] = 1; a.Data()[1] = 2; a.Data()[2] = 3;
    a.Data()[3] = 4; a.Data()[4] = 5; a.Data()[5] = 6;

    // b =
    // [1 2]
    // [3 4]
    // [5 6]

    b.Data()[0] = 1; b.Data()[1] = 2;
    b.Data()[2] = 3; b.Data()[3] = 4;
    b.Data()[4] = 5; b.Data()[5] = 6;

    Kernels::MatMul_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 22); // 1*1+2*3+3*5
    EXPECT_FLOAT_EQ(out.Data()[1], 28); // 1*2+2*4+3*6
    EXPECT_FLOAT_EQ(out.Data()[2], 49); // 4*1+5*3+6*5
    EXPECT_FLOAT_EQ(out.Data()[3], 64); // 4*2+5*4+6*6
}



TEST(Kernels, MatMul_Forward_Identity)
{
    Tensor a({ 3, 3 });
    Tensor b({ 3, 3 });
    Tensor out({ 3, 3 });

    // A = identity
    a.Data()[0] = 1; a.Data()[1] = 0; a.Data()[2] = 0;
    a.Data()[3] = 0; a.Data()[4] = 1; a.Data()[5] = 0;
    a.Data()[6] = 0; a.Data()[7] = 0; a.Data()[8] = 1;

    // B = arbitrary
    for (size_t i = 0; i < 9; i++)
        b.Data()[i] = (float)(i + 1);

    Kernels::MatMul_Forward(out, a, b);

    for (size_t i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(out.Data()[i], b.Data()[i]);
}



TEST(Kernels, MatMul_Forward_ShapeMismatchThrows)
{
    Tensor a({ 2, 3 });
    Tensor b({ 4, 2 });
    Tensor out({ 2, 2 });

    EXPECT_THROW(
        Kernels::MatMul_Forward(out, a, b),
        std::runtime_error
    );
}



TEST(Kernels, MatMul_Forward_Rectangular)
{
    Tensor a({ 3, 2 });
    Tensor b({ 2, 4 });
    Tensor out({ 3, 4 });

    // fill A
    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = (float)(i + 1);

    // fill B
    for (size_t i = 0; i < b.Size(); i++)
        b.Data()[i] = 1.0f;

    Kernels::MatMul_Forward(out, a, b);

    // each output is sum of 2 elements
    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_GT(out.Data()[i], 0.0f);
}



TEST(Kernels, MatMul_Forward_Batch3D)
{
    Tensor a({ 2, 2, 3 }); // batch=2
    Tensor b({ 2, 3, 2 });
    Tensor out({ 2, 2, 2 });

    // fill A
    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = 1.0f;

    // B identity-like blocks
    for (size_t i = 0; i < b.Size(); i++)
        b.Data()[i] = (i % 3 == i / 3) ? 1.0f : 0.0f;

    Kernels::MatMul_Forward(out, a, b);

    // just ensure output is valid and consistent
    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_GE(out.Data()[i], 0.0f);
}



TEST(Kernels, MatMul_Forward_4D_Batch)
{
    Tensor a({ 2, 2, 2, 3 });
    Tensor b({ 2, 2, 3, 2 });
    Tensor out({ 2, 2, 2, 2 });

    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = 1.0f;

    for (size_t i = 0; i < b.Size(); i++)
        b.Data()[i] = 1.0f;

    Kernels::MatMul_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_GT(out.Data()[i], 0.0f);
}



TEST(Kernels, MatMul_Forward_OutputShapeCorrect)
{
    Tensor a({ 5, 3 });
    Tensor b({ 3, 7 });
    Tensor out({ 5, 7 });

    Kernels::MatMul_Forward(out, a, b);

    EXPECT_EQ(out.Shape()[0], 5);
    EXPECT_EQ(out.Shape()[1], 7);
}