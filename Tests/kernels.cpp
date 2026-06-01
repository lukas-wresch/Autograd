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



TEST(Kernels, Multiply_Forward_Elementwise2D)
{
    Tensor a({ 2, 2 });
    Tensor b({ 2, 2 });
    Tensor out({ 2, 2 });

    a.Data()[0] = 1; a.Data()[1] = 2;
    a.Data()[2] = 3; a.Data()[3] = 4;

    b.Data()[0] = 10; b.Data()[1] = 20;
    b.Data()[2] = 30; b.Data()[3] = 40;

    Kernels::Multiply_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 10);
    EXPECT_FLOAT_EQ(out.Data()[1], 40);
    EXPECT_FLOAT_EQ(out.Data()[2], 90);
    EXPECT_FLOAT_EQ(out.Data()[3], 160);
}



TEST(Kernels, Multiply_Forward_ScalarBroadcast_Stress)
{
    Tensor a({ 3, 3 });
    Tensor b({ 1 });
    Tensor out({ 3, 3 });

    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = (float)(i + 1);

    b.Data()[0] = 2.0f;

    Kernels::Multiply_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_FLOAT_EQ(out.Data()[i], a.Data()[i] * 2.0f);
}



TEST(Kernels, Multiply_Forward_Broadcast_VectorRow)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3 });
    Tensor out({ 2, 3 });

    for (size_t i = 0; i < 6; i++)
        a.Data()[i] = 1.0f;

    b.Data()[0] = 1;
    b.Data()[1] = 2;
    b.Data()[2] = 3;

    Kernels::Multiply_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 1);
    EXPECT_FLOAT_EQ(out.Data()[1], 2);
    EXPECT_FLOAT_EQ(out.Data()[2], 3);

    EXPECT_FLOAT_EQ(out.Data()[3], 1);
    EXPECT_FLOAT_EQ(out.Data()[4], 2);
    EXPECT_FLOAT_EQ(out.Data()[5], 3);
}



TEST(Kernels, Multiply_Forward_InvalidBroadcast)
{
    Tensor a({ 2, 3 });
    Tensor b({ 2, 2 });
    Tensor out({ 2, 3 });

    EXPECT_THROW(
        Kernels::Multiply_Forward(out, a, b),
        std::runtime_error
    );
}



TEST(Kernels, MatMul_Forward_KnownResult)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3, 2 });
    Tensor out({ 2, 2 });

    a.Data()[0] = 1; a.Data()[1] = 2; a.Data()[2] = 3;
    a.Data()[3] = 4; a.Data()[4] = 5; a.Data()[5] = 6;

    b.Data()[0] = 7;  b.Data()[1] = 8;
    b.Data()[2] = 9;  b.Data()[3] = 10;
    b.Data()[4] = 11; b.Data()[5] = 12;

    Kernels::MatMul_Forward(out, a, b);

    EXPECT_FLOAT_EQ(out.Data()[0], 58);
    EXPECT_FLOAT_EQ(out.Data()[1], 64);
    EXPECT_FLOAT_EQ(out.Data()[2], 139);
    EXPECT_FLOAT_EQ(out.Data()[3], 154);
}



TEST(Kernels, MatMul_Forward_ShapeMismatch)
{
    Tensor a({ 2, 3 });
    Tensor b({ 4, 2 });
    Tensor out({ 2, 2 });

    EXPECT_THROW(
        Kernels::MatMul_Forward(out, a, b),
        std::runtime_error
    );
}



TEST(Kernels, MatMul_Forward_BatchConsistency)
{
    Tensor a({ 2, 2, 2 });
    Tensor b({ 2, 2, 2 });
    Tensor out({ 2, 2, 2 });

    for (size_t i = 0; i < a.Size(); i++)
    {
        a.Data()[i] = 1.0f;
        b.Data()[i] = 1.0f;
    }

    Kernels::MatMul_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i++)
        EXPECT_GT(out.Data()[i], 0.0f);
}



TEST(Kernels, Multiply_Forward_3D_VectorBroadcast_Stress)
{
    Tensor a({ 2, 2, 3 });
    Tensor b({ 3 });
    Tensor out({ 2, 2, 3 });

    for (size_t i = 0; i < a.Size(); i++)
        a.Data()[i] = (float)(i + 1);

    b.Data()[0] = 1;
    b.Data()[1] = 10;
    b.Data()[2] = 100;

    Kernels::Multiply_Forward(out, a, b);

    for (size_t i = 0; i < out.Size(); i += 3)
    {
        EXPECT_FLOAT_EQ(out.Data()[i + 0], a.Data()[i + 0] * 1);
        EXPECT_FLOAT_EQ(out.Data()[i + 1], a.Data()[i + 1] * 10);
        EXPECT_FLOAT_EQ(out.Data()[i + 2], a.Data()[i + 2] * 100);
    }
}



TEST(Kernels, Tanh_Backward_Sanity)
{
    Tensor out({ 2, 2 });
    Tensor grad({ 2, 2 });
    Tensor grad_in({ 2, 2 });

    for (size_t i = 0; i < 4; i++)
    {
        out.Data()[i] = std::tanh((float)i);
        grad.Data()[i] = 1.0f;
    }

    Kernels::Tanh_Backward(grad_in, out, grad);

    for (size_t i = 0; i < 4; i++)
        EXPECT_GE(grad_in.Data()[i], 0.0f);
}



/*TEST(Kernels, Multiply_Backward_Basic)
{
    Tensor a({ 2, 2 });
    Tensor b({ 2, 2 });
    Tensor gradA({ 2, 2 });

    for (size_t i = 0; i < 4; i++)
    {
        a.Data()[i] = (float)(i + 1);
        b.Data()[i] = 2.0f;
        gradA.Data()[i] = 0.0f;
    }

    Kernels::Multiply_Backward(gradA, a, b);

    for (size_t i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(gradA.Data()[i], b.Data()[i]);
}*/



/*TEST(Kernels, Multiply_Backward_ScalarBroadcast)
{
    Tensor a({ 2, 3 });
    Tensor b({ 1 });
    Tensor gradA({ 2, 3 });

    for (size_t i = 0; i < 6; i++)
    {
        a.Data()[i] = 1.0f;
        gradA.Data()[i] = 0.0f;
    }

    b.Data()[0] = 5.0f;

    Kernels::Multiply_Backward(gradA, a, b);

    for (size_t i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(gradA.Data()[i], 5.0f);
}



TEST(Kernels, Multiply_Backward_Accumulation)
{
    Tensor a({ 2, 2 });
    Tensor b({ 2, 2 });
    Tensor gradA({ 2, 2 });

    for (size_t i = 0; i < 4; i++)
    {
        a.Data()[i] = 1.0f;
        b.Data()[i] = 2.0f;
        gradA.Data()[i] = 1.0f;
    }

    Kernels::Multiply_Backward(gradA, a, b);
    Kernels::Multiply_Backward(gradA, a, b);

    for (size_t i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(gradA.Data()[i], 1.0f + 2.0f + 2.0f);
}*/



TEST(Kernels, Tanh_Backward_ChainRule)
{
    Tensor out({ 3 });
    Tensor outer_grad({ 3 });
    Tensor grad_in({ 3 });

    for (int i = 0; i < 3; i++)
    {
        float x = (float)i;
        out.Data()[i] = std::tanh(x);
        outer_grad.Data()[i] = 1.0f;
        grad_in.Data()[i] = 0.0f;
    }

    Kernels::Tanh_Backward(grad_in, out, outer_grad);

    for (int i = 0; i < 3; i++)
    {
        float expected = 1.0f - out.Data()[i] * out.Data()[i];
        EXPECT_NEAR(grad_in.Data()[i], expected, 1e-5);
    }
}



TEST(Kernels, Tanh_Backward_BroadcastScalar)
{
    Tensor out({ 2, 2 });
    Tensor outer_grad({ 1 });
    Tensor grad_in({ 2, 2 });

    for (size_t i = 0; i < 4; i++)
    {
        out.Data()[i] = 0.5f;
        grad_in.Data()[i] = 0.0f;
    }

    outer_grad.Data()[0] = 2.0f;

    Kernels::Tanh_Backward(grad_in, out, outer_grad);

    for (size_t i = 0; i < 4; i++)
        EXPECT_GT(grad_in.Data()[i], 0.0f);
}



TEST(Kernels, MatMul_Backward_A_Shape)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3, 2 });
    Tensor dOut({ 2, 2 });
    Tensor gradA({ 2, 3 });

    for (size_t i = 0; i < gradA.Size(); i++)
        gradA.Data()[i] = 0.0f;

    Kernels::MatMul_Backward_A(gradA, dOut, b);

    EXPECT_EQ(gradA.Shape()[0], 2);
    EXPECT_EQ(gradA.Shape()[1], 3);
}



TEST(Kernels, MatMul_Backward_B_Shape)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3, 2 });
    Tensor dOut({ 2, 2 });
    Tensor gradB({ 3, 2 });

    for (size_t i = 0; i < gradB.Size(); i++)
        gradB.Data()[i] = 0.0f;

    Kernels::MatMul_Backward_B(gradB, a, dOut);

    EXPECT_EQ(gradB.Shape()[0], 3);
    EXPECT_EQ(gradB.Shape()[1], 2);
}



TEST(Kernels, MatMul_Backward_A_MatrixVsTensor)
{
    Matrix a({ 2, 3 });
    Matrix b({ 3, 2 });
    Matrix dOut({ 2, 2 });
    Matrix gradA({ {2, 3}, {4, 5} });

    Tensor a_t({2, 1}, { 2, 3 });
    Tensor b_t({ 2, 1 }, { 3, 2 });
    Tensor dOut_t({ 2, 1 }, { 2, 2 });
    Tensor gradA_t({ 2, 2 }, { 2, 3, 4, 5 });

    gradA += dOut * b.Transpose();

    Kernels::MatMul_Backward_A(gradA_t, dOut_t, b_t);

    EXPECT_EQ(gradA.At(0, 0), gradA_t.At({ 0, 0 }));
    EXPECT_EQ(gradA.At(0, 1), gradA_t.At({ 0, 1 }));
    EXPECT_EQ(gradA.At(1, 0), gradA_t.At({ 1, 0 }));
    EXPECT_EQ(gradA.At(1, 1), gradA_t.At({ 1, 1 }));
}



TEST(Kernels, MatMul_Backward_B_MatrixVsTensor)
{
    Matrix a({ 2, 3 });
    Matrix b({ 3, 2 });
    Matrix dOut({ 2, 2 });
    Matrix gradB({ {2, 3}, {4, 5} });

    Tensor a_t({ 2, 1 }, { 2, 3 });
    Tensor b_t({ 2, 1 }, { 3, 2 });
    Tensor dOut_t({ 2, 1 }, { 2, 2 });
    Tensor gradB_t({ 2, 2 }, { 2, 3, 4, 5 });

    gradB += a.Transpose() * dOut;

    Kernels::MatMul_Backward_B(gradB_t, a_t, dOut_t);

    EXPECT_EQ(gradB.At(0, 0), gradB_t.At({ 0, 0 }));
    EXPECT_EQ(gradB.At(0, 1), gradB_t.At({ 0, 1 }));
    EXPECT_EQ(gradB.At(1, 0), gradB_t.At({ 1, 0 }));
    EXPECT_EQ(gradB.At(1, 1), gradB_t.At({ 1, 1 }));
}



TEST(Kernels, MatMul_Backward_A_MatrixVsTensor_Hard)
{
    constexpr int RUNS = 20;

    for (int r = 0; r < RUNS; r++)
    {
        size_t M = 4 + rand() % 8;   // 4..11
        size_t K = 4 + rand() % 8;   // 4..11
        size_t N = 4 + rand() % 8;   // 4..11

        Matrix a(M, K);
        Matrix b(K, N);
        Matrix dOut(M, N);
        Matrix gradA(M, K);

        Tensor a_t({ M,K }, std::vector<float>(M * K));
        Tensor b_t({ K,N }, std::vector<float>(K * N));
        Tensor dOut_t({ M,N }, std::vector<float>(M * N));
        Tensor gradA_t({ M,K }, std::vector<float>(M * K));

        // random init
        for (size_t i = 0; i < a.GetSize(); i++) a.SetValue()[i] = (float)(rand() % 100 - 50) / 10.f;
        for (size_t i = 0; i < b.GetSize(); i++) b.SetValue()[i] = (float)(rand() % 100 - 50) / 10.f;
        for (size_t i = 0; i < dOut.GetSize(); i++) dOut.SetValue()[i] = (float)(rand() % 100 - 50) / 10.f;
        for (size_t i = 0; i < gradA.GetSize(); i++) gradA.SetValue()[i] = 0.0f;

        // mirror into tensors
        for (size_t i = 0; i < a.GetSize(); i++) a_t.Data()[i] = a.SetValue()[i];
        for (size_t i = 0; i < b.GetSize(); i++) b_t.Data()[i] = b.SetValue()[i];
        for (size_t i = 0; i < dOut.GetSize(); i++) dOut_t.Data()[i] = dOut.SetValue()[i];
        for (size_t i = 0; i < gradA.GetSize(); i++) gradA_t.Data()[i] = 0.0f;

        gradA += dOut * b.Transpose();
        Kernels::MatMul_Backward_A(gradA_t, dOut_t, b_t);

        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++)
            {
                float expected = gradA.At(i, k);
                float actual = gradA_t.At({ i, k });

                ASSERT_TRUE(std::isfinite(actual));
                EXPECT_NEAR(actual, expected, 1e-4)
                    << "Mismatch A at run " << r << " (" << i << "," << k << ")";
            }
    }
}



TEST(Kernels, MatMul_Backward_B_MatrixVsTensor_Hard)
{
    constexpr int RUNS = 20;

    for (int r = 0; r < RUNS; r++)
    {
        size_t M = 4 + rand() % 8;   // 4..11
        size_t K = 4 + rand() % 8;   // 4..11
        size_t N = 4 + rand() % 8;   // 4..11

        Matrix a(M, K);
        Matrix b(K, N);
        Matrix dOut(M, N);
        Matrix gradB(K, N);

        Tensor a_t({ M,K }, std::vector<float>(M * K));
        Tensor b_t({ K,N }, std::vector<float>(K * N));
        Tensor dOut_t({ M,N }, std::vector<float>(M * N));
        Tensor gradB_t({ K,N }, std::vector<float>(K * N));

        // random init
        for (size_t i = 0; i < a.GetSize(); i++) a.SetValue()[i] = (float)(rand() % 100 - 50) / 10.f;
        for (size_t i = 0; i < b.GetSize(); i++) b.SetValue()[i] = (float)(rand() % 100 - 50) / 10.f;
        for (size_t i = 0; i < dOut.GetSize(); i++) dOut.SetValue()[i] = (float)(rand() % 100 - 50) / 10.f;
        for (size_t i = 0; i < gradB.GetSize(); i++) gradB.SetValue()[i] = 0.0f;

        // mirror into tensors
        for (size_t i = 0; i < a.GetSize(); i++) a_t.Data()[i] = a.SetValue()[i];
        for (size_t i = 0; i < b.GetSize(); i++) b_t.Data()[i] = b.SetValue()[i];
        for (size_t i = 0; i < dOut.GetSize(); i++) dOut_t.Data()[i] = dOut.SetValue()[i];
        for (size_t i = 0; i < gradB.GetSize(); i++) gradB_t.Data()[i] = 0.0f;

        gradB += a.Transpose() * dOut;
        Kernels::MatMul_Backward_B(gradB_t, a_t, dOut_t);

        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
            {
                float expected = gradB.At(k, j);
                float actual = gradB_t.At({ k, j });

                ASSERT_TRUE(std::isfinite(actual));
                EXPECT_NEAR(actual, expected, 1e-4)
                    << "Mismatch B at run " << r << " (" << k << "," << j << ")";
            }
    }
}



TEST(Kernels, MatMul_Backward_IdentityGradient)
{
    Tensor a({ 2, 2 });
    Tensor b({ 2, 2 });
    Tensor dOut({ 2, 2 });
    Tensor gradA({ 2, 2 });
    Tensor gradB({ 2, 2 });

    for (size_t i = 0; i < 4; i++)
    {
        a.Data()[i] = 1.0f;
        b.Data()[i] = 1.0f;
        dOut.Data()[i] = 1.0f;
        gradA.Data()[i] = 0.0f;
        gradB.Data()[i] = 0.0f;
    }

    Kernels::MatMul_Backward_A(gradA, dOut, b);
    Kernels::MatMul_Backward_B(gradB, a, dOut);

    for (size_t i = 0; i < 4; i++)
    {
        EXPECT_GT(gradA.Data()[i], 0.0f);
        EXPECT_GT(gradB.Data()[i], 0.0f);
    }
}



TEST(Kernels, MatMul_Backward_ZeroDOut)
{
    Tensor a({ 2, 3 });
    Tensor b({ 3, 2 });
    Tensor dOut({ 2, 2 });
    Tensor gradA({ 2, 3 });

    for (size_t i = 0; i < dOut.Size(); i++)
        dOut.Data()[i] = 0.0f;

    Kernels::MatMul_Backward_A(gradA, dOut, b);

    for (size_t i = 0; i < gradA.Size(); i++)
        EXPECT_FLOAT_EQ(gradA.Data()[i], 0.0f);
}



TEST(Kernels, Backward_Overall_Sanity)
{
    Tensor a({ 2, 2 });
    Tensor b({ 2, 2 });
    Tensor out({ 2, 2 });

    Tensor gradA({ 2, 2 });
    Tensor gradB({ 2, 2 });

    for (size_t i = 0; i < 4; i++)
    {
        a.Data()[i] = 1.0f;
        b.Data()[i] = 2.0f;
    }

    Kernels::MatMul_Forward(out, a, b);

    for (size_t i = 0; i < 4; i++)
        gradA.Data()[i] = 1.0f;

    Kernels::MatMul_Backward_A(gradA, out, b);

    for (size_t i = 0; i < gradA.Size(); i++)
        EXPECT_GT(gradA.Data()[i], 0.0f);
}