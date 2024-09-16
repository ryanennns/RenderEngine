#include <gtest/gtest.h>
#include "../include/Structs.h"
#include "../include/helpers.h"

TEST(LineTest, it_evaluates_lines_in_positive_direction)
{
    const Line line = Line{{0, 0, 0}, {1, 1, 1}};
    const int evaluatingBy = 5;

    auto evaluated = lineEvaluate(line, evaluatingBy);

    EXPECT_EQ(5, evaluated.x);
    EXPECT_EQ(5, evaluated.y);
    EXPECT_EQ(5, evaluated.z);
}

TEST(LineTest, it_evaluates_lines_in_negative_direction)
{
    const Line line = Line{{0, 0, 0}, {1, 1, 1}};
    const int evaluatingBy = -5;

    auto evaluated = lineEvaluate(line, evaluatingBy);

    EXPECT_EQ(-5, evaluated.x);
    EXPECT_EQ(-5, evaluated.y);
    EXPECT_EQ(-5, evaluated.z);
}