
#define _USE_MATH_DEFINES
#include <cmath>


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>

using namespace std;
namespace
{
    struct float2 { float x{}, y{}; };
    struct float3 { float x{}, y{}, z{}; };

    template <typename T>
    T lerp(T const & lhs, T const & rhs, float s) noexcept
    {
        return lhs + (rhs - lhs) * s;
    }

    float2 lerp(float2 const & lhs, float2 const & rhs, float s) noexcept
    {
        return { lerp(lhs.x, rhs.x, s), lerp(lhs.y, rhs.y, s) };
    }

    double sqr(double x) { return x * x; };


    void CreateDDM(std::vector<float2>& ddm, std::vector<float3> const & normal_map, float min_z)
    {
        ddm.resize(normal_map.size());
        for (size_t i = 0; i < normal_map.size(); ++ i)
        {
            float3 n = normal_map[i];
            n.z = std::max(n.z, min_z);
            ddm[i].x = n.x / n.z;
            ddm[i].y = n.y / n.z;
        }
    }

    void AccumulateDDM(std::vector<float>& height_map, std::vector<float2> const & ddm, uint32_t width, uint32_t height, int directions, int rings)
    {
        float const step = 2 * M_PI / directions;
        std::vector<float2> dxdy(directions);
        for (int i = 0; i < directions; ++ i)
        {
            const auto v = -i * step;
            dxdy[i].y = sin(v);
            dxdy[i].x = cos(v);
        }

        std::vector<float2> tmp_hm[2];
        tmp_hm[0].resize(ddm.size(), float2());
        tmp_hm[1].resize(ddm.size(), float2());
        int active = 0;
        for (int i = 1; i < rings; ++ i)
        {
            for (size_t j = 0; j < ddm.size(); ++ j)
            {
                int y = static_cast<int>(j / width);
                int x = static_cast<int>(j - y * width);

                for (int k = 0; k < directions; ++ k)
                {
                    const auto delta_x = dxdy[k].x * i;
                    const auto delta_y = dxdy[k].y * i;
                    float sample_x = x + delta_x;
                    float sample_y = y + delta_y;
                    int sample_x0 = static_cast<int>(floor(sample_x));
                    int sample_y0 = static_cast<int>(floor(sample_y));
                    int sample_x1 = sample_x0 + 1;
                    int sample_y1 = sample_y0 + 1;
                    float weight_x = sample_x - sample_x0;
                    float weight_y = sample_y - sample_y0;

                    sample_x0 %= width;
                    sample_y0 %= height;
                    sample_x1 %= width;
                    sample_y1 %= height;

                    float2 hl0 = lerp(tmp_hm[active][sample_y0 * width + sample_x0], tmp_hm[active][sample_y0 * width + sample_x1], weight_x);
                    float2 hl1 = lerp(tmp_hm[active][sample_y1 * width + sample_x0], tmp_hm[active][sample_y1 * width + sample_x1], weight_x);
                    float2 h = lerp(hl0, hl1, weight_y);
                    float2 ddl0 = lerp(ddm[sample_y0 * width + sample_x0], ddm[sample_y0 * width + sample_x1], weight_x);
                    float2 ddl1 = lerp(ddm[sample_y1 * width + sample_x0], ddm[sample_y1 * width + sample_x1], weight_x);
                    float2 dd = lerp(ddl0, ddl1, weight_y);

                    auto& v = tmp_hm[active == 0][j];
                    v.x += h.x + dd.x * delta_x;
                    v.y += h.y + dd.y * delta_y;
                }
            }

            active = static_cast<int>(active) == 0;
        }

        float const scale = 0.5F / (directions * rings);

        height_map.resize(ddm.size());
        for (size_t i = 0; i < ddm.size(); ++ i)
        {
            float2 const & h = tmp_hm[active][i];
            height_map[i] = (h.x + h.y) * scale;
        }
    }

    void CreateHeightMap(std::string const & in_file, std::string const & out_file, float min_z)
    {
        auto in_tex = cv::imread(in_file);

        auto const width = in_tex.cols;
        auto const height = in_tex.rows;
        {
            uint32_t the_width = width;
            uint32_t the_height = height;

                std::vector<float> heights;// (in_data.size());
            {
                std::vector<float3> normals(the_width * the_height);
                {
                    for (uint32_t y = 0; y < the_height; ++ y)
                    {
                        for (uint32_t x = 0; x < the_width; ++ x)
                        {
                            auto p = in_tex.at<cv::Vec3b>(y, x);

                            if (p[0] == 255 && p[1] == 255 && p[2] == 255)
                            {
                                normals[y * the_width + x].x = 0;
                                normals[y * the_width + x].y = 0;
                                normals[y * the_width + x].z = 1;
                            }
                            else
                            {
                                auto nx = p[2] - 127.5;
                                auto ny = -(p[1] - 127.5);
                                auto nz = p[0] - 127.5;

                                auto coeff = 1. / sqrt(sqr(nx) + sqr(ny) + sqr(nz));
                                normals[y * the_width + x].x = nx * coeff;
                                normals[y * the_width + x].y = ny * coeff;
                                normals[y * the_width + x].z = nz * coeff;
                            }
                        }
                    }
                }

                std::vector<float2> ddm;
                CreateDDM(ddm, normals, min_z);

                AccumulateDDM(heights/*[i]*/, ddm, the_width, the_height, 4, 9);

                the_width = std::max(the_width / 2, 1U);
                the_height = std::max(the_height / 2, 1U);
            }

            float min_height = +1e10F;
            float max_height = -1e10F;
            {
                for (float height : heights)
                {
                    min_height = std::min(min_height, height);
                    max_height = std::max(max_height, height);
                }
            }
            if (max_height - min_height > 1e-6F)
            {
                {
                    for (float & height : heights)
                    {
                        height = (height - min_height) / (max_height - min_height);
                    }
                }
            }

            cv::Mat data_block(height, width, CV_8UC1);

            {
                for (size_t j = 0; j < heights/*[i]*/.size(); ++ j)
                {
                    data_block.data[j] = static_cast<uint8_t>(std::clamp(static_cast<int>(heights/*[i]*/[j] * 255 + 0.5F), 0, 255));
                }
            }

            the_width = width;
            the_height = height;

            cv::imshow("Result", data_block);
            //*/

            cv::waitKey();

        }
    }
}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cout << "Usage: Normal2Height xxx.dds yyy.dds [min_z]" << endl;
        return 1;
    }

    std::string in_file = 
        argv[1];
    if (in_file.empty())
    {
        cout << "Couldn't locate " << in_file << endl;
        return 1;
    }

    float min_z = 1e-6F;
    if (argc >= 4)
    {
        min_z = static_cast<float>(atof(argv[3]));
    }

    CreateHeightMap(in_file, argv[2], min_z);

    cout << "Height map is saved to " << argv[2] << endl;

    return 0;
}
