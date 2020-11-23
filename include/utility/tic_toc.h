#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

//用于计算时间；
class TicToc
{
  public:
    TicToc()
    {
        //初始化的时候即对开始时间进行计时；
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
