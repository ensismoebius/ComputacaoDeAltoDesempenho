#include <CL/cl.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

/*
 * Inlinning functions so the call overhead
 * is avoided
 */
inline long double f(long double x)
{
    return std::sqrt(std::pow(100, 2) - std::pow(x, 2));
}

inline long double integrate(std::vector<long double> &f_values,
                             double lowerLimit,
                             double upperLimit,
                             int partitions)
{
    double h = (upperLimit - lowerLimit) / partitions;
    long double sum = 0;

    // Create an OpenCL context and queue
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);

    // Load the OpenCL kernel source code
    std::string kernelSource = "__kernel void integrate(__global float* f_values, float "
                               "lowerLimit, float h, int partitions) {\n"
                               "    int i = get_global_id(0);\n"
                               "    float x = lowerLimit + i * h;\n"
                               "    f_values[i] = sqrt(pow(100.0f, 2) - pow(x, 2));\n"
                               "}\n";

    // Create an OpenCL program and build it
    cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length() + 1));
    cl::Program program(context, sources);
    program.build(devices);

    // Create an OpenCL kernel
    cl::Kernel kernel(program, "integrate");

    // Create OpenCL buffers for f_values and sum
    cl::Buffer fBuffer(context, CL_MEM_WRITE_ONLY, sizeof(long double) * partitions);

    // Set kernel arguments
    kernel.setArg(0, fBuffer);
    kernel.setArg(1, static_cast<float>(lowerLimit));
    kernel.setArg(2, static_cast<float>(h));
    kernel.setArg(3, partitions);

    // Enqueue the kernel for execution
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(partitions));

    // Read the results back to the host
    queue.enqueueReadBuffer(fBuffer, CL_TRUE, 0, sizeof(long double) * partitions, f_values.data());

    // Calculate the sum
    for (int k = 0; k < partitions - 1; k++) {
        sum += f_values[k];
    }

    sum *= 2;
    sum += (f(lowerLimit) + f(upperLimit));

    return (h / 2) * sum;
}

inline void experiment(double lowerLimit, double upperLimit, double interval)
{
    int partitions = static_cast<int>(upperLimit / interval) + 1;
    std::vector<long double> f_values(partitions);

    double start = omp_get_wtime();
    long double result = integrate(f_values, lowerLimit, upperLimit, partitions);
    double end = omp_get_wtime();

    // Calculating total time taken by the program.
    double timeTaken = end - start;

    std::cout << std::fixed << std::setprecision(6) << interval << " \t " << partitions - 1
              << " \t " << timeTaken << " \t " << result << std::endl;
}

int main()
{
    double lowerLimit = 0, upperLimit = 100;

    std::cout << "Interval \t Partitions \t Time taken \t Result" << std::endl;
    experiment(lowerLimit, upperLimit, 0.000001);
    experiment(lowerLimit, upperLimit, 0.00001);
    experiment(lowerLimit, upperLimit, 0.0001);

    return 0;
}
