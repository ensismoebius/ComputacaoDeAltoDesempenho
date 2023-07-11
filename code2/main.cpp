#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

inline long double f(long double x)
{
    return std::sqrt(std::pow(100, 2) - std::pow(x, 2));
}

inline long double integrate(long double (*f)(long double),
                             double lowerLimit,
                             double upperLimit,
                             int partitions)
{
    double h = (upperLimit - lowerLimit) / partitions;
    long double sum = 0;

    int slotId, slotsAvailable;
    MPI_Comm_rank(MPI_COMM_WORLD, &slotId);
    MPI_Comm_size(MPI_COMM_WORLD, &slotsAvailable);

    int localIterations = (partitions - 2) / slotsAvailable;
    int remainingIterations = (partitions - 2) % slotsAvailable;

    int start = slotId * localIterations + std::min(slotId, remainingIterations) + 1;
    int end = start + localIterations + (slotId < remainingIterations ? 1 : 0);

    long double localSum = 0;

    for (int k = start; k < end; k++) {
        localSum += f(lowerLimit + k * h);
    }

    MPI_Reduce(&localSum, &sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (slotId == 0) {
        sum *= 2;
        sum += (f(lowerLimit) + f(upperLimit));
        sum *= (h / 2);
    }

    return sum;
}

inline void experiment(double lowerLimit, double upperLimit, double interval)
{
    long double result = 0;
    int partitions = upperLimit / interval;

    double start_time = MPI_Wtime();

    result = integrate(f, lowerLimit, upperLimit, partitions);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6) << interval << " \t " << partitions
                  << " \t " << elapsed_time << " \t " << result << std::endl;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    double lowerLimit = 0, upperLimit = 100;

    int slotId;
    MPI_Comm_rank(MPI_COMM_WORLD, &slotId);

    if (slotId == 0) {
        std::cout << "Interval \t Partitions \t Time taken \t Result" << std::endl;
    }

    experiment(lowerLimit, upperLimit, 0.000001);
    experiment(lowerLimit, upperLimit, 0.00001);
    experiment(lowerLimit, upperLimit, 0.0001);

    MPI_Finalize();
    return 0;
}
