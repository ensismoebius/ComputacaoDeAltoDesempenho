#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

/*
 * Inlinning functions so the call overhead
 * is avoided
 */
inline long double f(long double x)
{
    return std::sqrt(std::pow(100, 2) - std::pow(x, 2));
}

/*
 * Inlinning the functions so the call overhead
 * is avoided. Note that **inline** it is not a
 * command, the compiler may **ignore** the inline
 * request.
 */
inline long double integrate(long double (*f)(long double),
                             double lowerLimit,
                             double upperLimit,
                             int partitions)
{
    double h = (upperLimit - lowerLimit) / partitions;
    long double sum = 0;

    for (int k = 1; k < partitions - 1; k++) {
        sum += f(lowerLimit + k * h);
    }

    sum *= 2;
    sum += (f(lowerLimit) + f(upperLimit));
    return (h / 2) * sum;
}

inline void experiment(double lowerLimit, double upperLimit, double interval)
{
    long double result = 0;
    int partitions = upperLimit / interval;

    double start = MPI_Wtime();
    result = integrate(f, lowerLimit, upperLimit, partitions);
    double end = MPI_Wtime();

    // Calculating total time taken by the program.
    double timeTaken = end - start;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long double *allResults = nullptr;

    if (rank == 0) {
        allResults = new long double[size];
    }

    MPI_Gather(&result, 1, MPI_LONG_DOUBLE, allResults, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6) << interval << " \t " << partitions
                  << " \t " << timeTaken << " \t " << allResults[0] << std::endl;
        delete[] allResults;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    double lowerLimit = 0, upperLimit = 100;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Interval \t Partitions \t Time taken \t Result" << std::endl;
    }

    experiment(lowerLimit, upperLimit, 0.000001);
    experiment(lowerLimit, upperLimit, 0.00001);
    experiment(lowerLimit, upperLimit, 0.0001);

    MPI_Finalize();

    return 0;
}
