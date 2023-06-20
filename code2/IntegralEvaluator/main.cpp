#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

/*
 * Inlining functions so the call overhead
 * is avoided
 */
inline long double f(long double x)
{
    return std::sqrt(std::pow(100, 2) - std::pow(x, 2));
}

/*
 * Inlining the functions so the call overhead
 * is avoided. Note that **inline** is not a
 * command; the compiler may **ignore** the inline
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

inline void experiment(
    double lowerLimit, double upperLimit, double interval, int myRank, int numProcs)
{
    long double result = 0;
    int partitions = static_cast<int>(upperLimit / interval);
    int localPartitions = partitions / numProcs;
    int localStart = myRank * localPartitions;
    int localEnd = localStart + localPartitions;

    long double localSum = integrate(f,
                                     lowerLimit + localStart * interval,
                                     lowerLimit + localEnd * interval,
                                     localPartitions);

    // Gather results from all processes
    MPI_Reduce(&localSum, &result, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
        std::cout << std::fixed << std::setprecision(6) << interval << " \t " << partitions
                  << " \t " << result << std::endl;
    }
}

int main(int argc, char **argv)
{
    double lowerLimit = 0, upperLimit = 100;
    int numProcs, myRank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (myRank == 0) {
        std::cout << "Interval \t Partitions \t Result" << std::endl;
    }

    experiment(lowerLimit, upperLimit, 0.000001, myRank, numProcs);
    experiment(lowerLimit, upperLimit, 0.00001, myRank, numProcs);
    experiment(lowerLimit, upperLimit, 0.0001, myRank, numProcs);

    MPI_Finalize();

    return 0;
}
