#include <cmath>
#include <iostream>
#include <ostream>

#ifdef _OPENMP
	#include <omp.h>
#endif

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

#pragma default(none) shared(soma)
    {
#pragma omp parallel
        {
#pragma omp for reduction(+ : sum)
            for (int k = 1; k < partitions - 1; k++) {
                sum += f(lowerLimit + k * h);
            }
        }
#pragma omp barrier

        sum *= 2;
        sum += (f(lowerLimit) + f(upperLimit));
        return (h / 2) * sum;
    }
}

int main()
{
    double lowerLimit = 0, upperLimit = 100;

    int partitions = 0;

    partitions = upperLimit / 0.000001;
    std::cout << "Interval: " << 0.000001 << " Partitions: " << partitions
              << " - Result: " << integrate(f, lowerLimit, upperLimit, partitions) << std::endl;

    partitions = upperLimit / 0.00001;
    std::cout << "Interval: " << 0.00001 << " Partitions: " << partitions
              << " - Result: " << integrate(f, lowerLimit, upperLimit, partitions) << std::endl;

    partitions = upperLimit / 0.0001;
    std::cout << "Interval: " << 0.0001 << " Partitions: " << partitions
              << " - Result: " << integrate(f, lowerLimit, upperLimit, partitions) << std::endl;

    return 0;
}
