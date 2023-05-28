#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>

#ifdef _OPENMP
#include <omp.h>
#endif

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

/*
 * sum is shared by all threads
 * all the other variables are not
 */
#pragma default(none) shared(soma)
    {
/*
 * Starts a parallel zone, from now on the
 * instructions are replicated in several 
 * threads according to parallelization rules
 * bellow.
 */
#pragma omp parallel
        {
/*
 * For each thread sum is initialized to zero
 * then, after all calculations, the obtained values
 * are summed together.
 * For this directive it have to be possible to known
 * beforehand the number of iterations in execution time
 * i.e. the number of iterations cannot change while the
 * loop are being executed. Also the k variable must NOT
 * be modified within the loop otherwise it can be changed
 * like in k++ statement.
 */
#pragma omp for reduction(+ : sum)
            for (int k = 1; k < partitions - 1; k++) {
                sum += f(lowerLimit + k * h);
            }
        }
/*
 * Creates a barrier to the execution of the program
 * in order to ensure that all threads has been finished
 * until this point. Now it is possible to do 
 * another operations with the outputted sum.
 * This is optional in this **specific case** because
 * there is an implicit barrier in **for** directive.
 * It is important to note that THERE IS NO CODE associated
 * with this directive i.e. it just marks a point where all
 * threads must arrive.
 * Warning! All or none of the threads must encounter the 
 * barrier otherwise an DEADLOCK happens.
 */
#pragma omp barrier

        sum *= 2;
        sum += (f(lowerLimit) + f(upperLimit));
        return (h / 2) * sum;
    }
}

inline void experiment(double lowerLimit, double upperLimit, double interval)
{
    long double result = 0;
    int partitions = upperLimit / interval;

    double start = omp_get_wtime();
    result = integrate(f, lowerLimit, upperLimit, partitions);
    double end = omp_get_wtime();

    // Calculating total time taken by the program.
    double timeTaken = end - start;

    std::cout << std::fixed << std::setprecision(6) << interval << " \t " << partitions << " \t "
              << timeTaken << " \t " << result << std::endl;
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
