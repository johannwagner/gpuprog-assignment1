/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2018/19      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : Assignment 1
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description:
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>
#include <thread>
#include <cassert>
#include <atomic>
#include <mutex>


typedef std::chrono::milliseconds TimeT;

typedef struct {
    std::vector<double> &resultVector;
    size_t resultVectorEntry;
    size_t resultVectorEntriesToCalculate;
    std::vector<double> &leftVector;
    std::vector<double> &topMatrix;
    int topMatrixDimensionSize;

} MatrixMultiplicationThreadParams;

// initalization is ensured by zero intialization
const int MaxBarriers = 10;
const int NumberOfThreads = 16;

std::atomic<int> barrier_counter[MaxBarriers];
std::mutex mutex_barrier[MaxBarriers];

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Read transition matrix data from file
////////////////////////////////////////////////////////////////////////////////////////////////////
void
readMat(const std::string &fname, const int nmat, std::vector<double> &mat) {

    // open file
    std::ifstream file(fname, std::ios::binary);
    if (!file.good()) {
        std::cerr << "readMat() : Failed to open '" << fname << "'." << std::endl;
        exit(0);
    }

    // determine file size
    file.seekg(0, std::ios::end);
    auto fsize = file.tellg();
    file.seekg(0);
    assert(fsize / sizeof(double) == nmat * nmat);

    // read data
    mat.resize(fsize / sizeof(double));
    file.read((char *) mat.data(), fsize);

    file.close();
}

void multiplyMatricesThread(MatrixMultiplicationThreadParams *threadParams) {
    size_t topMatrixIterations = threadParams->topMatrix.size() / threadParams->topMatrixDimensionSize;

    std::cout << "Calculating workload from " << threadParams->resultVectorEntry << " to " << threadParams->resultVectorEntry + threadParams->resultVectorEntriesToCalculate << std::endl;
    std::cout << "resultVector.size() = " << threadParams->resultVector.size() << std::endl;

    for (size_t i = threadParams->resultVectorEntry;
         i < threadParams->resultVectorEntry + threadParams->resultVectorEntriesToCalculate; i++) {


        threadParams->resultVector[i] = 0;

        for (size_t j = 0; j < topMatrixIterations; j++) {

            threadParams->resultVector[i] +=
                    threadParams->topMatrix[j * threadParams->topMatrixDimensionSize + i] * threadParams->leftVector[j];
        }
    }
}

std::vector<double>
multiplyMatricesThreaded(std::vector<double> &leftVector, std::vector<double> &topMatrix,
                         int topMatrixDimensionSize) {


    size_t topMatrixIterations = topMatrix.size() / topMatrixDimensionSize;
    assert(leftVector.size() == topMatrixIterations);

    std::vector<double> resultVector(leftVector.size());
    std::vector<std::thread> threads;

    size_t numberToCalculate = leftVector.size() / NumberOfThreads;

    for (size_t i = 0; i < NumberOfThreads; i++) {

        auto *threadParams = new MatrixMultiplicationThreadParams{
                resultVector,
                i * numberToCalculate,
                numberToCalculate,
                leftVector,
                topMatrix,
                topMatrixDimensionSize
        };


        std::cout << "Creating Thread " << i << " with " << numberToCalculate << " workloads" << std::endl;

        threads.emplace_back(std::thread(multiplyMatricesThread, threadParams));
    }

    std::cout << "Created all threads, joining them now." << std::endl;


    for (int i = 0; i < NumberOfThreads; i++) {
        std::cout << "Joining.." << std::endl;
        threads[i].join();
        std::cout << "Joined.." << std::endl;
    }

    return resultVector;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Create stationary vector
////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<double>
multiplyMatrices(std::vector<double> &leftVector, std::vector<double> &topMatrix,
                 int topMatrixDimensionSize) {


    size_t topMatrixIterations = topMatrix.size() / topMatrixDimensionSize;
    assert(leftVector.size() == topMatrixIterations);

    std::vector<double> resultVector(leftVector.size());

    for (size_t i = 0; i < leftVector.size(); i++) {

        resultVector[i] = 0;

        for (size_t j = 0; j < topMatrixIterations; j++) {
            resultVector[i] += topMatrix[j * topMatrixDimensionSize + i] * leftVector[j];
        }
    }

    return resultVector;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//! Create stationary vector
////////////////////////////////////////////////////////////////////////////////////////////////////
void
initializeStationary(std::vector<double> &stationary) {

    for (size_t i = 0; i < stationary.size(); i++) {
        stationary[i] = (double) rand() / RAND_MAX;
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Compute Norm
////////////////////////////////////////////////////////////////////////////////////////////////////

double
computeNorm(std::vector<double> &matrix) {
    size_t lengthOfMatrix = matrix.size();
    double sum = 0;
    for (size_t i = 0; i < lengthOfMatrix; ++i) {
        sum += matrix[i] * matrix[i];
    }

    return sqrt(sum);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//! Scalar Multiplication
////////////////////////////////////////////////////////////////////////////////////////////////////


void
scalarMultiplication(std::vector<double> &matrix, double scalar) {
    for (double &i : matrix) {
        i *= scalar;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Applying Norm
////////////////////////////////////////////////////////////////////////////////////////////////////

void applyNorm(std::vector<double> &matrix, double matrixNorm) {
    scalarMultiplication(matrix, 1.0 / matrixNorm);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Verify stationary vector
////////////////////////////////////////////////////////////////////////////////////////////////////
double
verifyStationary(const int nmat, std::vector<double> &mat, std::vector<double> &stationary) {

    // right matrix-vector product
    std::vector<double> temp(nmat, 0.);
    for (int i = 0; i < nmat; ++i) {
        for (int j = 0; j < nmat; ++j) {
            temp[i] += mat[j * nmat + i] * stationary[j];
        }
    }

    // compute L_1 residual
    double err = 0.;
    for (int i = 0; i < (int) temp.size(); ++i) {

        std::cout << "temp[i] = " << temp[i] << std::endl;
        std::cout << "stationary[i] = " << stationary[i] << std::endl;
        err += std::abs(temp[i] - stationary[i]);
    }

    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple synchronization barrier
// @param id  unique id (uniqueness has to be ensured by used, id has to be <= than MaxBarriers)
// @param
////////////////////////////////////////////////////////////////////////////////////////////////////
void
barrier(const int id, const int num_threads) {

    barrier_counter[id]++;
    mutex_barrier[id].lock();
    while ((barrier_counter[id] % num_threads) != 0) {};
    mutex_barrier[id].unlock();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Implementation of power iteration
//! @num_iterations  number of iterations
//! @nmat  size of matrix
//! @mat   matrix (row-major order)
//! @stationary  stationary vector to be computed
////////////////////////////////////////////////////////////////////////////////////////////////////
double
powerIterationSerial(const int num_iterations, const int nmat,
                     std::vector<double> &mat, std::vector<double> &stationary) {

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_iterations; i++) {

        std::cout << "Iteration = " << i << std::endl;


        auto resultingVector = multiplyMatrices(stationary, mat, nmat);
        auto unnormedVectorSize = computeNorm(resultingVector);
        applyNorm(resultingVector, unnormedVectorSize);


        for (size_t j = 0; j < stationary.size(); j++) {
            stationary[j] = resultingVector[j];
        }

    }

    // TODO: implement power iteration

    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();

    return time;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Parallel implementation of power iteration
//! @num_iterations  number of iterations
//! @nmat  size of matrix
//! @mat   matrix (row-major order)
//! @stationary  stationary vector to be computed
////////////////////////////////////////////////////////////////////////////////////////////////////
double
powerIterationParallel(const int num_iterations, const int nmat,
                       std::vector<double> &mat, std::vector<double> &stationary) {

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_iterations; i++) {

        std::cout << "Iteration = " << i << std::endl;


        auto resultingVector = multiplyMatricesThreaded(stationary, mat, nmat);
        auto unnormedVectorSize = computeNorm(resultingVector);
        applyNorm(resultingVector, unnormedVectorSize);


        for (size_t j = 0; j < stationary.size(); j++) {
            stationary[j] = resultingVector[j];
        }

    }

    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();

    return time;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Parallel implementation of power iteration
//! @num_iterations  number of iterations
//! @nmat  size of matrix
//! @mat   matrix (row-major order)
//! @stationary  stationary vector to be computed
////////////////////////////////////////////////////////////////////////////////////////////////////
double
powerIterationParallelBarrier(const int num_iterations, const int nmat,
                              std::vector<double> &mat, std::vector<double> &stationary) {

    auto start = std::chrono::steady_clock::now();

    // TODO: implement power iteration in parallel without restarting threads every iteration

    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();

    return time;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main(int /*argc*/, char ** /*argv*/) {

    // grid resolution to use
    const int ntheta = 64;
    const int nphi = 128;
    const int nmat = ntheta * nphi;

    // load transition matrix
    std::vector<double> mat;
    std::stringstream fname;
    fname << "./data/mat_" << ntheta << "x" << nphi << ".dat";
    readMat(fname.str(), nmat, mat);


    std::vector<double> stationary(nmat);
    initializeStationary(stationary);

    std::cout << "mat.size() = " << mat.size() << std::endl;
    std::cout << "stationary.size() = " << stationary.size() << std::endl;


    //double timeserial = powerIterationSerial(10, nmat, mat, stationary);
    //std::cout << "time serial = " << timeserial << " ms." << std::endl;

    //auto error = verifyStationary(nmat, mat, stationary);
    //std::cout << "error = " << error << std::endl;


    double timeparallel = powerIterationParallel(10, nmat, mat, stationary);
    std::cout << "time parallel = " << timeparallel << " ms." << std::endl;

    auto error = verifyStationary(nmat, mat, stationary);
    std::cout << "error = " << error << std::endl;


    std::vector<double> stationary_barrier;
    double timebarrier = powerIterationParallelBarrier(50, nmat, mat, stationary_barrier);
    std::cout << "time barrier = " << timebarrier << " ms." << std::endl;


}
