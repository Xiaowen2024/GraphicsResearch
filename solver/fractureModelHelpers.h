#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
using namespace std;

#ifndef FRACTURE_MODEL_LIB_H
#define FRACTURE_MODEL_LIB_H

const double infinity = numeric_limits<double>::infinity();

// returns a random value in the range [rMin,rMax]
inline double random(double rMin, double rMax) {
    const double rRandMax = 1.0 / (double)RAND_MAX;
    double u = rRandMax * (double)rand();
    return u * (rMax - rMin) + rMin;
}

// use std::complex to implement 2D vectors
using Vec2D = complex<double>;
inline double length(Vec2D u) { return sqrt(norm(u)); }
inline double angleOf(Vec2D u) { return arg(u); }
inline Vec2D rotate90(Vec2D u) { return Vec2D(-imag(u), real(u)); }
inline double dot(Vec2D u, Vec2D v) { return real(u) * real(v) + imag(u) * imag(v); }
inline double cross(Vec2D u, Vec2D v) { return real(u) * imag(v) - imag(u) * real(v); }
inline vector<Vec2D> outerProduct(Vec2D u, Vec2D v) {
    return vector<Vec2D>{Vec2D(real(u) * real(v), real(u) * imag(v)), Vec2D(imag(u) * real(v), imag(u) * imag(v))};
}
inline vector<Vec2D> matrixAdd(vector<Vec2D> A, vector<Vec2D> B) {
    vector<Vec2D> result = {};
    for (int i = 0; i < A.size(); i++) {
        result.push_back(A[i] + B[i]);
    }
    return result;
}
inline vector<Vec2D> matrixSubstract(vector<Vec2D> A, vector<Vec2D> B) {
    vector<Vec2D> result = {};
    for (int i = 0; i < A.size(); i++) {
        result.push_back(A[i] - B[i]);
    }
    return result;
}
inline vector<Vec2D> matrixMultiply(vector<Vec2D> A, vector<Vec2D> B) {
    vector<Vec2D> result = {};
    for (int i = 0; i < A.size(); i++) {
        result.push_back(Vec2D(dot(A[i], B[0]), dot(A[i], B[1])));
    }
    return result;
}
inline Vec2D matrixVectorMultiply(vector<Vec2D> A, Vec2D v) {
    return Vec2D(dot(A[0], v), dot(A[1], v));
}

inline vector<Vec2D> scalarMultiplyMatrix(double scalar, vector<Vec2D> matrix) {
    vector<Vec2D> result;
    for (int i = 0; i < matrix.size(); i++) {
        result.push_back(scalar * matrix[i]);
    }
    return result;
}

inline Vec2D closestPoint(Vec2D x, Vec2D a, Vec2D b) {
    Vec2D u = b - a;
    double t = clamp(dot(x - a, u) / dot(u, u), 0.0, 1.0);
    return (1.0 - t) * a + t * b;
}

inline bool isSilhouette(Vec2D x, Vec2D a, Vec2D b, Vec2D c) {
    return cross(b - a, x - a) * cross(c - b, x - b) < 0;
}

inline double rayIntersection(Vec2D x, Vec2D v, Vec2D a, Vec2D b) {
    Vec2D u = b - a;
    Vec2D w = x - a;
    double d = cross(v, u);
    double s = cross(v, w) / d;
    double t = cross(u, w) / d;
    if (t > 0. && 0. <= s && s <= 1.) {
        return t;
    }
    return infinity;
}

inline vector<Vec2D> getSymmetricMatrix(Vec2D v) {
    if (sqrt(norm(v)) == 0) {
        return vector<Vec2D>{Vec2D{0, 0}, Vec2D{0, 0}};
    } else {
        vector<Vec2D> op = outerProduct(v, v);
        double length = sqrt(norm(v));
        return vector<Vec2D>{Vec2D{real(op[0]) / length, imag(op[0]) / length}, Vec2D{real(op[1]) / length, imag(op[1]) / length}};
    }
}

inline vector<Vec2D> getStress(double lam, double mu, double trace, double dudx, double dudy, double dvdx, double dvdy) {
    return matrixAdd(vector<Vec2D>{Vec2D{1 * lam * trace, 0}, Vec2D{0, 1 * lam * trace}}, vector<Vec2D>{Vec2D{2 * mu * dudx, 2 * mu * dudy}, Vec2D{2 * mu * dvdx, 2 * mu * dvdy}});
}

using Polyline = vector<Vec2D>;
vector<Polyline> newBoundaryDirichlet = {
    {
        {Vec2D(-0.2, 0), Vec2D(0.3, 0), Vec2D(0.5, 0.5), Vec2D(0.7, 0), Vec2D(1.2, 0), Vec2D(1.0, 1), Vec2D(0, 1), Vec2D(-0.2, 0)}
    }
};

double distancePolylines(Vec2D x, const vector<Polyline>& P);

double silhouetteDistancePolylines(Vec2D x, const vector<Polyline>& P);

Vec2D intersectPolylines(Vec2D x, Vec2D v, double r, const vector<Polyline>& P, Vec2D& n, bool& onBoundary);

Vec2D solve(Vec2D x0, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D)> g);


inline double signedAngle(Vec2D x, const vector<Polyline>& P) {
    double Theta = 0.;
    for (int i = 0; i < P.size(); i++)
        for (int j = 0; j < P[i].size() - 1; j++)
            Theta += arg((P[i][j + 1] - x) / (P[i][j] - x));
    return Theta;
}

Vec2D interpolateVec2D_BoundaryPoints(Vec2D v, vector<Polyline> mappings, double num_tol = 1e-3, bool print_in_bounds = false, bool print_out_bounds = false);

Vec2D displacement(Vec2D v);

bool insideDomain(Vec2D x, const vector<Polyline>& boundaryDirichlet, const vector<Polyline>& boundaryNeumann);

vector<Vec2D> getDeformationGradientAndStress(Vec2D point, double h, function<Vec2D(Vec2D)> deform, std::ofstream& strainFile, std::ofstream& neighbourFile, std::ofstream& stressFile, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann);

vector<pair<double, Vec2D>> eigenDecomposition(vector<Vec2D> A);

pair<vector<Vec2D>, vector<Vec2D>> forceDecomposition(vector<Vec2D> stress, vector<pair<double, Vec2D>> eigenpairs);

Vec2D getDirectHomogenousForce(vector<Vec2D> stressComponent, Vec2D normal);

vector<Vec2D> getSeparationTensor(Vec2D tensileForce, Vec2D compressiveForce, vector<Vec2D> neighbourTensileForces, vector<Vec2D> neighbourCompressiveForces);

Vec2D determineCrackPropagationDirection(vector<Vec2D> separationTensor, double threshold);

#endif
