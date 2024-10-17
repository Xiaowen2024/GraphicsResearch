#include "WalkOnStars.h"
#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
using namespace std;

using Vec2D = complex<double>;
using Polyline = vector<Vec2D>;

// Parameters
// num_points = 100
// amplitude = 0.1
// frequency = 2 * np.pi

// # Generate the 3D sinusoidal surface
// x = np.linspace(0, 1, num_points)
// y = np.linspace(0, 1, num_points)
// X, Y = np.meshgrid(x, y)
// Z = amplitude * np.sin(frequency * X) * np.sin(frequency * Y)

double interpolateHeight(Vec2D x){
    double r = abs(x);
    double theta = atan2(imag(x), real(x));
    double amplitude = 0.1;
    double frequency = 2 * M_PI;
    return amplitude * sin(frequency * theta) * (1 - r);
}

void createBoundary(int numPoints, double amplitude, double frequency, vector<Polyline>& boundaryDirichlet){
    vector<Vec2D> vec;
    for (int i = 0; i <= numPoints; ++i) {
        double t = (double)i / numPoints;
        double x = cos(2 * M_PI * t);
        double y = sin(2 * M_PI * t);
        double z = amplitude * sin(frequency * t);
        vec.push_back(Vec2D(x, y));
    }
    boundaryDirichlet.push_back(vec); 
}

vector<Polyline> boundaryDirichlet = {};
vector<Polyline> boundaryNeumann = {};

int main() {
    srand(time(NULL)); 
    std::ofstream out("out.csv"); 

    int numPoints = 100;
    double amplitude = 0.1;
    double frequency = 2 * M_PI;

    createBoundary(numPoints, amplitude, frequency, boundaryDirichlet);

    WalkOnStars w(boundaryDirichlet,boundaryNeumann, interpolateHeight); // Create an instance of WalkOnStars
    int s = 128; // Image size

    for (int j = 0; j < s; j++) {
        std::cerr << "row " << j << " of " << s << std::endl; 
        for (int i = 0; i < s; i++) {
            Vec2D x0(((double)i + 0.5) / ((double)s), ((double)j + 0.5) / ((double)s));
            double u = 0.0;

            // Check if the point is inside the domain and solve if it is
            if (w.insideDomain(x0, w.boundaryDirichlet, w.boundaryNeumann)) {
                u = w.solve(x0, w.boundaryDirichlet, w.boundaryNeumann, w.interpolate);
            }

            out << u;
            if (i < s - 1) out << ",";
        }
        out << std::endl;
    }

    out.close();
    return 0;
}