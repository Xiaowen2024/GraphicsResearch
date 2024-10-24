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

using Vec2D = Eigen::Vector2d;
using Polyline = vector<Vec2D>;

// Define saddle point surface as Z = X**2 - Y**2

void createSaddlePointBoundary(double x1, double y1, double x2, double y2, int numPoints, vector<Polyline>& boundaryDirichlet) {
    Polyline boundary;

    double xStep = (x2 - x1) / numPoints;
    double yStep = (y2 - y1) / numPoints;

    // Bottom edge (left to right)
    for (double x = x1; x <= x2; x += xStep) {
        boundary.push_back(Vec2D(x, y1));
    }

    // Right edge (bottom to top)
    for (double y = y1 + yStep; y <= y2; y += yStep) {
        boundary.push_back(Vec2D(x2, y));
    }

    // Top edge (right to left)
    for (double x = x2 - xStep; x >= x1; x -= xStep) {
        boundary.push_back(Vec2D(x, y2));
    }

    // Left edge (top to bottom)
    for (double y = y2 - yStep; y > y1; y -= yStep) {
        boundary.push_back(Vec2D(x1, y));
    }

    boundaryDirichlet.push_back(boundary);
}

// typedef double (*HeightFunction)(double, double);

// double interpolateSaddlePointHeight(double x, double y, HeightFunction heightFunc) {
//     return heightFunc(x, y);
// }


double getSaddlePointHeight(Vec2D x) {
    return x.x() * x.x() - x.y() * x.y();
}   

double interpolateHeight(Vec2D x){
    double r = x.norm();
    double theta = atan2(x.x(), x.y());
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
        vec.push_back(Vec2D(x, y));
    }
    boundaryDirichlet.push_back(vec); 
} 

int main() {
    srand(time(NULL)); 
    std::ofstream out("saddlePoint.csv"); 
    vector<Polyline> boundaryDirichlet;  
    vector<Polyline> boundaryNeumann;    
    createSaddlePointBoundary(-1.0, -1.0, 1.0, 1.0, 30, boundaryDirichlet);
    

    WalkOnStars w(boundaryDirichlet, boundaryNeumann, getSaddlePointHeight); // Create an instance of WalkOnStars
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
