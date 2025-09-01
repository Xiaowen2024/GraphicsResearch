
// Compile with: c++ -std=c++17 -O3 -pedantic -Wall -Xclang -fopenmp gradientEstimateLame.cpp -o gel -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -lomp -w
// c++ -std=c++17 -O3 -pedantic -Wall -Xclang -fopenmp gradientEstimateLame.cpp -o gel -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -I /opt/homebrew/opt/libomp/include -L /opt/homebrew/opt/libomp/lib -lomp -w

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <chrono> 
#include <omp.h>
using namespace std;
using namespace std::chrono; 

#include <Eigen/Dense> 
#include "fractureModelHelpers.cpp"
#include "risSampler.cpp"
#include "fractureModelHelpers.h"

std::random_device rd;  // Non-deterministic random number generator
std::mt19937 gen(rd()); // Mersenne Twister PRNG
std::uniform_real_distribution<double> dist(-M_PI, M_PI);

// use std::complex to implement 2D vectors
using Vec2D = complex<double>;

// Overload operator* for scalar multiplication with Vec2D
inline Vec2D operator*(double scalar, const Vec2D& vec) {
    return Vec2D(scalar * real(vec), scalar * imag(vec));
}

inline vector<Vec2D> operator*(double scalar, const vector<Vec2D>& vec) {
   vector<Vec2D> result;
   for (const auto& v : vec) {
      result.push_back(scalar * v);
   }
   return result;
}

pair<Vec2D, double> sampleNeumannBoundary() {
   // Sample a point on the Neumann boundary and return its PDF
   static std::random_device rd;   // Seed generator (non-deterministic)
   static std::mt19937 gen(rd());  // Mersenne Twister RNG
   std::uniform_real_distribution<double> dist(0.0, 1.0);
   double pdf;
   
   double x = dist(gen);
   double y = dist(gen);
   
   if (y < 0.5) {
      pdf = 0.5;  // Probability density for this specific point on the bottom edge
      return make_pair(Vec2D(x, 0), pdf);  // Bottom edge
   } else {
      pdf = 0.5;  // Probability density for this specific point on the top edge
      return make_pair(Vec2D(x, 1), pdf);    // Top edge
   }
}

using Polyline = vector<Vec2D>;

Vec2D interpolateVec2DBoundaryPoints(Vec2D v, vector<Polyline> originalPoints, vector<Polyline> displacedPoints, double num_tol=1e-3) { 
   for (int j = 0; j < originalPoints.size(); j++) {
      const Polyline& polyline = originalPoints[j];
      const Polyline& displacedPolyline = displacedPoints[j];
      for (int i = 0; i < polyline.size() - 1; i++) {
         Vec2D AP = v - polyline[i];
         Vec2D PB = v - polyline[i + 1];
         Vec2D AB = polyline[i + 1] - polyline[i];
         Vec2D displaced1 = displacedPolyline[i]; 
         Vec2D displaced2 = displacedPolyline[i + 1];
   
         if (abs(length(AP) + length(PB) - length(AB)) < num_tol) {
            Vec2D displaced = displaced1 + (displaced2 - displaced1) * length(AP) / length(AB);
            return displaced;
         }
      }   
   }
   
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   return nan;
}

Vec2D sampleRectangleBoundary() {
   // Define the rectangle's properties (unit square from 0,0 to 1,1)
   double min_x = 0.0;
   double min_y = 0.0;
   double max_x = 1.0;
   double max_y = 1.0;

   // Calculate the lengths of the sides
   double width = max_x - min_x;
   double height = max_y - min_y;

   // Calculate the total perimeter
   double perimeter = 2 * (width + height);

   // Initialize random number generator
   // std::random_device provides a non-deterministic seed
   std::random_device rd;
   // std::mt19937 is a Mersenne Twister pseudo-random number generator
   std::mt19937 gen(rd());
   // std::uniform_real_distribution generates uniformly distributed real numbers
   // in the range [0, perimeter)
   std::uniform_real_distribution<> distrib(0.0, perimeter);

   // Generate a random distance along the perimeter
   double random_distance = distrib(gen);

   double x, y;

   // Determine which side the point falls on and calculate its coordinates
   // Side 1: Bottom edge (from (0,0) to (1,0))
   if (random_distance < width) {
       x = min_x + random_distance;
       y = min_y;
   }
   // Side 2: Right edge (from (1,0) to (1,1))
   else if (random_distance < width + height) {
       x = max_x;
       y = min_y + (random_distance - width);
   }
   // Side 3: Top edge (from (1,1) to (0,1))
   else if (random_distance < 2 * width + height) {
       x = max_x - (random_distance - (width + height));
       y = max_y;
   }
   // Side 4: Left edge (from (0,1) to (0,0))
   else {
       x = min_x;
       y = max_y - (random_distance - (2 * width + height));
   }

   return Vec2D{x, y};
}

pair<Vec2D, double> rectangleBoundarySampler() {
   double min_x = 0.0, min_y = 0.0, max_x = 1.0, max_y = 1.0;
   double width = max_x - min_x, height = max_y - min_y;
   double perimeter = 2.0 * (width + height);

   static thread_local std::mt19937 gen((std::random_device())());
   static std::uniform_real_distribution<> distrib(0.0, perimeter);

   double random_distance = distrib(gen);
   double x, y;

   if (random_distance < width) {
       x = min_x + random_distance; y = min_y;
   } else if (random_distance < width + height) {
       x = max_x; y = min_y + (random_distance - width);
   } else if (random_distance < 2 * width + height) {
       x = max_x - (random_distance - (width + height));
       y = max_y;
   } else {
       x = min_x;
       y = max_y - (random_distance - (2 * width + height));
   }

   double pdf = 1.0 / perimeter;
   return make_pair(Vec2D{x, y}, 1 / pdf);
}


// vector<Polyline> boundaryDirichlet = {{ Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.49, 0), Vec2D(0.5, 0.2), Vec2D(0.51, 0), Vec2D(1, 0)}};
// vector<Polyline> displacedPoints =   {{ Vec2D(1.1, 0), Vec2D(1.1, 1), Vec2D(-0.1, 1), Vec2D(-0.1, 0)}};

// vector<Polyline> boundaryDirichlet = {{  Vec2D(0.0, 1.0), Vec2D(0, 0)}, {Vec2D(1, 0), Vec2D(1, 1)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.49, 0), Vec2D(0.5, 0.2), Vec2D(0.51, 0), Vec2D(1, 0)}, { Vec2D(1, 1), Vec2D(0, 1)} };
// vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1.0), Vec2D(-0.1, 0)}, {Vec2D(1.1, 0), Vec2D(1.1, 1)}};

// vector<Polyline> boundaryDirichlet = {{  Vec2D(0, 1), Vec2D(0, 0)}, {Vec2D(0.2, 0), Vec2D(0.2, 1)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.2, 0)}, { Vec2D(0.2, 1), Vec2D(0, 1)} };
// vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1.0), Vec2D(-0.1, 0)}, {Vec2D(0.3, 0), Vec2D(0.3, 1)}};

vector<Polyline> boundaryDirichlet = {{ Vec2D(0, 1), Vec2D(0, 0)}, { Vec2D(1, 0), Vec2D(1, 1)}};
vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(1, 0)}, { Vec2D(1, 1), Vec2D(0, 1)}};
vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1), Vec2D(-0.1, 0)}, { Vec2D(1.1, 0), Vec2D(1.1, 1)}};

bool isPointOnLine(Vec2D point, Vec2D lineStart, Vec2D lineEnd, double tolerance = 1e-5) {
   Vec2D AP = point - lineStart;
   Vec2D PB = point - lineEnd;
   Vec2D AB = lineEnd - lineStart;

   // Check if the sum of distances from the point to the line's endpoints equals the line's length
   return abs(length(AP) + length(PB) - length(AB)) < tolerance;
}

Vec2D getDirichletValue(Vec2D v, vector<Polyline> boundaryDirichlet, vector<Polyline> displacedPoints) { 
   const double tolerance = 1e-4; // Define a small tolerance for comparison
   Vec2D interpolatedDisplacement = interpolateVec2DBoundaryPoints(v, boundaryDirichlet, displacedPoints);
   if (!isnan(real(interpolatedDisplacement)) && !isnan(imag(interpolatedDisplacement))) {
      return interpolatedDisplacement - v;
   } else {
      cout << "Error: getDirichletValue called with point not on Dirichlet boundary: " << real(v) << ", " << imag(v) << endl;
      Vec2D nan = numeric_limits<double>::quiet_NaN();
      return nan;
   }
}

bool isOnDirichlet(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
   const double tolerance = 1e-4; // Define a small tolerance for comparison
   for (const auto& polyline : boundaryDirichlet) {
      for (size_t i = 0; i < polyline.size() - 1; ++i) {
         Vec2D lineStart = polyline[i];
         Vec2D lineEnd = polyline[i + 1];
         if (isPointOnLine(point, lineStart, lineEnd, tolerance)) {
            return true;
         }
      }
   }
   return false;
}


bool isOnNeumann(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
   const double tolerance = 1e-4; // Define a small tolerance for comparison
   for (const auto& polyline : boundaryNeumann) {
      for (size_t i = 0; i < polyline.size() - 1; ++i) {
         Vec2D lineStart = polyline[i];
         Vec2D lineEnd = polyline[i + 1];
         if (isPointOnLine(point, lineStart, lineEnd, tolerance)) {
            return true;
         }
      }
   }
   return false;
}

enum class BoundaryType {
   Dirichlet,  // For Dirichlet boundary conditions
   Neumann,    // For Neumann boundary conditions  
   None        // For points not on any boundary
};

BoundaryType getBoundaryTypeAtPoint(Vec2D point, const vector<Polyline>& boundaryDirichlet, const vector<Polyline>& boundaryNeumann) {
   if (isOnDirichlet(point, boundaryDirichlet, boundaryNeumann)) {
       return BoundaryType::Dirichlet;
   } else if (isOnNeumann(point, boundaryDirichlet, boundaryNeumann)) {
       return BoundaryType::Neumann;
   } else {
       return BoundaryType::None;
   }
}

vector<Vec2D> getNormal(Vec2D point) { 
   const double tolerance = 1e-3; // Define a small tolerance for comparison
   vector<Vec2D> normals;

   if (abs(real(point)) < tolerance) {
      normals.push_back(Vec2D(-1, 0));
   }
   if (abs(real(point) - 1) < tolerance) {
      normals.push_back(Vec2D(1, 0));
   }
   if (abs(imag(point)) < tolerance) {
      normals.push_back(Vec2D(0, -1));
   }
   if (abs(imag(point) - 1) < tolerance) {
      normals.push_back(Vec2D(0, 1));
   }
   if (isPointOnLine(point, Vec2D(0, 0), Vec2D(0.5, 0.2), tolerance)) {
      Vec2D lineStart = Vec2D(0, 0);
      Vec2D lineEnd = Vec2D(0.5, 0.2);
      Vec2D lineDirection = lineEnd - lineStart;
      Vec2D normal = Vec2D(imag(lineDirection), -real(lineDirection));
      normals.push_back(normal);
   } 
   if (isPointOnLine(point, Vec2D(0.5, 0.2), Vec2D(1, 0), tolerance)) {
      Vec2D lineStart = Vec2D(0.5, 0.2);
      Vec2D lineEnd = Vec2D(1, 0);
      Vec2D lineDirection = lineEnd - lineStart;
      Vec2D normal = Vec2D(imag(lineDirection), -real(lineDirection));
      normals.push_back(normal);
   }
   return normals;
}

bool written = false;

Vec2D getNeumannValue( Vec2D point, vector<Polyline> boundaryNeumann) {
   const double tolerance = 1e-5; // Define a small tolerance for comparison
   // Check if the point is on the Neumann boundary
   if (abs(imag(point) - 1) < tolerance) {
      return Vec2D(0, -0.1);
   }
   else if (abs(imag(point)) < tolerance) {
      return Vec2D(0, 0.1);
   }
   else if (abs(real(point)) < tolerance) {
      return Vec2D(0, 0);
   }
   else if (abs(real(point) - 1) < tolerance) {
      return Vec2D(0, 0);
   }
   else {
      cout << "Error: getNeumannValue called with point not on Neumann boundary: " << real(point) << ", " << imag(point) << endl;
      Vec2D nan = numeric_limits<double>::quiet_NaN();
      return nan;
   }
}

double get2DFreeSpaceGreenKernel( Vec2D x, Vec2D y ) {
   // Green's function for the Laplace operator in 2D
   // G(x,y) = 1/(2*pi) * log(|y-x|)
   if (x == y) {
      return 0.0;
   }
   return 1.0/(2.0*M_PI) * log(length(y-x));
}

double get2DPoissonKernel(Vec2D x, Vec2D y, Vec2D n) {
   // Poisson kernel for the unit disk
   // P(x,y) = 1/((2*pi) * /|y-x|^2) * (y - x) * n 
   // where n is the unit normal at y
   if (x == y) {
      return 0.0;
   }
   return 1.0/(2.0*M_PI) * dot(n, y - x)/ (length(y-x)*length(y-x));
}


double get2DFreeSpaceGreenKernelforBall( Vec2D y, Vec2D x, double R) {
   if (x == y) {
      return 0.0;
   }
   return 1.0/(2.0*M_PI) * log(R / length(y-x));
}

double get2DPoissonKernelforBall(double r) {
   return 1.0/ ((2.0*M_PI) * r);
}

double sampleSourceDistance(double rMin, double rMax) {
   static std::random_device rd;
   static std::mt19937 gen(rd());
   static std::uniform_real_distribution<double> dist(0.0, 1.0);

   double U = dist(gen); // Uniform sample in (0,1)
   return rMin * exp(U * log(rMax / rMin));
}

double kernelAii(double xi, double yi, double R, double poissonRatio) {
   double first = 3 / (8 - 12 * poissonRatio);
   double second = 5 * (xi - yi) * (xi - yi) / (R * R) + 1 - 4 * poissonRatio;
   return first * second;
}

double kernelAij(double xi, double yi, double xj, double yj, double R, double poissonRatio) {
   double first = 3 / (8 - 12 * poissonRatio);
   double second = 5 * (xi - yi) * (xj - yj ) / (R * R);
   return first * second;
}

vector<Vec2D> KKernel(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
   double dx = xi - xj;
   double dy = yi - yj;
   double R = sqrt(dx*dx + dy*dy);
   
   double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
   double ln_R = log(R);
   double dx_dx = (dx * dx) / (R * R);
   double dy_dy = (dy * dy) / (R * R);
   double dx_dy = (dx * dy) / (R * R);
   
   // Diagonal terms
   
   double first =  coeff * dx_dx;
   double second = coeff * ((3.0 - 4.0 * poissonRatio) * ln_R + dx_dy);
   double third = coeff * ((3.0 - 4.0 * poissonRatio) * ln_R + dx_dy);
   double fourth = coeff * dy_dy;
   return {Vec2D(first, second), 
           Vec2D(third, fourth)};
}

std::vector<Vec2D> TKernel(Vec2D x, Vec2D x0,
   double shearModulus, double poissonRatio, Vec2D normal) {

   // Vector r = x - x0
   Vec2D r_vec = x - x0;
   double r = length(r_vec);

   // Handle the singularity if r is very small (points are coincident)
   if (r < 1e-9) { // A small tolerance to prevent division by zero
      // In real BEM applications, special handling for coincident points is required.
      // For now, return a zero matrix or throw an error.
      std::cerr << "Warning: r is too small. Returning zero matrix." << std::endl;
      return {Vec2D(0, 0), Vec2D(0, 0)};
   }

   // d = 2 for 2D
   const double d = 2.0;

   // Constants a and b
   double a = 1.0 / (2.0 * shearModulus * M_PI);
   double b = 1.0 / (8.0 * (1.0 - poissonRatio) * shearModulus * M_PI);

   // Normalized vector r_hat = r_vec / r
   Vec2D r_hat = r_vec / r;

   // Term 1: (n^T * r)
   double n_dot_r = dot(normal, r_hat);

   // Term 2: Identity Matrix I (for 2D)
   // I = [1 0]
   //     [0 1]
   std::vector<Vec2D> I_matrix = {Vec2D(1, 0), Vec2D(0, 1)};

   // Term 3: n * r^T (outer product)
   // [n.x * r.x   n.x * r.y]
   // [n.y * r.x   n.y * r.y]
   std::vector<Vec2D> n_rT_matrix = {
      Vec2D(real(normal) * real(r_hat), real(normal) * imag(r_hat)),
      Vec2D(imag(normal) * real(r_hat), imag(normal) * imag(r_hat))
   };

   // Term 4: r * n^T (outer product)
   // [r.x * n.x   r.x * n.y]
   // [r.y * n.x   r.y * n.y]
   std::vector<Vec2D> r_nT_matrix = {
      Vec2D(real(r_hat) * real(normal), real(r_hat) * imag(normal)),
      Vec2D(imag(r_hat) * real(normal), imag(r_hat) * imag(normal))
   };

   // Calculate the first bracket term: [ (n^T r)I + nr^T - rn^T ]
   std::vector<Vec2D> bracket_term(2);
   bracket_term[0] = Vec2D(
      n_dot_r * real(I_matrix[0]) + real(n_rT_matrix[0]) - real(r_nT_matrix[0]),
      n_dot_r * imag(I_matrix[0]) + imag(n_rT_matrix[0]) - imag(r_nT_matrix[0])
   );
   bracket_term[1] = Vec2D(
      n_dot_r * real(I_matrix[1]) + real(n_rT_matrix[1]) - real(r_nT_matrix[1]),
      n_dot_r * imag(I_matrix[1]) + imag(n_rT_matrix[1]) - imag(r_nT_matrix[1])
   );

   // First main term of T
   // (mu * (a - 2b) / r^d) * bracket_term
   double factor1 = shearModulus * (a - 2.0 * b) / std::pow(r, d);
   std::vector<Vec2D> term1(2);
   term1[0] = bracket_term[0] * factor1;
   term1[1] = bracket_term[1] * factor1;

   // Second main term: (2 * mu * b * d / r^(d+2)) * (n^T r) * r r^T
   double factor2 = (2.0 * shearModulus * b * d) / std::pow(r, d + 2.0);

   // Calculate r r^T (outer product)
   // [r.x * r.x   r.x * r.y]
   // [r.y * r.x   r.y * r.y]
   std::vector<Vec2D> r_rT_matrix = {
      Vec2D(real(r_hat) * real(r_hat), real(r_hat) * imag(r_hat)),
      Vec2D(imag(r_hat) * real(r_hat), imag(r_hat) * imag(r_hat))
   };

   std::vector<Vec2D> term2(2);
   term2[0] = r_rT_matrix[0] * (factor2 * n_dot_r);
   term2[1] = r_rT_matrix[1] * (factor2 * n_dot_r);

   // Final T matrix = term1 + term2
   std::vector<Vec2D> T_matrix(2);
   T_matrix[0] = term1[0] + term2[0];
   T_matrix[1] = term1[1] + term2[1];

   return T_matrix;
}

vector<Vec2D> PKernel(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio, Vec2D normal) {
   double dx = xi - xj;
   double dy = yi - yj;
   double R = sqrt(dx*dx + dy*dy);
   
   double coeff = - 1.0 / (4.0 * M_PI * (1.0 - poissonRatio) * R);
   double dr_dn = dot(normal, Vec2D(dx, dy)) / R; 
   double dx_dx = (dx * dx) / (R * R);
   double dy_dy = (dy * dy) / (R * R);
   double dx_dy = (dx * dy) / (R * R);
   
   double first1 = dr_dn * 2 * dx_dx;
   double second1 = (1 - 2 * poissonRatio) * (real(normal) * dx - real(normal) * dx);
   double one = coeff * (first1 + second1);

   double first2 = dr_dn * ((1 - 2 * poissonRatio) + 2 * dx_dy);
   double second2 = (1 - 2 * poissonRatio) * (real(normal) * dy - imag(normal) * dx);
   double two = coeff * (first2 + second2);

   double first3 = dr_dn * ((1 - 2 * poissonRatio) + 2 * dx_dy);
   double second3 = (1 - 2 * poissonRatio) * (imag(normal) * dx - real(normal) * dy);
   double three = coeff * (first3 + second3);

   double first4 = dr_dn * 2 * dx_dy;
   double second4 = (1 - 2 * poissonRatio) * (imag(normal) * dy - imag(normal) * dy);
   double four = coeff * (first4 + second4);

   return {Vec2D(one, three), Vec2D(four, two)};
}

// double Gii(double xi, double yi, double xj, double yj, double R, double G, double poissonRatio) {
//    double first = - 1 / (8 * M_PI * G * (1 - poissonRatio));
//    double second = (xi - yi) * (xi - yi) / (R * R);
//    return first * second;
// }

// double Gij(double xi, double yi, double xj, double yj, double R, double G, double poissonRatio) {
//    double first = - 1 / (8 * M_PI * G);
//    double second = (3 - poissonRatio) * log(R) + (1 + poissonRatio) * (xi - yi) * (xj - yj) / (R * R);
//    return first * second;
// }


// double Gii(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
//    double dx = xi - xj;  // Fixed: should be xi - xj, not xi - yi
//    double dy = yi - yj;  // Fixed: should be yi - yj
//    double R = sqrt(dx*dx + dy*dy);
   
//    if (R == 0.0) return 0.0;  // Avoid singularity
   
//    double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
//    double term1 = -(3.0 - 4.0 * poissonRatio) * log(R);  // Fixed: missing (3-4ν)ln(R) term
//    double term2 = (dx * dx) / (R * R);  // For G11: (x1-ξ1)²/r²
   
//    return coeff * (term1 + term2);
// }

// // Off-diagonal components G12, G21
// double Gij(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
//    double dx = xi - xj;
//    double dy = yi - yj;
//    double R = sqrt(dx*dx + dy*dy);
   
//    if (R == 0.0) return 0.0;  // Avoid singularity
   
//    double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
//    double term = (dx * dy) / (R * R);  // Fixed: (x1-ξ1)(x2-ξ2)/r²
   
//    return coeff * term;
// }

// vector<Vec2D> Green2D(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
//    double dx = xi - xj;
//    double dy = yi - yj;
//    double R = sqrt(dx*dx + dy*dy);
   
//    double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
//    double ln_R = log(R);
//    double dx_dx = (dx * dx) / (R * R);
//    double dy_dy = (dy * dy) / (R * R);
//    double dx_dy = (dx * dy) / (R * R);
   
//    // Diagonal terms
//    double first = coeff * (-(3.0 - 4.0 * poissonRatio) * ln_R + dx_dx);  // G11
//    double second = coeff * (-(3.0 - 4.0 * poissonRatio) * ln_R + dy_dy);  // G22
   
//    // Off-diagonal terms
//    double third = coeff * dx_dy;  // G12
//    double fourth = coeff * dx_dy;  // G21
//    return {Vec2D(first, second), Vec2D(third, fourth)};
// }

double Gii(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
   double dx = xi - xj;  // Fixed: should be xi - xj, not xi - yi
   double dy = yi - yj;  // Fixed: should be yi - yj
   double R = sqrt(dx*dx + dy*dy);
   
   if (R == 0.0) return 0.0;  // Avoid singularity
   
   double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
   double term1 = -(3.0 - 4.0 * poissonRatio) * log(R);  // Fixed: missing (3-4ν)ln(R) term
   double term2 = (dx * dx) / (R * R);  // For G11: (x1-ξ1)²/r²
   
   return coeff * (term1 + term2);
}

// Off-diagonal components G12, G21
double Gij(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
   double dx = xi - xj;
   double dy = yi - yj;
   double R = sqrt(dx*dx + dy*dy);
   
   if (R == 0.0) return 0.0;  // Avoid singularity
   
   double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
   double term = (dx * dy) / (R * R);  // Fixed: (x1-ξ1)(x2-ξ2)/r²
   
   return coeff * term;
}

vector<Vec2D> Green2D(double xi, double yi, double xj, double yj, double shearModulus, double poissonRatio) {
   double dx = xi - xj;
   double dy = yi - yj;
   double R = sqrt(dx*dx + dy*dy);
   
   double coeff = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
   double ln_R = log(R);
   double dx_dx = (dx * dx) / (R * R);
   double dy_dy = (dy * dy) / (R * R);
   double dx_dy = (dx * dy) / (R * R);
   
   // Diagonal terms
   double first = coeff * (-(3.0 - 4.0 * poissonRatio) * ln_R + dx_dx);  // G11
   double second = coeff * (-(3.0 - 4.0 * poissonRatio) * ln_R + dy_dy);  // G22
   
   // Off-diagonal terms
   double third = coeff * dx_dy;  // G12
   double fourth = coeff * dx_dy;  // G21
   return {Vec2D(first, second), Vec2D(third, fourth)};
}

// CORRECTED Kelvin Kernel (Somigliana Tensor)
vector<Vec2D> kelvinKernel(double shearModulus, double poissonRatio, Vec2D x_minus_y) {
   // The vector r in the formula is y - x. Here we are given x - y. So, r = -x_minus_y.
   Vec2D r_vec(-real(x_minus_y), -imag(x_minus_y));

   double r_len_sq = real(r_vec) * real(r_vec) + imag(r_vec) * imag(r_vec);
   if (r_len_sq < 1e-12) {
       return { Vec2D(0,0), Vec2D(0,0) };
   }
   double r_len = std::sqrt(r_len_sq);

   // Standard formulation constants for 2D plane strain
   double constant = 1.0 / (8.0 * M_PI * shearModulus * (1.0 - poissonRatio));
   double log_factor = 3.0 - 4.0 * poissonRatio;

   // The log term must be negative (it comes from log(1/r))
   double log_val = -std::log(r_len);

   // The r_i * r_j / r^2 part
   double r1r1_over_r_sq = (real(r_vec) * real(r_vec)) / r_len_sq;
   double r1r2_over_r_sq = (real(r_vec) * imag(r_vec)) / r_len_sq;
   double r2r2_over_r_sq = (imag(r_vec) * imag(r_vec)) / r_len_sq;

   // Assemble the correct tensor components
   double k11 = constant * (log_factor * log_val + r1r1_over_r_sq);
   double k12 = constant * (r1r2_over_r_sq);
   double k21 = k12; // Symmetric
   double k22 = constant * (log_factor * log_val + r2r2_over_r_sq);

   return { Vec2D(k11, k12), Vec2D(k21, k22) };
}

std::vector<std::vector<Vec2D>> kelvinKernelGradient(double shearModulus, double poissonRatio, Vec2D x_minus_y) {
   // The vector r in the formula is y - x. Here we are given x - y.
   // So, r = -x_minus_y.
   Vec2D r(-real(x_minus_y), -imag(x_minus_y));

   double r_sq = std::norm(r); // For complex numbers, norm(r) is real(r)^2 + imag(r)^2
   if (r_sq < 1e-12) {
       return {{Vec2D(0,0), Vec2D(0,0)}, {Vec2D(0,0), Vec2D(0,0)}};
   }

   // This constant is different from the Kelvin kernel itself.
   // This is the derivative of the Kelvin Kernel from the original paper.
   double constant_factor = 1.0 / (4.0 * M_PI * shearModulus * (1.0 - poissonRatio));
   double C1 = (3.0 - 4.0 * poissonRatio);
   double C2 = 1.0; 

   // Pre-calculate common terms
   double r_x = real(r);
   double r_y = imag(r);
   double r_pow2_inv = 1.0 / r_sq;
   double r_pow4_inv = r_pow2_inv * r_pow2_inv;

   std::vector<std::vector<Vec2D>> dU_dy(2, std::vector<Vec2D>(2));

   for (int i = 0; i < 2; ++i) {
       for (int j = 0; j < 2; ++j) {
           // Helper values for r_i and r_j
           double ri = (i == 0) ? r_x : r_y;
           double rj = (j == 0) ? r_x : r_y;
           
           // Derivative w.r.t. y_x (k=0)
           double delta_ij = (i == j) ? 1.0 : 0.0;
           double delta_ik = (i == 0) ? 1.0 : 0.0; // k=0 for y_x derivative
           double delta_jk = (j == 0) ? 1.0 : 0.0;
           double rk = r_x;
           
           double term1_dx = -C1 * delta_ij * rk * r_pow2_inv;
           double term2_dx = C2 * ((delta_ik * rj + ri * delta_jk) * r_pow2_inv - 2 * ri * rj * rk * r_pow4_inv);
           double dU_ij_dyx = constant_factor * (term1_dx + term2_dx);
           
           // Derivative w.r.t. y_y (k=1)
           delta_ik = (i == 1) ? 1.0 : 0.0; // k=1 for y_y derivative
           delta_jk = (j == 1) ? 1.0 : 0.0;
           rk = r_y;

           double term1_dy = -C1 * delta_ij * rk * r_pow2_inv;
           double term2_dy = C2 * ((delta_ik * rj + ri * delta_jk) * r_pow2_inv - 2 * ri * rj * rk * r_pow4_inv);
           double dU_ij_dyy = constant_factor * (term1_dy + term2_dy);
           
           dU_dy[i][j] = Vec2D(dU_ij_dyx, dU_ij_dyy);
       }
   }

   return dU_dy;
}


std::vector<Vec2D> conormalDerivativeKelvinKernel(double shearModulus, double poissonRatio, Vec2D x_minus_y, Vec2D normal_y) {
   // Lamé constants for 2D plane strain
   double mu = shearModulus;
   double lambda = (poissonRatio * 2.0 * mu) / (1.0 - 2.0 * poissonRatio);

   // Get the gradients of the Kelvin kernel (Somigliana tensor)
   std::vector<std::vector<Vec2D>> dU_dy = kelvinKernelGradient(shearModulus, poissonRatio, x_minus_y);

   std::vector<Vec2D> traction_kernel(2); // The resulting 2x2 traction kernel

   // For each column k of the original kernel (force direction)
   for (int k = 0; k < 2; ++k) {
       // This column corresponds to a displacement field u^(k)
       
       // Divergence of u^(k): Div(u^(k)) = ∂U_xk/∂yx + ∂U_yk/∂yy
       double div_uk = real(dU_dy[0][k]) + imag(dU_dy[1][k]);
       
       // Term 1: lambda * Div(u^(k)) * n
       Vec2D term1(lambda * div_uk * real(normal_y),
                   lambda * div_uk * imag(normal_y));

       // Term 2: mu * (∇u + ∇u^T)n
       double n_x = real(normal_y);
       double n_y = imag(normal_y);
       
       double du_xx = real(dU_dy[0][k]);
       double du_xy = imag(dU_dy[0][k]);
       double du_yx = real(dU_dy[1][k]);
       double du_yy = imag(dU_dy[1][k]);
       
       Vec2D strain_dot_n(
           (2 * du_xx) * n_x + (du_xy + du_yx) * n_y,
           (du_yx + du_xy) * n_x + (2 * du_yy) * n_y
       );
       Vec2D term2(mu * real(strain_dot_n), mu * imag(strain_dot_n));

       // The final column of the traction kernel is the sum of the two terms
       traction_kernel[k] = term1 + term2;
   }
   
   return traction_kernel;
}

double poissonRatio = 0.3; 
double E = 1.0;
double mu = E / (2.0 * (1.0 + poissonRatio));
static bool hasPrintedGlobal = false; 

Vec2D getU(Vec2D startingPoint, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int depth, int maxDepth, Vec2D& accumulatedSecond,
int& accumulatedNumber) { 
   vector<Vec2D> normals = getNormal(startingPoint);
   bool onBoundary = (normals.size() > 0);
   double theta = random(-M_PI, M_PI);
   bool flipped = false;
   if (!hasPrintedGlobal && normals.size() == 1 && dot(normals[0], Vec2D(cos(theta), sin(theta))) > 0) {
      Vec2D u{cos(theta), sin(theta)};
      // cout << "u" << real(u) << ", " << imag(u) << endl;
      theta += M_PI; // Flip the direction to sample in the hemisphere opposite to the normal
      flipped = true;
   } 
   else if (!hasPrintedGlobal && normals.size() == 2 && (dot(normals[0], Vec2D(cos(theta), sin(theta))) > 0 || 
      dot(normals[1], Vec2D(cos(theta), sin(theta))) > 0)) {
      Vec2D u{cos(theta), sin(theta)};
      // cout << "u" << real(u) << ", " << imag(u) << endl;
      if (dot(normals[0], Vec2D(cos(theta), sin(theta))) > 0 && 
            dot(normals[1], Vec2D(cos(theta), sin(theta))) > 0) {
         theta += M_PI; // Flip the direction to avoid being on the same side of both normals
         flipped = true;
      } else {
         // Adjust theta to ensure it works with both normals
         theta = atan2(-imag(normals[0] + normals[1]), -real(normals[0] + normals[1]));
         flipped = true;
      }
   }
   Vec2D v{cos(theta), sin(theta)};
   Vec2D n{0, 0};
   double p;
   double r = distancePolylines(startingPoint, boundaryNeumann).first; // Get the distance to the nearest point on the Neumann boundary
   if (onBoundary) {
      p = 1 / (2 * M_PI * r); // Hemisphere sampling on the boundary
   } else {
      p = 1 / (4 * M_PI * r); // Uniform distribution over the distance to the boundary
   }
   Vec2D intersectionPoint = intersectPolylines(startingPoint, v, std::numeric_limits<double>::infinity(), boundaryNeumann, n, onBoundary);
   // Define p as the probability of getting this r
  
// Ensure this prints only once in the entire program
   if (!hasPrintedGlobal && (isinf(real(intersectionPoint)) || isinf(imag(intersectionPoint)) || isnan(real(intersectionPoint)) || isnan(imag(intersectionPoint)))) {
      cout << "startingPoint: " << real(startingPoint) << ", " << imag(startingPoint) << endl;
      cout << "v: " << real(v) << ", " << imag(v) << endl;
      hasPrintedGlobal = true;
      cout << "flipped: " << flipped << endl;
   }
   Vec2D q = getNeumannValue(intersectionPoint, boundaryNeumann); // Ensure u is defined
   vector<Vec2D> kernel =  KKernel(real(intersectionPoint), imag(intersectionPoint), real(startingPoint), imag(startingPoint), mu, poissonRatio);
   Vec2D second = Vec2D( 2 / p * real(matrixVectorMultiply(kernel, q)), 2 / p * imag(matrixVectorMultiply(kernel, q)));
   accumulatedSecond += second;
   accumulatedNumber += 1;
   if (depth >= maxDepth) {
      return 0.5 * second; // Corrected division
   } else {
      Vec2D res = getU(intersectionPoint, boundaryNeumann, getNeumannValue, depth + 1, maxDepth, accumulatedSecond, accumulatedNumber);
      return 2.0 * second - res;
   }
}


Vec2D getUDirichlet(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, int depth, int maxDepth) { 
   vector<Vec2D> normals = getNormal(startingPoint);
   bool onBoundary = (normals.size() > 0);
   double theta = random(-M_PI, M_PI);
   bool flipped = false;
   Vec2D dir = Vec2D(cos(theta), sin(theta));
   Vec2D n{0, 0};
   if (normals.size() == 1) {
      double value1 = dot(normals[0], dir);
      if (value1 > 0) {
          theta += M_PI;
          flipped = true;
      }
   } 
   else if (normals.size() == 2) {
      double value1 = dot(normals[0], dir);
      double value2 = dot(normals[1], dir);

      if (value1 > 0 || value2 > 0) {
         if (value1 > 0 && value2 > 0) {
            theta += M_PI;
         } else {
            theta = atan2(-imag(normals[0] + normals[1]), -real(normals[0] + normals[1]));
         }
         flipped = true;
      }
   }
   else if (normals.size() == 3) {
      double value1 = dot(normals[0], dir);
      double value2 = dot(normals[1], dir);
      double value3 = dot(normals[2], dir);

      if (value1 > 0 || value2 > 0 || value3 > 0) {
         if (value1 > 0 && value2 > 0 && value3 > 0) {
            theta += M_PI;
         } else {
            theta = atan2(-imag(normals[0] + normals[1] + normals[2]), -real(normals[0] + normals[1] + normals[2]));
         }
         flipped = true;
      }
   }
   Vec2D v{cos(theta), sin(theta)};
   double p;
   double r = distancePolylines(startingPoint, boundaryDirichlet).first;
   if (onBoundary) {
      p = 1 / (2 * M_PI * r);
   } else {
      p = 1 / (4 * M_PI * r);
   }

   Vec2D intersectionPoint = intersectPolylines(startingPoint, v, std::numeric_limits<double>::infinity(), boundaryDirichlet, n, onBoundary);
  
   if (!hasPrintedGlobal && (isinf(real(intersectionPoint)) || isinf(imag(intersectionPoint)) || isnan(real(intersectionPoint)) || isnan(imag(intersectionPoint)))) {
      cout << "startingPoint: " << real(startingPoint) << ", " << imag(startingPoint) << endl;
      cout << "v: " << real(v) << ", " << imag(v) << endl;
      cout << "intersectionPoint: " << real(intersectionPoint) << ", " << imag(intersectionPoint) << endl;
      double value1 = dot(normals[0], v);
      cout << "normal size : " << normals.size() << endl;
      if (normals.size() == 2) { 
         double value2 = dot(normals[1], v);
         cout << "value1: " << value1 << ", value2: " << value2 << endl;
      } else {
         cout << "value1: " << value1 << endl;
      }
      hasPrintedGlobal = true;
      cout << "flipped: " << flipped << endl;
   }

   Vec2D u = getDirichletValue(intersectionPoint, boundaryDirichlet, displacedPoints); // Ensure u is defined
   vector<Vec2D> ns = getNormal(intersectionPoint);
   Vec2D normal = ns.size() > 0 ? ns[0] : Vec2D(0, 0); // Use the first normal if available
   if (!hasPrintedGlobal && ns.size() == 0) {
      cout << "No normal found at intersection point: " << real(intersectionPoint) << ", " << imag(intersectionPoint) << endl;
   }
   vector<Vec2D> kernel =  conormalDerivativeKelvinKernel(mu, poissonRatio, Vec2D(real(startingPoint) - real(intersectionPoint), imag(startingPoint) - imag(intersectionPoint)), normal);
   if (depth >= maxDepth) {
      return u;
   } else {
      Vec2D res = getUDirichlet(intersectionPoint, boundaryDirichlet, getDirichletValue, depth + 1, maxDepth);
      return 2.0 * u - res;
   }
}

int generateRandomZeroOrOne() {
   static std::random_device rd;
   static std::mt19937 gen(rd());
   static std::uniform_real_distribution<double> dist(0.0, 1.0);

   double randomValue = dist(gen);
   return (randomValue < 2.0 / 3.0) ? 0 : 1;
}

Vec2D getMixedConditionResult(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {
   Vec2D nextPoint = sampleRectangleBoundary();
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
      if (depth >= maxDepth) {
         return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints);
      }
      else {
         int choice = generateRandomZeroOrOne();
         if (choice == 0) {
            Vec2D contribution = getMixedConditionResult(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
            Vec2D u = 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints) - (3 * contribution);
            return u;
         }
         else {
            Vec2D contribution = getMixedConditionResult(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
            Vec2D u = 3 * contribution + 4 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints);
            return u;
         }
      }
   }
   else {
      if (depth >= maxDepth) {
         return 2 * getNeumannValue(startingPoint, boundaryNeumann);
      }
      else {
         Vec2D contribution = getMixedConditionResult(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         Vec2D u = 2 * getNeumannValue(startingPoint, boundaryNeumann) - contribution;
         return u;
      }
   }
}

Vec2D getSourceTerm(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
   if (isOnDirichlet(point, boundaryDirichlet, boundaryNeumann)) {
      return getDirichletValue(point, boundaryDirichlet, displacedPoints);
   } else if (isOnNeumann(point, boundaryDirichlet, boundaryNeumann)) {
      return getNeumannValue(point, boundaryNeumann);
   }
}

Vec2D getMixedConditionResultKernel(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {

   Vec2D sample_contribution = Vec2D(0.0, 0.0);
   Vec2D currentPoint = sampleRectangleBoundary();
   vector<Vec2D> currentPointNormals = getNormal(currentPoint);
   Vec2D currentPointNormal = currentPointNormals.size() > 0 ? currentPointNormals[0] : Vec2D(0, 0); 
   vector<Vec2D> path_weight = isOnDirichlet(currentPoint, boundaryDirichlet, boundaryNeumann) ? kelvinKernel(mu, poissonRatio, startingPoint - currentPoint) : conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - currentPoint, currentPointNormal);

   for (int depth = 0; depth < maxDepth; ++depth) {

      if (depth == maxDepth - 1) {
         path_weight = 1/2 * path_weight; 
      }

      vector<Vec2D> kernel_for_next_weight = {Vec2D(1, 1), Vec2D(1, 1)};
      sample_contribution += matrixVectorMultiply(path_weight, getSourceTerm(currentPoint, boundaryDirichlet, boundaryNeumann));
      Vec2D nextPoint = sampleRectangleBoundary();
      vector<Vec2D> normals = getNormal(nextPoint);
      Vec2D normal = normals.size() > 0 ? normals[0] : Vec2D(0, 0); 
      if (isOnDirichlet(currentPoint, boundaryDirichlet, boundaryNeumann)) {
         int choice = generateRandomZeroOrOne();

         if (isOnDirichlet(nextPoint, boundaryDirichlet, boundaryNeumann)) {
            kernel_for_next_weight = - 8 * kelvinKernel(mu, poissonRatio, currentPoint - nextPoint);
         } else {
            kernel_for_next_weight = 8 * conormalDerivativeKelvinKernel(mu, poissonRatio, currentPoint - nextPoint, normal);
         }
      } else {
         kernel_for_next_weight = 2 * conormalDerivativeKelvinKernel(mu, poissonRatio, currentPoint - nextPoint, normal);
      }

      path_weight = matrixMultiply(path_weight, kernel_for_next_weight);
      currentPoint = nextPoint;
   }

   return sample_contribution + matrixVectorMultiply(path_weight, getSourceTerm(currentPoint, boundaryDirichlet, boundaryNeumann));
}

Vec2D getMixedConditionResultKernel2(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {

   Vec2D sample_contribution = Vec2D(0.0, 0.0);
   Vec2D currentPoint = sampleRectangleBoundary();
   vector<Vec2D> currentPointNormals = getNormal(currentPoint);
   Vec2D currentPointNormal = currentPointNormals.size() > 0 ? currentPointNormals[0] : Vec2D(0, 0); 
   vector<Vec2D> path_weight = isOnDirichlet(currentPoint, boundaryDirichlet, boundaryNeumann) ? kelvinKernel(mu, poissonRatio, startingPoint - currentPoint) : conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - currentPoint, currentPointNormal);

   for (int depth = 0; depth < maxDepth; ++depth) {

      if (depth == maxDepth - 1) {
         path_weight = 1/2 * path_weight; 
      }

      vector<Vec2D> kernel_for_next_weight = {Vec2D(1, 1), Vec2D(1, 1)};
      sample_contribution += matrixVectorMultiply(path_weight, getSourceTerm(currentPoint, boundaryDirichlet, boundaryNeumann));
      Vec2D nextPoint = sampleRectangleBoundary();
      vector<Vec2D> normals = getNormal(nextPoint);
      Vec2D normal = normals.size() > 0 ? normals[0] : Vec2D(0, 0); 
      if (isOnDirichlet(currentPoint, boundaryDirichlet, boundaryNeumann)) {
         int choice = generateRandomZeroOrOne();

         if (isOnDirichlet(nextPoint, boundaryDirichlet, boundaryNeumann)) {
            if (choice == 0) {
               kernel_for_next_weight = - 24 * kelvinKernel(mu, poissonRatio, currentPoint - nextPoint);
            } else {
               kernel_for_next_weight = vector<Vec2D>{Vec2D(3, 0), Vec2D(0, 3)};
            }
         } else {
            if (choice == 0) {
               kernel_for_next_weight = - 24 * conormalDerivativeKelvinKernel(mu, poissonRatio, currentPoint - nextPoint, normal);
            } else {
               kernel_for_next_weight = vector<Vec2D>{Vec2D(3, 0), Vec2D(0, 3)};
            }
         }
      } else {
         kernel_for_next_weight = 2 * conormalDerivativeKelvinKernel(mu, poissonRatio, currentPoint - nextPoint, normal);
      }

      path_weight = matrixMultiply(path_weight, kernel_for_next_weight);
      currentPoint = nextPoint;
   }

   return sample_contribution + matrixVectorMultiply(path_weight, getSourceTerm(currentPoint, boundaryDirichlet, boundaryNeumann));
}

Vec2D getMixedConditionResultKernel3(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {

   Vec2D nextPoint = sampleRectangleBoundary();
   vector<Vec2D> nextPointNormals = getNormal(nextPoint);
   Vec2D nextPointNormal = nextPointNormals.size() > 0 ? nextPointNormals[0] : Vec2D(0, 0); 

   if (depth == maxDepth) {
      if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
         return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints);
      } else {
         return getNeumannValue(startingPoint, boundaryNeumann);
      }
   }
   else {
      if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
         Vec2D contribution = getMixedConditionResultKernel3(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         int choice = generateRandomZeroOrOne();

         if (choice == 0) {
            return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints) + 12 * contribution;// could be adding 
         } else {
            return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints) - 3/2 * contribution; 
         }

      } else {
         Vec2D contribution = getMixedConditionResultKernel3(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         return 2 * getNeumannValue(nextPoint, boundaryNeumann) - contribution;
      }
   }
}

int num_resampling_candidates = 4;

Vec2D getMixedConditionResultKernel4(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {
   Vec2D nextPoint = sampleRectangleBoundary();
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
      auto [y_dirichlet, inv_pdf_d] = sample_boundary_ris(
         gen,
         [&](const Vec2D& y) {
            std::vector<Vec2D> kelvinMatrix = kelvinKernel(mu, poissonRatio, startingPoint - y);
            double frobenius_norm = 0.0;
            for (const Vec2D& val : kelvinMatrix) {
               frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
            }
             return abs(frobenius_norm);
         },
         rectangleBoundarySampler,
         num_resampling_candidates
     );
      if (!isnan(real(y_dirichlet)) && !isnan(imag(y_dirichlet))) {
         nextPoint = y_dirichlet;
      }
   } else {
      auto [y_neumann, inv_pdf_n] = sample_boundary_ris(
         gen,
         [&](const Vec2D& y) {
            vector<Vec2D> yNormals = getNormal(y);
            Vec2D yNormal = yNormals.size() > 0 ? yNormals[0] : Vec2D(0, 0); 
            std::vector<Vec2D> cornormalMatrix = conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - y, yNormal);
            double frobenius_norm = 0.0;
            for (const Vec2D& val : cornormalMatrix) {
               frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
            }
             return abs(frobenius_norm);
         },
         rectangleBoundarySampler,
         num_resampling_candidates
      );
      if (!isnan(real(y_neumann)) && !isnan(imag(y_neumann))) {
         nextPoint = y_neumann;
      }
   }

   vector<Vec2D> nextPointNormals = getNormal(nextPoint);
   Vec2D nextPointNormal = nextPointNormals.size() > 0 ? nextPointNormals[0] : Vec2D(0, 0); 

   if (depth == maxDepth) {
      if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
         return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints);
      } else {
         return getNeumannValue(startingPoint, boundaryNeumann);
      }
   }
   else {
      if (isOnDirichlet(nextPoint, boundaryDirichlet, boundaryNeumann)) {
         Vec2D contribution = getMixedConditionResultKernel4(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         int choice = generateRandomZeroOrOne();
         return 2 * getDirichletValue(nextPoint, boundaryDirichlet, displacedPoints) - contribution;

         if (choice == 0) {
            return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints) + 12 * contribution; // could be adding 
         } else {
            return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints) - 3/2 * contribution; 
         }

      } else {
         Vec2D contribution = getMixedConditionResultKernel4(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         return 2 * getNeumannValue(nextPoint, boundaryNeumann) - contribution;
      }
   }
}

Vec2D getMixedConditionResultKernel5(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {
   Vec2D nextPoint = sampleRectangleBoundary();
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
      auto [y_dirichlet, inv_pdf_d] = sample_boundary_ris(
         gen,
         [&](const Vec2D& y) {
            std::vector<Vec2D> kelvinMatrix = kelvinKernel(mu, poissonRatio, startingPoint - y);
            double frobenius_norm = 0.0;
            for (const Vec2D& val : kelvinMatrix) {
               frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
            }
             return abs(frobenius_norm);
         },
         rectangleBoundarySampler,
         num_resampling_candidates
     );
      if (!isnan(real(y_dirichlet)) && !isnan(imag(y_dirichlet))) {
         nextPoint = y_dirichlet;
      } 
   } else {
      auto [y_neumann, inv_pdf_n] = sample_boundary_ris(
         gen,
         [&](const Vec2D& y) {
            vector<Vec2D> yNormals = getNormal(y);
            Vec2D yNormal = yNormals.size() > 0 ? yNormals[0] : Vec2D(0, 0); 
            std::vector<Vec2D> cornormalMatrix = conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - y, yNormal);
            double frobenius_norm = 0.0;
            for (const Vec2D& val : cornormalMatrix) {
               frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
            }
             return abs(frobenius_norm);
         },
         rectangleBoundarySampler,
         num_resampling_candidates
      );
      if (!isnan(real(y_neumann)) && !isnan(imag(y_neumann))) {
         nextPoint = y_neumann;
      }
   }

   vector<Vec2D> nextPointNormals = getNormal(nextPoint);
   Vec2D nextPointNormal = nextPointNormals.size() > 0 ? nextPointNormals[0] : Vec2D(0, 0); 

   if (depth == maxDepth) {
      if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
         return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints);
      } else {
         return getNeumannValue(startingPoint, boundaryNeumann);
      }
   }
   else {
      if (isOnDirichlet(nextPoint, boundaryDirichlet, boundaryNeumann)) {
         Vec2D contribution = getMixedConditionResultKernel5(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         int choice = generateRandomZeroOrOne();
   
         if (choice == 0) {
            return 4 * getDirichletValue(nextPoint, boundaryDirichlet, displacedPoints) - 24 * matrixVectorMultiply(kelvinKernel(mu, poissonRatio, startingPoint - nextPoint), contribution); 
         } else {
            return 4 * getDirichletValue(nextPoint, boundaryDirichlet, displacedPoints) + 3 * contribution;
         }

      } else {
         Vec2D contribution = getMixedConditionResultKernel5(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
         return 2 * getNeumannValue(nextPoint, boundaryNeumann) - 8 * matrixVectorMultiply(conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - nextPoint, nextPointNormal), contribution); // why minus? 
      }
   }
}

bool isOnCorner(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
   const double tolerance = 1e-4;
   int boundaryCount = 0;

   // Check Dirichlet boundaries
   for (const auto& polyline : boundaryDirichlet) {
      for (size_t i = 0; i < polyline.size() - 1; ++i) {
         if (isPointOnLine(point, polyline[i], polyline[i + 1], tolerance)) {
            boundaryCount++;
            if (boundaryCount > 1) return true;
         }
      }
   }

   // Check Neumann boundaries
   for (const auto& polyline : boundaryNeumann) {
      for (size_t i = 0; i < polyline.size() - 1; ++i) {
         if (isPointOnLine(point, polyline[i], polyline[i + 1], tolerance)) {
            boundaryCount++;
            if (boundaryCount > 1) return true;
         }
      }
   }

   return false;
}


Vec2D fredholmEquationKnown(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann){
   if (isOnDirichlet(point, boundaryDirichlet, boundaryNeumann)){
      return 4 * getDirichletValue(point, boundaryDirichlet, displacedPoints);
   }
   else {
      double integralFreeTerm = 0.5;
      if (isOnCorner(point, boundaryDirichlet, boundaryNeumann)) {
         integralFreeTerm = 0.25;
      }
      return 1 / integralFreeTerm * getNeumannValue(point, boundaryNeumann);
   }
}



// CORRECTED Traction Kernel
// vector<Vec2D> computeTractionKernel2D(Vec2D x, Vec2D y, Vec2D normal_y, double shearModulus, double poissonRatio) {
//    // The vector r from the source 'y' to the field point 'x'
//    Vec2D r_vec(real(x) - real(y), imag(x) - imag(y));

//    double r_len_sq = std::norm(r_vec); 
//    if (r_len_sq < 1e-12) {
//        return { Vec2D(0,0), Vec2D(0,0) };
//    }
//    double r_len = std::sqrt(r_len_sq);

//    double dr_dn = (real(r_vec) * real(normal_y) + imag(r_vec) * imag(normal_y)) / r_len;

//    double constant_factor = -1.0 / (4.0 * M_PI * (1.0 - poissonRatio) * r_len);

//    double r_i_n_j_minus_r_j_n_i = real(r_vec) * imag(normal_y) - imag(r_vec) * real(normal_y);

//    double term1 = (1.0 - 2.0 * poissonRatio);
   
//    // Calculate all four components cleanly
//    double T11 = constant_factor * (dr_dn * (term1 + 2.0 * real(r_vec) * real(r_vec) / r_len_sq) + term1 * imag(r_vec) * imag(normal_y));
//    double T12 = constant_factor * (dr_dn * (2.0 * real(r_vec) * imag(r_vec) / r_len_sq) - term1 * r_i_n_j_minus_r_j_n_i);
//    double T21 = constant_factor * (dr_dn * (2.0 * real(r_vec) * imag(r_vec) / r_len_sq) + term1 * r_i_n_j_minus_r_j_n_i);
//    double T22 = constant_factor * (dr_dn * (term1 + 2.0 * imag(r_vec) * imag(r_vec) / r_len_sq) + term1 * real(r_vec) * real(normal_y));
   
//    return { Vec2D(T11, T12), Vec2D(T21, T22) };
// }

// vector<Vec2D> computeTractionKernel2D(Vec2D x, Vec2D y, Vec2D normal_y, double shearModulus, double poissonRatio) {
//    Vec2D r_vec(real(x) - real(y), imag(x) - imag(y));

//    double r_len_sq = std::norm(r_vec);
//    if (r_len_sq < 1e-12) {
//        return { Vec2D(0,0), Vec2D(0,0) };
//    }
//    double r_len = std::sqrt(r_len_sq);

//    double r_x = real(r_vec);
//    double r_y = imag(r_vec);
//    double n_x = real(normal_y);
//    double n_y = imag(normal_y);

//    double r_dot_n = r_x * n_x + r_y * n_y;

//    double constant_factor = (1.0 - 2.0 * poissonRatio) / (4.0 * M_PI * (1.0 - poissonRatio) * r_len);
   
//    // Calculate components based on the standard BEM formula T_ik = ...
//    double T11 = constant_factor * (n_x + (2.0 / (1.0 - 2.0 * poissonRatio)) * (r_x / r_len) * (r_x / r_len) * r_dot_n / r_len);
//    double T12 = constant_factor * (n_y - (2.0 / (1.0 - 2.0 * poissonRatio)) * (r_x / r_len) * (r_y / r_len) * r_dot_n / r_len);
//    double T21 = constant_factor * (n_x - (2.0 / (1.0 - 2.0 * poissonRatio)) * (r_y / r_len) * (r_x / r_len) * r_dot_n / r_len);
//    double T22 = constant_factor * (n_y + (2.0 / (1.0 - 2.0 * poissonRatio)) * (r_y / r_len) * (r_y / r_len) * r_dot_n / r_len);

//    return { Vec2D(T11, T12), Vec2D(T21, T22) };
// }


// CORRECTED and VERIFIED Traction Kernel T(x,y)
vector<Vec2D> computeTractionKernel2D(Vec2D x, Vec2D y, Vec2D normal_y, double shearModulus, double poissonRatio) {
   // The vector r from the source 'y' to the field point 'x'
   Vec2D r_vec(real(x) - real(y), imag(x) - imag(y));

   double r_len_sq = std::norm(r_vec);
   if (r_len_sq < 1e-12) {
       return { Vec2D(0,0), Vec2D(0,0) };
   }
   double r_len = std::sqrt(r_len_sq);

   // Common terms needed for the formula
   double r_x_norm = real(r_vec) / r_len; // This is r,1 in the formula
   double r_y_norm = imag(r_vec) / r_len; // This is r,2 in the formula
   double n_x = real(normal_y);
   double n_y = imag(normal_y);

   // The dot product of r_vec and the normal, divided by r_len: (r . n) / r
   double dr_dn = r_x_norm * n_x + r_y_norm * n_y;

   double const1 = -1.0 / (4.0 * M_PI * (1.0 - poissonRatio));
   double const2 = 1.0 - 2.0 * poissonRatio;

   // The cross term (r,1*n,2 - r,2*n,1)
   double cross_term = r_x_norm * n_y - r_y_norm * n_x;

   // Calculate each component cleanly from the standard formula
   double T11 = const1 / r_len * (dr_dn * (const2 + 2.0 * r_x_norm * r_x_norm) - const2 * cross_term);
   double T12 = const1 / r_len * (dr_dn * (2.0 * r_x_norm * r_y_norm) + const2 * cross_term);
   double T21 = const1 / r_len * (dr_dn * (2.0 * r_x_norm * r_y_norm) - const2 * cross_term);
   double T22 = const1 / r_len * (dr_dn * (const2 + 2.0 * r_y_norm * r_y_norm) - const2 * cross_term);

   return { Vec2D(T11, T12), Vec2D(T21, T22) };
}


Polyline fredholmEquationUnknown(Vec2D startingPoint, Vec2D nextPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, Vec2D nextPointNormal, double invPdf) {
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)){
      return  - invPdf * 4 * kelvinKernel(mu, poissonRatio, startingPoint - nextPoint);
   }
   else {
      return - invPdf * 2 * computeTractionKernel2D(startingPoint, nextPoint, nextPointNormal, mu, poissonRatio);
   }
}

Polyline solutionUnknown(Vec2D startingPoint, Vec2D nextPoint, double invPdf, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
   // if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
   //    return {Vec2D(0,0), Vec2D(0,0)};
   // }
   // else {
   //    return invPdf * kelvinKernel(mu, poissonRatio, startingPoint - nextPoint);
   // }
   return invPdf * kelvinKernel(mu, poissonRatio, startingPoint - nextPoint);
}

pair<Vec2D, double> importanceSample(Vec2D startingPoint){
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
      auto [y_dirichlet, inv_pdf_d] = sample_boundary_ris(
         gen,
         [&](const Vec2D& y) {
            std::vector<Vec2D> kelvinMatrix = kelvinKernel(mu, poissonRatio, startingPoint - y);
            double frobenius_norm = 0.0;
            for (const Vec2D& val : kelvinMatrix) {
               frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
            }
             return abs(frobenius_norm);
         },
         rectangleBoundarySampler,
         num_resampling_candidates
     );
      if (!isnan(real(y_dirichlet)) && !isnan(imag(y_dirichlet))) {
         return make_pair(y_dirichlet, inv_pdf_d);
      } 
   } else {
      auto [y_neumann, inv_pdf_n] = sample_boundary_ris(
         gen,
         [&](const Vec2D& y) {
            vector<Vec2D> yNormals = getNormal(y);
            Vec2D yNormal = yNormals.size() > 0 ? yNormals[0] : Vec2D(0, 0); 
            std::vector<Vec2D> cornormalMatrix = computeTractionKernel2D(startingPoint, y, yNormal, mu, poissonRatio);
            double frobenius_norm = 0.0;
            for (const Vec2D& val : cornormalMatrix) {
               frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
            }
             return abs(frobenius_norm);
         },
         rectangleBoundarySampler,
         num_resampling_candidates
      );
      if (!isnan(real(y_neumann)) && !isnan(imag(y_neumann))) {
         return make_pair(y_neumann, inv_pdf_n);
      }
   }

   Vec2D nan = numeric_limits<double>::quiet_NaN();
   return make_pair(nan, 0);
}

Vec2D getMixedConditionResultKernel6(Vec2D queryPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> geNeumannValue, int depth, int maxDepth) {
   Vec2D result = Vec2D{0, 0};
   pair<Vec2D, double> pair = importanceSample(queryPoint);
   Vec2D currentPoint = pair.first;
   double invPdf = pair.second;
   vector<Vec2D> currentPointNormals = getNormal(currentPoint);
   Vec2D currentPointNormal;
   if (currentPointNormals.size() > 0) {
      currentPointNormal = currentPointNormals[0];
   }
   else {
      cerr << "currentPointNormals size is 0" << endl;
      return Vec2D{0, 0};
   } 
   Vec2D pathWeight = invPdf * fredholmEquationKnown(currentPoint, boundaryDirichlet, boundaryNeumann);

   for (int i = 0; i < maxDepth; i++){
      if (i == maxDepth - 1) { pathWeight *= 1/2;}
      result += matrixVectorMultiply(solutionUnknown(queryPoint, currentPoint, 1.0, boundaryDirichlet, boundaryNeumann), pathWeight);
      Vec2D previousPoint = currentPoint;
      pair = importanceSample(previousPoint); 
      currentPoint = pair.first;
      currentPointNormals = getNormal(currentPoint);
      if (currentPointNormals.size() > 0) {
         currentPointNormal = currentPointNormals[0];
      } 
      else {
         cerr << "currentPointNormals size is 0" << endl;
         return Vec2D{0, 0};
      }

      double invPdf = pair.second;
      pathWeight = matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, currentPointNormal, invPdf), pathWeight);
   }
   
   return result;
}

// Vec2D getMixedConditionResultKernel7(Vec2D startingPoint, vector<Polyline> boundaryDirichlet,
//    vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,  
//    function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {

//    // These would be parameters for your solver
//    const float phi = 1.0f; // 1 for interior, -1 for exterior problems
//    const float k = 4.0f;   // A constant for the Dirichlet estimator [cite: 413]
//    const float p_k = 0.66f; // Probability for the Dirichlet estimator [cite: 413]

//    Vec2D finalResult = Vec2D{0, 0};

//    // --- INITIALIZATION ---
//    // The random walk starts on the boundary. We sample a starting point y_0.
//    // NOTE: The walk does NOT start from the interior 'startingPoint'. The walk estimates mu
//    // on the boundary, and 'startingPoint' is used in the last-step connection.
//    pair<Vec2D, double> firstStep = importanceSample(startingPoint); // You need a function to sample the boundary
//    Vec2D currentPoint = firstStep.first;
//    double invPdf_0 = firstStep.second;

//    // Initialize the pathWeight with the correct source term based on the type of the first point.
//    // This is the "baton" for the start of the relay race.
//    BoundaryType initialType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);
//    Vec2D pathWeight;
//    if (initialType == BoundaryType::Neumann) {
//       pathWeight = invPdf_0 * (2.0f * phi * getNeumannValue(currentPoint, boundaryNeumann));
//    } else { // DIRICHLET
//       pathWeight = invPdf_0 * (k * getDirichletValue(currentPoint, boundaryDirichlet, displacedPoints));
//    }

//    // --- THE RANDOM WALK ON THE BOUNDARY ---
//    Vec2D normal_prev;
//    for (int i = 0; i < maxDepth; i++) {
//    // --- 1. ACCUMULATE RESULT (The "Last-Step Connection") ---
//    // Connect the walker's current position back to the original query point.
//       finalResult += matrixVectorMultiply(solutionUnknown(startingPoint, currentPoint, 1.0), pathWeight);

//       // --- 2. RUSSIAN ROULETTE (Correct Path Termination) ---
//       const double absorptionProb = 0.2;
//       if (rand() / (RAND_MAX + 1.0) < absorptionProb) {
//          break; // Walk terminates
//       }
//       pathWeight *= (1.0f / (1.0f - absorptionProb));


//       // --- 3. UPDATE PATH WEIGHT (Conditional Logic) ---
//       Vec2D previousPoint = currentPoint;
//       BoundaryType currentType = getBoundaryTypeAtPoint(previousPoint, boundaryDirichlet, boundaryNeumann);
//       if (currentType == BoundaryType::Neumann) {
//          // Apply the update rule from your first image
//          pair<Vec2D, double> nextStep = importanceSample(previousPoint);
//          currentPoint = nextStep.first;
//          double invPdf = nextStep.second;
//          vector<Vec2D> normal_prev_list = getNormal(previousPoint);
//          if (normal_prev_list.size() > 0) {
//             normal_prev = normal_prev_list[0];
//          }
//          else {
//             cerr << "normal_prev size is 0" << endl;
//             return Vec2D{0, 0};
//          }

//          // Your `fredholmEquationUnknown` should be the kernel -2*phi*(∂G/∂n_x)
//          pathWeight = matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf), pathWeight);

//       } else { // currentType == DIRICHLET
//          std::random_device rd;
//          std::mt19937 gen(rd());
//          std::uniform_real_distribution<double> dist(0.0, 1.0);
//          if (dist(gen) < p_k) {
//             pair<Vec2D, double> nextStep = importanceSample(previousPoint);
//             currentPoint = nextStep.first;
//             double invPdf = nextStep.second;

//             // Here, `fredholmEquationUnknown` should be the kernel -k*G(x,y)
//             Polyline weightUpdate = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf);
//             pathWeight = matrixVectorMultiply((1.0f / p_k) * weightUpdate, pathWeight);

//          } else {
//             pathWeight *= (1.0f / (1.0f - p_k));
//          }
//       }
//    }
//    return finalResult;
// }

Vec2D getMixedConditionResultKernel8(Vec2D startingPoint, vector<Polyline> boundaryDirichlet,
   vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,
   function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {

   const float phi = 1.0f;
   const float k = 4.0f;
   const float p_k = 1.0f / 3.0f; 

   Vec2D finalResult_mu = Vec2D(0, 0); // Renamed to clarify it's an estimate for μ

   // --- INITIALIZATION ---
   Vec2D currentPoint = startingPoint;
   // The path starts with a weight of 1.0 (identity).
   Vec2D pathWeight = Vec2D(1.0, 1.0); // Assuming component-wise multiplication

   thread_local std::mt19937 rng(std::random_device{}());
   std::uniform_real_distribution<double> uni01(0.0, 1.0);

   // --- THE RANDOM WALK ON THE BOUNDARY ---
   for (int i = 0; i < maxDepth; i++) {
       // --- 1. ACCUMULATE EMISSION AT CURRENT POINT ---
       // Add the contribution from the current point, weighted by the path so far.
       BoundaryType currentType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);
       Vec2D emission;
       if (currentType == BoundaryType::Neumann) {
           emission = 2.0f * phi * getNeumannValue(currentPoint, boundaryNeumann);
       } else { // DIRICHLET
           emission = k * getDirichletValue(currentPoint, boundaryDirichlet, displacedPoints);
       }
       finalResult_mu += pathWeight * emission; // Assuming component-wise product

       // --- 2. PREPARE FOR NEXT STEP (UPDATE PATH WEIGHT) ---
       Vec2D previousPoint = currentPoint;
       
       // This logic calculates the kernel for the next bounce and updates the path weight.
       // It should stay the same.
       if (currentType == BoundaryType::Neumann) {
         pair<Vec2D, double> nextStep = importanceSample(previousPoint);
         currentPoint = nextStep.first;
         double invPdf = nextStep.second;
         Vec2D normal_prev = getNormal(previousPoint)[0];

         pathWeight = matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf), pathWeight);
     
      } else { // DIRICHLET
            if (uni01(rng)  < p_k) {
               pair<Vec2D, double> nextStep = importanceSample(previousPoint);
               currentPoint = nextStep.first;
               double invPdf = nextStep.second;
               Vec2D normal_prev = getNormal(previousPoint)[0];

               Polyline weightUpdate = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf);
               pathWeight = matrixVectorMultiply(weightUpdate, pathWeight);
            } else {
               pathWeight *= (1.0f / (1.0f - p_k));
               currentPoint = previousPoint;
            }
      }
   }

   return finalResult_mu;
}

Vec2D getMixedConditionResultKernel7(Vec2D startingPoint, vector<Polyline> boundaryDirichlet,
   vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,
   function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {

    const float phi = 1.0f;
    const float k = 4.0f;
    const float p_k = 1/3;

    Vec2D finalResult = Vec2D(0, 0);

    // --- INITIALIZATION ---
    pair<Vec2D, double> firstStep = rectangleBoundarySampler();
    Vec2D currentPoint = firstStep.first;
    double invPdf_0 = firstStep.second;

   BoundaryType initialType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);
    Vec2D pathWeight;
    if (initialType == BoundaryType::Neumann) {
       pathWeight = invPdf_0 * (2.0f * phi * getNeumannValue(currentPoint, boundaryNeumann));
    } else { // DIRICHLET
       pathWeight = invPdf_0 * (k * getDirichletValue(currentPoint, boundaryDirichlet, displacedPoints));
    }

   thread_local std::mt19937 rng( std::random_device{}() );
   std::uniform_real_distribution<double> uni01(0.0, 1.0);
    // --- THE RANDOM WALK ON THE BOUNDARY ---
    for (int i = 0; i < maxDepth; i++) {
        // --- 1. ACCUMULATE RESULT ---
        // The last-step kernel should not be scaled by the PDF.
        finalResult += matrixVectorMultiply(solutionUnknown(startingPoint, currentPoint, 1.0, boundaryDirichlet, boundaryNeumann), pathWeight);

      // --- 2. RUSSIAN ROULETTE ---
        const double absorptionProb = 0.2;
        if (uni01(rng) < absorptionProb) {
            break;
        }
        pathWeight *= (1.0f / (1.0f - absorptionProb));

        // --- 3. UPDATE PATH WEIGHT ---
        Vec2D previousPoint = currentPoint;
        BoundaryType currentType = getBoundaryTypeAtPoint(previousPoint, boundaryDirichlet, boundaryNeumann);

        if (currentType == BoundaryType::Neumann) {
            pair<Vec2D, double> nextStep = importanceSample(previousPoint);
            currentPoint = nextStep.first;
            double invPdf = nextStep.second;
            Vec2D normal_prev = getNormal(previousPoint)[0];

            pathWeight = matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf), pathWeight);
        
        } else { // DIRICHLET
            if (uni01(rng)  < p_k) {
                pair<Vec2D, double> nextStep = importanceSample(previousPoint);
                currentPoint = nextStep.first;
                double invPdf = nextStep.second;
                Vec2D normal_prev = getNormal(previousPoint)[0];

                Polyline weightUpdate = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf);
                pathWeight = matrixVectorMultiply((1.0f / p_k) * weightUpdate, pathWeight);
            } else {
                pathWeight *= (1.0f / (1.0f - p_k));
                currentPoint = previousPoint;
            }
        }
    }

    return finalResult;
}

Vec2D getMixedConditionResultKernel10(Vec2D startingPoint, vector<Polyline> boundaryDirichlet,vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,
   function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {
   const float phi = 1.0f;
   const float k = 4.0f;
   const float p_k = 1/3;
   Vec2D finalResult = Vec2D(0, 0);
   
   pair<Vec2D, double> firstStep = importanceSample(startingPoint);
   Vec2D currentPoint = firstStep.first;
   double invPdf_0 = firstStep.second;

   BoundaryType initialType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);

   Vec2D pathWeight;
   
   if (initialType == BoundaryType::Neumann) {
   
      pathWeight = invPdf_0 * (2.0f * phi * getNeumannValue(currentPoint, boundaryNeumann));
   
   } else { // DIRICHLET
   
      pathWeight = invPdf_0 * (k * getDirichletValue(currentPoint, boundaryDirichlet, displacedPoints));
   
   }
   
   
   
   thread_local std::mt19937 rng( std::random_device{}() );
   
   std::uniform_real_distribution<double> uni01(0.0, 1.0);
   
   // --- THE RANDOM WALK ON THE BOUNDARY ---
   
   for (int i = 0; i < maxDepth; i++) {
   
      // --- 1. ACCUMULATE RESULT ---
      
      // The last-step kernel should not be scaled by the PDF.
   
      finalResult += matrixVectorMultiply(solutionUnknown(startingPoint, currentPoint, 1.0, boundaryDirichlet, boundaryNeumann), pathWeight);
   
   
   
      // --- 2. RUSSIAN ROULETTE ---
      
      const double absorptionProb = 0.2;
      
      if (uni01(rng) < absorptionProb) {
      
       break;
      
      }
   
      pathWeight *= (1.0f / (1.0f - absorptionProb));
   
   
      
      // --- 3. UPDATE PATH WEIGHT ---
      
      Vec2D previousPoint = currentPoint;
      
      BoundaryType currentType = getBoundaryTypeAtPoint(previousPoint, boundaryDirichlet, boundaryNeumann);
      
      
      
      if (currentType == BoundaryType::Neumann) {
      
      pair<Vec2D, double> nextStep = importanceSample(previousPoint);
      
      currentPoint = nextStep.first;
      
      double invPdf = nextStep.second;
      
      Vec2D normal_prev = getNormal(previousPoint)[0];
      
      
      
      pathWeight = matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf), pathWeight);
      
      
      } else { // DIRICHLET
      
      if (uni01(rng) < p_k) {
      
      pair<Vec2D, double> nextStep = importanceSample(previousPoint);
      
      currentPoint = nextStep.first;
      
      double invPdf = nextStep.second;
      
      Vec2D normal_prev = getNormal(previousPoint)[0];
      
      
      
      Polyline weightUpdate = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf);
      
      pathWeight = matrixVectorMultiply((1.0f / p_k) * weightUpdate, pathWeight);
      
      } else {
      
      pathWeight *= (1.0f / (1.0f - p_k));
      
      currentPoint = previousPoint;
      
      }
   
    }
   
   }
   
   
   
   return finalResult;
   
   }

Vec2D getMixedConditionResultKernel9(Vec2D startingPoint, vector<Polyline> boundaryDirichlet,
   vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,
   function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {

    const float phi = 1.0f;
    const float k = 4.0f;
    const float p_k = 1/3;

    Vec2D finalResult = Vec2D(0, 0);

    pair<Vec2D, double> firstStep = importanceSample(startingPoint);
    Vec2D currentPoint = firstStep.first;
    double invPdf_0 = firstStep.second;

   BoundaryType initialType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);
    Vec2D pathWeight;
    if (initialType == BoundaryType::Neumann) {
       pathWeight = invPdf_0 * (2.0f * phi * getNeumannValue(currentPoint, boundaryNeumann));
    } else { // DIRICHLET
       pathWeight = invPdf_0 * (k * getDirichletValue(currentPoint, boundaryDirichlet, displacedPoints));
    }

   thread_local std::mt19937 rng( std::random_device{}() );
   std::uniform_real_distribution<double> uni01(0.0, 1.0);
    // --- THE RANDOM WALK ON THE BOUNDARY ---
    for (int i = 0; i < maxDepth; i++) {
        // --- 1. ACCUMULATE RESULT ---
        // The last-step kernel should not be scaled by the PDF.
        finalResult += matrixVectorMultiply(solutionUnknown(startingPoint, currentPoint, 1.0, boundaryDirichlet, boundaryNeumann), pathWeight);

      // --- 2. RUSSIAN ROULETTE ---
        const double absorptionProb = 0.2;
        if (uni01(rng) < absorptionProb) {
            break;
        }
        pathWeight *= (1.0f / (1.0f - absorptionProb));

        // --- 3. UPDATE PATH WEIGHT ---
        Vec2D previousPoint = currentPoint;
        BoundaryType currentType = getBoundaryTypeAtPoint(previousPoint, boundaryDirichlet, boundaryNeumann);

        if (currentType == BoundaryType::Neumann) {
            pair<Vec2D, double> nextStep = importanceSample(previousPoint);
            currentPoint = nextStep.first;
            double invPdf = nextStep.second;
            Vec2D normal_prev = getNormal(previousPoint)[0];

            pathWeight = invPdf * matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf), pathWeight);
        
        } else { // DIRICHLET
            if (uni01(rng)  < p_k) {
                pair<Vec2D, double> nextStep = importanceSample(previousPoint);
                currentPoint = nextStep.first;
                double invPdf = nextStep.second;
                Vec2D normal_prev = getNormal(previousPoint)[0];

                Polyline weightUpdate = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf);
                pathWeight = 1.0f / invPdf * matrixVectorMultiply(weightUpdate, pathWeight);
            } else {
                pathWeight *= (1.0f / (1.0f - p_k));
                currentPoint = previousPoint;
            }
        }
    }

    return finalResult;
}

Vec2D getCorrectMixedConditionResult(Vec2D startingPoint, vector<Polyline> boundaryDirichlet,
   vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,
   function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {

    const float phi = 1.0f;
    const float k = 2.0f; // Using the balanced value, as discussed
    const float p_k = 1.0f / 3.0f;

    Vec2D finalResult = Vec2D(0, 0);

    pair<Vec2D, double> firstStep = importanceSample(startingPoint); 
    Vec2D currentPoint = firstStep.first;
    double invPdf_0 = firstStep.second;

    // The path weight is initialized with the source term, scaled by the first invPdf.
    BoundaryType initialType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);
    Vec2D pathWeight;
    if (initialType == BoundaryType::Neumann) {
       pathWeight = invPdf_0 * (2.0f * phi * getNeumannValue(currentPoint, boundaryNeumann));
    } else { // DIRICHLET
       pathWeight = invPdf_0 * (k * getDirichletValue(currentPoint, boundaryDirichlet, displacedPoints));
    }

    thread_local std::mt19937 rng( std::random_device{}() );
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    // --- THE RANDOM WALK ---
    for (int i = 0; i < maxDepth; i++) {
        // --- 1. Accumulate Result ---
        finalResult += matrixVectorMultiply(solutionUnknown(startingPoint, currentPoint, 1.0, boundaryDirichlet, boundaryNeumann), pathWeight);

        // --- 2. Russian Roulette ---
        const double absorptionProb = 0.2;
        if (uni01(rng) < absorptionProb) {
            break;
        }
        pathWeight *= (1.0f / (1.0f - absorptionProb));

        // --- 3. Update Path Weight ---
        Vec2D previousPoint = currentPoint;
        BoundaryType currentType = getBoundaryTypeAtPoint(previousPoint, boundaryDirichlet, boundaryNeumann);

        pair<Vec2D, double> nextStep = importanceSample(previousPoint);
        currentPoint = nextStep.first;
        double invPdf = nextStep.second;
        Vec2D normal_prev = getNormal(previousPoint)[0];
        
        // Get the pure, unscaled kernel (assumes fredholm... does NOT multiply by invPdf)
        Polyline transitionKernel = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf);

        if (currentType == BoundaryType::Neumann) {
            pathWeight = invPdf * matrixVectorMultiply(transitionKernel, pathWeight);
        
        } else { // DIRICHLET
            if (uni01(rng) < p_k) {
                // CORRECTED UPDATE:
                pathWeight = (invPdf / p_k) * matrixVectorMultiply(transitionKernel, pathWeight);
            } else {
                pathWeight *= (1.0f / (1.0f - p_k));
                currentPoint = previousPoint;
            }
        }
    }

    return finalResult;
}
// Vec2D getOneStepResult(Vec2D startingPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int depth, int maxDepth) {
//    // startingPoint = sampleRectangleBoundary();
//    Vec2D nextPoint = sampleRectangleBoundary();
//    if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
//       auto [y_dirichlet, inv_pdf_d] = sample_boundary_ris(
//          gen,
//          [&](const Vec2D& y) {
//             std::vector<Vec2D> kelvinMatrix = kelvinKernel(mu, poissonRatio, startingPoint - y);
//             double frobenius_norm = 0.0;
//             for (const Vec2D& val : kelvinMatrix) {
//                frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
//             }
//              return abs(frobenius_norm);
//          },
//          rectangleBoundarySampler,
//          num_resampling_candidates
//      );
//       if (!isnan(real(y_dirichlet)) && !isnan(imag(y_dirichlet))) {
//          nextPoint = y_dirichlet;
//       } 
//    } else {
//       auto [y_neumann, inv_pdf_n] = sample_boundary_ris(
//          gen,
//          [&](const Vec2D& y) {
//             vector<Vec2D> yNormals = getNormal(y);
//             Vec2D yNormal = yNormals.size() > 0 ? yNormals[0] : Vec2D(0, 0); 
//             std::vector<Vec2D> cornormalMatrix = conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - y, yNormal);
//             double frobenius_norm = 0.0;
//             for (const Vec2D& val : cornormalMatrix) {
//                frobenius_norm += real(val) * real(val) + imag(val) * imag(val);
//             }
//              return abs(frobenius_norm);
//          },
//          rectangleBoundarySampler,
//          num_resampling_candidates
//       );
//       if (!isnan(real(y_neumann)) && !isnan(imag(y_neumann))) {
//          nextPoint = y_neumann;
//       }
//    }

//    vector<Vec2D> nextPointNormals = getNormal(nextPoint);
//    Vec2D nextPointNormal = nextPointNormals.size() > 0 ? nextPointNormals[0] : Vec2D(0, 0); 

//    if (depth == maxDepth) {
//       if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)) {
//          return 2 * getDirichletValue(startingPoint, boundaryDirichlet, displacedPoints);
//       } else {
//          return getNeumannValue(startingPoint, boundaryNeumann);
//       }
//    }
//    else {
//       if (isOnDirichlet(nextPoint, boundaryDirichlet, boundaryNeumann)) {
//          Vec2D contribution = getMixedConditionResultKernel5(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
//          int choice = generateRandomZeroOrOne();
   
//          if (choice == 0) {
//             return 4 * getDirichletValue(nextPoint, boundaryDirichlet, displacedPoints) - 24 * matrixVectorMultiply(kelvinKernel(mu, poissonRatio, startingPoint - nextPoint), contribution); 
//          } else {
//             return 4 * getDirichletValue(nextPoint, boundaryDirichlet, displacedPoints) + 3 * contribution;
//          }

//       } else {
//          Vec2D contribution = getMixedConditionResultKernel5(nextPoint, boundaryDirichlet, boundaryNeumann, getDirichletValue, geNeumannValue, depth + 1, maxDepth);
//          return 2 * getNeumannValue(nextPoint, boundaryNeumann) - 8 * matrixVectorMultiply(conormalDerivativeKelvinKernel(mu, poissonRatio, startingPoint - nextPoint, nextPointNormal), contribution); // why minus? 
//       }
//    }
// }

void solveGradientWOB( Vec2D x0,
              vector<Polyline> boundaryDirichlet, 
              vector<Polyline> boundaryNeumann,
              function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, std::ofstream& displacementFile, std::ofstream& gradientFile) { 
   const int nWalks = 100000; 

   Vec2D sumU = Vec2D(0, 0);

   for (int i = 0; i < nWalks; i++) {
      // --- STEP 1: Sample a new point 'y' on the boundary ---
      // This is the Monte Carlo integration step for u = ∫ Γμ dA.
      pair<Vec2D, double> sample = rectangleBoundarySampler();
      Vec2D y = sample.first;      // A random point on the boundary
      double invPdf = sample.second; // The inverse PDF for sampling y

      // --- STEP 2: Estimate μ at that new point y ---
      // This is where you call your existing function.
      // NOTE: The 'startingPoint' for the μ walk is now 'y'.
      Vec2D mu_at_y = getMixedConditionResultKernel10(y, boundaryDirichlet, boundaryNeumann, getDirichletValue, getNeumannValue, 6);

      // --- STEP 3: Calculate the Kelvin Kernel Γ(x,y) ---
      vector<Vec2D> kelvin_kernel = kelvinKernel(mu, poissonRatio, x0 - y);

      // --- STEP 4: Combine everything to get one sample of u ---
      // This is u_sample ≈ (1/pdf) * Γ(x,y) * μ(y)
      Vec2D u_sample = invPdf * matrixVectorMultiply(kelvin_kernel, mu_at_y);

      sumU += u_sample;
   }

   displacementFile << real(x0) << "," << imag(x0) << ",";
   if (isnan(real(sumU) /nWalks) || isnan(imag(sumU)/nWalks)) {
      cout << "Nan value encountered at x0: " << real(x0) << ", " << imag(x0) << endl;
   }
   displacementFile << real(sumU) /nWalks  << "," << imag(sumU)/nWalks << "\n";
}

vector<Vec2D> solveGradient( Vec2D x0,
   vector<Polyline> boundaryDirichlet, 
   vector<Polyline> boundaryNeumann,
   Vec2D source,
   function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getNeumannValue, std::ofstream& displacementFile, std::ofstream& gradientFile) { 
   const double eps = 0.000001; 
   const double rMin = 0.000001; 
   const int nWalks = 100000;
   const int maxSteps = 65536; 
   double sum_11 = 0.0; 
   double sum_12 = 0.0;
   double sum_21 = 0.0; 
   double sum_22 = 0.0;
   double sum_x = 0.0;
   double sum_y = 0.0;
   int i = 0;
   int walker = 0;
   int countX1 = 0;
   int countY1 = 0;
   int biggerX = 0;
   int smallerX = 0;
   int biggerY = 0;
   int smallerY = 0;

   double nextTheta = -1;
   int count = 0;
   Vec2D center;
   for( i = 0; i < nWalks; i++ ) {
   std::mt19937 generator(i);  
   std::uniform_real_distribution<double> dist(-M_PI, M_PI);
   Vec2D x = x0; 
   Vec2D n{ 0.0, 0.0 }; 
   bool onBoundary = false; 
   double r, dDirichlet, dSilhouette; 
   int steps = 0;
   Vec2D closestPoint;
   bool isStarting = true;
   Vec2D normal = Vec2D(0, 0);
   double neumann_contribution_x = 0;
   double neumann_contribution_y = 0;
   double source_contribution_x = 0;
   double source_contribution_y = 0;
   double raidus = 0;
   double alpha = 1;

   do { 
      center = x;
      auto p = distancePolylines( x, boundaryDirichlet );
      dDirichlet = p.first;
      closestPoint = p.second;
      dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
      r = max( rMin, min( dDirichlet, dSilhouette ));

      // compute single sample neumann contribution
      double theta = random( -M_PI, M_PI );
      if( onBoundary ) { // sample from a hemisphere around the normal
      theta = theta/2. + angleOf(n);
      alpha = 0.5; 
      }
      else {
      alpha = 1.0;
   }
  
   Vec2D v{ cos(theta), sin(theta) }; 
   x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary );
   if (isStarting) {
      isStarting = false;
      normal = v / length(v);
      raidus = dDirichlet; 
             
   }

   steps++;
   } 
   while(dDirichlet > eps && steps < maxSteps);

      if( steps >= maxSteps ) continue;
      // accumulatedDisplacement = { Vec2D(kernelAii(real(x0), real(x), r, poissonRatio), kernelAij(real(x0), real(x), imag(x0), imag(x), r, poissonRatio)),
                                 // Vec2D(kernelAij(imag(x0), imag(x), real(x0), real(x), r, poissonRatio), kernelAii(imag(x0), imag(x), r, poissonRatio))};            
      Vec2D ug = getDirichletValue(closestPoint, boundaryDirichlet, displacedPoints) - closestPoint;
      Vec2D estimated_u = ug;
      vector<Vec2D> estimated_normal = multiply(estimated_u, normal);
      estimated_normal = { Vec2D(2 * 1/raidus * real(estimated_normal[0]), 2 * 1/raidus * imag(estimated_normal[0])),
      Vec2D(2 * 1/raidus * real(estimated_normal[1]), 2 * 1/raidus * imag(estimated_normal[1])) };

      if (isnan(real(estimated_u)) || isnan(imag(estimated_u))) {
      continue;
      }
      walker += 1;
      sum_11 += real(estimated_normal[0]);
      sum_12 += imag(estimated_normal[0]);
      sum_21 += real(estimated_normal[1]);
      sum_22 += imag(estimated_normal[1]);
      sum_x += real(estimated_u);
      sum_y += imag(estimated_u);
   } 

   displacementFile << real(x0) << "," << imag(x0) << ",";
   displacementFile << sum_x /walker  << "," << sum_y/walker << "\n";
   gradientFile << real(x0) << "," << imag(x0) << ",";
   gradientFile << sum_11/walker << "," << sum_12/walker << "," << sum_21/walker << "," << sum_22/walker << "\n";
   Vec2D row1 = Vec2D(sum_11/walker, sum_12/walker);
   Vec2D row2 = Vec2D(sum_21/walker, sum_22/walker);
   return {row1, row2};
}

Vec2D interpolateVec2D_BoundaryPoints(Vec2D v, vector<Polyline> originalPoints, vector<Polyline> displacedPoints, double num_tol=1e-5) { 
   for (int i = 0; i < originalPoints[0].size() - 1; i++) {
      Vec2D AP = v - originalPoints[0][i];
      Vec2D PB = v - originalPoints[0][i + 1];
      Vec2D AB = originalPoints[0][i + 1] - originalPoints[0][i];
      Vec2D displaced1 = displacedPoints[0][i]; 
      Vec2D displaced2 = displacedPoints[0][i + 1];

      if (abs(length(AP) + length(PB) - length(AB)) < num_tol) {
         Vec2D displaced = displaced1 + (displaced2 - displaced1) * length(AP) / length(AB);
         return displaced;
      }
   }

   // cerr << "v is not on the boundary" << ", value: " << real(v) << " " << imag(v) << std::endl;
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   return nan;
}

Vec2D identity( Vec2D v ) {
  return v;
}

void storeEigenvectors(double dudx, double dudy, double dvdx, double dvdy, std::ofstream& eigenvectorFile) {
   // Create matrix
   Eigen::MatrixXf matrix(2, 2);
   matrix << dudx, dudy,
            dvdx, dvdy;

   // Compute eigenvalues and eigenvectors
   Eigen::EigenSolver<Eigen::MatrixXf> eigensolver;
   eigensolver.compute(matrix);

   // Get real parts of eigenvalues and eigenvectors
   Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().real();
   Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors().real();

   // Find index of maximum eigenvalue
   Eigen::Index maxIndex;
   eigenvalues.maxCoeff(&maxIndex);

   // Store the principal eigenvector
   Eigen::VectorXf principalEigenvector = eigenvectors.col(maxIndex);
   eigenvectorFile << "principalX,principalY\n";
   eigenvectorFile << principalEigenvector(0) << "," << principalEigenvector(1) << "\n";
}

void storeEigens(double dudx, double dudy, double dvdx, double dvdy, 
                  std::ofstream& eigenvalueFile, std::ofstream& eigenvalueFile2, std::ofstream& eigenvalueFile3) {
   // Create matrix
   Eigen::MatrixXf matrix(2, 2);
  
   matrix << dudx, dudy,
            dvdx, dvdy;

   // Calculate the transpose of the matrix
   Eigen::MatrixXf matrixTranspose = matrix.transpose();

   // Multiply the transpose of the matrix by the matrix
   Eigen::MatrixXf strain = 0.5 * (matrixTranspose * matrix - Eigen::MatrixXf::Identity(2, 2));

   // Compute eigenvalues and eigenvectors
   Eigen::EigenSolver<Eigen::MatrixXf> eigensolver;
   eigensolver.compute(strain);

   // Get real parts of eigenvalues and eigenvectors
   Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().real();
   Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors().real();

   // Find index of maximum eigenvalue
   Eigen::Index maxIndex;

   eigenvalues.maxCoeff(&maxIndex);
   eigenvalueFile << eigenvalues(maxIndex) << "\n";

   // Find index of maximum magnitude eigenvalue
   eigenvalues.array().abs().maxCoeff(&maxIndex);
   eigenvalueFile2 << eigenvalues(maxIndex) << "\n";

   // test: storing norm instead
   eigenvalueFile3 << strain.norm() << "\n";
}

vector<Vec2D> getDeformationGradientAndStress( Vec2D point, double h, function<Vec2D(Vec2D)> deform, 
                                             std::ofstream& strainFile, std::ofstream& neighbourFile, std::ofstream& eigenvalueFile, std::ofstream& eigenvectorFile,
                                             std::ofstream& eigenvalueFile2, std::ofstream& eigenvalueFile3
                                             ) {
   double x = real(point);
   double y = imag(point);
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   Vec2D solved_vec = nan; 
   if (!strainFile.is_open()) {
      std::cerr << "Unable to open file: " << std::endl;
      return vector<Vec2D>{solved_vec, solved_vec};
   }

   Vec2D left{ x - h/2, y };
   Vec2D right{ x + h/2, y };
   Vec2D top{ x, y + h/2 };
   Vec2D bottom{ x, y - h/2 };
   Vec2D farLeft{ x - h, y };
   Vec2D farRight{ x + h, y };
   Vec2D farTop{ x, y + h };
   Vec2D farBottom{ x, y - h };
   vector<Vec2D> neighbors = {left, right, top, bottom, farLeft, farRight, farTop, farBottom};
   vector<Vec2D> neighbors_deformed = {};
  
   for (int i = 0; i < neighbors.size(); i ++){
      if (insideDomain(neighbors[i], boundaryDirichlet, boundaryNeumann)){
         solved_vec = solve(neighbors[i], boundaryDirichlet, boundaryNeumann, deform);
         neighbors_deformed.push_back(solved_vec);
      }
      else {
         return vector<Vec2D>{nan, nan};
      }
   }
  
   neighbourFile << "leftX, leftY, rightX, rightY, topX, topY, bottomX, bottomY\n";
   neighbourFile << real(left) << "," << imag(left) << ",";
   neighbourFile << real(right) << "," << imag(right) << ",";
   neighbourFile << real(top) << "," << imag(top) << ",";
   neighbourFile << real(bottom) << "," << imag(bottom) << "\n";
  
   neighbourFile << "leftX_d, leftY_d, rightX_d rightY_d, topX_d, topY_d, bottomX_d, bottomY_d\n";
   neighbourFile << real(neighbors_deformed[0]) << "," << imag(neighbors_deformed[0]) << ",";
   neighbourFile << real(neighbors_deformed[1]) << "," << imag(neighbors_deformed[1]) << ",";
   neighbourFile << real(neighbors_deformed[2]) << "," << imag(neighbors_deformed[2]) << ",";
   neighbourFile << real(neighbors_deformed[3]) << "," << imag(neighbors_deformed[3]) << "\n";

   double dudx = (8 * real(neighbors_deformed[1]) - 8 * real(neighbors_deformed[0]) + real(neighbors_deformed[4]) - real(neighbors_deformed[5])) / (6 * h);
   double dudy = (8 * real(neighbors_deformed[2]) - 8 * real(neighbors_deformed[3]) + real(neighbors_deformed[7]) - real(neighbors_deformed[6])) / (6 * h);
   double dvdx = (8 * imag(neighbors_deformed[1]) - 8 * imag(neighbors_deformed[0]) + imag(neighbors_deformed[4]) - imag(neighbors_deformed[5])) / (6 * h);
   double dvdy = (8 * imag(neighbors_deformed[2]) - 8 * imag(neighbors_deformed[3]) + imag(neighbors_deformed[7]) - imag(neighbors_deformed[6])) / (6 * h);

   strainFile << "X,Y,F11,F12,F21,F22\n";
   strainFile << x << "," << y << ",";
   strainFile << dudx << "," << dudy << "," << dvdx << "," << dvdy << "\n";

   eigenvalueFile << x << "," << y << ", "; 
   eigenvalueFile2 << x << "," << y << ", ";
   eigenvalueFile3 << x << "," << y << ", ";

   return vector<Vec2D>{ Vec2D{dudx, dudy}, Vec2D{dvdx, dvdy}};
}

string double_to_str(double f) {
   std::string str = std::to_string (f);
   str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
   str.erase ( str.find_last_not_of('.') + 1, std::string::npos );
   return str;
}

int main( int argc, char** argv ) {
   string shape = "lame_wob_mixed_50";
   double h = 0.01;
   string fileName = shape; 
   int s = 16;

   std::ofstream gradientFile("../output/" + fileName + "_deformation_gradient.csv");
   std::ofstream displacementFile("../output/" + shape + "_displacements.csv");
   gradientFile << "X,Y,F11,F12,F21,F22\n";

   for( int j = 0; j < s; j++ )
   {
      for( int i = 0; i < s; i++ )
      {
      
         Vec2D x0(((double)i / (s - 1)),
                  ((double)j / (s - 1))
         );
         if (insideDomain(x0, boundaryDirichlet, boundaryNeumann)){
            solveGradientWOB(x0, boundaryDirichlet, boundaryNeumann, getDirichletValue, getNeumannValue, displacementFile, gradientFile);
         }
      }
   }
   
   vector<Vec2D> customValues;
   customValues.push_back(Vec2D(0.01, 0.1));
   customValues.push_back(Vec2D(0.01, 0.2));
   customValues.push_back(Vec2D(0.01, 0.3));
   customValues.push_back(Vec2D(0.01, 0.4));
   customValues.push_back(Vec2D(0.01, 0.5));
   customValues.push_back(Vec2D(0.01, 0.6));
   customValues.push_back(Vec2D(0.01, 0.7));
   customValues.push_back(Vec2D(0.01, 0.8));
   customValues.push_back(Vec2D(0.01, 0.9));
   customValues.push_back(Vec2D(0.99, 0.1));
   customValues.push_back(Vec2D(0.99, 0.2));
   customValues.push_back(Vec2D(0.99, 0.3));
   customValues.push_back(Vec2D(0.99, 0.4));
   customValues.push_back(Vec2D(0.99, 0.5));
   customValues.push_back(Vec2D(0.99, 0.6));
   customValues.push_back(Vec2D(0.99, 0.7));
   customValues.push_back(Vec2D(0.99, 0.8));
   customValues.push_back(Vec2D(0.99, 0.9));
   for (const auto& point : customValues) {
         if (insideDomain(point, boundaryDirichlet, boundaryNeumann)) {
               solveGradientWOB(point, boundaryDirichlet, boundaryNeumann, getDirichletValue, getNeumannValue, displacementFile, gradientFile);
         }
   }

   // for( int j = 0; j < s; j++ )
   // {
   //    for( int i = 0; i < s; i++ )
   //    {
   //       double i_scaled = i * 0.2;
   //       Vec2D x0(((double)i_scaled / (s - 0.2)),
   //                ((double)j / (s - 1))
   //       );
   //       if( insideDomain(x0, boundaryDirichlet, boundaryNeumann)){
   //          vector<Vec2D> gradient = solveGradient(x0, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //       }

   //       if (j == 0){
   //          Vec2D x1 = Vec2D(real(x0), 0.01);
   //          Vec2D x2 = Vec2D(real(x0), 0.041);
   //          Vec2D x3 = Vec2D(real(x0), 0.081);
   //          Vec2D x4 = Vec2D(real(x0), 0.99);
   //          Vec2D x5 = Vec2D(real(x0), 0.959);
   //          Vec2D x6 = Vec2D(real(x0), 0.919);
   //          if (insideDomain(x1, boundaryDirichlet, boundaryNeumann)) {
   //             solveGradient(x1, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //          }
   //          if (insideDomain(x2, boundaryDirichlet, boundaryNeumann)) {
   //             solveGradient(x2, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //          }
   //          if (insideDomain(x3, boundaryDirichlet, boundaryNeumann)) {
   //             solveGradient(x3, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //          }
   //          if (insideDomain(x4, boundaryDirichlet, boundaryNeumann)) {
   //             solveGradient(x4, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //          }
   //          if (insideDomain(x5, boundaryDirichlet, boundaryNeumann)) {
   //             solveGradient(x5, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //          }
   //          if (insideDomain(x6, boundaryDirichlet, boundaryNeumann)) {
   //             solveGradient(x6, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //          }
   //       }
   //    } 
   // }
   // vector<Vec2D> customValues;
   // customValues.push_back(Vec2D(0.4925, 0.051));
   // customValues.push_back(Vec2D(0.49, 0.01));
   // customValues.push_back(Vec2D(0.495, 0.11));
   // customValues.push_back(Vec2D(0.4975, 0.161));
   // customValues.push_back(Vec2D(0.505, 0.11));
   // customValues.push_back(Vec2D(0.5025, 0.051));
   // customValues.push_back(Vec2D(0.5075, 0.161));
   // customValues.push_back(Vec2D(0.51, 0.01));
   // customValues.push_back(Vec2D(0.5, 0.21));
   // customValues.push_back(Vec2D(0.5, 0.3));  
   // customValues.push_back(Vec2D(0.5, 0.4));
   // for (const auto& point : customValues) {
   //       if (insideDomain(point, boundaryDirichlet, boundaryNeumann)) {
   //          vector<Vec2D> gradient = solveGradient(point, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
   //       }
   // }
}
