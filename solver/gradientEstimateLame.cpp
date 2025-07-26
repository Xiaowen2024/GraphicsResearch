
// c++ -std=c++17 -O3 -pedantic -Wall gradientEstimateLame.cpp -o gel -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -w

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <chrono> 
using namespace std;
using namespace std::chrono; 

#include <Eigen/Dense> 
#include "fractureModelHelpers.cpp"
#include "fractureModelHelpers.h"

std::random_device rd;  // Non-deterministic random number generator
std::mt19937 gen(rd()); // Mersenne Twister PRNG
std::uniform_real_distribution<double> dist(-M_PI, M_PI);

// use std::complex to implement 2D vectors
using Vec2D = complex<double>;

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

Vec2D interpolateVec2DBoundaryPoints(Vec2D v, vector<Polyline> originalPoints, vector<Polyline> displacedPoints, double num_tol=1e-5) { 
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

// vector<Polyline> boundaryDirichlet = {{ Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.49, 0), Vec2D(0.5, 0.2), Vec2D(0.51, 0), Vec2D(1, 0)}};
// vector<Polyline> displacedPoints =   {{ Vec2D(1.1, 0), Vec2D(1.1, 1), Vec2D(-0.1, 1), Vec2D(-0.1, 0)}};

// vector<Polyline> boundaryDirichlet = {{  Vec2D(0.0, 1.0), Vec2D(0, 0)}, {Vec2D(1, 0), Vec2D(1, 1)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.49, 0), Vec2D(0.5, 0.2), Vec2D(0.51, 0), Vec2D(1, 0)}, { Vec2D(1, 1), Vec2D(0, 1)} };
// vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1.0), Vec2D(-0.1, 0)}, {Vec2D(1.1, 0), Vec2D(1.1, 1)}};

// vector<Polyline> boundaryDirichlet = {{  Vec2D(0, 1), Vec2D(0, 0)}, {Vec2D(0.2, 0), Vec2D(0.2, 1)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.2, 0)}, { Vec2D(0.2, 1), Vec2D(0, 1)} };
// vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1.0), Vec2D(-0.1, 0)}, {Vec2D(0.3, 0), Vec2D(0.3, 1)}};

vector<Polyline> boundaryDirichlet = {{  Vec2D(0, 1), Vec2D(0, 0), Vec2D(0.5, 0.2), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1)}};
vector<Polyline> boundaryNeumann =  {};
vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1), Vec2D(-0.1, 0), Vec2D(0.5, 0.2), Vec2D(1.1, 0), Vec2D(1.1, 1), Vec2D(-0.1, 1)}};

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
   else {
      // If no conditions are met, return NaN
      return {};
   }
   return normals;
}

bool written = false;

Vec2D getNeumannValue( Vec2D point, vector<Polyline> boundaryNeumann, vector<Polyline> displacedPoints) {
   const double tolerance = 1e-5; // Define a small tolerance for comparison
   // Check if the point is on the Neumann boundary
   if (abs(imag(point) - 1) < tolerance) {
      return Vec2D(0, -0.1);
   }
   else if (abs(imag(point)) < tolerance) {
      return Vec2D(0, 0.1);
   }
   else if (abs(real(point)) < tolerance) {
      return Vec2D(0.1, 0);
   }
   else if (abs(real(point) - 1) < tolerance) {
      return Vec2D(-0.1, 0);
   }
   else {
      // cout << "Error: getNeumannValue called with point not on Neumann boundary: " << real(point) << ", " << imag(point) << endl;
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

vector<Vec2D> kelvinKernel(double shearModulus, double poissonRatio, Vec2D x){
   double c1 = 1/2 * ( 1 / shearModulus  + 1 / (2 * shearModulus + poissonRatio));
   double c2 = 1/2 * ( 1 / shearModulus  - 1 / (2 * shearModulus + poissonRatio));
   
   double k11 = - c2 / (2 * M_PI) * real(x) * real(x) / (length(x) * length(x));
   double k12 = c1 / (2 * M_PI) * log(length(x)) - c2 / (2 * M_PI) * real(x) * imag(x) / (length(x) * length(x));
   double k21 = c1 / (2 * M_PI) * log(length(x)) - c2 / (2 * M_PI) * real(x) * imag(x) / (length(x) * length(x));
   double k22 = - c2 / (2 * M_PI) * imag(x) * imag(x) / (length(x) * length(x));

   return {Vec2D(k11, k12), 
           Vec2D(k21, k22)};
}
std::vector<std::vector<Vec2D>> kelvinKernelGradient(double shearModulus, double poissonRatio, Vec2D r) {
   // These c1, c2 need to be the same as in kelvinKernel
   double c1_val = 1.0 / (2.0 * shearModulus) + 1.0 / (2.0 * shearModulus + poissonRatio);
   double c2_val = 1.0 / (2.0 * shearModulus) - 1.0 / (2.0 * shearModulus + poissonRatio);

   double A = c1_val / (2.0 * M_PI); // Corresponds to gamma1 / (2pi)
   double B = - c2_val / (2.0 * M_PI); // Corresponds to -gamma2 / (2pi)

   double r_x = real(r);
   double r_y = imag(r);
   double r_sq = r_x * r_x + r_y * r_y;
   double r_pow4 = r_sq * r_sq;

   // Derivatives of 1/r^2 and ln(r)
   double d_lnr_drx = r_x / r_sq;
   double d_lnr_dry = r_y / r_sq;
   double d_r_sq_inv_drx = -2.0 * r_x / r_pow4;
   double d_r_sq_inv_dry = -2.0 * r_y / r_pow4;

   // Allocate 2x2 matrix for gradients: dGamma[row][col] is a Vec2D representing (dGamma_row_col/drx, dGamma_row_col/dry)
   std::vector<std::vector<Vec2D>> dGamma(2, std::vector<Vec2D>(2));

   // --- Derivatives of Gamma_11 ---
   // Gamma_11 = A * log_r + B * r_x^2 / r_sq
   dGamma[0][0].real(A * d_lnr_drx + B * (2.0 * r_x / r_sq - r_x * r_x * d_r_sq_inv_drx)); // d(Gamma_11)/dr_x
   dGamma[0][0].imag(A * d_lnr_dry + B * (-r_x * r_x * d_r_sq_inv_dry));                 // d(Gamma_11)/dr_y

   // --- Derivatives of Gamma_12 ---
   // Gamma_12 = B * r_x * r_y / r_sq
   dGamma[0][1].real(B * (r_y / r_sq - r_x * r_y * d_r_sq_inv_drx)); // d(Gamma_12)/dr_x
   dGamma[0][1].imag(B * (r_x / r_sq - r_x * r_y * d_r_sq_inv_dry)); // d(Gamma_12)/dr_y

   // --- Derivatives of Gamma_21 --- (Symmetric, Gamma_21 = Gamma_12)
   dGamma[1][0].real(dGamma[0][1].real()); // d(Gamma_21)/dr_x
   dGamma[1][0].imag(dGamma[0][1].imag()); // d(Gamma_21)/dr_y

   // --- Derivatives of Gamma_22 ---
   // Gamma_22 = A * log_r + B * r_y^2 / r_sq
   dGamma[1][1].real(A * d_lnr_drx + B * (-r_y * r_y * d_r_sq_inv_drx));                 // d(Gamma_22)/dr_x
   dGamma[1][1].imag(A * d_lnr_dry + B * (2.0 * r_y / r_sq - r_y * r_y * d_r_sq_inv_dry)); // d(Gamma_22)/dr_y

   return dGamma;
}

std::vector<Vec2D> conormalDerivativeKelvinKernel(double shearModulus, double poissonRatio, Vec2D x_minus_y, Vec2D normal_y) {
   // Lamé constants (assuming 2D plane strain for lambda)
   double mu = shearModulus;
   double lambda = (poissonRatio * 2.0 * mu) / (1.0 - 2.0 * poissonRatio);
   // If you are using plane stress or 3D, adjust lambda definition.

   // Calculate the gradients of the Kelvin kernel components
   // dGamma[m][k] is (dGamma_mk/drx, dGamma_mk/dry)
   std::vector<std::vector<Vec2D>> dGamma_dr = kelvinKernelGradient(shearModulus, poissonRatio, x_minus_y);

   std::vector<Vec2D> conormal_kernel(2); // This will be the 2x2 matrix for the double layer kernel

   // Calculate each component of the conormal derivative (dGamma/d_nu)_ik
   // For each column k of Gamma (k=0, 1)
   for (int k = 0; k < 2; ++k) { // k is the column index of Gamma (original force direction)
       // For each component i of the resulting traction vector (i=0, 1)
       for (int i = 0; i < 2; ++i) { // i is the row index of the conormal kernel
           // We need u_j^(k) = Gamma_jk.
           // (dGamma/d_nu)_ik = lambda * (Div(u^(k))) * nu_i + mu * Sum_j ( (du_i^(k)/dr_j) + (du_j^(k)/dr_i) ) * nu_j

           // Part 1: lambda * (Div(u^(k))) * nu_i
           // Div(u^(k)) = d(Gamma_0k)/drx + d(Gamma_1k)/dry
           double div_uk = - (dGamma_dr[0][k].real() + dGamma_dr[1][k].imag()); // Negative because d/dy = -d/dr

           double term1 = lambda * div_uk * (i == 0 ? real(normal_y) : imag(normal_y));

           // Part 2: mu * Sum_j ( (du_i^(k)/dr_j) + (du_j^(k)/dr_i) ) * nu_j
           double term2 = 0.0;
           for (int j = 0; j < 2; ++j) { // Sum over j=0, 1 (x, y components)
               // du_i^(k)/dr_j is d(Gamma_ik)/dr_j
               double du_i_drj = (j == 0 ? dGamma_dr[i][k].real() : dGamma_dr[i][k].imag());

               // du_j^(k)/dr_i is d(Gamma_jk)/dr_i
               double du_j_dri;
               if (i == 0) { // du_j^(k)/dr_x
                   du_j_dri = dGamma_dr[j][k].real();
               } else { // du_j^(k)/dr_y
                   du_j_dri = dGamma_dr[j][k].imag();
               }

               term2 += (du_i_drj + du_j_dri) * (j == 0 ? real(normal_y) : imag(normal_y));
           }
           term2 *= -mu; // Negative because d/dy = -d/dr

           // Assign the calculated component
           if (i == 0) { // Real part of conormal_kernel[k]
               conormal_kernel[k].real(term1 + term2);
           } else { // Imaginary part of conormal_kernel[k]
               conormal_kernel[k].imag(term1 + term2);
           }
       }
   }
   return conormal_kernel;
}

double poissonRatio = 0.3; 
double E = 1.0;
double mu = E / (2.0 * (1.0 + poissonRatio));
static bool hasPrintedGlobal = false; 

Vec2D getU(Vec2D startingPoint, vector<Polyline> boundaryNeumann, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getNeumannValue, int depth, int maxDepth, Vec2D& accumulatedSecond,
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
   Vec2D q = getNeumannValue(intersectionPoint, boundaryNeumann, displacedPoints); // Ensure u is defined
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
   if (!hasPrintedGlobal && normals.size() == 1 && dot(normals[0], Vec2D(cos(theta), sin(theta))) > 0) {
      Vec2D u{cos(theta), sin(theta)};
      theta += M_PI; // Flip the direction to sample in the hemisphere opposite to the normal
      flipped = true;
   } 
   else if (!hasPrintedGlobal && normals.size() == 2 && (dot(normals[0], Vec2D(cos(theta), sin(theta))) > 0 || 
      dot(normals[1], Vec2D(cos(theta), sin(theta))) > 0)) {
      Vec2D u{cos(theta), sin(theta)};
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
   double r = distancePolylines(startingPoint, boundaryDirichlet).first; // Get the distance to the nearest point on the Neumann boundary
   if (onBoundary) {
      p = 1 / (2 * M_PI * r); // Hemisphere sampling on the boundary
   } else {
      p = 1 / (4 * M_PI * r); // Uniform distribution over the distance to the boundary
   }
   Vec2D intersectionPoint = intersectPolylines(startingPoint, v, std::numeric_limits<double>::infinity(), boundaryDirichlet, n, onBoundary);
   // Define p as the probability of getting this r
  
   if (!hasPrintedGlobal && (isinf(real(intersectionPoint)) || isinf(imag(intersectionPoint)) || isnan(real(intersectionPoint)) || isnan(imag(intersectionPoint)))) {
      cout << "startingPoint: " << real(startingPoint) << ", " << imag(startingPoint) << endl;
      cout << "v: " << real(v) << ", " << imag(v) << endl;
      cout << "intersectionPoint: " << real(intersectionPoint) << ", " << imag(intersectionPoint) << endl;
      hasPrintedGlobal = true;
      cout << "flipped: " << flipped << endl;
   }

   Vec2D u = getDirichletValue(intersectionPoint, boundaryDirichlet, displacedPoints); // Ensure u is defined
   vector<Vec2D> ns = getNormal(intersectionPoint);
   Vec2D normal = ns.size() > 0 ? ns[0] : Vec2D(0, 0); // Use the first normal if available
   if (ns.size() == 0) {
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

void solveGradientWOB( Vec2D x0,
              vector<Polyline> boundaryDirichlet, 
              vector<Polyline> boundaryNeumann,
              function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getNeumannValue, std::ofstream& displacementFile, std::ofstream& gradientFile) { 
   const int nWalks = 100000; 
   Vec2D sumU = Vec2D(0.0, 0.0);
   int i = 0;

   #pragma omp parallel for reduction(+:sum)
   for( i = 0; i < nWalks; i++ ) {
      Vec2D uhat = getUDirichlet(x0, boundaryDirichlet, getDirichletValue, 0, 10);
      sumU += uhat;
      // Vec2D uhat = getU(x0, boundaryNeumann, getNeumannValue, 0, 5, accumulatedSecond, accumulatedNumber);
      // sumU += uhat + Vec2D(accumulatedSecond.real() / accumulatedNumber, accumulatedSecond.imag() / accumulatedNumber);
   } 

   displacementFile << real(x0) << "," << imag(x0) << ",";
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

   #pragma omp parallel for reduction(+:sum)
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
   // else {
   //    accumulatedAKernel = matrixMultiply(
   //       accumulatedAKernel, 
   //       { Vec2D(kernelAii(real(center), real(x), r, poissonRatio), kernelAij(real(center), real(x), imag(center), imag(x), r, poissonRatio)),
   //          Vec2D(kernelAij(imag(center), imag(x), real(center), real(x), r, poissonRatio), kernelAii(imag(center), imag(x), r, poissonRatio))}        
   //    );
   // }

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
   string shape = "lame_wob_dirichlet_nonconvex";
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
         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann)){
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
