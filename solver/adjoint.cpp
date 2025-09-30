
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

// Handle OpenMP include with fallback paths
#ifdef _OPENMP
  #include <omp.h>
#else
  // OpenMP stub for systems without OpenMP support
  #define omp_get_thread_num() 0
  #define omp_get_max_threads() 1
  #define omp_get_num_threads() 1
#endif

using namespace std;
using namespace std::chrono; 

#include <Eigen/Dense> 
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

vector<Vec2D> tensor3DVec2DMultiply(const vector<vector<Vec2D>>& tensor, const Vec2D& vec) {
    // The result is a 2-element vector of Vec2D objects.
    vector<Vec2D> result(2, Vec2D(0.0, 0.0));

    // Get the components of the input vector.
    double vx = vec.real();
    double vy = vec.imag();

    // Loop through the first dimension of the tensor.
    for (int i = 0; i < 2; ++i) {
        // For each row 'i' of the tensor, we calculate a new Vec2D.
        // The first component of this new Vec2D is the dot product of the
        // first row of the matrix slice with the input vector.
        double real_component = tensor[i][0].real() * vx + tensor[i][0].imag() * vy;
        
        // The second component of the new Vec2D is the dot product of the
        // second row of the matrix slice with the input vector.
        double imag_component = tensor[i][1].real() * vx + tensor[i][1].imag() * vy;
        
        result[i] = Vec2D(real_component, imag_component);
    }

    return result;
}

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
   return make_pair(Vec2D{x, y},  1 / pdf);
}

vector<Polyline> boundaryDirichlet = {{ Vec2D(0, 1), Vec2D(0, 0)}, { Vec2D(1, 0), Vec2D(1, 1)}};
vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(1, 0)}, { Vec2D(1, 1), Vec2D(0, 1)}};
vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1), Vec2D(-0.1, 0)}, { Vec2D(1.1, 0), Vec2D(1.1, 1)}};

std::vector<Vec2D> boundaryCorners = {
   {0.0, 0.0}, // Bottom-left
   {1.0, 0.0}, // Bottom-right
   {1.0, 1.0}, // Top-right
   {0.0, 1.0}  // Top-left
};

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
   // if (isPointOnLine(point, Vec2D(0, 0), Vec2D(0.5, 0.2), tolerance)) {
   //    Vec2D lineStart = Vec2D(0, 0);
   //    Vec2D lineEnd = Vec2D(0.5, 0.2);
   //    Vec2D lineDirection = lineEnd - lineStart;
   //    Vec2D normal = Vec2D(imag(lineDirection), -real(lineDirection));
   //    normals.push_back(normal);
   // } 
   // if (isPointOnLine(point, Vec2D(0.5, 0.2), Vec2D(1, 0), tolerance)) {
   //    Vec2D lineStart = Vec2D(0.5, 0.2);
   //    Vec2D lineEnd = Vec2D(1, 0);
   //    Vec2D lineDirection = lineEnd - lineStart;
   //    Vec2D normal = Vec2D(imag(lineDirection), -real(lineDirection));
   //    normals.push_back(normal);
   // }
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



double poissonRatio = 0.3; 
double E = 1.0;
double mu = E / (2.0 * (1.0 + poissonRatio));
static bool hasPrintedGlobal = false; 

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

int num_resampling_candidates = 4;

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
         return make_pair(y_neumann, inv_pdf_n);
      }
   }

   Vec2D nan = numeric_limits<double>::quiet_NaN();
   return make_pair(nan, 0);
}


vector<Vec2D> computeTractionKernel2D(Vec2D x, Vec2D y, Vec2D normal_y, double shearModulus, double poissonRatio) {
   // The vector r from the source 'y' to the field point 'x'
   Vec2D r_vec(real(x) - real(y), imag(x) - imag(y));

   double r_len_sq = std::norm(r_vec); 
   if (r_len_sq < 1e-12) {
       return { Vec2D(0,0), Vec2D(0,0) };
   }
   double r_len = std::sqrt(r_len_sq);

   double dr_dn = (real(r_vec) * real(normal_y) + imag(r_vec) * imag(normal_y)) / r_len;

   double constant_factor = -1.0 / (4.0 * M_PI * (1.0 - poissonRatio) * r_len);

   double r_i_n_j_minus_r_j_n_i = real(r_vec) * imag(normal_y) - imag(r_vec) * real(normal_y);

   double term1 = (1.0 - 2.0 * poissonRatio);
   
   // Calculate all four components cleanly
   double T11 = constant_factor * (dr_dn * (term1 + 2.0 * real(r_vec) * real(r_vec) / r_len_sq) + term1 * imag(r_vec) * imag(normal_y));
   double T12 = constant_factor * (dr_dn * (2.0 * real(r_vec) * imag(r_vec) / r_len_sq) - term1 * r_i_n_j_minus_r_j_n_i);
   double T21 = constant_factor * (dr_dn * (2.0 * real(r_vec) * imag(r_vec) / r_len_sq) + term1 * r_i_n_j_minus_r_j_n_i);
   double T22 = constant_factor * (dr_dn * (term1 + 2.0 * imag(r_vec) * imag(r_vec) / r_len_sq) + term1 * real(r_vec) * real(normal_y));
   
   return { Vec2D(T11, T12), Vec2D(T21, T22) };
}

// This corresponds to the known value part in the BIE formulation
Vec2D fredholmEquationKnown(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, float k){
   if (isOnDirichlet(point, boundaryDirichlet, boundaryNeumann)){
      return k * getDirichletValue(point, boundaryDirichlet, displacedPoints);
   }
   else {
      double integralFreeTerm = 0.5;
      if (isOnCorner(point, boundaryDirichlet, boundaryNeumann)) {
         integralFreeTerm = 0.25;
      }
      return 1 / integralFreeTerm * getNeumannValue(point, boundaryNeumann);
   }
}

// This correspondings to the unknwon integral part that needs recursive evaluation
Polyline fredholmEquationUnknown(Vec2D startingPoint, Vec2D nextPoint, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, Vec2D nextPointNormal, double invPdf, float k) {
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)){
      return -k * invPdf * kelvinKernel(mu, poissonRatio, startingPoint - nextPoint);
   }
   else {
      double integralFreeTerm = 0.5;
      if (isOnCorner(startingPoint, boundaryDirichlet, boundaryNeumann)) {
         integralFreeTerm = 0.25;
      }
      return - 1.0f / integralFreeTerm * invPdf * computeTractionKernel2D(startingPoint, nextPoint, nextPointNormal, mu, poissonRatio);
   }
}

// for boundary point, on dirichlet boundaries, the value we solve for is known so returns a matrix of 0
Polyline solutionBoundaryUnknown(Vec2D startingPoint, Vec2D nextPoint, double invPdf) {
   if (isOnDirichlet(startingPoint, boundaryDirichlet, boundaryNeumann)){
      return {Vec2D(0.0f, 0.0f), Vec2D(0.0f, 0.0f)};
   }
   else {
      return invPdf * kelvinKernel(mu, poissonRatio, startingPoint - nextPoint);
   }
}

// for interior point
Polyline solutionDomainUnknown(Vec2D startingPoint, Vec2D nextPoint, double invPdf) {
   return invPdf * kelvinKernel(mu, poissonRatio, startingPoint - nextPoint);
}

std::pair<Vec2D, double> importanceSampleBoundary(const Vec2D& x0, const std::vector<Vec2D>& boundaryCorners, std::mt19937& rng) {
   int numCorners = boundaryCorners.size();
   if (numCorners < 2) {
       return {{0,0}, 1.0};
   }

   // --- 1. Calculate the angle subtended by each edge from x0 ---
   std::vector<double> angles(numCorners);
   double totalAngle = 0.0;
   for (int i = 0; i < numCorners; ++i) {
       Vec2D v1 = boundaryCorners[i] - x0;
       Vec2D v2 = boundaryCorners[(i + 1) % numCorners] - x0; // Wrap around for the last edge
       
       // Angle between two vectors v1 and v2 is acos(dot(v1, v2) / (|v1| * |v2|))
       double dotProduct = (v1.real() * v2.real() + v1.imag() * v2.imag());
       double magProduct = std::abs(v1) * std::abs(v2);
       
       // Clamp the argument to acos to avoid floating point errors leading to NaN
       double cosTheta = std::clamp(dotProduct / magProduct, -1.0, 1.0);
       
       angles[i] = std::acos(cosTheta);
       totalAngle += angles[i];
   }
   
   // --- 2. Normalize angles to get probabilities for choosing each edge ---
   std::vector<double> edgeProbabilities(numCorners);
   if (totalAngle > 1e-9) { // Avoid division by zero
       for (int i = 0; i < numCorners; ++i) {
           edgeProbabilities[i] = angles[i] / totalAngle;
       }
   } else { // Fallback to uniform if x0 is somehow equidistant from all points (unlikely)
        for (int i = 0; i < numCorners; ++i) edgeProbabilities[i] = 1.0 / numCorners;
   }

   // --- 3. Choose an edge based on the calculated probabilities ---
   std::discrete_distribution<> edgeDist(edgeProbabilities.begin(), edgeProbabilities.end());
   int chosenEdgeIndex = edgeDist(rng);

   // --- 4. Sample a point uniformly along the chosen edge ---
   std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
   double t = uniform_dist(rng); // Random parameter along the edge
   
   Vec2D startCorner = boundaryCorners[chosenEdgeIndex];
   Vec2D endCorner = boundaryCorners[(chosenEdgeIndex + 1) % numCorners];
   Vec2D sampledPoint = startCorner + t * (endCorner - startCorner);
   
   // --- 5. Calculate the PDF and Inverse PDF ---
   double edgeLength = std::abs(endCorner - startCorner);
   
   double pdf = edgeProbabilities[chosenEdgeIndex] / edgeLength;
   double invPdf = 1.0 / pdf;
   return {sampledPoint, invPdf};
}

Vec2D getMixedConditionResultKernelForward(Vec2D evaluationPoint, Vec2D startingPoint, float invPDF, vector<Polyline> boundaryDirichlet,
      vector<Polyline> boundaryNeumann, function<Vec2D (Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue,
      function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, int maxDepth) {

      const float phi = 1.0f;
      const float k = 2.0f; // TODO: variable k could be set
      const float p_k = 1/3; 

      Vec2D finalResult = Vec2D(0, 0);

      // --- INITIALIZATION ---
      thread_local std::mt19937 rng(std::random_device{}());
    //   pair<Vec2D, double> firstStep = rectangleBoundarySampler(); // (startingPoint, boundaryCorners, rng);
    //   Vec2D currentPoint = firstStep.first;
    //   double invPdf_0 = firstStep.second;
      Vec2D currentPoint = startingPoint;
      double invPdf_0 = invPDF;
      BoundaryType initialType = getBoundaryTypeAtPoint(currentPoint, boundaryDirichlet, boundaryNeumann);
      Vec2D pathWeight = invPdf_0 * fredholmEquationKnown(currentPoint, boundaryDirichlet, boundaryNeumann, k);

      std::uniform_real_distribution<double> uni01(0.0, 1.0);
      // --- THE RANDOM WALK ON THE BOUNDARY ---
      for (int i = 0; i < maxDepth; i++) {
         if (i == maxDepth - 1){
            pathWeight = 0.5 * pathWeight;
         }

         // --- 1. ACCUMULATE RESULT ---
         finalResult += matrixVectorMultiply(solutionDomainUnknown(evaluationPoint, currentPoint, 1.0), pathWeight);
         
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
            pathWeight = matrixVectorMultiply(fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf, k), pathWeight);
         } else { // DIRICHLET
            Vec2D knownDirichletTerm = k * getDirichletValue(previousPoint, boundaryDirichlet, displacedPoints);
            // Correspond to the indirect double to indirect single layer transformation mentioned in the paper
            if (uni01(rng) < p_k) {
               pair<Vec2D, double> nextStep = importanceSample(previousPoint);
               currentPoint = nextStep.first;
               double invPdf = nextStep.second;
               Vec2D normal_prev = getNormal(previousPoint)[0];
               Polyline weightUpdate = fredholmEquationUnknown(previousPoint, currentPoint, boundaryDirichlet, boundaryNeumann, normal_prev, invPdf, k);
               pathWeight = matrixVectorMultiply((1.0f / p_k) * weightUpdate, pathWeight); // + knownDirichletTerm;
            } else {
               pathWeight = (1.0f / (1.0f - p_k)) * pathWeight;// + knownDirichletTerm;
               currentPoint = previousPoint;
            }
         }
      }

      return finalResult;
}

void solveGradientWOB( Vec2D x0,
              vector<Polyline> boundaryDirichlet, 
              vector<Polyline> boundaryNeumann,
              function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> getDirichletValue, function<Vec2D(Vec2D, vector<Polyline>)> getNeumannValue, std::ofstream& displacementFile, std::ofstream& gradientFile) { 
   const int nWalks = 100000; 
   Vec2D sumU = Vec2D(0.0, 0.0);
   vector<Vec2D> sumGradient = {Vec2D(0.0, 0.0), Vec2D(0.0, 0.0)};
   int i = 0;

   #pragma omp parallel for reduction(+:sumU)
   for (int i = 0; i < nWalks; i++) {
      // --- STEP 1: Sample a new point 'y' on the boundary ---
      // This is the Monte Carlo integration step for u = ∫ Γμ dA.
      thread_local std::mt19937 rng(std::random_device{}() + i);
      pair<Vec2D, double> sample = rectangleBoundarySampler();// importanceSampleBoundary(x0, boundaryCorners, rng);
      Vec2D y = sample.first;      // A random point on the boundary
      double invPdf = sample.second; // The inverse PDF for sampling y

      // --- STEP 2: Estimate μ at that new point y ---
      // TODO: the last argument (max depth) could be set by user.
      Vec2D mu_at_y = getMixedConditionResultKernelForward(x0, y, invPdf, boundaryDirichlet, boundaryNeumann, getDirichletValue, getNeumannValue, 2);

      // --- STEP 3: Calculate the Kelvin Kernel Γ(x,y) ---
      vector<Vec2D> kelvin_kernel = kelvinKernel(mu, poissonRatio, x0 - y);

      vector<Polyline> gradient = kelvinKernelGradient(mu, poissonRatio, x0 - y);
      
      // // --- STEP 4: Combine everything to get one sample of u ---
      //   Vec2D u_sample = invPdf * matrixVectorMultiply(kelvin_kernel, mu_at_y);
      vector<Vec2D> gradient_sample = tensor3DVec2DMultiply(gradient, mu_at_y);

      sumU += mu_at_y;
      sumGradient[0] += gradient_sample[0];
      sumGradient[1] += gradient_sample[1];
   }

   displacementFile << real(x0) << "," << imag(x0) << ",";
   if (isnan(real(sumU)/nWalks) || isnan(imag(sumU)/nWalks)) {
      cout << "Nan value encountered at x0: " << real(x0) << ", " << imag(x0) << endl;
   }
   displacementFile << real(sumU) /nWalks  << "," << imag(sumU)/nWalks << "\n";

   gradientFile << real(x0) << "," << imag(x0) << ",";
   if (isnan(real(sumGradient[0])/nWalks) || isnan(imag(sumGradient[0])/nWalks)) {
      cout << "Nan value encountered at x0: " << real(x0) << ", " << imag(x0) << endl;
   }
   gradientFile << real(sumGradient[0]) /nWalks  << "," << imag(sumGradient[0])/nWalks << ",";
   if (isnan(real(sumGradient[1])/nWalks) || isnan(imag(sumGradient[1])/nWalks)) {
      cout << "Nan value encountered at x0: " << real(x0) << ", " << imag(x0) << endl;
   }
   gradientFile << real(sumGradient[1]) /nWalks  << "," << imag(sumGradient[1])/nWalks << "\n";
}

int main( int argc, char** argv ) {
   // TODO: Change output name here
   string shape = "lame_wob_adjoint_7";
   double h = 0.01;
   string fileName = shape; 
   int s = 16;

   std::ofstream gradientFile("../output/" + fileName + "_displacement_gradient.csv");
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
}