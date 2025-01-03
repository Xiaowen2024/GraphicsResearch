// (Optimized) implementation of the "Walk on Stars" algorithm for Laplace equations.
// Corresponds to the estimator given in Equation 18 of Sawhney et al,
// "Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary
// Conditions" (2023), assuming no source term and zero-Neumann conditions.
// NOTE: this code makes a few shortcuts for the sake of code brevity; may
// be more suitable for tutorials than for production code/evaluation.
// To compile: c++ -std=c++17 -O3 -pedantic -Wall WoStDeformationGradientRect.cpp -o wost-dg-rect
// c++ -std=c++17 -O3 -pedantic -Wall WostTrouser.cpp -o wost-dg-trouser

// if you can't import Eigen, 
// do export CPLUS_INCLUDE_PATH=<your_eigen_path>:$CPLUS_INCLUDE_PATH
// working eigen path example: /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3
// or run (replacing with your path)
// c++ -std=c++17 -O3 -pedantic -Wall WostTrouser.cpp -o wost-dg-trouser -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <chrono> // Added for timing
using namespace std;
using namespace std::chrono; // Added for timing

#include <Eigen/Dense> 

// the constant "infinity" is used as a maximum value in several calculations
const double infinity = numeric_limits<double>::infinity();

// returns a random value in the range [rMin,rMax]
double random( double rMin, double rMax ) {
   const double rRandMax = 1.0/(double)RAND_MAX;
   double u = rRandMax*(double)rand();
   return u*(rMax-rMin) + rMin;
}

// use std::complex to implement 2D vectors
using Vec2D = complex<double>;
double length( Vec2D u ) { return sqrt( norm(u) ); }
double angleOf(Vec2D u) { return arg(u); }
Vec2D rotate90( Vec2D u ) { return Vec2D( -imag(u), real(u) ); }
double   dot(Vec2D u, Vec2D v) { return real(u)*real(v) + imag(u)*imag(v); }
double cross(Vec2D u, Vec2D v) { return real(u)*imag(v) - imag(u)*real(v); }

// returns the closest point to x on a segment with endpoints a and b
Vec2D closestPoint( Vec2D x, Vec2D a, Vec2D b ) {
   Vec2D u = b-a;
   double t = clamp( dot(x-a,u)/dot(u,u), 0.0, 1.0 );
   return (1.0-t)*a + t*b;
}

// returns true if the point b on the polyline abc is a silhoutte relative to x
bool isSilhouette( Vec2D x, Vec2D a, Vec2D b, Vec2D c ) {
   return cross(b-a,x-a) * cross(c-b,x-b) < 0;
}

// returns the time t at which the ray x+tv intersects segment ab,
// or infinity if there is no intersection
double rayIntersection( Vec2D x, Vec2D v, Vec2D a, Vec2D b ) {
   Vec2D u = b - a;
   Vec2D w = x - a;
   double d = cross(v,u);
   double s = cross(v,w) / d;
   double t = cross(u,w) / d;
   if (t > 0. && 0. <= s && s <= 1.) {
      return t;
   }
   return infinity;
}

using Polyline = vector<Vec2D>;

// returns distance from x to closest point on the given polylines P
double distancePolylines( Vec2D x, const vector<Polyline>& P ) {
   double d = infinity; // minimum distance so far
   // #pragma omp parallel for reduction(min:d)
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 0; j < P[i].size()-1; j++ ) { // iterate over segments
         Vec2D y = closestPoint( x, P[i][j], P[i][j+1] ); // distance to segment
         d = min( d, length(x-y) ); // update minimum distance
      }
   }
   return d;
}

// returns distance from x to closest silhouette point on the given polylines P
double silhouetteDistancePolylines( Vec2D x, const vector<Polyline>& P ){
   double d = infinity; // minimum distance so far
   // #pragma omp parallel for reduction(min:d)
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 1; j < P[i].size()-1; j++ ) { // iterate over segment pairs
         if( isSilhouette( x, P[i][j-1], P[i][j], P[i][j+1] )) {
            d = min( d, length(x-P[i][j]) ); // update minimum distance
         }
      }
   }
   return d;
}

// finds the first intersection y of the ray x+tv with the given polylines P,
// restricted to a ball of radius r around x.  The flag onBoundary indicates
// whether the first hit is on a boundary segment (rather than the sphere), and
// if so sets n to the normal at the hit point.
Vec2D intersectPolylines( Vec2D x, Vec2D v, double r,
                         const vector<Polyline>& P,
                         Vec2D& n, bool& onBoundary ) {
   double tMin = r; // smallest hit time so far
   n = Vec2D{ 0.0, 0.0 }; // first hit normal
   onBoundary = false; // will be true only if the first hit is on a segment
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 0; j < P[i].size()-1; j++ ) { // iterate over segments
         const double c = 1e-5; // ray offset (to avoid self-intersection)
         double t = rayIntersection( x + c*v, v, P[i][j], P[i][j+1] );
         if( t < tMin ) { // closest hit so far
            tMin = t;
            n = rotate90( P[i][j+1] - P[i][j] ); // get normal
            n /= length(n); // make normal unit length
            onBoundary = true;
         }
      }
   }
   return x + tMin*v; // first hit location
}

// solves a Laplace equation Delta u = 0 at x0, where the Dirichlet and Neumann
// boundaries are each given by a collection of polylines, the Neumann
// boundary conditions are all zero, and the Dirichlet boundary conditions
// are given by a function g that can be evaluated at any point in space
Vec2D solve( Vec2D x0, // evaluation point
              vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
              vector<Polyline> boundaryNeumann, // reflecting part of the boundary
              function<Vec2D(Vec2D)> g ) { // Dirichlet boundary values
   const double eps = 0.0001; // stopping tolerance
   const double rMin = 0.0001; // minimum step size
   const int nWalks = 65536; // number of Monte Carlo samples
   const int maxSteps = 65536; // maximum walk length
   double sum_x = 0.0; // running sum of boundary contributions
   double sum_y = 0.0;
   int i = 0;
   unsigned seed = 1;
   srand(seed);
  
   // #pragma omp parallel for reduction(+:sum)
   for( i = 0; i < nWalks; i++ ) {
      Vec2D x = x0; // start walk at the evaluation point
      Vec2D n{ 0.0, 0.0 }; // assume x0 is an interior point, and has no normal
      bool onBoundary = false; // flag whether x is on the interior or boundary

      double r, dDirichlet, dSilhouette; // radii used to define star shaped region
      int steps = 0;
      do { 
         // compute the radius of the largest star-shaped region
         dDirichlet = distancePolylines( x, boundaryDirichlet );
         dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
         r = max( rMin, min( dDirichlet, dSilhouette ));

         // intersect a ray with the star-shaped region boundary
         double theta = random( -M_PI, M_PI );
         if( onBoundary ) { // sample from a hemisphere around the normal
            theta = theta/2. + angleOf(n);
         }
         Vec2D v{ cos(theta), sin(theta) }; // unit ray direction
         x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary );

         steps++;
      }
      while(dDirichlet > eps && steps < maxSteps);
      //stop if we hit the Dirichlet boundary, or the walk is too long

      if( steps >= maxSteps ) cerr << "Hit max steps" << endl;


      Vec2D eval_vec = g(x);
      sum_x += real(eval_vec);
      sum_y += imag(eval_vec);
   }
   // std::cout << i << std::endl;
   return Vec2D(sum_x/nWalks, sum_y/nWalks);
}

// for simplicity, in this code we assume that the Dirichlet and Neumann
// boundary polylines form a collection of closed polygons (possibly with holes),
// and are given with consistent counter-clockwise orientation
vector<Polyline> boundaryDirichlet = {
   {
      {Vec2D(0, 0), Vec2D(0.4, 0), Vec2D(0.5, 0.3), Vec2D(0.6, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)}
   }
};
vector<Polyline> newBoundaryDirichlet = {
   {
      {Vec2D(-0.2, 0), Vec2D(0.3, 0), Vec2D(0.5, 0.5), Vec2D(0.7, 0), Vec2D(1.2, 0), Vec2D(1.0, 1), Vec2D(0, 1), Vec2D(-0.2, 0)}
   }
};

vector<Polyline> boundaryNeumann = {

};

// these routines are not used by WoSt itself, but are rather used to check
// whether a given evaluation point is actually inside the domain
double signedAngle( Vec2D x, const vector<Polyline>& P )
{
   double Theta = 0.;
   for( int i = 0; i < P.size(); i++ )
      for( int j = 0; j < P[i].size()-1; j++ )
         Theta += arg( (P[i][j+1]-x)/(P[i][j]-x) );
   return Theta;
}

Vec2D interpolateVec2D_BoundaryPoints(Vec2D v, vector<Polyline> mappings, double num_tol=1e-3, bool print_in_bounds=false, bool print_out_bounds=false) { 
   if (print_in_bounds) cout << real(v) << " " << imag(v) << " " << mappings[0].size() << std::endl;
   for (int i = 0; i < mappings[0].size() - 1; i++) {
      Vec2D AP = v - boundaryDirichlet[0][i];
      Vec2D PB = v - boundaryDirichlet[0][i + 1];
      Vec2D AB = boundaryDirichlet[0][i + 1] - boundaryDirichlet[0][i];

      Vec2D mapping1 = mappings[0][i]; 
      Vec2D mapping2 = mappings[0][i + 1];

      // check if v is the same as any of the boundary points
      if (abs(real(v) - real(boundaryDirichlet[0][i])) < num_tol && abs(imag(v) - imag(boundaryDirichlet[0][i])) < num_tol)
         { if (print_in_bounds) cout << "in bounds 1" << std::endl; return mapping1; }
      if (abs(real(v) - real(boundaryDirichlet[0][i + 1])) < num_tol && abs(imag(v) - imag(boundaryDirichlet[0][i + 1])) < num_tol)
         { if (print_in_bounds) cout << "in bounds 2" << std::endl; return mapping2; } 

      // check that v lies in the line segment between the boundary points
      // if (distance(A, C) + distance(B, C) == distance(A, B))
      if (abs(length(AP) + length(PB) - length(AB)) < num_tol) {
         // interpolate mapping between mapping1 and mapping2
         Vec2D mapping3 = mapping1 + (mapping2 - mapping1) * length(AP) / length(AB);
         if (print_in_bounds) cout << "in the boundary 3" << real(v) << " " << imag(v) << std::endl;
         return mapping3;
      }
   }

   cerr << "v is not in the boundary" << ", value: " << real(v) << " " << imag(v) << std::endl;
   return Vec2D(0,0);
}

Vec2D displacement(Vec2D v) { 
   vector<Polyline> displacedPoints = {
      {
         {Vec2D(-0.2, 0), Vec2D(0.3, 0), Vec2D(0.5, 0.5), Vec2D(0.7, 0), Vec2D(1.2, 0), Vec2D(1.0, 1), Vec2D(0, 1), Vec2D(-0.2, 0)}
      }
   }; 

   // create vector polyline of displacement vectors from boundary points
   vector<Polyline> displacementVectors = {{{}}};
   for (int i = 0; i < boundaryDirichlet[0].size(); i++) {
      Vec2D point = boundaryDirichlet[0][i];
      Vec2D deformed_vec = displacedPoints[0][i];
      Vec2D displacement_vec = deformed_vec - point;
      displacementVectors[0].push_back(displacement_vec);
   }

   double num_tol = 1;
   Vec2D interpolatedDisplacement = interpolateVec2D_BoundaryPoints(v, displacementVectors, num_tol);
   return interpolatedDisplacement;
}

// for the trouser shape 
Vec2D deformTrousers( Vec2D v ) {
   vector<Polyline> mappings = newBoundaryDirichlet; 
   
   // check if v is between any 2 consecutive points in the boundary and get the corresponding interpolation between the 2 points in the mapping
   double num_tol = 1e-3;
   Vec2D mapping = interpolateVec2D_BoundaryPoints(v, mappings, num_tol);
   return mapping;
}

// Returns true if the point x is contained in the region bounded by the Dirichlet
// and Neumann curves.  We assume these curves form a collection of closed polygons,
// and are given in a consistent counter-clockwise winding order.

// if inside the polygon, the signed angle sum should be close to 2 * PI otherwise it should be close to 0
bool insideDomain( Vec2D x,
                   const vector<Polyline>& boundaryDirichlet,
                   const vector<Polyline>& boundaryNeumann )
{
   double Theta = signedAngle( x, boundaryDirichlet ) +
                  signedAngle( x, boundaryNeumann );
   const double delta = 1e-4; // numerical tolerance
   return abs(Theta-2.*M_PI) < delta; // boundary winds around x exactly once
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

   // Compute eigenvalues and eigenvectors
   Eigen::EigenSolver<Eigen::MatrixXf> eigensolver;
   eigensolver.compute(matrix);

   // Get real parts of eigenvalues and eigenvectors
   Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().real();
   Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors().real();

   // Find index of maximum eigenvalue
   Eigen::Index maxIndex;

   // Store the principal eigenvalue squared
   eigenvalues.maxCoeff(&maxIndex);
   eigenvalueFile << eigenvalues(maxIndex) << "\n";

   // test: Store the principal eigenvalue squared
   // Find index of maximum magnitude eigenvalue
   eigenvalues.array().abs().maxCoeff(&maxIndex);
   eigenvalueFile2 << eigenvalues(maxIndex) << "\n";

   // test: storing norm instead
   eigenvalueFile3 << matrix.norm() << "\n";
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
   neighbourFile << "leftX, leftY, rightX, rightY, topX, topY, bottomX, bottomY\n";
   Vec2D left{ x - h/2, y };
   Vec2D right{ x + h/2, y };
   Vec2D top{ x, y + h/2 };
   Vec2D bottom{ x, y - h/2 };
   vector<Vec2D> neighbors = {left, right, top, bottom};
   vector<Vec2D> neighbors_deformed = {};
   for ( int i = 0; i < 4; i++ ) {
      if( insideDomain(neighbors[i], boundaryDirichlet, boundaryNeumann) ){
         solved_vec = solve(neighbors[i], boundaryDirichlet, boundaryNeumann, deform);
         neighbors_deformed.push_back(solved_vec);
      }
      else {
         return vector<Vec2D>{nan, nan};
      }
   }

   neighbourFile << real(neighbors_deformed[0]) << "," << imag(neighbors_deformed[0]) << ",";
   neighbourFile << real(neighbors_deformed[1]) << "," << imag(neighbors_deformed[1]) << ",";
   neighbourFile << real(neighbors_deformed[2]) << "," << imag(neighbors_deformed[2]) << ",";
   neighbourFile << real(neighbors_deformed[3]) << "," << imag(neighbors_deformed[3]) << "\n";

   double dudx = (real(neighbors_deformed[1]) - real(neighbors_deformed[0])) / h;
   double dudy = (real(neighbors_deformed[2]) - real(neighbors_deformed[3])) / h;
   double dvdx = (imag(neighbors_deformed[1]) - imag(neighbors_deformed[0])) / h;
   double dvdy = (imag(neighbors_deformed[2]) - imag(neighbors_deformed[3])) / h;


   strainFile << "X,Y,F11,F12,F21,F22\n";
   strainFile << x << "," << y << ",";
   strainFile << dudx << "," << dudy << "," << dvdx << "," << dvdy << "\n";

   eigenvalueFile << x << "," << y << ", "; 
   eigenvalueFile2 << x << "," << y << ", ";
   eigenvalueFile3 << x << "," << y << ", ";
   storeEigens(dudx, dudy, dvdx, dvdy, eigenvalueFile,eigenvalueFile2,eigenvalueFile3);

   return vector<Vec2D>{ Vec2D{dudx, dudy}, Vec2D{dvdx, dvdy}};
}

Vec2D deformRect( Vec2D v ) {
   double x = real(v);
   double y = imag(v);
   return Vec2D(x + 0.4 * x * x, y );
}

string double_to_str(double f) {
   std::string str = std::to_string (f);
   str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
   str.erase ( str.find_last_not_of('.') + 1, std::string::npos );
   return str;
}

int main( int argc, char** argv ) {
   // string shape = "rect100_seed_1_65536";
   string shape = "trousers";
   double size = 1; 
   double h = 0.01;
   // string fileName = shape + "_size_" + double_to_str(size) + "_h_" + double_to_str(h);
   string fileName = shape; 
   std::cout << fileName << std::endl;
   auto deform = deformTrousers;

   srand( time(NULL) );

   int s = 16; // make it smaller to speed up
   auto start = high_resolution_clock::now(); // Added for timing

   // shape.csv
   std::ofstream origFile("../output/" + shape + ".csv");
   std::ofstream strainFile("../output/" + fileName + "_strain.csv");
   std::ofstream neighbourFile("../output/" + shape + "_neighbour_displacements.csv");
   std::ofstream displacementFile("../output/" + shape + "_displacements.csv");
   std::ofstream eigenvaluesFile2("../output/" + shape + "_eigenvalues2.csv");
   std::ofstream eigenvaluesFile3("../output/" + shape + "_eigenvalues3.csv");
   std::ofstream eigenvaluesFile("../output/" + shape + "_eigenvalues.csv");std::ofstream eigenvectorFile("../output/" + shape + "_eigenvectors.csv");

   for( int j = 0; j < s; j++ )
   {
      cerr << "row " << j << " of " << s << endl;
      for( int i = 0; i < s; i++ )
      {
         Vec2D x0(((double)i / (s - 1)),// * 2 * size - size,
                  ((double)j / (s - 1))// * 2 * size - size 
         );
         double x = real(x0);
         double y = imag(x0);
         double lam = 1.0;
         double mu = 1.0;
         Vec2D left{ x - h/2, y };
         Vec2D right{ x + h/2, y };
         Vec2D top{ x, y + h/2 };
         Vec2D bottom{ x, y - h/2 };
         Vec2D solved_vec = NAN;
         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann) && insideDomain(left, boundaryDirichlet, boundaryNeumann) && insideDomain(right, boundaryDirichlet, boundaryNeumann) && insideDomain(top, boundaryDirichlet, boundaryNeumann) && insideDomain(bottom, boundaryDirichlet, boundaryNeumann) ){
            origFile << x << "," << y << "," << real(x0) << "," << imag(x0) << "\n";
            getDeformationGradientAndStress(x0, h, deform, strainFile, neighbourFile, eigenvaluesFile, eigenvectorFile, eigenvaluesFile2, eigenvaluesFile3);
            solved_vec = solve(x0, boundaryDirichlet, boundaryNeumann, deform);
            displacementFile << real(solved_vec) << "," << imag(solved_vec) << "\n";
         }
      } 
   }
   auto stop = high_resolution_clock::now(); // Added for timing
   auto duration = duration_cast<milliseconds>(stop - start); // Added for timing
   cout << "Time taken by function: " << duration.count() << " milliseconds" << endl; // Added for timing
}