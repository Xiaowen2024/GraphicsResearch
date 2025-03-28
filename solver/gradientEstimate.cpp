
// c++ -std=c++17 -O3 -pedantic -Wall gradientEstimate.cpp -o gradientEstimate -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -w

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

std::random_device rd;  // Non-deterministic random number generator
std::mt19937 gen(rd()); // Mersenne Twister PRNG
std::uniform_real_distribution<double> dist(-M_PI, M_PI);

// the constant "infinity" is used as a maximum value in several calculations
const double infinity = numeric_limits<double>::infinity();

// returns a random value in the range [rMin,rMax]
double random(double rMin, double rMax) {
    static std::random_device rd;   // Seed generator (non-deterministic)
    static std::mt19937 gen(rd());  // Mersenne Twister RNG
    std::uniform_real_distribution<double> dist(rMin, rMax);
    return dist(gen);
}

// use std::complex to implement 2D vectors
using Vec2D = complex<double>;
double length( Vec2D u ) { return sqrt( norm(u) ); }
double angleOf(Vec2D u) { return arg(u); }
Vec2D rotate90( Vec2D u ) { return Vec2D( -imag(u), real(u) ); }
double   dot(Vec2D u, Vec2D v) { return real(u)*real(v) + imag(u)*imag(v); }
double cross(Vec2D u, Vec2D v) { return real(u)*imag(v) - imag(u)*real(v); }
vector<Vec2D> multiply(Vec2D A, Vec2D B) {
      vector<Vec2D> result;
      Vec2D row1 = {real(A) * real(B), real(A) * imag(B)};
      Vec2D row2 = {imag(A) * real(B), imag(A) * imag(B)};
      result.push_back(row1);
      result.push_back(row2);
      return result;
}

vector<Vec2D> transpose(vector<Vec2D> matrix) {
   vector<Vec2D> result;
   Vec2D row1 = {real(matrix[0]), real(matrix[1])};
   Vec2D row2 = {imag(matrix[0]), imag(matrix[1])};
   result.push_back(row1);
   result.push_back(row2);
   return result;
}

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

// // returns the time t at which the ray x+tv intersects segment ab,
// // or infinity if there is no intersection
// double rayIntersection( Vec2D x, Vec2D v, Vec2D a, Vec2D b ) {
//    Vec2D u = b - a;
//    Vec2D w = x - a;
//    double d = cross(v,u);
//    double s = cross(v,w) / d;
//    double t = cross(u,w) / d;
//    if (t > 0. && 0. <= s && s <= 1.) {
//       return t;
//    }
//    return infinity;
// }

Vec2D deformShear( Vec2D v ) {
   double x = real(v);
   double y = imag(v);
   return Vec2D( 0.42222 * x + 0.27822 * y, 0.12322 * x  + 0.62222  * y);
}

using Polyline = vector<Vec2D>;

double rayIntersection(Vec2D x, Vec2D v, Vec2D a, Vec2D b) {
    Vec2D u = b - a;
    Vec2D w = x - a;
    double d = cross(v, u);

    if (fabs(d) < 1e-12) {
        return infinity;  
    }

    double s = cross(v, w) / d;
    double t = cross(u, w) / d;

    // Check if intersection is within the segment bounds
    if (t > 0.0 && s >= 0.0 && s <= 1.0) {
        return t;
    }
    return infinity;  // No valid intersection
}


Vec2D intersectPolylines(Vec2D x, Vec2D v, double r,
                         const vector<Polyline>& P,
                         Vec2D& n, bool& onBoundary) {
    double tMin = r;  // smallest hit time so far
    n = Vec2D{ 0.0, 0.0 };  // first hit normal
    onBoundary = false;  // will be true only if the first hit is on a segment
    
    for (int i = 0; i < P.size(); i++) {  // iterate over polylines
        for (int j = 0; j < P[i].size() - 1; j++) {  // iterate over segments
            const double c = 1e-5;  // ray offset (to avoid self-intersection)
            double t = rayIntersection(x + c * v, v, P[i][j], P[i][j+1]);
            
            // Check if t is valid and closer than the current minimum
            if (t < tMin && t > 0) {
                tMin = t;
                Vec2D edge = P[i][j + 1] - P[i][j];
                
                // Calculate the normal only if the edge length is non-zero
                if (length(edge) > 1e-6) {
                    n = rotate90(edge);
                    n /= length(n);  // make normal unit length
                }
                
                onBoundary = true;
            }
        }
    }
    
    // Return the intersection point
    return x + tMin * v;
}



// returns distance from x to closest point on the given polylines P
pair<double, Vec2D> distancePolylines( Vec2D x, const vector<Polyline>& P ) {
   double d = infinity; // minimum distance so far
   // #pragma omp parallel for reduction(min:d)
   Vec2D y;
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 0; j < P[i].size()-1; j++ ) { // iterate over segments
         // distance to segment
         Vec2D temp = closestPoint( x, P[i][j], P[i][j+1] );
         if (d > length(x-temp)) {
            y = closestPoint( x, P[i][j], P[i][j+1] );
            d = min( d, length(x-y) ); // update minimum distance
         }
        
      }
   }
   return make_pair(d, y);
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

// // finds the first intersection y of the ray x+tv with the given polylines P,
// // restricted to a ball of radius r around x.  The flag onBoundary indicates
// // whether the first hit is on a boundary segment (rather than the sphere), and
// // if so sets n to the normal at the hit point.
// Vec2D intersectPolylines( Vec2D x, Vec2D v, double r,
//                          const vector<Polyline>& P,
//                          Vec2D& n, bool& onBoundary ) {
//    double tMin = r; // smallest hit time so far
//    n = Vec2D{ 0.0, 0.0 }; // first hit normal
//    onBoundary = false; // will be true only if the first hit is on a segment
//    for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
//       for( int j = 0; j < P[i].size()-1; j++ ) { // iterate over segments
//          const double c = 1e-5; // ray offset (to avoid self-intersection)
//          double t = rayIntersection( x + c*v, v, P[i][j], P[i][j+1] );
//          if( t < tMin ) { // closest hit so far
//             tMin = t;
//             n = rotate90( P[i][j+1] - P[i][j] ); // get normal
//             n /= length(n); // make normal unit length
//             onBoundary = true;
//          }
//       }
//    }
//    return x + tMin*v; // first hit location
// }
Vec2D solve( Vec2D x0, // evaluation point
   vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
   vector<Polyline> boundaryNeumann, // reflecting part of the boundary
   function<Vec2D(Vec2D)> g ) { // Dirichlet boundary values
const double eps = 0.000001; // stopping tolerance
const double rMin = 0.000001; // minimum step size
const int nWalks = 10000000; // number of Monte Carlo samples
const int maxSteps = 65536; // maximum walk length
double sum_x = 0.0; // running sum of boundary contributions
double sum_y = 0.0;
int i = 0;
int walker = 0;
int countX1 = 0;
int countY1 = 0;
int biggerX = 0;
int smallerX = 0;
int biggerY = 0;
int smallerY = 0;
// unsigned seed = 30;
// srand(seed);

#pragma omp parallel for reduction(+:sum)
double nextTheta = -1;
bool isStarting = true;
for( i = 0; i < nWalks; i++ ) {
std::mt19937 generator(i);  
std::uniform_real_distribution<double> dist(-M_PI, M_PI);
Vec2D x = x0; // start walk at the evaluation point
Vec2D n{ 0.0, 0.0 }; // assume x0 is an interior point, and has no normal
bool onBoundary = false; // flag whether x is on the interior or boundary
double r, dDirichlet, dSilhouette; // radii used to define star shaped region
int steps = 0;
Vec2D closestPoint;
do { 
// compute the radius of the largest star-shaped region
auto p = distancePolylines( x, boundaryDirichlet );
dDirichlet = p.first;
closestPoint = p.second;
// dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
r = max( rMin, dDirichlet);

// intersect a ray with the star-shaped region boundary
double theta;
if (isStarting){
 if (nextTheta != -1){
    theta = nextTheta;
    nextTheta = -1;
 } 
 else {
    theta = random( -M_PI, M_PI );
    nextTheta = 2 * M_PI - theta;
 } 
 isStarting = false;
} 

theta = random( -M_PI, M_PI );
if( onBoundary ) { // sample from a hemisphere around the normal
 theta = theta/2. + angleOf(n);
}
Vec2D v{ cos(theta), sin(theta) }; // unit ray direction
x = x + r * v; //intersectPolylines( x, v, r, boundaryDirichlet, n, onBoundary );
steps++;
}
while(dDirichlet > eps);

// x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary );

// std::cout << "dDirichlet: " << length(x - closestPoint) << std::endl;

//stop if we hit the Dirichlet boundary, or the walk is too long

// if( steps >= maxSteps ) cerr << "Hit max steps" << endl;

Vec2D eval_vec = g(x);
// if (real(closestPoint) == 0 && imag(closestPoint) != 0) {
//    countX1 += 1;
// }
// if (real(closestPoint) != 0 && imag(closestPoint) == 0) {
//    countY1 += 1;
// }

// if (real(x) > 0.1){
//    biggerX += 1;
// }
// else if (real(x) < 0.1){
//    smallerX += 1;
// }

// if (imag(x) > 0.1){
//    biggerY += 1;
// }
// else if (imag(x) < 0.1){
//    smallerY += 1;
// }
// std::cout << "x0: " << real(x0) << ", " << imag(x0) << std::endl;
// std::cout << "closestPoint: " << real(closestPoint) << ", " << imag(closestPoint) << std::endl;
// std::cout << "x: " << real(eval_vec) << ", " << imag(eval_vec) << std::endl;
// assert (!isnan(real(eval_vec)) && !isnan(imag(eval_vec)));
// assert (real(eval_vec) == real(closestPoint) && imag(eval_vec) == imag(closestPoint));

// get rid of nan
if (isnan(real(eval_vec)) || isnan(imag(eval_vec))) {
continue;
}
walker += 1;
sum_x += real(eval_vec);
sum_y += imag(eval_vec);
} 
// cout << "count x1: " << countX1 << ", count y1: " << countY1 << endl;
// cout << "biggerX: " << biggerX << ", smallerX: " << smallerX << endl;
// cout << "biggerY: " << biggerY << ", smallerY: " << smallerY << endl;
// std::cout << "sum_x: " << sum_x << "sum_y: " << sum_y << "walker: " << walker << std::endl;
return Vec2D(sum_x/walker, sum_y/walker);
}


vector<Vec2D> solveGradient( Vec2D x0, // evaluation point
              vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
              vector<Polyline> boundaryNeumann, // reflecting part of the boundary
              function<Vec2D(Vec2D)> g, std::ofstream& gradientFile) { // Dirichlet boundary values
   const double eps = 0.000001; // stopping tolerance
   const double rMin = 0.000001; // minimum step size
   const int nWalks = 100000000; // number of Monte Carlo samples
   const int maxSteps = 65536; // maximum walk length
   double sum_11 = 0.0; 
   double sum_12 = 0.0;
   double sum_21 = 0.0; 
   double sum_22 = 0.0;
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
      double raidus = 0;
      do { 
         // compute the radius of the largest star-shaped region
         center = x;
         auto p = distancePolylines( x, boundaryDirichlet );
         dDirichlet = p.first;
         closestPoint = p.second;
         // dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
         r = max( rMin, dDirichlet);

         // intersect a ray with the star-shaped region boundary
         double theta;
         // if (isStarting){
         //    if (nextTheta != -1){
         //       theta = nextTheta;
         //       count += 1;
         //       nextTheta = -1;
         //    }
         //    else {
         //          int zone = nWalks % 6;
         //          double angle = 2 * M_PI / 6;
         //          theta = random( -M_PI + angle * zone , -M_PI + angle * (zone + 1));
         //          nextTheta = theta + M_PI;
         //    }
         //    isStarting = false;
         // } 
         // else {
         //    int zone = nWalks % 6;
         //    double angle = 2 * M_PI / 6;
         //    theta = random( -M_PI + angle * zone , -M_PI + angle * (zone + 1));
         // } 
         // int zone = nWalks % 6;
         // double angle = 2 * M_PI / 6;
         // theta = random( -M_PI + angle * zone , -M_PI + angle * (zone + 1));
         theta = random( -M_PI, M_PI );
         if( onBoundary ) { // sample from a hemisphere around the normal
            theta = theta/2. + angleOf(n);
         }
         Vec2D v{ cos(theta), sin(theta) }; // unit ray direction
         x = x + r * v; //intersectPolylines( x, v, r, boundaryDirichlet, n, onBoundary );
         if (isStarting){
            isStarting = false;
            normal = v / length(v);
            raidus = dDirichlet;
         }

         steps++;
      }
      while(dDirichlet > eps && steps < maxSteps);

      if( steps >= maxSteps ) continue;
      Vec2D estimated_u = g(closestPoint);
      vector<Vec2D> result = multiply(estimated_u, normal);
      result = { Vec2D(2 * 1/raidus * real(result[0]), 2 * 1/raidus * imag(result[0])),
      Vec2D(2 * 1/raidus * real(result[1]), 2 * 1/raidus * imag(result[1])) };
     
      if (isnan(real(estimated_u)) || isnan(imag(estimated_u))) {
         continue;
      }
      walker += 1;
      sum_11 += real(result[0]);
      sum_12 += imag(result[0]);
      sum_21 += real(result[1]);
      sum_22 += imag(result[1]);
   } 

   gradientFile << "X,Y,F11,F12,F21,F22\n";
   gradientFile << real(x0) << "," << imag(x0) << ",";
   gradientFile << sum_11/walker << "," << sum_12/walker << "," << sum_21/walker << "," << sum_22/walker << "\n";
   Vec2D row1 = Vec2D(sum_11/walker, sum_12/walker);
   Vec2D row2 = Vec2D(sum_21/walker, sum_22/walker);
   return {row1, row2};
}

// for simplicity, in this code we assume that the Dirichlet and Neumann
// boundary polylines form a collection of closed polygons (possibly with holes),
// and are given with consistent counter-clockwise orientation
vector<Polyline> boundaryDirichlet =    {{ Vec2D(0, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0) }};


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

Vec2D displacement(Vec2D v) { 
   // vector<Polyline> boundaryDirichlet = {   {{ Vec2D(0, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0) }}};
   // vector<Polyline> displacedPoints = {
   //    {
   //       {Vec2D(0, -0.1), Vec2D(1, -0.2), Vec2D(1, 1.2), Vec2D(0, 1.1), Vec2D(0, -0.1)}
   //    }
   // }; 
   //  vector<Polyline> displacedPoints = {
   //    {
   //       {Vec2D(-0.1, 0), Vec2D(1.2, 0), Vec2D(1.1, 1), Vec2D(-0.2, 1), Vec2D(-0.1, 0)}
   //    }
   // }; 
   
   vector<Polyline> displacedPoints = {   {{ Vec2D(0, 0), Vec2D(1.4, -0.4), Vec2D(1.2, 1.4), Vec2D(0, 1), Vec2D(0, 0) }}};

   // create vector polyline of displacement vectors from boundary points
   // vector<Polyline> displacementVectors = {{{}}};
   // for (int i = 0; i < boundaryDirichlet[0].size(); i++) {
   //    Vec2D point = boundaryDirichlet[0][i];
   //    Vec2D deformed_vec = displacedPoints[0][i];
   //    Vec2D displacement_vec = deformed_vec - point;
   //    displacementVectors[0].push_back(displacement_vec);
   // }
   

   Vec2D nan = numeric_limits<double>::quiet_NaN();

   Vec2D interpolatedDisplacement = interpolateVec2D_BoundaryPoints(v, boundaryDirichlet, displacedPoints);
   if (isnan(real(interpolatedDisplacement)) || isnan(imag(interpolatedDisplacement))) {
      return nan;
   }
   // std::cout << " original point: " << real(v) << ", " << imag(v) << std::endl;
   // std::cout << " interpolated displacement: " << real(interpolatedDisplacement) << ", " << imag(interpolatedDisplacement) << std::endl;
   return interpolatedDisplacement;
}

Vec2D deformFunc( Vec2D v ) {
  return v;
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

   

   // std::cout << "original x, y: " << x << ", " << y << std::endl;

   // std::cout << "original left: " << real(neighbors[0]) << ", " << imag(neighbors[0]) << std::endl;
   // std::cout << "deformed left: " << real(neighbors_deformed[0]) << ", " << imag(neighbors_deformed[0]) << std::endl;
   // std::cout << "original right: " << real(neighbors[1]) << ", " << imag(neighbors[1]) << std::endl;
   // std::cout << "deformed right: " << real(neighbors_deformed[1]) << ", " << imag(neighbors_deformed[1]) << std::endl;
   // std::cout << "original top: " << real(neighbors[2]) << ", " << imag(neighbors[2]) << std::endl;
   // std::cout << "deformed top: " << real(neighbors_deformed[2]) << ", " << imag(neighbors_deformed[2]) << std::endl;
   // std::cout << "original bottom: " << real(neighbors[3]) << ", " << imag(neighbors[3]) << std::endl;
   // std::cout << "deformed bottom: " << real(neighbors_deformed[3]) << ", " << imag(neighbors_deformed[3]) << std::endl;
  
   neighbourFile << "leftX_d, leftY_d, rightX_d rightY_d, topX_d, topY_d, bottomX_d, bottomY_d\n";
   neighbourFile << real(neighbors_deformed[0]) << "," << imag(neighbors_deformed[0]) << ",";
   neighbourFile << real(neighbors_deformed[1]) << "," << imag(neighbors_deformed[1]) << ",";
   neighbourFile << real(neighbors_deformed[2]) << "," << imag(neighbors_deformed[2]) << ",";
   neighbourFile << real(neighbors_deformed[3]) << "," << imag(neighbors_deformed[3]) << "\n";

   
   double dudx = (8 * real(neighbors_deformed[1]) - 8 * real(neighbors_deformed[0]) + real(neighbors_deformed[4]) - real(neighbors_deformed[5])) / (6 * h);
   double dudy = (8 * real(neighbors_deformed[2]) - 8 * real(neighbors_deformed[3]) + real(neighbors_deformed[7]) - real(neighbors_deformed[6])) / (6 * h);
   double dvdx = (8 * imag(neighbors_deformed[1]) - 8 * imag(neighbors_deformed[0]) + imag(neighbors_deformed[4]) - imag(neighbors_deformed[5])) / (6 * h);
   double dvdy = (8 * imag(neighbors_deformed[2]) - 8 * imag(neighbors_deformed[3]) + imag(neighbors_deformed[7]) - imag(neighbors_deformed[6])) / (6 * h);
   // double dudx = (real(neighbors_deformed[1]) - real(neighbors_deformed[0])) / h;
   // double dudy = (real(neighbors_deformed[2]) - real(neighbors_deformed[3])) / h;
   // double dvdx = (imag(neighbors_deformed[1]) - imag(neighbors_deformed[0])) / h;
   // double dvdy = (imag(neighbors_deformed[2]) - imag(neighbors_deformed[3])) / h;

   // if (dudx != 1.0 || dudy != 0.0 || dvdx != 0.0 || dvdy != 1.0) {
   //    std::cout << "dudx: " << dudx << ", dudy: " << dudy << ", dvdx: " << dvdx << ", dvdy: " << dvdy << std::endl;
   //    std::cout << "original x, y: " << x << ", " << y << std::endl;
   //    std::cout << "original left: " << real(neighbors[0]) << ", " << imag(neighbors[0]) << std::endl;
   //    std::cout << "deformed left: " << real(neighbors_deformed[0]) << ", " << imag(neighbors_deformed[0]) << std::endl;
   //    std::cout << "original right: " << real(neighbors[1]) << ", " << imag(neighbors[1]) << std::endl;
   //    std::cout << "deformed right: " << real(neighbors_deformed[1]) << ", " << imag(neighbors_deformed[1]) << std::endl;
   //    std::cout << "original top: " << real(neighbors[2]) << ", " << imag(neighbors[2]) << std::endl;
   //    std::cout << "deformed top: " << real(neighbors_deformed[2]) << ", " << imag(neighbors_deformed[2]) << std::endl;
   //    std::cout << "original bottom: " << real(neighbors[3]) << ", " << imag(neighbors[3]) << std::endl;
   //    std::cout << "deformed bottom: " << real(neighbors_deformed[3]) << ", " << imag(neighbors_deformed[3]) << std::endl;
   // }

   strainFile << "X,Y,F11,F12,F21,F22\n";
   strainFile << x << "," << y << ",";
   strainFile << dudx << "," << dudy << "," << dvdx << "," << dvdy << "\n";

   eigenvalueFile << x << "," << y << ", "; 
   eigenvalueFile2 << x << "," << y << ", ";
   eigenvalueFile3 << x << "," << y << ", ";
   // storeEigens(dudx, dudy, dvdx, dvdy, eigenvalueFile, eigenvalueFile2, eigenvalueFile3);

   return vector<Vec2D>{ Vec2D{dudx, dudy}, Vec2D{dvdx, dvdy}};
}

// Vec2D deformRect( Vec2D v ) {
//    double x = real(v);
//    double y = imag(v);
//    return Vec2D(x + 0.4 * x * x, y );
// }

string double_to_str(double f) {
   std::string str = std::to_string (f);
   str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
   str.erase ( str.find_last_not_of('.') + 1, std::string::npos );
   return str;
}

int main( int argc, char** argv ) {
   string shape = "gradient_estimate_displace";
   // // double size = 1; 
   double h = 0.01;
   // // // string fileName = shape + "_size_" + double_to_str(size) + "_h_" + double_to_str(h);
   string fileName = shape; 
   auto deform = displacement;

   int s = 16; // make it smaller to speed up
   // // auto start = high_resolution_clock::now(); // Added for timing

   //  shape.csv
   std::ofstream origFile("../output/" + shape + ".csv");
   std::ofstream gradientFile("../output/" + fileName + "deformation_gradient.csv");
   std::ofstream neighbourFile("../output/" + shape + "_neighbour_displacements.csv");
   std::ofstream displacementFile("../output/" + shape + "_displacements.csv");
   std::ofstream eigenvaluesFile2("../output/" + shape + "_eigenvalues2.csv");
   std::ofstream eigenvaluesFile3("../output/" + shape + "_eigenvalues3.csv");
   std::ofstream eigenvaluesFile("../output/" + shape + "_eigenvalues.csv");
   std::ofstream eigenvectorFile("../output/" + shape + "_eigenvectors.csv");

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
         Vec2D left{ x - h/2, y };
         Vec2D right{ x + h/2, y };
         Vec2D top{ x, y + h/2 };
         Vec2D bottom{ x, y - h/2 };
         Vec2D solved_vec = NAN;
         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann) && insideDomain(left, boundaryDirichlet, boundaryNeumann) && insideDomain(right, boundaryDirichlet, boundaryNeumann) && insideDomain(top, boundaryDirichlet, boundaryNeumann) && insideDomain(bottom, boundaryDirichlet, boundaryNeumann) ){
            vector<Vec2D> gradient = solveGradient(x0, boundaryDirichlet, boundaryNeumann, deform, gradientFile);
         }
      } 
   }
   // auto stop = high_resolution_clock::now(); // Added for timing
   // auto duration = duration_cast<milliseconds>(stop - start); // Added for timing
   // cout << "Time taken by function: " << duration.count() << " milliseconds" << endl; // Added for timing
   // Vec2D solved = solve(Vec2D(0.1, 0.1), boundaryDirichlet, boundaryNeumann, deformFunc);
   // vector<Vec2D> gradients = getDeformationGradientAndStress(Vec2D{0.1, 0.1}, 0.001, deformFunc, strainFile, neighbourFile, eigenvaluesFile, eigenvectorFile, eigenvaluesFile2, eigenvaluesFile3);
   // cout << "dudx: " << real(gradients[0]) << " dudy: " << imag(gradients[0]) << " dvdx: " << real(gradients[1]) << " dvdy: " << imag(gradients[1]) << endl;
   // cout << "Solved: (" << real(solved) << ", " << imag(solved) << ")" << endl;
   // vector<Vec2D> gradients[30];
   // vector<Vec2D> points;
   // std::uniform_real_distribution<double> dist(0.01, 0.99);
   // for (int i = 0; i < 30; ++i) {
   //    double x, y;
   //    do {
   //       x = dist(gen);
   //       y = dist(gen);
   //    } while (x == y);
   //    points.emplace_back(x, y);
   // }
   
   // int sampleSize = 30;
   // #pragma omp parallel for
   // for (int i = 0; i < sampleSize; ++i) {
   //     gradients[i] = getDeformationGradientAndStress(points[i], 0.1, deformFunc, strainFile, neighbourFile, eigenvaluesFile, eigenvectorFile, eigenvaluesFile2, eigenvaluesFile3);
   //     cout << "Point: (" << real(points[i]) << ", " << imag(points[i]) << ") ";
   //     cout << "dudx: " << real(gradients[i][0]) << " dudy: " << imag(gradients[i][0]) << " dvdx: " << real(gradients[i][1]) << " dvdy: " << imag(gradients[i][1]) << endl;
   // }

   // double sum_dudx = 0.0, sum_dudy = 0.0, sum_dvdx = 0.0, sum_dvdy = 0.0;
   // for (int i = 0; i < sampleSize; ++i) {
   //     sum_dudx += real(gradients[i][0]);
   //     sum_dudy += imag(gradients[i][0]);
   //     sum_dvdx += real(gradients[i][1]);
   //     sum_dvdy += imag(gradients[i][1]);
   // }

   // double mean_dudx = sum_dudx / sampleSize;
   // double mean_dudy = sum_dudy / sampleSize;
   // double mean_dvdx = sum_dvdx / sampleSize;
   // double mean_dvdy = sum_dvdy / sampleSize;

   // double var_dudx = 0.0, var_dudy = 0.0, var_dvdx = 0.0, var_dvdy = 0.0;
   // for (int i = 0; i < sampleSize; ++i) {
   //     var_dudx += (real(gradients[i][0]) - mean_dudx) * (real(gradients[i][0]) - mean_dudx);
   //     var_dudy += (imag(gradients[i][0]) - mean_dudy) * (imag(gradients[i][0]) - mean_dudy);
   //     var_dvdx += (real(gradients[i][1]) - mean_dvdx) * (real(gradients[i][1]) - mean_dvdx);
   //     var_dvdy += (imag(gradients[i][1]) - mean_dvdy) * (imag(gradients[i][1]) - mean_dvdy);
   // }

   // double stddev_dudx = sqrt(var_dudx / sampleSize);
   // double stddev_dudy = sqrt(var_dudy / sampleSize);
   // double stddev_dvdx = sqrt(var_dvdx / sampleSize);
   // double stddev_dvdy = sqrt(var_dvdy / sampleSize);

   // cout << "Mean dudx: " << mean_dudx << " dudy: " << mean_dudy << " dvdx: " << mean_dvdx << " dvdy: " << mean_dvdy << endl;
   // cout << "Standard Deviation dudx: " << stddev_dudx << " dudy: " << stddev_dudy << " dvdx: " << stddev_dvdx << " dvdy: " << stddev_dvdy << endl;

   // const int numRuns = 1000;
   // vector<Vec2D> results;
   // #pragma omp parallel for
   // for (int i = 0; i < numRuns; ++i) {
   //     Vec2D solved = solve(Vec2D(0.1, 0.1), boundaryDirichlet, boundaryNeumann, deformFunc);
   //     results.push_back(solved);
   // }

   // double sumReal = 0.0, sumImag = 0.0;
   // for (const auto& res : results) {
   //     sumReal += real(res);
   //     sumImag += imag(res);
   // }

   // double meanReal = sumReal / numRuns;
   // double meanImag = sumImag / numRuns;

   // double varianceReal = 0.0, varianceImag = 0.0;
   // for (const auto& res : results) {
   //     varianceReal += (real(res) - meanReal) * (real(res) - meanReal);
   //     varianceImag += (imag(res) - meanImag) * (imag(res) - meanImag);
   // }

   // varianceReal /= numRuns;
   // varianceImag /= numRuns;

   // cout << "Mean: (" << meanReal << ", " << meanImag << ")" << endl;
   // cout << "Variance: (" << varianceReal << ", " << varianceImag << ")" << endl;

   // Calculate RMSE between estimation and (0.1, 0.1)
   // double rmseReal = 0.0, rmseImag = 0.0;
   // for (const auto& res : results) {
   //     rmseReal += (real(res) - 0.1) * (real(res) - 0.1);
   //     rmseImag += (imag(res) - 0.1) * (imag(res) - 0.1);
   // }

   // rmseReal = sqrt(rmseReal / numRuns);
   // rmseImag = sqrt(rmseImag / numRuns);

   // cout << "RMSE: (" << rmseReal << ", " << rmseImag << ")" << endl;
   // vector<Vec2D> gradient = solveGradient(Vec2D(0.85, 0.85), boundaryDirichlet, boundaryNeumann, deformShear);
   // vector<Vec2D> trans = transpose(gradient);
   // vector<Vec2D> deformationGradient = { Vec2D{real(gradient[0]) + real(trans[0]), imag(gradient[0]) + imag(gradient[0])}, Vec2D{real(gradient[1]) + real(trans[1]), imag(gradient[1]) + imag(gradient[1])} };
   // cout << "dudx: " << real(deformationGradient[0]) << " dudy: " << imag(deformationGradient[0]) << " dvdx: " << real(deformationGradient[1]) << " dvdy: " << imag(deformationGradient[1]) << endl;
   // return 0;
//    vector<Vec2D> gradients = getDeformationGradientAndStress(Vec2D{0.1, 0.1}, 0.01, deformShear, strainFile, neighbourFile, eigenvaluesFile, eigenvectorFile, eigenvaluesFile2, eigenvaluesFile3);
//    cout << "dudx: " << real(gradients[0]) << " dudy: " << imag(gradients[0]) << " dvdx: " << real(gradients[1]) << " dvdy: " << imag(gradients[1]) << endl;
}
