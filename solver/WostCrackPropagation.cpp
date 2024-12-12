// (Optimized) implementation of the "Walk on Stars" algorithm for Laplace equations.
// Corresponds to the estimator given in Equation 18 of Sawhney et al,
// "Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary
// Conditions" (2023), assuming no source term and zero-Neumann conditions.
// NOTE: this code makes a few shortcuts for the sake of code brevity; may
// be more suitable for tutorials than for production code/evaluation.
// To compile: c++ -std=c++17 -O3 -pedantic -Wall WoStCrackPropagation.cpp -o wost-cp

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
double dot(Vec2D u, Vec2D v) { return real(u)*real(v) + imag(u)*imag(v); }
double cross(Vec2D u, Vec2D v) { return real(u)*imag(v) - imag(u)*real(v); }
vector<Vec2D> outerProduct(Vec2D u, Vec2D v) {
   vector<Vec2D> result;
   return vector<Vec2D>{ Vec2D(real(u) * real(v), real(u) * imag(v)), Vec2D(imag(u) * real(v), imag(u) * imag(v)) };
}
vector<Vec2D> matrixAdd(vector<Vec2D> A, vector<Vec2D> B) {
   vector<Vec2D> result = {};
   for (int i = 0; i < A.size(); i++) {
      result.push_back(A[i] + B[i]);
   }
   return result;
}
vector<Vec2D> matrixSubstract(vector<Vec2D> A, vector<Vec2D> B) {
   vector<Vec2D> result = {};
   for (int i = 0; i < A.size(); i++) {
      result.push_back(A[i] - B[i]);
   }
   return result;
}
vector<Vec2D> matrixMultiply(vector<Vec2D> A, vector<Vec2D> B) {
   vector<Vec2D> result = {};
   for (int i = 0; i < A.size(); i++) {
      result.push_back(Vec2D(dot(A[i], B[0]), dot(A[i], B[1])));
   }
   return result;
}

Vec2D matrixVectorMultiply(vector<Vec2D> A, Vec2D v) {
   return Vec2D(dot(A[0], v), dot(A[1], v));
}

vector<Vec2D> scalarMultiplyMatrix(double scalar, vector<Vec2D> matrix) {
   vector<Vec2D> result;
   for (int i = 0; i < matrix.size(); i++) {
      result.push_back(scalar * matrix[i]);
   }
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

// boundary geometry is represented by polylines
using Polyline = vector<Vec2D>;
vector<Polyline> newBoundaryDirichlet = {
   {
      {Vec2D(-0.2, 0), Vec2D(0.3, 0), Vec2D(0.5, 0.5), Vec2D(0.7, 0), Vec2D(1.2, 0), Vec2D(1.0, 1), Vec2D(0, 1), Vec2D(-0.2, 0)}
   }
};


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
   std::cout << i << std::endl;
   return Vec2D(sum_x/nWalks, sum_y/nWalks);
}

// for simplicity, in this code we assume that the Dirichlet and Neumann
// boundary polylines form a collection of closed polygons (possibly with holes),
// and are given with consistent counter-clockwise orientation
// vector<Polyline> boundaryDirichlet = {   {{ Vec2D(0, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0) }}};
// for crack propagation shape 
vector<Polyline> boundaryDirichlet = {   {{ Vec2D(0, 0), Vec2D(100, 0), Vec2D(100, 100), Vec2D(0, 100), Vec2D(0, 0) }}};
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
Vec2D deformCrackPropagation( Vec2D v ) {
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


vector<Vec2D> getStress(double lam, double mu, double trace, double dudx, double dudy, double dvdx, double dvdy) {
   return  matrixAdd(vector<Vec2D>{Vec2D{ 1 * lam * trace , 0}, Vec2D{0, 1 * lam * trace}}, vector<Vec2D>{Vec2D{2 * mu * dudx, 2 * mu * dudy}, Vec2D{2 * mu * dvdx, 2 * mu * dvdy}});
}


vector<Vec2D> getSymmetricMatrix(Vec2D v) {
   if (sqrt(norm(v)) == 0) {
      return vector<Vec2D>{Vec2D{0,  0}, Vec2D{0, 0}};
   }
   else {
      vector<Vec2D> op = outerProduct(v, v);
      double length = sqrt(norm(v));
      return vector<Vec2D>{Vec2D{real(op[0]) / length, imag(op[0]) / length}, Vec2D{real(op[1]) / length, imag(op[1]) / length}};
   }
}

vector<Vec2D> getDeformationGradientAndStress( Vec2D point, double h, function<Vec2D(Vec2D)> deform, std::ofstream& strainFile, std::ofstream& neighbourFile, std::ofstream& stressFile) {
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

   vector<Vec2D> stress = getStress(1.0, 0.1, dudx + dvdy, dudx, dudy, dvdx, dvdy);
   stressFile << "X,Y,Stress\n";
   stressFile << x << "," << y << "," << real(stress[0]) << imag(stress[0]) << real(stress[1]) << imag(stress[1]) << "\n";

   return stress;
}

vector<Vec2D> returnDeformationGradientAndStress( Vec2D point, double h, function<Vec2D(Vec2D)> deform) {
   double x = real(point);
   double y = imag(point);
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   Vec2D solved_vec = nan; 
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
   double dudx = (real(neighbors_deformed[1]) - real(neighbors_deformed[0])) / h;
   double dudy = (real(neighbors_deformed[2]) - real(neighbors_deformed[3])) / h;
   double dvdx = (imag(neighbors_deformed[1]) - imag(neighbors_deformed[0])) / h;
   double dvdy = (imag(neighbors_deformed[2]) - imag(neighbors_deformed[3])) / h;
   vector<Vec2D> stress = getStress(1.0, 0.1, dudx + dvdy, dudx, dudy, dvdx, dvdy);
   return stress;
}



vector<pair<double, Vec2D>> eigenDecomposition(vector<Vec2D> A) {
   double a = 1;
   double b = -real(A[0]) - imag(A[1]);
   double c = real(A[0]) * imag(A[1]) - imag(A[0]) * real(A[1]);
   double lambda1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
   double lambda2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
   Vec2D eigenvector1 = Vec2D(lambda1 - real(A[0]), -imag(A[0]));
   Vec2D eigenvector1Normalized = eigenvector1 / length(eigenvector1);
   Vec2D eigenvector2 = Vec2D(lambda2 - real(A[0]), -imag(A[0]));
   Vec2D eigenvector2Normalized = eigenvector2 / length(eigenvector2);
   return vector<pair<double, Vec2D>>{
      {lambda1, eigenvector1Normalized},
      {lambda2, eigenvector2Normalized}
   };
}

pair<vector<Vec2D>, vector<Vec2D>> forceDecomposition(vector<Vec2D> stress, vector<pair<double, Vec2D>> eigenpairs) {
   vector<Vec2D> m1 = getSymmetricMatrix(eigenpairs[0].second);
   vector<Vec2D> m2 = getSymmetricMatrix(eigenpairs[1].second);
   vector<Vec2D> tensileComponents = matrixAdd(scalarMultiplyMatrix(max(0.0, eigenpairs[0].first), m1), scalarMultiplyMatrix(max(0.0, eigenpairs[1].first), m2));
   vector<Vec2D> compressiveComponents = matrixAdd(scalarMultiplyMatrix(min(0.0, eigenpairs[0].first), m1), scalarMultiplyMatrix(min(0.0, eigenpairs[1].first), m2));
   return {tensileComponents, compressiveComponents};
}

Vec2D getDirectHomogenousForce(vector<Vec2D> stressComponent, Vec2D normal) {
   return matrixVectorMultiply(stressComponent, normal);
}

vector<Vec2D> getSeparationTensor(Vec2D tensileForce, Vec2D compressiveForce, vector<Vec2D> neighbourTensileForces, vector<Vec2D> neighbourCompressiveForces) {
   vector<Vec2D> sum =  matrixSubstract(getSymmetricMatrix(compressiveForce), getSymmetricMatrix(tensileForce));
   for (int i = 0; i < neighbourTensileForces.size(); i++) {
      sum = matrixAdd(sum, getSymmetricMatrix(neighbourTensileForces[i]));
      sum = matrixSubstract(sum, getSymmetricMatrix(neighbourCompressiveForces[i]));
   }
   return scalarMultiplyMatrix(1.0 / 2, sum);
}

Vec2D determineCrackPropagationDirection(vector<Vec2D> separationTensor, double threshold) {
   vector<pair<double, Vec2D>> eigenpairs = eigenDecomposition(separationTensor);
   sort(eigenpairs.begin(), eigenpairs.end(), [](const pair<double, Vec2D>& a, const pair<double, Vec2D>& b) {
      return a.first > b.first;
   });
   if  (eigenpairs[0].first > threshold) {
      return eigenpairs[0].second;
   }
   else {
      return Vec2D(0, 0);
   }
}

Vec2D deform( Vec2D v ) {
   double x = real(v);
   double y = imag(v);
   return Vec2D(x + 0.4 * x * x, y );
}


Vec2D getForce(Vec2D position, Vec2D normal, function<Vec2D(Vec2D)> deform, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
   double x = real(position);
   double y = imag(position);
   double h = 10;
   Vec2D left{ x - h/2, y };
   Vec2D right{ x + h/2, y };
   Vec2D top{ x, y + h/2 };
   Vec2D bottom{ x, y - h/2 };
   if( insideDomain(position, boundaryDirichlet, boundaryNeumann) && insideDomain(left, boundaryDirichlet, boundaryNeumann) && insideDomain(right, boundaryDirichlet, boundaryNeumann) && insideDomain(top, boundaryDirichlet, boundaryNeumann) && insideDomain(bottom, boundaryDirichlet, boundaryNeumann) ){
      vector<Vec2D> stress = returnDeformationGradientAndStress(position, h, deform);
      vector<pair<double, Vec2D>> eigenpairs = eigenDecomposition(stress);
      pair<vector<Vec2D>, vector<Vec2D>> forcePairs = forceDecomposition(stress, eigenpairs);
      vector<Vec2D> tensileComponents = forcePairs.first;
      cout << "tensile components" << tensileComponents[0] << " " << tensileComponents[1] << endl;
      vector<Vec2D> compressiveComponents = forcePairs.second;
      cout << "compressive components" << compressiveComponents[0] << " " << compressiveComponents[1] << endl;
      Vec2D tensileForce = getDirectHomogenousForce(tensileComponents, Vec2D{-1, 0});
      cout << "tensile force" << tensileForce << endl;
      return tensileForce;
   }
   return Vec2D(0, 0);
}

int main(){
   return 0;
}