#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include "fractureModelHelpers.h"
using namespace std;

using Vec2D = complex<double>;
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
   return Vec2D(sum_x/nWalks, sum_y/nWalks);
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

vector<Vec2D> getDeformationGradientAndStress( Vec2D point, double h, function<Vec2D(Vec2D)> deform, std::ofstream& strainFile, std::ofstream& neighbourFile, std::ofstream& stressFile, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
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


vector<Vec2D> returnStress( Vec2D point, double h, function<Vec2D(Vec2D)> deform, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann) {
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
   std:: cout << "dudx: " << dudx << " dudy: " << dudy << " dvdx: " << dvdx << " dvdy: " << dvdy << std::endl;
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
   vector<pair<double, Vec2D>> eigenpairs = vector<pair<double, Vec2D>>{
      {lambda1, eigenvector1Normalized},
      {lambda2, eigenvector2Normalized}
   };
   sort (eigenpairs.begin(), eigenpairs.end(), [](const pair<double, Vec2D>& a, const pair<double, Vec2D>& b) {
      return a.first > b.first;
   });
   return eigenpairs;
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

float getNormalStress(vector<Vec2D> stressTensor, Vec2D normal){
   return real(normal) * real(normal) * real(stressTensor[0]) + imag(normal) * imag(normal) * imag(stressTensor[1]) + 2 * real(normal) * imag(normal) * imag(stressTensor[0]);
}

vector<Vec2D> getSeparationTensor(Vec2D tensileForce, Vec2D compressiveForce, vector<Vec2D> neighbourTensileForces, vector<Vec2D> neighbourCompressiveForces) {
   vector<Vec2D> sum =  matrixSubstract(getSymmetricMatrix(compressiveForce), getSymmetricMatrix(tensileForce));
   for (int i = 0; i < neighbourTensileForces.size(); i++) {
      sum = matrixAdd(sum, getSymmetricMatrix(neighbourTensileForces[i]));
      sum = matrixSubstract(sum, getSymmetricMatrix(neighbourCompressiveForces[i]));
   }
   return scalarMultiplyMatrix(1.0 / 2, sum);
}


Vec2D determineCrackPropagationDirection(vector<Vec2D> stressTensor) {
   vector<pair<double, Vec2D>> eigenpairs = eigenDecomposition(stressTensor);
   return eigenpairs[0].second;
}
