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

Vec2D solve( Vec2D x0, // evaluation point
   vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
   vector<Polyline> boundaryNeumann, // reflecting part of the boundary
   function<Vec2D(Vec2D)> g ) { // Dirichlet boundary values
      const double eps = 0.000001; // stopping tolerance
      const double rMin = 0.000001; // minimum step size
      const int nWalks = 1000000; // number of Monte Carlo samples
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
      
      // #pragma omp parallel for reduction(+:sum)
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
         auto p = distancePolylines( x, boundaryDirichlet );
         dDirichlet = p.first;
         closestPoint = p.second;
         dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
         r = max( rMin, min( dDirichlet, dSilhouette ));
         double theta = random( -M_PI, M_PI );
         if( onBoundary ) { // sample from a hemisphere around the normal
            theta = theta/2. + angleOf(n);
         }
         Vec2D v{ cos(theta), sin(theta) }; // unit ray direction
         x =  intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary );
         steps++;
         }
         while(dDirichlet > eps && steps < maxSteps);

         Vec2D eval_vec = g(x);

         if (isnan(real(eval_vec)) || isnan(imag(eval_vec))) {
            continue;
         }
         walker += 1;
         sum_x += real(eval_vec);
         sum_y += imag(eval_vec);
      } 
      return Vec2D(sum_x/walker, sum_y/walker);
}

vector<Vec2D> solveGradient( Vec2D x0, // evaluation point
   vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
   vector<Polyline> boundaryNeumann, // reflecting part of the boundary
   function<Vec2D(Vec2D)> g = [](Vec2D) { return Vec2D(0.0, 0.0); }, 
   std::ofstream& displacementFile = *(new std::ofstream()), 
   std::ofstream& gradientFile = *(new std::ofstream())) { // Dirichlet boundary values
   const double eps = 0.000001; // stopping tolerance
   const double rMin = 0.000001; // minimum step size
   const int nWalks = 100000;//100000000; // number of Monte Carlo samples
   const int maxSteps = 65536; // maximum walk length
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

   // #pragma omp parallel for reduction(+:sum)
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
   center = x;
   auto p = distancePolylines( x, boundaryDirichlet );
   dDirichlet = p.first;
   closestPoint = p.second;
   dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
   r = max( rMin, min( dDirichlet, dSilhouette ));

   // intersect a ray with the star-shaped region boundary
   double theta = random( -M_PI, M_PI );
   if( onBoundary ) { // sample from a hemisphere around the normal
   theta = theta/2. + angleOf(n);
   }
   Vec2D v{ cos(theta), sin(theta) }; // unit ray direction
   x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary );
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
   gradientFile << "X,Y,F11,F12,F21,F22\n";
   gradientFile << real(x0) << "," << imag(x0) << ",";
   gradientFile << sum_11/walker << "," << sum_12/walker << "," << sum_21/walker << "," << sum_22/walker << "\n";
   Vec2D row1 = Vec2D(sum_11/walker, sum_12/walker);
   Vec2D row2 = Vec2D(sum_21/walker, sum_22/walker);
   return {row1, row2};
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

double getNormalStress(vector<Vec2D> stressTensor, Vec2D normal){
   double normalStress = 0;
   vector<Vec2D> inter = vectorMatrixMultiply(normal, stressTensor);
   normalStress = real(inter[0]) * real(normal) + imag(inter[0]) * imag(normal);
   return normalStress;
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
