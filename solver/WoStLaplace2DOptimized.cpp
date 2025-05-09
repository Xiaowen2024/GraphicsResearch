// (Optimized) implementation of the "Walk on Stars" algorithm for Laplace equations.
// Corresponds to the estimator given in Equation 18 of Sawhney et al,
// "Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary
// Conditions" (2023), assuming no source term and zero-Neumann conditions.
// NOTE: this code makes a few shortcuts for the sake of code brevity; may
// be more suitable for tutorials than for production code/evaluation.
// To compile: c++ -std=c++17 -O3 -pedantic -Wall WoStLaplace2D.cpp -o wost
#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <omp.h> 
#include <chrono> 
#include <Eigen/Dense>
using namespace std;
using namespace std::chrono; 

const double infinity = numeric_limits<double>::infinity();

// returns a random value in the range [rMin,rMax]
double random( double rMin, double rMax ) {
   const double rRandMax = 1.0/(double)RAND_MAX;
   double u = rRandMax*(double)rand();
   return u*(rMax-rMin) + rMin;
}

// use Eigen::Vector2d for optimization
using Vec2D = Eigen::Vector2d;
using Polyline = vector<Vec2D>;
double length( Vec2D u ) { return sqrt( u.squaredNorm() ); }
double angleOf(Vec2D u) { return atan2(u.y(), u.x()); }
double angleOfTwo(Vec2D u, Vec2D v) { 
    double dot = u.x() * v.x() + u.y() * v.y(); 
    double det = u.x() * v.y() - u.y() * v.x(); 
    return std::atan2(det, dot); 
}
Vec2D rotate90( Vec2D u ) { return Vec2D( -u.y(), u.x() ); }
double   dot(Vec2D u, Vec2D v) { return u.dot(v); }
double cross(Vec2D u, Vec2D v) { return (u.x() * v.y() - u.y() * v.x());; }

// returns the closest point to x on a segment with endpoints a and b
Vec2D closestPoint( Vec2D x, Vec2D a, Vec2D b ) {
   Vec2D u = b-a;
   double t = clamp( dot(x-a,u)/u.squaredNorm(), 0.0, 1.0 );
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

// returns distance from x to closest point on the given polylines P
double distancePolylines( Vec2D x, const vector<Polyline>& P ) {
   double d = infinity; // minimum distance so far
   #pragma omp parallel for reduction(min:d)
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
   #pragma omp parallel for reduction(min:d)
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
double solve( Vec2D x0, // evaluation point
              vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
              vector<Polyline> boundaryNeumann, // reflecting part of the boundary
              function<double(Vec2D)> g ) { // Dirichlet boundary values
   const double eps = 0.0001; // stopping tolerance
   const double rMin = 0.0001; // minimum step size
   const int nWalks = 65536; // number of Monte Carlo samples
   const int maxSteps = 65536; // maximum walk length

   double sum = 0.0; // running sum of boundary contributions
   #pragma omp parallel for reduction(+:sum)
   for( int i = 0; i < nWalks; i++ ) {
      Vec2D x = x0; // start walk at the evaluation point
      Vec2D n{ 0.0, 0.0 }; // assume x0 is an interior point, and has no normal
      bool onBoundary = false; // flag whether x is on the interior or boundary

      double r, dDirichlet, dSilhouette; // radii used to define star shaped region
      int steps = 0;
      do { // loop until the walk hits the Dirichlet boundary
           
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

      sum += g(x); // accumulate contribution of the boundary value
   }
   return sum/nWalks; // Monte Carlo estimate
}

double lines( Vec2D x ) {
   const double s = 8.0;
   return fmod( floor(s*x.x()), 2.0 );
}

// for simplicity, in this code we assume that the Dirichlet and Neumann
// boundary polylines form a collection of closed polygons (possibly with holes),
// and are given with consistent counter-clockwise orientation
vector<Polyline> boundaryDirichlet = {
};
vector<Polyline> boundaryNeumann = {

};

// these routines are not used by WoSt itself, but are rather used to check
// whether a given evaluation point is actually inside the domain
double signedAngle( Vec2D x, const vector<Polyline>& P )
{
   double Theta = 0.;
   for (const auto& polyline : P) {
         for (size_t j = 0; j < polyline.size() - 1; j++) {
               Vec2D v1 = (polyline[j + 1] - x).normalized(); // Vector from x to P[i][j+1]
               Vec2D v2 = (polyline[j] - x).normalized();     // Vector from x to P[i][j]

               // Compute the angle between the two vectors
               Theta += angleOfTwo(v1, v2);
         }
      }
}

// Returns true if the point x is contained in the region bounded by the Dirichlet
// and Neumann curves.  We assume these curves form a collection of closed polygons,
// and are given in a consistent counter-clockwise winding order.
bool insideDomain( Vec2D x,
                   const vector<Polyline>& boundaryDirichlet,
                   const vector<Polyline>& boundaryNeumann )
{
   double Theta = signedAngle( x, boundaryDirichlet ) +
                  signedAngle( x, boundaryNeumann );
   const double delta = 1e-4; // numerical tolerance
   return abs(Theta-2.*M_PI) < delta; // boundary winds around x exactly once
}

double getSaddlePointHeight(Vec2D x) {
    return x.x() * x.x() - x.y() * x.y();
}

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

int main( int argc, char** argv ) {
   srand( time(NULL) );
   ofstream out( "saddlePointEigen.csv" );

   int s = 128; // image size
   createSaddlePointBoundary(-1., -1., 1., 1., 30, boundaryDirichlet);
   auto start = high_resolution_clock::now(); // Added for timing
   #pragma omp parallel for
   for( int j = 0; j < s; j++ )
   {
      cerr << "row " << j << " of " << s << endl;
      for( int i = 0; i < s; i++ )
      {
         Vec2D x0( ((double)i+.5)/((double)s),
                   ((double)j+.5)/((double)s) );
         double u = 0.;
         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann) )
            u = solve( x0, boundaryDirichlet, boundaryNeumann, getSaddlePointHeight );
         out << u;
         if( i < s-1 ) out << ",";
      }
      out << endl;
   }
   auto stop = high_resolution_clock::now(); // Added for timing
   auto duration = duration_cast<milliseconds>(stop - start); // Added for timing
   cout << "Time taken by function: " << duration.count() << " milliseconds" << endl; // Added for timing
   return 0;
}
