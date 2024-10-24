// (Slow) implementation of the "Walk on Stars" algorithm for Laplace equations.
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
#include <Eigen/Dense>
#include "WalkOnStars.h"
using namespace std;

WalkOnStars::WalkOnStars(const vector<Polyline>& boundaryDirichlet,
                         const vector<Polyline>& boundaryNeumann,
                         function<double(Vec2D)> interpolate)
    : boundaryDirichlet(boundaryDirichlet),
      boundaryNeumann(boundaryNeumann),
      interpolate(interpolate) {}
// the constant "infinity" is used as a maximum value in several calculations
const double infinity = numeric_limits<double>::infinity();

// returns a random value in the range [rMin,rMax]
double WalkOnStars::random(double rMin, double rMax) {
   mt19937 rng(random_device{}());
   uniform_real_distribution<double> distribution(rMin, rMax);
   return distribution(rng);
}

// use std::complex to implement 2D vectors
using Vec2D =  Eigen::Matrix<double, 2, 1>;

// returns the closest point to x on a segment with endpoints a and b
Vec2D WalkOnStars::closestPoint( Vec2D x, Vec2D a, Vec2D b ) {
   Vec2D u = b - a; // direction vector from a to b
   double t = (x - a).dot(u) / u.squaredNorm(); // projection factor
   t = clamp(t, 0.0, 1.0); // clamp t to the range [0, 1]
   return a + t * u; // return the closest point on the segment
}  

// returns true if the point b on the polyline abc is a silhoutte relative to x
bool WalkOnStars::isSilhouette( Vec2D x, Vec2D a, Vec2D b, Vec2D c ) {
   return (b-a).cross(x-a).norm() * (c-b).cross(x-b).norm() < 0;
}

// returns the time t at which the ray x+tv intersects segment ab,
// or infinity if there is no intersection
double WalkOnStars::rayIntersection( Vec2D x, Vec2D v, Vec2D a, Vec2D b ) {
   Vec2D u = b - a;
   Vec2D w = x - a;
   double d = v.cross(u).norm();
   double s = v.cross(w).norm() / d;
   double t = u.cross(w).norm() / d;
   if (t > 0. && 0. <= s && s <= 1.) {
      return t;
   }
   return infinity;
}

// boundary geometry is represented by polylines
using Polyline = vector<Vec2D>;

// returns distance from x to closest point on the given polylines P
double WalkOnStars::distancePolylines( Vec2D x, const vector<Polyline>& P ) {
   double d = infinity; // minimum distance so far
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 0; j < P[i].size()-1; j++ ) { // iterate over segments
         Vec2D y = closestPoint( x, P[i][j], P[i][j+1] ); // distance to segment
         d = min( d, (x-y).norm() ); // update minimum distance
      }
   }
   return d;
}

// returns distance from x to closest silhouette point on the given polylines P
double WalkOnStars::silhouetteDistancePolylines( Vec2D x, const vector<Polyline>& P ){
   double d = infinity; // minimum distance so far
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 1; j < P[i].size()-1; j++ ) { // iterate over segment pairs
         if( WalkOnStars::isSilhouette( x, P[i][j-1], P[i][j], P[i][j+1] )) {
            d = min( d, (x-P[i][j]).norm() ); // update minimum distance
         }
      }
   }
   return d;
}

// finds the first intersection y of the ray x+tv with the given polylines P,
// restricted to a ball of radius r around x.  The flag onBoundary indicates
// whether the first hit is on a boundary segment (rather than the sphere), and
// if so sets n to the normal at the hit point.
Vec2D WalkOnStars::intersectPolylines( Vec2D x, Vec2D v, double r,
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
            n = (P[i][j+1] - P[i][j]).cross(Vec2D(0,0,1)); // get normal
            n /= n.norm(); // make normal unit length
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
double WalkOnStars::solve( Vec2D x0, // evaluation point
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
            theta = theta/2. + atan2(n.y(), n.x());
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


// these routines are not used by WoSt itself, but are rather used to check
// whether a given evaluation point is actually inside the domain
double WalkOnStars::signedAngle( Vec2D x, const vector<Polyline>& P )
{
   double Theta = 0.;
   for( int i = 0; i < P.size(); i++ )
      for( int j = 0; j < P[i].size()-1; j++ )
         Theta += atan2( (P[i][j+1]-x).y(), (P[i][j+1]-x).x() ) - atan2( (P[i][j]-x).y(), (P[i][j]-x).x() );
   return Theta;
}



// Returns true if the point x is contained in the region bounded by the Dirichlet
// and Neumann curves.  We assume these curves form a collection of closed polygons,
// and are given in a consistent counter-clockwise winding order.
bool WalkOnStars::insideDomain( Vec2D x,
                   const vector<Polyline>& boundaryDirichlet,
                   const vector<Polyline>& boundaryNeumann )
{
   double Theta = signedAngle( x, boundaryDirichlet ) +
                  signedAngle( x, boundaryNeumann );
   const double delta = 1e-4; // numerical tolerance
   return abs(Theta-2.*M_PI) < delta; // boundary winds around x exactly once
}
