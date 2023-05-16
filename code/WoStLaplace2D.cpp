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
using namespace std;

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

// boundary geometry is represented by polylines
using Polyline = vector<Vec2D>;

// returns distance from x to closest point on the given polylines P
double distancePolylines( Vec2D x, const vector<Polyline>& P ) {
   double d = infinity; // minimum distance so far
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
// restricted to a ball of radius r around x; also gives the unit normal n
// at y, or n=(0,0) if the ray first hits the boundary of the ball
Vec2D intersectPolylines( Vec2D x, Vec2D v, double r,
                         const vector<Polyline>& P,
                         Vec2D& n ) {
   double tMin = r; // smallest hit time so far
   n = Vec2D{ 0.0, 0.0 }; // first hit normal
   for( int i = 0; i < P.size(); i++ ) { // iterate over polylines
      for( int j = 0; j < P[i].size()-1; j++ ) { // iterate over segments
         double t = rayIntersection( x, v, P[i][j], P[i][j+1] );
         if( t < tMin ) { // closest hit so far
            tMin = t;
            n = rotate90( P[i][j+1] - P[i][j] ); // get normal
            n /= length(n); // make normal unit length
         }
      }
   }
   return x + tMin*v; // first hit location
}

// solves a Laplace equation Delta u = 0 at x0, where the Dirichlet and Neumann
// boundaries are each given by a collection of polylines, the Neumann
// boundary conditions are all zero, and the Dirichlet boundary conditions
// are given by a function g that can be evaluated at any point in space
double solve( Vec2D x0,
             vector<Polyline> boundaryDirichlet,
             vector<Polyline> boundaryNeumann,
             function<double(Vec2D)> g ) {
   const double eps = 0.01; // stopping tolerance
   const double rMin = 0.01; // minimum step size
   const int nWalks = 128; // number of Monte Carlo samples
   const int maxSteps = 32; // maximum walk length

   double sum = 0.0;
   for( int i = 0; i < nWalks; i++ ) {
      // assume we start on an interior point (hence have no normal)
      Vec2D x = x0;
      Vec2D n{ 0.0, 0.0 };
      bool onBoundary = false;

      double r;
      int steps = 0;
      do {
         // compute the radius of the largest star-shaped region
         double dDirichlet = distancePolylines( x, boundaryDirichlet );
         double dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
         r = max( rMin, min( dDirichlet, dSilhouette ));

         // intersect a ray with the star-shaped region boundary
         double theta = random( -M_PI, M_PI );
         if( onBoundary ) {
            theta = theta/2. + angleOf(n);
         }
         Vec2D v{ cos(theta), sin(theta) };
         x = intersectPolylines( x, v, r, boundaryNeumann, n );
         if( length(n) < .5 )
            onBoundary = false;
         else
            onBoundary = true;

         steps++;
      }
      while( r > eps && steps < maxSteps );

      sum += g(x);
   }
   return sum/nWalks; // Monte Carlo estimate
}

double lines( Vec2D x ) {
   const double s = 8.0;
   return fmod( floor(s*real(x)), 2.0 );
}

// for simplicity, in this code we assume that the Dirichlet and Neumann
// boundary polylines form a collection of closed polygons (possibly with holes),
// and are given with consistent counter-clockwise orientation
vector<Polyline> boundaryDirichlet = {
   {{ Vec2D(0.2, 0.2), Vec2D(0.6, 0.0), Vec2D(1.0, 0.2) }},
   {{ Vec2D(1.0, 1.0), Vec2D(0.6, 0.8), Vec2D(0.2, 1.0) }}
};
vector<Polyline> boundaryNeumann = {
   {{ Vec2D(1.0, 0.2), Vec2D(0.8, 0.6), Vec2D(1.0, 1.0) }},
   {{ Vec2D(0.2, 1.0), Vec2D(0.0, 0.6), Vec2D(0.2, 0.2) }}
};

// these routines are not used by WoSt itself, but are rather used to check
// whether a given evaluation point is actually inside the domain
double solidAngle( Vec2D x, const vector<Polyline>& P )
{
   double Theta = 0.;
   for( int i = 0; i < P.size(); i++ )
      for( int j = 0; j < P[i].size()-1; j++ )
         Theta += arg( (P[i][j+1]-x)/(P[i][j]-x) );
   return Theta;
}
bool insideDomain( Vec2D x,
                   const vector<Polyline>& PD,
                   const vector<Polyline>& PN )
{
   double Theta = solidAngle(x,PD) + solidAngle(x,PN);
   return abs(Theta-2.*M_PI) < 0.01;
}

int main( int argc, char** argv ) {
   srand( time(NULL) );
   ofstream out( "out.csv" );

   int s = 128; // image size
   for( int j = 0; j < s; j++ )
   {
      cerr << "row " << j << " of " << s << endl;
      for( int i = 0; i < s; i++ )
      {
         Vec2D x0( ((double)i+.5)/((double)s),
                   ((double)j+.5)/((double)s) );
         double u = 0.;
         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann) )
            u = solve( x0, boundaryDirichlet, boundaryNeumann, lines );
         out << u;
         if( i < s-1 ) out << ",";
      }
      out << endl;
   }
   return 0;
}