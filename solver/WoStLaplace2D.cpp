// (Optimized) implementation of the "Walk on Stars" algorithm for Laplace equations.
// Corresponds to the estimator given in Equation 18 of Sawhney et al,
// "Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary
// Conditions" (2023), assuming no source term and zero-Neumann conditions.
// NOTE: this code makes a few shortcuts for the sake of code brevity; may
// be more suitable for tutorials than for production code/evaluation.
// To compile: c++ -std=c++17 -O3 -pedantic -Wall WoStLaplace2D.cpp -o wost
// To compile the new version with Open MP: c++ -std=c++17 -O3 -pedantic -Wall -I/opt/homebrew/Cellar/libomp/18.1.1/include WoStLaplace2D.cpp -o wost-opt -L/opt/homebrew/Cellar/libomp/18.1.1/lib -lomp

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
// #include <omp.h> 
#include <chrono> // Added for timing
using namespace std;
using namespace std::chrono; // Added for timing

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
double solve( Vec2D x0, // evaluation point
              vector<Polyline> boundaryDirichlet, // absorbing part of the boundary
              vector<Polyline> boundaryNeumann, // reflecting part of the boundary
              function<double(Vec2D)> g ) { // Dirichlet boundary values
   const double eps = 0.0001; // stopping tolerance
   const double rMin = 0.0001; // minimum step size
   const int nWalks = 65536; // number of Monte Carlo samples
   const int maxSteps = 65536; // maximum walk length

   double sum = 0.0; // running sum of boundary contributions
   // #pragma omp parallel for reduction(+:sum)
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
   return fmod( floor(s*real(x)), 2.0 );
}

// for simplicity, in this code we assume that the Dirichlet and Neumann
// boundary polylines form a collection of closed polygons (possibly with holes),
// and are given with consistent counter-clockwise orientation
vector<Polyline> boundaryDirichlet = { {0, 0}, {1, 0}, { 1, 1}, {0, 1}};
vector<Polyline> boundaryNeumann = {

};

vector<double> cornerHeights;

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
   // cout << Theta << std::endl;
   return abs(Theta-2.*M_PI) < delta; // boundary winds around x exactly once
}

double getSaddlePointHeight(Vec2D x) {
    return real(x) * real(x) - imag(x) * imag(x);
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

void createStarBoundary(int num_points, double outer_radius, double inner_radius, vector<Polyline>& boundaryDirichlet) 
{
   vector<double> angles;
   for (int i = 0; i < num_points * 2; i++) {
      angles.push_back(2 * M_PI * i / (num_points * 2));
   }

   vector<Vec2D> boundary;
   for (int i = 0; i < num_points * 2; i++) {
      if (i % 2 == 0) {
         boundary.push_back(Vec2D(outer_radius * cos(angles[i]), outer_radius * sin(angles[i])));
      }  else {
         boundary.push_back(Vec2D(inner_radius * cos(angles[i]), inner_radius * sin(angles[i])));
      }
   }
   boundary.push_back(boundary[0]);
   cout << "boundary length: " << boundary.size() << std::endl;

   boundaryDirichlet.push_back(boundary);
   // remove middle 3 points 
   // boundaryDirichlet[0].erase(boundaryDirichlet[0].begin() + num_points - 1, boundaryDirichlet[0].begin() + num_points + 5);
}

void createBubbleSoapBoundary(int num_points, double radius, double perturbation_amplitude, std::vector<Polyline>& boundaryDirichlet) {
    Polyline boundary;
    
    // Calculate points around the circle with small random-like perturbations for the soap-bubble effect
    for (int i = 0; i < num_points; i++) {
        double angle = 2 * M_PI * i / num_points;
        
        // Perturb the radius slightly for a bubble effect
        double perturbed_radius = radius + perturbation_amplitude * std::sin(3 * angle) * std::cos(2 * angle);
        
        // Calculate x and y coordinates
        double x = perturbed_radius * std::cos(angle);
        double y = perturbed_radius * std::sin(angle);
        
        boundary.push_back(Vec2D(x, y));
    }
    
    // Ensure the boundary is closed by adding the first point to the end
    boundary.push_back(boundary[0]);
    
    std::cout << "Boundary length: " << boundary.size() << std::endl;
    boundaryDirichlet.push_back(boundary);
}


int start_vertex_count = 0;

double getStarHeight(Vec2D point){
   double tilt_angle = M_PI / 6;
   double randomValue = static_cast<double>(rand()) / RAND_MAX * 2;
   double height = 0;
   if (start_vertex_count == 0 || start_vertex_count == 10){
      height = (tilt_angle) * real(point);
   }
   else {
      height = randomValue * (tilt_angle) * real(point);
   }
   start_vertex_count += 1;
   return height;
}

double getBubbleHeight(Vec2D point) {
   double max_tilt_angle = M_PI / 6;
   double distance_from_center = sqrt(real(point) * real(point) + imag(point) * imag(point));

    // Generate a small random perturbation for the "bubble" effect
   double randomValue = static_cast<double>(rand()) / RAND_MAX * 2 - 1;

    // Calculate height based on distance and perturbation
   double height = distance_from_center * max_tilt_angle * (0.5 + randomValue * 0.5);

   return height;
}

double getCircleHeight(Vec2D point) {
    double wave_amplitude = 0.2; 
    double angle = atan2(real(point), real(point));  
    double baseHeight = length(point);
    return baseHeight * (1 + wave_amplitude * sin(angle));
}

double getCircleHeight2(Vec2D point) {
    double radiusFactor = length(point);
    return radiusFactor * M_PI; 
}


double getCircleHeight3(Vec2D point) {
    double tilt_factor = 0.1;  // Controls the amount of random variation
    double randomValue = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    double baseHeight = length(point);  // Distance from the origin, or radius
    return baseHeight * (1 + tilt_factor * randomValue);
}

void createCircleBoundary(int num_points, double radius, vector<Polyline>& boundaryDirichlet) {
   vector<Vec2D> boundary;
   for (int i = 0; i < num_points; i++) {
      double angle = 2 * M_PI * i / num_points;
      boundary.push_back(Vec2D(radius * cos(angle), radius * sin(angle)));
   }
   boundary.push_back(boundary[0]);
   boundaryDirichlet.push_back(boundary);
}

// calculate the signed area of polygon: xi * yi+1 - xi+1 * yi
bool checkOrder(vector<Polyline> boundary){
   double Theta = 0.;
   for (int k = 0; k < boundary.size(); k ++){
      int size = boundary[k].size();
      for( int i = 0; i < size; i++ ){
         Theta += real(boundary[k][i]) * imag(boundary[k][i+1]) - real(boundary[k][i+1]) * imag(boundary[k][i]);
      }
   }
   return Theta / 2 > 0;
}

void createBubbleBoundary(int num_points, double radius, vector<Polyline>& boundaryDirichlet) {
   vector<Vec2D> boundary;
   for (int i = 0; i < num_points; i++) {
      double angle = 2 * M_PI * i / num_points;
      boundary.push_back(Vec2D(radius * cos(angle), radius * sin(angle)));
   }
   boundary.push_back(boundary[0]); 
   boundaryDirichlet.push_back(boundary);
} 

double getBubbleSoapHeight(Vec2D point) {
   double radius = 1;
   double distanceFromCenter = sqrt(real(point) * real(point) + imag(point) * imag(point));
   double maxHeight = radius * 0.2;
   double height = maxHeight * exp(-pow(distanceFromCenter / radius, 2));
   return height;
}

double getBubbleHeightConstant(Vec2D point) {
   return 1;
}

double getRectangleHeightRandom(Vec2D point){
   double x = real(point);
   double y = imag(point);
   
   // Assuming corner heights are stored in cornerHeights in the order: bottom-left, bottom-right, top-right, top-left
   double height = 0.0;

   if (y == imag(boundaryDirichlet[0][0])) { // Bottom edge
      height = cornerHeights[0] + (cornerHeights[1] - cornerHeights[0]) * (x - real(boundaryDirichlet[0][0])) / (real(boundaryDirichlet[0][1]) - real(boundaryDirichlet[0][0]));
   } else if (y == imag(boundaryDirichlet[0][2])) { // Top edge
      height = cornerHeights[2] + (cornerHeights[3] - cornerHeights[2]) * (x - real(boundaryDirichlet[0][2])) / (real(boundaryDirichlet[0][3]) - real(boundaryDirichlet[0][2]));
   } else if (x == real(boundaryDirichlet[0][0])) { // Left edge
      height = cornerHeights[3] + (cornerHeights[0] - cornerHeights[3]) * (y - imag(boundaryDirichlet[0][0])) / (imag(boundaryDirichlet[0][3]) - imag(boundaryDirichlet[0][0]));
   } else if (x == real(boundaryDirichlet[0][1])) { // Right edge
      height = cornerHeights[1] + (cornerHeights[2] - cornerHeights[1]) * (y - imag(boundaryDirichlet[0][1])) / (imag(boundaryDirichlet[0][2]) - imag(boundaryDirichlet[0][1]));
   }

   return height;
}


double initializeRectangleHeightRandom(vector<double>& cornerHeights) {
   for (int i = 0; i < 4; i++) {
      cornerHeights.push_back(rand() / static_cast<double>(RAND_MAX));
   }
}

int main( int argc, char** argv ) {
   bool printBoundary = true;
   string shape = "random-height-rectangle";
   auto heightFunction = getRectangleHeightRandom;
   initializeRectangleHeightRandom(cornerHeights);
   srand( time(NULL) );
   ofstream out( "../output/" + shape + ".csv" );

   int s = 16; // make it smaller to speed up
   // createStarBoundary(5, 1.0, 0.5, boundaryDirichlet);
   // cout << "check boundary " << checkOrder(boundaryDirichlet) << std::endl;
   // createSaddlePointBoundary(-1.0, -1.0, 1.0, 1.0, 30, boundaryDirichlet);
   auto start = high_resolution_clock::now(); // Added for timing
   
   // solve using the Walk on Stars algorithm 
   #pragma omp parallel for
   for( int j = 0; j < s; j++ )
   {
      cerr << "row " << j << " of " << s << endl;
      for( int i = 0; i < s; i++ )
      {
         Vec2D x0(((double)i / (s - 1)) * 2 - 1,
                 ((double)j / (s - 1)) * 2 - 1);
         double u = numeric_limits<double>::quiet_NaN();
         
         // cout << real(x0) << " " << imag(x0) << std::endl;

         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann) ){
            u = solve( x0, boundaryDirichlet, boundaryNeumann, heightFunction );
            cout << "inside domain u: " << u << std::endl;
         }
         out << u;
         if( i < s-1 ) out << ",";
      }
      out << endl;
   }
   auto stop = high_resolution_clock::now(); // Added for timing
   auto duration = duration_cast<milliseconds>(stop - start); // Added for timing
   cout << "Time taken by function: " << duration.count() << " milliseconds" << endl; // Added for timing

   // print boundary dirichlet 
   if (!printBoundary) return 0;
   ofstream outBoundaryD( "../output/" + shape + "BoundaryDirichlet.csv" );
   for (int i = 0; i < boundaryDirichlet[0].size(); i++) {
      Vec2D point = boundaryDirichlet[0][i];
      double height = heightFunction(point);
      outBoundaryD << real(point) << "," << imag(point) << "," << height << std::endl;
   }
}
