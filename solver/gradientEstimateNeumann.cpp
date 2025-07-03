
// c++ -std=c++17 -O3 gradientEstimateNeumann.cpp -o ge -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -w

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

#include <Eigen/Dense> 

std::random_device rd; 
std::mt19937 gen(rd());
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
double length( Vec2D u ) { return sqrt( real(u) * real(u) + imag(u) * imag(u) ); }
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

// vector<Polyline> boundaryDirichlet = {{  Vec2D(0.3, 1), Vec2D(0, 1)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 1), Vec2D(0, 0), Vec2D(0.3, 0), Vec2D(0.3, 1)}};
// vector<Polyline> displacedPoints =  {{  Vec2D(0.4, 1), Vec2D(-0.1, 1)}};

vector<Polyline> boundaryDirichlet = {{  Vec2D(0, 1), Vec2D(0, 0)}, {Vec2D(0.3, 0), Vec2D(0.3, 1)}};
vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.3, 0)}, { Vec2D(0.3, 1), Vec2D(0, 1)} };
vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1.0), Vec2D(-0.1, 0)}, {Vec2D(0.4, 0), Vec2D(0.4, 1)}};

// vector<Polyline> boundaryDirichlet = {{  Vec2D(0.0, 1.0), Vec2D(0, 0)}, {Vec2D(0.2, 0), Vec2D(0.2, 1)}};
// vector<Polyline> boundaryNeumann =  {{ Vec2D(0, 0), Vec2D(0.2, 0)},  {Vec2D(0.2, 1), Vec2D(0, 1)}};
// vector<Polyline> displacedPoints =  {{  Vec2D(-0.1, 1.0), Vec2D(-0.1, 0)}, {Vec2D(0.3, 0), Vec2D(0.3, 1)}};

// vector<Polyline> boundaryDirichlet = {{ Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)}};
// vector<Polyline> boundaryNeumann = {{ Vec2D(0, 0), Vec2D(0.49, 0), Vec2D(0.5, 0.2), Vec2D(0.51, 0), Vec2D(1, 0)}};
// vector<Polyline> displacedPoints =  {{ Vec2D(1.1, 0), Vec2D(1.1, 1), Vec2D(-0.1, 1), Vec2D(-0.1, 0)}};

bool isCloseToNeumannBoundary(Vec2D x0, const vector<Polyline>& boundaryNeumann, double tolerance) {
   for (const auto& polyline : boundaryNeumann) {
      for (size_t i = 0; i < polyline.size() - 1; ++i) {
         Vec2D closest = closestPoint(x0, polyline[i], polyline[i + 1]);
         if (length(x0 - closest) <= tolerance) {
            return true;
         }
      }
   }
   return false;
}

bool isOnBoundary(Vec2D x0, const vector<Polyline>& boundaries, double tolerance) {
   for (const auto& polyline : boundaries) {
      for (size_t i = 0; i < polyline.size() - 1; ++i) {
         Vec2D closest = closestPoint(x0, polyline[i], polyline[i + 1]);
         if (length(x0 - closest) <= tolerance) {
            return true;
         }
      }
   }
   return false;
}

Vec2D intersectPolylines(Vec2D x, Vec2D v, double r,
                         const vector<Polyline>& P,
                         Vec2D& n, bool& onBoundary, bool& onNeumann) {
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
               onNeumann = isCloseToNeumannBoundary(x + tMin * v, boundaryNeumann, 1e-5);
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
      
      #pragma omp parallel for reduction(+:sum)
      double nextTheta = -1;
      bool isStarting = true;
      for( i = 0; i < nWalks; i++ ) {
         std::mt19937 generator(i);  
         std::uniform_real_distribution<double> dist(-M_PI, M_PI);
         Vec2D x = x0; // start walk at the evaluation point
         Vec2D n{ 0.0, 0.0 }; // assume x0 is an interior point, and has no normal
         bool onBoundary = false; // flag whether x is on the interior or boundary
         bool onNeumann = false;
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
            x =  intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary, onNeumann);
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
              function<Vec2D(Vec2D)> g, std::ofstream& displacementFile, std::ofstream& gradientFile) { // Dirichlet boundary values
   const double eps = 0.000001; // stopping tolerance
   const double rMin = 0.000001; // minimum step size
   int nWalks = 100000;
   const int maxSteps = 65536; 
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
      bool onNeumann = false; 
      double r, dDirichlet, dSilhouette; 
      int steps = 0;
      Vec2D closestPoint;
      bool isStarting = true;
      Vec2D normal = Vec2D(0, 0);
      Vec2D firstHitBoundary = Vec2D(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
      double radius = 0;
      do {  
         center = x;
         auto p = distancePolylines( x, boundaryDirichlet );
         dDirichlet = p.first;
         closestPoint = p.second;
         dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
         r = max( rMin, min( dDirichlet, dSilhouette ));
         double theta = random( -M_PI, M_PI );
         Vec2D origv{ cos(theta), sin(theta) };
         if ( onBoundary ) { // sample from a hemisphere around the normal
            theta = theta/2. + angleOf(n);
            if (isnan(real(firstHitBoundary)) || isnan(imag(firstHitBoundary))) {
               firstHitBoundary = x;
            }
         }
         Vec2D v{ cos(theta), sin(theta) };
         x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary, onNeumann);
         if (isStarting){
            isStarting = false;
            normal = origv / length(origv);
            radius = dDirichlet;
         }
         steps++;
      }  
      while(dDirichlet > eps && steps < maxSteps);
      
      if( steps >= maxSteps ) continue;

      if (isnan(real(firstHitBoundary)) || isnan(imag(firstHitBoundary))) {
         firstHitBoundary = closestPoint;
      }
      // cout << "---------------------------------" << "\n";
      Vec2D estimated_position = g(closestPoint);
      // cout << "Estimated position: " << real(estimated_position) << ", " << imag(estimated_position) << "\n";
      if (isnan(real(estimated_position)) || isnan(imag(estimated_position))) {
         cout << "x0: " << real(x0) << ", " << imag(x0) << "\n";
         cout << "Closest point: " << real(closestPoint) << ", " << imag(closestPoint) << "\n";
         cout << "First hit boundary: " << real(firstHitBoundary) << ", " << imag(firstHitBoundary) << "\n";
      }
      Vec2D estimated_displacement = estimated_position - firstHitBoundary;
      if (isnan(real(estimated_displacement)) || isnan(imag(estimated_displacement))) {
         cout << "x0: " << real(x0) << ", " << imag(x0) << "\n";
         cout << "estimated position: " << real(estimated_position) << ", " << imag(estimated_position) << "\n";
         cout << "Estimated displacement: " << real(estimated_displacement) << ", " << imag(estimated_displacement) << "\n";
      } 
      vector<Vec2D> estimated_gradient = multiply(estimated_displacement, normal);
      estimated_gradient = { Vec2D(2 * 1/radius * real(estimated_gradient[0]), 2 * 1/radius * imag(estimated_gradient[0])),
      Vec2D(2 * 1/radius * real(estimated_gradient[1]), 2 * 1/radius * imag(estimated_gradient[1])) };
      walker += 1;
      sum_11 += real(estimated_gradient[0]);
      sum_12 += imag(estimated_gradient[0]);
      sum_21 += real(estimated_gradient[1]);
      sum_22 += imag(estimated_gradient[1]);
      sum_x += real(estimated_displacement);
      sum_y += imag(estimated_displacement);
   } 
   
   displacementFile << real(x0) << "," << imag(x0) << ",";
   displacementFile << sum_x /walker  << "," << sum_y/walker << "\n";
   gradientFile << real(x0) << "," << imag(x0) << ",";
   gradientFile << sum_11/walker << "," << sum_12/walker << "," << sum_21/walker << "," << sum_22/walker << "\n";
   Vec2D row1 = Vec2D(sum_11/walker, sum_12/walker);
   Vec2D row2 = Vec2D(sum_21/walker, sum_22/walker);
   return {row1, row2};
}

double signedAngle( Vec2D x, const vector<Polyline>& P )
{
   double Theta = 0.;
   for( int i = 0; i < P.size(); i++ )
      for( int j = 0; j < P[i].size()-1; j++ )
         Theta += arg( (P[i][j+1]-x)/(P[i][j]-x) );
   return Theta;
}

Vec2D interpolateVec2DBoundaryPoints(Vec2D v, vector<Polyline> originalPoints, vector<Polyline> displacedPoints, double num_tol=1e-5) { 
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

Vec2D displacement(Vec2D v) { 
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   Vec2D interpolatedDisplacement = interpolateVec2DBoundaryPoints(v, boundaryDirichlet, displacedPoints);
   if (isnan(real(interpolatedDisplacement)) || isnan(imag(interpolatedDisplacement))) {
      return nan;
   }
   return interpolatedDisplacement;
}

Vec2D deformBoundary(Vec2D v) { 
   float x = real(v);
   if ( x < 0.5) {
      x = real(v) - 0.1 * (1 - imag(v));
   }
   else {
      x = real(v) + 0.1 * (1 - imag(v));
   }
   return Vec2D(x, imag(v));
}

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
   eigenvectorFile << "principalX, principalY\n";
   eigenvectorFile << principalEigenvector(0) << "," << principalEigenvector(1) << "\n";
}

string double_to_str(double f) {
   std::string str = std::to_string (f);
   str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
   str.erase ( str.find_last_not_of('.') + 1, std::string::npos );
   return str;
}

vector<Vec2D> generatePointsNearNeumannBoundary(Vec2D p1, Vec2D p2, Vec2D p3) {
   vector<Vec2D> nearbyPoints;

   int samplesPerSegment = 10;
   double offsetDistance = 0.01;

   // Generate points near segment 1
   for (int i = 0; i <= samplesPerSegment; ++i) {
       double t = static_cast<double>(i) / samplesPerSegment;
       Vec2D point = (1 - t) * p1 + t * p2;

       // Direction vector of the segment
       Vec2D dir = p2 - p1;
       // Perpendicular vector (rotate 90 degrees)
       Vec2D normal(-imag(dir), real(dir));
       normal /= abs(normal);  // Normalize

       nearbyPoints.push_back(point + offsetDistance * normal);
       nearbyPoints.push_back(point - offsetDistance * normal);
   }

   // Generate points near segment 2
   for (int i = 0; i <= samplesPerSegment; ++i) {
       double t = static_cast<double>(i) / samplesPerSegment;
       Vec2D point = (1 - t) * p2 + t * p3;

       Vec2D dir = p3 - p2;
       Vec2D normal(-imag(dir), real(dir));
       normal /= abs(normal);

       nearbyPoints.push_back(point + offsetDistance * normal);
       nearbyPoints.push_back(point - offsetDistance * normal);
   }

   return nearbyPoints;
}

int main( int argc, char** argv ) {
   string shape = "gradient_estimate_dirichlet_without_notch_0.3";
   string fileName = shape; 
   auto deform = displacement;
   int s = 16; 
   
   std::ofstream gradientFile("../output/" + fileName + "_deformation_gradient.csv");
   std::ofstream displacementFile("../output/" + fileName + "_displacement.csv");
   gradientFile << "X,Y,F11,F12,F21,F22\n";
   
   for (int j = 0; j < (s - 1) ; j+= 1) {
      for (int i = 0; i < (s - 1); i+= 1) {
         Vec2D x0((double)i / (s - 1), (double)j / (s - 1));
   
         double x = real(x0);
         double y = imag(x0);

         if (isOnBoundary(x0, boundaryDirichlet, 1e-8) || 
            isOnBoundary(x0, boundaryNeumann, 1e-8)) {
            continue;
         }
         else { 
            vector<Vec2D> gradient = solveGradient(x0, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
         }
       
         if (j == 0){
            Vec2D x1 = Vec2D(real(x0), 0.01);
            Vec2D x2 = Vec2D(real(x0), 0.041);
            Vec2D x3 = Vec2D(real(x0), 0.081);
            Vec2D x4 = Vec2D(real(x0), 0.99);
            Vec2D x5 = Vec2D(real(x0), 0.959);
            Vec2D x6 = Vec2D(real(x0), 0.919);
            if (insideDomain(x1, boundaryDirichlet, boundaryNeumann)) {
               solveGradient(x1, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
            }
            if (insideDomain(x2, boundaryDirichlet, boundaryNeumann)) {
               solveGradient(x2, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
            }
            if (insideDomain(x3, boundaryDirichlet, boundaryNeumann)) {
               solveGradient(x3, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
            }
            
            if (insideDomain(x4, boundaryDirichlet, boundaryNeumann)) {
               solveGradient(x4, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
            }
            if (insideDomain(x5, boundaryDirichlet, boundaryNeumann)) {
               solveGradient(x5, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
            }
            if (insideDomain(x6, boundaryDirichlet, boundaryNeumann)) {
               solveGradient(x6, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
            }
         }
      }
      
      Vec2D p1(0.1,  (double)j / (s - 1));
      Vec2D p2(0.01,  (double)j / (s - 1));
      Vec2D p3(0.99,  (double)j / (s - 1));
      Vec2D p4(0.9,  (double)j / (s - 1));
      if (insideDomain(p1, boundaryDirichlet, boundaryNeumann)) {
         solveGradient(p1, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      }
      if (insideDomain(p2, boundaryDirichlet, boundaryNeumann)) {
         solveGradient(p2, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      }
      if (insideDomain(p3, boundaryDirichlet, boundaryNeumann)) {
         solveGradient(p3, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      }
      if (insideDomain(p4, boundaryDirichlet, boundaryNeumann)) {
         solveGradient(p4, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      }
   }
      // if (insideDomain(p5, boundaryDirichlet, boundaryNeumann)) {
      //    solveGradient(p5, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      // }
      // if (insideDomain(p6, boundaryDirichlet, boundaryNeumann)) {
      //    solveGradient(p6, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      // }
      // if (insideDomain(p7, boundaryDirichlet, boundaryNeumann)) {
      //    solveGradient(p7, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      // }

      // vector<Vec2D> customValues;
      // customValues.push_back(Vec2D(0.01, 0.01));
      // customValues.push_back(Vec2D(0.01, 0.99));
      // customValues.push_back(Vec2D(0.19, 0.01));
      // customValues.push_back(Vec2D(0.19, 0.99));
      // customValues = generatePointsNearNeumannBoundary(Vec2D(0.48, 0), Vec2D(0.5, 0.2), Vec2D(0.52, 0));
      // customValues.push_back(Vec2D(0.5, 0.21));
      // customValues.push_back(Vec2D(0.49, 0.2));
      // customValues.push_back(Vec2D(0.51, 0.2));
      // customValues.push_back(Vec2D(0.01, 0.01));
      // customValues.push_back(Vec2D(0.05, 0.01));
      // customValues.push_back(Vec2D(0.05, 0.05));
      // customValues.push_back(Vec2D(0.01, 0.05));
      // customValues.push_back(Vec2D(0.99, 0.01));
      // customValues.push_back(Vec2D(0.99, 0.05));
      // customValues.push_back(Vec2D(0.95, 0.01));
      // customValues.push_back(Vec2D(0.95, 0.05));
      // customValues.push_back(Vec2D(0.99, 0.99));
      // customValues.push_back(Vec2D(0.99, 0.95));
      // customValues.push_back(Vec2D(0.95, 0.99));
      // customValues.push_back(Vec2D(0.95, 0.95));
      // customValues.push_back(Vec2D(0.01, 0.99));
      // customValues.push_back(Vec2D(0.05, 0.99));
      // customValues.push_back(Vec2D(0.01, 0.95));
      // customValues.push_back(Vec2D(0.05, 0.95));
      // customValues.push_back(Vec2D(0.475, 0.01));
      // customValues.push_back(Vec2D(0.47, 0.01));
      // customValues.push_back(Vec2D(0.465, 0.01));
      // customValues.push_back(Vec2D(0.525, 0.01));
      // customValues.push_back(Vec2D(0.53, 0.01));
      // customValues.push_back(Vec2D(0.535, 0.01));
      // customValues.push_back(Vec2D(0.9, 0.05));
      // customValues.push_back(Vec2D(0.93, 0.05));
      // customValues.push_back(Vec2D(0.88, 0.05));
      // customValues.push_back(Vec2D(0.95, 0.05));
      // customValues.push_back(Vec2D(0.15, 0.05));
      // customValues.push_back(Vec2D(0.12, 0.05));
      // customValues.push_back(Vec2D(0.1, 0.05));
      // customValues.push_back(Vec2D(0.08, 0.05));
      // customValues.push_back(Vec2D(0.06, 0.05));
      // customValues.push_back(Vec2D(0.04, 0.05));
      // customValues.push_back(Vec2D(0.19, 0.05));
      // customValues.push_back(Vec2D(0.85, 0.05));
      // customValues.push_back(Vec2D(0.9, 0.05));
      // customValues.push_back(Vec2D(0.93, 0.05));
      // customValues.push_back(Vec2D(0.88, 0.05));
      // customValues.push_back(Vec2D(0.95, 0.05));
      // customValues.push_back(Vec2D(0.15, 0.008));
      // customValues.push_back(Vec2D(0.12, 0.008));
      // customValues.push_back(Vec2D(0.1, 0.008));
      // customValues.push_back(Vec2D(0.08, 0.008));
      // customValues.push_back(Vec2D(0.06, 0.008));
      // customValues.push_back(Vec2D(0.04, 0.008));
      // customValues.push_back(Vec2D(0.19, 0.004));
      // customValues.push_back(Vec2D(0.19, 0.008));
      // customValues.push_back(Vec2D(0.19, 0.012));
      // customValues.push_back(Vec2D(0.18, 0.004));
      // customValues.push_back(Vec2D(0.18, 0.008));
      // customValues.push_back(Vec2D(0.18, 0.012));
      // customValues.push_back(Vec2D(0.17, 0.004));
      // customValues.push_back(Vec2D(0.17, 0.008));
      // customValues.push_back(Vec2D(0.17, 0.012));
      // customValues.push_back(Vec2D(0.5, 0.201));
      // customValues.push_back(Vec2D(0.49, 0.201));
      // customValues.push_back(Vec2D(0.51, 0.201));
      // customValues.push_back(Vec2D(0.5, 0.2001));
      // customValues.push_back(Vec2D(0.49, 0.2001));
      // customValues.push_back(Vec2D(0.51, 0.2001));
      // customValues.push_back(Vec2D(0.5, 0.20001));
      // customValues.push_back(Vec2D(0.49, 0.20001));
      // customValues.push_back(Vec2D(0.51, 0.20001));
      // customValues.push_back(Vec2D(0.5, 0.2000001));
      // customValues.push_back(Vec2D(0.48, 0.001));
      // customValues.push_back(Vec2D(0.49, 1.001));
      // customValues.push_back(Vec2D(0.52, 0.001));
      // customValues.push_back(Vec2D(0.51, 1.001));
     
   // for (const auto& polyline : boundaryDirichlet) {
   //    for (int i = 0; i < polyline.size() - 1; i++) {
   //       customValues.push_back(polyline[i] + Vec2D(0, 0.1));
   //       customValues.push_back(polyline[i] + Vec2D(0, 0.01));
   //       customValues.push_back(polyline[i] + Vec2D(0, 0.001));
   //       customValues.push_back(polyline[i] + Vec2D(0, 0.0001));
   //       customValues.push_back(polyline[i] + Vec2D(-0.01, 0));
   //       customValues.push_back(polyline[i] + Vec2D(-0.001, 0));
   //       customValues.push_back(polyline[i] + Vec2D(0.01, 0));
   //       customValues.push_back(polyline[i] + Vec2D(0.001, 0));
   //    }
   // }
   // vector<Vec2D> more = {
   //    {0.48, 0.01},
   //    {0.48, 0.015},
   //    {0.48, 0.02},
   //    {0.485, 0.01},
   //    {0.485, 0.015},
   //    {0.485, 0.02},
   //    {0.49, 0.01},
   //    {0.49, 0.015},
   //    {0.49, 0.02},
   //    {0.49, 0.11},
   //    {0.495, 0.11},
   //    {0.485, 0.11},
   //    {0.505, 0.11},
   //    {0.51, 0.11},
   //    {0.515, 0.11},
   //    {0.506, 0.01},
   //    {0.51, 0.01},
   //    {0.515, 0.01},
   //    {0.52, 0.01},
   //    {0.49, 0.021},
   //    {0.485,0.021},
   //    {0.48, 0.021},
   //    {0.506, 0.021},
   //    {0.51, 0.021},
   //    {0.515, 0.021},
   //    {0.52, 0.021},
   //    {0.49, 0.04},
   //    {0.485, 0.04},
   //    {0.48, 0.04},
   //    {0.506, 0.04},
   //    {0.51, 0.04},
   //    {0.515, 0.04},
   //    {0.52, 0.04},
   //    {0.49901, 0.01},
   //    {0.50099, 0.01},
   //    {0.49, 0.201}, 
   //    {0.495, 0.201}, 
   //    {0.498, 0.201}, 
   //    {0.502, 0.201}, 
   //    {0.5, 0.2005}, 
   //    {0.5, 0.201}, 
   //    {0.5, 0.2015}, 
   //    {0.505, 0.201}, 
   //    {0.51, 0.201}, 
   //    {0.49, 0.21}, 
   //    {0.495, 0.21}, 
   //    {0.498, 0.21}, 
   //    {0.502, 0.21}, 
   //    {0.5, 0.205}, 
   //    {0.5, 0.21}, 
   //    {0.5, 0.215}, 
   //    {0.505, 0.21}, 
   //    {0.51, 0.21}
   // };
   // customValues.insert(customValues.end(), more.begin(), more.end());

      // for (const auto& point : customValues) {
      //    if (insideDomain(point, boundaryDirichlet, boundaryNeumann)) {
      //       vector<Vec2D> gradient = solveGradient(point, boundaryDirichlet, boundaryNeumann, deform, displacementFile, gradientFile);
      //    }
      // }
   // }
}
