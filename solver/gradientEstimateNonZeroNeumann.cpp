
// c++ -std=c++17 -O3 -pedantic -Wall gradientEstimateNonZeroNeumann.cpp -o gradientEstimateNonZeroNeumann -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -w

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
double  dot(Vec2D u, Vec2D v) { return real(u)*real(v) + imag(u)*imag(v); }
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

vector<Polyline> boundaryDirichlet =  {{ Vec2D(0, 1), Vec2D(0, 0) }};

vector<Polyline> boundaryNeumann = {{ Vec2D(0, 0), Vec2D(0.5, 0), Vec2D(1, 0), Vec2D(1, 0.5), Vec2D(1, 1), Vec2D(0.5, 1),
                                     Vec2D(0, 1) }};

Vec2D getDirichletValue( Vec2D point ) {
   if (real(point) == 0) {
      return point;
   }
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   return nan;
}

Vec2D getNeumannValue( Vec2D point ) {
   if (real(point) == 1) {
      return Vec2D(0.5, 0);
   }
   Vec2D nan = numeric_limits<double>::quiet_NaN();
   return nan;
}

double get2DFreeSpaceGreenKernel( Vec2D x, Vec2D y ) {
   // Green's function for the Laplace operator in 2D
   // G(x,y) = 1/(2*pi) * log(|y-x|)
   if (x == y) {
      return 0.0;
   }
   return 1.0/(2.0*M_PI) * log(length(y-x));
}

double get2DPoissonKernel(Vec2D x, Vec2D y, Vec2D n) {
   // Poisson kernel for the unit disk
   // P(x,y) = 1/((2*pi) * /|y-x|^2) * (y - x) * n 
   // where n is the unit normal at y
   if (x == y) {
      return 0.0;
   }
   return 1.0/(2.0*M_PI) * dot(n, y - x)/ (length(y-x)*length(y-x));
}


double get2DFreeSpaceGreenKernelforBall( Vec2D x, Vec2D y, double r) {
   // Green's function for the Laplace operator in 2D for the ball with Radius r
   // G(x,y) = 1/(2*pi) * log(r / |y-x|)
   if (x == y) {
      return 0.0;
   }
   return 1.0/(2.0*M_PI) * log(r / length(y-x));
}

double get2DPoissonKernelforBall(double r) {
   // Poisson kernel for the ball with Radius r
   // P(x,y) = 1/((2*pi) * /|y-x|^2) * (y - x) * n 
   return 1.0/ ((2.0*M_PI) * r);
}

double sampleSourceDistance(double rMin, double rMax) {
   static std::random_device rd;
   static std::mt19937 gen(rd());
   static std::uniform_real_distribution<double> dist(0.0, 1.0);

   double U = dist(gen); // Uniform sample in (0,1)
   return rMin * exp(U * log(rMax / rMin));
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

// solver for a poisson problem with non-zero Neumann boundary conditions
vector<Vec2D> solveGradient( Vec2D x0,
              vector<Polyline> boundaryDirichlet, 
              vector<Polyline> boundaryNeumann,
              Vec2D source,
              function<Vec2D(Vec2D)> getDirichletValue, function<Vec2D(Vec2D)> getNeumannValue, std::ofstream& displacementFile, std::ofstream& gradientFile) { 
   const double eps = 0.000001; 
   const double rMin = 0.000001; 
   const int nWalks = 100000;
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
      double r, dDirichlet, dSilhouette; 
      int steps = 0;
      Vec2D closestPoint;
      bool isStarting = true;
      Vec2D normal = Vec2D(0, 0);
      double neumann_contribution_x = 0;
      double neumann_contribution_y = 0;
      double source_contribution_x = 0;
      double source_contribution_y = 0;
      double raidus = 0;
      do { 
         center = x;
         auto p = distancePolylines( x, boundaryDirichlet );
         dDirichlet = p.first;
         closestPoint = p.second;
         dSilhouette = silhouetteDistancePolylines( x, boundaryNeumann );
         r = max( rMin, min( dDirichlet, dSilhouette ));

         // compute single sample neumann contribution
         double theta = random( -M_PI, M_PI );
         if( onBoundary ) { // sample from a hemisphere around the normal
            Vec2D normal_derivative = getNeumannValue(x);
            if (!isnan(real(normal_derivative)) && !isnan(imag(normal_derivative))) {
               neumann_contribution_x += real(normal_derivative) * get2DFreeSpaceGreenKernelforBall(x0, x, r);
               neumann_contribution_y += imag(normal_derivative) * get2DFreeSpaceGreenKernelforBall(x0, x, r);
            }
            else {
               theta = theta/2. + angleOf(n);
            }
         }
         Vec2D v{ cos(theta), sin(theta) }; 
         x = intersectPolylines( x, v, r, boundaryNeumann, n, onBoundary );
         if (isStarting){
            isStarting = false;
            normal = v / length(v);
            raidus = dDirichlet;
         }

         // compute single sample source contribution
         double t_source = sampleSourceDistance(rMin, r);
         Vec2D y = x + t_source * Vec2D{ cos(theta), sin(theta) };
         if (length(y - x0) < length(x - x0)) {
            source_contribution_x += real(source) * get2DFreeSpaceGreenKernelforBall(x0, y, t_source);
            source_contribution_y += imag(source) * get2DFreeSpaceGreenKernelforBall(x0, y, t_source);
         }
         steps++;
      }
      while(dDirichlet > eps && steps < maxSteps);

      if( steps >= maxSteps ) continue;
      Vec2D estimated_u = Vec2D(real(getDirichletValue(closestPoint)) - neumann_contribution_x + source_contribution_x, imag(getDirichletValue(closestPoint)) - neumann_contribution_y + source_contribution_y);
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
   vector<Polyline> displacedPoints =  {{ Vec2D(1.4, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(-0.2, 0)}};
   // vector<Polyline> displacedPoints = {{ Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0), Vec2D(1.4, 0) }};

   Vec2D nan = numeric_limits<double>::quiet_NaN();

   Vec2D interpolatedDisplacement = interpolateVec2D_BoundaryPoints(v, boundaryDirichlet, displacedPoints);
   if (isnan(real(interpolatedDisplacement)) || isnan(imag(interpolatedDisplacement))) {
      return nan;
   }
   return interpolatedDisplacement;
}

Vec2D identity( Vec2D v ) {
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
  
   neighbourFile << "leftX_d, leftY_d, rightX_d rightY_d, topX_d, topY_d, bottomX_d, bottomY_d\n";
   neighbourFile << real(neighbors_deformed[0]) << "," << imag(neighbors_deformed[0]) << ",";
   neighbourFile << real(neighbors_deformed[1]) << "," << imag(neighbors_deformed[1]) << ",";
   neighbourFile << real(neighbors_deformed[2]) << "," << imag(neighbors_deformed[2]) << ",";
   neighbourFile << real(neighbors_deformed[3]) << "," << imag(neighbors_deformed[3]) << "\n";

   double dudx = (8 * real(neighbors_deformed[1]) - 8 * real(neighbors_deformed[0]) + real(neighbors_deformed[4]) - real(neighbors_deformed[5])) / (6 * h);
   double dudy = (8 * real(neighbors_deformed[2]) - 8 * real(neighbors_deformed[3]) + real(neighbors_deformed[7]) - real(neighbors_deformed[6])) / (6 * h);
   double dvdx = (8 * imag(neighbors_deformed[1]) - 8 * imag(neighbors_deformed[0]) + imag(neighbors_deformed[4]) - imag(neighbors_deformed[5])) / (6 * h);
   double dvdy = (8 * imag(neighbors_deformed[2]) - 8 * imag(neighbors_deformed[3]) + imag(neighbors_deformed[7]) - imag(neighbors_deformed[6])) / (6 * h);

   strainFile << "X,Y,F11,F12,F21,F22\n";
   strainFile << x << "," << y << ",";
   strainFile << dudx << "," << dudy << "," << dvdx << "," << dvdy << "\n";

   eigenvalueFile << x << "," << y << ", "; 
   eigenvalueFile2 << x << "," << y << ", ";
   eigenvalueFile3 << x << "," << y << ", ";

   return vector<Vec2D>{ Vec2D{dudx, dudy}, Vec2D{dvdx, dvdy}};
}

string double_to_str(double f) {
   std::string str = std::to_string (f);
   str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
   str.erase ( str.find_last_not_of('.') + 1, std::string::npos );
   return str;
}

int main( int argc, char** argv ) {
   string shape = "gradient_estimate_free_boundary_gravity_test";
   double h = 0.01;
   string fileName = shape; 
   auto deform = identity;
   int s = 16;

   std::ofstream gradientFile("../output/" + fileName + "_deformation_gradient.csv");
   std::ofstream displacementFile("../output/" + shape + "_displacements.csv");

   for( int j = 0; j < s; j++ )
   {
      cerr << "row " << j << " of " << s << endl;
      for( int i = 0; i < s; i++ )
      {
         Vec2D x0(((double)i / (s - 1)),
                  ((double)j / (s - 1))
         );
         double x = real(x0);
         double y = imag(x0);
         if( insideDomain(x0, boundaryDirichlet, boundaryNeumann)){
            vector<Vec2D> gradient = solveGradient(x0, boundaryDirichlet, boundaryNeumann, Vec2D(0, 0.0981), getDirichletValue, getNeumannValue, displacementFile, gradientFile);
         }
      } 
   }
}
