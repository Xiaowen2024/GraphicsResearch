2D Linear Elasticity Solver using Walk-on-Boundary
1. Overview
This project implements a C++ solver for a 2D linear elasticity problem with mixed boundary conditions. It uses a Monte Carlo method known as Walk-on-Boundary (WoB) to calculate the displacement field of a 2D elastic body.

The specific problem hard-coded in this solver is a unit square. The boundary conditions are:

Dirichlet: The left and right vertical sides are stretched horizontally. The left side is displaced by -0.1 in the x-direction, and the right side by +0.1.

Neumann: The top and bottom edges are either traction-free or with a small applied vertical traction to simulate pressure.

The solver computes the displacement vector (u_x, u_y) for a grid of points inside this domain and outputs the results to a CSV file.

2. How the Solver Works: The Theory
The solver is based on an Indirect Single-Layer Boundary Integral Equation (BIE) formulation. 

2.1. The Core Equation
The fundamental idea is that the displacement u at any interior point x can be represented by the a integral of μ (the source density), distributed over the entire boundary Γ. The equation is:

$$ u(x) = \int_{\Gamma} G(x, y) \mu(y) , d\Gamma(y) $$

u(x): The final displacement vector we want to find.

G(x, y): The Kelvin Kernel, which is the fundamental solution to the elasticity equations. 

μ(y): The unknown vector source density at boundary point y.

The entire challenge is to find the correct μ that satisfies all the mixed boundary conditions simultaneously.

2.2. The Monte Carlo Approach
The solver uses a two-step Monte Carlo process to solve this problem:

Step 1: Calculate the Source Density μ (The Forward Walk)

The source density μ is itself the solution to a boundary integral that combines the Dirichlet and Neumann conditions. The code uses getMixedConditionResultKernelForward to solve for the source density. 


Step 2: Calculate the Final Displacement u and gradient (The Main Integration)

The final displacement and gradient is calculated in solveGradientWOB function. 

3. Key Kernels Used
Kelvin Kernel (Somigliana Tensor): Implemented in kelvinKernel(...). This is the fundamental solution G. It is a 2x2 tensor and used in the final integration step inside solveGradientWOB.

Traction Kernel: Implemented in computeTractionKernel2D(...). This is the traction field of the Kelvin kernel, often denoted T or ∂G/∂n. It is used in the forward estimator.

4. Hard-coded Functions and Variables
This solver is configured for a specific geometry and problem setup. Many key parameters are hard-coded. To solve for other shape, please modify the following paramaters. 

4.1. Boundary and Domain Geometry
boundaryDirichlet: A global vector<Polyline> defining the left and right vertical sides of the unit square.

boundaryNeumann: A global vector<Polyline> defining the top edge and the bottom edge.

allBoundaryVertices: A hard-coded, ordered list of all vertices of the domain, used by the sampling functions.

4.2. Boundary Conditions
displacedPoints: A global vector<Polyline> that defines the new positions of the boundaryDirichlet lines. It's hard-coded to move the left side to x=-0.1 and the right side to x=1.1.

getDirichletValue(...): Calculates the prescribed displacement (u_x, u_y) for any point on the Dirichlet boundary by interpolating between the vertices defined in displacedPoints.

getNeumannValue(...): Calculates the prescribed traction (force). It's hard-coded to return (0, -0.1) on the top edge and (0, 0.1) on the bottom edge, simulating a small vertical pressure.

4.3. Other Core Functions
getNormal(point): Returns the outward-facing normal vector for a point on the boundary. Its logic is hard-coded for the specific segments of the concave shape.

uniformBoundarySampler(...): A general function that samples a random point from the entire boundary, with each point having an equal probability based on the total perimeter.

importanceSampleBoundary(...): A more advanced sampler that intelligently chooses points on the boundary. It gives a higher probability to sampling segments that subtend a larger angle from the perspective of the interior evaluation point x0. This greatly improves the efficiency of the main integration loop.

5. How to Compile and Run
5.1. Compilation
The code can be compiled from the command line using a C++17 compliant compiler (like g++ or clang++). You will need the Eigen library for linear algebra and OpenMP for parallelism.

# Example compilation command on macOS with Homebrew
c++ -std=c++17 -O3 -Wall -Xclang -fopenmp gradientEstimateLame.cpp -o gel -I /path/to/eigen/include -L /path/to/libomp/lib -lomp -w

Replace /path/to/eigen/include and /path/to/libomp/lib with the correct paths for your system.

In my case it is : c++ -std=c++17 -O3 -pedantic -Wall -Xclang -fopenmp gradientEstimateLame.cpp -o gel -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -I /opt/homebrew/opt/libomp/include -L /opt/homebrew/opt/libomp/lib -lomp -w

5.2. Running
The program takes no command-line arguments. It will run the simulation for a 16x16 grid of interior points.

./gel

5.3. Output
The program will generate two output files in the ../output/ directory:

lame_wob_xxx_displacements.csv: A CSV file containing the X and Y coordinates of each grid point and the calculated horizontal (dispX) and vertical (dispY) components of its displacement.

lame_wob_xxx_displacement_gradient.csv: A CSV file containing the each value of the 2 by 2 displacement gradient. 