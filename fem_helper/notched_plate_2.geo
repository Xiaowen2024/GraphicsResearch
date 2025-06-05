// notched_plate.geo

// Mesh resolution
res = 0.01;

// Points (bottom boundary with notch - Neumann)
Point(1) = {0, 0, res};
Point(2) = {0.49, 0, res};
Point(3) = {0.5, 0.2, res};    // notch tip
Point(4) = {0.51, 0, res};
Point(5) = {1, 0, res};

// Top-left-right boundary (Dirichlet)
Point(6) = {1, 1, res};
Point(7) = {0, 1, res};

// Edges (bottom Neumann)
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};

// Edges (Dirichlet)
Line(5) = {5, 6}; // right
Line(6) = {6, 7}; // top
Line(7) = {7, 1}; // left

// Define outer boundary
Line Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

// Tag boundaries for FEM
Physical Line("neumann") = {1, 2, 3, 4};
Physical Line("dirichlet") = {5, 6, 7};
Physical Surface("domain") = {1};