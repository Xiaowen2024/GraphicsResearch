// notched_plate.geo

// Mesh resolution
res = 0.01;

// Points (bottom boundary with notch - Neumann)
Point(1) = {0, 0, res};
Point(2) = {1, 0, res};
Point(3) = {1, 1, res};   
Point(4) = {0, 1, res};

// Edges (bottom Neumann)
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define outer boundary
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Tag boundaries for FEM
Physical Line("bottom") = {1};
Physical Line("right")  = {2};
Physical Line("top")    = {3};
Physical Line("left")   = {4};
Physical Surface("domain") = {1};