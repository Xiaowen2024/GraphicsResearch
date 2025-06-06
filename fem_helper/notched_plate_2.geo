// notched_plate.geo

// Mesh resolution
res = 0.01;
// Points from boundaryDirichlet and boundaryNeumann
Point(1) = {0.51, 0.0, 1.0};
Point(2) = {1.0, 0.0, 1.0};
Point(3) = {1.0, 1.0, 1.0};
Point(4) = {0.0, 1.0, 1.0};
Point(5) = {0.0, 0.0, 1.0};
Point(6) = {0.49, 0.0, 1.0};
Point(7) = {0.50, 0.2, 1.0}; // Neumann middle point

// BoundaryDirichlet lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {4, 5};
Line(4) = {5, 6};

// BoundaryNeumann lines
Line(5) = {6, 7};
Line(6) = {7, 1};

// Connecting top-left to top-right (assuming closed boundary loop)
Line(7) = {3, 4};

// Create Line Loop for surface
Line Loop(10) = {1, 2, 7, 3, 4, 5, 6};

// Create surface
Plane Surface(20) = {10};

// Optionally define physical groups for boundary conditions
Physical Line("Dirichlet") = {1, 2, 3, 4};
Physical Line("Neumann") = {5, 6, 7};
Physical Surface("Domain") = {20};
