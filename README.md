# Walk of Stars Method for Shape Generation

This README provides instructions on how to use the Walk of Stars method for generating heights for different shapes.

## Steps to Generate New Shapes

### 1. Modify WoStLaplace2D.cpp

#### a. Define Boundary Values

Define the boundary values for Dirichlet and Neumann conditions:

vector<Polyline> boundaryDirichlet = {
};
vector<Polyline> boundaryNeumann = {

};

either add values directly or add a function to populate the vector such as the createSaddlePointBoundary( ) in the script. 

2. Define height evaluation function so that when the random walker reaches a boundary point, the function returns the height at the specified location.
An example is getSaddlePointHeight(Vec2D x) in the script. 

3. In the main function, modify the passed in arguments to the solve function with the defined boundaryDirichlet, boundaryNeumann and heightEvaluationFunction. 

u = solve( x0, boundaryDirichlet, boundaryNeumann, heightEvaluationFunction);

Also change the output csv path for ofstream out( "saddlePointNe.csv" ); 

4. Compile by c++ -std=c++17 -O3 -pedantic -Wall WoStLaplace2D.cpp -o wost

5. Run by ./wost

After the program finishes, you should see a new csv file in the solver folder with the height values. 

Use visualizer.ipynb to visualize the height field by calling visualize_csv_output(csv_path, output_image_path). 
