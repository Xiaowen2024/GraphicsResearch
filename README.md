# Walk of Stars Method for Shape Generation

This README provides instructions on how to use the Walk of Stars method for generating heights for different shapes.

## Steps to Generate New Shapes

### 1. Modify `WoStLaplace2D.cpp`

#### a. Define Boundary Values

Define the boundary values for Dirichlet and Neumann conditions:

```cpp
vector<Polyline> boundaryDirichlet = {
    // Add boundary values here
};

vector<Polyline> boundaryNeumann = {
    // Add boundary values here
};
```

You can either add values directly or use a function to populate the vector, such as createSaddlePointBoundary() in the script.

#### b. Define Height Evaluation Function

Define a height evaluation function so that when the random walker reaches a boundary point, it returns the height at the specified location. An example is getSaddlePointHeight(Vec2D x) in the script.

#### c. Modify the Main Function

In the main function, modify the arguments passed to the solve function with your defined boundaryDirichlet, boundaryNeumann, and heightEvaluationFunction:

```cpp
u = solve( x0, boundaryDirichlet, boundaryNeumann, heightEvaluationFunction);
```

Also change the output csv path for ofstream out( "saddlePointNe.csv" ); 

### 2. Compile and Run

Compile the program using:

```bash
c++ -std=c++17 -O3 -pedantic -Wall WoStLaplace2D.cpp -o wost
```

Run the program with:

```bash
./wost
```

After the program finishes, you should see a new csv file in the solver folder with the height values.

### 3. Visualize the Output

Use visualizer.ipynb to visualize the height field by calling:

```python
visualize_csv_output(csv_path, output_image_path)
```