To try the Walk of Stars method on other shapes, do the following steps: 

In WoStLaplace2D.cpp: 

1. Define boundary values for
   
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

4. 
