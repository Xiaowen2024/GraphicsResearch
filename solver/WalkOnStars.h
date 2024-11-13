#pragma once

#include <complex>
#include <vector>
#include <functional>
#include <Eigen/Dense>
using namespace std;

class WalkOnStars {
public:
    using Vec2D = Eigen::Vector2d;
    using Polyline = vector<Vec2D>;

    WalkOnStars(const vector<Polyline>& boundaryDirichlet,
                const vector<Polyline>& boundaryNeumann,
                function<double(Vec2D)> interpolate);

    double solve(Vec2D x0, vector<Polyline> boundaryDirichlet, 
                          vector<Polyline> boundaryNeumann, 
                          function<double(Vec2D)> g);
                          
    bool insideDomain( Vec2D x,
                   const vector<Polyline>& boundaryDirichlet,
                   const vector<Polyline>& boundaryNeumann );
    vector<Polyline> boundaryDirichlet;
    vector<Polyline> boundaryNeumann;
    function<double(Vec2D)> interpolate;

private:
    function<double(Vec2D)> g;
    double random( double rMin, double rMax);
    Vec2D closestPoint( Vec2D x, Vec2D a, Vec2D b ); 
    Vec2D WalkOnStars::customCross(Vec2D u, Vec2D v);
    bool isSilhouette( Vec2D x, Vec2D a, Vec2D b, Vec2D c);
    double rayIntersection( Vec2D x, Vec2D v, Vec2D a, Vec2D b );
    double distancePolylines( Vec2D x, const vector<Polyline>& P );
    double silhouetteDistancePolylines( Vec2D x, const vector<Polyline>& P );
    Vec2D intersectPolylines( Vec2D x, Vec2D v, double r,
                         const vector<Polyline>& P,
                         Vec2D& n, bool& onBoundary );
    double signedAngle( Vec2D x, const vector<Polyline>& P );
};
