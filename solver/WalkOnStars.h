#pragma once

#include <complex>
#include <vector>
#include <functional>
using namespace std;

class WalkOnStars {
public:
    using Vec2D = std::complex<double>;
    using Polyline = std::vector<Vec2D>;

    WalkOnStars(const std::vector<Polyline>& boundaryDirichlet,
                const std::vector<Polyline>& boundaryNeumann,
                std::function<double(Vec2D)> interpolate);

    double solve(Vec2D x0, std::vector<Polyline> boundaryDirichlet, 
                          std::vector<Polyline> boundaryNeumann, 
                          std::function<double(Vec2D)> g);
    bool insideDomain( Vec2D x,
                   const std::vector<Polyline>& boundaryDirichlet,
                   const std::vector<Polyline>& boundaryNeumann );
    std::vector<Polyline> boundaryDirichlet;
    std::vector<Polyline> boundaryNeumann;
    std::function<double(Vec2D)> interpolate;

private:
    std::function<double(Vec2D)> g;
    double random( double rMin, double rMax);
    double length( Vec2D u );
    double angleOf(Vec2D u);
    Vec2D rotate90( Vec2D u );
    double dot(Vec2D u, Vec2D v);
    double cross(Vec2D u, Vec2D v);
    Vec2D closestPoint( Vec2D x, Vec2D a, Vec2D b );
    bool isSilhouette( Vec2D x, Vec2D a, Vec2D b, Vec2D c);
    double rayIntersection( Vec2D x, Vec2D v, Vec2D a, Vec2D b );
    double distancePolylines( Vec2D x, const std::vector<Polyline>& P );
    double silhouetteDistancePolylines( Vec2D x, const std::vector<Polyline>& P );
    Vec2D intersectPolylines( Vec2D x, Vec2D v, double r,
                         const std::vector<Polyline>& P,
                         Vec2D& n, bool& onBoundary );
    double signedAngle( Vec2D x, const std::vector<Polyline>& P );
    
};