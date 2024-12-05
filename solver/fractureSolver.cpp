#include "fractureModelHelpers.cpp"
#include "fractureModelHelpers.h"
#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <chrono> 
#include <SFML/Graphics.hpp>
using namespace std;
using namespace std::chrono; 

const int WIDTH = 800;
const int HEIGHT = 600;

void startCrackPropagation(vector<Polyline> boundaryDirichlet, function<Vec2D(Vec2D)> deform, Vec2D crackTip) {
    string shape = "rect-left-corner_h_0.1";
    int s = 16;
    double h = 0.1;
    std::ofstream strainFile("../output/deformation_gradient_" + shape + ".csv");
    std::ofstream neighbourFile("../output/deformation_gradient_" + shape + "_neighbour_displacements.csv");
    std::ofstream displacementFile("../output/deformation_gradient_" + shape + "_displacements.csv");
    std::ofstream stressFile ("../output/deformation_gradient_" + shape + "_stresses.csv");
    double x = real(crackTip);
    double y = imag(crackTip);
    double lam = 1.0;
    double mu = 1.0;
    Vec2D left{ x - h/2, y };
    Vec2D right{ x + h/2, y };
    Vec2D top{ x, y + h/2 };
    Vec2D bottom{ x, y - h/2 };
    Vec2D solved_vec = NAN;
    if (insideDomain(crackTip, boundaryDirichlet, boundaryNeumann) && insideDomain(left, boundaryDirichlet, boundaryNeumann) && insideDomain(right, boundaryDirichlet, boundaryNeumann) && insideDomain(top, boundaryDirichlet, boundaryNeumann) && insideDomain(bottom, boundaryDirichlet, boundaryNeumann) ){
        vector<Vec2D> stress = getDeformationGradientAndStress(crackTip, h, deform, strainFile, neighbourFile, stressFile);
        vector<pair<double, Vec2D>> eigenpairs = eigenDecomposition(stress);
        pair<vector<Vec2D>, vector<Vec2D>> forcePair = forceDecomposition(stress, eigenpairs);
        vector<pair<double, Vec2D>> tensilePair = eigenDecomposition(forcePair.first);
        vector<pair<double, Vec2D>> shearPair = eigenDecomposition(forcePair.second);

    }
}

vector<Polyline> boundaryDirichlet = { {{Vec2D(0, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)}} };
Vec2D crackTip = Vec2D(0.5, 0);

Vec2D deformLeftCorner( Vec2D point ) {
    double x = real(point);
    double y = imag(point);
    if (x == 0 && y == 0) {
        return Vec2D( x - (1 - x) * 0.2, y);
    }
    else {
        return Vec2D(x, y);
    }
}

int main() {
    // Initialize SFML window
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Crack Propagation");
    window.setFramerateLimit(30);

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
    }

    sf::ConvexShape shape;
    int pointSize = boundaryDirichlet[0].size();
    shape.setPointCount(pointSize); 

    // Set each point
    for (size_t i = 0; i < pointSize; ++i) {
        shape.setPoint(i, sf::Vector2f(real(boundaryDirichlet[0][i]), imag(boundaryDirichlet[0][i])));
    }

    shape.setFillColor(sf::Color::Cyan);
    shape.setOutlineThickness(2.f);
    shape.setOutlineColor(sf::Color::Blue);

    return 0;
}
