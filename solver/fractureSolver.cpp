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
#include <cmath>
using namespace std;
using namespace std::chrono; 

// c++ fractureSolver.cpp -std=c++17 -I/opt/homebrew/Cellar/sfml/2.6.2/include -o fsolver -L/opt/homebrew/Cellar/sfml/2.6.2/lib -lsfml-graphics -lsfml-window -lsfml-system

const int WIDTH = 800;
const int HEIGHT = 600;
vector<Polyline> boundaryDirichlet = { {{Vec2D(200, 150), Vec2D(200, 450), Vec2D(600, 450), Vec2D(600, 150), Vec2D(200, 150)}} };
vector<Polyline> boundaryNeumann = {};

void plotStressVector(vector<pair<double, Vec2D>> tensilePair, vector<pair<double, Vec2D>> compressivePair, Vec2D crackTip, sf::RenderWindow& window) {
    Vec2D tensileVector = tensilePair[0].second;
    Vec2D compressiveVector = compressivePair[0].second;
    cout << "Tensile vector: " << tensileVector << endl;
    cout << "Compressive vector: " << compressiveVector << endl;
    sf::VertexArray tensileLine(sf::Lines, 2);
    tensileLine[0].position = sf::Vector2f(real(crackTip), imag(crackTip));
    tensileLine[0].color = sf::Color::Red;
    tensileLine[1].position = sf::Vector2f(real(crackTip) + real(tensileVector) * 50, imag(crackTip) + imag(tensileVector) * 50);
    tensileLine[1].color = sf::Color::Red;

    sf::VertexArray compressiveLine(sf::Lines, 2);
    compressiveLine[0].position = sf::Vector2f(real(crackTip), imag(crackTip));
    compressiveLine[0].color = sf::Color::Yellow;
    compressiveLine[1].position = sf::Vector2f(real(crackTip) + real(compressiveVector) * 50, imag(crackTip) + imag(compressiveVector) * 50);
    compressiveLine[1].color = sf::Color::Yellow;

    window.draw(tensileLine);
    window.draw(compressiveLine);
}

pair<vector<pair<double, Vec2D>>, vector<pair<double, Vec2D>>> startCrackPropagation(vector<Polyline> boundaryDirichlet, function<Vec2D(Vec2D)> deform, Vec2D crackTip, sf::RenderWindow& window) {
    double h = 10;
    string shape = "rect-left-corner_h" + to_string(h);
    int s = 16;
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
        vector<Vec2D> stress = getDeformationGradientAndStress(crackTip, h, deform, strainFile, neighbourFile, stressFile, boundaryDirichlet, boundaryNeumann);
        vector<pair<double, Vec2D>> eigenpairs = eigenDecomposition(stress);
        pair<vector<Vec2D>, vector<Vec2D>> forcePair = forceDecomposition(stress, eigenpairs);
        vector<pair<double, Vec2D>> tensilePair = eigenDecomposition(forcePair.first);
        vector<pair<double, Vec2D>> compressivePair = eigenDecomposition(forcePair.second);
        return {tensilePair, compressivePair};
    }
}

Vec2D crackTip = Vec2D(400, 450);

Vec2D deformLeftCorner( Vec2D point ) {
    double x = real(point);
    double y = imag(point);
    if (x == 200 && y == 450) {
        return Vec2D( x - (600 - x) * 0.2, y);
    }
    else {
        return Vec2D(x, y);
    }
}

// calculate case a: long strip, central crack, tensile stress 
// TODO: need to confirm the correctness of normal stress
float calculateCaseAStressIntensityFactor(float crackLength, float planeWidth, vector<Vec2D> stressTensor, Vec2D normal){
    float normalStress = getNormalStress(stressTensor, normal);
    float alpha = crackLength / planeWidth;
    float secant = 1 / cos( M_PI * alpha / 2);
    float Y = sqrt(secant); 
    float K_I = Y * sqrt(alpha * M_PI) * normalStress;
    return K_I;
}

int main() {
    // Initialize SFML window
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Crack Propagation");
    window.setFramerateLimit(30);

    sf::ConvexShape shape;
    int pointSize = boundaryDirichlet[0].size();
    shape.setPointCount(pointSize); 

    for (size_t i = 0; i < pointSize; ++i) {
        shape.setPoint(i, sf::Vector2f(real(boundaryDirichlet[0][i]), imag(boundaryDirichlet[0][i])));
    }

    shape.setFillColor(sf::Color::White);
    shape.setOutlineThickness(2.f);
    shape.setOutlineColor(sf::Color::Blue);

    sf::CircleShape crackTipShape(5);
    crackTipShape.setFillColor(sf::Color::Red);
    crackTipShape.setPosition(real(crackTip) - crackTipShape.getRadius(), imag(crackTip) - crackTipShape.getRadius());
    window.draw(crackTipShape);

    // Start crack propagation
    startCrackPropagation(boundaryDirichlet, deformLeftCorner, crackTip, window);
    // pair<vector<pair<double, Vec2D>>, vector<pair<double, Vec2D>>> stressPair = startCrackPropagation(boundaryDirichlet, deformLeftCorner, crackTip, window);
    // plotStressVector(stressPair.first, stressPair.second, crackTip, window);

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.draw(crackTipShape);
        window.display();
    }

    return 0;
}
