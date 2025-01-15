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

const int WIDTH = 6;
const int HEIGHT = 4;
vector<Polyline> boundaryDirichlet = { {{Vec2D(-3, -2), Vec2D(3, -2), Vec2D(3, 2), Vec2D(-3, 2), Vec2D(-3, -2)}} };
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

float calciulateCriticalLength(float crackLength, float planeWidth, vector<Vec2D> stressTensor, Vec2D normal, float KIC){
    float normalStress = getNormalStress(stressTensor, normal);
    float alpha = crackLength / planeWidth;
    float secant = 1 / cos( M_PI * alpha / 2);
    float Y = sqrt(secant); 
    float denominator = Y * sqrt(M_PI) * normalStress;
    return (KIC / denominator) * (KIC / denominator);
}

// assume we have constant force during a single cycle of loading 
// the result is da/dN where N is the number of cycles 
// TODO: modify the formula according to static loading 
// for now assume each cycle is 1 second 
pair<Vec2D, float> calculateCrackGrowthDirectionAndRate(vector<Vec2D> stressTensor, float materialConstant, float parisExponent, float SIF) {
    Vec2D growthDirection = determineCrackPropagationDirection(stressTensor);
    // calculate the growth rate according to Paris's law 
    float rate = materialConstant * pow(SIF, parisExponent);
    return {growthDirection, rate};
}

vector<Vec2D> growHelper(Vec2D crackTip, float h, function<Vec2D(Vec2D)> deform, std::ofstream& strainFile, std::ofstream& neighbourFile, std::ofstream& stressFile, std::ofstream& displacementFile, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann){
    double x = real(crackTip);
    double y = imag(crackTip);
    Vec2D left{ x - h/2, y };
    Vec2D right{ x + h/2, y };
    Vec2D top{ x, y + h/2 };
    Vec2D bottom{ x, y - h/2 };
    Vec2D solved_vec = NAN;
    if( insideDomain(crackTip, boundaryDirichlet, boundaryNeumann) && insideDomain(left, boundaryDirichlet, boundaryNeumann) && insideDomain(right, boundaryDirichlet, boundaryNeumann) && insideDomain(top, boundaryDirichlet, boundaryNeumann) && insideDomain(bottom, boundaryDirichlet, boundaryNeumann) ){
        std::cout << "crackTip: " << real(crackTip) << ", " << imag(crackTip) << " inside domain" << std::endl;
        Vec2D displacedLocation = solve(crackTip, boundaryDirichlet, boundaryNeumann, deform);
        displacementFile << real(displacedLocation) << " " << imag(displacedLocation) << std::endl;
        vector<Vec2D> stressTensor = getDeformationGradientAndStress(crackTip, h, deform, strainFile, neighbourFile, stressFile, boundaryDirichlet, boundaryNeumann);
        return stressTensor;
    }
    else {
        return {};
    }
}

// assume the crack plane is on the x axis, the normal will therefore be (0, 1)
// asume crackStarting point is at (0, 0), crack tip is (0, 1), crack direction is (0, 1)
void growCrackTip(Vec2D crackTip, float crackLength, float planeWidth, Vec2D normal, float KIC, float materialConstant, float pairsExponent, float h, function<Vec2D(Vec2D)> deform, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann){
    string shape = "crackGrowthExperiment1";
    std::ofstream strainFile("../output/" + shape + "_strain.csv");
    std::ofstream neighbourFile("../output/" + shape + "_neighbour_displacements.csv");
    std::ofstream stressFile("../output/" + shape + "_stress.csv");
    std::ofstream displacementFile("../output/" + shape + "_displacement.csv");
    displacementFile << "displacedX, displacedY" << std::endl;
    vector<Vec2D> stressTensor = growHelper(crackTip, h, deform, strainFile, neighbourFile, stressFile, displacementFile, boundaryDirichlet, boundaryNeumann);
    if (stressTensor.empty()) {
        return;
    }
    std::cout << "stress magnitude: " << getNormalStress(stressTensor, normal) << std::endl;
    float SIF = calculateCaseAStressIntensityFactor(crackLength, planeWidth, stressTensor, normal);
    float criticalLength = calciulateCriticalLength(crackLength, planeWidth, stressTensor, normal, KIC);
    std::cout << "Stress Intensity Factor (SIF): " << SIF << std::endl;
    std::cout << "Critical Length: " << criticalLength << std::endl;
    while (crackLength < criticalLength && SIF < KIC){
        pair<Vec2D, float> growthInfo = calculateCrackGrowthDirectionAndRate(stressTensor, materialConstant, pairsExponent, SIF);
        if (dot(normal, growthInfo.first) * growthInfo.second <= 0){
            return;
        }
        crackTip = crackTip + Vec2D{dot(normal, growthInfo.first) * growthInfo.second * real(growthInfo.first), dot(normal, growthInfo.first) * growthInfo.second * imag(growthInfo.first)};
        stressTensor = growHelper(crackTip, h, deform, strainFile, neighbourFile, stressFile, displacementFile, boundaryDirichlet, boundaryNeumann);
        if (stressTensor.empty()) {
            return;
        }
        std::cout << "crack grew" << std::endl;
        std::cout << "stress magnitude: " << getNormalStress(stressTensor, normal) << std::endl;
        crackLength = crackLength + dot(normal, growthInfo.first) * growthInfo.second;
        criticalLength = calciulateCriticalLength(crackLength, planeWidth, stressTensor, normal, KIC);
        SIF = calculateCaseAStressIntensityFactor(crackLength, planeWidth, stressTensor, normal);
        std::cout << "Stress Intensity Factor (SIF): " << SIF << std::endl;
        std::cout << "Critical Length: " << criticalLength << std::endl;
        std::cout << "crackTip: " << real(crackTip) << ", " << imag(crackTip) << " with length: " << crackLength << endl;
    }
}


Vec2D deform(Vec2D point) {
    double x = real(point);
    double y = imag(point);
    if (x < 0){
        return Vec2D(x - 0.1, y);
    }
    else {
        return Vec2D(x + 0.1, y);
    }
}

int main( int argc, char** argv ) {
    // asumme KIC = 1, C = 12, N = 2
    growCrackTip(Vec2D{0, 0.01}, 0.01, 4, Vec2D{0, 1}, 1,  12, 2, 0.1, deform, boundaryDirichlet, boundaryNeumann);
}

// int render() {
//     // Initialize SFML window
//     sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Crack Propagation");
//     window.setFramerateLimit(30);

//     sf::ConvexShape shape;
//     int pointSize = boundaryDirichlet[0].size();
//     shape.setPointCount(pointSize); 

//     for (size_t i = 0; i < pointSize; ++i) {
//         shape.setPoint(i, sf::Vector2f(real(boundaryDirichlet[0][i]), imag(boundaryDirichlet[0][i])));
//     }

//     shape.setFillColor(sf::Color::White);
//     shape.setOutlineThickness(2.f);
//     shape.setOutlineColor(sf::Color::Blue);

//     // sf::CircleShape crackTipShape(5);
//     // crackTipShape.setFillColor(sf::Color::Red);
//     // crackTipShape.setPosition(real(crackTip) - crackTipShape.getRadius(), imag(crackTip) - crackTipShape.getRadius());
//     // window.draw(crackTipShape);

//     // Start crack propagation
//     // startCrackPropagation(boundaryDirichlet, deformLeftCorner, crackTip, window);
//     // pair<vector<pair<double, Vec2D>>, vector<pair<double, Vec2D>>> stressPair = startCrackPropagation(boundaryDirichlet, deformLeftCorner, crackTip, window);
//     // plotStressVector(stressPair.first, stressPair.second, crackTip, window);

//     // Main loop
//     while (window.isOpen()) {
//         sf::Event event;
//         while (window.pollEvent(event)) {
//             if (event.type == sf::Event::Closed)
//                 window.close();
//         }

//         // window.clear();
//         // window.draw(shape);
//         // window.draw(crackTipShape);
//         // window.display();
//     }

//     return 0;
// }
