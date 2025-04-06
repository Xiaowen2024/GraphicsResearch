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

vector<Polyline> boundaryDirichlet = {{ Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0) }};
vector<Polyline> boundaryNeumann = { {Vec2D(0, 0), Vec2D(0.5, 0.2), Vec2D(1, 0)} };

Vec2D interpolateVec2D_BoundaryPoints(Vec2D v, vector<Polyline> originalPoints, vector<Polyline> displacedPoints, double num_tol=1e-5) { 
    for (int i = 0; i < originalPoints[0].size() - 1; i++) {
       Vec2D AP = v - originalPoints[0][i];
       Vec2D PB = v - originalPoints[0][i + 1];
       Vec2D AB = originalPoints[0][i + 1] - originalPoints[0][i];
       Vec2D displaced1 = displacedPoints[0][i]; 
       Vec2D displaced2 = displacedPoints[0][i + 1];
 
       if (abs(length(AP) + length(PB) - length(AB)) < num_tol) {
          Vec2D displaced = displaced1 + (displaced2 - displaced1) * length(AP) / length(AB);
          return displaced;
       }
    }
 
    Vec2D nan = numeric_limits<double>::quiet_NaN();
    return nan;
}

Vec2D displacement(Vec2D v) { 
   
    vector<Polyline> displacedPoints =  {{ Vec2D(1.2, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(-0.2, 0) }};
 
    Vec2D nan = numeric_limits<double>::quiet_NaN();
 
    Vec2D interpolatedDisplacement = interpolateVec2D_BoundaryPoints(v, boundaryDirichlet, displacedPoints);
    if (isnan(real(interpolatedDisplacement)) || isnan(imag(interpolatedDisplacement))) {
       return nan;
    }
    return interpolatedDisplacement;
}

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

vector<Vec2D> calculateStrain(vector<Vec2D> displacementGradient){
    // strain = 1/2 ( displacementGradient + transpose(displacementGradient) )
    vector<Vec2D> strain;
    vector<Vec2D> transpose = getTransposeMatrix(displacementGradient);
    strain.push_back( Vec2D(1/2 * (real(displacementGradient[0]) + real(transpose[0])) , 1/2 * (imag(displacementGradient[0]) + imag(transpose[0]))) );
    strain.push_back( Vec2D(1/2 * (real(displacementGradient[1]) + real(transpose[1])) , 1/2 * (imag(displacementGradient[1]) + imag(transpose[1]))) );
    return strain;
}

vector<Vec2D> growHelper(Vec2D crackTip, float h, function<Vec2D(Vec2D)> deform, std::ofstream& strainFile, std::ofstream& displacementFile, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann){
    if( insideDomain(crackTip, boundaryDirichlet, boundaryNeumann) ){
        std::cout << "crackTip: " << real(crackTip) << ", " << imag(crackTip) << " inside domain" << std::endl;
        vector<Vec2D> displacementGradient = solveGradient(crackTip, boundaryDirichlet, boundaryNeumann, deform, strainFile, displacementFile);
        vector<Vec2D> strain = calculateStrain(displacementGradient);
        vector<Vec2D> stressTensor = getStress(1.0, 0.1, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
        return stressTensor;
    }
    return {};
}

// assume the crack plane is on the x axis, the normal will therefore be (0, 1)
// asume crackStarting point is at (0, 0), crack tip is (0, 1), crack direction is (0, 1)
void growCrackTip(Vec2D crackTip, float crackLength, float planeWidth, Vec2D normal, float KIC, float materialConstant, float pairsExponent, float h, function<Vec2D(Vec2D)> deform, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann){
    string shape = "crackGrowthExperiment1";
    std::ofstream strainFile("../output/" + shape + "_strain.csv");
    std::ofstream stressFile("../output/" + shape + "_stress.csv");
    std::ofstream displacementFile("../output/" + shape + "_displacement.csv");
    vector<Vec2D> stressTensor = growHelper(crackTip, h, deform, strainFile, displacementFile, boundaryDirichlet, boundaryNeumann);
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
        stressTensor = growHelper(crackTip, h, deform, strainFile, displacementFile, boundaryDirichlet, boundaryNeumann);
        if (stressTensor.empty()) {
            return;
        }
        std::cout << "crack grew" << std::endl;
        std::cout << "stress magnitude: " << getNormalStress(stressTensor, normal) << std::endl;
        crackLength = crackLength + dot(normal, growthInfo.first) * growthInfo.second;
        std::cout << "growth direction " << real(growthInfo.first) << " " << imag(growthInfo.second) << std::endl;
        criticalLength = calciulateCriticalLength(crackLength, planeWidth, stressTensor, normal, KIC);
        SIF = calculateCaseAStressIntensityFactor(crackLength, planeWidth, stressTensor, normal);
        std::cout << "Stress Intensity Factor (SIF): " << SIF << std::endl;
        std::cout << "Critical Length: " << criticalLength << std::endl;
        std::cout << "crackTip: " << real(crackTip) << ", " << imag(crackTip) << " with length: " << crackLength << endl;
    }
}

Vec2D initializeCrackTip(float crackLength, float planeWidth, Vec2D normal, float KIC, float materialConstant, float pairsExponent){
    Vec2D crackTip = Vec2D{0, 0};
    float SIF = calculateCaseAStressIntensityFactor(crackLength, planeWidth, stressTensor, normal);
    float criticalLength = calciulateCriticalLength(crackLength, planeWidth, stressTensor, normal, KIC);
    std::cout << "Stress Intensity Factor (SIF): " << SIF << std::endl;
    std::cout << "Critical Length: " << criticalLength << std::endl;
    return crackTip;
}

int main( int argc, char** argv ) {
    // asumme KIC = 1, C = 12, N = 2
    growCrackTip(Vec2D{0, 0.01}, 0.01, 4, Vec2D{0, 1}, 1,  12, 2, 0.1, displacement, boundaryDirichlet, boundaryNeumann);
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
