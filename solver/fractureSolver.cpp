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
// #include <SFML/Graphics.hpp>
#include <cmath>
using namespace std;
using namespace std::chrono; 

// c++ fractureSolver.cpp -std=c++17 -I/opt/homebrew/Cellar/sfml/2.6.2/include -o fsolver -L/opt/homebrew/Cellar/sfml/2.6.2/lib -lsfml-graphics -lsfml-window -lsfml-system -w
// c++ -std=c++17 -O3 -pedantic -Wall fractureSolver.cpp -o fsolver  -w

const int WIDTH = 6;
const int HEIGHT = 4;

vector<Polyline> boundaryDirichlet = {{ Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0) }};
vector<Polyline> boundaryNeumann = { {Vec2D(0, 0), Vec2D(0.5, 0.2), Vec2D(1, 0)} };
vector<Polyline> displacedPoints =  {{ Vec2D(1.2, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(-0.2, 0) }};

Vec2D interpolateVec2DBoundaryPoints(Vec2D v, vector<Polyline> originalPoints, vector<Polyline> displacedPoints, double num_tol=1e-5) { 
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
 
    Vec2D nan = numeric_limits<double>::quiet_NaN();
 
    Vec2D interpolatedDisplacement = interpolateVec2DBoundaryPoints(v, boundaryDirichlet, displacedPoints);
    if (isnan(real(interpolatedDisplacement)) || isnan(imag(interpolatedDisplacement))) {
       return nan;
    }
    return interpolatedDisplacement;
}

// void plotStressVector(vector<pair<double, Vec2D>> tensilePair, vector<pair<double, Vec2D>> compressivePair, Vec2D crackTip, sf::RenderWindow& window) {
//     Vec2D tensileVector = tensilePair[0].second;
//     Vec2D compressiveVector = compressivePair[0].second;
//     cout << "Tensile vector: " << tensileVector << endl;
//     cout << "Compressive vector: " << compressiveVector << endl;
//     sf::VertexArray tensileLine(sf::Lines, 2);
//     tensileLine[0].position = sf::Vector2f(real(crackTip), imag(crackTip));
//     tensileLine[0].color = sf::Color::Red;
//     tensileLine[1].position = sf::Vector2f(real(crackTip) + real(tensileVector) * 50, imag(crackTip) + imag(tensileVector) * 50);
//     tensileLine[1].color = sf::Color::Red;

//     sf::VertexArray compressiveLine(sf::Lines, 2);
//     compressiveLine[0].position = sf::Vector2f(real(crackTip), imag(crackTip));
//     compressiveLine[0].color = sf::Color::Yellow;
//     compressiveLine[1].position = sf::Vector2f(real(crackTip) + real(compressiveVector) * 50, imag(crackTip) + imag(compressiveVector) * 50);
//     compressiveLine[1].color = sf::Color::Yellow;

//     window.draw(tensileLine);
//     window.draw(compressiveLine);
// }

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
    strain.push_back( Vec2D(0.5 * (real(displacementGradient[0]) + real(transpose[0])) , 0.5 * (imag(displacementGradient[0]) + imag(transpose[0]))) );
    strain.push_back( Vec2D(0.5 * (real(displacementGradient[1]) + real(transpose[1])) , 0.5 * (imag(displacementGradient[1]) + imag(transpose[1]))) );
    // cout << "displacementGradient[0]: " << real(displacementGradient[0]) << " " << imag(displacementGradient[0]) << endl;
    // cout << "displacementGradient[1]: " << real(displacementGradient[1]) << " " << imag(displacementGradient[1]) << endl;
    // cout << "transpose[0]: " << real(transpose[0]) << " " << imag(transpose[0]) << endl;
    // cout << "transpose[1]: " << real(transpose[1]) << " " << imag(transpose[1]) << endl;
    // cout << real(displacementGradient[0]) + real(transpose[0]) << " " << imag(displacementGradient[0]) + imag(transpose[0]) << endl;
    // cout << real(displacementGradient[1]) + real(transpose[1]) << " " << imag(displacementGradient[1]) + imag(transpose[1]) << endl;
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

pair<Vec2D, double> adpativeSamplingHelper(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, double stepsize, double& maxStress, function<Vec2D(Vec2D)> deform){
    if (stepsize < 1e-3 ) {
        return {point, maxStress};
    } 
    Vec2D pointLeft = point + Vec2D{-stepsize, 0};
    Vec2D pointRight = point + Vec2D{stepsize, 0};
    vector<Vec2D> displacementGradientLeft = solveGradient(pointLeft, boundaryDirichlet, boundaryNeumann, deform);
    displacementGradientLeft[0] -= Vec2D(1, 0);
    displacementGradientLeft[1] -= Vec2D(0, 1);
    vector<Vec2D> strainLeft = calculateStrain(displacementGradientLeft);
    vector<Vec2D> stressTensorLeft = getStress(1.0, 0.1, real(strainLeft[0]) + imag(strainLeft[1]), real(strainLeft[0]), imag(strainLeft[0]), real(strainLeft[1]), imag(strainLeft[1]));
    auto principleStressLeft = eigenDecomposition(stressTensorLeft);
    double stressMagnitudeLeft = max(abs(principleStressLeft[0].first), abs(principleStressLeft[1].first));

    vector<Vec2D> displacementGradientRight = solveGradient(pointRight, boundaryDirichlet, boundaryNeumann, deform);
    displacementGradientRight[0] -= Vec2D(1, 0);
    displacementGradientRight[1] -= Vec2D(0, 1);
    vector<Vec2D> strainRight = calculateStrain(displacementGradientRight);
    vector<Vec2D> stressTensorRight = getStress(1.0, 0.1, real(strainRight[0]) + imag(strainRight[1]), real(strainRight[0]), imag(strainRight[0]), real(strainRight[1]), imag(strainRight[1]));
    auto principleStressRight = eigenDecomposition(stressTensorRight);
    double stressMagnitudeRight = max(abs(principleStressRight[0].first), abs(principleStressRight[1].first));

    double stressMagnitude = max(stressMagnitudeLeft, stressMagnitudeRight);
    if ( stressMagnitude > maxStress) {
        maxStress = stressMagnitude;
        stepsize /= 2;
        auto maxStressPairLeft = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, stepsize, maxStress, deform);
        auto maxStressPairRight = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, stepsize, maxStress, deform);
        if (maxStressPairLeft.second > maxStress) {
            return maxStressPairLeft;
        } else if (maxStressPairRight.second > maxStress) {
            return maxStressPairRight;
        }
        else {
            return {stressMagnitudeLeft > stressMagnitudeRight ? pointLeft : pointRight, maxStress};
        }
    }
    return {point, maxStress};
}

pair<Vec2D, double> initializeCrackTip(function<Vec2D(Vec2D)> deform, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, double materialStrength){
    Vec2D crackTip = numeric_limits<double>::quiet_NaN();
    double stepSize = 0.1;
    double maxStress = 0.0;
    double h = 1e-5;

    for (const auto& polyline : boundaryDirichlet) {
        for (size_t i = 0; i < polyline.size() - 1; ++i) {
            Vec2D startPoint = polyline[i];
            Vec2D endPoint = polyline[i + 1];
            Vec2D direction = (endPoint - startPoint) / length(endPoint - startPoint);
            double segmentLength = length(endPoint - startPoint);

            for (double t = stepSize; t < segmentLength; t += stepSize) {
                Vec2D point = startPoint + direction * t;
                Vec2D pointLeft = point + Vec2D{-h, -imag(direction) * h};
                Vec2D pointRight = point + Vec2D{h, imag(direction) * h};
                Vec2D pointTop = point + Vec2D{real(direction) * h, h};
                Vec2D pointBottom = point + Vec2D{-real(direction) * h, -h};
                vector<Vec2D> neighbors = {pointLeft, pointRight, pointTop, pointBottom};
                vector<Vec2D> neighborsDeformed;
                for (const auto& neighbor : neighbors) {
                    Vec2D interpolatedPoint = interpolateVec2DBoundaryPoints(neighbor, boundaryDirichlet, displacedPoints);
                    if (isnan(real(interpolatedPoint)) || isnan(imag(interpolatedPoint))) {
                        // std::cerr << "Error: Point is not on the boundary." << std::endl;
                        continue;
                    }
                    neighborsDeformed.push_back(interpolatedPoint);
                } 
                if (neighborsDeformed.size() != 4) {
                    // std::cerr << "Error: Not enough deformed neighbors." << std::endl;
                    continue;
                }
                double dudx = (real(neighborsDeformed[1]) - real(neighborsDeformed[0])) / (2 * h);
                double dudy = (real(neighborsDeformed[2]) - real(neighborsDeformed[3])) / (2 * h);
                double dvdx = (imag(neighborsDeformed[1]) - imag(neighborsDeformed[0])) / (2 * h);
                double dvdy = (imag(neighborsDeformed[2]) - imag(neighborsDeformed[3])) / (2 * h);

                vector<Vec2D> displacementGradient = {Vec2D(dudx, dudy), Vec2D(dvdx, dvdy)};
                displacementGradient[0] -= Vec2D(1, 0);
                displacementGradient[1] -= Vec2D(0, 1);
                // cout << "displacementGradient: " << real(displacementGradient[0]) << ", " << imag(displacementGradient[0]) << endl;
                // cout << "displacementGradient: " << real(displacementGradient[1]) << ", " << imag(displacementGradient[1]) << endl;
                vector<Vec2D> strain = calculateStrain(displacementGradient);
                // cout << "strain: " << real(strain[0]) << ", " << imag(strain[0]) << endl;
                // cout << "strain: " << real(strain[1]) << ", " << imag(strain[1]) << endl;
                vector<Vec2D> stressTensor = getStress(1.0, 0.1, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
                auto stressDecomposed = eigenDecomposition(stressTensor);
                double stressMagnitude = max(abs(stressDecomposed[0].first), abs(stressDecomposed[1].first));
                if (stressMagnitude > maxStress) {
                    maxStress = stressMagnitude;
                    crackTip = point;
                    stepSize = max(stepSize / 2, 0.01); 
                }
            }
        }
    }

    stepSize = 0.1;

    for (const auto& polyline : boundaryNeumann) {
        for (size_t i = 0; i < polyline.size() - 1; ++i) {
            Vec2D startPoint = polyline[i];
            Vec2D endPoint = polyline[i + 1];
            Vec2D direction = (endPoint - startPoint) / length(endPoint - startPoint);
            double segmentLength = length(endPoint - startPoint);

            double t = stepSize;
            while (t <= segmentLength) { 
                Vec2D point = startPoint + direction * t + Vec2D{0, 1e-3};
                cout << "point: " << real(point) << ", " << imag(point) << endl;
                vector<Vec2D> displacementGradient = solveGradient(point, boundaryDirichlet, boundaryNeumann, deform);
                displacementGradient[0] -= Vec2D(1, 0);
                displacementGradient[1] -= Vec2D(0, 1);
                cout << "displacementGradient: " << real(displacementGradient[0]) << ", " << imag(displacementGradient[0]) << endl;
                cout << "displacementGradient: " << real(displacementGradient[1]) << ", " << imag(displacementGradient[1]) << endl;
                vector<Vec2D> strain = calculateStrain(displacementGradient);
                cout << "strain: " << real(strain[0]) << ", " << imag(strain[0]) << endl;
                cout << "strain: " << real(strain[1]) << ", " << imag(strain[1]) << endl;
                vector<Vec2D> stressTensor = getStress(1.0, 0.1, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
                auto stressDecomposed = eigenDecomposition(stressTensor);
                double stressMagnitude = max(abs(stressDecomposed[0].first), abs(stressDecomposed[1].first));
                cout << "Stress magnitude after recursive sampling: " << stressMagnitude << endl;
                cout << "Max stress: " << maxStress << endl;

                if (stressMagnitude > maxStress) {
                    maxStress = stressMagnitude;
                    crackTip = point;   

                    double previousStress = maxStress;
                    stepSize /= 2;

                    pair<Vec2D, double> stressRecursiveResult = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, stepSize, previousStress, deform);
                    if (stressRecursiveResult.second > maxStress) {
                        maxStress = stressRecursiveResult.second;
                        crackTip = stressRecursiveResult.first;
                        cout << "Updated crack tip: " << real(crackTip) << ", " << imag(crackTip) << endl;
                    } 
                } else {
                    stepSize *= 1.5; 
                }

                t += stepSize;
            }
        }
    }

    if (maxStress > materialStrength) {
        std::cout << "Crack tip found at: " << real(crackTip) << ", " << imag(crackTip) << std::endl;
    } else {
        cout << maxStress << " is not greater than material strength: " << materialStrength << std::endl;
        std::cout << "No crack tip found." << std::endl;
    }

    return {crackTip, maxStress};
}


int main( int argc, char** argv ) {
    auto result = initializeCrackTip(displacement, boundaryDirichlet, boundaryNeumann, 0.5);
    cout << "Crack tip: " << real(result.first) << ", " << imag(result.first) << endl;
    cout << "Max stress: " << result.second << endl;
    // auto gradient = solveGradient(Vec2D(0.369611, 0.225), boundaryDirichlet, boundaryNeumann, displacement);
    // gradient[0] -= Vec2D(1, 0);
    // gradient[1] -= Vec2D(0, 1);
    // vector<Vec2D> strain = calculateStrain(gradient);
    // vector<Vec2D> stressTensor = getStress(1.0, 0.1, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
    // auto stressDecomposed = eigenDecomposition(stressTensor);
    // double stressMagnitude = max(abs(stressDecomposed[0].first), abs(stressDecomposed[1].first));
    // cout << "Stress magnitude: " << stressMagnitude << endl;
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
