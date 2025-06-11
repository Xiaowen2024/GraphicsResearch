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
#include <map>
#include "json.hpp"
#include <string>
using json = nlohmann::json;
// #include <SFML/Graphics.hp>
#include <cmath>
using namespace std;
using namespace std::chrono;

// c++ fractureSolver.cpp -std=c++17 -I/opt/homebrew/Cellar/sfml/2.6.2/include -o fsolver -L/opt/homebrew/Cellar/sfml/2.6.2/lib -lsfml-graphics -lsfml-window -lsfml-system -w
// c++ -std=c++17 -O3 -pedantic -Wall fractureSolver.cpp -o fsolver -w
// debug option: c++ -g -std=c++17 -O0 fractureSolver.cpp -o fsolver -I/usr/local/include/eigen3 -w

const int WIDTH = 6;
const int HEIGHT = 4;

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

Vec2D displacement(Vec2D v, vector<Polyline> boundaryDirichlet, vector<Polyline> displacedBoundaryDirichlet) {
    Vec2D nan = numeric_limits<double>::quiet_NaN();
    Vec2D interpolatedDisplacement = interpolateVec2DBoundaryPoints(v, boundaryDirichlet, displacedBoundaryDirichlet);
    if (isnan(real(interpolatedDisplacement)) || isnan(imag(interpolatedDisplacement))) {
       return nan;
    }
    return interpolatedDisplacement;
}


vector<Polyline> extractBoundaries(string inputFilePath, string key) {
    ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        cerr << "Error opening file: " << inputFilePath << endl;
        return {};
    }

    nlohmann::json data;
    inputFile >> data;

    vector<Polyline> boundaries;
    if (data.contains(key)) {
        Polyline polyline = {};
        for (const auto& point : data[key]) {
            double x = point["x"];
            nlohmann::basic_json<>::value_type y = point["y"];
            polyline.push_back(Vec2D(x, y));
        }
        if (polyline.size() > 0) {
            boundaries.push_back(polyline);
        }
    } else {
        cerr << "Key not found in JSON: " << key << endl;
    }
    inputFile.close();
    return boundaries;
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
double calculateCaseAStressIntensityFactor(double crackLength, double planeWidth, vector<Vec2D> stressTensor, Vec2D normal){
    double normalStress = getNormalStress(stressTensor, normal);
    double alpha = crackLength / planeWidth;
    double secant = 1 / cos( M_PI * alpha / 2);
    double Y = sqrt(secant);
    double K_I = Y * sqrt(alpha * M_PI) * normalStress;
    return K_I;
}

double calculateCaseAStressIntensityFactorForNotch(double crackLength, double planeWidth, vector<Vec2D> stressTensor, Vec2D normal){
    double normalStress = getNormalStress(stressTensor, normal);
    double alpha = crackLength / planeWidth;
    double Y = (1.12 + alpha * ( 2.91 * alpha - 0.64)) / ( 1 - 0.93 * alpha);
    double K_I = normalStress * sqrt( M_PI * crackLength) * Y;
    return K_I;
}


double calculateCriticalLength(double crackLength, double planeWidth, vector<Vec2D> stressTensor, Vec2D normal, double KIC){
    double normalStress = getNormalStress(stressTensor, normal);
    double alpha = crackLength / planeWidth;
    double secant = 1 / cos( M_PI * alpha / 2);
    double Y = sqrt(secant);
    double denominator = Y * sqrt(M_PI) * normalStress;
    return (KIC / denominator) * (KIC / denominator);
}

double calculateCriticalLengthForNotch(double crackLength, double planeWidth, vector<Vec2D> stressTensor, Vec2D normal, double KIC){
    double normalStress = getNormalStress(stressTensor, normal);
    double alpha = crackLength / planeWidth;
    double Y = (1.12 + alpha * ( 2.91 * alpha - 0.64)) / ( 1 - 0.93 * alpha);
    double denominator = normalStress * sqrt(M_PI) * Y;
    return (KIC / denominator) * (KIC / denominator);
}

// below are all under the assumption of plane strain
double calculateK1(double shearModulus, double poissonRatio, double distance, double displacement) {
    return shearModulus * sqrt( 2 * M_PI ) * displacement / ( sqrt(distance) * (2 - 2 * poissonRatio));
}

double calculateK2(double shearModulus, double poissonRatio, double distance, double displacement) {
    return shearModulus * sqrt( 2 * M_PI ) * displacement / ( sqrt(distance) * (2 - 2 * poissonRatio));
}

double calculateK3(double shearModulus, double poissonRatio, double distance, double displacement) {
    return shearModulus * sqrt( M_PI ) * displacement / ( sqrt(2 * distance) ) ;
}

double calculateEffectiveSIF (double shearModulus, double poissonRatio, double distance, double displacement1, double displacement2, double displacement3) {
    double K1 = calculateK1(shearModulus, poissonRatio, distance, displacement1);
    double K2 = calculateK2(shearModulus, poissonRatio, distance, displacement2);
    double K3 = calculateK3(shearModulus, poissonRatio, distance, displacement3);
    return sqrt( K1 * K1 + K2 * K2 + K3 * K3 / ( 1 - poissonRatio) );
}

double calculateCrackPropagationRate(double rayleighWaveSpeed, double KIC, double Keff) {
    return Keff > KIC ? rayleighWaveSpeed * ((1- KIC * KIC) / (Keff * Keff)) : 0;
}

Vec2D determineCrackPropagationDirection(double K1, double K2, double K3, double poissonRatio) {
    double K4 = K1 * K1 + K3 * K3 / ( 1 - poissonRatio);
    return 2 * atan( (K4 - sqrt( K4 * K4 + 8 *  K2 * K2)) / ( 4 * K2));
}

// assume we have constant force during a single cycle of loading
// the result is da/dN where N is the number of cycles
// for now assume each cycle is 1 second
pair<Vec2D, double> calculateCrackGrowthDirectionAndRateForCrackInitiation(vector<Vec2D> stressTensor, double materialConstant, double parisExponent, double SIF) {
    Vec2D growthDirection = determineCrackPropagationDirection(stressTensor);
    // calculate the growth rate according to Paris's law
    double rate = materialConstant * pow(SIF, parisExponent);
    return {growthDirection, rate};
}

vector<Vec2D> calculateStrain(vector<Vec2D> displacementGradient){
    vector<Vec2D> strain;
    vector<Vec2D> transpose = getTransposeMatrix(displacementGradient);
    strain.push_back( Vec2D(0.5 * (real(displacementGradient[0]) + real(transpose[0])) , 0.5 * (imag(displacementGradient[0]) + imag(transpose[0]))) );
    strain.push_back( Vec2D(0.5 * (real(displacementGradient[1]) + real(transpose[1])) , 0.5 * (imag(displacementGradient[1]) + imag(transpose[1]))) );
    return strain;
}

// using Griffth's energy release rate formula, we can get a lower bound for the length of the crack
double estimateCrackLength(double KIC, double stress){
    double denominator = M_PI * stress * stress;
    if (denominator == 0) {
        return 0;
    }
    return KIC * KIC / denominator;
}


// assume the crack plane is on the x axis, the normal will therefore be (0, 1)
// asume crackStarting point is at (0, 0), crack tip is (0, 1), crack direction is (0, 1)
void growCrackTip(Vec2D crackTip, Vec2D crackLength, Vec2D point, double displacement1, double displacement2, double displacement3, Vec2D normal, double shearModulus, double poissonRatio, double rayleighWaveSpeed, double KIC, function<Vec2D(Vec2D)> deform, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann){
    string shape = "crackGrowthExperiment1";
    std::ofstream strainFile("../output/" + shape + "_strain.csv");
    std::ofstream stressFile("../output/" + shape + "_stress.csv");
    std::ofstream displacementFile("../output/" + shape + "_displacement.csv");
    double distance = length(point - crackTip);
    double SIF = calculateEffectiveSIF(shearModulus, poissonRatio, distance, displacement1, displacement2, displacement3);

    if (SIF > KIC){ // TODO: change back to while loop
        double rate = calculateCrackPropagationRate(rayleighWaveSpeed, KIC, SIF);
        Vec2D growthDirection = determineCrackPropagationDirection(displacement1, displacement2, displacement3, poissonRatio);
        crackTip = crackTip + Vec2D{dot(normal, growthDirection) * rate * real(growthDirection), dot(normal, growthDirection) * rate * imag(growthDirection)};
        crackLength = crackLength + dot(normal, growthDirection) * rate;
    }
}

// pair<Vec2D, double> adpativeSamplingHelper(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, vector<Polyline>& displacedBoundaryDirichlet, double stepsize, double& maxStress, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> displacement, double lam, double mu){
    // if (stepsize < 1e-3 ) {
    //     return {point, maxStress};
    // }
    // Vec2D pointLeft = point + Vec2D{-stepsize, 0};
    // Vec2D pointRight = point + Vec2D{stepsize, 0};
    // Vec2D displacedPointLeft;
    // Vec2D displacedPointRight;
    // vector<Vec2D> displacementGradientLeft = solveGradient(pointLeft, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPointLeft);
    // vector<Vec2D> strainLeft = calculateStrain(displacementGradientLeft);
    // vector<Vec2D> stressTensorLeft = getStress(lam, mu, real(strainLeft[0]) + imag(strainLeft[1]), real(strainLeft[0]), imag(strainLeft[0]), real(strainLeft[1]), imag(strainLeft[1]));
    // auto principleStressLeft = eigenDecomposition(stressTensorLeft);
    // double stressMagnitudeLeft = abs(principleStressLeft[0].first);
    // vector<Vec2D> displacementGradientRight = solveGradient(pointRight, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPointRight);
    // vector<Vec2D> strainRight = calculateStrain(displacementGradientRight);
    // vector<Vec2D> stressTensorRight = getStress(lam, mu, real(strainRight[0]) + imag(strainRight[1]), real(strainRight[0]), imag(strainRight[0]), real(strainRight[1]), imag(strainRight[1]));
    // auto principleStressRight = eigenDecomposition(stressTensorRight);
    // double stressMagnitudeRight = abs(principleStressRight[0].first);

    // double stressMagnitude = max(stressMagnitudeLeft, stressMagnitudeRight);
    // if ( stressMagnitude > maxStress) {
    //     maxStress = stressMagnitude;
    //     stepsize /= 2;
    //     auto maxStressPairLeft = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, maxStress, stepsize, displacement, lam, mu);
    //     auto maxStressPairRight = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, maxStress, stepsize, displacement, lam, mu);
    //     if (maxStressPairLeft.second > maxStress) {
    //         return maxStressPairLeft;
    //     } else if (maxStressPairRight.second > maxStress) {
    //         return maxStressPairRight;
    //     }
    //     else {
    //         return {stressMagnitudeLeft > stressMagnitudeRight ? pointLeft : pointRight, maxStress};
    //     }
    // }
    // return {point, maxStress};
// }

// map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> findCrackTipsFromBoundaries(int round, function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> displacement, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, vector<Polyline> displacedBoundaryDirichlet, double materialStrength, double lam, double mu, std::ofstream& crackInformationFile) {
//     map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> crackTips;

    // for (const auto& boundary : {boundaryDirichlet, boundaryNeumann}) {
    //     for (const auto& polyline : boundary) {
    //         for (size_t i = 0; i < polyline.size(); ++i) {

    //             Vec2D originalPoint = polyline[i];
    //             Vec2D point = originalPoint;
    //             if (insideDomain(point + Vec2D(1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
    //                point = point + Vec2D(1e-3, 1e-3);
    //             }
    //             else if (insideDomain(point - Vec2D(1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
    //                point = point - Vec2D(1e-3, 1e-3);
    //             }
    //             else if (insideDomain(point + Vec2D(-1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
    //                point = point + Vec2D(-1e-3, 1e-3);
    //             }
    //             else if (insideDomain(point - Vec2D(-1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
    //                point = point - Vec2D(-1e-3, 1e-3);
    //             }
    //             else {
    //                 cout << "Round: " << round << endl;
    //                 cout << "Point is not inside the domain: " << real(point) << ", " << imag(point) << endl;
    //                 continue;
    //             }
    //             Vec2D displacedPoint = displacement(originalPoint, boundaryDirichlet, displacedBoundaryDirichlet);
    //             Vec2D displacedPointAnalytic =  displacement(originalPoint, boundaryDirichlet, displacedBoundaryDirichlet);
    //             vector<Vec2D> displacementGradient = solveGradient(point, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPoint);
    //             vector<Vec2D> strain = calculateStrain(displacementGradient);
    //             vector<Vec2D> stressTensor = getStress(lam, mu, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
    //             auto stressDecomposed = eigenDecomposition(stressTensor);
    //             double stressMagnitude = stressDecomposed[0].first;
    //             Vec2D direction = stressDecomposed[0].second;
    //             if (stressMagnitude > materialStrength) {
    //                 crackInformationFile << round << ", " << real(point) << ", " << imag(point) << ", " << real(displacedPointAnalytic) << ", " << imag(displacedPointAnalytic) << ", " << real(displacementGradient[0]) << ", " << imag(displacementGradient[0]) << ", " << real(displacementGradient[1]) << ", " << imag(displacementGradient[1]) << "\n";
    //                 crackTips.insert({originalPoint, {{displacedPointAnalytic, stressMagnitude}, direction}});
    //                 cout << "Round: " << round << ", Crack tip: " << real(originalPoint) << ", " << imag(originalPoint) << ", Displaced point: " << real(displacedPointAnalytic) << ", " << imag(displacedPointAnalytic) << ", Stress magnitude: " << stressMagnitude << endl;
    //             }
    //             else {
    //                 cout << "Round: " << round << ", Crack tip (failed): " << real(originalPoint) << ", " << imag(originalPoint) << ", Displaced point: " << real(displacedPointAnalytic) << ", " << imag(displacedPointAnalytic) << ", Stress magnitude: " << stressMagnitude << endl;
    //             }
    //         }
    //     }
    // }
    // return crackTips;
// }

map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> findCrackTips(
    int round,
    function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> displacement,
    vector<Polyline> boundaryDirichlet,
    vector<Polyline> boundaryNeumann,
    vector<Polyline> displacedBoundaryDirichlet,
    double materialStrength,
    double lam,
    double mu,
    std::ofstream& crackInformationFile
) {
    double maxStress = 0.0;
    map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> crackTips;
    double initialStep = 0.05;

    std::ofstream csvFile("/Users/xiaowenyuan/GraphicsResearch/fractureOutput/stress.csv");
    csvFile << "PointX, PointY, Stress, DirectionX, DirectionY, ClosestBoundary\n";

    std::vector<std::vector<Polyline>> boundaries = {boundaryDirichlet, boundaryNeumann};
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const auto& boundary = boundaries[i];
        bool isNeumann = (i == 1);
        for (const auto& polyline : boundary) {
            for (size_t i = 0; i + 1 < polyline.size(); ++i) {
                Vec2D p0 = polyline[i];
                Vec2D p1 = polyline[i + 1];
                Vec2D direction = p1 - p0;
                double length = std::abs(direction);
                direction = direction / length;
                if (isNeumann){
                    initialStep = 0.01;
                }

                for (double t = 0.0; t < length; t += initialStep) {
                    Vec2D coarsePoint = p0 + t * direction;

                    if (insideDomain(coarsePoint + Vec2D(1e-2, 1e-2), boundaryDirichlet, boundaryNeumann)) {
                        coarsePoint += Vec2D(1e-2, 1e-2);
                    } else if (insideDomain(coarsePoint - Vec2D(1e-2, 1e-2), boundaryDirichlet, boundaryNeumann)) {
                        coarsePoint -= Vec2D(1e-2, 1e-2);
                    } else if (insideDomain(coarsePoint + Vec2D(-1e-2, 1e-2), boundaryDirichlet, boundaryNeumann)) {
                        coarsePoint += Vec2D(-1e-2, 1e-2);
                    } else if (insideDomain(coarsePoint - Vec2D(-1e-2, 1e-2), boundaryDirichlet, boundaryNeumann)) {
                        coarsePoint -= Vec2D(-1e-2, 1e-2);
                    }
                    else {
                        cout << "Round: " << round << endl;
                        cout << "Coarse point is not inside the domain: " << real(coarsePoint) << ", " << imag(coarsePoint) << endl;
                        continue;
                    }

                    Vec2D displacedPoint;
                    vector<Vec2D> coarseGradient = solveGradient(coarsePoint, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPoint);
                    vector<Vec2D> strain = calculateStrain(coarseGradient);
                    vector<Vec2D> stressTensor = getStress(lam, mu, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
                    auto stressDecomposed = eigenDecomposition(stressTensor);
                    double coarseStress = abs(stressDecomposed[0].first);
                    Vec2D coarseDirection = stressDecomposed[0].second;

                    Vec2D closesBoundary = closestPoint(coarsePoint, p0, p1);
                    csvFile << real(coarsePoint) << "," << imag(coarsePoint) << "," << coarseStress << ","
                            << real(coarseDirection) << "," << imag(coarseDirection) << ", " << closesBoundary << "\n";

                    // if (coarseStress > maxStress) {
                    //     double refineStress = coarseStress;
                    //     double refineStep = 0.01;
                    //     auto [refinedPoint, refinedStress] = adpativeSamplingHelper(coarsePoint, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, refineStep, refineStress, displacement, lam, mu);
                    //     maxStress = max(coarseStress, refinedStress);
                    //     cout << "Refined stress: " << refinedStress << " at point: " << real(refinedPoint) << ", " << imag(refinedPoint) << endl;

                    //     if (refinedStress > materialStrength) {
                    //         Vec2D displaced = displacement(refinedPoint, boundaryDirichlet, displacedBoundaryDirichlet);
                    //         vector<Vec2D> refinedGradient = solveGradient(refinedPoint, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displaced);
                    //         vector<Vec2D> refinedStrain = calculateStrain(refinedGradient);
                    //         vector<Vec2D> refinedStressTensor = getStress(lam, mu, real(refinedStrain[0]) + imag(refinedStrain[1]), real(refinedStrain[0]), imag(refinedStrain[0]), real(refinedStrain[1]), imag(refinedStrain[1]));
                    //         auto refinedDecomposed = eigenDecomposition(refinedStressTensor);
                    //         Vec2D crackDirection = refinedDecomposed[0].second;

                    //         crackInformationFile << round << ", " << real(refinedPoint) << ", " << imag(refinedPoint) << ", "
                    //                             << real(displaced) << ", " << imag(displaced) << ", "
                    //                             << real(refinedGradient[0]) << ", " << imag(refinedGradient[0]) << ", "
                    //                             << real(refinedGradient[1]) << ", " << imag(refinedGradient[1]) << ", " << refinedStress << "\n";

                    //         Vec2D closestBoundary = closestPoint(refinedPoint, p0, p1);
                    //         crackTips.insert({closestBoundary, {{displaced, refinedStress}, crackDirection}});
                    //     }
                    //     else if (coarseStress > materialStrength){
                    //         crackInformationFile << round << ", " << real(coarsePoint) << ", " << imag(coarsePoint) << ", "
                    //         << real(displacedPoint) << ", " << imag(displacedPoint) << ", "
                    //         << real(coarseGradient[0]) << ", " << imag(coarseGradient[0]) << ", "
                    //         << real(coarseGradient[1]) << ", " << imag(coarseGradient[1]) << ", " << coarseStress << "\n";

                    //         Vec2D closestBoundary = closestPoint(coarsePoint, p0, p1);
                    //         crackTips.insert({closestBoundary, {{displacedPoint, coarseStress}, coarseDirection}});
                    //     }
                    // }
                    if (coarseStress > materialStrength){
                        crackInformationFile << round << ", " << real(coarsePoint) << ", " << imag(coarsePoint) << ", "
                        << real(displacedPoint) << ", " << imag(displacedPoint) << ", "
                        << real(coarseGradient[0]) << ", " << imag(coarseGradient[0]) << ", "
                        << real(coarseGradient[1]) << ", " << imag(coarseGradient[1]) << ", " << coarseStress << "\n";

                        Vec2D closestBoundary = closestPoint(coarsePoint, p0, p1);
                        if (crackTips.find(closestBoundary) != crackTips.end()) {
                            double existingStress = crackTips[closestBoundary].first.second;
                            if (coarseStress > existingStress) {
                                // Update the existing crack tip with the new one
                                crackTips[closestBoundary] = {{displacedPoint, coarseStress}, coarseDirection};
                            }
                        }
                        crackTips.insert({closestBoundary, {{displacedPoint, coarseStress}, coarseDirection}});
                    }
                }
            }
        }
    }
    return crackTips;
}

vector<vector<Polyline>> updateBoundaries(int round, string fileName, map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> crackTips, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, vector<Polyline> displacedBoundaryDirichlet, std::ofstream& boundaryUpdateFile) {
    vector<Polyline> boundaryDirichletUpdated;
    vector<Polyline> displacedBoundaryDirichletUpdated;
    vector<Polyline> boundaryNeumannUpdated;
    boundaryDirichletUpdated.push_back({});
    displacedBoundaryDirichletUpdated.push_back({});
    boundaryNeumannUpdated.push_back({});
    vector<Polyline> boundaryNeumannOriginal = boundaryNeumann;
    std::vector<std::vector<Polyline>> boundaries = {boundaryDirichlet, boundaryNeumann};
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const auto& boundary = boundaries[i];
        bool isNeumann = (i == 1);
        for (const auto& polyline : boundary) {
            for (const auto& point : polyline) {
                cout << "Processing point: " << real(point) << ", " << imag(point) << endl;
            auto it = find_if(crackTips.begin(), crackTips.end(), [&](const auto& entry) {
                return abs(real(entry.first) - real(point)) < 1e-5 && abs(imag(entry.first) - imag(point)) < 1e-5;
            });

            if (it != crackTips.end()) {
                auto displacedPoint = crackTips[point].first.first;
                Vec2D crackDirection = crackTips[point].second;
                Vec2D crackDirectionNormalized = crackDirection / length(crackDirection);
                Vec2D leftCrackPosition;
                Vec2D rightCrackPosition;
                if (imag(crackDirection) > 0) {
                    leftCrackPosition = displacedPoint - crackDirectionNormalized * 0.01;
                    rightCrackPosition = displacedPoint + crackDirectionNormalized * 0.01;
                } else {
                    leftCrackPosition = displacedPoint + crackDirectionNormalized * 0.01;
                    rightCrackPosition = displacedPoint - crackDirectionNormalized * 0.01;
                }

                Vec2D crackTipPosition1 = displacedPoint + rotate90(crackDirectionNormalized) * 0.02;
                Vec2D crackTipPosition2 = displacedPoint - rotate90(crackDirectionNormalized) * 0.02;
                bool inside1 = insideDomain(crackTipPosition1, displacedBoundaryDirichlet, boundaryNeumannOriginal);
                bool inside2 = insideDomain(crackTipPosition2, displacedBoundaryDirichlet, boundaryNeumannOriginal);
               
                
                if (!inside1 && !inside2) {
                    cout << "Crack tip position is not inside the domain: " << real(crackTipPosition1) << ", " << imag(crackTipPosition1) << endl;
                    continue;
                }
                Vec2D crackTipPosition = insideDomain(crackTipPosition1, boundaryDirichlet, boundaryNeumannOriginal) ? crackTipPosition1 : crackTipPosition2;

                boundaryNeumannUpdated[0].push_back(leftCrackPosition);
                boundaryNeumannUpdated[0].push_back(crackTipPosition);
                boundaryNeumannUpdated[0].push_back(rightCrackPosition);

                boundaryUpdateFile << round << ", " << real(point) << ", " << imag(point) << ", "
                                     << real(displacedPoint) << ", " << imag(displacedPoint) << ", "
                                     << real(leftCrackPosition) << ", " << imag(leftCrackPosition) << ", "
                                     << real(crackTipPosition) << ", " << imag(crackTipPosition) << ", "
                                     << real(rightCrackPosition) << ", " << imag(rightCrackPosition) << "\n";
            }
            else {
                if (isNeumann) {
                    cout << "Point is not a crack tip: " << real(point) << ", " << imag(point) << endl;
                    boundaryNeumannUpdated[0].push_back(point);
                } else {
                    // cout << "Point is not a crack tip in Dirichlet boundary: " << real(point) << ", " << imag(point) << endl;
                    boundaryDirichletUpdated[0].push_back(point);
                    auto index = &point - &polyline[0];
                    if (index < displacedBoundaryDirichlet[0].size()) {
                        displacedBoundaryDirichletUpdated[0].push_back(displacedBoundaryDirichlet[0][index]);
                    }
                }
            }
        }
    }
    boundaryDirichlet = boundaryDirichletUpdated;
    displacedBoundaryDirichlet = displacedBoundaryDirichletUpdated;
    boundaryNeumann = boundaryNeumannUpdated;
    }
    return {boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet};
}

void visualizeBoundaries(const vector<Polyline>& boundaries, const string& filename = "boundaries.svg") {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open output file.\n";
        return;
    }
    file << "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='400' viewBox='0 0 2 2'>\n";
    file << "<g fill='none' stroke='black' stroke-width='0.2'>\n";

    for (const auto& polyline : boundaries) {
        file << "<polyline points='";
        for (const auto& point : polyline) {
            file << real(point) << "," << (1 - imag(point)) << " "; // flip y for SVG coordinate system
        }
        file << "' />\n";
    }

    file << "</g>\n</svg>\n";
    file.close();
    cout << "SVG written to " << filename << endl;
}

int main( int argc, char** argv ) {
    string fileName = "neumannTipHorizontalStretchNotch_2";
    std::ofstream crackInformationFile("../fractureOutput/" + fileName + "CrackInfo.csv");
    std::ofstream boundaryUpdateFile("../fractureOutput/" + fileName + "BoundaryUpdates.csv");
    std::ofstream jsonFile("../fractureCases/" + fileName + "BoundaryRecords.json");
    crackInformationFile << "Round, CrackTipX, CrackTipY, DisplacedX, DisplacedY, DuDx, DuDy, DvDx, DvDy, Stress\n";
    boundaryUpdateFile << "Round, CrackTipX, CrackTipY, DisplacedX, DisplacedY, LeftCrackX, LeftCrackY, CrackTipX, CrackTipY, RightCrackX, RightCrackY\n";
    const string configPath = "../fractureCases/" + fileName + ".json";
    auto boundaryDirichlet = extractBoundaries(configPath, "boundaryDirichlet");
    auto boundaryNeumann = extractBoundaries(configPath, "boundaryNeumann");
    auto displacedBoundaryDirichlet = extractBoundaries(configPath, "displacedBoundaryDirichlet");

    // map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> crackTips;
    // crackTips.insert({Vec2D(0.5, 0.2), {{Vec2D(0.5 - 0.00118704, 0.000648081 + 0.205), 269.357}, Vec2D{0.99999, 0.00443}}});

    int count = 0;
    for (int i = 0; i < 1; i++) {
        auto results = findCrackTips(count, displacement, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, 200, 100, 80, crackInformationFile);
        cout << "Round: " << count << ", Found " << results.size() << " crack tips." << endl;
        if (results.empty()) {
            cout << "No crack tips found in round " << count << ". Skipping boundary update." << endl;
            return 0;
        }
        else {
            cout << "Crack tips found in round " << count << ": " << endl;
            for (const auto& tip : results) {
                cout << "Crack tip at: " << real(tip.first) << ", " << imag(tip.first) << ", Displaced point: " 
                     << real(tip.second.first.first) << ", " << imag(tip.second.first.first) 
                     << ", Stress magnitude: " << tip.second.first.second 
                     << ", Direction: " << real(tip.second.second) << ", " << imag(tip.second.second) << endl;
            }
        }
        // auto boundaries = updateBoundaries(count, fileName, results, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, boundaryUpdateFile);
        // boundaryDirichlet = boundaries[0];
        // boundaryNeumann = boundaries[1];
        // displacedBoundaryDirichlet = boundaries[2];
        count++;
    }

    // auto boundaries = updateBoundaries(0, fileName, crackTips, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, boundaryUpdateFile);
    // boundaryDirichlet = boundaries[0];
    // boundaryNeumann = boundaries[1];
    // displacedBoundaryDirichlet = boundaries[2];

    // Vec2D displacedPoint;
    // vector<Vec2D> coarseGradient = solveGradient(Vec2D(0.5, 0.201), boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPoint);
    // vector<Vec2D> strain = calculateStrain(coarseGradient);
    // vector<Vec2D> stressTensor = getStress(100, 80, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
    // auto stressDecomposed = eigenDecomposition(stressTensor);
    // cout << "Round: "  <<  "gradient: " << real(coarseGradient[0]) << ", " << imag(coarseGradient[1]) <<
    // ", " << real(coarseGradient[1]) <<   ", " << imag(coarseGradient[1]) << ", Stress magnitude: " << abs(stressDecomposed[0].first) << ", " << "coarse direction: " << real(stressDecomposed[0].second) << ", " << imag(stressDecomposed[0].second) << endl;

    // if (jsonFile.is_open()) {
    //     json j;
    //     j["boundaryDirichlet"].clear();
    //     j["boundaryNeumann"].clear();
    //     j["displacedBoundaryDirichlet"].clear();

    //     for (const auto& polyline : boundaryDirichlet) {
    //         for (const auto& point : polyline) {
    //             j["boundaryDirichlet"].push_back({{"x", real(point)}, {"y", imag(point)}});
    //         }
    //     }
    //     for (const auto& polyline : boundaryNeumann) {
    //         for (const auto& point : polyline) {
    //             j["boundaryNeumann"].push_back({{"x", real(point)}, {"y", imag(point)}});
    //         }
    //     }
    //     for (const auto& polyline : displacedBoundaryDirichlet) {
    //         for (const auto& point : polyline) {
    //             j["displacedBoundaryDirichlet"].push_back({{"x", real(point)}, {"y", imag(point)}});
    //         }
    //     }
    //     jsonFile << j.dump(4);
    //     jsonFile.close();
    // }
    // else {
    //     cerr << "Failed to open JSON file for writing.\n";
    // }

    // Vec2D displacedPoint;
    // double lam = 100; 
    // double mu = 80;
    // vector<Vec2D> coarseGradient = solveGradient(Vec2D(0.51, 0.21), boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPoint);
    // vector<Vec2D> strain = calculateStrain(coarseGradient);
    // vector<Vec2D> stressTensor = getStress(lam, mu, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
    // auto stressDecomposed = eigenDecomposition(stressTensor);
    // double coarseStress = abs(stressDecomposed[0].first);
    // Vec2D coarseDirection = stressDecomposed[0].second;

    // cout << "Round: "  <<  "gradient: " << real(coarseGradient[0]) << ", " << imag(coarseGradient[1]) << 
    // ", " << real(coarseGradient[1]) <<   ", " << imag(coarseGradient[1]) << ", Stress magnitude: " << coarseStress << ", " << "coarse direction: " << real(coarseDirection) << ", " << imag(coarseDirection) << endl;

    // int totalRound = 2;
    // for (int count = 0; count < totalRound; count ++){
    //     auto results = findCrackTips(count, displacement, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, 200, 100, 80, crackInformationFile);
    //     cout << "Round: " << count << ", Found " << results.size() << " crack tips." << endl;
    //     if (results.empty()) {
    //         cout << "No crack tips found in round " << count << ". Skipping boundary update." << endl;
    //         return 0;
    //     }
    //     updateBoundaries(count, fileName, results, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, boundaryUpdateFile);
    //     if (jsonFile.is_open()) {
    //         json j;
    //         for (const auto& polyline : boundaryDirichlet) {
    //             for (const auto& point : polyline) {
    //                 j["boundaryDirichlet"].push_back({{"x", real(point)}, {"y", imag(point)}});
    //             }
    //         }
    //         for (const auto& polyline : boundaryNeumann) {
    //             for (const auto& point : polyline) {
    //                 j["boundaryNeumann"].push_back({{"x", real(point)}, {"y", imag(point)}});
    //             }
    //         }
    //         for (const auto& polyline : displacedBoundaryDirichlet) {
    //             for (const auto& point : polyline) {
    //                 j["displacedBoundaryDirichlet"].push_back({{"x", real(point)}, {"y", imag(point)}});
    //               }
    //         }
    //         jsonFile << j.dump(4);
    //         jsonFile.close();
    //     } else {
    //         cerr << "Failed to open JSON file for writing.\n";
    //     }
    // }
}

