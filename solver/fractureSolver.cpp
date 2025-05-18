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
            double y = point["y"];
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

vector<double> getCrackOpeningDisplacement(Vec2D crackTip, Vec2D pointN1, Vec2D pointN2) {
    double displacement1 = abs(real(pointN1) - real(pointN2));
    double displacement2 = abs(imag(pointN1) - imag(pointN2));
    double displacement3 = 0;
    return {displacement1, displacement2, displacement3};
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

// pair<Vec2D, double> adpativeSamplingHelper(Vec2D point, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, double stepsize, double& maxStress, function<Vec2D(Vec2D)> deform, double lam, double mu){
//     if (stepsize < 1e-3 ) {
//         return {point, maxStress};
//     } 
//     Vec2D pointLeft = point + Vec2D{-stepsize, 0};
//     Vec2D pointRight = point + Vec2D{stepsize, 0};
//     vector<Vec2D> displacementGradientLeft = solveGradient(pointLeft, boundaryDirichlet, boundaryNeumann, deform);
//     displacementGradientLeft[0] -= Vec2D(1, 0);
//     displacementGradientLeft[1] -= Vec2D(0, 1);
//     vector<Vec2D> strainLeft = calculateStrain(displacementGradientLeft);
//     vector<Vec2D> stressTensorLeft = getStress(lam, mu, real(strainLeft[0]) + imag(strainLeft[1]), real(strainLeft[0]), imag(strainLeft[0]), real(strainLeft[1]), imag(strainLeft[1]));
//     auto principleStressLeft = eigenDecomposition(stressTensorLeft);
//     double stressMagnitudeLeft = abs(principleStressLeft[0].first);

//     vector<Vec2D> displacementGradientRight = solveGradient(pointRight, boundaryDirichlet, boundaryNeumann, deform);
//     displacementGradientRight[0] -= Vec2D(1, 0);
//     displacementGradientRight[1] -= Vec2D(0, 1);
//     vector<Vec2D> strainRight = calculateStrain(displacementGradientRight);
//     vector<Vec2D> stressTensorRight = getStress(lam, mu, real(strainRight[0]) + imag(strainRight[1]), real(strainRight[0]), imag(strainRight[0]), real(strainRight[1]), imag(strainRight[1]));
//     auto principleStressRight = eigenDecomposition(stressTensorRight);
//     double stressMagnitudeRight = abs(principleStressRight[0].first);

//     double stressMagnitude = max(stressMagnitudeLeft, stressMagnitudeRight);
//     if ( stressMagnitude > maxStress) {
//         maxStress = stressMagnitude;
//         stepsize /= 2;
//         auto maxStressPairLeft = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, stepsize, maxStress, deform, lam, mu);
//         auto maxStressPairRight = adpativeSamplingHelper(point, boundaryDirichlet, boundaryNeumann, stepsize, maxStress, deform, lam, mu);
//         if (maxStressPairLeft.second > maxStress) {
//             return maxStressPairLeft;
//         } else if (maxStressPairRight.second > maxStress) {
//             return maxStressPairRight;
//         }
//         else {
//             return {stressMagnitudeLeft > stressMagnitudeRight ? pointLeft : pointRight, maxStress};
//         }
//     }
//     return {point, maxStress};
// }

map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> findCrackTipsFromBoundaries(function<Vec2D(Vec2D, vector<Polyline>, vector<Polyline>)> displacement, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, vector<Polyline> displacedBoundaryDirichlet, double materialStrength, double lam, double mu, std::ofstream& crackInformationFile) {
    map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> crackTips;
    for (const auto& boundary : {boundaryDirichlet, boundaryNeumann}) {
        for (const auto& polyline : boundary) {
            for (size_t i = 0; i < polyline.size(); ++i) {
                Vec2D point = polyline[i];
                if (insideDomain(point + Vec2D(1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
                   point = point + Vec2D(1e-3, 1e-3);
                }
                else if (insideDomain(point - Vec2D(1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
                   point = point - Vec2D(1e-3, 1e-3);
                }
                else if (insideDomain(point + Vec2D(-1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
                   point = point + Vec2D(-1e-3, 1e-3);
                }
                else if (insideDomain(point - Vec2D(-1e-3, 1e-3), boundaryDirichlet, boundaryNeumann)) {
                   point = point - Vec2D(-1e-3, 1e-3);
                }
                else {
                    continue;
                } 
                Vec2D displacedPoint = displacement(point, boundaryDirichlet, displacedBoundaryDirichlet);
                vector<Vec2D> displacementGradient = solveGradient(point, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, displacement, displacedPoint);
                vector<Vec2D> strain = calculateStrain(displacementGradient);
                vector<Vec2D> stressTensor = getStress(lam, mu, real(strain[0]) + imag(strain[1]), real(strain[0]), imag(strain[0]), real(strain[1]), imag(strain[1]));
                auto stressDecomposed = eigenDecomposition(stressTensor);
                double stressMagnitude = stressDecomposed[0].first;
                Vec2D direction = stressDecomposed[0].second;
                if (stressMagnitude > materialStrength) {
                    crackInformationFile << 0 << ", " << real(point) << ", " << imag(point) << ", " << real(displacedPoint) << ", " << imag(displacedPoint) << ", " << real(displacementGradient[0]) << ", " << imag(displacementGradient[0]) << ", " << real(displacementGradient[1]) << ", " << imag(displacementGradient[1]) << "\n";
                    crackTips.insert({point, {{displacedPoint, stressMagnitude}, direction}});
                }
            }
        }
    }
    return crackTips;
}

void updateBoundaries(int round, string fileName, map<Vec2D, pair<pair<Vec2D, double>, Vec2D>, ComplexLess> crackTips, vector<Polyline> boundaryDirichlet, vector<Polyline> boundaryNeumann, vector<Polyline> displacedBoundaryDirichlet) {
    for (auto& polyline : boundaryDirichlet) {
        for (auto& point : polyline) {
            if (crackTips.find(point) != crackTips.end()) {
                auto displacedPoint = crackTips[point].first.first;
                auto it = find(displacedBoundaryDirichlet[0].begin(), displacedBoundaryDirichlet[0].end(), point);
                Vec2D crackDirection = crackTips[point].second;
                Vec2D crackDirectionNormalized = crackDirection / length(crackDirection);
                Vec2D leftCrackPosition;
                Vec2D rightCrackPosition;
                if (imag(crackDirection) > 0) {
                    leftCrackPosition = displacedPoint - crackDirectionNormalized * 0.05;
                    rightCrackPosition = displacedPoint + crackDirectionNormalized * 0.05;
                } else {
                    leftCrackPosition = displacedPoint + crackDirectionNormalized * 0.05;
                    rightCrackPosition = displacedPoint - crackDirectionNormalized * 0.05;
                }
                
                Vec2D crackTipPosition1 = displacedPoint + rotate90(crackDirectionNormalized) * 0.05;
                Vec2D crackTipPosition2 = displacedPoint - rotate90(crackDirectionNormalized) * 0.05;
                Vec2D crackTipPosition = insideDomain(crackTipPosition1, boundaryDirichlet, boundaryNeumann) ? crackTipPosition1 : crackTipPosition2;
                
                if (it != displacedBoundaryDirichlet[0].end()) {
                    *it = leftCrackPosition;
                    displacedBoundaryDirichlet[0].insert(it + 1, crackTipPosition);
                    displacedBoundaryDirichlet[0].insert(it + 2, rightCrackPosition);
                }
            }
        }
    }
}

int main( int argc, char** argv ) {
    string fileName = "dirichletHorizontalStretchNotch";
    std::ofstream crackInformationFile("../fractureOutput/" + fileName + "CrackInfo.csv");
    crackInformationFile << "Round, CrackTipX, CrackTipY, DisplacedX, DisplacedY, DuDx, DuDy, DvDx, DvDy\n";
    const string configPath = "../fractureCases/" + fileName + ".json";
    auto boundaryDirichlet = extractBoundaries(configPath, "boundaryDirichlet");
    auto boundaryNeumann = extractBoundaries(configPath, "boundaryNeumann");
    auto displacedBoundaryDirichlet = extractBoundaries(configPath, "displacedBoundaryDirichlet");

    auto results = findCrackTipsFromBoundaries(displacement, boundaryDirichlet, boundaryNeumann, displacedBoundaryDirichlet, 100, 100, 80, crackInformationFile);
    for (auto& result : results) {
        cout << "Crack tip: " << real(result.first) << ", " << imag(result.first) << endl;
        cout << "Stress magnitude: " << result.second.first.second << endl;
    }
}

