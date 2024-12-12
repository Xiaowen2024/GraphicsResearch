// g++ renderer.cpp -I/opt/homebrew/Cellar/sfml/2.6.2/include -o app -L/opt/homebrew/Cellar/sfml/2.6.2/lib -lsfml-graphics -lsfml-window -lsfml-system
#include <SFML/Graphics.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "WostCrackPropagation.cpp"


// Constants
const int WIDTH = 800;
const int HEIGHT = 600;
const int NUM_PARTICLES = 5000; // Number of particles
const float PARTICLE_RADIUS = 2.0f;
const float INTERACTION_RADIUS = 20.0f;
const float FAILURE_THRESHOLD = 10.0f; // Stress threshold for bond fail
using namespace std;



int main() {
    std::vector<sf::Vector2f> points = std::vector<sf::Vector2f>();
    points.push_back(sf::Vector2f(200, 150));
    points.push_back(sf::Vector2f(600, 150));
    points.push_back(sf::Vector2f(600, 450));
    points.push_back(sf::Vector2f(200, 450));

    sf::Vector2f crackTip = sf::Vector2f(400, 450);
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Crack Propagation (Boundary-Free)");
    window.setFramerateLimit(30); 
    sf::VertexArray vertices(sf::Points, points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        vertices[i].position = sf::Vector2f(points[i].x, points[i].y);
        vertices[i].color = sf::Color::Blue;
    }
    sf::VertexArray lines(sf::LinesStrip, points.size() + 1);
    for (size_t i = 0; i < points.size(); ++i) {
        lines[i].position = points[i];
        lines[i].color = sf::Color::Red;
    }
    
    lines[points.size()].position = points[0]; 
    lines[points.size()].color = sf::Color::Red;
    sf::VertexArray shape(sf::LinesStrip, points.size() + 1);
    for (size_t i = 0; i < points.size(); ++i) {
        shape[i].position = points[i];
        shape[i].color = sf::Color::Blue;
    }
    shape[points.size()].position = points[0];
    shape[points.size()].color = sf::Color::Blue;

    sf::CircleShape crackTipShape(5);
    crackTipShape.setFillColor(sf::Color::Green);
    crackTipShape.setPosition(crackTip.x - crackTipShape.getRadius(), crackTip.y - crackTipShape.getRadius());
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(vertices);
        window.draw(shape);
        window.draw(crackTipShape);
        window.display();
    }
    

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(vertices);
        window.draw(shape);
        window.display();
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(vertices);
        window.display();
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.display();
    }

    Vec2D crackTipForce = getForce(Vec2D(crackTip.x, crackTip.y), Vec2D(-1, 0));

    return 0;
}
