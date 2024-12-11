// g++ renderer.cpp -I/opt/homebrew/Cellar/sfml/2.6.2/include -o app -L/opt/homebrew/Cellar/sfml/2.6.2/lib -lsfml-graphics -lsfml-window -lsfml-system
#include <SFML/Graphics.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>


// Constants
const int WIDTH = 800;
const int HEIGHT = 600;
const int NUM_PARTICLES = 5000; // Number of particles
const float PARTICLE_RADIUS = 2.0f;
const float INTERACTION_RADIUS = 20.0f;
const float FAILURE_THRESHOLD = 10.0f; // Stress threshold for bond fail
using namespace std;

std::vector<sf::Vector2f> points = {
    sf::Vector2f(200, 150),
    sf::Vector2f(400, 150),
    sf::Vector2f(600, 450),
    sf::Vector2f(200, 450)
};

int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Crack Propagation (Boundary-Free)");
    window.setFramerateLimit(30); 
    sf::VertexArray vertices(sf::Points, points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        vertices[i].position = sf::Vector2f(points[i].x, points[i].y);
        vertices[i].color = sf::Color::White;
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

    return 0;
}
