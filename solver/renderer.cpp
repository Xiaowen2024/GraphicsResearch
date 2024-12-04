// g++ renderer.cpp -I/opt/homebrew/Cellar/sfml/2.6.2/include -o app -L/opt/homebrew/Cellar/sfml/2.6.2/lib -lsfml-graphics -lsfml-window -lsfml-system
#include <SFML/Graphics.hpp>



// int main()
// {
// 	sf::Window window(
// 		sf::VideoMode(640, 480),
// 		"Hello World");
// 	while (window.isOpen()) {
// 		sf::Event event;
// 		while (window.pollEvent(event))
// 			if (event.type == 
// 			sf::Event::Closed)
// 				window.close();
// 	}
// 	return 0;
// }



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
const float FAILURE_THRESHOLD = 10.0f; // Stress threshold for bond failure

struct Particle {
    sf::Vector2f position;
    sf::Vector2f velocity;
    bool isBroken = false; // Tracks if the particle is "cracked"
};

// Function to initialize particles randomly within the material
void initializeParticles(std::vector<Particle>& particles) {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle p;
        p.position = sf::Vector2f(std::rand() % WIDTH, std::rand() % HEIGHT);
        p.velocity = sf::Vector2f(0.0f, 0.0f);
        particles.push_back(p);
    }
}

// Compute stress between two particles
float computeStress(const Particle& p1, const Particle& p2) {
    float distance = std::hypot(p2.position.x - p1.position.x, p2.position.y - p1.position.y);
    if (distance < INTERACTION_RADIUS) {
        return 1.0f / distance; // Simplified stress calculation
    }
    return 0.0f;
}

// Simulate crack propagation
void simulateCracks(std::vector<Particle>& particles) {
    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            if (particles[i].isBroken || particles[j].isBroken) continue;

            float stress = computeStress(particles[i], particles[j]);
            if (stress > FAILURE_THRESHOLD) {
                particles[i].isBroken = true;
                particles[j].isBroken = true;
            }
        }
    }
}

int main() {
    // Initialize SFML window
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Crack Propagation (Boundary-Free)");
    window.setFramerateLimit(30);

    // Initialize particles
    std::vector<Particle> particles;
    initializeParticles(particles);

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Simulate cracks
        simulateCracks(particles);

        // Render particles
        window.clear(sf::Color::White);
        for (const auto& particle : particles) {
            sf::CircleShape circle(PARTICLE_RADIUS);
            circle.setPosition(particle.position);
            circle.setFillColor(particle.isBroken ? sf::Color::Red : sf::Color::Black);
            window.draw(circle);
        }

        window.display();
    }

    return 0;
}
