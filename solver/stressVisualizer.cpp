#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#include <complex>
using namespace std;
using Vec2D = complex<double>;
//clang++ stressVisualizer.cpp -framework OpenGL -framework GLUT -o stressVisualizer

// Function prototypes
std::string loadShaderSource(const std::string& filepath);
unsigned int compileShader(const std::string& source, GLenum shaderType);
unsigned int createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
void render();

// Global shader program ID
unsigned int shaderProgram;

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("OpenGL Shader Example");

    // Load and compile shaders
    shaderProgram = createShaderProgram("vertex.glsl", "frag.glsl");

    // Set the render function
    glutDisplayFunc(render);

    // Main loop
    glutMainLoop();

    return 0;
}

std::string loadShaderSource(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

unsigned int compileShader(const std::string& source, GLenum shaderType) {
    unsigned int shader = glCreateShader(shaderType);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }
    return shader;
}

unsigned int createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath) {
    std::string vertexCode = loadShaderSource(vertexPath);
    std::string fragmentCode = loadShaderSource(fragmentPath);

    unsigned int vertexShader = compileShader(vertexCode, GL_VERTEX_SHADER);
    unsigned int fragmentShader = compileShader(fragmentCode, GL_FRAGMENT_SHADER);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader Program Linking Failed:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}


Vec2D deform(Vec2D point) {
    double x = real(point);
    double y = imag(point);
    if (x < 0){
        return Vec2D(x - 0.1, y);
    }
    else if (x > 0){
        return Vec2D(x + 0.1, y);
    }
}

void render() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Use the shader program
    glUseProgram(shaderProgram);
 // { {{Vec2D(-3, -2), Vec2D(3, -2), Vec2D(3, 2), Vec2D(-3, 2), Vec2D(-3, -2)}} };
    // Define a rectangle using OpenGL coordinates
    float vertices[] = {
        -.5f, -.5f, 0.0f,
         .5f, -.5f, 0.0f,
         .5f,  .5f, 0.0f,
         .5f,  .5f, 0.0f,
        -.5f,  .5f, 0.0f,
        -.5f, -.5f, 0.0f,
    };

    int numVertices = sizeof(vertices) / sizeof(vertices[0]) / 3;

    for (int i = 0; i < numVertices; ++i) {
        // Extract the x, y coordinates of each vertex
        float x = vertices[i * 3];
        float y = vertices[i * 3 + 1];

        // Apply the deform function
        Vec2D originalPoint(x, y);
        Vec2D deformedPoint = deform(originalPoint);

        // Update the vertex array with deformed values
        vertices[i * 3] = static_cast<float>(real(deformedPoint));
        vertices[i * 3 + 1] = static_cast<float>(imag(deformedPoint));
    }

    // Enable and bind the vertex data
    unsigned int VBO, VAO;
    glGenVertexArraysAPPLE(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArrayAPPLE(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Draw the triangle
    glBindVertexArrayAPPLE(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Clean up
    glBindVertexArrayAPPLE(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Swap buffers
    glutSwapBuffers();
}
