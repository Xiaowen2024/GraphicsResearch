#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

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

void render() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Use the shader program
    glUseProgram(shaderProgram);

    // Define a triangle
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

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
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Clean up
    glBindVertexArrayAPPLE(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Swap buffers
    glutSwapBuffers();
}
