#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#include <GLFW/glfw3.h>
#include <complex>
using namespace std;
using Vec2D = complex<double>;
#include "fractureModelHelpers.cpp"
#include "fractureModelHelpers.h"
//c++ stressVisualizer.cpp -std=c++17 \
  -I/System/Library/Frameworks/OpenGL.framework/Headers \
  -I/System/Library/Frameworks/GLUT.framework/Headers \
  -I/opt/homebrew/Cellar/glfw/3.4/include \
  -o stressVisualizer \
  -L/System/Library/Frameworks/OpenGL.framework/Libraries \
  -L/opt/homebrew/Cellar/glfw/3.4/lib \
  -framework OpenGL \
  -framework GLUT \
  -lglfw -w


// Function prototypes
std::string loadShaderSource(const std::string& filepath);
unsigned int compileShader(const std::string& source, GLenum shaderType);
unsigned int createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
void render();

// Global shader program ID
unsigned int shaderProgram;


void checkGLError(const std::string& functionName) {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error in " << functionName << ": ";
        
        // Handle specific error codes
        switch (error) {
            case GL_INVALID_ENUM:
                std::cerr << "GL_INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                std::cerr << "GL_INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                std::cerr << "GL_INVALID_OPERATION";
                break;
            case GL_STACK_OVERFLOW:
                std::cerr << "GL_STACK_OVERFLOW";
                break;
            case GL_STACK_UNDERFLOW:
                std::cerr << "GL_STACK_UNDERFLOW";
                break;
            case GL_OUT_OF_MEMORY:
                std::cerr << "GL_OUT_OF_MEMORY";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                std::cerr << "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            default:
                std::cerr << "Unknown error";
        }
        std::cerr << std::endl;
    }
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

vector<Polyline> boundaryDirichlet = { {{Vec2D(-0.5, -0.5), Vec2D(0.5, -0.5), Vec2D(0.5, 0.5), Vec2D(-0.5, 0.5), Vec2D(-0.5, -0.5)}} };
vector<Polyline> boundaryNeumann = {};
const int width = 1;
const int height = 1;
std::vector<float> stressData;
std::vector<float> texCoords;
// void generateStressData() {
//     for ( float x = -0.5; x <= 0.5; x += 0.1 ) {
//         for ( float y = -0.5; y <= 0.5; y += 0.1 ) {
//             Vec2D point(x, y);
//             Vec2D deformedPoint = deform(point);
//             vector<Vec2D> stress = returnStress(deformedPoint, 0.01, deform, boundaryDirichlet, boundaryNeumann);
//             if (std::isnan(real(stress[0])) || std::isnan(imag(stress[0])) || std::isnan(real(stress[1])) || std::isnan(imag(stress[1]))) {
//                 continue;
//             } 
//             else {
//                 texCoords.push_back(x);
//                 texCoords.push_back(y);
//             }
//             float stressMagnitude = sqrt(real(stress[0]) * real(stress[0]) + imag(stress[0]) * imag(stress[0]) + real(stress[1]) * real(stress[1]) + imag(stress[1]) * imag(stress[1]));
//             int yIndex = (y + 0.5) * 10;
//             int xIndex = (x + 0.5) * 10;
//             stressData[ y * width + x ] = stressMagnitude;
//         }
//     } 
//     float maxStress = 0.0f;
//     for (const auto& stress : stressData) {
//         if (stress > maxStress) {
//             maxStress = stress;
//         }
//     }
//     std::cout << "Max stress: " << maxStress << std::endl;
    
//     for (int i = 0; i < stressData.size(); ++i) {
//         stressData[i] /= maxStress;
//         // std::cout << "Stress at " << i << stressData[i] << std::endl;
//     }
// }
void generateStressData() {
    vector<float> stressMagnitudeVector;
    for (float x = -0.5; x <= 0.5; x += 0.1) {
        for (float y = -0.5; y <= 0.5; y += 0.1) {
            Vec2D point(x, y);
            Vec2D deformedPoint = deform(point);
            vector<Vec2D> stress = returnStress(deformedPoint, 0.01, deform, boundaryDirichlet, boundaryNeumann);

           

            // Map the x, y coordinates to texture coordinates
            texCoords.push_back((x + 0.5) / 1.0);  // Normalize the x coordinate to [0, 1]
            texCoords.push_back((y + 0.5) / 1.0);  // Normalize the y coordinate to [0, 1]


            if (std::isnan(real(stress[0])) || std::isnan(imag(stress[0])) || std::isnan(real(stress[1])) || std::isnan(imag(stress[1]))) {
                stressMagnitudeVector.push_back(0.0f);
            }
            else {
                float stressMagnitude = sqrt(real(stress[0]) * real(stress[0]) + imag(stress[0]) * imag(stress[0]) + 
                                         real(stress[1]) * real(stress[1]) + imag(stress[1]) * imag(stress[1]));

                int yIndex = (y + 0.5) * 10;
                int xIndex = (x + 0.5) * 10;
                
                int offset = (y * width + x) * 4;
                stressData[offset] = stressMagnitude * 255;
                stressData[offset + 1] = stressMagnitude * 255;
                stressData[offset + 2] = stressMagnitude * 255;
                stressData[offset + 3] = 255;
                stressMagnitudeVector.push_back(stressMagnitude);
            }

            float maxStress = 0.0f;
            for (const auto& stress : stressData) {
                if (stress > maxStress) {
                    maxStress = stress;
                }
            }

            for (int i = 0; i < stressData.size(); i += 4) {
                stressData[i] /= maxStress;
                stressData[i + 1] /= maxStress;
                stressData[i + 2] /= maxStress;
            }
        }
    }
}


GLuint uploadStressTexture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    int texWidth = 10; 
    int texHeight = 10;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, texWidth, texHeight, 0, GL_RED, GL_FLOAT, stressData.data());
    checkGLError("glTexImage2D");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return texture;
}


int main(int argc, char** argv) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Core profile for OpenGL 3.3
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // Important for macOS
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

    glBindAttribLocation(shaderProgram, 0, "aPos");
    glBindAttribLocation(shaderProgram, 1, "aTexCoord");

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

GLuint stressTexture;
void initializeStressTexture(){
    generateStressData();
    // stressTexture = uploadStressTexture();
}

void cleanup() {
    glDeleteTextures(1, &stressTexture);
}

void render() {
    initializeStressTexture();
    glClear(GL_COLOR_BUFFER_BIT);
    glUniform1i(glGetUniformLocation(shaderProgram, "stressTexture"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, stressTexture);

    // Use the shader program
    glUseProgram(shaderProgram);

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

    // generate a vector mapping geometry coordinates, colors, and texture coordinates
    float combined[sizeof(vertices) + sizeof(texCoords)];
    for (int i = 0; i < sizeof(vertices) / sizeof(vertices[0]); i += 3) {
        int index = i / 3;
        combined[i * 5 / 3] = vertices[i];
        combined[i * 5 / 3 + 1] = vertices[i + 1];
        combined[i * 5 / 3 + 2] = vertices[i + 2];
        combined[i * 5 / 3 + 3] = texCoords[index * 2];
        combined[i * 5 / 3 + 4] = texCoords[index * 2 + 1];
    }
    // Enable and bind the vertex data
    unsigned int VBO, VAO;
    glGenVertexArraysAPPLE(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArrayAPPLE(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(combined), combined, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArrayAPPLE(VAO);
    glDrawArrays(GL_TRIANGLES, 0, numVertices);

    glBindVertexArrayAPPLE(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glutSwapBuffers();
}

