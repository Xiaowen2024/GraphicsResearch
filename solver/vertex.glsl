#version 120
attribute vec3 aPos;
attribute vec2 aTexCoord;
varying vec2 fragTexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    fragTexCoord = aTexCoord;
}
