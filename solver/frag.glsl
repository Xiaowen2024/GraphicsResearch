#version 120

uniform sampler2D stressTexture;
varying vec2 fragTexCoord;

void main() {
   gl_FragColor = texture2D(stressTexture, fragTexCoord);
}
