#version 120

uniform sampler2D stressTexture;

void main() {
   float stress = texture2D(stressTexture, gl_TexCoord[0].xy).r;
   // vec3 color = vec3(stress, 0.0, 1.0 - stress); 
   gl_FragColor = vec4(1, 1, 1, 1.0);
}




